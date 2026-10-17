#!/usr/bin/env Rscript
# Produces the speckle side of tests/speckle_1_10_0_propeller_golden.json.
# Driven by scripts/export_propeller_golden.py, which writes cases.json with
# per-case categories x samples counts. Each case is expanded to one row per
# cell and passed to speckle::propeller, and the limma internals that
# propeller uses are recomputed to export the empirical Bayes prior.

suppressPackageStartupMessages({
  library(jsonlite)
  library(limma)
  library(speckle)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("usage: export_propeller_golden.R <work-dir>")
}
work <- args[[1]]
cases <- fromJSON(file.path(work, "cases.json"), simplifyVector = FALSE)

expand_cells <- function(case) {
  counts <- matrix(
    unlist(case$counts),
    nrow = length(case$categories),
    byrow = TRUE
  )
  categories <- unlist(case$categories)
  samples <- unlist(case$samples)
  groups <- unlist(case$groups)
  clusters <- character(0)
  sample <- character(0)
  group <- character(0)
  for (j in seq_along(samples)) {
    for (i in seq_along(categories)) {
      n <- counts[i, j]
      clusters <- c(clusters, rep(categories[i], n))
      sample <- c(sample, rep(samples[j], n))
      group <- c(group, rep(groups[j], n))
    }
  }
  list(clusters = clusters, sample = sample, group = group)
}

prior_of <- function(cells, case) {
  prop.list <- getTransformedProps(cells$clusters, cells$sample, case$transform)
  group.coll <- table(cells$sample, cells$group)
  design <- matrix(as.integer(group.coll != 0), ncol = ncol(group.coll))
  robust <- isTRUE(case$robust) && nrow(prop.list$TransformedProps) > 2
  if (ncol(design) == 2) {
    fit <- lmFit(prop.list$TransformedProps, design)
    fit <- contrasts.fit(fit, contrasts = c(1, -1))
  } else {
    design[, 1] <- 1
    fit <- lmFit(prop.list$TransformedProps, design)
    fit <- fit[, seq_len(ncol(design))[-1]]
  }
  fit <- eBayes(fit, robust = robust, trend = isTRUE(case$trend))
  list(
    dfPrior = as.numeric(rep_len(fit$df.prior, nrow(fit))),
    s2Prior = as.numeric(rep_len(fit$s2.prior, nrow(fit))),
    proportions = unname(lapply(seq_len(nrow(prop.list$Proportions)),
      function(i) as.numeric(prop.list$Proportions[i, ]))),
    transformed = unname(lapply(seq_len(nrow(prop.list$TransformedProps)),
      function(i) as.numeric(prop.list$TransformedProps[i, ])))
  )
}

results <- list()
for (case in cases) {
  cells <- expand_cells(case)
  out <- suppressMessages(propeller(
    clusters = cells$clusters,
    sample = cells$sample,
    group = cells$group,
    transform = case$transform,
    robust = isTRUE(case$robust),
    trend = isTRUE(case$trend)
  ))
  categories <- unlist(case$categories)
  out <- out[categories, , drop = FALSE]
  baseline_column <- grep("^BaselineProp", colnames(out), value = TRUE)
  baseline_column <- baseline_column[length(baseline_column)]
  prop_mean_columns <- grep("^PropMean", colnames(out), value = TRUE)
  record <- list(
    name = case$name,
    baselineProp = as.numeric(out[[baseline_column]]),
    propMean = unname(lapply(prop_mean_columns, function(column) as.numeric(out[[column]]))),
    pValue = as.numeric(out$P.Value),
    fdr = as.numeric(out$FDR)
  )
  if ("Tstatistic" %in% colnames(out)) {
    record$tStatistic <- as.numeric(out$Tstatistic)
    record$propRatio <- as.numeric(out$PropRatio)
  } else {
    record$fStatistic <- as.numeric(out$Fstatistic)
  }
  results[[length(results) + 1L]] <- c(record, prior_of(cells, case))
}

versions <- list(
  r = paste(R.version$major, R.version$minor, sep = "."),
  speckle = as.character(packageVersion("speckle")),
  limma = as.character(packageVersion("limma"))
)
write(
  toJSON(list(versions = versions, results = results), auto_unbox = TRUE, digits = NA),
  file.path(work, "results.json")
)
