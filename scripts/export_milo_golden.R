#!/usr/bin/env Rscript
# Produces the miloR side of tests/milor_2_6_0_golden.json.
# Driven by scripts/export_milo_golden.py, which writes cases.json with
# reduced dimensions, an exact kNN graph (self excluded, one-based), the
# randomly sampled vertices, and per-cell sample and label metadata.
#
# R's random number generator cannot reproduce NumPy's draws, so the sampled
# vertices are injected: .refined_sampling refines them, and the body of
# makeNhoods builds the neighbourhoods of the refined vertices. countCells,
# testNhoods, and annotateNhoods then run unmodified, and the edgeR stages of
# testNhoods are recomputed to export their intermediates.
#
# A large case (the Wilk fixture) passes its coordinates and kNN rows as
# little-endian binary files, and CyteArc's refined index cells as
# indexVertices. Its graph comes from an ANN index rather than an exact
# search, so the neighbourhoods are built on CyteArc's index cells, R's own
# refinement of the same draws is exported for comparison, and the recall of
# the graph against BiocNeighbors' exact kNN is reported. Such a case may also
# ask for speckle's propeller on the same cells.

suppressPackageStartupMessages({
  library(jsonlite)
  library(Matrix)
  library(igraph)
  library(SingleCellExperiment)
  library(BiocNeighbors)
  library(edgeR)
  library(miloR)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("usage: export_milo_golden.R <work-dir>")
}
work <- args[[1]]
cases <- fromJSON(file.path(work, "cases.json"), simplifyVector = FALSE)

as_matrix <- function(rows) {
  do.call(rbind, lapply(rows, function(row) as.numeric(unlist(row))))
}

read_rows <- function(case, file, what, size, n_cols) {
  path <- file.path(work, file)
  values <- readBin(path, what, n = case$nCells * n_cols, size = size, endian = "little")
  matrix(values, nrow = case$nCells, ncol = n_cols, byrow = TRUE)
}

results <- list()
for (case in cases) {
  if (is.null(case$coordinatesFile)) {
    pcs <- as_matrix(case$coordinates)
    knn <- as_matrix(case$knn)
  } else {
    pcs <- read_rows(case, case$coordinatesFile, "double", 8L, case$dims)
    knn <- read_rows(case, case$knnFile, "integer", 4L, case$k)
  }
  n_cells <- nrow(pcs)
  cell_ids <- sprintf("cell%04d", seq_len(n_cells))
  rownames(pcs) <- cell_ids
  colnames(pcs) <- sprintf("PC%d", seq_len(ncol(pcs)))
  k <- ncol(knn)
  meta <- data.frame(
    sample = unlist(case$sample),
    label = unlist(case$label),
    row.names = cell_ids,
    stringsAsFactors = FALSE
  )
  sce <- SingleCellExperiment(
    assays = list(counts = Matrix(0, nrow = 1, ncol = n_cells, sparse = TRUE)),
    colData = meta,
    reducedDims = list(PCA = pcs)
  )
  colnames(sce) <- cell_ids
  milo <- Milo(sce)
  graph(milo) <- miloR:::.neighborsToKNNGraph(knn, directed = FALSE)
  milo@.k <- as.integer(k)

  random_vertices <- as.integer(unlist(case$randomVertices))
  r_refined <- NULL
  if (isTRUE(case$refined)) {
    r_refined <- unique(suppressMessages(
      miloR:::.refined_sampling(random_vertices, pcs, k)
    ))
  }
  if (!is.null(case$indexVertices)) {
    sampled <- as.integer(unlist(case$indexVertices))
  } else if (isTRUE(case$refined)) {
    sampled <- r_refined
  } else {
    sampled <- unique(random_vertices)
  }
  # makeNhoods sets column j to the closed graph neighbourhood of vertex j.
  x_graph <- graph(milo)
  member_lists <- neighborhood(x_graph, order = 1, nodes = sampled)
  nh_mat <- sparseMatrix(
    i = unlist(lapply(member_lists, as.integer)),
    j = rep(seq_along(sampled), lengths(member_lists)),
    x = 1,
    dims = c(n_cells, length(sampled)),
    dimnames = list(cell_ids, as.character(sampled))
  )
  nhoodIndex(milo) <- as(sampled, "list")
  nhoods(milo) <- nh_mat

  milo <- countCells(milo, meta.data = data.frame(colData(milo)), samples = "sample")
  counts <- as.matrix(nhoodCounts(milo))

  design_df <- as.data.frame(lapply(case$designDf, unlist), stringsAsFactors = FALSE)
  rownames(design_df) <- design_df$sample
  design_df <- design_df[colnames(counts), , drop = FALSE]
  if (!is.null(case$conditionLevels)) {
    design_df$condition <- factor(design_df$condition, levels = unlist(case$conditionLevels))
  }
  formula <- as.formula(case$formula)
  contrast <- unlist(case$modelContrasts)
  res <- suppressWarnings(suppressMessages(testNhoods(
    milo,
    design = formula,
    design.df = design_df,
    fdr.weighting = "k-distance",
    reduced.dim = "PCA",
    norm.method = case$normMethod,
    min.mean = case$minMean,
    model.contrasts = contrast
  )))
  res <- res[order(res$Nhood), ]
  annotated <- suppressMessages(annotateNhoods(milo, res, "label"))

  x_model <- model.matrix(formula, data = design_df)
  keep <- if (case$minMean > 0) rowMeans(counts) >= case$minMean else rep(TRUE, nrow(counts))
  dge <- DGEList(counts = counts[keep, , drop = FALSE], lib.size = colSums(counts[keep, , drop = FALSE]))
  if (case$normMethod != "logMS") dge <- calcNormFactors(dge, method = case$normMethod)
  dge <- estimateDisp(dge, x_model)
  fit <- glmQLFit(dge, x_model, robust = TRUE, legacy = TRUE)

  # graphSpatialFDR takes the distance to the k-th exact neighbour.
  kdist <- apply(findKNN(pcs, get.distance = TRUE, subset = sampled, k = k)[["distance"]], 1, max)

  members <- lapply(seq_len(ncol(nh_mat)), function(j) which(nh_mat[, j] > 0) - 1L)
  record <- list(
    name = case$name,
    refinedVertices = as.integer(sampled) - 1L,
    nhoodMembers = members,
    nhoodCounts = unname(lapply(seq_len(nrow(counts)), function(i) as.integer(counts[i, ]))),
    sampleOrder = colnames(counts),
    designColumns = colnames(x_model),
    normFactors = as.numeric(dge$samples$norm.factors),
    trendedDispersion = as.numeric(dge$trended.dispersion),
    dfPrior = as.numeric(rep_len(fit$df.prior, nrow(dge))),
    s2Prior = as.numeric(rep_len(fit$s2.prior, nrow(dge))),
    kDistance = as.numeric(kdist),
    logFc = as.numeric(res$logFC),
    logCpm = as.numeric(res$logCPM),
    f = as.numeric(res$F),
    pValue = as.numeric(res$PValue),
    fdr = as.numeric(res$FDR),
    spatialFdr = as.numeric(res$SpatialFDR),
    labelMajority = as.character(annotated$label),
    labelFraction = as.numeric(annotated$label_fraction)
  )
  if (!is.null(case$indexVertices) && !is.null(r_refined)) {
    record$rRefinedVertices <- as.integer(r_refined) - 1L
  }
  if (isTRUE(case$exactKnn)) {
    exact <- findKNN(pcs, k = k)[["index"]]
    hits <- vapply(seq_len(n_cells), function(i) length(intersect(exact[i, ], knn[i, ])), integer(1))
    record$knnRecall <- sum(hits) / (n_cells * k)
    record$indexRowExact <- hits[sampled] == k
  }
  if (isTRUE(case$propeller)) {
    suppressPackageStartupMessages(library(speckle))
    group_of <- setNames(as.character(design_df$condition), rownames(design_df))
    out <- suppressMessages(speckle::propeller(
      clusters = meta$label,
      sample = meta$sample,
      group = unname(group_of[meta$sample])
    ))
    record$propeller <- list(
      categories = rownames(out),
      tStatistic = as.numeric(out$Tstatistic),
      propRatio = as.numeric(out$PropRatio),
      pValue = as.numeric(out$P.Value),
      fdr = as.numeric(out$FDR)
    )
  }
  results[[length(results) + 1L]] <- record
}

versions <- list(
  r = paste(R.version$major, R.version$minor, sep = "."),
  miloR = as.character(packageVersion("miloR")),
  edgeR = as.character(packageVersion("edgeR")),
  limma = as.character(packageVersion("limma")),
  BiocNeighbors = as.character(packageVersion("BiocNeighbors")),
  igraph = as.character(packageVersion("igraph")),
  speckle = as.character(packageVersion("speckle"))
)
write(
  toJSON(list(versions = versions, results = results), auto_unbox = TRUE, digits = NA),
  file.path(work, "results.json")
)
