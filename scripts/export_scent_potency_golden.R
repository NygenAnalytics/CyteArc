# Compute SCENT's signaling entropy rate and CCAT for CyteArc's reference outputs.
#
# SCENT (GPL-3, https://github.com/aet21/SCENT) is run, never copied: these
# calls follow its vignette. The input directory holds log_values.csv (genes
# by cells, the values of scripts/export_scent_potency_golden.py prepare) and
# edges.csv (zero-based positions of each edge's genes in that order).
#
# Usage: Rscript scripts/export_scent_potency_golden.R <input directory> <output directory>

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript export_scent_potency_golden.R <input directory> <output directory>")
}
input <- args[[1]]
output <- args[[2]]
suppressPackageStartupMessages(library(SCENT))

values <- as.matrix(read.csv(file.path(input, "log_values.csv"), row.names = 1, check.names = FALSE))
edges <- read.csv(file.path(input, "edges.csv"))
genes <- rownames(values)
adjacency <- matrix(0, nrow = length(genes), ncol = length(genes), dimnames = list(genes, genes))
adjacency[cbind(edges$first + 1, edges$second + 1)] <- 1
adjacency[cbind(edges$second + 1, edges$first + 1)] <- 1

integrated <- DoIntegPPI(exp.m = values, ppiA.m = adjacency)
if (!setequal(rownames(integrated$expMC), genes)) {
  stop("SCENT's integration kept other genes than the prepared network's")
}
rate <- CompSRana(integrated, local = FALSE, mc.cores = 1)
ccat <- CompCCAT(exp = values, ppiA = adjacency)

scores <- data.frame(
  cell = colnames(values),
  signaling_entropy = sprintf("%.17g", rate$SR),
  ccat = sprintf("%.17g", ccat)
)
write.csv(scores, file.path(output, "scent_scores.csv"), row.names = FALSE, quote = TRUE)
writeLines(
  c(
    paste("SCENT", as.character(packageVersion("SCENT"))),
    paste("SCENT commit", packageDescription("SCENT")$RemoteSha),
    R.version.string
  ),
  file.path(output, "versions.txt")
)
