---
myst:
  html_meta:
    description: CyteArc regulon API for GRNBoost2 networks, cisTarget motif pruning, chromVAR motif activity, motif and CUT&RUN cistromes, enhancer-driven regulons, and co-regulated gene programs.
---

# Regulons API reference

`cytearc.regulons` builds transcription-factor regulons the way SCENIC and SCENIC+ do, and
co-regulated gene programs, from RNA, ATAC, and binding data such as CUT&RUN. Every set it
returns, from modules and regulons to programs, cistromes, and eRegulons, is a table with
`source` (the set) and `target` (a member) columns, which {meth}`ds.scores.aucell
<cytearc.datastore.namespaces.ScoresAccessor.aucell>` (genes) and {meth}`ds.scores.chromvar
<cytearc.datastore.namespaces.ScoresAccessor.chromvar>` (peaks) score per cell and
`set_overlaps` compares. Regions are named by their `chrom:start-end` ids, as ATAC peaks are.
Reading cisTarget ranking databases needs the `regulons` extra: `pip install "cytearc[regulons]"`.

The cell-level steps run on a `DataStore`: {meth}`ds.regulons.grn
<cytearc.datastore.namespaces.RegulonsAccessor.grn>` infers the co-expression network,
{meth}`ds.regulons.profiles <cytearc.datastore.namespaces.RegulonsAccessor.profiles>` sums
groups of cells such as metacells into profiles, and the scoring methods above write
`enrichment_scores` artifacts, which {meth}`EnrichmentResult.to_pandas
<cytearc.EnrichmentResult.to_pandas>` reads as a table. The functions on this page work on
those results and on reference files, such as cisTarget databases, JASPAR motifs, a genome
FASTA, a gene annotation, and peak files.

## RNA regulons (SCENIC)

GRNBoost2 importances become co-expression modules, motif enrichment keeps the targets that an
enriched, annotated motif of each module's factor recovers, and the leading edges merge into
`TF(+)` regulons.

```{eval-rst}
.. autofunction:: cytearc.regulons.infer_grn

.. autoclass:: cytearc.regulons.GrnParameters

.. autofunction:: cytearc.regulons.modules_from_adjacencies

.. autoclass:: cytearc.regulons.ModuleParameters

.. autoclass:: cytearc.regulons.RankingDatabase
    :members: ranks, total

.. autofunction:: cytearc.regulons.read_motif_annotations

.. autofunction:: cytearc.regulons.prune_modules

.. autoclass:: cytearc.regulons.PruningParameters

.. autofunction:: cytearc.regulons.regulons_from_motifs
```

## Motifs, genomes, and cistromes

```{eval-rst}
.. autoclass:: cytearc.regulons.IndexedFasta
    :members: fetch, encode, names, length, close

.. autofunction:: cytearc.regulons.gc_content

.. autoclass:: cytearc.regulons.Motif

.. autofunction:: cytearc.regulons.read_jaspar_motifs

.. autofunction:: cytearc.regulons.read_meme_motifs

.. autofunction:: cytearc.regulons.match_motifs

.. autofunction:: cytearc.regulons.motif_annotations

.. autofunction:: cytearc.regulons.cistromes_from_motifs

.. autofunction:: cytearc.regulons.read_peak_file

.. autofunction:: cytearc.regulons.binding_cistromes
```

## Enhancer-driven regulons (SCENIC+)

```{eval-rst}
.. autofunction:: cytearc.regulons.read_gene_annotation

.. autofunction:: cytearc.regulons.link_regions_to_genes

.. autofunction:: cytearc.regulons.region_gene_links

.. autofunction:: cytearc.regulons.build_eregulons

.. autoclass:: cytearc.regulons.EregulonParameters
```

## Gene programs and specificity

```{eval-rst}
.. autofunction:: cytearc.regulons.gene_programs

.. autoclass:: cytearc.regulons.GeneProgramParameters

.. autofunction:: cytearc.regulons.program_scores

.. autofunction:: cytearc.regulons.program_regulators

.. autofunction:: cytearc.regulons.set_overlaps

.. autofunction:: cytearc.regulons.regulon_specificity

.. autofunction:: cytearc.regulons.connection_specificity

.. autofunction:: cytearc.regulons.regulon_modules
```
