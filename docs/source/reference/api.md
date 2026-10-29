(api)=
# API reference

Public CyteArc surfaces for analysts:

- `DataStore` and its documented methods
- Graph-construction methods and the pipeline recipes `ds.pipeline.rna`, `ds.pipeline.atac`,
  `ds.pipeline.adt`, and `ds.pipeline.multimodal`
- `ArtifactRef`, `ArtifactStatus`, `PipelineRun`, and strict artifact-resolution errors
- `pipeline_listener`, `CancellationToken`, and `PipelineCancelled` for following and stopping
  pipeline runs
- `EnrichmentResult` and `read_gmt` for gene-set scoring
- `cytearc.regulons` for SCENIC-style regulons, chromVAR motif activity, cistromes from motifs or
  CUT&RUN peaks, region-to-gene links, and co-regulated gene programs
- `cytearc.perturbation` for pooled CRISPR screens: guide calls made from the guide counts or imported from Cell Ranger, the thresholds fitted to each guide, perturbation labels that name the genes each cell's guides target, the efficiency of each perturbation on its target gene, and the E-distance of each perturbation's cells from the control cells
- `cytearc.repertoire` for single-cell T cell and B cell receptors, starting with contigs imported from Cell Ranger and AIRR Rearrangement files and the ranked chains, receptor type, subtype and chain pairing of each cell
- Readers that ingest source formats; writers that create or export Zarr stores (and other exports)
- `cytearc.plotting`
- `cytearc.datasets` for the public example datasets that the tutorials download
- Documented integration metrics (`DataStore.metric_*`; `cytearc.metrics` holds the underlying functions)
- `MappingReference` / `MappingResult` for atlas-style mapping

Inheritance helpers (`BaseDataStore`, `GraphDataStore`, `MappingDatastore`) are listed under [DataStore API reference](/api/datastore.html) for completeness.
Prefer calling methods on `DataStore`.

## By analysis stage

| Stage | Page |
|---|---|
| Import and export | [Import and export API reference](/api/import_export.html) |
| DataStore (all stages) | [DataStore API reference](/api/datastore.html) |
| Graph construction | [Graph construction API reference](/api/graph_construction.html) |
| Artifacts, lineage, and summaries | [Artifacts, lineage, and summaries API reference](/api/artifacts.html) |
| Analysis pipeline | [Analysis pipeline API reference](/api/pipeline.html) |
| Assays and metadata | [Assays and metadata API reference](/api/assays.html) |
| Integration and metrics | [Integration and metrics API reference](/api/integration.html) |
| Mapping | [Mapping API reference](/api/mapping.html) |
| Regulons and gene programs | [Regulons API reference](/api/regulons.html) |
| CRISPR screens | [Perturbation API reference](/api/perturbation.html) |
| Immune receptor repertoires | [Repertoire API reference](/api/repertoire.html) |
| Plotting | [Plotting API reference](/api/plotting.html) |
| Example datasets | [Example datasets API reference](/api/datasets.html) |
| Utilities | [Utilities API reference](/api/utilities.html) |

The CyteBase catalog of cloud-hosted datasets is the separate `cytebase` package; see the
[CyteBase documentation](https://docs.nygen.io/CyteBase/) for its API. CyteArc opens CyteBase
datasets by URI, as [](../tutorials/cytebase_datasets.md) shows.

[](../scanpy.md), [](../seurat.md), and [](../tutorials/scrna_seq.ipynb) describe related workflows;
this table is this reference's own grouping and only partially overlaps those pages.
