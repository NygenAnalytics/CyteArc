(api)=
# API reference

Public CyteArc surfaces for analysts:

- `DataStore` and its documented methods
- Graph-construction methods and `ds.pipeline.run`
- `ArtifactRef`, `ArtifactStatus`, `PipelineRun`, and strict artifact-resolution errors
- `EnrichmentResult` and `read_gmt` for gene-set scoring
- Readers that ingest source formats; writers that create or export Zarr stores (and other exports)
- `cytearc.plotting`
- `cytearc.cytebase.Catalog`, metadata-only `DatasetEntry` objects, and direct access to cloud-hosted CyteArc DataStores
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
| Plotting | [Plotting API reference](/api/plotting.html) |
| CyteBase datasets | [CyteBase API reference](/api/cytebase.html) |
| Utilities | [Utilities API reference](/api/utilities.html) |

[](../scanpy.md), [](../seurat.md), and [](../tutorials/scrna_seq.ipynb) describe related workflows;
this table is this reference's own grouping and only partially overlaps those pages.
