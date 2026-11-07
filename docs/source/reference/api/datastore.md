---
myst:
  html_meta:
    description: CyteArc DataStore methods for opening stores, selecting cells and features, running analyses, and accessing saved results.
---

# DataStore API reference

`DataStore` is the primary analyst-facing object.
It inherits graph, mapping, and assay helpers from the classes below.
Use this page for analyst-facing methods and consult the inheritance appendix when extending CyteArc.
A `DataStore` is not designed for concurrent use from several threads; call its methods from one
thread at a time. Write a store from one process at a time.

Graph-construction methods are documented on {doc}`graph_construction`.
Artifact and run inspection, including the metadata-only `DataStore.summary()`, is documented on {doc}`artifacts`.
Mapping methods are on {doc}`mapping`, and integration metrics are on {doc}`integration`.
Those methods are excluded here rather than repeated.

`summary` is reserved for `DataStore.summary()` and cannot be used as an assay name.
Writers and `DataStore` opening reject that name before mutating store-level state.
Assays are also available as attributes such as `ds.RNA`; an assay named like a `DataStore`
attribute, such as `cells`, or starting with an underscore is available only through `get_assay(name)`.
Opening also rejects the removed `{assay}/state` group. Rebuild such a store with the current
release; CyteArc does not read or migrate its former analysis state.

## Mounting shared count matrices

Use `mount_datastore` when count matrices stay in a read-only source store and analysis artifacts should write to a separate target.
See [Remote stores and mounted analysis targets](../tutorials/remote-stores).

```{eval-rst}
.. autofunction:: cytearc.mount_datastore
```

```{eval-rst}
.. autoclass:: cytearc.datastore.datastore.DataStore
    :members:
    :inherited-members:
    :exclude-members: summary, snapshot_cell_selection
```

## Analysis namespaces

For a standard analysis, start with a pipeline recipe, such as `ds.pipeline.rna(label=...)`,
`ds.pipeline.atac(...)`, `ds.pipeline.adt(...)`, `ds.pipeline.multimodal(...)`, or
`ds.pipeline.perturbation(...)`
({doc}`pipeline`). Use the namespaces below when selecting individual analysis steps. Each method
keeps its scientific defaults, return type, and saved-result identity. Cell metadata stays on
`ds.cells`, and `ds.get_assay(name)` provides access to every assay, including one whose name
matches a namespace.

### `ds.artifacts`

`find` resolves one complete saved result. It searches computed results by default; use
`ds.artifacts.find("embedding", imported="X_umap", from_assay="RNA")` for an imported embedding.
See {doc}`artifacts` for supported filters and ambiguity handling.

```{eval-rst}
.. autoclass:: cytearc.datastore.artifact_accessor.ArtifactAccessor
    :members: find
```

### `ds.qc`

QC bounds are exclusive: `qc.filter(["RNA_nFeatures"], lows=[200], highs=[None])` removes a
cell with exactly 200 detected genes, which scanpy's `filter_cells(min_genes=200)` and CyteArc's
`features.detected(min_cells=...)` keep. Pass `keep_bounds=True` to `qc.filter` or
`select_cells` for inclusive bounds; the bounds of `qc.auto_filter` are always exclusive, and a
writable open with `min_features_per_cell=k` removes cells with at most `k` features.
`qc.auto_filter` chooses each metric's MAD policy from what CyteArc wrote and warns when it
falls back to a column's name; pass `metric_policies` for imported columns such as scanpy's
`total_counts` or `pct_counts_mt`.

`qc.hto_demultiplexing`, `qc.call_cells`, `qc.ambient_contamination` and `qc.doublets` save the numbers behind their warnings as {ref}`diagnostics <artifact_diagnostics>`, which their entries below list, and warn from them each time their artifact is computed or reused.

`qc.call_cells` calls cells in an unfiltered droplet matrix with EmptyDrops ([Cell calling](../tutorials/cell-calling)). Import every barcode with counts, such as `CrH5Reader(path, is_filtered=False, filtering_cutoff=0)`: the barcodes with at most `lower` counts form the ambient pool that the other barcodes are tested against. It reads every barcode that the assay measured, never the live `I` column, so a writable open's `min_features_per_cell` floor cannot remove the ambient pool; open raw stores with `min_features_per_cell=-1` to keep `I` complete too. It saves the statistics of every barcode as an assay-scoped `cell_calling` artifact (operation `run_cell_calling`; arrays `total`, `log_probability`, `p_value`, `fdr`, `limited`, and the feature-length `ambient` profile; attributes `retain`, `knee`, `inflection`, `alpha`, and `n_ambient`; diagnostic `goodTuringSlope`) and returns a `cell_selection` (operation `call_cells`, parameter `fdr`) whose inputs are that artifact and the tested barcodes (a `select_all_cells` selection, narrowed by `select_measured_cells` when the assay did not measure every barcode). Another `fdr` reuses the statistics. `qc.load_cell_calls` reads them, and `qc.barcode_ranks` gives the barcode-rank curve from the prepared `<assay>_nCounts` totals without reading counts.

```{eval-rst}
.. autoclass:: cytearc.datastore.namespaces.QcAccessor
    :members: filter, auto_filter, feature_percentage, doublets, hto_demultiplexing, call_cells, load_cell_calls, barcode_ranks
```

The same computation runs on an in-memory matrix:

```{eval-rst}
.. autofunction:: cytearc.quality_control.empty_drops

.. autofunction:: cytearc.quality_control.barcode_ranks

.. autofunction:: cytearc.quality_control.good_turing_proportions
```

#### Ambient RNA

These methods estimate and remove ambient RNA as SoupX describes it (Young and Behjati, 2020, GigaScience, [doi:10.1093/gigascience/giaa151](https://doi.org/10.1093/gigascience/giaa151)); see [Ambient RNA removal](../tutorials/ambient-rna). `qc.ambient_profile` saves an assay-scoped `ambient_profile` artifact: the summed `counts` of each feature over the chosen empty droplets and each feature's `fraction` of them. `qc.ambient_contamination` saves an assay-scoped, cell-aligned `ambient_contamination` artifact over its cell selection, whose `values` hold each cell's contamination fraction and, for an estimate from gene sets, whose `observed`, `expected`, `p_values`, `expressing`, and `used` arrays hold one column per gene set, in name order. It records the profile, the clustering, and the resolved feature indices of each gene set. An estimate of exactly 0, whose cells hold no counts of their gene sets although pure soup would give them some, is saved as computed: a correction with it leaves those cells uncorrected. Its diagnostics count such estimates as `zeroEstimates` and give up to five as `zeroEstimateExamples`, each with its cells, the pure-soup count and the one-sided 95% Poisson upper bound of the fraction, about 3 divided by that count, and `largestEstimate` holds the largest of the global and cluster estimates. A `DiagnosticWarning` names the estimates of 0, and another an estimate above `max_contamination`, each time the estimate is computed or reused. Loading or correcting an estimate saved without these diagnostics, as earlier releases saved them, raises `ValueError` naming the call that estimates it anew. All three methods read RNA assays of raw UMI counts, which CyteArc stores as unsigned integers, and raise `TypeError` for other assays.

`qc.correct_ambient` writes a derived RNA assay, published atomically like the assays of
`add_grouped_assay`. It keeps the source's features and count dtype, holds integer counts that
never exceed the source's, and measures only the cells of the estimate: when they are not every
cell, its membership column `<assay_label>_I` marks them and the other rows hold no counts. Its
attributes record `ambient_source_assay`, the `ambient_contamination` and `ambient_profile`
artifacts, `ambient_random_seed`, `ambient_corrected_cells`, and `ambient_removed_counts`. Each
cell loses its fraction of its total, or every count of its genes that the soup holds when they
are fewer, rounded down or up, so a cell's removed counts are its `<source>_nCounts` minus its
`<assay_label>_nCounts`. Each gene's share of that removal is rounded at random, as in the paper,
by systematic sampling with one draw per cell: a gene is rounded up with a probability equal to
the fractional part of its share, so summed over cells every gene loses its share of the soup on
average. A cell's draw depends only on `random_seed` (0 by default) and the cell's row in the
store, so the counts are reproducible, do not depend on the memory budget, and are the same for
a cell whichever other cells are corrected with it; another seed gives other counts and another
assay identity. The correction streams the cells in bounded row bands.

```{eval-rst}
.. automethod:: cytearc.datastore.namespaces.QcAccessor.ambient_profile
.. automethod:: cytearc.datastore.namespaces.QcAccessor.load_ambient_profile
.. automethod:: cytearc.datastore.namespaces.QcAccessor.ambient_contamination
.. automethod:: cytearc.datastore.namespaces.QcAccessor.load_ambient_contamination
.. automethod:: cytearc.datastore.namespaces.QcAccessor.correct_ambient

.. autoclass:: cytearc.quality_control.ambient.AmbientContamination
    :members:
```

### `ds.features`

```{eval-rst}
.. autoclass:: cytearc.datastore.namespaces.FeaturesAccessor
    :members: universe, hvgs, detected, prevalent_peaks, snapshot
```

### `ds.reduction`

```{eval-rst}
.. autoclass:: cytearc.datastore.namespaces.ReductionAccessor
```

### `ds.graph`

```{eval-rst}
.. autoclass:: cytearc.datastore.namespaces.GraphAccessor
```

### `ds.embeddings`

```{eval-rst}
.. autoclass:: cytearc.datastore.namespaces.EmbeddingsAccessor
    :members: umap, tsne
```

### `ds.integration`

```{eval-rst}
.. autoclass:: cytearc.datastore.namespaces.IntegrationAccessor
```

### `ds.clusters`

`clusters.leiden` runs CyteArc's own Leiden implementation, `backend="cytearc"`, by default
([Leiden clustering](../reference/methods#leiden-method)). It uses the datastore's `nthreads` and
returns the same labels for any thread count, numbered from 1 with the largest cluster first.
`backend="igraph"` and `backend="leidenalg"` call the GPL-licensed igraph and leidenalg packages,
which CyteArc does not install
([Optional GPL Leiden backends](../installation#installation-leiden-gpl)); without them, computing
a new clustering with either backend raises `ImportError` that explains how to install them. The
backend is part of the artifact identity: a clustering saved with `"igraph"`, the default of
earlier releases, is reused only by a call that passes `backend="igraph"`, and reusing it does not
need the packages.

`clusters.connectivity(graph, groups)` sums the graph's weight between each pair of clusters and returns the tables that `plots.cluster_connectivity` draws, without a layout or a figure, as a {py:class}`~cytearc.clustering.connectivity.ClusterConnectivity`: `nodes` holds each cluster's cells and incident weight, and `edges` each pair's `rawWeight` and `normalizedWeight`. It needs no plotting dependencies and saves nothing, so a diagnostic can call it once per clustering, such as each Leiden resolution. `groups` is a clustering or another categorical cell artifact, read for the graph's cells as the plot reads it, or a cell metadata column read over the graph's cells. Its filters keep every pair by default, while the plot's `minimum_edge_weight` defaults to 0.02; pass the plot's filters to get the edges it draws.

```{eval-rst}
.. autoclass:: cytearc.datastore.namespaces.ClustersAccessor
    :members: leiden, paris, load_paris, membership_strength, relabel_by_overlap, connectivity
```

### `ds.markers`

```{eval-rst}
.. autoclass:: cytearc.datastore.namespaces.MarkersAccessor
    :members: search, load, export_csv
```

### `ds.scores`

`chromvar` scores peak sets, such as motif matches or CUT&RUN peaks of one factor, on an
ATAC assay; see {doc}`regulons` for building those networks.

```{eval-rst}
.. autoclass:: cytearc.datastore.namespaces.ScoresAccessor
    :members: aucell, chromvar, waggr, load, cell_cycle
```

### `ds.regulons`

`grn` infers transcription-factor to target importances with GRNBoost2 on the assay's stored values, its raw counts or the values of an assay imported from normalized values; the functions in {doc}`regulons` turn them into regulons. `specificity` and `connection_specificity` give the regulon specificity scores of each group of cells and the connection specificity index of each pair of regulons from a stored score artifact, such as AUCell's, reading its scores a block of cells at a time, so their memory follows a block of scores and the groups and regulons rather than every cell's scores. They save nothing.

```{eval-rst}
.. autoclass:: cytearc.datastore.namespaces.RegulonsAccessor
    :members: grn, load_grn, profiles, specificity, connection_specificity
```

### `ds.perturbation`

`ds.perturbation` analyzes a pooled CRISPR screen: it calls the guides of each cell or imports Cell Ranger's calls, labels each cell by the genes that its guides target, measures each perturbation's effect on its target gene and its E-distance from the control cells, separates knocked-out cells from escapers with Mixscape, tests which genes each perturbation changes, and clusters the perturbations whose profiles agree. Each method saves an artifact, which a `load_*` method reads as a record of `cytearc.perturbation`. {doc}`perturbation` explains each method, its settings and checks, and its record.

```{eval-rst}
.. autoclass:: cytearc.datastore.namespaces.PerturbationAccessor
    :members: call_guides, load_guide_thresholds, import_guide_calls, load_guide_calls, labels, load_labels, efficiency, load_efficiency, edistance, load_edistance, signature, load_signature, mixscape, load_mixscape, associations, load_associations, profiles, load_profiles, cluster, load_clusters
```

### `ds.clones`

`import_clones` reads the clones that lineage tracing tools write, such as TREX's `clones.txt`, CellTag-multi's clone tables, Numbat's clone posteriors or LARRY's clone matrices, joins their cells to a cell selection by barcode, and saves a `clones` artifact that labels each cell with its clone. `from_column` saves the clones of a cell metadata column instead. `load_clones` reads the artifact as tables of cells and clones, and `export_clones` writes it as CoSpar's clone table or a Matrix Market file. `expansion` saves each cell's expansion category, by its clone's size within its group of cells, and `load_expansion` reads it as tables of cells, categories and rank windows; `diversity` and `rarefaction` measure each group's diversity indices and Hill numbers, rarefied and extrapolated, `overlap` and `track` compare groups by the clones they share and follow chosen clones across them, and `coupling` measures how strongly groups share clones against shuffled labels; these five save nothing. `fate_bias` saves, for each early cell, the share of its clone's later cells in each fate, and tests each clone's fates, and `load_fate_bias` reads it as tables of early cells, clones and fates. `import_barcodes` reads the expressed lineage barcodes that extraction tools write as reads, molecules or barcode counts, such as TREX's molecule tables, pycashier's and CellTag's reads, QuiCAT's and BARtab's counts or Cell Ranger's `molecule_info.h5`, and saves a `lineage_barcodes` artifact of each cell's molecules and barcodes, which `load_barcodes` reads as tables. `barcode_clones` calls clones from them by the published rules of LARRY, TREX, CellTagR or CellTag-multi, or by rules chosen one by one, and saves a `clones` artifact. `import_alleles` reads the alleles that lineage recorder pipelines write, such as CARLIN's and DARLIN's allele files, Cassiopeia's allele tables, PEtracer's tables and character matrices and DNA Typewriter's tape tables, and saves a `lineage_alleles` artifact of each cell's sites and their states, which `load_alleles` reads as tables and `export_characters` writes as a character matrix for Cassiopeia or LAML. `allele_clones` calls clones from the rare alleles, by DARLIN's frequency graph or CARLIN's p_clonal test against a bank of allele frequencies, and saves a `clones` artifact. `import_phylogeny` reads the cell phylogenies that lineage recorders reconstruct, from Newick and TreeData files, joins their leaves to a cell selection and saves a `cell_phylogeny` artifact, `load_phylogeny` reads it as tables of nodes and trees, and `phylogeny_clones` cuts it into a `clones` artifact at a time or a depth. {doc}`clones` lists the formats and describes the join of barcodes to cells, files from several runs, cells in several clones, the summaries and their records.

```{eval-rst}
.. autoclass:: cytearc.datastore.namespaces.ClonesAccessor
    :members: import_clones, from_column, load_clones, export_clones, expansion, load_expansion, diversity, rarefaction, overlap, track, coupling, fate_bias, load_fate_bias, import_barcodes, load_barcodes, barcode_clones, import_alleles, load_alleles, export_characters, allele_clones, import_phylogeny, load_phylogeny, phylogeny_clones
```

### `ds.repertoire`

`import_receptors` reads the T cell and B cell receptor contigs of a set of cells from Cell Ranger's contig annotations or AIRR Rearrangement files, joins them to the cells by barcode, and saves them as an `immune_receptors` artifact, which ranks each cell's chains and labels the cell with its receptor type, subtype and chain pairing. `clonotypes` groups the cells into clonotypes by identical chains, or into clonotype clusters by chains within a Hamming or Levenshtein distance, and saves a `clonotypes` artifact that labels each cell with its clonotype, which the clone summaries of `ds.clones` read as clones. `gene_usage` and `spectratype` count the genes and junction lengths of groups' chains, without saving anything. `export_airr` writes the contigs, with each cell's clonotype, as an AIRR Rearrangement file. {doc}`repertoire` describes the receptor model, the clonotype options, and the records that `load_receptors`, `load_clonotypes`, `gene_usage` and `spectratype` return.

```{eval-rst}
.. autoclass:: cytearc.datastore.namespaces.RepertoireAccessor
    :members: import_receptors, load_receptors, clonotypes, load_clonotypes, gene_usage, spectratype, export_airr
```

### `ds.variants`

`import_counts` reads the per-cell allele counts of mitochondrial and targeted variants that mgatk, maegatk, cellsnp-lite, MQuad, VarTrix, MiTo, giftwrap and the GoT tools write, joins their cells to a cell selection by barcode, and saves a `variant_counts` artifact of the sites, their alleles and each cell's depth, counts and calls. `load_counts` reads the artifact as tables of sites, alleles, cells and stored entries. `informative` scores every variant across cells with mgatk's statistics and selects the informative ones by Signac's or MAESTER's thresholds, saved as a `variant_selection` artifact that `load_informative` reads, and `frequency_clones` calls the clones that the informative variants mark from their allele frequencies, as Signac's `FindClonotypes` does. `genotypes` calls each cell's genotype at targeted sites by GoT's rules, GIFT's model or the tools' own calls, saved as a `variant_genotypes` artifact that `load_genotypes` reads, and `genotype_clones` calls the clones that the combinations of mutations define, by the likelihood of a model of allelic dropout or by exact agreement. {doc}`variants` lists the formats and describes count units, depth, the statistics, the presets and masks, the calls and methods of genotyping, and the `VariantCounts`, `VariantSelection` and `VariantGenotypes` records.

```{eval-rst}
.. autoclass:: cytearc.datastore.namespaces.VariantsAccessor
    :members: import_counts, load_counts, informative, load_informative, frequency_clones, genotypes, load_genotypes, genotype_clones
```

### `ds.imputation`

```{eval-rst}
.. autoclass:: cytearc.datastore.namespaces.ImputationAccessor
    :members: diffusion, load_diffusion, compute_imputed
```

### `ds.trajectory`

```{eval-rst}
.. autoclass:: cytearc.datastore.namespaces.TrajectoryAccessor
    :members: pseudotime, load_pseudotime, fate, load_fate, markers, load_markers, aggregation, load_aggregation, lineage_markers, load_lineage_markers, drivers, load_drivers, lineage_aggregation, load_lineage_aggregation, curves, load_curves, curves_in_layout, lineage_tree, load_lineage_tree
```

`pseudotime` orders cells by the potential of population balance analysis, which [Weinreb et al. (2018)](https://doi.org/10.1073/pnas.1714723115) define as half the pseudoinverse of the graph's random-walk Laplacian applied to the source-sink vector. The default, `method="untruncated"`, solves for that potential over all of the Laplacian's modes by MINRES on the symmetric Laplacian, to machine precision: on synthetic graphs of one million cells it took 28 to 29 s and 724 to 777 iterations in containers of 4 CPUs, also when an eighth of the cells was joined to the rest by 64 edges. `method="truncated"` keeps the `n_singular_vals - 1` smallest nonzero singular modes, 29 by default, which SciPy's `svds` finds with ARPACK; it is the potential of releases before 1.4.0 and reproduces their pseudotimes. When the graph's slow modes outnumber the modes it keeps, as on graphs of many weakly joined clusters, the truncated potential departs from the full one, to a correlation of 0.971 on a synthetic graph of 20,000 cells in 150 clusters, and its solve can run for hours on graphs of several hundred thousand cells. Both potentials drop the component of the source-sink vector along the degree vector, as the pseudoinverse does, since the range of the Laplacian is orthogonal to that vector.

`max_iterations` bounds the solve and `solver_tol` sets its tolerance, and each enters the artifact's record only when set. For the untruncated potential they are MINRES's iterations, five per cell by default, and its `rtol`, which stops the solve once the residual falls below `solver_tol` times the estimated norms of the Laplacian and of the potential; the potential moves smoothly with it, by about 1% of its range at `solver_tol=1e-8` on the graphs of one million cells. For the truncated potential they are the restarts of `svds`, ten per cell by default, and the relative tolerance of its singular values; a looser tolerance can make it miss singular modes, so leave `solver_tol` unset there unless a measurement shows the potential unchanged. The solve advances a progress bar and logs its count of operator products at `INFO` level every 30 seconds, and a solve that reaches `max_iterations` raises {py:exc}`~cytearc.trajectory.PseudotimeConvergenceError` with its counts.

The untruncated default is revision 2 of `run_pseudotime_scoring` ([Operation revisions](../developers/operation-revisions)). Pseudotimes saved by earlier releases record no method, so a default call computes them anew, the fate maps and lineage results built on them follow through their inputs, and `ds.artifacts.lineage` marks them stale. A pseudotime that records `method="truncated"` stays current and is reused.

A pseudotime scores one connected component of its graph: the largest by default, or with `component_policy="sources"` the one that holds the most source cells, the cells with a source label or a negative `ss_vec` value. `load_pseudotime` returns each cell's `component`, numbered by decreasing size, and the `component_sizes`, computed from the stored graph. A source or sink label that no selected cell holds, or no cell of the scored component, raises {py:exc}`~cytearc.trajectory.SourceSinkLabelsError` from `pseudotime` and from `fate`, naming the labels in `missing_sources` and `missing_sinks`.

```{eval-rst}
.. autoexception:: cytearc.trajectory.PseudotimeConvergenceError

.. autoexception:: cytearc.trajectory.SourceSinkLabelsError
```

The lineage methods resolve one fate map into its lineages: each scored cell
weighs in each lineage by its fate probability. `lineage_markers` and
`lineage_aggregation` follow features along each lineage, `drivers` correlates
features with fate probabilities, and `curves` fits one curve per lineage
through the graph's coordinates. `lineage_tree` is a diagnostic that proposes
candidate sinks from a cluster tree; check them before passing them to `fate`.
See [Resolve gene dynamics by lineage](../tutorials/lineage-dynamics) and
[Draw lineage curves and check candidate sinks](../tutorials/lineage-curves).

`curves_in_layout(curves, layout)` returns the curves drawn in a two-dimensional layout, such as a UMAP, as a table of each curve's points, `lineage`, `point`, `x` and `y`, which `plots.lineage_curves` draws. It averages the layout coordinates of the scored cells with each lineage's fate weights along the curve and smooths them with the curves' spline, needs no plotting dependencies and saves nothing, so a viewer of its own can draw the curves. The layout may come from any cell selection of the curves' cells.

`lineage_tree` saves its tree, so a store opened with `zarr_mode="r"` refuses to build a tree that it does not hold. {py:func}`cytearc.trajectory.build_cluster_tree` builds the same tree from the cells' coordinates and labels without a store, as a {py:class}`~cytearc.trajectory.ClusterTree`, with the clusters in the order that the saved tree gives them, so a caller can show a tree before saving it or from a read-only store.

```{eval-rst}
.. autofunction:: cytearc.trajectory.build_cluster_tree
```

### `ds.mapping`

```{eval-rst}
.. autoclass:: cytearc.datastore.namespaces.MappingAccessor
```

### `ds.compare`

```{eval-rst}
.. autoclass:: cytearc.datastore.namespaces.CompareAccessor
    :members: compute_bulk, test, load_tests
```

## Updating existing calls

Replace the method name using this table. Arguments, defaults, return values, and saved artifacts stay the same, with two later changes. The default Leiden backend of `clusters.leiden` is now `"cytearc"` instead of `"igraph"`, so pass `backend="igraph"` to keep reusing clusterings made with the earlier default (see `ds.clusters` above). `compare.compute_bulk` returns a `BulkProfiles` record instead of a frame or a tuple of frames, `count_detected=True` replaces `return_fraction=True`, and `return_n_cells` is removed because the record always holds each column's cells; the rows after `ds.make_bulk(...)` show how to read the record, which the selected analysis contracts below describe. The previous names are no longer methods on `DataStore`. Saved provenance retains its operation names, such as `run_umap`.

| Previous call | Current call |
|---|---|
| `ds.inspect_artifact(...)` | `ds.artifacts.inspect(...)` |
| `ds.lineage(...)` | `ds.artifacts.lineage(...)` |
| `ds.list_artifacts(...)` | `ds.artifacts.list(...)` |
| `ds.load_artifact(...)` | `ds.artifacts.load(...)` |
| `ds.load_cell_values(...)` | `ds.artifacts.load_values(...)` |
| `ds.metric_label_concordance(...)` | `ds.clusters.compute_concordance(...)` |
| `ds.run_leiden_clustering(...)` | `ds.clusters.leiden(...)` |
| `ds.load_paris_clustering(...)` | `ds.clusters.load_paris(...)` |
| `ds.calc_membership_strength(...)` | `ds.clusters.membership_strength(...)` |
| `ds.run_paris_clustering(...)` | `ds.clusters.paris(...)` |
| `ds.smart_label(...)` | `ds.clusters.relabel_by_overlap(...)` |
| `ds.make_bulk(...)` | `ds.compare.compute_bulk(...)` |
| `bulk = ds.compare.compute_bulk(...)` | `bulk = ds.compare.compute_bulk(...).profiles` |
| `bulk, n_cells = ds.compare.compute_bulk(..., return_n_cells=True)` | `result = ds.compare.compute_bulk(...)`, then `result.profiles` and `result.columns["n_cells"]` |
| `bulk, fraction = ds.compare.compute_bulk(..., return_fraction=True)` | `result = ds.compare.compute_bulk(..., count_detected=True)`, then `result.profiles`, `result.detected` (cells) and `result.detected_fraction` |
| `bulk["a_d1"]`, a column named by joined labels | `result.profiles[("a", "d1")]`, a column labelled by its group and sub-group |
| `ds.get_statistical_tests(...)` | `ds.compare.load_tests(...)` |
| `ds.run_statistical_testing(...)` | `ds.compare.test(...)` |
| `ds.build_embedding_initialization(...)` | `ds.embeddings.initialization(...)` |
| `ds.run_tsne(...)` | `ds.embeddings.tsne(...)` |
| `ds.run_umap(...)` | `ds.embeddings.umap(...)` |
| `ds.select_detected_features(...)` | `ds.features.detected(...)` |
| `ds.select_hvgs(...)` | `ds.features.hvgs(...)` |
| `ds.run_normalization(...)` | `ds.features.normalize(...)` |
| `ds.select_prevalent_peaks(...)` | `ds.features.prevalent_peaks(...)` |
| `ds.resolve_features(...)` | `ds.features.resolve(...)` |
| `ds.set_feature_selection(...)` | `ds.features.snapshot(...)` |
| `ds.select_all_features(...)` | `ds.features.universe(...)` |
| `ds.build_ann_index(...)` | `ds.graph.ann_index(...)` |
| `ds.build_connectivity_map(...)` | `ds.graph.connectivity(...)` |
| `ds.load_graph(...)` | `ds.graph.load(...)` |
| `ds.query_neighbors(...)` | `ds.graph.neighbors(...)` |
| `ds.get_imputed(...)` | `ds.imputation.compute_imputed(...)` |
| `ds.run_diffusion_operator(...)` | `ds.imputation.diffusion(...)` |
| `ds.load_diffusion_operator(...)` | `ds.imputation.load_diffusion(...)` |
| `ds.metric_proportional_batch_mixing(...)` | `ds.integration.compute_batch_mixing(...)` |
| `ds.metric_clisi(...)` | `ds.integration.compute_clisi(...)` |
| `ds.metric_cluster_separability(...)` | `ds.integration.compute_cluster_separability(...)` |
| `ds.metric_graph_connectivity(...)` | `ds.integration.compute_graph_connectivity(...)` |
| `ds.metric_graph_silhouette(...)` | `ds.integration.compute_graph_silhouette(...)` |
| `ds.metric_ilisi(...)` | `ds.integration.compute_ilisi(...)` |
| `ds.integrate_assays(...)` | `ds.integration.modalities(...)` |
| `ds.build_mapping_reference(...)` | `ds.mapping.build_reference(...)` |
| `ds.get_mapping_score(...)` | `ds.mapping.compute_scores(...)` |
| `ds.get_label_transfer(...)` | `ds.mapping.load_label_transfer(...)` |
| `ds.get_mapping_reference(...)` | `ds.mapping.load_reference(...)` |
| `ds.get_mapping_result(...)` | `ds.mapping.load_result(...)` |
| `ds.run_mapping(...)` | `ds.mapping.run(...)` |
| `ds.run_label_transfer(...)` | `ds.mapping.transfer_labels(...)` |
| `ds.export_markers_to_csv(...)` | `ds.markers.export_csv(...)` |
| `ds.get_markers(...)` | `ds.markers.load(...)` |
| `ds.run_marker_search(...)` | `ds.markers.search(...)` |
| `ds.auto_filter_cells(...)` | `ds.qc.auto_filter(...)` |
| `ds.run_doublet_detection(...)` | `ds.qc.doublets(...)` |
| `ds.run_feature_percentage(...)` | `ds.qc.feature_percentage(...)` |
| `ds.filter_cells(...)` | `ds.qc.filter(...)` |
| `ds.run_hto_demultiplexing(...)` | `ds.qc.hto_demultiplexing(...)` |
| `ds.run_custom_reduction(...)` | `ds.reduction.custom(...)` |
| `ds.run_harmony(...)` | `ds.reduction.harmony(...)` |
| `ds.run_lsi(...)` | `ds.reduction.lsi(...)` |
| `ds.run_pca(...)` | `ds.reduction.pca(...)` |
| `ds.run_aucell(...)` | `ds.scores.aucell(...)` |
| `ds.run_cell_cycle_scoring(...)` | `ds.scores.cell_cycle(...)` |
| `ds.get_enrichment(...)` | `ds.scores.load(...)` |
| `ds.run_waggr(...)` | `ds.scores.waggr(...)` |
| `ds.run_pseudotime_aggregation(...)` | `ds.trajectory.aggregation(...)` |
| `ds.run_fate_mapping(...)` | `ds.trajectory.fate(...)` |
| `ds.load_pseudotime_aggregation(...)` | `ds.trajectory.load_aggregation(...)` |
| `ds.load_fate_mapping(...)` | `ds.trajectory.load_fate(...)` |
| `ds.load_pseudotime_markers(...)` | `ds.trajectory.load_markers(...)` |
| `ds.load_pseudotime_scoring(...)` | `ds.trajectory.load_pseudotime(...)` |
| `ds.run_pseudotime_marker_search(...)` | `ds.trajectory.markers(...)` |
| `ds.run_pseudotime_scoring(...)` | `ds.trajectory.pseudotime(...)` |

## Store-bound plotting

`DataStore.plots` binds the datastore argument for canonical store-first functions in `cytearc.plotting`.
Array and DataFrame diagnostics such as `elbow`, `qc`, `graph_qc`, and `highly_variable_features` remain standalone.

```{eval-rst}
.. autoclass:: cytearc.datastore.plot_accessor.DataStorePlotAccessor
    :members:
```

## Selected analysis contracts

Feature producers return immutable {py:class}`~cytearc.ArtifactRef` values. Use
`features.resolve(assay, ref)` for strict read-only validation.
`features.universe(from_assay=...)` creates or reuses the canonical immutable all-feature
selection for granular workflows. It does not write a live feature metadata column.
`snapshot_cell_selection(cell_key)` captures the explicit immutable cell input for granular graph
construction. `features.normalize(cell_selection, features)` requires both exact refs. It saves the
values of the assay's `normed` with its configured `normMethod`. A `log_transform` or
`renormalize_subset` left as None is True for the RNA library-size normalizers that apply it, and
False for every other normalizer, whose scale CyteArc cannot know; a flag that the normalizer cannot
apply raises `ValueError` when True ({doc}`assays`).
Direct marker, WAGGR, AUCell, and pseudotime feature analyses likewise require exact refs.
Graph-derived methods require a graph ref and project feature selections through its named lineage edges.
They do not accept a separate feature selection.

Normalize a selection such as the highly variable genes rather than the whole feature universe.
The normalized artifact is a dense float32 matrix of every selected cell and feature, so the
27,998 genes of 3,696 cells take 414 MB before compression, and `reduction.pca` of more than 4096
features uses its randomized solver, which reads the normalized values once per pass.

```python
cells = ds.snapshot_cell_selection("I")
features = ds.features.hvgs(cells, top_n=2000, show_plot=False)
normalized = ds.features.normalize(cells, features)
diffusion = ds.imputation.diffusion(graph_ref, t=2)
imputed = ds.imputation.compute_imputed("CD4", diffusion)
membership = ds.clusters.membership_strength(cluster_ref, graph_ref)
doublets = ds.qc.doublets(cluster_ref, graph_ref)
```

`imputation.compute_imputed` starts with the features and aligns its returned array to the cell selection in the explicit diffusion-operator lineage. It accepts one name or `FeatureRef`, or a sequence, one-dimensional array, or Series of them. A string that exactly matches a live cell metadata column, including case, diffuses that column; it takes precedence over an assay feature of the same name. Other strings match assay feature names case-insensitively, and a `FeatureRef` finds a feature by name, ID, or index, as plots and `compare.test` find features. A name that several features share raises `ValueError` naming their IDs: `FeatureRef(value=feature_id, by="id")` imputes one of them, and `FeatureRef(value=name, reduction="mean")` their mean. A feature that the assay does not hold raises `KeyError`. Metadata columns need no assay, so `from_assay` is required only for features from an integrated graph's operator; a native graph's operator uses its own assay. A metadata column with a missing value in the operator's cell selection raises `ValueError`, because diffusion would spread its stored placeholder to neighbouring cells. `imputation.diffusion` raises `MemoryError` before persisting an operator that could not be formed or loaded again within the datastore memory budget. `imputation.load_diffusion(ref)` exposes the validated sparse operator for direct matrix work. Membership strength and doublet detection consume exact cluster and graph refs and return artifacts.

`compare.compute_bulk(groups, ...)` accepts a categorical cell artifact or a user-owned metadata column name.
Artifact inputs derive their cell selection from lineage; `cell_selection=` can restrict that
selection explicitly. A metadata column without `cell_selection=` snapshots the live `I` column.
`secondary_groups=` provides an optional nested grouping without writing artifact labels to a cell
column. Mean profiles fit the assay normalization once over every selected cell, so ATAC document
frequency and ADT CLR geometric means are shared by all groups. Each mean averages the group's rows
of the assay's `normed` values, so a cell without counts adds zeros and still counts toward its
group's size. A mean or floating-point sum that is not finite, such as one from a custom
normalization that returns NaN, raises `ValueError` naming the group and the feature index. RNA
means also raise `ValueError` when a selected cell's `<assay>_nCounts` total is negative or not
finite. Cells whose group or sub-group label is recorded as missing join no group, as values in
`null_vals` do.
The call returns a {py:class}`~cytearc.features.aggregation.BulkProfiles` record: `profiles`, the features-by-columns profiles; `columns`, one row per bulk column with its `n_cells` as int64, so a downstream count model can weight or drop small pseudobulk samples; and, with `count_detected=True`, `detected`, the int64 number of each column's cells whose raw count of each feature is above zero, which `detected_fraction` divides by `n_cells`. A bulk column is labelled by its group label, or by a pandas MultiIndex of the levels `group`, `subgroup` and `replicate` (numbered from 1) when the call passes `secondary_groups=` or `pseudo_reps` above 1, and `profiles`, `detected` and `columns` share these labels. Labels are never joined into text, so the groups `a_b` with sub-group `c` and `a` with sub-group `b_c` make two columns, and numeric labels stay numbers. A bulk column without cells, such as a group and sub-group combination that no cell has or a pseudo-replicate of a group with fewer cells than `pseudo_reps`, is left out rather than returned as a profile of zeros; `remove_empty_columns=False` keeps such columns, with NaN profiles, no detecting cells, and `n_cells` 0. `pseudo_reps` must be an integer of at least 1 and `feature_label` one of `"index"`, `"id"`, or `"name"`; other values raise before anything is read.
`add_grouped_assay(groups, assay_label=...)` similarly accepts a
pseudotime-aggregation ref or an explicit feature metadata column when constructing a new assay.
An aggregation ref groups only the features it clustered, and missing metadata values never form a
group. The source normalization is fitted on the cells that the source assay measured, and other
cells get zero means.

`add_grouped_assay` and `add_melded_assay` make the new assay visible only after its counts are
complete. A write that fails with an error removes the partial assay. A pending assay stays when
the write is interrupted, as by `KeyboardInterrupt`, when the process is killed, or when the final
write that publishes the assay fails, because that write is never undone. Opening ignores a pending
assay, repacking refuses it, and it blocks its name in every workspace; after confirming that no
process is writing it, remove it with `discard_interrupted_assay(assay_label)`.

The new assay has a row for every cell and measures the cells that its source assay measured. When
the source records them in its membership column `<source>_I`, the new assay's `<assay_label>_I` is
a copy, written with the assay and removed with a failed or discarded one; a source without the
column gives an assay without one, which measures every cell ([Assay membership](../developers/zarr-internals#assay-membership)). A cell
column that already has the name `<assay_label>_I` makes both methods raise `ValueError` before
they compute anything.

{py:meth}`cytearc.datastore.namespaces.QcAccessor.auto_filter` defaults to `method="mad"`
over the pooled selection. Supplying `sample_column` estimates MAD bounds separately per sample.
`n_mads` controls the bounds; groups with fewer than `min_cells_per_sample=20` active cells
are retained with a warning, including small pooled selections. Pass `method="gaussian"`
explicitly to use the former pooled Gaussian policy and configure its `min_p` and `max_p`
quantiles. Changing these probabilities with either MAD path raises an error.
{py:meth}`cytearc.datastore.namespaces.QcAccessor.doublets` scores and calls doublets the way
Scrublet does, within each sample of `sample_column`, the column of each cell's capture: doublets
are simulated from pairs of the sample's own cells and every cell is scored against its sample's
cells and simulated doublets only, so no doublet pairs cells of two captures. Each sample gets its
own automatic threshold on its simulated doublets' scores, or the given `threshold`. The
`doublet_score` artifact holds the score in `values`, the fraction of simulated doublets among a
cell's neighbors in `raw_scores`, and the calls in `predicted_doublets`; its `samples` attribute
lists each sample's threshold, calls, and estimated doublet rate
([Doublet detection](../reference/methods#doublet-method)):

```python
doublets = ds.qc.doublets(cluster_ref, graph_ref, sample_column="sample")
singlets = ds.select_cells(doublets, value="predicted_doublets", include=[False])
ds.artifacts.load(doublets).attrs["samples"]  # thresholds and calls per sample
```

`qc.filter` and `qc.auto_filter` apply the same rules as pipeline filtering. A cell whose
metric is recorded as missing in a nullable column or artifact never passes a filter and does not
inform automatic bounds. MAD filtering rejects missing sample labels among active cells. Automatic
bounds reject non-finite metric values, such as the undefined percentages of zero-count cells, and
every filter raises when no cell remains.
{py:meth}`cytearc.datastore.datastore.DataStore.select_cells` thresholds the numeric values of an
exact cell-aligned artifact, or retains categorical values with `include=[...]`, and composes the
result with its stored source selection. It reads the kind's canonical array
({ref}`cell_aligned_kinds`), such as `values` of a quality metric, `phase` of a cell-cycle
artifact, `labels` of a Paris cut, or `pseudotime` of a pseudotime artifact, or the other per-cell
array that `value=` names, such as `predicted_doublets` of a doublet score, and refuses a kind
that is not cell-aligned. The selection records the name of an array other than the canonical one.
An explicit
`cell_selection=` may narrow, but never widen, that source selection. Cells whose value the
artifact records as missing are never selected, and a selection that retains no cell raises.
Doublet detection, `markers.search`, `clusters.membership_strength`, and `clusters.relabel_by_overlap` reject label
artifacts with missing labels before reusing or writing a result. For `markers.search`, select
the labelled cells with `select_cells(labels, include=[...])` and freeze their labels with
`snapshot_cluster_labels(labels, cell_selection=...)`. For `clusters.relabel_by_overlap`, freeze both label
artifacts over one selection of the cells labelled in both. Doublet detection and
`clusters.membership_strength` need a label for every cell of the graph. Build the graph over labelled
cells only when some of its cells have no label, then freeze the labels over the cell selection
that the graph was built from, such as `run["analysis_cell_selection"]` of a pipeline run.

{py:meth}`cytearc.datastore.datastore.DataStore.snapshot_cluster_labels` freezes one label for each
cell of an explicit `cell_selection` into a datastore-scoped `cluster_labels` artifact. Label
consumers accept it like a clustering, among them `markers.search`, `compare.compute_bulk`, `select_cells`,
`compare.test`, `clusters.relabel_by_overlap`, and `clusters.compute_concordance`. The labels come from a
cell metadata column, such as an annotation or a condition, or from a cell-label artifact, such as
a clustering, a Paris cut, or imported clusters. An artifact is read for a subset of its own cell
selection, so its labels can be narrowed to the labelled cells or to some of its clusters.
Floating-point labels that are whole numbers within the int64 range, such as the float64 ids that
pandas writes for integer ids with missing values, are stored as int64, and other floating-point
labels raise `TypeError`. Every selected cell needs a label: a missing label, including a row that
a linked missing mask flags, or a blank label raises `ValueError`. Text is stored at the width of
the selected labels, and integer and boolean labels keep their dtype. The identity holds the source
column name or source artifact, the cell selection, and a fingerprint of the stored labels, so the
same labels reuse one artifact, also from a read-only store, and changed labels create a new one
while earlier snapshots keep their values. On a mounted store the
artifact is written to the target. The labels are not checked against the marker-group naming
rule; `markers.search` rejects a label that cannot name a stored marker group before it writes
anything.

For example, rank the genes that separate disease from normal cells within one annotated cell type
of a mounted store:

```python
ds = cytearc.mount_datastore(source_path, at=target_path, default_assay="RNA")
cell_type = np.asarray(ds.cells.fetch_all("cell_type"), dtype=object)
ds.cells.insert("is_t_cell", cell_type == "T cell", overwrite=True)
t_cells = ds.snapshot_cell_selection("is_t_cell")
disease = ds.snapshot_cluster_labels("disease", cell_selection=t_cells)
screen = ds.markers.search(disease, features=ds.features.universe(from_assay="RNA"))
ds.markers.load(screen, group_id="COVID-19")
```

This screen is descriptive: it treats every cell as an independent observation, although cells of
one donor are not. To test a condition, aggregate cells to donors, for example with
`compare.test(genes, CellField("disease"), cell_selection=t_cells, sample_by="donor_id")`,
as in [Compare biological conditions across donors](../tutorials/condition-comparisons).

Nullable metadata columns keep a stored placeholder in each row that their linked missing mask
flags. `cells.fetch` and `cells.fetch_all` return stored values, placeholders included.
`cells.to_pandas_dataframe`, `cells.head`, `get_cell_vals`, plots, and exports show those rows as
missing: numeric columns become float64 with `NaN`, and other columns hold a missing value.
`sift` and `multi_sift` never select them. `imputation.compute_imputed` and `cytearc.metrics.silhouette_scoring`
reject a live column with a missing value among the cells they use.

{py:meth}`cytearc.metadata.MetaData.insert` writes such a column. Values given only for the rows that
`key` selects (`"I"` by default) leave the other rows missing, holding empty text, `NaN`, 0,
`False`, or `NaT`; values for every row keep every row. Arrays, lists, and a pandas Series with a
`RangeIndex` are assigned by position. A Series with any other index is assigned by label, as pandas
aligns it: its labels must be exactly the `ids` of those rows, in any order, so annotations from a
table in another order land on the right cells; labels that are not exactly those ids, such as
repeated labels or the row positions of a frame read with `key`, raise `ValueError`, and
`values.to_numpy()` assigns them by position instead. `update_key` aligns a Series the same way.
`to_pandas_dataframe` raises `KeyError` for a column that the table does not hold, and `grep`
returns the stored names that match, so names that differ only in case stay distinct. `None`,
`NaN`, `pd.NA`, and `NaT` among object, categorical, nullable, or string values are missing too,
and those values are typed as imports type them: bool, int64, float64, complex128, or else text;
object integers outside the int64 range raise `ValueError`. An explicit `fill_value` is stored as a real value without a mask and must
fit the values: text for text, which widens the column, a bool for a Boolean column, an integer
within range for integers, a real number that is not a bool for floating-point values, and a
datetime without a time zone or a timedelta that the unit of datetime or timedelta values holds
exactly; anything else raises `ValueError` before the store changes. `I`, `ids`, and `names` never
hold a missing value. A boolean column with masked rows selects the same cells as one filled with
`False`, so selections snapshotted from it keep their identity, while metadata snapshots and
filters over it record its mask. Replacing a column reads the previous column, its attributes, and
its mask first; if writing the new column fails or is interrupted, the previous one is written back
before the error propagates. When that also fails, a note on the error names the column.

Pandas categorical values, a `pd.Categorical` or a Series of categorical dtype, are stored as a categorical column: each row's code, -1 for a missing row, in the smallest signed integer type that holds it, beside the categories, every category in its order with whether they are ordered ([Categorical columns](../developers/zarr-internals#categorical-columns)). The categories are typed together, as other values are, so categories that read alike once stored, such as `1` and `"1"`, raise `ValueError`, and an explicit `fill_value` must be one of them. `I`, `ids` and `names` store the labels of categorical values. Every read returns the labels of either layout: `fetch`, `fetch_all`, `to_pandas_dataframe`, `artifacts.load_values` of a `CellField`, groupings, plots and exports read a categorical column as they read a text column of the same labels, and the fingerprints that results record of those labels are the same, apart from the stamp of stored chunks that `clones.from_column` records. {py:meth}`cytearc.metadata.MetaData.fetch_categorical` returns a column's labels for the rows that `key` selects, or for every row with `key=None`, as a pandas `Categorical`: the codes and categories of a categorical column as stored, unused categories included, and for another column its distinct values as unordered categories, numbers ascending and text in the natural order of plots.

```python
ds.cells.insert("cell_type", pd.Categorical(labels, categories=["T cell", "B cell", "NK cell"]))
cell_type = ds.cells.fetch_categorical("cell_type", key=None)
cell_type.codes, cell_type.categories
```

The cell table reserves the membership column `<assay>_I` of every assay of the store, which only
imports, merges, and derived assays write ([Assay membership](../developers/zarr-internals#assay-membership)). `insert`, with or without
`overwrite`, `update_key`, and `reset_key` raise `ValueError` for that name whether or not the
column exists, and `drop` raises for a column that carries the membership role, because a store
without it counts every cell as measured. A plain column of that name without the role can be
dropped.

A cell that an assay did not measure holds zero counts of the assay, which are no measurement, so
an operation that reads the assay's values refuses it. Normalization, `integration.modalities`, HVG and
detected-feature selection, WAGGR, AUCell, marker search, `compare.compute_bulk`, the feature keys of
`compare.test`, prevalent peaks, cell-cycle scoring, feature percentages, HTO
demultiplexing, doublet detection, pseudotime markers and aggregation, the feature names of
`imputation.compute_imputed`, `mapping.run` of the query assay, and `to_anndata(matrix="normed")`
without a run check their cells against the membership column after their own argument and
assay-type checks and before they reuse or write anything, on a read-only store too, and raise
{py:exc}`~cytearc.metadata.membership.UnmeasuredCellsError`, a `ValueError` that names the
operation, the assay, the column, and how many of the selected cells the assay did not measure.
The pipeline recipes `pipeline.rna`, `pipeline.atac`, and `pipeline.adt` check their cells the
same way before they write anything; as recipes write, they first need a store opened with
`zarr_mode="r+"`.
`compare.compute_bulk` checks the cells whose counts it reads: the cells of its bulk columns, or every selected
cell for a mean that fits the assay's normalization over them, as ATAC and ADT means do; for a
metadata grouping without `cell_selection`, it finds them from the live `I` column before it
snapshots `I`. Quality control reads the assay of each metric that it filters on:
`qc.auto_filter`, `qc.filter` on such a column, and the filtering stage of a pipeline recipe
refuse unmeasured cells too. A cell
column is a metric of the assay whose preparation wrote it, `<assay>_nCounts`,
`<assay>_nFeatures`, or a percentage column that the assay records in its `percentFeatures`
attribute, such as `RNA_percentMito`; a `quality_metric` artifact of an assay is a metric of that
assay; other columns read no assay. Operations that read only other results or metadata (such as
PCA, graphs, clustering, and `snapshot_cluster_labels`) and derived assays do not check. Exports
that declare membership, `to_h5ad` and `to_anndata` without layers, keep unmeasured cells; `to_mtx`
and a `to_anndata` layer of another assay cannot declare it and refuse them.
{py:meth}`~cytearc.datastore.datastore.DataStore.select_measured_cells` keeps the measured cells of
a selection: pass its result as `cell_selection`, which `compare.compute_bulk`, `compare.test`,
and the QC filters also take directly, build graphs over it, or freeze labels over it with
`snapshot_cluster_labels(labels, cell_selection=...)`. A pipeline recipe takes that selection as
its `cells`, or a boolean column. Normalized export takes a `cell_key` column instead. Either
column must be True only for the cells of `I` that the assay measured, such as
`ds.cells.insert("RNA_measured", ds.cells.fetch_all("I") & ds.cells.fetch_all("RNA_I"))`; the
membership column alone also holds cells that a filter removed from `I`. `select_measured_cells`
returns the input selection itself when the assay measured every selected cell, so fully measured
stores keep their identities. Plots and `get_cell_vals` show the feature values of unmeasured cells
as missing, NaN, instead of zero, and normalize only the measured cells, so that a normalizer
fitted over the cells it reads, such as ATAC TF-IDF, gives a measured cell the value of a read of
the measured cells alone. Opening saved results does not rerun membership checks; a pipeline
recipe whose `cells` include unmeasured cells raises before it creates a run record.

```{eval-rst}
.. autoexception:: cytearc.metadata.membership.UnmeasuredCellsError
```

Saved {py:meth}`cytearc.datastore.namespaces.MarkersAccessor.search` calls return the exact immutable marker-table reference.
Pass that reference as `markers.load(marker=ref)` to select the exact feature-specific result.
Marker tables report `group_id` as a string and list groups in the same order as plot
categories: decimal labels first in numeric order, so `"2"` precedes `"10"`, then other labels in
natural order, so `"1_2"` precedes `"1_10"` and `"2_T"` precedes `"B cell"`.
`markers.export_csv` uses the same column order. An unknown `group_id` raises an error.
Fresh marker results include score, expression fractions, fold change, AUC, two-sided Mann-Whitney p-values, and Benjamini-Hochberg values adjusted within each one-versus-rest group over tested features.
These are cell-level marker statistics, not replicate-aware differential expression.
Statistics are stored at full float64 precision. `neg_log10_p_value` holds `-log10` of the
p-value; for the normal approximation it is computed in log space, so it stays finite and keeps
its order where `p_value` underflows to 0, near a z statistic of 38. A group whose cells can be
split from the other cells in at most 100,000 ways, such as a group of two cells among at most 447,
gets exact p-values from the permutation null of its rank sum, ties included, as `compare.test`
does; other groups keep the tie- and continuity-corrected normal approximation. A group of fewer
than two cells cannot be compared with the rest: `markers.search` skips it with a warning that
names it, records it in the table's `skipped_groups` metadata, keeps its cells in the rest of every
other group, and `markers.load` gives it no rows. A clustering without any group of at least two
cells and at least two other cells raises `ValueError` naming the groups.

Each row of `markers.load` names its feature by `feature_name` and by `feature_id`, the ID that the marker table froze, and gives the feature's index in the assay as `feature_index`. The rows of two features that share a name, such as the two MATR3 genes of the 10x Genomics human reference, `ENSG00000280987` and `ENSG00000015479`, stay apart by their IDs. `markers.export_csv` writes names only.

A marker table of operation revision 1, written before release 1.3.0, records neither
`neg_log10_p_value` nor `p_value_policy`. `markers.load`, `markers.export_csv`, and
`marker_heatmap` still read it, from read-only stores too, under an explicit policy: its
statistics keep the five-decimal rounding they were stored with, except for the p-values, which
come from the tie- and continuity-corrected normal approximation for every group, the p-value
policy that the readers report as `asymptotic_tie_and_continuity_corrected`. Its
`neg_log10_p_value` is derived as `-log10(p_value)`, and is `inf` where `p_value` underflowed to
0. The filters and `sort_by` apply as to any table. `markers.search` never reuses such a table: on
a writable store it recomputes it at revision 2, and on a read-only store it raises
`PermissionError`. A table without `p_value_policy` that does not have exactly the revision 1
columns, or that records `skipped_groups`, raises `ValueError`.

`markers.load` and `markers.export_csv` keep a row that passes every filter that is not `None`:
`min_frac_exp=0.2`, `min_auc=0.6`, and `max_p_adj=0.05` by default, on `frac_exp`, `auc`, and
`p_value_adjusted`. `min_score` defaults to `None`. `score` is a group's share of the mean dense
ranks summed over all groups, so a gene that is uniform across groups scores `1/n_groups`, and a
gene with distinct values in the `n` cells of one group and zero elsewhere scores
`(n + 3) / (n + 1 + 2 * n_groups)`. A fixed score threshold therefore hides perfect markers of
small groups and of fine partitions:
LILRA4 in the plasmacytoid dendritic cells of a 60-group partition of 1,000 PBMCs has an AUC of 1.0
and a score of 0.125. Pass `None` for every filter to read every tested feature, negative markers
included. `sort_by` orders the rows of each group: `"score"` (the default), `"auc"`, or
`"fold_change"` descending, or `"p_value"` ascending. Ties fall to the p-value, with
`neg_log10_p_value` ordering p-values that underflow to 0, except that ties in `"p_value"` fall to
the AUC so that positive markers come first; then to the feature name and index.
`fold_change` is the group's `mean` divided by `mean_rest`, the means of the values that the
search ranks. It is `inf` for a feature that no other cell expresses, so such features sort above
every finite ratio, and `NaN` when both means are 0 or either is negative, as signed counts can
give. Every other statistic is finite, except the derived `neg_log10_p_value` of a revision 1
table. The ranked values are the assay's normalized values, such
as library-size normalized RNA counts, log-scale with `log_transform=True`, CLR values of ADT
counts, which are log-scale too, or TF-IDF values of ATAC counts. No pseudocount is added, and
fold changes compare only within one assay and normalization. Marker tables record this policy in
their `fold_change_policy` metadata. `markers.load`, `markers.export_csv`, and `marker_heatmap`
raise `ValueError` for tables missing that policy; recompute them with `markers.search`.
Marker search reads raw counts of any storage dtype. Library-size markers require finite
non-negative counts and cell totals: a negative or non-finite normalized value of a tested feature,
or total of a selected cell (its `<assay>_nCounts` value, or with `renormalize_subset=True` its sum
over the tested features), raises `ValueError` before anything is written.
Every marker, pseudotime-marker, and pseudotime-aggregation search ranks or correlates the values
of the assay's `normed` with its configured `normMethod`. Their `log_transform` and
`renormalize_subset` default to False; `log_transform=True` takes `log1p` of the normalizer's own
output, and a flag that the normalizer cannot apply, such as `log_transform=True` with ADT CLR or
ATAC TF-IDF, raises `ValueError` ({doc}`assays`). The artifact records the resolved flags.

{py:meth}`cytearc.datastore.namespaces.TrajectoryAccessor.markers` leaves untested features with `r_value` 0.0 and `NaN` for `p_value` and `p_value_adjusted`, and adjusts p-values over tested features only.

{py:meth}`cytearc.datastore.namespaces.CompareAccessor.test` computes Mann-Whitney
p-values from the exact permutation null, ties included, when the two groups can be formed in at
most 100,000 ways, as in typical sample-level designs. Larger designs keep the marker-search normal
approximation. `StatisticalTestResult.p_value_method` records `"exact"` or `"asymptotic"`, and a
saved Mann-Whitney result without that record must be recomputed. A `StudyDesign` pairing column
applies only to the paired Wilcoxon test; for subjects nested within conditions, pass the subject
column as `sample_by`. Writing a result requires `zarr_mode='r+'` and is checked before any value
is computed, while a matching saved result is still reused from a read-only store.

{py:meth}`cytearc.datastore.namespaces.ScoresAccessor.waggr` and
{py:meth}`cytearc.datastore.namespaces.ScoresAccessor.aucell` match network targets to active feature
names without case sensitivity. A target that matches several active features, such as a gene
symbol shared by two feature ids, is ambiguous. The default `ambiguous_targets="drop"` removes its
edges before `tmin` pruning, logs a warning, and records the target in the artifact's
`dropped_ambiguous_targets` attribute; `ambiguous_targets="error"` raises instead. Both producers
require `zarr_mode='r+'` and an intact prepared dataset identity.

{py:meth}`cytearc.datastore.namespaces.TrajectoryAccessor.aggregation` aggregates and clusters
features before it starts an artifact. It raises `ValueError`, and leaves no incomplete artifact,
when too few features pass `min_exp` or when `n_clusters` is greater than one but smaller than the
number of disconnected components of the feature neighbour graph. Modules tied at the cut height are merged so that exactly
`n_clusters` modules are returned.
{py:meth}`cytearc.datastore.namespaces.TrajectoryAccessor.fate` treats `solver_tol` as an absolute
bound on the largest Bellman residual of each solved sink column, so large sink groups do not
loosen the solve.

{py:meth}`~cytearc.datastore.namespaces.IntegrationAccessor.modalities` is the public SNN/WNN graph integration entry point.
WNN requires two or more cell-aligned assays and stores one per-cell weight for each assay.
SNN also requires two or more explicit connectivity-map refs.
`ds.pipeline.multimodal(runs)` runs this integration over completed modality runs of the same
cells and records a durable run ({doc}`pipeline`).
The recommended workflow is in [Read RNA and protein together with CITE-seq](../tutorials/cite-seq); method comparison and diagnostics are in [Diagnose multimodal integration](../tutorials/multimodal-diagnostics).

## Analysis results

```{eval-rst}
.. autoclass:: cytearc.FateMappingResult
    :members:

.. autoclass:: cytearc.PseudotimeScoreResult
    :members:

.. autoclass:: cytearc.PseudotimeMarkerResult
    :members:

.. autoclass:: cytearc.PseudotimeAggregationResult
    :members:

.. autoclass:: cytearc.trajectory.LineageMarkerResult
    :members:

.. autoclass:: cytearc.trajectory.LineageDriverResult
    :members:

.. autoclass:: cytearc.trajectory.LineageAggregationResult
    :members:

.. autoclass:: cytearc.trajectory.LineageCurvesResult
    :members:

.. autoclass:: cytearc.trajectory.CurveProjection
    :members:

.. autoclass:: cytearc.trajectory.LineageTreeResult
    :members:

.. autoclass:: cytearc.trajectory.ClusterTree
    :members:

.. autoclass:: cytearc.EnrichmentResult
    :members:

.. autoclass:: cytearc.regulons.GrnResult
    :members:

.. autoclass:: cytearc.features.statistical.StatisticalTestResult
    :members:

.. autoclass:: cytearc.features.aggregation.BulkProfiles
    :members:

.. autoclass:: cytearc.clustering.ParisClusteringResult
    :members:

.. autoclass:: cytearc.clustering.ParisClusterDiagnostic
    :members:

.. autoclass:: cytearc.clustering.connectivity.ClusterConnectivity
    :members:

.. autoclass:: cytearc.quality_control.cell_calling.EmptyDropsResult
    :members:

.. autoclass:: cytearc.quality_control.cell_calling.BarcodeRanks
    :members:
```

## Inheritance appendix

These implementation classes are not intended for analysts to construct directly.

```{eval-rst}
.. autoclass:: cytearc.datastore.base_datastore.BaseDataStore
    :no-index:
```

```{eval-rst}
.. autoclass:: cytearc.datastore.graph_datastore.GraphDataStore
    :no-index:
```

```{eval-rst}
.. autoclass:: cytearc.datastore.mapping_datastore.MappingDatastore
    :no-index:
```
