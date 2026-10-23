---
myst:
  html_meta:
    description: Work with CyteArc assay types, normalized count blocks, cell and feature metadata, and metadata selections.
---

# Assays and metadata API reference

Each assay opens as the class of the preset that its store declares in `assayTypes`, such as
`RNA`, `ATAC`, `ADT`, `HTO`, `GeneActivity`, or `Assay` for a generic assay; see
[Assay types](../developers/zarr-internals#assay-types-attribute). `DataStore(assay_types=...)` accepts only these presets and only
names of assays in the store, and a read-only open accepts only the types that the store records.
An open assay carries the preset that its `DataStore` resolved as `assayType`, which merge,
subset, HTO demultiplexing, and `DataStore.summary()` read. An assay that measured only some cells
of its store marks them in the Boolean cell column `<assay>_I` ([Assay membership](../developers/zarr-internals#assay-membership)), which
`MetaData` refuses to write or, while it carries the membership role, to drop.

`Assay.score_features(feature_names, cell_key, ctrl_size, n_bins, rand_seed)` is computation-only.
It computes full-row-order averages blockwise in memory and is safe on read-only counts; it does
not plan artifacts or write metadata. With `log_transform=True` it scores the values of
`normed(log_transform=True)`, so only normalizers that take `log_transform` accept it (see the
table below). Persistent cell-cycle outputs belong to `DataStore.scores.cell_cycle`.

Feature-count percentages belong to
`DataStore.qc.feature_percentage(cell_selection, features)`. It derives the assay from the exact
feature-selection ref and returns an assay-scoped `quality_metric` ref whose `values` array is the
per-cell percentage. It does not add a metadata column. A read-only `DataStore` returns an
identical existing result but raises `PermissionError` instead of computing a new one.

Persisted normalization belongs to
`DataStore.features.normalize(cell_selection, features)`, which returns an immutable artifact ref.
It saves the values of `normed` with the configured normalizer. A flag left as None takes the
operation's default only for the RNA library-size normalizers `norm_lib_size` and
`norm_lib_size_log`, whose scale CyteArc knows, as
`cytearc.assay.normalization.default_normalization_flags` reports: `features.normalize` applies the
flags they take by default. The values of every other normalizer, `norm_dummy`, CLR, TF-IDF, and
custom normalizers, may be signed, already logged, or computed over chosen totals, so their flags
stay False unless passed. The artifact records the resolved flags. Assay normalization methods are
computation-only and require explicit feature indexes where they do not accept an `ArtifactRef`.

Every normalized path applies the assay's configured `normMethod`: `normed`, saved normalization,
HVG statistics, feature summaries, scores, marker search, pseudotime markers and aggregations,
and marker heatmaps. Paths that compute library-size values straight from the counts, such as the
subset writer of `features.normalize` and the RNA feature streams, run only when
`cytearc.assay.normalization.uses_library_size_normalization(assay)` holds: the normalizer is
`norm_lib_size` and `sf` is set.

The two normalization flags have one meaning on every path. `log_transform=True` takes `log1p` of
the configured normalizer's own output, in float64. `renormalize_subset=True` hands the normalizer
each cell's total over the selected features instead of over every feature. A normalizer accepts
only the flags it can apply, as `cytearc.assay.normalization.applicable_normalization_flags` reports;
a flag it cannot apply defaults to False, and passing True raises `ValueError`:

| Assay and normalizer | `log_transform` | `renormalize_subset` |
|---|---|---|
| RNA `norm_lib_size` (default) and custom normalizers | yes | yes, through `assay.scalar` |
| RNA `norm_lib_size_log` | no, its values are already logarithms | yes |
| `norm_dummy` (the generic default) of RNA, ADT, and generic assays, and custom ADT and generic normalizers | yes | no |
| ATAC `norm_tf_idf` (default) and custom normalizers | no, ATAC values are never logged | yes, as the term-frequency denominator |
| `norm_clr` (the ADT default) on any assay, built-in normalizers of another assay class (`norm_tf_idf` outside ATAC, library-size normalizers outside RNA), and `norm_dummy` of ATAC | no | no |

A custom RNA normalizer is a callable `(assay, counts)` that may read each cell's total from
`assay.scalar` while it builds its result; `normed(log_transform=True)` returns the `log1p` of what
it returns. Saved results need a module-level function or a callable with an `artifact_identity`
attribute, which provenance records. Pass `log_transform=True` to take logarithms of the counts
that `norm_dummy` keeps as they are.

Provenance identifies a module-level function defined outside CyteArc, such as one defined in a
notebook, by its module, its qualified name, and `code`, a digest of its bytecode, constants, names,
and default values. Editing the function, one of its constants, or a default gives saved results a
new identity, so they are computed again rather than reused; the same function moved elsewhere in
its file or defined again with the same code keeps its identity. Bytecode differs between Python
versions, so another Python version also computes such results again. The digest does not cover the
values of the globals that the function reads or the functions that it calls: give a normalizer
whose behavior depends on them an `artifact_identity` that changes with them. A lambda, a nested
function or closure, a method, a `functools.partial`, a callable object, and a function with a
default value that provenance cannot record, such as an arbitrary object, raise `ValueError` unless
they define `artifact_identity`. CyteArc's own normalizers are identified by name; operation
revisions record changes to them.

Releases before 1.3.0 identified such a function by name alone, so its results did not record its
code. Operation revision 2 of each operation that records a normalizer (`run_normalization`,
`summarize_rna_features`, `summarize_atac_features`, `run_marker_search`,
`run_pseudotime_marker_search`, `run_pseudotime_aggregation`, and `run_statistical_testing`)
applies to the results that record a normalizer defined outside CyteArc: their earlier results are
reported stale, and planning logs one line when it computes one of them again.

RNA library-size normalization divides each cell by its total, the `<assay>_nCounts` value or, with
`renormalize_subset=True`, its sum over the selected features, and a cell whose total is zero
normalizes to zeros. `cytearc.assay.normalization.library_size_divisors` holds this rule for every
library-size path: `normed` with `norm_lib_size` or `norm_lib_size_log`, normalization artifacts,
HVG statistics, feature-group scores, marker search, `compare.compute_bulk` means, WAGGR, and mapping and
doublet projections. Each raises `ValueError` naming the totals when a total of a selected cell is
negative or not finite. `normed` with a custom `normMethod`, which need not read the totals, does
not check them and replaces a zero total with 1.

```{eval-rst}
.. autoclass:: cytearc.assay.Assay
    :members:
```

```{eval-rst}
.. autoclass:: cytearc.assay.RNAassay
    :members:
```

```{eval-rst}
.. autoclass:: cytearc.assay.ATACassay
    :members:
```

```{eval-rst}
.. autoclass:: cytearc.assay.ADTassay
    :members:
```

```{eval-rst}
.. autofunction:: cytearc.assay.norm_dummy
```

```{eval-rst}
.. autoclass:: cytearc.metadata.MetaData
    :members:
```
