---
description: Learn when CyteArc computation changes require an operation revision and how revisions affect artifact reuse and lineage.
---

(operation_revisions)=
# Operation revisions

This page is for contributors who change what an analysis operation computes. It explains how
CyteArc stops reusing stored results that a fix makes wrong or outdated, and when a change needs
that.

## Why revisions exist

An artifact's identity is its scope, assay, kind, and canonical provenance: the producing
operation, its parameters, and its inputs. Planning reuses a complete artifact only when its
canonical provenance matches the request exactly. Artifact IDs are random, so an input recorded
in provenance is a specific stored result: recomputing an upstream artifact gives every
downstream artifact a new identity.

Exact reuse also means that a fix that changes the output of an operation, for unchanged
parameters and inputs, would keep returning results of the old code. The operation revision
registry in `cytearc/storage/operation_revisions.py` is the one explicit way to stop that. The
CyteArc version recorded in each artifact's `cytearc_version` attribute is diagnostic only and never
affects reuse.

## How a revision changes reuse

An operation with revisions has an ordered tuple of `OperationRevision` entries; an operation
that is not listed is at revision 1. Revision 1 is everything the operation computed before its
first entry; entries are numbered 2, 3, and so on. An entry has:

- `revision`: its number.
- `release`: the first CyteArc release that records it.
- `change`: one line that says what changed, shown in logs and lineage reports. A scoped
  revision also says which results it affects.
- `applies`: `None` when the change affects every artifact of the operation, or a predicate
  `(kind, parameters, inputs) -> bool` that selects the affected artifacts.
- `added_diagnostics`: the diagnostics that the operation saves from this revision on, empty unless the revision added some ({ref}`diagnostics_added_after_a_release`).

When CyteArc plans an artifact, `effective_revision` returns the highest revision whose `applies`
holds for the artifact's kind and serialized parameters and inputs, or 1. Provenance records it
as `"revision": n` only when it is 2 or more. An operation without revisions omits the field.
Adding a revision changes only the identities of the artifacts it applies to.

Reuse stays exact. An artifact that records no revision is revision 1, so it is reused only while
the effective revision for its parameters is 1. During the same scan, a complete artifact whose
provenance differs from the request only in its revision is a superseded match:

- it is never reused, and when no artifact matches exactly, planning logs one INFO line for the
  newest one, such as
  `Recomputing <operation>: artifact 1a2b3c4d5e6f is revision 1, current 2: <change>`;
- it stays listable, loadable, and traceable. `ArtifactStatus.revision`, `current_revision`,
  `is_current`, and `superseded_by` describe it, and `ds.artifacts.lineage` marks it `stale`. Nothing
  persisted changes.

A revision also reaches the results built from a superseded artifact, through their inputs rather
than their own revisions. A UMAP or a Leiden clustering built on a revision 1 connectivity map
records the current revision of its own operation, so `ArtifactStatus.is_current`, which judges
only an artifact's own revision, is True for it. Its input is never reused, though: it is computed
again with a new reference, which gives the result new inputs, so the result is computed again
too.

An artifact that records a revision newer than the running release knows is not reused either,
and its status is not current. A user may still pass a superseded artifact explicitly as an input;
CyteArc accepts it without a warning, and lineage shows it as stale.

## When a change needs a revision

An operation that no release has shipped takes no revisions: until a release records it, a change to its name, parameters, inputs, stored arrays or the diagnostics it declares makes the next call compute its artifacts anew, its loader refuses an artifact of the earlier form with a message that names the call which saves it anew, and the first release ships it at revision 1.

Decide with this ladder, in order:

1. The change only moves results by last-bit or reassociation noise, or breaks exact ties in a stated order where the earlier code left their order unspecified, and old results stay equally valid: add no revision. Each earlier result of a tie-breaking change was one of the equally valid outputs, so it stays valid and is reused. Record the change under {ref}`stable_identity_result_changes`.
2. Results become materially different, or were wrong, for an identifiable subset of artifacts:
   add a scoped revision whose `applies` selects that subset from recorded provenance.
3. Otherwise: add a whole-operation revision with `applies=None`.
4. A new reuse requirement that stops existing artifacts from being reused, such as a new required array or attribute, a new diagnostic ({ref}`diagnostics_added_after_a_release`), or a stricter reuse validator, also adds a revision, so that the log explains the recomputation.

Do not change recorded parameters or inputs to invalidate results. That changes the identity of
every artifact the operation writes without telling anyone why its stored results stopped
matching. Use the operation revision registry for these changes.

When a new default replaces a computation that callers may still ask for, keep the earlier computation reachable through an argument that the record names, and scope the revision to the records that do not name it. Records of earlier releases name nothing, so the revision supersedes them, while an artifact that asks for the earlier computation by name stays current and is reused. Revision 2 of `run_pseudotime_scoring` works this way: the untruncated potential became the default, and its predicate selects the pseudotimes that do not record `method="truncated"`.

(diagnostics_added_after_a_release)=
## Diagnostics added after a release

Every artifact saves the diagnostics that its operation declares, and reuse requires them. A diagnostic added to an operation that a release has shipped therefore needs a revision that applies to every artifact, with `applies=None`, and lists the diagnostic in `added_diagnostics`. Every artifact computed from that revision on records it or a later revision and saves the diagnostic, so planning computes each artifact of an earlier revision anew and logs why.

An artifact that records an earlier revision and lacks the diagnostic reads it as None, not measured. `ArtifactStatus.diagnostics`, which `ds.artifacts.inspect(ref)` and `require_complete_artifact` read, applies this rule, so such an artifact still serves the calls that pass it explicitly and the runs that reference it. Reading still refuses an artifact that lacks any other declared diagnostic, one that holds a key its operation does not declare, and one without diagnostics, as releases before 1.4.0 saved them. It also refuses an artifact that records the adding revision or a later one and lacks the diagnostic: every such artifact was saved with it, so only damage or a bug can have lost it.

The registry refuses a scoped revision that adds diagnostics, and `declare_diagnostics` refuses the declaration of an operation when a revision of the operation adds a diagnostic that the declaration lacks. Revision 3 of `select_hvgs` added `selectedDetectionRatio`, so a selection that CyteArc 1.4.0 saved reads the ratio as None.

## Rules for entries

- The registry is append-only. A released entry is never edited, renumbered, or removed, and its
  predicate never changes, because every later release judges stored artifacts against it.
- A predicate must be pure and total over recorded provenance. It receives serialized values:
  lists rather than tuples, artifact inputs as reference mappings, and only the keys that the
  release that wrote the artifact recorded. It must return a bool for any such record, including
  records written before the parameters it reads existed, and must not raise.

## Add a revision

1. Append the entry to the operation's tuple in `cytearc/storage/operation_revisions.py`, adding the
   operation if it has no revisions yet, with the next number and the release that will ship it.
2. Add a test that a new artifact of the operation records the revision, as
   `ArtifactStatus.revision`, and that an artifact of the earlier revision is recomputed, not
   reused. For a scoped revision, also test an artifact that its predicate leaves out.
3. For a revision that adds diagnostics, also test that an artifact of the earlier revision reads them as None and that an artifact of the new revision without them is refused.

(stable_identity_result_changes)=
## Results that change while identities stay the same

A change that moves results only by rounding or reassociation noise, or that breaks exact ties in a stated order where the earlier code left their order unspecified, needs no operation revision when existing results remain valid: an earlier result of a tie-breaking change is one of the equally valid outputs, so it is reused. Record the affected operations, the release, and the reason reuse remains valid here. A user can request recomputation with `invalidate_cache=True`.

- `run_feature_percentage`, 1.3.0: the denominator is the cell total that preparation stored as
  `<assay>_nCounts`, a sequential float64 sum of the cell's counts, instead of a pairwise float64
  sum of the counts read again. Integer counts sum exactly either way, so their percentages are
  bit-identical. Counts with fractional values can differ in the last bits of the total, which
  leaves every stored percentage equally valid.
- 1.3.0, `run_pca`: a fit of a single row block used IncrementalPCA's dense SVD, and every fit
  of up to 4096 features now uses the Gram solver, which fits of several blocks with at least as
  many rows as features used before. Both are exact, and the Gram solver now computes the `dims`
  leading eigenpairs instead of one more, so results move by rounding, and by float32 precision
  for unscaled single-block fits, which IncrementalPCA computed in float32. A single-block fit of
  more than 4096 features was exact too; such fits now use the randomized solver and record
  `solver="randomized"`, so a new fit never reuses them. Revision 2 marks the inexact
  IncrementalPCA fits over several blocks.
- `compare.compute_bulk` and `regulons.profiles`, 1.3.0: an assay read from `counts`, such as ATAC,
  ADT, or RNA with another normalizer, sums and counts every bulk column in one pass over its raw
  rows, and adds floating-point counts in float64 in row order within each column instead of one
  block of stored rows at a time. Integer counts, including integer-valued floating-point counts,
  sum exactly either way, and expressing fractions are unchanged; fractional floating-point counts
  can differ in the last bits. Neither operation stores its result, so no stored result is reused.
- `select_prevalent_peaks` and `integrate_assays` with `method="snn"`, 1.5.0: ties are broken in a stated order, with a stable sort, where the default sort left their order to the platform's sort routine. Peaks whose prevalence ties at the `top_n` cutoff are kept in feature order, and SNN candidates whose ranks tie are taken from the last candidate of a cell's row back, the last graph's first. Each earlier selection and integration broke such ties one of the equally valid ways, so it stays valid and is reused, and a new computation, which breaks ties the same way on every platform, can keep other peaks or neighbors among tied values than one saved before. A tie needs equal values: prevalences computed on different platforms can still differ in the last bit, since TF-IDF takes logarithms, so near-ties can still be ranked differently.
- `integrate_assays` with `method="wnn"`, 1.5.0: the distance of a cell from each modality's prediction sums its squared offsets in NumPy's pairwise order, as its distances to its candidates do, instead of taking a BLAS dot product, whose summation order depends on the BLAS library and, with Apple's Accelerate, on where each vector sits in memory. The two sums differ only by rounding, since a BLAS library may fuse each multiplication into its addition and float16 coordinates took NumPy's own float16 dot product, so modality and graph weights move by rounding, and candidates whose affinities nearly tie can trade places; stored integrations remain equally valid. The distances no longer depend on the BLAS library or on memory addresses.
