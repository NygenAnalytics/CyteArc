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

Decide with this ladder, in order:

1. The change only moves results by last-bit or reassociation noise, and old results stay equally
   valid: add no revision. Record the change under {ref}`stable_identity_result_changes`.
2. Results become materially different, or were wrong, for an identifiable subset of artifacts:
   add a scoped revision whose `applies` selects that subset from recorded provenance.
3. Otherwise: add a whole-operation revision with `applies=None`.
4. A new reuse requirement that stops existing artifacts from being reused, such as a new required
   array or attribute or a stricter reuse validator, also adds a revision, so that the log explains
   the recomputation.

Do not change recorded parameters or inputs to invalidate results. That changes the identity of
every artifact the operation writes without telling anyone why its stored results stopped
matching. Use the operation revision registry for these changes.

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

(stable_identity_result_changes)=
## Results that change while identities stay the same

A change that moves results only by rounding or reassociation noise needs no operation revision
when existing results remain valid. Record the affected operations, the release, and the reason
reuse remains valid here. A user can request recomputation with `invalidate_cache=True`.
