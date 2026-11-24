"""Operation revisions, the one explicit way to stop reusing stored results.

An artifact's identity is its scope, assay, kind, and canonical provenance:
the producing operation, its parameters, and its inputs. Reuse requires an
exact provenance match, so a fix that changes what an operation computes for
unchanged provenance must also change provenance, or stored results of the
old code would keep being reused. This module records each such fix as an
:class:`OperationRevision`.

Planning computes the effective revision of a new artifact with
:func:`effective_revision` and records it in provenance only when it is 2 or
more. A revision changes only the identities of the artifacts it applies to.
Artifacts that record no revision are revision 1. A revision after which the
operation saves new diagnostics lists them, so that artifacts of earlier
revisions, saved without them, still read.

The registry is append-only. A released :class:`OperationRevision` is never
edited, renumbered, or removed, and its predicate never changes: stored
artifacts are judged against it by every later release. A new release that
changes the same results again adds the next revision.

See ``docs/source/developers/operation_revisions.md`` for when a change needs
a revision and how to add one.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

type RevisionPredicate = Callable[[str, Mapping[str, Any], Mapping[str, Any]], bool]
"""Predicate over an artifact's kind and its recorded parameters and inputs."""


@dataclass(frozen=True, slots=True)
class OperationRevision:
    """One released change to what an operation computes.

    Attributes:
        revision: The revision number; an operation's first change is 2.
        release: The first CyteArc release that records it, such as ``"1.0.0"``.
        change: One line that says what changed.
        applies: A pure predicate that selects the affected artifacts, or None for all.
        added_diagnostics: The diagnostics that the operation saves from this
            revision on. An artifact of an earlier revision that lacks one
            reads it as None, not measured. Only a revision that applies to
            every artifact adds diagnostics.
    """

    revision: int
    release: str
    change: str
    applies: RevisionPredicate | None
    added_diagnostics: tuple[str, ...] = ()


def _repeats_cell_cycle_gene(
    kind: str,
    parameters: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> bool:
    """Select cell-cycle scores whose recorded gene indices repeat an index.

    Only such scores change when repeated genes count once. A record without
    the index lists, or with lists that hold other values, repeats none.
    """
    for key in ("s_gene_indices", "g2m_gene_indices"):
        indices = parameters.get(key)
        if not isinstance(indices, list) or not all(
            type(index) is int for index in indices
        ):
            continue
        if len(set(indices)) != len(indices):
            return True
    return False


def _fitted_by_incremental_pca(
    _kind: str,
    parameters: Mapping[str, Any],
    _inputs: Mapping[str, Any],
) -> bool:
    """Select the PCA artifacts that IncrementalPCA fitted over several blocks.

    Those fits, and only those, recorded ``incremental_block_rows``.
    """
    return "incremental_block_rows" in parameters


# Options that only the Harmony of releases before 1.2.0 accepted: Harmony2
# renamed nclust, max_iter_kmeans, and lamb, and removed cluster_fn.
_PRE_HARMONY2_OPTIONS = frozenset({"cluster_fn", "lamb", "max_iter_kmeans", "nclust"})


def _records_pre_harmony2_option(
    kind: str, parameters: Mapping[str, Any], inputs: Mapping[str, Any]
) -> bool:
    """Return whether a correction records an option that only Harmony before 1.2.0 took."""
    options = parameters.get("harmony_parameters")
    return isinstance(options, Mapping) and not _PRE_HARMONY2_OPTIONS.isdisjoint(
        options
    )


def _records_normalizer_outside_cytearc(
    kind: str, parameters: Mapping[str, Any], inputs: Mapping[str, Any]
) -> bool:
    """Return whether an artifact records a normalizer that CyteArc does not define.

    Such a ``normalization_method`` names a function of another module, which
    1.3.0 and later releases also identify by a digest of its code.
    """
    method = parameters.get("normalization_method")
    if not isinstance(method, Mapping) or "identity" in method:
        return False
    module = method.get("module")
    return (
        isinstance(module, str)
        and module != "cytearc"
        and not module.startswith("cytearc.")
    )


_HARMONY2 = OperationRevision(
    revision=2,
    release="1.3.0",
    change=(
        "Harmony2 replaced the earlier Harmony in 1.2.0 (this entry came in "
        "1.3.0); affects corrections that record nclust, max_iter_kmeans, lamb, "
        "or cluster_fn, options of the earlier Harmony only. Other earlier "
        "corrections record what 1.2 records, and the 1.2.0 payload contract "
        "refuses to reuse them"
    ),
    applies=_records_pre_harmony2_option,
)
_NORMALIZER_CODE = OperationRevision(
    revision=2,
    release="1.3.0",
    change=(
        "A normalizer defined outside CyteArc is identified by a digest of its "
        "code and defaults as well as its name, so results of an edited "
        "normalizer are not reused; affects results that record such a normalizer"
    ),
    applies=_records_normalizer_outside_cytearc,
)
# run_marker_search records the same change as its revision 3, after the
# marker table revision of the same release.
_NORMALIZER_CODE_AFTER_MARKER_TABLES = OperationRevision(
    revision=3,
    release=_NORMALIZER_CODE.release,
    change=_NORMALIZER_CODE.change,
    applies=_NORMALIZER_CODE.applies,
)
_EXACT_FEATURE_NEIGHBORS = OperationRevision(
    revision=2,
    release="1.5.0",
    change=(
        "Each feature's neighbors in the graph that forms the modules are its "
        "exact nearest features, the earlier of two equally near ones first, "
        "instead of the approximate neighbors of an HNSW search, whose modules "
        "differed between platforms; ann_params records only the distance, and "
        "modules whose median peaks tie are numbered in the order of the cut"
    ),
    applies=None,
)
# run_pseudotime_aggregation records the same change as its revision 3, after
# the normalizer revision of 1.3.0.
_EXACT_FEATURE_NEIGHBORS_AFTER_NORMALIZER_CODE = OperationRevision(
    revision=3,
    release=_EXACT_FEATURE_NEIGHBORS.release,
    change=_EXACT_FEATURE_NEIGHBORS.change,
    applies=_EXACT_FEATURE_NEIGHBORS.applies,
)


def _scores_the_default_potential(
    kind: str, parameters: Mapping[str, Any], inputs: Mapping[str, Any]
) -> bool:
    """Select the pseudotimes that do not record the method ``"truncated"``.

    Pseudotimes of releases before 1.4.0 record no method and hold the
    truncated potential, which the untruncated potential replaced as the
    default; a pseudotime that records ``"truncated"`` asked for that
    potential, which is unchanged.
    """
    return parameters.get("method") != "truncated"


def _finite_number(value: object) -> bool:
    """Return whether a recorded value is a finite number, not a special float."""
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


# The default HVG blacklist before 1.3.0. Its ``^CCN`` also removed CCN1 to
# CCN6, and its ``^HIST`` missed the current histone symbols.
_HVG_BLACKLIST_BEFORE_1_3 = (
    "^MT-|^RPS|^RPL|^MRPS|^MRPL|^CCN|^HLA-|^H2-|^HIST|"
    "^XIST$|^DDX3Y$|^USP9Y$|^EIF1AY$|^KDM5D$|^SRY$|^ZFY$|^UTY$|^TMSB4Y$|^NLGN4Y$"
)


def _hvg_selection_1_3(
    kind: str, parameters: Mapping[str, Any], inputs: Mapping[str, Any]
) -> bool:
    """Select HVG selections that the candidate filters and defaults of 1.3.0 change.

    A finite ``min_var`` or ``max_var`` now filters candidates before
    ``top_n``. Every selection with the earlier default detection bounds
    records ``min_cells`` 20; those bounds did not scale with the selected
    cells, and the ``max_cells`` they recorded cannot be told from a value
    passed explicitly. Selections with the earlier default blacklist excluded
    CCN1 to CCN6 and kept the current histone symbols.
    """
    variance_bound = _finite_number(parameters.get("min_var")) or _finite_number(
        parameters.get("max_var")
    )
    earlier_bounds = parameters.get("min_cells") == 20
    earlier_blacklist = parameters.get("blacklist") == _HVG_BLACKLIST_BEFORE_1_3
    return variance_bound or earlier_bounds or earlier_blacklist


def _chooses_its_doublet_thresholds(
    kind: str, parameters: Mapping[str, Any], inputs: Mapping[str, Any]
) -> bool:
    """Select the doublet scores that choose each sample's threshold automatically.

    Such scores record ``threshold`` as None. Scores of a given threshold call
    the same cells. Scores of revision 1, which called no doublets, record no
    ``threshold`` and are selected too, as every later revision selects them.
    """
    return parameters.get("threshold") is None


def _scores_batches_under_an_unrecorded_normalizer(
    kind: str, parameters: Mapping[str, Any], inputs: Mapping[str, Any]
) -> bool:
    """Select batch-aware HVG selections of the cytearc flavor that record no normalizer.

    Such a selection scored each batch's values under the session's
    normalizer and size factor without recording them, so a call under
    another normalizer or size factor reused it. The cytearc flavor records
    no ``flavor``, and the other flavors read the counts.
    """
    return (
        "flavor" not in parameters
        and "batch" in inputs
        and "normalization_method" not in parameters
    )


# Released revisions by operation. An operation that is not listed has no
# revisions, so every artifact of it is revision 1. Append a revision to the
# tuple of the operation it changes, adding the operation if needed; never
# edit a released entry.
_OPERATION_REVISIONS: dict[str, tuple[OperationRevision, ...]] = {
    "run_cell_cycle_scoring": (
        OperationRevision(
            revision=2,
            release="1.3.0",
            change=(
                "A gene index repeated in s_gene_indices or g2m_gene_indices, "
                "from a repeated name or names that differ only in case, counts "
                "once in the score; only scores that record a repeated index "
                "change."
            ),
            applies=_repeats_cell_cycle_gene,
        ),
    ),
    "run_hto_demultiplexing": (
        OperationRevision(
            revision=2,
            release="1.3.0",
            change=(
                "A singlet takes the one HTO that its negative-binomial cutoff "
                "calls positive instead of the HTO with the highest CLR value, "
                "which is stored as the max_id diagnostic beside second_id and "
                "margin."
            ),
            applies=None,
        ),
        OperationRevision(
            revision=3,
            release="1.4.0",
            change=(
                "A hashtag whose background cluster gives a cutoff of 0 is refused "
                "when the counts carry ambient tags, and hashtags without counts "
                "in the selected cells are left out instead of refused."
            ),
            applies=None,
        ),
    ),
    "run_pca": (
        OperationRevision(
            revision=2,
            release="1.3.0",
            change=(
                "IncrementalPCA fits over several row blocks, which recorded "
                "incremental_block_rows, left trailing components and variance "
                "ratios inexact; PCA now fits every block layout exactly up to "
                "4096 features and with a randomized solver above."
            ),
            applies=_fitted_by_incremental_pca,
        ),
    ),
    "run_doublet_detection": (
        OperationRevision(
            revision=2,
            release="1.3.0",
            change=(
                "Doublets are simulated from every selected cell of each sample "
                "and scored by Scrublet's likelihood of the simulated doublets "
                "among each cell's neighbors in its sample, with raw scores, "
                "per-sample thresholds, and predicted doublets."
            ),
            applies=None,
        ),
        OperationRevision(
            revision=3,
            release="1.5.0",
            change=(
                "A sample of 50,000 points or more is indexed in 4 parts of "
                "every fourth point, built at once on the workers and searched "
                "to a shared depth, and the automatic threshold takes Otsu's "
                "threshold when fewer than 5% of a sample's simulated doublets "
                "lie on either side of the histogram minimum and Otsu's "
                "threshold estimates a doublet rate of at most 50%"
            ),
            applies=None,
        ),
        OperationRevision(
            revision=4,
            release="1.5.0",
            change=(
                "An automatic threshold is taken only where it estimates a "
                "doublet rate of at most 50%: the histogram minimum, else Otsu's "
                "threshold, else none, and a sample without one calls no cell and "
                "records the threshold method 'none' and a threshold of None; "
                "affects scores with an automatic threshold"
            ),
            applies=_chooses_its_doublet_thresholds,
        ),
    ),
    "run_harmony": (_HARMONY2,),
    "run_lineage_aggregation": (_EXACT_FEATURE_NEIGHBORS,),
    "run_marker_search": (
        OperationRevision(
            revision=2,
            release="1.3.0",
            change=(
                "Marker tables store full-precision statistics and "
                "neg_log10_p_value, give exact p-values to groups that can be "
                "split from the other cells in at most 100,000 ways, and skip "
                "groups of fewer than two cells instead of failing"
            ),
            applies=None,
        ),
        _NORMALIZER_CODE_AFTER_MARKER_TABLES,
    ),
    "run_normalization": (_NORMALIZER_CODE,),
    "run_pseudotime_aggregation": (
        _NORMALIZER_CODE,
        _EXACT_FEATURE_NEIGHBORS_AFTER_NORMALIZER_CODE,
    ),
    "run_pseudotime_marker_search": (_NORMALIZER_CODE,),
    "run_pseudotime_scoring": (
        OperationRevision(
            revision=2,
            release="1.4.0",
            change=(
                "The default potential is the untruncated population balance "
                "potential, solved by MINRES, instead of the potential of the "
                "smallest singular modes, 30 with the null mode; affects "
                "pseudotimes that do not record method='truncated', which "
                "computes the earlier potential"
            ),
            applies=_scores_the_default_potential,
        ),
    ),
    "run_statistical_testing": (_NORMALIZER_CODE,),
    "summarize_atac_features": (_NORMALIZER_CODE,),
    "summarize_rna_features": (_NORMALIZER_CODE,),
    "select_hvgs": (
        OperationRevision(
            revision=2,
            release="1.3.0",
            change=(
                "min_var and max_var filter HVG candidates before top_n, and the "
                "default min_cells, max_cells, and blacklist changed; affects "
                "selections with a finite min_var or max_var, min_cells 20, or "
                "the earlier default blacklist"
            ),
            applies=_hvg_selection_1_3,
        ),
        OperationRevision(
            revision=3,
            release="1.5.0",
            change=(
                "Selections save the selectedDetectionRatio diagnostic, which "
                "earlier selections lack; for the same recorded parameters they "
                "select the same genes, since a new default min_cells changes the "
                "recorded min_cells and the restart of a stalled trend fit only "
                "completes fits that raised before"
            ),
            applies=None,
            added_diagnostics=("selectedDetectionRatio",),
        ),
        OperationRevision(
            revision=4,
            release="1.5.0",
            change=(
                "Batch-aware selections of the cytearc flavor record the "
                "normalization_method and size_factor of the values they score, so "
                "a call under another normalizer or size factor computes its own "
                "selection; affects such selections that record no normalizer"
            ),
            applies=_scores_batches_under_an_unrecorded_normalizer,
        ),
    ),
    "import_clones": (
        OperationRevision(
            revision=2,
            release="1.5.0",
            change=(
                "Imports save each file's rows in the rowsPerFile diagnostic, "
                "which earlier imports lack, so that a reused import warns of a "
                "file that holds no row; the same files give the same clones"
            ),
            applies=None,
            added_diagnostics=("rowsPerFile",),
        ),
    ),
    "import_immune_receptors": (
        OperationRevision(
            revision=2,
            release="1.5.0",
            change=(
                "Imports save each file's contigs in the contigsPerFile "
                "diagnostic, which earlier imports lack, so that a reused import "
                "warns of a file that holds no contig; the same files give the "
                "same receptors"
            ),
            applies=None,
            added_diagnostics=("contigsPerFile",),
        ),
    ),
    "import_lineage_barcodes": (
        OperationRevision(
            revision=2,
            release="1.5.0",
            change=(
                "Imports save each file's rows in the rowsPerFile diagnostic, "
                "which earlier imports lack, so that a reused import warns of a "
                "file that holds no row; the same files give the same barcodes"
            ),
            applies=None,
            added_diagnostics=("rowsPerFile",),
        ),
    ),
    "import_recorder_alleles": (
        OperationRevision(
            revision=2,
            release="1.5.0",
            change=(
                "Imports save each file's rows in the rowsPerFile diagnostic, "
                "which earlier imports lack, so that a reused import warns of a "
                "file that holds no row; the same files give the same alleles"
            ),
            applies=None,
            added_diagnostics=("rowsPerFile",),
        ),
    ),
}


def _validate_registry(
    registry: Mapping[str, tuple[OperationRevision, ...]],
) -> None:
    for operation, revisions in registry.items():
        numbers = [entry.revision for entry in revisions]
        if numbers != list(range(2, len(numbers) + 2)):
            raise ValueError(
                f"Revisions of {operation!r} must be numbered 2, 3, ... in order, "
                f"got {numbers}"
            )
        for entry in revisions:
            # Every artifact computed from such a revision on records it or a
            # later one, so one of an earlier revision predates its diagnostics.
            if entry.added_diagnostics and entry.applies is not None:
                raise ValueError(
                    f"Revision {entry.revision} of {operation!r} adds the "
                    f"diagnostics {list(entry.added_diagnostics)}, which every "
                    "artifact computed from it on saves, so it must apply to every "
                    "artifact, with applies=None"
                )


_validate_registry(_OPERATION_REVISIONS)
OPERATION_REVISIONS: Mapping[str, tuple[OperationRevision, ...]] = MappingProxyType(
    _OPERATION_REVISIONS
)
"""Released revisions by operation; an operation that is not listed has none."""


def applicable_revisions(
    operation: str,
    kind: str,
    parameters: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> tuple[OperationRevision, ...]:
    """Return the released revisions that apply to one artifact, oldest first."""
    return tuple(
        entry
        for entry in OPERATION_REVISIONS.get(operation, ())
        if entry.applies is None or entry.applies(kind, parameters, inputs)
    )


def effective_revision(
    operation: str,
    kind: str,
    parameters: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> int:
    """Return the highest applicable revision of ``operation`` for an artifact, or 1."""
    applicable = applicable_revisions(operation, kind, parameters, inputs)
    return applicable[-1].revision if applicable else 1


def diagnostics_added_after(operation: str, revision: int) -> frozenset[str]:
    """Return the diagnostics that revisions of ``operation`` after ``revision`` added.

    An artifact that records ``revision`` was saved before ``operation``
    saved them.
    """
    return frozenset(
        key
        for entry in OPERATION_REVISIONS.get(operation, ())
        if entry.revision > revision
        for key in entry.added_diagnostics
    )
