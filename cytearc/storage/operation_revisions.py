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
Artifacts that record no revision are revision 1.

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
    """

    revision: int
    release: str
    change: str
    applies: RevisionPredicate | None


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
    ),
    "run_harmony": (_HARMONY2,),
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
    "run_pseudotime_aggregation": (_NORMALIZER_CODE,),
    "run_pseudotime_marker_search": (_NORMALIZER_CODE,),
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
