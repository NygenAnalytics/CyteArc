"""Perturbation labels of pooled CRISPR screens: what each cell's guides target.

Each guide of a library targets one gene, and control guides, such as
non-targeting guides, target nothing in the genome. Two guides of one gene
perturb the same gene, so a cell's perturbation is the set of genes its called
guides target, apart from the control guides. A cell's label names that set:

- ``"Negative"`` when no guide is called;
- ``"control"`` when only control guides are called;
- the gene when the called guides target one gene, with or without control
  guides;
- ``"Multiple"`` when they target two or more genes, or with the
  ``"combinations"`` policy, the genes sorted and joined by ``"+"``.

These functions resolve the target of each guide and turn calls into the arrays
of a ``perturbation_labels`` artifact. They never touch storage.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cytearc.perturbation.guide_calls import MULTIPLE, NEGATIVE
from cytearc.utils.arguments import optional_name_argument

__all__ = [
    "CONTROL",
    "POLICIES",
    "RESERVED_TARGETS",
    "GuideTargets",
    "PerturbationLabelArrays",
    "PerturbationLabels",
    "guide_targets",
    "label_memory_bytes",
    "label_settings",
    "longest_label",
    "perturbation_label_arrays",
    "perturbation_label_frames",
]

#: The label of a cell whose called guides are all control guides.
CONTROL = "control"
#: Labels that no target may take, since they label cells without one target.
RESERVED_TARGETS = frozenset({CONTROL, NEGATIVE, MULTIPLE})
#: How a cell whose guides target several genes is labelled.
POLICIES = ("single", "combinations")
_SEPARATOR = "+"


@dataclass(frozen=True, slots=True)
class GuideTargets:
    """The target of every guide of a guide assay.

    Attributes:
        names: The distinct targets, sorted, control targets included.
        is_control: Whether each target is a control target.
        guide_codes: Each guide's position in ``names``, -1 for a guide
            without a target.
    """

    names: np.ndarray
    is_control: np.ndarray
    guide_codes: np.ndarray

    def guide_names(self) -> np.ndarray:
        """Return each guide's target name, an empty string without one."""
        names = np.full(len(self.guide_codes), "", dtype=object)
        targeted = self.guide_codes >= 0
        names[targeted] = self.names[self.guide_codes[targeted]]
        return names


def guide_targets(
    guide_ids: np.ndarray,
    column: np.ndarray | None,
    overrides: Mapping[str, str] | None,
    controls: tuple[str, ...],
    *,
    policy: str,
) -> GuideTargets:
    """Resolve and check the target of each guide.

    Args:
        guide_ids: The guide IDs in assay order.
        column: Each guide's target from a feature column, or None. A missing
            or blank value means the guide has no target there.
        overrides: Targets by guide ID that replace the column's, or None.
        controls: The target names of control guides.
        policy: ``"single"`` or ``"combinations"``.

    Returns:
        The distinct targets, which of them are controls, and each guide's
        target.

    Raises:
        ValueError: If an override names a guide that is not in the assay or
            gives an empty target, a target that is not a control is a
            reserved label or, with ``"combinations"``, holds ``"+"``, or a
            control matches no target.
    """
    ids = np.asarray(guide_ids, dtype=object).astype(str)
    names = np.full(len(ids), "", dtype=object)
    if column is not None:
        for index, value in enumerate(np.asarray(column, dtype=object)):
            text = "" if value is None or pd.isna(value) else str(value).strip()
            names[index] = text
    if overrides:
        positions = pd.Index(ids).get_indexer(list(overrides))
        unknown = [
            guide for guide, at in zip(overrides, positions, strict=True) if at < 0
        ]
        if unknown:
            raise ValueError(
                f"targets names {len(unknown)} guides that are not guides of the "
                "assay, for example " + ", ".join(repr(guide) for guide in unknown[:5])
            )
        for position, target in zip(positions, overrides.values(), strict=True):
            if not str(target).strip():
                raise ValueError(
                    f"targets gives guide {ids[position]!r} an empty target"
                )
            names[position] = str(target).strip()
    distinct = np.asarray(sorted({name for name in names if name}), dtype=object)
    # A control target never becomes a label: a cell with control guides only
    # is labelled "control" whatever their targets are called.
    reserved = sorted(
        RESERVED_TARGETS.intersection(distinct.tolist()).difference(controls)
    )
    if reserved:
        raise ValueError(
            "Targets conflict with labels reserved for cells without one target: "
            + ", ".join(repr(name) for name in reserved)
            + ". Give their guides other targets with targets, or name them in "
            "controls if their guides are control guides"
        )
    if policy == "combinations":
        # Control targets never join the genes of a combination either.
        joined = [
            name for name in distinct if _SEPARATOR in name and name not in controls
        ]
        if joined:
            raise ValueError(
                f"With policy='combinations', targets that are not controls must "
                f"not hold {_SEPARATOR!r}, which joins the genes of a combination, "
                f"but {joined[0]!r} does"
            )
    missing = [control for control in controls if control not in set(distinct)]
    if missing:
        raise ValueError(
            "controls name targets that no guide has: "
            + ", ".join(repr(control) for control in missing)
        )
    lookup = pd.Index(distinct.astype(str))
    guide_codes = np.where(names != "", lookup.get_indexer(names.astype(str)), -1)
    return GuideTargets(
        names=distinct,
        is_control=np.isin(distinct, list(controls)),
        guide_codes=guide_codes.astype(np.int32),
    )


def label_settings(
    controls: object,
    target_column: object,
    targets: object,
    policy: object,
) -> tuple[str, ...]:
    """Check the settings of perturbation labels, as ``labels`` takes them.

    The pipeline recipe checks the same settings before its run starts.
    Whether the target column exists, and the targets that it and
    ``targets`` give, are checked by :func:`guide_targets` against the guide
    assay.

    Args:
        controls: The targets of control guides.
        target_column: The name of the guide assay's feature column that holds
            each guide's target, or None.
        targets: Targets by guide ID, or None.
        policy: One of :data:`POLICIES`.

    Returns:
        The control targets, stripped, without repeats and sorted.

    Raises:
        TypeError: If a setting has the wrong type.
        ValueError: If ``controls`` names no target or a blank one, neither
            ``target_column`` nor ``targets`` is given, or ``policy`` is
            unknown.
    """
    if (
        isinstance(controls, str)
        or not isinstance(controls, Sequence)
        or not all(isinstance(name, str) for name in controls)
    ):
        raise TypeError(
            "controls must be a sequence of target names, such as ['Non-Targeting']"
        )
    control_names = tuple(sorted({name.strip() for name in controls}))
    if not control_names or "" in control_names:
        raise ValueError(
            "controls must name at least one target, and no name may be blank"
        )
    optional_name_argument(target_column, "target_column", noun="a feature column name")
    if targets is not None and (
        not isinstance(targets, Mapping)
        or not all(
            isinstance(guide, str) and isinstance(target, str)
            for guide, target in targets.items()
        )
    ):
        raise TypeError("targets must map guide IDs to target names, or be None")
    if target_column is None and not targets:
        raise ValueError(
            "Pass target_column, targets, or both, so that every called guide "
            "has a target"
        )
    if not isinstance(policy, str) or policy not in POLICIES:
        raise ValueError(f"policy must be 'single' or 'combinations', not {policy!r}")
    return control_names


def longest_label(targets: GuideTargets, width: int, *, policy: str) -> int:
    """Return the most characters that a cell's label can hold.

    Args:
        targets: The target of every guide, from :func:`guide_targets`.
        width: The most guides called in one cell.
        policy: ``"single"`` or ``"combinations"``, which joins the genes of a
            cell with several, at most ``width`` of the longest.
    """
    lengths = sorted(
        (len(str(name)) for name in targets.names[~targets.is_control]), reverse=True
    )
    longest = max([len(NEGATIVE), len(MULTIPLE), len(CONTROL), *lengths[:1]])
    if policy == "single":
        return longest
    joined = lengths[: max(1, width)]
    return max(longest, sum(joined) + len(joined) - 1)


def label_memory_bytes(
    n_cells: int, width: int, *, label_length: int, joined: bool
) -> int:
    """Return the bytes that :func:`perturbation_label_arrays` holds beside the codes.

    Args:
        n_cells: The cells.
        width: The most guides called in one cell, the width of the codes.
        label_length: The most characters of a label, as
            :func:`longest_label` gives it.
        joined: Whether a cell with several genes gets a label of its own,
            as ``policy="combinations"`` gives it.
    """
    # Each entry's target, control flag and call flag, and the flags that
    # check the targets, take at most 8 bytes, and each cell's sorted targets
    # are copied out at 4 bytes per entry while the entries are released. Each
    # cell's label, control flag and number of targets, as counted and as
    # stored, take 24 bytes, and a joined label is a string of its own of
    # about 64 bytes and at most 4 bytes per character.
    per_cell = 24 + (64 + 4 * label_length if joined else 0)
    return n_cells * (9 * width + per_cell)


@dataclass(frozen=True, slots=True)
class PerturbationLabelArrays:
    """The per-cell arrays of a ``perturbation_labels`` artifact.

    Attributes:
        values: Each cell's label.
        is_control: Whether each cell carries control guides only.
        n_targets: The number of genes each cell's called guides target.
        target_codes: Each cell's targets as positions in the targets, in
            ascending order, padded with -1.
    """

    values: np.ndarray
    is_control: np.ndarray
    n_targets: np.ndarray
    target_codes: np.ndarray


def perturbation_label_arrays(
    guide_codes: np.ndarray,
    targets: GuideTargets,
    *,
    policy: str,
) -> PerturbationLabelArrays:
    """Label each cell by the genes its called guides target.

    Args:
        guide_codes: Each cell's called guides as positions in the assay,
            padded with -1, as a ``guide_calls`` artifact holds them.
        targets: The target of every guide, from :func:`guide_targets`.
        policy: ``"single"`` labels a cell whose guides target several genes
            ``"Multiple"``; ``"combinations"`` joins the genes, sorted, with
            ``"+"``.

    Returns:
        The label, control flag, number of targets and target codes of each
        cell.

    Raises:
        ValueError: If a called guide has no target.
    """
    codes = np.asarray(guide_codes)
    guide_target = np.asarray(targets.guide_codes, dtype=np.int32)
    guide_control = np.zeros(len(guide_target), dtype=bool)
    targeted = guide_target >= 0
    guide_control[targeted] = targets.is_control[guide_target[targeted]]
    called = codes >= 0
    # Each entry's target and whether it is a control target. An entry padded
    # with -1 reads the last guide's, which the steps below set aside.
    keyed = guide_target[codes]
    control = guide_control[codes]
    untargeted = np.unique(codes[called & (keyed < 0)])
    if len(untargeted):
        raise ValueError(
            f"{len(untargeted)} called guides have no target; pass targets or "
            "choose a target_column that names them"
        )
    control &= called
    is_control = control.any(axis=1)
    # The distinct genes of each cell, in ascending order, padded at the end:
    # entries without a called gene guide take the pad, which sorts last. The
    # entries are sorted in place, so that no copy of them is made.
    pad = np.iinfo(np.int32).max
    np.copyto(keyed, pad, where=control)
    np.copyto(keyed, pad, where=~called)
    del called, control
    keyed.sort(axis=1)
    repeated = np.zeros(keyed.shape, dtype=bool)
    np.equal(keyed[:, 1:], keyed[:, :-1], out=repeated[:, 1:])
    np.copyto(keyed, pad, where=repeated)
    del repeated
    keyed.sort(axis=1)
    n_targets = np.count_nonzero(keyed != pad, axis=1).astype(np.int32)
    width = max(1, int(n_targets.max(initial=0)))
    target_codes = keyed[:, :width].copy()
    del keyed
    target_codes[target_codes == pad] = -1
    is_control &= n_targets == 0
    values = np.full(len(codes), NEGATIVE, dtype=object)
    values[is_control] = CONTROL
    one = n_targets == 1
    values[one] = targets.names[target_codes[one, 0]]
    several = np.flatnonzero(n_targets > 1)
    if policy == "single":
        values[several] = MULTIPLE
    else:
        for row in several:
            genes = target_codes[row]
            values[row] = _SEPARATOR.join(targets.names[genes[genes >= 0]])
    return PerturbationLabelArrays(
        values=values,
        is_control=is_control,
        n_targets=n_targets,
        target_codes=target_codes,
    )


@dataclass(frozen=True, slots=True)
class PerturbationLabels:
    """The perturbation labels of a set of cells.

    Attributes:
        cells: One row per cell, indexed by cell ``ids``, with ``label``,
            ``guide_label`` (the cell's label in the guide calls),
            ``is_control`` and ``n_targets``.
        targets: One row per cell and gene its called guides target, with the
            cell's ``ids`` and the ``target``; control guides add no row.
        guides: One row per guide of the assay, with ``guide``, ``target``
            (None for a guide without one) and ``is_control``.
        controls: The target names of control guides.
        policy: ``"single"`` or ``"combinations"``.
    """

    cells: pd.DataFrame = field(repr=False)
    targets: pd.DataFrame = field(repr=False)
    guides: pd.DataFrame = field(repr=False)
    controls: tuple[str, ...]
    policy: str


def perturbation_label_frames(
    cell_ids: np.ndarray,
    *,
    values: np.ndarray,
    guide_label: np.ndarray,
    is_control: np.ndarray,
    n_targets: np.ndarray,
    target_codes: np.ndarray,
    target_names: np.ndarray,
    target_is_control: np.ndarray,
    guide_ids: np.ndarray,
    guide_target_codes: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the per-cell, per-target and per-guide tables of stored labels."""
    index = pd.Index(np.asarray(cell_ids), name="ids")
    cells = pd.DataFrame(
        {
            "label": np.asarray(values, dtype=object),
            "guide_label": np.asarray(guide_label, dtype=object),
            "is_control": np.asarray(is_control, dtype=bool),
            "n_targets": np.asarray(n_targets),
        },
        index=index,
    )
    codes = np.asarray(target_codes)
    rows, slots = np.nonzero(codes >= 0)
    names = np.asarray(target_names, dtype=object)
    targets = pd.DataFrame(
        {
            "ids": np.asarray(cell_ids)[rows],
            "target": names[codes[rows, slots]],
        }
    )
    guide_codes = np.asarray(guide_target_codes)
    targeted = guide_codes >= 0
    guide_targets_text = np.full(len(guide_codes), None, dtype=object)
    guide_targets_text[targeted] = names[guide_codes[targeted]]
    guide_controls = np.zeros(len(guide_codes), dtype=bool)
    guide_controls[targeted] = np.asarray(target_is_control, dtype=bool)[
        guide_codes[targeted]
    ]
    guides = pd.DataFrame(
        {
            "guide": np.asarray(guide_ids, dtype=object),
            "target": pd.Series(guide_targets_text, dtype=object),
            "is_control": guide_controls,
        }
    )
    return cells, targets, guides
