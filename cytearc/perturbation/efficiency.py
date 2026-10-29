"""How strongly each perturbation changes the expression of its target gene.

A guide that knocks its gene down, by Cas9 cutting or CRISPR interference,
lowers the gene's expression in the cells that carry it, and one that
activates its gene raises it. Cell Ranger calls this check perturbation
efficiency. For each target gene, or each guide, these functions compare the
target gene's expression in the cells that carry the perturbation with its
expression in control cells:

- the mean UMIs of the gene in each group, as Cell Ranger reports them;
- the mean expression after scaling each cell to the median library size of
  the compared cells, and the fraction of the control expression that
  remains;
- a two-sided Mann-Whitney test on the scaled expression, with
  Benjamini-Hochberg adjustment across the tested rows.

They read the counts that a caller passes and never touch storage.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, mannwhitneyu

__all__ = [
    "EFFICIENCY_STATUSES",
    "NO_CELLS",
    "NO_CONTROL_EXPRESSION",
    "NO_FEATURE",
    "TESTED",
    "EfficiencyArrays",
    "PerturbationEfficiency",
    "efficiency_arrays",
    "efficiency_settings",
    "id_column_setting",
    "rows_setting",
    "target_features",
]

#: Row statuses, stored as positions in this tuple.
EFFICIENCY_STATUSES = ("tested", "no_feature", "no_cells", "no_control_expression")
#: The target gene was compared between perturbed and control cells.
TESTED = 0
#: The target matches no feature of the expression assay.
NO_FEATURE = 1
#: No cell carries the perturbation alone.
NO_CELLS = 2
#: The control cells hold no UMI of the target gene, so no fraction remains.
NO_CONTROL_EXPRESSION = 3


def rows_setting(by: object) -> str:
    """Check ``by``, the rows of a perturbation table: targets or guides."""
    if not isinstance(by, str) or by not in ("target", "guide"):
        raise ValueError(f"by must be 'target' or 'guide', not {by!r}")
    return by


def id_column_setting(id_column: object) -> str | None:
    """Check ``id_column``, the guide assay's column of target feature IDs.

    Whether the column exists is checked against the guide assay.
    """
    if id_column is not None and (not isinstance(id_column, str) or not id_column):
        raise TypeError("id_column must be a feature column name or None")
    return id_column


def efficiency_settings(by: object, id_column: object) -> None:
    """Check the settings of perturbation efficiency, as ``efficiency`` takes them.

    The pipeline recipe checks the same settings before its run starts.

    Raises:
        TypeError: If ``id_column`` is neither a column name nor None.
        ValueError: If ``by`` is neither ``"target"`` nor ``"guide"``.
    """
    rows_setting(by)
    id_column_setting(id_column)


@dataclass(frozen=True, slots=True)
class EfficiencyArrays:
    """One entry per target or guide.

    Attributes:
        status: The position of each row's status in
            :data:`EFFICIENCY_STATUSES`.
        n_cells: The cells that carry the perturbation alone.
        n_control_cells: The control cells.
        mean_umi: The mean UMIs of the target gene in the perturbed cells.
        control_mean_umi: The mean UMIs of the target gene in control cells.
        mean_expression: The mean scaled expression in the perturbed cells.
        control_mean_expression: The mean scaled expression in control cells.
        remaining_fraction: ``mean_expression`` over
            ``control_mean_expression``.
        log2_fold_change: The base-2 logarithm of ``remaining_fraction``,
            minus infinity when no UMI remains.
        p_value: The two-sided Mann-Whitney p-value.
        adjusted_p_value: The Benjamini-Hochberg adjustment across the rows
            with a p-value.
    """

    status: np.ndarray
    n_cells: np.ndarray
    n_control_cells: np.ndarray
    mean_umi: np.ndarray
    control_mean_umi: np.ndarray
    mean_expression: np.ndarray
    control_mean_expression: np.ndarray
    remaining_fraction: np.ndarray
    log2_fold_change: np.ndarray
    p_value: np.ndarray
    adjusted_p_value: np.ndarray


def target_features(
    target_names: np.ndarray,
    guide_target_codes: np.ndarray,
    guide_feature_ids: np.ndarray | None,
    feature_ids: np.ndarray,
    feature_names: np.ndarray,
) -> np.ndarray:
    """Match each target to a feature of the expression assay.

    A target takes the feature whose ID its guides give, when they give one
    ID and the feature IDs are unique. Otherwise it takes the one feature
    whose name equals the target, when no other feature shares that name.

    Args:
        target_names: The targets.
        guide_target_codes: Each guide's position in ``target_names``, -1
            for a guide without a target.
        guide_feature_ids: Each guide's target feature ID, or None to match by
            name only. A missing or blank ID gives nothing.
        feature_ids: The IDs of the expression features.
        feature_names: The names of the expression features.

    Returns:
        Each target's feature position, -1 without a match.
    """
    codes = np.asarray(guide_target_codes)
    matched = np.full(len(target_names), -1, dtype=np.int64)
    by_id = pd.Index(np.asarray(feature_ids).astype(str))
    if guide_feature_ids is not None and by_id.is_unique:
        column = np.asarray(guide_feature_ids, dtype=object)
        for code in range(len(target_names)):
            ids = {
                str(value).strip()
                for value in column[codes == code]
                if value is not None and not pd.isna(value) and str(value).strip()
            }
            if len(ids) == 1:
                matched[code] = by_id.get_indexer(list(ids))[0]
    names = np.asarray(feature_names).astype(str)
    distinct, name_counts = np.unique(names, return_counts=True)
    unique_names = pd.Index(distinct[name_counts == 1])
    position = pd.Series(np.arange(len(names)), index=names)
    for unmatched in np.flatnonzero(matched < 0):
        name = str(target_names[unmatched])
        if unique_names.get_indexer([name])[0] >= 0:
            matched[unmatched] = int(position[name])
    return matched


def _two_sided_p(perturbed: np.ndarray, control: np.ndarray) -> float:
    """Return the two-sided Mann-Whitney p-value, 1 when every value is equal."""
    values = np.concatenate([perturbed, control])
    if np.all(values == values[0]):
        return 1.0
    return float(
        mannwhitneyu(
            perturbed,
            control,
            alternative="two-sided",
            method="asymptotic",
            use_continuity=True,
        ).pvalue
    )


def efficiency_arrays(
    counts: np.ndarray,
    library_sizes: np.ndarray,
    groups: np.ndarray,
    control: np.ndarray,
    row_columns: np.ndarray,
) -> EfficiencyArrays:
    """Compare each row's target gene between its cells and the control cells.

    Args:
        counts: The UMIs of the target genes, cells x genes.
        library_sizes: The total UMIs of each cell.
        groups: The row each cell carries alone, or -1.
        control: Whether each cell is a control cell.
        row_columns: The column of each row's gene in ``counts``, or -1 when
            the row's target matches no feature.

    Returns:
        The status, cell counts, means, remaining fraction, fold change and
        p-values of every row.

    Raises:
        ValueError: If the arrays do not align, or there is no control cell.
    """
    counts = np.asarray(counts, dtype=np.float64)
    library = np.asarray(library_sizes, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)
    control = np.asarray(control, dtype=bool)
    row_columns = np.asarray(row_columns, dtype=np.int64)
    n_cells = counts.shape[0]
    if counts.ndim != 2 or len(library) != n_cells:
        raise ValueError("counts must be cells x genes, with one library size per cell")
    if len(groups) != n_cells or len(control) != n_cells:
        raise ValueError("groups and control must hold one entry per cell")
    if not control.any():
        raise ValueError("There is no control cell to compare with")
    compared = (groups >= 0) | control
    scale = float(np.median(library[compared]))
    with np.errstate(divide="ignore", invalid="ignore"):
        per_cell = np.where(library > 0, scale / library, 0.0)
    expression = counts * per_cell[:, None]
    n_rows = len(row_columns)
    order = np.argsort(groups, kind="stable")
    bounds = np.searchsorted(groups[order], np.arange(n_rows + 1))
    controls = np.flatnonzero(control)
    status = np.full(n_rows, TESTED, dtype=np.int8)
    shape = n_rows
    out = {
        name: np.full(shape, np.nan)
        for name in (
            "mean_umi",
            "control_mean_umi",
            "mean_expression",
            "control_mean_expression",
            "remaining_fraction",
            "log2_fold_change",
            "p_value",
        )
    }
    sizes = np.diff(bounds).astype(np.int64)
    for row in range(n_rows):
        column = row_columns[row]
        members = order[bounds[row] : bounds[row + 1]]
        if column < 0:
            status[row] = NO_FEATURE
            continue
        if not len(members):
            status[row] = NO_CELLS
            continue
        out["mean_umi"][row] = counts[members, column].mean()
        out["control_mean_umi"][row] = counts[controls, column].mean()
        perturbed = expression[members, column]
        reference = expression[controls, column]
        out["mean_expression"][row] = perturbed.mean()
        out["control_mean_expression"][row] = reference.mean()
        out["p_value"][row] = _two_sided_p(perturbed, reference)
        if out["control_mean_expression"][row] == 0:
            status[row] = NO_CONTROL_EXPRESSION
            continue
        fraction = out["mean_expression"][row] / out["control_mean_expression"][row]
        out["remaining_fraction"][row] = fraction
        with np.errstate(divide="ignore"):
            out["log2_fold_change"][row] = np.log2(fraction)
    adjusted = np.full(n_rows, np.nan)
    tested = np.isfinite(out["p_value"])
    if tested.any():
        adjusted[tested] = false_discovery_control(out["p_value"][tested], method="bh")
    return EfficiencyArrays(
        status=status,
        n_cells=sizes,
        n_control_cells=np.full(n_rows, len(controls), dtype=np.int64),
        adjusted_p_value=adjusted,
        **out,
    )


@dataclass(frozen=True, slots=True)
class PerturbationEfficiency:
    """How strongly each perturbation changes its target gene.

    Attributes:
        table: One row per target or guide, with ``target``, ``guide`` (for
            rows by guide), ``feature`` (the matched feature ID, None without
            one), ``status``, ``n_cells``, ``n_control_cells``, ``mean_umi``,
            ``control_mean_umi``, ``mean_expression``,
            ``control_mean_expression``, ``remaining_fraction``,
            ``log2_fold_change``, ``p_value`` and ``adjusted_p_value``.
        by: ``"target"`` or ``"guide"``.
        from_assay: The expression assay.
    """

    table: pd.DataFrame = field(repr=False)
    by: str
    from_assay: str
