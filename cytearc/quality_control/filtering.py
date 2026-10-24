from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Literal, TypedDict

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..metadata.selection import NamedCellArtifact
from ..utils.arrays import within_bounds
from ..utils.warnings import warn

__all__ = ["gaussian_quantile_bounds"]

FilterMethod = Literal["manual", "gaussian", "mad"]
MetricKind = Literal["counts", "features", "percentage"]

_MAD_SCALE = 1.4826
_COUNT_SUFFIXES = ("nCounts", "nFeatures")
_PERCENT_SUFFIXES = ("percentMito", "percentRibo")
_TRANSFORMS = ("log1p", "identity")
_BOUND_DIRECTIONS = ("two_sided", "lower", "upper")
# Automatic bounds that remove more than this fraction of the selected cells
# are reported, because they were set by a population other than the cells.
_MAJORITY_REMOVED_FRACTION = 0.5
# A log1p metric is strongly bimodal when the best split of its values into a
# low and a high group separates them by at least this Ashman's D; a single
# normal population gives about 2.7 and a uniform one about 3.5.
_BIMODAL_SEPARATION = 5.0
# Cells beyond an upper bound form a separate population when their median
# lies more than this many scaled MADs beyond the bound; the tail of a single
# normal population lies about 0.2 beyond it.
_SEPARATE_POPULATION_MADS = 3.0
# The fewest cells that a reported population holds.
_MIN_POPULATION_CELLS = 10
# The most values that the bimodality check sorts.
_SPLIT_SAMPLE_SIZE = 1 << 20
# The remedy that warnings about empty droplets give, shared with the
# min_features_per_cell floor of a writable DataStore open.
RAW_DROPLET_REMEDY = (
    "If this is a raw (unfiltered) droplet matrix, most of its barcodes are "
    "empty droplets, and bounds computed over them keep the empty droplets and "
    "remove the real cells: call cells first with qc.call_cells, or import the "
    "filtered matrix (for example Cell Ranger's filtered_feature_bc_matrix), "
    "before QC filtering."
)
# The remedy of the same warnings for cells that were already called from
# droplets, where empty droplets are not the likely cause.
CALLED_CELLS_REMEDY = (
    "These cells were called from droplets, so a second population is more "
    "likely doublets, another cell type, or barcodes that cell calling kept near "
    "the knee than empty droplets: check those cells before trusting the bounds."
)


@dataclass(frozen=True, slots=True)
class MetricPolicy:
    """How MAD filtering bounds one QC metric.

    Attributes:
        transform: ``"log1p"`` bounds ``log1p`` of the values, ``"identity"``
            the values themselves.
        bound_direction: ``"two_sided"``, ``"lower"``, or ``"upper"``: which
            bounds are applied.
        percentage: Whether the metric is a percentage, whose upper bound is
            kept within 0 to 100. Provenance does not record it.
    """

    transform: str
    bound_direction: str
    percentage: bool = False

    def record(self) -> dict[str, str]:
        """Return the policy as provenance records it."""
        return {"transform": self.transform, "bound_direction": self.bound_direction}


# The default policy of each metric kind: counts and feature counts are
# bounded on the log1p scale from both sides, percentages from above.
_KIND_POLICIES: Mapping[str, MetricPolicy] = MappingProxyType(
    {
        "counts": MetricPolicy("log1p", "two_sided"),
        "features": MetricPolicy("log1p", "two_sided"),
        "percentage": MetricPolicy("identity", "upper", percentage=True),
    }
)


class _MadProvenance(TypedDict):
    mad_scale: float
    metric_policies: dict[str, dict[str, str]]
    sample_sizes: dict[str, int]
    skip_reasons: dict[str, str]
    resolved_bounds: dict[str, dict[str, dict[str, object]]]
    warnings: list[str]


def gaussian_quantile_bounds(
    values: np.ndarray,
    min_p: float = 0.01,
    max_p: float = 0.99,
) -> tuple[float, float]:
    if not 0 < min_p < max_p < 1:
        raise ValueError("Gaussian filtering requires 0 < min_p < max_p < 1")
    median, deviation = float(np.median(values)), float(np.std(values))
    if deviation == 0:
        return median, median
    dist = norm(median, deviation)
    return float(dist.ppf(min_p)), float(dist.ppf(max_p))


def mad_bounds(values: np.ndarray, n_mads: float) -> tuple[float, float, float]:
    """Return ``(low, high, scaled_mad)`` for robust per-sample thresholds.

    ``scaled_mad`` is ``1.4826 * MAD``. Callers must treat a zero scaled MAD as a
    skip condition rather than applying a zero-width exclusive interval.
    """
    if not np.isfinite(n_mads) or n_mads <= 0:
        raise ValueError("n_mads must be finite and greater than 0")
    if not np.isfinite(values).all():
        raise ValueError("MAD input values must all be finite")
    median = float(np.median(values))
    scaled_mad = float(_MAD_SCALE * np.median(np.abs(values - median)))
    with np.errstate(over="ignore", invalid="ignore"):
        distance = float(n_mads * scaled_mad)
        low = float(median - distance)
        high = float(median + distance)
    if not np.isfinite([median, scaled_mad, low, high]).all():
        raise ValueError("n_mads produces non-finite MAD bounds")
    return low, high, scaled_mad


def _name_policy(attr: str) -> MetricPolicy:
    """Return the policy that a metric's name suggests.

    Names ending in ``nCounts`` or ``nFeatures`` are counts, names ending in
    ``percentMito`` or ``percentRibo`` are percentages, and other metrics get
    raw two-sided bounds.
    """
    for suffix in _COUNT_SUFFIXES:
        if attr == suffix or attr.endswith(f"_{suffix}"):
            return _KIND_POLICIES["counts"]
    for suffix in _PERCENT_SUFFIXES:
        if attr == suffix or attr.endswith(f"_{suffix}"):
            return _KIND_POLICIES["percentage"]
    return MetricPolicy("identity", "two_sided")


def validate_metric_policies(
    metric_policies: object,
    metrics: Sequence[str],
) -> dict[str, MetricPolicy]:
    """Check explicit MAD policies before any metric is read.

    Args:
        metric_policies: None, or a mapping from metric names to mappings with
            exactly the keys ``"transform"`` (``"log1p"`` or ``"identity"``)
            and ``"bound_direction"`` (``"two_sided"``, ``"lower"``, or
            ``"upper"``).
        metrics: The names of the metrics being filtered.

    Returns:
        The explicit policies by metric name, without the ``percentage``
        flag, which ``resolve_metric_policies`` sets.

    Raises:
        TypeError: If the policies or a policy is not a mapping.
        ValueError: If a name is not filtered, or a policy has other keys or
            values.
    """
    if metric_policies is None:
        return {}
    if not isinstance(metric_policies, Mapping):
        raise TypeError("metric_policies must map metric names to policies")
    unknown = [name for name in metric_policies if name not in metrics]
    if unknown:
        raise ValueError(
            f"metric_policies names metrics that are not filtered: {unknown!r}"
        )
    policies: dict[str, MetricPolicy] = {}
    for name, policy in metric_policies.items():
        if not isinstance(policy, Mapping):
            raise TypeError(
                f"The policy of {name!r} must map 'transform' and "
                "'bound_direction' to text"
            )
        if set(policy) != {"transform", "bound_direction"}:
            raise ValueError(
                f"The policy of {name!r} must have exactly the keys 'transform' "
                f"and 'bound_direction'; got {sorted(map(str, policy))!r}"
            )
        transform, direction = policy["transform"], policy["bound_direction"]
        if transform not in _TRANSFORMS:
            raise ValueError(
                f"The transform of {name!r} must be one of {list(_TRANSFORMS)!r}; "
                f"got {transform!r}"
            )
        if direction not in _BOUND_DIRECTIONS:
            raise ValueError(
                f"The bound_direction of {name!r} must be one of "
                f"{list(_BOUND_DIRECTIONS)!r}; got {direction!r}"
            )
        policies[name] = MetricPolicy(str(transform), str(direction))
    return policies


def resolve_metric_policies(
    metrics: Sequence[str],
    *,
    kinds: Mapping[str, MetricKind],
    explicit: Mapping[str, MetricPolicy],
) -> dict[str, MetricPolicy]:
    """Return the MAD policy of each metric, warning about guessed ones.

    An explicit policy wins. Otherwise a metric of a known kind, one that
    CyteArc wrote and so knows to be counts, feature counts, or a
    percentage, gets that kind's policy. Any other metric gets the policy its
    name suggests (``_name_policy``), and one warning names those metrics
    and how to set their policies. A metric is a percentage, whose upper
    bound stays within 0 to 100, when its kind or, for an unknown kind, its
    name says so.

    Args:
        metrics: The names of the metrics being filtered.
        kinds: The kind of each metric whose kind is known.
        explicit: Validated explicit policies by metric name.
    """
    policies: dict[str, MetricPolicy] = {}
    guessed: list[str] = []
    for name in metrics:
        kind = kinds.get(name)
        default = _name_policy(name) if kind is None else _KIND_POLICIES[kind]
        chosen = explicit.get(name)
        if chosen is None:
            chosen = default
            if kind is None:
                guessed.append(name)
        policies[name] = MetricPolicy(
            chosen.transform, chosen.bound_direction, default.percentage
        )
    if guessed:
        described = ", ".join(
            f"{name!r} ({_describe_policy(policies[name])})" for name in guessed
        )
        warn(
            f"QC metrics {guessed!r} were not written by CyteArc's preparation or "
            "by qc.feature_percentage, so their MAD policies come from their "
            f"names: {described}. Pass metric_policies to set them, for example "
            "{'transform': 'log1p', 'bound_direction': 'two_sided'} for counts "
            "or {'transform': 'identity', 'bound_direction': 'upper'} for a "
            "percentage."
        )
    return policies


def _describe_policy(policy: MetricPolicy) -> str:
    scale = "log1p values" if policy.transform == "log1p" else "raw values"
    sides = {
        "two_sided": "two-sided bounds",
        "lower": "a lower bound",
        "upper": "an upper bound",
    }[policy.bound_direction]
    return f"{scale} with {sides}"


def validated_work_scale(
    values: np.ndarray,
    *,
    attr: str,
    transform: str,
) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError(f"QC values in '{attr}' contain non-finite entries")
    if transform == "log1p":
        if (raw < 0).any():
            raise ValueError(f"QC values in '{attr}' must be non-negative before log1p")
        work = np.asarray(np.log1p(raw), dtype=float)
    else:
        work = raw
    return work


def from_work_scale(bound: float, transform: str) -> float:
    if transform == "log1p":
        with np.errstate(over="ignore", invalid="ignore"):
            resolved = float(np.expm1(bound))
    else:
        resolved = float(bound)
    if not np.isfinite(resolved):
        raise ValueError(
            f"MAD bound is non-finite after converting from the {transform} scale"
        )
    return resolved


def clamp_metric_bound(
    bound: float,
    *,
    transform: str,
    is_percent: bool,
) -> float:
    if not np.isfinite(bound):
        raise ValueError("Resolved MAD bounds must be finite")
    if transform == "log1p":
        resolved = max(0.0, bound)
    elif is_percent:
        resolved = min(100.0, max(0.0, bound))
    else:
        resolved = bound
    return resolved


def validated_sample_labels(
    sample_labels: np.ndarray,
    active: np.ndarray,
    *,
    label_name: str = "sample labels",
) -> np.ndarray:
    """Check the sample labels of active rows and return an object copy.

    Active labels must be present, non-blank, finite, and of one kind: text,
    UTF-8 bytes, booleans, integers, or floats. They are Python scalars in the
    returned copy.
    """
    labels = np.asarray(sample_labels)
    active_mask = np.asarray(active)
    if labels.ndim != 1 or active_mask.ndim != 1 or labels.shape != active_mask.shape:
        raise ValueError("Sample labels and active selection must be aligned vectors")
    normalized = labels.astype(object, copy=True)
    # A typed array holds one kind of label, so each distinct value needs one
    # check; object arrays can mix kinds and are checked cell by cell.
    if labels.dtype != object:
        kinds = {
            _sample_label_kind(value, label_name)
            for value in pd.unique(labels[active_mask])
        }
    else:
        kinds = set()
        for index in np.flatnonzero(active_mask):
            value = labels[index]
            if isinstance(value, np.generic):
                value = value.item()
            kinds.add(_sample_label_kind(value, label_name))
            normalized[index] = value
    if len(kinds) > 1:
        raise ValueError(
            f"{label_name} must use one consistent label type among active cells"
        )
    return normalized


def unique_label_keys(labels: Iterable[object], *, label_name: str) -> list[str]:
    """Return the text that provenance records for each distinct label.

    Byte strings decode as UTF-8 and other labels use ``str``, so sample,
    capture, and group labels share one key encoding.

    Raises:
        ValueError: If two distinct labels have the same key.
    """
    keys = []
    for label in labels:
        value = label.item() if isinstance(label, np.generic) else label
        keys.append(value.decode("utf-8") if isinstance(value, bytes) else str(value))
    if len(set(keys)) != len(keys):
        raise ValueError(f"{label_name} collide after provenance encoding")
    return keys


def _sample_label_kind(value: object, label_name: str) -> str:
    """Return the kind of one sample label, rejecting missing labels."""
    if isinstance(value, np.generic):
        value = value.item()
    missing = pd.isna(value)
    if isinstance(missing, bool | np.bool_) and bool(missing):
        raise ValueError(f"{label_name} contains missing labels among active cells")
    if isinstance(value, bytes):
        try:
            decoded = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{label_name} contains a non-UTF-8 bytes label") from exc
        if decoded.strip() == "":
            raise ValueError(f"{label_name} contains missing labels among active cells")
        return "bytes"
    if isinstance(value, str):
        if value.strip() == "":
            raise ValueError(f"{label_name} contains missing labels among active cells")
        return "str"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{label_name} must contain finite labels")
        return "float"
    raise TypeError(
        f"{label_name} contains unsupported label type {type(value).__name__!r}"
    )


def _apply_bounds(
    values: np.ndarray,
    low: float | str | None,
    high: float | str | None,
    *,
    keep_bounds: bool = False,
) -> np.ndarray:
    """Return a boolean mask for one-dimensional values within bounds."""
    resolved = np.asarray(values)
    if resolved.ndim != 1:
        raise ValueError("Filter values must be a one-dimensional array")
    return within_bounds(resolved, low, high, keep_bounds=keep_bounds)


def _split_two_groups(values: np.ndarray) -> tuple[float, float] | None:
    """Split values into the low and high group that best separate them.

    The split maximizes the between-group sum of squares, the exact
    one-dimensional two-means split, and only falls between distinct values.
    More than ``_SPLIT_SAMPLE_SIZE`` values are split by an evenly strided
    sample of them, which bounds the memory of the sort.

    Returns:
        Ashman's D, the difference of the group means over the root mean of
        their variances, and the largest value of the low group; None when
        every value is equal.
    """
    step = -(-values.size // _SPLIT_SAMPLE_SIZE)
    ordered = np.sort(np.asarray(values[::step], dtype=float))
    n_values = ordered.size
    distinct = ordered[1:] > ordered[:-1]
    if not distinct.any():
        return None
    prefix = np.cumsum(ordered - ordered.mean())[:-1]
    sizes = np.arange(1, n_values)
    score = np.where(distinct, prefix**2 / (sizes * (n_values - sizes)), -np.inf)
    n_low = int(np.argmax(score)) + 1
    low, high = ordered[:n_low], ordered[n_low:]
    spread = float(np.sqrt((low.var() + high.var()) / 2))
    separation = np.inf if spread == 0.0 else float((high.mean() - low.mean()) / spread)
    return separation, float(ordered[n_low - 1])


def _population_warning(
    *,
    attr: str,
    group_label: str,
    raw: np.ndarray,
    work: np.ndarray,
    low: float | None,
    high: float | None,
    high_work: float,
    scaled_mad: float,
    remedy: str,
) -> str | None:
    """Describe a second population that the MAD bounds of a metric miss.

    Two signs are reported for a log1p metric, such as counts of a raw
    droplet matrix, where empty droplets set the bounds: at least
    ``_MIN_POPULATION_CELLS`` cells above the upper bound whose median lies
    more than ``_SEPARATE_POPULATION_MADS`` scaled MADs beyond it, and a
    strongly bimodal metric whose low group the bounds mostly keep, when
    each group holds at least ``_MIN_POPULATION_CELLS`` cells. Removing a low
    group of debris below the bounds is not reported.
    """
    if high is not None:
        above = work[raw >= high]
        beyond = high_work + _SEPARATE_POPULATION_MADS * scaled_mad
        if above.size >= _MIN_POPULATION_CELLS and float(np.median(above)) > beyond:
            return (
                f"{group_label}: {above.size} cells above the upper {attr!r} "
                f"bound of {high:.6g} form a separate population (median "
                f"{float(np.median(raw[raw >= high])):.6g}), which the MAD bounds "
                f"remove as outliers. {remedy} Otherwise check whether "
                "those cells are a real population that the bounds should keep."
            )
    split = _split_two_groups(work)
    if split is None:
        return None
    separation, threshold = split
    in_low = work <= threshold
    n_low = int(np.count_nonzero(in_low))
    n_high = work.size - n_low
    if separation < _BIMODAL_SEPARATION or min(n_low, n_high) < _MIN_POPULATION_CELLS:
        return None
    kept = within_bounds(raw, low, high, keep_bounds=False)
    if 2 * int(np.count_nonzero(kept & in_low)) <= n_low:
        return None
    return (
        f"{group_label}: {attr!r} is strongly bimodal: {n_low} cells form a low "
        f"population (median {float(np.median(raw[in_low])):.6g}) and {n_high} a "
        f"high population (median {float(np.median(raw[~in_low])):.6g}), and the "
        "MAD bounds keep most of the low population. MAD bounds assume one "
        f"population of cells. {remedy}"
    )


def sample_aware_mad_mask(
    *,
    values_by_attr: dict[str, np.ndarray],
    sample_labels: np.ndarray | None,
    active: np.ndarray,
    n_mads: float,
    min_cells_per_sample: int,
    attrs: list[str],
    metric_policies: Mapping[str, MetricPolicy] | None = None,
    remedy: str = RAW_DROPLET_REMEDY,
) -> tuple[np.ndarray, _MadProvenance]:
    """Build one cell mask from pooled or per-sample MAD bounds.

    Inactive cells are left ``True`` in the returned mask so callers can
    intersect with the current selection via ``update_key``.
    ``metric_policies`` gives each metric's policy; without it, each metric
    gets the policy its name suggests. Each group's bounds of a log1p
    metric are checked for a second population that they miss (see
    ``_population_warning``), with a warning that provenance does not record.
    """
    n_cells = active.shape[0]
    pooled = sample_labels is None
    sample_labels = (
        np.full(n_cells, "all")
        if sample_labels is None
        else validated_sample_labels(sample_labels, active)
    )
    keep = np.ones(n_cells, dtype=bool)
    resolved_policies = (
        {attr: _name_policy(attr) for attr in attrs}
        if metric_policies is None
        else {attr: metric_policies[attr] for attr in attrs}
    )
    policies = {attr: policy.record() for attr, policy in resolved_policies.items()}
    raw_values_by_attr: dict[str, np.ndarray] = {}
    work_values_by_attr: dict[str, np.ndarray] = {}
    for attr in attrs:
        raw = np.asarray(values_by_attr[attr], dtype=float)
        raw_values_by_attr[attr] = raw
        work = np.empty(n_cells, dtype=float)
        work[active] = validated_work_scale(
            raw[active],
            attr=attr,
            transform=policies[attr]["transform"],
        )
        work_values_by_attr[attr] = work
    sample_sizes: dict[str, int] = {}
    skip_reasons: dict[str, str] = {}
    resolved_bounds: dict[str, dict[str, dict[str, object]]] = {}
    warnings: list[str] = []
    population_warnings: list[str] = []

    # Factorizing keeps first-seen order, which keeps provenance deterministic,
    # and turns per-sample label comparisons into integer comparisons.
    active_idx = np.flatnonzero(active)
    sample_codes, sample_uniques = pd.factorize(sample_labels[active])
    sample_keys = unique_label_keys(sample_uniques, label_name="Sample labels")

    for code, sample_key in enumerate(sample_keys):
        sample_idx = active_idx[sample_codes == code]
        sample_sizes[sample_key] = int(sample_idx.shape[0])
        resolved_bounds[sample_key] = {}
        group_label = "Selected cells" if pooled else f"Sample '{sample_key}'"

        if sample_idx.shape[0] < min_cells_per_sample:
            skip_reasons[sample_key] = "insufficient_cells"
            warnings.append(
                f"{group_label}: fewer than {min_cells_per_sample} "
                "active cells; retaining them without MAD filtering"
            )
            continue

        sample_keep = np.ones(sample_idx.shape[0], dtype=bool)
        for attr in attrs:
            policy = policies[attr]
            transform = policy["transform"]
            direction = policy["bound_direction"]
            raw = raw_values_by_attr[attr][sample_idx]
            work = work_values_by_attr[attr][sample_idx]
            low_t, high_t, scaled_mad = mad_bounds(work, n_mads)
            low: float | None = None
            high: float | None = None
            if scaled_mad == 0.0:
                resolved_bounds[sample_key][attr] = {
                    "low": None,
                    "high": None,
                    "skip_reason": "zero_mad",
                    "transform": transform,
                    "bound_direction": direction,
                    "scaled_mad": 0.0,
                }
                warnings.append(
                    f"{group_label}: zero MAD for '{attr}'; "
                    "retaining cells for this metric"
                )
            else:
                # A lower bound below zero already keeps every value, so only
                # an upper bound is kept within the range of a percentage.
                if direction != "upper":
                    low = clamp_metric_bound(
                        from_work_scale(low_t, transform),
                        transform=transform,
                        is_percent=False,
                    )
                if direction != "lower":
                    high = clamp_metric_bound(
                        from_work_scale(high_t, transform),
                        transform=transform,
                        is_percent=resolved_policies[attr].percentage,
                    )
                resolved_bounds[sample_key][attr] = {
                    "low": low,
                    "high": high,
                    "skip_reason": None,
                    "transform": transform,
                    "bound_direction": direction,
                    "scaled_mad": scaled_mad,
                }
                sample_keep &= _apply_bounds(raw, low, high)
            if transform == "log1p":
                message = _population_warning(
                    attr=attr,
                    group_label=group_label,
                    raw=raw,
                    work=work,
                    low=low,
                    high=high,
                    high_work=high_t,
                    scaled_mad=scaled_mad,
                    remedy=remedy,
                )
                if message is not None:
                    population_warnings.append(message)

        keep[sample_idx] = sample_keep

    for message in population_warnings:
        warn(message)

    provenance: _MadProvenance = {
        "mad_scale": _MAD_SCALE,
        "metric_policies": policies,
        "sample_sizes": sample_sizes,
        "skip_reasons": skip_reasons,
        "resolved_bounds": resolved_bounds,
        "warnings": warnings,
    }
    return keep, provenance


@dataclass(frozen=True, slots=True)
class CellFilterResult:
    """Rows retained by one QC filtering rule and the bounds it resolved.

    ``retained`` aligns with the rows given to ``filter_cell_metrics``.
    ``gaussian_bounds`` and ``mad_provenance`` hold the resolved Gaussian
    bounds or the MAD provenance of those methods. ``sample_labels`` holds the
    validated MAD sample labels, normalized to Python scalars.
    """

    retained: np.ndarray
    gaussian_bounds: dict[str, dict[str, float]] | None
    mad_provenance: _MadProvenance | None
    sample_labels: np.ndarray | None


def _check_filter_bound(value: object, name: str) -> None:
    if value is None or isinstance(value, str):
        return
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} values must be finite numbers, text, or None")
    if not np.isfinite(float(value)):
        raise ValueError(f"{name} values must be finite; use None for no bound")


def validate_named_cell_artifacts(
    values: Iterable[NamedCellArtifact] | None,
    *,
    expected_kind: str,
    label: str,
) -> list[NamedCellArtifact]:
    """Check that named cell artifacts reference one kind under unique names.

    Args:
        values: Named cell-aligned artifacts, or None for none.
        expected_kind: The artifact kind every value must reference.
        label: The argument name that error messages use.

    Returns:
        The values as a list.

    Raises:
        TypeError: If a value is not a ``NamedCellArtifact``.
        ValueError: If a value references another kind or repeats a name.
    """
    sources = list(values or ())
    names: set[str] = set()
    for source in sources:
        if not isinstance(source, NamedCellArtifact):
            raise TypeError(f"{label} must contain NamedCellArtifact values")
        if source.artifact.kind != expected_kind:
            raise ValueError(f"{label} must reference {expected_kind!r} artifacts")
        if source.name in names:
            raise ValueError(f"{label} must use unique semantic names")
        names.add(source.name)
    return sources


def validate_cell_filter_sources(
    attrs: Iterable[str],
    artifact_metrics: Iterable[NamedCellArtifact] | None = None,
    *,
    sample_column: str | None = None,
    sample_artifact: NamedCellArtifact | None = None,
) -> tuple[list[str], list[NamedCellArtifact], NamedCellArtifact | None]:
    """Check the metric and sample sources of a cell filter before any read.

    Metadata metrics are distinct column names. Artifact metrics reference
    ``quality_metric`` artifacts under unique names that no metadata metric
    uses. Sample labels come from a metadata column or from a named
    ``hto_identity`` artifact, not both, and a sample artifact name differs
    from every artifact metric name.

    Returns:
        The metadata metrics, the artifact metrics, and the sample artifact.

    Raises:
        TypeError: If a metric or sample source has an invalid type.
        ValueError: If names repeat or collide, an artifact has another kind,
            or both sample sources are given.
    """
    attrs_list = list(attrs)
    if any(not isinstance(attr, str) for attr in attrs_list):
        raise TypeError("attrs must contain only column names")
    if len(set(attrs_list)) != len(attrs_list):
        raise ValueError("attrs must not contain duplicate columns")
    metrics = validate_named_cell_artifacts(
        artifact_metrics,
        expected_kind="quality_metric",
        label="artifact_metrics",
    )
    sample = (
        None
        if sample_artifact is None
        else validate_named_cell_artifacts(
            [sample_artifact],
            expected_kind="hto_identity",
            label="sample_artifact",
        )[0]
    )
    if sample_column is not None and sample is not None:
        raise ValueError("sample_column and sample_artifact are mutually exclusive")
    metric_names = {source.name for source in metrics}
    if sample is not None and sample.name in metric_names:
        raise ValueError("Sample and metric artifact names must be distinct")
    collisions = sorted(metric_names.intersection(attrs_list))
    if collisions:
        raise ValueError(
            f"Metadata and artifact QC metrics must use distinct names: {collisions}"
        )
    return attrs_list, metrics, sample


def validate_filter_bounds(
    lows: Sequence[object],
    highs: Sequence[object],
    *,
    keep_bounds: object,
) -> None:
    """Check manual filtering bounds before any metadata is read.

    Numeric bounds must be finite and cannot be booleans. Text bounds compare
    text columns lexically. A lower bound cannot exceed its upper bound.
    """
    if not isinstance(keep_bounds, bool):
        raise TypeError("keep_bounds must be a boolean")
    for low, high in zip(lows, highs, strict=True):
        _check_filter_bound(low, "lows")
        _check_filter_bound(high, "highs")
        if low is None or high is None:
            continue
        if isinstance(low, str) != isinstance(high, str):
            raise TypeError("Paired lows and highs must both be numbers or both text")
        if low > high:  # type: ignore[operator]
            raise ValueError("A lower bound cannot exceed its upper bound")


def validate_bound_types(
    attr: str,
    dtype: np.dtype,
    low: object,
    high: object,
) -> None:
    """Check that a column's bounds can be compared with its values.

    Bounds of a numeric or boolean column must be numbers, and bounds of a
    text column must be text; None leaves a side open. Call it with bounds
    that ``validate_filter_bounds`` accepted.

    Raises:
        TypeError: If a bound does not match the column, or the column holds
            values that bounds cannot compare, such as dates.
    """
    kind = np.dtype(dtype).kind
    bounds = [bound for bound in (low, high) if bound is not None]
    if kind in "biuf":
        if any(isinstance(bound, str) for bound in bounds):
            raise TypeError(
                f"Column {attr!r} holds numeric values, so its bounds must be "
                f"numbers or None; got {[low, high]!r}"
            )
    elif kind in "OSUT":
        if any(not isinstance(bound, str) for bound in bounds):
            raise TypeError(
                f"Column {attr!r} holds text, so its bounds must be text or "
                f"None; got {[low, high]!r}"
            )
    elif bounds:
        raise TypeError(
            f"Column {attr!r} holds {np.dtype(dtype)} values, which bounds cannot "
            "filter"
        )


def _require_finite_metrics(
    values_by_attr: Mapping[str, np.ndarray],
    complete: np.ndarray,
) -> None:
    """Reject non-finite values among rows that inform automatic bounds."""
    n_complete = int(np.count_nonzero(complete))
    for attr, values in values_by_attr.items():
        array = np.asarray(values)
        if array.dtype.kind not in "fc":
            continue
        n_nonfinite = int(np.count_nonzero(~np.isfinite(array[complete])))
        if n_nonfinite:
            raise ValueError(
                f"QC metric {attr!r} has {n_nonfinite} non-finite value(s) among "
                f"{n_complete} selected cells with recorded values. Percentages "
                "are undefined for cells without counts, so exclude zero-count "
                "cells first: pass a cell selection that requires nCounts > 0."
            )


def filter_cell_metrics(
    values_by_attr: Mapping[str, np.ndarray],
    missing_by_attr: Mapping[str, np.ndarray],
    active: np.ndarray,
    *,
    method: FilterMethod,
    lows: Sequence[float | None] = (),
    highs: Sequence[float | None] = (),
    keep_bounds: bool = False,
    min_p: float = 0.01,
    max_p: float = 0.99,
    sample_labels: np.ndarray | None = None,
    sample_missing: np.ndarray | None = None,
    sample_label_name: str = "sample labels",
    n_mads: float = 3.0,
    min_cells_per_sample: int = 20,
    metric_policies: Mapping[str, MetricPolicy] | None = None,
    remedy: str = RAW_DROPLET_REMEDY,
) -> CellFilterResult:
    """Apply one QC filtering rule to metric rows aligned with ``active``.

    Every metric, mask, and label vector must align with ``active``. A row
    flagged in ``missing_by_attr`` has no recorded value for that metric. It
    never passes a filter and does not inform Gaussian or MAD bounds.
    Automatic bounds require finite metrics on the remaining active rows. MAD
    filtering rejects an active row flagged in ``sample_missing``, and takes
    each metric's policy from ``metric_policies``, or from its name without
    them. Manual bounds must match the type of their values. The result must
    retain at least one active row; automatic bounds that remove more than
    half of the active rows give a warning. ``remedy`` ends the warnings about
    a second population, which by default suspect a raw droplet matrix.
    """
    active_mask = np.asarray(active, dtype=bool)
    attrs = list(values_by_attr)
    metric_missing = np.zeros(active_mask.shape[0], dtype=bool)
    for missing in missing_by_attr.values():
        metric_missing |= np.asarray(missing, dtype=bool)
    complete = active_mask & ~metric_missing
    keep = ~metric_missing
    gaussian_bounds: dict[str, dict[str, float]] | None = None
    mad_provenance: _MadProvenance | None = None
    validated_labels: np.ndarray | None = None
    if method == "manual":
        for attr, low, high in zip(attrs, lows, highs, strict=True):
            values = np.asarray(values_by_attr[attr])
            validate_bound_types(attr, values.dtype, low, high)
            keep &= _apply_bounds(values, low, high, keep_bounds=keep_bounds)
    elif method in ("gaussian", "mad"):
        if not complete.any():
            raise ValueError(
                "Cell filtering has no selected cells with complete metrics"
            )
        if method == "mad" and sample_labels is not None:
            if sample_missing is not None and np.any(
                active_mask & np.asarray(sample_missing, dtype=bool)
            ):
                raise ValueError(
                    f"{sample_label_name} contains missing labels among active cells"
                )
            validated_labels = validated_sample_labels(
                sample_labels,
                active_mask,
                label_name=sample_label_name,
            )
        _require_finite_metrics(values_by_attr, complete)
        if method == "gaussian":
            gaussian_bounds = {}
            for attr in attrs:
                values = np.asarray(values_by_attr[attr])
                low, high = gaussian_quantile_bounds(values[complete], min_p, max_p)
                if not np.isfinite([low, high]).all():
                    raise ValueError(
                        f"QC metric {attr!r} produced non-finite Gaussian bounds"
                    )
                gaussian_bounds[attr] = {"low": low, "high": high}
                keep &= _apply_bounds(values, low, high, keep_bounds=low == high)
        elif attrs:
            # Typed labels keep their dtype so the second validation checks
            # each distinct label once instead of every cell.
            mad_keep, mad_provenance = sample_aware_mad_mask(
                values_by_attr=dict(values_by_attr),
                sample_labels=sample_labels,
                active=complete,
                n_mads=n_mads,
                min_cells_per_sample=min_cells_per_sample,
                attrs=attrs,
                metric_policies=metric_policies,
                remedy=remedy,
            )
            keep &= mad_keep
    else:
        raise ValueError("method must be 'manual', 'gaussian', or 'mad'")
    retained = np.asarray(active_mask & keep, dtype=bool)
    if not retained.any():
        raise ValueError("Cell filtering removed every selected cell")
    n_active = int(np.count_nonzero(active_mask))
    n_removed = n_active - int(np.count_nonzero(retained))
    if method != "manual" and n_removed > _MAJORITY_REMOVED_FRACTION * n_active:
        warn(
            f"Automatic QC bounds removed {n_removed} of {n_active} selected cells "
            f"({n_removed / n_active:.0%}), so most selected cells lie outside "
            "bounds meant to remove outliers. Check the resolved bounds and the "
            f"metric distributions. {remedy}"
        )
    return CellFilterResult(
        retained=retained,
        gaussian_bounds=gaussian_bounds,
        mad_provenance=mad_provenance,
        sample_labels=validated_labels,
    )
