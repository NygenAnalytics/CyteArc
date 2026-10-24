from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..matrix import ChunkedArray
from ..storage.refs import ArtifactRef
from ..utils.arrays import read_only_copy
from .curves import curve_values_at, project_onto_curve


@dataclass(frozen=True, slots=True)
class _PseudotimeAggregationFeatureIdentity:
    names: np.ndarray
    ids: np.ndarray

    def __post_init__(self) -> None:
        names = np.asarray(self.names)
        ids = np.asarray(self.ids)
        if names.ndim != 1 or ids.ndim != 1 or names.shape != ids.shape:
            raise ValueError("Frozen feature names and IDs must align")
        object.__setattr__(self, "names", read_only_copy(names))
        object.__setattr__(self, "ids", read_only_copy(ids))


class _PseudotimeAggregationIdentityCarrier:
    __slots__ = ("_feature_identity",)


@dataclass(frozen=True, slots=True, eq=False)
class FateMappingResult:
    """Fate probabilities loaded from an immutable artifact."""

    ref: ArtifactRef
    graph: ArtifactRef
    pseudotime: ArtifactRef
    sink_labels_artifact: ArtifactRef
    cell_selection: ArtifactRef
    sink_labels: tuple[Any, ...]
    values: np.ndarray = field(repr=False)
    valid: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("Fate probabilities must be two-dimensional")
        if self.valid.ndim != 1:
            raise ValueError("Fate validity must be one-dimensional")
        if self.values.shape[0] != self.valid.shape[0]:
            raise ValueError("Fate probabilities and validity rows must align")
        n_sinks = self.values.shape[1]
        if n_sinks != len(self.sink_labels):
            raise ValueError("Fate probability columns and sink labels must align")


@dataclass(frozen=True, slots=True, eq=False)
class PseudotimeScoreResult:
    """Pseudotime values loaded from an immutable artifact."""

    ref: ArtifactRef
    graph: ArtifactRef
    cell_selection: ArtifactRef
    values: np.ndarray = field(repr=False)
    valid: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        if self.values.ndim != 1 or self.valid.ndim != 1:
            raise ValueError("Pseudotime values and validity must be one-dimensional")
        if self.values.shape != self.valid.shape:
            raise ValueError("Pseudotime values and validity must have the same shape")


@dataclass(frozen=True, slots=True, eq=False)
class PseudotimeMarkerResult:
    """Pseudotime correlation table loaded from an immutable artifact."""

    ref: ArtifactRef
    table: pd.DataFrame = field(repr=False)
    assay: str
    cell_selection: ArtifactRef
    feature_selection: ArtifactRef
    pseudotime: ArtifactRef

    def __post_init__(self) -> None:
        if (
            not isinstance(self.feature_selection, ArtifactRef)
            or self.feature_selection.kind != "feature_selection"
            or self.feature_selection.scope != "assay"
            or self.feature_selection.assay != self.assay
        ):
            raise ValueError(
                "Pseudotime marker feature selection must belong to its assay"
            )
        required = {
            "feature_index",
            "feature_name",
            "r_value",
            "p_value",
        }
        missing = required.difference(self.table.columns)
        if missing:
            raise ValueError(
                "Pseudotime marker table is missing columns: "
                + ", ".join(sorted(missing))
            )


@dataclass(frozen=True, slots=True, eq=False)
class PseudotimeAggregationResult(_PseudotimeAggregationIdentityCarrier):
    """Lazy pseudotime aggregation loaded from an immutable artifact."""

    ref: ArtifactRef
    data: ChunkedArray = field(repr=False)
    feature_indices: np.ndarray = field(repr=False)
    feature_clusters: np.ndarray = field(repr=False)
    assay: str
    cell_selection: ArtifactRef
    feature_selection: ArtifactRef
    pseudotime: ArtifactRef

    def __post_init__(self) -> None:
        if (
            not isinstance(self.feature_selection, ArtifactRef)
            or self.feature_selection.kind != "feature_selection"
            or self.feature_selection.scope != "assay"
            or self.feature_selection.assay != self.assay
        ):
            raise ValueError(
                "Pseudotime aggregation feature selection must belong to its assay"
            )
        if len(self.data.shape) != 2:
            raise ValueError("Pseudotime aggregation data must be two-dimensional")
        if self.feature_indices.ndim != 1 or self.feature_clusters.ndim != 1:
            raise ValueError("Feature indices and clusters must be one-dimensional")
        n_features = int(self.data.shape[0])
        if len(self.feature_indices) != n_features:
            raise ValueError("Feature indices do not align with aggregation rows")
        if len(self.feature_clusters) != n_features:
            raise ValueError("Feature clusters do not align with aggregation rows")

    def _attach_feature_identity(
        self,
        names: np.ndarray,
        ids: np.ndarray,
    ) -> None:
        if hasattr(self, "_feature_identity"):
            raise RuntimeError("Frozen feature identity is already attached")
        identity = _PseudotimeAggregationFeatureIdentity(names=names, ids=ids)
        if identity.names.shape != (int(self.data.shape[0]),):
            raise ValueError(
                "Frozen feature identity does not align with aggregation rows"
            )
        object.__setattr__(self, "_feature_identity", identity)

    @property
    def feature_names(self) -> np.ndarray:
        """Frozen feature names aligned to the aggregation rows."""
        identity = getattr(self, "_feature_identity", None)
        if not isinstance(identity, _PseudotimeAggregationFeatureIdentity):
            raise RuntimeError(
                "Frozen feature names are available on results loaded with "
                "ds.trajectory.load_aggregation"
            )
        return identity.names

    @property
    def feature_ids(self) -> np.ndarray:
        """Frozen feature IDs aligned to the aggregation rows."""
        identity = getattr(self, "_feature_identity", None)
        if not isinstance(identity, _PseudotimeAggregationFeatureIdentity):
            raise RuntimeError(
                "Frozen feature IDs are available on results loaded with "
                "ds.trajectory.load_aggregation"
            )
        return identity.ids


_LINEAGE_TABLE_COLUMNS = frozenset(
    {
        "feature_index",
        "feature_name",
        "feature_id",
        "lineage",
        "r_value",
        "p_value",
        "p_value_adjusted",
    }
)


def _require_assay_features(
    feature_selection: ArtifactRef, assay: str, label: str
) -> None:
    if (
        not isinstance(feature_selection, ArtifactRef)
        or feature_selection.kind != "feature_selection"
        or feature_selection.scope != "assay"
        or feature_selection.assay != assay
    ):
        raise ValueError(f"{label} feature selection must belong to its assay")


def _require_table_columns(
    table: pd.DataFrame, required: frozenset[str], label: str
) -> None:
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            f"{label} table is missing columns: " + ", ".join(sorted(missing))
        )


@dataclass(frozen=True, slots=True, eq=False)
class LineageMarkerResult:
    """Fate-weighted associations of features with pseudotime, per lineage.

    ``table`` holds one row per tested or untested selected feature and
    lineage. ``effective_cells`` gives each lineage's Kish effective number of
    cells, which its p-values use.
    """

    ref: ArtifactRef
    table: pd.DataFrame = field(repr=False)
    assay: str
    lineages: tuple[Any, ...]
    effective_cells: np.ndarray = field(repr=False)
    cell_selection: ArtifactRef
    feature_selection: ArtifactRef
    fate: ArtifactRef

    def __post_init__(self) -> None:
        _require_assay_features(self.feature_selection, self.assay, "Lineage-marker")
        _require_table_columns(self.table, _LINEAGE_TABLE_COLUMNS, "Lineage-marker")
        if self.effective_cells.shape != (len(self.lineages),):
            raise ValueError("Effective cell counts must align with the lineages")
        object.__setattr__(
            self, "effective_cells", read_only_copy(self.effective_cells)
        )


@dataclass(frozen=True, slots=True, eq=False)
class LineageDriverResult:
    """Correlations of features with each lineage's fate probability.

    ``table`` adds the Fisher-z confidence interval of each correlation,
    ``ci_low`` and ``ci_high``. ``n_cells`` counts the cells that were
    correlated, and ``driver_cells`` is the selection they were drawn from.
    """

    ref: ArtifactRef
    table: pd.DataFrame = field(repr=False)
    assay: str
    lineages: tuple[Any, ...]
    n_cells: int
    cell_selection: ArtifactRef
    driver_cells: ArtifactRef
    feature_selection: ArtifactRef
    fate: ArtifactRef

    def __post_init__(self) -> None:
        _require_assay_features(self.feature_selection, self.assay, "Lineage-driver")
        _require_table_columns(
            self.table,
            _LINEAGE_TABLE_COLUMNS | {"ci_low", "ci_high"},
            "Lineage-driver",
        )
        if self.n_cells < 1:
            raise ValueError("Lineage drivers need at least one correlated cell")


@dataclass(frozen=True, slots=True, eq=False)
class LineageAggregationResult:
    """Fate-weighted feature profiles along each lineage, with feature modules.

    ``data`` holds the retained features by lineages by bins. Each lineage's
    bins split its fate weight into equal parts along pseudotime;
    ``bin_pseudotime`` is the weighted mean pseudotime of each bin and
    ``lineage_weight`` each lineage's total weight.
    """

    ref: ArtifactRef
    data: np.ndarray = field(repr=False)
    feature_indices: np.ndarray = field(repr=False)
    feature_clusters: np.ndarray = field(repr=False)
    feature_names: np.ndarray = field(repr=False)
    feature_ids: np.ndarray = field(repr=False)
    lineages: tuple[Any, ...]
    bin_pseudotime: np.ndarray = field(repr=False)
    lineage_weight: np.ndarray = field(repr=False)
    assay: str
    cell_selection: ArtifactRef
    feature_selection: ArtifactRef
    fate: ArtifactRef

    def __post_init__(self) -> None:
        _require_assay_features(
            self.feature_selection, self.assay, "Lineage-aggregation"
        )
        if self.data.ndim != 3 or self.data.shape[1] != len(self.lineages):
            raise ValueError("Lineage profiles must be features by lineages by bins")
        n_features = self.data.shape[0]
        for name in ("feature_indices", "feature_clusters", "feature_names"):
            if getattr(self, name).shape != (n_features,):
                raise ValueError(f"{name} does not align with the profiles")
        if self.feature_ids.shape != (n_features,):
            raise ValueError("feature_ids does not align with the profiles")
        if self.bin_pseudotime.shape != self.data.shape[1:]:
            raise ValueError("Bin pseudotime must be lineages by bins")
        if self.lineage_weight.shape != (len(self.lineages),):
            raise ValueError("Lineage weights must align with the lineages")
        for name in (
            "data",
            "feature_indices",
            "feature_clusters",
            "feature_names",
            "feature_ids",
            "bin_pseudotime",
            "lineage_weight",
        ):
            object.__setattr__(self, name, read_only_copy(getattr(self, name)))


@dataclass(frozen=True, slots=True, eq=False)
class CurveProjection:
    """Positions of cells on lineage curves.

    Each array is cells by lineages. ``progress`` is the rank of the nearest
    curve point, from zero at the start to one at the sink; ``pseudotime``
    the curve's fitted pseudotime there; and ``distance`` the Euclidean
    distance to it.
    """

    lineages: tuple[Any, ...]
    progress: np.ndarray = field(repr=False)
    pseudotime: np.ndarray = field(repr=False)
    distance: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        expected = self.progress.shape
        if len(expected) != 2 or expected[1] != len(self.lineages):
            raise ValueError("Curve positions must be rows by lineages")
        if self.pseudotime.shape != expected or self.distance.shape != expected:
            raise ValueError("Curve positions must be rows by lineages")
        for name in ("progress", "pseudotime", "distance"):
            object.__setattr__(self, name, read_only_copy(getattr(self, name)))


@dataclass(frozen=True, slots=True, eq=False)
class LineageCurvesResult:
    """Fate-weighted curves through reduced coordinates, one per lineage.

    ``points`` holds lineages by points by dimensions in the space of
    ``coordinates``, and ``curve_pseudotime`` the fitted pseudotime of each
    point. ``progress`` and ``distance`` place every scored cell on every
    curve, cells by lineages, and ``weights`` holds the fate probability
    with which each cell weighed in each curve's fit; unscored cells hold
    NaN and a false ``valid``.
    """

    ref: ArtifactRef
    fate: ArtifactRef
    coordinates: ArtifactRef
    cell_selection: ArtifactRef
    lineages: tuple[Any, ...]
    points: np.ndarray = field(repr=False)
    curve_pseudotime: np.ndarray = field(repr=False)
    progress: np.ndarray = field(repr=False)
    distance: np.ndarray = field(repr=False)
    weights: np.ndarray = field(repr=False)
    valid: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        n_lineages = len(self.lineages)
        if self.points.ndim != 3 or self.points.shape[0] != n_lineages:
            raise ValueError("Curve points must be lineages by points by dimensions")
        if self.points.shape[1] < 2:
            raise ValueError("Each curve needs at least two points")
        if self.curve_pseudotime.shape != self.points.shape[:2]:
            raise ValueError("Curve pseudotime must be lineages by points")
        if self.valid.ndim != 1:
            raise ValueError("Curve validity must be one-dimensional")
        expected = (self.valid.shape[0], n_lineages)
        if self.progress.shape != expected or self.distance.shape != expected:
            raise ValueError("Cell positions must be cells by lineages")
        if self.weights.shape != expected:
            raise ValueError("Cell weights must be cells by lineages")
        for name in (
            "points",
            "curve_pseudotime",
            "progress",
            "distance",
            "weights",
            "valid",
        ):
            object.__setattr__(self, name, read_only_copy(getattr(self, name)))

    @property
    def curve_progress(self) -> np.ndarray:
        """The rank of each curve point, from zero to one."""
        return np.linspace(0.0, 1.0, self.points.shape[1])

    def project(self, coordinates: np.ndarray, *, nthreads: int = 1) -> CurveProjection:
        """Place rows of coordinates on every curve.

        ``coordinates`` must lie in the space the curves were fitted in, with
        one column per dimension of ``points``, such as rows of the same
        reduction for cells that were not scored. Rows are projected in
        parallel over ``nthreads`` threads.

        Raises:
            ValueError: If the coordinates are not finite or have the wrong
                number of dimensions.
        """
        rows = np.asarray(coordinates, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != self.points.shape[2]:
            raise ValueError(
                f"coordinates must have {self.points.shape[2]} columns, one per "
                "curve dimension"
            )
        if not np.isfinite(rows).all():
            raise ValueError("coordinates must be finite")
        progress = self.curve_progress
        n_lineages = len(self.lineages)
        positions = np.empty((rows.shape[0], n_lineages), dtype=np.float64)
        distances = np.empty_like(positions)
        pseudotime = np.empty_like(positions)
        for lineage in range(n_lineages):
            position, distance = project_onto_curve(
                rows, self.points[lineage], progress, nthreads=nthreads
            )
            positions[:, lineage] = position
            distances[:, lineage] = distance
            pseudotime[:, lineage] = curve_values_at(
                position, progress, self.curve_pseudotime[lineage]
            )
        return CurveProjection(
            lineages=self.lineages,
            progress=positions,
            pseudotime=pseudotime,
            distance=distances,
        )


@dataclass(frozen=True, slots=True, eq=False)
class LineageTreeResult:
    """A minimum spanning tree over clusters and the lineages it proposes.

    ``edges`` lists the tree's edges by cluster label with their Slingshot
    distance. ``leaves`` are clusters joined to one other cluster; with a
    root, ``lineages`` are the paths from it to the other leaves. Leaves are
    candidates for sinks, not discovered fates: a tree must join every
    cluster, so similar terminal clusters can be chained.
    """

    ref: ArtifactRef
    labels: ArtifactRef
    coordinates: ArtifactRef
    cell_selection: ArtifactRef
    clusters: tuple[Any, ...]
    sizes: np.ndarray = field(repr=False)
    means: np.ndarray = field(repr=False)
    edges: pd.DataFrame = field(repr=False)
    root: Any
    ends: tuple[Any, ...]
    leaves: tuple[Any, ...]
    lineages: tuple[tuple[Any, ...], ...]

    def __post_init__(self) -> None:
        n_clusters = len(self.clusters)
        if self.sizes.shape != (n_clusters,):
            raise ValueError("Cluster sizes must align with the clusters")
        if self.means.ndim != 2 or self.means.shape[0] != n_clusters:
            raise ValueError("Cluster means must be clusters by dimensions")
        _require_table_columns(
            self.edges, frozenset({"source", "target", "length"}), "Lineage-tree"
        )
        if len(self.edges) != n_clusters - 1:
            raise ValueError("A tree over n clusters has n - 1 edges")
        for name in ("sizes", "means"):
            object.__setattr__(self, name, read_only_copy(getattr(self, name)))

    @property
    def candidate_sinks(self) -> tuple[Any, ...]:
        """Leaves other than the root, the clusters to check as sinks."""
        return tuple(leaf for leaf in self.leaves if not bool(leaf == self.root))
