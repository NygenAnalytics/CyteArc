"""Which lineages each cell follows, and where two lineages part.

The store holds a branching trajectory in two dimensions: a narrow trunk that
splits into branches A and B, which widen toward their ends, and a side
branch C off the middle of the trunk that no sink ends. The graph links each
cell to its five nearest neighbors with Gaussian weights. A pseudotime runs
from the trunk's first half to the second halves of A and B, its fate map
takes those halves as sinks, and curves follow both lineages. The known
lineages give each cell's expected membership: the trunk is shared, the ends
of A and B follow their own lineage, and C, whose cells split their fate
probabilities as trunk cells do, follows none.

Other layouts of the same trunk make the trunk split its probability
unevenly: a sink ten times smaller than the other, a branch four times longer
than the other, and fates of 4% and 7% of the trunk's probability beyond a
gap, with a short branch, a long branch or a large sink. A cell follows each
lineage while it keeps half the lineage's trunk probability, so tests check
that every trunk cell follows every lineage, that a cell that follows one
lineage alone holds it as its most likely fate, and where the rule lets late
cells keep a common lineage. Two trajectories in separate graph components,
with a fate map that a score directs, check that each lineage takes its trunk
from the components of its own sink cells. A slow test runs the lineage
tutorials' analysis of the pancreas fixture, whose trunk gives Delta 0.04
against 0.45 for Beta.
"""

import math
from dataclasses import dataclass, replace
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cytearc.metadata import MetaData
from cytearc.metadata.selection import cell_selection_indices
from cytearc.storage.artifacts import ArtifactRef, artifact_group, list_artifacts
from cytearc.storage.selections import (
    resolve_metadata_snapshot,
    resolve_stored_selection_artifact,
)
from cytearc.trajectory import LineageMembershipResult
from cytearc.trajectory.artifacts import LINEAGE_MEMBERSHIP_PAYLOAD
from cytearc.trajectory.membership import (
    BRANCH_POINT_COLUMNS,
    branch_point,
    branch_points,
    branch_window,
    distance_limits,
    TRUNK_SHARE,
    lineage_members,
    membership_labels,
    progress_bins,
    trunk_probabilities,
    weighted_median,
)

from .test_graph_coverage import _memory_graph_store
from .test_graph_feature_projection import _native_chain
from .test_trajectory_record_paths import (
    _copy_store,
    _refresh_fingerprint,
    _rewrite_inputs,
    _rewrite_parameters,
    _write_graph_payload,
)

_SOURCES = ["t1"]


def _knn_adjacency(points: np.ndarray, k: int = 5, scale: float = 0.15) -> np.ndarray:
    """Link each cell to its ``k`` nearest cells, with weights exp(-(d/scale)^2)."""
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argsort(distances, axis=1, kind="stable")[:, :k]
    rows = np.repeat(np.arange(len(points)), k)
    cols = nearest.ravel()
    adjacency = np.zeros((len(points), len(points)))
    adjacency[rows, cols] = np.exp(-((distances[rows, cols] / scale) ** 2))
    return adjacency


@dataclass(frozen=True)
class _Arm:
    """A branch that leaves the trunk's end, with the cells of its sink at its end.

    The branch holds ``n_cells`` cells in steps of ``step`` along ``heading``,
    starting ``gap`` beyond the trunk's end, and scatters across its heading
    by up to 0.3 at its end. Its last ``sink_cells`` cells take the label
    ``<name>2`` and the others ``<name>1``. A gap thins the graph's edges
    into the branch, so that the trunk gives its sink a smaller share.
    """

    name: str
    heading: tuple[float, float]
    n_cells: int = 40
    sink_cells: int = 20
    step: float = 0.075
    gap: float = 0.0


_UP = (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0))
_DOWN = (1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0))
_ACROSS = (1.0, 0.0)
_TWO_ARMS = (_Arm("a", _UP), _Arm("b", _DOWN))
# B's sink holds 2 cells against A's 20.
_SMALL_SINK = (_Arm("a", _UP), _Arm("b", _DOWN, sink_cells=2))
# D leaves the trunk's end straight ahead, beyond a gap, and ends in 2 cells.
_RARE_THIRD = (
    _Arm("a", _UP),
    _Arm("b", _DOWN),
    _Arm("d", _ACROSS, sink_cells=2, gap=0.25),
)
# B runs four times as long as A, with 140 cells before its sink.
_LONG_BRANCH = (_Arm("a", _UP), _Arm("b", _DOWN, n_cells=160))
# B, beyond a gap, takes 4% of the trunk's probability and holds most of its
# own weight on its branch, 60 cells before a sink of 20, or in a sink of 80.
_RARE_LONG_BRANCH = (_Arm("a", _UP), _Arm("b", _DOWN, n_cells=80, gap=0.25))
_RARE_LARGE_SINK = (
    _Arm("a", _UP),
    _Arm("b", _DOWN, n_cells=100, sink_cells=80, gap=0.25),
)
# D, beyond a gap, takes 7% of the trunk's probability, with 280 cells
# before its sink.
_RARE_LONG_THIRD = (
    _Arm("a", _UP),
    _Arm("b", _DOWN),
    _Arm("d", _ACROSS, n_cells=300, sink_cells=20, gap=0.12),
)


def _trajectory_cells(
    arms: tuple[_Arm, ...] = _TWO_ARMS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Each cell's arm, position along it, cluster label and coordinates.

    The trunk runs from (0, 0) along x in steps of 0.05 for 60 cells and
    scatters across by up to 0.01. Each arm then leaves the trunk's end, by
    default A and B at 45 degrees up and down for 40 cells each, in steps of
    0.075. C rises from the trunk's middle, at x = 1.5, in steps of 0.1 for
    5 cells, starting 0.2 above it.
    """
    rng = np.random.default_rng(210)
    arm_names, positions, labels, points = [], [], [], []
    trunk = np.arange(60)
    arm_names += ["trunk"] * 60
    positions += trunk.tolist()
    labels += ["t1"] * 30 + ["t2"] * 30
    points.append(np.column_stack([0.05 * trunk, rng.uniform(-0.01, 0.01, size=60)]))
    start = np.array([3.0, 0.0])
    for arm in arms:
        along = np.arange(1, arm.n_cells + 1)
        heading = np.asarray(arm.heading)
        across = np.array([-heading[1], heading[0]])
        spread = rng.uniform(-1.0, 1.0, size=arm.n_cells) * 0.3 * along / arm.n_cells
        arm_names += [arm.name] * arm.n_cells
        positions += along.tolist()
        labels += [f"{arm.name}1"] * (arm.n_cells - arm.sink_cells)
        labels += [f"{arm.name}2"] * arm.sink_cells
        points.append(
            start
            + np.outer(arm.gap + arm.step * along, heading)
            + np.outer(spread, across)
        )
    side = np.arange(5)
    arm_names += ["c"] * 5
    positions += side.tolist()
    labels += ["c"] * 5
    points.append(np.column_stack([np.full(5, 1.5), 0.2 + 0.1 * side]))
    return (
        np.asarray(arm_names, dtype=object),
        np.asarray(positions),
        np.asarray(labels, dtype=object),
        np.vstack(points),
    )


@dataclass(frozen=True)
class _Branching:
    store: Any
    graph: ArtifactRef
    selection: ArtifactRef
    arms: np.ndarray
    positions: np.ndarray
    labels: np.ndarray
    points: np.ndarray
    pseudotime: ArtifactRef
    fate: ArtifactRef
    curves: ArtifactRef

    def label_artifact(self) -> ArtifactRef:
        return resolve_metadata_snapshot(
            self.store.zw,
            values=self.labels,
            row_ids=np.asarray([f"c{row}" for row in range(len(self.labels))]),
            operation="test_cluster_labels",
            parameters={},
            inputs={"cell_selection": self.selection},
            source_columns=["cluster"],
        )


def _graph_of(
    arms: np.ndarray, positions: np.ndarray, labels: np.ndarray, points: np.ndarray
) -> _Branching:
    """Store cells and the graph of their five nearest neighbors, in a shuffled order."""
    order = np.random.default_rng(10).permutation(len(arms))
    arms, positions, labels, points = (
        arms[order],
        positions[order],
        labels[order],
        points[order],
    )
    store = _memory_graph_store()
    table = store.z.create_group("cellData")
    table.create_array("ids", data=np.asarray([f"c{row}" for row in range(len(arms))]))
    table.create_array("I", data=np.ones(len(arms), dtype=bool))
    store.cells = MetaData(table)
    selection = resolve_stored_selection_artifact(
        store.zw,
        table_path="cellData",
        id_column="ids",
        source_column="I",
        scope="datastore",
        kind="cell_selection",
        operation="test_selection",
        parameters={},
        inputs={},
    )
    graph, _neighbors, coordinates = _native_chain(
        store.zw, "RNA", cell_selection=selection
    )
    _write_graph_payload(store, graph, _knn_adjacency(points))
    artifact_group(store.zw, coordinates)["data"][:] = points
    return _Branching(
        store=store,
        graph=graph,
        selection=selection,
        arms=arms,
        positions=positions,
        labels=labels,
        points=points,
        pseudotime=None,  # type: ignore[arg-type]
        fate=None,  # type: ignore[arg-type]
        curves=None,  # type: ignore[arg-type]
    )


@cache
def _template(layout: tuple[_Arm, ...] = _TWO_ARMS) -> _Branching:
    labeled = _graph_of(*_trajectory_cells(layout))
    store = labeled.store
    clusters = labeled.label_artifact()
    sinks = [f"{arm.name}2" for arm in layout]
    pseudotime = store.trajectory.pseudotime(
        labeled.graph, source_sink=clusters, sources=_SOURCES, sinks=sinks
    )
    fate = store.trajectory.fate(pseudotime, clusters, sinks=sinks)
    curves = store.trajectory.curves(fate, n_points=40, df=6.0)
    return replace(labeled, pseudotime=pseudotime, fate=fate, curves=curves)


def _branching(layout: tuple[_Arm, ...] = _TWO_ARMS) -> _Branching:
    """Return a private copy of the store of a layout, built once."""
    template = _template(layout)
    return replace(template, store=_copy_store(template.store))


def _artifacts(store: Any, kind: str) -> list[ArtifactRef]:
    return list_artifacts(store.zw, scope="assay", assay="RNA", kind=kind)


def _trunk_by_hand(branching: _Branching) -> np.ndarray:
    """Each lineage's mean probability over the earliest tenth of the cells outside the sinks.

    The trajectory is one graph component, so every lineage takes the same
    cells: sorted by pseudotime, the first tenth of the scored cells outside
    the sinks, rounded up, and every cell at the pseudotime of the last.
    """
    store = branching.store
    fate = store.trajectory.load_fate(branching.fate)
    pseudotime = store.trajectory.load_pseudotime(branching.pseudotime).values
    cells = np.flatnonzero(
        fate.valid & ~np.isin(branching.labels, list(fate.sink_labels))
    )
    ordered = cells[np.argsort(pseudotime[cells], kind="stable")]
    last = pseudotime[ordered[-(-len(ordered) // 10) - 1]]
    trunk = cells[pseudotime[cells] <= last]
    return np.asarray(fate.values[trunk].mean(axis=0), dtype=np.float64)


# The rule's parts, by hand


def test_a_weighted_median_and_its_spread_by_hand() -> None:
    # Distances 1, 2, 3 and 10 of equal weight: the lower middle value is 2,
    # and the deviations from it, 1, 0, 1 and 8, have the median 1.
    assert weighted_median(np.array([3.0, 1.0, 10.0, 2.0]), np.ones(4)) == 2.0
    limits = distance_limits(
        np.array([[3.0], [1.0], [10.0], [2.0]]),
        np.full((4, 1), 0.5),
        np.ones((4, 1)),
        np.ones(4, dtype=bool),
        n_bins=1,
        score=3.0,
    )
    np.testing.assert_allclose(limits, [[2.0 + 3.0 * 1.4826 * 1.0]])
    # The cell at 10 weighs more than the rest together, so it is the median.
    assert (
        weighted_median(np.array([1.0, 2.0, 3.0, 10.0]), np.array([1, 1, 1, 5])) == 10.0
    )
    # Cells of zero weight count for nothing, and no weight gives no median.
    assert weighted_median(np.array([1.0, 5.0, 9.0]), np.array([0.0, 1.0, 0.0])) == 5.0
    assert math.isnan(weighted_median(np.array([1.0]), np.array([0.0])))
    with pytest.raises(ValueError, match="finite and non-negative"):
        weighted_median(np.array([1.0, 2.0]), np.array([1.0, -1.0]))


def test_distance_limits_take_the_median_spread_of_neighboring_bins() -> None:
    # Lineage 0 over four bins of progress. Bin 0 holds distances 1 and 1,
    # bin 1 distance 2, and bin 3 distance 3, each without deviation. Bin 2
    # holds a cell of the lineage at 0.5 and a side state at 9 of three
    # times its weight, which makes the bin's median 9 and its deviation 0.
    # Lineage 1 weighs only the last cell, and an unscored cell weighs none.
    distance = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.5, 0.0],
            [9.0, 0.0],
            [3.0, 4.0],
            [7.0, 7.0],
        ]
    )
    progress = np.array([0.1, 0.2, 0.3, 0.6, 0.6, 0.9, 0.5])[:, None].repeat(2, axis=1)
    weights = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [3.0, 0.0],
            [1.0, 2.0],
            [1.0, 1.0],
        ]
    )
    valid = np.array([True] * 6 + [False])

    limits = distance_limits(distance, progress, weights, valid, n_bins=4, score=2.0)

    # The spreads 1, 2, 9 and 3 give bins 0 and 1 the median of the first
    # three, 2, and bins 2 and 3 that of the last three, 3, so the side state
    # does not widen bin 2. Lineage 1's one spread, 4 in bin 3, serves the
    # bins whose three neighbors hold it, and the others have no limit.
    np.testing.assert_allclose(
        limits, [[2.0, 2.0, 3.0, 3.0], [np.nan, np.nan, 4.0, 4.0]]
    )
    # With two bins, each takes the median of both: 3 and 5 give 4.
    two = distance_limits(
        np.array([[1.0], [3.0], [5.0]]),
        np.array([[0.2], [0.3], [0.8]]),
        np.array([[1.0], [3.0], [1.0]]),
        np.ones(3, dtype=bool),
        n_bins=2,
        score=2.0,
    )
    np.testing.assert_allclose(two, [[4.0, 4.0]])
    no_weight = distance_limits(
        np.ones((2, 1)),
        np.full((2, 1), 0.5),
        np.zeros((2, 1)),
        np.ones(2, bool),
        n_bins=3,
        score=1.0,
    )
    assert np.isnan(no_weight).all()


def _trunk_layout() -> dict[str, Any]:
    """Cells of two graph components for the trunk of three lineages, A, B and C.

    Component 0 holds cells 0 to 29 outside the sinks, ordered 0 to 29 except
    cell 3, which ties with cell 2 at 2, then a cell of A's sink ordered -1,
    a cell of B's sink, a cell of the sink of D, a fate that the lineages
    leave out, ordered -2, and an unscored cell ordered -5. Component 1 holds
    cells 34 to 43 outside the sinks, ordered 50 to 59, and a cell of C's
    sink. Cells of component 0 give C either nothing or 1e-6, the rounding
    of a solver in a component without C's sink. Component 2 holds one
    unscored cell of A's sink.
    """
    n_cells = 46
    ordering = np.zeros(n_cells)
    ordering[:30] = np.arange(30)
    ordering[3] = 2.0
    ordering[[30, 31, 32, 33]] = [-1.0, 100.0, -2.0, -5.0]
    ordering[34:44] = 50.0 + np.arange(10)
    ordering[44] = 100.0
    ordering[45] = np.nan
    component = np.zeros(n_cells, dtype=np.int64)
    component[34:] = 1
    component[45] = 2
    valid = np.ones(n_cells, dtype=bool)
    valid[[33, 45]] = False
    sinks = np.zeros(n_cells, dtype=bool)
    sinks[[30, 31, 32, 44, 45]] = True
    lineage_sinks = np.zeros((n_cells, 3), dtype=bool)
    lineage_sinks[30, 0] = lineage_sinks[31, 1] = lineage_sinks[44, 2] = True
    lineage_sinks[45, 0] = True
    weights = np.zeros((n_cells, 3))
    weights[:30] = [0.5, 0.2, 0.0]
    weights[:30:2, 2] = 1e-6
    weights[[0, 1, 2, 3], 0] = [0.1, 0.1, 0.4, 0.6]
    weights[[0, 1, 2, 3], 1] = [0.06, 0.0, 0.03, 0.03]
    weights[[30, 31, 32, 44]] = [[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]]
    weights[33] = [0.9, 0.1, 0.0]
    weights[34:44] = [[0.0, 0.0, 0.25]] + [[0.0, 0.0, 0.6]] * 9
    weights[45] = np.nan
    return {
        "weights": weights,
        "ordering": ordering,
        "valid": valid,
        "sinks": sinks,
        "lineage_sinks": lineage_sinks,
        "component": component,
    }


def test_a_lineage_trunk_is_the_earliest_tenth_of_the_cells_that_can_reach_it() -> None:
    layout = _trunk_layout()

    components, trunk = trunk_probabilities(**layout, lineages=["A", "B", "C"])

    # Components 0 and 1 hold scored sink cells; component 2 holds only an
    # unscored one, so it has no row. A and B take the 30 scored cells of
    # component 0 outside the sinks: the sink cells of A and of D and the
    # unscored cell come earlier but stay out. A tenth of 30 is 3 cells,
    # ordered up to 2, and cell 3 ties with the third, so the trunk holds
    # cells 0 to 3. Their mean is 1.2 / 4 = 0.3 for A, whose median would be
    # 0.25, and 0.12 / 4 = 0.03 for B, which leaves out none of the cells of
    # no probability. C's sink lies in component 1 alone, whose earliest
    # tenth is its first cell, with 0.25; component 0, which gives C 0 or
    # 1e-6, has no trunk for C, nor component 1 for A and B.
    np.testing.assert_array_equal(components, [0, 1])
    np.testing.assert_allclose(
        trunk, [[0.3, 0.03, np.nan], [np.nan, np.nan, 0.25]], rtol=1e-12
    )
    assert TRUNK_SHARE == 0.1


def _members_by_trunk(weights: Any, trunk: Any, **cells: Any) -> np.ndarray:
    """Call the rule on cells of one graph component that lie on one curve, in one bin of limit 1.

    One sink cell of each lineage joins the component after the given cells,
    and the rule's calls of the given cells are returned.
    """
    weights = np.asarray(weights, dtype=np.float64)
    n_cells, n_lineages = weights.shape
    with_sinks = np.vstack([weights, np.eye(n_lineages)])
    lineage_sinks = np.vstack(
        [np.zeros((n_cells, n_lineages), dtype=bool), np.eye(n_lineages, dtype=bool)]
    )
    distance = cells.get("distance", np.full((n_cells, n_lineages), 0.1))
    valid = cells.get("valid", np.ones(n_cells, dtype=bool))
    members = lineage_members(
        with_sinks,
        np.vstack([distance, np.full((n_lineages, n_lineages), 0.1)]),
        np.full((n_cells + n_lineages, n_lineages), 0.5),
        np.concatenate([valid, np.ones(n_lineages, dtype=bool)]),
        np.ones((n_lineages, 1)),
        np.zeros(n_cells + n_lineages, dtype=np.int64),
        lineage_sinks,
        np.array([0]),
        np.asarray(trunk, dtype=np.float64)[None, :],
        band=cells.get("band", 0.5),
    )
    np.testing.assert_array_equal(members[n_cells:], np.eye(n_lineages, dtype=bool))
    return np.asarray(members[:n_cells])


def test_a_cell_follows_each_lineage_it_keeps_half_the_trunk_probability_of() -> None:
    # Two lineages whose trunk probabilities, 0.8 and 0.2, cover every sink.
    # Cell 9 lies off both curves and cell 11 is unscored.
    weights = [
        [0.8, 0.2],
        [0.5, 0.5],
        [0.92, 0.08],
        [0.84, 0.16],
        [0.88, 0.12],
        [0.6, 0.4],
        [0.7, 0.3],
        [0.38, 0.62],
        [0.0, 0.0],
        [0.8, 0.2],
        [0.41, 0.59],
        [0.8, 0.2],
    ]
    distance = np.full((12, 2), 0.1)
    distance[9] = 5.0
    valid = np.ones(12, dtype=bool)
    valid[11] = False

    members = _members_by_trunk(weights, [0.8, 0.2], distance=distance, valid=valid)

    # A cell follows A while its probability for A is at least 0.4, half of
    # 0.8, and B while its probability for B is at least 0.1. Cells 0, 1, 3
    # to 6 and 10 hold both; cell 2, with 0.08 for B, follows A alone, and
    # cell 7, with 0.38 for A, B alone. Cell 8 has no probability, cell 9
    # lies off the curves and cell 11 is unscored. Measured against each
    # cell's largest divided probability instead, cells 1, 5 and 10, which
    # divide to more than twice as much for B as for A, would follow B alone
    # while their probability for A is still above 0.4.
    np.testing.assert_array_equal(
        members,
        [[1, 1], [1, 1], [1, 0], [1, 1], [1, 1], [1, 1], [1, 1], [0, 1]]
        + [[0, 0], [0, 0], [1, 1], [0, 0]],
    )
    np.testing.assert_array_equal(
        membership_labels(members, ["A", 7]),
        ["shared", "shared", "A", "shared", "shared", "shared", "shared", "7"]
        + ["none", "none", "shared", "none"],
    )
    with pytest.raises(ValueError, match="'shared' cannot be told apart"):
        membership_labels(members, ["shared", "B"])
    # A band of 0.9 asks for 0.72 of A and 0.18 of B, which only cell 0, with
    # the trunk's split, holds together.
    tight = _members_by_trunk(
        weights, [0.8, 0.2], distance=distance, valid=valid, band=0.9
    )
    np.testing.assert_array_equal(tight.all(axis=1), np.arange(12) == 0)
    np.testing.assert_array_equal(
        progress_bins(np.array([0.0, 0.29, 0.3, 1.0, np.nan]), 10), [0, 2, 3, 9, 0]
    )


def test_curves_of_some_sinks_compare_a_cell_with_its_largest_divided_probability() -> (
    None
):
    # The curves follow A and B, with trunk probabilities 0.5 and 0.2, and
    # leave out a third sink that takes the trunk's other 0.3. Cell 0 has the
    # trunk's split. Cells 1 to 3 move toward the third sink, so all their
    # divided probabilities fall below one, and each follows the lineages
    # whose divided probability is at least half its largest.
    members = _members_by_trunk(
        [[0.5, 0.2], [0.1, 0.05], [0.1, 0.01], [0.3, 0.05]], [0.5, 0.2]
    )

    # Cell 1 divides to (0.2, 0.25) and follows both, though 0.1 and 0.05 lie
    # below half the trunk's 0.5 and 0.2. Cell 2 divides to (0.2, 0.05) and
    # follows A alone. Cell 3 divides to (0.6, 0.25), and 0.25 is under half
    # of 0.6, so it follows A alone, as it does by half the trunk's
    # probabilities.
    np.testing.assert_array_equal(members, [[1, 1], [1, 1], [1, 0], [1, 0]])


def test_a_rare_lineage_keeps_the_trunk_that_its_progenitors_share() -> None:
    # The trunk splits 0.70, 0.27 and 0.03, a tenth of the major share for
    # the rare fate, so a cell follows them while it keeps 0.35, 0.135 and
    # 0.015.
    trunk = [0.70, 0.27, 0.03]
    members = _members_by_trunk(
        [
            trunk,
            [0.97, 0.027, 0.003],
            [0.40, 0.10, 0.50],
            [0.10, 0.00, 0.90],
            [0.69, 0.30, 0.01],
            [0.20, 0.10, 0.70],
            [0.00, 0.70, 0.30],
        ],
        trunk,
    )

    # A cell with the trunk's split follows all three, though 0.03 is a tenth
    # of 0.70. A cell committed to the major lineage, whose other
    # probabilities fall in proportion, follows it alone. A cell with 0.40
    # for the major lineage and 0.50 for the rare one follows both and has
    # left the minor one. A cell close to the rare sink follows it alone,
    # and one that keeps a third of the trunk's rare share follows the other
    # two. A cell of 0.20, 0.10 and 0.70 has left the major and minor
    # lineages. A cell of 0.70 for the minor lineage and 0.30 for the rare
    # one follows both: the rare fate's divided probability, 10, does not
    # make it leave the minor lineage, whose probability is far above 0.135.
    np.testing.assert_array_equal(
        members,
        [[1, 1, 1], [1, 0, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1]],
    )


def test_a_lineage_that_no_cell_can_reach_from_the_trunk_is_refused() -> None:
    layout = _trunk_layout()
    # Without the cells of component 1 outside the sinks, C's sink cell
    # fills a component of its own.
    alone = {
        name: np.delete(values, np.arange(34, 44), axis=0)
        for name, values in layout.items()
    }
    with pytest.raises(
        ValueError, match="no probability to 'C': its sink cells.*without that lineage"
    ):
        trunk_probabilities(**alone, lineages=["A", "B", "C"])
    # A trunk of no probability for A and B is refused too, naming both.
    layout["weights"][:4, :2] = 0.0
    with pytest.raises(
        ValueError,
        match="no probability to 'A', 'B': their sink cells.*without those lineages",
    ):
        trunk_probabilities(**layout, lineages=["A", "B", "C"])
    # A lineage with a trunk in one component and none of probability in
    # another is refused, so that no cell is divided by zero.
    two = _trunk_layout()
    two["lineage_sinks"][44, 0] = True
    two["sinks"][44] = True
    with pytest.raises(ValueError, match="no probability to 'A'"):
        trunk_probabilities(**two, lineages=["A", "B", "C"])


def test_each_cell_is_measured_against_the_trunk_of_its_own_component() -> None:
    # Component 0 splits its trunk 0.9 and 0.1 between A and B, and component
    # 1 splits it 0.1 and 0.9. Component 2 holds only sink cells of A, and
    # component 3 holds sink cells of B but none of A, so its cell 5 reaches
    # A only by the solver's rounding.
    weights = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.5, 0.5],
            [0.5, 0.5],
            [1.0, 0.0],
            [1e-7, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    component = np.array([0, 1, 0, 1, 2, 3, 0, 0, 1, 1, 3])
    lineage_sinks = np.zeros((11, 2), dtype=bool)
    lineage_sinks[[4, 6, 9], 0] = True
    lineage_sinks[[7, 8, 10], 1] = True
    members = lineage_members(
        weights,
        np.full((11, 2), 0.1),
        np.full((11, 2), 0.5),
        np.ones(11, dtype=bool),
        np.ones((2, 1)),
        component,
        lineage_sinks,
        np.array([0, 1, 2, 3]),
        np.array([[0.9, 0.1], [0.1, 0.9], [np.nan, np.nan], [np.nan, 0.9]]),
        band=0.5,
    )

    # Each trunk cell follows both lineages of its component. A cell of 0.5
    # and 0.5 keeps half of A's 0.9 and of B's 0.1 in component 0, and half
    # of B's 0.9 and of A's 0.1 in component 1, so it follows both in each.
    # The sink cell of A in component 2, which has no trunk, follows A, and
    # the cell of component 3 follows neither lineage: its 1e-7 for A counts
    # as zero.
    np.testing.assert_array_equal(
        members[:6], [[1, 1], [1, 1], [1, 1], [1, 1], [1, 0], [0, 0]]
    )
    # Measured against component 0's trunk alone, the trunk of component 1
    # would leave A: its 0.1 is under half of 0.9.
    pooled = lineage_members(
        weights,
        np.full((11, 2), 0.1),
        np.full((11, 2), 0.5),
        np.ones(11, dtype=bool),
        np.ones((2, 1)),
        np.zeros(11, dtype=np.int64),
        lineage_sinks,
        np.array([0]),
        np.array([[0.9, 0.1]]),
        band=0.5,
    )
    assert not pooled[1, 0]


def test_a_branch_point_splits_shared_cells_from_parted_ones() -> None:
    pseudotime = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    shared = np.array([1, 1, 1, 0, 1, 0, 0], dtype=bool)
    parted = ~shared
    # Splits after 0.3 and after 0.5 each put one cell on the wrong side; the
    # earlier one wins, halfway between 0.3 and 0.4.
    assert branch_point(pseudotime, shared, parted) == pytest.approx(0.35)
    # The split falls only between different pseudotimes.
    tied = np.array([0.1, 0.2, 0.2, 0.3])
    assert branch_point(
        tied, np.array([1, 1, 0, 0], bool), np.array([0, 0, 1, 1], bool)
    ) == pytest.approx(0.15)
    assert math.isnan(branch_point(pseudotime, shared & False, parted))
    assert math.isnan(branch_point(pseudotime, shared, parted & False))
    with pytest.raises(ValueError, match="both shared and parted"):
        branch_point(pseudotime, shared, shared)
    # The window holds the share of the pair's cells nearest the point.
    window = branch_window(pseudotime, np.ones(7, bool), 0.35, 0.5)
    np.testing.assert_array_equal(window, [0, 1, 1, 1, 1, 0, 0])
    with pytest.raises(ValueError, match="no branch point"):
        branch_window(pseudotime, np.ones(7, bool), math.nan, 0.5)


def test_a_pair_whose_shared_cells_do_not_come_first_has_no_branch_point() -> None:
    pseudotime = np.array([0.0, 0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9])

    def split(pattern: str, times: np.ndarray = pseudotime) -> float:
        shared = np.array([cell == "S" for cell in pattern])
        return branch_point(times, shared, ~shared)

    # Four parted cells, two shared and two parted again: no stretch from
    # the first cell holds more shared than parted cells, so the best split
    # lies before the first cell, which misplaces only the two shared cells.
    assert math.isnan(split("PPPPSSPP"))
    # One parted cell and seven shared ones: the best split lies after the
    # last cell, which misplaces only the parted cell.
    assert math.isnan(split("PSSSSSSS"))
    # A shared start parts after it, halfway between 0.1 and 0.2.
    assert split("SSPPPPPP") == pytest.approx(0.15)
    # One shared cell after a parted one ties with no split at all, and the
    # earliest split wins, so the pair has no branch point.
    assert math.isnan(split("PSPP", pseudotime[:4]))
    # A split between cells that ties with the split after the last cell wins.
    assert split("SSPS", pseudotime[:4]) == pytest.approx(0.15)
    # A shared first cell at the same pseudotime as two parted cells cannot
    # be split from them, so it makes no shared start.
    tied = np.array([0.0, 0.0, 0.0, 0.5, 0.6])
    assert math.isnan(split("SPPSP", tied))
    assert split("SPPSP", np.array([0.0, 0.1, 0.2, 0.5, 0.6])) == pytest.approx(0.05)


_SHAPES = (np.zeros((3, 2)), np.full((3, 2), 0.5), np.ones((3, 2)), np.ones(3, bool))
# Components, lineage sinks, trunk components and trunk probabilities.
_TRUNK = (
    np.zeros(3, int),
    np.zeros((3, 2), bool),
    np.array([0]),
    np.array([[0.5, 0.5]]),
)
# Ordering, validity, sinks, lineage sinks and components of three cells.
_CELLS = (
    np.arange(3.0),
    np.ones(3, bool),
    np.zeros(3, bool),
    np.zeros((3, 2), bool),
    np.zeros(3, int),
)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: weighted_median(np.ones(2), np.ones(3)),
            "one-dimensional and aligned",
        ),
        (
            lambda: distance_limits(np.ones(3), *_SHAPES[1:], n_bins=2, score=1.0),
            "cells by lineages",
        ),
        (
            lambda: distance_limits(
                _SHAPES[0], np.ones((3, 1)), *_SHAPES[2:], n_bins=2, score=1.0
            ),
            "progress, distance and weights",
        ),
        (
            lambda: distance_limits(
                *_SHAPES[:3], np.ones(2, bool), n_bins=2, score=1.0
            ),
            "one entry per cell",
        ),
        (
            lambda: lineage_members(
                np.ones(3), *_SHAPES[:2], _SHAPES[3], np.ones((2, 1)), *_TRUNK, band=0.5
            ),
            "weights must be cells by lineages",
        ),
        (
            lambda: lineage_members(
                _SHAPES[2], *_SHAPES[:2], _SHAPES[3], np.ones((3, 1)), *_TRUNK, band=0.5
            ),
            "limits must be lineages by bins",
        ),
        (
            lambda: lineage_members(
                _SHAPES[2],
                *_SHAPES[:2],
                _SHAPES[3],
                np.ones((2, 1)),
                np.zeros(2, int),
                *_TRUNK[1:],
                band=0.5,
            ),
            "component must hold one entry per cell",
        ),
        (
            lambda: lineage_members(
                _SHAPES[2],
                *_SHAPES[:2],
                _SHAPES[3],
                np.ones((2, 1)),
                *_TRUNK[:3],
                np.array([[0.5, 0.0]]),
                band=0.5,
            ),
            "one positive probability or NaN per lineage",
        ),
        (
            lambda: lineage_members(
                _SHAPES[2],
                *_SHAPES[:2],
                _SHAPES[3],
                np.ones((2, 1)),
                *_TRUNK[:2],
                np.array([0, 0]),
                np.full((2, 2), 0.5),
                band=0.5,
            ),
            "one positive probability or NaN per lineage",
        ),
        (
            lambda: trunk_probabilities(np.ones(3), *_CELLS, lineages=["A", "B"]),
            "weights must be cells by lineages",
        ),
        (
            lambda: trunk_probabilities(_SHAPES[2], *_CELLS, lineages=["A"]),
            "one lineage per column",
        ),
        (
            lambda: trunk_probabilities(
                _SHAPES[2], np.ones(2), *_CELLS[1:], lineages=["A", "B"]
            ),
            "one entry per cell",
        ),
        (
            lambda: trunk_probabilities(
                _SHAPES[2],
                *_CELLS[:3],
                np.ones(3, bool),
                _CELLS[4],
                lineages=["A", "B"],
            ),
            "lineage_sinks must be cells by lineages",
        ),
        (lambda: membership_labels(np.ones((3, 2), bool), ["A"]), "cells by lineages"),
        (
            lambda: branch_points(
                np.ones((3, 2), bool), np.ones(3), np.ones((3, 4)), ["A", "B"]
            ),
            "one curve per lineage",
        ),
        (
            lambda: branch_points(
                np.ones(3, bool), np.ones(3), np.ones((2, 4)), ["A", "B"]
            ),
            "cells by lineages",
        ),
        (
            lambda: branch_window(np.ones(3), np.zeros(3, bool), 0.5, 0.5),
            "No cell follows either lineage",
        ),
    ],
)
def test_the_rule_refuses_inputs_of_other_shapes(call, message) -> None:
    with pytest.raises(ValueError, match=message):
        call()


# The membership of a branching trajectory


def test_membership_of_a_branching_trajectory_follows_its_known_lineages() -> None:
    branching = _branching()
    store = branching.store

    ref = store.trajectory.lineage_membership(branching.curves)
    result = store.trajectory.load_lineage_membership(ref)

    assert isinstance(result, LineageMembershipResult)
    assert result.lineages == ("a2", "b2")
    assert result.curves == branching.curves
    assert result.cell_selection == branching.selection
    labels = result.labels
    arms, positions = branching.arms, branching.positions
    # The trunk is shared, A and B follow their own lineage beyond their
    # first ten cells, and C follows none beyond its first cell, which lies
    # 0.2 from the trunk.
    assert set(labels[arms == "trunk"]) == {"shared"}
    assert set(labels[(arms == "a") & (positions > 10)]) == {"a2"}
    assert set(labels[(arms == "b") & (positions > 10)]) == {"b2"}
    assert set(labels[(arms == "c") & (positions > 0)]) == {"none"}
    # C's cells split their probability between A and B as trunk cells do.
    fate = store.trajectory.load_fate(branching.fate)
    side = (arms == "c") & (positions > 0)
    assert np.all(fate.values[side].min(axis=1) >= 0.5 * fate.values[side].max(axis=1))
    # And their distances to the curves overlap those of the ends of A and B,
    # so no distance from the curves alone tells them apart.
    curves = store.trajectory.load_curves(branching.curves)
    side_distance = curves.distance[side].min(axis=1)
    end_distance = curves.distance[(arms != "c") & (positions > 30)].min(axis=1)
    assert side_distance.min() < end_distance.max()
    # A second call reuses the membership.
    assert store.trajectory.lineage_membership(branching.curves) == ref
    assert len(_artifacts(store, "lineage_membership")) == 1


def test_membership_follows_its_recorded_rule() -> None:
    # B's sink is ten times smaller than A's, so the trunk splits its
    # probability unevenly and the scale of each lineage matters.
    branching = _branching(_SMALL_SINK)
    store = branching.store
    ref = store.trajectory.lineage_membership(
        branching.curves, band=0.6, distance_score=2.5, n_bins=8
    )
    result = store.trajectory.load_lineage_membership(ref)
    curves = store.trajectory.load_curves(branching.curves)

    assert store.artifacts.inspect(ref).parameters == {
        "band": 0.6,
        "distance_score": 2.5,
        "n_bins": 8,
    }

    # Each bin's spread is the weighted median distance of its cells plus 2.5
    # times their weighted median absolute deviation and 1.4826, each median
    # the smallest value whose weight, with that of smaller values, reaches
    # half; each limit is the median spread of three neighboring bins.
    def lower_median(values: np.ndarray, weights: np.ndarray) -> float:
        total = weights.sum()
        return min(
            value for value in values if weights[values <= value].sum() >= total / 2
        )

    spreads = np.full((2, 8), np.nan)
    for lineage in range(2):
        for index in range(8):
            in_bin = curves.valid & (curves.weights[:, lineage] > 0)
            in_bin &= (
                np.minimum(np.floor(np.nan_to_num(curves.progress[:, lineage]) * 8), 7)
                == index
            )
            if not in_bin.any():
                continue
            values = curves.distance[in_bin, lineage]
            weights = curves.weights[in_bin, lineage]
            center = lower_median(values, weights)
            spread = lower_median(np.abs(values - center), weights)
            spreads[lineage, index] = center + 2.5 * 1.4826 * spread
    for index in range(8):
        first = min(max(index - 1, 0), 5)
        expected_limits = np.nanmedian(spreads[:, first : first + 3], axis=1)
        np.testing.assert_allclose(
            result.distance_limits[:, index], expected_limits, rtol=1e-12
        )
    # A cell follows a lineage when it lies within some lineage's limit at
    # its bin and its probability for the lineage is at least 0.6 of the
    # trunk's, since the curves follow every sink.
    trunk_probability = _trunk_by_hand(branching)
    # The trajectory is one graph component, whose trunk probabilities the
    # membership saves.
    assert result.trunk_probabilities.shape == (1, 2)
    np.testing.assert_allclose(
        result.trunk_probabilities.to_numpy()[0], trunk_probability, rtol=1e-6
    )
    expected = np.zeros_like(result.members)
    by_largest = np.zeros_like(result.members)
    undivided = np.zeros_like(result.members)
    for cell in np.flatnonzero(curves.valid):
        on = any(
            curves.distance[cell, lineage]
            <= result.distance_limits[
                lineage, min(int(curves.progress[cell, lineage] * 8), 7)
            ]
            for lineage in range(2)
        )
        divided = curves.weights[cell] / trunk_probability
        for lineage in range(2):
            weight = curves.weights[cell, lineage]
            expected[cell, lineage] = (
                on and weight > 0 and weight >= 0.6 * trunk_probability[lineage]
            )
            by_largest[cell, lineage] = (
                on and weight > 0 and divided[lineage] >= 0.6 * divided.max()
            )
            undivided[cell, lineage] = (
                on and weight > 0 and weight >= 0.6 * curves.weights[cell].max()
            )
    np.testing.assert_array_equal(result.members, expected)
    # Without the division, no trunk cell would follow B, and measured
    # against each cell's largest divided probability, cells of B's branch
    # that keep more than 0.6 of the trunk's A probability would leave A.
    trunk = branching.arms == "trunk"
    assert expected[trunk].all() and not undivided[trunk, 1].any()
    assert np.any(expected[:, 0] & ~by_largest[:, 0])
    # A wider band calls fewer cells shared.
    wide = store.trajectory.load_lineage_membership(
        store.trajectory.lineage_membership(branching.curves, band=0.9)
    )
    narrow = store.trajectory.load_lineage_membership(
        store.trajectory.lineage_membership(branching.curves, band=0.3)
    )
    assert (wide.labels == "shared").sum() < (narrow.labels == "shared").sum()


def test_two_lineages_part_where_their_fate_probabilities_do() -> None:
    branching = _branching()
    store = branching.store
    result = store.trajectory.load_lineage_membership(
        store.trajectory.lineage_membership(branching.curves)
    )
    pseudotime = store.trajectory.load_pseudotime(branching.pseudotime).values
    curves = store.trajectory.load_curves(branching.curves)
    trunk_probability = _trunk_by_hand(branching)

    (row,) = result.branch_points.itertuples(index=False)
    assert (row.first, row.second) == ("a2", "b2")
    # By hand: the split with the fewest shared cells after it and parted
    # cells before it, over every pseudotime between two of the pair's cells.
    pair = result.members.any(axis=1)
    shared = result.members.all(axis=1)
    candidates = np.unique(pseudotime[pair])
    best, best_errors = math.nan, math.inf
    for before, after in zip(candidates[:-1], candidates[1:], strict=True):
        split = 0.5 * (before + after)
        errors = np.sum(shared & (pseudotime > split)) + np.sum(
            pair & ~shared & (pseudotime < split)
        )
        if errors < best_errors:
            best, best_errors = split, errors
    # It misplaces fewer cells than a split before the first cell, which
    # misplaces every shared cell, and no more than one after the last.
    assert best_errors < shared.sum() and best_errors <= (pair & ~shared).sum()
    assert row.pseudotime == pytest.approx(best)
    assert row.shared_cells == shared.sum()
    assert row.parted_cells == (pair & ~shared).sum()
    # It lies where the cells give up half the trunk's probability for one of
    # A and B: walking out along each branch, the last cell that keeps half
    # of both comes before it, and the next cell after it, within the
    # pseudotime of a few cells.
    for arm in ("a", "b"):
        on_arm = np.flatnonzero(branching.arms == arm)
        ordered = on_arm[np.argsort(branching.positions[on_arm])]
        within = np.all(curves.weights[ordered] >= 0.5 * trunk_probability, axis=1)
        last_shared = int(np.flatnonzero(within)[-1])
        assert np.all(within[: last_shared + 1])
        assert pseudotime[ordered[last_shared]] - 0.05 < row.pseudotime
        assert row.pseudotime < pseudotime[ordered[last_shared + 1]] + 0.05
    # The progress on each curve is where its fitted pseudotime reaches it.
    for lineage, progress in enumerate((row.first_progress, row.second_progress)):
        np.testing.assert_allclose(
            np.interp(
                progress, curves.curve_progress, curves.curve_pseudotime[lineage]
            ),
            row.pseudotime,
            atol=1e-9,
        )


@pytest.mark.parametrize(
    "layout",
    [
        _SMALL_SINK,
        _RARE_THIRD,
        _LONG_BRANCH,
        _RARE_LONG_BRANCH,
        _RARE_LARGE_SINK,
        _RARE_LONG_THIRD,
    ],
    ids=[
        "sink ten times smaller",
        "third fate of 4%",
        "branch four times longer",
        "fate of 4% with a long branch",
        "fate of 4% with a large sink",
        "third fate of 7% with a long branch",
    ],
)
def test_every_lineage_keeps_the_trunk_however_unevenly_it_splits(layout) -> None:
    branching = _branching(layout)
    store = branching.store
    fate = store.trajectory.load_fate(branching.fate).values
    pseudotime = store.trajectory.load_pseudotime(branching.pseudotime).values
    trunk = branching.arms == "trunk"
    # The trunk gives some fate less than half its largest probability, so a
    # band of half the largest probability would leave that lineage no trunk.
    split = fate[trunk].mean(axis=0)
    assert split.min() < 0.5 * split.max()

    result = store.trajectory.load_lineage_membership(
        store.trajectory.lineage_membership(branching.curves)
    )

    # Every trunk cell follows every lineage. Every cell that follows some
    # lineage follows exactly those whose probability is still at least half
    # the trunk's, so a cell that follows one lineage alone has given up
    # half of every other and holds that lineage as its most likely fate.
    assert result.members[trunk].all()
    followed = result.members.any(axis=1)
    np.testing.assert_array_equal(
        result.members[followed], fate[followed] >= 0.5 * _trunk_by_hand(branching)
    )
    alone = result.members.sum(axis=1) == 1
    np.testing.assert_array_equal(
        result.members[alone].argmax(axis=1), fate[alone].argmax(axis=1)
    )
    # Beyond its first half, nearly all of each branch follows its own
    # lineage, and every pair parts after the last trunk cell.
    for index, arm in enumerate(layout):
        late = (branching.arms == arm.name) & (branching.positions > arm.n_cells / 2)
        assert result.members[late, index].mean() >= 0.9
    for row in result.branch_points.itertuples(index=False):
        assert row.pseudotime > pseudotime[trunk].max()


def test_a_cell_keeps_a_common_lineage_while_it_keeps_half_its_trunk_probability() -> (
    None
):
    # A cell leaves a lineage only once its probability falls below half the
    # trunk's, so on a short or rare branch, cells keep a common fate that
    # they still hold at half its trunk probability. Three layouts show it.
    small = _branching(_SMALL_SINK)
    result = small.store.trajectory.load_lineage_membership(
        small.store.trajectory.lineage_membership(small.curves)
    )
    fate = small.store.trajectory.load_fate(small.fate).values
    trunk_a = _trunk_by_hand(small)[0]
    # The trunk gives A 0.78, and some of the last 20 cells of B's branch,
    # 6 of them when measured, still hold at least 0.39 of A and follow it.
    late_b = (small.arms == "b") & (small.positions > 20)
    keep_a = late_b & result.members[:, 0]
    assert keep_a.any() and np.all(fate[keep_a, 0] >= 0.5 * trunk_a)
    assert not np.any(
        late_b & ~keep_a & result.members.any(axis=1) & (fate[:, 0] >= 0.5 * trunk_a)
    )

    for layout, branch in ((_LONG_BRANCH, "b"), (_RARE_LONG_THIRD, "d")):
        branching = _branching(layout)
        store = branching.store
        result = store.trajectory.load_lineage_membership(
            store.trajectory.lineage_membership(branching.curves)
        )
        pseudotime = store.trajectory.load_pseudotime(branching.pseudotime).values
        trunk_probability = _trunk_by_hand(branching)
        fate = store.trajectory.load_fate(branching.fate).values
        (row,) = result.branch_points.loc[
            (result.branch_points["first"] == "a2")
            & (result.branch_points["second"] == "b2")
        ].itertuples(index=False)
        sinks = np.isin(branching.labels, ["a2", "b2"])
        # A and B part after the first cell of their sinks: 0.344 against
        # 0.333 when B's branch is four times longer, and 0.292 against 0.194
        # when D's long branch takes 7% of the trunk's probability. The cells
        # that both still follow after that first sink cell all lie on the
        # long branch and keep half the trunk's probability of A and B.
        assert row.pseudotime > pseudotime[sinks].min()
        late = result.members[:, 0] & result.members[:, 1]
        late &= pseudotime > pseudotime[sinks].min()
        assert late.any() and set(branching.arms[late]) == {branch}
        assert np.all(fate[late][:, :2] >= 0.5 * trunk_probability[:2])


def _two_components(
    first: tuple[_Arm, ...], second: tuple[_Arm, ...] | None, offset: float = 10.0
) -> tuple[_Branching, ArtifactRef, np.ndarray]:
    """Store two trajectories as two graph components, with a fate map that a score directs.

    The second trajectory lies 30 above the first, its trunk labeled u1 and
    u2. Without arms, the second component is a cluster of 12 cells that all
    hold the sink label z2. The score is each cell's x coordinate, plus
    ``offset`` in the second component, so that by default every cell of the
    second comes after the first. Arms of the same name in both components
    end in the same sink label.
    """
    arms, positions, labels, points = _trajectory_cells(first)
    if second is None:
        rng = np.random.default_rng(3)
        other = (
            np.full(12, "z", dtype=object),
            np.arange(12),
            np.full(12, "z2", dtype=object),
            np.array([3.0, 30.0]) + rng.uniform(-0.05, 0.05, size=(12, 2)),
        )
    else:
        other_arms, other_positions, other_labels, other_points = _trajectory_cells(
            second
        )
        renamed = {"t1": "u1", "t2": "u2"}
        other = (
            np.asarray([f"{arm}+" for arm in other_arms], dtype=object),
            other_positions,
            np.asarray(
                [renamed.get(label, label) for label in other_labels], dtype=object
            ),
            other_points + np.array([0.0, 30.0]),
        )
    branching = _graph_of(
        np.concatenate([arms, other[0]]),
        np.concatenate([positions, other[1]]),
        np.concatenate([labels, other[2]]),
        np.vstack([points, other[3]]),
    )
    later = branching.points[:, 1] > 15.0
    score = branching.points[:, 0] + offset * later
    sinks = list(
        dict.fromkeys(
            f"{arm.name}2" for arm in (*first, *(second or (_Arm("z", _UP),)))
        )
    )
    fate = branching.store.trajectory.fate(
        sink_labels=branching.label_artifact(),
        graph=branching.graph,
        direction=score,
        sinks=sinks,
    )
    return replace(branching, fate=fate), fate, later


def test_each_lineage_takes_its_scale_from_its_own_graph_component() -> None:
    # A fate map directed by a score over two trajectories that share no
    # edge: A and B in the first component, E and F in the second, where F's
    # branch starts beyond a gap and its sink holds 4 cells. The earliest
    # cells by the score all lie in the first component, where E and F, the
    # last sink, whose column holds one minus the solved ones, have no
    # probability.
    branching, fate, later = _two_components(
        _TWO_ARMS, (_Arm("e", _UP), _Arm("f", _DOWN, sink_cells=4, gap=0.15))
    )
    store = branching.store
    stored = store.trajectory.load_fate(fate)
    assert list(stored.sink_labels) == ["a2", "b2", "e2", "f2"]
    assert stored.valid.all()
    np.testing.assert_array_equal(stored.values[~later, 2:], 0.0)
    np.testing.assert_array_equal(stored.values[later, :2], 0.0)
    curves = store.trajectory.curves(fate, n_points=40, df=6.0)
    result = store.trajectory.load_lineage_membership(
        store.trajectory.lineage_membership(curves)
    )
    members = pd.DataFrame(result.members, columns=list(result.lineages))

    # Each trunk follows the two lineages of its own component and no other.
    first_trunk = branching.arms == "trunk"
    second_trunk = branching.arms == "trunk+"
    assert members.loc[first_trunk, ["a2", "b2"]].to_numpy().all()
    assert not members.loc[first_trunk, ["e2", "f2"]].to_numpy().any()
    assert members.loc[second_trunk, ["e2", "f2"]].to_numpy().all()
    assert not members.loc[second_trunk, ["a2", "b2"]].to_numpy().any()
    assert not members.loc[later, ["a2", "b2"]].to_numpy().any()
    assert not members.loc[~later, ["e2", "f2"]].to_numpy().any()
    # The pairs within a component part on its scores, and the pairs across
    # components share no cell.
    table = result.branch_points.set_index(["first", "second"])
    assert 0.0 < table.loc[("a2", "b2"), "pseudotime"] < 10.0
    assert 10.0 < table.loc[("e2", "f2"), "pseudotime"] < 20.0
    across = table.drop([("a2", "b2"), ("e2", "f2")])
    assert across["pseudotime"].isna().all() and (across["shared_cells"] == 0).all()


@pytest.mark.parametrize(
    ("first", "second", "offset"),
    [
        (_TWO_ARMS, (_Arm("a", _UP), _Arm("b", _DOWN, gap=0.25)), 10.0),
        (_TWO_ARMS, (_Arm("a", _UP), _Arm("b", _DOWN, gap=0.25)), 0.0),
        (
            (_Arm("a", _UP, gap=0.25), _Arm("b", _DOWN)),
            (_Arm("a", _UP), _Arm("b", _DOWN, gap=0.25)),
            10.0,
        ),
    ],
    ids=["even and rare, second later", "even and rare, same scores", "both rare"],
)
def test_a_sink_label_in_two_components_takes_a_trunk_in_each(
    first, second, offset
) -> None:
    # Both components end in sinks labeled a2 and b2, and their trunks
    # split the probability differently: in the second, B's branch starts
    # beyond a gap and takes about 4%.
    branching, fate, later = _two_components(first, second, offset)
    store = branching.store
    values = store.trajectory.load_fate(fate).values
    second_trunk = branching.arms == "trunk+"
    split = values[second_trunk].mean(axis=0)
    assert split[1] < 0.1 * split[0]

    result = store.trajectory.load_lineage_membership(
        store.trajectory.lineage_membership(
            store.trajectory.curves(fate, n_points=40, df=6.0)
        )
    )

    # Every trunk cell on the curves of either component follows both
    # lineages, as it keeps its own component's split.
    for trunk in (branching.arms == "trunk", second_trunk):
        followed = trunk & result.members.any(axis=1)
        assert followed.sum() >= 55 and result.members[followed].all()
    # Each component's trunk probabilities are the mean of its earliest
    # tenth outside the sinks by the score, the first ceil(n / 10) when
    # sorted, with every cell at the score of the last; the rows' order
    # follows the components' numbers, so compare them as a set.
    score = branching.points[:, 0] + offset * later
    expected = []
    for part in (~later, later):
        cells = np.flatnonzero(part & ~np.isin(branching.labels, ["a2", "b2"]))
        ordered = cells[np.argsort(score[cells], kind="stable")]
        last = score[ordered[-(-len(ordered) // 10) - 1]]
        expected.append(values[cells[score[cells] <= last]].mean(axis=0))
    stored = result.trunk_probabilities.to_numpy()
    assert result.trunk_probabilities.index.name == "component"
    assert list(result.trunk_probabilities.columns) == ["a2", "b2"]
    np.testing.assert_allclose(
        stored[np.argsort(stored[:, 0])],
        np.asarray(expected)[np.argsort(np.asarray(expected)[:, 0])],
        rtol=1e-6,
    )


def test_a_membership_saved_without_trunk_probabilities_is_computed_anew() -> None:
    # A membership in the form that releases before the trunk probabilities
    # saved: without their two arrays, with the fingerprint of the others.
    branching = _branching()
    store = branching.store
    earlier = store.trajectory.lineage_membership(branching.curves)
    group = artifact_group(store.zw, earlier)
    del group["trunk_components"]
    del group["trunk_probabilities"]
    _refresh_fingerprint(
        group,
        tuple(
            name
            for name in LINEAGE_MEMBERSHIP_PAYLOAD
            if name not in ("trunk_components", "trunk_probabilities")
        ),
    )

    # A call with the same curves and rule computes a new membership.
    current = store.trajectory.lineage_membership(branching.curves)
    assert current != earlier
    assert len(_artifacts(store, "lineage_membership")) == 2
    assert store.trajectory.load_lineage_membership(
        current
    ).trunk_probabilities.shape == (1, 2)
    # Loading the earlier one, or selecting its branch cells, names the call
    # that saves it anew.
    message = (
        r"saved by an earlier version of CyteArc, without the arrays "
        r"trunk_components, trunk_probabilities; call "
        r"ds\.trajectory\.lineage_membership\(curves, band=0\.5, "
        r"distance_score=5\.0, n_bins=10\) again, which saves it anew"
    )
    with pytest.raises(ValueError, match=message):
        store.trajectory.load_lineage_membership(earlier)
    with pytest.raises(ValueError, match=message):
        store.trajectory.branch_cells(earlier, ["a2", "b2"])


def test_a_lineage_whose_sink_cells_fill_a_graph_component_is_refused() -> None:
    # The second component holds only cells of Z's sink, so no cell outside
    # the sinks can reach Z and set its trunk probability.
    branching, fate, _later = _two_components(_TWO_ARMS, None)
    store = branching.store
    curves = store.trajectory.curves(fate, n_points=40, df=6.0)

    with pytest.raises(ValueError, match="no probability to 'z2'.*Fit the curves"):
        store.trajectory.lineage_membership(curves)
    assert not _artifacts(store, "lineage_membership")
    # Curves without Z call the membership of A and B.
    kept = store.trajectory.curves(fate, lineages=["a2", "b2"], n_points=40, df=6.0)
    result = store.trajectory.load_lineage_membership(
        store.trajectory.lineage_membership(kept)
    )
    assert result.members[branching.arms == "trunk"].all()
    assert not result.members[branching.arms == "z"].any()


def test_a_sink_that_the_curves_leave_out_stays_out_of_the_trunk() -> None:
    # The trunk extends backward into 20 cells of D's sink, which the fate
    # map's score, each cell's x coordinate, puts before every other cell.
    # The curves follow A and B only.
    arms, positions, labels, points = _trajectory_cells()
    behind = np.column_stack([-0.05 * np.arange(1, 21), np.zeros(20)])
    branching = _graph_of(
        np.concatenate([arms, np.full(20, "d", dtype=object)]),
        np.concatenate([positions, np.arange(20)]),
        np.concatenate([labels, np.full(20, "d2", dtype=object)]),
        np.vstack([points, behind]),
    )
    store = branching.store
    fate = store.trajectory.fate(
        sink_labels=branching.label_artifact(),
        graph=branching.graph,
        direction=branching.points[:, 0],
        sinks=["a2", "b2", "d2"],
    )
    curves = store.trajectory.curves(fate, lineages=["a2", "b2"], n_points=40, df=6.0)

    result = store.trajectory.load_lineage_membership(
        store.trajectory.lineage_membership(curves)
    )

    # D's cells would fill the earliest tenth of the cells outside A's and
    # B's sinks, and they give A and B nothing; the trunk leaves them out and
    # takes the trunk's first cells, so every trunk cell on the curves
    # follows both. The first six trunk cells, which reach D with
    # probability 0.77 to 0.93, lie off the curves of A and B.
    trunk = branching.arms == "trunk"
    followed = result.members.any(axis=1)
    assert (trunk & followed).sum() >= 50
    assert result.members[trunk & followed].all()
    assert not result.members[branching.arms == "d"].any()


def test_branch_cells_say_why_a_pair_has_no_branch_point(monkeypatch) -> None:
    from cytearc.datastore._operations import trajectory as operations

    branching = _branching()
    store = branching.store
    pseudotime = store.trajectory.load_pseudotime(branching.pseudotime).values
    rule = operations._lineage_members_impl

    # Every cell on the trajectory follows A, and only those after pseudotime
    # 0.9 also follow B, so the shared cells come after the parted ones.
    def late_shared(*args: Any, **kwargs: Any) -> np.ndarray:
        followed = rule(*args, **kwargs).any(axis=1)
        return np.column_stack([followed, followed & (pseudotime > 0.9)])

    monkeypatch.setattr(operations, "_lineage_members_impl", late_shared)
    membership = store.trajectory.lineage_membership(branching.curves)
    (row,) = store.trajectory.load_lineage_membership(
        membership
    ).branch_points.itertuples(index=False)
    assert math.isnan(row.pseudotime)
    assert math.isnan(row.first_progress) and math.isnan(row.second_progress)
    assert 0 < row.shared_cells < row.parted_cells
    with pytest.raises(
        ValueError,
        match=(
            rf"'a2' and 'b2' have no branch point: the cells that follow both, "
            rf"{row.shared_cells} of the pair's {row.shared_cells + row.parted_cells}, "
            "do not come before"
        ),
    ):
        store.trajectory.branch_cells(membership, ["a2", "b2"])

    # Every cell that follows one lineage follows both: the pair never parts.
    monkeypatch.setattr(
        operations,
        "_lineage_members_impl",
        lambda *args, **kwargs: np.repeat(
            rule(*args, **kwargs).any(axis=1)[:, None], 2, axis=1
        ),
    )
    together = store.trajectory.lineage_membership(
        branching.curves, invalidate_cache=True
    )
    with pytest.raises(ValueError, match="no branch point: every cell that follows"):
        store.trajectory.branch_cells(together, ["a2", "b2"])


def test_branch_cells_select_the_cells_around_where_two_lineages_part() -> None:
    branching = _branching()
    store = branching.store
    membership = store.trajectory.lineage_membership(branching.curves)
    result = store.trajectory.load_lineage_membership(membership)
    pseudotime = store.trajectory.load_pseudotime(branching.pseudotime).values
    point = float(result.branch_points["pseudotime"].iloc[0])

    selection = store.trajectory.branch_cells(membership, ["b2", "a2"], window=0.25)

    rows = cell_selection_indices(store.zw, branching.selection)
    selected = np.isin(rows, cell_selection_indices(store.zw, selection))
    pair = result.members.any(axis=1)
    assert selected.sum() == math.ceil(0.25 * pair.sum())
    assert not np.any(selected & ~pair)
    nearest = np.abs(pseudotime - point)
    assert nearest[selected].max() <= nearest[pair & ~selected].min()
    status = store.artifacts.inspect(selection)
    assert status.operation == "select_branch_cells"
    assert status.parameters == {"lineages": ["a2", "b2"], "window": 0.25}
    assert status.inputs["lineage_membership"] == membership.to_dict()
    # The cell readers read each cell's lineages.
    members = store.artifacts.load_values(membership, value="members", cell_ids=False)
    np.testing.assert_array_equal(members.values, result.members)


def test_membership_and_branch_cells_check_their_arguments() -> None:
    branching = _branching()
    store = branching.store
    membership = store.trajectory.lineage_membership(branching.curves)

    for arguments, error, message in (
        ({"band": 0.0}, ValueError, "band must be greater than 0.0"),
        ({"band": 1.5}, ValueError, "band must be at most 1.0"),
        ({"distance_score": -1.0}, ValueError, "distance_score must be greater"),
        ({"n_bins": 0}, ValueError, "n_bins must be"),
        ({"band": "half"}, TypeError, "band must be a real number"),
    ):
        with pytest.raises(error, match=message):
            store.trajectory.lineage_membership(branching.curves, **arguments)
    with pytest.raises(TypeError, match="curves must be an ArtifactRef"):
        store.trajectory.lineage_membership("curves")
    with pytest.raises(TypeError, match="lineages must be a list of two"):
        store.trajectory.branch_cells(membership, "a2")
    with pytest.raises(ValueError, match="name two lineages"):
        store.trajectory.branch_cells(membership, ["a2"])
    with pytest.raises(ValueError, match="'c' is not a lineage of the membership"):
        store.trajectory.branch_cells(membership, ["a2", "c"])
    with pytest.raises(ValueError, match="window must be above 0"):
        store.trajectory.branch_cells(membership, ["a2", "b2"], window=0.0)
    # With a band of 1 no cell's two probabilities match, so the pair never
    # shares a cell.
    unshared = store.trajectory.lineage_membership(branching.curves, band=1.0)
    assert math.isnan(
        store.trajectory.load_lineage_membership(unshared)
        .branch_points["pseudotime"]
        .iloc[0]
    )
    with pytest.raises(ValueError, match="no branch point: they never share a cell"):
        store.trajectory.branch_cells(unshared, ["a2", "b2"])
    assert len(_artifacts(store, "lineage_membership")) == 2


def test_the_membership_loader_refuses_changed_records() -> None:
    branching = _branching()
    store = branching.store
    ref = store.trajectory.lineage_membership(branching.curves)

    with pytest.raises(TypeError, match="ref must be an ArtifactRef"):
        store.trajectory.load_lineage_membership("membership")
    with pytest.raises(ValueError, match="ref must be a lineage_membership"):
        store.trajectory.load_lineage_membership(branching.curves)
    group = artifact_group(store.zw, ref)
    labels = np.asarray(group["values"][:]).astype(str)
    first_shared = int(np.flatnonzero(labels == "shared")[0])
    stored = group["values"]
    changed = labels.copy()
    changed[first_shared] = "a2"
    stored[:] = changed
    with pytest.raises(ValueError, match="payload is invalid"):
        store.trajectory.load_lineage_membership(ref)
    stored[:] = labels
    assert store.trajectory.load_lineage_membership(ref).ref == ref
    # Trunk probabilities must be positive or NaN, with one for each lineage,
    # even when the fingerprint is taken again.
    trunk = group["trunk_probabilities"]
    kept = np.asarray(trunk[:])
    for broken in (-kept, np.full_like(kept, np.nan), kept * np.inf):
        trunk[:] = broken
        _refresh_fingerprint(group, LINEAGE_MEMBERSHIP_PAYLOAD)
        with pytest.raises(ValueError, match="payload is invalid"):
            store.trajectory.load_lineage_membership(ref)
    trunk[:] = kept
    _refresh_fingerprint(group, LINEAGE_MEMBERSHIP_PAYLOAD)
    assert store.trajectory.load_lineage_membership(ref).ref == ref
    _rewrite_parameters(store, ref, band=2.0)
    with pytest.raises(ValueError, match="parameters are malformed"):
        store.trajectory.load_lineage_membership(ref)
    _rewrite_parameters(store, ref, band=0.5)
    _rewrite_inputs(store, ref, cell_selection=branching.graph)
    with pytest.raises(Exception, match="cell"):
        store.trajectory.load_lineage_membership(ref)


def test_a_sink_named_like_a_membership_label_is_refused() -> None:
    with pytest.raises(ValueError, match="'none' cannot be told apart"):
        membership_labels(np.zeros((1, 2), dtype=bool), ["none", "B"])
    frame = pd.DataFrame(columns=["first"])
    with pytest.raises(ValueError, match="missing columns"):
        LineageMembershipResult(
            ref=ArtifactRef(
                scope="datastore", kind="lineage_membership", artifact_id="0" * 64
            ),
            curves=ArtifactRef(
                scope="datastore", kind="lineage_curves", artifact_id="1" * 64
            ),
            cell_selection=ArtifactRef(
                scope="datastore", kind="cell_selection", artifact_id="2" * 64
            ),
            lineages=("A", "B"),
            labels=np.array(["A"]),
            members=np.zeros((1, 2), dtype=bool),
            valid=np.ones(1, dtype=bool),
            distance_limits=np.zeros((2, 3)),
            branch_points=frame,
            trunk_probabilities=pd.DataFrame([[0.5, 0.5]], columns=["A", "B"]),
        )
    with pytest.raises(ValueError, match="one column per lineage"):
        LineageMembershipResult(
            ref=ArtifactRef(
                scope="datastore", kind="lineage_membership", artifact_id="0" * 64
            ),
            curves=ArtifactRef(
                scope="datastore", kind="lineage_curves", artifact_id="1" * 64
            ),
            cell_selection=ArtifactRef(
                scope="datastore", kind="cell_selection", artifact_id="2" * 64
            ),
            lineages=("A", "B"),
            labels=np.array(["A"]),
            members=np.zeros((1, 2), dtype=bool),
            valid=np.ones(1, dtype=bool),
            distance_limits=np.zeros((2, 3)),
            branch_points=pd.DataFrame(
                [["A", "B", np.nan, np.nan, np.nan, 0, 0]],
                columns=list(BRANCH_POINT_COLUMNS),
            ),
            trunk_probabilities=pd.DataFrame([[0.5]], columns=["A"]),
        )


# Drivers where two lineages part, on the 1K PBMC store


def test_drivers_take_the_cells_where_two_lineages_part(
    datastore, lineage_curves, lineage_fate, leiden_clustering, mark_hvgs
):
    from cytearc.storage.feature_selection import read_feature_selection_indices

    membership = datastore.trajectory.lineage_membership(lineage_curves)
    result = datastore.trajectory.load_lineage_membership(membership)
    assert list(zip(result.branch_points["first"], result.branch_points["second"])) == [
        (2, 3),
        (2, 4),
        (3, 4),
    ]
    rows = cell_selection_indices(datastore.zw, result.cell_selection)
    # select_cells reads the labels of the membership.
    shared = datastore.select_cells(membership, include=["shared"])
    np.testing.assert_array_equal(
        np.isin(rows, cell_selection_indices(datastore.zw, shared)),
        result.labels == "shared",
    )
    parting = result.branch_points.dropna(subset=["pseudotime"])
    assert len(parting)
    first, second = parting["first"].iloc[0], parting["second"].iloc[0]

    branch = datastore.trajectory.branch_cells(membership, [first, second])
    drivers = datastore.trajectory.load_drivers(
        datastore.trajectory.drivers(lineage_fate, features=mark_hvgs, cells=branch)
    )

    assert drivers.driver_cells == branch
    fate = datastore.trajectory.load_fate(lineage_fate)
    labels = np.asarray(datastore.artifacts.load_values(leiden_clustering).values)
    in_branch = np.isin(rows, cell_selection_indices(datastore.zw, branch))
    # Drivers leave the sink cells out.
    correlated = in_branch & fate.valid & ~np.isin(labels, [2, 3, 4])
    assert drivers.n_cells == int(correlated.sum()) >= 4
    # Each correlation is Pearson's of the library-size normalized counts,
    # computed here from the raw counts, with a lineage's fate probability.
    parameters = datastore.artifacts.inspect(drivers.ref).parameters
    features = read_feature_selection_indices(
        datastore.zw, "RNA", datastore.features.resolve("RNA", mark_hvgs)
    ).astype(np.int64)
    cells = rows[correlated]
    counts = np.asarray(
        datastore.RNA.rawData._backing.oindex[cells, features], dtype=np.float64
    )
    totals = np.asarray(datastore.cells.fetch_all("RNA_nCounts"), dtype=np.float64)
    values = counts / totals[cells, np.newaxis] * parameters["size_factor"]
    if parameters["normalization"]["log_transform"]:
        values = np.log1p(values)
    for column, lineage in enumerate(fate.sink_labels):
        table = drivers.table[drivers.table["lineage"] == lineage].dropna(
            subset=["r_value"]
        )
        probability = np.asarray(fate.values[correlated, column], dtype=np.float64)
        centered = values - values.mean(axis=0)
        centered_probability = probability - probability.mean()
        with np.errstate(invalid="ignore", divide="ignore"):
            expected = (centered * centered_probability[:, None]).sum(axis=0) / np.sqrt(
                (centered**2).sum(axis=0) * (centered_probability**2).sum()
            )
        positions = np.searchsorted(features, table["feature_index"].to_numpy())
        np.testing.assert_allclose(
            table["r_value"].to_numpy(), expected[positions], rtol=0.0, atol=1e-6
        )


# The pancreas, where the progenitors give Delta a small share


@pytest.mark.slow
def test_every_pancreas_lineage_shares_the_progenitor_trunk(
    bastidas_ponce_data, tmp_path
) -> None:
    from cytearc.datastore.datastore import DataStore
    from cytearc.readers import H5adReader
    from cytearc.writers import H5adToZarr

    # The pancreas fixture analyzed as the lineage tutorials do: Ductal
    # cells are the source and the four endocrine types the sinks.
    reader = H5adReader(bastidas_ponce_data)
    try:
        H5adToZarr(reader, zarr_loc=str(tmp_path / "pancreas.zarr")).dump()
    finally:
        reader.h5.close()
    ds = DataStore(str(tmp_path / "pancreas.zarr"), default_assay="RNA", nthreads=2)
    run = ds.pipeline.rna(
        filtering=False,
        cell_cycle=False,
        hvg={"top_n": 2000},
        pca={"dims": 15},
        neighbors={"k": 11},
        leiden={"partitions": [0.5]},
        doublets=False,
        umap=False,
        markers=False,
    )
    types = np.asarray(ds.cells.fetch("clusters", key="I")).astype(str)
    fates = ["Alpha", "Beta", "Delta", "Epsilon"]
    source_sink = np.zeros(len(types))
    source_sink[types == "Ductal"] = -1 / (types == "Ductal").sum()
    source_sink[np.isin(types, fates)] = 1 / np.isin(types, fates).sum()
    pseudotime_ref = ds.trajectory.pseudotime(
        run["connectivity_map"], ss_vec=source_sink
    )
    pseudotime = ds.trajectory.load_pseudotime(pseudotime_ref)
    labels = ds.snapshot_cluster_labels(
        "clusters", cell_selection=pseudotime.cell_selection
    )
    fate_ref = ds.trajectory.fate(pseudotime_ref, labels, sinks=fates)
    fate = ds.trajectory.load_fate(fate_ref)
    trunk = np.isin(types, ["Ductal", "Ngn3 low EP", "Ngn3 high EP"])
    # The trunk gives Delta under a tenth of the Beta probability.
    split = fate.values[trunk].mean(axis=0)
    assert split[2] < 0.1 * split[1]

    membership = ds.trajectory.lineage_membership(ds.trajectory.curves(fate_ref))
    result = ds.trajectory.load_lineage_membership(membership)

    # Every Ductal and Ngn3 low EP cell on the curves follows all four
    # lineages, as do all but a few Ngn3 high EP cells at the trunk's end.
    followed = result.members.any(axis=1)
    early = np.isin(types, ["Ductal", "Ngn3 low EP"])
    assert result.members[early & followed].all()
    late = (types == "Ngn3 high EP") & followed
    assert result.members[late].all(axis=1).mean() > 0.99
    assert (trunk & followed).sum() > 0.95 * trunk.sum()
    # The trunk probabilities sum to one over the four fates, so a cell's
    # largest divided probability is at least one, and a cell follows a
    # lineage only while its probability stays at least half the trunk's.
    # The Ductal cells, the start of the trunk, give each fate within 0.002
    # of the trunk probability; 0.45 of their mean allows for that.
    ductal = fate.values[types == "Ductal"].mean(axis=0)
    scored = np.where(fate.valid[:, None], fate.values, 0.0)
    assert not np.any(result.members & (scored < 0.45 * ductal))
    # Delta, which the trunk gives 0.04, keeps Pre-endocrine cells as well as
    # its trunk and its sink.
    assert result.members[types == "Pre-endocrine", 2].sum() > 100
    # A cell that follows one lineage alone has given up half the trunk's
    # probability of every other, so that lineage is its most likely fate.
    alone = result.members.sum(axis=1) == 1
    np.testing.assert_array_equal(
        result.members[alone].argmax(axis=1), scored[alone].argmax(axis=1)
    )
    # Every pair parts among the middle 80% of the Pre-endocrine cells by
    # pseudotime, after every Ngn3 high EP cell.
    values = pseudotime.values
    low, high = np.quantile(values[types == "Pre-endocrine"], [0.1, 0.9])
    points = result.branch_points["pseudotime"].to_numpy()
    assert np.all(points > max(low, values[types == "Ngn3 high EP"].max()))
    assert np.all(points < high)
    # The cells around the point where Beta and Delta part are Pre-endocrine
    # and endpoint cells, none of the trunk.
    window = ds.trajectory.branch_cells(membership, ["Beta", "Delta"])
    selected = np.isin(
        cell_selection_indices(ds.zw, result.cell_selection),
        cell_selection_indices(ds.zw, window),
    )
    assert selected.any() and not (selected & trunk).any()
