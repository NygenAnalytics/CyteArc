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

The membership calls load their fate map and its graph once, and refuse a memory budget that cannot hold the graph before they read it. Copies of the store with arms of thousands of cells and a graph of 11 neighbors, stored as the pipeline stores a graph, check that each call runs within the budget that its refusal names.
"""

import gc
import math
import threading
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

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
    membership_bytes,
    membership_load_bytes,
    stored_membership_bytes,
    membership_labels,
    progress_bins,
    trunk_probabilities,
    weighted_median,
)
from cytearc.utils.memory_budget import MemoryBudgetError

from .test_graph_coverage import _memory_graph_store
from .test_graph_feature_projection import _native_chain
from .test_trajectory_record_paths import (
    _copy_store,
    _count_calls,
    _refresh_fingerprint,
    _rewrite_inputs,
    _rewrite_parameters,
    _rewrite_record,
    _set_budget,
    _write_graph_payload,
)
from tests.storage_helpers import by_cell_id

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


def _write_neighbor_graph(
    store: Any, graph: ArtifactRef, points: np.ndarray, k: int
) -> None:
    """Store each cell's ``k`` nearest cells, weighted exp(-(d/0.15)^2).

    The edges are uint32 pairs and the weights float32, as the pipeline
    stores a graph.
    """
    distances, nearest = cKDTree(points).query(points, k=k + 1)
    group = artifact_group(store.zw, graph)
    pairs = np.column_stack(
        (np.repeat(np.arange(len(points)), k), nearest[:, 1:].ravel())
    )
    group.create_array("edges", data=pairs.astype(np.uint32))
    group.create_array(
        "weights",
        data=np.exp(-((distances[:, 1:].ravel() / 0.15) ** 2)).astype(np.float32),
    )
    group.attrs.update({"n_cells": len(points), "n_neighbors": k})


def _graph_of(
    arms: np.ndarray,
    positions: np.ndarray,
    labels: np.ndarray,
    points: np.ndarray,
    neighbors: int | None = None,
) -> _Branching:
    """Store cells and the graph of their nearest neighbors, in a shuffled order.

    Without ``neighbors``, each cell links to its five nearest cells, stored
    as uint64 pairs and float64 weights; with it, to that many, stored as
    the pipeline stores a graph.
    """
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
    if neighbors is None:
        _write_graph_payload(store, graph, _knn_adjacency(points))
    else:
        _write_neighbor_graph(store, graph, points, neighbors)
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
def _template(
    layout: tuple[_Arm, ...] = _TWO_ARMS, neighbors: int | None = None
) -> _Branching:
    labeled = _graph_of(*_trajectory_cells(layout), neighbors=neighbors)
    store = labeled.store
    clusters = labeled.label_artifact()
    sinks = [f"{arm.name}2" for arm in layout]
    pseudotime = store.trajectory.pseudotime(
        labeled.graph, source_sink=clusters, sources=_SOURCES, sinks=sinks
    )
    fate = store.trajectory.fate(pseudotime, clusters, sinks=sinks)
    curves = store.trajectory.curves(fate, n_points=40, df=6.0)
    return replace(labeled, pseudotime=pseudotime, fate=fate, curves=curves)


def _branching(
    layout: tuple[_Arm, ...] = _TWO_ARMS, neighbors: int | None = None
) -> _Branching:
    """Return a private copy of the store of a layout, built once."""
    template = _template(layout, neighbors)
    return replace(template, store=_copy_store(template.store))


def _long_arms(n_cells: int) -> tuple[_Arm, ...]:
    """Arms A and B of ``n_cells`` cells each, half of them their sink, as long as the default arms."""
    return tuple(
        _Arm(
            name, heading, n_cells=n_cells, sink_cells=n_cells // 2, step=3.0 / n_cells
        )
        for name, heading in (("a", _UP), ("b", _DOWN))
    )


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


def test_a_cell_at_its_bins_distance_limit_lies_on_the_curve() -> None:
    # The one bin's limit is 1. A cell at distance 1 from both curves lies on
    # them and follows both lineages, whose trunk split it keeps; a cell the
    # smallest step farther lies off both curves and follows none.
    beyond = np.nextafter(1.0, 2.0)
    members = _members_by_trunk(
        [[0.8, 0.2], [0.8, 0.2]],
        [0.8, 0.2],
        distance=np.array([[1.0, 1.0], [beyond, beyond]]),
    )

    np.testing.assert_array_equal(members, [[1, 1], [0, 0]])


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
        direction=by_cell_id(branching.store.cells, score),
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


def test_curves_placed_a_cell_at_a_time_leave_the_unscored_cells_unplaced() -> None:
    # A far cluster of 20 cells is a graph component of its own, which the
    # pseudotime from t1 does not reach, so the fate map leaves it unscored.
    arms, positions, labels, points = _trajectory_cells()
    far = np.array([30.0, 30.0]) + np.random.default_rng(4).uniform(
        -0.05, 0.05, size=(20, 2)
    )
    branching = _graph_of(
        np.concatenate([arms, np.full(20, "far", dtype=object)]),
        np.concatenate([positions, np.arange(20)]),
        np.concatenate([labels, np.full(20, "f", dtype=object)]),
        np.vstack([points, far]),
    )
    store = branching.store
    clusters = branching.label_artifact()
    sinks = ["a2", "b2"]
    pseudotime = store.trajectory.pseudotime(
        branching.graph, source_sink=clusters, sources=_SOURCES, sinks=sinks
    )
    fate = store.trajectory.fate(pseudotime, clusters, sinks=sinks)
    unscored = ~store.trajectory.load_fate(fate).valid
    np.testing.assert_array_equal(unscored, branching.arms == "far")

    whole = store.trajectory.load_curves(
        store.trajectory.curves(fate, n_points=40, df=6.0)
    )
    # Blocks of one cell, some of which hold no scored cell.
    single = store.trajectory.load_curves(
        store.trajectory.curves(
            fate, n_points=40, df=6.0, batch_size=1, invalidate_cache=True
        )
    )
    for name in ("progress", "distance"):
        placed = getattr(single, name)
        np.testing.assert_array_equal(placed, getattr(whole, name))
        assert np.isnan(placed[unscored]).all()
        assert not np.isnan(placed[~unscored]).any()


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
        direction=by_cell_id(store.cells, branching.points[:, 0]),
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
    with pytest.raises(
        ValueError, match="^Lineage membership does not use its curves' cells$"
    ):
        store.trajectory.load_lineage_membership(ref)


@cache
def _membership_template() -> tuple[_Branching, ArtifactRef]:
    """The two-armed trajectory with its lineage membership, computed once."""
    branching = _branching()
    return branching, branching.store.trajectory.lineage_membership(branching.curves)


def _membership_copy() -> tuple[_Branching, ArtifactRef]:
    """A private copy of the store of the membership template, and its membership."""
    branching, ref = _membership_template()
    return replace(branching, store=_copy_store(branching.store)), ref


def _replace_array(group: Any, name: str, values: np.ndarray) -> None:
    del group[name]
    group.create_array(name, data=values)


def _add_attribute(group: Any) -> None:
    group.attrs["note"] = "edited by hand"


def _members_as_integers(group: Any) -> None:
    _replace_array(group, "members", np.asarray(group["members"][:], dtype=np.int8))


def _attribute_on_an_array(group: Any) -> None:
    group["valid"].attrs["unit"] = "flag"


def _trunk_components_in_a_column(group: Any) -> None:
    components = np.asarray(group["trunk_components"][:])
    _replace_array(group, "trunk_components", components.reshape(-1, 1))


def _trunk_components_as_floats(group: Any) -> None:
    components = np.asarray(group["trunk_components"][:], dtype=np.float64)
    _replace_array(group, "trunk_components", components)


def _no_graph_component(group: Any) -> None:
    _replace_array(group, "trunk_components", np.zeros(0, dtype=np.int64))
    _replace_array(group, "trunk_probabilities", np.zeros((0, 2)))


def _scored_cell_marked_unscored(group: Any) -> None:
    # The cell follows none and is labeled so, but its curves scored it.
    for name, value in (("valid", False), ("members", False), ("values", "none")):
        values = np.asarray(group[name][:])
        values[0] = value
        group[name][:] = values


def _relabeled_cell(group: Any, label: str, relabeled: str) -> None:
    labels = np.asarray(group["values"][:])
    labels[np.flatnonzero(labels == label)[0]] = relabeled
    group["values"][:] = labels


@pytest.mark.parametrize(
    "edit",
    [
        _add_attribute,
        _members_as_integers,
        _attribute_on_an_array,
        _trunk_components_in_a_column,
        _trunk_components_as_floats,
        _no_graph_component,
        _scored_cell_marked_unscored,
        lambda group: _relabeled_cell(group, "a2", "b2"),
        lambda group: _relabeled_cell(group, "shared", "a2"),
        lambda group: _relabeled_cell(group, "a2", "none"),
    ],
    ids=[
        "attribute-of-its-own",
        "members-as-integers",
        "attribute-on-an-array",
        "trunk-components-in-a-column",
        "trunk-components-as-floats",
        "no-graph-component",
        "scored-cell-marked-unscored",
        "one-lineage-labeled-as-another",
        "shared-cell-labeled-as-one-lineage",
        "member-labeled-none",
    ],
)
def test_a_hand_edited_membership_is_refused_and_computed_anew(edit) -> None:
    branching, ref = _membership_copy()
    store = branching.store
    group = artifact_group(store.zw, ref)
    edit(group)
    # The edited arrays are fingerprinted again, so that the checks of the
    # payload, not its fingerprint, have to find the edit.
    _refresh_fingerprint(group, LINEAGE_MEMBERSHIP_PAYLOAD)

    with pytest.raises(ValueError, match="^Lineage-membership artifact payload is"):
        store.trajectory.load_lineage_membership(ref)
    # A call with the same curves and rule does not reuse the edited
    # membership, and saves the arrays of the template again.
    again = store.trajectory.lineage_membership(branching.curves)
    assert again != ref
    assert len(_artifacts(store, "lineage_membership")) == 2
    template, _ = _membership_template()
    saved = artifact_group(template.store.zw, ref)
    for name in LINEAGE_MEMBERSHIP_PAYLOAD:
        np.testing.assert_array_equal(
            artifact_group(store.zw, again)[name][:], saved[name][:]
        )


def test_the_membership_loader_refuses_a_missing_or_unfinished_record() -> None:
    branching, ref = _membership_copy()
    store = branching.store
    absent = replace(ref, artifact_id="f" * 64)
    message = "^Lineage-membership artifact is unavailable or invalid$"
    with pytest.raises(ValueError, match=message):
        store.trajectory.load_lineage_membership(absent)
    with pytest.raises(ValueError, match=message):
        store.trajectory.branch_cells(absent, ["a2", "b2"])
    # A record of another operation, and an unfinished record, are refused
    # alike.
    _rewrite_record(store, ref, operation="call_lineage_curves")
    with pytest.raises(ValueError, match=message):
        store.trajectory.load_lineage_membership(ref)
    _rewrite_record(store, ref, operation="call_lineage_membership")
    assert store.trajectory.load_lineage_membership(ref).ref == ref
    artifact_group(store.zw, ref).attrs["complete"] = False
    with pytest.raises(ValueError, match=message):
        store.trajectory.load_lineage_membership(ref)


def test_a_membership_moved_out_of_its_curves_scope_is_refused() -> None:
    # A copy of the RNA membership among the store's own artifacts, as a
    # hand-edited store could hold it.
    branching, ref = _membership_copy()
    store = branching.store
    moved = ArtifactRef(
        scope="datastore", kind="lineage_membership", artifact_id=ref.artifact_id
    )
    source = artifact_group(store.zw, ref)
    target = store.zw.require_group(f"artifacts/lineage_membership/{ref.artifact_id}")
    target.attrs.update(dict(source.attrs))
    for name, array in source.arrays():
        target.create_array(name, data=np.asarray(array[:]))

    assert store.trajectory.load_lineage_membership(ref).ref == ref
    with pytest.raises(
        ValueError,
        match="^Lineage-membership artifact does not share its curves' scope$",
    ):
        store.trajectory.load_lineage_membership(moved)


def test_membership_refuses_a_small_budget_before_it_writes() -> None:
    branching, ref = _membership_copy()
    store = branching.store
    # A budget of the curves' three arrays of each cell and lineage, which
    # hold 8 bytes a value, leaves no room for the graph or the membership.
    n_cells = len(branching.arms)
    _set_budget(store, 3 * n_cells * 2 * 8)
    with pytest.raises(MemoryBudgetError) as refused:
        store.trajectory.lineage_membership(branching.curves, invalidate_cache=True)
    error = refused.value
    assert error.budget_bytes == 3 * n_cells * 2 * 8 < error.needed_bytes
    assert _artifacts(store, "lineage_membership") == [ref]
    # The budget that the refusal names computes the membership that a
    # budget of 64 MiB computed.
    _set_budget(store, error.needed_bytes)
    again = store.trajectory.lineage_membership(branching.curves, invalidate_cache=True)
    assert again != ref
    np.testing.assert_array_equal(
        store.trajectory.load_lineage_membership(again).members,
        store.trajectory.load_lineage_membership(ref).members,
    )


def test_the_membership_estimate_pins_its_bytes_per_cell() -> None:
    # The two-armed store's 145 cells on lineages a2 and b2 of a fate map of
    # those two sinks, whose longest label is "shared", 6 characters. The
    # inputs: the curves' progress, distances and weights, 3 * 2 * 8 = 48
    # bytes a cell, the probabilities 2 * 8 = 16, and 40 more, 145 * 104 =
    # 15,080. Calling holds 12 + max(2 * 44, 120) = 132 bytes a cell, 19,140;
    # building the labels 4 * 6 + 18 = 42, 6,090; writing them holds the
    # labels, 145 * 24 = 3,480, 2 * 2 + 20 = 24 bytes a cell, 3,480, and 64
    # KiB of the write's objects, 65,536, beside the check of the written
    # membership, three copies of the labels and 24 + 2 * 2 = 28 bytes a
    # cell, 10,440 + 4,060 = 14,500, and 240 KiB of Zarr's buffers and
    # objects, 245,760, which outgrows two copies of the labels: 72,496 +
    # 260,260 = 332,756, the most.
    assert membership_bytes(145, ["a2", "b2"], 2) == 15_080 + 332_756 == 347_836
    # Loading a saved membership holds its inputs and its check.
    assert membership_load_bytes(145, ["a2", "b2"], 2) == 15_080 + 260_260
    assert stored_membership_bytes(145, ["a2", "b2"]) == 14_500 + 245_760
    # Whatever the cells, writing and checking a membership hold the write's
    # and the check's objects, 65,536 + 245,760 = 311,296.
    fixed = 311_296
    # One cell of one lineage: inputs 24 + 8 + 40, and writing 24 + 22
    # beside the check 72 + 26, more than calling's 132.
    assert membership_bytes(1, ["a2"], 1) == 72 + 46 + 98 + fixed
    # One cell of three lineages: calling 12 + 132 beside writing 24 + 26 and
    # the check 72 + 30.
    assert membership_bytes(1, ["a2", "b2", "d2"], 3) == 136 + 50 + 102 + fixed
    # A sink named by 9 characters, or by a number of 9 digits, widens the
    # label to 36 bytes.
    assert membership_bytes(1, ["lineage_a", "b"], 2) == 104 + 60 + 136 + fixed
    assert membership_bytes(1, [123456789, 2], 2) == 104 + 60 + 136 + fixed
    assert membership_bytes(0, ["a2", "b2"], 2) == fixed


def test_a_membership_that_outweighs_its_graph_is_refused_once() -> None:
    # Sinks named by 41 characters on a graph of 4 neighbors: the membership,
    # which holds the labels at 164 bytes a cell, outweighs the graph's load,
    # so the check of the graph names the membership and its bytes, before
    # the graph is read. Inputs 15,080, beside 145 * 164 = 23,780 bytes of labels, 3,480
    # more and the write's 65,536 while they are written, and three copies of
    # the labels, 4,060 and the check's 245,760 while the written membership
    # is checked: 15,080 + 92,796 + 321,160.
    name = "a" * 40
    layout = (_Arm(name, _UP), _Arm("b" * 40, _DOWN))
    branching = _branching(layout, neighbors=4)
    store = branching.store
    lineages = [f"{name}2", f"{'b' * 40}2"]
    needed = membership_bytes(145, lineages, 2)
    assert needed == 15_080 + 92_796 + 321_160 == 429_036
    reads: list[str] = []
    original = store._store_to_sparse
    store._store_to_sparse = lambda *args, **kwargs: (
        reads.append("graph"),
        original(*args, **kwargs),
    )[1]
    _set_budget(store, needed - 1)
    with pytest.raises(MemoryBudgetError) as refused:
        store.trajectory.lineage_membership(branching.curves, invalidate_cache=True)
    assert refused.value.needed_bytes == needed
    assert refused.value.item == "the lineage membership of every cell"
    assert str(refused.value).startswith(
        f"Calling lineage membership of 145 cells needs about {needed} bytes in "
        "memory for the lineage membership of every cell, but"
    )
    assert reads == []
    # The budget that the refusal names runs the call.
    _set_budget(store, needed)
    store.trajectory.lineage_membership(branching.curves, invalidate_cache=True)
    assert reads == ["graph"]


# The calls whose loaders load the curves, their fate map and its
# pseudotime, which reads the graph to number its components.
_MEMBERSHIP_CALLS: dict[str, Callable[[_Branching, ArtifactRef], Any]] = {
    "lineage_membership": lambda branching, _ref: (
        branching.store.trajectory.lineage_membership(
            branching.curves, invalidate_cache=True
        )
    ),
    "reused lineage_membership": lambda branching, _ref: (
        branching.store.trajectory.lineage_membership(branching.curves)
    ),
    "load_lineage_membership": lambda branching, ref: (
        branching.store.trajectory.load_lineage_membership(ref)
    ),
    "branch_cells": lambda branching, ref: branching.store.trajectory.branch_cells(
        ref, ["a2", "b2"]
    ),
    "load_curves": lambda branching, _ref: branching.store.trajectory.load_curves(
        branching.curves
    ),
}


def test_each_membership_call_loads_the_fate_map_and_its_graph_once(
    monkeypatch,
) -> None:
    branching, ref = _membership_copy()
    calls = _count_calls(
        monkeypatch, branching.store, "_load_fate_inputs", "_load_graph_artifact"
    )
    loads = {}
    for name, call in _MEMBERSHIP_CALLS.items():
        calls.clear()
        call(branching, ref)
        loads[name] = dict(calls)
    assert loads == {
        name: {"_load_fate_inputs": 1, "_load_graph_artifact": 1}
        for name in _MEMBERSHIP_CALLS
    }


# The store's graph links each of its 145 cells to its 5 nearest cells, 725
# entries, which SciPy loads as int32 indices and float64 weights, 12 bytes
# an entry, beside 146 int32 row pointers of 584 bytes. Forming the fuzzy
# union holds the directed matrix, a copy of its transpose for the sum and
# for the product, a sum and a product sized for 1,450 entries, and a compact
# copy of the product, 7 * 725 * 12 + 5 * 584 = 63,820 bytes, beside the
# 64 KiB of Zarr's buffers and the load's objects, 65,536; it keeps the
# union, 1,450 * 12 + 584 = 17,984. Numbering its components is charged
# what SciPy 1.16 holds for float32 weights, three copies of the union and 4
# bytes an entry, and 12 bytes a cell, 3 * 17,984 + 4 * 1,450 + 12 * 145 =
# 61,492, less than forming the union. The positions of the graph's cells
# add 145 * 8 = 1,160 bytes, and what the pseudotime's labels leave held,
# up to an int32 code of the cell's source or sink label and whether it is a
# source, 145 * 5 = 725.
_GRAPH_LOAD_BYTES = 63_820 + 65_536 + 1_160 + 725
# Calling a membership names what it holds, and loading a saved one or
# selecting branch cells what checking it holds beside its inputs, as the
# estimate's pins derive, more than the graph's load.
_MEMBERSHIP_CHECK_BYTES = {
    "lineage_membership": 347_836,
    "reused lineage_membership": 347_836,
    "load_lineage_membership": 275_340,
    "branch_cells": 275_340,
}
# Calling a membership names the membership as what needs its bytes,
# loading a saved one or selecting branch cells the check of the saved
# membership, and the other calls the graph.
_MEMBERSHIP_ITEMS = {
    "lineage_membership": "the lineage membership of every cell",
    "reused lineage_membership": "the lineage membership of every cell",
    "load_lineage_membership": "the check of the saved lineage membership",
    "branch_cells": "the check of the saved lineage membership",
}
# Each call names itself in the refusal of the graph's load.
_GRAPH_LOAD_WORK = {
    "lineage_membership": "Calling lineage membership",
    "reused lineage_membership": "Calling lineage membership",
    "load_lineage_membership": "Loading a lineage membership",
    "branch_cells": "Selecting branch cells",
    "fate": "Mapping fates",
}


@pytest.mark.parametrize(
    "call", [*_MEMBERSHIP_CALLS, "load_fate", "load_pseudotime", "fate"]
)
def test_a_budget_below_the_graph_load_is_refused_before_the_graph_is_read(
    monkeypatch, call: str
) -> None:
    branching, ref = _membership_copy()
    store = branching.store
    labels = branching.label_artifact()
    calls = {
        **_MEMBERSHIP_CALLS,
        "load_fate": lambda branching, _ref: branching.store.trajectory.load_fate(
            branching.fate
        ),
        "load_pseudotime": lambda branching, _ref: (
            branching.store.trajectory.load_pseudotime(branching.pseudotime)
        ),
        "fate": lambda branching, _ref: branching.store.trajectory.fate(
            branching.pseudotime, labels, sinks=["a2", "b2"], invalidate_cache=True
        ),
    }
    selections = list_artifacts(store.zw, scope="datastore", kind="cell_selection")
    reads = _count_calls(monkeypatch, store, "_store_to_sparse")

    # Mapping the fates of the 2 sinks holds the union, 17,984 bytes, 64
    # bytes for each of the 145 cells and the solver's 2 * 1,450 * 12 + 145 *
    # 280 = 75,400, less than the graph's load, which the other calls name.
    needed = _MEMBERSHIP_CHECK_BYTES.get(call, _GRAPH_LOAD_BYTES)
    _set_budget(store, needed - 1)
    with pytest.raises(MemoryBudgetError) as refused:
        calls[call](branching, ref)
    work = _GRAPH_LOAD_WORK.get(call, "Loading a pseudotime")
    if call in _MEMBERSHIP_ITEMS:
        named = item = _MEMBERSHIP_ITEMS[call]
    else:
        named, item = "a graph of 725 edges", "the graph of 725 edges and its work"
    assert str(refused.value) == (
        f"{work} of 145 cells needs about {needed} bytes "
        f"in memory for {named}, but the operation limit is "
        f"{needed - 1} bytes. Raise mem_budget."
    )
    assert refused.value.needed_bytes == needed
    assert refused.value.item == item
    assert reads == {}
    assert _artifacts(store, "lineage_membership") == [ref]
    assert (
        list_artifacts(store.zw, scope="datastore", kind="cell_selection") == selections
    )
    # The budget that the refusal names reads the graph once and runs the call.
    _set_budget(store, refused.value.needed_bytes)
    calls[call](branching, ref)
    assert reads == {"_store_to_sparse": 1}


def test_the_membership_loaders_name_a_graph_that_outweighs_the_membership_check(
    monkeypatch,
) -> None:
    # The two-armed store with each of its 145 cells linked to its 26 nearest
    # cells, 3,770 stored edges of uint32 pairs and float32 weights, which
    # SciPy loads at 8 bytes an entry beside 146 int32 row pointers of 584
    # bytes. Forming the fuzzy union holds 7 * 3,770 * 8 + 5 * 584 = 214,040
    # bytes beside the load's 65,536; numbering the components of the union,
    # at most 7,540 entries and 60,904 bytes, holds it, three copies of it, 4
    # bytes an entry and 12 a cell, 275,516 bytes, less. Beside the cells'
    # positions and their labels' codes, 1,160 + 725 bytes, the load needs
    # 281,461 bytes, more than the 275,340 that checking the saved membership
    # holds with its inputs, so the loaders name the graph.
    branching = _branching(_TWO_ARMS, neighbors=26)
    store = branching.store
    ref = store.trajectory.lineage_membership(branching.curves)
    needed = 214_040 + 65_536 + 1_160 + 725
    assert needed == 281_461 > membership_load_bytes(145, ["a2", "b2"], 2)
    loaders = {
        "Loading a lineage membership": lambda: (
            store.trajectory.load_lineage_membership(ref)
        ),
        "Selecting branch cells": lambda: store.trajectory.branch_cells(
            ref, ["a2", "b2"]
        ),
    }
    reads = _count_calls(monkeypatch, store, "_store_to_sparse")
    for work, call in loaders.items():
        reads.clear()
        _set_budget(store, needed - 1)
        with pytest.raises(MemoryBudgetError) as refused:
            call()
        assert str(refused.value) == (
            f"{work} of 145 cells needs about {needed} bytes in memory for a "
            f"graph of 3770 edges, but the operation limit is {needed - 1} "
            "bytes. Raise mem_budget."
        )
        assert refused.value.item == "the graph of 3770 edges and its work"
        assert reads == {}
        # The budget that the refusal names reads the graph once and runs
        # the call.
        _set_budget(store, needed)
        call()
        assert reads == {"_store_to_sparse": 1}


def _full_collections() -> int:
    """Return how many full collections the cyclic garbage collector has run."""
    return int(gc.get_stats()[-1]["collections"])


def _allocations_near_peak(
    call: Callable[[], Any], peak: int
) -> tuple[tracemalloc.Snapshot | None, int, int]:
    """Run ``call`` traced with 25 frames, and snapshot its allocations near the traced ``peak``.

    From 80% of ``peak`` on, a profiler on every thread snapshots the traced
    allocations each time they rise 16 KiB above the last snapshot, and
    keeps the last one, with the traced bytes that it holds; the snapshots
    themselves are left out of those bytes. Returns the snapshot, those
    bytes and the run's traced peak, which the profiler and the snapshots
    add to.
    """
    lock = threading.Lock()
    state: dict[str, Any] = {"high": int(0.8 * peak), "own": 0, "snapshot": None}

    def watch(_frame: Any, event: str, _arg: Any) -> None:
        if event not in ("return", "c_return"):
            return
        with lock:
            traced = tracemalloc.get_traced_memory()[0] - start - state["own"]
            if traced < state["high"] + 16 * 1024:
                return
            state["snapshot"] = None
            base = tracemalloc.get_traced_memory()[0]
            state["high"] = base - start
            state["snapshot"] = tracemalloc.take_snapshot()
            state["own"] = tracemalloc.get_traced_memory()[0] - base

    tracemalloc.start(25)
    try:
        start = tracemalloc.get_traced_memory()[0]
        threading.setprofile_all_threads(watch)
        try:
            call()
        finally:
            threading.setprofile_all_threads(None)
        run_peak = tracemalloc.get_traced_memory()[1] - start
    finally:
        tracemalloc.stop()
    return state["snapshot"], state["high"], run_peak


def _over_budget_report(
    store: Any,
    calls: dict[str, Callable[[], Any]],
    over: dict[str, tuple[int, int]],
    full_collections: dict[str, tuple[int, int]],
    unbounded: int,
) -> str:
    """Name each call over its budget, with the largest allocations near its peak in a second run.

    The second run follows a full collection, which empties CPython's free
    lists, so that it allocates, and traces, every object that it holds.
    """
    lines = []
    for name, (peak, budget) in over.items():
        before, during = full_collections[name]
        lines.append(
            f"{name}: peak {peak} bytes, budget {budget} bytes "
            f"({peak / budget:.3f} times); full collections since the last "
            f"traced call {before}, during this one {during}"
        )
        _set_budget(store, budget)
        try:
            gc.collect()
            snapshot, held, run_peak = _allocations_near_peak(calls[name], peak)
        except Exception as error:  # noqa: BLE001 - the report names it instead
            lines.append(f"  a second run raised {error!r}")
            continue
        finally:
            _set_budget(store, unbounded)
        if snapshot is None:
            lines.append(
                f"  a second run, after a full collection, peaked at {run_peak} "
                "bytes, under 80% of the first"
            )
            continue
        lines.append(
            f"  a second run, after a full collection, peaked at {run_peak} bytes; "
            f"where it held {held}, the largest blocks were allocated at:"
        )
        for stat in snapshot.statistics("traceback")[:10]:
            lines.append(f"  {stat.size} bytes in {stat.count} blocks, innermost first")
            lines.extend(
                f"    {_short_path(frame.filename)}:{frame.lineno}"
                for frame in reversed(stat.traceback)
            )
        lines.append("  and by the line that allocated them:")
        for stat in snapshot.statistics("lineno")[:20]:
            frame = stat.traceback[0]
            lines.append(
                f"    {stat.size} bytes in {stat.count} blocks at "
                f"{_short_path(frame.filename)}:{frame.lineno}"
            )
    return "\n".join(lines)


def _short_path(path: str) -> str:
    """Return ``path`` from its package directory on, as a log line reads it."""
    if "site-packages/" in path:
        return path.rsplit("site-packages/", 1)[1]
    for package in ("cytearc", "tests"):
        if f"/{package}/" in path:
            return package + "/" + path.rsplit(f"/{package}/", 1)[1]
    return path


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_cells, neighbors, name",
    [
        (3_000, 11, ""),
        (6_000, 11, ""),
        (3_000, 4, "_" + "x" * 23),
        (3_000, 5, ""),
        (1_000, 5, "_" + "x" * 13),
    ],
    ids=[
        "6065-cells",
        "12065-cells",
        "long-labels-few-neighbors",
        "5-neighbors",
        "2065-cells-5-neighbors-16-character-sinks",
    ],
)
def test_trajectory_and_graph_calls_run_within_the_budget_they_name(
    n_cells: int, neighbors: int, name: str
) -> None:
    # Arms of 3,000 and 6,000 cells beside the trunk and the side branch, 6,065
    # and 12,065 cells, whose graph outweighs the tens of kilobytes that the
    # interpreter and Zarr hold for a call; and arms whose labels run to 26
    # characters on a graph of 4 neighbors, where the labels outweigh it.
    layout = tuple(
        _Arm(
            arm + name,
            heading,
            n_cells=n_cells,
            sink_cells=n_cells // 2,
            step=3.0 / n_cells,
        )
        for arm, heading in (("a", _UP), ("b", _DOWN))
    )
    branching = _branching(layout, neighbors=neighbors)
    store = branching.store
    sinks = [f"a{name}2", f"b{name}2"]
    clusters = store.trajectory.load_fate(branching.fate).sink_labels_artifact
    computed = store.trajectory.lineage_membership(branching.curves)
    members = store.trajectory.load_lineage_membership(computed).members
    window = cell_selection_indices(
        store.zw, store.trajectory.branch_cells(computed, sinks)
    )
    again = {"invalidate_cache": True}
    # A score keyed by cell ID, which directs a fate map of its own.
    score = by_cell_id(
        store.cells,
        3.0 * store.trajectory.load_pseudotime(branching.pseudotime).values + 7.0,
    )
    calls = {
        "membership": lambda: store.trajectory.lineage_membership(
            branching.curves, **again
        ),
        "reused membership": lambda: store.trajectory.lineage_membership(
            branching.curves
        ),
        "branch_cells": lambda: store.trajectory.branch_cells(computed, sinks, **again),
        "load_pseudotime": lambda: store.trajectory.load_pseudotime(
            branching.pseudotime
        ),
        "pseudotime": lambda: store.trajectory.pseudotime(
            branching.graph,
            source_sink=clusters,
            sources=["t1"],
            sinks=sinks,
            **again,
        ),
        "fate": lambda: store.trajectory.fate(
            branching.pseudotime, clusters, sinks=sinks, **again
        ),
        "fate by a score": lambda: store.trajectory.fate(
            sink_labels=clusters,
            graph=branching.graph,
            direction=score,
            sinks=sinks,
            **again,
        ),
        "curves": lambda: store.trajectory.curves(
            branching.fate, n_points=40, df=6.0, **again
        ),
        "diffusion": lambda: store.imputation.diffusion(branching.graph, t=2, **again),
        "graph.load": lambda: store.graph.load(branching.graph, symmetric=True),
        "graph.load nearest": lambda: store.graph.load(branching.graph, use_k=2),
    }

    def check(call_name: str, result: Any) -> None:
        if call_name == "membership":
            np.testing.assert_array_equal(
                store.trajectory.load_lineage_membership(result).members, members
            )
        elif call_name == "branch_cells":
            np.testing.assert_array_equal(
                cell_selection_indices(store.zw, result), window
            )

    _assert_calls_run_within_the_budget_they_name(store, calls, check=check)


def _assert_calls_run_within_the_budget_they_name(
    store: Any,
    calls: dict[str, Callable[[], Any]],
    *,
    check: Callable[[str, Any], None] | None = None,
) -> None:
    """Assert that each call runs within the budget that its refusals name, traced after a full collection.

    Each call starts at a budget of one byte and follows its refusals, each
    of which names a budget that runs the call or at which a later step
    refuses and names more. At the last budget the call runs again, traced,
    and ``check(name, result)`` checks what it returned at the store's own
    budget. A call over its budget is traced once more, since CPython grows
    its own tables, such as the one of interned strings, now and then as a
    session interns names, and the call that grows one holds the new table
    once, 7.7 MB when it makes room for 349,525 strings, which no step
    charges. A call over its budget in both runs is named with the
    allocations near its peak.
    """
    unbounded = store.memoryBytes
    peaks = {}
    # A full collection empties CPython's free lists, after which a call's
    # small objects are allocated, and traced, anew, so each traced call
    # follows one and is measured at its most; the report counts the full
    # collections around a call over its budget.
    full_collections = {}
    last_full = _full_collections()
    try:
        for call_name, call in calls.items():
            budget = 1
            for _ in range(5):
                _set_budget(store, budget)
                try:
                    result = call()
                    break
                except MemoryBudgetError as refused:
                    budget = refused.needed_bytes
            gc.collect()
            full = _full_collections()
            for _ in range(2):
                tracemalloc.start()
                try:
                    before, _ = tracemalloc.get_traced_memory()
                    result = call()
                    _, peak = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()
                peaks[call_name] = (peak - before, budget)
                if peak - before <= budget:
                    break
                gc.collect()
            full_collections[call_name] = (
                full - last_full,
                _full_collections() - full,
            )
            last_full = _full_collections()
            _set_budget(store, unbounded)
            if check is not None:
                check(call_name, result)
    finally:
        _set_budget(store, unbounded)
    over = {name: held for name, held in peaks.items() if held[0] > held[1]}
    assert not over, _over_budget_report(
        store, calls, over, full_collections, unbounded
    )


def _cells_with_ids(
    layout: tuple[_Arm, ...], ids: np.ndarray, chunk_rows: int
) -> tuple[Any, ArtifactRef, ArtifactRef, pd.Series]:
    """Store a layout's cells in a shuffled order, with ``ids`` as stored, and the graph of each cell's five nearest cells.

    Returns the store, the graph, a selection of every cell but those of the
    side branch C, so that its rows lie scattered among the store's, and a
    source/sink vector over the selected cells keyed by their IDs: the
    first half of the trunk its sources and the second halves of A and B its
    sinks. The IDs are stored in chunks of ``chunk_rows``.
    """
    arms, _positions, labels, points = _trajectory_cells(layout)
    order = np.random.default_rng(10).permutation(len(arms))
    arms, labels, points = arms[order], labels[order], points[order]
    store = _memory_graph_store()
    table = store.z.create_group("cellData")
    table.create_array("ids", data=ids, chunks=(chunk_rows,))
    table.create_array("I", data=np.ones(len(arms), dtype=bool))
    table.create_array("kept", data=arms != "c")
    store.cells = MetaData(table)
    selections = [
        resolve_stored_selection_artifact(
            store.zw,
            table_path="cellData",
            id_column="ids",
            source_column=column,
            scope="datastore",
            kind="cell_selection",
            operation="test_selection",
            parameters={},
            inputs={},
        )
        for column in ("I", "kept")
    ]
    graph, _neighbors, coordinates = _native_chain(
        store.zw, "RNA", cell_selection=selections[0]
    )
    _write_neighbor_graph(store, graph, points, 5)
    artifact_group(store.zw, coordinates)["data"][:] = points
    kept_labels = labels[arms != "c"]
    vector = np.zeros(len(kept_labels))
    vector[kept_labels == "t1"] = -1.0 / np.sum(kept_labels == "t1")
    sinks = np.isin(kept_labels, ["a2", "b2"])
    vector[sinks] = 1.0 / np.sum(sinks)
    return store, graph, selections[1], by_cell_id(store.cells, vector, key="kept")


def test_a_pseudotime_of_some_cells_reads_variable_width_ids_without_zarr_selections(
    monkeypatch,
) -> None:
    from .test_metadata_contracts import _refuse_zarr_selections

    # IDs of 26 characters, which NumPy keeps outside its 16-byte elements,
    # stored as fixed-width text, as CyteArc writes them, and as
    # variable-width text, as another tool writes them with Zarr's
    # dtype=str. Zarr's selection of scattered rows of the latter crashes
    # NumPy 2.1, and the pseudotime of some of the graph's cells reads their
    # IDs to check its source/sink snapshot.
    ids = np.asarray([f"cell-{row:05d}-" + "x" * 15 for row in range(145)])
    loaded = {}
    for kind, stored in (
        ("fixed", ids),
        ("variable", ids.astype(np.dtypes.StringDType())),
    ):
        if kind == "variable":
            _refuse_zarr_selections(monkeypatch)
        store, graph, kept, ss_vec = _cells_with_ids(_TWO_ARMS, stored, 32)
        ref = store.trajectory.pseudotime(graph, cells=kept, ss_vec=ss_vec)
        loaded[kind] = store.trajectory.load_pseudotime(ref)
    assert loaded["variable"].valid.sum() == 140
    np.testing.assert_array_equal(loaded["variable"].valid, loaded["fixed"].valid)
    np.testing.assert_array_equal(loaded["variable"].values, loaded["fixed"].values)


@pytest.mark.slow
@pytest.mark.parametrize(
    "character", ["x", "\N{GRINNING FACE}"], ids=["ascii", "beyond-ascii"]
)
def test_pseudotime_calls_that_read_variable_width_ids_run_within_the_budget_they_name(
    character: str,
) -> None:
    # 2,065 cells whose IDs of 300 characters, stored as variable-width text,
    # outweigh their graph of five neighbors: charged without their
    # characters, the pseudotime of some of the cells held 3.42 times the
    # budget that its refusals named with ASCII IDs and 7.23 times with IDs
    # beyond it, and its load 2.21 and 7.36 times.
    n_cells = 60 + 2 * 1_000 + 5
    ids = np.asarray([f"c{row:05d}".ljust(300, character) for row in range(n_cells)])
    store, graph, kept, ss_vec = _cells_with_ids(
        _long_arms(1_000), ids.astype(np.dtypes.StringDType()), 1_024
    )
    ref = store.trajectory.pseudotime(graph, cells=kept, ss_vec=ss_vec)
    _assert_calls_run_within_the_budget_they_name(
        store,
        {
            "pseudotime of some cells": lambda: store.trajectory.pseudotime(
                graph, cells=kept, ss_vec=ss_vec, invalidate_cache=True
            ),
            "load_pseudotime": lambda: store.trajectory.load_pseudotime(ref),
        },
    )


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


def _membership_of_three_lineages(**changed: Any) -> LineageMembershipResult:
    """A membership of two cells and lineages A, B and C, with some fields changed."""
    fields: dict[str, Any] = {
        "ref": ArtifactRef(
            scope="datastore", kind="lineage_membership", artifact_id="0" * 64
        ),
        "curves": ArtifactRef(
            scope="datastore", kind="lineage_curves", artifact_id="1" * 64
        ),
        "cell_selection": ArtifactRef(
            scope="datastore", kind="cell_selection", artifact_id="2" * 64
        ),
        "lineages": ("A", "B", "C"),
        "labels": np.array(["A", "shared"]),
        "members": np.array([[True, False, False], [True, True, False]]),
        "valid": np.ones(2, dtype=bool),
        "distance_limits": np.zeros((3, 4)),
        "branch_points": pd.DataFrame(
            [
                [first, second, np.nan, np.nan, np.nan, 0, 0]
                for first, second in (("A", "B"), ("A", "C"), ("B", "C"))
            ],
            columns=list(BRANCH_POINT_COLUMNS),
        ),
        "trunk_probabilities": pd.DataFrame(
            [[0.5, 0.25, 0.25]], columns=["A", "B", "C"]
        ),
    }
    return LineageMembershipResult(**{**fields, **changed})


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        (
            {"valid": np.ones((2, 1), dtype=bool)},
            "Membership validity must be one-dimensional",
        ),
        ({"labels": np.array(["A"])}, "Membership labels must hold one label per cell"),
        (
            {"labels": np.array([["A", "shared"]])},
            "Membership labels must hold one label per cell",
        ),
        (
            {"members": np.zeros((2, 2), dtype=bool)},
            "Membership must be cells by lineages",
        ),
        ({"members": np.zeros((3, 3), dtype=bool)}, "Membership must be cells by"),
        ({"distance_limits": np.zeros(3)}, "Distance limits must be lineages by bins"),
        (
            {"distance_limits": np.zeros((2, 4))},
            "Distance limits must be lineages by bins",
        ),
        # Three lineages part in three pairs, not one.
        (
            {
                "branch_points": pd.DataFrame(
                    [["A", "B", np.nan, np.nan, np.nan, 0, 0]],
                    columns=list(BRANCH_POINT_COLUMNS),
                )
            },
            "Branch points need one row per pair of lineages",
        ),
    ],
)
def test_a_membership_refuses_arrays_that_do_not_fit_its_cells_and_lineages(
    changed, message
) -> None:
    with pytest.raises(ValueError, match=f"^{message}"):
        _membership_of_three_lineages(**changed)


def test_a_membership_of_fitting_arrays_holds_read_only_copies() -> None:
    members = np.array([[True, False, False], [True, True, False]])
    result = _membership_of_three_lineages(members=members)
    members[0, 0] = False
    assert result.members.tolist() == [[True, False, False], [True, True, False]]
    assert not result.members.flags.writeable
    assert len(result.branch_points) == 3


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

# The cell types of the pancreas file in its order, with its colors.
PANCREAS_COLORS = {
    "Ductal": "#8fbc8f",
    "Ngn3 low EP": "#f4a460",
    "Ngn3 high EP": "#fdbf6f",
    "Pre-endocrine": "#ff7f00",
    "Beta": "#b2df8a",
    "Alpha": "#1f78b4",
    "Delta": "#6a3d9a",
    "Epsilon": "#cab2d6",
}


@pytest.mark.slow
def test_every_pancreas_lineage_shares_the_progenitor_trunk(
    bastidas_ponce_data, tmp_path
) -> None:
    from cytearc.datastore.datastore import DataStore
    from cytearc.plotting._display import stored_artifact_scale
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
        run["connectivity_map"], ss_vec=by_cell_id(ds.cells, source_sink)
    )
    pseudotime = ds.trajectory.load_pseudotime(pseudotime_ref)
    labels = ds.snapshot_cluster_labels(
        "clusters", cell_selection=pseudotime.cell_selection
    )
    # The sink labels keep the file's order and colors of the cell types.
    sink_scale = stored_artifact_scale(ds, labels)
    assert sink_scale is not None
    assert sink_scale.palette == PANCREAS_COLORS
    assert sink_scale.order == tuple(PANCREAS_COLORS)
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


def test_labels_are_read_and_let_go_before_the_graph_and_kept_as_codes(
    monkeypatch,
) -> None:
    import weakref

    from cytearc.datastore._operations import trajectory as operations

    # Sinks named by 41 characters: the labels' text is read, matched against
    # the listed labels and let go before the graph is read, and the fate
    # map keeps which sink each cell holds as a byte.
    layout = (_Arm("a" * 40, _UP), _Arm("b" * 40, _DOWN))
    branching = _branching(layout, neighbors=4)
    store = branching.store
    texts: list[weakref.ref[Any]] = []
    resolve = operations.resolve_cell_aligned_artifact

    def resolved(*args: Any, **kwargs: Any) -> Any:
        result = resolve(*args, **kwargs)
        if np.asarray(result.values).dtype.kind == "U":
            texts.append(weakref.ref(result.values))
        return result

    alive_at_read: list[list[bool]] = []
    load = store._load_graph_artifact

    def read(*args: Any, **kwargs: Any) -> Any:
        alive_at_read.append([text() is not None for text in texts])
        return load(*args, **kwargs)

    monkeypatch.setattr(operations, "resolve_cell_aligned_artifact", resolved)
    monkeypatch.setattr(store, "_load_graph_artifact", read)
    inputs = store._load_fate_inputs(branching.fate)
    assert alive_at_read == [[False]]
    assert len(texts) == 2 and all(text() is None for text in texts)
    assert inputs.sink_values.codes.dtype == np.int8
    np.testing.assert_array_equal(
        np.asarray(inputs.sink_values == f"{'a' * 40}2"),
        branching.labels == f"{'a' * 40}2",
    )


def test_a_label_read_that_outweighs_the_graph_is_refused_before_it_reads() -> None:
    # Labels of 41 characters on a graph of 4 neighbors: the graph's check
    # passes, and reading the labels, which needs more, refuses before the
    # graph is read.
    layout = (_Arm("a" * 40, _UP), _Arm("b" * 40, _DOWN))
    branching = _branching(layout, neighbors=4)
    store = branching.store
    reads: list[str] = []
    original = store._store_to_sparse
    store._store_to_sparse = lambda *args, **kwargs: (
        reads.append("graph"),
        original(*args, **kwargs),
    )[1]
    _set_budget(store, 1)
    with pytest.raises(MemoryBudgetError) as graph_check:
        store.trajectory.load_pseudotime(branching.pseudotime)
    _set_budget(store, graph_check.value.needed_bytes)
    with pytest.raises(MemoryBudgetError) as label_read:
        store.trajectory.load_pseudotime(branching.pseudotime)
    assert label_read.value.needed_bytes > graph_check.value.needed_bytes
    assert "values' of the metadata_snapshot artifact" in str(label_read.value)
    assert reads == []
    _set_budget(store, label_read.value.needed_bytes)
    store.trajectory.load_pseudotime(branching.pseudotime)
    assert reads == ["graph"]
