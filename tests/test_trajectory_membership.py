"""Which lineages each cell follows, and where two lineages part.

The store holds a branching trajectory in two dimensions: a narrow trunk that
splits into branches A and B, which widen toward their ends, and a side
branch C off the middle of the trunk that no sink ends. The graph links each
cell to its five nearest neighbours with Gaussian weights. A pseudotime runs
from the trunk's first half to the second halves of A and B, its fate map
takes those halves as sinks, and curves follow both lineages. The known
lineages give each cell's expected membership: the trunk is shared, the ends
of A and B follow their own lineage, and C, whose cells split their fate
probabilities as trunk cells do, follows none.

Other layouts of the same trunk make the trunk split its probability
unevenly: a sink ten times smaller than the other, a third branch beyond a
gap that takes 4% of the trunk's probability, and a branch four times longer
than the other. Two trajectories in separate graph components, with a fate
map that a score directs, check that each lineage takes its scale from its
own curve. A slow test runs the lineage tutorials' analysis of the pancreas
fixture, whose trunk gives Delta 0.04 against 0.45 for Beta.
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
from cytearc.trajectory.membership import (
    branch_point,
    branch_points,
    branch_window,
    distance_limits,
    lineage_members,
    membership_labels,
    progress_bins,
    weighted_median,
)

from .test_graph_coverage import _memory_graph_store
from .test_graph_feature_projection import _native_chain
from .test_trajectory_record_paths import (
    _copy_store,
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
    """Store cells and the graph of their five nearest neighbours, in a shuffled order."""
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
    labelled = _graph_of(*_trajectory_cells(layout))
    store = labelled.store
    clusters = labelled.label_artifact()
    sinks = [f"{arm.name}2" for arm in layout]
    pseudotime = store.trajectory.pseudotime(
        labelled.graph, source_sink=clusters, sources=_SOURCES, sinks=sinks
    )
    fate = store.trajectory.fate(pseudotime, clusters, sinks=sinks)
    curves = store.trajectory.curves(fate, n_points=40, df=6.0)
    return replace(labelled, pseudotime=pseudotime, fate=fate, curves=curves)


def _branching(layout: tuple[_Arm, ...] = _TWO_ARMS) -> _Branching:
    """Return a private copy of the store of a layout, built once."""
    template = _template(layout)
    return replace(template, store=_copy_store(template.store))


def _artifacts(store: Any, kind: str) -> list[ArtifactRef]:
    return list_artifacts(store.zw, scope="assay", assay="RNA", kind=kind)


def _divided_probabilities(
    branching: _Branching, limits: np.ndarray, n_bins: int
) -> np.ndarray:
    """Divide each lineage's probabilities by the mean at the start of its curve, by hand.

    The start of a curve holds the scored cells outside the sinks whose
    progress on it lies below ``1 / n_bins`` and whose distance from it is
    within its first limit.
    """
    curves = branching.store.trajectory.load_curves(branching.curves)
    sinks = {str(lineage) for lineage in curves.lineages}
    scale = []
    for lineage in range(len(curves.lineages)):
        start = [
            cell
            for cell in np.flatnonzero(curves.valid)
            if branching.labels[cell] not in sinks
            and curves.progress[cell, lineage] * n_bins < 1
            and curves.distance[cell, lineage] <= limits[lineage, 0]
        ]
        scale.append(np.mean(curves.weights[start, lineage]))
    return np.asarray(curves.weights / np.asarray(scale))


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


def test_distance_limits_take_the_median_spread_of_neighbouring_bins() -> None:
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
    # bins whose three neighbours hold it, and the others have no limit.
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


def test_membership_rescales_each_lineage_by_the_start_of_its_curve() -> None:
    # Two lineages over two bins of progress, each with the limit 1. Cells 0
    # to 4 lie in the first bin of both curves and cell 12 in the first bin
    # of A's curve only. Of them, cell 2 is a sink cell, cell 3 lies off
    # both curves and cell 4 is unscored, so the cells at the start of A's
    # curve are 0, 1 and 12, with mean probability (0.9 + 0.7 + 0.6) / 3 =
    # 11 / 15 for A, and those at the start of B's curve are 0 and 1, with
    # mean (0.1 + 0.3) / 2 = 0.2 for B.
    weights = np.array(
        [
            [0.9, 0.1],
            [0.7, 0.3],
            [1.0, 0.0],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.5, 0.5],
            [0.9, 0.1],
            [0.84, 0.16],
            [0.0, 0.0],
            [0.8, 0.2],
            [0.87, 0.13],
            [0.6, 0.4],
        ]
    )
    distance = np.full((13, 2), 0.1)
    distance[[3, 10]] = 5.0
    distance[4, 1] = 5.0
    progress = np.full((13, 2), 0.7)
    progress[:5] = 0.1
    progress[12, 0] = 0.1
    valid = np.ones(13, dtype=bool)
    valid[4] = False
    sinks = np.zeros(13, dtype=bool)
    sinks[2] = True
    limits = np.ones((2, 2))

    members = lineage_members(
        weights, distance, progress, valid, limits, sinks, band=0.5, lineages=["A", 7]
    )

    # Divided by 11 / 15 and 0.2, a cell whose larger divided probability is
    # A's also follows B when p_B / p_A >= 0.5 * 0.2 / (11 / 15) = 3 / 22,
    # about 0.136, and one whose larger is B's also follows A when
    # p_A / p_B >= 0.5 * (11 / 15) / 0.2 = 11 / 6. Cells 5 and 11, with
    # ratios 0.25 and 0.149, follow both, as cell 8 with 0.19 does; cells 0
    # and 7, with 0.111, follow A only, as the sink cell 2 does. Cell 6
    # divides to (0.68, 2.5) and follows B only, and cell 12, whose ratio
    # p_A / p_B is 1.5, does too; cell 1, with 2.33, follows both. Cells 3
    # and 10 lie off the curves, cell 4 is unscored, and cell 9 has no
    # probability, so they follow none.
    np.testing.assert_array_equal(
        members,
        [
            [1, 0],
            [1, 1],
            [1, 0],
            [0, 0],
            [0, 0],
            [1, 1],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 0],
            [0, 0],
            [1, 1],
            [0, 1],
        ],
    )
    # Compared with each cell's largest probability, as before the rescale,
    # cells 1, 5, 8 and 11 would follow A alone and cells 6 and 12 both.
    np.testing.assert_array_equal(
        membership_labels(members, ["A", 7]),
        ["A", "shared", "A", "none", "none", "shared", "7"]
        + ["A", "shared", "none", "none", "shared", "7"],
    )
    with pytest.raises(ValueError, match="'shared' cannot be told apart"):
        membership_labels(members, ["shared", "B"])
    np.testing.assert_array_equal(
        progress_bins(np.array([0.0, 0.29, 0.3, 1.0, np.nan]), 10), [0, 2, 3, 9, 0]
    )


def test_a_rare_lineage_keeps_the_trunk_that_its_progenitors_share() -> None:
    # Three lineages over two bins. Cells 0 and 1 lie in the first bin: a
    # trunk cell that splits 0.70, 0.27 and 0.03, a tenth of the major share
    # for the rare fate, and a cell of no probability, which halves every
    # mean alike.
    trunk = np.array([0.70, 0.27, 0.03])
    weights = np.array(
        [
            trunk,
            [0.0, 0.0, 0.0],
            trunk,
            [0.97, 0.027, 0.003],
            [0.40, 0.10, 0.50],
            [0.10, 0.00, 0.90],
            [0.69, 0.30, 0.01],
        ]
    )
    distance = np.full((7, 3), 0.1)
    progress = np.full((7, 3), 0.5)
    valid = np.ones(7, dtype=bool)
    sinks = np.zeros(7, dtype=bool)
    progress[[0, 1]] = 0.1

    members = lineage_members(
        weights,
        distance,
        progress,
        valid,
        np.ones((3, 2)),
        sinks,
        band=0.5,
        lineages=["major", "minor", "rare"],
    )

    # Divided by half the trunk's split, a cell with the trunk's split scores
    # 2 for every lineage and follows all three, though 0.03 is a tenth of
    # 0.70. A cell committed to the major lineage, whose other probabilities
    # fall in proportion, divides to (2.77, 0.2, 0.2) and follows it alone.
    # A cell with 0.5 for the rare lineage divides to (1.14, 0.74, 33) and
    # follows it alone, as one close to its sink does. A cell that keeps a
    # third of the trunk's rare share divides to (1.97, 2.22, 0.67) and
    # follows the major and minor lineages.
    np.testing.assert_array_equal(
        members,
        [
            [1, 1, 1],
            [0, 0, 0],
            [1, 1, 1],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 0],
        ],
    )
    # Without the cell of no probability, the trunk cell alone sets the
    # scale, and no call changes.
    alone = lineage_members(
        weights[[0, 2, 3, 4, 5, 6]],
        distance[:6],
        progress[[0, 2, 3, 4, 5, 6]],
        valid[:6],
        np.ones((3, 2)),
        sinks[:6],
        band=0.5,
        lineages=["major", "minor", "rare"],
    )
    np.testing.assert_array_equal(alone, members[[0, 2, 3, 4, 5, 6]])


def test_membership_refuses_a_lineage_that_no_cell_at_its_start_reaches() -> None:
    # One bin. Cell 0 is a progenitor of lineages A and B, cell 1 a sink cell
    # of C, cell 2 an unscored cell with a probability for C, and cell 3 a
    # cell off every curve with a probability for C.
    weights = np.array(
        [[0.6, 0.4, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.5, 0.5]]
    )
    distance = np.full((4, 3), 0.1)
    distance[3] = 5.0
    progress = np.full((4, 3), 0.5)
    valid = np.array([True, True, False, True])
    sinks = np.array([False, True, False, False])
    with pytest.raises(ValueError, match="no such cell has a probability for 'C'"):
        lineage_members(
            weights,
            distance,
            progress,
            valid,
            np.ones((3, 1)),
            sinks,
            band=0.5,
            lineages=["A", "B", "C"],
        )
    # Without the progenitor no lineage has a cell at its start, and every
    # such lineage is named.
    with pytest.raises(ValueError, match="for 'A', 'B', 'C'"):
        lineage_members(
            weights[[1, 2, 3]],
            distance[[1, 2, 3]],
            progress[[1, 2, 3]],
            valid[[1, 2, 3]],
            np.ones((3, 1)),
            sinks[[1, 2, 3]],
            band=0.5,
            lineages=["A", "B", "C"],
        )


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
                np.ones(3),
                *_SHAPES[:2],
                _SHAPES[3],
                np.ones((2, 1)),
                np.zeros(3, bool),
                band=0.5,
                lineages=["A", "B"],
            ),
            "weights must be cells by lineages",
        ),
        (
            lambda: lineage_members(
                _SHAPES[2],
                *_SHAPES[:2],
                _SHAPES[3],
                np.ones((3, 1)),
                np.zeros(3, bool),
                band=0.5,
                lineages=["A", "B"],
            ),
            "limits must be lineages by bins",
        ),
        (
            lambda: lineage_members(
                _SHAPES[2],
                *_SHAPES[:2],
                _SHAPES[3],
                np.ones((2, 1)),
                np.zeros(2, bool),
                band=0.5,
                lineages=["A", "B"],
            ),
            "sinks must hold one entry per cell",
        ),
        (
            lambda: lineage_members(
                _SHAPES[2],
                *_SHAPES[:2],
                _SHAPES[3],
                np.ones((2, 1)),
                np.zeros(3, bool),
                band=0.5,
                lineages=["A"],
            ),
            "one lineage per column",
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
    # half; each limit is the median spread of three neighbouring bins.
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
    # its bin and its probability, divided by the mean at the start of the
    # lineage's curve, is at least 0.6 of its largest divided probability.
    divided = _divided_probabilities(branching, result.distance_limits, 8)
    expected = np.zeros_like(result.members)
    undivided = np.zeros_like(result.members)
    for cell in np.flatnonzero(curves.valid):
        on = any(
            curves.distance[cell, lineage]
            <= result.distance_limits[
                lineage, min(int(curves.progress[cell, lineage] * 8), 7)
            ]
            for lineage in range(2)
        )
        for lineage in range(2):
            weight = curves.weights[cell, lineage]
            expected[cell, lineage] = (
                on
                and weight > 0
                and divided[cell, lineage] >= 0.6 * divided[cell].max()
            )
            undivided[cell, lineage] = (
                on and weight > 0 and weight >= 0.6 * curves.weights[cell].max()
            )
    np.testing.assert_array_equal(result.members, expected)
    # Without the division, no trunk cell would follow B.
    trunk = branching.arms == "trunk"
    assert expected[trunk].all() and not undivided[trunk, 1].any()
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
    divided = _divided_probabilities(branching, result.distance_limits, 10)

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
    # It lies where A's and B's divided probabilities leave the band: walking
    # out along each branch, the last cell whose smaller divided probability
    # is at least half the larger one comes before it, and the next cell
    # after it, within the pseudotime of a few cells.
    for arm in ("a", "b"):
        on_arm = np.flatnonzero(branching.arms == arm)
        ordered = on_arm[np.argsort(branching.positions[on_arm])]
        within = divided[ordered].min(axis=1) >= 0.5 * divided[ordered].max(axis=1)
        last_shared = int(np.flatnonzero(within)[-1])
        assert np.all(within[: last_shared + 1])
        assert pseudotime[ordered[last_shared]] - 0.05 < row.pseudotime
        assert row.pseudotime < pseudotime[ordered[last_shared + 1]] + 0.05
    # The progress on each curve is where its fitted pseudotime reaches it.
    curves = store.trajectory.load_curves(branching.curves)
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
    [_SMALL_SINK, _RARE_THIRD, _LONG_BRANCH],
    ids=["sink ten times smaller", "third fate of 4%", "branch four times longer"],
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

    # Every trunk cell follows every lineage, and beyond its first half each
    # branch follows no other lineage, and nearly all of it its own.
    assert result.members[trunk].all()
    for index, arm in enumerate(layout):
        late = (branching.arms == arm.name) & (branching.positions > arm.n_cells / 2)
        assert not np.delete(result.members[late], index, axis=1).any()
        assert result.members[late, index].mean() >= 0.9
    # Every pair parts after the last trunk cell and before the first cell of
    # either sink.
    for row in result.branch_points.itertuples(index=False):
        sinks = np.isin(branching.labels, [row.first, row.second])
        assert pseudotime[trunk].max() < row.pseudotime < pseudotime[sinks].min()


def _two_components(
    first: tuple[_Arm, ...], second: tuple[_Arm, ...] | None
) -> tuple[_Branching, ArtifactRef, np.ndarray]:
    """Store two trajectories as two graph components, with a fate map that a score directs.

    The second trajectory lies 30 above the first, its trunk labelled u1 and
    u2. Without arms, the second component is a cluster of 12 cells that all
    hold the sink label z2. The score is each cell's x coordinate, plus 10 in
    the second component, so every cell of the second comes after the first.
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
    score = branching.points[:, 0] + 10.0 * later
    sinks = [f"{arm.name}2" for arm in (*first, *(second or (_Arm("z", _UP),)))]
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
    # sink holds 2 cells. The earliest cells by the score all lie in the
    # first component, where E and F have no probability.
    branching, fate, later = _two_components(
        _TWO_ARMS, (_Arm("e", _UP), _Arm("f", _DOWN, sink_cells=2))
    )
    store = branching.store
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


def test_a_lineage_whose_curve_starts_among_its_sink_cells_is_refused() -> None:
    # The second component holds only cells of Z's sink, so no cell outside
    # the sinks sets the scale of Z's probabilities.
    branching, fate, _later = _two_components(_TWO_ARMS, None)
    store = branching.store
    curves = store.trajectory.curves(fate, n_points=40, df=6.0)

    with pytest.raises(ValueError, match="probability for 'z2'.*fit the curves"):
        store.trajectory.lineage_membership(curves)
    assert not _artifacts(store, "lineage_membership")
    # Curves without Z call the membership of A and B.
    kept = store.trajectory.curves(fate, lineages=["a2", "b2"], n_points=40, df=6.0)
    result = store.trajectory.load_lineage_membership(
        store.trajectory.lineage_membership(kept)
    )
    assert result.members[branching.arms == "trunk"].all()
    assert not result.members[branching.arms == "z"].any()


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
    # lineages, as do all but a few Ngn3 high EP cells at the trunk's end,
    # and no endpoint cell follows another lineage than its own.
    followed = result.members.any(axis=1)
    early = np.isin(types, ["Ductal", "Ngn3 low EP"])
    assert result.members[early & followed].all()
    late = (types == "Ngn3 high EP") & followed
    assert result.members[late].all(axis=1).mean() > 0.99
    assert (trunk & followed).sum() > 0.95 * trunk.sum()
    for column, endpoint in enumerate(fates):
        cells = types == endpoint
        assert not np.delete(result.members[cells], column, axis=1).any()
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
