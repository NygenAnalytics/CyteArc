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
_SINKS = ["a2", "b2"]


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


def _trajectory_cells() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Each cell's arm, position along it, cluster label and coordinates.

    The trunk runs from (0, 0) along x in steps of 0.05 for 60 cells and
    scatters across by up to 0.01. A and B leave its end at 45 degrees up
    and down for 40 cells each, in steps of 0.075, and scatter across their
    direction by up to 0.3 at their ends. C rises from the trunk's middle,
    at x = 1.5, in steps of 0.1 for 5 cells, starting 0.2 above it.
    """
    rng = np.random.default_rng(210)
    arms, positions, labels, points = [], [], [], []
    trunk = np.arange(60)
    arms += ["trunk"] * 60
    positions += trunk.tolist()
    labels += ["t1"] * 30 + ["t2"] * 30
    points.append(np.column_stack([0.05 * trunk, rng.uniform(-0.01, 0.01, size=60)]))
    start = np.array([3.0, 0.0])
    for arm, direction in (("a", 1.0), ("b", -1.0)):
        along = np.arange(1, 41)
        heading = np.array([1.0, direction]) / math.sqrt(2.0)
        across = np.array([-direction, 1.0]) / math.sqrt(2.0)
        spread = rng.uniform(-1.0, 1.0, size=40) * 0.3 * along / 40
        arms += [arm] * 40
        positions += along.tolist()
        labels += [f"{arm}1"] * 20 + [f"{arm}2"] * 20
        points.append(
            start + np.outer(0.075 * along, heading) + np.outer(spread, across)
        )
    side = np.arange(5)
    arms += ["c"] * 5
    positions += side.tolist()
    labels += ["c"] * 5
    points.append(np.column_stack([np.full(5, 1.5), 0.2 + 0.1 * side]))
    return (
        np.asarray(arms, dtype=object),
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


@cache
def _template() -> _Branching:
    arms, positions, labels, points = _trajectory_cells()
    # The cells lie in the table in a shuffled order.
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
    labelled = _Branching(
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
    clusters = labelled.label_artifact()
    pseudotime = store.trajectory.pseudotime(
        graph, source_sink=clusters, sources=_SOURCES, sinks=_SINKS
    )
    fate = store.trajectory.fate(pseudotime, clusters, sinks=_SINKS)
    curves = store.trajectory.curves(fate, n_points=40, df=6.0)
    return replace(labelled, pseudotime=pseudotime, fate=fate, curves=curves)


def _branching() -> _Branching:
    """Return a private copy of the store, built once."""
    template = _template()
    return replace(template, store=_copy_store(template.store))


def _artifacts(store: Any, kind: str) -> list[ArtifactRef]:
    return list_artifacts(store.zw, scope="assay", assay="RNA", kind=kind)


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


def test_membership_takes_a_band_of_probabilities_on_the_trajectory() -> None:
    weights = np.array(
        [[0.5, 0.5], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5], [0.0, 0.0], [0.5, 0.5]]
    )
    # Cells 0 to 2 and 4 lie on curve 0 and cell 3 on neither; cell 5 is unscored.
    distance = np.array(
        [[0.1, 5.0], [0.1, 5.0], [0.1, 5.0], [9.0, 9.0], [0.1, 0.1], [0.1, 0.1]]
    )
    progress = np.full((6, 2), 0.5)
    valid = np.array([True, True, True, True, True, False])
    limits = np.array([[1.0], [1.0]])

    members = lineage_members(weights, distance, progress, valid, limits, band=0.5)

    # 0.3 is less than half of 0.7, and 0.4 at least half of 0.6.
    np.testing.assert_array_equal(
        members,
        [[1, 1], [1, 0], [1, 1], [0, 0], [0, 0], [0, 0]],
    )
    np.testing.assert_array_equal(
        membership_labels(members, ["A", 7]),
        ["shared", "A", "shared", "none", "none", "none"],
    )
    with pytest.raises(ValueError, match="'shared' cannot be told apart"):
        membership_labels(members, ["shared", "B"])
    np.testing.assert_array_equal(
        progress_bins(np.array([0.0, 0.29, 0.3, 1.0, np.nan]), 10), [0, 2, 3, 9, 0]
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
    with pytest.raises(ValueError, match="both shared and parted"):
        branch_point(pseudotime, shared, shared)
    # The window holds the share of the pair's cells nearest the point.
    window = branch_window(pseudotime, np.ones(7, bool), 0.35, 0.5)
    np.testing.assert_array_equal(window, [0, 1, 1, 1, 1, 0, 0])
    with pytest.raises(ValueError, match="no branch point"):
        branch_window(pseudotime, np.ones(7, bool), math.nan, 0.5)


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
                np.ones(3), *_SHAPES[:2], _SHAPES[3], np.ones((2, 1)), band=0.5
            ),
            "weights must be cells by lineages",
        ),
        (
            lambda: lineage_members(
                _SHAPES[2], *_SHAPES[:2], _SHAPES[3], np.ones((3, 1)), band=0.5
            ),
            "limits must be lineages by bins",
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
    branching = _branching()
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
    # its bin and its probability is at least 0.6 of its largest.
    expected = np.zeros_like(result.members)
    for cell in np.flatnonzero(curves.valid):
        on = any(
            curves.distance[cell, lineage]
            <= result.distance_limits[
                lineage, min(int(curves.progress[cell, lineage] * 8), 7)
            ]
            for lineage in range(2)
        )
        largest = curves.weights[cell].max()
        for lineage in range(2):
            weight = curves.weights[cell, lineage]
            expected[cell, lineage] = on and weight > 0 and weight >= 0.6 * largest
    np.testing.assert_array_equal(result.members, expected)
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
    fate = store.trajectory.load_fate(branching.fate).values

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
    assert row.pseudotime == pytest.approx(best)
    assert row.shared_cells == shared.sum()
    assert row.parted_cells == (pair & ~shared).sum()
    # It lies where A's and B's probabilities leave the band: walking out
    # along each branch, the last cell whose smaller probability is at least
    # half the larger one comes before it, and the next cell after it, within
    # the pseudotime of a few cells.
    for arm in ("a", "b"):
        on_arm = np.flatnonzero(branching.arms == arm)
        ordered = on_arm[np.argsort(branching.positions[on_arm])]
        within = fate[ordered].min(axis=1) >= 0.5 * fate[ordered].max(axis=1)
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
    with pytest.raises(ValueError, match="no branch point"):
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
