"""Trajectory calls that score some of a graph's cells.

The store holds a branching trajectory in two dimensions: a trunk that splits
into branches A and B, a branch C off the middle of the trunk, a separate ring
of cells, and two cells that the graph leaves out. The graph links each cell
to its five nearest neighbours with Gaussian weights, in one direction per
pair, as a k-nearest-neighbour graph does. Scoring some of the graph's cells
with ``cells=`` must give what a graph built by hand on those cells alone
gives, a graph that holds the graph's edges among them with their weights.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cache
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import cytearc.plotting as splt
from cytearc.metadata import MetaData
from cytearc.metadata.selection import cell_selection_indices
from cytearc.storage.artifacts import ArtifactRef, artifact_group, list_artifacts
from cytearc.storage.feature_selection import read_feature_selection_indices
from cytearc.storage.selections import (
    resolve_metadata_snapshot,
    resolve_stored_selection_artifact,
)
from cytearc.trajectory.curves import layout_rows
from cytearc.utils.arrays import sorted_places

from .clone_helpers import write_selection_embedding
from .test_graph_coverage import _memory_graph_store
from .test_graph_feature_projection import _native_chain
from .test_trajectory_record_paths import (
    _copy_store,
    _rewrite_inputs,
    _write_graph_payload,
)

# Each arm's cells by label, in order along the arm.
_ARMS = {
    "trunk": ((0.0, 0.0), (0.1, 0.0), 24),
    "a": ((2.4, 0.05), (0.1, 0.1), 20),
    "b": ((2.4, -0.05), (0.1, -0.1), 20),
    "c": ((1.2, 0.15), (0.0, 0.1), 14),
}
_RING = 8
# The table rows of the two cells that the graph leaves out.
_OUTSIDE_ROWS = (5, 40)
_SOURCES = ["t1"]
_SINKS = ["a2", "b2"]


def _cluster_labels(arm: str, n_cells: int) -> list[str]:
    """Each arm's first half and second half are clusters of their own."""
    names = {"trunk": "t", "a": "a", "b": "b"}
    if arm not in names:
        return [arm] * n_cells
    half = n_cells // 2
    return [f"{names[arm]}1"] * half + [f"{names[arm]}2"] * (n_cells - half)


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
class _Store:
    """The store and the facts of its cells, by table row."""

    store: Any
    graph: ArtifactRef
    graph_cells: ArtifactRef
    coordinates: ArtifactRef
    #: Selections of some cells: ``part`` (trunk, A and B), ``split`` (A, B
    #: and the trunk without its last four cells) and ``wide`` (every cell of
    #: the table, the two outside the graph among them).
    selections: dict[str, ArtifactRef]
    #: Graphs built on the cells of ``part`` and ``split`` alone, which hold
    #: the graph's edges among them, and their coordinates.
    alone: dict[str, ArtifactRef]
    alone_coordinates: dict[str, ArtifactRef]
    arms: np.ndarray
    labels: np.ndarray
    points: np.ndarray
    adjacency: np.ndarray
    graph_rows: np.ndarray

    def rows(self, name: str) -> np.ndarray:
        """The table rows of a selection's cells, ascending."""
        if name == "graph":
            return self.graph_rows
        return cell_selection_indices(self.store.zw, self.selections[name])

    def selection(self, name: str) -> ArtifactRef:
        return self.graph_cells if name == "graph" else self.selections[name]

    def label_artifact(self, name: str) -> ArtifactRef:
        """Cluster labels of the cells of a selection."""
        rows = self.rows(name)
        return resolve_metadata_snapshot(
            self.store.zw,
            values=self.labels[rows],
            row_ids=np.asarray([f"c{row}" for row in rows]),
            operation="test_cluster_labels",
            parameters={},
            inputs={"cell_selection": self.selection(name)},
            source_columns=["cluster"],
        )

    def layout(self, name: str) -> ArtifactRef:
        """A layout of a selection's cells: the coordinates turned by 30 degrees."""
        angle = np.deg2rad(30.0)
        turn = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        return write_selection_embedding(
            self.store, self.selection(name), self.points[self.rows(name)] @ turn.T
        )

    def artifacts(self, kind: str) -> list[ArtifactRef]:
        return list_artifacts(self.store.zw, scope="assay", assay="RNA", kind=kind)


def _selection(store: Any, column: str, flags: np.ndarray) -> ArtifactRef:
    store.z["cellData"].create_array(column, data=flags)
    return resolve_stored_selection_artifact(
        store.zw,
        table_path="cellData",
        id_column="ids",
        source_column=column,
        scope="datastore",
        kind="cell_selection",
        operation=f"test_{column}_cells",
        parameters={},
        inputs={},
    )


def _chain(
    store: Any,
    selection: ArtifactRef,
    adjacency: np.ndarray,
    points: np.ndarray,
) -> tuple[ArtifactRef, ArtifactRef]:
    """Store a graph and its two-dimensional coordinates over a selection."""
    graph, _neighbors, coordinates = _native_chain(
        store.zw, "RNA", cell_selection=selection
    )
    _write_graph_payload(store, graph, adjacency)
    artifact_group(store.zw, coordinates)["data"][:] = points
    return graph, coordinates


@cache
def _template() -> _Store:
    arms: list[str] = []
    labels: list[str] = []
    points: list[np.ndarray] = []
    for arm, (start, step, n_cells) in _ARMS.items():
        arms += [arm] * n_cells
        labels += _cluster_labels(arm, n_cells)
        points.append(np.asarray(start) + np.outer(np.arange(n_cells), step))
    angles = 2.0 * np.pi * np.arange(_RING) / _RING
    points.append(
        np.column_stack([4.0 + 0.1 * np.cos(angles), 3.0 + 0.1 * np.sin(angles)])
    )
    arms += ["ring"] * _RING
    labels += ["ring"] * _RING
    arm_of = np.asarray(arms, dtype=object)
    along = np.vstack(points)
    in_part = np.isin(arm_of, ["trunk", "a", "b"])
    # The trunk's last four cells join it to A and B.
    in_split = in_part & ~((arm_of == "trunk") & (along[:, 0] > 1.95))
    # The graph's cells lie in the table in a shuffled order, around the two
    # cells it leaves out.
    order = np.random.default_rng(208).permutation(len(arms))
    n_table = len(arms) + len(_OUTSIDE_ROWS)
    graph_rows = np.setdiff1d(np.arange(n_table), _OUTSIDE_ROWS)
    table_arms = np.full(n_table, "outside", dtype=object)
    table_labels = np.full(n_table, "outside", dtype=object)
    table_points = np.zeros((n_table, 2))
    masks = {"part": np.zeros(n_table, dtype=bool), "split": np.zeros(n_table, bool)}
    table_arms[graph_rows] = arm_of[order]
    table_labels[graph_rows] = np.asarray(labels, dtype=object)[order]
    table_points[graph_rows] = along[order]
    masks["part"][graph_rows] = in_part[order]
    masks["split"][graph_rows] = in_split[order]

    store = _memory_graph_store()
    table = store.z.create_group("cellData")
    table.create_array("ids", data=np.asarray([f"c{row}" for row in range(n_table)]))
    table.create_array("I", data=np.ones(n_table, dtype=bool))
    store.cells = MetaData(table)
    in_graph = np.ones(n_table, dtype=bool)
    in_graph[list(_OUTSIDE_ROWS)] = False
    graph_cells = _selection(store, "in_graph", in_graph)
    adjacency = _knn_adjacency(table_points[graph_rows])
    graph, coordinates = _chain(store, graph_cells, adjacency, table_points[graph_rows])
    selections = {
        "part": _selection(store, "part", masks["part"]),
        "split": _selection(store, "split", masks["split"]),
        "wide": _selection(store, "wide", np.ones(n_table, dtype=bool)),
    }
    alone: dict[str, ArtifactRef] = {}
    alone_coordinates: dict[str, ArtifactRef] = {}
    for name in ("part", "split"):
        places = sorted_places(np.flatnonzero(masks[name]), graph_rows)
        assert (places >= 0).all()
        alone[name], alone_coordinates[name] = _chain(
            store,
            selections[name],
            adjacency[np.ix_(places, places)],
            table_points[graph_rows][places],
        )
    return _Store(
        store=store,
        graph=graph,
        graph_cells=graph_cells,
        coordinates=coordinates,
        selections=selections,
        alone=alone,
        alone_coordinates=alone_coordinates,
        arms=table_arms,
        labels=table_labels,
        points=table_points,
        adjacency=adjacency,
        graph_rows=graph_rows,
    )


def _store() -> _Store:
    """Return a private copy of the store, built once."""
    template = _template()
    return replace(template, store=_copy_store(template.store))


def _labelled(trajectory: _Store, **overrides: Any) -> dict[str, Any]:
    return {
        "source_sink": trajectory.label_artifact("graph"),
        "sources": _SOURCES,
        "sinks": _SINKS,
        **overrides,
    }


# Pseudotime


@pytest.mark.parametrize("method", ["untruncated", "truncated"])
def test_scoring_some_cells_equals_scoring_a_graph_of_those_cells(method: str) -> None:
    trajectory = _store()
    store = trajectory.store
    part = trajectory.selections["part"]
    method_arguments = {"method": method}
    if method == "truncated":
        method_arguments["n_singular_vals"] = 12

    some = store.trajectory.pseudotime(
        trajectory.graph, cells=part, **_labelled(trajectory), **method_arguments
    )
    alone = store.trajectory.pseudotime(
        trajectory.alone["part"],
        **_labelled(trajectory, source_sink=trajectory.label_artifact("part")),
        **method_arguments,
    )

    scored, reference = (store.trajectory.load_pseudotime(ref) for ref in (some, alone))
    np.testing.assert_allclose(scored.values, reference.values, rtol=0.0, atol=1e-10)
    np.testing.assert_array_equal(scored.valid, reference.valid)
    np.testing.assert_array_equal(scored.component, reference.component)
    assert scored.component_sizes == reference.component_sizes == (64,)
    # The pseudotime runs from the trunk's start to the ends of A and B.
    rows = trajectory.rows("part")
    ends = np.isin(trajectory.labels[rows], _SINKS)
    assert scored.values[trajectory.labels[rows] == "t1"].max() < 0.5
    assert scored.values[ends].min() > 0.5
    # The record names the graph and the cells that it scores.
    assert scored.graph == trajectory.graph
    assert scored.cell_selection == part
    inputs = store.artifacts.inspect(some).inputs
    assert inputs["connectivity_map"] == trajectory.graph.to_dict()
    assert inputs["cell_selection"] == part.to_dict()
    # A second call reuses the artifact.
    assert (
        store.trajectory.pseudotime(
            trajectory.graph, cells=part, **_labelled(trajectory), **method_arguments
        )
        == some
    )


def test_the_subgraph_of_the_cells_holds_the_graphs_edges_among_them() -> None:
    trajectory = _store()
    store = trajectory.store
    ref = store.trajectory.pseudotime(
        trajectory.graph, cells=trajectory.selections["part"], **_labelled(trajectory)
    )

    subgraph = store._load_pseudotime_scoring_with_graph(ref).graph.toarray()

    places = sorted_places(trajectory.rows("part"), trajectory.graph_rows)
    whole = store.graph.load(trajectory.graph, symmetric=True).toarray()
    np.testing.assert_array_equal(subgraph, whole[np.ix_(places, places)])
    # The fuzzy union of the stored edges a -> b and b -> a, by hand.
    directed = trajectory.adjacency[np.ix_(places, places)]
    np.testing.assert_allclose(
        subgraph, directed + directed.T - directed * directed.T, rtol=0.0, atol=1e-6
    )
    np.testing.assert_array_equal(
        subgraph, store.graph.load(trajectory.alone["part"], symmetric=True).toarray()
    )


def test_cells_that_split_the_graph_are_scored_by_their_own_components() -> None:
    trajectory = _store()
    store = trajectory.store
    split_rows = trajectory.rows("split")
    labels = trajectory.labels[split_rows]
    # Sources at the start of A and sinks at the ends of A and B, which the
    # trunk, cut off from them, does not reach.
    source_sink = np.zeros(len(split_rows))
    source_sink[labels == "a1"] = -1.0 / np.count_nonzero(labels == "a1")
    source_sink[labels == "a2"] = 0.5 / np.count_nonzero(labels == "a2")
    source_sink[labels == "b2"] = 0.5 / np.count_nonzero(labels == "b2")

    some = store.trajectory.load_pseudotime(
        store.trajectory.pseudotime(
            trajectory.graph, cells=trajectory.selections["split"], ss_vec=source_sink
        )
    )
    alone = store.trajectory.load_pseudotime(
        store.trajectory.pseudotime(trajectory.alone["split"], ss_vec=source_sink)
    )

    # The graph is connected among the trunk, A and B, but the trunk's first
    # 20 cells lie 0.5 from A and B, beyond any of their five neighbours.
    assert some.component_sizes == alone.component_sizes == (40, 20)
    np.testing.assert_array_equal(
        some.valid, np.isin(trajectory.arms[split_rows], ["a", "b"])
    )
    np.testing.assert_array_equal(some.valid, alone.valid)
    np.testing.assert_array_equal(some.component, alone.component)
    np.testing.assert_allclose(some.values, alone.values, rtol=0.0, atol=1e-10)
    assert np.isnan(some.values[~some.valid]).all()


def test_the_graphs_own_cells_give_the_pseudotime_without_cells() -> None:
    trajectory = _store()
    store = trajectory.store
    every = store.trajectory.pseudotime(trajectory.graph, **_labelled(trajectory))

    assert (
        store.trajectory.pseudotime(
            trajectory.graph, cells=trajectory.graph_cells, **_labelled(trajectory)
        )
        == every
    )
    assert len(trajectory.artifacts("pseudotime")) == 1
    # A pseudotime of every cell records the graph's selection as before.
    assert (
        store.artifacts.inspect(every).inputs["cell_selection"]
        == trajectory.graph_cells.to_dict()
    )


def test_cells_outside_the_graph_are_refused() -> None:
    trajectory = _store()
    store = trajectory.store
    wide = trajectory.selections["wide"]
    n_wide = len(trajectory.rows("wide"))

    message = (
        f"cells holds 2 of its {n_wide} cells outside the graph's cells; pass a "
        "selection of the graph's cells"
    )
    with pytest.raises(ValueError, match=message):
        store.trajectory.pseudotime(
            trajectory.graph, cells=wide, **_labelled(trajectory)
        )
    with pytest.raises(ValueError, match=message):
        store.trajectory.lineage_tree(
            trajectory.graph, trajectory.label_artifact("graph"), cells=wide
        )
    with pytest.raises(ValueError, match="cells must be a cell_selection artifact"):
        store.trajectory.pseudotime(
            trajectory.graph,
            cells=trajectory.label_artifact("part"),
            **_labelled(trajectory),
        )
    with pytest.raises(TypeError, match="cells must be a cell-selection ArtifactRef"):
        store.trajectory.lineage_tree(
            trajectory.graph, trajectory.label_artifact("graph"), cells="part"
        )
    assert trajectory.artifacts("pseudotime") == []
    assert trajectory.artifacts("lineage_tree") == []


def test_labels_of_the_graph_or_of_the_cells_give_the_same_scores() -> None:
    trajectory = _store()
    store = trajectory.store
    part = trajectory.selections["part"]

    of_graph, of_cells = (
        store.trajectory.load_pseudotime(
            store.trajectory.pseudotime(
                trajectory.graph,
                cells=part,
                **_labelled(trajectory, source_sink=trajectory.label_artifact(name)),
            )
        )
        for name in ("graph", "part")
    )

    assert of_graph.ref != of_cells.ref
    np.testing.assert_array_equal(of_graph.values, of_cells.values)
    # Labels of other cells, though they hold every cell scored, are refused.
    with pytest.raises(
        ValueError, match="Source/sink labels do not match the graph selection or cells"
    ):
        store.trajectory.pseudotime(
            trajectory.graph,
            cells=trajectory.selections["split"],
            **_labelled(trajectory, source_sink=trajectory.label_artifact("part")),
        )


def test_a_vector_of_the_scored_cells_scores_them() -> None:
    trajectory = _store()
    store = trajectory.store
    labels = trajectory.labels[trajectory.rows("part")]
    source_sink = np.where(labels == "t1", -1.0 / np.count_nonzero(labels == "t1"), 0.0)
    for sink in _SINKS:
        source_sink[labels == sink] = 0.5 / np.count_nonzero(labels == sink)

    some = store.trajectory.pseudotime(
        trajectory.graph, cells=trajectory.selections["part"], ss_vec=source_sink
    )
    alone = store.trajectory.pseudotime(trajectory.alone["part"], ss_vec=source_sink)

    np.testing.assert_allclose(
        store.trajectory.load_pseudotime(some).values,
        store.trajectory.load_pseudotime(alone).values,
        rtol=0.0,
        atol=1e-10,
    )
    # The vector's snapshot holds the scored cells.
    snapshot = store.artifacts.inspect(some).input_ref("source_sink")
    assert (
        store.artifacts.inspect(snapshot).inputs["cell_selection"]
        == trajectory.selections["part"].to_dict()
    )
    with pytest.raises(ValueError, match="ss_vec"):
        store.trajectory.pseudotime(
            trajectory.graph,
            cells=trajectory.selections["part"],
            ss_vec=np.zeros(len(trajectory.graph_rows)),
        )


def test_a_record_of_cells_outside_its_graph_is_refused() -> None:
    trajectory = _store()
    store = trajectory.store
    ref = store.trajectory.pseudotime(
        trajectory.graph, cells=trajectory.selections["part"], **_labelled(trajectory)
    )
    _rewrite_inputs(store, ref, cell_selection=trajectory.selections["wide"])

    with pytest.raises(
        ValueError, match="Pseudotime graph and cell selection do not match"
    ):
        store.trajectory.load_pseudotime(ref)


# Fate, curves and the tree


@dataclass(frozen=True)
class _Lineages:
    trajectory: _Store
    some: dict[str, ArtifactRef]
    alone: dict[str, ArtifactRef]


def _lineages() -> _Lineages:
    """Pseudotime, fate and curves of the cells of ``part``, and on their graph alone."""
    trajectory = _store()
    store = trajectory.store
    part = trajectory.selections["part"]
    some_pseudotime = store.trajectory.pseudotime(
        trajectory.graph, cells=part, **_labelled(trajectory)
    )
    alone_pseudotime = store.trajectory.pseudotime(
        trajectory.alone["part"],
        **_labelled(trajectory, source_sink=trajectory.label_artifact("part")),
    )
    some_fate = store.trajectory.fate(
        some_pseudotime, trajectory.label_artifact("graph"), sinks=_SINKS
    )
    alone_fate = store.trajectory.fate(
        alone_pseudotime, trajectory.label_artifact("part"), sinks=_SINKS
    )
    return _Lineages(
        trajectory=trajectory,
        some={
            "pseudotime": some_pseudotime,
            "fate": some_fate,
            "curves": store.trajectory.curves(some_fate, n_points=30, df=6.0),
        },
        alone={
            "pseudotime": alone_pseudotime,
            "fate": alone_fate,
            "curves": store.trajectory.curves(alone_fate, n_points=30, df=6.0),
        },
    )


def test_fate_of_some_cells_equals_fate_on_a_graph_of_those_cells() -> None:
    lineages = _lineages()
    store = lineages.trajectory.store
    part = lineages.trajectory.selections["part"]

    some, alone = (
        store.trajectory.load_fate(group["fate"])
        for group in (lineages.some, lineages.alone)
    )

    np.testing.assert_allclose(some.values, alone.values, rtol=0.0, atol=1e-6)
    np.testing.assert_array_equal(some.valid, alone.valid)
    assert some.cell_selection == part
    assert some.graph == lineages.trajectory.graph
    # Sink labels of the scored cells give the same probabilities.
    of_cells = store.trajectory.load_fate(
        store.trajectory.fate(
            lineages.some["pseudotime"],
            lineages.trajectory.label_artifact("part"),
            sinks=_SINKS,
        )
    )
    np.testing.assert_array_equal(of_cells.values, some.values)
    with pytest.raises(
        ValueError, match="Sink labels do not match the pseudotime cell selection"
    ):
        store.trajectory.fate(
            lineages.some["pseudotime"],
            lineages.trajectory.label_artifact("split"),
            sinks=_SINKS,
        )


def test_curves_of_some_cells_equal_curves_on_a_graph_of_those_cells() -> None:
    lineages = _lineages()
    trajectory = lineages.trajectory
    store = trajectory.store

    some, alone = (
        store.trajectory.load_curves(group["curves"])
        for group in (lineages.some, lineages.alone)
    )

    for name in ("points", "curve_pseudotime", "progress", "distance", "weights"):
        np.testing.assert_allclose(
            getattr(some, name), getattr(alone, name), rtol=0.0, atol=1e-6
        )
    np.testing.assert_array_equal(some.valid, alone.valid)
    # The curves read the graph's coordinates at the cells they follow.
    assert some.coordinates == trajectory.coordinates
    assert some.cell_selection == trajectory.selections["part"]
    # Each curve ends at the last cell of its branch, (4.3, 1.95) for A and
    # (4.3, -1.95) for B, and starts at the trunk's first cell, (0, 0).
    np.testing.assert_allclose(
        some.points[:, -1], [[4.3, 1.95], [4.3, -1.95]], atol=0.1
    )
    np.testing.assert_allclose(some.points[:, 0], [[0.0, 0.0], [0.0, 0.0]], atol=0.1)
    # Coordinates of the scored cells give the same curves.
    np.testing.assert_allclose(
        store.trajectory.load_curves(
            store.trajectory.curves(
                lineages.some["fate"],
                coordinates=trajectory.alone_coordinates["part"],
                n_points=30,
                df=6.0,
            )
        ).points,
        some.points,
        rtol=0.0,
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="cells of the trajectory or of its graph"):
        store.trajectory.curves(
            lineages.some["fate"],
            coordinates=trajectory.alone_coordinates["split"],
        )


def test_a_tree_of_some_cells_equals_a_tree_on_a_graph_of_those_cells() -> None:
    trajectory = _store()
    store = trajectory.store
    part = trajectory.selections["part"]

    some_ref = store.trajectory.lineage_tree(
        trajectory.graph, trajectory.label_artifact("graph"), cells=part, root="t1"
    )
    alone_ref = store.trajectory.lineage_tree(
        trajectory.alone["part"], trajectory.label_artifact("part"), root="t1"
    )
    some, alone = (
        store.trajectory.load_lineage_tree(ref) for ref in (some_ref, alone_ref)
    )

    assert some.clusters == alone.clusters == ("a1", "a2", "b1", "b2", "t1", "t2")
    np.testing.assert_array_equal(some.sizes, alone.sizes)
    np.testing.assert_allclose(some.means, alone.means, rtol=0.0, atol=1e-12)
    pd.testing.assert_frame_equal(some.edges, alone.edges)
    assert some.lineages == alone.lineages
    assert {lineage[-1] for lineage in some.lineages} == {"a2", "b2"}
    assert some.cell_selection == part
    assert some.coordinates == trajectory.coordinates
    # Without cells, the tree joins C and the ring too.
    every = store.trajectory.load_lineage_tree(
        store.trajectory.lineage_tree(
            trajectory.graph, trajectory.label_artifact("graph"), root="t1"
        )
    )
    assert set(every.clusters) == {"a1", "a2", "b1", "b2", "c", "ring", "t1", "t2"}


def test_a_tree_needs_a_label_for_every_cell_it_joins() -> None:
    trajectory = _store()
    store = trajectory.store

    with pytest.raises(ValueError, match="labels must hold the cells of the graph"):
        store.trajectory.lineage_tree(
            trajectory.graph,
            trajectory.label_artifact("part"),
            cells=trajectory.selections["split"],
        )
    assert trajectory.artifacts("lineage_tree") == []


def test_records_of_some_cells_are_checked_when_loaded() -> None:
    lineages = _lineages()
    trajectory = lineages.trajectory
    store = trajectory.store
    part = trajectory.selections["part"]
    tree = store.trajectory.lineage_tree(
        trajectory.graph, trajectory.label_artifact("graph"), cells=part, root="t1"
    )
    # Each record names inputs of the cells of split, which lack four cells
    # of part.
    _rewrite_inputs(store, tree, labels=trajectory.label_artifact("split"))
    with pytest.raises(ValueError, match="Lineage-tree labels do not hold its cells"):
        store.trajectory.load_lineage_tree(tree)
    _rewrite_inputs(
        store,
        tree,
        labels=trajectory.label_artifact("graph"),
        coordinates=trajectory.alone_coordinates["split"],
    )
    with pytest.raises(
        ValueError, match="Lineage-tree coordinates do not hold its cells"
    ):
        store.trajectory.load_lineage_tree(tree)
    _rewrite_inputs(
        store, lineages.some["fate"], sink_labels=trajectory.label_artifact("split")
    )
    with pytest.raises(
        ValueError, match="Fate-map sink labels use a different cell selection"
    ):
        store.trajectory.load_fate(lineages.some["fate"])
    _rewrite_inputs(
        store,
        lineages.some["pseudotime"],
        source_sink=trajectory.label_artifact("split"),
    )
    with pytest.raises(
        ValueError, match="Pseudotime source/sink input uses a different selection"
    ):
        store.trajectory.load_pseudotime(lineages.some["pseudotime"])
    # The rows of a trajectory's cells among coordinates that lack two of them.
    with pytest.raises(
        ValueError, match="2 cells of the trajectory lie outside the cells of its"
    ):
        store._rows_of_selection(trajectory.selections["wide"], trajectory.graph_cells)


# Layouts


def test_a_layout_of_the_graph_draws_curves_of_some_of_its_cells() -> None:
    lineages = _lineages()
    trajectory = lineages.trajectory
    store = trajectory.store

    drawn = store.trajectory.curves_in_layout(
        lineages.some["curves"], trajectory.layout("graph")
    )

    pd.testing.assert_frame_equal(
        drawn,
        store.trajectory.curves_in_layout(
            lineages.alone["curves"], trajectory.layout("part")
        ),
    )
    with pytest.raises(ValueError) as raised:
        store.trajectory.curves_in_layout(
            lineages.some["curves"], trajectory.layout("split")
        )
    assert raised.value.args == (
        f"The layout lacks 4 of the trajectory's {len(trajectory.rows('part'))} "
        "cells; draw an embedding that holds them, such as a UMAP of the "
        "trajectory's graph",
    )


@pytest.mark.parametrize("name", ["lineage_curves", "lineage_tree"])
def test_lineage_plots_draw_some_of_a_layouts_cells(name: str) -> None:
    lineages = _lineages()
    trajectory = lineages.trajectory
    store = trajectory.store
    part = trajectory.selections["part"]
    if name == "lineage_curves":
        arguments: dict[str, Any] = {"curves": lineages.some["curves"]}
    else:
        arguments = {
            "tree": store.trajectory.lineage_tree(
                trajectory.graph,
                trajectory.label_artifact("graph"),
                cells=part,
                root="t1",
            )
        }
    plot: Callable[..., Any] = getattr(splt, name)

    whole = plot(store, layout=trajectory.layout("graph"), show=False, **arguments)
    own = plot(store, layout=trajectory.layout("part"), show=False, **arguments)
    try:
        for table in own.tables:
            pd.testing.assert_frame_equal(whole.tables[table], own.tables[table])
        # Only the trajectory's cells are drawn, where the layout places them.
        (whole_axis,), (own_axis,) = whole.axes.values(), own.axes.values()
        np.testing.assert_array_equal(
            whole_axis.collections[0].get_offsets(),
            own_axis.collections[0].get_offsets(),
        )
        assert whole.provenance.n_cells == len(trajectory.rows("part"))
    finally:
        whole.close()
        own.close()
    plt.close("all")


def test_layout_rows_find_a_trajectorys_cells_in_a_layout_of_more() -> None:
    # Layout cells 0, 2, 3, 7 and 9 hold the trajectory's cells 2, 7 and 9.
    np.testing.assert_array_equal(
        layout_rows(np.array([0, 2, 3, 7, 9]), np.array([2, 7, 9])), [1, 3, 4]
    )
    np.testing.assert_array_equal(
        layout_rows(np.array([4, 5]), np.array([4, 5])), [0, 1]
    )
    with pytest.raises(ValueError, match="The layout lacks 2 of the trajectory's 3"):
        layout_rows(np.array([0, 2, 3]), np.array([1, 2, 8]))
    # Cells before the first cell of the layout, beyond its last, between two
    # of its cells, and among none.
    np.testing.assert_array_equal(
        sorted_places(np.array([0, 4, 11]), np.array([3, 5, 10])), [-1, -1, -1]
    )
    np.testing.assert_array_equal(
        sorted_places(np.array([3], dtype=np.uint32), np.array([], dtype=np.int64)),
        [-1],
    )
    with pytest.raises(TypeError, match="sorted_places needs a one-dimensional"):
        sorted_places(np.array([0.5]), np.array([0, 1]))
    # Against each value's index in a dictionary of the other array's values.
    rng = np.random.default_rng(208)
    for _ in range(20):
        among = np.unique(rng.integers(0, 200, size=60))
        values = np.unique(rng.integers(0, 200, size=40))
        index = {int(value): place for place, value in enumerate(among)}
        np.testing.assert_array_equal(
            sorted_places(values, among),
            [index.get(int(value), -1) for value in values],
        )


# The feature calls on the 1K PBMC store


def _by_hand(datastore, cells: np.ndarray, features: np.ndarray, parameters):
    """Library-size normalized counts of some cells, computed from the raw counts."""
    counts = np.asarray(
        datastore.RNA.rawData._backing.oindex[cells, features], dtype=np.float64
    )
    totals = np.asarray(datastore.cells.fetch_all("RNA_nCounts"), dtype=np.float64)
    values = counts / totals[cells, np.newaxis] * parameters["size_factor"]
    return np.log1p(values) if parameters["normalization"]["log_transform"] else values


def _read_features(datastore, features: ArtifactRef) -> np.ndarray:
    """The feature indices of a feature selection, ascending."""
    return read_feature_selection_indices(
        datastore.zw, "RNA", datastore.features.resolve("RNA", features)
    ).astype(np.int64)


def _pearson(values: np.ndarray, regressor: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0)
    regressor = regressor - regressor.mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        return (centered * regressor[:, None]).sum(axis=0) / np.sqrt(
            (centered**2).sum(axis=0) * (regressor**2).sum()
        )


def test_the_feature_calls_follow_a_pseudotime_of_some_cells(
    datastore, connectivity_graph, leiden_clustering, mark_hvgs, umap
):
    # Leiden clusters 1 to 4 of the session graph, whose subgraph is connected.
    part = datastore.select_cells(leiden_clustering, include=[1, 2, 3, 4])
    rows = cell_selection_indices(datastore.zw, part)
    labels = np.asarray(
        datastore.artifacts.load_values(
            leiden_clustering, cell_selection=part, cell_ids=False
        ).values
    )
    pseudotime = datastore.trajectory.pseudotime(
        connectivity_graph,
        cells=part,
        source_sink=leiden_clustering,
        sources=[1],
        sinks=[2],
    )
    scored = datastore.trajectory.load_pseudotime(pseudotime)
    assert scored.cell_selection == part
    assert scored.component_sizes == (len(rows),)
    features = _read_features(datastore, mark_hvgs)

    markers = datastore.trajectory.markers(pseudotime, features=mark_hvgs, min_cells=1)
    loaded_markers = datastore.trajectory.load_markers(markers)
    assert loaded_markers.cell_selection == part
    tested = loaded_markers.table.dropna(subset=["r_value"])
    parameters = datastore.artifacts.inspect(markers).parameters
    by_hand = _by_hand(datastore, rows, features, parameters)
    expected = _pearson(by_hand, scored.values)
    positions = np.searchsorted(features, tested.feature_index.to_numpy())
    np.testing.assert_allclose(
        tested.r_value.to_numpy(), expected[positions], rtol=0.0, atol=1e-6
    )
    aggregation = datastore.trajectory.aggregation(
        pseudotime, features=mark_hvgs, n_clusters=4, window_size=40, chunk_size=10
    )
    assert datastore.trajectory.load_aggregation(aggregation).cell_selection == part

    fate = datastore.trajectory.fate(pseudotime, leiden_clustering, sinks=[2, 3, 4])
    fate_result = datastore.trajectory.load_fate(fate)
    assert fate_result.cell_selection == part
    assert fate_result.valid.all()
    weights = np.asarray(fate_result.values, dtype=np.float64)
    lineage_markers = datastore.trajectory.load_lineage_markers(
        datastore.trajectory.lineage_markers(fate, features=mark_hvgs)
    )
    assert lineage_markers.cell_selection == part
    np.testing.assert_allclose(
        lineage_markers.effective_cells,
        weights.sum(axis=0) ** 2 / (weights**2).sum(axis=0),
        rtol=1e-10,
    )
    lineage_aggregation = datastore.trajectory.load_lineage_aggregation(
        datastore.trajectory.lineage_aggregation(
            fate, features=mark_hvgs, n_bins=8, n_clusters=4
        )
    )
    assert lineage_aggregation.cell_selection == part
    np.testing.assert_allclose(
        lineage_aggregation.lineage_weight, weights.sum(axis=0), rtol=1e-10
    )

    # Drivers among the cells of cluster 1, which no sink holds.
    source_cells = datastore.select_cells(leiden_clustering, include=[1])
    drivers = datastore.trajectory.load_drivers(
        datastore.trajectory.drivers(fate, features=mark_hvgs, cells=source_cells)
    )
    in_source = labels == 1
    assert drivers.cell_selection == part
    assert drivers.driver_cells == source_cells
    assert drivers.n_cells == int(in_source.sum())
    driving = by_hand[in_source]
    for column, lineage in enumerate([2, 3, 4]):
        table = drivers.table[drivers.table["lineage"] == lineage]
        tested = table.dropna(subset=["r_value"])
        expected = _pearson(driving, weights[in_source, column])
        positions = np.searchsorted(features, tested.feature_index.to_numpy())
        np.testing.assert_allclose(
            tested.r_value.to_numpy(), expected[positions], rtol=0.0, atol=1e-6
        )

    # Curves in the graph's coordinates, drawn in the UMAP of every graph cell.
    curves = datastore.trajectory.curves(fate, n_points=25)
    loaded_curves = datastore.trajectory.load_curves(curves)
    assert loaded_curves.cell_selection == part
    assert loaded_curves.progress.shape == (len(rows), 3)
    drawn = datastore.trajectory.curves_in_layout(curves, umap)
    assert drawn.shape == (75, 4)
    assert np.isfinite(drawn[["x", "y"]].to_numpy()).all()
