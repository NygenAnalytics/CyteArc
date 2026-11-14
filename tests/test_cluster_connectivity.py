"""The graph weight between clusters, as tables and as the figure draws it.

A hand-built graph of three clusters gives weights worked out by hand. On a
store of 60 cells, ``clusters.connectivity`` gives the weights that an
independent count of the graph's edges between clusters gives, the tables that
``plots.cluster_connectivity`` draws, and the same tables in a process that
cannot import matplotlib.
"""

import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cytearc.clustering.connectivity import (
    ClusterConnectivity,
    check_edge_filters,
    cluster_connectivity_tables,
    filter_cluster_edges,
    label_codes,
)
from cytearc.metadata.artifacts import plan_cell_data_artifact
from cytearc.metadata.selection import cell_selection_indices
from cytearc.storage.arrays import MISSING_MASK_PREFIX
from cytearc.storage.artifact_writer import artifact_transaction
from cytearc.storage.artifacts import ArtifactRef
from tests.clone_helpers import write_selection_embedding
from tests.storage_helpers import (
    insert_nullable_cell_column,
    open_store,
    write_count_store,
)


def _graph(n_cells, weighted_edges):
    """A symmetric graph holding each weighted edge in both directions."""
    rows, columns, data = [], [], []
    for first, second, weight in weighted_edges:
        rows.extend((first, second))
        columns.extend((second, first))
        data.extend((weight, weight))
    return sparse.coo_matrix((data, (rows, columns)), shape=(n_cells, n_cells))


# Cells 0-2 are cluster A, 3-4 cluster B and 5-6 cluster C.
LABELS = np.array(["A", "A", "A", "B", "B", "C", "C"], dtype=object)
EDGES = [
    (0, 1, 1.0),
    (3, 4, 2.0),
    (5, 6, 0.5),
    (0, 3, 0.6),
    (2, 4, 0.4),
    (1, 5, 0.3),
    (4, 6, 0.9),
]
# Weighted degrees: A 1.6 + 1.3 + 0.4, B 2.6 + 3.3, C 0.8 + 1.4.
INCIDENT = {"A": 3.3, "B": 5.9, "C": 2.2}
# Edges between clusters: A-B 0.6 + 0.4, A-C 0.3, B-C 0.9.
RAW = {("A", "B"): 1.0, ("A", "C"): 0.3, ("B", "C"): 0.9}


def _normalized(pair):
    first, second = pair
    return RAW[pair] / np.sqrt(INCIDENT[first] * INCIDENT[second])


def _tables(graph, categories=("A", "B", "C")):
    return cluster_connectivity_tables(
        graph, label_codes(LABELS, list(categories)), list(categories)
    )


def test_tables_sum_each_undirected_edge_once_by_hand():
    # The edge between cells 0 and 3 is split into two entries each way, and
    # cell 2 has a self-loop: the entries are summed and the loop left out.
    split = [edge for edge in EDGES if edge[:2] != (0, 3)]
    split += [(0, 3, 0.25), (0, 3, 0.35), (2, 2, 2.5)]
    graph = _graph(7, split)
    assert graph.nnz == 2 * len(split)

    tables = _tables(graph)

    assert isinstance(tables, ClusterConnectivity)
    nodes = tables.nodes
    assert nodes.columns.tolist() == ["category", "nCells", "incidentWeight"]
    assert nodes["category"].tolist() == ["A", "B", "C"]
    assert nodes["nCells"].tolist() == [3, 2, 2]
    np.testing.assert_allclose(nodes["incidentWeight"], [3.3, 5.9, 2.2])
    edges = tables.edges
    assert edges.columns.tolist() == [
        "source",
        "target",
        "rawWeight",
        "normalizedWeight",
    ]
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    assert list(zip(edges["source"], edges["target"], strict=True)) == pairs
    np.testing.assert_allclose(edges["rawWeight"], [RAW[pair] for pair in pairs])
    np.testing.assert_allclose(
        edges["normalizedWeight"], [_normalized(pair) for pair in pairs]
    )


def test_tables_follow_the_order_of_their_categories():
    tables = _tables(_graph(7, EDGES), categories=("C", "A", "B"))

    assert tables.nodes["category"].tolist() == ["C", "A", "B"]
    assert tables.nodes["nCells"].tolist() == [2, 3, 2]
    # The source of each pair comes first among the categories.
    pairs = [("C", "A"), ("C", "B"), ("A", "B")]
    assert list(zip(tables.edges["source"], tables.edges["target"])) == pairs
    np.testing.assert_allclose(
        tables.edges["rawWeight"],
        [RAW[tuple(sorted(pair))] for pair in pairs],
    )


def test_a_cluster_without_edges_has_no_weight_and_no_pair():
    # Cell 7, cluster D, has no edge.
    labels = np.append(LABELS, "D")
    tables = cluster_connectivity_tables(
        _graph(8, EDGES), label_codes(labels, ["A", "B", "C", "D"]), "ABCD"
    )

    assert tables.nodes["incidentWeight"].iloc[-1] == 0.0
    assert tables.nodes["nCells"].iloc[-1] == 1
    assert "D" not in set(tables.edges["source"]) | set(tables.edges["target"])
    assert len(tables.edges) == 3


def test_a_graph_without_edges_between_clusters_gives_an_empty_edge_table():
    tables = _tables(_graph(7, EDGES[:3]))

    assert tables.edges.empty
    assert tables.edges.columns.tolist() == [
        "source",
        "target",
        "rawWeight",
        "normalizedWeight",
    ]
    np.testing.assert_allclose(tables.nodes["incidentWeight"], [2.0, 4.0, 1.0])


def test_filters_keep_a_floor_and_the_strongest_pairs_of_each_cluster():
    tables = _tables(_graph(7, EDGES))
    # Normalized weights by hand: B-C 0.250, A-B 0.227, A-C 0.111. A pair at
    # the floor, A-B, is kept.
    floor = tables.edges["normalizedWeight"].iloc[0]
    assert floor == pytest.approx(_normalized(("A", "B")))

    kept = filter_cluster_edges(
        tables, minimum_edge_weight=floor, max_edges_per_node=None
    )
    assert kept.edges[["source", "target"]].values.tolist() == [
        ["A", "B"],
        ["B", "C"],
    ]
    assert kept.nodes is tables.nodes
    # B-C takes B and C's one pair; A-B and A-C then meet a full cluster.
    capped = filter_cluster_edges(tables, minimum_edge_weight=0.0, max_edges_per_node=1)
    assert capped.edges[["source", "target"]].values.tolist() == [["B", "C"]]
    every = filter_cluster_edges(tables, minimum_edge_weight=0.0, max_edges_per_node=2)
    pd.testing.assert_frame_equal(every.edges, tables.edges)
    none = filter_cluster_edges(tables, minimum_edge_weight=0.0, max_edges_per_node=0)
    assert none.edges.empty
    assert none.edges.columns.tolist() == tables.edges.columns.tolist()


@pytest.mark.parametrize(
    ("categories", "kept"),
    [(["A", "B", "C"], ["A", "B"]), (["A", "C", "B"], ["A", "C"])],
)
def test_capped_ties_fall_to_the_order_of_the_nodes(categories, kept):
    # One cell in each cluster: A-B and A-C have equal raw and normalized
    # weights, so the cap keeps the pair that comes first among the nodes.
    labels = np.array(["A", "B", "C"], dtype=object)
    tables = cluster_connectivity_tables(
        _graph(3, [(0, 1, 1.0), (0, 2, 1.0)]),
        label_codes(labels, categories),
        categories,
    )

    capped = filter_cluster_edges(tables, minimum_edge_weight=0.0, max_edges_per_node=1)

    assert capped.edges[["source", "target"]].values.tolist() == [kept]


@pytest.mark.parametrize(
    ("minimum_edge_weight", "max_edges_per_node", "message"),
    [
        (-0.1, None, "minimum_edge_weight must be finite and non-negative"),
        (np.nan, None, "minimum_edge_weight must be finite and non-negative"),
        (np.inf, None, "minimum_edge_weight must be finite and non-negative"),
        (0.0, -1, "max_edges_per_node must be a non-negative integer or None"),
        (0.0, 1.5, "max_edges_per_node must be a non-negative integer or None"),
        (0.0, True, "max_edges_per_node must be a non-negative integer or None"),
    ],
)
def test_invalid_filters_are_refused(minimum_edge_weight, max_edges_per_node, message):
    with pytest.raises(ValueError) as raised:
        check_edge_filters(minimum_edge_weight, max_edges_per_node)
    assert raised.value.args == (message,)
    with pytest.raises(ValueError) as raised:
        filter_cluster_edges(
            _tables(_graph(7, EDGES)),
            minimum_edge_weight=minimum_edge_weight,
            max_edges_per_node=max_edges_per_node,
        )
    assert raised.value.args == (message,)


@pytest.mark.parametrize(
    ("graph", "codes", "categories", "error", "message"),
    [
        (
            _graph(7, EDGES).toarray(),
            label_codes(LABELS, "ABC"),
            "ABC",
            TypeError,
            "graph must be a scipy sparse matrix",
        ),
        (
            _graph(8, EDGES),
            label_codes(LABELS, "ABC"),
            "ABC",
            ValueError,
            "Graph shape must match the selected cells; expected (7, 7), got (8, 8)",
        ),
        (
            _graph(7, [*EDGES, (1, 2, np.inf)]),
            label_codes(LABELS, "ABC"),
            "ABC",
            ValueError,
            "Graph contains non-finite edge weights",
        ),
        (
            _graph(7, [*EDGES, (1, 2, -0.5)]),
            label_codes(LABELS, "ABC"),
            "ABC",
            ValueError,
            "Graph edge weights must be non-negative",
        ),
        (
            _graph(7, EDGES),
            label_codes(LABELS, "ABC").astype(float),
            "ABC",
            TypeError,
            "codes must be integers",
        ),
        (
            _graph(7, EDGES),
            label_codes(LABELS, "ABC")[:, None],
            "ABC",
            ValueError,
            "codes must be one-dimensional",
        ),
        (
            _graph(7, EDGES),
            label_codes(LABELS, "ABC"),
            "AB",
            ValueError,
            "codes must be positions among the categories",
        ),
        (
            _graph(7, EDGES),
            label_codes(LABELS, "ABC") - 1,
            "ABC",
            ValueError,
            "codes must be positions among the categories",
        ),
        (
            _graph(7, EDGES),
            label_codes(LABELS, "ABC"),
            "ABCA",
            ValueError,
            "categories must name each cluster once",
        ),
    ],
)
def test_malformed_graphs_and_codes_are_refused(
    graph, codes, categories, error, message
):
    with pytest.raises(error) as raised:
        cluster_connectivity_tables(graph, codes, list(categories))
    assert raised.value.args == (message,)


def test_label_codes_refuse_labels_outside_the_categories():
    labels = np.array(["B", "A", "B"], dtype=object)

    np.testing.assert_array_equal(label_codes(labels, ["A", "B"]), [1, 0, 1])
    with pytest.raises(ValueError) as raised:
        label_codes(labels, ["A"])
    assert raised.value.args == ("Could not map every group value to a category",)
    assert isinstance(raised.value.__cause__, KeyError)
    unhashable = np.empty(1, dtype=object)
    unhashable[0] = ["A"]
    with pytest.raises(ValueError) as raised:
        label_codes(unhashable, ["A"])
    assert isinstance(raised.value.__cause__, TypeError)


# A store of 60 cells. Every tenth cell is left out of the graph, which holds
# the 54 kept cells; each block of 20 cells expresses its own four genes.
N_CELLS = 60
LEFT_OUT = np.arange(N_CELLS) % 10 == 0
KEPT = np.flatnonzero(~LEFT_OUT)
BLOCKS = np.repeat(np.array(["block 1", "block 2", "block 3"], dtype=object), 20)


@pytest.fixture(scope="module")
def clustered(tmp_path_factory):
    """A graph of the kept cells, its Leiden clusters, block labels, and a layout."""
    rng = np.random.default_rng(11)
    rates = np.ones((3, 12))
    for block in range(3):
        rates[block, 4 * block : 4 * block + 4] = 4.0
    counts = rng.poisson(rates[np.repeat(np.arange(3), 20)])
    path = tmp_path_factory.mktemp("cluster_connectivity") / "store.zarr"
    write_count_store(str(path), {"RNA": counts}, "uint16")
    store = open_store(path)
    store.cells.insert("block", BLOCKS)
    store.cells.insert("kept", ~LEFT_OUT)
    store.cells.insert("half", np.arange(N_CELLS) < N_CELLS // 2)
    store.cells.insert("every_cell", np.ones(N_CELLS, dtype=bool))
    # Cell 1, a kept cell, has no block label in this column.
    insert_nullable_cell_column(
        store, "block_with_gap", BLOCKS.astype(str), np.arange(N_CELLS) == 1
    )
    kept = store.snapshot_cell_selection("kept")
    with warnings.catch_warnings():
        # The few cells of the store draw warnings from the neighbor search.
        warnings.simplefilter("ignore")
        normalized = store.features.normalize(
            kept, store.features.universe(from_assay="RNA")
        )
        reduction = store.reduction.pca(normalized, dims=3)
        neighbors = store.graph.neighbors(store.graph.ann_index(reduction), k=5)
        graph = store.graph.connectivity(neighbors)
        leiden = store.clusters.leiden(graph, resolution=1.0)
    positions = cell_selection_indices(store.zw, kept)
    layout = write_selection_embedding(
        store, kept, np.column_stack([positions, positions % 7])
    )
    return SimpleNamespace(
        store=store,
        graph=graph,
        leiden=leiden,
        layout=layout,
        kept=kept,
        every=store.snapshot_cell_selection("every_cell"),
        half=store.snapshot_cell_selection("half"),
    )


def _expected(store, graph, labels):
    """Each cluster's cells and incident weight, and each pair's weights, counted apart."""
    matrix = sparse.csr_matrix(store.graph.load(graph, symmetric=True), dtype=float)
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()
    categories = sorted(set(labels.tolist()))
    codes = np.searchsorted(categories, labels)
    membership = sparse.csr_matrix(
        (np.ones(len(labels)), (np.arange(len(labels)), codes)),
        shape=(len(labels), len(categories)),
    )
    # For a symmetric graph, entry (a, b) of M' G M sums each edge between
    # clusters a and b once; each row of G sums a cell's weighted degree.
    between = (membership.T @ matrix @ membership).toarray()
    incident = membership.T @ np.asarray(matrix.sum(axis=1)).ravel()
    first, second = np.nonzero(np.triu(between, k=1))
    nodes = pd.DataFrame(
        {
            "category": categories,
            "nCells": np.bincount(codes, minlength=len(categories)),
            "incidentWeight": incident,
        }
    )
    edges = pd.DataFrame(
        {
            "source": np.asarray(categories, dtype=object)[first],
            "target": np.asarray(categories, dtype=object)[second],
            "rawWeight": between[first, second],
            "normalizedWeight": between[first, second]
            / np.sqrt(incident[first] * incident[second]),
        }
    )
    return nodes, edges


def _assert_tables(actual, nodes, edges):
    assert actual.nodes["category"].tolist() == nodes["category"].tolist()
    np.testing.assert_array_equal(actual.nodes["nCells"], nodes["nCells"])
    np.testing.assert_allclose(actual.nodes["incidentWeight"], nodes["incidentWeight"])
    assert actual.edges[["source", "target"]].values.tolist() == (
        edges[["source", "target"]].values.tolist()
    )
    np.testing.assert_allclose(actual.edges["rawWeight"], edges["rawWeight"])
    np.testing.assert_allclose(
        actual.edges["normalizedWeight"], edges["normalizedWeight"]
    )


def test_connectivity_counts_the_graphs_edges_between_leiden_clusters(clustered):
    store = clustered.store
    labels = store.artifacts.load_values(clustered.leiden).values

    tables = store.clusters.connectivity(clustered.graph, clustered.leiden)

    nodes, edges = _expected(store, clustered.graph, labels)
    assert len(nodes) > 1 and len(edges) > 0
    _assert_tables(tables, nodes, edges)
    assert tables.nodes["nCells"].sum() == len(KEPT)


@pytest.mark.parametrize(
    "groups", ["block", "block over the kept cells", "block over every cell"]
)
def test_connectivity_reads_a_column_or_labels_of_more_cells_for_the_graphs_cells(
    clustered, groups
):
    store = clustered.store
    selections = {
        "block over the kept cells": clustered.kept,
        "block over every cell": clustered.every,
    }
    source = (
        groups
        if groups == "block"
        else store.snapshot_cluster_labels("block", cell_selection=selections[groups])
    )

    tables = store.clusters.connectivity(clustered.graph, source)

    nodes, edges = _expected(store, clustered.graph, BLOCKS[KEPT])
    assert nodes["category"].tolist() == ["block 1", "block 2", "block 3"]
    assert len(edges) > 0
    _assert_tables(tables, nodes, edges)


@pytest.mark.parametrize(
    "filters",
    [
        {"minimum_edge_weight": 0.0},
        {"minimum_edge_weight": 0.05, "max_edges_per_node": 1},
    ],
)
def test_connectivity_equals_the_tables_that_the_plot_draws(clustered, filters):
    store = clustered.store

    tables = store.clusters.connectivity(clustered.graph, clustered.leiden, **filters)
    drawn = store.plots.cluster_connectivity(
        graph=clustered.graph,
        groups=clustered.leiden,
        layout=clustered.layout,
        show=False,
        **filters,
    )

    try:
        pd.testing.assert_frame_equal(
            drawn.tables["nodes"][["category", "nCells"]],
            tables.nodes[["category", "nCells"]],
        )
        pd.testing.assert_frame_equal(drawn.tables["edges"], tables.edges)
        assert drawn.provenance.extras["n_aggregated_edges"] == len(
            store.clusters.connectivity(clustered.graph, clustered.leiden).edges
        )
    finally:
        drawn.close()


def test_connectivity_filters_as_the_domain_filter_does(clustered):
    store = clustered.store
    every = store.clusters.connectivity(clustered.graph, clustered.leiden)

    capped = store.clusters.connectivity(
        clustered.graph,
        clustered.leiden,
        minimum_edge_weight=0.01,
        max_edges_per_node=2,
    )

    expected = filter_cluster_edges(
        every, minimum_edge_weight=0.01, max_edges_per_node=2
    )
    pd.testing.assert_frame_equal(capped.edges, expected.edges)
    pd.testing.assert_frame_equal(capped.nodes, every.nodes)
    assert (
        pd.concat([capped.edges["source"], capped.edges["target"]]).value_counts().max()
        <= 2
    )


def test_connectivity_refuses_labels_that_miss_some_of_the_graphs_cells(clustered):
    store = clustered.store
    half = store.snapshot_cluster_labels("block", cell_selection=clustered.half)

    with pytest.raises(ValueError) as raised:
        store.clusters.connectivity(clustered.graph, half)

    assert raised.value.args == (
        "The cluster_labels artifact of groups holds no label for some cells of "
        "the graph; compute it on cells that include the graph's, such as the "
        "graph's cell selection",
    )


def _masked_labels(store, selection, values, missing):
    """Save cluster labels of a selection's cells with a linked mask of the missing ones."""
    planned = plan_cell_data_artifact(
        store.zw,
        scope="datastore",
        kind="cluster_labels",
        operation="test_cluster_connectivity",
        parameters={"case": "masked"},
        inputs={},
        execution_options={},
        cell_selection=selection,
        arrays={"values": (values.shape, None)},
    )
    mask_name = f"{MISSING_MASK_PREFIX}values"
    with artifact_transaction(store.zw, planned) as group:
        group.create_array("values", data=values)
        group.create_array(mask_name, data=missing)
        group["values"].attrs["missing_mask"] = mask_name
    return planned.ref


@pytest.mark.parametrize("artifact", [False, True])
def test_connectivity_refuses_missing_labels(clustered, artifact):
    store = clustered.store
    # Kept cell 1 has no label; the artifact stores a placeholder for it.
    groups = (
        _masked_labels(
            store,
            clustered.kept,
            BLOCKS[KEPT].astype(str),
            KEPT == 1,
        )
        if artifact
        else "block_with_gap"
    )

    with pytest.raises(ValueError) as raised:
        store.clusters.connectivity(clustered.graph, groups)

    assert raised.value.args == ("groups contains missing labels",)


@pytest.mark.parametrize(
    ("graph", "groups", "options", "error", "message"),
    [
        (
            "graph",
            "block",
            {"minimum_edge_weight": -1.0},
            ValueError,
            "minimum_edge_weight must be finite and non-negative",
        ),
        (
            "graph",
            "block",
            {"max_edges_per_node": -2},
            ValueError,
            "max_edges_per_node must be a non-negative integer or None",
        ),
        ("dict", "block", {}, TypeError, "graph must be an ArtifactRef"),
        (
            "graph",
            3,
            {},
            TypeError,
            "groups must be an ArtifactRef or a cell column name",
        ),
        (
            "graph",
            ArtifactRef(
                scope="assay", assay="ADT", kind="cluster_labels", artifact_id="5" * 64
            ),
            {},
            ValueError,
            "groups and graph must belong to the same assay",
        ),
    ],
)
def test_connectivity_refuses_invalid_arguments_before_reading(
    clustered, graph, groups, options, error, message
):
    graph_argument = clustered.graph if graph == "graph" else clustered.graph.to_dict()

    with pytest.raises(error) as raised:
        clustered.store.clusters.connectivity(graph_argument, groups, **options)

    assert raised.value.args == (message,)


def test_connectivity_refuses_groups_that_hold_no_labels(clustered):
    with pytest.raises(ValueError) as raised:
        clustered.store.clusters.connectivity(clustered.graph, clustered.layout)

    assert raised.value.args == (
        "Grouping artifacts must contain categorical cell labels",
    )


@pytest.fixture
def without_matplotlib(monkeypatch):
    # A None entry in sys.modules makes `import matplotlib` raise ImportError.
    for name in [name for name in sys.modules if name.split(".")[0] == "matplotlib"]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "matplotlib", None)


def test_connectivity_needs_no_matplotlib(clustered, without_matplotlib):
    store = clustered.store
    labels = store.artifacts.load_values(clustered.leiden).values

    tables = store.clusters.connectivity(clustered.graph, clustered.leiden)

    _assert_tables(tables, *_expected(store, clustered.graph, labels))
    with pytest.raises(ImportError):
        store.plots.cluster_connectivity(
            graph=clustered.graph,
            groups=clustered.leiden,
            layout=clustered.layout,
            show=False,
        )
