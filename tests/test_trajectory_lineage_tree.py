import numpy as np
import pytest

from cytearc.trajectory import build_cluster_tree
from cytearc.trajectory import lineage_tree
from cytearc.trajectory.lineage_tree import (
    SpanningTree,
    cluster_distances,
    cluster_moments,
    labeled_tree,
    spanning_tree,
    tree_lineages,
)

# Cluster centers of a Y: a stem 0-1-2 that forks into 3-4 and 5-6.
_Y_CENTERS = np.array(
    [
        [0.0, 0.0],
        [2.0, 0.0],
        [4.0, 0.0],
        [6.0, 2.0],
        [8.0, 4.0],
        [6.0, -2.0],
        [8.0, -4.0],
    ]
)
_Y_EDGES = [[0, 1], [1, 2], [2, 3], [2, 5], [3, 4], [5, 6]]


def _clustered_cells(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Forty-one cells in four 3D clusters, the last a single cell, shuffled."""
    rng = np.random.default_rng(seed)
    codes = np.repeat([0, 1, 2, 3], [12, 15, 13, 1])
    centers = np.array(
        [[0.0, 0.0, 0.0], [5.0, 1.0, -2.0], [1.0, 6.0, 3.0], [9.0, 9.0, 9.0]]
    )
    spread = np.array(
        [[1.0, 0.5, 2.0], [0.3, 1.5, 0.7], [2.0, 2.0, 0.2], [1.0, 1.0, 1.0]]
    )
    data = centers[codes] + rng.normal(size=(codes.size, 3)) * spread[codes]
    order = rng.permutation(codes.size)
    return codes[order], data[order]


def _blocks(data: np.ndarray, cuts: tuple[int, ...]) -> list[tuple[int, np.ndarray]]:
    return [
        (start, data[start:stop])
        for start, stop in zip(cuts[:-1], cuts[1:], strict=True)
    ]


def _prim_edges(distances: np.ndarray) -> list[list[int]]:
    """Grow a minimum spanning tree from cluster 0, one cheapest edge at a time."""
    n_clusters = distances.shape[0]
    joined = {0}
    edges = []
    while len(joined) < n_clusters:
        _, first, second = min(
            (distances[a, b], a, b)
            for a in joined
            for b in range(n_clusters)
            if b not in joined
        )
        joined.add(second)
        edges.append(sorted((first, second)))
    return sorted(edges)


def _y_tree_distances(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    codes = np.repeat(np.arange(7), 30)
    data = _Y_CENTERS[codes] + rng.normal(scale=0.3, size=(codes.size, 2))
    _, means, covariances = cluster_moments(
        _blocks(data, (0, 50, 210)), codes, n_clusters=7, n_dims=2
    )
    return cluster_distances(means, covariances)


def _line_distances(positions: list[float]) -> np.ndarray:
    values = np.asarray(positions)
    return np.square(values[:, np.newaxis] - values[np.newaxis, :])


@pytest.mark.parametrize(
    "cuts",
    [
        (0, 41),
        (0, 1, 7, 7, 20, 41),
        tuple(range(42)),
    ],
)
def test_cluster_moments_match_numpy_over_any_block_partition(cuts):
    codes, data = _clustered_cells(0)

    counts, means, covariances = cluster_moments(
        _blocks(data, cuts), codes, n_clusters=4, n_dims=3
    )

    np.testing.assert_array_equal(counts, [12, 15, 13, 1])
    assert counts.dtype == np.int64
    for cluster in range(3):
        rows = data[codes == cluster]
        np.testing.assert_allclose(
            means[cluster], rows.mean(axis=0), rtol=1e-13, atol=1e-14
        )
        np.testing.assert_allclose(
            covariances[cluster], np.cov(rows, rowvar=False), rtol=1e-12, atol=1e-14
        )
    np.testing.assert_array_equal(means[3], data[codes == 3][0])
    np.testing.assert_array_equal(covariances[3], 0.0)


def test_cluster_moments_stay_accurate_far_from_the_origin():
    codes, data = _clustered_cells(1)
    shifted = data + 1.0e8

    _, means, covariances = cluster_moments(
        _blocks(shifted, tuple(range(42))), codes, n_clusters=4, n_dims=3
    )

    for cluster in range(3):
        rows = shifted[codes == cluster]
        np.testing.assert_allclose(means[cluster], rows.mean(axis=0), rtol=1e-14)
        # A one-pass sum of squares would lose every digit at this offset.
        np.testing.assert_allclose(
            covariances[cluster],
            np.cov(data[codes == cluster], rowvar=False),
            atol=1e-6,
        )


def _moment_inputs(name: str) -> tuple[list, np.ndarray, int]:
    data = np.arange(12.0).reshape(6, 2)
    codes = np.array([0, 1, 1, 2, 0, 2])
    blocks = [(0, data)]
    n_clusters = 3
    if name == "matrix codes":
        codes = codes.reshape(2, 3)
    elif name == "negative code":
        codes = np.array([0, 1, -1, 2, 0, 2])
    elif name == "large code":
        codes = np.array([0, 1, 3, 2, 0, 2])
    elif name == "empty cluster":
        codes = np.array([0, 0, 0, 2, 0, 2])
    elif name == "gap":
        blocks = [(0, data[:2]), (3, data[3:])]
    elif name == "dimensions":
        blocks = [(0, data[:, :1])]
    elif name == "flat":
        blocks = [(0, data[:, 0])]
    elif name == "beyond":
        blocks = [(0, data), (6, data[:1])]
    elif name == "non-finite":
        broken = data.copy()
        broken[4, 0] = np.inf
        blocks = [(0, broken)]
    elif name == "short":
        blocks = [(0, data[:5])]
    return blocks, codes, n_clusters


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("matrix codes", "codes must give a cluster index for every cell"),
        ("negative code", "codes must give a cluster index for every cell"),
        ("large code", "codes must give a cluster index for every cell"),
        ("empty cluster", "Every cluster needs at least one cell"),
        ("gap", "Coordinate blocks must cover the cells in order"),
        ("dimensions", "Coordinate blocks must cover the cells in order"),
        ("flat", "Coordinate blocks must cover the cells in order"),
        ("beyond", "Coordinate blocks extend beyond the cells"),
        ("non-finite", "Coordinates must be finite"),
        ("short", "Coordinate blocks must cover every cell"),
    ],
)
def test_cluster_moments_reject_invalid_codes_and_blocks(name, message):
    blocks, codes, n_clusters = _moment_inputs(name)

    with pytest.raises(ValueError, match=message):
        cluster_moments(blocks, codes, n_clusters=n_clusters, n_dims=2)


def test_cluster_distances_match_the_slingshot_formula():
    rng = np.random.default_rng(2)
    means = rng.normal(size=(5, 3)) * 4.0
    factors = rng.normal(size=(5, 3, 3))
    covariances = factors @ np.transpose(factors, (0, 2, 1)) + 0.1 * np.eye(3)

    distances = cluster_distances(means, covariances)

    for first in range(5):
        for second in range(5):
            difference = means[first] - means[second]
            pooled = covariances[first] + covariances[second]
            expected = difference @ np.linalg.inv(pooled) @ difference
            assert distances[first, second] == pytest.approx(expected, rel=1e-10, abs=0)
    np.testing.assert_array_equal(distances, distances.T)
    np.testing.assert_array_equal(np.diag(distances), 0.0)


def test_cluster_distances_use_the_diagonal_when_pooled_covariance_is_singular():
    means = np.array([[0.0, 5.0], [3.0, 5.0], [1.0, 2.0], [4.0, 4.0]])
    covariances = np.array(
        [
            np.diag([1.0, 0.0]),
            np.diag([2.0, 0.0]),
            # Rank one along (2, 1); doubled, its Cholesky factor hits an
            # exact zero pivot.
            [[0.5, 0.25], [0.25, 0.125]],
            [[0.5, 0.25], [0.25, 0.125]],
        ]
    )

    distances = cluster_distances(means, covariances)

    # The second dimension has no variance in either cluster, and the means
    # agree there, so it adds nothing.
    assert distances[0, 1] == pytest.approx(9.0 / 3.0)
    assert distances[2, 3] == pytest.approx(9.0 / 1.0 + 4.0 / 0.25)


def test_cluster_distances_are_infinite_where_means_differ_without_variance():
    means = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 3.0, 0.0],
            [3.0, 3.0, 7.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ]
    )
    covariances = np.array(
        [
            [[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            np.zeros((3, 3)),
        ]
    )

    distances = cluster_distances(means, covariances)

    # Only the diagonal of the singular pooled matrix is used, so the
    # covariance between the first two dimensions is ignored too, and the
    # third dimension adds nothing where the means agree.
    assert distances[0, 1] == pytest.approx(9.0 / 3.0 + 9.0 / 3.0)
    # Means that differ in a dimension without variance are infinitely far
    # apart, as the t-statistic is.
    assert distances[0, 2] == np.inf
    assert distances[2, 0] == np.inf
    # Two single cells at one position are identical; at two positions they
    # are infinitely far apart, not identical.
    assert distances[3, 4] == 0.0
    assert distances[3, 5] == np.inf


def test_spanning_tree_joins_single_cells_through_clusters_that_vary():
    rng = np.random.default_rng(0)
    coordinates = np.vstack(
        [
            rng.normal(0.0, 1.0, size=(50, 3)),
            rng.normal(0.0, 1.0, size=(50, 3)) + [6.0, 0.0, 0.0],
            [[0.5, 0.0, 0.0]],
            [[100.0, 100.0, 100.0]],
        ]
    )
    codes = np.repeat([0, 1, 2, 3], [50, 50, 1, 1])
    _, means, covariances = cluster_moments(
        [(0, coordinates)], codes, n_clusters=4, n_dims=3
    )

    distances = cluster_distances(means, covariances)
    tree = spanning_tree(distances)

    assert distances[2, 3] == np.inf
    # The far single cell joins the cluster it is nearest to, measured by
    # that cluster's spread, never the other single cell.
    nearest = int(np.argmin(distances[3, :2]))
    assert [min(3, nearest), max(3, nearest)] in tree.edges.tolist()
    assert [2, 3] not in tree.edges.tolist()
    assert np.isfinite(tree.lengths).all()


@pytest.mark.parametrize(
    ("ends", "labels", "pair"),
    [
        ((), None, "0 and 2"),
        ((), ["A", "B", "C"], "'A' and 'C'"),
        ((2,), ["A", "B", "C"], "'A' and 'C'"),
    ],
)
def test_spanning_tree_rejects_clusters_reachable_only_across_infinity(
    ends, labels, pair
):
    distances = np.array(
        [[0.0, 1.0, np.inf], [1.0, 0.0, np.inf], [np.inf, np.inf, 0.0]]
    )

    with pytest.raises(
        ValueError, match=f"join clusters {pair} only across an infinite"
    ):
        spanning_tree(distances, ends=ends, labels=labels)


def test_cluster_distances_use_the_diagonal_for_singular_covariances_that_factor():
    # Cluster 0 is two cells along (1, 1), cluster 1 a single cell, so the
    # pooled covariance is [[2, 2], [2, 2]], which has rank one.
    coordinates = np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 4.0]])
    _, means, covariances = cluster_moments(
        [(0, coordinates)], np.array([0, 0, 1]), n_clusters=2, n_dims=2
    )
    np.testing.assert_array_equal(covariances[0], [[2.0, 2.0], [2.0, 2.0]])

    distances = cluster_distances(means, covariances)

    assert distances[0, 1] == pytest.approx(4.0 / 2.0 + 9.0 / 2.0)


def test_spanning_tree_joins_a_y_of_clusters_and_traces_its_branches():
    distances = _y_tree_distances()

    tree = spanning_tree(distances, root=0)
    unrooted = spanning_tree(distances)

    assert isinstance(tree, SpanningTree)
    assert tree.edges.tolist() == _Y_EDGES
    assert tree.edges.dtype == np.int64
    np.testing.assert_array_equal(
        tree.lengths, distances[tree.edges[:, 0], tree.edges[:, 1]]
    )
    np.testing.assert_array_equal(tree.leaves, [0, 4, 6])
    assert tree.lineages == ((0, 1, 2, 3, 4), (0, 1, 2, 5, 6))
    assert unrooted.edges.tolist() == _Y_EDGES
    np.testing.assert_array_equal(unrooted.leaves, [0, 4, 6])
    assert unrooted.lineages == ()


def test_spanning_tree_orders_root_lineages_by_leaf():
    distances = _y_tree_distances()

    from_fork = spanning_tree(distances, root=2)
    from_tip = spanning_tree(distances, root=4)

    assert from_fork.lineages == ((2, 1, 0), (2, 3, 4), (2, 5, 6))
    # A root that is itself a leaf starts every path but ends none.
    assert from_tip.lineages == ((4, 3, 2, 1, 0), (4, 3, 2, 5, 6))
    np.testing.assert_array_equal(from_tip.leaves, [0, 4, 6])


@pytest.mark.parametrize("seed", range(4))
def test_spanning_tree_matches_a_brute_force_minimum_spanning_tree(seed):
    rng = np.random.default_rng(10 + seed)
    n_clusters = int(rng.integers(3, 9))
    upper = np.triu(rng.uniform(0.5, 10.0, size=(n_clusters, n_clusters)), 1)
    distances = upper + upper.T

    tree = spanning_tree(distances)

    assert tree.edges.tolist() == _prim_edges(distances)
    expected_leaves = np.flatnonzero(np.bincount(tree.edges.ravel()) == 1)
    np.testing.assert_array_equal(tree.leaves, expected_leaves)


def test_spanning_tree_makes_ends_leaves_that_break_a_chain():
    # On a line the tree is the chain 0-1-2-3, but cluster 1 is declared an
    # end, so it hangs off its nearest other cluster.
    distances = _line_distances([0.0, 0.9, 2.0, 3.0])

    chain = spanning_tree(distances, root=0)
    supervised = spanning_tree(distances, root=0, ends=[1])

    assert chain.edges.tolist() == [[0, 1], [1, 2], [2, 3]]
    assert chain.lineages == ((0, 1, 2, 3),)
    assert supervised.edges.tolist() == [[0, 1], [0, 2], [2, 3]]
    np.testing.assert_allclose(supervised.lengths, [0.81, 4.0, 1.0])
    np.testing.assert_array_equal(supervised.leaves, [1, 3])
    assert supervised.lineages == ((0, 1), (0, 2, 3))


def test_spanning_tree_joins_each_end_to_an_inner_cluster_never_to_another_end():
    distances = _line_distances([0.0, 1.0, 5.0, 5.1])

    tree = spanning_tree(distances, ends=(2, 3))

    assert tree.edges.tolist() == [[0, 1], [1, 2], [1, 3]]
    np.testing.assert_array_equal(tree.leaves, [0, 2, 3])


def test_spanning_tree_with_one_inner_cluster_attaches_every_end_to_it():
    distances = _line_distances([0.0, 1.0, 3.0])

    tree = spanning_tree(distances, root=0, ends=[2, 1])

    assert tree.edges.tolist() == [[0, 1], [0, 2]]
    np.testing.assert_array_equal(tree.lengths, [1.0, 9.0])
    np.testing.assert_array_equal(tree.leaves, [1, 2])
    assert tree.lineages == ((0, 1), (0, 2))


@pytest.mark.parametrize(
    ("distances", "root", "expected_edges"),
    [
        (
            np.array([[0.0, 0.0, 4.0], [0.0, 0.0, 5.0], [4.0, 5.0, 0.0]]),
            0,
            [[0, 1], [0, 2]],
        ),
        (
            np.array([[0.0, 1e-9, 4.0], [1e-9, 0.0, 5.0], [4.0, 5.0, 0.0]]),
            0,
            [[0, 1], [0, 2]],
        ),
        (
            np.array([[0.0, 5.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            2,
            [[0, 2], [1, 2]],
        ),
        (np.zeros((4, 4)), 0, None),
    ],
    ids=["zero edge", "edge below 1e-8", "forest", "all equal"],
)
def test_spanning_tree_joins_clusters_with_equal_means(distances, root, expected_edges):
    tree = spanning_tree(distances, root=root)

    assert len(tree.edges) == distances.shape[0] - 1
    if expected_edges is not None:
        assert tree.edges.tolist() == expected_edges
    assert len(tree.lineages) == len(tree.leaves) - int(root in tree.leaves)


@pytest.mark.parametrize(
    ("root", "ends", "message"),
    [
        (None, [1, 1], "ends must not repeat a cluster"),
        (None, [3], "ends must index clusters"),
        (None, [-1], "ends must index clusters"),
        (3, (), "root must index a cluster"),
        (-1, (), "root must index a cluster"),
        (1, [1], "The root cannot also be an end"),
        (None, [0, 1, 2], "At least one cluster must not be an end"),
    ],
)
def test_spanning_tree_rejects_invalid_roots_and_ends(root, ends, message):
    with pytest.raises(ValueError, match=message):
        spanning_tree(_line_distances([0.0, 1.0, 2.0]), root=root, ends=ends)


def test_tree_lineages_trace_breadth_first_paths_from_the_root():
    edges = np.array([[0, 1], [1, 2], [1, 3], [3, 4]])

    unrooted_leaves, unrooted = tree_lineages(5, edges, None)
    leaves, from_tip = tree_lineages(5, edges, 0)
    _, from_inner = tree_lineages(5, edges, 3)

    np.testing.assert_array_equal(unrooted_leaves, [0, 2, 4])
    assert unrooted == ()
    np.testing.assert_array_equal(leaves, [0, 2, 4])
    assert leaves.dtype == np.int64
    assert from_tip == ((0, 1, 2), (0, 1, 3, 4))
    assert from_inner == ((3, 1, 0), (3, 1, 2), (3, 4))


def test_labeled_tree_names_the_y_by_its_cluster_labels():
    # The Y's stem a-b-c forks into d-e and f-g; a, e and g end one branch
    # each, and with the root a the lineages run from it to e and to g.
    clusters = ["a", "b", "c", "d", "e", "f", "g"]
    sizes = np.arange(1, 8)
    lengths = np.array([2.0, 2.0, 8.0, 8.0, 8.0, 8.0])

    tree = labeled_tree(
        clusters, sizes, _Y_CENTERS, np.array(_Y_EDGES), lengths, root="a", ends=["g"]
    )

    assert tree.clusters == tuple(clusters)
    np.testing.assert_array_equal(tree.sizes, sizes)
    np.testing.assert_array_equal(tree.means, _Y_CENTERS)
    assert tree.edges.values.tolist() == [
        ["a", "b", 2.0],
        ["b", "c", 2.0],
        ["c", "d", 8.0],
        ["c", "f", 8.0],
        ["d", "e", 8.0],
        ["f", "g", 8.0],
    ]
    assert tree.root == "a"
    assert tree.ends == ("g",)
    assert tree.leaves == ("a", "e", "g")
    assert tree.lineages == (("a", "b", "c", "d", "e"), ("a", "b", "c", "f", "g"))
    # The module names the function in US spelling only.
    assert "labeled_tree" in lineage_tree.__all__
    assert not hasattr(lineage_tree, "labelled_tree")


def test_tree_lineages_of_a_single_cluster_have_no_leaves():
    leaves, lineages = tree_lineages(1, np.empty((0, 2), dtype=np.int64), 0)

    assert leaves.size == 0
    assert lineages == ()


# A tree built without a store, as ds.trajectory.lineage_tree builds it.


def test_a_tree_built_by_hand_over_three_clusters_on_a_line():
    # Three cells about each center, 0, 12 and 20, each cluster with a sample
    # variance of 1, so Slingshot's distance between clusters is the squared
    # gap of their means over 2: 72 from c1 to c2, 32 from c2 to c10 and 200
    # from c1 to c10. The cells come in an order that mixes the clusters.
    centers = {"c1": 0.0, "c2": 12.0, "c10": 20.0}
    cells = [
        (label, center + offset)
        for offset in (-1.0, 0.0, 1.0)
        for label, center in centers.items()
    ]
    labels = np.asarray([label for label, _ in cells], dtype=object)
    coordinates = np.asarray([[value] for _, value in cells])

    tree = build_cluster_tree(coordinates, labels, root="c1")

    # The clusters follow their natural order, as the saved trees do.
    assert tree.clusters == ("c1", "c2", "c10")
    np.testing.assert_array_equal(tree.sizes, [3, 3, 3])
    np.testing.assert_allclose(tree.means, [[0.0], [12.0], [20.0]])
    assert tree.edges[["source", "target"]].values.tolist() == [
        ["c1", "c2"],
        ["c2", "c10"],
    ]
    np.testing.assert_allclose(tree.edges["length"], [72.0, 32.0])
    assert tree.leaves == ("c1", "c10")
    assert tree.lineages == (("c1", "c2", "c10"),)
    assert tree.candidate_sinks == ("c10",)

    # Held as a leaf, c2 joins its nearest cluster, c10, after c1 and c10
    # are joined across their distance of 200.
    held = build_cluster_tree(coordinates, labels, root="c1", ends=["c2"])
    assert held.edges[["source", "target"]].values.tolist() == [
        ["c1", "c10"],
        ["c2", "c10"],
    ]
    np.testing.assert_allclose(held.edges["length"], [200.0, 32.0])
    assert held.ends == ("c2",)
    assert held.leaves == ("c1", "c2")
    assert held.lineages == (("c1", "c10", "c2"),)
    # Its arrays are read-only, as a saved tree's are.
    with pytest.raises(ValueError, match="read-only"):
        held.means[0, 0] = 1.0


def test_build_cluster_tree_refuses_what_lineage_tree_refuses():
    labels = np.asarray(["a", "a", "b", "b"], dtype=object)
    coordinates = np.asarray([[0.0, 1.0], [1.0, 0.0], [5.0, 6.0], [6.0, 5.0]])

    for arguments, error, message in (
        ({"ends": ("b",)}, TypeError, "ends must be a list"),
        ({"root": "a", "ends": ["a"]}, ValueError, "root cannot also be an end"),
        ({"root": "z"}, ValueError, "root 'z' is not a cluster label"),
        ({"ends": ["z"]}, ValueError, "End 'z' is not a cluster label"),
        ({"ends": ["a", "a"]}, ValueError, "must not contain duplicate labels"),
        ({"root": [1]}, TypeError, "root must contain scalar labels"),
        ({"ends": ["a", "b"]}, ValueError, "At least one cluster must not be an end"),
    ):
        with pytest.raises(error, match=message):
            build_cluster_tree(coordinates, labels, **arguments)
    for cells, cell_labels, error, message in (
        (coordinates, np.asarray(["a"] * 4), ValueError, "at least two clusters"),
        (
            coordinates,
            np.asarray(["a", None, "b", "b"], dtype=object),
            ValueError,
            "Every cell needs a label",
        ),
        (
            coordinates,
            np.asarray([1.0, np.nan, 2.0, 2.0]),
            ValueError,
            "Every cell needs a label",
        ),
        (coordinates, labels[:3], ValueError, "one label for each row"),
        (coordinates, labels.reshape(2, 2), ValueError, "one label for each row"),
        (coordinates[:, 0], labels, ValueError, "cells by dimensions"),
        (
            np.where(coordinates > 5.5, np.inf, coordinates),
            labels,
            ValueError,
            "Coordinates must be finite",
        ),
        (coordinates.astype(str), labels, TypeError, "coordinates must hold numbers"),
    ):
        with pytest.raises(error, match=message):
            build_cluster_tree(cells, cell_labels)
    # Two clusters of one cell each, at different positions, are infinitely
    # far apart, so no tree joins them.
    with pytest.raises(ValueError, match="only across an infinite distance"):
        build_cluster_tree([[0.0], [1.0]], ["a", "b"])
