"""Compiled WNN and SNN integration against the NumPy loops they replaced.

The references below are the per-cell WNN loop and the per-row SNN merge that
the compiled kernels replaced, kept as they were. The kernels repeat their
arithmetic and summation order, so every comparison is exact: the same
columns, the same float32 weights bit for bit, and the same modality weights.
"""

import numpy as np
import pytest
from scipy.sparse import coo_matrix, csr_matrix

from cytearc.neighbors import graph as neighbor_graph
from cytearc.neighbors import integration


def _legacy_kernel_affinity(
    distances: np.ndarray, nearest_distance: float, bandwidth: float
) -> np.ndarray:
    adjusted = np.maximum(distances - nearest_distance, 0.0)
    tolerance = 8.0 * np.finfo(np.float64).eps * nearest_distance
    if bandwidth <= tolerance:
        return (adjusted <= tolerance).astype(np.float64)
    return np.exp(-(adjusted / bandwidth))


def _legacy_prediction_affinity(
    point: np.ndarray, estimate: np.ndarray, nearest_distance: float, bandwidth: float
) -> float:
    estimate_distance = float(np.linalg.norm(point - estimate))
    return float(
        _legacy_kernel_affinity(
            np.asarray([estimate_distance]), nearest_distance, bandwidth
        )[0]
    )


def _legacy_wnn(
    modalities: list[tuple[str, np.ndarray, np.ndarray]], *, l2_normalize: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score one cell at a time, as WNN integration did before its kernels."""
    indices = [np.asarray(values) for _name, values, _embedding in modalities]
    embeddings = []
    for _name, _values, values in modalities:
        embedding = np.asarray(values)
        if not np.issubdtype(embedding.dtype, np.floating):
            embedding = embedding.astype(np.float64)
        embeddings.append(embedding)
    inverse_norms = (
        [integration._inverse_row_norms(embedding) for embedding in embeddings]
        if l2_normalize
        else [None] * len(modalities)
    )
    n_cells = indices[0].shape[0]
    nk = min(values.shape[1] for values in indices)
    columns = np.empty((n_cells, nk), dtype=np.uint32)
    data = np.empty((n_cells, nk), dtype=np.float32)
    weights_out = np.empty((n_cells, len(modalities)), dtype=np.float32)
    for cell in range(n_cells):
        rows = [values[cell] for values in indices]
        mixed = rows[0]
        for neighbors in rows[1:]:
            mixed = np.union1d(mixed, neighbors)
        positions = [np.searchsorted(mixed, neighbors) for neighbors in rows]
        points, candidates_list, distances, nearest, bandwidths = [], [], [], [], []
        for embedding, inverse, own in zip(
            embeddings, inverse_norms, positions, strict=True
        ):
            candidates = embedding[mixed]
            point = embedding[cell]
            if inverse is not None:
                candidates = candidates * inverse[mixed, np.newaxis]
                point = point * inverse[cell]
            modality_distances = np.linalg.norm(point - candidates, axis=1)
            ranked = np.sort(modality_distances[own])
            nearest_distance = float(ranked[0])
            points.append(point)
            candidates_list.append(candidates)
            distances.append(modality_distances)
            nearest.append(nearest_distance)
            bandwidths.append(float(ranked[-1] - nearest_distance))
        within = [
            _legacy_prediction_affinity(
                point, candidates[own].mean(axis=0), nearest_distance, bandwidth
            )
            for point, candidates, own, nearest_distance, bandwidth in zip(
                points, candidates_list, positions, nearest, bandwidths, strict=True
            )
        ]
        scores = np.full((len(modalities), len(modalities)), -np.inf)
        for target in range(len(modalities)):
            for source, source_positions in enumerate(positions):
                if source == target:
                    continue
                cross = _legacy_prediction_affinity(
                    points[target],
                    candidates_list[target][source_positions].mean(axis=0),
                    nearest[target],
                    bandwidths[target],
                )
                scores[target, source] = np.clip(
                    within[target] / (cross + 1e-4), 0, 200
                )
        finite_scores = scores[np.isfinite(scores)]
        if finite_scores.size == 0:
            raise FloatingPointError(
                "WNN integration produced non-finite modality weights"
            )
        max_score = float(finite_scores.max())
        strengths = np.zeros(scores.shape, dtype=np.float64)
        finite = np.isfinite(scores)
        strengths[finite] = np.exp(scores[finite] - max_score)
        modality_strengths = strengths.sum(axis=1)
        weights = modality_strengths / modality_strengths.sum()
        weights_out[cell] = weights
        combined = weights[0] * _legacy_kernel_affinity(
            distances[0], nearest[0], bandwidths[0]
        )
        for weight, values, nearest_distance, bandwidth in zip(
            weights[1:], distances[1:], nearest[1:], bandwidths[1:], strict=True
        ):
            combined += weight * _legacy_kernel_affinity(
                values, nearest_distance, bandwidth
            )
        selected = np.lexsort((mixed, -combined))[:nk]
        columns[cell] = mixed[selected]
        data[cell] = np.maximum(
            np.clip(combined[selected], 0.0, 1.0), np.finfo(np.float32).tiny
        )
    if not np.all(np.isfinite(data)):
        raise FloatingPointError("WNN integration produced non-finite graph weights")
    return columns, data, weights_out


def _neighbors(
    rng: np.random.Generator, n_cells: int, k: int, dtype: type = np.int64
) -> np.ndarray:
    """Return ``k`` distinct neighbours of every cell, without the cell itself."""
    indices = np.empty((n_cells, k), dtype=dtype)
    for cell in range(n_cells):
        choice = rng.choice(n_cells - 1, size=k, replace=False)
        choice[choice >= cell] += 1
        indices[cell] = choice
    return indices


def _modalities(
    rng: np.random.Generator,
    n_cells: int,
    counts: tuple[int, ...],
    dims: tuple[int, ...],
    dtype: type,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return clustered modalities whose coordinates share four groups."""
    labels = rng.integers(4, size=n_cells)
    modalities = []
    for index, (k, d) in enumerate(zip(counts, dims, strict=True)):
        centers = rng.normal(scale=3.0, size=(4, d))
        coordinates = (centers[labels] + rng.normal(size=(n_cells, d))).astype(dtype)
        index_dtype = np.uint32 if index % 2 == 0 else np.int64
        modalities.append(
            (f"m{index}", _neighbors(rng, n_cells, k, index_dtype), coordinates)
        )
    return modalities


def _assert_matches_legacy(
    modalities: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    l2_normalize: bool = True,
    nthreads: int = 1,
) -> None:
    columns, data, weights = _legacy_wnn(modalities, l2_normalize=l2_normalize)
    graph, modality_weights = integration._wnn_integration_many(
        modalities, nthreads, l2_normalize=l2_normalize
    )
    n_cells, nk = columns.shape
    np.testing.assert_array_equal(graph.row, np.repeat(np.arange(n_cells), nk))
    np.testing.assert_array_equal(graph.col, columns.ravel())
    # Bit for bit: the float32 bytes are compared.
    assert graph.data.tobytes() == data.tobytes()
    assert modality_weights.tobytes() == weights.tobytes()
    assert graph.data.dtype == modality_weights.dtype == np.float32


@pytest.mark.parametrize("l2_normalize", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    ("counts", "dims"),
    [
        ((6, 6), (5, 3)),
        ((9, 4), (12, 7)),
        ((5, 7, 3), (4, 9, 2)),
    ],
    ids=["two", "two-unequal", "three-unequal"],
)
def test_wnn_kernels_match_the_per_cell_loop(
    monkeypatch, counts, dims, dtype, l2_normalize
):
    rng = np.random.default_rng(len(counts) * 100 + sum(dims))
    modalities = _modalities(rng, 90, counts, dims, dtype)
    # Blocks of 32 cells put block edges inside the data.
    monkeypatch.setattr(integration, "_WNN_BLOCK_CELLS", 32)

    _assert_matches_legacy(modalities, l2_normalize=l2_normalize)


@pytest.mark.parametrize("l2_normalize", [True, False])
def test_wnn_kernels_match_on_duplicate_cells_and_ties(l2_normalize):
    rng = np.random.default_rng(3)
    modalities = _modalities(rng, 60, (8, 8), (3, 4), np.float64)
    for _name, _indices, coordinates in modalities:
        # Identical cells and coordinates on a coarse grid tie distances,
        # affinities, and bandwidths.
        coordinates[1] = coordinates[0]
        coordinates[2] = coordinates[0]
        coordinates[:] = np.round(coordinates)
    modalities[1][2][:30] = 0.0

    _assert_matches_legacy(modalities, l2_normalize=l2_normalize)


@pytest.mark.parametrize("l2_normalize", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_wnn_kernels_match_on_wide_and_single_coordinates(dtype, l2_normalize):
    # 300 coordinates sum in NumPy's split pairwise order, and twelve
    # neighbours of a single coordinate average pairwise.
    rng = np.random.default_rng(8)
    modalities = _modalities(rng, 40, (12, 12), (300, 1), dtype)

    _assert_matches_legacy(modalities, l2_normalize=l2_normalize)


def test_wnn_kernels_do_not_depend_on_the_thread_count(monkeypatch):
    rng = np.random.default_rng(12)
    modalities = _modalities(rng, 120, (7, 7, 5), (6, 4, 3), np.float32)
    monkeypatch.setattr(integration, "_WNN_BLOCK_CELLS", 50)
    single, single_weights = integration._wnn_integration_many(modalities, 1)

    for threads in (2, 3):
        graph, weights = integration._wnn_integration_many(modalities, threads)
        np.testing.assert_array_equal(graph.col, single.col)
        np.testing.assert_array_equal(graph.data, single.data)
        np.testing.assert_array_equal(weights, single_weights)
    _assert_matches_legacy(modalities, nthreads=3)


def test_wnn_scores_cells_in_compiled_kernels_on_nthreads(monkeypatch):
    import numba

    rng = np.random.default_rng(13)
    modalities = _modalities(rng, 40, (5, 5), (4, 3), np.float32)
    expected = integration._wnn_integration_many(modalities, 1)
    previous = numba.get_num_threads()
    distances = integration._candidate_distances
    threads: list[int] = []

    def recorded(*args):
        threads.append(numba.get_num_threads())
        return distances(*args)

    def per_cell(*_args):
        raise AssertionError("WNN scored a cell in a NumPy loop")

    monkeypatch.setattr(integration, "_candidate_distances", recorded)
    monkeypatch.setattr(np, "union1d", per_cell)
    monkeypatch.setattr(integration, "_integrate_cells", per_cell)

    graph, weights = integration._wnn_integration_many(modalities, 2)

    # The kernels ran on the requested threads, which are restored afterwards.
    assert threads == [min(2, numba.config.NUMBA_NUM_THREADS)] * 2
    assert numba.get_num_threads() == previous
    np.testing.assert_array_equal(graph.col, expected[0].col)
    np.testing.assert_array_equal(weights, expected[1])


def test_snn_merges_blocks_of_rows_without_a_per_row_sort(monkeypatch):
    rng = np.random.default_rng(15)
    graphs = [_graph(rng, _neighbors(rng, 30, 4)) for _ in range(2)]
    expected = _legacy_merge(graphs)

    def per_row(*_args):
        raise AssertionError("SNN merged a row in a NumPy loop")

    monkeypatch.setattr(neighbor_graph, "weight_sort_indices", per_row)
    merged = neighbor_graph.merge_graphs(graphs)

    np.testing.assert_array_equal(merged.col, expected.col)
    np.testing.assert_array_equal(merged.data, expected.data)


@pytest.mark.parametrize("l2_normalize", [True, False])
@pytest.mark.parametrize("dtype", [np.float16, np.dtype(">f8")], ids=str)
def test_wnn_scores_other_float_types_in_their_own_precision(dtype, l2_normalize):
    rng = np.random.default_rng(21)
    modalities = _modalities(rng, 30, (5, 5), (4, 3), np.float64)
    modalities[0] = (
        modalities[0][0],
        modalities[0][1],
        modalities[0][2].astype(dtype),
    )

    _assert_matches_legacy(modalities, l2_normalize=l2_normalize)


@pytest.mark.parametrize("l2_normalize", [True, False])
def test_wnn_scores_extended_precision_as_the_loop_did(l2_normalize):
    rng = np.random.default_rng(22)
    modalities = _modalities(rng, 25, (4, 4), (3, 2), np.longdouble)

    _assert_matches_legacy(modalities, l2_normalize=l2_normalize)


@pytest.mark.parametrize(
    ("dtype", "scale"),
    [(np.float64, 1e200), (np.float16, 1e3)],
    ids=["float64", "float16"],
)
@pytest.mark.parametrize(
    ("overflowing", "message"),
    [
        ((True, False), "non-finite graph weights"),
        ((True, True), "non-finite modality weights"),
    ],
)
def test_wnn_kernels_report_non_finite_weights_like_the_loop(
    dtype, scale, overflowing, message
):
    # Squared distances of these coordinates overflow; float16 coordinates
    # are scored one cell at a time.
    rng = np.random.default_rng(4)
    modalities = [
        (name, indices, (coordinates * (scale if overflows else 1.0)).astype(dtype))
        for (name, indices, coordinates), overflows in zip(
            _modalities(rng, 8, (3, 3), (3, 2), np.float64), overflowing, strict=True
        )
    ]
    # Every cell's neighbours, so overflowing distances reach the graph.
    for _name, indices, _coordinates in modalities:
        indices[:] = [
            [neighbor for neighbor in range(8) if neighbor != cell][:3]
            for cell in range(8)
        ]

    with np.errstate(over="ignore", invalid="ignore"):
        for run in (
            lambda: _legacy_wnn(modalities, l2_normalize=False),
            lambda: integration._wnn_integration_many(
                modalities, 1, l2_normalize=False
            ),
        ):
            with pytest.raises(FloatingPointError, match=message):
                run()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pairwise_sum_adds_in_numpy_reduction_order(dtype):
    rng = np.random.default_rng(1)
    zero = dtype(0)
    for count in (1, 5, 7, 8, 9, 16, 61, 128, 129, 136, 255, 300, 1000, 4097):
        values = (rng.normal(size=count + 3) * 1e3).astype(dtype)
        expected = np.add.reduce(values[3:])

        for kernel in (integration._pairwise_sum, integration._pairwise_sum.py_func):
            total = zero + kernel(values, 3, count, zero)
            assert np.asarray(total, dtype=dtype).tobytes() == expected.tobytes()
        if count <= 128:
            run = integration._pairwise_run.py_func(values, 3, count, zero)
            assert np.asarray(zero + run, dtype=dtype).tobytes() == expected.tobytes()


def test_candidate_pools_union_rows_and_locate_neighbors():
    stacked = np.array([[4, 1, 7, 1, 9, 4], [2, 3, 5, 0, 8, 6]], dtype=np.int64)
    pools = np.full(stacked.shape, -1, dtype=np.int64)
    sizes = np.empty(2, dtype=np.int64)
    positions = np.empty(stacked.shape, dtype=np.int64)

    integration._candidate_pools.py_func(
        stacked, np.argsort(stacked, axis=1), pools, sizes, positions
    )

    np.testing.assert_array_equal(sizes, [4, 6])
    np.testing.assert_array_equal(pools[0, :4], [1, 4, 7, 9])
    np.testing.assert_array_equal(pools[1], [0, 2, 3, 5, 6, 8])
    np.testing.assert_array_equal(positions[0], [1, 0, 2, 0, 3, 1])
    np.testing.assert_array_equal(positions[1], [1, 2, 3, 0, 5, 4])


@pytest.mark.parametrize("normalize", [False, True])
def test_candidate_distances_kernel_matches_numpy_norms(normalize):
    rng = np.random.default_rng(2)
    embedding = rng.normal(size=(6, 3)).astype(np.float32)
    inverse = integration._inverse_row_norms(embedding) if normalize else None
    zero = np.float64(0) if normalize else np.float32(0)
    pools = np.array([[1, 2, 5, 0], [0, 3, 4, 0]], dtype=np.int64)
    sizes = np.array([3, 2], dtype=np.int64)
    distances = np.full((2, 4), np.nan, dtype=zero.dtype)

    integration._candidate_distances.py_func(
        embedding,
        inverse,
        1,
        pools,
        sizes,
        zero,
        np.empty((2, 6), dtype=zero.dtype),
        distances,
    )

    values = embedding if inverse is None else embedding * inverse[:, np.newaxis]
    for row, cell in enumerate((1, 2)):
        size = sizes[row]
        expected = np.linalg.norm(values[cell] - values[pools[row, :size]], axis=1)
        np.testing.assert_array_equal(distances[row, :size], expected)
        np.testing.assert_array_equal(distances[row, size:], 0)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("dims", [1, 3])
def test_prediction_offsets_kernel_matches_numpy_means(dims, normalize):
    rng = np.random.default_rng(6)
    embedding = rng.normal(size=(12, dims)).astype(np.float32)
    inverse = integration._inverse_row_norms(embedding) if normalize else None
    zero = np.float64(0) if normalize else np.float32(0)
    stacked = np.array(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [0, 2, 4, 6, 8, 10, 11, 9, 7, 5, 3]],
        dtype=np.int64,
    )
    starts = np.array([0, 2], dtype=np.int64)
    counts = np.array([2, 9], dtype=np.int64)
    offsets = np.empty((2, 2, dims), dtype=zero.dtype)

    integration._prediction_offsets.py_func(
        embedding,
        inverse,
        0,
        stacked,
        starts,
        counts,
        zero,
        np.empty((2, 18), dtype=zero.dtype),
        offsets,
    )

    values = embedding if inverse is None else embedding * inverse[:, np.newaxis]
    for cell in range(2):
        for source, (start, count) in enumerate(zip(starts, counts, strict=True)):
            members = stacked[cell, start : start + count]
            expected = values[cell] - values[members].mean(axis=0)
            np.testing.assert_array_equal(offsets[cell, source], expected)


def test_candidates_rank_like_a_lexsort_with_nan_last():
    precedes = integration._precedes.py_func
    nan = float("nan")

    assert precedes(0.5, 3, 0.25, 1)
    assert not precedes(0.25, 1, 0.5, 3)
    assert precedes(0.5, 1, 0.5, 3)
    assert not precedes(0.5, 3, 0.5, 1)
    assert precedes(0.0, 9, nan, 1)
    assert not precedes(nan, 1, 0.0, 9)
    assert precedes(nan, 1, nan, 2)
    assert not precedes(nan, 2, nan, 1)


def test_select_candidates_kernel_matches_lexsort_and_clipping():
    tiny = float(np.finfo(np.float32).tiny)
    combined = np.array(
        [
            [0.5, np.nan, 1.5, 0.5, -0.25, 0.0],
            [1e-45, 0.75, np.nan, 0.75, 0.2, 0.9],
            [0.9, 0.8, 0.7, 0.6, 0.1, 0.0],
            [-0.5, 0.2, np.nan, 0.3, 0.0, 0.0],
        ]
    )
    pools = np.array(
        [
            [3, 5, 8, 9, 12, 14],
            [1, 2, 4, 6, 7, 0],
            [0, 1, 2, 3, 4, 5],
            [2, 4, 6, 8, 0, 0],
        ],
        dtype=np.int64,
    )
    sizes = np.array([6, 5, 5, 4], dtype=np.int64)
    columns = np.empty((4, 4), dtype=np.uint32)
    weights = np.empty((4, 4), dtype=np.float32)

    integration._select_candidates.py_func(
        combined,
        pools,
        sizes,
        np.empty((4, 4), dtype=np.int64),
        columns,
        weights,
    )

    for row, size in enumerate(sizes):
        selected = np.lexsort((pools[row, :size], -combined[row, :size]))[:4]
        np.testing.assert_array_equal(columns[row], pools[row, selected])
        expected = np.maximum(np.clip(combined[row, selected], 0.0, 1.0), tiny)
        assert weights[row].tobytes() == expected.astype(np.float32).tobytes()
    np.testing.assert_array_equal(columns[0], [8, 3, 9, 14])
    assert weights[0, 0] == 1 and weights[0, 3] == np.float32(tiny)
    # Kept NaN stays NaN, and a negative affinity becomes the smallest weight.
    np.testing.assert_array_equal(columns[3], [8, 4, 2, 6])
    assert weights[3, 2] == np.float32(tiny) and np.isnan(weights[3, 3])


def _legacy_shared_counts(indices: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [
                len(set(row.tolist()) & set(indices[neighbor].tolist()))
                for neighbor in row
            ]
            for row in indices
        ],
        dtype=np.float64,
    )


def _legacy_merge(graphs: list[csr_matrix]) -> coo_matrix:
    """Merge one row at a time, as SNN integration did before its kernels."""
    n_cells = graphs[0].shape[0]
    nk = int(np.diff(graphs[0].indptr)[0])
    neighbor_rows = [graph.indices.reshape(n_cells, nk) for graph in graphs]
    weight_rows = [graph.data.reshape(n_cells, nk) for graph in graphs]
    snns = [_legacy_shared_counts(rows) / (nk - 1) for rows in neighbor_rows]
    columns = np.empty(
        n_cells * nk, dtype=np.result_type(*(rows.dtype for rows in neighbor_rows))
    )
    data = np.empty(
        n_cells * nk, dtype=np.result_type(*(rows.dtype for rows in weight_rows))
    )
    for row in range(n_cells):
        merged, weights = neighbor_graph.weight_sort_indices(
            np.hstack([rows[row] for rows in neighbor_rows]),
            np.hstack([rows[row] for rows in weight_rows]),
            np.hstack(
                [
                    rows[row] + snn[row]
                    for rows, snn in zip(weight_rows, snns, strict=True)
                ]
            ),
            nk,
        )
        columns[row * nk : (row + 1) * nk] = merged
        data[row * nk : (row + 1) * nk] = weights
    rows_out = np.repeat(np.arange(n_cells), nk)
    return coo_matrix((data, (rows_out, columns)), shape=graphs[0].shape)


def _graph(
    rng: np.random.Generator,
    neighbors: np.ndarray,
    *,
    weight_dtype: type = np.float32,
    ties: bool = False,
) -> csr_matrix:
    n_cells, k = neighbors.shape
    if ties:
        # Quarter steps give many equal ranks.
        weights = rng.integers(0, 5, size=(n_cells, k)) / 4
    else:
        weights = rng.random((n_cells, k))
    return csr_matrix(
        (
            weights.astype(weight_dtype).ravel(),
            (np.repeat(np.arange(n_cells), k), neighbors.ravel()),
        ),
        shape=(n_cells, n_cells),
    )


def _sharing_neighbors(
    rng: np.random.Generator, neighbors: np.ndarray, shared: int
) -> np.ndarray:
    """Return other neighbour rows that keep ``shared`` of each row's neighbours."""
    n_cells, k = neighbors.shape
    rows = np.empty_like(neighbors)
    for cell in range(n_cells):
        others = np.setdiff1d(np.arange(n_cells), [cell, *neighbors[cell]])
        rows[cell, :shared] = neighbors[cell, :shared]
        rows[cell, shared:] = rng.choice(others, size=k - shared, replace=False)
    return rows


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("n_graphs", [1, 2, 3])
def test_snn_merge_matches_the_per_row_merge(monkeypatch, n_graphs, ties):
    rng = np.random.default_rng(10 + n_graphs)
    first = _neighbors(rng, 70, 6)
    # Edges that graphs share reach the merge as repeated candidates.
    rows = [first] + [_sharing_neighbors(rng, first, 3) for _ in range(n_graphs - 1)]
    graphs = [_graph(rng, neighbors, ties=ties) for neighbors in rows]
    expected = _legacy_merge(graphs)
    # Blocks of 16 rows put block edges inside the data.
    monkeypatch.setattr(neighbor_graph, "_MERGE_BLOCK_ROWS", 16)

    merged = neighbor_graph.merge_graphs(graphs)

    np.testing.assert_array_equal(merged.row, expected.row)
    np.testing.assert_array_equal(merged.col, expected.col)
    np.testing.assert_array_equal(merged.data, expected.data)
    assert merged.col.dtype == expected.col.dtype
    assert merged.data.dtype == expected.data.dtype


def test_snn_merge_matches_with_mixed_index_and_weight_dtypes():
    rng = np.random.default_rng(31)
    neighbors = _neighbors(rng, 50, 5)
    first = _graph(rng, neighbors, ties=True)
    second = _graph(
        rng,
        _sharing_neighbors(rng, neighbors, 2),
        weight_dtype=np.float64,
        ties=True,
    )
    second.indices = second.indices.astype(np.int64)
    second.indptr = second.indptr.astype(np.int64)

    expected = _legacy_merge([first, second])
    merged = neighbor_graph.merge_graphs([first, second])

    np.testing.assert_array_equal(merged.col, expected.col)
    np.testing.assert_array_equal(merged.data, expected.data)
    assert merged.data.dtype == np.float64


def test_snn_merge_rejects_rows_with_too_few_distinct_neighbors():
    # A non-canonical row lists neighbour 1 three times.
    graph = csr_matrix(
        (
            np.ones(12),
            np.array([1, 1, 1, 0, 2, 3, 0, 1, 3, 0, 1, 2]),
            np.arange(0, 13, 3),
        ),
        shape=(4, 4),
    )

    with pytest.raises(ValueError, match="Row 0 of the merged graphs has 1 distinct"):
        neighbor_graph.merge_graphs([graph])


@pytest.mark.parametrize(
    "kernel",
    [
        neighbor_graph._count_shared_neighbors,
        neighbor_graph._count_shared_neighbors.py_func,
    ],
    ids=["jit", "python"],
)
def test_shared_neighbor_counts_count_repeated_neighbors_once(kernel):
    rng = np.random.default_rng(14)
    indices = rng.integers(0, 9, size=(9, 5))
    # Repeated values inside rows, as a set intersection counts them.
    indices[:, 1] = indices[:, 0]
    counts = np.empty(indices.shape, dtype=np.int64)

    kernel(indices, counts)

    np.testing.assert_array_equal(counts, _legacy_shared_counts(indices))


def test_sorted_shared_neighbor_kernel_merges_sorted_rows():
    indices = np.array([[1, 2, 2, 3], [0, 3, 3, 2], [3, 3, 3, 0], [0, 1, 2, 2]])
    counts = np.empty(indices.shape, dtype=np.int64)

    neighbor_graph._count_sorted_shared_neighbors.py_func(
        indices, np.sort(indices, axis=1), counts
    )

    np.testing.assert_array_equal(counts, _legacy_shared_counts(indices))


def test_merge_ranked_rows_kernel_keeps_first_distinct_candidates():
    candidates = np.array([[5, 7, 5, 9], [2, 2, 2, 2]], dtype=np.int32)
    weights = np.array([[0.5, 0.25, 0.75, 1.0], [0.1, 0.2, 0.3, 0.4]], np.float32)
    order = np.array([[3, 1, 0, 2], [0, 1, 2, 3]])
    columns = np.zeros((2, 2), dtype=np.int32)
    kept_weights = np.zeros((2, 2), dtype=np.float32)
    counts = np.empty(2, dtype=np.int64)

    neighbor_graph._merge_ranked_rows.py_func(
        candidates, weights, order, columns, kept_weights, counts
    )

    # Read from the end of each order, row 0 keeps 5 at its best-ranked copy.
    np.testing.assert_array_equal(columns[0], [5, 7])
    np.testing.assert_array_equal(kept_weights[0], [0.75, 0.25])
    np.testing.assert_array_equal(counts, [2, 1])
    assert columns[1, 0] == 2 and kept_weights[1, 0] == np.float32(0.4)
