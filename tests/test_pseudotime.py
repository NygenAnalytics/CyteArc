import numpy as np
import pandas as pd
import pytest
import zarr
from scipy.sparse import csr_matrix

from cytearc.trajectory.feature_dynamics import (
    aggregate_feature_profiles,
    knn_clustering,
)
from cytearc.trajectory.feature_dynamics import (
    scatter_feature_clusters as _scatter_feature_clusters,
    validate_pseudotime_regressor as _validated_pseudotime_regressor,
)
from cytearc.trajectory.pseudotime import (
    make_source_sink_vector as _make_source_sink_vector,
    random_walk_laplacian_transpose as _random_walk_laplacian_transpose,
    select_pseudotime_component as _select_pseudotime_component,
    truncated_pba_potential as _truncated_pba_potential,
    validate_source_sink_labels as _validate_source_sink_labels,
    validate_source_sink_vector as _validate_source_sink_vector,
)
from cytearc.trajectory.parameters import AGGREGATION_ANN_DEFAULTS
from cytearc.trajectory.results import PseudotimeMarkerResult


def _metadata_values(table):
    return {
        column: np.asarray(table.fetch_all(column)).copy() for column in table.columns
    }


@pytest.fixture(scope="module")
def few_features(datastore, detected_features):
    """Forty detected features: enough for each check, and quick to analyze."""
    from cytearc.storage.artifacts import artifact_group

    detected = np.asarray(
        artifact_group(datastore.zw, detected_features)["values"][:], dtype=bool
    )
    return datastore.features.snapshot(
        from_assay="RNA",
        feature_indexes=np.flatnonzero(detected)[:40].tolist(),
    )


def _pearson_markers(datastore, pseudotime, feature_indices, normalization):
    """Correlate lib-size normalized counts with pseudotime over scored cells."""
    from cytearc.storage.selections import read_stored_selection_indices

    scored = datastore.trajectory.load_pseudotime(pseudotime)
    cells = read_stored_selection_indices(
        datastore.zw,
        scored.cell_selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )[scored.valid]
    counts = np.asarray(
        datastore.RNA.rawData._backing.oindex[cells, feature_indices],
        dtype=np.float64,
    )
    totals = np.asarray(datastore.cells.fetch_all("RNA_nCounts"), dtype=np.float64)
    values = counts / totals[cells, np.newaxis] * normalization["size_factor"]
    if normalization["log_transform"]:
        values = np.log1p(values)
    ptime = scored.values[scored.valid]
    return np.array(
        [np.corrcoef(column, ptime)[0, 1] for column in values.T], dtype=np.float64
    )


def _assert_metadata_unchanged(table, before):
    assert set(table.columns) == set(before)
    for column, values in before.items():
        np.testing.assert_array_equal(table.fetch_all(column), values)


def test_source_sink_vector_supports_one_sided_supervision():
    labels = pd.Series(["source", "source", "other", "other"])
    vector = _make_source_sink_vector(labels, ["source"], [])

    np.testing.assert_array_equal(vector, [-1.0, -1.0, 1.0, 1.0])
    assert vector.sum() == pytest.approx(0.0)


def test_source_sink_labels_reject_missing_and_overlap():
    labels = pd.Series(["a", "b", "c"])

    with pytest.raises(ValueError, match="overlap"):
        _validate_source_sink_labels(labels, ["a"], ["a"], "test cells")
    with pytest.raises(ValueError, match="Missing sources"):
        _validate_source_sink_labels(labels, ["missing"], [], "test cells")


def test_source_sink_vector_rejects_unbalanced_and_nonfinite_values():
    for values, message in (
        (np.array([-2.0, 1.0]), "The values in ss_vec must sum to zero"),
        (np.array([-1.0, 2.0]), "The values in ss_vec must sum to zero"),
        # The tolerance scales with the vector, so a rounding residue passes
        # but a residue of the same size on a unit vector does not.
        (np.array([-1.0, 1.0 + 1e-9]), "The values in ss_vec must sum to zero"),
        (np.array([-1.0, np.nan]), "ss_vec must contain only finite values"),
        (np.array([-1.0, np.inf]), "ss_vec must contain only finite values"),
    ):
        with pytest.raises(ValueError, match=message):
            _validate_source_sink_vector(values, 2, "ss_vec")
    with pytest.raises(TypeError, match="ss_vec must contain numeric values"):
        _validate_source_sink_vector(np.array(["source", "sink"]), 2, "ss_vec")

    column = _validate_source_sink_vector(
        np.array([[-1.0], [1.0]]),
        2,
        "ss_vec",
    )
    np.testing.assert_array_equal(column, [-1.0, 1.0])
    balanced = _validate_source_sink_vector(
        np.array([-1e12, 1e12 + 1e-3]),
        2,
        "ss_vec",
    )
    np.testing.assert_array_equal(balanced, [-1e12, 1e12 + 1e-3])


def test_source_sink_vector_rejects_wrong_shapes():
    for values, message in (
        (np.zeros((2, 2)), r"must be one-dimensional or have shape \(2, 1\)"),
        (np.zeros((1, 2)), r"must be one-dimensional or have shape \(2, 1\)"),
        (np.zeros(3), r"Size mismatch between ss_vec \(3\) and graph \(2\)"),
    ):
        with pytest.raises(ValueError, match=message):
            _validate_source_sink_vector(values, 2, "ss_vec")


def test_raw_source_sink_scoring_round_trips_and_validates_snapshot(
    datastore,
    connectivity_graph,
):
    from scipy.sparse.csgraph import connected_components

    from cytearc.storage.artifacts import ArtifactRef, artifact_group

    graph = datastore.graph.load(
        connectivity_graph,
        symmetric=True,
        upper_only=False,
    )
    _, components = connected_components(graph, directed=False)
    component_sizes = np.bincount(components)
    retained = np.flatnonzero(components == int(np.argmax(component_sizes)))
    source_sink = np.zeros(graph.shape[0], dtype=np.float64)
    source_sink[retained[0]] = -1.0
    source_sink[retained[-1]] = 1.0

    with pytest.raises(TypeError, match="min_max_norm_ptime"):
        datastore.trajectory.pseudotime(
            connectivity_graph,
            ss_vec=source_sink,
            min_max_norm_ptime=1,
        )

    ref = datastore.trajectory.pseudotime(
        connectivity_graph,
        ss_vec=source_sink,
    )
    result = datastore.trajectory.load_pseudotime(ref)
    assert result.graph == connectivity_graph
    np.testing.assert_array_equal(
        result.valid, components == np.argmax(component_sizes)
    )

    status = datastore.artifacts.inspect(ref)
    snapshot = ArtifactRef.from_dict(status.inputs["source_sink"])
    snapshot_status = datastore.artifacts.inspect(snapshot)
    assert snapshot_status.operation == "snapshot_pseudotime_source_sink"
    assert snapshot_status.parameters == {"shape": [graph.shape[0]]}
    np.testing.assert_array_equal(
        artifact_group(datastore.zw, snapshot)["values"][:],
        source_sink,
    )

    snapshot_group = artifact_group(datastore.zw, snapshot)
    original_provenance = dict(snapshot_group.attrs["provenance"])
    corrupted = dict(original_provenance)
    corrupted["parameters"] = {"shape": [graph.shape[0], 1]}
    snapshot_group.attrs["provenance"] = corrupted
    try:
        with pytest.raises(ValueError, match="snapshot is invalid"):
            datastore.trajectory.load_pseudotime(ref)
    finally:
        snapshot_group.attrs["provenance"] = original_provenance

    datastore.trajectory.load_pseudotime(ref)


def test_raw_source_sink_scoring_owns_the_vector_before_validation_and_snapshot(
    datastore,
    connectivity_graph,
    monkeypatch,
):
    import cytearc.datastore._operations.trajectory as trajectory_operations
    from scipy.sparse.csgraph import connected_components

    graph = datastore.graph.load(
        connectivity_graph,
        symmetric=True,
        upper_only=False,
    )
    _, components = connected_components(graph, directed=False)
    component_sizes = np.bincount(components)
    retained = np.flatnonzero(components == int(np.argmax(component_sizes)))
    source_sink = np.zeros(graph.shape[0], dtype=np.float64)
    source_sink[retained[0]] = -1.0
    source_sink[retained[-1]] = 1.0
    expected = source_sink.copy()
    original_validator = trajectory_operations._validate_source_sink_vector_impl
    original_potential = trajectory_operations._pba_potential_impl
    received: list[np.ndarray] = []

    def mutate_caller_after_validation(values, n_cells, context):
        validated = original_validator(values, n_cells, context)
        if context == "ss_vec":
            source_sink[:] *= -1.0
        return validated

    def capture_source_sink(graph, values, **bounds):
        received.append(np.array(values, copy=True))
        return original_potential(graph, values, **bounds)

    monkeypatch.setattr(
        trajectory_operations,
        "_validate_source_sink_vector_impl",
        mutate_caller_after_validation,
    )
    monkeypatch.setattr(
        trajectory_operations,
        "_pba_potential_impl",
        capture_source_sink,
    )
    ref = datastore.trajectory.pseudotime(
        connectivity_graph,
        ss_vec=source_sink,
        invalidate_cache=True,
    )
    result = datastore.trajectory.load_pseudotime(ref)

    assert len(received) == 1
    np.testing.assert_array_equal(received[0], expected[result.valid])
    np.testing.assert_array_equal(source_sink, -expected)


def test_source_sink_vector_rejects_fully_labeled_cells():
    labels = pd.Series(["source", "sink"])
    with pytest.raises(ValueError, match="All selected cells"):
        _make_source_sink_vector(labels, ["source"], ["sink"])


def test_largest_component_selection_is_deterministic_on_ties():
    adjacency = np.zeros((6, 6), dtype=float)
    for start in (0, 3):
        adjacency[start, start + 1] = adjacency[start + 1, start] = 1.0
        adjacency[start + 1, start + 2] = adjacency[start + 2, start + 1] = 1.0
    selected_indices = np.array([10, 11, 12, 1, 2, 3])

    retained, sizes = _select_pseudotime_component(
        csr_matrix(adjacency),
        selected_indices,
        "largest",
    )

    assert sizes == [3, 3]
    np.testing.assert_array_equal(retained, [False, False, False, True, True, True])
    with pytest.raises(ValueError, match="connected components"):
        _select_pseudotime_component(
            csr_matrix(adjacency),
            selected_indices,
            "error",
        )


def test_largest_component_selection_keeps_a_unique_largest_component():
    adjacency = np.zeros((6, 6), dtype=float)
    # A pair, a path of three cells, and an isolated cell.
    for left, right in ((0, 1), (2, 3), (3, 4)):
        adjacency[left, right] = adjacency[right, left] = 1.0
    # The pair holds the smallest selected index, but the path is larger.
    selected_indices = np.array([1, 2, 30, 31, 32, 0])

    retained, sizes = _select_pseudotime_component(
        csr_matrix(adjacency),
        selected_indices,
        "largest",
    )

    # Sizes are listed largest first.
    assert sizes == [3, 2, 1]
    np.testing.assert_array_equal(retained, [False, False, True, True, True, False])


@pytest.mark.parametrize("seed", range(12))
def test_largest_component_selection_matches_a_scan_of_every_component(seed):
    from scipy.sparse.csgraph import connected_components

    rng = np.random.default_rng(seed)
    # Components of a few repeated sizes tie often; singletons are components.
    sizes = rng.choice([1, 2, 3, 3, 5], size=int(rng.integers(2, 40)))
    n_cells = int(sizes.sum())
    order = rng.permutation(n_cells)
    rows, columns = [], []
    start = 0
    for size in sizes.tolist():
        members = order[start : start + size]
        rows.extend(members[:-1].tolist())
        columns.extend(members[1:].tolist())
        start += size
    graph = csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(n_cells, n_cells))
    # Unsorted selected indices, which repeat when seeds are odd.
    selected = rng.permutation(np.arange(100, 100 + n_cells))
    if seed % 2:
        selected[rng.integers(0, n_cells, size=3)] = selected[0]

    retained, component_sizes = _select_pseudotime_component(graph, selected, "largest")

    n_components, labels = connected_components(
        graph, directed=False, return_labels=True
    )
    counts = np.bincount(labels, minlength=n_components).tolist()
    # Every component scanned in label order, the earlier selection rule.
    expected = min(
        range(n_components),
        key=lambda label: (-counts[label], int(selected[labels == label].min())),
    )
    assert component_sizes == sorted(counts, reverse=True)
    if n_components > 1:
        np.testing.assert_array_equal(retained, labels == expected)


@pytest.mark.parametrize(
    ("shape", "n_selected"), [((6, 6), 5), ((6, 5), 6), ((5, 6), 5)]
)
def test_component_selection_rejects_graphs_that_do_not_match_the_selection(
    shape, n_selected
):
    with pytest.raises(ValueError, match="one row per selected cell index"):
        _select_pseudotime_component(
            csr_matrix(shape), np.arange(n_selected), "largest"
        )


def test_truncated_pba_operator_matches_dense_svd_reference():
    adjacency = np.zeros((8, 8), dtype=float)
    weights = [1.0, 2.0, 1.5, 3.0, 0.75, 2.5, 1.25]
    for index, weight in enumerate(weights):
        adjacency[index, index + 1] = weight
        adjacency[index + 1, index] = weight
    laplacian_transpose = _random_walk_laplacian_transpose(csr_matrix(adjacency))
    source_sink = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    k = 5

    actual = _truncated_pba_potential(
        laplacian_transpose,
        k,
        17,
        source_sink,
    )

    left, singular_values, right_t = np.linalg.svd(
        laplacian_transpose.toarray(),
        full_matrices=False,
    )
    retained_modes = np.argsort(singular_values)[:k]
    retained_modes = retained_modes[np.argsort(singular_values[retained_modes])]
    nonzero_modes = retained_modes[1:]
    expected = left[:, nonzero_modes] @ (
        (1.0 / singular_values[nonzero_modes])
        * (right_t[nonzero_modes, :] @ source_sink)
    )

    # ARPACK converges to machine precision, and the smallest nonzero singular
    # value, about 0.08, amplifies its rounding about tenfold.
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


def test_truncated_pba_logs_svd_stage_start():
    from loguru import logger

    adjacency = np.zeros((6, 6), dtype=float)
    for index in range(5):
        adjacency[index, index + 1] = 1.0
        adjacency[index + 1, index] = 1.0
    laplacian_transpose = _random_walk_laplacian_transpose(csr_matrix(adjacency))
    source_sink = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="DEBUG",
    )
    try:
        _truncated_pba_potential(laplacian_transpose, 3, 7, source_sink)
    finally:
        logger.remove(sink)

    # A path of six cells has five edges each way and six diagonal entries.
    assert messages == [
        "Pseudotime scoring: calculating SVD (shape=(6, 6), nnz=16, k=3)"
    ]


def test_dense_pba_reference_orders_a_path_from_source_to_sink():
    adjacency = np.zeros((7, 7), dtype=float)
    for index, weight in enumerate([1.0, 2.0, 1.5, 3.0, 0.75, 2.5]):
        adjacency[index, index + 1] = weight
        adjacency[index + 1, index] = weight
    laplacian_transpose = _random_walk_laplacian_transpose(csr_matrix(adjacency))
    source_sink = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    # The transposed random-walk Laplacian is I - A D^-1: column j of the
    # adjacency is divided by the degree of cell j.
    degree = adjacency.sum(axis=1)
    np.testing.assert_allclose(
        laplacian_transpose.toarray(),
        np.eye(7) - adjacency / degree[np.newaxis, :],
        rtol=0.0,
        atol=1e-15,
    )
    potential = np.linalg.pinv(laplacian_transpose.toarray().T) @ source_sink

    assert potential[0] < potential[-1]
    assert np.all(np.diff(potential) > 0)
    with pytest.raises(ValueError, match="contains isolated cells"):
        _random_walk_laplacian_transpose(csr_matrix(np.pad(adjacency, (0, 1))))


def test_marker_search_returns_an_explicit_artifact_without_feature_writes(
    datastore,
    pseudotime_scoring,
    few_features,
):
    from cytearc.storage.artifacts import (
        ArtifactRef,
        artifact_path,
        fingerprint_stored_strings,
    )

    feature_metadata_before = _metadata_values(datastore.RNA.feats)
    ref = datastore.trajectory.markers(
        pseudotime_scoring,
        features=few_features,
        min_cells=1,
    )
    result = datastore.trajectory.load_markers(ref)

    assert isinstance(ref, ArtifactRef)
    assert isinstance(result, PseudotimeMarkerResult)
    assert result.ref == ref
    assert result.pseudotime == pseudotime_scoring
    assert result.feature_selection == few_features
    # Each stored correlation is the Pearson r of the recorded normalization of
    # one selected feature with the pseudotime of the scored cells.
    parameters = datastore.artifacts.inspect(ref).parameters or {}
    tested = result.table.dropna(subset=["r_value"])
    assert len(tested) > 30
    np.testing.assert_allclose(
        tested.r_value.to_numpy(),
        _pearson_markers(
            datastore,
            pseudotime_scoring,
            tested.feature_index.to_numpy(),
            {
                "size_factor": parameters["size_factor"],
                "log_transform": parameters["normalization"]["log_transform"],
            },
        ),
        rtol=0.0,
        atol=1e-6,
    )
    _assert_metadata_unchanged(datastore.RNA.feats, feature_metadata_before)
    group = datastore.zw[artifact_path(ref)]
    assert set(group.array_keys()) == {
        "feature_ids",
        "feature_names",
        "p_value",
        "p_value_adjusted",
        "r_value",
    }
    inputs = datastore.artifacts.inspect(ref).inputs or {}
    assert inputs["ordered_feature_ids_fingerprint"] == fingerprint_stored_strings(
        group["feature_ids"]
    )
    assert inputs["ordered_feature_names_fingerprint"] == (
        fingerprint_stored_strings(group["feature_names"])
    )
    assert "p_value_adjusted" in result.table.columns


def test_marker_search_validates_refs_before_reading(
    datastore,
    pseudotime_scoring,
    detected_features,
    auto_filter_cells,
):
    with pytest.raises(TypeError, match="pseudotime must be an ArtifactRef"):
        datastore.trajectory.markers("ptime", features=detected_features)
    with pytest.raises(TypeError, match="features must be an ArtifactRef"):
        datastore.trajectory.markers(pseudotime_scoring, features="genes")
    with pytest.raises(ValueError, match="assay-scoped feature selection"):
        datastore.trajectory.markers(
            pseudotime_scoring,
            features=auto_filter_cells,
        )


def _scaled_by_totals(assay, counts):
    """A custom RNA normalizer that reads the totals ``normed`` hands it."""
    return 50.0 * counts / assay.scalar.reshape(-1, 1)


def _logged_scaled_by_totals(assay, counts):
    """``_scaled_by_totals`` that takes ``log1p`` of its own output."""
    return np.log1p(_scaled_by_totals(assay, counts))


def test_marker_search_logs_the_configured_normalizer(
    datastore,
    pseudotime_scoring,
    few_features,
    monkeypatch,
):
    from cytearc.assay.normalization import norm_clr, norm_dummy

    monkeypatch.setattr(datastore.RNA, "normMethod", _scaled_by_totals)
    ref = datastore.trajectory.markers(
        pseudotime_scoring,
        features=few_features,
        min_cells=1,
        log_transform=True,
    )

    status = datastore.artifacts.inspect(ref)
    # _scaled_by_totals is defined outside CyteArc, so its results record the
    # revision that identifies a normalizer by its code.
    assert status.revision == 2
    assert status.parameters["normalization"] == {
        "log_transform": True,
        "renormalize_subset": False,
    }
    # A flag of None means its default, as an omitted flag does.
    assert (
        datastore.trajectory.markers(
            pseudotime_scoring,
            features=few_features,
            min_cells=1,
            log_transform=True,
            renormalize_subset=None,
        )
        == ref
    )
    tested = datastore.trajectory.load_markers(ref).table.dropna(subset=["r_value"])
    np.testing.assert_allclose(
        tested.r_value.to_numpy(),
        _pearson_markers(
            datastore,
            pseudotime_scoring,
            tested.feature_index.to_numpy(),
            {"size_factor": 50.0, "log_transform": True},
        ),
        rtol=0.0,
        atol=1e-6,
    )
    # CLR values are already log ratios, and norm_dummy reads no totals.
    monkeypatch.setattr(datastore.RNA, "normMethod", norm_clr)
    with pytest.raises(ValueError, match="does not support log_transform"):
        datastore.trajectory.markers(
            pseudotime_scoring, features=few_features, log_transform=True
        )
    monkeypatch.setattr(datastore.RNA, "normMethod", norm_dummy)
    with pytest.raises(ValueError, match="does not support renormalize_subset"):
        datastore.trajectory.aggregation(
            pseudotime_scoring, features=few_features, renormalize_subset=True
        )


@pytest.mark.slow
def test_aggregation_logs_the_configured_normalizer(
    datastore,
    pseudotime_scoring,
    few_features,
    monkeypatch,
):
    arguments = {
        "features": few_features,
        "n_clusters": 3,
        "n_neighbors": 5,
        "window_size": 50,
        "chunk_size": 10,
    }
    monkeypatch.setattr(datastore.RNA, "normMethod", _scaled_by_totals)
    logged = datastore.trajectory.aggregation(
        pseudotime_scoring, log_transform=True, **arguments
    )
    # Normalizers defined outside CyteArc record the code-digest revision.
    assert datastore.artifacts.inspect(logged).revision == 2

    # The same values from a normalizer that logs its own output.
    monkeypatch.setattr(datastore.RNA, "normMethod", _logged_scaled_by_totals)
    direct = datastore.trajectory.aggregation(pseudotime_scoring, **arguments)
    assert datastore.artifacts.inspect(direct).revision == 2
    np.testing.assert_allclose(
        np.asarray(datastore.trajectory.load_aggregation(logged).data),
        np.asarray(datastore.trajectory.load_aggregation(direct).data),
        rtol=1e-6,
        atol=1e-6,
    )


def test_marker_loader_rejects_each_tampered_record(
    datastore,
    pseudotime_markers,
    detected_features,
):
    from cytearc.storage.artifacts import artifact_group

    with pytest.raises(TypeError, match="ref must be an ArtifactRef"):
        datastore.trajectory.load_markers("markers")
    with pytest.raises(ValueError, match="assay-scoped pseudotime_markers"):
        datastore.trajectory.load_markers(detected_features)
    group = artifact_group(datastore.zw, pseudotime_markers)
    original = group.attrs["provenance"]
    unrelated = detected_features.to_dict()
    try:
        for section, key, value, message in (
            (
                "parameters",
                "min_cells",
                0,
                "parameters do not match the required contract",
            ),
            (
                "inputs",
                "ordered_feature_ids_fingerprint",
                1,
                "identities are malformed",
            ),
            (
                "inputs",
                "ordered_feature_ids_fingerprint",
                "x",
                "ID identity has changed",
            ),
            ("inputs", "dataset_fingerprint", "x", "dataset identity has changed"),
            ("inputs", "cell_selection", unrelated, "cell selection does not match"),
        ):
            provenance = dict(original)
            provenance[section] = dict(provenance[section]) | {key: value}
            group.attrs["provenance"] = provenance
            with pytest.raises(ValueError, match=message):
                datastore.trajectory.load_markers(pseudotime_markers)
    finally:
        group.attrs["provenance"] = original
    assert datastore.trajectory.load_markers(pseudotime_markers).ref == (
        pseudotime_markers
    )


def test_trajectory_feature_selection_indices_are_read_blockwise(
    datastore,
    detected_features,
    monkeypatch,
):
    import cytearc.datastore._operations.trajectory as trajectory_operations
    from cytearc.storage.artifacts import artifact_path

    selection_values_path = f"{artifact_path(detected_features)}/values"
    expected = np.flatnonzero(
        np.asarray(datastore.zw[selection_values_path][:], dtype=bool)
    )
    original_getitem = zarr.Array.__getitem__

    def reject_full_selection_read(array, key):
        if array.path == selection_values_path and (
            key is Ellipsis
            or (
                isinstance(key, slice)
                and key.start is None
                and key.stop is None
                and key.step is None
            )
        ):
            raise AssertionError("trajectory materialized the full feature selection")
        return original_getitem(array, key)

    monkeypatch.setattr(zarr.Array, "__getitem__", reject_full_selection_read)

    resolved, indices = trajectory_operations._resolve_feature_indices(
        datastore,
        datastore.RNA,
        detected_features,
    )

    assert resolved == detected_features
    assert indices.dtype == np.int64
    assert 0 < len(expected) < datastore.RNA.feats.N
    np.testing.assert_array_equal(indices, expected)


def test_marker_loader_uses_frozen_feature_names(
    datastore,
    pseudotime_markers,
):
    expected = datastore.trajectory.load_markers(
        pseudotime_markers
    ).table.feature_name.to_numpy(copy=True)
    live_names = datastore.RNA.feats._get_array("names")
    original = np.asarray(live_names[:]).copy()
    try:
        renamed = original.astype(str)
        renamed[:] = [f"renamed_{index}" for index in range(len(renamed))]
        live_names[:] = renamed
        loaded = datastore.trajectory.load_markers(pseudotime_markers)
        np.testing.assert_array_equal(
            loaded.table.feature_name.to_numpy(),
            expected,
        )
    finally:
        live_names[:] = original


def test_trajectory_loaders_do_not_depend_on_live_normalization_settings(
    datastore,
    pseudotime_markers,
    pseudotime_aggregation,
    pseudotime_scoring,
    detected_features,
    monkeypatch,
):
    marker_before = datastore.trajectory.load_markers(pseudotime_markers)
    aggregation_before = datastore.trajectory.load_aggregation(pseudotime_aggregation)
    monkeypatch.setattr(datastore.RNA, "sf", float(datastore.RNA.sf) * 2.0)

    marker_after = datastore.trajectory.load_markers(pseudotime_markers)
    aggregation_after = datastore.trajectory.load_aggregation(pseudotime_aggregation)
    pd.testing.assert_frame_equal(marker_after.table, marker_before.table)
    np.testing.assert_array_equal(
        aggregation_after.feature_indices,
        aggregation_before.feature_indices,
    )
    np.testing.assert_array_equal(
        aggregation_after.feature_clusters,
        aggregation_before.feature_clusters,
    )

    # A new search with the arguments of the stored one does follow the live
    # size factor.
    rerun = datastore.trajectory.markers(
        pseudotime_scoring,
        features=detected_features,
    )
    assert rerun != pseudotime_markers
    size_factors = [
        (datastore.artifacts.inspect(ref).parameters or {})["size_factor"]
        for ref in (pseudotime_markers, rerun)
    ]
    assert size_factors[1] == 2.0 * size_factors[0]


def test_aggregation_provenance_records_resolved_ann_defaults(
    datastore,
    pseudotime_aggregation,
    few_features,
):
    status = datastore.artifacts.inspect(pseudotime_aggregation)
    expected = dict(AGGREGATION_ANN_DEFAULTS)
    expected["dim"] = 10
    assert (status.parameters or {})["ann_params"] == expected

    result = datastore.trajectory.load_aggregation(pseudotime_aggregation)
    arguments = {
        "pseudotime": result.pseudotime,
        "features": few_features,
        "n_clusters": 3,
        "window_size": 50,
        "chunk_size": 10,
    }
    default = datastore.trajectory.aggregation(**arguments)
    overridden = datastore.trajectory.aggregation(
        **arguments, ann_params={"random_seed": 445}
    )
    assert overridden != default
    default_parameters = datastore.artifacts.inspect(default).parameters or {}
    overridden_parameters = datastore.artifacts.inspect(overridden).parameters or {}
    assert default_parameters["ann_params"] == expected
    assert overridden_parameters["ann_params"] == {
        **expected,
        "random_seed": 445,
    }
    # Only the requested ANN seed differs between the two records.
    for parameters in (default_parameters, overridden_parameters):
        parameters.pop("ann_params")
    assert overridden_parameters == default_parameters
    assert (
        datastore.trajectory.aggregation(**arguments, ann_params={"random_seed": 445})
        == overridden
    )


def test_aggregation_takes_n_neighbors_and_saves_it_under_the_released_key(
    datastore,
    pseudotime_aggregation,
    few_features,
):
    result = datastore.trajectory.load_aggregation(pseudotime_aggregation)
    arguments = {
        "pseudotime": result.pseudotime,
        "features": few_features,
        "n_clusters": 3,
        "window_size": 50,
        "chunk_size": 10,
    }
    four = datastore.trajectory.aggregation(**arguments, n_neighbors=4)

    # 1.3.0 saved the argument as n_neighbours, and the record keeps that key
    # so that the results it computed are still reused.
    parameters = datastore.artifacts.inspect(four).parameters or {}
    assert parameters["n_neighbours"] == 4
    assert "n_neighbors" not in parameters
    assert datastore.trajectory.aggregation(**arguments, n_neighbors=4) == four
    assert datastore.trajectory.aggregation(**arguments, n_neighbors=3) != four

    with pytest.raises(TypeError, match="unexpected keyword argument 'n_neighbours'"):
        datastore.trajectory.aggregation(**arguments, n_neighbours=4)
    with pytest.raises(ValueError, match="^n_neighbors must be at least 1$"):
        datastore.trajectory.aggregation(**arguments, n_neighbors=0)
    with pytest.raises(TypeError, match="^n_neighbors must be an integer$"):
        datastore.trajectory.aggregation(**arguments, n_neighbors=4.0)


@pytest.mark.parametrize(
    ("batch_size", "error_type"),
    [
        (True, TypeError),
        (False, TypeError),
        (0, ValueError),
        (-1, ValueError),
        (1.5, TypeError),
    ],
)
def test_trajectory_feature_batch_sizes_are_strictly_positive_integers(
    datastore,
    pseudotime_scoring,
    detected_features,
    batch_size,
    error_type,
):
    with pytest.raises(error_type, match="gene_batch_size"):
        datastore.trajectory.markers(
            pseudotime_scoring,
            features=detected_features,
            gene_batch_size=batch_size,
        )
    with pytest.raises(error_type, match="batch_size"):
        datastore.trajectory.aggregation(
            pseudotime_scoring,
            features=detected_features,
            batch_size=batch_size,
        )


def test_aggregation_rejects_nonmapping_ann_parameters(
    datastore,
    pseudotime_scoring,
    detected_features,
):
    with pytest.raises(TypeError, match="ann_params must be a mapping"):
        datastore.trajectory.aggregation(
            pseudotime_scoring,
            features=detected_features,
            ann_params=[("M", 20)],  # type: ignore[arg-type]
        )


def test_trajectory_operations_require_boolean_cache_invalidation(
    datastore,
    connectivity_graph,
    pseudotime_scoring,
    detected_features,
):
    with pytest.raises(TypeError, match="invalidate_cache"):
        datastore.trajectory.pseudotime(
            connectivity_graph,
            invalidate_cache=1,
        )
    with pytest.raises(TypeError, match="invalidate_cache"):
        datastore.trajectory.fate(
            pseudotime_scoring,
            pseudotime_scoring,
            sinks=[1],
            invalidate_cache=1,
        )
    with pytest.raises(TypeError, match="invalidate_cache"):
        datastore.trajectory.markers(
            pseudotime_scoring,
            features=detected_features,
            invalidate_cache=1,
        )
    with pytest.raises(TypeError, match="invalidate_cache"):
        datastore.trajectory.aggregation(
            pseudotime_scoring,
            features=detected_features,
            invalidate_cache=1,
        )


def test_raw_source_sink_scoring_refuses_read_only_store_before_snapshot(
    datastore,
    connectivity_graph,
):
    from scipy.sparse.csgraph import connected_components

    from cytearc.datastore.datastore import DataStore

    graph = datastore.graph.load(
        connectivity_graph,
        symmetric=True,
        upper_only=False,
    )
    _, components = connected_components(graph, directed=False)
    retained = np.flatnonzero(components == int(np.argmax(np.bincount(components))))
    source_sink = np.zeros(graph.shape[0], dtype=np.float64)
    source_sink[retained[1]] = -1.0
    source_sink[retained[-2]] = 1.0
    snapshots = datastore.artifacts.list(kind="metadata_snapshot", scope="datastore")

    read_only = DataStore(datastore.zarr_loc, default_assay="RNA", zarr_mode="r")
    with pytest.raises(PermissionError, match="zarr_mode='r\\+'"):
        read_only.trajectory.pseudotime(
            connectivity_graph,
            ss_vec=source_sink,
        )
    assert (
        datastore.artifacts.list(kind="metadata_snapshot", scope="datastore")
        == snapshots
    )


def test_trajectory_feature_producers_refuse_read_only_stores_before_computing(
    datastore,
    pseudotime_scoring,
    detected_features,
    monkeypatch,
):
    from cytearc.datastore.datastore import DataStore

    read_only = DataStore(datastore.zarr_loc, default_assay="RNA", zarr_mode="r")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a read-only store must refuse before computing")

    monkeypatch.setattr(
        type(read_only.get_assay("RNA")),
        "_aggregate_ordering_profiles",
        forbidden,
    )
    monkeypatch.setattr(
        "cytearc.features.markers.find_markers_by_regression", forbidden
    )
    with pytest.raises(PermissionError, match="run_pseudotime_aggregation"):
        read_only.trajectory.aggregation(
            pseudotime_scoring,
            features=detected_features,
            window_size=17,
            n_clusters=3,
        )
    with pytest.raises(PermissionError, match="run_pseudotime_marker_search"):
        read_only.trajectory.markers(
            pseudotime_scoring,
            features=detected_features,
            min_cells=7,
        )


def test_marker_identity_change_during_computation_discards_the_artifact(
    datastore,
    pseudotime_scoring,
    few_features,
    monkeypatch,
):
    import cytearc.features.markers as marker_algorithms

    before = set(
        datastore.artifacts.list(
            kind="pseudotime_markers",
            from_assay="RNA",
        )
    )
    live_names = datastore.RNA.feats._get_array("names")
    original_names = np.asarray(live_names[:]).copy()
    renamed = original_names.astype(str)
    renamed[:] = [f"changed_{index}" for index in range(len(renamed))]
    original_search = marker_algorithms.find_markers_by_regression

    def mutate_after_computation(*args, **kwargs):
        result = original_search(*args, **kwargs)
        live_names[:] = renamed
        return result

    monkeypatch.setattr(
        marker_algorithms,
        "find_markers_by_regression",
        mutate_after_computation,
    )
    try:
        with pytest.raises(ValueError, match="identities changed"):
            datastore.trajectory.markers(
                pseudotime_scoring,
                features=few_features,
                min_cells=1,
                invalidate_cache=True,
            )
    finally:
        live_names[:] = original_names

    created = (
        set(
            datastore.artifacts.list(
                kind="pseudotime_markers",
                from_assay="RNA",
            )
        )
        - before
    )
    # A failed write deletes its incomplete slot.
    assert created == set()


def test_marker_normalization_change_during_computation_discards_the_artifact(
    datastore,
    pseudotime_scoring,
    few_features,
    monkeypatch,
):
    import cytearc.features.markers as marker_algorithms

    before = set(
        datastore.artifacts.list(
            kind="pseudotime_markers",
            from_assay="RNA",
        )
    )
    original_size_factor = datastore.RNA.sf
    assert original_size_factor is not None
    original_search = marker_algorithms.find_markers_by_regression

    def mutate_after_computation(*args, **kwargs):
        result = original_search(*args, **kwargs)
        datastore.RNA.sf = original_size_factor * 2
        return result

    monkeypatch.setattr(
        marker_algorithms,
        "find_markers_by_regression",
        mutate_after_computation,
    )
    try:
        with pytest.raises(ValueError, match="normalization settings changed"):
            datastore.trajectory.markers(
                pseudotime_scoring,
                features=few_features,
                min_cells=1,
                invalidate_cache=True,
            )
    finally:
        datastore.RNA.sf = original_size_factor

    created = (
        set(
            datastore.artifacts.list(
                kind="pseudotime_markers",
                from_assay="RNA",
            )
        )
        - before
    )
    # A failed write deletes its incomplete slot.
    assert created == set()


def test_aggregation_normalization_change_discards_the_artifact(
    datastore,
    pseudotime_scoring,
    few_features,
    monkeypatch,
):
    before = set(
        datastore.artifacts.list(
            kind="pseudotime_aggregation",
            from_assay="RNA",
        )
    )
    original_size_factor = datastore.RNA.sf
    assert original_size_factor is not None
    original_writer = datastore.RNA._write_aggregated_ordering_group

    def mutate_after_aggregation(*args, **kwargs):
        result = original_writer(*args, **kwargs)
        datastore.RNA.sf = original_size_factor * 2
        return result

    monkeypatch.setattr(
        datastore.RNA,
        "_write_aggregated_ordering_group",
        mutate_after_aggregation,
    )
    try:
        with pytest.raises(ValueError, match="normalization settings changed"):
            datastore.trajectory.aggregation(
                pseudotime_scoring,
                features=few_features,
                window_size=10,
                chunk_size=5,
                n_neighbors=11,
                n_clusters=3,
                invalidate_cache=True,
            )
    finally:
        datastore.RNA.sf = original_size_factor

    created = (
        set(
            datastore.artifacts.list(
                kind="pseudotime_aggregation",
                from_assay="RNA",
            )
        )
        - before
    )
    # A failed write deletes its incomplete slot.
    assert created == set()


def test_incomplete_pseudotime_marker_artifact_is_recomputed(
    datastore,
    pseudotime_scoring,
    few_features,
    monkeypatch,
):
    import cytearc.features.markers as marker_algorithms
    from cytearc.storage.artifacts import artifact_path

    arguments = {
        "pseudotime": pseudotime_scoring,
        "features": few_features,
        "min_cells": 1,
        "gene_batch_size": 8,
    }
    first = datastore.trajectory.markers(**arguments)
    old_artifact = datastore.zw[artifact_path(first)]
    del old_artifact["p_value_adjusted"]
    feature_metadata_before = _metadata_values(datastore.RNA.feats)

    original = marker_algorithms.find_markers_by_regression
    calls = 0

    def tracked_marker_search(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        marker_algorithms,
        "find_markers_by_regression",
        tracked_marker_search,
    )
    second = datastore.trajectory.markers(**arguments)

    assert calls == 1
    assert second != first
    assert "p_value_adjusted" not in old_artifact
    assert "p_value_adjusted" in datastore.zw[artifact_path(second)]
    _assert_metadata_unchanged(datastore.RNA.feats, feature_metadata_before)


def test_regressor_validation_points_to_component_validity_key():
    with pytest.raises(ValueError, match="ptime__valid"):
        _validated_pseudotime_regressor(
            np.array([0.0, np.nan]),
            2,
            "ptime",
            "I",
            has_validity_column=True,
        )


@pytest.mark.parametrize(
    ("values", "expected_size", "error_type", "match"),
    [
        (["early", "late"], 2, TypeError, "must be numeric"),
        (np.array([[0.0, 1.0]]), 1, ValueError, "one-dimensional"),
        (np.array([0.0]), 2, ValueError, "selects 2 cells"),
        (np.array([0.0, np.inf]), 2, ValueError, "non-finite"),
        (np.array([0.5, 0.5]), 2, ValueError, "distinct values"),
    ],
)
def test_regressor_validation_rejects_invalid_columns(
    values,
    expected_size,
    error_type,
    match,
):
    with pytest.raises(error_type, match=match):
        _validated_pseudotime_regressor(
            values,
            expected_size,
            "ptime",
            "I",
            has_validity_column=False,
        )


class _ShapeOnlyArray:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape


def test_knn_clustering_rejects_infeasible_parameters():
    with pytest.raises(ValueError, match="two-dimensional"):
        knn_clustering(_ShapeOnlyArray((3,)), 1, 1, 1)
    with pytest.raises(ValueError, match="At least two"):
        knn_clustering(_ShapeOnlyArray((1, 3)), 1, 1, 1)
    with pytest.raises(TypeError, match="n_neighbors"):
        knn_clustering(_ShapeOnlyArray((3, 3)), True, 2, 1)
    with pytest.raises(ValueError, match="n_neighbors"):
        knn_clustering(_ShapeOnlyArray((3, 3)), np.int64(3), np.int32(2), 1)
    with pytest.raises(ValueError, match="n_clusters"):
        knn_clustering(_ShapeOnlyArray((3, 3)), np.int32(1), np.int64(4), 1)
    with pytest.raises(TypeError, match="n_clusters"):
        knn_clustering(_ShapeOnlyArray((3, 3)), 1, np.bool_(True), 1)
    with pytest.raises(ValueError, match="effective bin count"):
        knn_clustering(
            _ShapeOnlyArray((3, 3)),
            1,
            2,
            1,
            {"dim": 2},
        )
    with pytest.raises(ValueError, match="smaller than the feature count"):
        knn_clustering(
            _ShapeOnlyArray((3, 3)),
            1,
            2,
            1,
            {"max_elements": 2},
        )


def test_feature_cluster_scatter_honors_custom_unassigned_value():
    values = _scatter_feature_clusters(
        5,
        np.array([1, 3]),
        np.array([1, 2]),
        0,
    )

    np.testing.assert_array_equal(values, [0, 1, 0, 2, 0])
    with pytest.raises(ValueError, match="misaligned"):
        _scatter_feature_clusters(3, np.array([0, 1]), np.array([1]), 0)
    with pytest.raises(ValueError, match="conflicts"):
        _scatter_feature_clusters(3, np.array([0]), np.array([1]), 1)


def test_aggregate_feature_profiles_orders_filters_and_bins():
    values = np.array(
        [
            [3.0, 5.0, 0.01],
            [1.0, 5.0, 0.01],
            [2.0, 5.0, 0.01],
            [0.0, 5.0, 0.01],
        ],
        dtype=float,
    )
    ordering = np.array([3, 1, 2, 0])
    feature_indices = np.array([10, 11, 12])

    binned, valid = aggregate_feature_profiles(
        values,
        ordering,
        feature_indices,
        min_expression=0.1,
        window_size=2,
        n_bins=2,
        smooth=False,
        z_scale=False,
    )

    np.testing.assert_array_equal(valid, [True, False, False])
    assert binned.shape == (3, 2)
    np.testing.assert_allclose(binned[0], [0.5, 2.5])
    np.testing.assert_array_equal(binned[1], 0.0)
    np.testing.assert_array_equal(binned[2], 0.0)

    z_binned, z_valid = aggregate_feature_profiles(
        values,
        ordering,
        feature_indices,
        min_expression=0.1,
        window_size=2,
        n_bins=2,
        smooth=False,
        z_scale=True,
    )
    np.testing.assert_array_equal(z_valid, valid)
    # The ordered profile 0, 1, 2, 3 has mean 1.5 and population standard
    # deviation sqrt(1.25), so its two bins average to -+1 / sqrt(1.25).
    np.testing.assert_allclose(z_binned[0], np.array([-1.0, 1.0]) / np.sqrt(1.25))
    np.testing.assert_array_equal(z_binned[1:], 0.0)

    with pytest.raises(ValueError, match="non-finite values"):
        aggregate_feature_profiles(
            np.array([[1.0, np.nan], [2.0, 3.0]]),
            np.array([0, 1]),
            np.array([0, 1]),
            min_expression=0.0,
            window_size=1,
            n_bins=1,
            smooth=False,
            z_scale=False,
        )


def test_aggregate_feature_profiles_rejects_profiles_that_lose_finiteness():
    import warnings

    options = {"min_expression": 0.0, "z_scale": False}
    # Finite values near the float64 limit overflow the rolling-window sums.
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="Smoothed feature profiles contain non-finite"),
    ):
        aggregate_feature_profiles(
            np.array([[1e308, 0.0], [1e308, 1.0], [1e308, 2.0]]),
            np.arange(3),
            np.array([4, 7]),
            window_size=2,
            n_bins=1,
            smooth=True,
            **options,
        )
    # More bins than ordered cells leave an empty bin without a mean.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(
            ValueError, match="Binned feature profiles contain non-finite"
        ):
            aggregate_feature_profiles(
                np.array([[1.0, 2.0], [3.0, 5.0]]),
                np.arange(2),
                np.array([4, 7]),
                window_size=1,
                n_bins=3,
                smooth=False,
                **options,
            )


def _ring_profiles(n_features: int) -> np.ndarray:
    theta = 2 * np.pi * np.arange(n_features) / n_features
    return 10.0 * np.column_stack([np.cos(theta), np.sin(theta)])


def test_knn_clustering_reports_disconnected_feature_graph():
    from cytearc.matrix import ChunkedArray

    rng = np.random.default_rng(3)
    centers = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]])
    profiles = np.repeat(centers, 5, axis=0) + rng.normal(scale=0.01, size=(15, 2))

    with pytest.raises(ValueError, match="3 disconnected components") as error:
        knn_clustering(ChunkedArray.from_numpy(profiles), 2, 2, 1)
    assert "n_neighbors" in str(error.value)

    labels = knn_clustering(ChunkedArray.from_numpy(profiles), 2, 3, 1)
    assert sorted(np.unique(labels)) == [1, 2, 3]
    for group in range(3):
        assert np.unique(labels[5 * group : 5 * group + 5]).size == 1


def test_knn_clustering_never_exceeds_requested_clusters_on_tied_heights():
    from cytearc.matrix import ChunkedArray

    n_features = 12
    # The two nearest neighbors of each ring point are its ring neighbors,
    # so the feature graph is a cycle whose Paris merges tie in height.
    profiles = ChunkedArray.from_numpy(_ring_profiles(n_features))
    for n_clusters in range(1, n_features + 1):
        labels = knn_clustering(profiles, 2, n_clusters, 1)
        assert sorted(np.unique(labels)) == list(range(1, n_clusters + 1))


def _aggregation_artifacts(datastore) -> set:
    return set(
        datastore.artifacts.list(
            kind="pseudotime_aggregation",
            from_assay="RNA",
        )
    )


def test_aggregation_rejects_infeasible_clustering_before_starting_artifact(
    datastore,
    pseudotime_scoring,
    few_features,
):
    before = _aggregation_artifacts(datastore)

    # With one neighbor, mutual nearest pairs split the feature graph into
    # far more components than two modules can hold.
    with pytest.raises(ValueError, match="disconnected components"):
        datastore.trajectory.aggregation(
            pseudotime_scoring,
            features=few_features,
            window_size=10,
            chunk_size=5,
            n_neighbors=1,
            n_clusters=2,
            invalidate_cache=True,
        )

    assert _aggregation_artifacts(datastore) == before


def test_aggregation_min_exp_filter_failure_creates_no_artifact(
    datastore,
    pseudotime_scoring,
    few_features,
):
    before = _aggregation_artifacts(datastore)

    with pytest.raises(ValueError, match="min_exp") as error:
        datastore.trajectory.aggregation(
            pseudotime_scoring,
            features=few_features,
            min_exp=1e12,
            window_size=10,
            chunk_size=5,
            n_neighbors=3,
            n_clusters=2,
            invalidate_cache=True,
        )

    assert "Only 0 of" in str(error.value)
    assert _aggregation_artifacts(datastore) == before


@pytest.mark.slow
def test_aggregation_and_marker_search_complete_within_a_small_budget(tmp_path):
    from scipy.sparse import random as sparse_random

    from cytearc.datastore.datastore import DataStore
    from cytearc.writers import SparseToZarr
    from tests.fixtures_datastore import build_neighborhood_graph

    # All 2000 features form one 32 MB read group, so a 128 MiB budget holds
    # the reads but not every normalized value together with its scratch,
    # about 256 MB.
    n_cells, n_features = 4_000, 2_000
    budget = 128 * 1024**2
    rng = np.random.default_rng(7)
    counts = sparse_random(
        n_cells,
        n_features,
        density=0.05,
        format="lil",
        random_state=rng,
        data_rvs=lambda size: rng.integers(1, 20, size=size),
    ).astype(np.uint32)
    ordering = np.linspace(0.0, 1.0, n_cells)
    counts[:, :20] = rng.poisson(5 * ordering[:, None], size=(n_cells, 20))
    path = str(tmp_path / "budget.zarr")
    SparseToZarr(
        counts.tocsr(),
        path,
        cell_ids=[f"c{i}" for i in range(n_cells)],
        feature_ids=[f"g{i}" for i in range(n_features)],
        nthreads=2,
    ).dump()
    del counts
    datastore = DataStore(
        path,
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=2,
        mem_budget=budget,
    )
    cells = datastore.snapshot_cell_selection("I")
    features = datastore.features.universe(from_assay="RNA")
    graph = build_neighborhood_graph(
        datastore,
        cell_selection=cells,
        features=datastore.features.hvgs(
            cells,
            top_n=100,
            show_plot=False,
            min_cells=5,
            max_cells=np.inf,
        ),
        dims=5,
        k=10,
        n_centroids=10,
        local_cache=False,
    )
    source_sink = np.zeros(n_cells)
    source_sink[[0, -1]] = [-1.0, 1.0]
    pseudotime = datastore.trajectory.pseudotime(graph, ss_vec=source_sink)
    # Every cell is scored, so both analyses read the counts of all 4000 cells.
    assert datastore.trajectory.load_pseudotime(pseudotime).valid.all()

    for run in (
        lambda: datastore.trajectory.aggregation(
            pseudotime,
            features=features,
            n_clusters=5,
        ),
        lambda: datastore.trajectory.markers(
            pseudotime,
            features=features,
        ),
    ):
        ref = run()
        assert datastore.artifacts.inspect(ref).complete


# Solver bounds, progress, and graph components of pseudotime scoring


def _clusters_on_a_tree(n_cells: int, n_clusters: int, seed: int = 0) -> csr_matrix:
    """Return a neighbor graph of Gaussian clusters, each one step from another.

    Neighboring clusters share few edges, so the smallest singular values of
    the graph's Laplacian lie close together, as on a large atlas.
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    centers = np.zeros((n_clusters, 10))
    for index in range(1, n_clusters):
        step = rng.normal(size=10)
        centers[index] = centers[rng.integers(0, index)] + 5.0 * step / np.linalg.norm(
            step
        )
    points = centers[rng.integers(0, n_clusters, n_cells)] + rng.normal(
        size=(n_cells, 10)
    )
    distances, indices = cKDTree(points).query(points, k=11)
    weights = np.exp(-((distances[:, 1:] / np.median(distances[:, -1])) ** 2))
    rows = np.repeat(np.arange(n_cells), 10)
    graph = csr_matrix(
        (weights.ravel(), (rows, indices[:, 1:].ravel())), shape=(n_cells, n_cells)
    )
    return graph.maximum(graph.T).tocsr()


def _first_to_last_source_sink(n_cells: int) -> np.ndarray:
    source_sink = np.zeros(n_cells)
    source_sink[[0, -1]] = [-1.0, 1.0]
    return source_sink


def _count_svds_products(monkeypatch) -> list[int]:
    """Count the Laplacian products that svds takes, apart from the solve's count.

    Each call appends one count to the returned list.
    """
    from scipy.sparse.linalg import LinearOperator, aslinearoperator

    import cytearc.trajectory.pseudotime as pseudotime_module

    original = pseudotime_module.svds
    counts: list[int] = []

    def counting_svds(operator, **arguments):
        inner = aslinearoperator(operator)
        counts.append(0)

        def matvec(values):
            counts[-1] += 1
            return inner.matvec(values)

        def rmatvec(values):
            counts[-1] += 1
            return inner.rmatvec(values)

        counted = LinearOperator(
            shape=inner.shape,
            matvec=matvec,
            rmatvec=rmatvec,
            matmat=inner.matmat,
            dtype=inner.dtype,
        )
        return original(counted, **arguments)

    monkeypatch.setattr(pseudotime_module, "svds", counting_svds)
    return counts


def test_a_bounded_solve_stops_with_the_counts_of_the_solve(monkeypatch):
    import re

    from scipy.sparse.linalg import ArpackNoConvergence

    from cytearc.trajectory import PseudotimeConvergenceError

    laplacian_transpose = _random_walk_laplacian_transpose(_clusters_on_a_tree(600, 60))
    counts = _count_svds_products(monkeypatch)

    with pytest.raises(PseudotimeConvergenceError) as raised:
        _truncated_pba_potential(
            laplacian_transpose,
            30,
            4444,
            _first_to_last_source_sink(600),
            max_iterations=1,
        )

    error = raised.value
    cause = error.__cause__
    assert isinstance(error, RuntimeError)
    assert isinstance(cause, ArpackNoConvergence)
    # ARPACK counts its first basis as an iteration, so it reports one more
    # iteration than the restarts it ran, and the modes that converged.
    iterations, converged = map(
        int, re.search(r"\((\d+) iterations, (\d+)/30 ", str(cause)).groups()
    )
    assert (error.n_cells, error.n_modes, error.max_iterations) == (600, 30, 1)
    assert error.solver_tol is None
    assert error.restarts == iterations - 1 == 1
    assert error.converged_modes == converged == len(cause.eigenvalues)
    assert error.operator_products == counts[0]
    # A first basis of 2 * 30 + 1 vectors takes 62 products of L Lᵀ and the
    # restart at most 31 more, each two products of the Laplacian.
    assert 0 < error.operator_products <= 2 * (62 + 31)
    assert "max_iterations=1" in str(error)
    assert "fewer modes do not help" in str(error)


def test_counting_the_products_leaves_the_potential_unchanged(monkeypatch):
    import cytearc.trajectory.pseudotime as pseudotime_module

    laplacian_transpose = _random_walk_laplacian_transpose(
        _clusters_on_a_tree(300, 10, seed=1)
    )
    source_sink = _first_to_last_source_sink(300)
    counted = _truncated_pba_potential(laplacian_transpose, 10, 7, source_sink)
    # A bound that the solve meets changes nothing it computes.
    bounded = _truncated_pba_potential(
        laplacian_transpose, 10, 7, source_sink, max_iterations=10_000
    )

    # Without the counting operator, svds reads the matrix as it did before.
    monkeypatch.setattr(
        pseudotime_module, "_counted_operator", lambda matrix, _progress: matrix
    )
    plain = _truncated_pba_potential(laplacian_transpose, 10, 7, source_sink)

    np.testing.assert_array_equal(counted, plain)
    np.testing.assert_array_equal(bounded, plain)


def test_the_solver_bounds_reach_svds(monkeypatch):
    import cytearc.trajectory.pseudotime as pseudotime_module

    received: list[dict] = []
    original = pseudotime_module.svds

    def recording_svds(operator, **arguments):
        received.append({name: arguments[name] for name in ("tol", "maxiter")})
        return original(operator, **arguments)

    monkeypatch.setattr(pseudotime_module, "svds", recording_svds)
    laplacian_transpose = _random_walk_laplacian_transpose(
        _clusters_on_a_tree(300, 10, seed=1)
    )
    source_sink = _first_to_last_source_sink(300)
    default = _truncated_pba_potential(laplacian_transpose, 10, 7, source_sink)
    bounded = _truncated_pba_potential(
        laplacian_transpose, 10, 7, source_sink, solver_tol=1e-8, max_iterations=500
    )

    # SciPy's own defaults: machine precision and ten restarts per cell.
    assert received == [
        {"tol": 0.0, "maxiter": None},
        {"tol": 1e-8, "maxiter": 500},
    ]
    np.testing.assert_allclose(bounded, default, rtol=0.0, atol=1e-9)


def test_a_result_without_the_null_mode_names_a_set_solver_tolerance(monkeypatch):
    import cytearc.trajectory.pseudotime as pseudotime_module

    laplacian_transpose = _random_walk_laplacian_transpose(_clusters_on_a_tree(40, 4))
    original = pseudotime_module.svds

    def search_that_missed_the_null_mode(operator, **arguments):
        # Drop the smallest singular value, as a loose search can.
        left, values, right = original(operator, **arguments)
        order = np.argsort(values)[1:]
        return left[:, order], values[order], right[order]

    monkeypatch.setattr(pseudotime_module, "svds", search_that_missed_the_null_mode)
    source_sink = _first_to_last_source_sink(40)

    with pytest.raises(ValueError, match="null mode$"):
        _truncated_pba_potential(laplacian_transpose, 5, 7, source_sink)
    with pytest.raises(
        ValueError, match="can miss at solver_tol=0.01; lower solver_tol or leave"
    ):
        _truncated_pba_potential(
            laplacian_transpose, 5, 7, source_sink, solver_tol=0.01
        )


@pytest.mark.parametrize(
    ("bounds", "error", "message"),
    [
        ({"solver_tol": 0.0}, ValueError, "greater than 0 and less than 1"),
        ({"solver_tol": 1.0}, ValueError, "greater than 0 and less than 1"),
        ({"solver_tol": float("nan")}, ValueError, "solver_tol must be finite"),
        ({"solver_tol": "1e-4"}, TypeError, "solver_tol must be a real number"),
        ({"solver_tol": True}, TypeError, "solver_tol must be a real number"),
        ({"max_iterations": 0}, ValueError, "max_iterations must be at least 1"),
        ({"max_iterations": 2.0}, TypeError, "max_iterations must be an integer"),
        ({"max_iterations": True}, TypeError, "max_iterations must be an integer"),
    ],
)
def test_solver_bounds_are_checked_before_the_solve(
    bounds, error, message, monkeypatch
):
    import cytearc.trajectory.pseudotime as pseudotime_module

    def unreachable(*_args, **_kwargs):
        raise AssertionError("the bounds were not checked before the solve")

    monkeypatch.setattr(pseudotime_module, "svds", unreachable)
    laplacian_transpose = _random_walk_laplacian_transpose(_clusters_on_a_tree(40, 4))

    with pytest.raises(error, match=message):
        _truncated_pba_potential(
            laplacian_transpose, 3, 7, _first_to_last_source_sink(40), **bounds
        )


@pytest.mark.parametrize("max_iterations", [200, None], ids=["bounded", "unbounded"])
def test_the_solve_reports_progress_to_observers_and_the_log(
    max_iterations, monkeypatch
):
    from loguru import logger

    import cytearc.trajectory.pseudotime as pseudotime_module
    from cytearc.utils.logging import configure_output, progress_enabled
    from cytearc.utils.progress import observe_progress

    # Report every product, so a quick solve reports as a long one does.
    monkeypatch.setattr(pseudotime_module, "_PROGRESS_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(pseudotime_module, "_BAR_INTERVAL_SECONDS", 0.0)
    counts = _count_svds_products(monkeypatch)
    laplacian_transpose = _random_walk_laplacian_transpose(_clusters_on_a_tree(40, 4))
    observed: list[tuple[str | None, int, int | None]] = []
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="INFO",
        filter=lambda record: record["message"].startswith("Pseudotime scoring: "),
    )
    displayed = progress_enabled()
    # Observers receive the bar whether or not it is displayed.
    configure_output(progress=False)
    try:
        with observe_progress(
            lambda _bar, description, completed, total: observed.append(
                (description, completed, total)
            )
        ):
            _truncated_pba_potential(
                laplacian_transpose,
                5,
                7,
                _first_to_last_source_sink(40),
                max_iterations=max_iterations,
            )
    finally:
        configure_output(progress=displayed)
        logger.remove(sink)

    products = counts[0]
    assert products > 0
    assert {description for description, _, _ in observed} == {"Pseudotime solve"}
    completed = [done for _, done, _ in observed]
    assert completed == sorted(completed)
    record_counts = [int(message.split()[2]) for message in messages]
    assert record_counts == list(range(1, products + 1))
    if max_iterations is None:
        assert {total for _, _, total in observed} == {None}
        assert completed[-1] == products
        assert messages[-1].endswith("; set max_iterations to bound the solve")
    else:
        # A basis of 20 vectors takes 21 products of L Lᵀ and each of 200
        # restarts at most 15 more, each two products of the Laplacian.
        bound = 2 * (21 + 200 * 15)
        assert {total for _, _, total in observed} == {bound}
        # The solve converged, so its bar completes.
        assert completed[-2:] == [products, bound]
        assert f"{products} of at most {bound} operator products" in messages[-1]
        assert messages[-1].endswith("cells (max_iterations=200)")


def test_graph_components_are_numbered_by_decreasing_size():
    from cytearc.trajectory.pseudotime import pseudotime_components

    # Paths of 3, 5, and 4 cells, laid out in a shuffled cell order.
    order = np.random.default_rng(3).permutation(12)
    paths = [order[:3], order[3:8], order[8:]]
    adjacency = np.zeros((12, 12))
    for members in paths:
        for first, second in zip(members[:-1], members[1:], strict=True):
            adjacency[first, second] = adjacency[second, first] = 1.0
    graph = csr_matrix(adjacency)
    expected = np.empty(12, dtype=np.int64)
    for number, members in enumerate((paths[1], paths[2], paths[0])):
        expected[members] = number

    component, sizes, scored = pseudotime_components(graph, np.arange(12), "largest")

    assert sizes == (5, 4, 3)
    np.testing.assert_array_equal(component, expected)
    assert scored == 0

    # "sources" scores the component that holds the most source cells.
    source_cells = np.isin(np.arange(12), paths[0][:2])
    _, _, scored = pseudotime_components(
        graph, np.arange(12), "sources", source_cells=source_cells
    )
    assert scored == 2
    retained, listed = _select_pseudotime_component(
        graph, np.arange(12), "sources", source_cells=source_cells
    )
    np.testing.assert_array_equal(retained, np.isin(np.arange(12), paths[0]))
    assert listed == [5, 4, 3]
    # A tie goes to the lower number, the larger component.
    source_cells[paths[2][:2]] = True
    _, _, scored = pseudotime_components(
        graph, np.arange(12), "sources", source_cells=source_cells
    )
    assert scored == 1


@pytest.mark.parametrize(
    ("policy", "source_cells", "message"),
    [
        ("sources", None, "needs the source cells"),
        ("sources", np.zeros(4, dtype=bool), "needs at least one source cell"),
        ("sources", np.ones(3, dtype=bool), "one entry per graph row"),
        ("sources", np.ones(4), "one entry per graph row"),
        ("largest", np.ones(4, dtype=bool), "applies only to component_policy"),
        ("smallest", None, "'largest', 'sources' or 'error'"),
    ],
)
def test_component_selection_refuses_unusable_source_cells(
    policy, source_cells, message
):
    from cytearc.trajectory.pseudotime import pseudotime_components

    with pytest.raises(ValueError, match=message):
        pseudotime_components(
            csr_matrix(np.ones((4, 4))),
            np.arange(4),
            policy,
            source_cells=source_cells,
        )


def test_missing_source_and_sink_labels_raise_a_typed_error():
    from cytearc.trajectory import SourceSinkLabelsError

    labels = pd.Series(["a", "b", "c"])

    with pytest.raises(SourceSinkLabelsError) as raised:
        _validate_source_sink_labels(labels, ["a", "x"], ["y", "c"], "test cells")
    error = raised.value
    assert (error.missing_sources, error.missing_sinks) == (("x",), ("y",))
    assert (error.missing_from, error.component_sizes) == ("labels", None)
    assert str(error) == (
        "Source/sink labels were not found in test cells. "
        "Missing sources: ['x']; missing sinks: ['y']"
    )

    # The error is a ValueError, so earlier handlers still catch it.
    with pytest.raises(ValueError, match="missing sinks: \\['z'\\]") as raised:
        _validate_source_sink_labels(
            labels,
            [],
            ["z"],
            "the retained connected component",
            missing_from="scored component",
            component_sizes=[3, 2],
        )
    assert isinstance(raised.value, SourceSinkLabelsError)
    assert raised.value.missing_from == "scored component"
    assert raised.value.component_sizes == (3, 2)
    assert str(raised.value).endswith(
        "The graph's connected components hold [3, 2] cells"
    )

    # The sizes go with the scored component and only with it.
    for arguments in (
        {"missing_from": "scored component"},
        {"component_sizes": [3]},
        {"missing_from": "graph", "component_sizes": [3]},
    ):
        with pytest.raises(ValueError, match="component_sizes|missing_from"):
            _validate_source_sink_labels(labels, ["a"], [], "cells", **arguments)


# Pseudotime scoring on stored graphs


def _pseudotime_parameters(method: str = "untruncated", **changes):
    """Return the parameters that a pseudotime of the Y graph records."""
    parameters = {
        "method": method,
        "sources": ["root"],
        "sinks": ["A", "B"],
        "min_max_norm_ptime": True,
        "component_policy": "largest",
    }
    if method == "truncated":
        parameters |= {"n_singular_vals": 3, "random_seed": 4444}
    return parameters | changes


def _method_arguments(method: str) -> dict:
    """Return the arguments that ask a pseudotime of the Y graph for ``method``."""
    if method == "truncated":
        return {"method": "truncated", "n_singular_vals": 3}
    return {"method": method}


@pytest.mark.parametrize("method", ["untruncated", "truncated"])
def test_a_pseudotime_records_its_method_and_the_parameters_of_its_method(method):
    from .test_trajectory_record_paths import _Y_LABELS, _trajectory_store, _y_graph

    trajectory = _trajectory_store(_y_graph())
    labels = trajectory.labels(_Y_LABELS)
    arguments = {"source_sink": labels, "sources": ["root"], "sinks": ["A", "B"]}
    ref = trajectory.store.trajectory.pseudotime(
        trajectory.graph, **arguments, **_method_arguments(method)
    )

    status = trajectory.store.artifacts.inspect(ref)
    assert status.parameters == _pseudotime_parameters(method)
    assert set(status.inputs or {}) == {
        "connectivity_map",
        "source_sink",
        "cell_selection",
    }
    # The untruncated potential is revision 2 of pseudotime scoring, and the
    # truncated potential, which earlier releases computed, revision 1.
    assert (status.revision, status.is_current) == (
        (2, True) if method == "untruncated" else (1, True)
    )
    if method == "untruncated":
        # A call that leaves the method unset asks for the same artifact.
        assert trajectory.store.trajectory.pseudotime(
            trajectory.graph, **arguments
        ) == (ref)


@pytest.mark.parametrize("method", ["untruncated", "truncated"])
def test_a_set_solver_bound_enters_the_record_and_makes_another_artifact(method):
    from .test_trajectory_record_paths import _Y_LABELS, _trajectory_store, _y_graph

    trajectory = _trajectory_store(_y_graph())
    store = trajectory.store
    arguments = {
        "source_sink": trajectory.labels(_Y_LABELS),
        "sources": ["root"],
        "sinks": ["A", "B"],
        **_method_arguments(method),
    }
    default = store.trajectory.pseudotime(trajectory.graph, **arguments)
    unset = store.trajectory.pseudotime(
        trajectory.graph, solver_tol=None, max_iterations=None, **arguments
    )
    bounded = store.trajectory.pseudotime(
        trajectory.graph, max_iterations=500, **arguments
    )
    tolerant = store.trajectory.pseudotime(
        trajectory.graph, solver_tol=1e-6, **arguments
    )

    assert unset == default
    assert len({default, bounded, tolerant}) == 3
    assert store.artifacts.inspect(bounded).parameters == _pseudotime_parameters(
        method, max_iterations=500
    )
    assert store.artifacts.inspect(tolerant).parameters == _pseudotime_parameters(
        method, solver_tol=1e-6
    )
    # Each is reused by a call that sets its bound again.
    assert (
        store.trajectory.pseudotime(trajectory.graph, max_iterations=500, **arguments)
        == bounded
    )
    expected = store.trajectory.load_pseudotime(default).values
    np.testing.assert_array_equal(
        store.trajectory.load_pseudotime(bounded).values, expected
    )
    np.testing.assert_allclose(
        store.trajectory.load_pseudotime(tolerant).values,
        expected,
        rtol=0.0,
        atol=1e-6,
    )


@pytest.mark.parametrize("method", ["untruncated", "truncated"])
def test_a_solve_stopped_at_its_bound_saves_no_pseudotime(method):
    from cytearc.trajectory import PseudotimeConvergenceError

    from .test_trajectory_record_paths import _trajectory_store

    trajectory = _trajectory_store(_clusters_on_a_tree(300, 30).toarray())
    values = np.full(300, "middle", dtype=object)
    values[:5] = "start"
    values[-5:] = "end"

    with pytest.raises(PseudotimeConvergenceError) as raised:
        trajectory.store.trajectory.pseudotime(
            trajectory.graph,
            source_sink=trajectory.labels(values.astype(str)),
            sources=["start"],
            sinks=["end"],
            method=method,
            max_iterations=1,
        )

    error = raised.value
    assert (error.method, error.n_cells, error.max_iterations) == (method, 300, 1)
    if method == "truncated":
        assert (error.n_modes, error.restarts) == (30, 1)
    else:
        # SciPy's symmetry check takes two products and the one iteration one.
        assert (error.operator_products, error.n_modes, error.restarts) == (
            3,
            None,
            None,
        )
    assert trajectory.artifacts("pseudotime") == []


def test_loaded_pseudotime_numbers_the_graph_components_by_size():
    from scipy.sparse import block_diag
    from scipy.sparse.csgraph import connected_components

    from .test_trajectory_record_paths import (
        _Y_LABELS,
        _adjacency,
        _path,
        _trajectory_store,
        _y_graph,
    )

    # A path of 4 cells, a pair, and a Y of 5 cells, in a shuffled cell order.
    blocks = block_diag(
        (
            csr_matrix(_path(4)),
            csr_matrix(_adjacency(2, [(0, 1)])),
            csr_matrix(_y_graph()),
        )
    ).toarray()
    order = np.random.default_rng(5).permutation(11)
    trajectory = _trajectory_store(blocks[np.ix_(order, order)])
    block_labels = np.array(["other"] * 6 + list(_Y_LABELS))
    ref = trajectory.store.trajectory.pseudotime(
        trajectory.graph,
        source_sink=trajectory.labels(block_labels[order]),
        sources=["root"],
        sinks=["A", "B"],
    )

    result = trajectory.store.trajectory.load_pseudotime(ref)

    # Cells 0 to 3 of the blocks form the path, 4 and 5 the pair, the rest the Y.
    block_component = np.array([1] * 4 + [2] * 2 + [0] * 5)
    assert result.component_sizes == (5, 4, 2)
    np.testing.assert_array_equal(result.component, block_component[order])
    np.testing.assert_array_equal(result.valid, result.component == 0)
    graph = trajectory.store.graph.load(trajectory.graph, symmetric=True)
    _, scipy_labels = connected_components(graph, directed=False)
    assert sorted(np.bincount(scipy_labels).tolist(), reverse=True) == [5, 4, 2]

    connected = _trajectory_store(_y_graph())
    whole = connected.store.trajectory.load_pseudotime(
        connected.store.trajectory.pseudotime(
            connected.graph,
            source_sink=connected.labels(_Y_LABELS),
            sources=["root"],
            sinks=["A", "B"],
        )
    )
    assert whole.component_sizes == (5,)
    np.testing.assert_array_equal(whole.component, np.zeros(5))


def _path_and_y() -> tuple[np.ndarray, np.ndarray]:
    """Return a path of 8 cells beside a Y of 5, and labels for the Y."""
    from scipy.sparse import block_diag

    from .test_trajectory_record_paths import _Y_LABELS, _path, _y_graph

    adjacency = block_diag((csr_matrix(_path(8)), csr_matrix(_y_graph()))).toarray()
    return adjacency, np.concatenate((np.full(8, "other"), _Y_LABELS))


def test_the_sources_policy_scores_the_component_that_holds_the_sources():
    from cytearc.trajectory import SourceSinkLabelsError

    from .test_trajectory_record_paths import (
        _Y_LABELS,
        _Y_SOURCE_SINK,
        _trajectory_store,
        _y_graph,
    )

    adjacency, labels = _path_and_y()
    trajectory = _trajectory_store(adjacency)
    store = trajectory.store
    arguments = {
        "source_sink": trajectory.labels(labels),
        "sources": ["root"],
        "sinks": ["A", "B"],
    }

    # The largest component, the path, holds none of the labels.
    with pytest.raises(SourceSinkLabelsError) as raised:
        store.trajectory.pseudotime(trajectory.graph, **arguments)
    assert (raised.value.missing_sources, raised.value.missing_sinks) == (
        ("root",),
        ("A", "B"),
    )
    assert raised.value.missing_from == "scored component"
    assert raised.value.component_sizes == (8, 5)

    ref = store.trajectory.pseudotime(
        trajectory.graph, component_policy="sources", **arguments
    )
    result = store.trajectory.load_pseudotime(ref)
    np.testing.assert_array_equal(result.valid, np.arange(13) >= 8)
    np.testing.assert_array_equal(result.component, np.repeat([0, 1], [8, 5]))
    assert result.component_sizes == (8, 5)
    assert np.isnan(result.values[:8]).all()
    # The Y is scored as it is on a graph of the Y alone.
    alone = _trajectory_store(_y_graph())
    alone_ref = alone.store.trajectory.pseudotime(
        alone.graph,
        source_sink=alone.labels(_Y_LABELS),
        sources=["root"],
        sinks=["A", "B"],
    )
    np.testing.assert_allclose(
        result.values[8:],
        alone.store.trajectory.load_pseudotime(alone_ref).values,
        rtol=0.0,
        atol=1e-12,
    )

    # A custom vector's sources are its negative values.
    from_vector = store.trajectory.pseudotime(
        trajectory.graph,
        ss_vec=np.concatenate((np.zeros(8), _Y_SOURCE_SINK)),
        component_policy="sources",
    )
    np.testing.assert_array_equal(
        store.trajectory.load_pseudotime(from_vector).valid, np.arange(13) >= 8
    )


@pytest.mark.parametrize(
    ("sources", "sinks", "expected"),
    [
        # The path holds two source cells and the Y one, so the path is scored.
        (["early", "root"], ["late"], (("root",), (), "scored component", (8, 5))),
        (["root"], ["late", "A"], ((), ("late",), "scored component", (8, 5))),
        (["absent"], ["A"], (("absent",), (), "labels", None)),
    ],
    ids=["sources-in-two-components", "sink-outside", "label-absent"],
)
def test_the_sources_policy_names_labels_outside_the_scored_component(
    sources, sinks, expected
):
    from cytearc.trajectory import SourceSinkLabelsError

    from .test_trajectory_record_paths import _trajectory_store

    adjacency, labels = _path_and_y()
    labels[[0, 1]] = "early"
    labels[7] = "late"
    trajectory = _trajectory_store(adjacency)

    with pytest.raises(SourceSinkLabelsError) as raised:
        trajectory.store.trajectory.pseudotime(
            trajectory.graph,
            source_sink=trajectory.labels(labels),
            sources=sources,
            sinks=sinks,
            component_policy="sources",
        )

    error = raised.value
    assert (
        error.missing_sources,
        error.missing_sinks,
        error.missing_from,
        error.component_sizes,
    ) == expected
    assert trajectory.artifacts("pseudotime") == []


# The untruncated potential


def _random_walk_laplacian(adjacency: np.ndarray) -> np.ndarray:
    """Return the dense random-walk Laplacian ``I - D^-1 W`` of a graph."""
    degree = adjacency.sum(axis=1)
    return np.eye(len(degree)) - adjacency / degree[:, np.newaxis]


def _pseudoinverse_potential(adjacency: np.ndarray, source_sink: np.ndarray):
    """Return NumPy's dense ``pinv(L) R'`` and ``R'``, the treated source-sink vector.

    ``R'`` is ``R`` without its component along the degree vector, which
    spans the left null space of the random-walk Laplacian ``L``.
    """
    degree = adjacency.sum(axis=1)
    laplacian = _random_walk_laplacian(adjacency)
    # A connected graph's Laplacian has one null mode, which a cutoff far
    # above rounding and far below the other singular values drops.
    assert np.linalg.matrix_rank(laplacian, rtol=1e-10) == len(degree) - 1
    projected = source_sink - degree * (degree @ source_sink) / (degree @ degree)
    return np.linalg.pinv(laplacian, rtol=1e-10) @ projected, projected


def _weighted_path(weights: list[float]) -> np.ndarray:
    adjacency = np.zeros((len(weights) + 1,) * 2)
    for index, weight in enumerate(weights):
        adjacency[index, index + 1] = adjacency[index + 1, index] = weight
    return adjacency


def _random_connected_graph(n_cells: int, seed: int) -> np.ndarray:
    """Return a weighted graph of a random spanning tree and random extra edges."""
    rng = np.random.default_rng(seed)
    adjacency = np.zeros((n_cells, n_cells))
    for cell in range(1, n_cells):
        other = int(rng.integers(0, cell))
        adjacency[cell, other] = adjacency[other, cell] = rng.uniform(0.1, 1.0)
    for _ in range(2 * n_cells):
        first, second = rng.integers(0, n_cells, size=2)
        if first != second:
            adjacency[first, second] = adjacency[second, first] = rng.uniform(0.1, 1.0)
    return adjacency


def _small_graphs() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Graphs whose degrees differ, each with a source-sink vector that sums to zero."""
    from .test_trajectory_record_paths import _Y_SOURCE_SINK, _y_graph

    path = _weighted_path([1.0, 2.0, 1.5, 3.0, 0.75, 2.5, 1.25])
    trees = _clusters_on_a_tree(300, 10, seed=1).toarray()
    groups = pd.Series(np.arange(300) // 30)
    random_graph = _random_connected_graph(60, seed=4)
    return {
        "weighted-path": (path, _first_to_last_source_sink(8)),
        "y": (_y_graph(), _Y_SOURCE_SINK),
        "clusters-on-a-tree": (trees, _make_source_sink_vector(groups, [0], [9])),
        "random": (random_graph, np.random.default_rng(5).normal(size=60)),
    }


@pytest.mark.parametrize("name", ["weighted-path", "y", "clusters-on-a-tree", "random"])
def test_the_untruncated_potential_equals_the_dense_pseudoinverse(name):
    from cytearc.trajectory import pba_potential

    adjacency, source_sink = _small_graphs()[name]
    if name == "random":
        source_sink = source_sink - source_sink.mean()
    degree = adjacency.sum(axis=1)
    # The degrees differ and R has a component along them, so the treatment
    # of that component decides the potential.
    assert np.ptp(degree) > 0.1
    assert abs(degree @ source_sink) > 1e-3 * np.linalg.norm(degree)
    expected, projected = _pseudoinverse_potential(adjacency, source_sink)

    potential = pba_potential(csr_matrix(adjacency), source_sink)

    np.testing.assert_allclose(
        potential, expected, rtol=0.0, atol=1e-8 * np.abs(expected).max()
    )
    # Removing a constant instead, which also balances the degree-weighted
    # sum, would give another potential.
    balanced = source_sink - (degree @ source_sink) / degree.sum()
    other = np.linalg.pinv(_random_walk_laplacian(adjacency), rtol=1e-10) @ balanced
    assert np.abs(other - expected).max() > 1e-3 * np.abs(expected).max()
    # The pseudoinverse drops the component itself.
    np.testing.assert_allclose(
        np.linalg.pinv(_random_walk_laplacian(adjacency), rtol=1e-10) @ source_sink,
        expected,
        rtol=0.0,
        atol=1e-10 * np.abs(expected).max(),
    )
    assert abs(projected @ degree) < 1e-12 * np.linalg.norm(degree)


def test_the_untruncated_potential_orders_a_path_from_source_to_sink():
    from cytearc.trajectory import pba_potential

    from .test_trajectory_record_paths import _trajectory_store

    # Stored edge weights are united with the fuzzy union, so they stay in (0, 1].
    adjacency = _weighted_path([0.9, 0.3, 0.6, 1.0, 0.45, 0.8, 0.2, 0.7])
    source_sink = _first_to_last_source_sink(9)

    potential = pba_potential(csr_matrix(adjacency), source_sink)

    assert np.all(np.diff(potential) > 0)
    trajectory = _trajectory_store(adjacency)
    store = trajectory.store
    values = store.trajectory.load_pseudotime(
        store.trajectory.pseudotime(trajectory.graph, ss_vec=source_sink)
    ).values
    assert (values[0], values[-1]) == (0.0, 1.0)
    assert np.all(np.diff(values) > 0)
    scored = pba_potential(
        store.graph.load(trajectory.graph, symmetric=True, upper_only=False),
        source_sink,
    )
    np.testing.assert_allclose(
        values, (scored - scored.min()) / np.ptp(scored), rtol=0.0, atol=1e-12
    )


def test_the_untruncated_potential_finishes_where_the_truncated_solve_cannot():
    from scipy.sparse import diags, identity

    from cytearc.trajectory import PseudotimeConvergenceError, pba_potential

    # 400 clusters of about ten cells, each joined to another by few edges,
    # pack the smallest singular values of the Laplacian closely.
    graph = _clusters_on_a_tree(4000, 400)
    source_sink = _first_to_last_source_sink(4000)

    with pytest.raises(PseudotimeConvergenceError) as raised:
        _truncated_pba_potential(
            _random_walk_laplacian_transpose(graph),
            30,
            4444,
            source_sink,
            max_iterations=250,
        )
    # A restart takes up to 62 products, so the singular value solve stopped
    # after more than 10,000.
    assert raised.value.converged_modes < 30
    assert raised.value.operator_products > 10_000

    bounded = pba_potential(graph, source_sink, max_iterations=250)

    # The bound changes nothing that the solve computes.
    np.testing.assert_array_equal(bounded, pba_potential(graph, source_sink))
    degree = np.asarray(graph.sum(axis=1)).ravel()
    laplacian = identity(4000) - diags(1.0 / degree) @ graph
    projected = source_sink - degree * (degree @ source_sink) / (degree @ degree)
    residual = np.linalg.norm(laplacian @ bounded - projected)
    assert residual < 1e-9 * np.linalg.norm(projected)
    assert abs(bounded.mean()) < 1e-12 * np.abs(bounded).max()


def _count_minres_products(monkeypatch) -> list[int]:
    """Count the operator products that minres takes, apart from the solve's count.

    Each call appends one count to the returned list.
    """
    from scipy.sparse.linalg import LinearOperator, aslinearoperator

    import cytearc.trajectory.pseudotime as pseudotime_module

    original = pseudotime_module.minres
    counts: list[int] = []

    def counting_minres(operator, right_hand_side, **arguments):
        inner = aslinearoperator(operator)
        counts.append(0)

        def matvec(values):
            counts[-1] += 1
            return inner.matvec(values)

        counted = LinearOperator(shape=inner.shape, matvec=matvec, dtype=inner.dtype)
        return original(counted, right_hand_side, **arguments)

    monkeypatch.setattr(pseudotime_module, "minres", counting_minres)
    return counts


def test_a_bounded_untruncated_solve_stops_with_the_counts_of_the_solve(monkeypatch):
    from scipy.sparse.linalg import LinearOperator, minres

    from cytearc.trajectory import PseudotimeConvergenceError, pba_potential

    graph = _clusters_on_a_tree(600, 60)
    source_sink = _first_to_last_source_sink(600)
    counts = _count_minres_products(monkeypatch)

    with pytest.raises(PseudotimeConvergenceError) as raised:
        pba_potential(graph, source_sink, max_iterations=5)

    error = raised.value
    assert isinstance(error, RuntimeError)
    assert (error.method, error.n_cells, error.max_iterations) == (
        "untruncated",
        600,
        5,
    )
    assert (error.solver_tol, error.n_modes, error.converged_modes) == (
        None,
        None,
        None,
    )
    assert error.restarts is None
    # SciPy's symmetry check takes two products and each iteration one.
    assert error.operator_products == counts[0] == 5 + 2
    # The residual is that of SciPy's solve of the symmetric system after
    # five iterations.
    degree = np.asarray(graph.sum(axis=1)).ravel()
    scale = 1.0 / np.sqrt(degree)
    symmetric = LinearOperator(
        shape=graph.shape,
        matvec=lambda values: values - scale * (graph @ (scale * values)),
        dtype=np.float64,
    )
    projected = source_sink - degree * (degree @ source_sink) / (degree @ degree)
    right_hand_side = projected / scale
    solution, info = minres(symmetric, right_hand_side, rtol=0.0, maxiter=5)
    assert info == 5
    expected = np.linalg.norm(symmetric @ solution - right_hand_side) / np.linalg.norm(
        right_hand_side
    )
    assert error.relative_residual == pytest.approx(expected, rel=1e-8)
    assert "MINRES reached max_iterations=5 after 7 operator products" in str(error)
    assert "Raise max_iterations" in str(error)


def test_the_untruncated_solver_bounds_reach_minres(monkeypatch):
    import cytearc.trajectory.pseudotime as pseudotime_module
    from cytearc.trajectory import pba_potential

    received: list[dict] = []
    original = pseudotime_module.minres

    def recording_minres(operator, right_hand_side, **arguments):
        received.append(
            {name: arguments[name] for name in ("rtol", "maxiter", "check")}
        )
        return original(operator, right_hand_side, **arguments)

    monkeypatch.setattr(pseudotime_module, "minres", recording_minres)
    graph = _clusters_on_a_tree(300, 10, seed=1)
    source_sink = _first_to_last_source_sink(300)
    default = pba_potential(graph, source_sink)
    bounded = pba_potential(graph, source_sink, solver_tol=1e-8, max_iterations=500)

    # Machine precision by default, SciPy's bound of five iterations per cell,
    # and SciPy's check that the operator is symmetric.
    assert received == [
        {"rtol": 0.0, "maxiter": None, "check": True},
        {"rtol": 1e-8, "maxiter": 500, "check": True},
    ]
    # A looser tolerance moves the potential smoothly, far less than its range.
    np.testing.assert_allclose(bounded, default, rtol=0.0, atol=1e-4 * np.ptp(default))


@pytest.mark.parametrize(
    ("bounds", "error", "message"),
    [
        ({"solver_tol": 0.0}, ValueError, "greater than 0 and less than 1"),
        ({"solver_tol": 1.0}, ValueError, "greater than 0 and less than 1"),
        ({"solver_tol": float("nan")}, ValueError, "solver_tol must be finite"),
        ({"solver_tol": "1e-4"}, TypeError, "solver_tol must be a real number"),
        ({"max_iterations": 0}, ValueError, "max_iterations must be at least 1"),
        ({"max_iterations": 2.0}, TypeError, "max_iterations must be an integer"),
        ({"max_iterations": True}, TypeError, "max_iterations must be an integer"),
    ],
)
def test_untruncated_solver_bounds_are_checked_before_the_solve(
    bounds, error, message, monkeypatch
):
    import cytearc.trajectory.pseudotime as pseudotime_module
    from cytearc.trajectory import pba_potential

    def unreachable(*_args, **_kwargs):
        raise AssertionError("the bounds were not checked before the solve")

    monkeypatch.setattr(pseudotime_module, "minres", unreachable)

    with pytest.raises(error, match=message):
        pba_potential(
            _clusters_on_a_tree(40, 4), _first_to_last_source_sink(40), **bounds
        )


@pytest.mark.parametrize("max_iterations", [200, None], ids=["bounded", "unbounded"])
def test_the_untruncated_solve_reports_progress_to_observers_and_the_log(
    max_iterations, monkeypatch
):
    from loguru import logger

    import cytearc.trajectory.pseudotime as pseudotime_module
    from cytearc.trajectory import pba_potential
    from cytearc.utils.logging import configure_output, progress_enabled
    from cytearc.utils.progress import observe_progress

    # Report every product, so a quick solve reports as a long one does.
    monkeypatch.setattr(pseudotime_module, "_PROGRESS_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(pseudotime_module, "_BAR_INTERVAL_SECONDS", 0.0)
    counts = _count_minres_products(monkeypatch)
    observed: list[tuple[str | None, int, int | None]] = []
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="INFO",
        filter=lambda record: record["message"].startswith("Pseudotime scoring: "),
    )
    displayed = progress_enabled()
    # Observers receive the bar whether or not it is displayed.
    configure_output(progress=False)
    try:
        with observe_progress(
            lambda _bar, description, completed, total: observed.append(
                (description, completed, total)
            )
        ):
            pba_potential(
                _clusters_on_a_tree(40, 4),
                _first_to_last_source_sink(40),
                max_iterations=max_iterations,
            )
    finally:
        configure_output(progress=displayed)
        logger.remove(sink)

    products = counts[0]
    assert products > 2
    assert {description for description, _, _ in observed} == {"Pseudotime solve"}
    completed = [done for _, done, _ in observed]
    assert completed == sorted(completed)
    record_counts = [int(message.split()[2]) for message in messages]
    assert record_counts == list(range(1, products + 1))
    assert "solving for the untruncated potential of 40 cells" in messages[-1]
    if max_iterations is None:
        assert {total for _, _, total in observed} == {None}
        assert completed[-1] == products
        assert messages[-1].endswith("; set max_iterations to bound the solve")
    else:
        # Two products of SciPy's symmetry check and one per iteration.
        bound = 200 + 2
        assert {total for _, _, total in observed} == {bound}
        # The solve converged, so its bar completes.
        assert completed[-2:] == [products, bound]
        assert f"{products} of at most {bound} operator products" in messages[-1]
        assert messages[-1].endswith("cells (max_iterations=200)")


def _unusable_inputs():
    path = csr_matrix(_weighted_path([0.5, 1.0, 0.25, 0.75, 1.0]))
    directed = path.toarray()
    directed[np.tril_indices(6, -1)] = 0.0
    directed[5, 0] = 1.0
    two_paths = csr_matrix(
        np.block(
            [[path.toarray(), np.zeros((6, 6))], [np.zeros((6, 6)), path.toarray()]]
        )
    )
    isolated = path.toarray()
    isolated[4, 5] = isolated[5, 4] = 0.0
    source_sink = _first_to_last_source_sink(6)
    return {
        "dense-graph": (
            path.toarray(),
            source_sink,
            TypeError,
            "graph must be a SciPy sparse matrix",
        ),
        "rectangular-graph": (
            csr_matrix(np.ones((6, 5))),
            source_sink,
            ValueError,
            "graph must be square",
        ),
        "directed-graph": (
            csr_matrix(directed),
            source_sink,
            ValueError,
            "graph must be symmetric",
        ),
        "two-components": (
            two_paths,
            _first_to_last_source_sink(12),
            ValueError,
            "graph must be connected, but it has 2 connected components",
        ),
        "isolated-cell": (
            csr_matrix(isolated),
            source_sink,
            ValueError,
            "contains isolated cells",
        ),
        "short-vector": (path, source_sink[:5], ValueError, "one finite value"),
        "missing-value": (
            path,
            np.r_[np.nan, source_sink[1:]],
            ValueError,
            "one finite value for each of the 6 cells",
        ),
    }


@pytest.mark.parametrize(
    "case",
    [
        "dense-graph",
        "rectangular-graph",
        "directed-graph",
        "two-components",
        "isolated-cell",
        "short-vector",
        "missing-value",
    ],
)
def test_the_untruncated_potential_refuses_unusable_graphs_and_vectors(case):
    from cytearc.trajectory import pba_potential

    graph, source_sink, error, message = _unusable_inputs()[case]

    with pytest.raises(error, match=message):
        pba_potential(graph, source_sink)


def test_the_untruncated_potential_is_linear_up_to_the_float64_range():
    from cytearc.trajectory import pba_potential

    graph = _clusters_on_a_tree(300, 10, seed=1)
    source_sink = _first_to_last_source_sink(300)
    potential = pba_potential(graph, source_sink)

    # A vector near the largest float64 values would overflow the norms of
    # an unscaled solve, so the solve scales it to a largest magnitude of 1.
    np.testing.assert_allclose(
        pba_potential(graph, 1e300 * source_sink), 1e300 * potential, rtol=1e-12
    )
    np.testing.assert_array_equal(pba_potential(graph, np.zeros(300)), np.zeros(300))
    # A vector along the degree vector has no component in the range of L.
    degree = np.asarray(graph.sum(axis=1)).ravel()
    np.testing.assert_allclose(pba_potential(graph, degree), 0.0, rtol=0.0, atol=1e-12)


def test_the_convergence_error_carries_the_fields_of_its_solve():
    import pickle

    from cytearc.trajectory import PseudotimeConvergenceError

    fields = {
        "n_cells": 600,
        "operator_products": 186,
        "max_iterations": 1,
        "solver_tol": None,
    }
    truncated = PseudotimeConvergenceError(
        method="truncated", n_modes=30, converged_modes=0, restarts=1, **fields
    )
    untruncated = PseudotimeConvergenceError(
        method="untruncated", relative_residual=2.5e-3, **fields
    )

    for error in (truncated, untruncated):
        restored = pickle.loads(pickle.dumps(error))
        assert type(restored) is PseudotimeConvergenceError
        assert (restored.args, vars(restored)) == (error.args, vars(error))
    assert "0 of 30 singular modes converged" in str(truncated)
    assert "fewer modes do not help" in str(truncated)
    assert "with a relative residual of 2.5e-03 (solver_tol=None)" in str(untruncated)
    for arguments in (
        {"method": "truncated", "n_modes": 30, "converged_modes": 0},
        {
            "method": "truncated",
            "n_modes": 30,
            "converged_modes": 0,
            "restarts": 1,
            "relative_residual": 1e-3,
        },
        {"method": "untruncated"},
        {"method": "untruncated", "relative_residual": 1e-3, "restarts": 1},
        {"method": "exact", "relative_residual": 1e-3},
    ):
        with pytest.raises(ValueError, match="must be|method must be one of"):
            PseudotimeConvergenceError(**fields, **arguments)


def test_the_truncated_method_reproduces_the_earlier_potential_bit_for_bit():
    from .test_trajectory_record_paths import _trajectory_store

    trajectory = _trajectory_store(_clusters_on_a_tree(120, 6, seed=2).toarray())
    store = trajectory.store
    source_sink = _first_to_last_source_sink(120)

    ref = store.trajectory.pseudotime(
        trajectory.graph, ss_vec=source_sink, method="truncated"
    )

    # Releases before 1.4.0 scored every pseudotime so: the 30 smallest
    # singular modes from the seed 4444, scaled to [0, 1].
    graph = store.graph.load(trajectory.graph, symmetric=True, upper_only=False)
    earlier = _truncated_pba_potential(
        _random_walk_laplacian_transpose(graph), 30, 4444, source_sink
    )
    earlier = np.clip((earlier - earlier.min()) / np.ptp(earlier), 0.0, 1.0)
    np.testing.assert_array_equal(store.trajectory.load_pseudotime(ref).values, earlier)
    # The record is the earlier one with the method.
    assert store.artifacts.inspect(ref).parameters == {
        "method": "truncated",
        "n_singular_vals": 30,
        "sources": [],
        "sinks": [],
        "min_max_norm_ptime": True,
        "random_seed": 4444,
        "component_policy": "largest",
    }


def test_saved_default_pseudotimes_are_computed_again_and_truncated_ones_reused():
    from cytearc.storage.artifacts import artifact_group
    from cytearc.storage.operation_revisions import OPERATION_REVISIONS

    from .test_trajectory_record_paths import _trajectory_store

    trajectory = _trajectory_store(_clusters_on_a_tree(120, 6, seed=2).toarray())
    store = trajectory.store
    source_sink = _first_to_last_source_sink(120)
    # A release before 1.4.0 saved the truncated potential by default, with no
    # method and no revision in its record.
    earlier = store.trajectory.pseudotime(
        trajectory.graph, ss_vec=source_sink, method="truncated"
    )
    group = artifact_group(store.zw, earlier)
    provenance = dict(group.attrs["provenance"])
    provenance["parameters"] = {
        key: value for key, value in provenance["parameters"].items() if key != "method"
    }
    assert "revision" not in provenance
    group.attrs["provenance"] = provenance

    default = store.trajectory.pseudotime(trajectory.graph, ss_vec=source_sink)
    truncated = store.trajectory.pseudotime(
        trajectory.graph, ss_vec=source_sink, method="truncated"
    )

    (revision,) = OPERATION_REVISIONS["run_pseudotime_scoring"]
    assert len({earlier, default, truncated}) == 3
    old = store.artifacts.inspect(earlier)
    assert (old.revision, old.current_revision, old.is_current) == (1, 2, False)
    assert old.superseded_by == (revision,)
    markdown = store.artifacts.lineage(earlier).to_markdown()
    assert "- Status: `stale`" in markdown
    assert f"- Revision: `1, current 2: {revision.change}`" in markdown
    new = store.artifacts.inspect(default)
    assert (new.revision, new.is_current) == (2, True)
    assert new.parameters["method"] == "untruncated"
    # The earlier pseudotime still loads, with the potential of the truncated
    # method, which an explicit call reuses and keeps current.
    np.testing.assert_array_equal(
        store.trajectory.load_pseudotime(earlier).values,
        store.trajectory.load_pseudotime(truncated).values,
    )
    again = store.trajectory.pseudotime(
        trajectory.graph, ss_vec=source_sink, method="truncated"
    )
    assert again == truncated
    kept = store.artifacts.inspect(truncated)
    assert (kept.revision, kept.current_revision, kept.is_current) == (1, 1, True)
    assert kept.superseded_by == ()
    assert store.trajectory.pseudotime(trajectory.graph, ss_vec=source_sink) == default


@pytest.mark.parametrize(
    ("parameters", "expected"),
    [
        (_pseudotime_parameters("untruncated"), 2),
        (_pseudotime_parameters("truncated"), 1),
        # Records of releases before 1.4.0 hold no method.
        (
            {
                key: value
                for key, value in _pseudotime_parameters("truncated").items()
                if key != "method"
            },
            2,
        ),
        # A record of another release may lack any parameter, or hold any value.
        ({}, 2),
        ({"method": ["truncated"]}, 2),
    ],
)
def test_the_pseudotime_revision_applies_to_pseudotimes_without_the_truncated_method(
    parameters, expected
):
    from cytearc.storage.operation_revisions import (
        OPERATION_REVISIONS,
        effective_revision,
    )

    inputs = {"connectivity_map": {"artifact_id": "a" * 64}}

    assert (
        effective_revision("run_pseudotime_scoring", "pseudotime", parameters, inputs)
        == expected
    )
    (revision,) = OPERATION_REVISIONS["run_pseudotime_scoring"]
    assert (revision.revision, revision.release) == (2, "1.4.0")
