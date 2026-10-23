"""Default ANN index settings, the recorded query search depth, and query threads.

The index's structure depends on ``ann_efc``, ``ann_m``, the seed and the
insertion order; the search depth ``ef`` only changes queries. A query
searches ``max(index ann_ef, 2 * k)`` candidates by default and records the
depth it used as its ``ann_ef`` only when that is not the index's own depth,
``max(index ann_ef, k + 1)``, which every query searched before queries chose
a depth. Queries run on every datastore thread, which never changes the
neighbors.
"""

import inspect
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from cytearc import DataStore
from cytearc.datastore.namespaces import EmbeddingsAccessor, GraphAccessor
from cytearc.embeddings.imported import write_imported_coordinates
from cytearc.neighbors.index import instantiate_knn_index, query_search_depth
from cytearc.neighbors.stages import AnnIndexStage
from cytearc.storage.artifact_writer import artifact_transaction, plan_artifact
from cytearc.storage.artifacts import ArtifactRef, artifact_group, fingerprint_array
from tests.storage_helpers import write_count_store
from tests.test_imported_coordinates import (
    _SOURCE_DIGEST,
    _graph_store,
    _root_with_selection,
)

N_CELLS = 120


def _counts() -> np.ndarray:
    rng = np.random.default_rng(31)
    counts = rng.poisson(rng.gamma(1.0, 3.0, size=40), size=(N_CELLS, 40))
    counts[:, 0] += 3
    return counts


def _open(zarr_loc: Path, **options) -> DataStore:
    return DataStore(
        str(zarr_loc),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
        **options,
    )


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> tuple[Path, ArtifactRef]:
    zarr_loc = tmp_path_factory.mktemp("ann_query_settings") / "store.zarr"
    write_count_store(str(zarr_loc), {"RNA": _counts()}, "uint16")
    store = _open(zarr_loc)
    cells = store.snapshot_cell_selection()
    normalized = store.features.normalize(
        cells, store.features.universe(from_assay="RNA")
    )
    return zarr_loc, store.reduction.pca(normalized, dims=5)


@pytest.fixture
def store(template, tmp_path) -> tuple[DataStore, ArtifactRef]:
    zarr_loc, pca = template
    target = tmp_path / "store.zarr"
    shutil.copytree(zarr_loc, target)
    return _open(target), pca


def _configured(monkeypatch) -> list[tuple[int, int]]:
    """Record the search depth and threads of every configured index."""
    calls: list[tuple[int, int]] = []
    configure = AnnIndexStage.configure

    def recording(index, *, ef, threads):
        calls.append((ef, threads))
        return configure(index, ef=ef, threads=threads)

    monkeypatch.setattr(AnnIndexStage, "configure", staticmethod(recording))
    return calls


def test_ann_index_defaults_build_and_record_the_robust_settings(store) -> None:
    datastore, pca = store
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(
            GraphAccessor.ann_index
        ).parameters.items()
        if name.startswith("ann_")
    }

    # ef_construction is at least twice M, so the lowest layer's 2 * M links
    # are chosen from enough candidates.
    assert defaults == {
        "ann_metric": "l2",
        "ann_efc": 200,
        "ann_ef": 200,
        "ann_m": 24,
        "ann_parallel": False,
    }
    ann = datastore.graph.ann_index(pca)
    assert datastore.artifacts.inspect(ann).parameters == {
        "ann_metric": "l2",
        "ann_efc": 200,
        "ann_ef": 200,
        "ann_m": 24,
        "rand_state": 4466,
        "ann_parallel": False,
    }


@pytest.mark.parametrize(
    ("k", "index_ef", "requested", "expected"),
    [
        (11, 50, None, 50),
        (25, 50, None, 50),
        (26, 50, None, 52),
        (30, 50, None, 60),
        (11, 80, None, 80),
        (50, 80, None, 100),
        (1, 1, None, 2),
        (11, 50, 7, 12),
        (11, 50, 12, 12),
        (11, 50, 100, 100),
        (30, 500, 40, 40),
        (np.int64(11), np.int64(50), np.int64(13), 13),
    ],
)
def test_query_search_depth_resolves_one_integer(k, index_ef, requested, expected):
    depth = query_search_depth(k, index_ef=index_ef, requested=requested)

    assert depth == expected
    assert type(depth) is int


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"k": 0, "index_ef": 50}, ValueError, "k must be at least 1"),
        ({"k": 11, "index_ef": 0}, ValueError, "index_ef must be at least 1"),
        ({"k": 11, "index_ef": 50, "requested": 0}, ValueError, "ann_ef must be"),
        ({"k": 11, "index_ef": 50, "requested": 1.5}, TypeError, "ann_ef must be"),
        ({"k": True, "index_ef": 50}, TypeError, "k must be an integer"),
    ],
)
def test_query_search_depth_rejects_invalid_arguments(arguments, error, message):
    k = arguments.pop("k")
    with pytest.raises(error, match=message):
        query_search_depth(k, **arguments)


def test_neighbors_search_and_record_the_resolved_depth(store, monkeypatch) -> None:
    datastore, pca = store
    calls = _configured(monkeypatch)
    ann = datastore.graph.ann_index(pca)
    # Indexes whose smallest depth is the earlier default of 50, and 80.
    floor_50 = datastore.graph.ann_index(pca, ann_ef=50)
    floor_80 = datastore.graph.ann_index(pca, ann_ef=80)

    cases = [
        # (index, k, requested ann_ef, depth searched, ann_ef recorded)
        (ann, 11, None, 200, None),
        (ann, 30, None, 200, None),
        (ann, 101, None, 202, 202),
        (floor_50, 11, None, 50, None),
        (floor_50, 30, None, 60, 60),
        (ann, 11, 7, 12, 12),
        (ann, 11, 100, 100, 100),
        (floor_80, 11, None, 80, None),
        (floor_80, 45, None, 90, 90),
    ]
    refs = []
    for index, k, requested, depth, recorded in cases:
        calls.clear()
        ref = datastore.graph.neighbors(index, k=k, ann_ef=requested)
        refs.append(ref)
        assert calls == [(depth, 1)]
        # The index's own depth is recorded as no ann_ef at all.
        expected = {"k": k, "distance_metric": "l2"}
        if recorded is not None:
            expected["ann_ef"] = recorded
        assert datastore.artifacts.inspect(ref).parameters == expected
    assert len(set(refs)) == len(refs)

    calls.clear()
    # The default depth and the same depth given explicitly are one request,
    # so neither recomputes; nor does a depth below k + 1, which searches
    # k + 1 candidates.
    assert datastore.graph.neighbors(ann, k=11) == refs[0]
    assert datastore.graph.neighbors(ann, k=11, ann_ef=200) == refs[0]
    assert datastore.graph.neighbors(ann, k=11, ann_ef=3) == refs[5]
    assert datastore.graph.neighbors(floor_50, k=11, ann_ef=50) == refs[3]
    assert datastore.graph.neighbors(floor_80, k=11, ann_ef=80) == refs[7]
    assert calls == []


def _previous_release_neighbors(
    datastore: DataStore, ann: ArtifactRef, k: int
) -> tuple[ArtifactRef, np.ndarray]:
    """Save neighbors of ``ann`` as releases did before queries chose a depth.

    Those releases searched the index's own ``ann_ef`` and recorded only ``k``
    and the metric. The indices are ones that no search returns, so a call
    that returns this artifact reused it.
    """
    record = datastore.artifacts.inspect(ann)
    assert record.inputs is not None
    planned = plan_artifact(
        datastore.zw,
        scope="assay",
        assay="RNA",
        kind="neighbors",
        operation="query_neighbors",
        parameters={"k": k, "distance_metric": "l2"},
        inputs={
            "ann_index": ann.to_dict(),
            "coordinates": record.inputs["coordinates"],
        },
        execution_options={
            "nthreads": 1,
            "batch_size": N_CELLS,
            "invalidate_cache": False,
        },
    )
    assert not planned.reused
    indices = (np.arange(N_CELLS)[:, None] + np.arange(1, k + 1)) % N_CELLS
    with artifact_transaction(datastore.zw, planned) as group:
        group.create_array("indices", data=indices.astype(np.uint32))
        group.create_array("distances", data=np.ones((N_CELLS, k), dtype=np.float32))
        group.attrs["n_cells"] = N_CELLS
        group.attrs["n_neighbors"] = k
        group.attrs["self_hit_rate"] = 100.0
    return planned.ref, indices


def test_index_depth_queries_reuse_neighbors_saved_before_depths_were_recorded(
    store, tmp_path, monkeypatch
) -> None:
    datastore, pca = store
    # Earlier releases built indexes whose smallest depth was 50.
    ann = datastore.graph.ann_index(pca, ann_ef=50)
    assert datastore.artifacts.inspect(ann).parameters["ann_ef"] == 50
    saved, indices = _previous_release_neighbors(datastore, ann, k=11)
    calls = _configured(monkeypatch)
    # The store fixture copies the template to this path.
    read_only = _open(tmp_path / "store.zarr", zarr_mode="r")

    # Eleven neighbours search max(50, 2 * 11) = 50 candidates, the index's
    # own depth, which the saved query searched too.
    for opened in (datastore, read_only):
        assert opened.graph.neighbors(ann, k=11) == saved
        assert opened.graph.neighbors(ann, k=11, ann_ef=50) == saved
        np.testing.assert_array_equal(
            artifact_group(opened.zw, saved)["indices"][:], indices
        )
    assert calls == []


def test_a_deeper_default_query_records_its_depth_and_is_computed_again(
    store, tmp_path, monkeypatch
) -> None:
    datastore, pca = store
    # An index whose smallest depth is the earlier default of 50.
    ann = datastore.graph.ann_index(pca, ann_ef=50)
    saved, indices = _previous_release_neighbors(datastore, ann, k=30)
    calls = _configured(monkeypatch)
    read_only = _open(tmp_path / "store.zarr", zarr_mode="r")

    # Thirty neighbours search max(50, 2 * 30) = 60 candidates, more than the
    # saved query's 50, so a read-only store holds no such neighbors.
    with pytest.raises(
        PermissionError,
        match=r"query_neighbors requires a DataStore opened with zarr_mode='r\+'",
    ):
        read_only.graph.neighbors(ann, k=30)
    # The index's own depth, asked for explicitly, is the saved query.
    assert read_only.graph.neighbors(ann, k=30, ann_ef=50) == saved
    assert calls == []

    deeper = datastore.graph.neighbors(ann, k=30)

    assert calls == [(60, 1)]
    assert deeper != saved
    assert datastore.artifacts.inspect(deeper).parameters == {
        "k": 30,
        "distance_metric": "l2",
        "ann_ef": 60,
    }
    assert not np.array_equal(
        artifact_group(datastore.zw, deeper)["indices"][:], indices
    )
    assert datastore.graph.neighbors(ann, k=30, ann_ef=60) == deeper
    assert datastore.graph.neighbors(ann, k=30, ann_ef=50) == saved
    assert calls == [(60, 1)]


def test_an_explicit_index_depth_has_the_default_identity(store, monkeypatch) -> None:
    datastore, pca = store
    ann = datastore.graph.ann_index(pca)
    calls = _configured(monkeypatch)

    explicit = datastore.graph.neighbors(ann, k=11, ann_ef=200)

    # The index's ann_ef is the depth a default query of eleven neighbours
    # searches, so the explicit query records no ann_ef and the default query
    # returns its artifact.
    assert calls == [(200, 1)]
    assert datastore.artifacts.inspect(explicit).parameters == {
        "k": 11,
        "distance_metric": "l2",
    }
    assert datastore.graph.neighbors(ann, k=11) == explicit
    assert calls == [(200, 1)]


def test_the_index_depth_is_at_least_k_plus_one(store, monkeypatch) -> None:
    datastore, pca = store
    shallow = datastore.graph.ann_index(pca, ann_ef=8)
    calls = _configured(monkeypatch)

    # hnswlib searches at least k + 1 candidates, so this index's own depth
    # for eleven neighbours is 12, which records no ann_ef, while the default
    # depth is 2 * 11.
    own = datastore.graph.neighbors(shallow, k=11, ann_ef=8)
    assert calls == [(12, 1)]
    assert datastore.artifacts.inspect(own).parameters == {
        "k": 11,
        "distance_metric": "l2",
    }
    assert datastore.graph.neighbors(shallow, k=11, ann_ef=12) == own
    default = datastore.graph.neighbors(shallow, k=11)
    assert calls == [(12, 1), (22, 1)]
    assert datastore.artifacts.inspect(default).parameters == {
        "k": 11,
        "distance_metric": "l2",
        "ann_ef": 22,
    }


def test_neighbors_validate_the_requested_depth_before_reading(
    store, monkeypatch
) -> None:
    datastore, pca = store
    ann = datastore.graph.ann_index(pca)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("coordinates must not be read")

    monkeypatch.setattr(datastore, "_coordinate_source", forbidden)
    for value, error in ((0, ValueError), (-5, ValueError), (2.5, TypeError)):
        with pytest.raises(error, match="ann_ef must be"):
            datastore.graph.neighbors(ann, k=5, ann_ef=value)
    with pytest.raises(TypeError, match="ann_ef must be an integer"):
        datastore.graph.neighbors(ann, k=5, ann_ef=True)


def test_ann_ef_changes_queries_but_not_the_index(tmp_path) -> None:
    rng = np.random.default_rng(5)
    values = rng.normal(size=(2_000, 8)).astype(np.float32)
    saved = []
    for ef in (10, 500):
        index = instantiate_knn_index("l2", 8, 2_000, 60, 12, 4466, ef, 1)
        index.add_items(values)
        path = tmp_path / f"index_{ef}.bin"
        index.save_index(str(path))
        saved.append(path.read_bytes())

    # hnswlib builds with ef_construction alone, so the indices are equal.
    assert saved[0] == saved[1]


def test_queries_use_the_datastore_threads_with_identical_results(
    store, monkeypatch
) -> None:
    datastore, pca = store
    calls = _configured(monkeypatch)
    # A serial index is built on one thread whatever the datastore's count.
    ann = datastore.graph.ann_index(pca, ann_m=8, ann_efc=40)
    results = []
    for threads in (1, 2):
        monkeypatch.setattr(datastore, "nthreads", threads)
        calls.clear()
        ref = datastore.graph.neighbors(ann, k=20, ann_ef=21, invalidate_cache=True)
        assert calls == [(21, threads)]
        assert datastore.artifacts.inspect(ref).execution_options["nthreads"] == (
            threads
        )
        group = artifact_group(datastore.zw, ref)
        results.append((group["indices"][:], group["distances"][:]))

    # Each cell's search is independent, so two threads return exactly the
    # neighbors and distances of one.
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])


def test_ann_index_and_initialization_accept_seed_zero(store) -> None:
    datastore, pca = store

    ann = datastore.graph.ann_index(pca, rand_state=0)
    initialization = datastore.embeddings.initialization(
        pca, n_centroids=10, rand_state=0
    )

    assert datastore.artifacts.inspect(ann).parameters["rand_state"] == 0
    assert datastore.artifacts.inspect(initialization).parameters["rand_state"] == 0
    assert ann != datastore.graph.ann_index(pca)
    assert (
        inspect.signature(EmbeddingsAccessor.initialization)
        .parameters["rand_state"]
        .default
        == 4466
    )
    for call in (
        lambda: datastore.graph.ann_index(pca, rand_state=-1),
        lambda: datastore.embeddings.initialization(pca, rand_state=-1),
    ):
        with pytest.raises(ValueError, match="rand_state must be at least 0"):
            call()


def _self_hit_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [
        str(item.message)
        for item in caught
        if "found themselves among their" in str(item.message)
    ]


def test_a_low_self_hit_rate_warns(store) -> None:
    datastore, _pca = store
    # The two genes with the most zeros leave most cells without counts, so
    # their normalized values coincide, and a cell among more than k + 1
    # equal neighbours is often not returned for itself.
    sparsest = np.argsort((_counts() == 0).sum(axis=0))[-2:]
    cells = datastore.snapshot_cell_selection()
    sparse = datastore.features.snapshot(
        from_assay="RNA", feature_indexes=sparsest.tolist()
    )
    normalized = datastore.features.normalize(cells, sparse)
    values = datastore.artifacts.load(normalized)["data"][:]
    duplicated = len(values) - len(np.unique(values, axis=0))
    assert duplicated > N_CELLS // 2
    ann = datastore.graph.ann_index(normalized)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        neighbors = datastore.graph.neighbors(ann, k=3)

    path = datastore.artifacts.inspect(neighbors).path
    recorded = float(datastore.zw[path].attrs["self_hit_rate"])
    assert recorded < 99.5
    (message,) = _self_hit_warnings(caught)
    assert message.startswith(
        f"Only {recorded:.2f}% of cells found themselves among their 4 nearest "
        "neighbours in the ANN index (below 99.5%)"
    )
    # A reused query does not warn again.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert datastore.graph.neighbors(ann, k=3) == neighbors
    assert _self_hit_warnings(caught) == []


def test_a_complete_self_hit_rate_does_not_warn(store) -> None:
    datastore, pca = store
    ann = datastore.graph.ann_index(pca)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        neighbors = datastore.graph.neighbors(ann, k=5)

    path = datastore.artifacts.inspect(neighbors).path
    assert datastore.zw[path].attrs["self_hit_rate"] == 100.0
    assert _self_hit_warnings(caught) == []


def _clustered(n_cells: int, dims: int, clusters: int, seed: int) -> np.ndarray:
    """Well-separated mixture components with a decaying PCA-like spectrum."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(clusters, dims)) * 3.0
    labels = rng.integers(clusters, size=n_cells)
    scales = 1.0 / np.sqrt(np.arange(1, dims + 1))
    values = centers[labels] * scales + rng.normal(size=(n_cells, dims)) * scales
    return values.astype(np.float32)


@pytest.mark.slow
def test_default_index_finds_the_exact_neighbors_of_separated_clusters() -> None:
    # The former defaults, ann_efc=50 and ann_m=48, missed every true
    # neighbour of 48 of these cells (recall 0.9968); exact KNN is the
    # reference.
    n_cells, dims, k = 20_000, 21, 11
    coordinates = _clustered(n_cells, dims, 25, seed=0)
    root, selection, cell_ids, mask = _root_with_selection(n_cells=n_cells)
    imported = write_imported_coordinates(
        root,
        assay="RNA",
        dimreduc_key="pca",
        role="pca",
        coordinates=coordinates,
        source_digest=_SOURCE_DIGEST,
        payload_fingerprints={"data": fingerprint_array(coordinates)},
        source_cell_ids=cell_ids[mask],
        cell_selection=selection,
    )
    datastore = _graph_store(root)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        neighbors = datastore.graph.neighbors(datastore.graph.ann_index(imported), k=k)

    assert _self_hit_warnings(caught) == []
    found = artifact_group(datastore.zw, neighbors)["indices"][:].astype(np.int64)
    exact = (
        NearestNeighbors(n_neighbors=k + 1)
        .fit(coordinates)
        .kneighbors(coordinates, return_distance=False)[:, 1:]
    )
    hits = (found[:, :, None] == exact[:, None, :]).any(axis=2).sum(axis=1)
    assert hits.sum() / (n_cells * k) >= 0.995
    assert np.count_nonzero(hits == 0) / n_cells < 0.0005
