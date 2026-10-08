import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from cytearc import DataStore, mount_datastore
from cytearc.cytebase._embeddings import embedding
from cytearc.embeddings.imported import write_imported_embedding
from cytearc.storage import operation_revisions, validation_scope
from cytearc.storage.artifact_writer import (
    finish_artifact,
    plan_artifact,
    start_artifact,
)
from cytearc.storage.artifacts import ArtifactRef, fingerprint_array
from cytearc.storage.errors import ArtifactResolutionError
from cytearc.storage.operation_revisions import OperationRevision
from tests.storage_helpers import write_count_store
from tests.test_graph_operation_paths import N_CELLS, open_store, write_graph_template


@pytest.fixture
def store():
    datastore = object.__new__(DataStore)
    datastore.z = zarr.open_group(store=MemoryStore(), mode="w")
    datastore.workspace = None
    datastore._defaultAssay = "RNA"
    return datastore


def _record(
    store,
    *,
    kind="embedding",
    operation="run_umap",
    parameters=None,
    inputs=None,
    assay="RNA",
    complete=True,
):
    planned = plan_artifact(
        store.zw,
        scope="datastore" if assay is None else "assay",
        assay=assay,
        kind=kind,
        operation=operation,
        parameters={} if parameters is None else parameters,
        inputs={} if inputs is None else inputs,
        execution_options={},
        invalidate_cache=True,
    )
    group = start_artifact(store.zw, planned)
    if complete:
        finish_artifact(group, planned)
    return planned.ref


def test_find_returns_one_match_with_method_omitted(store):
    expected = _record(store, parameters={"min_dist": 0.25})
    _record(store, operation="run_tsne", parameters={"box_h": 0.7})

    assert store.artifacts.find("embedding", min_dist=0.25) == expected
    assert store.artifacts.find("embedding", method="umap") == expected


def test_find_no_match_reports_available_parameters_and_input_refs(store):
    graph = _record(store, kind="connectivity_map", operation="build_connectivity_map")
    available = _record(store, parameters={"min_dist": 0.25}, inputs={"graph": graph})

    with pytest.raises(ArtifactResolutionError) as caught:
        store.artifacts.find("embedding", method="umap", graph=graph, min_dist=0.5)

    assert caught.value.code == "no_matching_result"
    assert caught.value.context["matches"] == 0
    assert available.artifact_id in str(caught.value)
    assert graph.artifact_id in str(caught.value)
    assert "'min_dist': 0.25" in str(caught.value)


def test_find_ambiguity_includes_every_match_and_can_be_narrowed(store):
    first = _record(store, parameters={"min_dist": 0.25})
    second = _record(store, parameters={"min_dist": 0.5})

    with pytest.raises(ArtifactResolutionError) as caught:
        store.artifacts.find("embedding")

    assert caught.value.code == "ambiguous_result"
    assert caught.value.context["matches"] == 2
    assert first.artifact_id in str(caught.value)
    assert second.artifact_id in str(caught.value)
    assert store.artifacts.find("embedding", min_dist=0.5) == second


def test_find_method_omitted_considers_all_producers(store):
    umap = _record(store)
    tsne = _record(store, operation="run_tsne")

    with pytest.raises(ArtifactResolutionError) as caught:
        store.artifacts.find("embedding")

    assert caught.value.code == "ambiguous_result"
    assert store.artifacts.find("embedding", method="umap") == umap
    assert store.artifacts.find("embedding", method="tsne") == tsne


def test_find_defaults_to_default_assay_and_datastore_scopes(store):
    rna = _record(store)
    adt = _record(store, assay="ADT")
    assert store.artifacts.find("embedding") == rna
    assert store.artifacts.find("embedding", from_assay="ADT") == adt

    integrated = _record(store, assay=None)
    with pytest.raises(ArtifactResolutionError) as caught:
        store.artifacts.find("embedding")

    assert caught.value.code == "ambiguous_result"
    assert caught.value.context["matches"] == 2
    assert rna.artifact_id in str(caught.value)
    assert integrated.artifact_id in str(caught.value)
    assert adt.artifact_id not in str(caught.value)
    assert store.artifacts.find("embedding", from_assay="RNA") == rna


@pytest.mark.parametrize(
    ("result", "method", "kind", "operation", "input_name"),
    [
        ("embedding", "umap", "embedding", "run_umap", "graph"),
        ("clusters", "leiden", "cluster_labels", "run_leiden_clustering", "graph"),
        ("clusters", "paris", "cluster_cut", "cut_paris_hierarchy", "connectivity_map"),
    ],
)
def test_find_graph_filter_uses_integrated_scope_and_correct_input_name(
    store, result, method, kind, operation, input_name
):
    graph = _record(
        store, kind="integrated_graph", operation="integrate_assays", assay=None
    )
    other_graph = _record(
        store, kind="integrated_graph", operation="integrate_assays", assay=None
    )
    expected = _record(
        store, kind=kind, operation=operation, inputs={input_name: graph}, assay=None
    )
    _record(
        store,
        kind=kind,
        operation=operation,
        inputs={input_name: other_graph},
        assay=None,
    )
    _record(store, kind=kind, operation=operation, inputs={input_name: graph})

    assert store.artifacts.find(result, method=method, graph=graph) == expected


@pytest.mark.parametrize("method", ["wnn", "snn"])
def test_find_integrated_graph_method_filters_recorded_method(store, method):
    refs = {
        name: _record(
            store,
            kind="integrated_graph",
            operation="integrate_assays",
            parameters={"method": name},
            assay=None,
        )
        for name in ("wnn", "snn")
    }
    assert store.artifacts.find("graph", method=method) == refs[method]
    with pytest.raises(ArtifactResolutionError, match="2 complete graph results"):
        store.artifacts.find("graph")


def test_find_excludes_incomplete_and_imported_records(store):
    incomplete = _record(store, complete=False)
    imported = _record(store, operation="import_embedding")
    assert set(store.artifacts.list(kind="embedding")) == {incomplete, imported}

    with pytest.raises(ArtifactResolutionError) as caught:
        store.artifacts.find("embedding")

    assert caught.value.code == "no_matching_result"
    assert incomplete.artifact_id not in str(caught.value)
    assert imported.artifact_id not in str(caught.value)
    expected = _record(store)
    assert store.artifacts.find("embedding") == expected


def test_find_keeps_superseded_computed_results_as_candidates(store, monkeypatch):
    old = _record(store)
    monkeypatch.setattr(
        operation_revisions,
        "OPERATION_REVISIONS",
        {
            "run_umap": (
                OperationRevision(2, "1.0.0", "Updated embedding computation.", None),
            )
        },
    )
    assert not store.artifacts.inspect(old).is_current
    assert store.artifacts.find("embedding") == old
    current = _record(store)
    assert store.artifacts.inspect(current).is_current

    with pytest.raises(ArtifactResolutionError) as caught:
        store.artifacts.find("embedding")

    assert caught.value.code == "ambiguous_result"
    assert old.artifact_id in str(caught.value)
    assert current.artifact_id in str(caught.value)
    assert "revision=1" in str(caught.value)
    assert "revision=2" in str(caught.value)


@pytest.mark.parametrize(
    ("result", "method", "kind", "operation", "parameter"),
    [
        ("clusters", "leiden", "cluster_labels", "run_leiden_clustering", "resolution"),
        *[
            ("embedding", "umap", "embedding", "run_umap", parameter)
            for parameter in (
                "spread",
                "min_dist",
                "repulsion_strength",
                "initial_alpha",
                "negative_sample_rate",
                "dens_lambda",
                "dens_frac",
                "dens_var_shift",
            )
        ],
        *[
            ("embedding", "tsne", "embedding", "run_tsne", parameter)
            for parameter in ("lambda_scale", "box_h")
        ],
        *[
            ("graph", None, "connectivity_map", "build_connectivity_map", parameter)
            for parameter in ("local_connectivity", "bandwidth")
        ],
    ],
)
def test_find_float_filters_accept_integers_while_list_stays_exact(
    store, result, method, kind, operation, parameter
):
    expected = _record(
        store, kind=kind, operation=operation, parameters={parameter: 1.0}
    )
    assert store.artifacts.find(result, method=method, **{parameter: 1}) == expected
    assert store.artifacts.find(result, method=method, **{parameter: 1.0}) == expected
    assert store.artifacts.list(kind=kind, parameters={parameter: 1}) == []
    assert store.artifacts.list(kind=kind, parameters={parameter: 1.0}) == [expected]


@pytest.mark.parametrize("filters", [{"umap_dims": 2.0}, {"parallel": 0}])
def test_find_preserves_integer_and_boolean_parameter_identity(store, filters):
    expected = _record(store, parameters={"umap_dims": 2, "parallel": False})
    assert store.artifacts.find("embedding", umap_dims=2, parallel=False) == expected
    with pytest.raises(ArtifactResolutionError) as caught:
        store.artifacts.find("embedding", method="umap", **filters)
    assert caught.value.code == "no_matching_result"


@pytest.mark.parametrize("imported", [None, "X_umap"])
@pytest.mark.parametrize("outcome", ["one", "none", "ambiguous"])
def test_find_shares_validation_scope_and_resets_it(
    store, monkeypatch, outcome, imported
):
    operation = "run_umap" if imported is None else "import_dimreduc"
    parameters = {} if imported is None else {"dimreduc_key": imported}
    expected = _record(
        store, operation=operation, parameters={**parameters, "min_dist": 0.25}
    )
    if outcome == "ambiguous":
        _record(store, operation=operation, parameters={**parameters, "min_dist": 0.5})
    original_list = store._artifacts_list
    original_inspect = store._artifacts_inspect
    scopes = []
    inspected = []

    def list_with_scope(**kwargs):
        scopes.append(validation_scope._RESULTS.get())
        return original_list(**kwargs)

    def inspect_with_scope(ref):
        scopes.append(validation_scope._RESULTS.get())
        inspected.append(ref)
        return original_inspect(ref)

    monkeypatch.setattr(store, "_artifacts_list", list_with_scope)
    monkeypatch.setattr(store, "_artifacts_inspect", inspect_with_scope)
    assert validation_scope._RESULTS.get() is None
    if outcome == "one":
        assert store.artifacts.find("embedding", imported=imported) == expected
    else:
        with pytest.raises(ArtifactResolutionError):
            filters = {"min_dist": 0.75} if outcome == "none" else {}
            store.artifacts.find("embedding", imported=imported, **filters)
        assert expected in inspected
    assert len(scopes) > 1
    assert scopes[0] is not None
    assert all(scope is scopes[0] for scope in scopes)
    assert validation_scope._RESULTS.get() is None


@pytest.fixture(scope="module")
def produced_store(tmp_path_factory):
    location = tmp_path_factory.mktemp("discovery") / "store.zarr"
    refs = write_graph_template(location)
    datastore = open_store(location)
    refs["connectivity"] = datastore.graph.connectivity(
        refs["rna_neighbors"], local_connectivity=1, bandwidth=1
    )
    refs["clusters"] = datastore.clusters.leiden(refs["wnn"], resolution=1)
    initialization = np.random.default_rng(82).normal(size=(N_CELLS, 2))
    refs["umap"] = datastore.embeddings.umap(
        refs["wnn"], initialization, n_epochs=10, min_dist=1, spread=2, nthreads=1
    )
    return datastore, refs


@pytest.mark.parametrize(
    ("result", "method", "key", "parameters"),
    [
        ("clusters", "leiden", "clusters", {"resolution": 1}),
        ("embedding", "umap", "umap", {"min_dist": 1, "spread": 2}),
        ("graph", None, "connectivity", {"local_connectivity": 1, "bandwidth": 1}),
    ],
)
def test_find_matches_real_producer_normalization(
    produced_store, result, method, key, parameters
):
    datastore, refs = produced_store
    expected = refs[key]
    filters = {"from_assay": "RNA"} if result == "graph" else {"graph": refs["wnn"]}
    recorded = datastore.artifacts.inspect(expected).parameters
    for parameter, value in parameters.items():
        assert type(recorded[parameter]) is float
        assert recorded[parameter] == float(value)
    assert (
        datastore.artifacts.find(result, method=method, **filters, **parameters)
        == expected
    )
    assert (
        datastore.artifacts.list(
            kind=expected.kind,
            scope=expected.scope,
            from_assay=expected.assay,
            parameters=parameters,
        )
        == []
    )


def test_find_reads_real_results_from_read_only_store(produced_store):
    datastore, refs = produced_store
    read_only = open_store(datastore.zarr_loc, zarr_mode="r")
    assert read_only.zw.read_only
    assert (
        read_only.artifacts.find(
            "clusters", graph=refs["wnn"], method="leiden", resolution=1
        )
        == refs["clusters"]
    )
    assert (
        read_only.artifacts.find(
            "embedding", graph=refs["wnn"], method="umap", min_dist=1, spread=2
        )
        == refs["umap"]
    )


def _import_embedding(store, *, key="X_umap", assay="RNA", token=0, dims=2):
    coordinates = np.arange(store.cells.N * dims, dtype=np.float32).reshape(
        store.cells.N, dims
    )
    return write_imported_embedding(
        store.zw,
        assay=assay,
        dimreduc_key=key,
        role="umap",
        coordinates=coordinates,
        source_digest=bytes([token]) * 32,
        payload_fingerprints={"values": fingerprint_array(coordinates)},
        source_cell_ids=store.cells.fetch_all("ids"),
        cell_selection=store.snapshot_cell_selection(),
    )


@pytest.fixture
def imported_store(tmp_path):
    location = tmp_path / "source.zarr"
    write_count_store(
        str(location),
        {"RNA": np.ones((6, 4)), "ADT": np.ones((6, 3))},
        "uint16",
    )
    datastore = open_store(location)
    return datastore, _import_embedding(datastore)


def test_find_imported_embedding_uses_source_key_assay_and_exact_parameters(
    imported_store,
):
    datastore, expected = imported_store
    adt = _import_embedding(datastore, assay="ADT")
    assert datastore.artifacts.find("embedding", imported="X_umap") == expected
    assert (
        datastore.artifacts.find(
            "embedding", imported="X_umap", from_assay="RNA", role="umap", dims=2
        )
        == expected
    )
    assert (
        datastore.artifacts.find("embedding", imported="X_umap", from_assay="ADT")
        == adt
    )
    assert embedding(datastore) == expected
    assert embedding(datastore, assay="ADT") == adt
    loaded = datastore.artifacts.load_values(expected)
    np.testing.assert_array_equal(
        loaded.values,
        np.arange(12, dtype=np.float32).reshape(6, 2),
    )
    assert loaded.cell_ids.tolist() == [f"cell{index}" for index in range(6)]
    with pytest.raises(ArtifactResolutionError) as caught:
        datastore.artifacts.find("embedding", imported="X_umap", dims=2.0)
    assert caught.value.code == "no_matching_result"


def test_find_imported_embedding_reports_missing_and_excludes_computed_results(
    imported_store,
):
    datastore, imported = imported_store
    computed = _record(datastore, parameters={"dimreduc_key": "X_missing"})
    incomplete = _record(
        datastore,
        operation="import_dimreduc",
        parameters={"dimreduc_key": "X_missing"},
        complete=False,
    )
    assert datastore.artifacts.find("embedding") == computed
    assert datastore.artifacts.find("embedding", imported="X_umap") == imported
    with pytest.raises(ArtifactResolutionError) as caught:
        datastore.artifacts.find("embedding", imported="X_missing")
    assert caught.value.code == "no_matching_result"
    assert caught.value.context["imported"] == "X_missing"
    assert "imported='X_missing'" in str(caught.value)
    assert imported.artifact_id in str(caught.value)
    assert computed.artifact_id not in str(caught.value)
    assert incomplete.artifact_id not in str(caught.value)


def test_find_imported_embedding_ambiguity_keeps_all_refs_and_can_be_narrowed(
    imported_store,
):
    datastore, first = imported_store
    second = _import_embedding(datastore, token=1, dims=3)
    with pytest.raises(ArtifactResolutionError) as caught:
        datastore.artifacts.find("embedding", imported="X_umap")
    assert caught.value.code == "ambiguous_result"
    assert caught.value.context["matches"] == 2
    assert first.artifact_id in str(caught.value)
    assert second.artifact_id in str(caught.value)
    assert datastore.artifacts.find("embedding", imported="X_umap", dims=3) == second
    with pytest.raises(ValueError, match="ambiguous"):
        embedding(datastore)


def test_find_imported_embedding_reads_mount_source_and_reports_target_ambiguity(
    imported_store, tmp_path
):
    source, original = imported_store
    target = tmp_path / "analysis.zarr"
    mounted = mount_datastore(
        source.zarr_loc,
        at=str(target),
        default_assay="RNA",
        min_features_per_cell=-1,
        nthreads=1,
    )
    reopened = open_store(target, zarr_mode="r")
    assert reopened.zw.read_only
    assert reopened.artifacts.find("embedding", imported="X_umap") == original
    loaded = reopened.artifacts.load_values(original)
    np.testing.assert_array_equal(
        loaded.values,
        np.arange(12, dtype=np.float32).reshape(6, 2),
    )
    assert loaded.cell_ids.tolist() == [f"cell{index}" for index in range(6)]
    local = _import_embedding(mounted, token=1)
    reopened = open_store(target, zarr_mode="r")
    with pytest.raises(ArtifactResolutionError) as caught:
        reopened.artifacts.find("embedding", imported="X_umap")
    assert caught.value.code == "ambiguous_result"
    assert original.artifact_id in str(caught.value)
    assert local.artifact_id in str(caught.value)
    assert source.artifacts.find("embedding", imported="X_umap") == original


@pytest.mark.parametrize("imported", [False, 1, ["X_umap"], b"X_umap"])
def test_find_imported_embedding_requires_a_string(store, imported):
    with pytest.raises(TypeError, match="imported must be a string"):
        store.artifacts.find("embedding", imported=imported)


def test_find_imported_embedding_rejects_empty_source_key(store):
    with pytest.raises(ValueError, match="non-empty source embedding key"):
        store.artifacts.find("embedding", imported="")


@pytest.mark.parametrize("result", ["clusters", "graph"])
def test_find_imported_only_supports_embeddings(store, result):
    with pytest.raises(ValueError, match="only supported for embeddings"):
        store.artifacts.find(result, imported="X_umap")


@pytest.mark.parametrize(
    "filters",
    [
        {"method": "umap"},
        {
            "graph": ArtifactRef(
                scope="assay",
                assay="RNA",
                kind="connectivity_map",
                artifact_id="a" * 64,
            )
        },
    ],
)
def test_find_imported_embedding_rejects_computed_selectors(store, filters):
    with pytest.raises(ValueError, match="cannot be combined with graph= or method="):
        store.artifacts.find("embedding", imported="X_umap", **filters)


@pytest.mark.parametrize("dimreduc_key", ["X_umap", "X_other"])
def test_find_imported_embedding_does_not_overwrite_an_explicit_source_key(
    store, dimreduc_key
):
    with pytest.raises(ValueError, match="Use imported= to select"):
        store.artifacts.find("embedding", imported="X_umap", dimreduc_key=dimreduc_key)
