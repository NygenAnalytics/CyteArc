"""Map a query onto a reference by feature name when the two use other IDs."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cytearc.datastore.datastore import DataStore
from cytearc.mapping.projection import NO_QUERY_BATCH_FINGERPRINT, plan_projection
from cytearc.mapping.reference import MappingReference, reference_feature_keys
from cytearc.storage.artifacts import ArtifactRef, artifact_group
from cytearc.storage.errors import ArtifactResolutionError
from cytearc.storage.feature_selection import resolve_feature_selection
from tests.storage_helpers import write_count_store
from tests.test_mapping_reference import (
    MAPPING_N_GENES,
    MAPPING_REFERENCE_GENES,
    MappingStoreSource,
    build_mapping_store,
    mapping_counts,
)

# The reference names each gene by an Ensembl-style ID and by a symbol. Its
# feature selection keeps the first 26 genes, whose symbols are unique; two of
# the genes outside it share the symbol SHARED.
IDS = [f"ENSG{index:011d}" for index in range(MAPPING_N_GENES)]
SYMBOLS = [f"GENE{index}" for index in range(MAPPING_N_GENES)]
REFERENCE_NAMES = [*SYMBOLS]
REFERENCE_NAMES[26] = REFERENCE_NAMES[27] = "SHARED"
SELECTED = SYMBOLS[:MAPPING_REFERENCE_GENES]


@pytest.fixture(scope="module")
def named_source(tmp_path_factory) -> tuple[MappingStoreSource, ArtifactRef]:
    """Return the reference store and a second reference whose names repeat.

    The second reference selects 20 of the genes and both genes named SHARED.
    """
    path = tmp_path_factory.mktemp("mapping_feature_names") / "reference.zarr"
    source = build_mapping_store(
        path,
        with_references=True,
        with_symphony=False,
        feature_ids=IDS,
        feature_names=REFERENCE_NAMES,
    )
    store = DataStore(str(path), default_assay="RNA")
    features = store.features.snapshot(
        from_assay="RNA", feature_indexes=[*range(20), 26, 27]
    )
    reduction = store.reduction.pca(
        store.features.normalize(source.cells, features), dims=4, local_cache=False
    )
    neighbors = store.graph.neighbors(
        store.graph.ann_index(reduction), coordinates=reduction, k=5
    )
    return source, store.mapping.build_reference(neighbors)


def _reference(
    source: MappingStoreSource, ref: ArtifactRef | None = None
) -> MappingReference:
    store = DataStore(str(source.path), default_assay="RNA", zarr_mode="r")
    return store.mapping.load_reference(source.reference if ref is None else ref)


def _query(
    path: Path, counts: np.ndarray, ids: list[str], names: list[str]
) -> tuple[DataStore, ArtifactRef]:
    write_count_store(
        str(path),
        {"RNA": counts},
        "uint32",
        feature_ids={"RNA": ids},
        feature_names={"RNA": names},
    )
    query = DataStore(str(path), default_assay="RNA", min_features_per_cell=0)
    return query, query.snapshot_cell_selection("I")


def _store_files(path: str) -> dict[str, bytes]:
    return {
        str(file.relative_to(path)): file.read_bytes()
        for file in Path(path).rglob("*")
        if file.is_file()
    }


def _overlap(query: DataStore, projection: ArtifactRef) -> ArtifactRef:
    inputs = query.artifacts.inspect(projection).inputs or {}
    return ArtifactRef.from_dict(inputs["feature_selection"])


def _assert_same_projection(
    first: tuple[DataStore, ArtifactRef],
    second: tuple[DataStore, ArtifactRef],
    reference: MappingReference,
) -> None:
    results = [
        store.mapping.load_result(ref, reference=reference, load_arrays=True)
        for store, ref in (first, second)
    ]
    for name in ("indices", "distances", "uninformative"):
        np.testing.assert_array_equal(
            getattr(results[0], name), getattr(results[1], name)
        )
    assert results[0].diagnostics == results[1].diagnostics


def test_names_map_a_query_whose_ids_follow_another_scheme(named_source, tmp_path):
    source, _ = named_source
    reference = _reference(source)
    counts = mapping_counts()
    by_id = _query(tmp_path / "ids.zarr", counts, IDS, SYMBOLS)
    by_name = _query(tmp_path / "symbols.zarr", counts, SYMBOLS, SYMBOLS)
    query, cells = by_name

    with pytest.raises(ValueError, match="pass feature_key='names'"):
        query.mapping.run(reference, cells)
    id_projection = by_id[0].mapping.run(reference, by_id[1])
    name_projection = query.mapping.run(reference, cells, feature_key="names")

    _assert_same_projection(
        (by_id[0], id_projection), (query, name_projection), reference
    )
    transfers = [
        store.mapping.load_label_transfer(
            store.mapping.transfer_labels(
                projection, reference=reference, reference_labels="mapping_batch"
            )
        )
        for store, projection in ((by_id[0], id_projection), (query, name_projection))
    ]
    pd.testing.assert_frame_equal(transfers[0].evidence, transfers[1].evidence)

    status = query.artifacts.inspect(name_projection)
    assert status.parameters == {
        "save_k": 3,
        "missing_feature_policy": "reference_mean",
        "correction_method": "none",
        "feature_key": "names",
    }
    overlap = _overlap(query, name_projection)
    assert query.artifacts.inspect(overlap).parameters == {"feature_key": "names"}
    np.testing.assert_array_equal(
        artifact_group(query.zw, overlap)["values"][:],
        np.isin(SYMBOLS, SELECTED),
    )
    # A read-only store returns the saved match by name.
    read_only = DataStore(str(query.zarr_loc), default_assay="RNA", zarr_mode="r")
    assert read_only.mapping.run(reference, cells, feature_key="names") == (
        name_projection
    )


def test_query_features_that_share_a_name_add_their_counts(named_source, tmp_path):
    source, _ = named_source
    reference = _reference(source)
    counts = mapping_counts()
    split = [3, 10]
    # Each of two genes splits its counts between two query features named
    # after it, the second of them appended after the other genes.
    first_part = np.random.default_rng(5).integers(0, counts[:, split] + 1)
    split_counts = np.column_stack([counts, counts[:, split] - first_part])
    split_counts[:, split] = first_part
    names = [*SYMBOLS, *(SYMBOLS[index] for index in split)]
    split_query = _query(
        tmp_path / "split.zarr",
        split_counts,
        [f"query-{index}" for index in range(len(names))],
        names,
    )
    # The other query holds each gene's counts summed by hand.
    summed = split_counts[:, :MAPPING_N_GENES].copy()
    summed[:, split] += split_counts[:, MAPPING_N_GENES:]
    np.testing.assert_array_equal(summed, counts)
    summed_query = _query(tmp_path / "summed.zarr", summed, SYMBOLS, SYMBOLS)

    projections = [
        (store, store.mapping.run(reference, cells, feature_key="names"))
        for store, cells in (split_query, summed_query)
    ]

    _assert_same_projection(projections[0], projections[1], reference)
    store, projection = projections[0]
    result = store.mapping.load_result(projection, reference=reference)
    assert result.diagnostics["featureCoverage"] == 1.0
    np.testing.assert_array_equal(
        artifact_group(store.zw, _overlap(store, projection))["values"][:],
        np.isin(names, SELECTED),
    )


def test_reference_features_that_share_a_name_are_refused(named_source, tmp_path):
    source, shared = named_source
    reference = _reference(source, shared)
    query, cells = _query(tmp_path / "query.zarr", mapping_counts(), IDS, SYMBOLS)
    before = _store_files(query.zarr_loc)

    with pytest.raises(
        ValueError, match=re.escape(f"'SHARED' names {IDS[26]}, {IDS[27]}.")
    ):
        query.mapping.run(reference, cells, feature_key="names")

    assert _store_files(query.zarr_loc) == before
    # The IDs still tell every reference feature apart.
    projection = query.mapping.run(reference, cells)
    result = query.mapping.load_result(projection, reference=reference)
    assert result.diagnostics["featureCoverage"] == 1.0


@pytest.mark.parametrize("policy", ["reference_mean", "zero", "error"])
def test_reference_names_missing_from_the_query_follow_the_policy(
    named_source, tmp_path, policy
):
    source, _ = named_source
    reference = _reference(source)
    # The query measures 20 of the 26 reference genes.
    counts = mapping_counts()[:, :20]
    by_id = _query(tmp_path / "ids.zarr", counts, IDS[:20], SYMBOLS[:20])
    by_name = _query(tmp_path / "symbols.zarr", counts, SYMBOLS[:20], SYMBOLS[:20])
    calls = (
        (by_id, {}),
        (by_name, {"feature_key": "names"}),
    )

    if policy == "error":
        for (store, cells), key in calls:
            with pytest.raises(ValueError, match="missing 6 required reference"):
                store.mapping.run(
                    reference, cells, missing_feature_policy=policy, **key
                )
        return
    projections = []
    for (store, cells), key in calls:
        # The reference renormalizes each cell over its features.
        with pytest.warns(UserWarning, match=r"measures only 76\.9% of them"):
            projection = store.mapping.run(
                reference, cells, missing_feature_policy=policy, **key
            )
        projections.append((store, projection))

    _assert_same_projection(projections[0], projections[1], reference)
    store, projection = projections[1]
    result = store.mapping.load_result(projection, reference=reference)
    assert result.diagnostics["featureCoverage"] == 20 / 26


def test_matching_by_id_keeps_the_projection_identity(named_source, tmp_path):
    source, _ = named_source
    reference = _reference(source)
    query, cells = _query(tmp_path / "query.zarr", mapping_counts(), IDS, SYMBOLS)

    by_default = query.mapping.run(reference, cells)
    assert query.mapping.run(reference, cells, feature_key="ids") == by_default
    by_name = query.mapping.run(reference, cells, feature_key="names")

    # A match by ID records what projections recorded before features could
    # match by name, so their identities hold.
    assert query.artifacts.inspect(by_default).parameters == {
        "save_k": 3,
        "missing_feature_policy": "reference_mean",
        "correction_method": "none",
    }
    assert query.artifacts.inspect(_overlap(query, by_default)).parameters == {}
    # The IDs and the names select the same query features, yet a match by
    # name is a separate projection with a separate overlap.
    assert by_name != by_default
    assert _overlap(query, by_name) != _overlap(query, by_default)
    np.testing.assert_array_equal(
        artifact_group(query.zw, _overlap(query, by_name))["values"][:],
        artifact_group(query.zw, _overlap(query, by_default))["values"][:],
    )
    _assert_same_projection((query, by_default), (query, by_name), reference)


@pytest.mark.parametrize(
    ("feature_key", "message"),
    [
        ("ids", "Projection feature_key is unsupported"),
        (["names"], "Projection feature_key is unsupported"),
        (None, "matches features by another key than the projection"),
    ],
)
def test_projection_loader_rejects_a_feature_key_that_its_overlap_lacks(
    named_source, tmp_path, feature_key, message
):
    source, _ = named_source
    reference = _reference(source)
    query, cells = _query(tmp_path / "query.zarr", mapping_counts(), SYMBOLS, SYMBOLS)
    projection = query.mapping.run(reference, cells, feature_key="names")
    group = artifact_group(query.zw, projection)
    provenance = dict(group.attrs["provenance"])
    parameters = dict(provenance["parameters"])
    if feature_key is None:
        del parameters["feature_key"]
    else:
        parameters["feature_key"] = feature_key
    provenance["parameters"] = parameters
    group.attrs["provenance"] = provenance

    with pytest.raises(ValueError, match=re.escape(message)):
        query.mapping.load_result(projection, reference=reference)


@pytest.mark.parametrize("parameters", [{"feature_key": "ids"}, {"key": "names"}])
def test_overlap_selection_records_only_a_match_by_name(
    named_source, tmp_path, parameters
):
    source, _ = named_source
    reference = _reference(source)
    query, cells = _query(tmp_path / "query.zarr", mapping_counts(), SYMBOLS, SYMBOLS)
    overlap = _overlap(query, query.mapping.run(reference, cells, feature_key="names"))
    resolve_feature_selection(query.zw, "RNA", overlap)
    group = artifact_group(query.zw, overlap)
    provenance = dict(group.attrs["provenance"])
    provenance["parameters"] = parameters
    group.attrs["provenance"] = provenance

    with pytest.raises(
        ArtifactResolutionError, match="provenance does not match its operation"
    ):
        resolve_feature_selection(query.zw, "RNA", overlap)


@pytest.mark.parametrize(
    ("feature_key", "error", "message"),
    [
        (1, TypeError, "feature_key must be a string"),
        ("symbols", ValueError, "feature_key must be 'ids' or 'names'"),
        ("IDS", ValueError, "feature_key must be 'ids' or 'names'"),
    ],
)
def test_mapping_run_rejects_unknown_feature_keys(
    named_source, tmp_path, feature_key, error, message
):
    source, _ = named_source
    reference = _reference(source)
    query, cells = _query(tmp_path / "query.zarr", mapping_counts(), IDS, SYMBOLS)

    with pytest.raises(error, match=re.escape(message)):
        query.mapping.run(reference, cells, feature_key=feature_key)


def test_reference_feature_keys_read_the_selected_names(named_source):
    source, shared = named_source
    reference = _reference(source)

    assert reference_feature_keys(reference, "ids") is reference.feature_ids
    names = reference_feature_keys(reference, "names")
    np.testing.assert_array_equal(names, SELECTED)
    assert not names.flags.writeable
    with pytest.raises(ValueError, match="feature_key must be 'ids' or 'names'"):
        reference_feature_keys(reference, "symbols")
    with pytest.raises(ValueError, match="share names"):
        reference_feature_keys(_reference(source, shared), "names")


def test_shared_reference_names_are_listed_with_their_ids(named_source, monkeypatch):
    source, _ = named_source
    reference = _reference(source)
    # Thirteen names, each shared by two adjacent reference features.
    names = np.asarray([f"PAIR{index // 2}" for index in range(26)])

    def blocks(*_args, **_kwargs):
        yield names, np.ones(len(names), dtype=bool)

    monkeypatch.setattr(
        "cytearc.mapping.reference.iter_feature_selection_blocks", blocks
    )
    with pytest.raises(ValueError, match="share names") as caught:
        reference_feature_keys(reference, "names")
    message = str(caught.value)
    # The first five shared names in sorted order, each with its features' IDs.
    assert f"apart: 'PAIR0' names {IDS[0]}, {IDS[1]}; 'PAIR1' names" in message
    assert f"'PAIR12' names {IDS[24]}, {IDS[25]}; and 8 more. Map by" in message

    def short_blocks(*_args, **_kwargs):
        yield names[:25], np.ones(25, dtype=bool)

    monkeypatch.setattr(
        "cytearc.mapping.reference.iter_feature_selection_blocks", short_blocks
    )
    with pytest.raises(ValueError, match="feature IDs do not match the selected"):
        reference_feature_keys(reference, "names")


def test_projection_plan_rejects_unknown_feature_keys(named_source, tmp_path):
    source, _ = named_source
    reference = _reference(source)
    query, cells = _query(tmp_path / "query.zarr", mapping_counts(), IDS, SYMBOLS)
    overlap = _overlap(query, query.mapping.run(reference, cells))

    with pytest.raises(ValueError, match="feature_key must be 'ids' or 'names'"):
        plan_projection(
            query.zw,
            query_assay="RNA",
            n_cells=int(query.cells.N),
            save_k=3,
            missing_feature_policy="reference_mean",
            correction_method="none",
            cell_selection=cells,
            feature_selection=overlap,
            query_dataset_fingerprint=query._ensure_dataset_fingerprint("RNA"),
            query_batch_fingerprint=NO_QUERY_BATCH_FINGERPRINT,
            query_batch_count=1,
            mapping_reference=reference.external_ref,
            reference=reference,
            reference_cell_count=reference.selected_cell_count,
            feature_key="symbols",
        )
