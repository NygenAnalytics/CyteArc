"""Ordered row IDs are fingerprinted once while their stored bytes are unchanged.

Every public call validates its cell selection against the current ordered cell
IDs. The fingerprint of the IDs is reused only while every stored byte that
decides their values is the same, so each way of changing the IDs is still
detected: an in-place chunk write, a rewritten array, a deleted chunk, a change
made through another handle, and a write while the fingerprint is computed.
"""

from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore, MemoryStore

import cytearc.storage.artifacts as artifacts_module
import cytearc.storage.content_stamps as stamps_module
from cytearc.storage.artifacts import fingerprint_stored_strings, fingerprint_strings
from cytearc.storage.content_stamps import (
    cached_for_content,
    clear_content_cache,
    stored_content_stamp,
)
from cytearc.storage.errors import ArtifactResolutionError
from cytearc.storage.selections import (
    resolve_generated_selection_artifact,
    validate_cell_selection,
)
from cytearc.storage.validation_scope import validation_scope


@pytest.fixture
def decodes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the number of values each full fingerprint of stored text decodes."""
    calls: list[int] = []
    original = artifacts_module.fingerprint_text_blocks

    def counting(n_values, source_dtype, blocks):  # type: ignore[no-untyped-def]
        calls.append(int(n_values))
        return original(n_values, source_dtype, blocks)

    monkeypatch.setattr(artifacts_module, "fingerprint_text_blocks", counting)
    return calls


def _ids(n: int = 50, offset: int = 0) -> np.ndarray:
    return np.asarray([f"cell{index + offset:05d}" for index in range(n)])


def _cell_table(store: object | None = None) -> zarr.Group:
    root = zarr.open_group(store=MemoryStore() if store is None else store, mode="w")
    cells = root.create_group("cellData")
    cells.create_array("ids", data=_ids(), chunks=(8,))
    return root


def _selection(root: zarr.Group):  # type: ignore[no-untyped-def]
    values = np.zeros(50, dtype=bool)
    values[::3] = True
    return resolve_generated_selection_artifact(
        root,
        scope="datastore",
        kind="cell_selection",
        values=values,
        row_ids=root["cellData/ids"],
        operation="test_selection",
        parameters={},
        inputs={},
        source_column="I",
    )[0]


def _validate(root: zarr.Group, ref) -> None:  # type: ignore[no-untyped-def]
    # Each public call opens its own validation scope.
    with validation_scope():
        validate_cell_selection(root, ref)


def _assert_row_identity_mismatch(root: zarr.Group, ref) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ArtifactResolutionError) as caught:
        _validate(root, ref)
    assert caught.value.code == "row_identity_mismatch"


def test_reused_fingerprint_equals_a_fresh_one(decodes: list[int]) -> None:
    root = _cell_table()
    ids = root["cellData/ids"]
    expected = fingerprint_strings(_ids())

    assert fingerprint_stored_strings(ids) == expected
    assert fingerprint_stored_strings(root["cellData/ids"]) == expected
    # Only the first call decoded the IDs.
    assert decodes == [50]


def test_each_public_call_validates_without_decoding_the_ids_again(
    decodes: list[int],
) -> None:
    root = _cell_table()
    ref = _selection(root)
    decodes.clear()

    for _ in range(3):
        _validate(root, ref)

    # Creating the selection fingerprinted the IDs as they are stored.
    assert decodes == []


def test_pipeline_views_check_row_identity_without_decoding_the_ids(
    decodes: list[int],
) -> None:
    from tests.test_pipeline_run_foundation import _completed_run, _root

    root = _root()
    run = _completed_run(root)
    decodes.clear()

    # Before, building a view hashed the live IDs three times and every read
    # hashed them again.
    cells = run.cells
    features = run.features
    cells.fetch("clusters")
    cells.head(1)
    cells.to_pandas_dataframe(["ids", "clusters"])
    features.fetch_all("names")
    assert decodes == []

    root["cellData/ids"][:] = np.asarray(["c2", "c1", "c3", "c4"])
    with pytest.raises(ArtifactResolutionError) as caught:
        cells.fetch("clusters")
    assert caught.value.code == "row_identity_mismatch"
    assert decodes == [4]


def test_in_place_chunk_writes_are_detected(decodes: list[int]) -> None:
    root = _cell_table()
    ref = _selection(root)
    _validate(root, ref)

    # A raw Zarr write keeps the metadata document and replaces one chunk.
    reordered = _ids()
    reordered[[0, 1]] = reordered[[1, 0]]
    root["cellData/ids"][:] = reordered
    _assert_row_identity_mismatch(root, ref)

    # Writing the original values back restores the identity.
    root["cellData/ids"][:] = _ids()
    _validate(root, ref)


def test_a_rewritten_ids_array_is_detected() -> None:
    root = _cell_table()
    ref = _selection(root)
    _validate(root, ref)
    cells = root["cellData"]

    # Same shape, dtype, chunks, and codecs; other values.
    del cells["ids"]
    cells.create_array("ids", data=_ids(offset=1), chunks=(8,))
    _assert_row_identity_mismatch(root, ref)

    # The same values encoded differently keep the identity.
    del cells["ids"]
    cells.create_array("ids", data=_ids(), chunks=(5,), compressors=None)
    _validate(root, ref)


def test_a_deleted_chunk_is_detected() -> None:
    from zarr.core.sync import sync

    store = MemoryStore()
    root = _cell_table(store)
    ref = _selection(root)
    _validate(root, ref)

    # A lost chunk object reads as empty IDs.
    sync(store.delete("cellData/ids/c/2"))
    _assert_row_identity_mismatch(root, ref)


def test_a_prepared_store_detects_ids_changed_outside_cytearc(
    datastore_ephemeral,  # type: ignore[no-untyped-def]
) -> None:
    store = datastore_ephemeral
    selection = store.snapshot_cell_selection("I")
    assert store.select_measured_cells("RNA", cell_selection=selection) == selection

    # Prepared IDs are protected from CyteArc's own writes, so only a raw
    # write through another handle can change them.
    raw = zarr.open_group(store.zw.store_path.store.root, mode="r+")
    ids = np.asarray(raw["cellData/ids"][:])
    raw["cellData/ids"][:] = np.roll(ids, 1)
    with pytest.raises(ArtifactResolutionError) as caught:
        store.select_measured_cells("RNA", cell_selection=selection)
    assert caught.value.code == "row_identity_mismatch"


def test_changes_made_through_another_handle_are_detected(tmp_path: Path) -> None:
    root = _cell_table(LocalStore(str(tmp_path / "store.zarr")))
    ref = _selection(root)
    _validate(root, ref)

    other = zarr.open_group(str(tmp_path / "store.zarr"), mode="r+")
    reversed_ids = _ids()[::-1].copy()
    other["cellData/ids"][:] = reversed_ids
    _assert_row_identity_mismatch(root, ref)


def test_a_write_while_the_fingerprint_is_computed_is_never_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _cell_table()
    ids = root["cellData/ids"]
    original = artifacts_module.fingerprint_text_blocks
    writes = iter([_ids(offset=7)])

    def write_during(n_values, source_dtype, blocks):  # type: ignore[no-untyped-def]
        digest = original(n_values, source_dtype, blocks)
        replacement = next(writes, None)
        if replacement is not None:
            root["cellData/ids"][:] = replacement
        return digest

    monkeypatch.setattr(artifacts_module, "fingerprint_text_blocks", write_during)
    first = fingerprint_stored_strings(ids)
    assert first == fingerprint_strings(_ids())

    # The first value was not cached, so restoring the original bytes and
    # asking again computes the fingerprint afresh.
    root["cellData/ids"][:] = _ids()
    computed: list[int] = []

    def counting(n_values, source_dtype, blocks):  # type: ignore[no-untyped-def]
        computed.append(int(n_values))
        return original(n_values, source_dtype, blocks)

    monkeypatch.setattr(artifacts_module, "fingerprint_text_blocks", counting)
    assert fingerprint_stored_strings(root["cellData/ids"]) == first
    assert computed == [50]


def _stamp_inputs() -> dict[str, zarr.Array]:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    other = zarr.open_group(store=MemoryStore(), mode="w")
    return {
        "base": root.create_array("ids", data=_ids(), chunks=(8,)),
        "copy_elsewhere": other.create_group("deep").create_array(
            "ids", data=_ids(), chunks=(8,)
        ),
        "other_values": root.create_array("other", data=_ids(offset=1), chunks=(8,)),
        "other_chunks": root.create_array("rechunked", data=_ids(), chunks=(5,)),
        "other_codec": root.create_array(
            "plain", data=_ids(), chunks=(8,), compressors=None
        ),
        "other_attributes": root.create_array(
            "labelled", data=_ids(), chunks=(8,), attributes={"note": 1}
        ),
    }


def test_stamps_follow_the_stored_bytes_not_the_location() -> None:
    arrays = _stamp_inputs()
    stamps = {name: stored_content_stamp(array) for name, array in arrays.items()}

    assert stamps["base"] == stamps["copy_elsewhere"]
    others = [stamp for name, stamp in stamps.items() if name != "copy_elsewhere"]
    assert len(set(others)) == len(others)


def test_stamps_include_how_missing_chunks_read() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    array = root.create_array("ids", data=_ids(), chunks=(8,))
    strict = array.with_config({"read_missing_chunks": False})

    assert stored_content_stamp(strict) != stored_content_stamp(array)


def test_stamps_do_not_depend_on_how_many_objects_are_fetched_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    array = _stamp_inputs()["base"]
    expected = stored_content_stamp(array)

    monkeypatch.setattr(stamps_module, "_STAMP_BATCH_OBJECTS", 3)
    assert stored_content_stamp(array) == expected


def test_stamps_cover_every_shard_and_empty_arrays() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    sharded = root.create_array("sharded", data=np.arange(40), chunks=(4,), shards=(8,))
    before = stored_content_stamp(sharded)
    sharded[39] = -1
    assert stored_content_stamp(sharded) != before
    assert stamps_module._object_keys(sharded) == [f"c/{index}" for index in range(5)]

    empty = root.create_array("empty", shape=(0,), dtype="<U3", chunks=(4,))
    assert stamps_module._object_keys(empty) == []
    assert fingerprint_stored_strings(empty) == fingerprint_strings(
        np.asarray([], dtype="U3")
    )


def test_cached_values_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stamps_module, "_CACHE_ENTRIES", 2)
    arrays = list(_stamp_inputs().values())[2:5]
    computed: list[int] = []

    def compute(index: int) -> int:
        computed.append(index)
        return index

    for index, array in enumerate(arrays):
        assert cached_for_content("index", array, lambda: compute(index)) == index
    # The oldest entry was dropped; the newest two are reused.
    for index, array in reversed(list(enumerate(arrays))):
        assert cached_for_content("index", array, lambda: compute(index)) == index
    assert computed == [0, 1, 2, 0]


def test_purposes_keep_separate_values() -> None:
    array = _stamp_inputs()["base"]

    assert cached_for_content("first", array, lambda: 1) == 1
    assert cached_for_content("second", array, lambda: 2) == 2
    assert cached_for_content("first", array, lambda: 3) == 1


def test_in_memory_ids_are_fingerprinted_every_time(decodes: list[int]) -> None:
    ids = _ids()

    assert fingerprint_stored_strings(ids) == fingerprint_stored_strings(ids)
    assert decodes == [50, 50]


def test_cleared_values_are_computed_again() -> None:
    array = _stamp_inputs()["base"]

    assert cached_for_content("value", array, lambda: 1) == 1
    assert cached_for_content("value", array, lambda: 2) == 1
    clear_content_cache()
    assert cached_for_content("value", array, lambda: 2) == 2
