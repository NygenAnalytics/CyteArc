import json
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import MemoryStore

import cytearc.metadata as metadata
import cytearc.metadata.selection as selection
from cytearc.metadata.selection import (
    CellField,
    FeatureRef,
    NamedCellArtifact,
    NormalizationSpec,
    category_codes,
    cell_value_array,
    grouping_value_name,
    resolve_cell_aligned_artifact,
    resolve_complete_labels,
    resolve_grouping,
    valid_category_mask,
)
from cytearc.metadata.queries import (
    column_constant_within,
    column_partition_digest,
    columns_same_partition,
    level_key,
    missing_frame_values,
    reduce_observation_units,
)
from cytearc.metadata.rows import MetaDataRowBlock as implementation_row_block
from cytearc.metadata.rows import (
    apply_missing_mask,
    array_row_selection_parts,
    iter_metadata_column_blocks,
    metadata_missing_mask,
    read_array_rows_chunkwise,
    read_metadata_missing_rows,
    read_metadata_rows,
    read_metadata_rows_chunkwise,
    text_read_bytes,
    text_row_selection_parts,
    text_width,
)
from cytearc.metadata.table import MetaData as implementation_metadata
from cytearc.storage import ArtifactRef
from cytearc.storage.errors import ArtifactResolutionError
from tests.memory_helpers import held_within_charge, recording_ledger, traced_charge
from tests.signature_contracts import signature_digest


_METHODS = {
    "__init__",
    "__repr__",
    "_column_names",
    "_expand_to_rows",
    "_get_array",
    "_get_size",
    "_save",
    "active_index",
    "default_block_rows",
    "drop",
    "fetch",
    "fetch_all",
    "fetch_categorical",
    "get_dtype",
    "get_index_by",
    "grep",
    "head",
    "index_to_bool",
    "insert",
    "iter_row_blocks",
    "multi_sift",
    "reset_key",
    "sift",
    "to_pandas_dataframe",
    "update_key",
}


def _metadata_fixture() -> metadata.MetaData:
    group = zarr.open_group(store=MemoryStore(), mode="w")
    group.create_array(
        "I",
        data=np.array([True, False, True, True]),
        chunks=(2,),
    )
    group.create_array(
        "ids",
        data=np.array(["a", "b", "c", "d"]),
        chunks=(2,),
    )
    group.create_array(
        "names",
        data=np.array(["Alpha", "Beta", "Alpine", "Delta"]),
        chunks=(2,),
    )
    group.create_array(
        "score",
        data=np.array([0.5, 2.0, 3.5, 5.0]),
        chunks=(2,),
    )
    return metadata.MetaData(group)


def test_metadata_facade_exports_canonical_objects():
    assert metadata.__all__ == ["CellValues", "MetaData", "MetaDataRowBlock"]
    assert metadata.CellValues is selection.CellValues
    assert metadata.MetaData is implementation_metadata
    assert metadata.MetaDataRowBlock is implementation_row_block
    assert metadata.CellValues.__module__ == "cytearc.metadata"
    assert metadata.MetaData.__module__ == "cytearc.metadata"
    assert metadata.MetaDataRowBlock.__module__ == "cytearc.metadata"


def test_metadata_method_ownership_and_signatures_remain_stable():
    assert _METHODS <= set(metadata.MetaData.__dict__)
    methods = {name: getattr(metadata.MetaData, name) for name in _METHODS}

    assert signature_digest(methods) == (
        "0485f8a7c7ba4947b6507686f11e2c3ab306c76445b1d0987a4ccf90b780aa84"
    )


def test_metadata_rows_and_queries_match_table_contract():
    table = _metadata_fixture()

    assert table.default_block_rows() == 2
    blocks = list(table.iter_row_blocks(columns=["score"], block_rows=2))
    np.testing.assert_array_equal(
        np.concatenate([block.active_global_indices for block in blocks]),
        [0, 2, 3],
    )
    np.testing.assert_allclose(
        np.concatenate([block.values["score"] for block in blocks]),
        [0.5, 3.5, 5.0],
    )
    np.testing.assert_array_equal(
        table.sift("score", 1.0, 4.0), [False, True, True, False]
    )
    # Matching ignores case; the names come back as stored.
    assert table.grep("^AL") == ["Alpha", "Alpine"]
    assert table.head(2)["score"].tolist() == [0.5, 2.0]


def test_metadata_fetch_reads_only_the_requested_column():
    from tests.store_probes import RecordingStore

    store = RecordingStore()
    group = zarr.open_group(store=store, mode="w")
    for index in range(20):
        group.create_array(f"value{index}", data=np.arange(3))
    table = metadata.MetaData(group)
    store.reset()
    np.testing.assert_array_equal(table.fetch_all("value4"), np.arange(3))
    metadata_reads = [
        key
        for operation, key in store.ops
        if operation == "get" and key.endswith("zarr.json")
    ]
    assert metadata_reads and set(metadata_reads) == {"value4/zarr.json"}
    group.create_array("added", data=np.ones(3))
    np.testing.assert_array_equal(table.fetch_all("added"), np.ones(3))
    del group["added"]
    with pytest.raises(
        KeyError, match=r"^'added does not exist in the metadata columns\.'$"
    ):
        table.fetch_all("added")
    group.create_array("nested/hidden", data=np.ones(3))
    with pytest.raises(
        KeyError,
        match="nested/hidden does not exist in the metadata columns. .*look for "
        "'nested_hidden' or a numbered variant",
    ):
        table.fetch_all("nested/hidden")


def test_metadata_row_helpers_read_permutations_and_missing_masks():
    table = _metadata_fixture()
    group = table.locations["primary"]
    group.create_array(
        "__cytearc_missing__score",
        data=np.array([False, True, False, True]),
        chunks=(2,),
    )
    group["score"].attrs["missing_mask"] = "__cytearc_missing__score"

    np.testing.assert_allclose(
        read_metadata_rows(table, "score", np.array([3, 2])),
        [5.0, 3.5],
    )
    np.testing.assert_array_equal(
        metadata_missing_mask(table, "score")[:], [False, True, False, True]
    )
    np.testing.assert_array_equal(
        read_metadata_missing_rows(table, "score", np.array([3, 2])),
        [True, False],
    )
    assert "__cytearc_missing__score" not in table.columns


def test_apply_missing_mask_shows_masked_rows_as_missing():
    missing = np.array([False, True, False])
    counts = np.array([3, 0, 5], dtype=np.int64)

    assert apply_missing_mask(counts, None) is counts
    unmasked = apply_missing_mask(counts, np.zeros(3, dtype=bool))
    assert unmasked.dtype == np.int64
    numeric = apply_missing_mask(counts, missing)
    assert numeric.dtype == np.float64
    np.testing.assert_array_equal(numeric, [3.0, np.nan, 5.0])
    np.testing.assert_array_equal(counts, [3, 0, 5])
    np.testing.assert_array_equal(
        apply_missing_mask(np.array([True, True, True]), missing),
        [True, False, True],
    )
    assert apply_missing_mask(np.array(["a", "", "b"]), missing).tolist() == [
        "a",
        None,
        "b",
    ]
    assert apply_missing_mask(counts, missing, labels=True).tolist() == [3, None, 5]
    assert apply_missing_mask(
        np.array([True, False, True]), missing, labels=True
    ).tolist() == [True, None, True]
    with pytest.raises(ValueError, match="does not align"):
        apply_missing_mask(counts, np.array([True]))


def test_metadata_frames_show_masked_rows_as_missing_and_fetch_stays_raw():
    table = _metadata_fixture()
    group = table.locations["primary"]
    missing = np.array([False, True, True, False])
    for name, values in (
        ("donor", np.array([1, 0, 0, 2], dtype=np.int64)),
        ("site", np.array(["A", "", "", "B"])),
        ("flag", np.array([True, False, False, False])),
    ):
        group.create_array(name, data=values, chunks=(2,))
        group.create_array(f"__cytearc_missing__{name}", data=missing, chunks=(2,))
        group[name].attrs["missing_mask"] = f"__cytearc_missing__{name}"

    np.testing.assert_array_equal(table.fetch_all("donor"), [1, 0, 0, 2])
    np.testing.assert_array_equal(table.fetch("site"), ["A", "", "B"])
    frame = table.to_pandas_dataframe(["donor", "site", "flag", "score"], key="I")
    assert frame["donor"].dtype == np.float64
    assert frame["flag"].dtype == "boolean"
    np.testing.assert_array_equal(
        frame.isna().to_numpy(),
        [[False, False, False, False], [True, True, True, False], [False] * 4],
    )
    assert frame["donor"].tolist()[::2] == [1.0, 2.0]
    assert frame["site"].tolist()[::2] == ["A", "B"]
    assert frame["flag"].tolist()[::2] == [True, False]
    head = table.head(2)
    assert head["donor"].isna().tolist() == [False, True]
    assert head["site"].isna().tolist() == [False, True]
    assert head["score"].tolist() == [0.5, 2.0]
    boolean = missing_frame_values(np.array([True, True]), np.array([True, False]))
    assert boolean.dtype == "boolean"
    assert boolean.isna().tolist() == [True, False]


def test_metadata_row_helpers_preserve_noncontiguous_order_without_span():
    class TrackingArray:
        def __init__(self, values):
            self._values = np.asarray(values)
            self.shape = self._values.shape
            self.requests: list[tuple[str, object]] = []

        def __getitem__(self, key):
            self.requests.append(("getitem", key))
            return self._values[key]

        def get_orthogonal_selection(self, selection):
            self.requests.append(("orthogonal", selection))
            (indices,) = selection
            return self._values[np.asarray(indices)]

    class TrackingMeta:
        N = 6
        columns = ["score"]

        def __init__(self, array):
            self._array = array

        def _get_array(self, column):
            assert column == "score"
            return self._array

        def default_block_rows(self, column="I"):
            return self.N

    array = TrackingArray([10, 20, 30, 40, 50, 60])
    table = TrackingMeta(array)
    rows = np.array([5, 1, 4], dtype=np.int64)
    np.testing.assert_array_equal(
        read_metadata_rows(table, "score", rows), [60, 20, 50]
    )
    assert array.requests[0][0] == "orthogonal"
    assert not any(kind == "getitem" for kind, _ in array.requests)

    contiguous = TrackingArray([10, 20, 30, 40])
    contiguous_table = TrackingMeta(contiguous)
    np.testing.assert_array_equal(
        read_metadata_rows(contiguous_table, "score", np.array([1, 2, 3])),
        [20, 30, 40],
    )
    assert contiguous.requests == [("getitem", slice(1, 4))]


def test_chunkwise_metadata_rows_preserve_order_and_decode_one_chunk():
    class ArrayMetadata:
        shards = None

    class TrackingArray:
        def __init__(self, values, chunk_rows):
            self._values = np.asarray(values)
            self.shape = self._values.shape
            self.dtype = self._values.dtype
            self.chunks = (chunk_rows,)
            self.metadata = ArrayMetadata()
            self.requests: list[tuple[str, object]] = []

        def __getitem__(self, key):
            self.requests.append(("getitem", key))
            return self._values[key]

        def get_orthogonal_selection(self, selection):
            self.requests.append(("orthogonal", selection))
            (indices,) = selection
            return self._values[np.asarray(indices)]

    class TrackingMeta:
        columns = ["score"]

        def __init__(self, array):
            self._array = array
            self.N = int(array.shape[0])

        def _get_array(self, column):
            assert column == "score"
            return self._array

        def default_block_rows(self, column="I"):
            _ = column
            return int(self._array.chunks[0])

    array = TrackingArray([10, 20, 30, 40, 50, 60], chunk_rows=2)
    table = TrackingMeta(array)
    rows = np.array([5, 0, 4, 1, 2], dtype=np.int64)
    np.testing.assert_array_equal(
        read_metadata_rows_chunkwise(table, "score", rows),
        [60, 10, 50, 20, 30],
    )

    for kind, request in array.requests:
        if kind == "getitem":
            assert isinstance(request, slice)
            selected = np.arange(request.start, request.stop)
        else:
            assert isinstance(request, tuple)
            selected = np.asarray(request[0])
        assert np.unique(selected // array.chunks[0]).size == 1

    # The bytes charged per row cover every value of a two-dimensional row.
    group = zarr.open_group(store=MemoryStore(), mode="w")
    narrow = group.create_array("narrow", shape=(6,), dtype="f8", chunks=(2,))
    wide = group.create_array("wide", shape=(6, 4), dtype="f8", chunks=(2, 4))
    assert array_row_selection_parts(wide)[1] > array_row_selection_parts(narrow)[1]


def test_reading_text_rows_counts_a_string_for_each_row():
    group = zarr.open_group(store=MemoryStore(), mode="w")
    text = group.create_array("text", shape=(6,), dtype="<U5", chunks=(2,))
    numbers = group.create_array("numbers", shape=(6,), dtype="f8", chunks=(2,))
    chunk, per_row = array_row_selection_parts(text)
    # A row of five characters is held three times, 20 bytes each, beside 16
    # int64 positions, and then as a string of 64 bytes and four per character.
    assert text_width(text) == 5
    assert per_row == 3 * 20 + 16 * 8
    assert text_row_selection_parts(text) == (chunk, per_row + 64 + 4 * 5)
    assert text_read_bytes(text, 3) == chunk + 3 * (per_row + 64 + 4 * 5)
    # Reading no row still decodes a chunk.
    assert chunk > 0
    assert text_read_bytes(text, 0) == chunk
    # Values that are not text have no characters, but each becomes a string.
    chunk, per_row = array_row_selection_parts(numbers)
    assert text_width(numbers) == 0
    assert text_read_bytes(numbers, 3) == chunk + 3 * (per_row + 64)


def test_metadata_column_blocks_respect_source_chunk_boundaries():
    class ArrayMetadata:
        shards = None

    class TrackingArray:
        def __init__(self):
            self._values = np.arange(7)
            self.shape = self._values.shape
            self.dtype = self._values.dtype
            self.chunks = (3,)
            self.metadata = ArrayMetadata()
            self.requests: list[slice] = []

        def __getitem__(self, key):
            assert isinstance(key, slice)
            self.requests.append(key)
            return self._values[key]

    class TrackingMeta:
        N = 7
        columns = ["score"]

        def __init__(self, array):
            self._array = array

        def _get_array(self, column):
            assert column == "score"
            return self._array

        def default_block_rows(self, column="I"):
            _ = column
            return int(self._array.chunks[0])

    array = TrackingArray()
    table = TrackingMeta(array)
    values = list(iter_metadata_column_blocks(table, "score", block_rows=2))
    np.testing.assert_array_equal(np.concatenate(values), np.arange(7))
    assert max(value.size for value in values) <= 2
    for request in array.requests:
        first_bin = request.start // array.chunks[0]
        last_bin = (request.stop - 1) // array.chunks[0]
        assert first_bin == last_bin


def test_metadata_row_blocks_remain_pickle_resolvable():
    block = metadata.MetaDataRowBlock(
        start=0,
        stop=2,
        active_global_indices=np.array([0]),
        values={"score": np.array([0.5])},
    )

    restored = pickle.loads(pickle.dumps(block))

    assert type(restored) is metadata.MetaDataRowBlock
    assert (restored.start, restored.stop) == (0, 2)
    np.testing.assert_array_equal(restored.active_global_indices, [0])
    assert list(restored.values) == ["score"]
    np.testing.assert_array_equal(restored.values["score"], [0.5])


def _artifact(kind: str, token: str) -> ArtifactRef:
    return ArtifactRef(
        scope="datastore",
        kind=kind,
        artifact_id=token * 64,
    )


def test_metadata_selection_value_contracts_reject_ambiguous_inputs():
    np.testing.assert_array_equal(
        valid_category_mask(np.array([b"", b"a", np.bytes_(b" ")], dtype=object)),
        [False, True, False],
    )
    with pytest.raises(ValueError, match="one-dimensional"):
        valid_category_mask(np.ones((2, 2)))
    with pytest.raises(ValueError, match="missing mask must align"):
        valid_category_mask(["a", "b"], missing_mask=[True])
    with pytest.raises(ValueError, match="by must be"):
        FeatureRef("gene", by="label")
    with pytest.raises(ValueError, match="reduction must be"):
        FeatureRef("gene", reduction="median")
    for value in (1.9, np.float64(2.0), True, np.bool_(True), "1", np.array(1)):
        with pytest.raises(TypeError, match="FeatureRef value must be an integer"):
            FeatureRef(value, by="index")
    index_ref = FeatureRef(np.int64(1), by="index")
    assert type(index_ref.value) is int and index_ref == FeatureRef(1, by="index")
    for by in ("name", "id"):
        for value in (1, np.int64(1), None, b"gene"):
            with pytest.raises(TypeError, match="FeatureRef value must be a string"):
                FeatureRef(value, by=by)
    assert type(FeatureRef(np.str_("gene"), by="id").value) is str
    with pytest.raises(ValueError, match="non-empty name"):
        NamedCellArtifact("", _artifact("cluster_labels", "a"))
    with pytest.raises(ValueError, match="surrounding whitespace"):
        NamedCellArtifact(" labels ", _artifact("cluster_labels", "a"))
    with pytest.raises(TypeError, match="ArtifactRef"):
        NamedCellArtifact("labels", object())
    with pytest.raises(ValueError, match="categorical cell labels"):
        grouping_value_name("embedding")
    with pytest.raises(ValueError, match="categorical cell labels"):
        grouping_value_name("connectivity_map")
    assert grouping_value_name("cluster_cut") == "labels"
    assert cell_value_array("cell_cycle") == ("phase", True)
    assert cell_value_array("cell_cycle", "s_score") == ("s_score", False)
    with pytest.raises(ValueError, match="no per-cell array 'values'"):
        cell_value_array("cell_cycle", "values")
    with pytest.raises(ValueError, match="'normalized' is not a cell-aligned"):
        cell_value_array("normalized", "data")
    with pytest.raises(TypeError, match="value must be a string or None"):
        cell_value_array("cell_cycle", b"phase")
    with pytest.raises(ValueError, match="transform"):
        NormalizationSpec(transform="sqrt")


def test_cell_aligned_artifact_validation_reports_each_broken_contract(monkeypatch):
    artifact = _artifact("cluster_labels", "b")
    source = _artifact("cell_selection", "c")
    target = _artifact("cell_selection", "d")

    with pytest.raises(TypeError, match="artifact must be"):
        resolve_cell_aligned_artifact(None, object())
    with pytest.raises(ValueError, match="Expected a 'cell_cycle'"):
        resolve_cell_aligned_artifact(None, artifact, expected_kind="cell_cycle")
    with pytest.raises(ValueError, match="value_name must be"):
        resolve_cell_aligned_artifact(None, artifact, value_name="")
    with pytest.raises(ValueError, match="ndim must be"):
        resolve_cell_aligned_artifact(None, artifact, ndim=0)
    # The table names every value array: a listed kind reads only its per-cell
    # arrays, and an unlisted kind has no canonical array to fall back to.
    with pytest.raises(ValueError, match="no per-cell array 'labels'"):
        resolve_cell_aligned_artifact(None, artifact, value_name="labels")
    with pytest.raises(ValueError, match="'cell_selection' is not a cell-aligned"):
        resolve_cell_aligned_artifact(None, source)

    monkeypatch.setattr(
        selection,
        "inspect_artifact",
        lambda *_: SimpleNamespace(exists=False, complete=False, inputs={}),
    )
    with pytest.raises(ValueError, match="unavailable or incomplete"):
        resolve_cell_aligned_artifact(None, artifact)

    # The recorded cell selection is read with the one strict input reader.
    status = SimpleNamespace(exists=True, complete=True, inputs={})
    monkeypatch.setattr(selection, "inspect_artifact", lambda *_: status)
    with pytest.raises(
        ArtifactResolutionError, match="cluster_labels artifact has no 'cell_selection'"
    ) as missing:
        resolve_cell_aligned_artifact(None, artifact)
    assert missing.value.code == "corrupt_payload"

    status.inputs = {"cell_selection": {"type": "wrong"}}
    with pytest.raises(ArtifactResolutionError, match="malformed 'cell_selection'"):
        resolve_cell_aligned_artifact(None, artifact)

    status.inputs = {"cell_selection": source.to_dict()}
    with pytest.raises(TypeError, match="cell_selection must be"):
        resolve_cell_aligned_artifact(None, artifact, cell_selection=object())

    monkeypatch.setattr(selection, "artifact_group", lambda *_: {})
    with pytest.raises(ValueError, match="no 'values' value array"):
        resolve_cell_aligned_artifact(None, artifact)

    # The selection's cells are counted, not read, to check the array's
    # shape and to admit the read before any cell row is read.
    monkeypatch.setattr(selection, "_selection_count", lambda *_: 2)
    bad_array = SimpleNamespace(ndim=1, shape=(3,))
    monkeypatch.setattr(selection, "artifact_group", lambda *_: {"values": bad_array})
    monkeypatch.setattr(selection, "as_zarr_array", lambda value, **_: value)
    with pytest.raises(ValueError, match="one value per source-selected cell"):
        resolve_cell_aligned_artifact(None, artifact)
    with pytest.raises(ValueError, match="one row per source-selected cell"):
        resolve_cell_aligned_artifact(None, artifact, ndim=None)

    good_array = SimpleNamespace(ndim=1, shape=(2,), attrs={})
    monkeypatch.setattr(selection, "artifact_group", lambda *_: {"values": good_array})

    def out_of_bounds(_root, ref):
        return np.array([1, 3]) if ref == source else np.array([5])

    monkeypatch.setattr(selection, "cell_selection_indices", out_of_bounds)
    with pytest.raises(ValueError, match="must be a subset"):
        resolve_cell_aligned_artifact(None, artifact, cell_selection=target)

    def not_a_member(_root, ref):
        return np.array([1, 3]) if ref == source else np.array([2])

    monkeypatch.setattr(selection, "cell_selection_indices", not_a_member)
    with pytest.raises(ValueError, match="must be a subset"):
        resolve_cell_aligned_artifact(None, artifact, cell_selection=target)

    # The linked mask is resolved before any value is read, so that the
    # memory admission counts it.
    monkeypatch.setattr(
        selection, "cell_selection_indices", lambda *_: np.array([1, 3])
    )
    monkeypatch.setattr(
        selection,
        "read_rows",
        lambda *_args, **_kwargs: np.array([1]),
    )
    with pytest.raises(ValueError, match="values do not match"):
        resolve_cell_aligned_artifact(None, artifact)


def test_resolve_grouping_validates_field_kind_and_missing_mask(monkeypatch):
    cells = SimpleNamespace(N=2)
    with pytest.raises(ValueError, match="must be categorical"):
        resolve_grouping(None, cells, CellField("score", kind="continuous"))

    monkeypatch.setattr(
        selection, "read_metadata_rows", lambda *_: np.array(["a", "b"])
    )
    monkeypatch.setattr(
        selection, "read_metadata_missing_rows", lambda *_: np.array([True])
    )
    with pytest.raises(ValueError, match="missing mask does not align"):
        resolve_grouping(None, cells, CellField("group", kind="categorical"))


def test_case_insensitive_index_matches_the_text_of_any_value():
    from cytearc.metadata.table import CaseInsensitiveIndex

    index = CaseInsensitiveIndex(["Gene", "GENE", None, 7, np.nan, "other"])

    assert index.positions("gene") == [0, 1]
    assert index.positions(7) == [3]
    assert index.positions("missing") == []
    index.positions("gene").append(9)
    assert index.positions("gene") == [0, 1]


def test_metadata_table_fill_and_error_contracts():
    empty = zarr.open_group(store=MemoryStore(), mode="w")
    with pytest.raises(ValueError, match="empty zarr group"):
        metadata.MetaData(empty)

    primary = zarr.open_group(store=MemoryStore(), mode="w")
    primary.create_array("I", data=np.array([True, False, True, False]), chunks=(2,))
    primary.create_array("ids", data=np.array(["a", "b", "c", "d"]), chunks=(2,))
    primary.create_array("score", data=np.arange(4.0), chunks=(2,))
    primary.create_array("x_value", data=np.arange(4), chunks=(2,))
    table = metadata.MetaData(primary)

    # Listing a group reads every column's metadata, so a corrupt column fails loudly.
    corrupt = zarr.open_group(store=MemoryStore(), mode="w")
    corrupt.create_array("value", data=np.arange(4), chunks=(2,))
    sync(
        corrupt.store.set(
            "value/zarr.json", default_buffer_prototype().buffer.from_bytes(b"{")
        )
    )
    with pytest.raises(json.JSONDecodeError):
        table._get_size(corrupt)
    with pytest.raises(ValueError, match="empty zarr group"):
        table._get_size(empty)
    assert table._has_column("I") and not table._has_column("unknown")
    with pytest.raises(KeyError, match="does not exist"):
        table._get_array("unknown")
    with pytest.raises(KeyError, match="does not exist"):
        table.drop("unknown")
    with pytest.raises(TypeError, match="boolean type column"):
        table._bool_array("score")
    assert list(table.locations) == ["primary"]

    with pytest.raises(ValueError, match="Expected shape"):
        table._save("value", np.arange(2))
    with pytest.raises(ValueError, match="missing mask is of shape"):
        table._save("value", np.arange(4), np.zeros(3, dtype=bool))

    # Rows that the key leaves out hold a placeholder and are flagged missing,
    # whatever the sign of the supplied integers.
    for supplied, expected in (([5, 6], [5, 0, 6, 0]), ([-2, -1], [-2, 0, -1, 0])):
        values, missing = table._expand_to_rows(supplied, "I", name="x")
        np.testing.assert_array_equal(values, expected)
        np.testing.assert_array_equal(missing, [False, True, False, True])
    # An explicit fill is a real value that must fit the dtype.
    values, missing = table._expand_to_rows(np.array([-2, -1]), "I", -9, name="x")
    np.testing.assert_array_equal(values, [-2, -9, -1, -9])
    assert not missing.any()
    for fill in (np.nan, "bad"):
        with pytest.raises(ValueError, match="fill_value .* does not fit"):
            table._expand_to_rows(np.array([1, 2]), "I", fill, name="x")
    with pytest.raises(ValueError, match="incorrect length"):
        table._expand_to_rows(np.array([1]), "I", 0, name="x")

    with pytest.raises(TypeError, match="value_targets"):
        table.get_index_by("a", "ids")
    np.testing.assert_array_equal(table.get_index_by(["A"], "ids", key="I"), [0])
    # With a key, indices are positions among the selected rows.
    np.testing.assert_array_equal(
        table.get_index_by(["C", "a"], "ids", key="I"), [1, 0]
    )
    # Values and targets that are not text match as text instead of raising.
    numeric = table.get_index_by([2, "missing"], "x_value")
    assert numeric.dtype == np.int64
    assert numeric.tolist() == [2]
    assert table.get_index_by(["missing"], "ids").dtype == np.int64
    np.testing.assert_array_equal(
        table.index_to_bool(np.array([1]), invert=True), [True, False, True, True]
    )

    with pytest.raises(ValueError, match="protected column"):
        table.insert("I", np.ones(4, dtype=bool))
    with pytest.raises(ValueError, match="already exists"):
        table.insert("score", np.arange(4.0))
    table.insert("list_values", [1, 2, 3, 4])
    with pytest.raises(ValueError, match="protected name"):
        table.drop("ids")
    np.testing.assert_array_equal(
        table.multi_sift(["score"], [0.0], [3.0]),
        [False, True, True, False],
    )
    assert repr(table) == "MetaData of 2(4) elements"

    with pytest.raises(TypeError, match="boolean type column"):
        table.active_index("score")


def _masked_metadata() -> metadata.MetaData:
    group = zarr.open_group(store=MemoryStore(), mode="w")
    group.create_array("I", data=np.ones(4, dtype=bool))
    group.create_array("ids", data=np.array(["a", "b", "c", "d"]))
    group.create_array("names", data=np.array(["a", "b", "c", "d"]))
    for name, values in (
        ("zeta", np.array([1, 0, 7, 2], dtype=np.int64)),
        ("count", np.array([5, 0, 7, 2], dtype=np.int64)),
        ("label", np.array(["a", "", "b", "a"])),
        ("donor", np.array(["x", "", "y", "x"])),
    ):
        array = group.create_array(name, data=values)
        group.create_array(
            f"__cytearc_missing__{name}", data=np.array([False, True, False, False])
        )
        array.attrs["missing_mask"] = f"__cytearc_missing__{name}"
    return metadata.MetaData(group)


def test_metadata_columns_are_listed_in_a_stable_order():
    table = _masked_metadata()

    assert table.columns == ["I", "ids", "names", "count", "donor", "label", "zeta"]


def test_sift_never_passes_rows_with_a_linked_missing_mask():
    table = _masked_metadata()

    np.testing.assert_array_equal(
        table.sift("count", -1, 10), [True, False, True, True]
    )
    np.testing.assert_array_equal(
        table.multi_sift(["count", "zeta"], [-1, -1], [10, 10]),
        [True, False, True, True],
    )


def test_partition_helpers_treat_masked_rows_as_missing():
    table = _masked_metadata()
    digest = column_partition_digest(table, "label")

    assert (digest.nMissing, digest.nLevels, digest.nRows) == (1, 3, 4)
    # Masked rows hold "" placeholders but pair up as one missing level.
    assert columns_same_partition(table, "label", "donor") == (
        True,
        "a = x; missing = missing; b = y",
    )
    assert column_constant_within(table, "donor", "label")
    units = reduce_observation_units(table, "donor", ["label"])
    assert units["donor"].isna().tolist() == [False, True, False]
    assert units["label"].isna().tolist() == [False, True, False]
    assert units.dropna()["label"].tolist() == ["a", "b"]


def test_insert_keeps_an_explicit_boolean_fill_value():
    table = _metadata_fixture()

    table.insert("keep", np.array([True, False, False]), fill_value=True)
    table.insert("drop_rows", np.array([True, False, False]))

    np.testing.assert_array_equal(table.fetch_all("keep"), [True, True, False, False])
    assert metadata_missing_mask(table, "keep") is None
    # Without a fill, the row outside the key holds False and is flagged missing.
    np.testing.assert_array_equal(
        table.fetch_all("drop_rows"), [True, False, False, False]
    )
    np.testing.assert_array_equal(
        metadata_missing_mask(table, "drop_rows")[:], [False, True, False, False]
    )


def test_head_reads_only_the_requested_rows(monkeypatch):
    table = _metadata_fixture()

    def fail(_column):
        raise AssertionError("head must not read whole columns")

    monkeypatch.setattr(table, "fetch_all", fail)
    assert table.head(2)["score"].tolist() == [0.5, 2.0]


@pytest.mark.parametrize("name", ["a/b", "a\\b"], ids=["slash", "backslash"])
def test_insert_rejects_zarr_path_separators_before_writing(name):
    table = _metadata_fixture()

    with pytest.raises(ValueError, match="path separators; use 'a_b' instead"):
        table.insert(name, np.arange(4.0))

    assert set(table._group.keys()) == {"I", "ids", "names", "score"}
    assert not table._is_public(name)
    with pytest.raises(KeyError, match="look for 'a_b' or a numbered variant"):
        table.fetch_all(name)
    with pytest.raises(KeyError, match="'a_b_2'"):
        table.drop(name)


@pytest.mark.parametrize("name", ["r/k", "r\\k"], ids=["slash", "backslash"])
def test_key_writes_reject_zarr_path_separators_before_writing(name):
    table = _metadata_fixture()

    with pytest.raises(ValueError, match="path separators; use 'r_k' instead"):
        table.reset_key(name)
    with pytest.raises(ValueError, match="path separators; use 'r_k' instead"):
        table.update_key(np.ones(4, dtype=bool), name)

    assert set(table._group.keys()) == {"I", "ids", "names", "score"}


def test_lookups_of_names_that_are_not_text_raise_key_errors():
    table = _metadata_fixture()

    for column in (123, None):
        with pytest.raises(KeyError, match="does not exist in the metadata columns"):
            table.fetch_all(column)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["__cytearc_missing__score", "__cytearc_missing__new"])
def test_writes_reject_the_missing_mask_prefix(name):
    table = _metadata_fixture()

    with pytest.raises(ValueError, match="reserves for missing-value masks"):
        table.insert(name, np.ones(4, dtype=bool))
    with pytest.raises(ValueError, match="reserves for missing-value masks"):
        table.reset_key(name)

    assert set(table._group.keys()) == {"I", "ids", "names", "score"}


@pytest.mark.parametrize("name", ["", ".", ".."])
def test_lookups_of_names_zarr_cannot_store_raise_key_errors(name):
    table = _metadata_fixture()

    with pytest.raises(KeyError, match="does not exist in the metadata columns"):
        table.fetch_all(name)
    with pytest.raises(KeyError, match="does not exist in the metadata columns"):
        table.drop(name)


def test_writes_over_a_nested_group_ask_for_a_new_import():
    table = _metadata_fixture()
    table._group.create_array("Baseline (ml/min)", data=np.ones(4))

    for write in (
        lambda: table.drop("Baseline (ml"),
        lambda: table.insert("Baseline (ml", np.zeros(4), overwrite=True),
        lambda: table.reset_key("Baseline (ml"),
    ):
        with pytest.raises(TypeError, match="import the source again"):
            write()
    assert list(table._group["Baseline (ml"].array_keys()) == ["min)"]


def test_nested_group_from_an_older_import_asks_for_a_new_import():
    table = _metadata_fixture()
    # Older imports let Zarr nest a source column named with '/' into groups.
    table._group.create_array("Baseline (ml/min)", data=np.ones(4))

    assert "Baseline (ml" in table.columns
    with pytest.raises(TypeError, match="nested group named 'Baseline \\(ml'"):
        table.fetch_all("Baseline (ml")
    with pytest.raises(TypeError, match="import the source again"):
        table.head()


class _SelectionArray:
    """A vector whose vectorized selections may be absent or unsupported.

    ``orthogonal`` and ``coordinate`` are None to omit that selection, True to
    support it, or the exception type that the selection raises.
    """

    def __init__(self, values, *, orthogonal=None, coordinate=None):
        self._values = np.asarray(values)
        self.shape = self._values.shape
        self.requests: list[str] = []
        for kind, behavior in (("orthogonal", orthogonal), ("coordinate", coordinate)):
            if behavior is not None:
                setattr(self, f"get_{kind}_selection", self._selection(kind, behavior))

    def _selection(self, kind, behavior):
        def select(selection):
            self.requests.append(kind)
            if behavior is not True:
                raise behavior(f"{kind} selection is unsupported")
            (indices,) = selection
            return self._values[np.asarray(indices)]

        return select

    def __getitem__(self, key):
        self.requests.append("getitem")
        return self._values[key]


@pytest.mark.parametrize(
    ("options", "requests"),
    [
        (
            {"orthogonal": NotImplementedError, "coordinate": True},
            ["orthogonal", "coordinate"],
        ),
        (
            {"orthogonal": AttributeError, "coordinate": TypeError},
            ["orthogonal", "coordinate", "getitem"],
        ),
        ({"coordinate": NotImplementedError}, ["coordinate", "getitem"]),
        ({}, ["getitem"]),
    ],
    ids=["orthogonal_unsupported", "both_unsupported", "coordinate_only", "neither"],
)
def test_row_reader_falls_back_through_the_selections_a_backend_supports(
    options, requests
):
    array = _SelectionArray([10, 20, 30, 40, 50, 60], **options)
    table = SimpleNamespace(_get_array=lambda _column: array)

    np.testing.assert_array_equal(
        read_metadata_rows(table, "score", np.array([5, 1, 4])), [60, 20, 50]
    )
    assert array.requests == requests

    # Only errors that mean "unsupported" fall back; others propagate.
    failing = _SelectionArray([10, 20, 30], orthogonal=RuntimeError, coordinate=True)
    table = SimpleNamespace(_get_array=lambda _column: failing)
    with pytest.raises(RuntimeError, match="orthogonal selection is unsupported"):
        read_metadata_rows(table, "score", np.array([2, 0]))
    assert failing.requests == ["orthogonal"]


def test_row_readers_reject_malformed_and_out_of_range_rows():
    table = _metadata_fixture()

    for read in (read_metadata_rows, read_metadata_rows_chunkwise):
        with pytest.raises(ValueError, match="row indices must be one-dimensional"):
            read(table, "score", np.array([[0, 2]]))
        empty = read(table, "score", np.array([], dtype=np.int64))
        assert empty.shape == (0,)
        assert empty.dtype == np.float64
    for rows in ([1, 4], [-1, 2]):
        with pytest.raises(IndexError, match="Metadata row indices are out of bounds"):
            read_metadata_rows(table, "score", np.array(rows))
    # The chunkwise reader reads each requested row exactly once.
    with pytest.raises(IndexError, match="out-of-range index"):
        read_metadata_rows_chunkwise(table, "score", np.array([1, 4]))
    with pytest.raises(ValueError, match="cannot contain duplicate indexes"):
        read_metadata_rows_chunkwise(table, "score", np.array([2, 2]))


def test_chunkwise_row_reader_rejects_a_selection_of_the_wrong_shape():
    class ArrayMetadata:
        shards = None

    class ShortArray:
        """A chunked array whose selections drop their last value."""

        def __init__(self):
            self._values = np.arange(4.0)
            self.shape = self._values.shape
            self.dtype = self._values.dtype
            self.chunks = (4,)
            self.metadata = ArrayMetadata()

        def get_orthogonal_selection(self, selection):
            (indices,) = selection
            return self._values[np.asarray(indices)][:-1]

    table = SimpleNamespace(_get_array=lambda _column: ShortArray())

    with pytest.raises(ValueError, match="selection returned an invalid shape"):
        read_metadata_rows_chunkwise(table, "score", np.array([0, 2]))


def test_block_iterators_reject_nonpositive_block_sizes():
    table = _metadata_fixture()

    for block_rows in (0, -3):
        with pytest.raises(ValueError, match="block_rows must be >= 1"):
            list(iter_metadata_column_blocks(table, "score", block_rows=block_rows))
        with pytest.raises(ValueError, match="block_rows must be >= 1"):
            list(table.iter_row_blocks(columns=["score"], block_rows=block_rows))


def test_single_row_blocks_cover_every_row_in_order():
    table = _metadata_fixture()

    blocks = list(table.iter_row_blocks(columns=["ids", "score"], block_rows=1))

    assert [(block.start, block.stop) for block in blocks] == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
    ]
    assert [block.active_global_indices.tolist() for block in blocks] == [
        [0],
        [],
        [2],
        [3],
    ]
    assert [block.values["ids"].tolist() for block in blocks] == [
        ["a"],
        [],
        ["c"],
        ["d"],
    ]
    assert [block.values["score"].tolist() for block in blocks] == [
        [0.5],
        [],
        [3.5],
        [5.0],
    ]


def test_missing_mask_lookup_of_an_unknown_column_raises_a_key_error():
    table = _masked_metadata()

    with pytest.raises(KeyError, match="unknown does not exist in the metadata"):
        metadata_missing_mask(table, "unknown")
    with pytest.raises(KeyError, match="unknown does not exist in the metadata"):
        read_metadata_missing_rows(table, "unknown", np.array([0]))
    # Reserved and unmasked columns exist but have no mask.
    assert metadata_missing_mask(table, "ids") is None
    np.testing.assert_array_equal(
        metadata_missing_mask(table, "count")[:], [False, True, False, False]
    )


def test_level_keys_merge_every_missing_value_and_decode_bytes():
    missing = level_key(None)

    # Missing scalars of every kind share one level.
    for value in (np.nan, np.float32("nan"), pd.NA, pd.NaT, np.datetime64("NaT", "D")):
        assert level_key(value) is missing
    # Bytes compare with their UTF-8 text; NumPy scalars with Python values.
    assert level_key(b"donor") == level_key(np.bytes_(b"donor")) == "donor"
    assert level_key(b"\xff") == "�"
    assert level_key(np.int64(3)) == 3
    assert type(level_key(np.int64(3))) is int
    assert level_key(np.str_("x")) == "x"
    # Unhashable labels compare by representation, even those whose
    # missing-value test is ambiguous.
    for value in ([1, 2], [1], (1, 2), {1}, {"a": 1}, np.array([1, 2])):
        assert level_key(value) == ("__cytearc_repr__", repr(value))
    assert level_key([1, 2]) != level_key((1, 2))
    assert level_key("missing") is not missing


def _partition_metadata() -> metadata.MetaData:
    group = zarr.open_group(store=MemoryStore(), mode="w")
    group.create_array("I", data=np.ones(6, dtype=bool), chunks=(4,))
    group.create_array("none", data=np.zeros(6, dtype=bool), chunks=(4,))
    group.create_array("ids", data=np.array([f"c{i}" for i in range(6)]), chunks=(4,))
    for name, values in (
        ("sample", ["s1", "s1", "s2", "s2", "s3", "s3"]),
        ("donor", ["d1", "d1", "d2", "d2", "d2", "d2"]),
        ("site", ["x", "x", "y", "y", "y", "y"]),
    ):
        group.create_array(name, data=np.array(values), chunks=(4,))
    return metadata.MetaData(group)


def test_partition_helpers_detect_coarser_and_varying_columns():
    table = _partition_metadata()

    # Donor merges samples s2 and s3: the partitions differ in both orders.
    assert columns_same_partition(table, "sample", "donor") == (False, "")
    assert columns_same_partition(table, "donor", "sample") == (False, "")
    # Renamed labels form the same partition and list each correspondence.
    assert columns_same_partition(table, "donor", "site") == (True, "d1 = x; d2 = y")
    assert column_partition_digest(table, "donor") == column_partition_digest(
        table, "site"
    )
    assert column_constant_within(table, "donor", "sample")
    assert not column_constant_within(table, "sample", "donor")
    units = reduce_observation_units(table, "sample", ["donor"])
    assert units.to_dict(orient="list") == {
        "sample": ["s1", "s2", "s3"],
        "donor": ["d1", "d2", "d2"],
    }


def test_observation_units_of_an_empty_selection_keep_their_columns():
    table = _partition_metadata()

    units = reduce_observation_units(
        table, "sample", ["donor", "sample"], cell_key="none"
    )

    assert units.empty
    assert list(units.columns) == ["sample", "donor"]


def test_category_mask_accepts_labels_whose_missing_test_is_ambiguous():
    values = np.empty(4, dtype=object)
    values[:] = [["a", "b"], "c", None, " "]

    np.testing.assert_array_equal(
        valid_category_mask(values), [True, True, False, False]
    )


def test_category_codes_name_labels_by_text_in_natural_order():
    values = np.asarray(["b10", "b2", " ", "", "b2", "a"], dtype="<U3")
    codes, names = category_codes(values, missing_mask=[False] * 5 + [True])
    assert names == ["b2", "b10"]
    assert codes.tolist() == [1, 0, -1, -1, 0, -1]
    assert codes.dtype == np.int64
    # Values that read alike share a code; numbers sort by value.
    mixed = np.asarray([2, "2", 10, None, float("nan"), 1.5], dtype=object)
    codes, names = category_codes(mixed)
    assert names == ["1.5", "2", "10"]
    assert codes.tolist() == [1, 1, 2, -1, -1, 0]
    integers, names = category_codes(np.asarray([3, 1, 3]))
    assert (integers.tolist(), names) == ([1, 0, 1], ["1", "3"])
    # Unhashable labels are coded by their text.
    unhashable = np.empty(3, dtype=object)
    unhashable[:] = [["a", "b"], "c", None]
    codes, names = category_codes(unhashable)
    assert (codes.tolist(), names) == ([0, 1, -1], ["['a', 'b']", "c"])
    empty, names = category_codes(np.asarray([], dtype=object))
    assert (empty.tolist(), names) == ([], [])
    with pytest.raises(ValueError, match="one-dimensional"):
        category_codes(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="must align"):
        category_codes(["a", "b"], missing_mask=[True])


def test_complete_labels_require_an_artifact_ref():
    with pytest.raises(TypeError, match="^clusters must be an ArtifactRef$"):
        resolve_complete_labels(None, "clusters", name="clusters")


@pytest.mark.parametrize("kind", ["fixed", "variable"])
def test_stored_rows_are_fingerprinted_a_chunk_at_a_time_as_read_whole(kind) -> None:
    import tracemalloc

    from cytearc.metadata.rows import fingerprint_stored_rows
    from cytearc.storage.artifacts import fingerprint_strings

    values = np.asarray(
        [f"cell{index:05d}" + "x" * (index % 9) for index in range(400)]
    )
    if kind == "variable":
        values = values.astype(np.dtypes.StringDType())
    root = zarr.open_group(store=MemoryStore(), mode="w")
    stored = root.create_array("ids", data=values, chunks=(50,))
    rng = np.random.default_rng(4)
    selections = [
        np.arange(400),
        np.array([399]),
        np.array([], dtype=np.int64),
        np.sort(rng.choice(400, 77, replace=False)),
    ]
    for rows in selections:
        assert fingerprint_stored_rows(stored, rows) == fingerprint_strings(
            values[rows]
        )
    # Rows out of order or repeated would fingerprint other text.
    for rows in (np.array([3, 1]), np.array([2, 2])):
        with pytest.raises(ValueError, match="distinct and ascending"):
            fingerprint_stored_rows(stored, rows)
    with pytest.raises(ValueError, match="must be one-dimensional"):
        fingerprint_stored_rows(stored, np.zeros((2, 2), dtype=np.int64))
    # One chunk of 1,000 of 40,000 rows is held at a time: well below the
    # rows read whole, 4 bytes for each of the widest value's 15 characters.
    many = np.asarray([f"cell{index:011d}" for index in range(40_000)])
    if kind == "variable":
        many = many.astype(np.dtypes.StringDType())
    stored = root.create_array("many", data=many, chunks=(1_000,))
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    fingerprint_stored_rows(stored, np.arange(40_000))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak - before < 40_000 * 15 * 4 // 4


# Text of variable width, which another tool can write


def _variable_text(n_values: int) -> np.ndarray:
    """Values of 9 to 40 characters, every fifth beyond ASCII.

    NumPy keeps a value of up to 15 bytes in UTF-8 inside its 16-byte
    element and a longer one in an arena beside the array, and the values
    that NumPy before 2.3.2 mishandles under a selection of scattered rows
    are the longer ones.
    """
    return np.asarray(
        [
            f"cell{index:05d}" + ("é" if index % 5 == 0 else "") + "x" * (index % 31)
            for index in range(n_values)
        ]
    )


def _refuse_zarr_selections(monkeypatch) -> None:
    """Make Zarr's selections of scattered rows of variable-width text fail, as they do on NumPy 2.1.

    Zarr reads a slice through its orthogonal selection too, which indexes
    each chunk with slices and stays, as do selections of other values.
    """
    orthogonal = zarr.Array.get_orthogonal_selection

    coordinate = zarr.Array.get_coordinate_selection

    def scattered(self, selection):
        parts = selection if isinstance(selection, tuple) else (selection,)
        return self.dtype.kind == "T" and not all(
            isinstance(part, slice) for part in parts
        )

    def sliced(self, selection, *args, **kwargs):
        if scattered(self, selection):
            raise AssertionError("Zarr selected scattered rows of variable-width text")
        return orthogonal(self, selection, *args, **kwargs)

    def coordinates(self, selection, *args, **kwargs):
        if self.dtype.kind == "T":
            raise AssertionError("Zarr selected scattered rows of variable-width text")
        return coordinate(self, selection, *args, **kwargs)

    monkeypatch.setattr(zarr.Array, "get_orthogonal_selection", sliced)
    monkeypatch.setattr(zarr.Array, "get_coordinate_selection", coordinates)


def test_scattered_rows_of_variable_width_text_are_read_a_chunk_span_at_a_time(
    monkeypatch,
) -> None:
    from cytearc.metadata.rows import fingerprint_stored_rows
    from cytearc.storage.artifacts import fingerprint_strings
    from cytearc.storage.partition import read_variable_text_rows

    values = _variable_text(400)
    root = zarr.open_group(store=MemoryStore(), mode="w")
    stored = root.create_array(
        "ids", data=values.astype(np.dtypes.StringDType()), chunks=(50,)
    )
    _refuse_zarr_selections(monkeypatch)
    spans: list[slice] = []
    getitem = zarr.Array.__getitem__

    def recorded(self, selection):
        spans.append(selection)
        return getitem(self, selection)

    monkeypatch.setattr(zarr.Array, "__getitem__", recorded)

    def chunks_of(selections: list[slice]) -> list[int]:
        """The stored chunk of 50 rows that each read takes a slice of."""
        assert all(isinstance(span, slice) for span in selections)
        assert all(span.start // 50 == (span.stop - 1) // 50 for span in selections)
        return [span.start // 50 for span in selections]

    scattered = np.sort(np.random.default_rng(4).choice(400, 77, replace=False))

    # Distinct ascending rows, as chunkwise reads and fingerprints take them:
    # one slice of each stored chunk that holds some of them.
    np.testing.assert_array_equal(
        read_array_rows_chunkwise(stored, scattered), values[scattered]
    )
    assert chunks_of(spans) == np.unique(scattered // 50).tolist()
    assert fingerprint_stored_rows(stored, scattered) == fingerprint_strings(
        values[scattered]
    )
    # Rows out of order and repeated, across chunks, through a table's column:
    # a slice of each chunk, from its first selected row to its last.
    rows = np.array([399, 3, 3, 120, 51, 50, 49, 0, 399])
    table = SimpleNamespace(_get_array=lambda column: stored)
    spans.clear()
    np.testing.assert_array_equal(read_metadata_rows(table, "ids", rows), values[rows])
    assert sorted((span.start, span.stop) for span in spans) == [
        (0, 50),
        (50, 52),
        (120, 121),
        (399, 400),
    ]
    chunks_of(spans)
    # Text in memory, and no rows.
    in_memory = values.astype(np.dtypes.StringDType())
    np.testing.assert_array_equal(
        read_variable_text_rows(in_memory, rows), values[rows]
    )
    empty = read_variable_text_rows(stored, np.array([], dtype=np.int64))
    assert empty.shape == (0,) and empty.dtype == stored.dtype


# Reads of scattered rows of the issue's IDs, and a copy of those rows. On
# NumPy 2.1, Zarr's selection of these rows crashes the interpreter.
_SCATTERED_TEXT_READS = """
import json
import numpy as np
import zarr
from zarr.storage import MemoryStore
from cytearc.metadata.rows import read_array_rows_chunkwise
from cytearc.storage.copy import copy_zarr_group_tree

values = np.asarray([f"cell{index:05d}" + "x" * (index % 9) for index in range(400)])
root = zarr.open_group(store=MemoryStore(), mode="w")
root.create_array("ids", data=values.astype(np.dtypes.StringDType()), chunks=(50,))
rows = np.sort(np.random.default_rng(4).choice(400, 77, replace=False))
read = read_array_rows_chunkwise(root["ids"], rows)
copy = zarr.open_group(store=MemoryStore(), mode="w")
copy_zarr_group_tree(root, copy, row_indices=rows)
print(json.dumps([read.tolist(), copy["ids"][:].tolist()]))
"""


def test_reading_scattered_variable_width_text_does_not_crash_the_interpreter() -> None:
    import subprocess
    import sys

    # A crash would stop the test run, so the reads run in a process of their own.
    result = subprocess.run(
        [sys.executable, "-c", _SCATTERED_TEXT_READS],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"reading scattered rows of variable-width text exited with "
        f"{result.returncode}:\n{result.stderr[-3000:]}"
    )
    values = np.asarray(
        [f"cell{index:05d}" + "x" * (index % 9) for index in range(400)]
    )
    expected = values[np.sort(np.random.default_rng(4).choice(400, 77, replace=False))]
    read, copied = json.loads(result.stdout)
    assert read == copied == expected.tolist()


def _sized_bytes(utf8: int) -> int:
    """The bytes that NumPy's arena takes for a value of ``utf8`` UTF-8 bytes before it grows (NEP 55): none up to 15 bytes, then the bytes behind a size of a byte up to 255 bytes and of 8 beyond."""
    if utf8 <= 15:
        return 0
    return utf8 + (1 if utf8 <= 255 else 8)


def _arena_bytes(utf8: int) -> int:
    """The bytes of a value of ``utf8`` UTF-8 bytes that the extent of variable-width text counts in NumPy's arena: none up to 15 bytes, then its bytes."""
    return utf8 if utf8 > 15 else 0


@pytest.mark.parametrize("part_bytes", [None, 1], ids=["parts", "one value a part"])
@pytest.mark.parametrize(
    "values",
    [
        ["a", "", "c" * 23],
        ["a", "b\u00e9", "\u00ff" * 9],
        ["\u0100", "x"],
        ["\u4e2d\u6587", "plain"],
        ["\N{GRINNING FACE}x", "\u00e9", "\u4e2d"],
        ["\u00e9" * 5, "\u4e2d", "abcdefgh"],
        ["x" * 300, "y" * 20, "\u4e2d" * 90],
        ["\u00e9" * 300 + "\N{GRINNING FACE}", "a", "b" * 280],
        ["", "", "abc"],
        ["", ""],
        [],
    ],
    ids=[
        "ascii",
        "latin-1",
        "beyond-latin-1",
        "bmp",
        "astral",
        "longest-in-characters-not-bytes",
        "beyond-255-bytes",
        "long-beyond-ascii",
        "a-chunk-of-empty-values-first",
        "empty-values",
        "none",
    ],
)
def test_the_extent_of_variable_width_text_bounds_each_value(
    values, part_bytes, monkeypatch
) -> None:
    import cytearc.storage.geometry as geometry_module
    from cytearc.storage.geometry import variable_text_extent

    if part_bytes is not None:
        # Each value is measured in a part of its own.
        monkeypatch.setattr(geometry_module, "_TEXT_MEASURE_BYTES", part_bytes)
    texts = np.asarray(values, dtype=np.dtypes.StringDType())
    root = zarr.open_group(store=MemoryStore(), mode="w")
    stored = root.create_array("ids", data=texts, chunks=(2,))
    utf8 = [len(value.encode("utf-8")) for value in values]
    for array in (stored, texts):
        extent = variable_text_extent(array)
        assert extent.characters == max(map(len, values), default=0)
        assert extent.utf8Bytes == max(utf8, default=0)
        assert extent.widest == max(
            (ord(character) for value in values for character in value), default=0
        )
        assert extent.values == len(values)
        assert extent.characterTotal == sum(map(len, values))
        assert extent.arenaBytes == sum(map(_arena_bytes, utf8))
        assert text_width(array) == extent.characters


@pytest.mark.parametrize("n_values", [19_250, 20_250])
@pytest.mark.parametrize("encoded_bytes", [1, 15, 16, 255, 256, 1_000])
def test_numpy_holds_variable_width_values_as_their_extent_counts(
    encoded_bytes, n_values
) -> None:
    import gc
    import tracemalloc

    from cytearc.storage.geometry import variable_text_extent

    # Values of one length in UTF-8, made from Python strings as Zarr's
    # decode makes them; at 19,250 values on NumPy 2.1 and 20,250 on NumPy
    # 2.5 the arena has just grown by a quarter.
    strings = np.asarray(
        [
            (f"{index:06d}" + "x" * encoded_bytes)[:encoded_bytes]
            for index in range(n_values)
        ],
        dtype=object,
    )
    gc.collect()
    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        values = strings.astype(np.dtypes.StringDType())
        held = tracemalloc.get_traced_memory()[0] - before
    finally:
        tracemalloc.stop()
    extent = variable_text_extent(values)
    charged = len(values) * extent.valueBytes()
    assert extent.keptBytes() <= charged
    # NumPy keeps each value's 16-byte element and, beyond 15 bytes, its
    # sized bytes in an arena that it grows by a quarter at a time (NEP 55),
    # so it holds at least the elements and the sized bytes. The extent
    # counts the elements and the values' bytes, and leaves out the sizes and
    # the arena's growth, at most a quarter of the arena beside them.
    sized = _sized_bytes(encoded_bytes)
    least = len(values) * (16 + sized)
    assert least <= held
    assert held_within_charge(held, extent.keptBytes() + extent.arenaBytes // 4)


def test_a_column_of_variable_width_text_is_measured_once_in_an_operation(
    monkeypatch,
) -> None:
    import cytearc.storage.content_stamps as content_stamps
    import cytearc.storage.geometry as geometry_module
    from cytearc.metadata.rows import read_rows
    from cytearc.storage.validation_scope import validation_scope
    from cytearc.utils.memory_budget import MemoryLedger

    values = _variable_text(400)
    root = zarr.open_group(store=MemoryStore(), mode="w")
    stored = root.create_array(
        "ids", data=values.astype(np.dtypes.StringDType()), chunks=(10,)
    )
    measured: list[int] = []
    measure = geometry_module._measured_text_extent

    def counted(array, ledger):
        measured.append(1)
        return measure(array, ledger)

    def no_stamp(array):
        raise AssertionError("a read of variable-width text stamped its column")

    monkeypatch.setattr(geometry_module, "_measured_text_extent", counted)
    monkeypatch.setattr(content_stamps, "stored_content_stamp", no_stamp)
    rows = np.arange(0, 400, 3)

    def read() -> np.ndarray:
        return read_rows(MemoryLedger(None, "Reading"), stored, rows, item="the IDs")

    # The chunks alone need no measure.
    geometry = geometry_module.array_geometry(stored)
    assert geometry is not None and geometry.text is None and measured == []
    # In an operation, the column is measured for its first read and its
    # later reads, and another opening of the array, reuse the measure.
    with validation_scope():
        assert read().tolist() == values[rows].tolist()
        read()
        assert text_width(root["ids"]) == max(map(len, values))
    assert len(measured) == 1
    # Each operation measures it anew, and so does each read outside one.
    stored[1] = "d" * 90
    with validation_scope():
        assert text_width(stored) == 90
        # An array of the same path in another store is another column.
        other = zarr.open_group(store=MemoryStore(), mode="w").create_array(
            "ids", data=np.asarray(["e" * 7]).astype(np.dtypes.StringDType())
        )
        assert text_width(other) == 7
    read()
    read()
    assert len(measured) == 5


def test_measuring_variable_width_text_is_charged_before_and_as_it_reads(
    tmp_path, monkeypatch
) -> None:
    from cytearc.metadata.rows import read_rows, read_rows_bytes
    from cytearc.storage.geometry import variable_text_extent
    from cytearc.utils.memory_budget import MemoryBudgetError, MemoryLedger

    values = _random_text("latin-1", 120, 6_000)
    root = zarr.open_group(tmp_path / "ids.zarr", mode="w")
    stored = root.create_array(
        "ids", data=values.astype(np.dtypes.StringDType()), chunks=(2_000,)
    )
    # The measure holds what it charges, chunk by chunk.
    traced = traced_charge(lambda ledger: variable_text_extent(stored, ledger))
    assert held_within_charge(traced.held, traced.charged)
    # A chunk whose values cannot all be decoded within the budget, even as
    # empty strings, is refused before it is read.
    reads: list[object] = []
    getitem = zarr.Array.__getitem__
    monkeypatch.setattr(
        zarr.Array,
        "__getitem__",
        lambda self, key: reads.append(key) or getitem(self, key),
    )
    with pytest.raises(MemoryBudgetError, match="a measure of the text of 'ids'"):
        variable_text_extent(stored, MemoryLedger(100_000, "Reading the IDs"))
    assert reads == []
    # A chunk that fits as empty strings but not as its values is refused
    # once it is measured, before the next is read.
    with pytest.raises(MemoryBudgetError, match="a measure of the text of 'ids'"):
        variable_text_extent(stored, MemoryLedger(1_000_000, "Reading the IDs"))
    assert len(reads) == 1
    monkeypatch.undo()
    # Outside an operation, a read of a row measures the column first, which
    # holds more than reading the row and which its planner counts.
    rows = np.array([3_000])
    traced = traced_charge(
        lambda ledger: read_rows(ledger, stored, rows, item="the IDs")
    )
    assert held_within_charge(traced.held, traced.charged)
    assert held_within_charge(traced.held, read_rows_bytes(stored, len(rows)))
    rows = np.arange(0, 6_000, 7)
    # Text read from memory stays charged as NumPy keeps it.
    ledger = MemoryLedger(None, "Reading the IDs")
    taken = read_rows(ledger, values.astype(np.dtypes.StringDType()), rows, item="x")
    utf8 = [len(value.encode("utf-8")) for value in taken.tolist()]
    assert ledger.held_bytes >= 16 * len(utf8) + sum(map(_arena_bytes, utf8))


# Values of random digits and then one character repeated, which compress as
# identifiers do, and values of random characters, which barely compress:
# ASCII, up to U+00FF, and beyond U+FFFF.
_TEXT_KINDS = {"ascii": "x", "latin-1": "\u00e9", "astral": "\N{GRINNING FACE}"}
_ALPHABETS = {
    "ascii": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "latin-1": "".join(map(chr, range(0xC0, 0x100))),
    "astral": "".join(map(chr, range(0x1F600, 0x1F650))),
}


def _digits_then(character: str, characters: int, n_values: int) -> np.ndarray:
    rng = np.random.default_rng(characters)
    digits = min(8, characters)
    return np.asarray(
        [
            "".join(rng.choice(list("0123456789"), digits))
            + character * (characters - digits)
            for _ in range(n_values)
        ]
    )


def _random_text(kind: str, characters: int, n_values: int) -> np.ndarray:
    rng = np.random.default_rng(characters)
    letters = rng.choice(list(_ALPHABETS[kind]), size=(n_values, characters))
    return np.asarray(["".join(row) for row in letters])


def _chunk_bytes(values: list[str]) -> tuple[int, int]:
    """The least and the most bytes that decoding a stored chunk of ``values`` holds, from known bytes.

    At the least, the chunk's Python strings and their references, as
    ``sys.getsizeof`` gives them, and NumPy's elements and sized bytes (NEP
    55). At the most, also the chunk's stored bytes, a 4-byte count and a
    4-byte length and the UTF-8 bytes of each value, both as decompressed and
    as stored without compressing, the UTF-8 copy that each string beyond
    ASCII keeps, and the quarter that NumPy grows its arena by.
    """
    import sys

    utf8 = [len(value.encode("utf-8")) for value in values]
    strings = sum(map(sys.getsizeof, values)) + 8 * len(values)
    sized = sum(map(_sized_bytes, utf8))
    least = strings + 16 * len(values) + sized
    stored = 4 + sum(4 + length for length in utf8)
    copies = sum(
        length + 1
        for value, length in zip(values, utf8, strict=True)
        if not value.isascii()
    )
    most = 2 * stored + strings + copies + 16 * len(values) + sized + sized // 4 + 1
    return least, most


@pytest.mark.parametrize(
    ("kind", "characters", "n_values", "chunks", "shards", "random"),
    [
        # Short values in one chunk, which a read of any of them decodes.
        ("ascii", 6, 20_000, 20_000, None, False),
        # Values of 16 bytes or more, which NumPy keeps beside its elements.
        ("latin-1", 12, 4_000, 1_024, None, False),
        # Long values that barely compress, in one chunk.
        ("ascii", 120, 4_000, 4_000, None, True),
        ("latin-1", 120, 4_000, 4_000, None, True),
        ("astral", 120, 4_000, 4_000, None, True),
        # Columns of 40 chunks, and shards of 5 chunks.
        ("ascii", 50, 20_000, 500, None, True),
        ("latin-1", 50, 20_000, 1_000, 5_000, True),
        # Short values, whose strings and NumPy values are charged exactly,
        # so that only what Zarr's read leaves of its objects lies between
        # what the whole read as strings holds and its charge: the objects of
        # the read itself in one shard of 5 chunks, and those of each chunk in
        # 40 chunks.
        ("ascii", 6, 20_000, 4_000, 20_000, False),
        ("ascii", 6, 20_000, 500, None, False),
    ],
    ids=[
        "short",
        "short-in-the-arena",
        "long-ascii",
        "long-latin-1",
        "long-astral",
        "forty-chunks",
        "sharded",
        "short-in-one-shard",
        "short-in-forty-chunks",
    ],
)
def test_reads_of_variable_width_text_hold_what_they_charge(
    kind, characters, n_values, chunks, shards, random, tmp_path
) -> None:
    import os

    from cytearc.metadata.rows import (
        array_row_selection_parts,
        read_rows,
        read_rows_bytes,
        read_text_rows,
        read_whole,
        read_whole_text,
        text_read_bytes,
    )
    from cytearc.storage.geometry import variable_text_extent
    from cytearc.storage.validation_scope import validation_scope

    values = (
        _random_text(kind, characters, n_values)
        if random
        else _digits_then(_TEXT_KINDS[kind], characters, n_values)
    )
    root = zarr.open_group(tmp_path / "ids.zarr", mode="w")
    stored = root.create_array(
        "ids",
        data=values.astype(np.dtypes.StringDType()),
        chunks=(chunks,),
        **({} if shards is None else {"shards": (shards,)}),
    )
    rng = np.random.default_rng(1)
    scattered = np.sort(rng.choice(n_values, n_values // 3, replace=False))
    one = np.array([chunks // 2])
    # Two rows at either end of a chunk, which a read takes the chunk's span of.
    two = np.array([chunks, 2 * chunks - 1]) if n_values >= 2 * chunks else one
    fixed, per_row = array_row_selection_parts(stored)
    # Each read, and what the planners of reads state it holds beside its
    # own charge.
    reads = {
        "whole": (lambda ledger: read_whole(ledger, root, "ids", item="the IDs"), None),
        "one row": (
            lambda ledger: read_rows(ledger, stored, one, item="the IDs"),
            read_rows_bytes(stored, 1),
        ),
        "two rows": (
            lambda ledger: read_rows(ledger, stored, two, item="the IDs"),
            read_rows_bytes(stored, len(two)),
        ),
        "scattered rows": (
            lambda ledger: read_rows(ledger, stored, scattered, item="the IDs"),
            min(
                read_rows_bytes(stored, len(scattered)),
                fixed + len(scattered) * per_row,
            ),
        ),
        "rows as strings": (
            lambda ledger: read_text_rows(ledger, stored, scattered, item="the IDs"),
            text_read_bytes(stored, len(scattered)),
        ),
        "whole as strings": (
            lambda ledger: read_whole_text(ledger, root, "ids", item="the IDs"),
            text_read_bytes(stored, len(values)),
        ),
    }
    traced = {}
    # An operation measures the column once, before its reads.
    with validation_scope():
        extent = variable_text_extent(stored)
        for name, (read, planned) in reads.items():
            traced[name] = traced_charge(read)
            assert held_within_charge(traced[name].held, traced[name].charged), name
            assert planned is None or held_within_charge(traced[name].held, planned), (
                name
            )
            # What the read returns stays charged, beside the few KiB of
            # objects that Zarr keeps after a read and up to a quarter of the
            # arena more, which NumPy grows the arena by and the charge leaves
            # out.
            assert held_within_charge(
                traced[name].kept, traced[name].kept_charged + extent.arenaBytes // 4
            ), name
    if shards is None:
        # One row decodes its chunk, which holds at least the known bytes of
        # the chunk's values, its stored bytes as compressed among them, and
        # is charged no more than twice their most, beside the row itself and
        # the allowance of a read of one chunk, 128 KiB and 4 KiB.
        chunk = values[(one[0] // chunks) * chunks :][:chunks].tolist()
        key = stored.metadata.encode_chunk_key((int(one[0]) // chunks,))
        compressed = os.path.getsize(tmp_path / "ids.zarr" / "ids" / key)
        least, most = _chunk_bytes(chunk)
        row = 3 * (16 + _arena_bytes(len(values[one[0]].encode("utf-8"))))
        assert traced["one row"].held >= compressed + least
        assert traced["one row"].charged <= 2 * most + row + 132 * 1024


def test_a_read_of_variable_width_labels_is_charged_by_their_characters(
    tmp_path,
) -> None:
    from cytearc.metadata.rows import read_whole
    from cytearc.storage.geometry import array_geometry

    # 20,000 labels of 71 characters in one chunk, which a read decodes whole.
    labels = _digits_then("x", 71, 20_000)
    root = zarr.open_group(tmp_path / "labels.zarr", mode="w")
    stored = root.create_array(
        "labels", data=labels.astype(np.dtypes.StringDType()), chunks=(20_000,)
    )
    traced = traced_charge(
        lambda ledger: read_whole(ledger, root, "labels", item="the labels")
    )
    assert held_within_charge(traced.held, traced.charged)
    # A charge at NumPy's 16-byte elements, as such a read had, is passed.
    elements = array_geometry(stored)
    assert elements is not None and traced.held > elements.wholeReadBytes(1)


def test_one_long_label_does_not_widen_the_charge_of_every_label(tmp_path) -> None:
    from cytearc.metadata.rows import read_rows, read_whole

    # 20,000 labels of 15 characters and one of 5,000, in chunks of 5,000:
    # the read holds each label as long as it is, so its charge counts every
    # label's own bytes, and the longest only in its own chunk.
    labels = [f"cell-{index:08d}-x" for index in range(20_000)]
    labels[123] = "y" * 5_000
    root = zarr.open_group(tmp_path / "labels.zarr", mode="w")
    stored = root.create_array(
        "labels",
        data=np.asarray(labels, dtype=np.dtypes.StringDType()),
        chunks=(5_000,),
    )
    whole = traced_charge(lambda ledger: read_whole(ledger, root, "labels", item="v"))
    assert held_within_charge(whole.held, whole.charged)
    # Every chunk decoded at once, each at most as its own values, beside
    # the read's result: no more than the known bytes of the labels, four
    # times over, and far from 20,000 labels of 5,000 characters.
    _, most = _chunk_bytes(labels)
    assert whole.charged <= 4 * most
    # A read of a few rows far from the long label holds and is charged
    # like any other.
    rows = traced_charge(
        lambda ledger: read_rows(ledger, stored, np.arange(10_000, 10_010), item="v")
    )
    assert held_within_charge(rows.held, rows.charged)
    assert rows.charged <= 4 * most


@pytest.mark.parametrize("kind", list(_TEXT_KINDS))
def test_variable_width_text_reads_as_python_strings(kind) -> None:
    import sys

    from cytearc.metadata.rows import read_text_rows, read_whole_text

    values = _digits_then(_TEXT_KINDS[kind], 30, 50)
    root = zarr.open_group(store=MemoryStore(), mode="w")
    stored = root.create_array(
        "ids", data=values.astype(np.dtypes.StringDType()), chunks=(8,)
    )
    rows = np.array([49, 0, 17, 18, 3])
    in_memory = values.astype(np.dtypes.StringDType())
    for read, expected in (
        (lambda ledger: read_text_rows(ledger, stored, rows, item="IDs"), values[rows]),
        (lambda ledger: read_whole_text(ledger, root, "ids", item="IDs"), values),
        # Text in memory, which has no chunks, is measured as it is.
        (
            lambda ledger: read_text_rows(ledger, in_memory, rows, item="IDs"),
            values[rows],
        ),
    ):
        ledger = recording_ledger("Reading")
        strings = read(ledger)
        assert strings.dtype == object
        assert strings.tolist() == expected.tolist()
        # The ledger keeps the strings and their references charged, at
        # least their characters and a reference each, and lets go of the
        # stored rows, which would add their elements and UTF-8 bytes.
        characters = sum(map(len, strings)) + 8 * len(strings)
        actual = sum(map(sys.getsizeof, strings)) + 8 * len(strings)
        rows_kept = sum(16 + len(value.encode("utf-8")) for value in strings)
        assert characters <= ledger.held_bytes < actual + rows_kept
    # No row reads no string and keeps nothing charged.
    ledger = recording_ledger("Reading")
    empty = read_text_rows(ledger, stored, np.array([], dtype=np.int64), item="IDs")
    assert empty.dtype == object and empty.size == 0 and ledger.held_bytes == 0
    numbers = root.create_array("numbers", data=np.arange(5), chunks=(5,))
    with pytest.raises(TypeError, match="IDs must be stored as text, not as int64"):
        read_text_rows(recording_ledger("Reading"), numbers, np.arange(2), item="IDs")
    with pytest.raises(TypeError, match="IDs must be stored as text, not as int64"):
        read_whole_text(recording_ledger("Reading"), root, "numbers", item="IDs")


def test_rows_of_variable_width_text_in_memory_are_planned_by_their_longest_value() -> (
    None
):
    from cytearc.metadata.rows import array_row_selection_parts, read_rows_bytes

    # 1,000 values of 100 bytes and one of 300: NumPy keeps each in its
    # 16-byte element and the longest's bytes behind an 8-byte size in its
    # arena (NEP 55), and a selection holds each row three times. The fixed
    # bytes are the measure's, which keeps 8 bytes four times for each value.
    values = np.asarray(["x" * 100] * 1_000 + ["y" * 300]).astype(
        np.dtypes.StringDType()
    )
    fixed, per_row = array_row_selection_parts(values)
    assert fixed >= 32 * len(values)
    assert per_row >= 3 * (16 + 300 + 8)
    # A read of every row holds each value's element and sized bytes, and
    # their strings too as Python strings, each at least its characters and a
    # reference.
    least = 16 * len(values) + 1_000 * 101 + 308
    strings = sum(map(len, values.tolist())) + 8 * len(values)
    assert read_rows_bytes(values, len(values)) >= least
    assert text_read_bytes(values, len(values)) >= least + strings
    assert text_row_selection_parts(values)[1] >= per_row + 300


def test_a_whole_read_of_a_scalar_keeps_its_value_charged() -> None:
    from cytearc.metadata.rows import read_whole

    root = zarr.open_group(store=MemoryStore(), mode="w")
    root.create_array("count", data=np.asarray(7, dtype=np.int64))
    ledger = recording_ledger("Reading")
    value = read_whole(ledger, root, "count", item="the count")
    assert value.shape == () and int(value) == 7
    assert ledger.held_bytes == 8
