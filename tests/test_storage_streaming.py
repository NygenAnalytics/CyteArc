import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from cytearc.metadata import MetaData
from cytearc.storage.arrays import MetadataBlock, create_streamed_metadata_column
from cytearc.storage.artifacts import (
    artifact_group,
    fingerprint_strings,
)
from cytearc.storage.schema import (
    create_empty_cell_data,
    create_empty_zarr_count_assay,
)
from cytearc.storage.selections import (
    fingerprint_selected_stored_strings,
    resolve_stored_selection_artifact,
)


def test_streamed_metadata_column_writes_contiguous_blocks_and_mask():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    group = root.create_group("cellData")
    values = create_streamed_metadata_column(
        group,
        "cluster",
        shape=4,
        dtype="U2",
        blocks=(
            MetadataBlock(0, np.array(["a", ""], dtype="U2"), np.array([0, 1])),
            MetadataBlock(2, np.array(["bb", "a"], dtype="U2"), np.array([0, 0])),
        ),
        hasMissing=True,
    )

    assert values[:].tolist() == ["a", "", "bb", "a"]
    missing_name = values.attrs["missing_mask"]
    assert group[missing_name][:].tolist() == [False, True, False, False]


def test_streamed_metadata_column_casts_blocks_and_marks_unflagged_rows_present():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    values = create_streamed_metadata_column(
        root,
        "count",
        shape=5,
        dtype=np.int16,
        blocks=(
            MetadataBlock(0, np.array([1, 2], dtype=np.int64), np.array([0, 1])),
            # A block without a mask holds only present values.
            MetadataBlock(2, np.array([3.0, 4.0, 5.0])),
        ),
        chunkSize=2,
        hasMissing=True,
    )

    assert values.dtype == np.dtype(np.int16)
    assert values.chunks == (2,)
    np.testing.assert_array_equal(values[:], [1, 2, 3, 4, 5])
    mask = root[values.attrs["missing_mask"]]
    np.testing.assert_array_equal(mask[:], [False, True, False, False, False])


@pytest.mark.parametrize(
    ("options", "blocks", "message"),
    [
        ({"shape": -1}, (), "shape must be non-negative"),
        ({"chunkSize": 0}, (), "chunkSize must be positive"),
        (
            {},
            (MetadataBlock(1, np.array([], dtype=np.int32)),),
            "Metadata blocks must be contiguous; expected 0, received 1",
        ),
        (
            {},
            (MetadataBlock(0, np.zeros((1, 1), dtype=np.int32)),),
            "Metadata blocks must be one-dimensional",
        ),
        (
            {},
            (MetadataBlock(0, np.arange(4, dtype=np.int32)),),
            "Metadata block exceeds declared shape",
        ),
        (
            {},
            (MetadataBlock(0, np.arange(3, dtype=np.int32), np.zeros(3)),),
            "A missing mask was supplied but hasMissing is false",
        ),
        (
            {"hasMissing": True},
            (MetadataBlock(0, np.arange(3, dtype=np.int32), np.zeros(2)),),
            "Missing mask must align with metadata values",
        ),
        (
            {},
            (MetadataBlock(0, np.arange(2, dtype=np.int32)),),
            "Metadata column is incomplete: wrote 2 of 3 rows",
        ),
    ],
    ids=[
        "negative-shape",
        "zero-chunks",
        "gap",
        "two-dimensional",
        "overflow",
        "unexpected-mask",
        "misaligned-mask",
        "incomplete",
    ],
)
def test_streamed_metadata_column_rejects_malformed_blocks(options, blocks, message):
    root = zarr.open_group(store=MemoryStore(), mode="w")
    arguments = {"shape": 3, "dtype": np.int32, **options}
    with pytest.raises(ValueError, match=f"^{message}"):
        create_streamed_metadata_column(root, "bad", blocks=blocks, **arguments)


def test_internal_missing_columns_are_not_public_metadata():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    group = create_empty_cell_data(root, None, 2, "U2", "U2")
    group["ids"][:] = np.array(["c1", "c2"])
    group["names"][:] = np.array(["c1", "c2"])
    create_streamed_metadata_column(
        group,
        "label",
        shape=2,
        dtype="U1",
        blocks=(MetadataBlock(0, np.array(["a", ""], dtype="U1"), np.array([0, 1])),),
        hasMissing=True,
    )

    metadata = MetaData(group)
    # The mask is stored beside the column, but only the column is public.
    assert "__cytearc_missing__label" in group
    assert metadata.columns == ["I", "ids", "names", "label"]


def test_empty_assay_schema_accepts_blockwise_feature_metadata():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    counts, feature_data = create_empty_zarr_count_assay(
        root,
        "RNA",
        None,
        3,
        2,
        "U2",
        "U6",
        dtype=np.uint32,
    )
    feature_data["ids"][:] = np.array(["g1", "g2"])
    feature_data["names"][:] = np.array(["Gene 1", "Gene 2"])

    assert counts.shape == (3, 2)
    assert feature_data["ids"][:].tolist() == ["g1", "g2"]
    assert feature_data["I"][:].tolist() == [True, True]


def test_stored_selection_artifact_copies_values_blockwise():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    cells = create_empty_cell_data(root, None, 4, "U2", "U2")
    cells["ids"][:] = np.array(["c1", "c2", "c3", "c4"])
    cells["names"][:] = cells["ids"][:]
    cells["I"][:] = np.array([True, False, True, True])

    ref = resolve_stored_selection_artifact(
        root,
        table_path="cellData",
        id_column="ids",
        source_column="I",
        scope="datastore",
        kind="cell_selection",
        operation="import_cell_selection",
        parameters={"source": "seurat"},
        inputs={},
    )

    assert artifact_group(root, ref)["values"][:].tolist() == [
        True,
        False,
        True,
        True,
    ]


def test_selected_stored_string_fingerprint_matches_materialized_values():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    ids = root.create_array(
        "ids",
        data=np.array(["cell-1", "cell-2", "cell-3", "cell-4"], dtype="U6"),
        chunks=(2,),
    )
    selection = root.create_array(
        "selection",
        data=np.array([True, False, True, True]),
        chunks=(2,),
    )

    digest, count = fingerprint_selected_stored_strings(ids, selection)

    assert count == 3
    assert digest == fingerprint_strings(ids[:][selection[:]])


def _whole_cell_table(root: zarr.Group, ids: np.ndarray) -> zarr.Group:
    """Write a cell table as imports wrote it before it was written in blocks: each column whole."""
    from cytearc.storage.arrays import create_zarr_obj_array, encode_metadata_values
    from cytearc.storage.layout import column_chunk_rows

    group = root.create_group("cellData")
    for name, values in (("ids", ids), ("names", ids), ("I", np.ones(len(ids), bool))):
        stored = encode_metadata_values(values, values.dtype, name=name)
        create_zarr_obj_array(
            group,
            name,
            stored,
            stored.dtype,
            chunk_size=column_chunk_rows(len(stored), max(1, stored.dtype.itemsize)),
        )
    return group


_CELL_ID_KINDS = {
    # 250,000 IDs of up to 11 characters are three chunks of 100,000 rows.
    "unicode": np.array([f"cell-{row}" for row in range(250_000)]),
    "wide-unicode": np.array([f"c{row}" for row in range(30)], dtype="U40"),
    "bytes": np.array([f"cell-{row}".encode() for row in range(120_000)]),
    "utf8-bytes": np.array([f"célula-{row}".encode() for row in range(1_000)]),
    "objects": np.array([f"cell-{row}" for row in range(150_000)], dtype=object),
    "objects-with-bytes": np.array(
        [f"c{row}" if row % 2 else f"c{row}".encode() for row in range(1_000)],
        dtype=object,
    ),
    "numbers": np.arange(300_000, dtype=np.int64),
    "empty": np.array([], dtype="U3"),
    "empty-objects": np.array([], dtype=object),
}


@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("ordered", [False, True], ids=["in-place", "ordered"])
@pytest.mark.parametrize("kind", list(_CELL_ID_KINDS))
def test_a_cell_table_written_a_block_at_a_time_stores_what_a_whole_write_stored(
    kind: str, ordered: bool, threads: int
) -> None:
    from cytearc.storage.async_execution import register_store_threads
    from cytearc.storage.schema import create_cell_data

    ids = _CELL_ID_KINDS[kind]
    order = np.random.default_rng(6).permutation(len(ids)) if ordered else None
    whole_store, block_store = MemoryStore(), MemoryStore()
    register_store_threads(block_store, threads)
    _whole_cell_table(
        zarr.open_group(store=whole_store, mode="w"),
        ids if order is None else ids[order],
    )
    blocks = create_cell_data(
        zarr.open_group(store=block_store, mode="w"), None, ids, ids, order=order
    )
    # The stores hold the same arrays, metadata and chunks byte for byte.
    assert sorted(block_store._store_dict) == sorted(whole_store._store_dict)
    for key, value in whole_store._store_dict.items():
        assert block_store._store_dict[key].to_bytes() == value.to_bytes(), key
    # Text is stored decoded, row r holding the ID of the order's row r.
    if len(ids) and kind != "numbers":
        row = len(ids) - 1
        source = ids[row if order is None else order[row]]
        expected = source.decode() if isinstance(source, bytes) else str(source)
        assert blocks["ids"][row] == expected == blocks["names"][row]
    assert blocks["I"][:].all()


def test_a_cell_table_reads_names_apart_from_ids_and_refuses_misaligned_ones() -> None:
    from cytearc.storage.schema import create_cell_data

    root = zarr.open_group(store=MemoryStore(), mode="w")
    ids = np.array(["a", "b", "c"])
    names = np.array(["x", "yy", "zzz"], dtype=object)
    group = create_cell_data(root, None, ids, names, order=np.array([2, 0, 1]))
    assert group["ids"][:].tolist() == ["c", "a", "b"]
    assert group["names"][:].tolist() == ["zzz", "x", "yy"]
    assert group["names"].dtype == np.dtype("U3")
    with pytest.raises(ValueError, match="one name for each of its 3 cells; found 2"):
        create_cell_data(root, "w1", ids, names[:2])
    for order in (np.array([0, 1]), np.array([0.0, 1.0, 2.0])):
        with pytest.raises(ValueError, match="one integer row for each of its 3"):
            create_cell_data(root, "w2", ids, ids, order=order)
    # Nothing was written for the refused tables.
    assert "w1" not in root and "w2" not in root


def _astral_ids(n_rows: int, width: int) -> np.ndarray:
    """Return IDs of ``width`` random characters beyond the Basic Multilingual Plane, which barely compress."""
    codes = np.random.default_rng(9).integers(
        0x10000, 0x10FFFF, size=(n_rows, width), dtype=np.uint32
    )
    return codes.view(f"<U{width}").ravel()


# 350,000 IDs of 10 characters, 40 bytes each, are three chunks of 100,000
# rows, 4,000,000 bytes, and 50,000 rows of a fourth, which Zarr fills before
# encoding it; a codec's buffer of a chunk takes at most a 128th more and
# 1,024 bytes: 4,032,274.
_ID_CHUNK = 4_000_000
_ID_ENCODED = 4_032_274


@pytest.mark.parametrize(
    ("kind", "ordered", "threads", "expected", "least"),
    [
        # In place, a block lends its rows to Zarr: the last one holds the
        # filled chunk and its encoded buffer.
        ("unicode", False, 1, _ID_CHUNK + _ID_ENCODED, 2 * _ID_CHUNK),
        # Gathered in an order, the last block's 50,000 IDs are a copy
        # beside them.
        ("unicode", True, 1, 2_000_000 + _ID_CHUNK + _ID_ENCODED, 10_000_000),
        # Two threads write the last two chunks at once beside their
        # 150,000 gathered IDs.
        (
            "unicode",
            True,
            2,
            6_000_000 + _ID_CHUNK + 2 * _ID_ENCODED,
            6_000_000 + 3 * _ID_CHUNK,
        ),
        # Objects become text, which the last block holds as its 50,000
        # rows, beside the filled chunk and its buffer: more than a whole
        # block's references and two copies of its text, 88 bytes a row.
        ("objects", True, 1, 2_000_000 + _ID_CHUNK + _ID_ENCODED, 10_000_000),
    ],
)
def test_a_cell_table_write_holds_what_its_size_helper_charges(
    kind: str, ordered: bool, threads: int, expected: int, least: int
) -> None:
    from contextlib import nullcontext

    from zarr.codecs import BloscCodec

    from cytearc.storage.async_execution import register_store_threads
    from cytearc.storage.schema import cell_table_write_bytes, create_cell_data
    from tests.memory_helpers import (
        chunkless_store,
        encodes_held_together,
        traced_charge,
    )

    ids = _astral_ids(350_000, 10)
    if kind == "objects":
        ids = ids.astype(object)
    order = np.random.default_rng(7).permutation(350_000) if ordered else None
    charge = cell_table_write_bytes(ids, ids, threads, order=order)
    assert charge == expected

    def write(_ledger: object) -> None:
        store = chunkless_store()
        register_store_threads(store, threads)
        create_cell_data(
            zarr.open_group(store=store, mode="w"), None, ids, ids, order=order
        )

    together = (
        encodes_held_together(BloscCodec, threads) if threads > 1 else nullcontext()
    )
    with together:
        held = traced_charge(write).held
    # Beside the IDs and the order, the write holds at least its gathered
    # rows and its chunks' buffers, each the chunk's bytes, and at most the
    # charge with Zarr's objects, tens of kilobytes.
    assert least <= held <= charge + 128 * 1024


def test_a_cell_table_reads_a_list_of_ids_once_for_its_ids_and_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cytearc.storage.schema as schema

    # A list that the caller gives as both is one source, whose blocks are
    # made text once for both columns.
    ids = [f"cell-{row}" for row in range(5)]
    made: list[int] = []
    encode = schema.encode_metadata_values

    def recorded(values: object, *args: object, **kwargs: object) -> np.ndarray:
        made.append(len(np.asarray(values)))
        return encode(values, *args, **kwargs)

    monkeypatch.setattr(schema, "encode_metadata_values", recorded)
    group = schema.create_cell_data(
        zarr.open_group(store=MemoryStore(), mode="w"), None, ids, ids
    )
    assert made == [5]
    assert group["ids"][:].tolist() == ids == group["names"][:].tolist()


@pytest.mark.parametrize(
    ("kind", "expected", "least"),
    [
        # 200,000 IDs of one ASCII character, bytes as an H5AD reader keeps
        # them, gathered in an order: a block of 100,000 holds a reference to
        # each, the text it becomes, 4 bytes each, and NumPy's length of
        # each, 8 bytes, as the text's width is measured: more than Zarr's
        # writes of the two whole chunks of 400,000 bytes and the stored
        # block beside each.
        ("short-bytes", 100_000 * (8 + 4 + 8), 100_000 * (8 + 4 + 8)),
        # IDs of an e with an acute accent and a letter, bytes that are not
        # ASCII, become Python strings before they are text: at most 60
        # bytes and four a character each, with a reference, beside the
        # bytes as NumPy bytes and the text twice, which the stored text
        # that Zarr writes, 1,607,274 at most, does not reach.
        ("accented-bytes", 100_000 * (8 + 68 + 3 * 8), 1_607_274 + 128 * 1024),
    ],
)
def test_a_cell_table_of_bytes_holds_no_more_than_its_size_helper_charges(
    kind: str, expected: int, least: int
) -> None:
    from cytearc.storage.async_execution import register_store_threads
    from cytearc.storage.schema import cell_table_write_bytes, create_cell_data
    from tests.memory_helpers import chunkless_store, traced_charge

    letters = [chr(97 + row % 26) for row in range(200_000)]
    texts = letters if kind == "short-bytes" else [f"é{letter}" for letter in letters]
    ids = np.array([text.encode() for text in texts], dtype=object)
    order = np.random.default_rng(8).permutation(200_000)
    assert cell_table_write_bytes(ids, ids, 1, order=order) == expected

    def write(_ledger: object) -> None:
        store = chunkless_store()
        register_store_threads(store, 1)
        create_cell_data(
            zarr.open_group(store=store, mode="w"), None, ids, ids, order=order
        )

    held = traced_charge(write).held
    assert least <= held <= expected + 128 * 1024
