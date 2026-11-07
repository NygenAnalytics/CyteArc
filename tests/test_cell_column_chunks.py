"""Chunk the cell and feature columns of new stores by their rows and by 16 MiB of values.

Every writer of a store's cell and feature columns takes its chunks from
``storage.layout.column_chunk_rows``: a chunk holds at most 100,000 rows, no
more than the table's rows and no more than 16 MiB of values as stored, but
always one row. Stores written before keep their chunks and are read as
stored. Each expected chunk below is derived by
hand from 16 MiB, 16,777,216 bytes, and fixed-width text's four bytes a
character of its widest value.
"""

import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import zarr
from scipy.sparse import csr_matrix

from cytearc.metadata.rows import read_rows
from cytearc.metadata.table import MetaData
from cytearc.storage.geometry import array_geometry
from cytearc.storage.layout import (
    COLUMN_CHUNK_BYTES,
    PROFILE_METADATA_CHUNK,
    column_chunk_rows,
)
from cytearc.utils.memory_budget import MemoryLedger
from tests import full_path
from tests.storage_helpers import rechunk_column

if TYPE_CHECKING:
    from cytearc.datastore.datastore import DataStore

# A text column of 42 characters is the narrowest whose 100,000 rows pass
# 16 MiB: 16,777,216 // (4 * 42) = 99,864 rows to a chunk, against 102,300
# for 41 characters.
_WIDE = 42
_WIDE_ROWS = 99_864
# One label of 100,000 characters stores every row at 400,000 bytes, so a
# chunk holds 16,777,216 // 400,000 = 41 rows of it.
_LABEL_WIDTH = 100_000
_LABEL_ROWS = 41


@pytest.mark.parametrize(
    ("n_rows", "itemsize", "max_rows", "expected"),
    [
        # A column of no rows or one row is one chunk of one row.
        (0, 24, PROFILE_METADATA_CHUNK, 1),
        (1, 24, PROFILE_METADATA_CHUNK, 1),
        # A small store's column is one chunk of its rows.
        (48, 24, PROFILE_METADATA_CHUNK, 48),
        (100_000, 24, PROFILE_METADATA_CHUNK, 100_000),
        (100_001, 24, PROFILE_METADATA_CHUNK, 100_000),
        # Booleans and float64 values never reach 16 MiB in 100,000 rows.
        (10_000_000, 1, PROFILE_METADATA_CHUNK, 100_000),
        (10_000_000, 8, PROFILE_METADATA_CHUNK, 100_000),
        (100_001, 4 * 41, PROFILE_METADATA_CHUNK, 100_000),
        (100_001, 4 * _WIDE, PROFILE_METADATA_CHUNK, _WIDE_ROWS),
        (100_000, 4 * _WIDE, PROFILE_METADATA_CHUNK, _WIDE_ROWS),
        (_WIDE_ROWS, 4 * _WIDE, PROFILE_METADATA_CHUNK, _WIDE_ROWS),
        (1_000_000, 4 * 40, PROFILE_METADATA_CHUNK, 100_000),
        (1_000_000, 4 * 200, PROFILE_METADATA_CHUNK, 20_971),
        (48, 4 * _LABEL_WIDTH, PROFILE_METADATA_CHUNK, _LABEL_ROWS),
        # Values of 16 MiB or more take a chunk each, and half of it two.
        (3, COLUMN_CHUNK_BYTES, PROFILE_METADATA_CHUNK, 1),
        (3, COLUMN_CHUNK_BYTES + 4, PROFILE_METADATA_CHUNK, 1),
        (3, COLUMN_CHUNK_BYTES // 2, PROFILE_METADATA_CHUNK, 2),
        # A writer's own cap on rows is clipped the same way.
        (100_000, 24, 65_536, 65_536),
        (48, 24, 65_536, 48),
        (1_000_000, 4 * 200, 65_536, 20_971),
    ],
)
def test_a_chunk_holds_at_most_the_rows_and_16_mib(
    n_rows: int, itemsize: int, max_rows: int, expected: int
) -> None:
    assert COLUMN_CHUNK_BYTES == 16 * 1024 * 1024
    assert column_chunk_rows(n_rows, itemsize, max_rows=max_rows) == expected


@pytest.mark.parametrize(
    ("n_rows", "itemsize", "max_rows", "message"),
    [
        (-1, 24, PROFILE_METADATA_CHUNK, "cannot have -1 rows"),
        (48, 0, PROFILE_METADATA_CHUNK, "at least one byte each, not 0"),
        (48, 24, 0, "at least one row, not 0"),
    ],
)
def test_the_rule_refuses_a_column_it_cannot_chunk(
    n_rows: int, itemsize: int, max_rows: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        column_chunk_rows(n_rows, itemsize, max_rows=max_rows)


def _chunks(group: zarr.Group) -> dict[str, tuple[int, ...]]:
    return {name: tuple(group[name].chunks) for name in sorted(group.array_keys())}


@pytest.mark.parametrize(
    ("n_rows", "narrow", "wide"),
    [
        (1, 1, 1),
        (64, 64, 64),
        (100_000, 100_000, _WIDE_ROWS),
        (100_001, 100_000, _WIDE_ROWS),
    ],
)
def test_each_writer_chunks_new_cell_and_feature_columns_by_rows_and_bytes(
    tmp_path: Path, n_rows: int, narrow: int, wide: int
) -> None:
    from cytearc.storage.copy import copy_zarr_group_tree
    from cytearc.storage.schema import (
        _write_feature_columns,
        create_cell_data,
        create_empty_cell_data,
        create_empty_zarr_count_assay,
    )
    from cytearc.writers._store import write_membership_column, write_metadata_column

    ids = np.asarray([f"c{index}" for index in range(n_rows)])
    labels = np.asarray([f"{index:0{_WIDE}d}" for index in range(n_rows)])
    with_missing = labels.astype(object)
    with_missing[0] = None
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    create_cell_data(root, None, ids=ids, names=ids)
    cells = root["cellData"]
    write_metadata_column(cells, "label", labels)
    write_metadata_column(cells, "label_with_missing", with_missing)
    write_membership_column(cells, "RNA", np.arange(n_rows) % 2 == 0)
    MetaData(cells).insert("inserted", labels)
    MetaData(cells).insert("score", np.arange(n_rows, dtype=np.float64))
    expected = {
        "I": (narrow,),
        "RNA_I": (narrow,),
        "__cytearc_missing__label_with_missing": (wide,),
        "ids": (narrow,),
        "inserted": (wide,),
        "label": (wide,),
        "label_with_missing": (wide,),
        "names": (narrow,),
        "score": (narrow,),
    }
    assert _chunks(cells) == expected
    np.testing.assert_array_equal(cells["label"][-2:], labels[-2:])
    np.testing.assert_array_equal(cells["inserted"][-2:], labels[-2:])

    copied = zarr.open_group(tmp_path / "copy.zarr", mode="w").create_group("cellData")
    copy_zarr_group_tree(cells, copied, chunk_rows=column_chunk_rows)
    assert _chunks(copied) == expected
    for name in expected:
        np.testing.assert_array_equal(copied[name][-2:], cells[name][-2:])

    # A feature table written and copied by the same writers takes the same
    # chunks; its IDs here are the 42-character labels.
    features = root.create_group("RNA/featureData")
    _write_feature_columns(features, labels, ids, "fast_local")
    write_metadata_column(features, "label", labels)
    write_metadata_column(features, "label_with_missing", with_missing)
    MetaData(features).insert("inserted", labels)
    MetaData(features).insert("score", np.arange(n_rows, dtype=np.float64))
    feature_chunks = {
        "I": (narrow,),
        "__cytearc_missing__label_with_missing": (wide,),
        "ids": (wide,),
        "inserted": (wide,),
        "label": (wide,),
        "label_with_missing": (wide,),
        "names": (narrow,),
        "score": (narrow,),
    }
    assert _chunks(features) == feature_chunks
    np.testing.assert_array_equal(features["ids"][-2:], labels[-2:])
    np.testing.assert_array_equal(features["names"][-2:], ids[-2:])
    copied_features = zarr.open_group(tmp_path / "copy.zarr", mode="r+").create_group(
        "RNA/featureData"
    )
    copy_zarr_group_tree(features, copied_features, chunk_rows=column_chunk_rows)
    assert _chunks(copied_features) == feature_chunks
    for name in feature_chunks:
        np.testing.assert_array_equal(copied_features[name][-2:], features[name][-2:])

    empty = create_empty_cell_data(
        zarr.open_group(tmp_path / "empty.zarr", mode="w"),
        None,
        n_rows,
        f"U{_WIDE}",
        "U7",
    )
    assert _chunks(empty) == {"I": (narrow,), "ids": (wide,), "names": (narrow,)}
    _counts, empty_features = create_empty_zarr_count_assay(
        zarr.open_group(tmp_path / "empty.zarr", mode="r+"),
        "RNA",
        None,
        1,
        n_rows,
        f"U{_WIDE}",
        "U7",
        np.uint32,
    )
    assert _chunks(empty_features) == {
        "I": (narrow,),
        "ids": (wide,),
        "names": (narrow,),
    }


@pytest.mark.parametrize(
    ("character", "as_bytes"),
    [("a", True), ("é", True), ("€", False), ("\U0001f600", True)],
    ids=["ascii-utf8", "two-byte-utf8", "three-byte-str", "four-byte-utf8"],
)
def test_text_is_chunked_by_its_characters_whatever_its_utf8_bytes(
    tmp_path: Path, character: str, as_bytes: bool
) -> None:
    from cytearc.storage.schema import _write_feature_columns, create_cell_data
    from cytearc.writers._store import write_metadata_column

    # Every value is stored at four bytes a character, so 42 characters take
    # 99,864 rows to a chunk whatever bytes they take as UTF-8, also as the
    # IDs and names that the importers pass to create_cell_data and to the
    # feature table.
    n_rows = _WIDE_ROWS + 1
    texts = [character * _WIDE] * n_rows
    values = np.asarray([text.encode() for text in texts] if as_bytes else texts)
    cells = zarr.open_group(tmp_path / "store.zarr", mode="w").create_group("cellData")
    write_metadata_column(cells, "label", values)
    MetaData(cells).insert("inserted", values)
    root = zarr.open_group(tmp_path / "ids.zarr", mode="w")
    created = create_cell_data(root, None, ids=values, names=values)
    features = root.create_group("RNA/featureData")
    _write_feature_columns(features, values, values, "fast_local")
    for group, name in (
        (cells, "label"),
        (cells, "inserted"),
        (created, "ids"),
        (created, "names"),
        (features, "ids"),
        (features, "names"),
    ):
        assert group[name].dtype == np.dtype(f"U{_WIDE}")
        assert group[name].chunks == (_WIDE_ROWS,)
        assert group[name][-1] == character * _WIDE


@pytest.mark.parametrize("table", ["cellData", "RNA/featureData"])
def test_a_value_wider_than_16_mib_is_a_chunk_of_its_own(
    tmp_path: Path, table: str
) -> None:
    from cytearc.writers._store import write_metadata_column

    # 4,194,305 characters are 16,777,220 bytes, four more than 16 MiB.
    width = COLUMN_CHUNK_BYTES // 4 + 1
    values = np.asarray(["x" * width, "y"], dtype=object)
    group = zarr.open_group(tmp_path / "store.zarr", mode="w").create_group(table)
    write_metadata_column(group, "label", values)
    MetaData(group).insert("inserted", values)
    for name in ("label", "inserted"):
        assert group[name].dtype.itemsize == COLUMN_CHUNK_BYTES + 4
        assert group[name].chunks == (1,)
        assert str(group[name][1]) == "y"
        assert len(str(group[name][0])) == width


@pytest.fixture
def old_store(tmp_path: Path) -> Path:
    """A published store of 484 cells and 65,908 peaks, whose columns are stored in 100,000-row chunks."""
    path = tmp_path / "atac.zarr"
    with tarfile.open(full_path("500_pbmc_atac.zarr.tar.gz"), "r:gz") as archive:
        archive.extractall(path, filter="data")
    return path


_OLD_COLUMNS = ("ATAC_nCounts", "ATAC_nFeatures", "I", "ids", "names")
_OLD_FEATURE_COLUMNS = ("I", "dropOuts", "ids", "nCells", "names")
_N_PEAKS = 65_908


def test_an_old_store_is_read_as_stored_and_keeps_its_chunks(
    old_store: Path,
) -> None:
    from cytearc.datastore.datastore import DataStore

    root = zarr.open_group(old_store, mode="r")
    cells, features = root["cellData"], root["ATAC/featureData"]
    before = _chunks(cells)
    assert before == dict.fromkeys(_OLD_COLUMNS, (PROFILE_METADATA_CHUNK,))
    features_before = _chunks(features)
    assert features_before == dict.fromkeys(
        _OLD_FEATURE_COLUMNS, (PROFILE_METADATA_CHUNK,)
    )
    stored_ids = np.asarray(cells["ids"][:])
    assert stored_ids.shape == (484,) and stored_ids.dtype == np.dtype("U18")
    stored_peaks = np.asarray(features["ids"][:])
    assert stored_peaks.shape == (_N_PEAKS,)

    store = DataStore(str(old_store), default_assay="ATAC", zarr_mode="r", nthreads=1)
    ledger = MemoryLedger(None, "Reading IDs")
    ids = store.cells._get_array("ids")
    rows = np.arange(0, 484, 10)
    read = read_rows(ledger, ids, rows, item="IDs")
    np.testing.assert_array_equal(read, stored_ids[rows])
    peaks = store.ATAC.feats._get_array("ids")
    rows = np.arange(0, _N_PEAKS, 1_000)
    read = read_rows(ledger, peaks, rows, item="peaks")
    np.testing.assert_array_equal(read, stored_peaks[rows])
    del store
    root = zarr.open_group(old_store, mode="r")
    assert _chunks(root["cellData"]) == before
    assert _chunks(root["ATAC/featureData"]) == features_before


def test_a_new_column_written_into_an_old_store_takes_the_new_chunks(
    old_store: Path,
) -> None:
    from cytearc.datastore.datastore import DataStore

    store = DataStore(
        str(old_store), default_assay="ATAC", min_features_per_cell=0, nthreads=1
    )
    labels = np.asarray([f"label{index % 7}" for index in range(store.cells.N)])
    store.cells.insert("label", labels)
    store.cells.insert("score", np.arange(store.cells.N, dtype=np.float64))
    regions = np.asarray([f"region{index % 5}" for index in range(_N_PEAKS)])
    store.ATAC.feats.insert("region", regions)
    store.ATAC.feats.insert("weight", np.arange(_N_PEAKS, dtype=np.float64))
    root = zarr.open_group(old_store, mode="r")
    after = _chunks(root["cellData"])
    # The columns already stored keep their chunks; the new ones are one chunk
    # of the store's 484 cells or of its 65,908 peaks.
    for name in ("ATAC_nCounts", "ATAC_nFeatures", "ids", "names"):
        assert after[name] == (PROFILE_METADATA_CHUNK,), name
    assert after["label"] == after["score"] == (484,)
    features = _chunks(root["ATAC/featureData"])
    for name in _OLD_FEATURE_COLUMNS:
        assert features[name] == (PROFILE_METADATA_CHUNK,), name
    assert features["region"] == features["weight"] == (_N_PEAKS,)
    reopened = DataStore(str(old_store), default_assay="ATAC", nthreads=1)
    np.testing.assert_array_equal(reopened.cells.fetch_all("label"), labels)
    np.testing.assert_array_equal(reopened.ATAC.feats.fetch_all("region"), regions)
    # Each column is costed by its own chunks: 100,000 IDs of 72 bytes, and
    # 484 labels of 24; 100,000 peaks of 100 bytes, and 65,908 regions of 28.
    old, new = (reopened.cells._get_array(name) for name in ("ids", "label"))
    assert array_geometry(old).nominalChunkBytes() == PROFILE_METADATA_CHUNK * 72
    assert array_geometry(new).nominalChunkBytes() == 484 * 24
    old, new = (reopened.ATAC.feats._get_array(name) for name in ("ids", "region"))
    assert array_geometry(old).nominalChunkBytes() == PROFILE_METADATA_CHUNK * 100
    assert array_geometry(new).nominalChunkBytes() == _N_PEAKS * 28


# The features of each sparse store.
_N_FEATURES = 64


def _sparse_store(path: Path, prefix: str) -> "DataStore":
    from cytearc.datastore.datastore import DataStore
    from cytearc.writers import SparseToZarr

    counts = np.random.default_rng(len(prefix)).integers(1, 9, (48, _N_FEATURES))
    SparseToZarr(
        csr_matrix(counts.astype(np.uint32)),
        str(path),
        [f"{prefix}{index}" for index in range(48)],
        [f"f{index}" for index in range(_N_FEATURES)],
        assay_name="RNA",
        nthreads=1,
    ).dump()
    store = DataStore(
        str(path), default_assay="RNA", min_features_per_cell=-1, nthreads=1
    )
    store.cells.insert(
        "wide", np.asarray(["w" * _LABEL_WIDTH] + ["v"] * 47, dtype=object)
    )
    # A wide column in chunks of 100,000 rows would allocate a chunk of 40 GB
    # as it is written, so a narrow column first checks that feature columns
    # take their rows: code without the rule fails here, not out of memory.
    store.RNA.feats.insert("narrow", np.asarray(["n"] * _N_FEATURES, dtype=object))
    assert store.zw["RNA/featureData/narrow"].chunks == (_N_FEATURES,)
    store.RNA.feats.insert(
        "wide",
        np.asarray(["w" * _LABEL_WIDTH] + ["v"] * (_N_FEATURES - 1), dtype=object),
    )
    return store


def _cell_chunks(path: Path, workspace: str = "") -> dict[str, tuple[int, ...]]:
    root = zarr.open_group(path, mode="r")
    return _chunks(root[f"{workspace}cellData"])


def _feature_chunks(path: Path, assay: str = "RNA") -> dict[str, tuple[int, ...]]:
    return _chunks(zarr.open_group(path, mode="r")[f"{assay}/featureData"])


def _ruled(chunks: dict[str, tuple[int, ...]], rows: int) -> dict[str, tuple[int, ...]]:
    """The chunks that the rule gives the same columns: 41 rows to the wide column and its mask, and ``rows`` to the others."""
    return {
        name: (_LABEL_ROWS,) if name.endswith("wide") else (rows,) for name in chunks
    }


def test_store_imports_and_copies_chunk_cell_and_feature_columns_by_rows_and_bytes(
    tmp_path: Path,
) -> None:
    from cytearc.datastore.datastore import mount_datastore
    from cytearc.merge import DataStoreMerge
    from cytearc.tools.repack_zarr import repack_store
    from cytearc.writers import SubsetZarr

    source = _sparse_store(tmp_path / "source.zarr", "a")
    written = _cell_chunks(tmp_path / "source.zarr")
    assert written == {
        "I": (48,),
        "RNA_nCounts": (48,),
        "RNA_nFeatures": (48,),
        "ids": (48,),
        "names": (48,),
        "wide": (_LABEL_ROWS,),
    }
    features = _feature_chunks(tmp_path / "source.zarr")
    assert set(features) == {
        "I",
        "dropOuts",
        "ids",
        "nCells",
        "names",
        "narrow",
        "wide",
    }
    assert features == _ruled(features, _N_FEATURES)

    SubsetZarr(
        str(tmp_path / "subset.zarr"), [source.RNA], cell_idx=np.arange(45), nthreads=1
    ).dump()
    subset = _cell_chunks(tmp_path / "subset.zarr")
    assert subset == {
        name: (_LABEL_ROWS,) if name == "wide" else (45,)
        for name in ("I", "ids", "names", "wide")
    }
    subset_features = _feature_chunks(tmp_path / "subset.zarr")
    assert {"I", "ids", "names", "wide"} <= set(subset_features)
    assert subset_features == _ruled(subset_features, _N_FEATURES)

    repack_store(str(tmp_path / "source.zarr"), str(tmp_path / "repacked.zarr"))
    assert _cell_chunks(tmp_path / "repacked.zarr") == written
    assert _feature_chunks(tmp_path / "repacked.zarr") == features

    mount_datastore(
        str(tmp_path / "source.zarr"),
        at=str(tmp_path / "mounted.zarr"),
        min_features_per_cell=-1,
        nthreads=1,
    )
    assert _cell_chunks(tmp_path / "mounted.zarr") == written
    assert _feature_chunks(tmp_path / "mounted.zarr") == features

    other = _sparse_store(tmp_path / "other.zarr", "b")
    merger = DataStoreMerge(
        [source, other], str(tmp_path / "merged.zarr"), ["a", "b"], nthreads=1
    )
    merger.plan()
    # The plan costs a band of the wide column at the chunk that the write
    # stores: at least its 41 values of 400,000 bytes and their encoding, and
    # less than a band of all 96 cells' values and their encoding would hold.
    assert merger._metadataPlan is not None
    cost = merger._metadataPlan.peak_write_bytes_at(1, chunk_rows=96)
    assert 2 * _LABEL_ROWS * 4 * _LABEL_WIDTH <= cost < 2 * 96 * 4 * _LABEL_WIDTH
    merger.dump()
    merged = _cell_chunks(tmp_path / "merged.zarr")
    wide = {name for name in merged if name.endswith("wide")}
    assert wide and {merged[name] for name in wide} == {(_LABEL_ROWS,)}
    assert {chunks for name, chunks in merged.items() if name not in wide} == {(96,)}
    # The merged features take the rule too, the wide column's in bands of the
    # chunk it stores.
    merged_features = _feature_chunks(tmp_path / "merged.zarr")
    assert {"I", "ids", "names", "wide"} <= set(merged_features)
    assert merged_features == _ruled(merged_features, _N_FEATURES)
    np.testing.assert_array_equal(
        zarr.open_group(tmp_path / "merged.zarr", mode="r")["RNA/featureData/wide"][:],
        source.RNA.feats.fetch_all("wide"),
    )
    # A resumed merge finds every column chunked as its write chunks it.
    resumed = DataStoreMerge(
        [source, other], str(tmp_path / "merged.zarr"), ["a", "b"], nthreads=1
    ).plan()
    assert resumed.cellDataAction == "skip"


def test_an_h5ad_import_chunks_its_cell_and_feature_columns_by_rows_and_bytes(
    tmp_path: Path,
) -> None:
    import anndata

    from cytearc.readers import H5adReader
    from cytearc.writers import H5adToZarr

    n_cells = n_features = 48
    # A wide column in chunks of 100,000 rows would allocate a chunk of 40 GB
    # as it is written, so a narrow import first checks that the h5ad import
    # gives feature columns their rows: code without the rule fails here, not
    # out of memory.
    anndata.AnnData(
        csr_matrix(np.ones((n_cells, n_features), dtype=np.float32)),
        obs=pd.DataFrame(index=[f"c{index}" for index in range(n_cells)]),
        var=pd.DataFrame(
            {"narrow": ["n"] * n_features},
            index=[f"f{index}" for index in range(n_features)],
        ),
    ).write_h5ad(tmp_path / "narrow.h5ad")
    narrow = H5adReader(str(tmp_path / "narrow.h5ad"))
    try:
        H5adToZarr(narrow, str(tmp_path / "narrow.zarr"), nthreads=1).dump()
    finally:
        narrow.close()
    assert _feature_chunks(tmp_path / "narrow.zarr")["narrow"] == (n_features,)
    # A nullable integer column holds a missing value, which a mask flags.
    score = pd.array(list(range(n_cells)), dtype="Int64")
    score[3] = pd.NA
    adata = anndata.AnnData(
        csr_matrix(np.ones((n_cells, n_features), dtype=np.float32)),
        obs=pd.DataFrame(
            {
                "wide": ["w" * _LABEL_WIDTH] + ["v"] * (n_cells - 1),
                "score": score,
                "RNA_I": np.arange(n_cells) % 2 == 0,
            },
            index=[f"c{index}" for index in range(n_cells)],
        ),
        var=pd.DataFrame(
            {
                "wide": ["w" * _LABEL_WIDTH] + ["v"] * (n_features - 1),
                "score": score,
            },
            index=[f"f{index}" for index in range(n_features)],
        ),
        uns={"cytearc": {"assayMembership": {"RNA": "RNA_I"}}},
    )
    adata.write_h5ad(tmp_path / "cells.h5ad")
    reader = H5adReader(str(tmp_path / "cells.h5ad"))
    try:
        H5adToZarr(reader, str(tmp_path / "imported.zarr"), nthreads=1).dump()
    finally:
        reader.close()
    assert _cell_chunks(tmp_path / "imported.zarr") == {
        "I": (48,),
        "RNA_I": (48,),
        "__cytearc_missing__score": (48,),
        "ids": (48,),
        "names": (48,),
        "score": (48,),
        "wide": (_LABEL_ROWS,),
    }
    assert _feature_chunks(tmp_path / "imported.zarr") == {
        "I": (48,),
        "__cytearc_missing__score": (48,),
        "ids": (48,),
        "names": (48,),
        "score": (48,),
        "wide": (_LABEL_ROWS,),
    }


def _seurat_import(path: Path) -> None:
    from cytearc.readers.seurat import SeuratReader
    from cytearc.writers.seurat import SeuratToZarr

    with SeuratReader(
        Path(full_path("seurat_assay5_synthetic.rds")), reductions=[]
    ) as reader:
        assert len(reader.cellIds) == 300
        SeuratToZarr(reader, str(path), mem_budget="1G", nthreads=1).dump()


def test_a_seurat_import_chunks_its_cell_and_feature_columns_by_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cytearc.writers.seurat as seurat_writer

    _seurat_import(tmp_path / "seurat.zarr")
    chunks = _cell_chunks(tmp_path / "seurat.zarr")
    assert {"ids", "names", "I"} <= set(chunks)
    assert set(chunks.values()) == {(300,)}
    # The feature IDs, names and metadata columns are one chunk of the
    # features' rows.
    features = _feature_chunks(tmp_path / "seurat.zarr")
    n_features = zarr.open_group(tmp_path / "seurat.zarr", mode="r")[
        "RNA/featureData/ids"
    ].shape[0]
    metadata = set(features) - {"ids", "names", "I"}
    assert metadata
    assert n_features < 65_536
    assert set(features.values()) == {(n_features,)}
    # Seurat's columns take the rule with its own cap of 65,536 rows.
    assert seurat_writer._COLUMN_CHUNK_ROWS(1_000_000, 4) == 65_536
    assert seurat_writer._COLUMN_CHUNK_ROWS(1_000_000, 4 * 200) == 20_971
    # The rule that the tables pass reaches every cell and feature metadata
    # column; the IDs and names take the rule of the empty columns.
    monkeypatch.setattr(seurat_writer, "_COLUMN_CHUNK_ROWS", lambda _rows, _size: 7)
    _seurat_import(tmp_path / "ruled.zarr")
    ruled = _cell_chunks(tmp_path / "ruled.zarr")
    assert {ruled[name] for name in set(ruled) - {"ids", "names", "I"}} == {(7,)}
    ruled_features = _feature_chunks(tmp_path / "ruled.zarr")
    assert {ruled_features[name] for name in metadata} == {(7,)}
    assert {ruled_features[name] for name in ("ids", "names", "I")} == {(n_features,)}


def test_csv_and_cell_ranger_imports_chunk_their_cell_and_feature_columns_by_rows(
    tmp_path: Path,
) -> None:
    from cytearc.readers import CSVReader
    from cytearc.writers import CSVtoZarr
    from tests.variant_truth import tapestri_truth, write_tapestri

    path = tmp_path / "counts.csv"
    path.write_text("cell,batch,g0,g1\nc0,a,1,0\nc1,b,0,2\nc2,a,3,1\n")
    reader = CSVReader(str(path), id_column=0, cell_data_cols=["batch"], batch_size=2)
    CSVtoZarr(reader, str(tmp_path / "csv.zarr"), assay_name="RNA", nthreads=1).dump()
    assert _cell_chunks(tmp_path / "csv.zarr") == dict.fromkeys(
        ("I", "batch", "ids", "names"), (3,)
    )
    assert _feature_chunks(tmp_path / "csv.zarr") == dict.fromkeys(
        ("I", "ids", "names"), (2,)
    )

    # A Tapestri file's cell columns reach the store through Cell Ranger's
    # writer, which Matrix Market imports share.
    from cytearc.readers import TapestriReader
    from cytearc.writers import TapestriToZarr

    truth = tapestri_truth()
    file = write_tapestri(truth, tmp_path / "run.dna+protein.h5")
    reader = TapestriReader(str(file))
    try:
        TapestriToZarr(reader, str(tmp_path / "tapestri.zarr"), nthreads=1).dump()
    finally:
        reader.close()
    tapestri = _cell_chunks(tmp_path / "tapestri.zarr")
    assert set(tapestri) == {"I", "filtered", "ids", "names", "sample_name"}
    assert set(tapestri.values()) == {(len(truth.row_ids),)}
    for assay in ("CNV", "ADT"):
        chunks = _feature_chunks(tmp_path / "tapestri.zarr", assay)
        n_features = zarr.open_group(tmp_path / "tapestri.zarr", mode="r")[
            f"{assay}/featureData/ids"
        ].shape[0]
        assert {"I", "ids", "names"} <= set(chunks)
        assert set(chunks.values()) == {(n_features,)}


def test_a_derived_assay_chunks_its_membership_and_features_as_new_columns(
    tmp_path: Path,
) -> None:
    from cytearc.storage.schema import create_cell_data, derived_assay_transaction
    from cytearc.writers._store import write_membership_column
    from tests.storage_helpers import finalize_test_counts

    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    ids = np.asarray([f"c{index}" for index in range(5)])
    create_cell_data(root, None, ids=ids, names=ids)
    write_membership_column(root["cellData"], "RNA", np.arange(5) % 2 == 0)
    # The source membership and I are stored in 100,000-row chunks, as in a
    # store written before.
    for name in ("RNA_I", "I"):
        rechunk_column(root["cellData"], name)
    assert root["cellData"]["RNA_I"].chunks == (PROFILE_METADATA_CHUNK,)
    with derived_assay_transaction(
        root, "SCORES", None, operation="add_grouped_assay", membership="RNA_I"
    ) as transaction:
        counts = transaction.create_counts(5, ["f0"], ["f0"], np.float64)
        counts[:] = 1.0
        finalize_test_counts(counts)
    derived = root["cellData"]["SCORES_I"]
    assert derived.chunks == (5,)
    np.testing.assert_array_equal(derived[:], np.arange(5) % 2 == 0)
    assert _feature_chunks(tmp_path / "store.zarr", "SCORES") == dict.fromkeys(
        ("I", "ids", "names"), (1,)
    )


@pytest.mark.parametrize(
    ("width", "n_names", "expected"),
    [(5, 48, 48), (40, 1_000_000, 100_000), (200, 1_000_000, 20_971)],
)
def test_phylogeny_node_names_take_the_chunks_of_cell_columns(
    width: int, n_names: int, expected: int
) -> None:
    from cytearc.datastore._operations.phylogeny import _name_rows

    assert _name_rows(width, n_names) == expected
