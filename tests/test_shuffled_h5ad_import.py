"""H5AD imports that shuffle cells write row ``r`` of every per-cell array from file row ``order[r]``.

Counts are checked against the file's own values; every other per-cell array
against an import of the same file in its own order, whose layout the other
H5AD tests pin. Stores compare file for file apart from the identifiers and
creation times of their artifacts, which every import makes anew.
"""

import hashlib
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest
import zarr
from loguru import logger
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from zarr.storage import MemoryStore

import cytearc.readers._row_spill as row_spill
import cytearc.writers.h5ad as h5ad_writer
from cytearc.readers import H5adReader
from cytearc.readers._sparse import SparseRowStore
from cytearc.storage.budget import ResourceBudget
from cytearc.storage.cell_order import (
    SOURCE_ROW,
    SOURCE_ROW_ROLE,
    recorded_shuffle_seed,
    shuffled_order,
)
from cytearc.storage.count_matrix import CountMatrixPolicy
from cytearc.storage.io_policy import StorageIoPolicy
from cytearc.storage.refs import artifact_path
from cytearc.utils import count_values
from cytearc.utils.arguments import quoted_value
from cytearc.utils.memory_budget import MemoryBudgetError
from cytearc.writers import H5adToZarr
from tests.row_store_helpers import (
    asked_bytes,
    file_bytes,
    finalized_bytes,
    free_disk,
)

anndata = pytest.importorskip("anndata")

_N_CELLS = 157
_N_FEATURES = 12
# Even features are genes and odd ones antibodies, which a split stores as
# RNA and ADT.
_FEATURE_TYPES = [
    "Gene Expression" if f % 2 == 0 else "Antibody Capture" for f in range(_N_FEATURES)
]


@contextmanager
def _warnings() -> Iterator[list[str]]:
    messages: list[str] = []
    handler = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        yield messages
    finally:
        logger.remove(handler)


def _counts(n_cells: int = _N_CELLS) -> np.ndarray:
    """Counts with empty cells, an empty feature, and a count past uint16."""
    counts = np.random.default_rng(1).poisson(0.9, (n_cells, _N_FEATURES))
    counts = counts.astype(np.float32)
    counts[[3, 17][: n_cells // 10]] = 0
    counts[:, 6] = 0
    counts[min(40, n_cells - 1), 3] = 70_000
    return counts


def _obs(n_cells: int, *, duplicates: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    ids = [f"cell-{row}" for row in range(n_cells)]
    if duplicates:
        ids[9] = ids[30] = ids[2]
    score = rng.normal(size=n_cells)
    score[[5 % n_cells, 50 % n_cells]] = np.nan
    stage = pd.Categorical(
        rng.choice(["early", "mid", "late"], n_cells),
        categories=["early", "mid", "late", "unused"],
        ordered=True,
    )
    stage[[1 % n_cells, 8 % n_cells]] = np.nan
    clusters = pd.Categorical(rng.choice(["a", "b", "c"], n_cells))
    clusters[[2 % n_cells, 60 % n_cells]] = np.nan
    nullable = pd.array(rng.integers(0, 9, n_cells), dtype="Int64")
    nullable[[0, 4 % n_cells]] = pd.NA
    return pd.DataFrame(
        {
            "score": score,
            "count": rng.integers(0, 1000, n_cells),
            "flag": rng.random(n_cells) < 0.5,
            "label": [f"L{value}" for value in rng.integers(0, 50, n_cells)],
            "stage": stage,
            "nullable": nullable,
            "clusters": clusters,
            "RNA_I": rng.random(n_cells) < 0.9,
            # A source column that a shuffled import's own column displaces.
            SOURCE_ROW: np.arange(n_cells)[::-1].copy(),
        },
        index=pd.Index(ids),
    )


def _write_source(
    path: Path, encoding: str, *, n_cells: int = _N_CELLS, duplicates: bool = False
) -> np.ndarray:
    """Write an AnnData file of every kind of per-cell array; return its counts."""
    counts = _counts(n_cells)
    matrix = {"csr": csr_matrix, "csc": csc_matrix, "dense": np.asarray}[encoding](
        counts
    )
    rng = np.random.default_rng(3)
    pca = rng.normal(size=(n_cells, 5))
    pca[7 % n_cells, 2] = -0.0
    anndata.AnnData(
        matrix,
        obs=_obs(n_cells, duplicates=duplicates),
        var=pd.DataFrame(
            {
                "feature_types": _FEATURE_TYPES,
                "feature_name": [f"gene-{f}" for f in range(_N_FEATURES)],
            },
            index=[f"feature-{f}" for f in range(_N_FEATURES)],
        ),
        obsm={"X_umap": rng.normal(size=(n_cells, 2)).astype(np.float32), "X_pca": pca},
        uns={"cytearc": {"assayMembership": {"RNA": "RNA_I"}}},
    ).write_h5ad(path)
    return counts


def _reader(path: Path, **options: Any) -> H5adReader:
    return H5adReader(
        str(path),
        feature_name_key="feature_name",
        embedding_roles={"X_umap": "umap", "X_pca": "coordinates"},
        cluster_keys=("clusters",),
        **options,
    )


def _import(
    source: Path,
    destination: Any,
    *,
    reader_options: dict[str, Any] | None = None,
    **options: Any,
) -> tuple[Any, Any]:
    """Import ``source`` into ``destination``; return the import's result and writer."""
    reader = _reader(source, **(reader_options or {}))
    try:
        settings = {"mem_budget": "256M", "nthreads": 1, **options}
        if "assay_split_key" in settings:
            settings.setdefault("analysis_assay", "RNA")
        writer = H5adToZarr(
            reader,
            str(destination) if isinstance(destination, Path) else destination,
            **settings,
        )
        result = writer.dump()
    finally:
        reader.close()
    return result, writer


def _open(destination: Any) -> zarr.Group:
    if isinstance(destination, MemoryStore):
        return zarr.open_group(store=destination, mode="r")
    return zarr.open_group(str(destination), mode="r")


def _cell_arrays(root: zarr.Group) -> dict[str, np.ndarray]:
    cells = root["cellData"]
    return {name: np.asarray(array[...]) for name, array in cells.arrays()}


def _artifact_arrays(root: zarr.Group, ref: Any) -> dict[str, np.ndarray]:
    group = root[artifact_path(ref)]
    return {name: np.asarray(array[...]) for name, array in group.arrays()}


def _assert_rows_in_order(
    shuffled: np.ndarray, plain: np.ndarray, order: np.ndarray, name: str
) -> None:
    expected = plain[order]
    assert shuffled.dtype == expected.dtype, name
    # Byte views compare NaN, negative zero and text exactly.
    np.testing.assert_array_equal(
        np.ascontiguousarray(shuffled).view(np.uint8),
        np.ascontiguousarray(expected).view(np.uint8),
        err_msg=name,
    )


def _assert_shuffled_import(
    plain_root: zarr.Group,
    shuffled_root: zarr.Group,
    plain_result: Any,
    shuffled_result: Any,
    order: np.ndarray,
    counts: np.ndarray,
    seed: int,
) -> None:
    """Check every per-cell array of the shuffled store against the plain store's rows."""
    n_cells = order.size
    for assay in plain_result.assayNames:
        stored = np.asarray(shuffled_root[f"{assay}/counts"][...])
        columns = (
            np.arange(_N_FEATURES)
            if len(plain_result.assayNames) == 1
            else np.flatnonzero(
                np.array(_FEATURE_TYPES)
                == ("Gene Expression" if assay == "RNA" else "Antibody Capture")
            )
        )
        # The file's own values, row by row in shuffled order.
        np.testing.assert_array_equal(stored, counts[order][:, columns], err_msg=assay)
        if f"{assay}/countsT" in plain_root:
            np.testing.assert_array_equal(
                np.asarray(shuffled_root[f"{assay}/countsT"][...]), stored.T
            )
        else:
            assert f"{assay}/countsT" not in shuffled_root
        assert (
            shuffled_root[f"{assay}/counts"].dtype
            == plain_root[f"{assay}/counts"].dtype
        )
        for name, array in plain_root[f"{assay}/countSummaries"].arrays():
            values = np.asarray(array[...])
            found = np.asarray(shuffled_root[f"{assay}/countSummaries/{name}"][...])
            if values.shape[:1] == (n_cells,):
                _assert_rows_in_order(found, values, order, name)
            else:
                np.testing.assert_array_equal(found, values, err_msg=name)
    plain_cells = _cell_arrays(plain_root)
    shuffled_cells = _cell_arrays(shuffled_root)
    # The plain import keeps the file's sourceRow column, which the shuffled
    # import replaces with its own.
    assert set(shuffled_cells) == set(plain_cells)
    for name, values in plain_cells.items():
        if name == SOURCE_ROW:
            continue
        if values.shape == (n_cells,):
            _assert_rows_in_order(shuffled_cells[name], values, order, name)
        else:
            np.testing.assert_array_equal(shuffled_cells[name], values, err_msg=name)
        assert dict(shuffled_root[f"cellData/{name}"].attrs) == dict(
            plain_root[f"cellData/{name}"].attrs
        ), name
    np.testing.assert_array_equal(shuffled_cells[SOURCE_ROW], order)
    assert shuffled_cells[SOURCE_ROW].dtype == np.int64
    assert dict(shuffled_root[f"cellData/{SOURCE_ROW}"].attrs) == {
        "role": SOURCE_ROW_ROLE
    }
    # The file's column of that name imports only in the file's order.
    assert "role" not in plain_root[f"cellData/{SOURCE_ROW}"].attrs
    assert recorded_shuffle_seed(shuffled_root["cellData"]) == seed
    assert recorded_shuffle_seed(plain_root["cellData"]) is None
    # Embeddings and cluster labels follow the same order.
    for kind in ("embeddingArtifacts", "clusterArtifacts"):
        plain_refs = getattr(plain_result, kind)
        shuffled_refs = getattr(shuffled_result, kind)
        assert set(shuffled_refs) == set(plain_refs)
        for key, ref in plain_refs.items():
            plain_arrays = _artifact_arrays(plain_root, ref)
            found = _artifact_arrays(shuffled_root, shuffled_refs[key])
            assert set(found) == set(plain_arrays)
            for name, values in plain_arrays.items():
                if values.shape[:1] == (n_cells,):
                    _assert_rows_in_order(found[name], values, order, f"{key}/{name}")
                else:
                    np.testing.assert_array_equal(found[name], values)


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("encoding", ["csr", "csc", "dense"])
def test_every_per_cell_array_follows_the_shuffled_order(
    tmp_path, encoding, split
) -> None:
    source = tmp_path / "source.h5ad"
    counts = _write_source(source, encoding)
    options = {"assay_split_key": "feature_types"} if split else {}
    with _warnings() as plain_warnings:
        plain_result, _ = _import(
            source, tmp_path / "plain.zarr", shuffle_cells=False, **options
        )
    with _warnings() as shuffled_warnings:
        shuffled_result, writer = _import(
            source,
            tmp_path / "shuffled.zarr",
            shuffle_cells=True,
            shuffle_seed=11,
            reader_options={"temp_dir": str(tmp_path)},
            **options,
        )
    assert writer.shuffleCells is True and writer.shuffleSeed == 11
    _assert_shuffled_import(
        _open(tmp_path / "plain.zarr"),
        _open(tmp_path / "shuffled.zarr"),
        plain_result,
        shuffled_result,
        shuffled_order(_N_CELLS, 11),
        counts,
        11,
    )
    skipped = "Skipped source cell metadata column 'sourceRow' because CyteArc stores"
    assert any(message.startswith(skipped) for message in shuffled_warnings)
    assert not any(SOURCE_ROW in message for message in plain_warnings)
    # The spill is gone once the import ends.
    assert not [path for path in tmp_path.iterdir() if path.name.startswith("cytearc-")]


@pytest.mark.filterwarnings("ignore:Observation names are not unique")
def test_categorical_columns_and_renamed_duplicate_ids_follow_the_order(
    tmp_path,
) -> None:
    source = tmp_path / "source.h5ad"
    counts = _write_source(source, "csr", duplicates=True)
    renamed = {"reader_options": {"duplicate_cell_ids": "make_unique"}}
    plain_result, _ = _import(
        source,
        tmp_path / "plain.zarr",
        categorical_columns=True,
        shuffle_cells=False,
        **renamed,
    )
    shuffled_result, _ = _import(
        source,
        tmp_path / "shuffled.zarr",
        categorical_columns=True,
        shuffle_cells=True,
        **renamed,
    )
    plain = _open(tmp_path / "plain.zarr")
    # Repeated IDs are renamed in the file's order before they are shuffled.
    ids = np.asarray(plain["cellData/ids"][...])
    assert ids[[2, 9, 30]].tolist() == ["cell-2", "cell-2-1", "cell-2-2"]
    assert "__cytearc_categories__stage" in plain["cellData"]
    _assert_shuffled_import(
        plain,
        _open(tmp_path / "shuffled.zarr"),
        plain_result,
        shuffled_result,
        shuffled_order(_N_CELLS, 0),
        counts,
        0,
    )


def _write_unsorted(path: Path, encoding: str) -> np.ndarray:
    """Write float64 counts with repeated coordinates out of order; return their sums.

    Float sums depend on the order of their terms, so the shuffled store
    must sum each repeated coordinate in the file's order.
    """
    rng = np.random.default_rng(5)
    n_cells, n_features = 60, 9
    base = coo_matrix(rng.poisson(0.6, (n_cells, n_features)) * 0.1)
    repeats = rng.random(base.nnz) < 0.4
    rows = np.concatenate([base.row, base.row[repeats], base.row[repeats]])
    columns = np.concatenate([base.col, base.col[repeats], base.col[repeats]])
    values = np.concatenate(
        [base.data, base.data[repeats] * 1e-9, base.data[repeats] * 3.3]
    )
    major, minor = (rows, columns) if encoding == "csr" else (columns, rows)
    # Each vector holds its entries in a random order.
    order = np.lexsort((rng.random(major.size), major))
    pointers = np.concatenate(
        (
            [0],
            np.cumsum(
                np.bincount(
                    major, minlength=n_cells if encoding == "csr" else n_features
                )
            ),
        )
    )
    with h5py.File(path, "w") as h5:
        group = h5.create_group("X")
        group.attrs["encoding-type"] = f"{encoding}_matrix"
        group.attrs["shape"] = (n_cells, n_features)
        group.create_dataset("data", data=values[order])
        group.create_dataset("indices", data=minor[order].astype(np.int32))
        group.create_dataset("indptr", data=pointers.astype(np.int64))
        h5.create_group("obs").create_dataset(
            "_index", data=np.array([f"c{row}".encode() for row in range(n_cells)])
        )
        h5.create_group("var").create_dataset(
            "_index", data=np.array([f"g{f}".encode() for f in range(n_features)])
        )
    summed = coo_matrix((values, (rows, columns)), shape=(n_cells, n_features))
    return summed.toarray()


@pytest.mark.parametrize("encoding", ["csr", "csc"])
def test_repeated_coordinates_sum_as_the_file_order_sums_them(
    tmp_path, encoding
) -> None:
    source = tmp_path / "unsorted.h5ad"
    summed = _write_unsorted(source, encoding)
    stores = {}
    for shuffle in (False, True):
        reader = H5adReader(str(source))
        try:
            H5adToZarr(
                reader,
                str(tmp_path / f"{shuffle}.zarr"),
                nthreads=1,
                values="normalized",
                shuffle_cells=shuffle,
                shuffle_seed=4,
            ).dump()
        finally:
            reader.close()
        stores[shuffle] = np.asarray(
            _open(tmp_path / f"{shuffle}.zarr")["RNA/counts"][...]
        )
    order = shuffled_order(summed.shape[0], 4)
    np.testing.assert_allclose(stores[False], summed, rtol=1e-12)
    # Bit for bit the plain import's sums, in shuffled order.
    _assert_rows_in_order(stores[True], stores[False], order, "counts")


def _store_files(path: Path, results: list[Any]) -> dict[str, str]:
    """Digest every file of a store, its artifacts' identifiers and creation times masked."""
    names: dict[str, str] = {}
    for result in results:
        names[result.cellSelection.artifact_id] = "cell_selection"
        for kind in ("embeddingArtifacts", "clusterArtifacts"):
            for key, ref in getattr(result, kind).items():
                names[ref.artifact_id] = f"{kind}:{key}"
    created = re.compile(rb'"created_at_ns": *[0-9]+')
    digests = {}
    for directory, _folders, files in os.walk(path):
        for name in files:
            full = os.path.join(directory, name)
            relative = os.path.relpath(full, path)
            with open(full, "rb") as stream:
                content = created.sub(b'"created_at_ns": 0', stream.read())
            for artifact_id, label in names.items():
                relative = relative.replace(artifact_id, label)
                content = content.replace(artifact_id.encode(), label.encode())
            digests[relative] = hashlib.sha256(content).hexdigest()
    return digests


def test_one_seed_gives_the_same_store_and_another_seed_another_order(tmp_path) -> None:
    source = tmp_path / "source.h5ad"
    _write_source(source, "csr")
    results = {}
    for name, seed in (("first", 6), ("second", 6), ("other", 7)):
        results[name], _ = _import(
            source, tmp_path / f"{name}.zarr", shuffle_cells=True, shuffle_seed=seed
        )
    first = _store_files(tmp_path / "first.zarr", [results["first"]])
    assert first == _store_files(tmp_path / "second.zarr", [results["second"]])
    assert "cellData/sourceRow/zarr.json" in first
    other = _store_files(tmp_path / "other.zarr", [results["other"]])
    assert set(other) == set(first)
    assert other["cellData/sourceRow/c/0"] != first["cellData/sourceRow/c/0"]


def test_an_import_shuffles_with_seed_0_unless_told_not_to(tmp_path) -> None:
    source = tmp_path / "source.h5ad"
    _write_source(source, "csr")
    results = {}
    for name, options in (
        ("default", {}),
        ("seed0", {"shuffle_cells": True, "shuffle_seed": 0}),
        ("plain", {"shuffle_cells": False}),
        ("explicit", {"shuffle_cells": False, "shuffle_seed": 9}),
    ):
        results[name], writer = _import(source, tmp_path / f"{name}.zarr", **options)
        assert writer.shuffleCells is (name in ("default", "seed0"))
        assert writer.shuffleSeed == options.get("shuffle_seed", 0)
    stores = {
        name: _store_files(tmp_path / f"{name}.zarr", [result])
        for name, result in results.items()
    }
    # The default is the shuffle with seed 0, and its source column named
    # sourceRow gives way to the cells' rows in the file.
    assert stores["default"] == stores["seed0"]
    cells = _open(tmp_path / "default.zarr")["cellData"]
    assert recorded_shuffle_seed(cells) == 0
    assert cells[SOURCE_ROW].attrs["role"] == SOURCE_ROW_ROLE
    np.testing.assert_array_equal(cells[SOURCE_ROW][...], shuffled_order(_N_CELLS, 0))
    # shuffle_cells=False writes the file's order, whatever the seed, and
    # keeps the source column.
    assert stores["plain"] == stores["explicit"] != stores["default"]
    cells = _open(tmp_path / "plain.zarr")["cellData"]
    assert "cellOrder" not in cells.attrs
    assert "role" not in cells[SOURCE_ROW].attrs
    np.testing.assert_array_equal(cells[SOURCE_ROW][...], np.arange(_N_CELLS)[::-1])


def _spy_on_scan_processes(monkeypatch) -> list[Any]:
    """Scan every matrix in processes when asked, and record each window's spill."""
    monkeypatch.setattr(count_values, "_SCAN_PROCESS_VALUES", 0)
    seen: list[Any] = []
    original = H5adReader._scan_in_processes

    def spy(self, windows, maxBytes, featureGroups, **kwargs):
        seen.append((list(windows), kwargs.get("spills")))
        return original(self, windows, maxBytes, featureGroups, **kwargs)

    monkeypatch.setattr(H5adReader, "_scan_in_processes", spy)
    return seen


def test_a_failed_scan_in_processes_leaves_no_rows_behind(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source.h5ad"
    _write_source(source, "csr")
    monkeypatch.setattr(count_values, "_SCAN_PROCESS_VALUES", 0)
    windows: list[Any] = []

    def fail(self, scanned, maxBytes, featureGroups, **kwargs):
        windows.append(scanned)
        raise RuntimeError("a scan process failed")

    monkeypatch.setattr(H5adReader, "_scan_in_processes", fail)
    with pytest.raises(RuntimeError, match="a scan process failed"):
        _import(
            source,
            tmp_path / "refused.zarr",
            mem_budget="1G",
            nthreads=3,
            shuffle_cells=True,
            reader_options={"temp_dir": str(tmp_path)},
        )
    assert len(windows[0]) == 3
    assert not [path for path in tmp_path.iterdir() if path.name.startswith("cytearc-")]
    assert not (tmp_path / "refused.zarr").exists()


def test_the_disk_check_of_a_spill_in_processes_counts_the_part_of_each(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source.h5ad"
    _write_source(source, "csr")
    seen = _spy_on_scan_processes(monkeypatch)
    options = {
        "mem_budget": "1G",
        "nthreads": 3,
        "shuffle_cells": True,
        "reader_options": {"temp_dir": str(tmp_path)},
    }
    with finalized_bytes() as peaks:
        _import(source, MemoryStore(), **options)
    (peak,) = peaks
    ((windows, _spills),) = seen
    assert len(windows) == 3
    with free_disk(0), pytest.raises(OSError, match="which has 0 free") as full:
        _import(source, MemoryStore(), **options)
    required = asked_bytes(full.value)
    # The check counts the record of each scan process's part beside that of
    # the store's own part, so it covers every file that the spill wrote.
    assert peak <= required
    # One byte short, the import is refused before a process starts.
    with free_disk(required - 1), pytest.raises(OSError, match=f"needs {required} "):
        _import(source, MemoryStore(), **options)
    assert len(seen) == 1
    assert not [path for path in tmp_path.iterdir() if path.name.startswith("cytearc-")]


@pytest.mark.parametrize("encoding", ["csr", "dense"])
def test_scan_processes_each_spill_their_own_rows(
    tmp_path, monkeypatch, encoding
) -> None:
    source = tmp_path / "source.h5ad"
    counts = _write_source(source, encoding)
    plain_result, _ = _import(source, tmp_path / "plain.zarr", shuffle_cells=False)
    seen = _spy_on_scan_processes(monkeypatch)
    # Each scan process needs room for a window and a buffer of pending rows.
    shuffled_result, _ = _import(
        source,
        tmp_path / "shuffled.zarr",
        mem_budget="1G",
        nthreads=3,
        shuffle_cells=True,
    )
    ((windows, spills),) = seen
    assert len(windows) == 3
    # Each process writes its own part with the destinations of its rows.
    order = shuffled_order(_N_CELLS, 0)
    destinations = np.argsort(order)
    assert [spill.part for spill in spills] == [1, 2, 3]
    for (start, stop), spill in zip(windows, spills, strict=True):
        np.testing.assert_array_equal(spill.destinations, destinations[start:stop])
        assert spill.bufferBytes > 0
    _assert_shuffled_import(
        _open(tmp_path / "plain.zarr"),
        _open(tmp_path / "shuffled.zarr"),
        plain_result,
        shuffled_result,
        order,
        counts,
        0,
    )


_SHARD_BANDS = {
    "mem_budget": 1024**2,
    "nthreads": 4,
    "policy": CountMatrixPolicy(unitBytes=12, chunkBytes=12),
    "io": StorageIoPolicy(readWorkers=2),
}


@pytest.mark.parametrize("direct", [True, False])
def test_producer_processes_read_and_delete_their_buckets(
    tmp_path, monkeypatch, direct
) -> None:
    """Several producers read their windows of shuffled rows, writing or sending bands."""
    counts = np.random.default_rng(8).integers(0, 5, (40, 3)).astype(np.uint32)
    counts[4:8] = 0
    source = tmp_path / "bands.h5ad"
    anndata.AnnData(
        csr_matrix(counts),
        obs=pd.DataFrame(index=[f"c{row}" for row in range(40)]),
        var=pd.DataFrame(index=["f0", "f1", "f2"]),
    ).write_h5ad(source)
    windows_seen: list[list[tuple[int, int]]] = []
    remaining: list[set[int]] = []
    original_messages = h5ad_writer._producer_messages
    original_release = H5adReader._release_shuffled_rows

    def messages(target, windows, *args, **kwargs):
        windows_seen.append(list(windows))
        return original_messages(target, windows, *args, **kwargs)

    def release(self):
        # The buckets that the producers left, before the release removes them.
        store = self._convertedCsr
        starts = store.bucket_starts.tolist()
        remaining.append(
            {
                bucket
                for bucket, first in enumerate(starts[:-1])
                if (store.directory / f"rows-{first:012d}").exists()
            }
        )
        bucket_starts.append(starts)
        directory.append(store.directory)
        return original_release(self)

    bucket_starts: list[list[int]] = []
    directory: list[Path] = []
    # Buckets of about three rows, so producer windows split some of them.
    monkeypatch.setattr(row_spill, "_SPILL_READ_BYTES", 600)
    monkeypatch.setattr(h5ad_writer, "_producer_messages", messages)
    monkeypatch.setattr(H5adReader, "_release_shuffled_rows", release)
    destination = str(tmp_path / "shuffled.zarr") if direct else MemoryStore()
    reader = H5adReader(str(source), temp_dir=str(tmp_path))
    try:
        writer = H5adToZarr(reader, destination, shuffle_cells=True, **_SHARD_BANDS)
        assert reader._convertedCsr is not None
        writer.resources = ResourceBudget(1_000_000, 4)
        writer._write_counts(batch_size=2)
    finally:
        reader.close()
    assert writer._lastImportProducerCount >= 2
    ((windows,),) = [windows_seen]
    assert len(windows) == writer._lastImportProducerCount
    ((starts,),) = [bucket_starts]
    # A producer deletes each bucket wholly inside its window; the buckets
    # that straddle two windows remain until the release removes them.
    straddling = {
        bucket
        for bucket, (first, last) in enumerate(
            zip(starts[:-1], starts[1:], strict=True)
        )
        if not any(start <= first and last <= stop for start, stop in windows)
    }
    assert straddling, "the test needs a bucket across two windows"
    assert remaining == [straddling]
    assert not directory[0].exists()
    order = shuffled_order(40, 0)
    np.testing.assert_array_equal(_open(destination)["RNA/counts"][...], counts[order])


@pytest.mark.parametrize("direct", [True, False], ids=["writers", "readers"])
@pytest.mark.parametrize("encoding", ["csr", "dense"])
def test_import_processes_hold_their_row_pointers_and_leave_the_cell_order_to_the_parent(
    tmp_path, monkeypatch, encoding, direct
) -> None:
    import cytearc.storage.sharding as sharding
    from cytearc.storage.schema import load_count_array

    counts = np.random.default_rng(5).integers(0, 5, (12, 3)).astype(np.uint32)
    source = tmp_path / "source.h5ad"
    anndata.AnnData(
        csr_matrix(counts) if encoding == "csr" else counts,
        obs=pd.DataFrame(index=[f"c{row}" for row in range(12)]),
        var=pd.DataFrame(index=["f0", "f1", "f2"]),
    ).write_h5ad(source)
    # Processes write their own windows of a destination path, and send their
    # bands to the parent for a store in memory.
    destination = str(tmp_path / "shuffled.zarr") if direct else MemoryStore()

    class Planned(Exception):
        pass

    def planned(*_args, **_kwargs):
        raise Planned

    reader = H5adReader(str(source), temp_dir=str(tmp_path))
    try:
        writer = H5adToZarr(reader, destination, shuffle_cells=True, **_SHARD_BANDS)
        destinations = {
            name: load_count_array(writer.z, name, None) for name in writer.assayNames
        }

        def producers(budget: int) -> int:
            writer._lastImportProducerCount = 0
            writer.resources = ResourceBudget(budget, 4)
            try:
                # The admission alone, which leaves the shuffled rows for the
                # next probe and for the write.
                writer._write_sparse_counts(2, destinations, {})
            except (Planned, MemoryError):
                pass
            return writer._lastImportProducerCount

        def smallest_process_budget() -> int:
            """Return the smallest budget at which two processes read the shuffled rows."""
            with monkeypatch.context() as patch:
                # Writer processes start through the first; the parent writes
                # through the second, beside reader processes or alone.
                patch.setattr(H5adToZarr, "_write_parallel_count_windows", planned)
                patch.setattr(sharding, "write_sparse_bands", planned)
                low, high = 1_000, 1_000_000
                assert producers(low) < 2
                assert producers(high) == 2
                while high - low > 1:
                    middle = (low + high) // 2
                    if producers(middle) == 2:
                        high = middle
                    else:
                        low = middle
            return high

        budget = smallest_process_budget()
        # Two processes read the shuffled rows (0, 8) and (8, 12). The parent
        # alone holds the order of the cells, which places the other per-cell
        # arrays, so an order that takes 1,000 bytes more raises the budget at
        # which the two start by 1,000 bytes.
        order = writer._cellOrder
        with monkeypatch.context() as patch:
            patch.setattr(
                writer, "_cellOrder", np.append(order, np.zeros(125, order.dtype))
            )
            assert smallest_process_budget() == budget + 1_000
        # Each process reads the shuffled rows through its own reader, which
        # holds their row pointers as the parent's does, so pointers that take
        # 1,000 bytes more raise that budget by 3,000 bytes.
        pointers = H5adReader.materialized_csr_bytes
        with monkeypatch.context() as patch:
            patch.setattr(
                H5adReader,
                "materialized_csr_bytes",
                lambda self: pointers(self) + 1_000,
            )
            assert smallest_process_budget() == budget + 3_000
        if encoding == "dense":
            # The parent alone holds the count of the values in each of the
            # file's rows, which its scan recorded.
            counted = H5adReader._sparse_import_resident_bytes
            with monkeypatch.context() as patch:
                patch.setattr(
                    H5adReader,
                    "_sparse_import_resident_bytes",
                    lambda self: counted(self) + 1_000,
                )
                assert smallest_process_budget() == budget + 1_000
        # At that budget the two processes read every count.
        writer.resources = ResourceBudget(budget, 4)
        writer._write_counts(batch_size=2)
    finally:
        reader.close()
    assert writer._lastImportProducerCount == 2
    np.testing.assert_array_equal(
        _open(destination)["RNA/counts"][...], counts[shuffled_order(12, 0)]
    )


def test_a_single_producer_deletes_every_bucket_as_it_reads(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source.h5ad"
    counts = _write_source(source, "csr")
    deleted: list[int] = []
    original = SparseRowStore._delete

    def delete(self, bucket, start, stop):
        deleted.append(bucket)
        return original(self, bucket, start, stop)

    monkeypatch.setattr(SparseRowStore, "_delete", delete)
    # Buckets of about ten rows.
    monkeypatch.setattr(row_spill, "_SPILL_READ_BYTES", 5_000)
    reader = _reader(source, temp_dir=str(tmp_path))
    try:
        writer = H5adToZarr(
            reader,
            str(tmp_path / "shuffled.zarr"),
            mem_budget="64M",
            nthreads=1,
            shuffle_cells=True,
        )
        store = reader._convertedCsr
        assert store is not None and store.bucket_starts.size > 3
        n_buckets = store.bucket_starts.size - 1
        writer.dump()
    finally:
        reader.close()
    assert sorted(deleted) == list(range(n_buckets))
    np.testing.assert_array_equal(
        _open(tmp_path / "shuffled.zarr")["RNA/counts"][...],
        counts[shuffled_order(_N_CELLS, 0)],
    )


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("encoding", ["csr", "csc", "dense"])
def test_the_values_are_decided_from_the_files_first_cells(
    tmp_path, encoding, split
) -> None:
    source = tmp_path / "source.h5ad"
    counts = _write_source(source, encoding)
    options = {"assay_split_key": "feature_types"} if split else {}
    seen = []
    for shuffle in (False, True):
        reader = H5adReader(str(source), feature_name_key="feature_name")
        try:
            writer = H5adToZarr(
                reader,
                MemoryStore(),
                nthreads=1,
                shuffle_cells=shuffle,
                **options,
            )
            seen.append(
                {name: cells.toarray() for name, cells in writer._first_cells().items()}
            )
        finally:
            reader.close()
    plain, shuffled = seen
    columns = {
        "RNA": np.array(_FEATURE_TYPES) == "Gene Expression",
        "ADT": np.array(_FEATURE_TYPES) == "Antibody Capture",
    }
    for name, cells in plain.items():
        expected = counts[:100] if not split else counts[:100][:, columns[name]]
        np.testing.assert_array_equal(cells, expected)
        np.testing.assert_array_equal(shuffled[name], cells)


def test_counts_are_written_once_from_the_rows_of_their_own_import(tmp_path) -> None:
    source = tmp_path / "source.h5ad"
    _write_source(source, "csr")
    reader = H5adReader(str(source))
    try:
        writer = H5adToZarr(reader, MemoryStore(), nthreads=1, shuffle_cells=True)
        writer._write_counts()
        with pytest.raises(RuntimeError, match="written once and released"):
            writer._write_counts()
        # After the release the reader yields the file's rows again.
        first = next(reader.consume_row_range(5, 0, 5)).toarray()
        np.testing.assert_array_equal(first, _counts()[:5])
        # A second shuffled import of the reader replaces the first's rows.
        replaced = H5adToZarr(reader, MemoryStore(), nthreads=1, shuffle_cells=True)
        H5adToZarr(
            reader, MemoryStore(), nthreads=1, shuffle_cells=True, shuffle_seed=1
        )
        with pytest.raises(RuntimeError, match="another import of the reader replaced"):
            replaced._write_counts()
    finally:
        reader.close()


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"shuffle_cells": 1}, TypeError, "shuffle_cells must be a boolean"),
        ({"shuffle_seed": -1}, ValueError, "shuffle_seed must be at least 0"),
        ({"shuffle_seed": 2.0}, TypeError, "shuffle_seed must be an integer"),
    ],
)
def test_bad_shuffle_arguments_are_refused_before_the_store_exists(
    tmp_path, options, error, message
) -> None:
    source = tmp_path / "source.h5ad"
    _write_source(source, "csr", n_cells=8)
    reader = H5adReader(str(source))
    try:
        with pytest.raises(error, match=message):
            H5adToZarr(reader, str(tmp_path / "store.zarr"), nthreads=1, **options)
    finally:
        reader.close()
    assert not (tmp_path / "store.zarr").exists()


def test_a_budget_without_room_for_the_spill_is_refused_with_the_bytes_it_needs(
    tmp_path,
) -> None:
    source = tmp_path / "source.h5ad"
    _write_source(source, "csr")
    nonzeros = np.count_nonzero(_counts(), axis=1)
    rows, longest = _N_CELLS, int(nonzeros.max())
    # The reader keeps the cell IDs, the feature IDs, and the feature IDs
    # again as the names, which the reader finds no column of; the import
    # holds the declared RNA membership, one boolean a cell, and the order and
    # its inverse beside the scan too.
    features = [f"feature-{f}" for f in range(_N_FEATURES)]
    held = (
        _id_bytes([f"cell-{row}" for row in range(rows)])
        + _id_bytes(features)
        + _text_bytes(features)
        + rows
        + 2 * 8 * rows
    )
    # What the scan may hold beside them.
    share = 5_488
    budget = held + share
    # Buckets read in a thirty-second of what the scan may hold, at 44 bytes
    # for each float32 entry read and 80 for each row.
    row_bytes = nonzeros.mean() * 44 + 80
    buckets = -(-rows // max(1, int((share // 32) // row_bytes)))
    # Row counts of the store; the writer's counts and marks, a few numbers
    # per bucket, and the destinations of its rows; a scan window of the
    # longest row and what appending it holds.
    least = (
        (rows + 1) * 32 + rows * 9 + (buckets + 1) * 25 + 8 * rows + longest * (72 + 96)
    )
    reader = H5adReader(str(source))
    try:
        with pytest.raises(MemoryBudgetError) as caught:
            H5adToZarr(
                reader,
                str(tmp_path / "refused.zarr"),
                mem_budget=budget,
                nthreads=1,
                shuffle_cells=True,
            )
        # The refusal states what the import needs in all, as a budget that
        # lets it pass the spill.
        needed = held + least
        item = (
            f"the row spill of a cell shuffle beside the {held} bytes of its cell "
            "order, IDs, names and other arrays"
        )
        assert caught.value.needed_bytes == needed
        assert caught.value.budget_bytes == budget
        assert caught.value.item == item
        assert str(caught.value) == (
            f"Importing {quoted_value(str(source))} holds about {needed} bytes for "
            f"{item}, more than the memory budget of {budget} bytes. Raise "
            f"mem_budget to at least {needed} bytes."
        )
        assert reader._convertedCsr is None
        # With that much, the spill fits; a later step of so small a budget
        # refuses for itself.
        with pytest.raises(MemoryError) as caught:
            H5adToZarr(
                reader,
                str(tmp_path / "admitted.zarr"),
                mem_budget=held + least,
                nthreads=1,
                shuffle_cells=True,
            )
        assert "row spill" not in str(caught.value)
    finally:
        reader.close()
    assert not (tmp_path / "refused.zarr").exists()


def test_a_disk_without_room_for_the_spill_is_refused_before_it_is_written(
    tmp_path, monkeypatch
) -> None:
    import cytearc.readers._sparse as sparse

    source = tmp_path / "source.h5ad"
    _write_source(source, "csr")
    actual: list[tuple[int, int]] = []
    original_finalize = SparseRowStore.finalize

    def finalize(self):
        original_finalize(self)
        actual.append((int(self._fileBytes.sum()), file_bytes(self.directory)))

    monkeypatch.setattr(SparseRowStore, "finalize", finalize)
    in_tmp = {"reader_options": {"temp_dir": str(tmp_path)}}
    _import(source, MemoryStore(), shuffle_cells=True, **in_tmp)
    ((spilled, peak),) = actual
    real_usage = sparse.shutil.disk_usage

    def usage(free: int):
        def disk_usage(path):
            return real_usage(path)._replace(free=free)

        return disk_usage

    monkeypatch.setattr(sparse.shutil, "disk_usage", usage(0))
    with pytest.raises(OSError, match=r"needs (\d+) temporary bytes") as caught:
        _import(source, MemoryStore(), shuffle_cells=True, **in_tmp)
    required = int(re.search(r"needs (\d+) temporary", str(caught.value)).group(1))
    # The estimate measured on the file's first rows covers every file that
    # the spill wrote, the record of the store's own part and the writer's
    # among them. It stays well below the bytes of the entries before
    # compression beside the records: 26 bytes for each cell, its count and
    # mark in each of the two parts and its pointer in the finalized record,
    # and under 8 KiB of the records' headers, the settings and the block of
    # a bucket.
    entries = int(np.count_nonzero(_counts()))
    assert spilled < peak <= required
    assert required < entries * (4 + 1) + _N_CELLS * (2 * 9 + 8) + 8192
    monkeypatch.setattr(sparse.shutil, "disk_usage", usage(required - 1))
    with pytest.raises(OSError, match=f"needs {required} temporary bytes"):
        _import(source, MemoryStore(), shuffle_cells=True, **in_tmp)
    monkeypatch.setattr(sparse.shutil, "disk_usage", usage(required))
    _import(source, MemoryStore(), shuffle_cells=True, **in_tmp)
    assert not [path for path in tmp_path.iterdir() if path.name.startswith("cytearc-")]


def test_a_disk_that_fills_while_dense_rows_spill_is_refused(
    tmp_path, monkeypatch
) -> None:
    import cytearc.readers._sparse as sparse

    source = tmp_path / "source.h5ad"
    _write_source(source, "dense")
    real_usage = sparse.shutil.disk_usage
    checks: list[int] = []

    def disk_usage(path):
        # The check before the spill finds room; the disk then fills.
        checks.append(1)
        return real_usage(path)._replace(free=1 << 40 if len(checks) == 1 else 10)

    monkeypatch.setattr(sparse.shutil, "disk_usage", disk_usage)
    with pytest.raises(OSError, match="temporary bytes .* which has 10 free"):
        _import(
            source,
            MemoryStore(),
            shuffle_cells=True,
            reader_options={"temp_dir": str(tmp_path)},
        )
    assert not [path for path in tmp_path.iterdir() if path.name.startswith("cytearc-")]


def test_the_check_of_rows_held_in_memory_charges_every_row_and_one_block(
    tmp_path,
) -> None:
    source = tmp_path / "source.h5ad"
    _write_source(source, "csr", n_cells=20)
    reader = _reader(source)
    try:
        writer = H5adToZarr(reader, MemoryStore(), nthreads=1, shuffle_cells=True)
        # Twenty rows and a block of four, of 40 bytes each, beside the 300
        # bytes the import holds throughout; a block of more rows than the
        # cells holds the twenty.
        for block_rows, needed in ((4, (20 + 4) * 40 + 300), (50, 40 * 40 + 300)):
            writer.resources = ResourceBudget(needed, 1)
            writer._check_shuffled_rows_memory(40, block_rows, "rows", 300)
            writer.resources = ResourceBudget(needed - 1, 1)
            with pytest.raises(
                MemoryBudgetError, match="Writing rows exceeds"
            ) as caught:
                writer._check_shuffled_rows_memory(40, block_rows, "rows", 300)
            assert caught.value.needed_bytes == needed
            assert caught.value.budget_bytes == needed - 1
            assert caught.value.item == "rows"
    finally:
        reader.close()


def test_destination_rows_must_be_a_permutation(tmp_path) -> None:
    source = tmp_path / "source.h5ad"
    _write_source(source, "csr", n_cells=6)
    reader = H5adReader(str(source))
    shape = (
        "destinationRows must hold one integer row for each of the 6 cells; "
        "found dtype {} and shape {}"
    )
    permutation = (
        "destinationRows must be a permutation of the rows, each row from 0 once"
    )
    try:
        for rows, message in (
            (np.arange(5), shape.format("int64", (5,))),
            (np.ones(6) * 0.5, shape.format("float64", (6,))),
            (np.zeros(6, dtype=int), permutation),
            (np.arange(6) - 1, permutation),
        ):
            with pytest.raises(ValueError) as refused:
                reader.count_value_ranges(1 << 30, destinationRows=rows)
            assert str(refused.value) == message
        assert reader._convertedCsr is None
    finally:
        reader.close()


@pytest.mark.parametrize("n_cells", [0, 1, 2])
@pytest.mark.parametrize("encoding", ["csr", "csc", "dense"])
def test_files_of_no_one_or_two_cells_shuffle(tmp_path, encoding, n_cells) -> None:
    counts = np.arange(1, 3 * n_cells + 1, dtype=np.float32).reshape(n_cells, 3)
    matrix = {"csr": csr_matrix, "csc": csc_matrix, "dense": np.asarray}[encoding]
    source = tmp_path / "small.h5ad"
    anndata.AnnData(
        matrix(counts),
        obs=pd.DataFrame(index=[f"c{row}" for row in range(n_cells)]),
        var=pd.DataFrame(index=["a", "b", "c"]),
    ).write_h5ad(source)
    reader = H5adReader(str(source), temp_dir=str(tmp_path))
    try:
        H5adToZarr(
            reader, str(tmp_path / "small.zarr"), nthreads=1, shuffle_cells=True
        ).dump()
    finally:
        reader.close()
    root = _open(tmp_path / "small.zarr")
    order = shuffled_order(n_cells, 0)
    np.testing.assert_array_equal(root["RNA/counts"][...], counts[order])
    np.testing.assert_array_equal(root[f"cellData/{SOURCE_ROW}"][...], order)
    assert not [path for path in tmp_path.iterdir() if path.name.startswith("cytearc-")]


def test_a_csc_conversion_is_redone_in_the_order_that_an_import_needs(tmp_path) -> None:
    source = tmp_path / "source.h5ad"
    counts = _write_source(source, "csc")
    reader = H5adReader(str(source), temp_dir=str(tmp_path))
    try:
        reader.materialize_csc(1 << 30)
        plain = reader._convertedCsr
        assert plain is not None
        # A bad order is refused before the conversion in hand is given up.
        with pytest.raises(ValueError, match="destinationRows must be a permutation"):
            reader.materialize_csc(1 << 30, destinationRows=np.zeros(_N_CELLS, int))
        with pytest.raises(ValueError, match="destinationRows must hold one integer"):
            reader.materialize_csc(1 << 30, destinationRows=np.arange(3))
        assert reader._convertedCsr is plain
        # A shuffled import converts the columns again, to destination rows.
        H5adToZarr(
            reader, str(tmp_path / "shuffled.zarr"), nthreads=1, shuffle_cells=True
        ).dump()
        assert not plain.directory.exists()
        assert reader._convertedCsr is None
        # Once its counts are written, the reader converts in the file's order.
        first = next(reader.consume_row_range(_N_CELLS, 0, _N_CELLS)).toarray()
        np.testing.assert_array_equal(first, counts)
        with pytest.raises(ValueError, match="consume row range is outside"):
            reader._file_first_rows(_N_CELLS + 1)
        with pytest.raises(ValueError, match="consume row range is outside"):
            reader._file_first_rows(-1)
    finally:
        reader.close()
    np.testing.assert_array_equal(
        _open(tmp_path / "shuffled.zarr")["RNA/counts"][...],
        counts[shuffled_order(_N_CELLS, 0)],
    )


def _temporary_folders(folder: Path) -> list[Path]:
    return [path for path in folder.iterdir() if path.name.startswith("cytearc-")]


@pytest.mark.parametrize("shuffled_first", [True, False])
@pytest.mark.parametrize("encoding", ["csr", "csc", "dense"])
def test_an_import_in_the_files_order_reads_the_file_whatever_another_import_left(
    tmp_path, encoding, shuffled_first
) -> None:
    source = tmp_path / "source.h5ad"
    counts = _write_source(source, encoding)
    reader = H5adReader(str(source), temp_dir=str(tmp_path))
    try:

        def shuffled() -> H5adToZarr:
            return H5adToZarr(reader, MemoryStore(), nthreads=1, shuffle_cells=True)

        def plain() -> H5adToZarr:
            return H5adToZarr(
                reader, str(tmp_path / "plain.zarr"), nthreads=1, shuffle_cells=False
            )

        if shuffled_first:
            shuffled_writer = shuffled()
            plain_writer = plain()
        else:
            # Built before the shuffled import spills, dumped after it.
            plain_writer = plain()
            shuffled_writer = shuffled()
        plain_writer.dump()
        # The plain import released the shuffled rows, which the shuffled
        # import cannot replace with the file's order.
        with pytest.raises(RuntimeError, match="written once and released"):
            shuffled_writer.dump()
    finally:
        reader.close()
    root = _open(tmp_path / "plain.zarr")
    np.testing.assert_array_equal(
        root["cellData/ids"][...], [f"cell-{row}" for row in range(_N_CELLS)]
    )
    np.testing.assert_array_equal(root["RNA/counts"][...], counts)
    np.testing.assert_array_equal(root["RNA/countsT"][...], counts.T)
    assert SOURCE_ROW not in root["cellData"] or (
        "role" not in root[f"cellData/{SOURCE_ROW}"].attrs
    )
    assert not _temporary_folders(tmp_path)


def _write_fractional(path: Path, encoding: str, n_cells: int = 150) -> np.ndarray:
    """Write values whose first 100 cells do not show what they are; column 0 is the row plus a half."""
    values = np.zeros((n_cells, 5), dtype=np.float32)
    values[:, 0] = np.arange(n_cells) + 0.5
    values[::3, 2] = 2
    matrix = {"csr": csr_matrix, "csc": csc_matrix, "dense": np.asarray}[encoding]
    anndata.AnnData(
        matrix(values),
        obs=pd.DataFrame(index=[f"c{row}" for row in range(n_cells)]),
        var=pd.DataFrame(index=[f"g{column}" for column in range(5)]),
    ).write_h5ad(path)
    return values


@pytest.mark.parametrize("failure", ["values", "cell data"])
@pytest.mark.parametrize("encoding", ["csr", "csc", "dense"])
def test_a_shuffled_import_refused_after_its_spill_leaves_no_rows_for_a_retry(
    tmp_path, monkeypatch, encoding, failure
) -> None:
    source = tmp_path / "source.h5ad"
    values = _write_fractional(source, encoding)
    options: dict[str, Any] = {}
    if failure == "cell data":
        # The refusal comes once the destination exists.
        options["values"] = "normalized"

        def refuse(self) -> None:
            raise OSError("the destination refused a write")

        monkeypatch.setattr(H5adToZarr, "_ini_cell_data", refuse)
    reader = H5adReader(str(source), temp_dir=str(tmp_path))
    try:
        with pytest.raises(
            ValueError if failure == "values" else OSError,
            match="not whole numbers" if failure == "values" else "refused a write",
        ):
            H5adToZarr(
                reader,
                str(tmp_path / "refused.zarr"),
                nthreads=1,
                shuffle_cells=True,
                **options,
            )
        assert reader._convertedCsr is None
        assert not reader._shuffledRows
        assert not _temporary_folders(tmp_path)
        monkeypatch.undo()
        H5adToZarr(
            reader,
            str(tmp_path / "retry.zarr"),
            nthreads=1,
            values="normalized",
            shuffle_cells=False,
        ).dump()
    finally:
        reader.close()
    root = _open(tmp_path / "retry.zarr")
    np.testing.assert_array_equal(
        root["cellData/ids"][...], [f"c{row}" for row in range(values.shape[0])]
    )
    np.testing.assert_array_equal(root["RNA/counts"][...], values)


@pytest.mark.parametrize("encoding", ["csr", "csc", "dense"])
def test_interleaved_shuffled_imports_never_write_each_others_rows(
    tmp_path, encoding
) -> None:
    source = tmp_path / "source.h5ad"
    counts = _write_source(source, encoding)
    reader = H5adReader(str(source), temp_dir=str(tmp_path))
    try:
        first = H5adToZarr(
            reader, MemoryStore(), nthreads=1, shuffle_cells=True, shuffle_seed=0
        )
        second = H5adToZarr(
            reader,
            str(tmp_path / "second.zarr"),
            nthreads=1,
            shuffle_cells=True,
            shuffle_seed=1,
        )
        # The second import's rows replaced the first's, in another order.
        with pytest.raises(RuntimeError, match="another import of the reader replaced"):
            first.dump()
        second.dump()
        with pytest.raises(RuntimeError, match="another import of the reader replaced"):
            first.dump()
    finally:
        reader.close()
    root = _open(tmp_path / "second.zarr")
    order = shuffled_order(_N_CELLS, 1)
    assert not np.array_equal(order, shuffled_order(_N_CELLS, 0))
    np.testing.assert_array_equal(
        root["cellData/ids"][...], [f"cell-{row}" for row in order]
    )
    np.testing.assert_array_equal(root["RNA/counts"][...], counts[order])
    assert not _temporary_folders(tmp_path)


def _id_bytes(ids: list[str]) -> int:
    """Bytes of the reader's object array of IDs, which h5py reads as bytes objects.

    CPython's bytes object takes 33 bytes and one per byte of its value, and
    the array a reference of 8 bytes for each.
    """
    return sum(8 + 33 + len(value.encode()) for value in ids)


def _text_bytes(names: list[str]) -> int:
    """Bytes of the reader's NumPy array of feature names, 4 a character of the longest."""
    return len(names) * 4 * max(len(name) for name in names)


@pytest.mark.parametrize("output", ["embedding", "cluster labels"])
def test_per_cell_arrays_that_do_not_fit_are_refused_before_anything_is_written(
    tmp_path, output
) -> None:
    n_cells = 600
    ids = [f"c{row}" for row in range(n_cells)]
    rng = np.random.default_rng(5)
    counts = rng.poisson(1.0, (n_cells, 20)).astype(np.float32)
    obs = pd.DataFrame(index=ids)
    obsm = {}
    options: dict[str, Any] = {}
    if output == "embedding":
        obsm["X_wide"] = rng.normal(size=(n_cells, 1000))
        options["embedding_roles"] = {"X_wide": "coordinates"}
        row_bytes = 1000 * 8
        sized_bytes = row_bytes
    else:
        # Labels a thousand characters long, held as text 4 bytes a
        # character with a missing mark, in blocks sized for 64 bytes a row.
        obs["long"] = [("x" * 999) + str(row % 3) for row in range(n_cells)]
        options["cluster_keys"] = ("long",)
        row_bytes = 4 * 1000 + 1
        sized_bytes = 64
    source = tmp_path / "source.h5ad"
    anndata.AnnData(
        csr_matrix(counts),
        obs=obs,
        var=pd.DataFrame(index=[f"g{column}" for column in range(20)]),
        obsm=obsm,
    ).write_h5ad(source)
    destination = tmp_path / "refused.zarr"
    reader = H5adReader(str(source), temp_dir=str(tmp_path), **options)
    try:
        with pytest.raises(MemoryBudgetError) as caught:
            H5adToZarr(
                reader,
                str(destination),
                mem_budget="4M",
                nthreads=1,
                shuffle_cells=True,
            )
        assert reader._convertedCsr is None
    finally:
        reader.close()
    assert not destination.exists()
    assert not _temporary_folders(tmp_path)
    budget = caught.value.budget_bytes
    # The order of the cells and the IDs and names that the reader keeps are
    # held throughout: the cell IDs, the feature IDs, and the feature IDs
    # again as the names, which the reader finds no column of. A block takes
    # the most rows that an eighth of the budget sizes it for.
    features = [f"g{column}" for column in range(20)]
    block = min(n_cells, budget // (8 * sized_bytes))
    needed = (
        8 * n_cells
        + _id_bytes(ids)
        + _id_bytes(features)
        + _text_bytes(features)
        + (n_cells + block) * row_bytes
    )
    assert caught.value.needed_bytes == needed
    assert needed > budget
    assert "in shuffled order" in caught.value.item


def test_what_a_shuffled_import_holds_beside_a_gathered_array_is_charged(
    tmp_path,
) -> None:
    source = tmp_path / "source.h5ad"
    _write_source(source, "dense")
    reader = _reader(source)
    try:
        writer = H5adToZarr(reader, MemoryStore(), nthreads=1, shuffle_cells=True)
        resident = writer._shuffled_resident_bytes()
    finally:
        reader.close()
    ids = [f"cell-{row}" for row in range(_N_CELLS)]
    features = [f"feature-{f}" for f in range(_N_FEATURES)]
    names = [f"gene-{f}" for f in range(_N_FEATURES)]
    # The order, the IDs and names that the reader keeps, the declared RNA
    # membership, one boolean a cell, and the dense matrix's count of values
    # in each row.
    assert resident == (
        8 * _N_CELLS
        + _id_bytes(ids)
        + _id_bytes(features)
        + _text_bytes(names)
        + _N_CELLS
        + 8 * (_N_CELLS + 1)
    )


@pytest.mark.parametrize("shuffle_cells", [False, True], ids=["file", "shuffled"])
def test_an_import_reads_each_id_and_name_column_once(
    tmp_path, monkeypatch, shuffle_cells
) -> None:
    source = tmp_path / "source.h5ad"
    _write_source(source, "csr")
    whole_reads: list[tuple[str, str]] = []
    original = H5adReader._read_column

    def spy(self, group, key, start=0, stop=None):
        if start == 0 and stop is None:
            whole_reads.append((group, key))
        return original(self, group, key, start, stop)

    monkeypatch.setattr(H5adReader, "_read_column", spy)
    # A split reads the feature IDs and names, and a shuffled import of
    # embeddings the cell IDs, before the import checks them all.
    _import(
        source,
        tmp_path / "store.zarr",
        shuffle_cells=shuffle_cells,
        assay_split_key="feature_types",
    )

    for column in [("obs", "_index"), ("var", "_index"), ("var", "feature_name")]:
        assert whole_reads.count(column) == 1, column


def test_embeddings_gathered_in_another_order_than_the_ids_fail_their_check(
    tmp_path,
) -> None:
    from cytearc.storage.errors import ArtifactResolutionError

    source = tmp_path / "source.h5ad"
    _write_source(source, "csr")
    reader = _reader(source)
    try:
        writer = H5adToZarr(reader, MemoryStore(), nthreads=1, shuffle_cells=True)
        # The rows of each embedding are gathered through another order than
        # the store's IDs were written in.
        writer._cellOrder = writer._cellOrder[::-1].copy()
        with pytest.raises(
            ArtifactResolutionError, match="cell IDs do not match the selected"
        ):
            writer.dump()
    finally:
        reader.close()


def test_a_disk_without_room_for_dense_rows_is_refused_before_any_is_written(
    tmp_path, monkeypatch
) -> None:
    import cytearc.readers._sparse as sparse

    source = tmp_path / "source.h5ad"
    _write_source(source, "dense")
    actual: list[tuple[int, int]] = []
    appended: list[int] = []
    original_finalize = SparseRowStore.finalize
    original_append = SparseRowStore.append_rows

    def finalize(self):
        original_finalize(self)
        actual.append((int(self._fileBytes.sum()), file_bytes(self.directory)))

    def append_rows(self, rows, matrix):
        appended.append(int(matrix.nnz))
        return original_append(self, rows, matrix)

    monkeypatch.setattr(SparseRowStore, "finalize", finalize)
    monkeypatch.setattr(SparseRowStore, "append_rows", append_rows)
    in_tmp = {"reader_options": {"temp_dir": str(tmp_path)}}
    _import(source, MemoryStore(), shuffle_cells=True, **in_tmp)
    ((spilled, peak),) = actual
    real_usage = sparse.shutil.disk_usage

    def usage(free: int):
        def disk_usage(path):
            return real_usage(path)._replace(free=free)

        return disk_usage

    appended.clear()
    monkeypatch.setattr(sparse.shutil, "disk_usage", usage(0))
    with pytest.raises(OSError, match=r"needs (\d+) temporary bytes") as caught:
        _import(source, MemoryStore(), shuffle_cells=True, **in_tmp)
    assert not appended
    required = int(re.search(r"needs (\d+) temporary", str(caught.value)).group(1))
    # Estimated from the first rows' values beside zeros and their encoding:
    # it covers every file that the spill wrote, and stays well below every
    # value of the matrix before compression beside the records: the count
    # and mark of each cell in each of the two parts and its pointer, and
    # under 8 KiB of the records' headers, the settings and a bucket's block.
    assert spilled < peak <= required
    assert required < _N_CELLS * _N_FEATURES * (4 + 1) + _N_CELLS * (2 * 9 + 8) + 8192
    monkeypatch.setattr(sparse.shutil, "disk_usage", usage(required - 1))
    with pytest.raises(OSError, match=f"needs {required} temporary bytes"):
        _import(source, MemoryStore(), shuffle_cells=True, **in_tmp)
    assert not appended
    monkeypatch.setattr(sparse.shutil, "disk_usage", usage(required))
    _import(source, MemoryStore(), shuffle_cells=True, **in_tmp)
    assert not _temporary_folders(tmp_path)
