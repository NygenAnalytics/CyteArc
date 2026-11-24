"""Seurat imports that shuffle cells write row ``r`` of every per-cell array from cell ``order[r]``.

Each shuffled store is checked against an import of the same object in its
own order, whose layout the other Seurat tests pin: counts, ``countsT``,
count summaries, cell IDs, names, metadata columns, membership, the active
identity and reductions. Stores compare file for file apart from the
identifiers and creation times of their artifacts, which every import makes
anew.
"""

import dataclasses
import hashlib
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from loguru import logger
from scipy.sparse import csc_matrix
from zarr.storage import MemoryStore

import cytearc.readers._row_spill as row_spill
from cytearc.readers._rds import R_INT_NA
from cytearc.readers._sparse import SparseRowStore
from cytearc.readers.seurat import SeuratReader
from cytearc.storage.arrays import MISSING_MASK_PREFIX
from cytearc.storage.cell_order import (
    SOURCE_ROW,
    SOURCE_ROW_ROLE,
    recorded_shuffle_seed,
    shuffled_order,
)
from cytearc.storage.count_matrix import CountMatrixPolicy
from cytearc.storage.refs import artifact_path
from cytearc.utils.memory_budget import MemoryBudgetError
from cytearc.writers.seurat import SeuratToZarr
from tests.storage_helpers import drawn_shuffles, shuffle_refusal
from tests.test_seurat_import import _ordered_factor
from tests.test_seurat_reader import _Wire, _write_fragment_matrix_fixture

_N_CELLS = 41
_GENES = [f"g{index}" for index in range(9)]
_PROTEINS = ["p1", "p2"]


@contextmanager
def _warnings() -> Iterator[list[str]]:
    messages: list[str] = []
    handler = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        yield messages
    finally:
        logger.remove(handler)


def _cells(n_cells: int) -> list[str]:
    # IDs of several lengths, so their stored text is padded.
    return [f"cell-{index}" + "x" * (index % 4) for index in range(n_cells)]


def _gene_counts(n_cells: int) -> np.ndarray:
    """Gene counts, genes by cells, with an empty cell, an empty gene and a large count."""
    counts = np.random.default_rng(1).poisson(0.9, (len(_GENES), n_cells))
    counts[6] = 0
    if n_cells:
        counts[:, 3 % n_cells] = 0
        counts[2, 5 % n_cells] = 70_000
    return counts


def _members(n_cells: int) -> list[int]:
    """The cells of the protein assay: two of every three."""
    return [index for index in range(n_cells) if index % 3 != 1]


def _counts_matrix(wire: _Wire, counts: np.ndarray, cells: list[str], sparse: bool):
    if not sparse:
        # R matrices are column-major: one column of genes per cell.
        return wire.matrix(
            counts.T.ravel().tolist(), counts.shape, rows=_GENES, columns=cells
        )
    matrix = csc_matrix(counts.astype(np.float64))
    return wire.s4(
        [
            ("i", wire.integer_vector(matrix.indices.tolist())),
            ("p", wire.integer_vector(matrix.indptr.tolist())),
            ("Dim", wire.integer_vector(list(matrix.shape))),
            ("Dimnames", wire.dimnames(_GENES, cells)),
            ("x", wire.real_vector(matrix.data.tolist())),
            ("factors", wire.vector([])),
            ("class", wire.string_vector(["dgCMatrix"])),
        ]
    )


def _protein_assay(wire: _Wire, cells: list[str]) -> bytes:
    members = _members(len(cells))
    reads = np.random.default_rng(2).integers(0, 30, (len(_PROTEINS), len(members)))
    return wire.s4(
        [
            (
                "layers",
                wire.vector(
                    [wire.matrix(reads.T.ravel().tolist(), reads.shape)],
                    names=["counts"],
                ),
            ),
            (
                "cells",
                wire.logmap(
                    [1] * len(members), [cells[index] for index in members], ["counts"]
                ),
            ),
            ("features", wire.logmap([1, 1], _PROTEINS, ["counts"])),
            (
                "meta.data",
                wire.data_frame([("kind", wire.string_vector(["a", "b"]))], 2),
            ),
            ("class", wire.string_vector(["Assay5"])),
        ]
    )


def _metadata(wire: _Wire, cells: list[str], *, source_column: bool) -> bytes:
    rng = np.random.default_rng(3)
    n_cells = len(cells)
    logical = rng.integers(0, 2, n_cells).tolist()
    integer = rng.integers(-50, 50, n_cells).tolist()
    real = rng.normal(size=n_cells).tolist()
    character: list[str | None] = [
        "label-" + "y" * int(value) for value in rng.integers(0, 5, n_cells)
    ]
    factor = rng.integers(1, 4, n_cells).tolist()
    ordered = rng.integers(1, 3, n_cells).tolist()
    for row in (0, 7 % n_cells) if n_cells else ():
        logical[row] = R_INT_NA
        integer[row] = R_INT_NA
        real[row] = float("nan")
        character[row] = None
        factor[row] = R_INT_NA
        ordered[row] = R_INT_NA
    if n_cells:
        real[min(9, n_cells - 1)] = -0.0
    columns = [
        ("logical", wire.logical_vector(logical)),
        ("integer", wire.integer_vector(integer)),
        ("real", wire.real_vector(real)),
        ("character", wire.string_vector(character)),
        ("group", wire.factor(factor, ["x", "y", "z", "unused"])),
        ("stage", _ordered_factor(wire, ordered, ["early", "late"])),
    ]
    if source_column:
        # A source column that a shuffled import's own column displaces.
        columns.append((SOURCE_ROW, wire.integer_vector(list(range(n_cells))[::-1])))
    return wire.data_frame(columns, cells)


def _reductions(wire: _Wire, cells: list[str]) -> bytes:
    rng = np.random.default_rng(4)
    n_cells = len(cells)
    if not n_cells:
        return wire.vector([], names=[])
    pca = rng.normal(size=(n_cells, 3))
    pca[2 % n_cells, 1] = -0.0
    loadings = rng.normal(size=(len(_GENES), 3))
    full = wire.s4(
        [
            (
                "cell.embeddings",
                wire.matrix(
                    pca.T.ravel().tolist(),
                    pca.shape,
                    rows=cells,
                    columns=["PC_1", "PC_2", "PC_3"],
                    real=True,
                ),
            ),
            (
                "feature.loadings",
                wire.matrix(
                    loadings.T.ravel().tolist(),
                    loadings.shape,
                    rows=_GENES,
                    columns=["PC_1", "PC_2", "PC_3"],
                    real=True,
                ),
            ),
            ("assay.used", wire.string_vector(["RNA"])),
            ("global", wire.logical_vector([0])),
            ("stdev", wire.real_vector([3.0, 2.0, 1.0])),
            ("key", wire.string_vector(["PC_"])),
            ("class", wire.string_vector(["DimReduc"])),
        ]
    )
    # A layout of some of the cells, listed out of the object's order.
    subset = rng.permutation(n_cells)[: max(1, n_cells // 2)]
    umap = rng.normal(size=(subset.size, 2))
    partial = wire.s4(
        [
            (
                "cell.embeddings",
                wire.matrix(
                    umap.T.ravel().tolist(),
                    umap.shape,
                    rows=[cells[index] for index in subset],
                    columns=["UMAP_1", "UMAP_2"],
                    real=True,
                ),
            ),
            ("feature.loadings", wire.matrix([], (0, 0), real=True)),
            ("assay.used", wire.string_vector(["RNA"])),
            ("global", wire.logical_vector([1])),
            ("key", wire.string_vector(["UMAP_"])),
            ("class", wire.string_vector(["DimReduc"])),
        ]
    )
    # A layout of every cell, listed in reverse.
    tsne = rng.normal(size=(n_cells, 2))
    reordered = wire.s4(
        [
            (
                "cell.embeddings",
                wire.matrix(
                    tsne.T.ravel().tolist(),
                    tsne.shape,
                    rows=cells[::-1],
                    columns=["tSNE_1", "tSNE_2"],
                    real=True,
                ),
            ),
            ("feature.loadings", wire.matrix([], (0, 0), real=True)),
            ("assay.used", wire.string_vector(["RNA"])),
            ("global", wire.logical_vector([1])),
            ("key", wire.string_vector(["tSNE_"])),
            ("class", wire.string_vector(["DimReduc"])),
        ]
    )
    return wire.vector([full, partial, reordered], names=["pca", "umap", "tsne"])


def _write_object(
    path: Path,
    *,
    n_cells: int = _N_CELLS,
    sparse: bool = True,
    source_column: bool = True,
) -> Path:
    """Write a Seurat object of every kind of per-cell array."""
    wire = _Wire()
    cells = _cells(n_cells)
    rna = wire.s4(
        [
            ("counts", _counts_matrix(wire, _gene_counts(n_cells), cells, sparse)),
            (
                "meta.features",
                wire.data_frame([("symbol", wire.string_vector(_GENES))], _GENES),
            ),
            ("class", wire.string_vector(["Assay"])),
        ]
    )
    identity = np.random.default_rng(5).integers(1, 3, n_cells).tolist()
    if n_cells:
        identity[1 % n_cells] = R_INT_NA
    # The identities name their cells in another order than the object's.
    named = np.random.default_rng(6).permutation(n_cells)
    root = wire.s4(
        [
            (
                "assays",
                wire.vector([rna, _protein_assay(wire, cells)], names=["RNA", "ADT"]),
            ),
            ("meta.data", _metadata(wire, cells, source_column=source_column)),
            ("active.assay", wire.string_vector(["RNA"])),
            (
                "active.ident",
                wire.factor(
                    [identity[index] for index in named],
                    ["zero", "one"],
                    names=[cells[index] for index in named],
                ),
            ),
            ("reductions", _reductions(wire, cells)),
            ("graphs", wire.vector([], names=[])),
            ("class", wire.string_vector(["Seurat"])),
        ]
    )
    path.write_bytes(wire.document(root))
    return path


def _import(
    source: Path,
    destination: Any,
    *,
    batch_size: int | None = None,
    temp_dir: Path | None = None,
    **options: Any,
) -> tuple[Any, SeuratToZarr]:
    """Import ``source`` into ``destination``; return the result and the writer."""
    with SeuratReader(source, temp_dir=temp_dir) as reader:
        settings = {
            "mem_budget": "64M",
            "nthreads": 1,
            "policy": CountMatrixPolicy(unitBytes=4096, chunkBytes=1024),
            **options,
        }
        writer = SeuratToZarr(
            reader,
            str(destination) if isinstance(destination, Path) else destination,
            **settings,
        )
        result = writer.dump(batch_size=batch_size)
    return result, writer


def _open(destination: Any) -> zarr.Group:
    if isinstance(destination, MemoryStore):
        return zarr.open_group(store=destination, mode="r")
    return zarr.open_group(str(destination), mode="r")


def _arrays(group: zarr.Group) -> dict[str, np.ndarray]:
    return {name: np.asarray(array[...]) for name, array in group.arrays()}


def _assert_rows_in_order(
    shuffled: np.ndarray, plain: np.ndarray, order: np.ndarray, name: str
) -> None:
    expected = plain[order]
    assert shuffled.dtype == expected.dtype, name
    assert shuffled.shape == expected.shape, name
    # Byte views compare NaN, negative zero and text exactly.
    np.testing.assert_array_equal(
        np.ascontiguousarray(shuffled).view(np.uint8),
        np.ascontiguousarray(expected).view(np.uint8),
        err_msg=name,
    )


def _selected_rows(
    root: zarr.Group, ref: Any, values: np.ndarray
) -> dict[str, np.ndarray]:
    """Return each row of an artifact over a cell selection, by its cell's ID."""
    group = root[artifact_path(ref)]
    selection = root[artifact_path(_selection_ref(group))]
    mask = np.asarray(selection["values"][...], dtype=bool)
    ids = np.asarray(root["cellData/ids"][...])[mask]
    assert ids.size == values.shape[0]
    return {str(cell): values[index] for index, cell in enumerate(ids)}


def _selection_ref(group: zarr.Group) -> Any:
    from cytearc.storage.refs import ArtifactRef

    provenance = group.attrs["provenance"]
    return ArtifactRef.from_dict(provenance["inputs"]["cell_selection"])


def _assert_shuffled_import(
    plain_root: zarr.Group,
    shuffled_root: zarr.Group,
    plain_result: Any,
    shuffled_result: Any,
    order: np.ndarray,
    seed: int,
) -> None:
    """Check every per-cell array of the shuffled store against the plain store's rows."""
    n_cells = order.size
    assert shuffled_result.assayNames == plain_result.assayNames
    for assay in plain_result.assayNames:
        counts = np.asarray(plain_root[f"{assay}/counts"][...])
        _assert_rows_in_order(
            np.asarray(shuffled_root[f"{assay}/counts"][...]), counts, order, assay
        )
        if f"{assay}/countsT" in plain_root:
            _assert_rows_in_order(
                np.asarray(shuffled_root[f"{assay}/countsT"][...]).T,
                np.asarray(plain_root[f"{assay}/countsT"][...]).T,
                order,
                f"{assay}/countsT",
            )
        else:
            assert f"{assay}/countsT" not in shuffled_root
        plain_summaries = _arrays(plain_root[f"{assay}/countSummaries"])
        found = _arrays(shuffled_root[f"{assay}/countSummaries"])
        assert set(found) == set(plain_summaries)
        for name, values in plain_summaries.items():
            if values.shape[:1] == (n_cells,):
                _assert_rows_in_order(found[name], values, order, name)
            else:
                np.testing.assert_array_equal(found[name], values, err_msg=name)
        assert (
            _arrays(shuffled_root[f"{assay}/featureData"]).keys()
            == _arrays(plain_root[f"{assay}/featureData"]).keys()
        )
    plain_cells = _arrays(plain_root["cellData"])
    shuffled_cells = _arrays(shuffled_root["cellData"])
    # The plain import keeps the object's sourceRow column and its missing
    # mask, which the shuffled import replaces with its own column.
    source_columns = {SOURCE_ROW, f"{MISSING_MASK_PREFIX}{SOURCE_ROW}"}
    assert set(shuffled_cells) == (set(plain_cells) - source_columns) | {SOURCE_ROW}
    for name, values in plain_cells.items():
        if name in source_columns:
            continue
        _assert_rows_in_order(shuffled_cells[name], values, order, name)
        assert dict(shuffled_root[f"cellData/{name}"].attrs) == dict(
            plain_root[f"cellData/{name}"].attrs
        ), name
    np.testing.assert_array_equal(shuffled_cells[SOURCE_ROW], order)
    assert shuffled_cells[SOURCE_ROW].dtype == np.int64
    assert dict(shuffled_root[f"cellData/{SOURCE_ROW}"].attrs) == {
        "role": SOURCE_ROW_ROLE
    }
    assert recorded_shuffle_seed(shuffled_root["cellData"]) == seed
    assert recorded_shuffle_seed(plain_root["cellData"]) is None
    # The active identity follows the same order.
    plain_identity = _arrays(plain_root[artifact_path(plain_result.activeIdentity)])
    shuffled_identity = _arrays(
        shuffled_root[artifact_path(shuffled_result.activeIdentity)]
    )
    assert set(shuffled_identity) == set(plain_identity)
    for name, values in plain_identity.items():
        _assert_rows_in_order(shuffled_identity[name], values, order, name)
    # Reductions over every cell follow the order; one over some of the cells
    # keeps each cell's row, in the store's order of its cells.
    assert set(shuffled_result.reductionArtifacts) == set(
        plain_result.reductionArtifacts
    )
    for key, ref in plain_result.reductionArtifacts.items():
        plain_arrays = _arrays(plain_root[artifact_path(ref)])
        shuffled_ref = shuffled_result.reductionArtifacts[key]
        found = _arrays(shuffled_root[artifact_path(shuffled_ref)])
        assert set(found) == set(plain_arrays)
        for name, values in plain_arrays.items():
            if values.shape[:1] == (n_cells,):
                _assert_rows_in_order(found[name], values, order, f"{key}/{name}")
            elif name in ("values", "data"):
                plain_rows = _selected_rows(plain_root, ref, values)
                shuffled_rows = _selected_rows(shuffled_root, shuffled_ref, found[name])
                assert shuffled_rows.keys() == plain_rows.keys()
                for cell, row in plain_rows.items():
                    np.testing.assert_array_equal(
                        shuffled_rows[cell].view(np.uint8), row.view(np.uint8)
                    )
            else:
                np.testing.assert_array_equal(found[name], values)


@pytest.mark.parametrize("sparse", [True, False], ids=["sparse", "dense"])
def test_every_per_cell_array_follows_the_shuffled_order(tmp_path, sparse) -> None:
    source = _write_object(tmp_path / "object.rds", sparse=sparse)
    with _warnings() as plain_warnings:
        plain_result, _ = _import(
            source, tmp_path / "plain.zarr", batch_size=7, shuffle_cells=False
        )
    with _warnings() as shuffled_warnings:
        shuffled_result, writer = _import(
            source,
            tmp_path / "shuffled.zarr",
            batch_size=7,
            temp_dir=tmp_path,
            shuffle_cells=True,
            shuffle_seed=11,
        )
    assert writer.shuffleCells is True and writer.shuffleSeed == 11
    plain = _open(tmp_path / "plain.zarr")
    _assert_shuffled_import(
        plain,
        _open(tmp_path / "shuffled.zarr"),
        plain_result,
        shuffled_result,
        shuffled_order(_N_CELLS, 11),
        11,
    )
    # The fixture's arrays are what the other Seurat tests pin.
    assert plain["RNA/counts"].shape == (_N_CELLS, len(_GENES))
    np.testing.assert_array_equal(plain["RNA/counts"][...], _gene_counts(_N_CELLS).T)
    assert not plain["cellData/ADT_I"][...][1::3].any()
    assert {"logical", "integer", "real", "character", "group", "stage"} <= set(
        plain["cellData"].array_keys()
    )
    assert set(plain_result.reductionArtifacts) == {"pca", "umap", "tsne"}
    skipped = "Skipped source cell metadata column 'sourceRow' because CyteArc stores"
    assert any(message.startswith(skipped) for message in shuffled_warnings)
    assert not any(SOURCE_ROW in message for message in plain_warnings)
    # The spill is gone once the import ends.
    assert not list(tmp_path.glob("cytearc-sparse-*"))


def test_cells_of_many_buckets_are_read_in_batches_and_deleted(
    tmp_path, monkeypatch
) -> None:
    source = _write_object(tmp_path / "object.rds")
    plain_result, _ = _import(source, tmp_path / "plain.zarr", shuffle_cells=False)
    # Buckets of a few cells each, read in batches that cross them.
    monkeypatch.setattr(row_spill, "_SPILL_READ_BYTES", 600)
    stores: list[SparseRowStore] = []
    deleted: list[int] = []
    original_init = SparseRowStore.__init__
    original_delete = SparseRowStore._delete

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        stores.append(self)

    def delete(self, bucket, start, stop):
        deleted.append(bucket)
        original_delete(self, bucket, start, stop)

    monkeypatch.setattr(SparseRowStore, "__init__", init)
    monkeypatch.setattr(SparseRowStore, "_delete", delete)
    shuffled_result, _ = _import(
        source,
        tmp_path / "shuffled.zarr",
        batch_size=5,
        temp_dir=tmp_path,
        shuffle_cells=True,
    )
    # One store per assay, each of several buckets, every one read and deleted.
    assert [store.operation for store in stores] == ["shuffled Seurat import"] * 2
    buckets = sum(store.bucket_starts.size - 1 for store in stores)
    assert all(store.bucket_starts.size > 3 for store in stores)
    assert len(deleted) == buckets
    _assert_shuffled_import(
        _open(tmp_path / "plain.zarr"),
        _open(tmp_path / "shuffled.zarr"),
        plain_result,
        shuffled_result,
        shuffled_order(_N_CELLS, 0),
        0,
    )
    assert not list(tmp_path.glob("cytearc-sparse-*"))


def test_cells_counted_from_fragments_shuffle(tmp_path) -> None:
    # Fragment counts reach the import through a row store of their own.
    source = _write_fragment_matrix_fixture(tmp_path / "fragments.rds")
    plain_result, _ = _import(source, tmp_path / "plain.zarr", shuffle_cells=False)
    seed = next(
        seed
        for seed in range(100)
        if not np.array_equal(shuffled_order(3, seed), np.arange(3))
    )
    shuffled_result, _ = _import(
        source,
        tmp_path / "shuffled.zarr",
        temp_dir=tmp_path,
        shuffle_cells=True,
        shuffle_seed=seed,
    )
    plain = _open(tmp_path / "plain.zarr")
    assert np.asarray(plain["ATAC/counts"][...]).any()
    _assert_shuffled_import(
        plain,
        _open(tmp_path / "shuffled.zarr"),
        plain_result,
        shuffled_result,
        shuffled_order(3, seed),
        seed,
    )
    assert not list(tmp_path.glob("cytearc-sparse-*"))


def _store_files(path: Path, results: list[Any]) -> dict[str, str]:
    """Digest every file of a store, its artifacts' identifiers and creation times masked."""
    names: dict[str, str] = {}
    root = _open(path)
    for result in results:
        names[result.cellSelection.artifact_id] = "cell_selection"
        names[result.activeIdentity.artifact_id] = "active_identity"
        for key, ref in result.reductionArtifacts.items():
            names[ref.artifact_id] = f"reduction:{key}"
            # A reduction of some of the cells has a selection of its own.
            selection = _selection_ref(root[artifact_path(ref)])
            names.setdefault(selection.artifact_id, f"selection:{key}")
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
    source = _write_object(tmp_path / "object.rds")
    results = {}
    for name, seed in (("first", 6), ("second", 6), ("other", 7)):
        results[name], _ = _import(
            source, tmp_path / f"{name}.zarr", shuffle_cells=True, shuffle_seed=seed
        )
    first = _store_files(tmp_path / "first.zarr", [results["first"]])
    assert first == _store_files(tmp_path / "second.zarr", [results["second"]])
    assert "cellData/sourceRow/zarr.json" in first
    other = _store_files(tmp_path / "other.zarr", [results["other"]])
    assert other["cellData/sourceRow/c/0"] != first["cellData/sourceRow/c/0"]
    np.testing.assert_array_equal(
        _open(tmp_path / "other.zarr")["cellData/sourceRow"][...],
        shuffled_order(_N_CELLS, 7),
    )


def test_an_import_shuffles_with_seed_0_unless_told_not_to(tmp_path) -> None:
    source = _write_object(tmp_path / "object.rds")
    results = {}
    for name, options in (
        ("default", {}),
        ("seed0", {"shuffle_cells": True, "shuffle_seed": 0}),
        ("plain", {"shuffle_cells": False}),
        ("explicit", {"shuffle_cells": False, "shuffle_seed": 9}),
    ):
        results[name], writer = _import(
            source, tmp_path / f"{name}.zarr", temp_dir=tmp_path, **options
        )
        assert writer.shuffleCells is (name in ("default", "seed0"))
    stores = {
        name: _store_files(tmp_path / f"{name}.zarr", [result])
        for name, result in results.items()
    }
    # The default is the shuffle with seed 0, and the object's column named
    # sourceRow gives way to each cell's position among the object's cells.
    assert stores["default"] == stores["seed0"]
    cells = _open(tmp_path / "default.zarr")["cellData"]
    assert recorded_shuffle_seed(cells) == 0
    assert cells[SOURCE_ROW].attrs["role"] == SOURCE_ROW_ROLE
    np.testing.assert_array_equal(cells[SOURCE_ROW][...], shuffled_order(_N_CELLS, 0))
    # shuffle_cells=False writes the object's order, whatever the seed.
    assert stores["plain"] == stores["explicit"] != stores["default"]
    cells = _open(tmp_path / "plain.zarr")["cellData"]
    assert "cellOrder" not in cells.attrs
    # The source's own column of that name is imported as any other.
    assert "role" not in cells[SOURCE_ROW].attrs
    np.testing.assert_array_equal(cells[SOURCE_ROW][...], np.arange(_N_CELLS)[::-1])
    assert not list(tmp_path.glob("cytearc-sparse-*"))


@pytest.mark.parametrize("n_cells", [0, 1, 2])
def test_objects_of_no_one_or_two_cells_shuffle(tmp_path, n_cells) -> None:
    source = _write_object(tmp_path / "object.rds", n_cells=n_cells)
    plain_result, _ = _import(source, tmp_path / "plain.zarr", shuffle_cells=False)
    shuffled_result, _ = _import(
        source,
        tmp_path / "shuffled.zarr",
        temp_dir=tmp_path,
        shuffle_cells=True,
        shuffle_seed=3,
    )
    _assert_shuffled_import(
        _open(tmp_path / "plain.zarr"),
        _open(tmp_path / "shuffled.zarr"),
        plain_result,
        shuffled_result,
        shuffled_order(n_cells, 3),
        3,
    )


def test_a_shuffled_import_dumps_once(tmp_path) -> None:
    source = _write_object(tmp_path / "object.rds")
    with SeuratReader(source, temp_dir=tmp_path) as reader:
        writer = SeuratToZarr(
            reader, MemoryStore(), mem_budget="64M", nthreads=1, shuffle_cells=True
        )
        writer.dump()
        with pytest.raises(RuntimeError, match="written once and removed"):
            writer.dump()
    assert not list(tmp_path.glob("cytearc-sparse-*"))


def test_a_failed_dump_removes_the_shuffled_cells(tmp_path, monkeypatch) -> None:
    source = _write_object(tmp_path / "object.rds")
    with SeuratReader(source, temp_dir=tmp_path) as reader:
        writer = SeuratToZarr(
            reader, MemoryStore(), mem_budget="64M", nthreads=1, shuffle_cells=True
        )
        # One store for each assay's shuffled cells.
        assert len(list(tmp_path.glob("cytearc-sparse-*"))) == 2

        def fail(*_args: Any, **_kwargs: Any) -> None:
            raise OSError("the destination failed")

        monkeypatch.setattr(SeuratToZarr, "_write_counts", fail)
        with pytest.raises(OSError, match="the destination failed"):
            writer.dump()
        assert not list(tmp_path.glob("cytearc-sparse-*"))
        with pytest.raises(RuntimeError, match="written once and removed"):
            writer.dump()


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"shuffle_cells": 1}, TypeError, "shuffle_cells must be a boolean"),
        (
            {"shuffle_cells": True, "shuffle_seed": -1},
            ValueError,
            "shuffle_seed must be from 0 to 2**64 - 1, not -1",
        ),
        (
            {"shuffle_cells": True, "shuffle_seed": 2**64},
            ValueError,
            "shuffle_seed must be from 0 to 2**64 - 1, not 18446744073709551616",
        ),
        (
            {"shuffle_cells": True, "shuffle_seed": 1.5},
            TypeError,
            "shuffle_seed must be an integer",
        ),
    ],
)
def test_bad_shuffle_arguments_are_refused_before_the_store_exists(
    tmp_path, options, error, message
) -> None:
    source = _write_object(tmp_path / "object.rds", n_cells=5)
    with SeuratReader(source, temp_dir=tmp_path) as reader:
        with pytest.raises(error) as refused:
            SeuratToZarr(reader, str(tmp_path / "refused.zarr"), **options)
    assert str(refused.value) == message
    assert not (tmp_path / "refused.zarr").exists()
    assert not list(tmp_path.glob("cytearc-sparse-*"))


def test_a_budget_without_room_for_the_spill_is_refused_with_the_bytes_it_needs(
    tmp_path,
) -> None:
    n_cells = 5
    source = _write_object(tmp_path / "object.rds", n_cells=n_cells)
    refusals = {}
    for shuffle in (False, True):
        with SeuratReader(source, temp_dir=tmp_path) as reader:
            with pytest.raises(MemoryBudgetError) as caught:
                SeuratToZarr(
                    reader,
                    str(tmp_path / "refused.zarr"),
                    mem_budget=900,
                    nthreads=1,
                    shuffle_cells=shuffle,
                )
        refusals[shuffle] = caught.value
    plain, shuffled = refusals[False], refusals[True]
    assert plain.item == "reading one cell of assay 'RNA'"
    assert shuffled.item == "shuffling one cell of assay 'RNA'"
    assert shuffled.budget_bytes == plain.budget_bytes == 900
    assert str(shuffled).startswith(
        "Assay 'RNA' counts cannot be read and sent to their shuffled rows"
    )
    # Each bucket of the shuffled cells reads in a quarter of what the import
    # leaves, and holds the longest cell whole: every gene of RNA, float64
    # values of 3 * 8 + 32 bytes each as they are read, and 112 bytes for the
    # cell's piece and pointers. Beside them, the import holds ADT's source,
    # whose protein and cell names count their UTF-8 bytes and 8 more and
    # whose layer maps each protein and member cell by an int64 index; RNA's
    # prefix sums of each cell's stored values, 8 bytes a cell and one; and
    # the order and its inverse, 16 bytes a cell.
    bucket = 4 * (len(_GENES) * (3 * 8 + 32) + 112)
    names = [*_PROTEINS, *_cells(n_cells)]
    adt = (
        sum(len(name.encode()) + 8 for name in names)
        + 8 * len(_PROTEINS)
        + 8 * len(_members(n_cells))
    )
    held = adt + 8 * (n_cells + 1) + 16 * n_cells
    assert shuffled.needed_bytes == held + bucket
    assert not (tmp_path / "refused.zarr").exists()
    assert not list(tmp_path.glob("cytearc-sparse-*"))


@pytest.mark.parametrize("n_cells", [40, 3000])
def test_a_shuffled_import_names_one_budget_for_a_bucket_and_one_cell(
    tmp_path, n_cells
) -> None:
    source = _write_object(tmp_path / "object.rds", n_cells=n_cells)
    attempts = iter(range(100))

    def refusal(budget: int) -> MemoryBudgetError | None:
        with SeuratReader(source, temp_dir=tmp_path) as reader:
            try:
                SeuratToZarr(
                    reader,
                    str(tmp_path / f"store-{next(attempts)}.zarr"),
                    mem_budget=budget,
                    nthreads=1,
                    shuffle_cells=True,
                )
            except MemoryBudgetError as error:
                return error
        return None

    # The refusals of the earlier steps lead from 900 bytes to the shuffle's.
    refused = refusal(900)
    while refused is not None and refused.item != "shuffling one cell of assay 'RNA'":
        refused = refusal(refused.needed_bytes)
    assert refused is not None
    named = refused.needed_bytes
    # Beside ADT's source, RNA's prefix sums and the order and its inverse, as
    # the three-cell import holds them, a bucket reads the longest cell: every
    # gene of RNA, float64 values of 3 * 8 + 32 bytes each as they are read,
    # and 112 bytes for the cell's piece and pointers. Forty cells and more
    # need more to read one cell and send it, and the first refusal names
    # that budget: the least, though a larger budget sizes larger buckets,
    # which need less beside them.
    names = [*_PROTEINS, *_cells(n_cells)]
    adt = (
        sum(len(name.encode()) + 8 for name in names)
        + 8 * len(_PROTEINS)
        + 8 * len(_members(n_cells))
    )
    held = adt + 8 * (n_cells + 1) + 16 * n_cells
    assert named > held + 4 * (len(_GENES) * (3 * 8 + 32) + 112)
    # With the budget it names, the import reads and spills one cell, and a
    # later step refuses for itself; one byte less is refused again, naming
    # the same budget.
    later = refusal(named)
    assert later is None or "one cell of assay" not in later.item
    below = refusal(named - 1)
    assert below is not None
    assert (below.item, below.needed_bytes) == (
        "shuffling one cell of assay 'RNA'",
        named,
    )
    assert not list(tmp_path.glob("cytearc-sparse-*"))


@pytest.mark.parametrize("shuffle_cells", [False, True], ids=["object", "shuffled"])
def test_a_seurat_import_charges_its_counts_t_write_with_its_sources(
    tmp_path, monkeypatch, shuffle_cells
) -> None:
    from cytearc.storage import sharding

    charged: list[int] = []
    preflight = sharding.preflight_counts_t_spec

    def recorded(*args: Any, residentBytes: int = 0, **options: Any) -> Any:
        charged.append(int(residentBytes))
        return preflight(*args, residentBytes=residentBytes, **options)

    monkeypatch.setattr(sharding, "preflight_counts_t_spec", recorded)
    n_cells = 5
    source = _write_object(tmp_path / "object.rds", n_cells=n_cells)
    with SeuratReader(source, temp_dir=tmp_path) as reader:
        SeuratToZarr(
            reader,
            str(tmp_path / "store.zarr"),
            nthreads=1,
            shuffle_cells=shuffle_cells,
        ).dump()
    # RNA's countsT runs beside ADT's source, whose protein and cell names
    # count their UTF-8 bytes and 8 more and whose layer maps each protein
    # and member cell by an int64 index; RNA's and ADT's prefix sums of each
    # cell's stored values, 8 bytes a cell and one each; and a shuffled
    # import's order, 8 bytes a cell. The scans let go of the order's
    # inverse, and the counts writes of the shuffled cells.
    names = [*_PROTEINS, *_cells(n_cells)]
    adt = (
        sum(len(name.encode()) + 8 for name in names)
        + 8 * len(_PROTEINS)
        + 8 * len(_members(n_cells))
    )
    held = adt + 2 * 8 * (n_cells + 1) + (8 * n_cells if shuffle_cells else 0)
    # The layout fit plans the write and the write admits itself, both with
    # that charge.
    assert len(charged) >= 2
    assert set(charged) == {held}


def test_a_budget_without_room_for_the_shuffle_is_refused_before_it_runs(
    tmp_path, monkeypatch
) -> None:
    source = _write_object(tmp_path / "object.rds")
    # The count sources that the import holds when it shuffles. RNA's
    # dgCMatrix stays in the file. ADT's counts stitch its layer into the
    # object's cells: the source keeps each protein and cell name, counted as
    # its UTF-8 bytes and 8 more, and the layer maps each protein and each of
    # its cells by an int64 index.
    names = [*_PROTEINS, *_cells(_N_CELLS)]
    held = (
        sum(len(name.encode()) + 8 for name in names)
        + 8 * len(_PROTEINS)
        + 8 * len(_members(_N_CELLS))
    )
    # Drawing the order and its inverse holds 24 bytes a cell beside them.
    needed = held + 24 * _N_CELLS
    drawn = drawn_shuffles(monkeypatch)

    def attempt(destination: Path, budget: int) -> None:
        with SeuratReader(source, temp_dir=tmp_path) as reader:
            SeuratToZarr(
                reader,
                str(destination),
                mem_budget=budget,
                nthreads=1,
                shuffle_cells=True,
            ).dump()

    with pytest.raises(MemoryBudgetError) as caught:
        attempt(tmp_path / "refused.zarr", needed - 1)
    refusal = caught.value
    assert (refusal.needed_bytes, refusal.budget_bytes) == (needed, needed - 1)
    assert str(refusal) == shuffle_refusal(_N_CELLS, held, needed - 1)
    # The import is refused before it draws the order or writes anything.
    assert drawn == []
    assert not (tmp_path / "refused.zarr").exists()
    # With the budget it names, the import draws the order, and a later step
    # of so small a budget refuses for itself, if at all.
    try:
        attempt(tmp_path / "admitted.zarr", needed)
    except MemoryError as later:
        assert "random order" not in str(later)
    assert drawn == [_N_CELLS]
    assert not list(tmp_path.glob("cytearc-sparse-*"))


def test_a_disk_without_room_for_the_spill_is_refused_before_it_is_written(
    tmp_path, monkeypatch
) -> None:
    import cytearc.readers._sparse as sparse

    source = _write_object(tmp_path / "object.rds")
    real_usage = sparse.shutil.disk_usage

    def disk_usage(path):
        # Only the store's disk is full; the reader's scratch files fit.
        usage = real_usage(path)
        return usage._replace(free=10) if "cytearc-sparse-" in str(path) else usage

    monkeypatch.setattr(sparse.shutil, "disk_usage", disk_usage)
    with pytest.raises(OSError, match=r"^Shuffled Seurat import needs \d+ temporary"):
        _import(source, MemoryStore(), temp_dir=tmp_path, shuffle_cells=True)
    assert not list(tmp_path.glob("cytearc-sparse-*"))


def _write_wide_object(
    path: Path, *, dims: int = 400, text_width: int = 1, n_cells: int = _N_CELLS
) -> Path:
    """Write an object of sparse RNA counts, a text column and a reduction of ``dims`` dimensions."""
    wire = _Wire()
    cells = _cells(n_cells)
    rna = wire.s4(
        [
            ("counts", _counts_matrix(wire, _gene_counts(n_cells), cells, True)),
            (
                "meta.features",
                wire.data_frame([("symbol", wire.string_vector(_GENES))], _GENES),
            ),
            ("class", wire.string_vector(["Assay"])),
        ]
    )
    pca = np.random.default_rng(4).normal(size=(n_cells, dims))
    wide = wire.s4(
        [
            (
                "cell.embeddings",
                wire.matrix(
                    pca.T.ravel().tolist(),
                    pca.shape,
                    rows=cells,
                    columns=[f"PC_{k + 1}" for k in range(dims)],
                    real=True,
                ),
            ),
            ("feature.loadings", wire.matrix([], (0, 0), real=True)),
            ("assay.used", wire.string_vector(["RNA"])),
            ("global", wire.logical_vector([1])),
            ("key", wire.string_vector(["PC_"])),
            ("class", wire.string_vector(["DimReduc"])),
        ]
    )
    text = ["t" * (text_width - 1) + str(index % 3) for index in range(n_cells)]
    root = wire.s4(
        [
            ("assays", wire.vector([rna], names=["RNA"])),
            (
                "meta.data",
                wire.data_frame([("text", wire.string_vector(text))], cells),
            ),
            ("active.assay", wire.string_vector(["RNA"])),
            ("active.ident", wire.factor([1] * n_cells, ["a"], names=cells)),
            ("reductions", wire.vector([wide], names=["pca"])),
            ("graphs", wire.vector([], names=[])),
            ("class", wire.string_vector(["Seurat"])),
        ]
    )
    path.write_bytes(wire.document(root))
    return path


@pytest.mark.parametrize(
    ("options", "item"),
    [
        ({"dims": 4000}, "the coordinates of reduction 'pca' in shuffled order"),
        (
            {"dims": 2, "text_width": 4000},
            "cell metadata column 'text' in shuffled order",
        ),
    ],
)
def test_per_cell_arrays_that_do_not_fit_are_refused_before_the_destination_exists(
    tmp_path, options, item
) -> None:
    source = _write_wide_object(tmp_path / "object.rds", **options)
    destination = tmp_path / "refused.zarr"
    with SeuratReader(source, temp_dir=tmp_path) as reader:
        with pytest.raises(MemoryBudgetError) as caught:
            SeuratToZarr(
                reader,
                str(destination),
                mem_budget="1M",
                nthreads=1,
                shuffle_cells=True,
            )
        # The same budget imports the object in its own order.
        SeuratToZarr(
            reader,
            str(tmp_path / "plain.zarr"),
            mem_budget="1M",
            nthreads=1,
            shuffle_cells=False,
        ).dump()
    assert caught.value.item == item
    assert caught.value.needed_bytes > caught.value.budget_bytes
    assert not destination.exists()
    assert not list(tmp_path.glob("cytearc-sparse-*"))


def test_what_a_shuffled_import_holds_beside_a_gathered_array_is_charged(
    tmp_path,
) -> None:
    dims = 400
    source = _write_wide_object(tmp_path / "object.rds", dims=dims)
    with SeuratReader(source, temp_dir=tmp_path) as reader:
        writer = SeuratToZarr(
            reader, MemoryStore(), mem_budget="64M", nthreads=1, shuffle_cells=True
        )
        stores = list(writer._shuffledRows.values())
        resident = writer._source_resident_bytes()
        # Beside each array, the import holds the order of the cells, 8 bytes
        # a cell, the row pointers of the shuffled cells, and the prefix sums
        # of each cell's stored values, 8 bytes a cell and one each. The
        # dgCMatrix stays in the file, so its source holds nothing.
        pointers = 8 * (_N_CELLS + 1)
        assert [store.indptr.nbytes for store in stores] == [pointers]
        assert resident == 8 * _N_CELLS + pointers + pointers
        width = max(len(cell) for cell in _cells(_N_CELLS))
        budget = 100_000
        # Blocks of the reduction take an eighth of the budget at the 8 bytes
        # a value it is read in. Each row holds its coordinates, its cell
        # name as text of 4 bytes a character, and 16 bytes of its place in
        # the rows' order; beside them, 9 bytes a cell of store rows and mask.
        block = min(_N_CELLS, budget // (8 * dims * 8))
        row_bytes = dims * 8 + 4 * width + 16
        needed = resident + 9 * _N_CELLS + (_N_CELLS + block) * row_bytes
        assert needed > budget
        for memory in (budget, needed - 1):
            writer.resources = dataclasses.replace(writer.resources, memoryBytes=memory)
            with pytest.raises(MemoryBudgetError) as caught:
                writer._check_shuffled_holds(np.dtype(f"U{width}"))
            assert (
                caught.value.item
                == "the coordinates of reduction 'pca' in shuffled order"
            )
            block = min(_N_CELLS, memory // (8 * dims * 8))
            assert caught.value.needed_bytes == (
                resident + 9 * _N_CELLS + (_N_CELLS + block) * row_bytes
            )
        writer._release_shuffled_rows()


def test_the_check_of_rows_held_in_memory_charges_every_row_and_one_block(
    tmp_path,
) -> None:
    source = _write_wide_object(tmp_path / "object.rds", dims=2)
    with SeuratReader(source, temp_dir=tmp_path) as reader:
        writer = SeuratToZarr(
            reader, MemoryStore(), mem_budget="64M", nthreads=1, shuffle_cells=True
        )
        # Rows of 40 bytes beside 300 held throughout: every cell's row and a
        # block of four, or of every row when a block holds more rows than
        # there are; a reduction of 20 cells holds its own rows alone.
        for block_rows, n_rows, needed in (
            (4, None, (_N_CELLS + 4) * 40 + 300),
            (100, None, 2 * _N_CELLS * 40 + 300),
            (100, 20, 2 * 20 * 40 + 300),
        ):
            writer.resources = dataclasses.replace(writer.resources, memoryBytes=needed)
            writer._check_shuffled_rows_memory(
                40, block_rows, "rows", 300, n_rows=n_rows
            )
            writer.resources = dataclasses.replace(
                writer.resources, memoryBytes=needed - 1
            )
            with pytest.raises(MemoryBudgetError) as caught:
                writer._check_shuffled_rows_memory(
                    40, block_rows, "rows", 300, n_rows=n_rows
                )
            assert (caught.value.needed_bytes, caught.value.item) == (needed, "rows")
        writer._release_shuffled_rows()


def test_a_shuffled_import_without_an_assays_cells_refuses_its_counts(
    tmp_path,
) -> None:
    source = _write_object(tmp_path / "object.rds")
    with SeuratReader(source, temp_dir=tmp_path) as reader:
        writer = SeuratToZarr(
            reader, MemoryStore(), mem_budget="64M", nthreads=1, shuffle_cells=True
        )
        # The RNA cells were lost, so its counts would read the object's order.
        writer._shuffledRows.pop("RNA").close()
        with pytest.raises(RuntimeError, match="shuffled cells of assay 'RNA'"):
            writer.dump()
    assert not list(tmp_path.glob("cytearc-sparse-*"))


@pytest.mark.parametrize("reduction", ["pca", "umap", "tsne"])
def test_reductions_gathered_in_another_order_than_the_ids_fail_their_check(
    tmp_path, monkeypatch, reduction
) -> None:
    from cytearc.storage.errors import ArtifactResolutionError

    source = _write_object(tmp_path / "object.rds")
    original = SeuratToZarr._shuffled_reduction_cells

    def reversed_rows(self, selected, cell_selection):
        selection, order = original(self, selected, cell_selection)
        return selection, order[::-1].copy()

    # The rows of the reduction, of every cell in the object's order, of
    # some cells, or of every cell in another order, are gathered in another
    # order than the store's IDs were written in.
    monkeypatch.setattr(SeuratToZarr, "_shuffled_reduction_cells", reversed_rows)
    with SeuratReader(source, temp_dir=tmp_path, reductions=[reduction]) as reader:
        writer = SeuratToZarr(
            reader, MemoryStore(), mem_budget="64M", nthreads=1, shuffle_cells=True
        )
        with pytest.raises(
            ArtifactResolutionError, match="cell IDs do not match the selected"
        ):
            writer.dump()


def test_a_shuffled_import_keeps_no_bucket_of_its_converted_cells(tmp_path) -> None:
    from cytearc.readers._seurat.sources import BaseMatrixSource, _matrix_sources

    # Fragment counts reach the import through a row store of their own.
    source = _write_fragment_matrix_fixture(tmp_path / "fragments.rds")
    with SeuratReader(source, temp_dir=tmp_path) as reader:
        SeuratToZarr(
            reader, MemoryStore(), mem_budget="64M", nthreads=1, shuffle_cells=True
        )._release_shuffled_rows()
        stores = [
            current._rowStore
            for current in _matrix_sources(reader.get_assay("ATAC").counts)
            if isinstance(current, BaseMatrixSource) and current._rowStore is not None
        ]
        assert stores
        # The scan read every converted cell, but the shuffled counts write
        # reads none, so no bucket stays held through it.
        assert all(store._cache is None for store in stores)
        # Reads in the object's order load buckets again.
        SeuratToZarr(
            reader, MemoryStore(), mem_budget="64M", nthreads=1, shuffle_cells=False
        ).dump()


def test_imports_of_one_reader_each_hold_their_own_shuffled_cells(tmp_path) -> None:
    source = _write_object(tmp_path / "object.rds")
    with SeuratReader(source, temp_dir=tmp_path) as reader:
        shuffled = SeuratToZarr(
            reader,
            str(tmp_path / "shuffled.zarr"),
            mem_budget="64M",
            nthreads=1,
            shuffle_cells=True,
        )
        # The shuffled cells are the writer's, so an import of the same reader
        # in the object's order reads the object, and the shuffled one still
        # writes its own.
        SeuratToZarr(
            reader,
            str(tmp_path / "plain.zarr"),
            mem_budget="64M",
            nthreads=1,
            shuffle_cells=False,
        ).dump()
        shuffled.dump()
    counts = _gene_counts(_N_CELLS).T
    order = shuffled_order(_N_CELLS, 0)
    np.testing.assert_array_equal(
        _open(tmp_path / "plain.zarr")["RNA/counts"][...], counts
    )
    np.testing.assert_array_equal(
        _open(tmp_path / "shuffled.zarr")["RNA/counts"][...], counts[order]
    )
    assert not list(tmp_path.glob("cytearc-sparse-*"))


def test_a_disk_without_room_for_dense_cells_is_refused_before_any_is_written(
    tmp_path, monkeypatch
) -> None:
    import cytearc.readers._sparse as sparse

    source = _write_object(tmp_path / "object.rds", sparse=False)
    appended: list[int] = []
    original_append = SparseRowStore.append_rows

    def append_rows(self, rows, matrix):
        appended.append(int(matrix.nnz))
        return original_append(self, rows, matrix)

    monkeypatch.setattr(SparseRowStore, "append_rows", append_rows)
    real_usage = sparse.shutil.disk_usage

    def disk_usage(path):
        usage = real_usage(path)
        return usage._replace(free=0) if "cytearc-sparse-" in str(path) else usage

    monkeypatch.setattr(sparse.shutil, "disk_usage", disk_usage)
    with pytest.raises(OSError, match=r"^Shuffled Seurat import needs \d+ temporary"):
        _import(source, MemoryStore(), temp_dir=tmp_path, shuffle_cells=True)
    # The dense RNA cells, the first sent, were refused from the estimate of
    # their first cells, before any was sent.
    assert not appended
    assert not list(tmp_path.glob("cytearc-sparse-*"))


def test_cell_rows_keep_repeated_coordinates_in_the_blocks_order() -> None:
    from scipy.sparse import coo_matrix, csr_matrix

    from cytearc.writers.seurat import _cell_rows

    rows = np.array([2, 0, 2, 1, 0, 2])
    columns = np.array([3, 1, 0, 3, 1, 3])
    values = np.array([1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
    converted = _cell_rows(coo_matrix((values, (rows, columns)), shape=(4, 5)))
    assert converted.format == "csr" and converted.shape == (4, 5)
    # Every entry stays, each row's in the order the block lists them.
    np.testing.assert_array_equal(converted.indptr, [0, 2, 3, 6, 6])
    np.testing.assert_array_equal(converted.indices, [1, 1, 3, 3, 0, 3])
    np.testing.assert_array_equal(converted.data, [2.0, 5.0, 4.0, 1.5, 3.0, 6.0])
    dense = np.arange(12.0).reshape(3, 4) % 3
    np.testing.assert_array_equal(_cell_rows(dense).toarray(), dense)
    given = csr_matrix(dense)
    assert _cell_rows(given) is given


def test_repeated_coordinates_sum_as_the_objects_order_sums_them(
    tmp_path, monkeypatch
) -> None:
    from scipy.sparse import coo_matrix

    from cytearc.readers._seurat.sources import CompressedMatrixSource

    original = CompressedMatrixSource.read_cells

    def read_cells(self, start: int, stop: int):
        # Each count arrives as three parts, 1e16, the count and -1e16,
        # whose float sum depends on the order of its terms. Each cell lists
        # its parts in a scrambled order of its own, whatever window reads it.
        block = original(self, start, stop).tocoo()
        big = np.full(block.nnz, 1e16)
        rows = np.concatenate([block.row] * 3)
        columns = np.concatenate([block.col] * 3)
        values = np.concatenate([big, block.data, -big])
        parts = np.repeat(np.arange(3), block.nnz)
        keys = ((rows + start) * 7919 + columns * 104_729 + parts * 31) % 1009
        order = np.argsort(keys, kind="stable")
        return coo_matrix(
            (values[order], (rows[order], columns[order])), shape=block.shape
        )

    monkeypatch.setattr(CompressedMatrixSource, "read_cells", read_cells)
    source = _write_object(tmp_path / "object.rds")
    plain_result, _ = _import(
        source, tmp_path / "plain.zarr", batch_size=7, shuffle_cells=False
    )
    shuffled_result, _ = _import(
        source,
        tmp_path / "shuffled.zarr",
        batch_size=7,
        temp_dir=tmp_path,
        shuffle_cells=True,
        shuffle_seed=2,
    )
    plain = _open(tmp_path / "plain.zarr")
    counts = np.asarray(plain["RNA/counts"][...])
    # The order of the terms changed some sums, so another order of the
    # parts would store other counts.
    assert not np.array_equal(counts, _gene_counts(_N_CELLS).T)
    # Bit for bit the plain import's sums, in shuffled order.
    _assert_rows_in_order(
        np.asarray(_open(tmp_path / "shuffled.zarr")["RNA/counts"][...]),
        counts,
        shuffled_order(_N_CELLS, 2),
        "RNA/counts",
    )
