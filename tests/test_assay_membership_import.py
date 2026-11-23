"""Assay membership survives export and import only when it is declared.

An export of one assay keeps that assay's ``<assay>_I`` membership column as
an ordinary ``obs`` column and declares it in
``uns["cytearc"]["assayMembership"]``; the membership columns of other assays
describe assays that the file does not hold and are not exported. An import
restores a declared column as the membership of an imported assay of that
name, skips any other column named for an imported assay with a warning, and
keeps the column of an assay that it does not import as ordinary metadata.
A merge's record of which features each source measured travels the same
way, declared in ``uns["cytearc"]["measuredFeatures"]``.
"""

import json
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import zarr
from scipy.sparse import csr_matrix
from zarr.storage import MemoryStore

import cytearc
from cytearc import DataStore
from cytearc.merge import DataStoreMerge
from cytearc.metadata.membership import (
    measured_feature_columns,
    membership_attributes,
)
from cytearc.readers import CSVReader, H5adReader
from cytearc.utils.logging import logger
from cytearc.writers import CrToZarr, CSVtoZarr, H5adToZarr, SparseToZarr
from tests.test_writer_paths import _CountsReader, _sentinel_store, _untouched


@contextmanager
def _captured_warnings() -> Iterator[list[str]]:
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    try:
        yield messages
    finally:
        logger.remove(sink)


def _store(path: Path, assays: dict[str, np.ndarray], prefix: str) -> DataStore:
    """Write and prepare a store whose assays share one set of cells."""
    (n_cells,) = {len(values) for values in assays.values()}
    cells = [f"{prefix}{index}" for index in range(n_cells)]
    first, *others = assays
    SparseToZarr(
        csr_matrix(assays[first]),
        str(path),
        cells,
        [f"{first}{index}" for index in range(assays[first].shape[1])],
        assay_name=first,
        nthreads=1,
    ).dump()
    for name in others:
        scratch = path.parent / f"{path.stem}_{name}.zarr"
        SparseToZarr(
            csr_matrix(assays[name]),
            str(scratch),
            cells,
            [f"{name}{index}" for index in range(assays[name].shape[1])],
            assay_name=name,
            nthreads=1,
        ).dump()
        shutil.copytree(scratch / name, path / name)
        zarr.open_group(str(path), mode="r+").attrs["assayTypes"] = {
            assay: assay for assay in assays
        }
        shutil.rmtree(scratch)
    return DataStore(
        str(path), default_assay=first, min_features_per_cell=-1, nthreads=1
    )


def _counts(n_cells: int, n_features: int, seed: int) -> np.ndarray:
    return (
        np.random.default_rng(seed)
        .integers(1, 9, (n_cells, n_features))
        .astype(np.uint32)
    )


def _rows(store: DataStore, column: str) -> dict[str, bool]:
    ids = store.cells.fetch_all("ids").astype(str)
    return dict(zip(ids, store.cells.fetch_all(column).tolist(), strict=True))


def _partial_sources(base: Path) -> list[DataStore]:
    """Write a source with RNA and ADT and a source with ADT only."""
    return [
        _store(
            base / "both.zarr", {"RNA": _counts(4, 3, 1), "ADT": _counts(4, 2, 2)}, "b"
        ),
        _store(base / "adt.zarr", {"ADT": _counts(3, 2, 3)}, "a"),
    ]


def _merged_with_partial_rna(
    tmp_path: Path, sources: list[DataStore] | None = None
) -> DataStore:
    """Merge a source with RNA and ADT and a source with ADT only."""
    merged = str(tmp_path / "merged.zarr")
    DataStoreMerge(
        sources or _partial_sources(tmp_path), merged, ["both", "adt"], nthreads=1
    ).dump()
    return DataStore(merged, default_assay="RNA", min_features_per_cell=-1, nthreads=1)


@pytest.fixture(scope="module")
def partial_merge(tmp_path_factory) -> tuple[list[DataStore], DataStore]:
    """The sources of ``_merged_with_partial_rna`` and their merged store."""
    base = tmp_path_factory.mktemp("partial_merge")
    sources = _partial_sources(base)
    return sources, _merged_with_partial_rna(base, sources)


def _attributes(store: DataStore, column: str) -> dict[str, object]:
    return dict(store.cells._get_array(column).attrs)


@pytest.mark.slow
def test_a_merged_store_round_trips_through_h5ad_and_merges_again(
    tmp_path, partial_merge
) -> None:
    _sources, merged = partial_merge
    membership = _rows(merged, "RNA_I")
    assert sorted(membership.values()) == [False] * 3 + [True] * 4

    exported = tmp_path / "merged.h5ad"
    cytearc.to_h5ad(merged.RNA, str(exported))
    with h5py.File(exported, "r") as h5:
        declared = h5["uns/cytearc/assayMembership"]
        # The file holds RNA only, so it declares and writes RNA's membership.
        assert {key: declared[key][()].decode() for key in declared} == {"RNA": "RNA_I"}
        assert declared.attrs["encoding-type"] == "dict"
        assert "RNA_I" in h5["obs"]
        assert "ADT_I" not in h5["obs"]
        assert "ADT_I" not in list(h5["obs"].attrs["column-order"])

    reader = H5adReader(str(exported))
    try:
        assert reader.assay_membership() == {"RNA": "RNA_I"}
        H5adToZarr(reader, str(tmp_path / "imported.zarr"), nthreads=1).dump()
    finally:
        reader.close()
    imported = DataStore(
        str(tmp_path / "imported.zarr"), min_features_per_cell=-1, nthreads=1
    )
    # RNA, which the file declares and the import writes, keeps its membership.
    assert _rows(imported, "RNA_I") == membership
    assert _attributes(imported, "RNA_I") == membership_attributes("RNA")
    # ADT's membership described an assay absent from the file.
    assert "ADT_I" not in imported.cells.columns

    # Before, the imported RNA_I lacked its attributes and merge refused it.
    other = _store(tmp_path / "other.zarr", {"RNA": _counts(2, 3, 4)}, "o")
    remerged = str(tmp_path / "remerged.zarr")
    DataStoreMerge(
        [imported, other], remerged, ["imported", "other"], nthreads=1
    ).dump()
    result = DataStore(remerged, min_features_per_cell=-1, nthreads=1)
    expected = {f"imported__{cell}": value for cell, value in membership.items()}
    expected.update({"other__o0": True, "other__o1": True})
    assert _rows(result, "RNA_I") == expected
    assert "orig_RNA_I" not in result.cells.columns
    assert "orig_ADT_I" not in result.cells.columns


@pytest.mark.slow
def test_to_anndata_declares_the_membership_of_live_metadata(
    tmp_path, partial_merge
) -> None:
    _sources, merged = partial_merge
    adata = merged.to_anndata(from_assay="RNA", cell_key="I")
    # Only the exported assay's membership is written and declared.
    assert adata.uns["cytearc"] == {"assayMembership": {"RNA": "RNA_I"}}
    assert "RNA_I" in adata.obs.columns
    assert "ADT_I" not in adata.obs.columns
    adt = merged.to_anndata(from_assay="ADT")
    assert adt.uns["cytearc"] == {"assayMembership": {"ADT": "ADT_I"}}
    assert "RNA_I" not in adt.obs.columns

    # A store without membership columns declares nothing.
    plain = _store(tmp_path / "plain.zarr", {"RNA": _counts(3, 2, 5)}, "p")
    assert "cytearc" not in plain.to_anndata().uns
    cytearc.to_h5ad(plain.RNA, str(tmp_path / "plain.h5ad"))
    with h5py.File(tmp_path / "plain.h5ad", "r") as h5:
        assert "uns" not in h5


def _foreign_h5ad(path: Path, obs: dict[str, object], uns: dict | None = None) -> Path:
    import anndata

    n_cells = len(next(iter(obs.values())))
    adata = anndata.AnnData(
        csr_matrix(_counts(n_cells, 2, 6).astype(np.float32)),
        obs=pd.DataFrame(obs, index=[f"c{index}" for index in range(n_cells)]),
        var=pd.DataFrame(index=["f0", "f1"]),
        uns=uns or {},
    )
    adata.write_h5ad(path)
    return path


def test_a_declaration_restores_only_the_assays_it_names(tmp_path) -> None:
    source = _foreign_h5ad(
        tmp_path / "declared.h5ad",
        {"RNA_I": [True, False, True], "ADT_I": [False, True, True]},
        uns={"cytearc": {"assayMembership": {"RNA": "RNA_I"}}},
    )
    messages: dict[str, list[str]] = {}
    for assay in ("RNA", "ADT"):
        reader = H5adReader(str(source))
        try:
            with _captured_warnings() as messages[assay]:
                H5adToZarr(
                    reader,
                    str(tmp_path / f"{assay.lower()}.zarr"),
                    assay_name=assay,
                    nthreads=1,
                    shuffle_cells=False,
                ).dump()
        finally:
            reader.close()
    assert not any("RNA_I" in message for message in messages["RNA"])

    rna = zarr.open_group(str(tmp_path / "rna.zarr"), mode="r")["cellData"]
    np.testing.assert_array_equal(rna["RNA_I"][:], [True, False, True])
    assert dict(rna["RNA_I"].attrs) == membership_attributes("RNA")
    # The declaration names RNA only, so an ADT import skips ADT_I, and the
    # column of an assay that it does not import stays ordinary.
    adt = zarr.open_group(str(tmp_path / "adt.zarr"), mode="r")["cellData"]
    assert "ADT_I" not in adt
    assert (
        "Skipped source cell metadata column 'ADT_I' because CyteArc reserves it for "
        "the membership of the imported assay 'ADT'"
    ) in messages["ADT"]
    np.testing.assert_array_equal(adt["RNA_I"][:], [True, False, True])
    assert "role" not in adt["RNA_I"].attrs


@pytest.mark.parametrize(
    ("obs", "declared", "message"),
    [
        (
            {"RNA_I": [1, 0, 1]},
            {"RNA": "RNA_I"},
            "'RNA_I' is declared as the membership of assay 'RNA', but it has dtype",
        ),
        (
            {"RNA_I": pd.array([True, None, False], dtype="boolean")},
            {"RNA": "RNA_I"},
            "'RNA_I' is declared as the membership of assay 'RNA', but it has "
            "missing values",
        ),
        (
            {"other": [1, 2, 3]},
            {"RNA": "RNA_I"},
            "declares 'RNA_I' as the membership of assay 'RNA', but obs has no",
        ),
        (
            {"RNA_I": [True, False, True]},
            {"RNA": "membership"},
            r"maps 'RNA' to 'membership'",
        ),
        (
            {"RNA_I": [True, False, True]},
            "RNA_I",
            "uns/cytearc/assayMembership must be a mapping",
        ),
        (
            {"RNA_I": [True, False, True]},
            {"RNA": ["RNA_I"]},
            "uns/cytearc/assayMembership/RNA must hold one text value",
        ),
    ],
    ids=["not-boolean", "missing", "absent", "other-column", "string", "list"],
)
def test_a_malformed_declaration_is_refused_before_writing(
    tmp_path, obs, declared, message
) -> None:
    source = _foreign_h5ad(
        tmp_path / "malformed.h5ad",
        obs,
        uns={"cytearc": {"assayMembership": declared}},
    )
    destination = _sentinel_store()
    reader = H5adReader(str(source))
    try:
        with pytest.raises(ValueError, match=message):
            H5adToZarr(reader, destination, nthreads=1)
    finally:
        reader.close()
    assert _untouched(destination)


def test_a_record_of_measured_features_round_trips_through_h5ad(tmp_path) -> None:
    # The second panel holds ADT2, which the first lacks.
    sources = [
        _store(tmp_path / "x.zarr", {"ADT": _counts(3, 2, 1)}, "x"),
        _store(tmp_path / "y.zarr", {"ADT": _counts(2, 3, 2)}, "y"),
    ]
    merged_path = str(tmp_path / "merged.zarr")
    DataStoreMerge(sources, merged_path, ["x", "y"], nthreads=1).dump()
    merged = DataStore(
        merged_path, default_assay="ADT", min_features_per_cell=-1, nthreads=1
    )
    record = {"measured_in_x": "x", "measured_in_y": "y"}
    assert measured_feature_columns(merged.ADT.feats) == record
    values = {column: merged.ADT.feats.fetch_all(column).tolist() for column in record}
    assert values == {
        "measured_in_x": [True, True, False],
        "measured_in_y": [True, True, True],
    }
    assert merged.to_anndata(from_assay="ADT").uns["cytearc"] == {
        "assayMembership": {"ADT": "ADT_I"},
        "measuredFeatures": {"ADT": record},
        "cellSources": "dataset",
    }
    lacking = dict(
        zip(
            merged.cells.fetch_all("ids").astype(str),
            np.isnan(merged.get_cell_vals("ADT", "I", "ADT2")),
            strict=True,
        )
    )
    # Only the cells of source y measured ADT2.
    assert sorted(str(cell) for cell, missing in lacking.items() if not missing) == [
        "y__y0",
        "y__y1",
    ]

    exported = tmp_path / "merged.h5ad"
    cytearc.to_h5ad(merged.ADT, str(exported))
    for assay, categorical in (("ADT", True), ("text", False), ("Protein", True)):
        reader = H5adReader(str(exported))
        try:
            assert reader.measured_features() == {"ADT": record}
            assert reader.cell_sources() == "dataset"
            H5adToZarr(
                reader,
                str(tmp_path / f"{assay}.zarr"),
                assay_name="ADT" if assay == "text" else assay,
                categorical_columns=categorical,
                nthreads=1,
            ).dump()
        finally:
            reader.close()
    for name in ("ADT", "text"):
        imported = DataStore(
            str(tmp_path / f"{name}.zarr"),
            default_assay="ADT",
            min_features_per_cell=-1,
            nthreads=1,
        )
        # The import restores the record and the column of cell sources that
        # it names, as codes or as text, so the store still reads which
        # features each cell's source measured.
        assert measured_feature_columns(imported.ADT.feats) == record
        assert {
            column: imported.ADT.feats.fetch_all(column).tolist() for column in record
        } == values
        assert imported.ADT.feats._get_array("measured_in_x").attrs["sourceColumn"] == (
            "dataset"
        )
        assert imported.cells._get_array("dataset").attrs["role"] == "cell_sources"
        assert (
            dict(
                zip(
                    imported.cells.fetch_all("ids").astype(str),
                    np.isnan(imported.get_cell_vals("ADT", "I", "ADT2")),
                    strict=True,
                )
            )
            == lacking
        )
    # The record describes an assay named ADT; an assay imported under another
    # name keeps the columns as ordinary metadata.
    protein = zarr.open_group(str(tmp_path / "Protein.zarr"), mode="r")
    assert "role" not in protein["Protein/featureData/measured_in_x"].attrs


def _var_h5ad(path: Path, var: dict[str, object], uns: dict) -> Path:
    import anndata

    n_features = len(next(iter(var.values())))
    adata = anndata.AnnData(
        csr_matrix(_counts(3, n_features, 7).astype(np.float32)),
        obs=pd.DataFrame(index=["c0", "c1", "c2"]),
        var=pd.DataFrame(var, index=[f"f{index}" for index in range(n_features)]),
        uns=uns,
    )
    adata.write_h5ad(path)
    return path


@pytest.mark.parametrize(
    ("var", "declared", "message"),
    [
        (
            {"measured_in_a": [1, 0]},
            {"RNA": {"measured_in_a": "a"}},
            "Column 'measured_in_a' is declared as the record of which features a "
            "source of assay 'RNA' measured, but it has dtype int64, not bool",
        ),
        (
            {"measured_in_a": pd.array([True, None], dtype="boolean")},
            {"RNA": {"measured_in_a": "a"}},
            "Column 'measured_in_a' is declared as the record of which features a "
            "source of assay 'RNA' measured, but it has missing values",
        ),
        (
            {"other": [True, False]},
            {"RNA": {"measured_in_a": "a"}},
            "The file declares 'measured_in_a' as the record of which features a "
            "source of assay 'RNA' measured, but var has no such column",
        ),
        (
            {"measured_in_a": [True, False]},
            {"RNA": {"measured_in_a": "b"}},
            "maps column 'measured_in_a' of assay 'RNA' to the source 'b'",
        ),
        (
            {"measured_in_a": [True, False]},
            "measured_in_a",
            "uns/cytearc/measuredFeatures must be a mapping from assay name",
        ),
        (
            {"measured_in_a": [True, False]},
            {"RNA": "measured_in_a"},
            "uns/cytearc/measuredFeatures/RNA must be a mapping from column name",
        ),
        (
            {"measured_in_a": [True, False]},
            {"RNA": {"measured_in_a": ["a"]}},
            "uns/cytearc/measuredFeatures/RNA/measured_in_a must hold one text value",
        ),
    ],
    ids=[
        "not-boolean",
        "missing",
        "absent",
        "other-source",
        "string",
        "assay-not-mapping",
        "list",
    ],
)
def test_a_malformed_record_of_measured_features_is_refused_before_writing(
    tmp_path, var, declared, message
) -> None:
    source = _var_h5ad(
        tmp_path / "malformed.h5ad",
        var,
        uns={"cytearc": {"measuredFeatures": declared}},
    )
    destination = _sentinel_store()
    reader = H5adReader(str(source))
    try:
        with pytest.raises(ValueError, match=re.escape(message)):
            H5adToZarr(reader, destination, nthreads=1)
    finally:
        reader.close()
    assert _untouched(destination)


@pytest.mark.parametrize(
    ("declared", "named"),
    [
        (
            None,
            "declares no obs column of each cell's source; export the store "
            "again with this release",
        ),
        (
            "study",
            "obs lacks the column 'study' that it declares as each cell's source",
        ),
    ],
)
def test_a_record_without_its_column_of_cell_sources_is_refused_before_writing(
    tmp_path, declared, named
) -> None:
    uns: dict = {"cytearc": {"measuredFeatures": {"RNA": {"measured_in_a": "a"}}}}
    if declared is not None:
        uns["cytearc"]["cellSources"] = declared
    source = _var_h5ad(tmp_path / "unnamed.h5ad", {"measured_in_a": [True, False]}, uns)
    destination = _sentinel_store()
    reader = H5adReader(str(source))
    try:
        with pytest.raises(ValueError) as refused:
            H5adToZarr(reader, destination, nthreads=1)
    finally:
        reader.close()
    assert str(refused.value) == (
        "The file declares which features each source of a merge measured, for "
        f"assays ['RNA'], but {named}"
    )
    assert _untouched(destination)


def test_a_record_whose_column_of_cell_sources_the_import_skips_is_refused(
    tmp_path,
) -> None:
    import anndata

    # CyteArc reserves the cell column I, so an import skips an obs column of
    # that name, which the record names as the column of cell sources.
    adata = anndata.AnnData(
        csr_matrix(_counts(3, 2, 7).astype(np.float32)),
        obs=pd.DataFrame({"I": ["a", "a", "b"]}, index=["c0", "c1", "c2"]),
        var=pd.DataFrame({"measured_in_a": [True, False]}, index=["f0", "f1"]),
        uns={
            "cytearc": {
                "measuredFeatures": {"RNA": {"measured_in_a": "a"}},
                "cellSources": "I",
            }
        },
    )
    source = tmp_path / "reserved.h5ad"
    adata.write_h5ad(source)
    destination = _sentinel_store()
    reader = H5adReader(str(source))
    try:
        with pytest.raises(ValueError) as refused:
            H5adToZarr(reader, destination, nthreads=1)
    finally:
        reader.close()
    assert str(refused.value) == (
        "The file declares which features each source of a merge measured, for "
        "assays ['RNA'], but the import would skip the obs column 'I' that it "
        "declares as each cell's source, whose name CyteArc reserves or cannot "
        "store; rename it in the file and import it again"
    )
    assert _untouched(destination)


def test_an_import_names_the_column_of_sources_by_the_key_it_writes(
    tmp_path,
) -> None:
    import anndata

    # The obs column 'lab\\study' is stored as 'lab_study', which the record
    # must name for the store to find each cell's source.
    counts = np.array([[4, 2], [3, 5], [6, 1]], dtype=np.float32)
    adata = anndata.AnnData(
        csr_matrix(counts),
        obs=pd.DataFrame(
            {"lab\\study": pd.Categorical(["x", "x", "y"])}, index=["c0", "c1", "c2"]
        ),
        var=pd.DataFrame(
            {"measured_in_x": [True, True], "measured_in_y": [True, False]},
            index=["f0", "f1"],
        ),
        uns={
            "cytearc": {
                "measuredFeatures": {
                    "RNA": {"measured_in_x": "x", "measured_in_y": "y"}
                },
                "cellSources": "lab\\study",
            }
        },
    )
    source = tmp_path / "renamed.h5ad"
    adata.write_h5ad(source)
    reader = H5adReader(str(source))
    try:
        H5adToZarr(
            reader, str(tmp_path / "renamed.zarr"), nthreads=1, shuffle_cells=False
        ).dump()
    finally:
        reader.close()
    store = DataStore(
        str(tmp_path / "renamed.zarr"), min_features_per_cell=-1, nthreads=1
    )
    assert store.RNA.feats._get_array("measured_in_y").attrs["sourceColumn"] == (
        "lab_study"
    )
    # Only source x measured f1, so the cell of y has no value of it.
    values = store.get_cell_vals("RNA", "I", "f1")
    np.testing.assert_array_equal(np.isnan(values), [False, False, True])


def test_csv_and_cell_ranger_imports_reserve_membership_names(tmp_path) -> None:
    path = tmp_path / "counts.csv"
    path.write_text(
        "cell,RNA_I,ADT_I,g0,g1\nc0,True,False,1,0\nc1,False,True,0,2\nc2,True,True,3,1\n"
    )
    reader = CSVReader(
        str(path), id_column=0, cell_data_cols=["RNA_I", "ADT_I"], batch_size=2
    )
    store = MemoryStore()
    with _captured_warnings() as messages:
        CSVtoZarr(
            reader, store, assay_name="RNA", nthreads=1, shuffle_cells=False
        ).dump()
    columns = zarr.open_group(store=store, mode="r")["cellData"]
    assert "RNA_I" not in columns
    assert any(
        "'RNA_I' because CyteArc reserves it for the membership of the imported "
        "assay 'RNA'" in message
        for message in messages
    )
    np.testing.assert_array_equal(columns["ADT_I"][:], [False, True, True])

    # Cell Ranger and Matrix Market imports share one writer.
    values = np.eye(3, dtype=np.uint8)
    cell_ranger = MemoryStore()
    with _captured_warnings() as messages:
        CrToZarr(
            _CountsReader(
                values,
                assay="ADT",
                cell_columns=(
                    ("ADT_I", np.array([True, False, True])),
                    ("RNA_I", np.array([False, False, True])),
                ),
            ),
            zarr_loc=cell_ranger,
            nthreads=1,
            shuffle_cells=False,
        ).dump()
    columns = zarr.open_group(store=cell_ranger, mode="r")["cellData"]
    assert "ADT_I" not in columns
    assert any("imported assay 'ADT'" in message for message in messages)
    np.testing.assert_array_equal(columns["RNA_I"][:], [False, False, True])


class _Documents(MemoryStore):
    """A MemoryStore that keeps every ``zarr.json`` document written, in order."""

    def __init__(
        self,
        store_dict=None,
        *,
        read_only: bool = False,
        documents: list[tuple[str, dict]] | None = None,
    ) -> None:
        super().__init__(store_dict, read_only=read_only)
        self.documents = [] if documents is None else documents

    def with_read_only(self, read_only: bool = False) -> "_Documents":
        return type(self)(
            self._store_dict, read_only=read_only, documents=self.documents
        )

    async def set(self, key, value, byte_range=None) -> None:
        await super().set(key, value, byte_range)
        if key.rsplit("/", 1)[-1] == "zarr.json":
            self.documents.append((key, json.loads(value.to_bytes())))

    def first_attributes(self, key: str) -> dict:
        """Return the attributes of the first document written at ``key``."""
        return next(
            document.get("attributes", {})
            for written, document in self.documents
            if written == key
        )


def _write_declared_membership(store: _Documents) -> str:
    from cytearc.writers._store import write_membership_column

    cells = zarr.open_group(store=store, mode="w").create_group("cellData")
    write_membership_column(cells, "RNA", np.array([True, False, True]))
    return "RNA"


type _PartialMerge = tuple[list[DataStore], DataStore]


def _import_seurat_membership(
    store: _Documents, tmp_path: Path, _partial: _PartialMerge
) -> str:
    from cytearc.readers.seurat import SeuratReader
    from cytearc.storage.count_matrix import CountMatrixPolicy
    from cytearc.writers.seurat import SeuratToZarr
    from tests.test_seurat_import import _write_partial_fixture

    source = _write_partial_fixture(tmp_path / "partial.rds")
    with SeuratReader(source) as reader:
        SeuratToZarr(
            reader,
            store,
            mem_budget="64M",
            nthreads=1,
            policy=CountMatrixPolicy(unitBytes=4096, chunkBytes=1024),
        ).dump(batch_size=1)
    return "ADT"


def _merge_membership(
    store: _Documents, _tmp_path: Path, partial: _PartialMerge
) -> str:
    sources, _merged = partial
    DataStoreMerge(sources, store, ["both", "adt"], nthreads=1).dump()
    return "RNA"


def _derive_membership(
    store: _Documents, _tmp_path: Path, _partial: _PartialMerge
) -> str:
    from cytearc.storage.schema import create_cell_data, derived_assay_transaction
    from cytearc.writers._store import write_membership_column
    from tests.storage_helpers import finalize_test_counts

    root = zarr.open_group(store=store, mode="w")
    ids = np.array(["c0", "c1", "c2"])
    create_cell_data(root, None, ids=ids, names=ids)
    write_membership_column(root["cellData"], "RNA", np.array([True, False, True]))
    with derived_assay_transaction(
        root, "SCORES", None, operation="add_grouped_assay", membership="RNA_I"
    ) as transaction:
        counts = transaction.create_counts(3, ["f0"], ["f0"], np.float64)
        counts[:] = 1.0
        finalize_test_counts(counts)
    return "SCORES"


def _subset_membership(
    store: _Documents, _tmp_path: Path, partial: _PartialMerge
) -> str:
    from cytearc.writers import SubsetZarr

    _sources, merged = partial
    SubsetZarr(
        store,
        [merged.RNA, merged.ADT],
        cell_ids=merged.cells.fetch_all("ids")[:5],
        nthreads=1,
    ).dump()
    return "RNA"


@pytest.mark.parametrize(
    "write",
    [
        lambda store, _tmp_path, _partial: _write_declared_membership(store),
        _import_seurat_membership,
        _merge_membership,
        _subset_membership,
        _derive_membership,
    ],
    ids=["h5ad", "seurat", "merge", "subset", "derived"],
)
def test_a_membership_column_carries_its_role_from_its_first_write(
    tmp_path, partial_merge, write
) -> None:
    store = _Documents()
    assay = write(store, tmp_path, partial_merge)

    key = f"cellData/{assay}_I/zarr.json"
    attributes = store.first_attributes(key)
    # Before, the attributes followed the values in a later metadata write,
    # so a reader in between found a column without its role.
    assert {name: attributes.get(name) for name in ("role", "assay")} == (
        membership_attributes(assay)
    )
    # The column keeps them in every later metadata write too.
    for written, document in store.documents:
        if written == key:
            assert membership_attributes(assay).items() <= (
                document.get("attributes", {}).items()
            )
