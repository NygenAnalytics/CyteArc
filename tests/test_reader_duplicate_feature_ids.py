"""Repeated feature IDs in the count readers: rejected, or renamed on request."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr
from scipy.sparse import csr_matrix
from zarr.storage import MemoryStore

from cytearc.readers import (
    CrDirReader,
    CrH5Reader,
    H5adReader,
    MtxReader,
    inspect_h5ad,
    inspect_mtx,
)
from cytearc.readers._text import unique_feature_identifiers
from cytearc.storage.count_matrix import CountMatrixPolicy
from cytearc.utils.logging import logger
from cytearc.writers import CrToZarr, H5adToZarr, MtxToZarr

_POLICY = CountMatrixPolicy(unitBytes=16, chunkBytes=8)
_VALUES = np.array([[1, 0, 2, 0], [0, 3, 0, 4], [5, 0, 0, 6]], dtype=np.uint16)
# Gene symbols used as IDs repeat; the third ID collides with the name that
# the second occurrence of GENE would take first.
_IDS = ["GENE", "OTHER", "GENE-1", "GENE"]
_RENAMED = ["GENE", "OTHER", "GENE-1", "GENE-2"]
_NAMES = ["Gene a", "Other", "Gene b", "Gene c"]


@contextmanager
def _warnings() -> Iterator[list[str]]:
    messages: list[str] = []
    handler = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        yield messages
    finally:
        logger.remove(handler)


def _feature_ids(root: zarr.Group, assay: str = "RNA") -> list[str]:
    return [str(value) for value in root[f"{assay}/featureData/ids"][:]]


def _feature_names(root: zarr.Group, assay: str = "RNA") -> list[str]:
    return [str(value) for value in root[f"{assay}/featureData/names"][:]]


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["A", "B", "A", "A", "C", "B"], ["A", "B", "A-1", "A-2", "C", "B-1"]),
        # A suffix that names another ID is skipped, wherever that ID is.
        (["A", "A", "A-1"], ["A", "A-2", "A-1"]),
        (["A-1", "A", "A"], ["A-1", "A", "A-2"]),
        (["A", "A", "A-1", "A-1"], ["A", "A-2", "A-1", "A-1-1"]),
        ([b"g", b"g"], ["g", "g-1"]),
    ],
)
def test_make_unique_renames_later_occurrences_in_file_order(
    values: list[object], expected: list[str]
) -> None:
    array = np.asarray(values, dtype=object)
    with _warnings() as messages:
        resolved = unique_feature_identifiers(array, "Feature IDs", "make_unique")
    assert list(resolved) == expected
    renamed = sum(
        str(before) != after
        for before, after in zip(
            [value.decode() if isinstance(value, bytes) else value for value in values],
            expected,
            strict=True,
        )
    )
    assert len(messages) == 1
    assert f"{renamed} later occurrences were renamed" in messages[0]
    assert "Feature names are unchanged" in messages[0]


def test_identifiers_without_repeats_are_returned_unchanged() -> None:
    values = np.asarray([b"a", "b", 3], dtype=object)
    with _warnings() as messages:
        for mode in ("error", "make_unique"):
            assert unique_feature_identifiers(values, "Feature IDs", mode) is values
    assert messages == []


def test_repeated_or_blank_identifiers_raise() -> None:
    repeated = np.asarray(["A", "A"], dtype=object)
    with pytest.raises(
        ValueError,
        match=(
            "Feature IDs must contain unique values; pass "
            "duplicate_feature_ids='make_unique'"
        ),
    ):
        unique_feature_identifiers(repeated, "Feature IDs", "error")
    # Renaming never fills in missing or blank IDs.
    for blank in (["A", "A", " "], ["A", None, "A"]):
        with pytest.raises(ValueError, match="must contain non-empty values"):
            unique_feature_identifiers(
                np.asarray(blank, dtype=object), "Feature IDs", "make_unique"
            )


def _write_mex(
    directory: Path,
    ids: list[str],
    *,
    barcodes: list[str] | None = None,
    reference: str | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    matrix = csr_matrix(_VALUES)
    entries = [
        f"{feature + 1} {cell + 1} {int(value)}"
        for cell in range(matrix.shape[0])
        for feature, value in zip(
            matrix.indices[matrix.indptr[cell] : matrix.indptr[cell + 1]],
            matrix.data[matrix.indptr[cell] : matrix.indptr[cell + 1]],
            strict=True,
        )
    ]
    (directory / "matrix.mtx").write_text(
        "%%MatrixMarket matrix coordinate integer general\n"
        f"{_VALUES.shape[1]} {_VALUES.shape[0]} {len(entries)}\n"
        + "\n".join(entries)
        + "\n"
    )
    (directory / "features.tsv").write_text(
        "".join(
            f"{feature_id}\t{name}\tGene Expression\n"
            for feature_id, name in zip(ids, _NAMES, strict=True)
        )
    )
    cells = barcodes or [f"cell-{index}" for index in range(_VALUES.shape[0])]
    (directory / "barcodes.tsv").write_text("".join(f"{cell}\n" for cell in cells))
    if reference is not None:
        (directory / "feature_reference.csv").write_text(reference)
    return directory


@pytest.mark.parametrize("reader_type", ["MtxReader", "CrDirReader"])
def test_matrix_market_readers_reject_repeated_feature_ids_by_default(
    tmp_path: Path, reader_type: str
) -> None:
    directory = _write_mex(tmp_path / "mex", _IDS)
    with pytest.raises(ValueError, match="Feature IDs must contain unique values"):
        if reader_type == "MtxReader":
            MtxReader(inspect_mtx(directory)[0])
        else:
            CrDirReader(str(directory))


@pytest.mark.parametrize("reader_type", ["MtxReader", "CrDirReader"])
def test_matrix_market_readers_rename_repeated_feature_ids_on_request(
    tmp_path: Path, reader_type: str
) -> None:
    directory = _write_mex(tmp_path / "mex", _IDS)
    with _warnings() as messages:
        reader = (
            MtxReader(inspect_mtx(directory)[0], duplicate_feature_ids="make_unique")
            if reader_type == "MtxReader"
            else CrDirReader(str(directory), duplicate_feature_ids="make_unique")
        )
    store = MemoryStore()
    try:
        assert reader.feature_ids() == _RENAMED
        assert reader.feature_ids("RNA") == _RENAMED
        assert reader.feature_names() == _NAMES
        MtxToZarr(reader, store, mem_budget="64M", nthreads=1, policy=_POLICY).dump()
    finally:
        reader.close()
    assert [message.strip() for message in messages if "renamed" in message] == [
        "Feature IDs repeat 1 values: 1 later occurrences were renamed with a "
        "numbered suffix, for example 'GENE' to 'GENE-2'. Feature names are "
        "unchanged."
    ]
    root = zarr.open_group(store=store, mode="r")
    assert _feature_ids(root) == _RENAMED
    assert _feature_names(root) == _NAMES
    np.testing.assert_array_equal(root["RNA/counts"][:], _VALUES)


def test_make_unique_renames_ids_across_assays(tmp_path: Path) -> None:
    # Uniqueness spans every feature, so an antibody that repeats the ID of an
    # earlier gene is renamed in its own assay.
    directory = tmp_path / "cite"
    directory.mkdir()
    (directory / "matrix.mtx").write_text(
        "%%MatrixMarket matrix coordinate integer general\n3 2 3\n1 1 5\n2 1 1\n3 2 7\n"
    )
    (directory / "features.tsv").write_text(
        "CD4\tCD4\tGene Expression\n"
        "CD8A\tCD8A\tGene Expression\n"
        "CD4\tCD4\tAntibody Capture\n"
    )
    (directory / "barcodes.tsv").write_text("cell-0\ncell-1\n")
    reader = MtxReader(inspect_mtx(directory)[0], duplicate_feature_ids="make_unique")
    try:
        assert reader.feature_ids("RNA") == ["CD4", "CD8A"]
        assert reader.feature_ids("ADT") == ["CD4-1"]
        assert reader.feature_names("ADT") == ["CD4"]
    finally:
        reader.close()


def test_matrix_market_feature_reference_joins_the_ids_of_the_sidecar(
    tmp_path: Path,
) -> None:
    # The reference describes the features whose sidecar ID is GENE, which
    # are both occurrences, and never names a renamed ID.
    directory = _write_mex(
        tmp_path / "mex", _IDS, reference="id,sequence\nGENE,ACGT\nOTHER,TTTT\n"
    )
    reader = MtxReader(inspect_mtx(directory)[0], duplicate_feature_ids="make_unique")
    try:
        columns = dict(reader.get_feature_columns())
    finally:
        reader.close()
    assert list(columns["sequence"]) == ["ACGT", "TTTT", None, "ACGT"]

    renamed = _write_mex(
        tmp_path / "renamed", _IDS, reference="id,sequence\nGENE-2,ACGT\n"
    )
    with pytest.raises(ValueError, match="not matrix features: GENE-2"):
        MtxReader(inspect_mtx(renamed)[0], duplicate_feature_ids="make_unique")


def test_matrix_market_cell_ids_stay_unique_when_feature_ids_are_renamed(
    tmp_path: Path,
) -> None:
    directory = _write_mex(
        tmp_path / "mex", _IDS, barcodes=["cell-0", "cell-0", "cell-2"]
    )
    with pytest.raises(ValueError, match="Cell IDs must contain unique values"):
        MtxReader(inspect_mtx(directory)[0], duplicate_feature_ids="make_unique")


def _write_cellranger(
    path: Path,
    ids: list[str],
    *,
    legacy: bool = False,
    barcodes: list[str] | None = None,
) -> Path:
    matrix = csr_matrix(_VALUES)
    cells = barcodes or [f"cell-{index}" for index in range(_VALUES.shape[0])]
    with h5py.File(path, mode="w") as handle:
        group = handle.create_group("GRCh38" if legacy else "matrix")
        group.create_dataset("data", data=matrix.data.astype(np.int32))
        group.create_dataset("indices", data=matrix.indices.astype(np.int64))
        group.create_dataset("indptr", data=matrix.indptr.astype(np.int64))
        group.create_dataset(
            "barcodes", data=np.array([cell.encode() for cell in cells])
        )
        encoded_ids = np.array([value.encode() for value in ids])
        encoded_names = np.array([value.encode() for value in _NAMES])
        if legacy:
            group.create_dataset("genes", data=encoded_ids)
            group.create_dataset("gene_names", data=encoded_names)
        else:
            features = group.create_group("features")
            features.create_dataset("id", data=encoded_ids)
            features.create_dataset("name", data=encoded_names)
            features.create_dataset(
                "feature_type", data=np.array([b"Gene Expression"] * len(ids))
            )
    return path


@pytest.mark.parametrize("legacy", [False, True], ids=["v3", "v2"])
def test_cellranger_h5_reader_rejects_repeated_feature_ids_by_default(
    tmp_path: Path, legacy: bool
) -> None:
    path = _write_cellranger(tmp_path / "counts.h5", _IDS, legacy=legacy)
    with pytest.raises(
        ValueError, match="Cell Ranger feature IDs must contain unique values"
    ):
        CrH5Reader(str(path))


@pytest.mark.parametrize("legacy", [False, True], ids=["v3", "v2"])
def test_cellranger_h5_reader_renames_repeated_feature_ids_on_request(
    tmp_path: Path, legacy: bool
) -> None:
    path = _write_cellranger(tmp_path / "counts.h5", _IDS, legacy=legacy)
    with _warnings() as messages:
        reader = CrH5Reader(str(path), duplicate_feature_ids="make_unique")
    store = MemoryStore()
    try:
        assert reader.duplicateFeatureIds == "make_unique"
        assert reader.feature_ids() == _RENAMED
        assert reader.feature_ids("RNA") == _RENAMED
        assert reader.feature_names() == _NAMES
        CrToZarr(reader, store, mem_budget="64M", nthreads=1, policy=_POLICY).dump()
    finally:
        reader.close()
    assert sum("Cell Ranger feature IDs repeat 1 values" in m for m in messages) == 1
    root = zarr.open_group(store=store, mode="r")
    assert _feature_ids(root) == _RENAMED
    assert _feature_names(root) == _NAMES
    np.testing.assert_array_equal(root["RNA/counts"][:], _VALUES)


def test_cellranger_h5_barcodes_stay_unique_when_feature_ids_are_renamed(
    tmp_path: Path,
) -> None:
    path = _write_cellranger(
        tmp_path / "counts.h5", _IDS, barcodes=["cell-0", "cell-1", "cell-0"]
    )
    with pytest.raises(
        ValueError, match="Cell Ranger barcodes must contain unique values"
    ):
        CrH5Reader(str(path), duplicate_feature_ids="make_unique")


def _write_sparse_h5ad(
    path: Path,
    ids: list[str],
    *,
    names: list[str] | None = None,
    cells: list[str] | None = None,
) -> Path:
    matrix = csr_matrix(_VALUES)
    with h5py.File(path, mode="w") as handle:
        group = handle.create_group("X")
        group.attrs["encoding-type"] = "csr_matrix"
        group.attrs["encoding-version"] = "0.1.0"
        group.attrs["shape"] = matrix.shape
        group.create_dataset("data", data=matrix.data)
        group.create_dataset("indices", data=matrix.indices.astype(np.int64))
        group.create_dataset("indptr", data=matrix.indptr.astype(np.int64))
        obs = handle.create_group("obs")
        obs.create_dataset(
            "_index",
            data=np.array(
                [
                    cell.encode()
                    for cell in (
                        cells or [f"cell-{i}" for i in range(_VALUES.shape[0])]
                    )
                ]
            ),
        )
        var = handle.create_group("var")
        var.create_dataset("_index", data=np.array([value.encode() for value in ids]))
        if names is not None:
            var.create_dataset(
                "gene_short_name", data=np.array([name.encode() for name in names])
            )
    return path


def _texts(values: np.ndarray) -> list[str]:
    return [
        value.decode() if isinstance(value, bytes) else str(value) for value in values
    ]


def test_h5ad_reader_rejects_repeated_feature_ids_by_default(tmp_path: Path) -> None:
    path = _write_sparse_h5ad(tmp_path / "counts.h5ad", _IDS, names=_NAMES)
    reader = H5adReader(str(path))
    try:
        with pytest.raises(
            ValueError, match="H5AD feature IDs must contain unique values"
        ):
            reader.feat_ids()
        with pytest.raises(ValueError, match="H5AD feature IDs must contain unique"):
            H5adToZarr(reader, MemoryStore(), nthreads=1)
    finally:
        reader.close()


@pytest.mark.parametrize("with_names", [True, False], ids=["names", "ids-as-names"])
def test_h5ad_reader_renames_repeated_feature_ids_on_request(
    tmp_path: Path, with_names: bool
) -> None:
    path = _write_sparse_h5ad(
        tmp_path / "counts.h5ad", _IDS, names=_NAMES if with_names else None
    )
    with _warnings() as messages:
        reader = H5adReader(str(path), duplicate_feature_ids="make_unique")
        store = MemoryStore()
        try:
            assert reader._clone_kwargs()["duplicate_feature_ids"] == "make_unique"
            assert _texts(reader.feat_ids()) == _RENAMED
            # Without a name column the names are the IDs as the file holds
            # them.
            assert _texts(reader.feat_names()) == (_NAMES if with_names else _IDS)
            H5adToZarr(reader, store, nthreads=1, policy=_POLICY).dump()
        finally:
            reader.close()
    # The IDs are resolved once, however often the import reads them.
    assert sum("H5AD feature IDs repeat 1 values" in m for m in messages) == 1
    root = zarr.open_group(store=store, mode="r")
    assert _feature_ids(root) == _RENAMED
    assert _feature_names(root) == (_NAMES if with_names else _IDS)
    np.testing.assert_array_equal(root["RNA/counts"][:], _VALUES)


def test_h5ad_reader_from_inspect_takes_the_duplicate_feature_ids_option(
    tmp_path: Path,
) -> None:
    path = _write_sparse_h5ad(tmp_path / "counts.h5ad", _IDS)
    inspection = inspect_h5ad(str(path))
    reader = H5adReader.from_inspect(inspection, duplicate_feature_ids="make_unique")
    try:
        assert _texts(reader.feat_ids()) == _RENAMED
    finally:
        reader.close()


def test_h5ad_cell_ids_stay_unique_when_feature_ids_are_renamed(
    tmp_path: Path,
) -> None:
    path = _write_sparse_h5ad(
        tmp_path / "counts.h5ad", _IDS, cells=["cell-0", "cell-0", "cell-2"]
    )
    reader = H5adReader(str(path), duplicate_feature_ids="make_unique")
    try:
        with pytest.raises(ValueError, match="H5AD cell IDs must contain unique"):
            reader.cell_ids()
    finally:
        reader.close()


@pytest.mark.parametrize("value", ["unique", "Error", None, True])
def test_readers_reject_an_unknown_duplicate_feature_ids_option(
    tmp_path: Path, value: object
) -> None:
    directory = _write_mex(tmp_path / "mex", _IDS)
    h5 = _write_cellranger(tmp_path / "counts.h5", _IDS)
    h5ad = _write_sparse_h5ad(tmp_path / "counts.h5ad", _IDS)
    message = "duplicate_feature_ids must be 'error' or 'make_unique'"
    with pytest.raises(ValueError, match=message):
        MtxReader(inspect_mtx(directory)[0], duplicate_feature_ids=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=message):
        CrDirReader(str(directory), duplicate_feature_ids=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=message):
        CrH5Reader(str(h5), duplicate_feature_ids=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=message):
        H5adReader(str(h5ad), duplicate_feature_ids=value)  # type: ignore[arg-type]
