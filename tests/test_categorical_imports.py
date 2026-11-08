"""H5AD imports store source categoricals as categorical columns with ``categorical_columns=True``.

Each check compares a categorical import with codes, categories and labels
written out by hand, or with a default import of the same file, whose label
columns every release reads, so an import that stored, typed or fingerprinted
a categorical differently fails.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest
import zarr
from scipy.sparse import csr_matrix

from cytearc.metadata.encoding import (
    missing_placeholder,
    stored_categorical_codes,
    stored_categorical_values,
)
from cytearc.metadata.rows import metadata_column_fingerprint, metadata_missing_mask
from cytearc.readers import H5adReader
from cytearc.storage.categorical import CategoricalColumn
from cytearc.storage.identity import read_dataset_fingerprint
from cytearc.storage.selections import snapshot_run_metadata
from cytearc.writers import H5adToZarr
from tests.storage_helpers import open_store

_TEXT = h5py.string_dtype()
_N_CELLS = 12
_GENES = [f"g{index}" for index in range(8)]
_PROTEINS = [f"p{index}" for index in range(4)]
_CELL_TYPES = [
    "T/NK",
    "B cell",
    "Zellé",
    "中文",
    "an unused category, wider than any label",
]

# Each obs categorical as the file holds it, its codes, categories and order,
# and what a categorical import stores: its codes, -1 in missing rows, its
# categories and their dtype, and the labels that every row reads, with the
# placeholder of their dtype in missing rows, as a default import stores them.
_OBS: dict[str, dict[str, Any]] = {
    "cell_type": {
        "codes": [0, 1, -1, 2, 3, 0, 1, -1, 2, 3, 0, 1],
        "categories": np.array(_CELL_TYPES, dtype=_TEXT),
        "ordered": False,
        "stored_codes": [0, 1, -1, 2, 3, 0, 1, -1, 2, 3, 0, 1],
        "stored_categories": _CELL_TYPES,
        "kind": "U",
        "labels": [
            "T/NK",
            "B cell",
            "",
            "Zellé",
            "中文",
            "T/NK",
            "B cell",
            "",
            "Zellé",
            "中文",
            "T/NK",
            "B cell",
        ],
    },
    "stage": {
        "codes": [0, 1, 2] * 4,
        "categories": np.array(["early", "mid", "late", "never"], dtype=_TEXT),
        "ordered": True,
        "stored_codes": [0, 1, 2] * 4,
        "stored_categories": ["early", "mid", "late", "never"],
        "kind": "U",
        "labels": ["early", "mid", "late"] * 4,
    },
    "clusters": {
        "codes": [0, 1, 2, -1, 0, 1, 2, 0, 1, 2, 0, 1],
        "categories": np.array([3, 1, 2, 7]),
        "ordered": False,
        "stored_codes": [0, 1, 2, -1, 0, 1, 2, 0, 1, 2, 0, 1],
        "stored_categories": [3, 1, 2, 7],
        "kind": "i",
        "labels": [3, 1, 2, 0, 3, 1, 2, 3, 1, 2, 3, 1],
    },
    "level": {
        "codes": [0, 1] * 6,
        "categories": np.array([1.5, 0.5]),
        "ordered": False,
        "stored_codes": [0, 1] * 6,
        "stored_categories": [1.5, 0.5],
        "kind": "f",
        "labels": [1.5, 0.5] * 6,
    },
    "flag": {
        "codes": [1, 0] * 6,
        "categories": np.array([True, False]),
        "ordered": False,
        "stored_codes": [1, 0] * 6,
        "stored_categories": [True, False],
        "kind": "b",
        "labels": [False, True] * 6,
    },
    # No row holds a category, so a default import stores the column as
    # text, and a categorical import its categories.
    "unheld": {
        "codes": [-1] * 12,
        "categories": np.array([5, 60]),
        "ordered": False,
        "stored_codes": [-1] * 12,
        "stored_categories": ["5", "60"],
        "kind": "U",
        "labels": [""] * 12,
    },
    # A code past the last category is missing.
    "past": {
        "codes": [0, 1, 2] * 4,
        "categories": np.array(["a", "b"], dtype=_TEXT),
        "ordered": False,
        "stored_codes": [0, 1, -1] * 4,
        "stored_categories": ["a", "b"],
        "kind": "U",
        "labels": ["a", "b", ""] * 4,
    },
    # AnnData 0.7 kept the categories of integer codes in obs/__categories,
    "legacy_07": {
        "codes": [2, 1, 0, -1] * 3,
        "categories": np.array([b"low", b"mid", b"high"]),
        "ordered": True,
        "stored_codes": [2, 1, 0, -1] * 3,
        "stored_categories": ["low", "mid", "high"],
        "kind": "U",
        "labels": ["high", "mid", "low", ""] * 3,
    },
    # and AnnData 0.6 in uns/<column>_categories.
    "legacy_06": {
        "codes": [1, 0] * 6,
        "categories": np.array([b"d1", b"d2"]),
        "ordered": False,
        "stored_codes": [1, 0] * 6,
        "stored_categories": ["d1", "d2"],
        "kind": "U",
        "labels": ["d2", "d1"] * 6,
    },
}

# Each var categorical as the file holds it.
_VAR: dict[str, tuple[list[int], np.ndarray]] = {
    "feature_types": (
        [0] * 8 + [1] * 4,
        np.array(["Gene Expression", "Antibody Capture"], dtype=_TEXT),
    ),
    "symbol": (
        [0, 1] * 4 + [2, 2, -1, 3],
        np.array(["A", "B", "a protein symbol", "中"], dtype=_TEXT),
    ),
    "chromosome": ([0, 1, 0, 1, 0, 1, 2, 2] + [-1] * 4, np.array([1, 2, 23])),
}

# What each assay of the split by feature_types stores for each var
# categorical: its codes, every category, and the labels of its features.
_SPLIT: dict[tuple[str, str], tuple[list[int], list[Any], list[Any]]] = {
    ("RNA", "feature_types"): (
        [0] * 8,
        ["Gene Expression", "Antibody Capture"],
        ["Gene Expression"] * 8,
    ),
    ("ADT", "feature_types"): (
        [1] * 4,
        ["Gene Expression", "Antibody Capture"],
        ["Antibody Capture"] * 4,
    ),
    ("RNA", "symbol"): (
        [0, 1] * 4,
        ["A", "B", "a protein symbol", "中"],
        ["A", "B"] * 4,
    ),
    ("ADT", "symbol"): (
        [2, 2, -1, 3],
        ["A", "B", "a protein symbol", "中"],
        ["a protein symbol", "a protein symbol", "", "中"],
    ),
    ("RNA", "chromosome"): (
        [0, 1, 0, 1, 0, 1, 2, 2],
        [1, 2, 23],
        [1, 2, 1, 2, 1, 2, 23, 23],
    ),
    # No protein holds a chromosome, so the labels and categories of the
    # proteins' column are text.
    ("ADT", "chromosome"): ([-1] * 4, ["1", "2", "23"], [""] * 4),
}


def _categorical(
    group: h5py.Group,
    name: str,
    codes: np.ndarray,
    categories: np.ndarray,
    *,
    ordered: bool = False,
) -> None:
    """Write a categorical column as AnnData 0.8 and later write one."""
    column = group.create_group(name)
    column.attrs["encoding-type"] = "categorical"
    column.attrs["encoding-version"] = "0.2.0"
    column.attrs["ordered"] = ordered
    column.create_dataset("codes", data=codes)
    column.create_dataset("categories", data=categories)


def _write_counts(h5: h5py.File, n_cells: int, n_features: int) -> None:
    counts = csr_matrix(
        (np.arange(n_cells * n_features).reshape(n_cells, n_features) % 4 + 1).astype(
            np.float32
        )
    )
    matrix = h5.create_group("X")
    matrix.attrs["encoding-type"] = "csr_matrix"
    matrix.attrs["shape"] = counts.shape
    matrix.create_dataset("data", data=counts.data)
    matrix.create_dataset("indices", data=counts.indices)
    matrix.create_dataset("indptr", data=counts.indptr)
    h5.create_group("obsm")


def _write_index(group: h5py.Group, names: list[str]) -> None:
    group.attrs["_index"] = "_index"
    group.create_dataset("_index", data=np.array(names, dtype=_TEXT))


def _write_source(path: Path) -> None:
    """Write an H5AD file whose obs and var hold categoricals of every kind and encoding."""
    with h5py.File(path, "w") as h5:
        _write_counts(h5, _N_CELLS, len(_GENES) + len(_PROTEINS))
        obs = h5.create_group("obs")
        _write_index(obs, [f"cell-{index}" for index in range(_N_CELLS)])
        for name, column in _OBS.items():
            if not name.startswith("legacy"):
                _categorical(
                    obs,
                    name,
                    np.array(column["codes"], dtype=np.int8),
                    column["categories"],
                    ordered=column["ordered"],
                )
        for name in ("legacy_07", "legacy_06"):
            obs.create_dataset(name, data=np.array(_OBS[name]["codes"], dtype=np.int8))
        kept = obs.create_group("__categories").create_dataset(
            "legacy_07", data=_OBS["legacy_07"]["categories"]
        )
        kept.attrs["ordered"] = True
        h5.create_group("uns").create_dataset(
            "legacy_06_categories", data=_OBS["legacy_06"]["categories"]
        )
        obs.create_dataset(
            "plain",
            data=np.array([f"text {index}" for index in range(_N_CELLS)], dtype=_TEXT),
        )
        obs.attrs["column-order"] = np.array([*_OBS, "plain"], dtype=_TEXT)
        var = h5.create_group("var")
        _write_index(var, _GENES + _PROTEINS)
        for name, (codes, categories) in _VAR.items():
            _categorical(var, name, np.array(codes, dtype=np.int8), categories)
        var.attrs["column-order"] = np.array(list(_VAR), dtype=_TEXT)


def _import(source: Path, destination: Path, **options: Any) -> None:
    reader = H5adReader(str(source))
    try:
        H5adToZarr(reader, str(destination), nthreads=1, **options).dump()
    finally:
        reader.close()


def _tables_digest(path: Path) -> str:
    """Digest every array of the cell and feature tables: values, dtype, shape, chunks and attributes."""
    root = zarr.open_group(str(path), mode="r")
    tables = [
        "cellData",
        *(
            f"{name}/featureData"
            for name in sorted(root.group_keys())
            if "featureData" in root[name]
        ),
    ]
    digest = hashlib.sha256()
    for table in tables:
        group = root[table]
        for key in sorted(group.array_keys()):
            array = group[key]
            values = np.ascontiguousarray(array[:])
            digest.update(
                json.dumps(
                    [
                        table,
                        key,
                        values.dtype.str,
                        list(array.shape),
                        list(array.chunks),
                        array.attrs.asdict(),
                    ],
                    sort_keys=True,
                ).encode()
            )
            digest.update(values.tobytes())
    return digest.hexdigest()


@pytest.fixture(scope="module")
def source(tmp_path_factory) -> Path:
    """The source file, whose obs and var hold categoricals of every kind."""
    path = tmp_path_factory.mktemp("categorical_source") / "source.h5ad"
    _write_source(path)
    return path


@pytest.fixture(scope="module")
def imports(source: Path, tmp_path_factory) -> dict[str, Any]:
    """A default and a categorical import of the source file, each split by feature_types."""
    folder = tmp_path_factory.mktemp("categorical_imports")
    found: dict[str, Any] = {}
    for layout, options in (
        ("text", {}),
        ("categorical", {"categorical_columns": True}),
    ):
        path = folder / f"{layout}.zarr"
        _import(source, path, assay_split_key="feature_types", **options)
        # As the import wrote the tables, before the first open prepares them.
        found[f"{layout} digest"] = _tables_digest(path)
        found[layout] = open_store(path)
    return found


def _missing_rows(table: Any, name: str) -> list[bool] | None:
    mask = metadata_missing_mask(table, name)
    return None if mask is None else np.asarray(mask[:]).tolist()


def test_source_categoricals_store_codes_beside_every_category(imports) -> None:
    cells = imports["categorical"].cells
    group = imports["categorical"].zw["cellData"]

    for name, column in _OBS.items():
        array = cells._get_array(name)
        assert isinstance(array, CategoricalColumn), name
        # A few categories take int8 codes, -1 in each missing row.
        assert group[name].dtype == np.dtype(np.int8), name
        assert group[name][:].tolist() == column["stored_codes"], name
        assert array.categories.tolist() == column["stored_categories"], name
        assert array.dtype.kind == column["kind"], name
        assert array.ordered is column["ordered"], name
        flagged = [code == -1 for code in column["stored_codes"]]
        assert _missing_rows(cells, name) == (flagged if any(flagged) else None), name
        categorical = cells.fetch_categorical(name, key=None)
        assert categorical.categories.tolist() == column["stored_categories"], name
    # Numbers keep the type that a column of their labels stores.
    assert [cells._get_array(name).dtype for name in ("clusters", "level", "flag")] == [
        np.dtype(np.int64),
        np.dtype(np.float64),
        np.dtype(bool),
    ]
    # A column of values keeps its layout.
    assert isinstance(cells._get_array("plain"), zarr.Array)
    assert "__cytearc_categories__plain" not in group


def test_categorical_imports_read_the_labels_of_a_default_import(imports) -> None:
    text, categorical = imports["text"].cells, imports["categorical"].cells
    for name, column in _OBS.items():
        labels = categorical.fetch_all(name)
        stored = text.fetch_all(name)
        assert isinstance(text._get_array(name), zarr.Array), name
        assert labels.tolist() == column["labels"] == stored.tolist(), name
        assert labels.dtype.kind == stored.dtype.kind == column["kind"], name
        assert _missing_rows(categorical, name) == _missing_rows(text, name), name
        assert metadata_column_fingerprint(categorical, name) == (
            metadata_column_fingerprint(text, name)
        ), name
        # The attributes that plots, merges and exports read, beside the
        # link to the categories.
        attributes = dict(categorical._get_array(name).attrs)
        assert attributes.pop("categories") == f"__cytearc_categories__{name}"
        assert attributes == dict(text._get_array(name).attrs), name
    assert categorical._get_array("stage").attrs["levels"] == [
        "early",
        "mid",
        "late",
        "never",
    ]
    columns = [*_OBS, "plain"]
    pd.testing.assert_frame_equal(
        categorical.to_pandas_dataframe(columns), text.to_pandas_dataframe(columns)
    )


def test_each_assay_of_a_split_keeps_its_own_categories(imports) -> None:
    for (assay, name), (codes, categories, labels) in _SPLIT.items():
        text = getattr(imports["text"], assay).feats
        categorical = getattr(imports["categorical"], assay).feats
        array = categorical._get_array(name)
        assert isinstance(array, CategoricalColumn), (assay, name)
        assert array.codes[:].tolist() == codes, (assay, name)
        assert array.categories.tolist() == categories, (assay, name)
        assert categorical.fetch_all(name).tolist() == labels, (assay, name)
        assert text.fetch_all(name).tolist() == labels, (assay, name)
        # Labels stored as values take the width of the assay's own labels.
        assert array.stored_labels_dtype() == text._get_array(name).dtype, (
            assay,
            name,
        )
        assert metadata_column_fingerprint(categorical, name) == (
            metadata_column_fingerprint(text, name)
        ), (assay, name)
    widths = {
        assay: getattr(imports["categorical"], assay)
        .feats._get_array("symbol")
        .stored_labels_dtype()
        for assay in ("RNA", "ADT")
    }
    assert widths == {"RNA": np.dtype("U1"), "ADT": np.dtype("U16")}


def test_snapshots_and_dataset_identities_match_a_default_import(imports) -> None:
    found = {}
    for layout in ("text", "categorical"):
        store = imports[layout]
        snapshots = {
            "cellData": snapshot_run_metadata(
                store.zw,
                table_path="cellData",
                id_column="ids",
                columns=list(_OBS),
                axis="cell",
            )
        }
        for assay in ("RNA", "ADT"):
            snapshots[assay] = snapshot_run_metadata(
                store.zw,
                table_path=f"{assay}/featureData",
                id_column="ids",
                columns=list(_VAR),
                axis="feature",
                assay=assay,
            )
        found[layout] = (
            {
                table: store.artifacts.inspect(ref).inputs["column_fingerprints"]
                for table, ref in snapshots.items()
            },
            {
                assay: read_dataset_fingerprint(store.zw[assay])
                for assay in ("RNA", "ADT")
            },
        )
        cell_snapshot = store.artifacts.load(snapshots["cellData"])
        assert cell_snapshot["cell_type"][:].tolist() == _OBS["cell_type"]["labels"]

    assert found["categorical"] == found["text"]


def test_exports_read_each_categorical_from_either_import(imports) -> None:
    exported = {
        layout: imports[layout].to_anndata() for layout in ("text", "categorical")
    }

    obs = exported["categorical"].obs
    pd.testing.assert_frame_equal(obs, exported["text"].obs)
    pd.testing.assert_frame_equal(exported["categorical"].var, exported["text"].var)
    assert obs["cell_type"].cat.categories.tolist() == _CELL_TYPES
    assert obs["stage"].cat.ordered
    assert obs["legacy_07"].cat.ordered


def test_subsets_and_merges_read_each_categorical_from_either_import(
    imports, tmp_path: Path
) -> None:
    from cytearc.merge import DataStoreMerge
    from cytearc.writers import SubsetZarr

    text, categorical = imports["text"], imports["categorical"]
    rows = np.array([0, 3, 4, 9])
    SubsetZarr(
        str(tmp_path / "subset.zarr"), [categorical.RNA], cell_idx=rows, nthreads=1
    ).dump()
    DataStoreMerge(
        [text, categorical],
        str(tmp_path / "merged.zarr"),
        ["text", "categorical"],
        nthreads=1,
    ).dump()

    subset = open_store(tmp_path / "subset.zarr")
    merged = open_store(tmp_path / "merged.zarr")
    merged_rows = {cell: row for row, cell in enumerate(merged.cells.fetch_all("ids"))}
    for name, column in _OBS.items():
        # A subset copies the codes of its rows beside every category.
        array = subset.cells._get_array(name)
        assert isinstance(array, CategoricalColumn), name
        assert array.codes[:].tolist() == [column["stored_codes"][row] for row in rows]
        assert array.categories.tolist() == column["stored_categories"], name
        source = categorical.cells._get_array(name).attrs
        for key in ("levels", "ordered", "display"):
            assert array.attrs.get(key) == source.get(key), (name, key)
        # A merge writes the labels of either import as values.
        values = merged.cells.fetch_all(f"orig_{name}")
        for layout in ("text", "categorical"):
            order = [
                merged_rows[f"{layout}__cell-{index}"] for index in range(_N_CELLS)
            ]
            assert values[order].tolist() == column["labels"], (name, layout)
        assert merged.cells._get_array(f"orig_{name}").attrs.get("levels") == (
            text.cells._get_array(name).attrs.get("levels")
        )


def test_a_default_import_stores_the_tables_of_earlier_releases(
    source: Path, tmp_path: Path
) -> None:
    _import(source, tmp_path / "default.zarr", assay_split_key="feature_types")

    # The digest of the tables that the import wrote from the source file
    # before it could write categorical columns, which every release reads.
    assert _tables_digest(tmp_path / "default.zarr") == (
        "5cdda353e73c93b5bb498d0ecfb3d77e1eb5c1e7ffde1cb139c08c8d6d6276a1"
    )


def test_an_import_without_categorical_columns_stores_label_columns(
    source: Path, imports, tmp_path: Path
) -> None:
    _import(
        source,
        tmp_path / "explicit.zarr",
        assay_split_key="feature_types",
        categorical_columns=False,
    )

    assert _tables_digest(tmp_path / "explicit.zarr") == imports["text digest"]
    assert imports["categorical digest"] != imports["text digest"]


def _write_compound_source(path: Path) -> None:
    """Write tables as AnnData 0.6 wrote them: compound datasets, with categories in uns."""
    with h5py.File(path, "w") as h5:
        _write_counts(h5, _N_CELLS, 4)
        obs = np.zeros(
            _N_CELLS, dtype=[("index", "S8"), ("batch", "i1"), ("score", "f8")]
        )
        obs["index"] = [f"cell-{index}".encode() for index in range(_N_CELLS)]
        obs["batch"] = [0, 1, 2, -1] * 3
        obs["score"] = np.arange(_N_CELLS) / 2
        h5.create_dataset("obs", data=obs)
        var = np.zeros(4, dtype=[("index", "S4"), ("group", "i1")])
        var["index"] = [b"g0", b"g1", b"g2", b"g3"]
        var["group"] = [1, 0, 1, 0]
        h5.create_dataset("var", data=var)
        uns = h5.create_group("uns")
        uns.create_dataset("batch_categories", data=np.array([b"b1", b"b2", b"b3"]))
        uns.create_dataset("group_categories", data=np.array([b"g", b"h"]))


def test_compound_tables_of_anndata_0_6_import_their_categoricals(
    tmp_path: Path,
) -> None:
    source = tmp_path / "compound.h5ad"
    _write_compound_source(source)
    stores = {}
    for layout, options in (
        ("text", {}),
        ("categorical", {"categorical_columns": True}),
    ):
        _import(source, tmp_path / f"{layout}.zarr", **options)
        stores[layout] = open_store(tmp_path / f"{layout}.zarr")

    for axis, name, codes, categories, labels in (
        (
            "cells",
            "batch",
            [0, 1, 2, -1] * 3,
            ["b1", "b2", "b3"],
            ["b1", "b2", "b3", ""] * 3,
        ),
        ("features", "group", [1, 0, 1, 0], ["g", "h"], ["h", "g", "h", "g"]),
    ):
        if axis == "cells":
            text, categorical = stores["text"].cells, stores["categorical"].cells
        else:
            text, categorical = (
                stores["text"].RNA.feats,
                stores["categorical"].RNA.feats,
            )
        array = categorical._get_array(name)
        assert isinstance(array, CategoricalColumn), name
        assert array.codes[:].tolist() == codes, name
        assert array.categories.tolist() == categories, name
        assert (
            categorical.fetch_all(name).tolist()
            == labels
            == text.fetch_all(name).tolist()
        )
        assert metadata_column_fingerprint(categorical, name) == (
            metadata_column_fingerprint(text, name)
        )
    # The numbers of a compound table are values.
    assert isinstance(stores["categorical"].cells._get_array("score"), zarr.Array)


def _size_source(path: Path, n_cells: int) -> None:
    """Write cells with one categorical of 300 labels of 20 characters, 2% of them missing."""
    rng = np.random.default_rng(7)
    codes = rng.integers(0, 300, n_cells).astype(np.int16)
    codes[rng.random(n_cells) < 0.02] = -1
    with h5py.File(path, "w") as h5:
        counts = csr_matrix(
            (
                np.ones(n_cells, np.float32),
                (np.arange(n_cells), np.zeros(n_cells, int)),
            ),
            shape=(n_cells, 2),
        )
        matrix = h5.create_group("X")
        matrix.attrs["encoding-type"] = "csr_matrix"
        matrix.attrs["shape"] = counts.shape
        matrix.create_dataset("data", data=counts.data)
        matrix.create_dataset("indices", data=counts.indices)
        matrix.create_dataset("indptr", data=counts.indptr)
        obs = h5.create_group("obs")
        _write_index(obs, [f"cell-{index}" for index in range(n_cells)])
        _categorical(
            obs,
            "cell_type",
            codes,
            np.array([f"cell type {index:010d}" for index in range(300)], dtype=_TEXT),
        )
        obs.attrs["column-order"] = np.array(["cell_type"], dtype=_TEXT)
        var = h5.create_group("var")
        _write_index(var, ["g0", "g1"])


def _column_bytes(path: Path, name: str) -> tuple[int, int]:
    """The bytes of a cell column's values, its mask and categories included, uncompressed and in stored chunks.

    The chunks leave out each array's metadata document, which holds the
    same ``levels`` and ``display`` attributes in either layout.
    """
    cells = zarr.open_group(str(path), mode="r")["cellData"]
    uncompressed = 0
    stored = 0
    for prefix in ("", "__cytearc_missing__", "__cytearc_categories__"):
        key = f"{prefix}{name}"
        if key not in cells:
            continue
        uncompressed += int(cells[key].nbytes)
        stored += sum(
            file.stat().st_size
            for file in (path / "cellData" / key).rglob("*")
            if file.is_file() and file.name != "zarr.json"
        )
    return uncompressed, stored


def test_categorical_imports_store_fewer_bytes(tmp_path: Path) -> None:
    source = tmp_path / "cells.h5ad"
    _size_source(source, 20_000)
    _import(source, tmp_path / "text.zarr")
    _import(source, tmp_path / "categorical.zarr", categorical_columns=True)

    text = _column_bytes(tmp_path / "text.zarr", "cell_type")
    categorical = _column_bytes(tmp_path / "categorical.zarr", "cell_type")
    # Text holds 80 bytes a row, int16 codes 2 bytes beside 300 categories.
    assert text[0] == 20_000 * 81
    assert categorical[0] == 20_000 * 3 + 300 * 80
    assert categorical[1] * 3 < text[1]
    cells = {
        layout: open_store(tmp_path / f"{layout}.zarr").cells
        for layout in ("text", "categorical")
    }
    assert cells["categorical"].fetch_all("cell_type").tolist() == (
        cells["text"].fetch_all("cell_type").tolist()
    )


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_the_option_is_validated_before_anything_is_written(
    source: Path, tmp_path: Path, value: object
) -> None:
    destination = tmp_path / "refused.zarr"
    reader = H5adReader(str(source))
    try:
        with pytest.raises(TypeError, match="categorical_columns must be a boolean"):
            H5adToZarr(reader, str(destination), nthreads=1, categorical_columns=value)
    finally:
        reader.close()

    assert not destination.exists()


def test_categorical_columns_take_the_storage_profile_of_the_import(
    source: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "cloud.zarr"
    _import(
        source,
        destination,
        assay_split_key="feature_types",
        categorical_columns=True,
        profile="cloud",
    )

    cells = zarr.open_group(str(destination), mode="r")["cellData"]
    expected = cells["plain"].compressors
    assert type(expected[0]).__name__ == "ZstdCodec"
    for key in (
        "cell_type",
        "__cytearc_categories__cell_type",
        "__cytearc_missing__cell_type",
    ):
        assert cells[key].compressors == expected, key


def _single_categorical_source(path: Path, categories: np.ndarray) -> None:
    with h5py.File(path, "w") as h5:
        _write_counts(h5, 4, 2)
        obs = h5.create_group("obs")
        _write_index(obs, [f"cell-{index}" for index in range(4)])
        _categorical(obs, "label", np.array([0, 1, 1, 0], dtype=np.int8), categories)
        obs.attrs["column-order"] = np.array(["label"], dtype=_TEXT)
        var = h5.create_group("var")
        _write_index(var, ["g0", "g1"])


@pytest.mark.parametrize(
    ("categories", "message"),
    [
        (np.array([0.5, np.nan]), "NaN among its categories"),
        (
            np.array(["a", "a", "b"], dtype=_TEXT),
            "categories that read alike once stored, .* 'a'",
        ),
    ],
)
def test_categories_that_a_categorical_column_cannot_hold_are_refused(
    tmp_path: Path, categories: np.ndarray, message: str
) -> None:
    source = tmp_path / "source.h5ad"
    _single_categorical_source(source, categories)
    # A default import stores their labels.
    _import(source, tmp_path / "text.zarr")

    with pytest.raises(ValueError, match=f"Column 'label' has {message}"):
        _import(source, tmp_path / "categorical.zarr", categorical_columns=True)


# The codes and categories that a categorical column stores.

_CASES: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "text": (
        np.array([1, -1, 0, 1, 2], dtype=np.int8),
        np.array([b"x", "Zellé".encode(), b"an unused category"], dtype=object),
    ),
    "str": (np.array([0, 1, 0]), np.array(["中文", "b"], dtype=object)),
    "fixed_width_bytes": (np.array([0, 1, -1]), np.array([b"ab", b"c"], dtype="S2")),
    "fixed_width_text": (np.array([1, 1]), np.array(["x", "a longer one"])),
    "integers": (np.array([0, 2, -1, 1]), np.array([3, 1, 4])),
    "unsigned_integers": (np.array([1, 0]), np.array([7, 9], dtype=np.uint8)),
    "floats": (np.array([1, 0, -1]), np.array([0.5, -1.25])),
    "booleans": (np.array([1, 0, 1]), np.array([False, True])),
    "integers_none_held": (np.array([-1, -1, 5]), np.array([3, 10])),
    "codes_past_the_categories": (
        np.array([0, 5, 1, -3], dtype=np.int64),
        np.array(["a", "b"]),
    ),
    "unsigned_codes": (np.array([0, 1, 255], dtype=np.uint8), np.array(["a", "b"])),
    "no_categories": (np.array([-1, 0]), np.array([], dtype=object)),
    "no_rows": (np.array([], dtype=np.int8), np.array(["a", "b"])),
}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_stored_codes_and_categories_read_the_stored_labels(name: str) -> None:
    source_codes, source_categories = _CASES[name]
    codes, categories = stored_categorical_codes(
        source_codes, source_categories, name="column"
    )
    labels, missing = stored_categorical_values(
        source_codes, source_categories, name="column"
    )

    assert codes.dtype == np.dtype(np.int64)
    assert (codes == -1).tolist() == missing.tolist()
    assert codes[~missing].tolist() == source_codes[~missing].tolist()
    assert len(categories) == len(source_categories)
    # Code -1 takes the placeholder at the end, as a categorical column reads it.
    decoded = np.append(categories, missing_placeholder(categories.dtype))[codes]
    assert decoded.dtype.kind == labels.dtype.kind
    assert decoded.astype(labels.dtype).tobytes() == labels.tobytes()


def test_stored_codes_and_categories_written_out_by_hand() -> None:
    codes, categories = stored_categorical_codes(*_CASES["text"], name="c")
    assert codes.tolist() == [1, -1, 0, 1, 2]
    assert categories.tolist() == ["x", "Zellé", "an unused category"]
    assert categories.dtype == np.dtype("U18")

    codes, categories = stored_categorical_codes(*_CASES["integers"], name="c")
    assert categories.dtype == np.dtype(np.int64)
    assert categories.tolist() == [3, 1, 4]

    # No row holds a category, so the labels, and the categories, are text.
    codes, categories = stored_categorical_codes(
        *_CASES["integers_none_held"], name="c"
    )
    assert codes.tolist() == [-1, -1, -1]
    assert categories.tolist() == ["3", "10"]

    codes, categories = stored_categorical_codes(
        *_CASES["codes_past_the_categories"], name="c"
    )
    assert codes.tolist() == [0, -1, 1, -1]
    assert categories.tolist() == ["a", "b"]


@pytest.mark.parametrize(
    ("codes", "categories", "error", "message"),
    [
        (
            np.array([0.0, 1.0]),
            np.array(["a", "b"]),
            TypeError,
            "integer category codes",
        ),
        (
            np.array([0, 1]),
            np.array([["a", "b"]]),
            ValueError,
            "one category per position",
        ),
        (
            np.array([0, -1]),
            np.array([1, "x"], dtype=object),
            ValueError,
            "more than one type: the ones that rows hold are stored as int64",
        ),
        (
            np.array([0, 1]),
            np.array([0.5, np.nan]),
            ValueError,
            "NaN among its categories",
        ),
        (np.array([0, 1]), np.array(["a", "a"]), ValueError, "read alike once stored"),
        (
            np.array([0, 1]),
            np.array([b"b", "b"], dtype=object),
            ValueError,
            "read alike once stored",
        ),
        (
            np.array([0]),
            np.array([1, 2**70], dtype=object),
            ValueError,
            "outside the int64 range",
        ),
    ],
)
def test_categories_that_cannot_be_stored_as_categories_are_refused(
    codes: np.ndarray, categories: np.ndarray, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=f"Column 'c' .*{message}"):
        stored_categorical_codes(codes, categories, name="c")
