"""H5AD imports store source categoricals as categorical columns by default.

Each check compares a categorical import with codes, categories and labels
written out by hand, or with an import of the same file with
``categorical_columns=False``, whose label columns every release reads, so an
import that stored, typed or fingerprinted a categorical differently fails.
"""

import hashlib
import json
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
from scipy.sparse import csr_matrix

from cytearc.metadata.encoding import (
    UnstorableCategoriesError,
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
from tests.test_categorical_columns import _opens_before_categorical_columns

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
# placeholder of their dtype in missing rows, as an import of label columns
# stores them.
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
    # No row holds a category, so an import of label columns stores the
    # column as text, and a categorical import its categories.
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
        H5adToZarr(
            reader, str(destination), nthreads=1, **options, shuffle_cells=False
        ).dump()
    finally:
        reader.close()


@contextmanager
def _warnings() -> Iterator[list[str]]:
    """Collect the warnings logged inside the block."""
    messages: list[str] = []
    handler = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        yield messages
    finally:
        logger.remove(handler)


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
    """An import of label columns and a default import of the source file, each split by feature_types."""
    folder = tmp_path_factory.mktemp("categorical_imports")
    found: dict[str, Any] = {}
    for layout, options in (
        ("text", {"categorical_columns": False}),
        ("categorical", {}),
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


def test_categorical_imports_read_the_labels_of_an_import_of_label_columns(
    imports,
) -> None:
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


def test_snapshots_and_dataset_identities_match_an_import_of_label_columns(
    imports,
) -> None:
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
        str(tmp_path / "subset.zarr"),
        [categorical.RNA],
        cell_ids=categorical.cells.fetch_all("ids")[rows],
        nthreads=1,
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
        # Four categories over the subset's four cells take a reserved entry.
        padded = subset.zw[f"cellData/__cytearc_categories__{name}"].attrs.get("padded")
        assert padded is (True if len(column["stored_categories"]) == 4 else None)
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


def test_categories_that_number_the_rows_take_a_reserved_entry(
    imports, tmp_path: Path
) -> None:
    # The proteins' symbol column has four categories over four proteins.
    text, categorical = imports["text"].zw, imports["categorical"].zw
    stored = categorical["ADT/featureData/__cytearc_categories__symbol"]
    assert stored[:].tolist() == ["A", "B", "a protein symbol", "中", ""]
    assert stored.attrs["padded"] is True
    genes = categorical["RNA/featureData/__cytearc_categories__symbol"]
    assert "padded" not in genes.attrs
    # A release before categorical columns refuses the table rather than read
    # its codes as values, and opens the table of label columns.
    assert not _opens_before_categorical_columns(categorical["ADT/featureData"])
    assert _opens_before_categorical_columns(text["ADT/featureData"])

    # Four cells of a categorical with four categories, two of them unused.
    source = tmp_path / "four.h5ad"
    _single_categorical_source(source, np.array(["a", "b", "c", "d"], dtype=_TEXT))
    stores = {}
    for layout, options in (
        ("text", {"categorical_columns": False}),
        ("categorical", {}),
    ):
        _import(source, tmp_path / f"{layout}.zarr", **options)
        stores[layout] = open_store(tmp_path / f"{layout}.zarr")

    cells = {layout: store.cells for layout, store in stores.items()}
    group = stores["categorical"].zw["cellData"]
    assert group["__cytearc_categories__label"][:].tolist() == ["a", "b", "c", "d", ""]
    assert cells["categorical"].fetch_all("label").tolist() == ["a", "b", "b", "a"]
    assert cells["text"].fetch_all("label").tolist() == ["a", "b", "b", "a"]
    assert cells["categorical"].fetch_categorical(
        "label", key=None
    ).categories.tolist() == ["a", "b", "c", "d"]
    assert metadata_column_fingerprint(cells["categorical"], "label") == (
        metadata_column_fingerprint(cells["text"], "label")
    )
    assert not _opens_before_categorical_columns(group)
    assert _opens_before_categorical_columns(stores["text"].zw["cellData"])


def test_an_import_without_categorical_columns_stores_the_tables_of_earlier_releases(
    imports,
) -> None:
    # The digest of the tables that the import wrote from the source file
    # before it could write categorical columns, which every release reads.
    assert imports["text digest"] == (
        "5cdda353e73c93b5bb498d0ecfb3d77e1eb5c1e7ffde1cb139c08c8d6d6276a1"
    )


def test_a_default_import_stores_what_categorical_columns_true_stores(
    source: Path, imports, tmp_path: Path
) -> None:
    with _warnings() as messages:
        _import(
            source,
            tmp_path / "explicit.zarr",
            assay_split_key="feature_types",
            categorical_columns=True,
        )

    assert _tables_digest(tmp_path / "explicit.zarr") == imports["categorical digest"]
    assert imports["categorical digest"] != imports["text digest"]
    # Every categorical of the file fits the layout, so nothing falls back.
    assert not [message for message in messages if "categorical" in message]


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
        ("text", {"categorical_columns": False}),
        ("categorical", {}),
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
    _import(source, tmp_path / "text.zarr", categorical_columns=False)
    _import(source, tmp_path / "categorical.zarr")

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


# Categoricals that a categorical column cannot store: their codes, their
# categories, the labels that their rows read, and the reason that the
# warning of an import gives.
_FALLBACKS: dict[str, tuple[list[int], np.ndarray, list[Any], str]] = {
    "many": (
        [0, 1, 10_000, -1],
        np.array([f"c{index:05d}" for index in range(10_001)], dtype=_TEXT),
        ["c00000", "c00001", "c10000", ""],
        "Column 'many' has 10,001 categories, more than the 10,000 that an "
        "import stores as a categorical column",
    ),
    "nan": (
        [0, 2, 2, 0],
        np.array([0.5, np.nan, 1.5]),
        [0.5, 1.5, 1.5, 0.5],
        "Column 'nan' has NaN among its categories; a categorical column marks "
        "a missing row with code -1 and cannot store NaN as a category",
    ),
    "alike": (
        [0, 2, 2, 0],
        np.array(["a", "a", "b"], dtype=_TEXT),
        ["a", "b", "b", "a"],
        "Column 'alike' has categories that read alike once stored, as text or "
        "numbers, so a categorical column cannot tell them apart, such as 'a'",
    ),
    # No row holds the category beyond int64, so the rows' labels are int64.
    "beyond_int64": (
        [0, 0, -1, 0],
        np.array([7, 2**63], dtype=np.uint64),
        [7, 7, 0, 7],
        "Column 'beyond_int64' has categories that cannot all be stored as "
        "int64, the type of the labels that its rows hold",
    ),
}
# Categoricals beside them that a categorical column stores: its codes,
# categories and labels.
_KEPT: dict[str, tuple[list[int], np.ndarray, list[Any]]] = {
    "kept": ([1, 0, 0, 1], np.array(["x", "y"], dtype=_TEXT), ["y", "x", "x", "y"]),
    # As many categories as an import stores as a categorical column.
    "at_the_limit": (
        [0, 9_999, 9_999, -1],
        np.array([f"c{index:04d}" for index in range(10_000)], dtype=_TEXT),
        ["c0000", "c9999", "c9999", ""],
    ),
}


def _fallback_source(path: Path) -> None:
    """Write four cells and two genes with the categoricals of both tables."""
    with h5py.File(path, "w") as h5:
        _write_counts(h5, 4, 2)
        obs = h5.create_group("obs")
        _write_index(obs, [f"cell-{index}" for index in range(4)])
        columns = {**_FALLBACKS, **_KEPT}
        for name, (codes, categories, *_) in columns.items():
            _categorical(obs, name, np.array(codes, dtype=np.int16), categories)
        obs.attrs["column-order"] = np.array(list(columns), dtype=_TEXT)
        var = h5.create_group("var")
        _write_index(var, ["g0", "g1"])
        # The feature names that the reader looks for, so it warns of nothing.
        var.create_dataset("gene_short_name", data=np.array(["G0", "G1"], dtype=_TEXT))
        _categorical(
            var, "symbol", np.array([0, 2], dtype=np.int8), np.array([1.0, np.nan, 2.0])
        )
        var.attrs["column-order"] = np.array(["gene_short_name", "symbol"], dtype=_TEXT)


@pytest.fixture(scope="module")
def fallback_source(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("fallback_source") / "source.h5ad"
    _fallback_source(path)
    return path


@pytest.fixture(scope="module")
def label_import(fallback_source: Path, tmp_path_factory) -> tuple[Path, list[str]]:
    """An import of the fallback source with categorical_columns=False, and its warnings."""
    destination = tmp_path_factory.mktemp("label_import") / "labels.zarr"
    with _warnings() as messages:
        _import(fallback_source, destination, categorical_columns=False)
    return destination, messages


def test_an_import_of_label_columns_stores_every_categorical_without_warnings(
    label_import,
) -> None:
    destination, messages = label_import
    assert messages == []
    cells = zarr.open_group(str(destination), mode="r")["cellData"]
    for name in (*_FALLBACKS, *_KEPT):
        assert "categories" not in cells[name].attrs, name
        assert f"__cytearc_categories__{name}" not in cells, name


@pytest.mark.parametrize(
    "options",
    [{}, {"categorical_columns": True}],
    ids=["default", "categorical_columns_true"],
)
def test_categoricals_that_a_categorical_column_cannot_store_import_as_labels(
    fallback_source: Path, label_import, tmp_path: Path, options: dict[str, Any]
) -> None:
    with _warnings() as messages:
        _import(fallback_source, tmp_path / "store.zarr", **options)

    # Each warning names the column, its table and the reason.
    assert sorted(messages) == sorted(
        [
            *(
                f"Stored cell metadata column {name!r} as a column of its labels, "
                f"not as a categorical column. {reason}"
                for name, (*_, reason) in _FALLBACKS.items()
            ),
            "Stored RNA feature metadata column 'symbol' as a column of its labels, "
            "not as a categorical column. Column 'symbol' has NaN among its "
            "categories; a categorical column marks a missing row with code -1 "
            "and cannot store NaN as a category",
        ]
    )
    store = open_store(tmp_path / "store.zarr")
    labels = zarr.open_group(str(label_import[0]), mode="r")
    stored = zarr.open_group(str(tmp_path / "store.zarr"), mode="r")
    tables = [("cellData", name) for name in _FALLBACKS] + [
        ("RNA/featureData", "symbol")
    ]
    for table, name in tables:
        # The column of an import of label columns, array for array:
        # values, dtype, chunks, attributes and missing mask.
        assert f"__cytearc_categories__{name}" not in stored[table], name
        for key in (name, f"__cytearc_missing__{name}"):
            assert (key in stored[table]) is (key in labels[table]), key
            if key not in labels[table]:
                continue
            assert stored[table][key].dtype == labels[table][key].dtype, key
            assert stored[table][key].chunks == labels[table][key].chunks, key
            assert dict(stored[table][key].attrs) == dict(labels[table][key].attrs)
            np.testing.assert_array_equal(
                stored[table][key][:], labels[table][key][:], err_msg=key
            )
    for name, (_codes, _categories, expected, _reason) in _FALLBACKS.items():
        assert isinstance(store.cells._get_array(name), zarr.Array), name
        assert store.cells.fetch_all(name).tolist() == expected, name
    assert store.RNA.feats.fetch_all("symbol").tolist() == [1.0, 2.0]
    # More categories than an import keeps with a column: no levels.
    assert "levels" not in stored["cellData/many"].attrs
    # The categoricals beside them are categorical columns, at the limit too.
    for name, (codes, categories, expected) in _KEPT.items():
        array = store.cells._get_array(name)
        assert isinstance(array, CategoricalColumn), name
        assert array.codes[:].tolist() == codes, name
        assert array.categories.tolist() == categories.tolist(), name
        assert store.cells.fetch_all(name).tolist() == expected, name
        assert array.attrs["levels"] == categories.tolist(), name


def test_the_column_writer_stores_categories_of_more_than_one_type_as_labels(
    tmp_path: Path,
) -> None:
    from cytearc.readers._h5ad_columns import CategoryCodes
    from cytearc.writers.h5ad import _write_column

    # An HDF5 dataset holds values of one type, so the categories of an H5AD
    # file share one; the column writer falls back for mixed ones all the same.
    group = zarr.open_group(str(tmp_path / "table.zarr"), mode="w")
    column = CategoryCodes(
        np.array([0, 0, -1, 0], dtype=np.int8), np.array([3, "x"], dtype=object)
    )
    with _warnings() as messages:
        missing = _write_column(
            group, "mixed", column, profile="fast_local", table="cell", categorical=True
        )

    assert messages == [
        "Stored cell metadata column 'mixed' as a column of its labels, not as a "
        "categorical column. Column 'mixed' has categories of more than one type: "
        "the ones that rows hold are stored as int64, which cannot hold the others"
    ]
    assert missing.tolist() == [False, False, True, False]
    assert group["mixed"].dtype == np.dtype(np.int64)
    assert group["mixed"][:].tolist() == [3, 3, 0, 3]
    assert "categories" not in group["mixed"].attrs
    assert "__cytearc_categories__mixed" not in group


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
            UnstorableCategoriesError,
            "more than one type: the ones that rows hold are stored as int64",
        ),
        (
            np.array([0, 1]),
            np.array([0.5, np.nan]),
            UnstorableCategoriesError,
            "NaN among its categories",
        ),
        (
            np.array([0, 1]),
            np.array(["a", "a"]),
            UnstorableCategoriesError,
            "read alike once stored",
        ),
        (
            np.array([0, 1]),
            np.array([b"b", "b"], dtype=object),
            UnstorableCategoriesError,
            "read alike once stored",
        ),
        (
            np.array([0]),
            np.array([1, 2**70], dtype=object),
            UnstorableCategoriesError,
            "cannot all be stored as int64, the type of the labels that its rows hold",
        ),
        (
            np.array([0]),
            np.array([1, 2**63], dtype=np.uint64),
            UnstorableCategoriesError,
            "cannot all be stored as int64",
        ),
        (
            np.array([0]),
            np.array([b"a", b"\xff"], dtype=object),
            UnstorableCategoriesError,
            "cannot all be stored as text",
        ),
    ],
)
def test_categories_that_cannot_be_stored_as_categories_are_refused(
    codes: np.ndarray, categories: np.ndarray, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=f"Column 'c' .*{message}") as refused:
        stored_categorical_codes(codes, categories, name="c")

    # Only categories that a categorical column cannot store take the error
    # that an import falls back from; the others are refused as before.
    assert (refused.type is UnstorableCategoriesError) is (
        error is UnstorableCategoriesError
    )
    assert issubclass(refused.type, ValueError) is (error is not TypeError)


@pytest.mark.parametrize(
    ("codes", "categories", "kind", "labels"),
    [
        # The second category is used, so the labels of the rows fail too.
        (
            np.array([1]),
            np.array([1, 2**63], dtype=np.uint64),
            "int64",
            "holds integers outside the int64 range, such as 9223372036854775808; "
            "pass them as a NumPy array of a dtype that holds them, such as uint64, "
            "or as text",
        ),
        (
            np.array([1]),
            np.array([b"a", b"\xff"], dtype=object),
            "text",
            "holds text that is not valid UTF-8; pass str values or UTF-8 encoded bytes",
        ),
    ],
    ids=["integer_beyond_int64", "invalid_utf8"],
)
def test_a_used_category_that_its_labels_cannot_hold_is_refused_as_labels_too(
    codes: np.ndarray, categories: np.ndarray, kind: str, labels: str
) -> None:
    with pytest.raises(UnstorableCategoriesError) as unstorable:
        stored_categorical_codes(codes, categories, name="c")
    with pytest.raises(ValueError) as refused:
        stored_categorical_values(codes, categories, name="c")
    assert str(unstorable.value) == (
        f"Column 'c' has categories that cannot all be stored as {kind}, the type "
        "of the labels that its rows hold"
    )
    # A column of labels refuses them with its own error, which an import
    # does not fall back from.
    assert refused.type is ValueError
    assert str(refused.value) == f"Column 'c' {labels}"
