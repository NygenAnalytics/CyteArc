"""Exports write categorical columns as AnnData categoricals of their codes beside every category.

Each check compares an export with codes, categories and labels written out
by hand, with the store it came from, or with what exports wrote for text
columns before, so an export that expanded a categorical column to text,
reordered or dropped a category, took the reserved entry for a category or
changed a text column fails.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest

import cytearc
from cytearc.metadata.rows import metadata_column_fingerprint, metadata_missing_mask
from cytearc.readers import H5adReader
from cytearc.storage.categorical import CategoricalColumn
from cytearc.writers import H5adToZarr
from tests.storage_helpers import open_store, write_count_store

anndata = pytest.importorskip("anndata")

_N_CELLS = 12
_N_FEATURES = 4

# Each categorical column: its categories in their order, whether they are
# ordered, and each row's code, -1 in a missing row.
_CELL_COLUMNS: dict[str, tuple[list[Any], bool, list[int]]] = {
    # Text beyond ASCII and with a slash, missing rows, and an unused
    # category wider than any label.
    "cell_type": (
        ["T/NK", "B cell", "Mono", "élan", "an unused category"],
        False,
        [0, 1, -1, 2, 0, 3, 0, 1, -1, 2, 0, 3],
    ),
    # Ordered text whose last category no row holds.
    "stage": (["early", "mid", "late", "never"], True, [2, 0, 1] * 4),
    # Integers in an order of their own, with missing rows and an unused one.
    "clusters": ([3, 1, 7], False, [0, 1, -1, 0, 1, 1, -1, 0, 0, 1, 1, 0]),
    "level": ([1.5, 0.5, 2.25], False, [1, 0] * 6),
    "flag": ([True, False], False, [1, 0] * 6),
    # As many categories as cells: the stored categories end in a reserved
    # entry, empty text, which no row names and which is no category.
    "padded": (
        [f"p{index}" for index in range(_N_CELLS)],
        False,
        [11, 10, 9, -1, 7, 6, 5, 4, 3, 2, 1, 0],
    ),
    # No row holds a category.
    "unheld": (["x", "y"], False, [-1] * _N_CELLS),
}
_FEATURE_COLUMNS: dict[str, tuple[list[Any], bool, list[int]]] = {
    "biotype": (["lncRNA", "protein_coding", "pseudogene"], False, [1, -1, 0, 1]),
    # As many integer categories as features: the reserved entry is 0.
    "chromosome": ([23, 1, 2, 5], False, [1, 0, -1, 2]),
}
# The dtype of the categories of each column of numbers.
_NUMBER_DTYPES = {
    "clusters": np.dtype(np.int64),
    "level": np.dtype(np.float64),
    "flag": np.dtype(bool),
    "chromosome": np.dtype(np.int64),
}

# Columns of values beside them, which export as they did before.
_SITES = ["A", None, "B", "A", None, "B", "A", "B", "A", None, "B", "A"]
_DONORS = ["d2", "d10", "d2", "d10", "d2", "d10", "d2", "d10", "d2", "d10", "d2", "d1"]
_COUNTS = [5, None, 7, 5, 7, 7, None, 5, 5, 7, 7, 5]
_PASSED = [True, None, False, True, False, None, True, False, True, False, True, True]
# Labels with the attributes that a default import gives a source categorical.
_KINDS = ["y", "x"] * 6
_KIND_LEVELS = ["y", "x", "z"]
_KIND_COLORS = ["#112233", "#445566", "#778899"]
_STRANDS = ["+", "-", "+", "+"]
# The store rows of the cells of the cell selection.
_CHOSEN = [0, 2, 3, 7, 11]


def _labels(categories: list[Any], codes: list[int]) -> list[Any]:
    """Each row's category, None in a missing row."""
    return [None if code < 0 else categories[code] for code in codes]


def _build_store(path: Path) -> Any:
    """Write and open a store whose cells and features hold categorical columns and columns of values."""
    counts = np.arange(_N_CELLS * _N_FEATURES).reshape(_N_CELLS, _N_FEATURES) % 5 + 1
    write_count_store(str(path), {"RNA": counts}, "uint16")
    store = open_store(path)
    for table, columns in (
        (store.cells, _CELL_COLUMNS),
        (store.RNA.feats, _FEATURE_COLUMNS),
    ):
        for name, (categories, ordered, codes) in columns.items():
            table.insert(
                name,
                pd.Categorical.from_codes(
                    codes, categories=categories, ordered=ordered
                ),
            )
            assert isinstance(table._get_array(name), CategoricalColumn), name
    store.cells.insert("site", np.array(_SITES, dtype=object))
    store.cells.insert("donor", np.array(_DONORS))
    store.cells.insert("count", np.array(_COUNTS, dtype=object))
    store.cells.insert("passed", np.array(_PASSED, dtype=object))
    store.cells.insert("kind", np.array(_KINDS))
    store.zw["cellData/kind"].attrs.update(
        {
            "levels": _KIND_LEVELS,
            "ordered": True,
            "display": {
                "kind": "categorical",
                "categories": [
                    {"value": level, "label": level, "color": color}
                    for level, color in zip(_KIND_LEVELS, _KIND_COLORS, strict=True)
                ],
            },
        }
    )
    store.cells.insert("chosen", np.isin(np.arange(_N_CELLS), _CHOSEN))
    store.RNA.feats.insert("strand", np.array(_STRANDS))
    return store


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> Any:
    return _build_store(tmp_path_factory.mktemp("categorical_exports") / "store.zarr")


def _plain(value: Any) -> Any:
    """``value`` as JSON values: arrays and NumPy scalars as lists and Python scalars."""
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _file_digest(path: Path, excluded: set[str]) -> str:
    """Digest every group and dataset of an H5AD file outside ``excluded``: attributes, dtypes, shapes and values."""
    records: list[Any] = []

    def visit(name: str, node: Any) -> None:
        if any(name == left or name.startswith(f"{left}/") for left in excluded):
            return
        attributes = {key: _plain(node.attrs[key]) for key in sorted(node.attrs)}
        if isinstance(node, h5py.Group):
            records.append([name, attributes])
            return
        text = h5py.check_string_dtype(node.dtype) is not None
        values = node.asstr()[()] if text else node[()]
        records.append(
            [
                name,
                attributes,
                "text" if text else node.dtype.str,
                list(node.shape),
                _plain(values),
            ]
        )

    with h5py.File(path, "r") as h5:
        visit("", h5)
        h5.visititems(visit)
    records.sort(key=lambda record: record[0])
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def _categorical_paths() -> set[str]:
    return {
        *(f"obs/{name}" for name in _CELL_COLUMNS),
        *(f"var/{name}" for name in _FEATURE_COLUMNS),
    }


def _assert_exported(
    column: pd.Series,
    categories: list[Any],
    ordered: bool,
    codes: list[int],
    name: str,
) -> None:
    """Check an exported column against the categories, order and codes written by hand."""
    assert isinstance(column.dtype, pd.CategoricalDtype), name
    assert column.cat.categories.tolist() == categories, name
    assert column.cat.ordered is ordered, name
    assert column.cat.codes.tolist() == codes, name
    assert [None if pd.isna(label) else label for label in column] == _labels(
        categories, codes
    ), name
    if name in _NUMBER_DTYPES:
        assert column.cat.categories.dtype == _NUMBER_DTYPES[name], name


def _assert_stored_categorical(
    group: h5py.Group, categories: list[Any], ordered: bool, codes: list[int], name: str
) -> None:
    """Check a column of an H5AD file against AnnData's categorical encoding of hand-written values."""
    node = group[name]
    assert isinstance(node, h5py.Group), name
    assert node.attrs["encoding-type"] == "categorical", name
    assert node.attrs["encoding-version"] == "0.2.0", name
    assert bool(node.attrs["ordered"]) is ordered, name
    stored = node["categories"]
    if h5py.check_string_dtype(stored.dtype) is not None:
        assert stored.attrs["encoding-type"] == "string-array", name
        assert stored.asstr()[()].tolist() == categories, name
    else:
        assert stored.attrs["encoding-type"] == "array", name
        assert stored[()].tolist() == categories, name
        assert stored.dtype == _NUMBER_DTYPES[name], name
    assert node["codes"].dtype.kind == "i", name
    assert node["codes"][()].tolist() == codes, name


def _every_column() -> list[tuple[str, str, tuple[list[Any], bool, list[int]]]]:
    return [
        *(("obs", name, column) for name, column in _CELL_COLUMNS.items()),
        *(("var", name, column) for name, column in _FEATURE_COLUMNS.items()),
    ]


def test_to_anndata_keeps_each_categorical_column_as_its_categorical(
    store, tmp_path
) -> None:
    exported = store.to_anndata()

    for table, name, (categories, ordered, codes) in _every_column():
        _assert_exported(
            getattr(exported, table)[name], categories, ordered, codes, name
        )
    # AnnData writes and reads the categoricals that the object holds.
    exported.write_h5ad(tmp_path / "written.h5ad")
    written = anndata.read_h5ad(tmp_path / "written.h5ad")
    for table, name, (categories, ordered, codes) in _every_column():
        _assert_exported(
            getattr(written, table)[name], categories, ordered, codes, name
        )


def test_to_h5ad_writes_each_categorical_column_in_anndatas_encoding(
    store, tmp_path
) -> None:
    path = tmp_path / "streamed.h5ad"
    cytearc.to_h5ad(store.RNA, str(path), nthreads=1)

    with h5py.File(path, "r") as h5:
        for table, name, (categories, ordered, codes) in _every_column():
            _assert_stored_categorical(h5[table], categories, ordered, codes, name)
    exported = anndata.read_h5ad(path)
    for table, name, (categories, ordered, codes) in _every_column():
        _assert_exported(
            getattr(exported, table)[name], categories, ordered, codes, name
        )
    # The file holds the columns that to_anndata returns; only the object
    # names its indexes.
    expected = store.to_anndata()
    for table, names in (("obs", _CELL_COLUMNS), ("var", _FEATURE_COLUMNS)):
        pd.testing.assert_frame_equal(
            getattr(exported, table)[list(names)],
            getattr(expected, table)[list(names)],
            check_names=False,
        )


def test_the_reserved_entry_of_a_categories_array_is_no_category(
    store, tmp_path
) -> None:
    cytearc.to_h5ad(store.RNA, str(tmp_path / "streamed.h5ad"), nthreads=1)
    exported = store.to_anndata()

    for table, name, placeholder in (
        (store.zw["cellData"], "padded", ""),
        (store.zw["RNA/featureData"], "chromosome", 0),
    ):
        stored = table[f"__cytearc_categories__{name}"]
        assert stored.attrs["padded"] is True
        assert stored[-1] == placeholder
        frame = exported.obs if name == "padded" else exported.var
        assert placeholder not in frame[name].cat.categories.tolist()
        assert len(frame[name].cat.categories) == stored.shape[0] - 1
    with h5py.File(tmp_path / "streamed.h5ad", "r") as h5:
        assert h5["obs/padded/categories"].shape == (_N_CELLS,)
        assert h5["var/chromosome/categories"][()].tolist() == [23, 1, 2, 5]


def test_exports_of_some_cells_and_features_keep_every_category(
    store, tmp_path
) -> None:
    chosen = store.snapshot_cell_selection("chosen")
    features = [3, 0, 2]
    exported = store.to_anndata(cell_selection=chosen, feature_indexes=features)
    path = tmp_path / "chosen.h5ad"
    cytearc.to_h5ad(store.RNA, str(path), nthreads=1, cell_selection=chosen)
    streamed = anndata.read_h5ad(path)

    assert exported.obs_names.tolist() == [f"cell{row}" for row in _CHOSEN]
    assert exported.var_names.tolist() == [f"RNA{index}" for index in features]
    for name, (categories, ordered, codes) in _CELL_COLUMNS.items():
        chosen_codes = [codes[row] for row in _CHOSEN]
        _assert_exported(exported.obs[name], categories, ordered, chosen_codes, name)
        _assert_exported(streamed.obs[name], categories, ordered, chosen_codes, name)
    for name, (categories, ordered, codes) in _FEATURE_COLUMNS.items():
        _assert_exported(
            exported.var[name],
            categories,
            ordered,
            [codes[index] for index in features],
            name,
        )
        # A file holds every feature.
        _assert_exported(streamed.var[name], categories, ordered, codes, name)


def test_a_categorical_import_of_an_export_keeps_codes_labels_and_fingerprints(
    store, tmp_path
) -> None:
    path = tmp_path / "streamed.h5ad"
    cytearc.to_h5ad(store.RNA, str(path), nthreads=1)
    reader = H5adReader(str(path))
    try:
        H5adToZarr(
            reader,
            str(tmp_path / "imported.zarr"),
            nthreads=1,
            categorical_columns=True,
        ).dump()
    finally:
        reader.close()
    imported = open_store(tmp_path / "imported.zarr")

    for source, target, columns in (
        (store.cells, imported.cells, _CELL_COLUMNS),
        (store.RNA.feats, imported.RNA.feats, _FEATURE_COLUMNS),
    ):
        for name, (categories, ordered, codes) in columns.items():
            stored = target._get_array(name)
            assert isinstance(stored, CategoricalColumn), name
            assert stored.codes[:].tolist() == codes, name
            assert stored.categories.tolist() == categories, name
            assert stored.ordered is ordered, name
            assert stored.dtype.kind == source._get_array(name).dtype.kind, name
            # As many categories as rows take the reserved entry again.
            assert stored.categories_array.attrs.get("padded") is (
                True if len(categories) == len(codes) else None
            ), name
            assert target.fetch_all(name).tolist() == source.fetch_all(name).tolist()
            mask = metadata_missing_mask(target, name)
            missing = [code < 0 for code in codes]
            assert (None if mask is None else mask[:].tolist()) == (
                missing if any(missing) else None
            ), name
            assert metadata_column_fingerprint(target, name) == (
                metadata_column_fingerprint(source, name)
            ), name


def test_text_columns_export_as_before(store, tmp_path) -> None:
    exported = store.to_anndata()
    path = tmp_path / "streamed.h5ad"
    cytearc.to_h5ad(store.RNA, str(path), nthreads=1)

    obs = exported.obs
    for name, values in (("site", _SITES), ("donor", _DONORS)):
        assert not isinstance(obs[name].dtype, pd.CategoricalDtype), name
        assert [None if pd.isna(value) else value for value in obs[name]] == values
    assert obs["count"].dtype == np.float64
    assert [None if pd.isna(value) else value for value in obs["count"]] == _COUNTS
    assert obs["passed"].dtype == "boolean"
    assert [None if pd.isna(value) else value for value in obs["passed"]] == _PASSED
    assert obs["kind"].cat.categories.tolist() == _KIND_LEVELS
    assert obs["kind"].cat.ordered
    assert obs["kind"].tolist() == _KINDS
    assert exported.uns["kind_colors"].tolist() == _KIND_COLORS
    assert exported.var["strand"].tolist() == _STRANDS
    with h5py.File(path, "r") as h5:
        # Text with missing rows is a categorical of the sorted labels.
        assert h5["obs/site"].attrs["encoding-type"] == "categorical"
        assert h5["obs/site/categories"].asstr()[()].tolist() == ["A", "B"]
        assert h5["obs/site/codes"].dtype == np.int32
        assert h5["obs/donor"].attrs["encoding-type"] == "string-array"
    # Every element of the file but the categorical columns, as the export
    # wrote it before categorical columns exported as categoricals (#225).
    assert _file_digest(path, _categorical_paths()) == (
        "51c0ee0971cac6c9166861a037e049dd47d04a6bdfa39fc890e806aaa06b5eef"
    )


def test_an_imported_categorical_column_exports_with_its_colors(tmp_path) -> None:
    from tests.test_h5ad_import_fidelity import (
        _STAGE_COLORS,
        _assert_categorical,
        _categorical_adata,
        _generated_colors,
    )

    source = _categorical_adata()
    source.write_h5ad(tmp_path / "source.h5ad")
    reader = H5adReader(str(tmp_path / "source.h5ad"))
    try:
        H5adToZarr(
            reader, str(tmp_path / "store.zarr"), nthreads=1, categorical_columns=True
        ).dump()
    finally:
        reader.close()
    imported = open_store(tmp_path / "store.zarr")
    assert isinstance(imported.cells._get_array("stage"), CategoricalColumn)
    cytearc.to_h5ad(imported.RNA, str(tmp_path / "streamed.h5ad"), nthreads=1)

    for exported in (
        imported.to_anndata(),
        anndata.read_h5ad(tmp_path / "streamed.h5ad"),
    ):
        for column in ("stage", "clusters_int", "flag", "level", "named", "short"):
            _assert_categorical(exported.obs[column], source.obs[column])
        _assert_categorical(exported.var["kind"], source.var["kind"])
        assert exported.uns["stage_colors"].tolist() == _STAGE_COLORS
        assert exported.uns["named_colors"].tolist() == ["#ff0000", "#0000ff"]
        assert exported.uns["clusters_int_colors"].tolist() == _generated_colors(
            [3, 1, 2, 7]
        )


def test_an_export_refuses_a_categorical_column_whose_mask_or_codes_are_corrupt(
    tmp_path,
) -> None:
    for corruption, message in (
        ("mask", "has a missing mask that disagrees with its codes"),
        ("codes", "holds the code 5, but its codes name 5 categories"),
    ):
        store = _build_store(tmp_path / f"{corruption}.zarr")
        cells = store.zw["cellData"]
        if corruption == "mask":
            mask = cells["__cytearc_missing__cell_type"]
            flags = mask[:]
            flags[2] = False
            mask[:] = flags
        else:
            codes = cells["cell_type"][:]
            codes[0] = 5
            cells["cell_type"][:] = codes
        with pytest.raises(ValueError, match=message):
            store.to_anndata()
        with pytest.raises(ValueError, match=message):
            cytearc.to_h5ad(store.RNA, str(tmp_path / f"{corruption}.h5ad"))
        assert not (tmp_path / f"{corruption}.h5ad").exists()


def test_a_plan_checks_a_categorical_column_as_it_checks_values(tmp_path) -> None:
    import dataclasses

    from cytearc.writers.export import H5adColumn, h5ad_frame, write_h5ad_plan
    from tests.test_writers import _plan

    rows = [np.asarray([[1, 0], [0, 2], [0, 3]], dtype=np.int32)]
    labels = pd.Categorical.from_codes([1, -1, 0], categories=["b", "a"], ordered=True)
    good = H5adColumn("labels", lambda: (labels, None))
    short = H5adColumn("labels", lambda: (labels[:2], None))
    masked = H5adColumn("labels", lambda: (labels, np.zeros(3, dtype=bool)))
    index = H5adColumn("_index", lambda: (pd.Categorical(["c0", "c1", "c2"]), None))
    path = tmp_path / "export.h5ad"
    for plan, error, message in (
        (_plan(rows, obs=[short]), ValueError, "Column 'labels' has 2 rows"),
        (
            _plan(rows, obs=[masked]),
            ValueError,
            "Column 'labels' is a categorical with a missing mask",
        ),
        (
            dataclasses.replace(_plan(rows), obs_index=index),
            TypeError,
            "Index '_index' must hold values, not a categorical",
        ),
    ):
        with pytest.raises(error, match=message):
            write_h5ad_plan(plan, path)
        assert not path.exists()
    with pytest.raises(TypeError, match="must hold values, not a categorical"):
        h5ad_frame(index, (), 3)

    write_h5ad_plan(_plan(rows, obs=[good]), path)
    with h5py.File(path, "r") as h5:
        assert h5["obs/labels/categories"].asstr()[()].tolist() == ["b", "a"]
        assert h5["obs/labels/codes"][()].tolist() == [1, -1, 0]
        assert bool(h5["obs/labels"].attrs["ordered"])
    # The object holds the categorical that the file stores.
    frame = h5ad_frame(_plan(rows, obs=[good]).obs_index, (good,), 3)
    pd.testing.assert_series_equal(
        anndata.read_h5ad(path).obs["labels"], frame["labels"], check_names=False
    )
