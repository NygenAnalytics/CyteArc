"""H5AD import fidelity: categoricals, colors, indexes, IDs, text, and matrix choice."""

import hashlib
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
from scipy.sparse import csc_matrix, csr_matrix

import cytearc
from cytearc.metadata.artifacts import categorical_display
from cytearc.readers import H5adReader, inspect_h5ad
from cytearc.readers._h5ad_columns import (
    H5adCategories,
    category_colors,
    column_categories,
    hex_color,
)
from cytearc.storage.count_matrix import DEFAULT_COUNT_MATRIX_POLICY
from cytearc.utils.memory_budget import MemoryBudgetError
from cytearc.writers import H5adToZarr

anndata = pytest.importorskip("anndata")

_STAGES = ["early", "mid", "late", "unused"]
_STAGE_COLORS = ["#000000", "#888888", "#ff0000", "#00ff00"]


def _text(*values: bytes) -> np.ndarray:
    """Return byte strings as AnnData writes text, of variable length, which h5py reads as bytes."""
    return np.array(values, dtype=h5py.string_dtype())


@contextmanager
def _warnings() -> Iterator[list[str]]:
    messages: list[str] = []
    handler = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        yield messages
    finally:
        logger.remove(handler)


def _write(path: Path, adata) -> str:
    adata.write_h5ad(path)
    return str(path)


def _import(path: str, destination: Path, **reader_kwargs):
    """Import ``path`` with its inspected keys and open the store."""
    reader = H5adReader.from_inspect(inspect_h5ad(path), **reader_kwargs)
    try:
        result = H5adToZarr(
            reader,
            str(destination),
            nthreads=1,
            analysis_assay=(
                "RNA" if reader.embeddingRoles or reader.clusterKeys else None
            ),
            shuffle_cells=False,
        ).dump()
    finally:
        reader.close()
    store = cytearc.DataStore(str(destination), min_features_per_cell=0, nthreads=1)
    return store, result


def _categorical_adata() -> "anndata.AnnData":
    """Cells with an ordered stage, integer clusters with a missing label, and more."""
    n_cells = 8
    obs = pd.DataFrame(index=[f"cell{index}" for index in range(n_cells)])
    obs["stage"] = pd.Categorical(
        ["late", "early", "mid", "late", "early", "mid", "late", "early"],
        categories=_STAGES,
        ordered=True,
    )
    clusters = pd.Categorical([3, 1, 2, 3, 1, 2, 3, 1], categories=[3, 1, 2, 7])
    clusters[0] = np.nan
    obs["clusters_int"] = clusters
    obs["flag"] = pd.Categorical(
        [True, False] * 4, categories=[True, False], ordered=False
    )
    obs["level"] = pd.Categorical([1.5, 0.5] * 4, categories=[1.5, 0.5])
    obs["named"] = pd.Categorical(["x", "y"] * 4, categories=["x", "y"])
    obs["short"] = pd.Categorical(["p", "q"] * 4, categories=["p", "q", "r"])
    obs["plain"] = [f"text{index}" for index in range(n_cells)]
    var = pd.DataFrame(index=[f"gene{index}" for index in range(4)])
    var["kind"] = pd.Categorical(["x", "y", "x", "y"], categories=["y", "x", "z"])
    counts = np.arange(n_cells * 4, dtype=np.float32).reshape(n_cells, 4) % 5
    adata = anndata.AnnData(X=csr_matrix(counts), obs=obs, var=var)
    adata.uns["stage_colors"] = np.asarray(_STAGE_COLORS, dtype=object)
    adata.uns["named_colors"] = ["red", "#00F"]
    # Fewer colors than categories are not used, as Scanpy does not use them.
    adata.uns["short_colors"] = ["#111111", "#222222"]
    return adata


@pytest.fixture(scope="module")
def categorical_store(tmp_path_factory):
    """A store imported from an H5AD file with many kinds of categoricals."""
    directory = tmp_path_factory.mktemp("categoricals")
    source = _categorical_adata()
    path = _write(directory / "source.h5ad", source)
    store, _ = _import(path, directory / "store.zarr")
    return source, path, store


def _generated_colors(levels: list) -> list[str]:
    generated = {
        category["value"]: category["color"]
        for category in categorical_display(np.asarray(levels, dtype=object))[
            "categories"
        ]
    }
    return [generated[level] for level in levels]


def test_import_keeps_categories_order_unused_levels_and_colors(
    categorical_store,
) -> None:
    _source, _path, store = categorical_store
    root = zarr.open_group(store.zw.store, mode="r")
    cells = root["cellData"]

    stage = dict(cells["stage"].attrs)
    assert stage["levels"] == _STAGES
    assert stage["ordered"] is True
    assert stage["display"] == {
        "kind": "categorical",
        "categories": [
            {"value": level, "label": level, "color": color}
            for level, color in zip(_STAGES, _STAGE_COLORS, strict=True)
        ],
    }
    # The integer categories keep their type and source order; one label is
    # missing, so the display names its missing color.
    clusters = dict(cells["clusters_int"].attrs)
    assert clusters["levels"] == [3, 1, 2, 7]
    assert clusters["ordered"] is False
    assert clusters["display"] == {
        "kind": "categorical",
        "categories": [
            {"value": level, "label": str(level), "color": color}
            for level, color in zip(
                [3, 1, 2, 7], _generated_colors([3, 1, 2, 7]), strict=True
            )
        ],
        "missing_label": "NA",
        "missing_color": "#bdbdbd",
    }
    assert store.cells.fetch_all("clusters_int").tolist() == [0, 1, 2, 3, 1, 2, 3, 1]
    assert dict(cells["flag"].attrs)["levels"] == [True, False]
    assert dict(cells["level"].attrs)["levels"] == [1.5, 0.5]
    # Named and short hexadecimal colors become #rrggbb.
    assert [
        category["color"] for category in cells["named"].attrs["display"]["categories"]
    ] == ["#ff0000", "#0000ff"]
    assert [
        category["color"] for category in cells["short"].attrs["display"]["categories"]
    ] == _generated_colors(["p", "q", "r"])
    # A text column that is not categorical gets no attributes.
    assert "levels" not in cells["plain"].attrs
    # Feature categoricals keep their categories, without a display, beside
    # the link of the categorical column that a default import stores.
    kind = dict(root["RNA/featureData/kind"].attrs)
    assert kind == {
        "levels": ["y", "x", "z"],
        "ordered": False,
        "categories": "__cytearc_categories__kind",
    }


def test_plots_read_imported_category_order_and_colors(categorical_store) -> None:
    from cytearc.plotting._display import stored_categorical_scale

    _source, _path, store = categorical_store
    scale = stored_categorical_scale(store, "stage")

    assert scale is not None
    assert scale.order == tuple(_STAGES)
    assert scale.palette == dict(zip(_STAGES, _STAGE_COLORS, strict=True))


def _assert_categorical(series: pd.Series, source: pd.Series) -> None:
    assert isinstance(series.dtype, pd.CategoricalDtype)
    assert series.cat.categories.tolist() == source.cat.categories.tolist()
    assert series.cat.ordered == source.cat.ordered
    assert series.astype(object).where(series.notna(), None).tolist() == (
        source.astype(object).where(source.notna(), None).tolist()
    )


@pytest.mark.parametrize("route", ["to_anndata", "to_h5ad"])
def test_exports_restore_imported_categoricals_and_colors(
    categorical_store, tmp_path, route
) -> None:
    source, _path, store = categorical_store
    if route == "to_anndata":
        exported = store.to_anndata()
        # AnnData writes the categoricals and colors that the object holds.
        exported.write_h5ad(tmp_path / "written.h5ad")
        exported = anndata.read_h5ad(tmp_path / "written.h5ad")
    else:
        cytearc.to_h5ad(store.RNA, str(tmp_path / "streamed.h5ad"))
        exported = anndata.read_h5ad(tmp_path / "streamed.h5ad")

    for column in ("stage", "clusters_int", "flag", "level", "named", "short"):
        _assert_categorical(exported.obs[column], source.obs[column])
    _assert_categorical(exported.var["kind"], source.var["kind"])
    assert exported.uns["stage_colors"].tolist() == _STAGE_COLORS
    assert exported.uns["named_colors"].tolist() == ["#ff0000", "#0000ff"]
    assert exported.uns["clusters_int_colors"].tolist() == _generated_colors(
        [3, 1, 2, 7]
    )
    assert not isinstance(exported.obs["plain"].dtype, pd.CategoricalDtype)
    assert exported.obs["plain"].tolist() == source.obs["plain"].tolist()


def test_export_appends_values_that_the_stored_levels_omit(tmp_path) -> None:
    from cytearc.writers.export import H5adCategorical, _explicit_categorical

    with _warnings() as messages:
        categorical, colors = _explicit_categorical(
            "stage",
            np.asarray(["b", "z", "a", "y", ""]),
            np.asarray([False, False, False, False, True]),
            H5adCategorical(
                levels=("b", "a"), ordered=True, colors={"b": "#000001", "a": "#000002"}
            ),
        )

    assert categorical.categories.tolist() == ["b", "a", "y", "z"]
    assert categorical.ordered
    assert categorical.astype(object).tolist()[:4] == ["b", "z", "a", "y"]
    assert pd.isna(categorical[4])
    # Not every category has a color, so none are written.
    assert colors is None
    assert any(
        "holds 2 values that are not among its stored categories, such as 'y'" in m
        for m in messages
    )


def test_cluster_labels_keep_categories_and_export_with_them(tmp_path) -> None:
    source = _categorical_adata()
    path = _write(tmp_path / "clusters.h5ad", source)
    store, result = _import(
        path, tmp_path / "store.zarr", cluster_keys=("stage", "plain")
    )
    ref = result.clusterArtifacts["stage"]
    # Labels that were not a categorical keep no categories.
    assert "levels" not in dict(
        store.artifacts.load(result.clusterArtifacts["plain"])["values"].attrs
    )

    attributes = dict(store.artifacts.load(ref)["values"].attrs)
    assert attributes["levels"] == _STAGES
    assert attributes["ordered"] is True
    assert [
        category["color"] for category in attributes["display"]["categories"]
    ] == _STAGE_COLORS
    exported = store.to_anndata(
        cell_selection=result.cellSelection, obs={"stage_labels": ref}
    )
    _assert_categorical(exported.obs["stage_labels"], source.obs["stage"])
    assert exported.uns["stage_labels_colors"].tolist() == _STAGE_COLORS


def test_imported_analyses_name_the_sha256_of_the_source_file(tmp_path) -> None:
    # The source's digest identifies every analysis that an import saves, so
    # a saved import is reused only for the same bytes.
    path = _write(tmp_path / "clusters.h5ad", _categorical_adata())
    store, result = _import(path, tmp_path / "store.zarr", cluster_keys=("stage",))
    inputs = store.artifacts.inspect(result.clusterArtifacts["stage"]).inputs
    expected = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert inputs["source_digest"] == {"bytes_hex": expected}


def test_categoricals_with_many_categories_keep_no_levels(tmp_path, monkeypatch):
    import cytearc.writers.h5ad as h5ad_writer

    monkeypatch.setattr(h5ad_writer, "MAX_STORED_CATEGORIES", 3)
    source = _categorical_adata()
    store, _ = _import(_write(tmp_path / "many.h5ad", source), tmp_path / "s.zarr")
    cells = zarr.open_group(store.zw.store, mode="r")["cellData"]

    # Four categories exceed the limit; two do not.
    assert "levels" not in cells["stage"].attrs
    assert cells["named"].attrs["levels"] == ["x", "y"]


@pytest.mark.parametrize(
    ("dtype", "categories", "levels"),
    [
        (np.dtype("U3"), [b"a", 1, "b"], ["a", "1", "b"]),
        (np.dtype(bool), [True, False], [True, False]),
        (np.dtype(np.int64), [3, 1], [3, 1]),
        (np.dtype(np.uint8), [3, 1], [3, 1]),
        (np.dtype(np.float64), [1, 0.5], [1.0, 0.5]),
        # Categories that the column cannot hold exactly keep no levels.
        (np.dtype(np.int64), ["a", 1], None),
        (np.dtype(np.float64), [float("inf")], None),
        (np.dtype(np.float64), [True], None),
        (np.dtype("M8[ns]"), [1], None),
    ],
)
def test_stored_levels_follow_the_stored_column_type(dtype, categories, levels):
    from cytearc.writers.h5ad import _stored_levels

    assert _stored_levels(dtype, categories) == levels


def test_display_metadata_that_would_be_invalid_is_not_stored() -> None:
    from cytearc.writers.h5ad import _categorical_attributes

    # A display holds hexadecimal colors only.
    attributes = _categorical_attributes(
        np.dtype("U1"),
        H5adCategories(("a", "b"), False),
        colors=("red", "#000002"),
        has_missing=False,
        display=True,
    )
    assert attributes == {"levels": ["a", "b"], "ordered": False}


def test_categories_that_compare_equal_keep_one_level_with_the_first_color() -> None:
    from cytearc.metadata.artifacts import levels_display
    from cytearc.writers.h5ad import _categorical_attributes

    # Two categories that compare equal cannot both be display values, so a
    # float column keeps one level for 1 and 1.0, with the color of the first.
    assert levels_display([1, 1.0], None, False) is None
    attributes = _categorical_attributes(
        np.dtype(np.float64),
        H5adCategories((1, 1.0, 2), True),
        colors=("#000001", "#000002", "#000003"),
        has_missing=False,
        display=True,
    )
    assert attributes == {
        "levels": [1.0, 2.0],
        "ordered": True,
        "display": {
            "kind": "categorical",
            "categories": [
                {"value": 1.0, "label": "1.0", "color": "#000001"},
                {"value": 2.0, "label": "2.0", "color": "#000003"},
            ],
        },
    }


@pytest.mark.parametrize(
    ("text", "color"),
    [
        ("#ABC", "#aabbcc"),
        ("#abcd", "#aabbcc"),
        ("#1F77B4", "#1f77b4"),
        ("#1f77b4ff", "#1f77b4"),
        (" #1f77b4 ", "#1f77b4"),
        ("red", "#ff0000"),
        ("tab:blue", "#1f77b4"),
        ("#12345", None),
        ("#", None),
        ("#ggg", None),
        ("not a color", None),
    ],
)
def test_hex_color_reads_hexadecimal_and_named_colors(text, color) -> None:
    assert hex_color(text) == color


def test_hex_color_needs_matplotlib_only_for_named_colors(monkeypatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "matplotlib.colors", None)
    assert hex_color("#00ff00") == "#00ff00"
    assert hex_color("red") is None


def test_category_colors_follow_scanpy_rules(tmp_path) -> None:
    path = tmp_path / "colors.h5"
    with h5py.File(path, "w") as h5:
        uns = h5.create_group("uns")
        uns.create_dataset("a_colors", data=np.asarray([b"#000000", b"#ffffff"]))
        uns.create_dataset("b_colors", data=np.asarray([1, 2]))
        uns.create_dataset("c_colors", data=np.asarray([b"#000000", b"nonsense"]))
        uns.create_dataset("d_colors", data=np.zeros((2, 2)))
        uns.create_group("e_colors")
        uns.create_dataset("f_colors", data=_text(b"#00FF00", b"#ff\xe90000"))
    with h5py.File(path, "r") as h5:
        # More colors than categories: the first ones are used.
        assert category_colors(h5, "a", 1) == ("#000000",)
        assert category_colors(h5, "a", 2) == ("#000000", "#ffffff")
        assert category_colors(h5, "a", 3) is None
        assert category_colors(h5, "a", 0) is None
        assert category_colors(h5, "b", 2) is None
        assert category_colors(h5, "c", 2) is None
        assert category_colors(h5, "d", 2) is None
        assert category_colors(h5, "e", 1) is None
        assert category_colors(h5, "missing", 1) is None
        # Bytes that are not UTF-8 text are no color, which the import used
        # to stop at with a UnicodeDecodeError after it created the store.
        assert category_colors(h5, "f", 1) == ("#00ff00",)
        assert category_colors(h5, "f", 2) is None


def test_legacy_categoricals_keep_their_categories(tmp_path) -> None:
    path = tmp_path / "legacy.h5"
    with h5py.File(path, "w") as h5:
        obs = h5.create_group("obs")
        obs.create_dataset("batch", data=np.asarray([1, 0, 1], dtype=np.int8))
        obs.create_dataset("score", data=np.asarray([0.5, 1.5, 2.5]))
        obs.create_dataset("other", data=np.asarray([0, 1, 1], dtype=np.int8))
        # AnnData 0.6 categories of a group's column, kept in uns.
        obs.create_dataset("donor", data=np.asarray([1, 1, 0], dtype=np.int8))
        categories = obs.create_group("__categories")
        ordered = categories.create_dataset("batch", data=np.asarray([b"B", b"A"]))
        ordered.attrs["ordered"] = True
        compound = np.zeros(
            2, dtype=[("index", "S2"), ("group", "i1"), ("value", "f8")]
        )
        compound["group"] = [0, 1]
        h5.create_dataset("var", data=compound)
        uns = h5.create_group("uns")
        uns.create_dataset("group_categories", data=np.asarray([b"g", b"h"]))
        uns.create_dataset("donor_categories", data=np.asarray([b"d1", b"d2"]))
        nullable = obs.create_group("nullable")
        nullable.create_dataset("values", data=np.asarray([1, 2, 3]))
        nullable.create_dataset("mask", data=np.asarray([False, False, True]))
    with h5py.File(path, "r") as h5:
        groups = ("__categories",)

        def categories(table: str, key: str) -> H5adCategories | None:
            return column_categories(
                h5, h5[table], key, groups, expectation="unused", place="unused"
            )

        assert categories("obs", "batch") == H5adCategories(("B", "A"), True)
        assert categories("obs", "donor") == H5adCategories(("d1", "d2"), False)
        assert categories("obs", "score") is None
        # Integer codes without stored categories are plain integers.
        assert categories("obs", "other") is None
        assert categories("obs", "nullable") is None
        # AnnData 0.6 kept the categories of a compound column in uns.
        assert categories("var", "group") == H5adCategories(("g", "h"), False)
        assert categories("var", "value") is None


@pytest.mark.filterwarnings("ignore:Observation names are not unique")
def test_index_that_repeats_the_names_is_left_out_and_exports_work(tmp_path):
    n_cells, n_genes = 6, 4
    var = pd.DataFrame(index=[f"SYM{index}" for index in range(n_genes)])
    var["gene_ids"] = [f"ENSG{index}" for index in range(n_genes)]
    var["gene_name"] = [f"SYM{index}" for index in range(n_genes)]
    obs = pd.DataFrame(index=[f"c{index % 3}" for index in range(n_cells)])
    obs["barcode"] = [f"bc{index}" for index in range(n_cells)]
    counts = csr_matrix(np.arange(n_cells * n_genes).reshape(n_cells, n_genes) % 3)
    path = _write(
        tmp_path / "index.h5ad",
        anndata.AnnData(X=counts.astype(np.float32), obs=obs, var=var),
    )
    inspection = inspect_h5ad(path)
    assert (inspection.featureIdsKey, inspection.featureNameKey) == (
        "gene_ids",
        "gene_name",
    )
    assert inspection.cellIdsKey == "barcode"

    store, _ = _import(path, tmp_path / "store.zarr")

    # The var index holds the names, so it is left out; the obs index holds
    # other values and imports as source_index, never as _index.
    assert set(store.RNA.feats.columns) == {"I", "ids", "names", "dropOuts", "nCells"}
    assert "_index" not in store.cells.columns
    assert store.cells.fetch_all("source_index").tolist() == list(obs.index)
    cytearc.to_h5ad(store.RNA, str(tmp_path / "streamed.h5ad"))
    streamed = anndata.read_h5ad(tmp_path / "streamed.h5ad")
    assert streamed.var["gene_short_name"].tolist() == list(var["gene_name"])
    assert streamed.obs["source_index"].tolist() == list(obs.index)
    store.to_anndata().write_h5ad(tmp_path / "written.h5ad")
    assert anndata.read_h5ad(tmp_path / "written.h5ad").obs_names.tolist() == list(
        obs["barcode"]
    )


def test_unnamed_index_imports_under_a_free_source_index_name(tmp_path) -> None:
    n_genes = 3
    var = pd.DataFrame(index=[f"alias{index}" for index in range(n_genes)])
    var["gene_ids"] = [f"ENSG{index}" for index in range(n_genes)]
    var["gene_name"] = [f"SYM{index}" for index in range(n_genes)]
    var["source_index"] = ["taken"] * n_genes
    obs = pd.DataFrame(index=["a", "b"])
    path = _write(
        tmp_path / "alias.h5ad",
        anndata.AnnData(
            X=csr_matrix(np.ones((2, n_genes), dtype=np.float32)), obs=obs, var=var
        ),
    )
    store, _ = _import(path, tmp_path / "store.zarr")

    assert store.RNA.feats.fetch_all("source_index_2").tolist() == list(var.index)
    assert store.RNA.feats.fetch_all("source_index").tolist() == ["taken"] * n_genes


def test_missing_feature_names_take_the_feature_ids(tmp_path) -> None:
    var = pd.DataFrame(index=[f"ENSG{index}" for index in range(5)])
    names = pd.Categorical(["A", None, "C", "D", None])
    var["gene_symbols"] = names
    path = _write(
        tmp_path / "names.h5ad",
        anndata.AnnData(
            X=csr_matrix(np.ones((2, 5), dtype=np.float32)),
            obs=pd.DataFrame(index=["a", "b"]),
            var=var,
        ),
    )
    with _warnings() as messages:
        store, _ = _import(path, tmp_path / "store.zarr")

    assert store.RNA.feats.fetch_all("names").tolist() == [
        "A",
        "ENSG1",
        "C",
        "D",
        "ENSG4",
    ]
    assert (
        sum(
            "2 feature names in 'gene_symbols' are missing or blank" in message
            for message in messages
        )
        == 1
    )


def _duplicated_cells(tmp_path: Path, **columns) -> str:
    obs = pd.DataFrame(index=["c0", "c1", "c0", "c2"])
    for name, values in columns.items():
        obs[name] = values
    adata = anndata.AnnData(
        X=csr_matrix(np.arange(8, dtype=np.float32).reshape(4, 2)),
        obs=obs,
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    adata.obsm["X_umap"] = np.arange(8, dtype=np.float32).reshape(4, 2)
    return _write(tmp_path / "duplicates.h5ad", adata)


@pytest.mark.filterwarnings("ignore:Observation names are not unique")
def test_repeated_cell_ids_import_only_when_renamed_on_request(tmp_path) -> None:
    # A unique float score and a unique count are not IDs.
    path = _duplicated_cells(
        tmp_path, score=[0.1, 0.2, 0.3, 0.4], n_counts=[10, 20, 30, 40]
    )
    with _warnings() as messages:
        inspection = inspect_h5ad(path)

    assert inspection.cellIdsKey == "_index"
    assert any(
        "The cell IDs in the obs index '_index' repeat values" in message
        and "duplicate_cell_ids='make_unique'" in message
        for message in messages
    )
    assert not any("generated IDs" in message for message in messages)
    reader = H5adReader.from_inspect(inspection)
    try:
        with pytest.raises(
            ValueError,
            match="H5AD cell IDs must contain unique values; pass "
            "duplicate_cell_ids='make_unique' to rename repeated IDs",
        ):
            reader.cell_ids()
    finally:
        reader.close()

    store, result = _import(
        path,
        tmp_path / "store.zarr",
        duplicate_cell_ids="make_unique",
        embedding_roles={"X_umap": "umap"},
    )
    assert store.cells.fetch_all("ids").tolist() == ["c0", "c1", "c0-1", "c2"]
    # The embedding aligns to the renamed IDs.
    np.testing.assert_array_equal(
        store.artifacts.load(result.embeddingArtifacts["X_umap"])["values"][:],
        np.arange(8, dtype=np.float32).reshape(4, 2),
    )


@pytest.mark.filterwarnings("ignore:Observation names are not unique")
def test_duplicate_cell_ids_option_is_validated(tmp_path) -> None:
    path = _duplicated_cells(tmp_path)
    with pytest.raises(
        ValueError, match="duplicate_cell_ids must be 'error' or 'make_unique'"
    ):
        H5adReader(path, duplicate_cell_ids="unique")  # type: ignore[arg-type]
    reader = H5adReader(path, duplicate_cell_ids="make_unique")
    try:
        assert reader._clone_kwargs()["duplicate_cell_ids"] == "make_unique"
    finally:
        reader.close()


@pytest.mark.filterwarnings("ignore:Observation names are not unique")
def test_preferred_integer_cell_ids_are_identifiers(tmp_path) -> None:
    path = _duplicated_cells(tmp_path, cell_id=[5, 6, 7, 8])

    assert inspect_h5ad(path).cellIdsKey == "cell_id"


def _h5ad_with(path: Path, members: dict[str, Any]) -> str:
    """Write a four-cell, three-gene H5AD file, then replace the given members.

    Each key is a member's path in the file. A dict writes a group of those
    datasets, such as a categorical index's codes and categories, and None
    removes the member.
    """
    adata = anndata.AnnData(csr_matrix(np.ones((4, 3), dtype=np.float32)))
    adata.obs_names = [f"c{index}" for index in range(4)]
    adata.var_names = [f"g{index}" for index in range(3)]
    adata.write_h5ad(path)
    with h5py.File(path, "r+") as h5:
        for location, data in members.items():
            if location in h5:
                del h5[location]
            if isinstance(data, dict):
                group = h5.create_group(location)
                for name, values in data.items():
                    group.create_dataset(name, data=values)
            elif data is not None:
                h5.create_dataset(location, data=data)
    return str(path)


_ID_METHODS = ("cell_ids", "feat_ids", "feat_names")
# What the reader gives and the import stores for the file that _h5ad_with
# writes before it replaces any member.
_WRITTEN_IDS = {
    "cell_ids": ["c0", "c1", "c2", "c3"],
    "feat_ids": ["g0", "g1", "g2"],
    "feat_names": ["g0", "g1", "g2"],
}


def _read_and_stored_ids(
    path: str, destination: Path, **reader_kwargs: Any
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return the reader's IDs and names, and those that an import of the file stores.

    The import runs on a reader of its own, in the file's order.
    """
    reader = H5adReader(path, **reader_kwargs)
    try:
        read = {method: getattr(reader, method)() for method in _ID_METHODS}
    finally:
        reader.close()
    reader = H5adReader(path, **reader_kwargs)
    try:
        H5adToZarr(reader, str(destination), nthreads=1, shuffle_cells=False).dump()
    finally:
        reader.close()
    store = cytearc.DataStore(str(destination), min_features_per_cell=0, nthreads=1)
    stored = {
        "cell_ids": store.cells.fetch_all("ids"),
        "feat_ids": store.RNA.feats.fetch_all("ids"),
        "feat_names": store.RNA.feats.fetch_all("names"),
    }
    return read, stored


@pytest.mark.parametrize(
    ("members", "reader_kwargs", "expected"),
    [
        pytest.param(
            {"obs/_index": np.array([101, 102, 103, 104])},
            {},
            {"cell_ids": ["101", "102", "103", "104"]},
            id="integer-cell-ids",
        ),
        pytest.param(
            {"obs/_index": np.array([b"c0", b"c1", b"c2", b"c3"], dtype="S2")},
            {},
            {"cell_ids": ["c0", "c1", "c2", "c3"]},
            id="fixed-length-byte-cell-ids",
        ),
        pytest.param(
            {"var/_index": np.array([7, 8, 9])},
            {},
            {"feat_ids": ["7", "8", "9"], "feat_names": ["7", "8", "9"]},
            id="integer-feature-ids-as-names",
        ),
        pytest.param(
            {
                "var/_index": np.array([b"g0", b"g1", b"g2"], dtype="S2"),
                "var/symbol": np.array([b"A", b"BB", b"C"], dtype="S2"),
            },
            {"feature_name_key": "symbol"},
            {"feat_ids": ["g0", "g1", "g2"], "feat_names": ["A", "BB", "C"]},
            id="fixed-length-byte-feature-ids-and-names",
        ),
        pytest.param(
            {"var/symbol": np.array([10, 20, 30])},
            {"feature_name_key": "symbol"},
            {"feat_names": ["10", "20", "30"]},
            id="integer-feature-names",
        ),
        pytest.param(
            {
                "obs/_index": np.array(
                    [text.encode() for text in ("café", "中文", "😀", "c3")]
                )
            },
            {},
            {"cell_ids": ["café", "中文", "😀", "c3"]},
            id="non-ascii-utf8-bytes",
        ),
        pytest.param(
            {
                "obs/_index": np.array([-1, 0, 2**62, 2**63 - 1]),
                "var/_index": np.array([2**64 - 1, 0, 7], dtype=np.uint64),
            },
            {},
            {
                "cell_ids": ["-1", "0", "4611686018427387904", "9223372036854775807"],
                "feat_ids": ["18446744073709551615", "0", "7"],
                "feat_names": ["18446744073709551615", "0", "7"],
            },
            id="negative-and-largest-integers",
        ),
        pytest.param(
            {
                "obs/_index": {
                    "codes": np.array([3, 2, 1, 0], dtype=np.int8),
                    "categories": np.array([7, 8, 9, 10]),
                }
            },
            {},
            {"cell_ids": ["10", "9", "8", "7"]},
            id="categorical-index-of-integers",
        ),
        pytest.param(
            {"obs/_index": None, "var/_index": None},
            {},
            {
                "cell_ids": ["cell_0", "cell_1", "cell_2", "cell_3"],
                "feat_ids": ["feature_0", "feature_1", "feature_2"],
                "feat_names": ["feature_0", "feature_1", "feature_2"],
            },
            id="no-index",
        ),
        pytest.param(
            {
                "var/_index": np.array([5, 6, 7]),
                "var/symbol": np.array([b"A", b"", b"C"]),
            },
            {"feature_name_key": "symbol"},
            {"feat_ids": ["5", "6", "7"], "feat_names": ["A", "6", "C"]},
            id="blank-name-takes-an-integer-id",
        ),
        pytest.param(
            {"var/symbol": np.array([1.5, np.nan, 2.0])},
            {"feature_name_key": "symbol"},
            {"feat_names": ["1.5", "g1", "2.0"]},
            id="missing-number-name-takes-the-id",
        ),
        pytest.param(
            {"obs/_index": np.array([5, 5, 6, 7])},
            {"duplicate_cell_ids": "make_unique"},
            {"cell_ids": ["5", "5-1", "6", "7"]},
            id="renamed-integer-cell-ids",
        ),
        pytest.param(
            {"var/_index": np.array([7, 7, 8])},
            {"duplicate_feature_ids": "make_unique"},
            {"feat_ids": ["7", "7-1", "8"], "feat_names": ["7", "7", "8"]},
            id="renamed-integer-feature-ids",
        ),
    ],
)
def test_reader_ids_and_names_are_the_text_that_an_import_stores(
    tmp_path, members, reader_kwargs, expected
) -> None:
    path = _h5ad_with(tmp_path / "ids.h5ad", members)

    read, stored = _read_and_stored_ids(path, tmp_path / "ids.zarr", **reader_kwargs)

    for method in _ID_METHODS:
        # The reader used to give the values as the file holds them, such as
        # 101 or b"c0", where the store holds "101" or "c0".
        assert read[method].dtype.kind == "U", method
        assert read[method].tolist() == expected.get(method, _WRITTEN_IDS[method])
        np.testing.assert_array_equal(read[method], stored[method], err_msg=method)


def test_assay_features_are_the_text_that_a_split_import_stores(tmp_path) -> None:
    path = _h5ad_with(
        tmp_path / "split.h5ad",
        {
            "var/_index": np.array([7, 8, 9]),
            "var/feature_types": np.array(
                [b"Gene Expression", b"Antibody Capture", b"Gene Expression"]
            ),
        },
    )
    reader = H5adReader(path)
    try:
        assays = reader.assay_feature_slices("feature_types")
    finally:
        reader.close()
    reader = H5adReader(path)
    try:
        H5adToZarr(
            reader,
            str(tmp_path / "split.zarr"),
            assay_split_key="feature_types",
            nthreads=1,
            shuffle_cells=False,
        ).dump()
    finally:
        reader.close()
    store = cytearc.DataStore(
        str(tmp_path / "split.zarr"),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
    )

    assert {
        name: (features.featureIds.tolist(), features.featureNames.tolist())
        for name, features in assays.items()
    } == {"RNA": (["7", "9"], ["7", "9"]), "ADT": (["8"], ["8"])}
    for name, features in assays.items():
        feats = getattr(store, name).feats
        np.testing.assert_array_equal(features.featureIds, feats.fetch_all("ids"))
        np.testing.assert_array_equal(features.featureNames, feats.fetch_all("names"))


@pytest.mark.parametrize("shuffle_cells", [False, True], ids=["file", "shuffled"])
@pytest.mark.parametrize(
    ("members", "reader_kwargs", "method", "message"),
    [
        pytest.param(
            {"obs/_index": np.array([b"caf\xe9", b"c1", b"c2", b"c3"])},
            {},
            "cell_ids",
            "H5AD cell IDs must be UTF-8 text; row 1 of the obs index '_index' of "
            "'refused.h5ad' holds b'caf\\xe9'",
            id="latin-1-cell-id",
        ),
        pytest.param(
            # The third ID ends within a character of two bytes.
            {"obs/_index": np.array([b"c0", b"c1", b"caf\xc3", b"c3"])},
            {},
            "cell_ids",
            "H5AD cell IDs must be UTF-8 text; row 3 of the obs index '_index' of "
            "'refused.h5ad' holds b'caf\\xc3'",
            id="cut-character-in-a-later-cell-id",
        ),
        pytest.param(
            # Row 2 holds the category that is not UTF-8, the fourth.
            {
                "obs/_index": {
                    "codes": np.array([0, 3, 1, 2], dtype=np.int8),
                    "categories": np.array([b"c0", b"c1", b"c2", b"c\xff"]),
                }
            },
            {},
            "cell_ids",
            "H5AD cell IDs must be UTF-8 text; row 2 of the obs index '_index' of "
            "'refused.h5ad' holds b'c\\xff'",
            id="latin-1-category-of-a-categorical-index",
        ),
        pytest.param(
            {"var/_index": np.array([b"g\xe9", b"g1", b"g2"])},
            {},
            "feat_ids",
            "H5AD feature IDs must be UTF-8 text; row 1 of the var index '_index' "
            "of 'refused.h5ad' holds b'g\\xe9'",
            id="latin-1-feature-id",
        ),
        pytest.param(
            # A UTF-16 surrogate, which UTF-8 never encodes.
            {"var/_index": np.array([b"g0", b"g1", b"\xed\xa0\x80"])},
            {},
            "feat_ids",
            "H5AD feature IDs must be UTF-8 text; row 3 of the var index '_index' "
            "of 'refused.h5ad' holds b'\\xed\\xa0\\x80'",
            id="surrogate-feature-id",
        ),
        pytest.param(
            {"var/symbol": np.array([b"A", b"B\xe9", b"C"])},
            {"feature_name_key": "symbol"},
            "feat_names",
            "H5AD feature names must be UTF-8 text; row 2 of the var column "
            "'symbol' of 'refused.h5ad' holds b'B\\xe9'",
            id="latin-1-feature-name",
        ),
        pytest.param(
            {
                "obs/_index": {
                    "values": np.array([b"c0", b"", b"c2", b"c3"]),
                    "mask": np.array([False, True, False, False]),
                }
            },
            {},
            "cell_ids",
            "H5AD cell IDs must contain non-empty values; row 2 of the obs index "
            "'_index' of 'refused.h5ad' is missing or blank",
            id="missing-cell-id",
        ),
        pytest.param(
            {"obs/_index": np.array([0.5, np.nan, 2.5, np.nan])},
            {},
            "cell_ids",
            "H5AD cell IDs must contain non-empty values; rows 2 and 4 of the obs "
            "index '_index' of 'refused.h5ad' are missing or blank",
            id="nan-in-a-float-cell-index",
        ),
        pytest.param(
            {"var/_index": np.array([np.nan, 1.5, 2.5], dtype=np.float32)},
            {},
            "feat_ids",
            "H5AD feature IDs must contain non-empty values; row 1 of the var index "
            "'_index' of 'refused.h5ad' is missing or blank",
            id="nan-in-a-float32-feature-index",
        ),
        pytest.param(
            # A nullable integer column shows its missing row as NaN.
            {
                "obs/barcode_number": {
                    "values": np.array([10, 11, 12, 13]),
                    "mask": np.array([False, False, True, False]),
                }
            },
            {"cell_ids_key": "barcode_number"},
            "cell_ids",
            "H5AD cell IDs must contain non-empty values; row 3 of the obs column "
            "'barcode_number' of 'refused.h5ad' is missing or blank",
            id="missing-number-in-a-cell-id-column",
        ),
        pytest.param(
            # A space, and an ideographic space, which counts as blank too.
            {"var/_index": np.array([b"g0", b" ", "　".encode()])},
            {},
            "feat_ids",
            "H5AD feature IDs must contain non-empty values; rows 2 and 3 of the "
            "var index '_index' of 'refused.h5ad' are missing or blank",
            id="blank-feature-ids",
        ),
        pytest.param(
            {"obs/_index": np.array([5, 5, 6, 7])},
            {},
            "cell_ids",
            "H5AD cell IDs must contain unique values; pass "
            "duplicate_cell_ids='make_unique' to rename repeated IDs",
            id="repeated-integer-cell-id",
        ),
    ],
)
def test_reader_refuses_the_ids_and_names_that_an_import_refuses(
    tmp_path, monkeypatch, members, reader_kwargs, method, message, shuffle_cells
) -> None:
    # A relative path, which the messages name as given.
    monkeypatch.chdir(tmp_path)
    path = _h5ad_with(Path("refused.h5ad"), members)
    temporary = tmp_path / "temporary"
    temporary.mkdir()

    reader = H5adReader(path, **reader_kwargs)
    try:
        with pytest.raises(ValueError) as refused_read:
            getattr(reader, method)()
    finally:
        reader.close()
    reader = H5adReader(path, temp_dir=str(temporary), **reader_kwargs)
    try:
        with pytest.raises(ValueError) as refused_import:
            H5adToZarr(reader, "refused.zarr", nthreads=1, shuffle_cells=shuffle_cells)
    finally:
        reader.close()

    # A ValueError of its own, not the UnicodeDecodeError of a decode.
    assert type(refused_read.value) is ValueError
    assert str(refused_read.value) == message
    assert type(refused_import.value) is ValueError
    assert str(refused_import.value) == message
    # The import refused before it wrote anything.
    assert not (tmp_path / "refused.zarr").exists()
    assert not list(temporary.iterdir())


@pytest.mark.parametrize(
    ("members", "reader_kwargs", "message"),
    [
        pytest.param(
            {"var/_index": np.array([b"g0", b"g\xe9", b"g2"])},
            {},
            "H5AD feature IDs must be UTF-8 text; row 2 of the var index '_index' "
            "of 'split.h5ad' holds b'g\\xe9'",
            id="feature-id",
        ),
        pytest.param(
            {"var/symbol": np.array([b"A", b"B", b"C\xe9"])},
            {"feature_name_key": "symbol"},
            "H5AD feature names must be UTF-8 text; row 3 of the var column "
            "'symbol' of 'split.h5ad' holds b'C\\xe9'",
            id="feature-name",
        ),
        pytest.param(
            {"var/_index": np.array([1.5, 2.5, np.nan])},
            {},
            "H5AD feature IDs must contain non-empty values; row 3 of the var index "
            "'_index' of 'split.h5ad' is missing or blank",
            id="missing-feature-id",
        ),
        pytest.param(
            {"var/description": _text(b"a gene", b"an antib\xf6dy", b"a gene")},
            {},
            "H5AD feature metadata must be UTF-8 text; row 2 of the var column "
            "'description' of 'split.h5ad' holds b'an antib\\xf6dy'",
            id="feature-metadata",
        ),
    ],
)
@pytest.mark.parametrize("shuffle_cells", [False, True], ids=["file", "shuffled"])
def test_an_import_split_by_feature_type_refuses_before_it_writes(
    tmp_path, monkeypatch, members, reader_kwargs, message, shuffle_cells
) -> None:
    monkeypatch.chdir(tmp_path)
    types = np.array([b"Gene Expression", b"Antibody Capture", b"Gene Expression"])
    path = _h5ad_with(Path("split.h5ad"), {"var/feature_types": types, **members})

    reader = H5adReader(path, **reader_kwargs)
    try:
        with pytest.raises(ValueError) as refused:
            H5adToZarr(
                reader,
                "split.zarr",
                assay_split_key="feature_types",
                nthreads=1,
                shuffle_cells=shuffle_cells,
            )
    finally:
        reader.close()

    assert type(refused.value) is ValueError
    assert str(refused.value) == message
    assert not (tmp_path / "split.zarr").exists()


@pytest.mark.parametrize(
    ("members", "message"),
    [
        pytest.param(
            {"obs/_index": np.array([b"c0", b"caf\xe9", b"c2", b"c3"])},
            "H5AD tables must hold UTF-8 text; row 2 of the obs index '_index' of "
            "'inspected.h5ad' holds b'caf\\xe9'",
            id="cell-index",
        ),
        pytest.param(
            {"var/_index": np.array([b"g\xe9", b"g1", b"g2"], dtype="S3")},
            "H5AD tables must hold UTF-8 text; row 1 of the var index '_index' of "
            "'inspected.h5ad' holds b'g\\xe9'",
            id="fixed-length-feature-index",
        ),
        pytest.param(
            # Not an ID or a name, but an import refuses it as metadata.
            {"var/description": np.array([b"tea", b"milk", b"caf\xe9"])},
            "H5AD tables must hold UTF-8 text; row 3 of the var column "
            "'description' of 'inspected.h5ad' holds b'caf\\xe9'",
            id="feature-column",
        ),
    ],
)
def test_inspection_refuses_text_that_is_not_utf8(
    tmp_path, monkeypatch, members, message
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _h5ad_with(Path("inspected.h5ad"), members)

    with pytest.raises(ValueError) as refused:
        inspect_h5ad(path)

    # Inspection used to raise the UnicodeDecodeError of a decode.
    assert type(refused.value) is ValueError
    assert str(refused.value) == message


@pytest.mark.parametrize(
    ("members", "title", "description", "warning"),
    [
        pytest.param(
            {
                "uns/title": np.bytes_(b"Caf\xe9 atlas"),
                "uns/description": np.bytes_(b"ok"),
            },
            None,
            "ok",
            "Leaving out uns/title of 'inspected.h5ad': it holds b'Caf\\xe9 atlas', "
            "which is not UTF-8 text",
            id="title",
        ),
        pytest.param(
            # One element of variable length, as AnnData writes a list. The
            # citation stands in for it, as for a file without a description.
            {
                "uns/description": _text(b"\xffcells"),
                "uns/citation": np.bytes_(b"Smith 2026"),
            },
            None,
            "Smith 2026",
            "Leaving out uns/description of 'inspected.h5ad': it holds "
            "b'\\xffcells', which is not UTF-8 text",
            id="description",
        ),
        pytest.param(
            {"uns/citation": np.bytes_(b"S\xe3o Paulo")},
            None,
            None,
            "Leaving out uns/citation of 'inspected.h5ad': it holds b'S\\xe3o Paulo', "
            "which is not UTF-8 text",
            id="citation",
        ),
    ],
)
def test_inspection_leaves_out_uns_text_that_is_not_utf8_and_the_file_imports(
    tmp_path, monkeypatch, members, title, description, warning
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _h5ad_with(Path("inspected.h5ad"), members)

    # An import never reads them, so inspection used to refuse a file that
    # imports, with the UnicodeDecodeError of a decode.
    with _warnings() as messages:
        inspection = inspect_h5ad(path)

    assert (inspection.title, inspection.description) == (title, description)
    assert [message.rstrip("\n") for message in messages] == [warning]
    store, _result = _import(path, tmp_path / "inspected.zarr")
    assert store.cells.fetch_all("ids").tolist() == _WRITTEN_IDS["cell_ids"]


@pytest.mark.parametrize("shuffle_cells", [False, True], ids=["file", "shuffled"])
@pytest.mark.parametrize(
    ("members", "reader_kwargs", "columns", "message"),
    [
        pytest.param(
            {"obs/note": _text(b"tea", b"caf\xe9", b"milk", b"water")},
            {},
            "get_cell_columns",
            "H5AD cell metadata must be UTF-8 text; row 2 of the obs column 'note' "
            "of 'refused.h5ad' holds b'caf\\xe9'",
            id="text-cell-column",
        ),
        pytest.param(
            # The third value ends within a character of two bytes.
            {"obs/note": np.array([b"tea", b"milk", b"caf\xc3", b"water"])},
            {},
            "get_cell_columns",
            "H5AD cell metadata must be UTF-8 text; row 3 of the obs column 'note' "
            "of 'refused.h5ad' holds b'caf\\xc3'",
            id="cut-character-in-a-fixed-width-column",
        ),
        pytest.param(
            # Row 2 is missing, so an import never stores its bytes.
            {
                "obs/note": {
                    "values": _text(b"tea", b"\xff", b"caf\xe9", b"water"),
                    "mask": np.array([False, True, False, False]),
                }
            },
            {},
            "get_cell_columns",
            "H5AD cell metadata must be UTF-8 text; row 3 of the obs column 'note' "
            "of 'refused.h5ad' holds b'caf\\xe9'",
            id="nullable-column-past-a-missing-row",
        ),
        pytest.param(
            {
                "obs/kind": {
                    "codes": np.array([0, 1, 0, 1], dtype=np.int8),
                    "categories": _text(b"x", b"y\xe9"),
                }
            },
            {},
            "get_cell_columns",
            "H5AD cell metadata must be UTF-8 text; category 2 of the obs column "
            "'kind' of 'refused.h5ad' holds b'y\\xe9'",
            id="categorical-column",
        ),
        pytest.param(
            # No row holds the third category, which an import stores too.
            {
                "obs/kind": {
                    "codes": np.array([0, 0, 1, -1], dtype=np.int8),
                    "categories": _text(b"x", b"y", b"z\xe9"),
                }
            },
            {},
            "get_cell_columns",
            "H5AD cell metadata must be UTF-8 text; category 3 of the obs column "
            "'kind' of 'refused.h5ad' holds b'z\\xe9'",
            id="unused-category",
        ),
        pytest.param(
            # AnnData 0.7 kept the categories of codes beside the columns.
            {
                "obs/batch": np.array([0, 1, 0, 1], dtype=np.int8),
                "obs/__categories": {"batch": np.array([b"one", b"tw\xf6"])},
            },
            {},
            "get_cell_columns",
            "H5AD cell metadata must be UTF-8 text; category 2 of the obs column "
            "'batch' of 'refused.h5ad' holds b'tw\\xf6'",
            id="categories-beside-the-columns",
        ),
        pytest.param(
            # AnnData 0.6 kept them in uns.
            {
                "obs/batch": np.array([1, 0, 1, 0], dtype=np.int8),
                "uns/batch_categories": np.array([b"on\xe9", b"two"]),
            },
            {},
            "get_cell_columns",
            "H5AD cell metadata must be UTF-8 text; category 1 of the obs column "
            "'batch' of 'refused.h5ad' holds b'on\\xe9'",
            id="categories-in-uns",
        ),
        pytest.param(
            # AnnData 0.6 wrote a table as one compound dataset.
            {
                "obs": np.array(
                    [
                        (b"c0", b"tea"),
                        (b"c1", b"milk"),
                        (b"c2", b"soda"),
                        (b"c3", b"t\xe9"),
                    ],
                    dtype=[("index", "S2"), ("note", "S4")],
                )
            },
            {},
            "get_cell_columns",
            "H5AD cell metadata must be UTF-8 text; row 4 of the obs column 'note' "
            "of 'refused.h5ad' holds b't\\xe9'",
            id="compound-table",
        ),
        pytest.param(
            # The cell IDs come from another column, so the index is metadata.
            {
                "obs/_index": _text(b"c0", b"caf\xe9", b"c2", b"c3"),
                "obs/barcode": _text(b"b0", b"b1", b"b2", b"b3"),
            },
            {"cell_ids_key": "barcode"},
            "get_cell_columns",
            "H5AD cell metadata must be UTF-8 text; row 2 of the obs index '_index' "
            "of 'refused.h5ad' holds b'caf\\xe9'",
            id="index-as-metadata",
        ),
        pytest.param(
            {
                "obs/_index": {
                    "codes": np.array([0, 1, 2, 3], dtype=np.int8),
                    "categories": _text(b"c0", b"c1", b"c\xff", b"c3"),
                },
                "obs/barcode": _text(b"b0", b"b1", b"b2", b"b3"),
            },
            {"cell_ids_key": "barcode"},
            "get_cell_columns",
            "H5AD cell metadata must be UTF-8 text; category 3 of the obs index "
            "'_index' of 'refused.h5ad' holds b'c\\xff'",
            id="categorical-index-as-metadata",
        ),
        pytest.param(
            {"var/description": _text(b"tea", b"milk", b"caf\xe9")},
            {},
            "get_feat_columns",
            "H5AD feature metadata must be UTF-8 text; row 3 of the var column "
            "'description' of 'refused.h5ad' holds b'caf\\xe9'",
            id="text-feature-column",
        ),
        pytest.param(
            {
                "var/chromosome": {
                    "codes": np.array([0, 1, 0], dtype=np.int8),
                    "categories": _text(b"chr\xe9", b"chr2"),
                }
            },
            {},
            "get_feat_columns",
            "H5AD feature metadata must be UTF-8 text; category 1 of the var "
            "column 'chromosome' of 'refused.h5ad' holds b'chr\\xe9'",
            id="categorical-feature-column",
        ),
        pytest.param(
            {"obs/leiden": _text(b"1", b"2", b"\xff", b"1")},
            {"cluster_keys": ("leiden",)},
            None,
            "H5AD cluster labels must be UTF-8 text; row 3 of the obs column "
            "'leiden' of 'refused.h5ad' holds b'\\xff'",
            id="text-cluster-labels",
        ),
        pytest.param(
            {
                "obs/leiden": {
                    "codes": np.array([0, 1, 1, 0], dtype=np.int8),
                    "categories": _text(b"T cell", b"B c\xe9ll"),
                }
            },
            {"cluster_keys": ("leiden",)},
            None,
            "H5AD cluster labels must be UTF-8 text; category 2 of the obs column "
            "'leiden' of 'refused.h5ad' holds b'B c\\xe9ll'",
            id="categorical-cluster-labels",
        ),
    ],
)
def test_an_import_refuses_metadata_text_that_is_not_utf8_before_it_writes(
    tmp_path, monkeypatch, members, reader_kwargs, columns, message, shuffle_cells
) -> None:
    # A relative path, which the messages name as given.
    monkeypatch.chdir(tmp_path)
    path = _h5ad_with(Path("refused.h5ad"), members)
    temporary = tmp_path / "temporary"
    temporary.mkdir()

    if columns is not None:
        reader = H5adReader(path, **reader_kwargs)
        try:
            with pytest.raises(ValueError) as refused_read:
                dict(getattr(reader, columns)())
        finally:
            reader.close()
        assert type(refused_read.value) is ValueError
        assert str(refused_read.value) == message
    reader = H5adReader(path, temp_dir=str(temporary), **reader_kwargs)
    try:
        with pytest.raises(ValueError) as refused_import:
            H5adToZarr(reader, "refused.zarr", nthreads=1, shuffle_cells=shuffle_cells)
    finally:
        reader.close()

    # A ValueError of its own: the import used to raise a UnicodeDecodeError,
    # or a ValueError that named no row, after it created the destination.
    assert type(refused_import.value) is ValueError
    assert str(refused_import.value) == message
    assert not (tmp_path / "refused.zarr").exists()
    assert not list(temporary.iterdir())


@pytest.mark.parametrize("block_rows", [1, 2, 3, 65_536])
def test_the_text_check_reads_blocks_of_rows_and_counts_rows_across_them(
    tmp_path, monkeypatch, block_rows
) -> None:
    import cytearc.readers._h5ad_columns as columns_module
    from cytearc.readers._h5ad_columns import check_column_text

    monkeypatch.setattr(columns_module, "_TEXT_CHECK_ROWS", block_rows)
    decoded: list[int] = []
    first_undecodable = columns_module._first_undecodable

    def counted(values: list[bytes]) -> int | None:
        decoded.append(len(values))
        return first_undecodable(values)

    monkeypatch.setattr(columns_module, "_first_undecodable", counted)
    path = tmp_path / "blocks.h5"
    with h5py.File(path, "w") as h5:
        obs = h5.create_group("obs")
        obs.create_dataset("text", data=_text(b"a", b"b", b"c", b"d", b"caf\xe9", b"f"))
        # The missing rows 1 and 3 hold bytes that are not UTF-8 text.
        nullable = obs.create_group("nullable")
        nullable.create_dataset(
            "values", data=_text(b"\xff", b"b", b"\xfe", b"d", b"e", b"\xfd")
        )
        nullable.create_dataset(
            "mask", data=np.array([True, False, True, False, False, False])
        )
        kind = obs.create_group("kind")
        kind.create_dataset("codes", data=np.zeros(6, dtype=np.int8))
        kind.create_dataset("categories", data=_text(b"a", b"b", b"c", b"d\xe9"))
        # An import skips a column of two dimensions, and a missing-value mask
        # that does not align with its values, so the check passes them.
        obs.create_dataset("table", data=np.array([[b"\xff"] * 2] * 6))
        misaligned = obs.create_group("misaligned")
        misaligned.create_dataset("values", data=_text(*[b"\xff"] * 6))
        misaligned.create_dataset("mask", data=np.zeros(5, dtype=bool))
        # Codes with numbers for categories hold no text.
        obs.create_dataset("level", data=np.array([0, 1, 0, 1, 0, 1], dtype=np.int8))
        obs.create_group("__categories").create_dataset("level", data=[0.5, 1.5])
    with h5py.File(path, "r") as h5:

        def check(key: str) -> None:
            check_column_text(
                h5,
                h5["obs"],
                key,
                ("__categories",),
                expectation="Text must be UTF-8",
                place=f"the column {key!r}",
            )

        for key, location in (
            ("text", "row 5 of the column 'text' holds b'caf\\xe9'"),
            ("nullable", "row 6 of the column 'nullable' holds b'\\xfd'"),
            ("kind", "category 4 of the column 'kind' holds b'd\\xe9'"),
        ):
            with pytest.raises(ValueError) as refused:
                check(key)
            assert str(refused.value) == f"Text must be UTF-8; {location}"
        for key in ("table", "misaligned", "level"):
            check(key)

    # No check held more than a block of rows at a time.
    assert decoded and max(decoded) <= block_rows


def test_text_decodes_in_one_pass_exactly_when_each_value_decodes() -> None:
    from cytearc.readers._h5ad_columns import _first_undecodable

    def first_undecodable(values: list[bytes]) -> int | None:
        for position, value in enumerate(values):
            try:
                value.decode("utf-8")
            except UnicodeDecodeError:
                return position
        return None

    # Characters split across two values decode once the values are put
    # together, but neither value decodes alone.
    assert _first_undecodable([b"caf\xc3", b"\xa9"]) == 0
    assert _first_undecodable([b"\xe2\x82", b"\xac"]) == 0
    assert _first_undecodable([b"\xf0\x9f\x98", b"\x80"]) == 0
    assert _first_undecodable([b"ok", b"\xa9"]) == 1
    assert _first_undecodable([b"a\nb", b"", "é".encode(), "😀".encode()]) is None
    assert _first_undecodable([]) is None
    # Lead, continuation and invalid bytes beside ASCII and the joining byte.
    alphabet = [0x00, 0x0A, 0x41, 0x80, 0xA9, 0xBF, 0xC0, 0xC3, 0xE2, 0xED, 0xF0, 0xFF]
    rng = np.random.default_rng(7)
    for _ in range(3_000):
        values = [
            bytes(rng.choice(alphabet, size=int(rng.integers(0, 4))).tolist())
            for _ in range(int(rng.integers(0, 5)))
        ]
        assert _first_undecodable(values) == first_undecodable(values), values


@pytest.mark.parametrize("shuffle_cells", [False, True], ids=["file", "shuffled"])
def test_reader_metadata_columns_are_the_text_that_an_import_stores(
    tmp_path, shuffle_cells
) -> None:
    path = _h5ad_with(
        tmp_path / "text.h5ad",
        {
            "obs/note": _text("café".encode(), b"tea", "中文".encode(), b""),
            "obs/fixed": np.array([b"a", b"bb", b"a", b"ccc"]),
            "obs/maybe": {
                "values": _text(b"x", b"", "ü".encode(), b"z"),
                "mask": np.array([False, True, False, False]),
            },
            "obs/kind": {
                "codes": np.array([1, -1, 0, 1], dtype=np.int8),
                "categories": _text("Zellé".encode(), b"T cell"),
            },
            "obs/count": np.array([3, 1, 4, 1]),
            "var/chromosome": np.array([b"1", b"2", b"X"]),
        },
    )
    expected = {
        "note": ["café", "tea", "中文", ""],
        "fixed": ["a", "bb", "a", "ccc"],
        "maybe": ["x", None, "ü", "z"],
        "kind": ["T cell", None, "Zellé", "T cell"],
    }
    reader = H5adReader(path)
    try:
        cells = dict(reader.get_cell_columns())
        features = dict(reader.get_feat_columns())
    finally:
        reader.close()

    for name, values in expected.items():
        # The reader used to give byte strings, such as b"caf\xc3\xa9".
        assert cells[name].dtype == object, name
        assert cells[name].tolist() == values, name
    assert cells["count"].tolist() == [3, 1, 4, 1]
    assert features["chromosome"].tolist() == ["1", "2", "X"]

    reader = H5adReader(path)
    try:
        H5adToZarr(
            reader,
            str(tmp_path / "text.zarr"),
            nthreads=1,
            shuffle_cells=shuffle_cells,
        ).dump()
    finally:
        reader.close()
    store = cytearc.DataStore(
        str(tmp_path / "text.zarr"), min_features_per_cell=0, nthreads=1
    )
    stored = store.cells.to_pandas_dataframe(list(expected))
    # The stored cells in the file's order.
    rows = (
        np.argsort(store.cells.fetch_all("sourceRow")) if shuffle_cells else slice(None)
    )
    for name, values in expected.items():
        held = stored[name].to_numpy()[rows]
        assert [None if pd.isna(value) else value for value in held] == values, name
    assert store.RNA.feats.fetch_all("chromosome").tolist() == ["1", "2", "X"]


def test_categories_that_are_not_utf8_are_refused_by_position(tmp_path) -> None:
    path = tmp_path / "categories.h5"
    with h5py.File(path, "w") as h5:
        obs = h5.create_group("obs")
        kind = obs.create_group("kind")
        kind.create_dataset("codes", data=np.array([0, 1], dtype=np.int8))
        kind.create_dataset("categories", data=_text(b"a", b"b\xe9"))
        obs.create_dataset("batch", data=np.array([1, 1], dtype=np.int8))
        legacy = obs.create_group("__categories")
        legacy.create_dataset("batch", data=np.array([b"\xff", b"b"]))
    with h5py.File(path, "r") as h5:
        for key, location in (
            ("kind", "category 2 of the column 'kind' holds b'b\\xe9'"),
            # No row holds the first category.
            ("batch", "category 1 of the column 'batch' holds b'\\xff'"),
        ):
            with pytest.raises(ValueError) as refused:
                column_categories(
                    h5,
                    h5["obs"],
                    key,
                    ("__categories",),
                    expectation="Categories must be UTF-8 text",
                    place=f"the column {key!r}",
                )
            # It used to raise the UnicodeDecodeError of a decode.
            assert type(refused.value) is ValueError
            assert str(refused.value) == (f"Categories must be UTF-8 text; {location}")


def test_categories_read_as_text_and_python_numbers_and_other_values_as_read(
    tmp_path,
) -> None:
    path = tmp_path / "categories.h5"
    with h5py.File(path, "w") as h5:
        obs = h5.create_group("obs")
        for key, categories in (
            ("text", _text(b"T cell", b"B cell")),
            ("number", np.array([7, 3], dtype=np.int16)),
        ):
            group = obs.create_group(key)
            group.create_dataset("codes", data=np.array([0, 1], dtype=np.int8))
            group.create_dataset("categories", data=categories)
        # h5py reads each value of a variable-length sequence as an array.
        sequences = obs.create_group("sequence")
        sequences.create_dataset("codes", data=np.array([0, 1], dtype=np.int8))
        stored = sequences.create_dataset(
            "categories", (2,), dtype=h5py.vlen_dtype(np.int64)
        )
        stored[0] = np.array([1, 2])
        stored[1] = np.array([3])
    with h5py.File(path, "r") as h5:
        found = {
            key: column_categories(
                h5,
                h5["obs"],
                key,
                ("__categories",),
                expectation="Categories must be UTF-8 text",
                place=f"the column {key!r}",
            )
            for key in ("text", "number", "sequence")
        }
    assert found["text"] == H5adCategories(("T cell", "B cell"), False)
    assert found["number"] == H5adCategories((7, 3), False)
    assert [type(value) for value in found["number"].values] == [int, int]
    first, second = found["sequence"].values
    assert isinstance(first, np.ndarray) and first.tolist() == [1, 2]
    assert isinstance(second, np.ndarray) and second.tolist() == [3]


@pytest.mark.parametrize("shuffle_cells", [False, True], ids=["file", "shuffled"])
@pytest.mark.parametrize(
    ("types", "message"),
    [
        pytest.param(
            np.array([b"Gene Expression", b"Antibody Captur\xe9", b"Gene Expression"]),
            "H5AD feature types must be UTF-8 text; row 2 of the var column "
            "'feature_types' of 'split.h5ad' holds b'Antibody Captur\\xe9'",
            id="text",
        ),
        pytest.param(
            # The feature types are read row by row, so the refusal names the
            # first row that holds the category.
            {
                "codes": np.array([0, 0, 1], dtype=np.int8),
                "categories": _text(b"Gene Expression", b"Antibody Captur\xe9"),
            },
            "H5AD feature types must be UTF-8 text; row 3 of the var column "
            "'feature_types' of 'split.h5ad' holds b'Antibody Captur\\xe9'",
            id="categorical",
        ),
    ],
)
def test_a_split_import_refuses_feature_types_that_are_not_utf8(
    tmp_path, monkeypatch, types, message, shuffle_cells
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _h5ad_with(Path("split.h5ad"), {"var/feature_types": types})

    reader = H5adReader(path)
    try:
        with pytest.raises(ValueError) as refused_read:
            reader.feature_types("feature_types")
        with pytest.raises(ValueError) as refused_import:
            H5adToZarr(
                reader,
                "split.zarr",
                assay_split_key="feature_types",
                nthreads=1,
                shuffle_cells=shuffle_cells,
            )
    finally:
        reader.close()

    for refused in (refused_read, refused_import):
        # Both used to raise the UnicodeDecodeError of a decode.
        assert type(refused.value) is ValueError
        assert str(refused.value) == message
    assert not (tmp_path / "split.zarr").exists()


@pytest.mark.parametrize("shuffle_cells", [False, True], ids=["file", "shuffled"])
def test_an_import_refuses_a_membership_column_name_that_is_not_utf8(
    tmp_path, monkeypatch, shuffle_cells
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _h5ad_with(
        Path("refused.h5ad"),
        {"uns/cytearc/assayMembership/RNA": np.bytes_(b"RNA\xe9_I")},
    )
    message = (
        "H5AD membership column names must be UTF-8 text; "
        "uns/cytearc/assayMembership/RNA of 'refused.h5ad' holds b'RNA\\xe9_I'"
    )

    reader = H5adReader(path)
    try:
        with pytest.raises(ValueError) as refused_read:
            reader.assay_membership()
        with pytest.raises(ValueError) as refused_import:
            H5adToZarr(reader, "refused.zarr", nthreads=1, shuffle_cells=shuffle_cells)
    finally:
        reader.close()

    for refused in (refused_read, refused_import):
        assert type(refused.value) is ValueError
        assert str(refused.value) == message
    assert not (tmp_path / "refused.zarr").exists()


def test_a_membership_declaration_of_a_number_is_refused_as_its_text(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _h5ad_with(Path("number.h5ad"), {"uns/cytearc/assayMembership/RNA": 7})

    reader = H5adReader(path)
    try:
        with pytest.raises(ValueError) as refused:
            reader.assay_membership()
    finally:
        reader.close()

    # The number reads as the text '7', which is not the column 'RNA_I'.
    assert str(refused.value) == (
        "uns['cytearc']['assayMembership'] maps 'RNA' to '7'; CyteArc declares "
        "each assay's membership column as '<assay>_I'. Remove the entry from the "
        "file or correct it."
    )


def test_colors_that_are_not_utf8_import_as_colors_that_are_not_colors(
    tmp_path,
) -> None:
    kind = {
        "codes": np.array([0, 1, 0, 1], dtype=np.int8),
        "categories": _text(b"a", b"b"),
    }
    displays = {}
    warnings = {}
    for name, members in (
        ("plain", {"obs/kind": kind}),
        (
            "nonsense",
            {"obs/kind": kind, "uns/kind_colors": _text(b"#f00", b"nonsense")},
        ),
        ("bytes", {"obs/kind": kind, "uns/kind_colors": _text(b"#f00", b"#0\xe90")}),
    ):
        path = _h5ad_with(tmp_path / f"{name}.h5ad", members)
        reader = H5adReader(path)
        try:
            with _warnings() as messages:
                H5adToZarr(
                    reader,
                    str(tmp_path / f"{name}.zarr"),
                    nthreads=1,
                    shuffle_cells=False,
                ).dump()
        finally:
            reader.close()
        root = zarr.open_group(str(tmp_path / f"{name}.zarr"), mode="r")
        displays[name] = root["cellData/kind"].attrs["display"]
        warnings[name] = messages

    # Colors that are not colors take the palette, without a warning, as a
    # file without colors does; the import used to stop at bytes that are not
    # UTF-8 with a UnicodeDecodeError after it created the store.
    assert displays["bytes"] == displays["nonsense"] == displays["plain"]
    assert warnings["bytes"] == warnings["nonsense"] == warnings["plain"]


def _rename(table: str, name: bytes) -> Any:
    """Return an edit that adds a member named ``name`` to ``table``."""

    def edit(h5: h5py.File) -> None:
        h5[table].create_dataset(name, data=np.arange(len(h5[table]["_index"])))

    return edit


def _attribute(table: str, name: str, value: Any, *, text: bool = False) -> Any:
    """Return an edit that sets an attribute of ``table``, as text of variable length with ``text``."""

    def edit(h5: h5py.File) -> None:
        if text:
            h5[table].attrs.create(name, value, dtype=h5py.string_dtype())
        else:
            h5[table].attrs[name] = value

    return edit


def _nested_member(h5: h5py.File) -> None:
    # Old AnnData nested a column named "a/b" into groups, which also hold a
    # member whose name is not UTF-8 text.
    group = h5["obs"].create_group("a")
    group.create_dataset("b", data=np.arange(4))
    group.create_dataset(b"x\xe9", data=np.arange(4))
    h5["obs"].attrs.create("column-order", ["a/b"], dtype=h5py.string_dtype())


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        pytest.param(
            _attribute("obs", "column-order", np.array([b"k\xe9"])),
            "H5AD column names must be UTF-8 text; the column-order attribute of "
            "the obs table of 'refused.h5ad' holds b'k\\xe9'",
            id="fixed-width-column-order",
        ),
        pytest.param(
            # h5py reads text of variable length that is not UTF-8 as lone
            # surrogates, which reached the store's column names unrefused.
            _attribute(
                "var", "column-order", np.array([b"k\xe9"], dtype=object), text=True
            ),
            "H5AD column names must be UTF-8 text; the column-order attribute of "
            "the var table of 'refused.h5ad' holds b'k\\xe9'",
            id="column-order-of-variable-length",
        ),
        pytest.param(
            _attribute("var", "_index", np.bytes_(b"_ind\xe9")),
            "H5AD column names must be UTF-8 text; the _index attribute of the var "
            "table of 'refused.h5ad' holds b'_ind\\xe9'",
            id="fixed-width-index-attribute",
        ),
        pytest.param(
            _attribute("obs", "_index", b"_ind\xe9", text=True),
            "H5AD column names must be UTF-8 text; the _index attribute of the obs "
            "table of 'refused.h5ad' holds b'_ind\\xe9'",
            id="index-attribute-of-variable-length",
        ),
        pytest.param(
            _rename("obs", b"n\xe9"),
            "H5AD column names must be UTF-8 text; the obs table of 'refused.h5ad' "
            "holds a member named b'n\\xe9'",
            id="cell-column-name",
        ),
        pytest.param(
            _rename("var", b"n\xe9"),
            "H5AD column names must be UTF-8 text; the var table of 'refused.h5ad' "
            "holds a member named b'n\\xe9'",
            id="feature-column-name",
        ),
        pytest.param(
            _nested_member,
            "H5AD column names must be UTF-8 text; the obs table of 'refused.h5ad' "
            "holds a member named b'a/x\\xe9'",
            id="nested-member-name",
        ),
    ],
)
def test_table_names_that_are_not_utf8_are_refused_when_the_reader_opens(
    tmp_path, monkeypatch, edit, message
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _h5ad_with(Path("refused.h5ad"), {})
    with h5py.File(path, "r+") as h5:
        edit(h5)

    with pytest.raises(ValueError) as refused_read:
        H5adReader(path)
    with pytest.raises(ValueError) as refused_inspection:
        inspect_h5ad(path)

    # Both used to raise the UnicodeDecodeError of a decode or a TypeError, or
    # took a name of variable length for a column name that no store holds.
    for refused in (refused_read, refused_inspection):
        assert type(refused.value) is ValueError
        assert str(refused.value) == message


def test_names_outside_the_tables_that_are_not_utf8_leave_the_file_importable(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _h5ad_with(Path("names.h5ad"), {})
    with h5py.File(path, "r+") as h5:
        layers = h5.require_group("layers")
        layers.create_dataset(b"c\xe9", data=np.ones((4, 3), dtype=np.float32))
        layers.create_dataset("counts", data=np.ones((4, 3), dtype=np.float32))
        sparse = layers.create_group(b"s\xe9")
        matrix = csr_matrix(np.ones((4, 3), dtype=np.float32))
        sparse.attrs.update({"encoding-type": "csr_matrix", "shape": matrix.shape})
        for part in ("data", "indices", "indptr"):
            sparse.create_dataset(part, data=getattr(matrix, part))
        h5["uns"].create_dataset(b"u\xe9", data=np.arange(3))
        h5["obsm"].create_dataset(b"X_\xe9", data=np.ones((4, 2)))

    with _warnings() as messages:
        inspection = inspect_h5ad(path)

    # A matrix that no reader argument can name is no candidate, and the
    # search for matrices used to raise a TypeError at its path.
    warned = [message.rstrip("\n") for message in messages]
    for layer in ("c", "s"):
        assert (
            f"Ignoring matrix candidate b'layers/{layer}\\xe9': its path is not UTF-8 "
            "text" in warned
        )
    assert inspection.layers == ("counts",)
    assert "layers/counts" in inspection.matrixCandidates
    store, _result = _import(path, tmp_path / "names.zarr")
    assert store.cells.fetch_all("ids").tolist() == _WRITTEN_IDS["cell_ids"]


@pytest.mark.parametrize(
    ("members", "warning", "ids"),
    [
        pytest.param(
            {"obs/_index": np.array([0.5, 1.5, 2.5, 3.5])},
            "The cell IDs in the obs index '_index' are floating-point numbers, and "
            "no other column holds unique text IDs. An import stores the text of "
            "each as the cell's ID, such as '0.5'",
            {"cell_ids": ["0.5", "1.5", "2.5", "3.5"]},
            id="floating-point-cell-ids",
        ),
        pytest.param(
            # The example is the text of the float32 value as an import stores it.
            {"obs/_index": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)},
            "The cell IDs in the obs index '_index' are floating-point numbers, and "
            "no other column holds unique text IDs. An import stores the text of "
            "each as the cell's ID, such as '0.10000000149011612'",
            {
                "cell_ids": [
                    "0.10000000149011612",
                    "0.20000000298023224",
                    "0.30000001192092896",
                    "0.4000000059604645",
                ]
            },
            id="float32-cell-ids",
        ),
        pytest.param(
            {"obs/_index": np.array([0.5, np.nan, 2.5, np.nan])},
            "The cell IDs in the obs index '_index' include missing or blank "
            "values, the first in row 2, and no other column holds unique text "
            "IDs. An import refuses cells without an ID",
            "H5AD cell IDs must contain non-empty values; rows 2 and 4 of the obs "
            "index '_index' of 'inspected.h5ad' are missing or blank",
            id="nan-cell-ids",
        ),
        pytest.param(
            # Text with two missing rows, which inspection does not take for
            # unique IDs.
            {
                "obs/_index": {
                    "values": _text(b"c0", b"", b"c2", b""),
                    "mask": np.array([False, True, False, True]),
                }
            },
            "The cell IDs in the obs index '_index' include missing or blank "
            "values, the first in row 2, and no other column holds unique text "
            "IDs. An import refuses cells without an ID",
            "H5AD cell IDs must contain non-empty values; rows 2 and 4 of the obs "
            "index '_index' of 'inspected.h5ad' are missing or blank",
            id="missing-text-cell-ids",
        ),
        pytest.param(
            {"obs/_index": np.array([1.5, 1.5, 2.5, 3.5])},
            "The cell IDs in the obs index '_index' repeat values, and no other "
            "column holds unique text IDs. Pass duplicate_cell_ids='make_unique' "
            "to H5adReader, also through H5adReader.from_inspect, to rename "
            "repeated IDs",
            "H5AD cell IDs must contain unique values; pass "
            "duplicate_cell_ids='make_unique' to rename repeated IDs",
            id="repeated-floating-point-cell-ids",
        ),
        pytest.param(
            {
                "obs/_index": {
                    "values": _text(b"c0", b"c1", b"c2", b"c3"),
                    "mask": np.zeros(3, dtype=bool),
                }
            },
            "The cell IDs in the obs index '_index' cannot be read, and no other "
            "column holds unique text IDs. An import refuses the file",
            "Column '/obs/_index' has a missing-value mask that does not align "
            "with its values",
            id="unreadable-cell-ids",
        ),
        pytest.param(
            {"obs/_index": None},
            "No cell ID column found; generated IDs will be used",
            {"cell_ids": ["cell_0", "cell_1", "cell_2", "cell_3"]},
            id="no-cell-index",
        ),
        pytest.param(
            {"var/_index": np.array([7, 8, 9])},
            "The feature IDs in the var index '_index' are integers, and no other "
            "column holds text feature IDs or names. An import stores the text of "
            "each as the feature's ID and name, such as '7'",
            {"feat_ids": ["7", "8", "9"], "feat_names": ["7", "8", "9"]},
            id="integer-feature-ids",
        ),
        pytest.param(
            {"var/_index": np.array([np.nan, 1.5, 2.5])},
            "The feature IDs in the var index '_index' include missing or blank "
            "values, the first in row 1, and no other column holds text feature "
            "IDs or names. An import refuses features without an ID",
            "H5AD feature IDs must contain non-empty values; row 1 of the var "
            "index '_index' of 'inspected.h5ad' is missing or blank",
            id="nan-feature-ids",
        ),
        pytest.param(
            {"var/_index": np.array([7, 7, 9])},
            "The feature IDs in the var index '_index' repeat values, and no other "
            "column holds text feature IDs or names. Pass "
            "duplicate_feature_ids='make_unique' to H5adReader, also through "
            "H5adReader.from_inspect, to rename repeated IDs",
            "H5AD feature IDs must contain unique values; pass "
            "duplicate_feature_ids='make_unique' to rename repeated IDs",
            id="repeated-integer-feature-ids",
        ),
        pytest.param(
            {"var/_index": None},
            "No feature ID or name column found; generated IDs will be used",
            {"feat_ids": ["feature_0", "feature_1", "feature_2"]},
            id="no-feature-index",
        ),
        pytest.param(
            # Unique integer and text IDs need no warning.
            {"obs/_index": np.array([5, 6, 7, 8])},
            None,
            {"cell_ids": ["5", "6", "7", "8"], "feat_ids": ["g0", "g1", "g2"]},
            id="integer-cell-ids",
        ),
        pytest.param(
            # One missing value leaves the other IDs unique, so inspection
            # takes the index for the IDs.
            {
                "obs/_index": {
                    "values": _text(b"c0", b"", b"c2", b"c3"),
                    "mask": np.array([False, True, False, False]),
                }
            },
            "The cell IDs in the obs index '_index' include missing or blank "
            "values, the first in row 2. An import refuses cells without an ID",
            "H5AD cell IDs must contain non-empty values; row 2 of the obs index "
            "'_index' of 'inspected.h5ad' is missing or blank",
            id="one-missing-cell-id-of-unique-ones",
        ),
        pytest.param(
            {"obs/_index": _text(b"c0", b"c1", b"c2", b" ")},
            "The cell IDs in the obs index '_index' include missing or blank "
            "values, the first in row 4. An import refuses cells without an ID",
            "H5AD cell IDs must contain non-empty values; row 4 of the obs index "
            "'_index' of 'inspected.h5ad' is missing or blank",
            id="one-blank-cell-id-of-unique-ones",
        ),
        pytest.param(
            # The index repeats, so inspection takes the barcodes for the IDs.
            {
                "obs/_index": _text(b"c0", b"c0", b"c2", b"c3"),
                "obs/barcode": _text(b"AC", b"", b"GT", b"TT"),
            },
            "The cell IDs in the obs column 'barcode' include missing or blank "
            "values, the first in row 2. An import refuses cells without an ID",
            "H5AD cell IDs must contain non-empty values; row 2 of the obs column "
            "'barcode' of 'inspected.h5ad' is missing or blank",
            id="blank-cell-id-in-a-barcode-column",
        ),
        pytest.param(
            {
                "var/_index": {
                    "values": _text(b"g0", b"g1", b""),
                    "mask": np.array([False, False, True]),
                }
            },
            "The feature IDs in the var index '_index' include missing or blank "
            "values, the first in row 3. An import refuses features without an ID",
            "H5AD feature IDs must contain non-empty values; row 3 of the var "
            "index '_index' of 'inspected.h5ad' is missing or blank",
            id="one-missing-feature-id-of-unique-ones",
        ),
        pytest.param(
            # An ID column wins over the index of symbols.
            {"var/gene_ids": _text(b"E0", b" ", b"E2")},
            "The feature IDs in the var column 'gene_ids' include missing or "
            "blank values, the first in row 2. An import refuses features without "
            "an ID",
            "H5AD feature IDs must contain non-empty values; row 2 of the var "
            "column 'gene_ids' of 'inspected.h5ad' is missing or blank",
            id="blank-feature-id-in-an-id-column",
        ),
    ],
)
def test_inspection_warns_what_an_import_does_with_an_index_of_ids(
    tmp_path, monkeypatch, members, warning, ids
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _h5ad_with(Path("inspected.h5ad"), members)

    with _warnings() as messages:
        inspection = inspect_h5ad(path)

    # Inspection used to warn that the IDs of a floating-point index repeat,
    # and that a feature index of numbers left the features generated IDs,
    # and it took IDs with one missing or blank value without a warning.
    id_warnings = [message.rstrip("\n") for message in messages if " ID" in message]
    assert id_warnings == ([] if warning is None else [warning])
    reader = H5adReader.from_inspect(inspection)
    try:
        if isinstance(ids, str):
            with pytest.raises(ValueError, match=re.escape(ids)):
                reader._check_ids_and_names()
        else:
            for method, values in ids.items():
                assert getattr(reader, method)().tolist() == values, method
    finally:
        reader.close()


# What the reader keeps of the IDs and names of the file that _h5ad_with
# writes: four cell IDs and three feature IDs as bytes objects, 33 bytes and
# one a character each with a reference of 8, and the feature IDs again as
# the names, which the reader finds no column of, at 4 bytes a character.
_WRITTEN_ID_BYTES = 4 * (8 + 33 + 2) + 3 * (8 + 33 + 2) + 3 * 4 * 2


_SPLIT_TYPES = np.array([b"Gene Expression", b"Antibody Capture", b"Gene Expression"])


@pytest.mark.parametrize(
    ("members", "options", "beside_ids"),
    [
        pytest.param({}, {}, 0, id="ids"),
        # Each assay of a split keeps the IDs and names of its features, 4
        # bytes a character, and the import its features' columns in the
        # file and the assay and column of each, 8 bytes each.
        pytest.param(
            {"var/feature_types": _SPLIT_TYPES},
            {"assay_split_key": "feature_types"},
            3 * 4 * 2 * 2 + 3 * 8 + 2 * 3 * 8,
            id="split",
        ),
        # A declared assay membership, one boolean a cell.
        pytest.param(
            {
                "obs/RNA_I": np.ones(4, dtype=bool),
                "uns/cytearc/assayMembership/RNA": "RNA_I",
            },
            {},
            4,
            id="membership",
        ),
    ],
)
@pytest.mark.parametrize("shuffle_cells", [False, True], ids=["file", "shuffled"])
def test_the_scan_leaves_room_for_what_the_import_holds(
    tmp_path, monkeypatch, members, options, beside_ids, shuffle_cells
) -> None:
    path = _h5ad_with(tmp_path / "ids.h5ad", members)
    budgets: dict[str, int] = {}
    scan = H5adReader.count_value_ranges
    convert = H5adReader.materialize_csc

    def scan_spy(self, maxBytes, *args, **kwargs):
        budgets.setdefault("scan", maxBytes)
        return scan(self, maxBytes, *args, **kwargs)

    def convert_spy(self, maxBytes, **kwargs):
        budgets.setdefault("conversion", maxBytes)
        return convert(self, maxBytes, **kwargs)

    monkeypatch.setattr(H5adReader, "count_value_ranges", scan_spy)
    monkeypatch.setattr(H5adReader, "materialize_csc", convert_spy)
    reader = H5adReader(path)
    try:
        H5adToZarr(
            reader,
            str(tmp_path / "ids.zarr"),
            mem_budget="64M",
            nthreads=1,
            shuffle_cells=shuffle_cells,
            **options,
        )
    finally:
        reader.close()

    # A shuffle also holds the order and its inverse, 8 bytes a cell each.
    held = _WRITTEN_ID_BYTES + beside_ids + (2 * 8 * 4 if shuffle_cells else 0)
    remaining = 64 * 1024**2 - held
    assert budgets == {"scan": remaining, "conversion": remaining}


@pytest.mark.parametrize(
    ("members", "options", "held", "held_items"),
    [
        pytest.param({}, {}, _WRITTEN_ID_BYTES, "IDs and names", id="ids"),
        # The split's features, as the scan budget test counts them.
        pytest.param(
            {"var/feature_types": _SPLIT_TYPES},
            {"assay_split_key": "feature_types"},
            _WRITTEN_ID_BYTES + 3 * 4 * 2 * 2 + 3 * 8 + 2 * 3 * 8,
            "IDs, names and other arrays",
            id="split",
        ),
    ],
)
def test_a_scan_refusal_states_what_the_import_needs_beside_the_ids(
    tmp_path, monkeypatch, members, options, held, held_items
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _h5ad_with(Path("ids.h5ad"), members)
    # Each cell holds three values, which the scan reads at 72 bytes each.
    vector = 3 * 72
    budget = held + vector - 1
    reader = H5adReader(path)
    try:
        with pytest.raises(MemoryBudgetError) as refused:
            H5adToZarr(
                reader,
                "refused.zarr",
                mem_budget=budget,
                nthreads=1,
                shuffle_cells=False,
                **options,
            )
        # The scan passes with the bytes that the refusal names.
        with pytest.raises(MemoryError) as later:
            H5adToZarr(
                reader,
                "admitted.zarr",
                mem_budget=refused.value.needed_bytes,
                nthreads=1,
                shuffle_cells=False,
                **options,
            )
    finally:
        reader.close()

    needed = held + vector
    item = (
        f"the 3 values of the longest compressed vector beside the {held} bytes "
        f"of its {held_items}"
    )
    assert refused.value.needed_bytes == needed
    assert refused.value.budget_bytes == budget
    assert refused.value.item == item
    assert str(refused.value) == (
        f"Importing 'ids.h5ad' holds about {needed} bytes for {item}, more than "
        f"the memory budget of {budget} bytes. Raise mem_budget to at least "
        f"{needed} bytes."
    )
    assert "compressed vector" not in str(later.value)
    assert not (tmp_path / "refused.zarr").exists()


def _write_vectors(path: Path, lengths: list[int], encoding: str) -> str:
    """Write a CSR or CSC file of ones whose rows, or columns, hold ``lengths`` values.

    Its 30 cells are ``c0`` to ``c29`` and its 30 genes ``g0`` to ``g29``.
    """
    values = np.zeros((30, 30), dtype=np.float32)
    for vector, length in enumerate(lengths):
        values[vector, :length] = 1
    matrix = csr_matrix(values) if encoding == "csr" else csc_matrix(values.T)
    adata = anndata.AnnData(matrix)
    adata.obs_names = [f"c{index}" for index in range(30)]
    adata.var_names = [f"g{index}" for index in range(30)]
    adata.write_h5ad(path)
    return str(path)


# What the reader keeps of the IDs and names of a file that _write_vectors
# writes: 30 cell IDs and 30 feature IDs as bytes objects, and the feature
# IDs again as the names, at 4 bytes a character of the longest.
_VECTOR_FILE_ID_BYTES = 2 * sum(8 + 33 + len(f"c{index}") for index in range(30)) + (
    30 * 4 * 3
)


@pytest.mark.parametrize("encoding", ["csr", "csc"])
def test_a_scan_refusal_names_a_longer_vector_after_the_one_it_refused(
    tmp_path, encoding
) -> None:
    # The first vector holds 3 values, more than the scan's room, and the
    # vector of 6 values comes long after the pointers that the scan read.
    lengths = [3] + [1] * 19 + [6] + [1] * 9
    path = _write_vectors(tmp_path / f"{encoding}.h5ad", lengths, encoding)
    held = _VECTOR_FILE_ID_BYTES

    def construct(budget: int) -> MemoryError:
        with pytest.raises(MemoryError) as refused:
            H5adToZarr(
                reader,
                str(tmp_path / f"{budget}.zarr"),
                mem_budget=budget,
                nthreads=1,
                shuffle_cells=False,
            )
        return refused.value

    reader = H5adReader(path)
    try:
        refusal = construct(held + 3 * 72 - 1)
        below = construct(held + 6 * 72 - 1)
        later = construct(held + 6 * 72)
        # The scan runs at the room that the named budget leaves it.
        (value_range,) = reader.count_value_ranges(6 * 72)
    finally:
        reader.close()

    assert held == 2980
    assert isinstance(refusal, MemoryBudgetError)
    assert refusal.needed_bytes == held + 6 * 72
    assert refusal.item == (
        f"the 6 values of the longest compressed vector beside the {held} bytes "
        "of its IDs and names"
    )
    assert isinstance(below, MemoryBudgetError)
    assert (below.needed_bytes, below.item) == (refusal.needed_bytes, refusal.item)
    # A later step refuses so small a budget.
    assert "compressed vector" not in str(later)
    assert value_range.maximum == 1


def _counts_t_resident_spy(monkeypatch) -> list[int]:
    """Record the bytes held beside each admission and write of countsT."""
    import cytearc.storage.sharding as sharding

    seen: list[int] = []
    preflight = sharding.preflight_counts_t_spec

    def spy(*args, residentBytes=0, **kwargs):
        seen.append(residentBytes)
        return preflight(*args, residentBytes=residentBytes, **kwargs)

    monkeypatch.setattr(sharding, "preflight_counts_t_spec", spy)
    return seen


@pytest.mark.parametrize("shuffle_cells", [False, True], ids=["file", "shuffled"])
def test_counts_t_is_planned_and_written_beside_what_the_import_holds(
    tmp_path, monkeypatch, shuffle_cells
) -> None:
    path = _h5ad_with(tmp_path / "ids.h5ad", {})
    with h5py.File(path, "r") as h5:
        pointers = 5 * h5["X/indptr"].dtype.itemsize
    seen = _counts_t_resident_spy(monkeypatch)
    reader = H5adReader(path)
    try:
        H5adToZarr(
            reader, str(tmp_path / "ids.zarr"), nthreads=1, shuffle_cells=shuffle_cells
        ).dump()
    finally:
        reader.close()

    # Beside the IDs and names, an import in the file's order holds the
    # file's row pointers, which plan its counts write. A shuffle holds its
    # order and, until its counts are written, the int64 pointers of its
    # shuffled rows.
    if shuffle_cells:
        held = _WRITTEN_ID_BYTES + 8 * 4
        assert seen == [held + 8 * 5, held]
    else:
        assert seen == [_WRITTEN_ID_BYTES + pointers] * 2
    root = zarr.open_group(str(tmp_path / "ids.zarr"), mode="r")
    np.testing.assert_array_equal(root["RNA/countsT"][:], np.ones((3, 4)))


def test_a_conversion_again_in_the_files_order_leaves_room_for_what_the_import_holds(
    tmp_path, monkeypatch
) -> None:
    path = _write_vectors(tmp_path / "csc.h5ad", [3, 1, 2] + [1] * 27, "csc")
    budgets: list[int] = []
    convert = H5adReader.materialize_csc

    def convert_spy(self, maxBytes, **kwargs):
        budgets.append(maxBytes)
        return convert(self, maxBytes, **kwargs)

    monkeypatch.setattr(H5adReader, "materialize_csc", convert_spy)
    reader = H5adReader(path)
    try:
        in_order = H5adToZarr(
            reader,
            str(tmp_path / "order.zarr"),
            mem_budget="64M",
            nthreads=1,
            shuffle_cells=False,
        )
        # A shuffled import of the same reader replaces the conversion with
        # its shuffled rows, so the first import converts the columns again.
        H5adToZarr(
            reader, str(tmp_path / "shuffled.zarr"), mem_budget="64M", nthreads=1
        )
        in_order._write_counts()
    finally:
        reader.close()

    # Each conversion runs beside the IDs and names, and a shuffled one
    # beside its order and the inverse of it too.
    in_file_order = 64 * 1024**2 - _VECTOR_FILE_ID_BYTES
    assert budgets == [in_file_order, in_file_order - 2 * 8 * 30, in_file_order]
    counts = zarr.open_group(str(tmp_path / "order.zarr"), mode="r")["RNA/counts"]
    np.testing.assert_array_equal(np.count_nonzero(counts[:], axis=0)[:3], [3, 1, 2])


def test_a_conversion_again_in_the_files_order_refuses_in_the_imports_budget(
    tmp_path,
) -> None:
    from cytearc.storage.budget import ResourceBudget

    path = _write_vectors(tmp_path / "csc.h5ad", [3, 1, 2] + [1] * 27, "csc")
    reader = H5adReader(path)
    try:
        # What the conversion needs at least on its own, as the reader names it.
        with pytest.raises(MemoryBudgetError) as alone:
            reader.materialize_csc(1)
        least = alone.value.needed_bytes
        in_order = H5adToZarr(
            reader,
            str(tmp_path / "order.zarr"),
            mem_budget="64M",
            nthreads=1,
            shuffle_cells=False,
        )
        # A shuffled import replaces the conversion, so the first import
        # converts the columns again, within a budget one byte too small.
        H5adToZarr(
            reader, str(tmp_path / "shuffled.zarr"), mem_budget="64M", nthreads=1
        )
        needed = _VECTOR_FILE_ID_BYTES + least
        in_order.resources = ResourceBudget(needed - 1, 1)
        with pytest.raises(MemoryBudgetError) as refused:
            in_order._write_counts()
    finally:
        reader.close()

    assert refused.value.needed_bytes == needed
    assert refused.value.budget_bytes == needed - 1
    # The first row holds an entry of every column, the most of any row.
    assert refused.value.item == (
        "one row of the CSC row conversion beside the "
        f"{_VECTOR_FILE_ID_BYTES} bytes of its IDs and names"
    )


def _cells_with_ids(path: Path, ids: list[str]) -> str:
    """Write 600 cells of Poisson counts over 50 genes, with the given cell IDs."""
    counts = np.random.default_rng(4).poisson(1.0, size=(600, 50)).astype(np.float32)
    adata = anndata.AnnData(csr_matrix(counts))
    adata.obs_names = ids
    adata.var_names = [f"g{index}" for index in range(50)]
    adata.write_h5ad(path)
    return str(path)


def test_the_count_layout_charges_the_ids_that_the_reader_keeps(tmp_path) -> None:
    from cytearc.storage.sharding import CountLayoutMemoryError

    short = [f"c{row}" for row in range(600)]
    long = [f"cell-{row:06d}-ACGTACGTACGTACGT" for row in range(600)]
    needed: dict[str, int] = {}
    for name, ids in (("short", short), ("long", long)):
        reader = H5adReader(_cells_with_ids(tmp_path / f"{name}.h5ad", ids))

        def construct(budget: int, reader: H5adReader = reader) -> H5adToZarr:
            # ADT writes no countsT, so its counts write alone sizes the layout.
            return H5adToZarr(
                reader,
                str(tmp_path / f"{name}-{budget}.zarr"),
                assay_name="ADT",
                mem_budget=budget,
                nthreads=1,
                shuffle_cells=False,
            )

        try:
            with pytest.raises(CountLayoutMemoryError) as refused:
                construct(64 * 1024)
            needed[name] = refused.value.needed_bytes
            # One byte less than the budget it names is refused the same way,
            # and that budget fits the layout.
            with pytest.raises(CountLayoutMemoryError) as below:
                construct(needed[name] - 1)
            assert below.value.needed_bytes == needed[name]
            assert construct(needed[name]).policy == DEFAULT_COUNT_MATRIX_POLICY
        finally:
            reader.close()

    # The reader keeps each cell ID as a bytes object of its characters, so
    # the longer IDs need exactly their extra characters more.
    extra = sum(len(text) for text in long) - sum(len(text) for text in short)
    assert needed["long"] - needed["short"] == extra == 14_510


def test_imported_analyses_take_blocks_of_what_the_budget_leaves_beside_the_ids(
    tmp_path,
) -> None:
    rng = np.random.default_rng(8)
    n_cells = 20_000
    adata = anndata.AnnData(
        csr_matrix(rng.poisson(0.5, (n_cells, 10)).astype(np.float32))
    )
    adata.obs_names = [f"cell-{row:06d}-ACGTACGTACGTACGT" for row in range(n_cells)]
    adata.var_names = [f"g{index}" for index in range(10)]
    adata.obs["clusters"] = pd.Categorical(rng.integers(0, 5, n_cells).astype(str))
    adata.obsm["X_wide"] = rng.normal(size=(n_cells, 64)).astype(np.float32)
    path = tmp_path / "analyses.h5ad"
    adata.write_h5ad(path)
    with h5py.File(path, "r") as h5:
        pointers = int(h5["X/indptr"].nbytes)
    budget = 8 * 1024**2
    reader = H5adReader(
        str(path),
        embedding_roles={"X_wide": "coordinates"},
        cluster_keys=("clusters",),
    )
    try:
        result = H5adToZarr(
            reader,
            str(tmp_path / "analyses.zarr"),
            mem_budget=budget,
            nthreads=1,
            shuffle_cells=False,
        ).dump()
    finally:
        reader.close()
    store = cytearc.DataStore(
        str(tmp_path / "analyses.zarr"), min_features_per_cell=0, nthreads=1
    )
    options = {
        name: store.artifacts.inspect(ref).execution_options
        for name, ref in (
            ("embedding", result.embeddingArtifacts["X_wide"]),
            ("labels", result.clusterArtifacts["clusters"]),
        )
    }

    # The import holds the cell IDs of 28 characters and the feature IDs as
    # bytes objects, the feature IDs again as the names, at 4 bytes a
    # character, and the file's row pointers. A block takes an eighth of
    # what is left: rows of 64 float32 values, and labels at 64 bytes a row.
    held = n_cells * (8 + 33 + 28) + 10 * (8 + 33 + 2) + 10 * 4 * 2 + pointers
    left = budget - held
    assert options["embedding"]["block_rows"] == left // (8 * 64 * 4)
    assert options["labels"]["block_rows"] == left // (8 * 64)


def test_the_import_charges_what_tracing_finds_the_reader_keeps(tmp_path) -> None:
    import gc
    import tracemalloc

    rng = np.random.default_rng(6)
    barcodes = [
        "".join(rng.choice(list("ACGT"), 16)) + f"-{row}" for row in range(50_000)
    ]
    adata = anndata.AnnData(csr_matrix((50_000, 20), dtype=np.float32))
    adata.obs_names = barcodes
    adata.var_names = [f"ENSG{index:011d}" for index in range(20)]
    adata.var["gene_short_name"] = [f"GENE{index}" for index in range(20)]
    path = str(tmp_path / "barcodes.h5ad")
    adata.write_h5ad(path)
    reader = H5adReader(path)
    try:
        gc.collect()
        tracemalloc.start()
        try:
            reader._check_ids_and_names()
            gc.collect()
            kept, _peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        writer = H5adToZarr(reader, str(tmp_path / "barcodes.zarr"), nthreads=1)
        charged = writer._kept_bytes()
    finally:
        reader.close()

    # What the allocations that the reader keeps add up to, which the import
    # charges, within 1%.
    assert kept > 50_000 * (8 + 33 + 16)
    assert abs(charged - kept) <= kept // 100


def test_the_reader_counts_the_renamed_ids_beside_those_of_the_file(
    tmp_path,
) -> None:
    path = _h5ad_with(
        tmp_path / "repeated.h5ad",
        {"obs/_index": np.array([b"c0", b"c0", b"c1", b"c2"])},
    )
    reader = H5adReader(path, duplicate_cell_ids="make_unique")
    try:
        held = reader._id_bytes()
    finally:
        reader.close()

    # The renamed IDs are str objects of ASCII, 41 bytes and one a character
    # each, with a reference of 8.
    renamed = sum(8 + 41 + len(name) for name in ["c0", "c0-1", "c1", "c2"])
    assert held == _WRITTEN_ID_BYTES + renamed


@pytest.mark.parametrize(
    ("rows", "named"),
    [
        ([0], "row 1"),
        ([1, 3], "rows 2 and 4"),
        ([0, 1, 2], "rows 1, 2 and 3"),
        ([4, 5, 6, 7, 8], "rows 5, 6, 7, 8 and 9"),
        (list(range(1_237)), "rows 1, 2, 3, 4, 5 and 1,232 more"),
    ],
)
def test_refusals_name_the_first_rows_and_how_many_more(rows, named) -> None:
    from cytearc.readers._text import _named_rows

    assert _named_rows(rows) == named


@pytest.mark.parametrize(
    ("members", "expected"),
    [
        pytest.param(
            {"obs/_index": np.array([b"nan", b"None", b"NaN", b"c3"])},
            {"cell_ids": ["nan", "None", "NaN", "c3"]},
            id="text-that-reads-as-missing",
        ),
        pytest.param(
            {"obs/_index": np.array([0.5, 1.5, np.inf, -0.0])},
            {"cell_ids": ["0.5", "1.5", "inf", "-0.0"]},
            id="numbers-without-nan",
        ),
        pytest.param(
            {"var/_index": np.array([0.25, 0.5, 0.75], dtype=np.float32)},
            {
                "feat_ids": ["0.25", "0.5", "0.75"],
                "feat_names": ["0.25", "0.5", "0.75"],
            },
            id="float32-feature-ids",
        ),
        pytest.param(
            {
                "obs/_index": np.array(
                    [text.encode() for text in ("café", "中文", "😀", "c3")]
                )
            },
            {"cell_ids": ["café", "中文", "😀", "c3"]},
            id="non-ascii-utf8-bytes",
        ),
        pytest.param({}, {}, id="written-by-anndata"),
    ],
)
def test_a_shuffled_import_stores_the_ids_and_names_that_an_import_in_order_stores(
    tmp_path, members, expected
) -> None:
    path = _h5ad_with(tmp_path / "ids.h5ad", members)
    stored = {}
    for shuffle_cells in (False, True):
        reader = H5adReader(path)
        try:
            H5adToZarr(
                reader,
                str(tmp_path / f"{shuffle_cells}.zarr"),
                nthreads=1,
                shuffle_cells=shuffle_cells,
            ).dump()
        finally:
            reader.close()
        store = cytearc.DataStore(
            str(tmp_path / f"{shuffle_cells}.zarr"), min_features_per_cell=0, nthreads=1
        )
        cell_ids = store.cells.fetch_all("ids")
        if shuffle_cells:
            # Each cell's ID at its row in the file.
            in_file_order = np.empty_like(cell_ids)
            in_file_order[store.cells.fetch_all("sourceRow")] = cell_ids
            cell_ids = in_file_order
        stored[shuffle_cells] = {
            "cell_ids": cell_ids.tolist(),
            "feat_ids": store.RNA.feats.fetch_all("ids").tolist(),
            "feat_names": store.RNA.feats.fetch_all("names").tolist(),
        }

    for method in _ID_METHODS:
        assert stored[True][method] == expected.get(method, _WRITTEN_IDS[method])
        assert stored[False][method] == stored[True][method]


@pytest.mark.parametrize(
    ("index", "ids"),
    [
        (np.array([101, 102, 103, 104]), ["101", "102", "103", "104"]),
        (np.array([b"c0", b"c1", b"c2", b"c3"], dtype="S2"), ["c0", "c1", "c2", "c3"]),
    ],
    ids=["integers", "fixed-length-bytes"],
)
def test_values_keyed_by_the_reader_cell_ids_insert_into_the_store(
    tmp_path, index, ids
) -> None:
    # Issue 396: values keyed by the reader's IDs were refused, because the
    # reader gave 101 or b"c0" where the store holds "101" or "c0".
    path = _h5ad_with(tmp_path / "ids.h5ad", {"obs/_index": index})
    reader = H5adReader.from_inspect(inspect_h5ad(path))
    try:
        cell_ids = reader.cell_ids()
        # The import shuffles the cells, as every import does by default.
        H5adToZarr(reader, str(tmp_path / "ids.zarr"), nthreads=1).dump()
    finally:
        reader.close()
    store = cytearc.DataStore(
        str(tmp_path / "ids.zarr"), min_features_per_cell=0, nthreads=1
    )
    rows = store.cells.fetch_all("sourceRow")
    assert not np.array_equal(rows, np.arange(4))

    store.cells.insert("score", pd.Series(np.arange(4.0), index=cell_ids))

    assert cell_ids.tolist() == ids
    # Each cell gets the value of its row in the file, wherever it is stored.
    np.testing.assert_array_equal(store.cells.fetch_all("score"), rows.astype(float))
    # Matching stays exact: the IDs as the file holds them name no cell.
    with pytest.raises(ValueError, match="4 labels are not such ids"):
        store.cells.insert("raw", pd.Series(np.arange(4.0), index=index.tolist()))
    assert "raw" not in store.cells.columns


def _write_matrices(path: Path, matrices: dict, n_cells: int, n_genes: int) -> str:
    """Write an H5AD file whose ``matrices`` are its candidate count matrices."""
    adata = anndata.AnnData(
        X=matrices.pop("X", None),
        obs=pd.DataFrame(index=[f"c{index}" for index in range(n_cells)]),
        var=pd.DataFrame(index=[f"g{index}" for index in range(n_genes)]),
    )
    for name, values in matrices.items():
        adata.layers[name] = values
    return _write(path, adata)


def test_inspection_imports_total_counts_of_velocity_data(tmp_path) -> None:
    rng = np.random.default_rng(0)
    spliced = rng.poisson(3, size=(30, 10))
    unspliced = rng.poisson(1, size=(30, 10))
    ambiguous = rng.poisson(0.2, size=(30, 10))

    def normalized(values):
        totals = values.sum(axis=1, keepdims=True).clip(1)
        return csr_matrix(np.log1p(values / totals * 1e4).astype(np.float32))

    path = _write_matrices(
        tmp_path / "velocity.h5ad",
        {
            "X": normalized(spliced + unspliced),
            "spliced": normalized(spliced),
            "unspliced": normalized(unspliced),
            "ambiguous": csr_matrix(ambiguous),
            "matrix": csr_matrix(spliced + unspliced + ambiguous),
        },
        30,
        10,
    )
    with _warnings() as messages:
        inspection = inspect_h5ad(path)

    # Before, ties fell to the alphabetically first layer, the ambiguous reads.
    assert inspection.matrixKey == "layers/matrix"
    assert inspection.matrixCandidates[:2] == ("layers/matrix", "layers/ambiguous")
    (warning,) = [
        m.rstrip() for m in messages if "Several matrices hold integer counts" in m
    ]
    assert warning.startswith(
        "Several matrices hold integer counts: layers/matrix (about "
    )
    assert "layers/ambiguous (about " in warning
    assert warning.endswith(
        "inspect_h5ad chose layers/matrix; review the choice and pass matrix_key "
        "to choose another"
    )
    with _warnings() as forced:
        assert (
            inspect_h5ad(path, matrix_key="layers/ambiguous").matrixKey
            == "layers/ambiguous"
        )
    assert not any("Several matrices" in message for message in forced)


@pytest.mark.parametrize(
    ("layers", "chosen"),
    [
        # A counts layer outranks integer X, which outranks spliced counts.
        ({"X": 2, "counts": 1, "spliced": 3}, "layers/counts"),
        ({"X": 1, "spliced": 3, "unspliced": 5}, "X"),
        ({"spliced": 1, "unspliced": 5, "ambiguous": 9}, "layers/spliced"),
        # Matrices of one tier rank by their total counts.
        ({"first": 1, "second": 4}, "layers/second"),
        ({"raw_counts": 1, "counts": 3}, "layers/counts"),
    ],
)
def test_inspection_prefers_conventional_count_names(tmp_path, layers, chosen):
    rng = np.random.default_rng(1)
    matrices = {
        name: csr_matrix(rng.poisson(scale, size=(20, 6)).astype(np.float32))
        for name, scale in layers.items()
    }
    path = _write_matrices(tmp_path / "counts.h5ad", matrices, 20, 6)

    assert inspect_h5ad(path).matrixKey == chosen


def test_inspection_skips_a_null_x(tmp_path) -> None:
    rng = np.random.default_rng(2)
    counts = csr_matrix(rng.poisson(2, size=(20, 6)).astype(np.float32))
    path = _write_matrices(tmp_path / "null_x.h5ad", {"counts": counts}, 20, 6)
    # anndata before 0.13 writes X=None as a null dataspace without a shape.
    with h5py.File(path, "r+") as h5:
        if "X" in h5:
            del h5["X"]
        h5.create_dataset("X", data=h5py.Empty("f4"))
        h5["X"].attrs["encoding-type"] = "null"
        h5["X"].attrs["encoding-version"] = "0.1.0"

    assert inspect_h5ad(path).matrixKey == "layers/counts"


def test_inspection_counts_the_cells_of_a_column_when_the_index_holds_no_rows(
    tmp_path,
) -> None:
    path = tmp_path / "cells.h5ad"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("X", data=np.ones((3, 2), dtype=np.float32))
        # The conventional name ranks first, but it holds two cells.
        h5.create_group("layers").create_dataset(
            "counts", data=np.ones((2, 2), dtype=np.float32)
        )
        obs = h5.create_group("obs")
        # The index names a group of no known encoding, which holds no rows.
        obs.attrs["_index"] = "cell_id"
        obs.create_group("cell_id").create_dataset("x", data=np.arange(5))
        obs.create_dataset("batch", data=np.array([1, 2, 3]))
        var = h5.create_group("var")
        var.attrs["_index"] = "_index"
        var.create_dataset("_index", data=_text(b"g0", b"g1"))

    with _warnings() as messages:
        inspection = inspect_h5ad(str(path))

    # The obs table holds the three cells of its column 'batch'.
    assert inspection.matrixKey == "X"
    assert inspection.nCells == 3
    assert (
        "Ignoring matrix candidate layers/counts: 2 rows do not match obs length 3"
        in [message.rstrip("\n") for message in messages]
    )


def test_a_column_named_categories_is_read_as_a_column(tmp_path) -> None:
    path = tmp_path / "categories.h5ad"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("X", data=np.ones((3, 2), dtype=np.float32))
        obs = h5.create_group("obs")
        obs.attrs["_index"] = "_index"
        obs.create_dataset("_index", data=_text(b"c0", b"c1", b"c2"))
        obs.create_dataset("categories", data=_text(b"a", b"b", b"a"))
        # AnnData 0.7 kept the categories of integer codes in '__categories'.
        obs.create_dataset("batch", data=np.array([0, 1, 0], dtype=np.int8))
        obs.create_group("__categories").create_dataset("batch", data=_text(b"x", b"y"))
        var = h5.create_group("var")
        var.attrs["_index"] = "_index"
        var.create_dataset("_index", data=_text(b"g0", b"g1"))

    inspection = inspect_h5ad(str(path))

    # A dataset named 'categories' is a column, not a group of categories.
    assert inspection.categoryNamesKey == "__categories"
    reader = H5adReader.from_inspect(inspection)
    try:
        columns = {name: values.tolist() for name, values in reader.get_cell_columns()}
    finally:
        reader.close()
    assert columns == {"batch": ["x", "y", "x"], "categories": ["a", "b", "a"]}


def test_inspection_reads_a_title_and_description_of_numbers_as_text(
    tmp_path,
) -> None:
    path = tmp_path / "numbers.h5ad"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("X", data=np.ones((3, 2), dtype=np.float32))
        uns = h5.create_group("uns")
        uns.create_dataset("title", data=np.int64(2024))
        uns.create_dataset("description", data=np.array([1.5]))

    inspection = inspect_h5ad(str(path))

    assert (inspection.title, inspection.description) == ("2024", "1.5")


def test_inspection_samples_values_across_the_whole_matrix(tmp_path) -> None:
    import cytearc.readers._h5ad_inspect as inspection_module

    # Whole values first, fractions after them: neither the first 101 values,
    # which inspection used to read, nor the first window of 1,024 sparse
    # values or 20 dense rows holds a fraction.
    values = np.ones((40, 50), dtype=np.float32)
    values[21:] = 0.5
    dense = values.copy()
    dense[:10] = 0
    path = tmp_path / "sampled.h5ad"
    with h5py.File(path, "w") as h5:
        matrix = csr_matrix(values)
        group = h5.create_group("X")
        group.attrs.update({"encoding-type": "csr_matrix", "shape": matrix.shape})
        group.create_dataset("data", data=matrix.data)
        group.create_dataset("indices", data=matrix.indices)
        group.create_dataset("indptr", data=matrix.indptr)
        h5.create_group("layers").create_dataset("dense", data=dense)
        h5.create_group("obs").create_dataset(
            "_index", data=np.asarray([f"c{index}".encode() for index in range(40)])
        )

    with h5py.File(path, "r") as h5:
        candidates = inspection_module._matrix_candidates(h5)
    # Matrices without counts estimate no total and rank by their name.
    assert candidates == [
        inspection_module._MatrixCandidate("X", "csr", (40, 50), integerLike=False),
        inspection_module._MatrixCandidate(
            "layers/dense", "dense", (40, 50), integerLike=False
        ),
    ]


@pytest.mark.parametrize(
    ("values", "integer_like", "total"),
    [
        (np.asarray([1, 2, 3], dtype=np.int32), True, 6.0),
        (np.asarray([1, -2, 3], dtype=np.int32), False, 0.0),
        (np.asarray([0.0, 2.0, 4.0]), True, 6.0),
        (np.asarray([1.0, 2.0, np.nan]), False, 0.0),
        # Infinity is non-negative, and rounds to itself.
        (np.asarray([1.0, 2.0, np.inf]), False, 0.0),
        (np.asarray([1.0, 2.0, -3.0]), False, 0.0),
        (np.asarray([1.0, 2.5]), False, 0.0),
        # A value within 1e-8 of a whole number is not whole.
        (np.asarray([1.0, 2.0 + 1e-10]), False, 0.0),
        (np.asarray([], dtype=np.float32), False, 0.0),
    ],
)
def test_sampled_counts_must_be_finite_non_negative_and_whole(
    tmp_path, values, integer_like, total
):
    from cytearc.readers._h5ad_inspect import _sparse_counts

    path = tmp_path / "values.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("data", data=values)
    with h5py.File(path, "r") as h5:
        assert _sparse_counts(h5["data"]) == (integer_like, total)


@pytest.mark.parametrize(
    "values",
    [
        [0.0, 1.0, 7.0],
        [1.0, 2.0 + 1e-10],
        [3.0 - 1e-12, 1.0],
        [1e-9],
        [4.0, 0.5],
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_inspection_tells_whole_numbers_as_the_import_stores_them(
    values, dtype
) -> None:
    from cytearc.readers._h5ad_inspect import _sampled_counts
    from cytearc.utils.count_values import CountValueRange

    sample = np.asarray(values, dtype=dtype)
    stored = CountValueRange()
    stored.update(sample)
    assert _sampled_counts(sample) is stored.integral


def test_sample_windows_spread_over_the_values() -> None:
    from cytearc.readers._h5ad_inspect import _sample_windows

    assert _sample_windows(0, 4) == []
    assert _sample_windows(3, 10) == [slice(0, 3)]
    windows = _sample_windows(100_000, 1024)
    assert len(windows) == 32
    assert windows[0] == slice(0, 1024)
    assert windows[-1] == slice(100_000 - 1024, 100_000)
    # Every window holds 1,024 values, and they start at the multiples of
    # 98,976 / 31, rounded down.
    assert {window.stop - window.start for window in windows} == {1024}
    assert set(np.diff([window.start for window in windows]).tolist()) == {3192, 3193}


def test_dense_matrix_without_columns_holds_no_counts(tmp_path) -> None:
    from cytearc.readers._h5ad_inspect import _dense_counts

    path = tmp_path / "dense.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("wide", data=np.zeros((3, 0)))
        h5.create_dataset("counts", data=np.asarray([[1, 2], [3, 4]], dtype=np.uint8))
    with h5py.File(path, "r") as h5:
        assert _dense_counts(h5["wide"]) == (False, 0.0)
        assert _dense_counts(h5["counts"]) == (True, 10.0)


def test_coordinates_role_imports_graph_coordinates(tmp_path) -> None:
    rng = np.random.default_rng(3)
    latent = rng.normal(size=(12, 3)).astype(np.float32)
    adata = anndata.AnnData(
        X=csr_matrix(rng.poisson(2, size=(12, 5)).astype(np.float32)),
        obs=pd.DataFrame(index=[f"c{index}" for index in range(12)]),
        var=pd.DataFrame(index=[f"g{index}" for index in range(5)]),
    )
    adata.obsm["X_scVI"] = latent
    path = _write(tmp_path / "latent.h5ad", adata)
    store, result = _import(
        path, tmp_path / "store.zarr", embedding_roles={"X_scVI": "coordinates"}
    )
    ref = result.embeddingArtifacts["X_scVI"]

    assert ref.kind == "imported_coordinates"
    np.testing.assert_array_equal(store.artifacts.load(ref)["data"][:], latent)
    status = store.artifacts.inspect(ref)
    assert status.parameters["role"] == "coordinates"
    assert status.parameters["dimreduc_key"] == "X_scVI"
    ann = store.graph.ann_index(ref, ann_efc=10, ann_ef=10, ann_m=4)
    neighbors = store.graph.neighbors(ann, k=3)
    assert store.artifacts.load(neighbors)["indices"].shape == (12, 3)


@pytest.mark.parametrize(
    ("name", "missing"),
    [
        (None, True),
        (float("nan"), True),
        (np.float32("nan"), True),
        (1.5, False),
        (b" ", True),
        ("", True),
        ("GENE", False),
        (b"GENE", False),
    ],
)
def test_feature_names_that_count_as_missing(name, missing) -> None:
    from cytearc.readers.h5ad import _is_missing_name

    assert _is_missing_name(name) is missing


@pytest.mark.parametrize(
    ("key", "tier"),
    [
        ("raw/X", 0),
        ("layers/counts", 0),
        ("layers/Raw_Counts", 0),
        ("X", 1),
        ("layers/spliced", 2),
        ("layers/matrix", 3),
        ("raw/layers/counts", 3),
        ("layers/unspliced", 4),
        ("layers/ambiguous", 4),
    ],
)
def test_count_matrix_names_rank_by_convention(key, tier) -> None:
    from cytearc.readers._h5ad_inspect import _name_tier

    assert _name_tier(key) == tier


def test_sampled_values_must_be_numbers() -> None:
    from cytearc.readers._h5ad_inspect import _sampled_counts

    assert _sampled_counts(np.asarray([True, False])) is False
    # Stored text is not counts, even when it spells whole numbers.
    assert _sampled_counts(np.asarray([b"1", b"2"])) is False
    assert _sampled_counts(np.asarray([], dtype=np.float32)) is False
    assert _sampled_counts(np.asarray([0, 3], dtype=np.uint8)) is True


def test_inspection_without_counts_warns_about_no_competition(tmp_path) -> None:
    rng = np.random.default_rng(2)
    normalized = csr_matrix(rng.random((6, 4)).astype(np.float32))
    path = _write_matrices(
        tmp_path / "normalized.h5ad",
        {"X": normalized, "scaled": normalized.copy()},
        6,
        4,
    )
    with _warnings() as messages:
        inspection = inspect_h5ad(path)

    assert inspection.integerLike is False
    assert (inspection.matrixKey, inspection.matrixCandidates) == (
        "X",
        ("X", "layers/scaled"),
    )
    assert messages == []

    # Counts in a layer that the features do not describe cannot be chosen,
    # so they do not compete with the chosen matrix either.
    with h5py.File(path, "r+") as h5:
        h5["layers"].create_dataset("counts", data=rng.poisson(2, size=(6, 5)))
    with _warnings() as messages:
        inspection = inspect_h5ad(path)

    assert inspection.integerLike is False
    assert (inspection.matrixKey, inspection.matrixCandidates) == (
        "X",
        ("layers/counts", "X", "layers/scaled"),
    )
    assert [message.rstrip() for message in messages] == [
        "Ignoring matrix candidate layers/counts: feature metadata group `var` "
        "length 4 does not match feature count 5"
    ]


def test_tables_without_an_index_import_every_column(tmp_path) -> None:
    path = tmp_path / "no_index.h5ad"
    with h5py.File(path, "w") as h5:
        matrix = csr_matrix(np.asarray([[1, 0], [0, 2]], dtype=np.uint8))
        group = h5.create_group("X")
        group.attrs.update({"encoding-type": "csr_matrix", "shape": matrix.shape})
        group.create_dataset("data", data=matrix.data)
        group.create_dataset("indices", data=matrix.indices)
        group.create_dataset("indptr", data=matrix.indptr)
        h5.create_group("obs").create_dataset("barcode", data=np.asarray([b"a", b"b"]))
        var = h5.create_group("var")
        var.create_dataset("gene_ids", data=np.asarray([b"g0", b"g1"]))
        var.create_dataset("chromosome", data=np.asarray([b"1", b"2"]))
    reader = H5adReader(str(path), cell_ids_key="barcode", feature_ids_key="gene_ids")
    try:
        assert reader._index_member("var") is None
        assert dict(reader.get_feat_columns())["chromosome"].tolist() == ["1", "2"]
        assert list(dict(reader.get_cell_columns())) == []
    finally:
        reader.close()


def test_categoricals_whose_categories_the_column_cannot_hold_keep_nothing():
    from cytearc.writers.h5ad import _categorical_attributes

    assert (
        _categorical_attributes(
            np.dtype(np.int64),
            H5adCategories(("a", "b"), False),
            colors=None,
            has_missing=False,
            display=True,
        )
        == {}
    )
