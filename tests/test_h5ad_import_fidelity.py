"""H5AD import fidelity: categoricals, colors, indexes, IDs, and matrix choice."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import zarr
from loguru import logger
from scipy.sparse import csr_matrix

import cytearc
from cytearc.metadata.artifacts import categorical_display
from cytearc.readers import H5adReader, inspect_h5ad
from cytearc.readers._h5ad_columns import (
    H5adCategories,
    category_colors,
    column_categories,
    hex_color,
)
from cytearc.writers import H5adToZarr

anndata = pytest.importorskip("anndata")

_STAGES = ["early", "mid", "late", "unused"]
_STAGE_COLORS = ["#000000", "#888888", "#ff0000", "#00ff00"]


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
    assert np.asarray(cells["clusters_int"][:]).tolist() == [0, 1, 2, 3, 1, 2, 3, 1]
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
    # Feature categoricals keep their categories, without a display.
    kind = dict(root["RNA/featureData/kind"].attrs)
    assert kind == {"levels": ["y", "x", "z"], "ordered": False}


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


def test_categoricals_with_many_categories_keep_no_levels(tmp_path, monkeypatch):
    import cytearc.writers.h5ad as h5ad_writer

    monkeypatch.setattr(h5ad_writer, "_MAX_STORED_CATEGORIES", 3)
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
    from cytearc.writers.h5ad import _categorical_attributes, _categorical_display

    # Two categories that compare equal cannot both be display values.
    assert _categorical_display([1, 1.0], None, False) is None
    attributes = _categorical_attributes(
        np.dtype(np.float64),
        H5adCategories((1, 1.0), False),
        colors=None,
        has_missing=False,
        display=True,
    )
    assert attributes == {"levels": [1.0, 1.0], "ordered": False}


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
        assert column_categories(h5, h5["obs"], "batch", groups) == H5adCategories(
            ("B", "A"), True
        )
        assert column_categories(h5, h5["obs"], "donor", groups) == H5adCategories(
            ("d1", "d2"), False
        )
        assert column_categories(h5, h5["obs"], "score", groups) is None
        # Integer codes without stored categories are plain integers.
        assert column_categories(h5, h5["obs"], "other", groups) is None
        assert column_categories(h5, h5["obs"], "nullable", groups) is None
        # AnnData 0.6 kept the categories of a compound column in uns.
        assert column_categories(h5, h5["var"], "group", groups) == H5adCategories(
            ("g", "h"), False
        )
        assert column_categories(h5, h5["var"], "value", groups) is None


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
        assert dict(reader.get_feat_columns())["chromosome"].tolist() == [b"1", b"2"]
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
