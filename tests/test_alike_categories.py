"""Categoricals whose categories read alike once stored keep one category per label.

An H5AD file can hold categories that read alike once CyteArc types them:
text, numbers or booleans that repeat, which h5py writes although pandas
refuses them, and the float categories -0.0 and 0.0, which differ in the file
and compare equal. Each label of such a categorical has one level, at the
place and with the color of its first category, in the attributes that an
import stores and in what exports, plots and merges read. Imports of CyteArc
1.3.0 and 1.4.0 stored one level per category and no display; a store whose
attributes are rewritten as CyteArc 1.4.0 wrote them stands in for theirs.
Each check compares with levels, codes and colors written out by hand, or
with a twin column whose file names each label once.
"""

from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import pytest
import zarr

matplotlib.use("Agg")

import cytearc  # noqa: E402
import cytearc.plotting as splt  # noqa: E402
from cytearc.metadata.artifacts import (  # noqa: E402
    distinct_level_positions,
    recorded_categories,
)
from cytearc.readers import H5adReader  # noqa: E402
from cytearc.storage.artifacts import artifact_group  # noqa: E402
from cytearc.storage.categorical import CategoricalColumn  # noqa: E402
from cytearc.writers import H5adToZarr  # noqa: E402
from tests.storage_helpers import open_store  # noqa: E402
from tests.test_categorical_imports import (  # noqa: E402
    _TEXT,
    _categorical,
    _warnings,
    _write_counts,
    _write_index,
)

anndata = pytest.importorskip("anndata")

_N_CELLS = 6
_GENES = ["g0", "g1", "g2", "g3"]

# Each obs categorical whose categories read alike once stored, as the file
# holds it: its codes, categories, order and uns colors, or None. Then, written
# out by hand, the levels that name each label once, in the order of its first
# category, each row's code among them, -1 when missing, and their colors.
_ALIKE: dict[str, dict[str, Any]] = {
    # Text that repeats, with a missing row and a color for every category.
    "cell_type": {
        "codes": [0, 2, 1, 4, -1, 3],
        "categories": np.array(["T", "B", "T", "NK", "B"], dtype=_TEXT),
        "ordered": False,
        "colors": ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#00ffff"],
        "levels": ["T", "B", "NK"],
        "level_codes": [0, 0, 1, 1, -1, 2],
        "level_colors": ["#ff0000", "#00ff00", "#ffff00"],
    },
    # Ordered text that repeats, with an unused category and no colors.
    "stage": {
        "codes": [0, 1, 2, 3, 0, 2],
        "categories": np.array(["early", "mid", "mid", "late", "unused"], dtype=_TEXT),
        "ordered": True,
        "colors": None,
        "levels": ["early", "mid", "late", "unused"],
        "level_codes": [0, 1, 1, 2, 0, 1],
        "level_colors": None,
    },
    # -0.0 and 0.0 differ in the file and compare equal; -0.0 comes first.
    "zero": {
        "codes": [0, 2, 1, 2, 0, 1],
        "categories": np.array([-0.0, 1.5, 0.0]),
        "ordered": False,
        "colors": ["#000001", "#000002", "#000003"],
        "levels": [-0.0, 1.5],
        "level_codes": [0, 0, 1, 0, 0, 1],
        "level_colors": ["#000001", "#000002"],
    },
    # Integers that repeat, with a missing row.
    "count": {
        "codes": [0, 1, 2, 2, -1, 0],
        "categories": np.array([7, 3, 7]),
        "ordered": False,
        "colors": ["#000010", "#000020", "#000030"],
        "levels": [7, 3],
        "level_codes": [0, 1, 0, 0, -1, 0],
        "level_colors": ["#000010", "#000020"],
    },
    # Booleans that repeat.
    "flag": {
        "codes": [0, 1, 2, 1, 2, 0],
        "categories": np.array([False, True, False]),
        "ordered": False,
        "colors": None,
        "levels": [False, True],
        "level_codes": [0, 1, 0, 1, 0, 0],
        "level_colors": None,
    },
}
# Ordered cluster labels that repeat, which the import keeps as an artifact.
_CLUSTERS: dict[str, Any] = {
    "codes": [1, 0, 2, 2, 1, -1],
    "categories": np.array(["c1", "c2", "c1"], dtype=_TEXT),
    "ordered": True,
    "colors": ["#100000", "#200000", "#300000"],
    "levels": ["c1", "c2"],
    "level_codes": [1, 0, 0, 0, 1, -1],
    "level_colors": ["#100000", "#200000"],
}
# A feature categorical that repeats; feature columns keep no display.
_BIOTYPE: dict[str, Any] = {
    "codes": [0, 2, 1, -1],
    "categories": np.array(["coding", "lnc", "coding"], dtype=_TEXT),
    "ordered": False,
    "levels": ["coding", "lnc"],
    "level_codes": [0, 0, 1, -1],
}
# The attributes that CyteArc 1.4.0 stored for these columns, as its import of
# this file shows: one level per category and no display.
_EARLIER_CELL_ATTRIBUTES: dict[str, dict[str, Any]] = {
    "cell_type": {
        "missing_mask": "__cytearc_missing__cell_type",
        "levels": ["T", "B", "T", "NK", "B"],
        "ordered": False,
    },
    "stage": {"levels": ["early", "mid", "mid", "late", "unused"], "ordered": True},
    "zero": {"levels": [-0.0, 1.5, 0.0], "ordered": False},
    "count": {
        "missing_mask": "__cytearc_missing__count",
        "levels": [7, 3, 7],
        "ordered": False,
    },
    "flag": {"levels": [False, True, False], "ordered": False},
}
_EARLIER_CLUSTER_ATTRIBUTES = {
    "missing_mask": "__cytearc_missing__values",
    "levels": ["c1", "c2", "c1"],
    "ordered": True,
}
_EARLIER_BIOTYPE_ATTRIBUTES = {
    "missing_mask": "__cytearc_missing__biotype",
    "levels": ["coding", "lnc", "coding"],
    "ordered": False,
}


def _twin(name: str) -> str:
    """The column whose file names each label of ``name`` once."""
    return f"{name}_twin"


def _write_source(path: Path) -> None:
    """Write six cells and four genes with the categoricals above and their twins."""
    with h5py.File(path, "w") as h5:
        _write_counts(h5, _N_CELLS, len(_GENES))
        obs = h5.create_group("obs")
        _write_index(obs, [f"cell-{index}" for index in range(_N_CELLS)])
        uns = h5.create_group("uns")
        columns = {**_ALIKE, "clusters": _CLUSTERS}
        for name, spec in columns.items():
            twin = (
                spec["level_codes"],
                np.array(spec["levels"], dtype=spec["categories"].dtype),
                spec["level_colors"],
            )
            for key, (codes, categories, colors) in (
                (name, (spec["codes"], spec["categories"], spec["colors"])),
                (_twin(name), twin),
            ):
                _categorical(
                    obs,
                    key,
                    np.array(codes, dtype=np.int8),
                    categories,
                    ordered=spec["ordered"],
                )
                if colors is not None:
                    uns.create_dataset(
                        f"{key}_colors", data=np.array(colors, dtype=_TEXT)
                    )
        obs.attrs["column-order"] = np.array(
            [key for name in columns for key in (name, _twin(name))], dtype=_TEXT
        )
        var = h5.create_group("var")
        _write_index(var, _GENES)
        # The feature names that the reader looks for, so it warns of nothing.
        var.create_dataset(
            "gene_short_name", data=np.array([name.upper() for name in _GENES], _TEXT)
        )
        _categorical(
            var,
            "biotype",
            np.array(_BIOTYPE["codes"], dtype=np.int8),
            _BIOTYPE["categories"],
        )
        var.attrs["column-order"] = np.array(["gene_short_name", "biotype"], _TEXT)


def _import(source: Path, destination: Path, **options: Any) -> dict[str, Any]:
    """Import the file with ``clusters`` as cluster labels, and open the store."""
    with _warnings() as messages:
        reader = H5adReader(str(source), cluster_keys=("clusters",))
        try:
            result = H5adToZarr(
                reader,
                str(destination),
                nthreads=1,
                analysis_assay="RNA",
                shuffle_cells=False,
                **options,
            ).dump()
        finally:
            reader.close()
    return {
        "store": open_store(destination),
        "clusters": result.clusterArtifacts["clusters"],
        "cells": result.cellSelection,
        "warnings": messages,
    }


@pytest.fixture(scope="module")
def source(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("alike_source") / "source.h5ad"
    _write_source(path)
    return path


@pytest.fixture(scope="module")
def imports(source: Path, tmp_path_factory) -> dict[str, dict[str, Any]]:
    """An import of label columns and a default import, which stores categorical columns."""
    folder = tmp_path_factory.mktemp("alike_imports")
    return {
        "labels": _import(source, folder / "labels.zarr", categorical_columns=False),
        "categorical": _import(source, folder / "categorical.zarr"),
    }


@pytest.fixture(scope="module")
def earlier(source: Path, tmp_path_factory) -> dict[str, Any]:
    """A store whose attributes are those that CyteArc 1.4.0 stored.

    CyteArc 1.4.0 imported label columns, which ``categorical_columns=False``
    still stores, and kept one level per category.
    """
    path = tmp_path_factory.mktemp("alike_earlier") / "earlier.zarr"
    imported = _import(source, path, categorical_columns=False)
    root = zarr.open_group(str(path), mode="r+")
    for name, attributes in _EARLIER_CELL_ATTRIBUTES.items():
        root["cellData"][name].attrs.put(attributes)
    root["RNA/featureData/biotype"].attrs.put(_EARLIER_BIOTYPE_ATTRIBUTES)
    artifact_group(root, imported["clusters"])["values"].attrs.put(
        _EARLIER_CLUSTER_ATTRIBUTES
    )
    return {**imported, "store": open_store(path)}


def _decoded(spec: dict[str, Any]) -> list[Any]:
    """Each row's label as the file decodes it, None when missing."""
    return [None if code < 0 else spec["categories"][code] for code in spec["codes"]]


def _exported(store: Any, folder: Path, route: str) -> Any:
    """Export a store and read the file back with AnnData."""
    path = folder / "export.h5ad"
    if route == "to_anndata":
        # AnnData writes the categoricals and colors that the object holds.
        store.to_anndata().write_h5ad(path)
    else:
        cytearc.to_h5ad(store.RNA, str(path))
    return anndata.read_h5ad(path)


def _categorical_of(series: Any) -> tuple[list[Any], list[int], bool]:
    """The categories, codes and order of an exported categorical."""
    return (
        series.cat.categories.tolist(),
        series.cat.codes.tolist(),
        bool(series.cat.ordered),
    )


@pytest.mark.parametrize("layout", ["labels", "categorical"])
def test_an_import_stores_one_level_per_label_with_its_first_color(
    imports, layout: str
) -> None:
    store = imports[layout]["store"]
    cells = store.zw["cellData"]
    for name, spec in _ALIKE.items():
        # A categorical column cannot tell the categories apart, so either
        # layout stores the column of labels, each row as its category reads.
        assert isinstance(store.cells._get_array(name), zarr.Array), name
        labels = store.cells.fetch_all(name).tolist()
        missing = [code < 0 for code in spec["codes"]]
        assert [
            None if absent else label
            for label, absent in zip(labels, missing, strict=True)
        ] == _decoded(spec), name
        attributes = dict(cells[name].attrs)
        assert attributes["levels"] == spec["levels"], name
        assert attributes["ordered"] is spec["ordered"], name
        twin = dict(cells[_twin(name)].attrs)
        # Plots draw the levels in their order, with the colors of the twin.
        assert attributes["display"] == twin["display"], name
        display = attributes["display"]["categories"]
        assert [category["value"] for category in display] == spec["levels"], name
        if spec["level_colors"] is not None:
            colors = [category["color"] for category in display]
            assert colors == spec["level_colors"], name
        assert ("missing_label" in attributes["display"]) is any(missing), name
        # The twin, whose categories differ, is a categorical column by default.
        twin_array = store.cells._get_array(_twin(name))
        categorical = isinstance(twin_array, CategoricalColumn)
        assert categorical is (layout == "categorical"), name
    # The labels keep the sign that their category gives them, and the level
    # of -0.0 and 0.0 is the first of them, -0.0.
    assert np.signbit(store.cells.fetch_all("zero")).tolist() == [
        True,
        False,
        False,
        False,
        True,
        False,
    ]
    assert np.signbit(cells["zero"].attrs["levels"]).tolist() == [True, False]
    # The cluster labels keep their levels, order and colors on their values.
    values = artifact_group(store.zw, imports[layout]["clusters"])["values"]
    clusters = dict(values.attrs)
    assert clusters["levels"] == _CLUSTERS["levels"]
    assert clusters["ordered"] is True
    assert [category["color"] for category in clusters["display"]["categories"]] == (
        _CLUSTERS["level_colors"]
    )
    biotype = dict(store.zw["RNA/featureData/biotype"].attrs)
    assert biotype["levels"] == _BIOTYPE["levels"]
    assert "display" not in biotype
    assert store.RNA.feats.fetch_all("biotype").tolist() == [
        "coding",
        "coding",
        "lnc",
        "",
    ]


def test_plots_read_the_levels_and_colors_of_an_import(imports) -> None:
    from cytearc.plotting._display import (
        stored_artifact_scale,
        stored_categorical_scale,
    )

    imported = imports["categorical"]
    scale = stored_categorical_scale(imported["store"], "cell_type")
    assert scale is not None
    assert scale.order == tuple(_ALIKE["cell_type"]["levels"])
    assert scale.palette == dict(
        zip(
            _ALIKE["cell_type"]["levels"],
            _ALIKE["cell_type"]["level_colors"],
            strict=True,
        )
    )
    clusters = stored_artifact_scale(imported["store"], imported["clusters"])
    assert clusters is not None
    assert clusters.order == ("c1", "c2")


@pytest.mark.parametrize("layout", ["labels", "categorical"])
def test_an_import_warns_only_of_the_categoricals_that_it_stores_as_labels(
    imports, layout: str
) -> None:
    messages = imports[layout]["warnings"]
    if layout == "labels":
        # An import of label columns stores every categorical so, quietly.
        assert messages == []
        return
    # Each categorical whose categories read alike is stored as labels with
    # one warning, which names it and one of the categories; giving each of
    # its labels one level adds no warning.
    prefixes = [
        f"Stored {table} metadata column {name!r} as a column of its labels, not "
        f"as a categorical column. Column {name!r} has categories that read alike "
        "once stored, as text or numbers, so a categorical column cannot tell "
        "them apart, such as "
        for table, name in [
            *(("cell", name) for name in _ALIKE),
            ("RNA feature", "biotype"),
        ]
    ]
    assert len(messages) == len(prefixes)
    for prefix in prefixes:
        assert sum(message.startswith(prefix) for message in messages) == 1, prefix


@pytest.mark.parametrize("layout", ["labels", "categorical"])
@pytest.mark.parametrize("route", ["to_anndata", "to_h5ad"])
def test_exports_write_one_category_per_label(
    imports, tmp_path: Path, layout: str, route: str
) -> None:
    exported = _exported(imports[layout]["store"], tmp_path, route)

    for name, spec in _ALIKE.items():
        categorical = _categorical_of(exported.obs[name])
        assert categorical == (spec["levels"], spec["level_codes"], spec["ordered"])
        # The column exports as its twin does.
        assert categorical == _categorical_of(exported.obs[_twin(name)]), name
        colors = exported.uns[f"{name}_colors"].tolist()
        assert colors == exported.uns[f"{_twin(name)}_colors"].tolist(), name
        if spec["level_colors"] is not None:
            assert colors == spec["level_colors"], name
    assert np.signbit(exported.obs["zero"].cat.categories).tolist() == [True, False]
    assert _categorical_of(exported.var["biotype"]) == (
        _BIOTYPE["levels"],
        _BIOTYPE["level_codes"],
        False,
    )


def test_an_export_imports_as_categorical_columns_of_the_same_labels(
    imports, tmp_path: Path
) -> None:
    store = imports["categorical"]["store"]
    cytearc.to_h5ad(store.RNA, str(tmp_path / "export.h5ad"))
    reader = H5adReader(str(tmp_path / "export.h5ad"))
    with _warnings() as messages:
        try:
            H5adToZarr(
                reader, str(tmp_path / "again.zarr"), nthreads=1, shuffle_cells=False
            ).dump()
        finally:
            reader.close()

    # The file names each label once, so a default import stores each column
    # as a categorical column of the levels, quietly, reading the same labels.
    again = open_store(tmp_path / "again.zarr")
    for name, spec in _ALIKE.items():
        assert not [message for message in messages if repr(name) in message], name
        column = again.cells._get_array(name)
        assert isinstance(column, CategoricalColumn), name
        assert column.categories.tolist() == spec["levels"], name
        assert column.codes[:].tolist() == spec["level_codes"], name
        labels = again.cells.fetch_all(name).tolist()
        assert labels == store.cells.fetch_all(name).tolist(), name


def test_cluster_labels_export_one_category_per_label(imports) -> None:
    imported = imports["categorical"]
    exported = imported["store"].to_anndata(
        cell_selection=imported["cells"], obs={"labels": imported["clusters"]}
    )
    assert _categorical_of(exported.obs["labels"]) == (
        _CLUSTERS["levels"],
        _CLUSTERS["level_codes"],
        True,
    )
    assert exported.uns["labels_colors"].tolist() == _CLUSTERS["level_colors"]


def _snapshot_categories(store: Any, labels: Any) -> dict[str, Any]:
    """Snapshot ``labels`` over the cells but the last two, which miss some labels.

    Returns the attributes that the snapshot keeps.
    """
    from tests.storage_helpers import by_cell_id

    labeled = np.arange(_N_CELLS) < _N_CELLS - 2
    store.cells.insert("labeled", by_cell_id(store.cells, labeled), overwrite=True)
    snapshot = store.snapshot_cluster_labels(
        labels, cell_selection=store.snapshot_cell_selection("labeled")
    )
    return dict(store.artifacts.load(snapshot)["values"].attrs)


def test_snapshots_keep_one_level_per_label_from_either_layout(
    source: Path, tmp_path: Path
) -> None:
    # The twins name each label once, so the default import stores them as
    # categorical columns, and an import of label columns as labels with
    # their levels.
    names = ["cell_type", "stage", "count", "flag"]
    found: dict[str, dict[str, dict[str, Any]]] = {}
    for layout, options in (
        ("labels", {"categorical_columns": False}),
        ("categorical", {}),
    ):
        store = _import(source, tmp_path / f"{layout}.zarr", **options)["store"]
        found[layout] = {
            name: _snapshot_categories(store, _twin(name)) for name in names
        }
        if layout == "labels":
            displays = {
                name: store.zw["cellData"][_twin(name)].attrs["display"]
                for name in names
            }
        else:
            assert all(
                isinstance(store.cells._get_array(_twin(name)), CategoricalColumn)
                for name in names
            )

    for name in names:
        spec = _ALIKE[name]
        assert found["labels"][name] == {
            "levels": spec["levels"],
            "ordered": spec["ordered"],
            "display": displays[name],
        }, name
        assert found["categorical"][name] == found["labels"][name], name


def test_snapshots_of_an_earlier_store_keep_one_level_per_label(
    source: Path, tmp_path: Path
) -> None:
    imported = _import(source, tmp_path / "earlier.zarr", categorical_columns=False)
    root = zarr.open_group(str(tmp_path / "earlier.zarr"), mode="r+")
    root["cellData"]["stage"].attrs.put(_EARLIER_CELL_ATTRIBUTES["stage"])
    artifact_group(root, imported["clusters"])["values"].attrs.put(
        _EARLIER_CLUSTER_ATTRIBUTES
    )
    store = open_store(tmp_path / "earlier.zarr")

    assert _snapshot_categories(store, "stage") == {
        "levels": ["early", "mid", "late", "unused"],
        "ordered": True,
    }
    # The snapshot keeps the levels but not the link of the clusters' missing
    # mask, since each of its cells has a label.
    assert _snapshot_categories(store, imported["clusters"]) == {
        "levels": ["c1", "c2"],
        "ordered": True,
    }


def test_the_earlier_store_holds_the_attributes_of_cytearc_1_4_0(earlier) -> None:
    root = earlier["store"].zw
    for name, attributes in _EARLIER_CELL_ATTRIBUTES.items():
        assert dict(root["cellData"][name].attrs) == attributes, name
    biotype = root["RNA/featureData/biotype"]
    assert dict(biotype.attrs) == _EARLIER_BIOTYPE_ATTRIBUTES
    values = artifact_group(root, earlier["clusters"])["values"]
    assert dict(values.attrs) == _EARLIER_CLUSTER_ATTRIBUTES


@pytest.mark.parametrize("route", ["to_anndata", "to_h5ad"])
def test_a_store_of_an_earlier_release_exports_one_category_per_label(
    earlier, tmp_path: Path, route: str
) -> None:
    store = earlier["store"]
    exported = _exported(store, tmp_path, route)

    for name, spec in _ALIKE.items():
        assert _categorical_of(exported.obs[name]) == (
            spec["levels"],
            spec["level_codes"],
            spec["ordered"],
        ), name
        # The release kept no display, so the export has no colors to write.
        assert f"{name}_colors" not in exported.uns, name
    assert np.signbit(exported.obs["zero"].cat.categories).tolist() == [True, False]
    assert _categorical_of(exported.var["biotype"]) == (
        _BIOTYPE["levels"],
        _BIOTYPE["level_codes"],
        False,
    )
    labels = store.to_anndata(
        cell_selection=earlier["cells"], obs={"labels": earlier["clusters"]}
    )
    assert _categorical_of(labels.obs["labels"]) == (
        _CLUSTERS["levels"],
        _CLUSTERS["level_codes"],
        True,
    )


def _legend(result: Any) -> list[str]:
    return [text.get_text() for text in result.figure.legends[0].get_texts()]


def test_plots_of_an_earlier_store_draw_ordered_levels_once(earlier) -> None:
    import matplotlib.pyplot as plt

    from cytearc.datastore._pipeline.fields import _frozen_column_display
    from cytearc.plotting._display import (
        stored_artifact_scale,
        stored_categorical_scale,
    )

    store = earlier["store"]
    assert stored_categorical_scale(store, "stage") == splt.CategoricalScale(
        order=("early", "mid", "late", "unused")
    )
    assert stored_artifact_scale(store, earlier["clusters"]) == (
        splt.CategoricalScale(order=("c1", "c2"))
    )
    try:
        # The plotted cells hold three of the four stages and both clusters.
        assert _legend(splt.composition(store, category_by="stage", show=False)) == [
            "early",
            "mid",
            "late",
        ]
        assert _legend(
            splt.composition(store, categories=earlier["clusters"], show=False)
        ) == ["c1", "c2", "NA"]
    finally:
        plt.close("all")
    # A run that freezes the column records its levels as the display of
    # its field.
    display = _frozen_column_display(store.zw, "stage")
    assert display is not None
    assert [category["value"] for category in display["categories"]] == [
        "early",
        "mid",
        "late",
        "unused",
    ]


def test_a_merge_with_an_earlier_store_keeps_one_level_per_label(
    earlier, imports, tmp_path: Path
) -> None:
    from cytearc.merge import DataStoreMerge

    with _warnings() as messages:
        DataStoreMerge(
            [earlier["store"], imports["categorical"]["store"]],
            str(tmp_path / "merged.zarr"),
            ["earlier", "current"],
            nthreads=1,
        ).dump()

    # The sources agree on every attribute once each label has one level.
    assert [message for message in messages if "differ" in message] == []
    merged = open_store(tmp_path / "merged.zarr")
    cells = merged.zw["cellData"]
    for name, spec in _ALIKE.items():
        attributes = dict(cells[f"orig_{name}"].attrs)
        assert attributes["levels"] == spec["levels"], name
        assert attributes["ordered"] is spec["ordered"], name
    biotype = dict(merged.zw["RNA/featureData/biotype"].attrs)
    assert biotype["levels"] == _BIOTYPE["levels"]
    exported = merged.to_anndata()
    stage = exported.obs["orig_stage"]
    assert stage.cat.categories.tolist() == _ALIKE["stage"]["levels"]
    assert stage.cat.ordered


@pytest.mark.parametrize(
    ("levels", "positions"),
    [
        ([], []),
        (["a", "b"], [0, 1]),
        (["b", "a", "b", "c", "a"], [0, 1, 3]),
        ([-0.0, 1.5, 0.0], [0, 1]),
        ([0.0, -0.0], [0]),
        ([7, 3, 7], [0, 1]),
        ([False, True, False, True], [0, 1]),
        # Equal numbers of two types read alike; text never reads as a number.
        ([1, 1.0, 2], [0, 2]),
        (["1", 1], [0, 1]),
    ],
)
def test_distinct_level_positions_keep_the_first_of_levels_that_read_alike(
    levels: list[Any], positions: list[int]
) -> None:
    assert distinct_level_positions(levels) == positions


def test_recorded_categories_read_repeated_levels_as_the_first() -> None:
    recorded = recorded_categories({"levels": ["b", "a", "b", "c"], "ordered": True})
    assert recorded.levels == ("b", "a", "c")
    assert recorded.ordered_levels == ("b", "a", "c")
    zeros = recorded_categories({"levels": [-0.0, 1.5, 0.0]}).levels
    assert zeros == (-0.0, 1.5)
    assert zeros is not None and np.signbit(zeros[0])


def test_merges_compare_and_keep_the_first_of_levels_that_read_alike() -> None:
    from zarr.storage import MemoryStore

    from cytearc.merge.metadata import (
        _merged_column_attributes,
        _reconciled_cell_attributes,
        _SourceConflict,
    )

    root = zarr.open_group(store=MemoryStore(), mode="w")

    def array(name: str, attributes: dict[str, Any]) -> Any:
        stored = root.create_array(name, data=np.zeros(2))
        stored.attrs.update(attributes)
        return stored

    earlier = array("earlier", {"levels": ["b", "a", "b"], "ordered": True})
    current = array("current", {"levels": ["b", "a"], "ordered": True})
    text_labels = np.dtype("U1")
    assert _reconciled_cell_attributes([earlier, current], [], text_labels) == (
        {"levels": ["b", "a"], "ordered": True},
        [],
    )
    # Two sources that each repeat a level keep it once.
    assert _reconciled_cell_attributes([earlier, earlier], [], text_labels) == (
        {"levels": ["b", "a"], "ordered": True},
        [],
    )
    assert _merged_column_attributes([earlier, current], "kind") == {
        "levels": ["b", "a"],
        "ordered": True,
    }
    # Levels in another order keep the first source's order, which is then
    # no order that every source agrees on.
    reordered = array("reordered", {"levels": ["a", "b"], "ordered": True})
    assert _reconciled_cell_attributes([earlier, reordered], [], text_labels) == (
        {"levels": ["b", "a"], "ordered": False},
        ["ordered"],
    )
    with pytest.raises(_SourceConflict, match="levels"):
        _merged_column_attributes([earlier, reordered], "kind")
    # Levels that are not a list are compared as stored.
    text = array("text", {"levels": "bab"})
    assert _reconciled_cell_attributes([text, text], [], text_labels) == (
        {"levels": "bab"},
        [],
    )
