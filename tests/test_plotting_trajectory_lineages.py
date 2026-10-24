"""Lineage plots: profile heatmaps, feature trends, curves, and cluster trees.

The session results profile three lineages of the 1K PBMC store, which end in
Leiden clusters 2, 3, and 4, and root a cluster tree at Leiden cluster 1. Each
plot is drawn through ``datastore.plots`` and ``cytearc.plotting`` and checked
against the loaded result it draws.
"""

import dataclasses
import inspect
import re
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

import cytearc.plotting as splt
from cytearc.datastore.datastore import DataStore
from cytearc.datastore.namespaces import TrajectoryAccessor
from cytearc.datastore.plot_accessor import DataStorePlotAccessor
from cytearc.plotting import lineages as lineage_plots
from cytearc.plotting._data import _resolve_layout
from cytearc.plotting._style import (
    DEFAULT_PANEL_INCHES,
    colormap_palette,
    default_point_size,
    padded_square_limits,
)
from cytearc.storage import ArtifactRef
from cytearc.trajectory.curves import smooth_against_progress

# Lineages are drawn under the string form of their sink labels.
LINEAGES = ("2", "3", "4")
N_BINS = 12
N_POINTS = 40
TREE_ROOT = 1
PLOTS = ("lineage_heatmap", "lineage_trends", "lineage_curves", "lineage_tree")


@pytest.fixture(autouse=True)
def _figures_are_closed():
    """No test leaves a figure open; a plot call that raises closes its own figure."""
    before = set(plt.get_fignums())
    yield
    leaked = set(plt.get_fignums()) - before
    for number in leaked:
        plt.close(number)
    assert not leaked, f"Figures left open: {sorted(leaked)}"


@pytest.fixture
def closing():
    """Register results whose figures close at teardown, even after a failure.

    It tears down before the autouse leak check, which therefore sees only
    figures that no result owns.
    """
    results: list[splt.PlotResult] = []

    def register(result: splt.PlotResult) -> splt.PlotResult:
        results.append(result)
        return result

    yield register
    for result in results:
        result.close()


@pytest.fixture(scope="module")
def loaded_aggregation(datastore, lineage_aggregation):
    return datastore.trajectory.load_lineage_aggregation(lineage_aggregation)


@pytest.fixture(scope="module")
def loaded_curves(datastore, lineage_curves):
    return datastore.trajectory.load_curves(lineage_curves)


@pytest.fixture(scope="module")
def loaded_fate(datastore, loaded_curves):
    return datastore.trajectory.load_fate(loaded_curves.fate)


@pytest.fixture(scope="module")
def loaded_tree(datastore, lineage_tree):
    return datastore.trajectory.load_lineage_tree(lineage_tree)


@pytest.fixture(scope="module")
def layout(datastore, umap, loaded_curves, loaded_tree):
    """The session UMAP, which embeds exactly the cells of the curves and the tree.

    Cluster labels are matched to the layout rows by cell-table row, not by
    position, so the expected values do not assume an alignment.
    """
    coordinates, cell_indices, selection = _resolve_layout(datastore, umap)
    assert selection == loaded_curves.cell_selection == loaded_tree.cell_selection
    values = datastore.artifacts.load_values(
        loaded_tree.labels, cell_selection=selection
    )
    by_cell = dict(
        zip(
            np.asarray(values.cell_idx).tolist(),
            np.asarray(values.values).tolist(),
            strict=True,
        )
    )
    return SimpleNamespace(
        ref=umap,
        coordinates=coordinates,
        cell_indices=cell_indices,
        selection=selection,
        labels=np.asarray([by_cell[cell] for cell in cell_indices.tolist()]),
    )


@pytest.fixture(scope="module")
def feature_names(loaded_aggregation) -> list[str]:
    return np.asarray(loaded_aggregation.feature_names).astype(str).tolist()


def _module_peak_order(loaded: Any, columns: list[int]) -> list[int]:
    """Feature rows by module, then by the peak of the chosen lineages' profiles.

    ``sorted`` is stable, so features that tie keep their stored order.
    """
    data = np.asarray(loaded.data)[:, columns, :]
    peaks = data.reshape(data.shape[0], -1).argmax(axis=1)
    clusters = np.asarray(loaded.feature_clusters)
    return sorted(range(data.shape[0]), key=lambda row: (clusters[row], peaks[row]))


def _feature_row(feature_names: list[str], name: str) -> int:
    return [label.lower() for label in feature_names].index(name.lower())


def _expected_curve(
    loaded_curves: Any,
    loaded_fate: Any,
    coordinates: np.ndarray,
    lineage: int,
    df: float,
) -> np.ndarray:
    """A lineage's curve in the layout, smoothed with the lineage's fate weights."""
    column = loaded_curves.lineages.index(lineage)
    sink = loaded_fate.sink_labels.index(lineage)
    valid = np.asarray(loaded_curves.valid, dtype=bool)
    return smooth_against_progress(
        coordinates[valid],
        np.asarray(loaded_curves.progress)[valid, column],
        np.asarray(loaded_fate.values, dtype=np.float64)[valid, sink],
        n_points=loaded_curves.points.shape[1],
        degrees_of_freedom=df,
    )


def _tree_leaves(loaded_tree: Any) -> set[Any]:
    """Clusters that join exactly one other cluster in the tree's edges."""
    ends = [*loaded_tree.edges["source"], *loaded_tree.edges["target"]]
    return {cluster for cluster in loaded_tree.clusters if ends.count(cluster) == 1}


def _other_ref(ref: ArtifactRef) -> ArtifactRef:
    return dataclasses.replace(ref, artifact_id="0" * 64)


# Forwarding through the datastore accessor.

_FAKE_AGGREGATION = ArtifactRef(
    scope="assay", assay="RNA", kind="lineage_aggregation", artifact_id="a" * 64
)
_FAKE_CURVES = ArtifactRef(
    scope="assay", assay="RNA", kind="lineage_curves", artifact_id="b" * 64
)
_FAKE_TREE = ArtifactRef(
    scope="assay", assay="RNA", kind="lineage_tree", artifact_id="c" * 64
)
_FAKE_EMBEDDING = ArtifactRef(
    scope="assay", assay="RNA", kind="embedding", artifact_id="d" * 64
)


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        (
            "lineage_heatmap",
            {
                "aggregation": _FAKE_AGGREGATION,
                "lineages": [3, 2],
                "show_features": ["CD3E"],
                "vmax": 3.0,
                "theme": "paper",
                "show": False,
            },
        ),
        (
            "lineage_trends",
            {
                "aggregation": _FAKE_AGGREGATION,
                "features": ["CD3E", "LYZ"],
                "n_columns": 1,
                "line_width": 1.0,
                "show_legend": False,
            },
        ),
        (
            "lineage_curves",
            {
                "curves": _FAKE_CURVES,
                "layout": _FAKE_EMBEDDING,
                "lineages": [4],
                "df": 4.0,
                "point_size": 2.0,
                "target": object(),
            },
        ),
        (
            "lineage_tree",
            {
                "tree": _FAKE_TREE,
                "layout": _FAKE_EMBEDDING,
                "show_cells": False,
                "node_size": 9.0,
                "labels": False,
                "figsize": (3.0, 3.0),
            },
        ),
    ],
)
def test_plot_accessor_forwards_lineage_plots_with_the_bound_store(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    kwargs: dict[str, Any],
):
    store = object.__new__(DataStore)
    accessor = DataStorePlotAccessor(store)
    sentinel = object()
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def canonical(*call_args: Any, **call_kwargs: Any) -> object:
        calls.append((call_args, call_kwargs))
        return sentinel

    monkeypatch.setattr(splt, name, canonical)
    method = getattr(accessor, name)
    bound = inspect.signature(method).bind(**kwargs)
    bound.apply_defaults()

    assert method(**kwargs) is sentinel
    # Every declared argument, defaults included, is passed by keyword.
    assert calls == [((store,), dict(bound.arguments))]


@pytest.fixture(scope="module")
def plot_arguments(
    lineage_aggregation, lineage_curves, lineage_tree, umap, feature_names
):
    return {
        "lineage_heatmap": {"aggregation": lineage_aggregation},
        "lineage_trends": {
            "aggregation": lineage_aggregation,
            "features": feature_names[:2],
        },
        "lineage_curves": {"curves": lineage_curves, "layout": umap},
        "lineage_tree": {"tree": lineage_tree, "layout": umap},
    }


@pytest.mark.parametrize("name", PLOTS)
def test_accessor_and_function_draw_the_same_plot(
    monkeypatch: pytest.MonkeyPatch,
    closing,
    datastore,
    plot_arguments,
    name: str,
):
    # Recording replaces showing, so no figure is displayed.
    shown: list[splt.PlotResult] = []
    monkeypatch.setattr(splt.PlotResult, "show", lambda result: shown.append(result))
    arguments = plot_arguments[name]

    bound = closing(getattr(datastore.plots, name)(**arguments))
    direct = closing(getattr(splt, name)(datastore, show=False, **arguments))

    # show defaults to True, so only the bound call showed its figure.
    assert shown == [bound]
    assert isinstance(bound, splt.PlotResult)
    assert isinstance(direct, splt.PlotResult)
    assert bound.owns_figure and direct.owns_figure
    assert bound.figure is not direct.figure
    assert list(bound.axes) == list(direct.axes)
    assert len(bound.figure.axes) == len(direct.figure.axes)
    assert bound.tables.keys() == direct.tables.keys()
    for table in bound.tables:
        pd.testing.assert_frame_equal(bound.tables[table], direct.tables[table])
    assert bound.legends == direct.legends
    assert bound.scales == direct.scales
    assert bound.provenance == direct.provenance
    assert bound.theme == direct.theme == "notebook"


# Private helpers.


@pytest.mark.parametrize(
    ("value", "expected"),
    [(2, 2.0), (0.25, 0.25), (np.float64(3.5), 3.5), (1e-9, 1e-9)],
)
def test_positive_returns_a_float(value, expected):
    resolved = lineage_plots._positive(value, "line_width")

    assert resolved == expected
    assert type(resolved) is float


@pytest.mark.parametrize("value", [True, False, "2", None, [1.0], np.bool_(True)])
def test_positive_rejects_values_that_are_not_real_numbers(value):
    with pytest.raises(TypeError, match="node_size must be a real number"):
        lineage_plots._positive(value, "node_size")


@pytest.mark.parametrize("value", [0, 0.0, -1, -0.5, np.nan, np.inf, -np.inf])
def test_positive_rejects_values_that_are_not_positive_and_finite(value):
    with pytest.raises(ValueError, match="df must be positive and finite"):
        lineage_plots._positive(value, "df")


@pytest.mark.parametrize("value", [np.int64(40), np.float32(40.0)])
def test_positive_accepts_numpy_scalars(value):
    assert lineage_plots._positive(value, "node_size") == 40.0


@pytest.mark.parametrize(
    ("available", "lineages", "expected"),
    [
        ((2, 3, 4), None, [0, 1, 2]),
        ((2, 3, 4), [4, 2], [2, 0]),
        ((2, 3, 4), (3,), [1]),
        ((2, 3, 4), [np.int64(4), 3.0], [2, 1]),
        (("erythroid", "myeloid"), ["myeloid"], [1]),
    ],
)
def test_lineage_selection_returns_columns_in_the_requested_order(
    available, lineages, expected
):
    assert lineage_plots._lineage_selection(available, lineages) == expected


@pytest.mark.parametrize(
    ("lineages", "error", "message"),
    [
        ("2", TypeError, "lineages must be a list of lineage labels or None"),
        (b"2", TypeError, "lineages must be a list of lineage labels or None"),
        ({2, 3}, TypeError, "lineages must be a list of lineage labels or None"),
        (2, TypeError, "lineages must be a list of lineage labels or None"),
        (
            np.array([2, 3]),
            TypeError,
            "lineages must be a list of lineage labels or None",
        ),
        ([], ValueError, "lineages must name at least one lineage"),
        ([2, 2], ValueError, "lineages must not repeat a lineage"),
        ([2, 3, 2.0], ValueError, "lineages must not repeat a lineage"),
        ([7], ValueError, "Lineage 7 is not in the result"),
        (["2"], ValueError, "Lineage '2' is not in the result"),
    ],
)
def test_lineage_selection_rejects_invalid_lineages(lineages, error, message):
    with pytest.raises(error, match=message):
        lineage_plots._lineage_selection((2, 3, 4), lineages)


def test_lineage_palette_spreads_a_colormap_over_the_lineages():
    palette = lineage_plots._lineage_palette(["4", "2", "3"], "viridis")

    assert list(palette) == ["4", "2", "3"]
    assert palette == colormap_palette(["4", "2", "3"], "viridis")
    assert len(set(palette.values())) == 3
    assert palette["4"] == to_hex(plt.get_cmap("viridis")(0.0))
    assert palette["3"] == to_hex(plt.get_cmap("viridis")(1.0))


def test_layout_for_returns_the_coordinates_of_the_trajectory_cells(datastore, layout):
    coordinates, cell_indices = lineage_plots._layout_for(
        datastore, layout.ref, layout.selection
    )

    np.testing.assert_array_equal(coordinates, layout.coordinates)
    np.testing.assert_array_equal(cell_indices, layout.cell_indices)
    assert coordinates.shape == (len(cell_indices), 2)
    with pytest.raises(ValueError, match="exact cell-selection artifact"):
        lineage_plots._layout_for(datastore, layout.ref, _other_ref(layout.selection))


# Lineage heatmap.


def test_lineage_heatmap_orders_features_by_module_and_peak(
    closing, datastore, lineage_aggregation, lineage_fate, mark_hvgs, loaded_aggregation
):
    loaded = loaded_aggregation
    n_features = loaded.data.shape[0]
    order = _module_peak_order(loaded, [0, 1, 2])
    flat = np.asarray(loaded.data).reshape(n_features, -1)

    result = closing(
        splt.lineage_heatmap(datastore, aggregation=lineage_aggregation, show=False)
    )

    assert result.owns_figure is True
    assert set(result.axes) == {
        "feature_clusters",
        "colorbar",
        *((panel, name) for panel in ("heatmap", "pseudotime") for name in LINEAGES),
    }
    assert len(result.figure.axes) == len(result.axes)
    profiles = result.tables["profiles"]
    assert profiles.shape == (n_features, len(LINEAGES) * N_BINS)
    assert profiles.index.name == "feature"
    assert list(profiles.columns.names) == ["lineage", "bin"]
    assert list(profiles.columns) == [
        (name, index) for name in LINEAGES for index in range(N_BINS)
    ]
    np.testing.assert_array_equal(profiles.to_numpy(), flat[order])
    assert profiles.index.tolist() == np.asarray(loaded.feature_names)[order].tolist()
    features = result.tables["features"]
    assert list(features.columns) == ["feature_index", "feature", "cluster"]
    np.testing.assert_array_equal(
        features["feature_index"], np.asarray(loaded.feature_indices)[order]
    )
    np.testing.assert_array_equal(features["feature"], profiles.index)
    np.testing.assert_array_equal(
        features["cluster"], np.asarray(loaded.feature_clusters)[order]
    )
    bins = result.tables["bins"]
    assert list(bins.columns) == ["lineage", "bin", "pseudotime"]
    assert bins["lineage"].tolist() == [
        name for name in LINEAGES for _ in range(N_BINS)
    ]
    assert bins["bin"].tolist() == list(range(N_BINS)) * len(LINEAGES)
    np.testing.assert_array_equal(
        bins["pseudotime"], np.asarray(loaded.bin_pseudotime).ravel()
    )

    bin_pseudotime = np.asarray(loaded.bin_pseudotime)
    for position, name in enumerate(LINEAGES):
        heatmap = result.axes[("heatmap", name)]
        np.testing.assert_array_equal(
            heatmap.images[0].get_array(), profiles[name].to_numpy()
        )
        first, last = bin_pseudotime[position, [0, -1]]
        assert heatmap.get_xlabel() == f"{first:.2g} to {last:.2g}"
        assert list(heatmap.get_yticks()) == []
        strip = result.axes[("pseudotime", name)]
        np.testing.assert_array_equal(
            strip.images[0].get_array(), bin_pseudotime[[position]]
        )
        # Every strip shares one pseudotime color range.
        assert strip.images[0].get_clim() == pytest.approx(
            (bin_pseudotime.min(), bin_pseudotime.max())
        )
        assert strip.get_title() == name
    lefts = [result.axes[("heatmap", name)].get_position().x0 for name in LINEAGES]
    assert lefts == sorted(lefts)

    modules = sorted(set(np.asarray(loaded.feature_clusters).tolist()))
    palette = colormap_palette(modules, "tab20")
    cluster_ax = result.axes["feature_clusters"]
    np.testing.assert_array_equal(
        cluster_ax.images[0].get_array().ravel(),
        [modules.index(cluster) for cluster in features["cluster"]],
    )
    assert [to_hex(color) for color in cluster_ax.images[0].get_cmap().colors] == [
        palette[module] for module in modules
    ]
    assert cluster_ax.get_ylabel() == f"{n_features} features"
    centres = {text.get_text(): text.get_position()[1] for text in cluster_ax.texts}
    assert set(centres) == {str(module) for module in modules}
    for module in modules:
        rows = np.flatnonzero(features["cluster"].to_numpy() == module)
        # Each module is one contiguous block of rows, labelled at its centre.
        assert rows.max() - rows.min() + 1 == rows.size
        assert centres[str(module)] == pytest.approx(rows.mean())
    assert result.axes["colorbar"].get_ylabel() == "standardized expression"

    assert result.scales == (
        splt.ColorScale(cmap="coolwarm", vmin=-2.0, vmax=2.0),
        splt.CategoricalScale(order=tuple(modules), palette=palette),
    )
    assert result.legends == (
        splt.LegendSpec(
            kind="colorbar",
            label="standardized expression",
            extras={"vmin": -2.0, "vmax": 2.0},
        ),
        splt.LegendSpec(kind="categorical", label="feature modules"),
    )
    assert result.provenance.assay == "RNA"
    assert result.provenance.renderer == "matplotlib"
    assert result.provenance.notes == ("lineage_heatmap", "aggregated")
    assert result.provenance.extras == {
        "aggregation": lineage_aggregation.to_dict(),
        "fate": lineage_fate.to_dict(),
        "cell_selection": loaded.cell_selection.to_dict(),
        "feature_selection": mark_hvgs.to_dict(),
        "lineages": list(LINEAGES),
        "n_features": n_features,
        "n_bins": N_BINS,
        "show_features": [],
    }
    assert result.theme == "notebook"
    assert tuple(result.figure.get_size_inches()) == pytest.approx(
        (1.2 + 2.4 * len(LINEAGES), 8.0)
    )


@pytest.mark.parametrize(
    ("lineages", "columns"),
    [([4, 2], [2, 0]), ((3,), [1])],
)
def test_lineage_heatmap_draws_the_chosen_lineages_in_order(
    closing, datastore, lineage_aggregation, loaded_aggregation, lineages, columns
):
    loaded = loaded_aggregation
    names = [str(lineage) for lineage in lineages]
    order = _module_peak_order(loaded, columns)
    expected = np.asarray(loaded.data)[:, columns, :].reshape(len(order), -1)[order]

    result = closing(
        datastore.plots.lineage_heatmap(
            aggregation=lineage_aggregation, lineages=lineages, show=False
        )
    )

    profiles = result.tables["profiles"]
    assert profiles.columns.get_level_values("lineage").unique().tolist() == names
    np.testing.assert_array_equal(profiles.to_numpy(), expected)
    np.testing.assert_array_equal(
        result.tables["features"]["feature_index"],
        np.asarray(loaded.feature_indices)[order],
    )
    drawn_pseudotime = np.asarray(loaded.bin_pseudotime)[columns]
    np.testing.assert_array_equal(
        result.tables["bins"]["pseudotime"], drawn_pseudotime.ravel()
    )
    assert set(result.axes) == {
        "feature_clusters",
        "colorbar",
        *((panel, name) for panel in ("heatmap", "pseudotime") for name in names),
    }
    lefts = [result.axes[("heatmap", name)].get_position().x0 for name in names]
    assert lefts == sorted(lefts)
    for name in names:
        strip = result.axes[("pseudotime", name)]
        assert strip.images[0].get_clim() == pytest.approx(
            (drawn_pseudotime.min(), drawn_pseudotime.max())
        )
    assert result.provenance.extras["lineages"] == names
    assert tuple(result.figure.get_size_inches()) == pytest.approx(
        (1.2 + 2.4 * len(names), 8.0)
    )


def test_lineage_heatmap_labels_requested_features_on_the_last_panel(
    closing, datastore, lineage_aggregation, feature_names
):
    requested = [feature_names[5].lower(), "NOT_A_FEATURE", feature_names[17]]

    result = closing(
        splt.lineage_heatmap(
            datastore,
            aggregation=lineage_aggregation,
            show_features=requested,
            feature_label_fontsize=7,
            show=False,
        )
    )

    rows = result.tables["profiles"].index.str.lower().tolist()
    last = result.axes[("heatmap", LINEAGES[-1])]
    labels = last.get_yticklabels()
    # Features keep the caller's spelling and order; unknown names are skipped.
    assert [label.get_text() for label in labels] == [
        feature_names[5].lower(),
        feature_names[17],
    ]
    assert {label.get_fontsize() for label in labels} == {7}
    assert list(last.get_yticks()) == [
        rows.index(feature_names[5].lower()),
        rows.index(feature_names[17].lower()),
    ]
    assert last.yaxis.get_ticks_position() == "right"
    for name in LINEAGES[:-1]:
        assert list(result.axes[("heatmap", name)].get_yticks()) == []
    assert result.provenance.extras["show_features"] == requested


def test_lineage_heatmap_feature_labels_clear_the_colorbar(
    closing, datastore, lineage_aggregation, feature_names
):
    requested = feature_names[:5]
    result = closing(
        splt.lineage_heatmap(
            datastore,
            aggregation=lineage_aggregation,
            show_features=requested,
            show=False,
        )
    )
    result.figure.canvas.draw()
    renderer = result.figure.canvas.get_renderer()
    colorbar = result.axes["colorbar"].get_window_extent(renderer)
    # The labels may sit on any panel, as long as every one stays clear.
    labels = [
        label.get_window_extent(renderer)
        for axis in result.figure.axes
        if axis is not result.axes["colorbar"]
        for label in axis.get_yticklabels()
        if label.get_visible() and label.get_text() in requested
    ]
    assert len(labels) == len(requested)
    assert not any(label.overlaps(colorbar) for label in labels)


def test_lineage_heatmap_without_colorbar_in_a_custom_style(
    closing, datastore, lineage_aggregation, loaded_aggregation
):
    result = closing(
        datastore.plots.lineage_heatmap(
            aggregation=lineage_aggregation,
            figsize=(6.0, 5.0),
            vmin=-1.0,
            vmax=3.0,
            heatmap_cmap="RdBu_r",
            pseudotime_cmap="magma",
            clusterbar_cmap="Set2",
            tick_fontsize=6,
            axis_fontsize=9,
            theme="dark",
            show_legend=False,
            show=False,
        )
    )

    assert "colorbar" not in result.axes
    assert len(result.figure.axes) == 1 + 2 * len(LINEAGES)
    assert tuple(result.figure.get_size_inches()) == (6.0, 5.0)
    heatmap = result.axes[("heatmap", "2")]
    image = heatmap.images[0]
    assert image.get_cmap().name == "RdBu_r"
    assert (image.norm.vmin, image.norm.vmax) == (-1.0, 3.0)
    assert heatmap.xaxis.label.get_fontsize() == 6
    strip = result.axes[("pseudotime", "2")]
    assert strip.images[0].get_cmap().name == "magma"
    assert strip.title.get_fontsize() == 9
    cluster_ax = result.axes["feature_clusters"]
    assert cluster_ax.yaxis.label.get_fontsize() == 9
    assert {text.get_fontsize() for text in cluster_ax.texts} == {6}
    modules = sorted(set(np.asarray(loaded_aggregation.feature_clusters).tolist()))
    assert result.scales == (
        splt.ColorScale(cmap="RdBu_r", vmin=-1.0, vmax=3.0),
        splt.CategoricalScale(
            order=tuple(modules), palette=colormap_palette(modules, "Set2")
        ),
    )
    assert result.legends[0].extras == {"vmin": -1.0, "vmax": 3.0}
    # The dark theme leaves the figure background transparent.
    assert result.theme == "dark"
    assert result.figure.patch.get_alpha() == 0


@pytest.mark.parametrize("aggregation", ["latest", None, 3])
def test_lineage_heatmap_requires_an_aggregation_ref(datastore, aggregation):
    with pytest.raises(TypeError, match="aggregation must be an ArtifactRef"):
        splt.lineage_heatmap(datastore, aggregation=aggregation, show=False).close()


def test_lineage_heatmap_rejects_a_ref_of_another_kind(datastore, lineage_curves):
    with pytest.raises(ValueError, match="lineage_aggregation artifact"):
        splt.lineage_heatmap(datastore, aggregation=lineage_curves, show=False).close()


def test_lineage_heatmap_shows_unscaled_profiles_as_mean_expression(
    closing, datastore, lineage_fate, mark_hvgs
):
    unscaled = datastore.trajectory.lineage_aggregation(
        lineage_fate, features=mark_hvgs, n_bins=12, n_clusters=5, z_scale=False
    )
    data = np.asarray(datastore.trajectory.load_lineage_aggregation(unscaled).data)

    result = closing(splt.lineage_heatmap(datastore, aggregation=unscaled, show=False))
    chosen = closing(
        splt.lineage_heatmap(
            datastore, aggregation=unscaled, vmin=0.5, vmax=1.5, show=False
        )
    )

    # Means are not z-scores, so the colours span the data, not -2 to 2.
    low, high = float(data.min()), float(np.percentile(data, 99.0))
    assert result.axes["colorbar"].get_ylabel() == "mean expression"
    assert result.scales[0] == splt.ColorScale(cmap="coolwarm", vmin=low, vmax=high)
    assert result.legends[0] == splt.LegendSpec(
        kind="colorbar", label="mean expression", extras={"vmin": low, "vmax": high}
    )
    assert chosen.scales[0] == splt.ColorScale(cmap="coolwarm", vmin=0.5, vmax=1.5)


def test_lineage_heatmap_spans_sparse_unscaled_profiles_to_their_maximum(
    closing, monkeypatch, datastore, lineage_fate, mark_hvgs
):
    unscaled = datastore.trajectory.lineage_aggregation(
        lineage_fate, features=mark_hvgs, n_bins=12, n_clusters=5, z_scale=False
    )
    loaded = datastore.trajectory.load_lineage_aggregation(unscaled)
    # Fewer than one value in a hundred rises above zero, so the 99th
    # percentile is the minimum.
    sparse = np.zeros(loaded.data.shape)
    sparse[0, 0, 0] = 3.0
    monkeypatch.setattr(
        TrajectoryAccessor,
        "load_lineage_aggregation",
        lambda accessor, ref: dataclasses.replace(loaded, data=sparse),
    )

    result = closing(splt.lineage_heatmap(datastore, aggregation=unscaled, show=False))

    assert result.scales[0] == splt.ColorScale(cmap="coolwarm", vmin=0.0, vmax=3.0)


@pytest.mark.parametrize(
    ("vmin", "vmax"),
    [(2.0, 2.0), (1.0, -1.0), (np.nan, 2.0), (-2.0, np.inf), (-np.inf, 2.0)],
)
def test_lineage_heatmap_requires_finite_increasing_limits(
    datastore, lineage_aggregation, vmin, vmax
):
    with pytest.raises(ValueError, match="vmax must be greater than vmin, and both"):
        splt.lineage_heatmap(
            datastore,
            aggregation=lineage_aggregation,
            vmin=vmin,
            vmax=vmax,
            show=False,
        ).close()


def test_lineage_heatmap_rejects_a_loaded_result_of_another_artifact(
    monkeypatch: pytest.MonkeyPatch,
    datastore,
    lineage_aggregation,
    loaded_aggregation,
):
    requests: list[ArtifactRef] = []

    def load(accessor: Any, ref: ArtifactRef) -> Any:
        requests.append(ref)
        return dataclasses.replace(loaded_aggregation, ref=_other_ref(ref))

    monkeypatch.setattr(TrajectoryAccessor, "load_lineage_aggregation", load)

    with pytest.raises(ValueError, match="does not match the request"):
        splt.lineage_heatmap(
            datastore, aggregation=lineage_aggregation, show=False
        ).close()
    assert requests == [lineage_aggregation]


# Lineage trends.


def test_lineage_trends_draw_each_lineage_against_bin_pseudotime(
    closing, datastore, lineage_aggregation, loaded_aggregation, feature_names
):
    loaded = loaded_aggregation
    # Features match case-insensitively and keep the caller's spelling.
    features = [feature_names[0], feature_names[10].lower(), feature_names[20]]

    result = closing(
        splt.lineage_trends(
            datastore,
            aggregation=lineage_aggregation,
            features=features,
            axis="pseudotime",
            show=False,
        )
    )

    assert result.owns_figure is True
    assert list(result.axes) == features
    assert len(result.figure.axes) == len(features)
    assert tuple(result.figure.get_size_inches()) == pytest.approx(
        (DEFAULT_PANEL_INCHES * 3, DEFAULT_PANEL_INCHES * 0.8)
    )
    palette = colormap_palette(list(LINEAGES), "tab10")
    trends = result.tables["trends"]
    assert list(trends.columns) == [
        "feature",
        "lineage",
        "progress",
        "pseudotime",
        "value",
    ]
    assert result.provenance.extras["axis"] == "pseudotime"
    assert len(trends) == len(features) * len(LINEAGES) * N_BINS
    bin_pseudotime = np.asarray(loaded.bin_pseudotime)
    data = np.asarray(loaded.data)
    for feature in features:
        row = _feature_row(feature_names, feature)
        axis = result.axes[feature]
        assert axis.get_title() == feature
        assert axis.get_xlabel() == "Pseudotime"
        lines = axis.get_lines()
        assert [line.get_label() for line in lines] == list(LINEAGES)
        for column, (name, line) in enumerate(zip(LINEAGES, lines, strict=True)):
            part = trends[(trends["feature"] == feature) & (trends["lineage"] == name)]
            np.testing.assert_array_equal(part["pseudotime"], bin_pseudotime[column])
            np.testing.assert_array_equal(part["value"], data[row, column])
            np.testing.assert_array_equal(line.get_xdata(), bin_pseudotime[column])
            np.testing.assert_array_equal(line.get_ydata(), data[row, column])
            assert line.get_color() == palette[name]
            assert line.get_linewidth() == 2.0
    legend = result.axes[features[0]].get_legend()
    assert [text.get_text() for text in legend.get_texts()] == list(LINEAGES)
    assert all(result.axes[feature].get_legend() is None for feature in features[1:])
    assert result.legends == (splt.LegendSpec(kind="categorical", label="lineage"),)
    assert result.scales == (splt.CategoricalScale(order=LINEAGES, palette=palette),)
    assert result.provenance.assay == "RNA"
    assert result.provenance.notes == ("lineage_trends", "aggregated")
    assert result.provenance.extras == {
        "aggregation": lineage_aggregation.to_dict(),
        "lineages": list(LINEAGES),
        "features": features,
        "axis": "pseudotime",
    }


def test_lineage_trends_space_bins_by_lineage_progress_by_default(
    closing, datastore, lineage_aggregation, loaded_aggregation, feature_names
):
    feature = feature_names[0]
    result = closing(
        splt.lineage_trends(
            datastore, aggregation=lineage_aggregation, features=[feature], show=False
        )
    )

    progress = (np.arange(N_BINS) + 0.5) / N_BINS
    axis = result.axes[feature]
    assert axis.get_xlabel() == "Lineage progress"
    for line in axis.get_lines():
        np.testing.assert_allclose(line.get_xdata(), progress)
    trends = result.tables["trends"]
    for name in LINEAGES:
        np.testing.assert_allclose(
            trends.loc[trends["lineage"] == name, "progress"], progress
        )
    assert result.provenance.extras["axis"] == "progress"


def test_lineage_trends_reject_an_unknown_axis_and_repeated_features(
    datastore, lineage_aggregation, feature_names
):
    with pytest.raises(ValueError, match="axis must be 'progress' or 'pseudotime'"):
        splt.lineage_trends(
            datastore,
            aggregation=lineage_aggregation,
            features=[feature_names[0]],
            axis="time",
            show=False,
        )
    with pytest.raises(ValueError, match="features must not repeat a feature"):
        splt.lineage_trends(
            datastore,
            aggregation=lineage_aggregation,
            features=[feature_names[0], feature_names[0].upper()],
            show=False,
        )


def test_lineage_trends_draw_chosen_lineages_in_a_custom_style(
    closing, datastore, lineage_aggregation, loaded_aggregation, feature_names
):
    features = feature_names[3:5]

    result = closing(
        datastore.plots.lineage_trends(
            aggregation=lineage_aggregation,
            features=features,
            lineages=[4, 2],
            lineage_cmap="Set1",
            line_width=1.5,
            figsize=(5.0, 3.0),
            theme="paper",
            show_legend=False,
            show=False,
        )
    )

    palette = colormap_palette(["4", "2"], "Set1")
    assert tuple(result.figure.get_size_inches()) == (5.0, 3.0)
    for feature in features:
        axis = result.axes[feature]
        assert [line.get_label() for line in axis.get_lines()] == ["4", "2"]
        assert [line.get_color() for line in axis.get_lines()] == [
            palette["4"],
            palette["2"],
        ]
        assert {line.get_linewidth() for line in axis.get_lines()} == {1.5}
        assert axis.get_legend() is None
        # The paper theme sets smaller titles.
        assert axis.title.get_fontsize() == splt.THEMES["paper"]["axes.titlesize"]
    trends = result.tables["trends"]
    assert trends["lineage"].unique().tolist() == ["4", "2"]
    row = _feature_row(feature_names, features[0])
    first = trends[(trends["feature"] == features[0]) & (trends["lineage"] == "4")]
    np.testing.assert_array_equal(first["value"], loaded_aggregation.data[row, 2])
    np.testing.assert_array_equal(
        first["pseudotime"], loaded_aggregation.bin_pseudotime[2]
    )
    assert result.scales == (splt.CategoricalScale(order=("4", "2"), palette=palette),)
    assert result.provenance.extras["lineages"] == ["4", "2"]
    assert result.theme == "paper"


@pytest.mark.parametrize(
    ("n_features", "n_columns", "grid"),
    [(3, None, (1, 3)), (3, 2, (2, 2)), (2, 6, (1, 2)), (1, None, (1, 1))],
)
def test_lineage_trends_lay_panels_out_in_columns(
    closing, datastore, lineage_aggregation, feature_names, n_features, n_columns, grid
):
    features = feature_names[:n_features]
    n_rows, n_grid_columns = grid

    result = closing(
        splt.lineage_trends(
            datastore,
            aggregation=lineage_aggregation,
            features=features,
            n_columns=n_columns,
            show=False,
        )
    )

    assert tuple(result.figure.get_size_inches()) == pytest.approx(
        (
            DEFAULT_PANEL_INCHES * n_grid_columns,
            DEFAULT_PANEL_INCHES * 0.8 * n_rows,
        )
    )
    # Grid cells without a panel are removed from the figure.
    assert len(result.figure.axes) == n_features
    for position, feature in enumerate(features):
        spec = result.axes[feature].get_subplotspec()
        assert (spec.rowspan.start, spec.colspan.start) == divmod(
            position, n_grid_columns
        )


def test_lineage_trends_draw_into_caller_axes(
    datastore, lineage_aggregation, feature_names
):
    features = feature_names[:2]
    figure, axes = plt.subplots(1, 2)
    try:
        listed = splt.lineage_trends(
            datastore,
            aggregation=lineage_aggregation,
            features=features,
            target=list(axes),
            show=False,
        )
        assert listed.owns_figure is False
        assert listed.figure is figure
        assert [listed.axes[feature] for feature in features] == list(axes)
        listed.close()
        assert plt.fignum_exists(figure.number)

        mapped = datastore.plots.lineage_trends(
            aggregation=lineage_aggregation,
            features=features,
            target={features[0]: axes[1], features[1]: axes[0]},
            show=False,
        )
        assert mapped.axes[features[0]] is axes[1]
        assert axes[1].get_title() == features[0]
        # axes[1] keeps the lines of the first call and adds those of the second.
        assert len(axes[1].get_lines()) == 2 * len(LINEAGES)

        with pytest.raises(ValueError, match="figsize is invalid"):
            splt.lineage_trends(
                datastore,
                aggregation=lineage_aggregation,
                features=features,
                target=list(axes),
                figsize=(4.0, 4.0),
                show=False,
            ).close()
    finally:
        plt.close(figure)


def test_lineage_trends_repeated_feature_draws_one_panel_per_axes(
    closing, datastore, lineage_aggregation, feature_names
):
    feature = feature_names[0]
    try:
        result = closing(
            splt.lineage_trends(
                datastore,
                aggregation=lineage_aggregation,
                features=[feature, feature],
                show=False,
            )
        )
    except ValueError:
        # Rejecting the repeat, as lineages=[...] does, also fixes the figure.
        return
    assert len(result.figure.axes) == len(result.axes)
    assert all(len(axis.get_lines()) == len(LINEAGES) for axis in result.axes.values())


@pytest.mark.parametrize("aggregation", ["latest", None])
def test_lineage_trends_require_an_aggregation_ref(
    datastore, aggregation, feature_names
):
    with pytest.raises(TypeError, match="aggregation must be an ArtifactRef"):
        splt.lineage_trends(
            datastore,
            aggregation=aggregation,
            features=feature_names[:1],
            show=False,
        ).close()


@pytest.mark.parametrize(
    "features", ["CD3E", [], (), {"CD3E"}, None, np.array(["CD3E"])]
)
def test_lineage_trends_require_a_sequence_of_feature_names(
    datastore, lineage_aggregation, features
):
    with pytest.raises(TypeError, match="features must be a non-empty sequence"):
        splt.lineage_trends(
            datastore, aggregation=lineage_aggregation, features=features, show=False
        ).close()


def test_lineage_trends_require_retained_features(
    datastore, lineage_aggregation, feature_names
):
    # An assay feature outside the aggregation's feature selection.
    assay_names = np.asarray(datastore.RNA.feats.fetch_all("names")).astype(str)
    retained = {name.lower() for name in feature_names}
    unprofiled = next(
        name for name in assay_names.tolist() if name.lower() not in retained
    )

    for missing in ("NOT_A_FEATURE", unprofiled):
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Feature '{missing}' is not a retained feature of the aggregation"
            ),
        ):
            splt.lineage_trends(
                datastore,
                aggregation=lineage_aggregation,
                features=[feature_names[0], missing],
                show=False,
            ).close()


@pytest.mark.parametrize("n_columns", [0, -1, True, False, 1.5, "2"])
def test_lineage_trends_require_a_positive_integer_column_count(
    datastore, lineage_aggregation, feature_names, n_columns
):
    with pytest.raises(ValueError, match="n_columns must be a positive integer"):
        splt.lineage_trends(
            datastore,
            aggregation=lineage_aggregation,
            features=feature_names[:2],
            n_columns=n_columns,
            show=False,
        ).close()


@pytest.mark.parametrize(
    ("line_width", "error", "message"),
    [
        (0, ValueError, "line_width must be positive and finite"),
        (-1.0, ValueError, "line_width must be positive and finite"),
        (np.nan, ValueError, "line_width must be positive and finite"),
        (np.inf, ValueError, "line_width must be positive and finite"),
        (True, TypeError, "line_width must be a real number"),
        ("2", TypeError, "line_width must be a real number"),
        (None, TypeError, "line_width must be a real number"),
    ],
)
def test_lineage_trends_require_a_positive_line_width(
    datastore, lineage_aggregation, feature_names, line_width, error, message
):
    with pytest.raises(error, match=message):
        splt.lineage_trends(
            datastore,
            aggregation=lineage_aggregation,
            features=feature_names[:1],
            line_width=line_width,
            show=False,
        ).close()


# Lineage curves.


def test_lineage_curves_draw_fate_weighted_curves_in_the_layout(
    closing,
    datastore,
    lineage_curves,
    lineage_fate,
    loaded_curves,
    loaded_fate,
    layout,
):
    result = closing(
        splt.lineage_curves(
            datastore, curves=lineage_curves, layout=layout.ref, show=False
        )
    )

    assert result.owns_figure is True
    assert list(result.axes) == ["curves"]
    # The default figure leaves room for the legend beside the panel.
    assert tuple(result.figure.get_size_inches()) == pytest.approx(
        (DEFAULT_PANEL_INCHES + 1.4, DEFAULT_PANEL_INCHES)
    )
    table = result.tables["curves"]
    assert list(table.columns) == ["lineage", "point", "x", "y"]
    assert len(table) == len(LINEAGES) * N_POINTS
    assert table["lineage"].unique().tolist() == list(LINEAGES)
    palette = colormap_palette(list(LINEAGES), "tab10")
    axis = result.axes["curves"]
    lines = axis.get_lines()
    assert len(lines) == 2 * len(LINEAGES)
    for position, name in enumerate(LINEAGES):
        rows = table[table["lineage"] == name]
        assert rows["point"].tolist() == list(range(N_POINTS))
        # The curve's own degrees of freedom smooth it in the layout.
        expected = _expected_curve(
            loaded_curves, loaded_fate, layout.coordinates, int(name), 10.0
        )
        np.testing.assert_allclose(rows[["x", "y"]].to_numpy(), expected)
        curve, end = lines[2 * position], lines[2 * position + 1]
        np.testing.assert_allclose(curve.get_xydata(), expected)
        assert curve.get_label() == name
        assert curve.get_color() == end.get_color() == palette[name]
        assert curve.get_linewidth() == 2.5
        np.testing.assert_allclose(end.get_xydata(), expected[[-1]])
        assert end.get_marker() == "o"
        assert end.get_markersize() == 2.5 * 2.5
    cells = axis.collections[0]
    np.testing.assert_array_equal(cells.get_offsets(), layout.coordinates)
    assert cells.get_sizes().tolist() == [default_point_size(len(layout.cell_indices))]
    assert cells.get_alpha() == 0.6
    assert to_hex(cells.get_facecolor()[0]) == "#d0d0d0"
    xlim, ylim = padded_square_limits(
        layout.coordinates[:, 0], layout.coordinates[:, 1]
    )
    assert axis.get_xlim() == pytest.approx(xlim)
    assert axis.get_ylim() == pytest.approx(ylim)
    assert axis.get_title() == "Lineage curves"
    legend = axis.get_legend()
    assert [text.get_text() for text in legend.get_texts()] == list(LINEAGES)
    assert result.legends == (splt.LegendSpec(kind="categorical", label="lineage"),)
    assert result.scales == (splt.CategoricalScale(order=LINEAGES, palette=palette),)
    assert result.provenance.n_cells == len(layout.cell_indices)
    assert result.provenance.notes == ("lineage_curves", "artifact")
    assert result.provenance.extras == {
        "curves": lineage_curves.to_dict(),
        "layout": layout.ref.to_dict(),
        "fate": lineage_fate.to_dict(),
        "cell_selection": layout.selection.to_dict(),
        "lineages": list(LINEAGES),
        "df": 10.0,
    }


def test_lineage_curves_draw_chosen_lineages_with_their_own_smoothing(
    closing, datastore, lineage_curves, loaded_curves, loaded_fate, layout
):
    result = closing(
        datastore.plots.lineage_curves(
            curves=lineage_curves,
            layout=layout.ref,
            lineages=[4, 3],
            df=4.0,
            lineage_cmap="Dark2",
            line_width=1.0,
            point_size=3.0,
            point_alpha=0.25,
            cell_color="#123456",
            figsize=(4.0, 4.0),
            theme="dark",
            show_legend=False,
            show=False,
        )
    )

    table = result.tables["curves"]
    assert table["lineage"].unique().tolist() == ["4", "3"]
    palette = colormap_palette(["4", "3"], "Dark2")
    for name in ("4", "3"):
        expected = _expected_curve(
            loaded_curves, loaded_fate, layout.coordinates, int(name), 4.0
        )
        default = _expected_curve(
            loaded_curves, loaded_fate, layout.coordinates, int(name), 10.0
        )
        drawn = table[table["lineage"] == name][["x", "y"]].to_numpy()
        np.testing.assert_allclose(drawn, expected)
        assert not np.allclose(drawn, default)
    axis = result.axes["curves"]
    curves = axis.get_lines()[::2]
    assert [line.get_color() for line in curves] == [palette["4"], palette["3"]]
    assert {line.get_linewidth() for line in curves} == {1.0}
    assert {line.get_markersize() for line in axis.get_lines()[1::2]} == {2.5}
    cells = axis.collections[0]
    assert cells.get_sizes().tolist() == [3.0]
    assert cells.get_alpha() == 0.25
    assert to_hex(cells.get_facecolor()[0]) == "#123456"
    assert axis.get_legend() is None
    assert tuple(result.figure.get_size_inches()) == (4.0, 4.0)
    assert result.provenance.extras["lineages"] == ["4", "3"]
    assert result.provenance.extras["df"] == 4.0
    assert result.theme == "dark"
    assert result.figure.patch.get_alpha() == 0


def test_lineage_curves_draw_into_a_caller_axes(datastore, lineage_curves, layout):
    figure, axis = plt.subplots()
    try:
        result = splt.lineage_curves(
            datastore,
            curves=lineage_curves,
            layout=layout.ref,
            lineages=[2],
            target=axis,
            show=False,
        )
        assert result.owns_figure is False
        assert result.figure is figure
        assert result.axes == {"curves": axis}
        assert len(axis.get_lines()) == 2
        result.close()
        assert plt.fignum_exists(figure.number)
    finally:
        plt.close(figure)


def test_lineage_curves_reject_df_the_spline_cannot_spend(
    datastore, lineage_curves, umap
):
    with pytest.raises(ValueError, match="df"):
        splt.lineage_curves(
            datastore, curves=lineage_curves, layout=umap, df=1.0, show=False
        ).close()


@pytest.mark.parametrize("curves", ["latest", None])
def test_lineage_curves_require_a_curves_ref(datastore, umap, curves):
    with pytest.raises(TypeError, match="curves must be an ArtifactRef"):
        splt.lineage_curves(datastore, curves=curves, layout=umap, show=False).close()


def test_lineage_curves_reject_refs_of_another_kind(
    datastore, lineage_curves, lineage_tree, umap
):
    with pytest.raises(ValueError, match="ref must be a lineage_curves artifact"):
        splt.lineage_curves(
            datastore, curves=lineage_tree, layout=umap, show=False
        ).close()
    with pytest.raises(TypeError, match="layout must be an ArtifactRef"):
        splt.lineage_curves(
            datastore, curves=lineage_curves, layout="umap", show=False
        ).close()
    with pytest.raises(ValueError, match="layout must identify an embedding artifact"):
        splt.lineage_curves(
            datastore, curves=lineage_curves, layout=lineage_curves, show=False
        ).close()


@pytest.mark.parametrize(
    ("argument", "value", "error", "message"),
    [
        ("line_width", 0.0, ValueError, "line_width must be positive and finite"),
        ("line_width", True, TypeError, "line_width must be a real number"),
        ("point_alpha", -0.1, ValueError, "point_alpha must be between zero and one"),
        ("point_alpha", 1.5, ValueError, "point_alpha must be between zero and one"),
        ("point_alpha", np.nan, ValueError, "point_alpha must be between zero and"),
        ("point_alpha", np.inf, ValueError, "point_alpha must be between zero and"),
        ("df", 0.0, ValueError, "df must be greater than 2 and less than 32"),
        ("df", -2.0, ValueError, "df must be greater than 2 and less than 32"),
        ("df", 2.0, ValueError, "df must be greater than 2 and less than 32"),
        ("df", 32, ValueError, "df must be greater than 2 and less than 32"),
        ("df", np.nan, ValueError, "df must be greater than 2 and less than 32"),
        ("df", np.inf, ValueError, "df must be greater than 2 and less than 32"),
        ("df", True, TypeError, "df must be a real number"),
        ("df", "4", TypeError, "df must be a real number"),
        ("point_size", 0, ValueError, "point_size must be positive and finite"),
        ("point_size", -1.0, ValueError, "point_size must be positive and finite"),
        ("point_size", np.nan, ValueError, "point_size must be positive and finite"),
        ("point_size", True, TypeError, "point_size must be a real number"),
        ("point_size", "3", TypeError, "point_size must be a real number"),
    ],
)
def test_lineage_curves_validate_drawing_arguments(
    datastore, lineage_curves, umap, argument, value, error, message
):
    with pytest.raises(error, match=message):
        splt.lineage_curves(
            datastore,
            curves=lineage_curves,
            layout=umap,
            show=False,
            **{argument: value},
        ).close()


# Lineage tree.


def test_lineage_tree_places_clusters_at_their_layout_medians(
    closing, datastore, lineage_tree, loaded_tree, layout
):
    result = closing(
        splt.lineage_tree(datastore, tree=lineage_tree, layout=layout.ref, show=False)
    )

    clusters = list(loaded_tree.clusters)
    assert loaded_tree.root == TREE_ROOT
    leaves = _tree_leaves(loaded_tree)
    candidates = leaves - {TREE_ROOT}
    assert result.owns_figure is True
    assert list(result.axes) == ["tree"]
    nodes = result.tables["nodes"]
    assert list(nodes.columns) == ["cluster", "x", "y", "size", "candidate_sink"]
    assert nodes["cluster"].tolist() == clusters
    medians = np.asarray(
        [np.median(layout.coordinates[layout.labels == c], axis=0) for c in clusters]
    )
    np.testing.assert_allclose(nodes[["x", "y"]].to_numpy(), medians)
    assert nodes["size"].tolist() == [int((layout.labels == c).sum()) for c in clusters]
    assert nodes["candidate_sink"].tolist() == [c in candidates for c in clusters]
    pd.testing.assert_frame_equal(result.tables["edges"], loaded_tree.edges)
    assert result.tables["edges"] is not loaded_tree.edges

    axis = result.axes["tree"]
    position = {
        cluster: (x, y)
        for cluster, x, y in zip(nodes["cluster"], nodes["x"], nodes["y"], strict=True)
    }
    edge_lines = axis.get_lines()
    assert len(edge_lines) == len(clusters) - 1
    for line, source, target in zip(
        edge_lines,
        loaded_tree.edges["source"],
        loaded_tree.edges["target"],
        strict=True,
    ):
        np.testing.assert_allclose(
            line.get_xydata(), [position[source], position[target]]
        )
        assert line.get_color() == "#555555"
        assert line.get_linewidth() == 1.5
    cells, *markers = axis.collections
    np.testing.assert_array_equal(cells.get_offsets(), layout.coordinates)
    assert cells.get_sizes().tolist() == [default_point_size(len(layout.cell_indices))]
    assert cells.get_alpha() == 0.5
    assert to_hex(cells.get_facecolor()[0]) == "#d0d0d0"
    assert len(markers) == len(clusters)
    for cluster, marker in zip(clusters, markers, strict=True):
        np.testing.assert_allclose(marker.get_offsets(), [position[cluster]])
        assert marker.get_sizes().tolist() == [80.0]
        expected = (
            "#59a14f"
            if cluster == TREE_ROOT
            else "#e15759"
            if cluster in candidates
            else "#4e79a7"
        )
        assert to_hex(marker.get_facecolor()[0]) == expected
    assert [text.get_text() for text in axis.texts] == [str(c) for c in clusters]
    for text, cluster in zip(axis.texts, clusters, strict=True):
        assert text.xy == pytest.approx(position[cluster])
    xlim, ylim = padded_square_limits(
        layout.coordinates[:, 0], layout.coordinates[:, 1]
    )
    assert axis.get_xlim() == pytest.approx(xlim)
    assert axis.get_ylim() == pytest.approx(ylim)
    assert axis.get_title() == "Cluster tree"
    inner = set(clusters) - leaves - {TREE_ROOT}
    assert candidates and inner
    roles = ("root", "candidate sink", "cluster")
    palette = {"root": "#59a14f", "candidate sink": "#e15759", "cluster": "#4e79a7"}
    assert result.legends == (
        splt.LegendSpec(kind="categorical", label="cluster role"),
    )
    assert result.scales == (splt.CategoricalScale(order=roles, palette=palette),)
    legend = axis.get_legend()
    assert [text.get_text() for text in legend.get_texts()] == list(roles)
    assert [
        to_hex(handle.get_markerfacecolor()) for handle in legend.legend_handles
    ] == [palette[role] for role in roles]
    assert tuple(result.figure.get_size_inches()) == pytest.approx(
        (DEFAULT_PANEL_INCHES + 1.6, DEFAULT_PANEL_INCHES)
    )
    assert result.provenance.n_cells == len(layout.cell_indices)
    assert result.provenance.notes == ("lineage_tree", "artifact")
    assert set(loaded_tree.candidate_sinks) == candidates
    assert result.provenance.extras == {
        "tree": lineage_tree.to_dict(),
        "layout": layout.ref.to_dict(),
        "cell_selection": layout.selection.to_dict(),
        "root": TREE_ROOT,
        "candidate_sinks": list(loaded_tree.candidate_sinks),
    }


def test_lineage_tree_without_cells_or_labels_in_custom_colors(
    closing, datastore, lineage_tree, loaded_tree, layout
):
    result = closing(
        datastore.plots.lineage_tree(
            tree=lineage_tree,
            layout=layout.ref,
            show_cells=False,
            node_size=20.0,
            node_color="#000000",
            root_color="#111111",
            leaf_color="#222222",
            edge_color="#333333",
            labels=False,
            figsize=(4.0, 4.0),
            theme="paper",
            show=False,
        )
    )

    axis = result.axes["tree"]
    candidates = set(loaded_tree.candidate_sinks)
    # Only the cluster markers remain, one per cluster.
    assert len(axis.collections) == len(loaded_tree.clusters)
    for cluster, marker in zip(loaded_tree.clusters, axis.collections, strict=True):
        assert marker.get_sizes().tolist() == [20.0]
        expected = (
            "#111111"
            if cluster == TREE_ROOT
            else "#222222"
            if cluster in candidates
            else "#000000"
        )
        assert to_hex(marker.get_facecolor()[0]) == expected
    assert {line.get_color() for line in axis.get_lines()} == {"#333333"}
    assert len(axis.texts) == 0
    assert tuple(result.figure.get_size_inches()) == (4.0, 4.0)
    assert result.theme == "paper"
    assert axis.title.get_fontsize() == splt.THEMES["paper"]["axes.titlesize"]
    legend = axis.get_legend()
    assert [
        to_hex(handle.get_markerfacecolor()) for handle in legend.legend_handles
    ] == ["#111111", "#222222", "#000000"]
    assert result.scales[0].palette == {
        "root": "#111111",
        "candidate sink": "#222222",
        "cluster": "#000000",
    }


def test_lineage_tree_without_a_legend_keeps_the_default_panel(
    closing, datastore, lineage_tree, layout
):
    result = closing(
        splt.lineage_tree(
            datastore,
            tree=lineage_tree,
            layout=layout.ref,
            show_legend=False,
            show=False,
        )
    )

    assert result.axes["tree"].get_legend() is None
    assert tuple(result.figure.get_size_inches()) == pytest.approx(
        (DEFAULT_PANEL_INCHES + 0.4, DEFAULT_PANEL_INCHES)
    )
    assert result.legends == (
        splt.LegendSpec(kind="categorical", label="cluster role"),
    )


def test_lineage_tree_draws_cells_into_a_caller_axes(datastore, lineage_tree, layout):
    figure, axis = plt.subplots()
    try:
        result = splt.lineage_tree(
            datastore,
            tree=lineage_tree,
            layout=layout.ref,
            point_size=5.0,
            point_alpha=0.2,
            cell_color="#abcdef",
            target=axis,
            show=False,
        )
        assert result.owns_figure is False
        assert result.figure is figure
        assert result.axes == {"tree": axis}
        cells = axis.collections[0]
        assert cells.get_sizes().tolist() == [5.0]
        assert cells.get_alpha() == 0.2
        assert to_hex(cells.get_facecolor()[0]) == "#abcdef"
        result.close()
        assert plt.fignum_exists(figure.number)
    finally:
        plt.close(figure)


def test_lineage_tree_without_a_root_marks_every_leaf(
    closing,
    monkeypatch: pytest.MonkeyPatch,
    datastore,
    lineage_tree,
    loaded_tree,
    layout,
):
    unrooted = dataclasses.replace(loaded_tree, root=None, lineages=())
    monkeypatch.setattr(
        TrajectoryAccessor, "load_lineage_tree", lambda accessor, ref: unrooted
    )

    result = closing(
        splt.lineage_tree(datastore, tree=lineage_tree, layout=layout.ref, show=False)
    )

    leaves = _tree_leaves(loaded_tree)
    assert TREE_ROOT not in leaves
    nodes = result.tables["nodes"]
    assert nodes["candidate_sink"].tolist() == [
        cluster in leaves for cluster in loaded_tree.clusters
    ]
    markers = result.axes["tree"].collections[1:]
    for cluster, marker in zip(loaded_tree.clusters, markers, strict=True):
        expected = "#e15759" if cluster in leaves else "#4e79a7"
        assert to_hex(marker.get_facecolor()[0]) == expected
    # Without a root the legend names only the roles drawn.
    legend = result.axes["tree"].get_legend()
    assert [text.get_text() for text in legend.get_texts()] == [
        "candidate sink",
        "cluster",
    ]
    assert result.scales[0].order == ("candidate sink", "cluster")
    assert result.provenance.extras["root"] is None
    assert result.provenance.extras["candidate_sinks"] == list(loaded_tree.leaves)
    assert set(loaded_tree.leaves) == leaves


@pytest.mark.parametrize("tree", ["latest", None])
def test_lineage_tree_requires_a_tree_ref(datastore, umap, tree):
    with pytest.raises(TypeError, match="tree must be an ArtifactRef"):
        splt.lineage_tree(datastore, tree=tree, layout=umap, show=False).close()


def test_lineage_tree_rejects_refs_of_another_kind(
    datastore, lineage_tree, lineage_curves, umap
):
    with pytest.raises(ValueError, match="ref must be a lineage_tree artifact"):
        splt.lineage_tree(
            datastore, tree=lineage_curves, layout=umap, show=False
        ).close()
    with pytest.raises(TypeError, match="layout must be an ArtifactRef"):
        splt.lineage_tree(datastore, tree=lineage_tree, layout=None, show=False).close()


@pytest.mark.parametrize(
    ("argument", "value", "error", "message"),
    [
        ("node_size", 0, ValueError, "node_size must be positive and finite"),
        ("node_size", -80.0, ValueError, "node_size must be positive and finite"),
        ("node_size", np.nan, ValueError, "node_size must be positive and finite"),
        ("node_size", np.inf, ValueError, "node_size must be positive and finite"),
        ("node_size", True, TypeError, "node_size must be a real number"),
        ("node_size", "80", TypeError, "node_size must be a real number"),
        ("point_alpha", -0.5, ValueError, "point_alpha must be between zero and one"),
        ("point_alpha", 2.0, ValueError, "point_alpha must be between zero and one"),
        ("point_alpha", np.nan, ValueError, "point_alpha must be between zero and"),
        ("point_size", 0.0, ValueError, "point_size must be positive and finite"),
        ("point_size", -1.0, ValueError, "point_size must be positive and finite"),
        ("point_size", False, TypeError, "point_size must be a real number"),
    ],
)
def test_lineage_tree_validates_drawing_arguments(
    datastore, lineage_tree, umap, argument, value, error, message
):
    with pytest.raises(error, match=message):
        splt.lineage_tree(
            datastore, tree=lineage_tree, layout=umap, show=False, **{argument: value}
        ).close()


def test_lineage_tree_validates_point_size_without_cells(datastore, lineage_tree, umap):
    with pytest.raises(ValueError, match="point_size must be positive and finite"):
        splt.lineage_tree(
            datastore,
            tree=lineage_tree,
            layout=umap,
            show_cells=False,
            point_size=-1.0,
            show=False,
        ).close()


# Arguments shared by the plots.


@pytest.mark.parametrize("name", PLOTS)
def test_lineage_plots_reject_an_unknown_theme(datastore, plot_arguments, name):
    with pytest.raises(KeyError, match="Unknown theme 'sepia'"):
        getattr(splt, name)(
            datastore, theme="sepia", show=False, **plot_arguments[name]
        ).close()


@pytest.mark.parametrize(
    "name", ["lineage_heatmap", "lineage_trends", "lineage_curves"]
)
@pytest.mark.parametrize(
    ("lineages", "error", "message"),
    [
        ([7], ValueError, "Lineage 7 is not in the result"),
        ([3, 3], ValueError, "lineages must not repeat a lineage"),
        ([], ValueError, "lineages must name at least one lineage"),
        ("2", TypeError, "lineages must be a list of lineage labels or None"),
    ],
)
def test_lineage_plots_validate_the_chosen_lineages(
    datastore, plot_arguments, name, lineages, error, message
):
    with pytest.raises(error, match=message):
        getattr(splt, name)(
            datastore, lineages=lineages, show=False, **plot_arguments[name]
        ).close()


@pytest.mark.parametrize("name", ["lineage_curves", "lineage_tree"])
def test_layout_plots_require_the_trajectory_cell_selection(
    monkeypatch: pytest.MonkeyPatch, datastore, plot_arguments, layout, name
):
    """A layout of other cells is refused although its coordinates fit."""
    requested: list[ArtifactRef] = []

    def resolve(store: Any, ref: ArtifactRef) -> tuple[Any, Any, ArtifactRef]:
        requested.append(ref)
        return layout.coordinates, layout.cell_indices, _other_ref(layout.selection)

    monkeypatch.setattr(lineage_plots, "_resolve_layout", resolve)

    with pytest.raises(ValueError, match="exact cell-selection artifact"):
        getattr(splt, name)(datastore, show=False, **plot_arguments[name]).close()
    assert requested == [layout.ref]
