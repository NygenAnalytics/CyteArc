import inspect
import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import cytearc.plotting as splt
import cytearc.plotting.diagnostics as diagnostics
from cytearc.plotting._figure import PlotResult
from cytearc.plotting.diagnostics import hvg_plot_options


# A steep drop followed by a slow tail has its only corner at the third PC, 1.0.
_CORNERED_VARIANCE = np.array([9.0, 5.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5])


def _qc_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "groups": ["a"] * 4 + ["b"] * 4,
            "nCounts": [10.0, 12.0, 11.0, 13.0, 20.0, 22.0, 21.0, 23.0],
            "nFeatures": [4.0, 5.0, 5.0, 6.0, 8.0, 9.0, 9.0, 10.0],
        }
    )


def _graph():
    return sparse.csr_matrix(
        np.array(
            [
                [0.0, 0.5, 0.0],
                [0.5, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
    )


def _hvg_arguments() -> tuple[np.ndarray, ...]:
    return (
        np.array([1.0, 2.0, 4.0, 8.0]),
        np.array([2.0, 3.0, 8.0, 12.0]),
        np.array([10, 20, 30, 40]),
        np.array([False, True, False, True]),
    )


def test_diagnostics_show_default_and_suppression(monkeypatch):
    shown = []

    def track_show(result):
        shown.append(result)

    monkeypatch.setattr(PlotResult, "show", track_show)
    defaults = [
        splt.qc(_qc_data(), max_points=0),
        splt.elbow([0.5, 0.3, 0.2]),
        splt.graph_qc(_graph()),
        splt.highly_variable_features(*_hvg_arguments()),
    ]
    suppressed = [
        splt.qc(_qc_data(), max_points=0, show=False),
        splt.elbow([0.5, 0.3, 0.2], show=False),
        splt.graph_qc(_graph(), show=False),
        splt.highly_variable_features(*_hvg_arguments(), show=False),
    ]

    assert shown == defaults
    for result in defaults + suppressed:
        result.close()


def test_qc_result_contains_tables_metadata_and_artists():
    from matplotlib.colors import to_rgba

    data = _qc_data()
    result = splt.qc(data, max_points=5, seed=13, show=False)

    assert isinstance(result, PlotResult)
    assert result.owns_figure is True
    assert list(result.axes) == ["nCounts", "nFeatures"]
    pd.testing.assert_frame_equal(result.tables["data"], data)
    summary = result.tables["summary"]
    assert list(zip(summary["group"], summary["metric"], strict=True)) == [
        ("a", "nCounts"),
        ("b", "nCounts"),
        ("a", "nFeatures"),
        ("b", "nFeatures"),
    ]
    assert summary["count"].tolist() == [4, 4, 4, 4]
    # Medians of [10, 12, 11, 13], [20, 22, 21, 23], [4, 5, 5, 6], [8, 9, 9, 10].
    np.testing.assert_allclose(summary["median"], [11.5, 21.5, 5.0, 9.0])
    assert result.provenance.n_cells == len(data)
    assert result.provenance.notes == ("qc",)
    assert result.provenance.extras["displayed_points"] == {
        "nCounts": 5,
        "nFeatures": 5,
    }
    assert result.provenance.extras["seed"] == 13
    assert result.legends[0].kind == "categorical"
    assert result.legends[0].extras == {"categories": ["a", "b"]}
    # Two groups take the first two tab20 colors.
    assert result.scales[0].palette == {"a": "#1f77b4", "b": "#aec7e8"}

    sampled_rows: dict[str, list[int]] = {}
    for metric, ax in result.axes.items():
        assert ax.get_ylabel() == metric
        assert [label.get_text() for label in ax.get_xticklabels()] == ["a", "b"]
        violins = ax.collections[:2]
        for violin, color in zip(violins, ("#1f77b4", "#aec7e8"), strict=True):
            np.testing.assert_allclose(
                violin.get_facecolor()[0],
                to_rgba(color, alpha=0.6),
            )
        strips = ax.collections[2:]
        assert len(strips) == 2
        rows: list[int] = []
        for position, (strip, group) in enumerate(zip(strips, "ab", strict=True)):
            offsets = np.asarray(strip.get_offsets())
            np.testing.assert_allclose(strip.get_facecolor()[0], (0, 0, 0, 0.4))
            assert np.all(np.abs(offsets[:, 0] - position) <= 0.4)
            group_values = data.loc[data["groups"] == group, metric]
            for value in offsets[:, 1]:
                matches = group_values.index[group_values == value]
                assert len(matches) >= 1
                rows.append(int(matches[0]))
        assert len(rows) == 5
        sampled_rows[metric] = rows
    # One seed samples the same cells for every metric.
    np.testing.assert_array_equal(
        data.loc[sorted(sampled_rows["nCounts"]), "nFeatures"].sort_values(),
        np.sort(
            np.concatenate(
                [
                    np.asarray(strip.get_offsets())[:, 1]
                    for strip in result.axes["nFeatures"].collections[2:]
                ]
            )
        ),
    )
    result.close()


def test_elbow_result_records_detection_and_line_artists():
    from matplotlib.colors import to_rgba

    values = _CORNERED_VARIANCE
    result = splt.elbow(values, show=False)

    # PCs are numbered from 1, as dims counts them. The fields are named for PC
    # numbers, so code that read the 0-based "component" and "elbow" fails.
    table = result.tables["variance_explained"]
    assert table.columns.tolist() == ["pc", "variance_explained", "is_elbow"]
    np.testing.assert_array_equal(table["pc"], [1, 2, 3, 4, 5, 6, 7, 8])
    np.testing.assert_array_equal(table["variance_explained"], values)
    assert table.loc[table["is_elbow"], "pc"].tolist() == [3]
    assert table.loc[table["is_elbow"], "variance_explained"].tolist() == [1.0]
    assert result.provenance.extras == {
        "elbow_pc": 3,
        "n_components": 8,
        "sensitivity": 1.0,
    }
    assert result.legends[0].label == "Elbow"
    assert result.legends[0].extras == {"pc": 3}
    # The default width spends a quarter inch on each component.
    assert result.figure.get_size_inches() == pytest.approx((2.0, 2.0))

    ax = result.axes["elbow"]
    curve, marker = ax.lines
    np.testing.assert_array_equal(curve.get_xdata(), [1, 2, 3, 4, 5, 6, 7, 8])
    np.testing.assert_array_equal(curve.get_ydata(), values)
    np.testing.assert_array_equal(marker.get_xdata(), [3, 3])
    assert to_rgba(marker.get_color()) == to_rgba("red")
    np.testing.assert_array_equal(ax.get_xticks(), [1, 2, 3, 4, 5, 6, 7, 8])
    result.figure.canvas.draw()
    assert [label.get_text() for label in ax.get_xticklabels()] == [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
    ]
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["Elbow"]
    assert ax.get_xlabel() == "Principal components"
    assert ax.get_ylabel() == "% Variance explained"
    result.close()


def _variance_with_corner_at(last_steep_pc: int, n_pcs: int) -> np.ndarray:
    # A steep straight drop from 50% to 5% over the first last_steep_pc PCs,
    # then a tail that loses 0.1% per PC, so the curve bends at last_steep_pc.
    steep = np.linspace(50.0, 5.0, last_steep_pc)
    tail = 5.0 - 0.1 * np.arange(1, n_pcs - last_steep_pc + 1)
    return np.concatenate([steep, tail])


@pytest.mark.parametrize("last_steep_pc", [2, 3, 7, 10])
@pytest.mark.parametrize("n_pcs", [12, 30])
def test_elbow_marks_the_number_of_the_pc_where_the_curve_bends(last_steep_pc, n_pcs):
    values = _variance_with_corner_at(last_steep_pc, n_pcs)
    result = splt.elbow(values, show=False)

    # The marked PC is the last PC of the drop, read straight as dims.
    assert result.provenance.extras["elbow_pc"] == last_steep_pc
    assert result.legends[0].extras == {"pc": last_steep_pc}
    ax = result.axes["elbow"]
    np.testing.assert_array_equal(ax.lines[1].get_xdata(), [last_steep_pc] * 2)
    table = result.tables["variance_explained"]
    (elbow_row,) = table.index[table["is_elbow"]]
    assert table.loc[elbow_row, "pc"] == last_steep_pc
    assert table.loc[elbow_row, "variance_explained"] == 5.0
    np.testing.assert_array_equal(table["pc"], np.arange(n_pcs) + 1)
    # The first PC sits at tick 1 and the last at tick n_pcs.
    assert ax.lines[0].get_xdata()[0] == 1
    assert ax.get_xticks()[[0, -1]].tolist() == [1, n_pcs]
    # The default width spends a quarter inch on each PC.
    width = {12: 3.0, 30: 7.5}[n_pcs]
    assert result.figure.get_size_inches() == pytest.approx((width, 2.0))
    result.close()


def test_elbow_of_the_issue_curve_names_the_second_pc():
    # Percent of variance for five PCs; the curve bends at the second PC, 12%.
    result = splt.elbow(np.array([40.0, 12.0, 6.0, 5.0, 4.5]), show=False)

    ax = result.axes["elbow"]
    result.figure.canvas.draw()
    assert [label.get_text() for label in ax.get_xticklabels()] == [
        "1",
        "2",
        "3",
        "4",
        "5",
    ]
    assert ax.lines[0].get_xdata()[0] == 1
    np.testing.assert_array_equal(ax.lines[1].get_xdata(), [2, 2])
    assert result.provenance.extras["elbow_pc"] == 2
    table = result.tables["variance_explained"]
    assert table.loc[
        table["is_elbow"], ["pc", "variance_explained"]
    ].values.tolist() == [[2, 12.0]]
    result.close()


def test_graph_qc_result_contains_graph_tables_and_histograms():
    result = splt.graph_qc(_graph(), show=False)

    assert result.provenance.n_cells == 3
    # The 99.5th percentile of degrees [1, 2, 1] is 1.99; the limit adds five.
    assert result.provenance.extras == {
        "graph_shape": [3, 3],
        "n_edges": 4,
        "degree_clip_limit": pytest.approx(6.99),
    }
    assert result.tables["node_degrees"]["node"].tolist() == [0, 1, 2]
    assert result.tables["node_degrees"]["degree"].tolist() == [1, 2, 1]
    assert result.tables["degree_frequencies"].to_dict("list") == {
        "degree": [1, 2],
        "frequency": [2, 1],
    }
    assert sorted(result.tables["edge_weights"]["edge_weight"]) == [
        0.5,
        0.5,
        1.0,
        1.0,
    ]

    degree_ax = result.axes["node_degree"]
    assert [
        (patch.get_x() + patch.get_width() / 2, patch.get_height())
        for patch in degree_ax.patches
    ] == [(1.0, 2), (2.0, 1)]
    assert degree_ax.get_xlim() == pytest.approx((0.0, 6.99))
    assert len(degree_ax.texts) == 0
    weight_patches = result.axes["edge_weight"].patches
    heights = [patch.get_height() for patch in weight_patches]
    assert len(heights) == 30
    # Thirty bins span the observed weights [0.5, 1.0]: both ends hold two edges.
    assert heights[0] == heights[-1] == 2
    assert sum(heights) == 4
    assert weight_patches[0].get_x() == pytest.approx(0.5)
    assert weight_patches[-1].get_x() + weight_patches[-1].get_width() == (
        pytest.approx(1.0)
    )
    result.close()


def test_highly_variable_features_result_contains_feature_table_and_scatters():
    args = _hvg_arguments()
    result = splt.highly_variable_features(*args, show=False)

    table = result.tables["features"]
    assert table["selected"].tolist() == [False, True, False, True]
    np.testing.assert_allclose(table["log2_mean_nonzero"], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        table["log2_corrected_variance"],
        [1.0, np.log2(3.0), 3.0, np.log2(12.0)],
    )
    assert result.provenance.extras["n_features"] == 4
    assert result.provenance.extras["n_selected"] == 2
    assert result.provenance.extras["max_expressing_cells"] == 40.0
    assert result.legends[0].kind == "categorical"

    ax = result.axes["highly_variable_features"]
    unselected, selected = ax.collections
    # Unselected features 0 and 2 then selected features 1 and 3, colored by
    # their expressing-cell counts.
    np.testing.assert_allclose(unselected.get_offsets(), [[0.0, 1.0], [2.0, 3.0]])
    np.testing.assert_allclose(
        selected.get_offsets(),
        [[1.0, np.log2(3.0)], [3.0, np.log2(12.0)]],
    )
    np.testing.assert_array_equal(unselected.get_array(), [10.0, 30.0])
    np.testing.assert_array_equal(selected.get_array(), [20.0, 40.0])
    assert unselected.get_sizes().tolist() == [3]
    assert selected.get_sizes().tolist() == [30]
    assert unselected.get_cmap().name == "winter"
    assert selected.get_cmap().name == "magma_r"
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "Not selected",
        "Selected",
    ]
    result.close()


@pytest.mark.parametrize(
    ("data", "kwargs", "error", "message"),
    [
        pytest.param(
            pd.DataFrame({"value": [1.0]}),
            {},
            KeyError,
            "groups",
            id="missing-groups",
        ),
        pytest.param(
            pd.DataFrame(columns=["groups", "value"]),
            {},
            ValueError,
            "at least one row",
            id="empty",
        ),
        pytest.param(
            _qc_data(),
            {"max_points": -1},
            ValueError,
            "max_points",
            id="negative-max-points",
        ),
        pytest.param(
            pd.DataFrame({"groups": ["a"]}),
            {},
            ValueError,
            "metric column",
            id="missing-metric",
        ),
        pytest.param(
            pd.DataFrame(
                [["a", 1.0, 2.0]],
                columns=["groups", "metric", "metric"],
            ),
            {},
            ValueError,
            "unique",
            id="duplicate-metric",
        ),
        pytest.param(
            pd.DataFrame({"groups": ["a"], "metric": ["not-numeric"]}),
            {},
            TypeError,
            "must be numeric",
            id="non-numeric",
        ),
    ],
)
def test_qc_rejects_malformed_dataframes(data, kwargs, error, message):
    with pytest.raises(error, match=message):
        splt.qc(data, show=False, **kwargs)


def test_qc_samples_its_points_with_any_seed_of_numpy():
    data = _qc_data()

    # pandas draws the shown points with NumPy's RandomState, which takes
    # seeds from 0 to 2**32 - 1.
    result = splt.qc(data, max_points=5, seed=np.uint32(2**32 - 1), show=False)

    assert result.provenance.extras["seed"] == 2**32 - 1
    assert type(result.provenance.extras["seed"]) is int
    assert result.provenance.extras["displayed_points"] == {
        "nCounts": 5,
        "nFeatures": 5,
    }
    result.close()
    for seed, error, message in (
        (2**32, ValueError, "^seed must be at most 4294967295$"),
        (-1, ValueError, "^seed must be at least 0$"),
        (1.5, TypeError, "^seed must be an integer$"),
    ):
        with pytest.raises(error, match=message):
            splt.qc(data, max_points=5, seed=seed, show=False)


def test_qc_vertical_layout_single_group_titles_and_figure_size():
    import matplotlib.pyplot as plt

    data = pd.DataFrame(
        {
            "groups": ["only"] * 4,
            "metricA": [1.0, 2.0, 3.0, 4.0],
            "metricB": [10.0, 12.0, 14.0, 16.0],
        }
    )
    result = splt.qc(
        data,
        color="#123456",
        max_points=0,
        show_on_single_row=False,
        sup_title="QC overview",
        show=False,
    )
    result.figure.canvas.draw()

    assert result.figure.get_size_inches() == pytest.approx((3.0, 6.0))
    assert result.figure._suptitle.get_text() == "QC overview"
    assert result.scales[0].order == ("only",)
    assert result.scales[0].palette == {"only": "#123456"}
    assert result.axes["metricA"].get_position().y0 > (
        result.axes["metricB"].get_position().y0
    )
    assert result.axes["metricA"].get_title() == "Median: 2.5"
    assert result.axes["metricB"].get_title() == "Median: 13.0"
    assert all(len(axis.get_xticks()) == 0 for axis in result.axes.values())
    figure_number = result.figure.number
    result.close()
    assert not plt.fignum_exists(figure_number)


def test_qc_takes_a_given_size_and_titles_a_median_only_of_finite_values():
    data = pd.DataFrame(
        {
            "groups": ["only"] * 4,
            "metricA": [1.0, 2.0, 3.0, 4.0],
            "metricB": [np.nan] * 4,
        }
    )
    result = splt.qc(data, max_points=0, figsize=(4.5, 3.0), show=False)
    try:
        assert result.figure.get_size_inches() == pytest.approx((4.5, 3.0))
        # A single group's panel says its median, which a metric without a
        # finite value does not have.
        assert result.axes["metricA"].get_title() == "Median: 2.5"
        assert result.axes["metricB"].get_title() == ""
    finally:
        result.close()


def test_qc_orders_natural_and_missing_groups_without_subsampling():
    groups = ["group10", "group2", None] * 3
    data = pd.DataFrame(
        {
            "groups": groups,
            "metric": np.arange(len(groups), dtype=float),
        }
    )
    result = splt.qc(data, max_points=100, show=False)

    assert result.scales[0].order == ("group2", "group10", "NA")
    assert result.provenance.extras["displayed_points"] == {"metric": len(data)}
    assert result.tables["summary"]["group"].tolist() == [
        "group10",
        "group2",
        "NA",
    ]
    result.close()


def test_elbow_without_detection_uses_data_driven_width():
    # A straight line has no point of maximum curvature, so no elbow. Its
    # length matters: kneed 0.8.1, the oldest supported, finds an elbow in
    # the rounding of a straight line of 12 values.
    values = np.linspace(1.0, 0.1, 10)
    result = splt.elbow(values, show=False)

    # A quarter inch for each of the 10 PCs, beside the default height.
    assert result.figure.get_size_inches() == pytest.approx((2.5, 2.0))
    assert result.provenance.extras["elbow_pc"] is None
    assert result.legends == ()
    assert len(result.axes["elbow"].lines) == 1
    table = result.tables["variance_explained"]
    assert not table["is_elbow"].any()
    np.testing.assert_array_equal(table["pc"], np.arange(1, 11))
    result.close()


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([], id="empty"),
        pytest.param([[1.0, 0.5]], id="two-dimensional"),
        pytest.param([1.0, np.inf], id="non-finite"),
    ],
)
def test_elbow_rejects_malformed_variance_arrays(values):
    with pytest.raises(ValueError, match="variance_explained"):
        splt.elbow(values, show=False)


@pytest.mark.parametrize(
    ("values", "width"),
    [
        pytest.param([5.0], 2.0, id="one-pc"),
        pytest.param([5.0, 1.0], 2.0, id="two-pcs"),
        pytest.param([3.0, 3.0, 3.0], 2.0, id="equal-values"),
        # Twelve PCs take their quarter inch each, above the floor.
        pytest.param([0.0] * 12, 3.0, id="twelve-zeros"),
    ],
)
def test_elbow_without_a_curve_marks_no_elbow_and_warns_nothing(
    values, width, monkeypatch
):
    def unreachable():
        raise AssertionError("an elbow plot without a curve searched for a knee")

    # kneed divides by the range of the values and needs a point between the
    # curve's ends, so these values are not searched, and draw without kneed.
    monkeypatch.setattr(diagnostics, "require_kneed", unreachable)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = splt.elbow(values, show=False)

    assert result.provenance.extras["elbow_pc"] is None
    assert result.legends == ()
    ax = result.axes["elbow"]
    assert len(ax.lines) == 1
    assert ax.get_legend() is None
    table = result.tables["variance_explained"]
    assert not table["is_elbow"].any()
    np.testing.assert_array_equal(table["pc"], np.arange(1, len(values) + 1))
    np.testing.assert_array_equal(table["variance_explained"], values)
    assert result.figure.get_size_inches() == pytest.approx((width, 2.0))
    result.close()


def test_elbow_searches_every_curve_of_three_values_or_more(monkeypatch):
    searched = []
    locator = diagnostics.require_kneed()

    def counted():
        searched.append(True)
        return locator

    monkeypatch.setattr(diagnostics, "require_kneed", counted)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Three PCs hold one PC between the curve's ends, and one value
        # that differs from the others makes a curve.
        for values in ([5.0, 2.0, 1.0], [3.0, 3.0, 2.9]):
            splt.elbow(values, show=False).close()
        assert len(searched) == 2
        splt.elbow([5.0, 1.0], show=False).close()
    assert len(searched) == 2


def test_elbow_keeps_only_kneeds_warning_of_no_knee_from_its_caller(monkeypatch):
    real = diagnostics.require_kneed()

    def warning_as_kneed_081(message, category):
        class Locator(real):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                warnings.warn(message, category)

        return Locator

    # kneed 0.8.1, the oldest supported, warns that it found no knee, which
    # elbow_pc already says; the plot leaves that warning out.
    monkeypatch.setattr(
        diagnostics,
        "require_kneed",
        lambda: warning_as_kneed_081("No knee/elbow found", UserWarning),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = splt.elbow([5.0, 2.0, 1.0], show=False)
    assert result.provenance.extras["elbow_pc"] is None
    result.close()

    # Any other warning of the search reaches the caller.
    for message, category in (
        ("invalid value encountered in divide", RuntimeWarning),
        ("No knee/elbow found in the first pass", UserWarning),
    ):
        monkeypatch.setattr(
            diagnostics,
            "require_kneed",
            lambda m=message, c=category: warning_as_kneed_081(m, c),
        )
        with pytest.warns(category) as caught:
            splt.elbow([5.0, 2.0, 1.0], show=False).close()
        assert [str(warning.message) for warning in caught] == [message]


@pytest.mark.parametrize(
    ("n_pcs", "width"),
    [(1, 2.0), (4, 2.0), (7, 2.0), (8, 2.0), (9, 2.25), (30, 7.5)],
)
def test_elbow_is_a_quarter_inch_wide_per_pc_and_at_least_two_inches(n_pcs, width):
    values = np.linspace(10.0, 1.0, n_pcs) ** 2
    result = splt.elbow(values, show=False)

    assert result.figure.get_size_inches() == pytest.approx((width, 2.0))
    result.close()


@pytest.mark.parametrize("theme", ["notebook", "paper"])
def test_elbow_of_one_pc_is_wide_enough_for_its_axis_label(theme):
    result = splt.elbow([5.0], theme=theme, show=False)
    figure = result.figure
    figure.canvas.draw()

    label = result.axes["elbow"].xaxis.label
    extent = label.get_window_extent(figure.canvas.get_renderer())
    # "Principal components" lies within the figure, not cut at either side.
    assert label.get_text() == "Principal components"
    assert extent.x0 >= figure.bbox.x0
    assert extent.x1 <= figure.bbox.x1
    result.close()


def test_elbow_keeps_the_width_that_figsize_gives():
    result = splt.elbow([5.0, 1.0], figsize=(0.5, 3.0), show=False)

    assert result.figure.get_size_inches() == pytest.approx((0.5, 3.0))
    result.close()


def test_graph_qc_clips_single_degree_outlier_and_records_limit():
    node_count = 1001
    leaves = np.arange(1, node_count, dtype=np.int64)
    rows = np.concatenate((np.zeros(node_count - 1, dtype=np.int64), leaves))
    columns = np.concatenate((leaves, np.zeros(node_count - 1, dtype=np.int64)))
    graph = sparse.csr_matrix(
        (np.ones(len(rows), dtype=float), (rows, columns)),
        shape=(node_count, node_count),
    )
    result = splt.graph_qc(graph, show=False)

    assert result.provenance.extras["degree_clip_limit"] == pytest.approx(6.0)
    assert result.axes["node_degree"].get_xlim() == pytest.approx((0.0, 6.0))
    assert [text.get_text() for text in result.axes["node_degree"].texts] == [
        "plot is clipped (max degree: 1000)"
    ]
    assert result.tables["node_degrees"]["degree"].max() == 1000
    result.close()


def test_graph_qc_rejects_malformed_sparse_adapters():
    class MissingData:
        shape = (2, 2)

    class BrokenDegreeCalculation:
        shape = (2, 2)
        data = np.array([1.0])

        def __ne__(self, other):
            return object()

    with pytest.raises(TypeError, match="two-dimensional sparse matrix"):
        splt.graph_qc(object(), show=False)
    with pytest.raises(TypeError, match="expose sparse edge weights"):
        splt.graph_qc(MissingData(), show=False)
    with pytest.raises(TypeError, match="non-zero degree calculation"):
        splt.graph_qc(BrokenDegreeCalculation(), show=False)


@pytest.mark.parametrize(
    ("arguments", "kwargs", "message"),
    [
        pytest.param(
            (
                np.ones((2, 2)),
                np.ones(4),
                np.ones(4),
                np.zeros(4, dtype=bool),
            ),
            {},
            "one-dimensional",
            id="dimensions",
        ),
        pytest.param(
            (
                np.ones(3),
                np.ones(4),
                np.ones(4),
                np.zeros(4, dtype=bool),
            ),
            {},
            "matching lengths",
            id="lengths",
        ),
        pytest.param(
            _hvg_arguments(),
            {"point_sizes": (3,)},
            "point_sizes",
            id="point-sizes",
        ),
        pytest.param(
            _hvg_arguments(),
            {"colormaps": ("viridis",)},
            "colormaps",
            id="colormaps",
        ),
    ],
)
def test_highly_variable_features_rejects_malformed_adapters(
    arguments,
    kwargs,
    message,
):
    with pytest.raises(ValueError, match=message):
        splt.highly_variable_features(*arguments, show=False, **kwargs)


@pytest.mark.parametrize(
    ("colormaps", "error", "message"),
    [
        (
            ("winter", 3),
            TypeError,
            "colormaps[1] must be None, a Matplotlib colormap name or a registered "
            "Colormap, not int",
        ),
        (
            ("not_a_colormap", "magma_r"),
            ValueError,
            "colormaps[0] must name a Matplotlib colormap, such as 'viridis'; "
            "'not_a_colormap' is not one",
        ),
        # A name has two letters, but it is one colormap, not two.
        (
            "ab",
            TypeError,
            "colormaps must be a pair of colormaps, for the features not "
            "selected and the selected ones",
        ),
        (
            3,
            TypeError,
            "colormaps must be a pair of colormaps, for the features not "
            "selected and the selected ones",
        ),
    ],
)
def test_highly_variable_features_name_the_colormap_they_refuse(
    colormaps, error, message
):
    with pytest.raises(error) as raised:
        splt.highly_variable_features(
            *_hvg_arguments(), colormaps=colormaps, show=False
        )

    assert raised.value.args == (message,)


def test_highly_variable_features_draw_with_colormaps_they_are_given(
    registered_colormap,
):
    from matplotlib.colors import ListedColormap

    not_selected = registered_colormap(
        ListedColormap(["#000000", "#ffffff"], name="cytearc_black_white")
    )
    result = splt.highly_variable_features(
        *_hvg_arguments(), colormaps=(not_selected, "plasma"), show=False
    )
    # None draws Matplotlib's default colormap, viridis, as it always has.
    default = splt.highly_variable_features(
        *_hvg_arguments(), colormaps=(None, "plasma"), show=False
    )

    unselected, selected = result.axes["highly_variable_features"].collections
    assert unselected.get_cmap() == not_selected
    assert selected.get_cmap().name == "plasma"
    assert default.axes["highly_variable_features"].collections[0].get_cmap().name == (
        "viridis"
    )
    # The results record each colormap by its name, and None as None.
    assert result.provenance.extras["colormaps"] == ["cytearc_black_white", "plasma"]
    assert default.provenance.extras["colormaps"] == [None, "plasma"]
    # The legend takes the middle color of the colormaps drawn: white of the
    # two-color map, plasma's #cc4778, and viridis's #21918c for None.
    assert _legend_colors(result) == {
        "Not selected": "#ffffff",
        "Selected": "#cc4778",
    }
    assert _legend_colors(default) == {
        "Not selected": "#21918c",
        "Selected": "#cc4778",
    }
    result.close()
    default.close()


def _legend_markers(result: PlotResult) -> dict[str, object]:
    legend = result.axes["highly_variable_features"].get_legend()
    return {
        text.get_text(): handle
        for text, handle in zip(legend.get_texts(), legend.legend_handles, strict=True)
    }


def _legend_colors(result: PlotResult) -> dict[str, str]:
    from matplotlib.colors import to_hex

    return {
        label: to_hex(marker.get_markerfacecolor())
        for label, marker in _legend_markers(result).items()
    }


@pytest.mark.parametrize("order", ["undetected-first", "reversed"])
def test_highly_variable_features_legend_draws_each_group_whatever_comes_first(
    order,
):
    from matplotlib.colors import to_hex, to_rgba

    # The first feature is not selected and no cell detects it: the log2 of
    # its mean is -inf, so its point takes the colormap's transparent color
    # for values that are not finite, which the legend copied.
    arrays = (
        np.array([0.0, 2.0, 4.0, 8.0, 1.0, 16.0]),
        np.array([1.0, 3.0, 8.0, 12.0, 2.0, 20.0]),
        np.array([0, 20, 30, 40, 5, 900]),
        np.array([False, True, False, True, False, True]),
    )
    if order == "reversed":
        # Now the first features are a selected one, detected in 900 cells,
        # and a feature not selected, in 5.
        arrays = tuple(array[::-1] for array in arrays)
    result = splt.highly_variable_features(*arrays, show=False)

    markers = _legend_markers(result)
    assert list(markers) == ["Not selected", "Selected"]
    not_selected, selected = markers["Not selected"], markers["Selected"]
    # Both markers have one readable size, where the points take 3 and 30
    # square points.
    assert not_selected.get_markersize() == selected.get_markersize() == 6
    # Each marker takes the middle color of its group's colormap, winter's
    # #0080bf and magma_r's #b5367a, at the alpha of the group's points.
    assert to_hex(not_selected.get_markerfacecolor()) == "#0080bf"
    assert to_hex(selected.get_markerfacecolor()) == "#b5367a"
    assert to_rgba(not_selected.get_markerfacecolor())[3] == 1.0
    assert to_rgba(selected.get_markerfacecolor())[3] == 1.0
    assert not_selected.get_alpha() == 0.6
    assert selected.get_alpha() == 0.8
    # The selected points' black edge marks the selected marker alone.
    assert to_rgba(selected.get_markeredgecolor()) == to_rgba("black")
    assert selected.get_markeredgewidth() == 0.5
    assert not_selected.get_markeredgecolor() == "none"
    assert not_selected.get_marker() == selected.get_marker() == "o"
    # The points keep their sizes, and the undetected one its transparent
    # color, which the legend no longer shows.
    result.figure.canvas.draw()
    unselected_points, selected_points = result.axes[
        "highly_variable_features"
    ].collections
    assert unselected_points.get_sizes().tolist() == [3.0]
    assert selected_points.get_sizes().tolist() == [30.0]
    undetected = 0 if order == "undetected-first" else 2
    assert to_rgba(unselected_points.get_facecolors()[undetected]) == (0, 0, 0, 0)
    result.close()


def test_hvg_plot_options_return_the_options_as_the_plot_draws_them(
    registered_colormap,
):
    from matplotlib.colors import ListedColormap

    grays = registered_colormap(
        ListedColormap(["#444444", "#bbbbbb"], name="cytearc_grays")
    )
    checked = hvg_plot_options(
        label_size=np.float32(9),
        figsize=np.array([5, 3]),
        point_sizes=[0, 2.5],
        colormaps=(grays, None),
        theme="paper",
    )

    assert checked == {
        "label_size": 9.0,
        "figsize": (5.0, 3.0),
        "point_sizes": (0.0, 2.5),
        "colormaps": ("cytearc_grays", None),
        "theme": "paper",
    }
    sizes = (checked["label_size"], *checked["figsize"], *checked["point_sizes"])
    assert all(type(size) is float for size in sizes)
    # Options left out stay out, as features.hvgs passes only those given.
    assert hvg_plot_options(label_size=9) == {"label_size": 9.0}
    assert hvg_plot_options() == {}


def test_hvg_plot_options_check_every_option_of_the_plot():
    options = {
        name: parameter.default
        for name, parameter in inspect.signature(
            splt.highly_variable_features
        ).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY and name != "show"
    }

    # features.hvgs forwards each of these keywords to the plot.
    assert set(options) == {
        "label_size",
        "figsize",
        "point_sizes",
        "colormaps",
        "theme",
    }
    assert hvg_plot_options(**options) == {
        "label_size": 12.0,
        "figsize": (4.5, 4.0),
        "point_sizes": (3.0, 30.0),
        "colormaps": ("winter", "magma_r"),
        "theme": "notebook",
    }
    with pytest.raises(TypeError) as raised:
        hvg_plot_options(title="HVGs", show=False)
    assert raised.value.args == (
        "highly_variable_features() got unexpected keyword arguments: 'show', 'title'",
    )


_PAIR_OF_SIZES = (
    "point_sizes must be a pair of marker areas in points squared, for the "
    "features not selected and the selected ones"
)
_PAIR_OF_LENGTHS = "figsize must be a pair of numbers, the width and height in inches"


@pytest.mark.parametrize(
    ("option", "value", "error", "message"),
    [
        ("label_size", "big", TypeError, "label_size must be a real number"),
        ("label_size", True, TypeError, "label_size must be a real number"),
        ("label_size", np.nan, ValueError, "label_size must be finite"),
        ("label_size", 0, ValueError, "label_size must be positive"),
        ("figsize", "wide", TypeError, _PAIR_OF_LENGTHS),
        ("figsize", np.array(4.0), TypeError, _PAIR_OF_LENGTHS),
        ("figsize", (4.0,), ValueError, "figsize must contain two values"),
        ("figsize", (10, 8, "cm"), ValueError, "figsize must contain two values"),
        ("figsize", ("4", 3), TypeError, "figsize[0] must be a real number"),
        ("figsize", (0, 4), ValueError, "figsize[0] must be positive"),
        ("figsize", (4, -1e-9), ValueError, "figsize[1] must be positive"),
        ("figsize", (4, np.inf), ValueError, "figsize[1] must be finite"),
        ("point_sizes", 3, TypeError, _PAIR_OF_SIZES),
        ("point_sizes", (3,), ValueError, "point_sizes must contain two values"),
        ("point_sizes", ("a", "b"), TypeError, "point_sizes[0] must be a real number"),
        ("point_sizes", (-1, 30), ValueError, "point_sizes[0] must be at least 0"),
        ("point_sizes", (3, np.nan), ValueError, "point_sizes[1] must be finite"),
        (
            "theme",
            "sepia",
            KeyError,
            "Unknown theme 'sepia'. Choose from: ['dark', 'minimal', 'notebook', "
            "'paper']",
        ),
    ],
)
def test_hvg_plot_options_name_the_value_they_refuse(option, value, error, message):
    import matplotlib.pyplot as plt

    with pytest.raises(error) as raised:
        hvg_plot_options(**{option: value})
    assert raised.value.args == (message,)

    # The plot refuses the value with the same error before it opens a figure.
    figures = set(plt.get_fignums())
    with pytest.raises(error) as drawn:
        splt.highly_variable_features(*_hvg_arguments(), show=False, **{option: value})
    assert drawn.value.args == (message,)
    assert set(plt.get_fignums()) == figures


def test_highly_variable_features_hide_points_of_no_area_and_take_the_default_panel():
    result = splt.highly_variable_features(
        *_hvg_arguments(), point_sizes=(0, 0), figsize=None, show=False
    )

    # A marker area of 0 hides a group's points; the legend still names it.
    unselected, selected = result.axes["highly_variable_features"].collections
    assert unselected.get_sizes().tolist() == selected.get_sizes().tolist() == [0.0]
    assert list(_legend_markers(result)) == ["Not selected", "Selected"]
    # None takes CyteArc's default panel of 3.2 inches, 0.4 wider for its
    # labels.
    assert result.figure.get_size_inches() == pytest.approx((3.6, 3.2))
    result.close()


@pytest.mark.parametrize(
    ("cmap", "error", "message"),
    [
        (
            3,
            TypeError,
            "cmap must be a seaborn palette or Matplotlib colormap name, or a "
            "list of colors, not 3",
        ),
        (
            "not_a_palette",
            ValueError,
            "cmap must be a seaborn palette or Matplotlib colormap name, or a "
            "list of colors, not 'not_a_palette'",
        ),
        (
            ["#ff0000", "not_a_color"],
            ValueError,
            "cmap must be a seaborn palette or Matplotlib colormap name, or a "
            "list of colors, not ['#ff0000', 'not_a_color']",
        ),
    ],
)
def test_qc_names_the_palette_it_refuses(cmap, error, message):
    with pytest.raises(error) as raised:
        splt.qc(_qc_data(), cmap=cmap, show=False)

    assert raised.value.args == (message,)


@pytest.mark.parametrize(
    ("cmap", "colors"),
    [
        # seaborn's deep palette, and a list of colors, in group order.
        ("deep", {"a": "#4c72b0", "b": "#dd8452"}),
        (["#ff0000", "#0000ff"], {"a": "#ff0000", "b": "#0000ff"}),
    ],
)
def test_qc_takes_any_palette_that_seaborn_reads(cmap, colors):
    result = splt.qc(_qc_data(), cmap=cmap, max_points=0, show=False)

    assert result.scales[0].palette == colors
    result.close()


def test_highly_variable_features_accepts_empty_feature_arrays():
    empty_float = np.array([], dtype=float)
    empty_bool = np.array([], dtype=bool)
    result = splt.highly_variable_features(
        empty_float,
        empty_float,
        empty_float,
        empty_bool,
        show=False,
    )

    assert result.tables["features"].empty
    assert result.provenance.extras["n_features"] == 0
    assert result.provenance.extras["max_expressing_cells"] == 0
    result.close()


def test_diagnostic_default_show_closes_all_owned_results():
    import matplotlib.pyplot as plt

    existing = set(plt.get_fignums())
    results = [
        splt.qc(_qc_data(), max_points=0),
        splt.elbow([1.0, 0.5, 0.25]),
        splt.graph_qc(_graph()),
        splt.highly_variable_features(*_hvg_arguments()),
    ]

    assert all(result.owns_figure for result in results)
    assert all(not plt.fignum_exists(result.figure.number) for result in results)
    # No figure the plots opened stays open, and the results keep their tables.
    assert set(plt.get_fignums()) == existing
    qc, elbow, graph, features = (result.tables for result in results)
    np.testing.assert_array_equal(qc["summary"]["median"], [11.5, 21.5, 5.0, 9.0])
    np.testing.assert_array_equal(
        elbow["variance_explained"]["variance_explained"], [1.0, 0.5, 0.25]
    )
    assert graph["node_degrees"]["degree"].tolist() == [1, 2, 1]
    assert features["features"]["selected"].tolist() == [False, True, False, True]
