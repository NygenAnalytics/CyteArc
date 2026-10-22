"""Violin widths, bounded density estimates, and category label checks."""

import sys

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import cytearc.metadata.selection as selection_module
import cytearc.plotting as splt
from cytearc.metadata.selection import valid_category_mask
from tests.test_plotting_category_colors import _bind_store


def _violin_store(groups, values, split=None):
    columns = {
        "I": np.ones(len(groups), dtype=bool),
        "group": np.asarray(groups, dtype=object),
        "metric": np.asarray(values, dtype=np.float64),
    }
    if split is not None:
        columns["batch"] = np.asarray(split, dtype=object)
    return _bind_store(columns, np.random.default_rng(0), None)


def _body_widths(ax):
    """Horizontal extent of each violin body, in category-axis units."""
    return [
        float(np.ptp(path.vertices[:, 0]))
        for collection in ax.collections
        if type(collection).__name__ in ("PolyCollection", "FillBetweenPolyCollection")
        for path in collection.get_paths()
    ]


def test_split_violins_scale_every_half_to_the_full_width():
    rng = np.random.default_rng(0)
    n = 400
    # Group "zeros" piles up at zero, so its density peak is far higher than
    # that of group "spread", whose values range over 0 to 45.
    groups = np.repeat(["spread", "zeros"], n)
    values = np.concatenate((rng.uniform(0.0, 45.0, n), rng.normal(0.0, 0.01, n)))
    split = np.tile(["b1", "b2"], n)
    store = _violin_store(groups, values, split)

    result = splt.distribution(
        store,
        "metric",
        grouping=splt.CellField("group"),
        split_by="batch",
        max_points=0,
        violin_inner=None,
        show=False,
    )
    try:
        widths = _body_widths(result.axes["metric"])
        # Two groups of two halves, each as wide as half the violin's width.
        assert widths == pytest.approx([0.4] * 4, abs=1e-6)
    finally:
        result.close()


def _capture_violin_data(monkeypatch):
    import seaborn

    captured = []
    violinplot = seaborn.violinplot

    def capture(*args, data=None, **kwargs):
        captured.append(data)
        return violinplot(*args, data=data, **kwargs)

    monkeypatch.setattr(seaborn, "violinplot", capture)
    return captured


def test_violin_densities_use_a_bounded_sample_that_keeps_each_range(monkeypatch):
    rng = np.random.default_rng(1)
    groups = np.repeat(["big", "small"], [5_000, 30])
    values = rng.gamma(2.0, 1.0, size=len(groups))
    store = _violin_store(groups, values)
    captured = _capture_violin_data(monkeypatch)

    bounded = splt.distribution(
        store,
        "metric",
        grouping=splt.CellField("group"),
        max_density_points=1_000,
        show=False,
    )
    again = splt.distribution(
        store,
        "metric",
        grouping=splt.CellField("group"),
        max_density_points=1_000,
        show=False,
    )
    exact = splt.distribution(
        store,
        "metric",
        grouping=splt.CellField("group"),
        max_density_points=None,
        show=False,
    )
    try:
        frame = captured[0]
        big = frame.loc[frame["group"] == "big", "value"].to_numpy()
        small = frame.loc[frame["group"] == "small", "value"].to_numpy()
        assert len(big) == 1_000
        # The sample keeps the group's extremes, so the violin spans its range.
        assert big.min() == values[:5_000].min()
        assert big.max() == values[:5_000].max()
        assert np.isin(big, values[:5_000]).all()
        # A violin within the bound keeps every value.
        np.testing.assert_array_equal(np.sort(small), np.sort(values[5_000:]))
        pd.testing.assert_frame_equal(captured[1], frame)
        assert len(captured[2]) == len(values)
        assert "subsampled_density" in bounded.provenance.notes
        assert bounded.provenance.extras["max_density_points"] == 1_000
        assert bounded.provenance.extras["approximate"] is True
        assert "subsampled_density" not in exact.provenance.notes
        assert exact.provenance.extras["max_density_points"] is None
    finally:
        for result in (bounded, again, exact):
            result.close()


def test_small_violins_render_exactly_as_without_a_bound(monkeypatch):
    rng = np.random.default_rng(2)
    groups = np.repeat(["a", "b", "c"], 50)
    store = _violin_store(groups, rng.normal(size=150), np.tile(["x", "y"], 75))
    captured = _capture_violin_data(monkeypatch)

    plots = [
        splt.distribution(
            store,
            "metric",
            grouping=splt.CellField("group"),
            split_by="batch",
            max_density_points=limit,
            show=False,
        )
        for limit in (None, 20_000, 25)
    ]
    try:
        # Halves of 25 values meet a bound of 25 and keep every value too.
        for frame in captured[1:]:
            pd.testing.assert_frame_equal(frame, captured[0])
        reference = plots[0].axes["metric"]
        for result in plots[1:]:
            ax = result.axes["metric"]
            assert len(ax.collections) == len(reference.collections)
            for drawn, expected in zip(ax.collections, reference.collections):
                for path, expected_path in zip(
                    drawn.get_paths(), expected.get_paths(), strict=True
                ):
                    np.testing.assert_array_equal(path.vertices, expected_path.vertices)
            assert "subsampled_density" not in result.provenance.notes
    finally:
        for result in plots:
            result.close()


def test_density_bound_must_hold_both_extremes():
    store = _violin_store(["a", "b"], [1.0, 2.0])
    open_figures = plt.get_fignums()
    with pytest.raises(
        ValueError, match="max_density_points must be None or at least 2"
    ):
        splt.distribution(
            store,
            "metric",
            grouping=splt.CellField("group"),
            max_density_points=1,
            show=False,
        )
    # The refused call opens no figure.
    assert plt.get_fignums() == open_figures


def _reference_mask(values, missing_mask=None):
    """The original per-value check that the dtype paths must reproduce."""

    def missing(value):
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False

    def blank(value):
        if isinstance(value, str):
            return value.strip() == ""
        if isinstance(value, bytes | np.bytes_):
            return value.strip() == b""
        return False

    array = np.asarray(values, dtype=object)
    valid = np.asarray([not missing(value) and not blank(value) for value in array])
    if missing_mask is not None:
        valid &= ~np.asarray(missing_mask, dtype=bool)
    return valid


def _mixed_inputs():
    whitespace = [
        chr(code)
        for code in range(sys.maxunicode + 1)
        if not 0xD800 <= code <= 0xDFFF and chr(code).isspace()
    ]
    text = np.asarray(
        [
            "",
            " ",
            "a",
            " a ",
            "​",
            "\x00a",
            "a b",
            *whitespace,
            *(character * 3 + "x" for character in whitespace),
        ]
    )
    raw = np.asarray([bytes([code]) for code in range(256)] + [b" \t", b" x "])
    objects = np.empty(14, dtype=object)
    objects[:] = [
        None,
        np.nan,
        pd.NA,
        pd.NaT,
        "",
        "  ",
        b" ",
        np.str_("x"),
        ["a", "b"],
        ("a", 1),
        [np.nan],
        np.array([np.nan]),
        1.5,
        True,
    ]
    return {
        "text": text,
        "bytes": raw,
        "float": np.array([0.0, -0.0, np.nan, np.inf, -np.inf, 2.5]),
        "float32": np.array([np.nan, 1.0], dtype=np.float32),
        "complex": np.array([complex(np.nan, 0), complex(0, np.nan), 1 + 2j]),
        "int": np.array([0, -1, 7], dtype=np.int16),
        "unsigned": np.array([0, 3], dtype=np.uint8),
        "bool": np.array([True, False]),
        "object": objects,
        "datetime": np.array(["NaT", "2020-01-01"], dtype="datetime64[ns]"),
        "timedelta": np.array([1, "NaT"], dtype="timedelta64[s]"),
        "list": ["a", None, " ", 3.0, float("nan"), "\x00"],
    }


@pytest.mark.parametrize("kind", sorted(_mixed_inputs()))
def test_category_mask_by_dtype_matches_the_per_value_check(kind):
    values = _mixed_inputs()[kind]
    expected = _reference_mask(values)
    np.testing.assert_array_equal(valid_category_mask(values), expected)
    missing = np.zeros(len(values), dtype=bool)
    missing[::2] = True
    np.testing.assert_array_equal(
        valid_category_mask(values, missing_mask=missing),
        _reference_mask(values, missing),
    )


def test_typed_category_masks_skip_the_per_value_check(monkeypatch):
    def refuse(_value):
        raise AssertionError("typed arrays are checked by dtype")

    monkeypatch.setattr(selection_module, "_is_missing_label", refuse)
    monkeypatch.setattr(selection_module, "_is_blank_label", refuse)
    labels = np.asarray(["Ductal", "  ", "Alpha", ""] * 50_000)
    mask = valid_category_mask(labels)
    np.testing.assert_array_equal(mask[:4], [True, False, True, False])
    assert int(mask.sum()) == 100_000
    assert valid_category_mask(np.array([1.0, np.nan])).tolist() == [True, False]
    assert valid_category_mask(np.array([b"", b"x"])).tolist() == [False, True]
    assert valid_category_mask(np.array([3, 4])).tolist() == [True, True]
