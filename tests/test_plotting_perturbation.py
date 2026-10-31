import inspect
import warnings
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.markers import MarkerStyle
from matplotlib.text import Annotation, Text
from scipy.stats import beta, false_discovery_control, norm

from cytearc.perturbation.associations import PerturbationAssociations
from cytearc.perturbation.clusters import PerturbationClusters, profile_distances
from cytearc.perturbation.edistance import PerturbationEDistance
from cytearc.perturbation.efficiency import PerturbationEfficiency
from cytearc.perturbation.guide_calls import (
    GuideCalls,
    guide_call_arrays,
    guide_calls_frames,
)
from cytearc.perturbation.guide_thresholds import (
    METHOD_PARAMETERS,
    GuideThresholds,
    guide_threshold_frames,
)
from cytearc.perturbation.mixscape import MixscapeResult
from cytearc.perturbation.mixtures import (
    FIT_STATUSES,
    gaussian_posterior,
    poisson_gaussian_posterior,
)
from cytearc.plotting._contracts import CategoricalScale, ColorScale
from cytearc.plotting._figure import LegendSpec, compose_results
from cytearc.plotting._style import default_point_size, foreground_color
from cytearc.plotting.perturbation import (
    association_calibration,
    association_volcano,
    edistance,
    efficiency,
    guide_calls,
    guide_thresholds,
    mixscape,
    perturbation_clusters,
)
from cytearc.storage.refs import ArtifactRef

CALLS = ArtifactRef(
    scope="assay", assay="CRISPR", kind="guide_calls", artifact_id="a" * 64
)
THRESHOLDS = ArtifactRef(
    scope="assay", assay="CRISPR", kind="guide_thresholds", artifact_id="b" * 64
)
CALLED = to_rgba("#2b6cb0")
NOT_CALLED = to_rgba("#9e9e9e")
POSTERIORS = {
    "poisson_gaussian": poisson_gaussian_posterior,
    "gaussian": gaussian_posterior,
}

# Each fit is a status, the four parameters of the posterior (None where the
# fit has none), and the cells at each nonzero UMI count. The thresholds follow
# from the parameters.
ONE_SAMPLE = {
    None: {
        "g1": (
            "fitted",
            (0.4, 5.5, 1.0, 0.1),
            {1: 60, 2: 25, 3: 8, 20: 3, 45: 6, 90: 2, 1500: 1},
        ),
        "g2": ("too_sparse", None, {1: 1}),
        "g3": ("no_positive", (0.4, 14.0, 0.5, 0.01), {1: 40, 2: 10, 40: 2}),
    }
}
TWO_SAMPLES = {
    "s1": {
        "g1": (
            "fitted",
            (0.1, 0.6, 0.12, 0.2),
            {1: 50, 2: 20, 3: 6, 4: 3, 6: 4, 10: 8, 12: 5},
        ),
        "g2": ("no_fit", None, {1: 20, 2: 3}),
        "g3": ("no_positive", (0.1, 3.0, 0.2, 0.05), {1: 30, 2: 6, 5: 1}),
    },
    "s2": {
        "g1": (
            "fitted",
            (0.1, 1.5, 0.15, 0.2),
            {1: 40, 2: 12, 4: 2, 9: 6, 30: 7, 35: 3},
        ),
        "g2": ("no_cells", None, {}),
        "g3": ("fitted", (0.0, 0.30103, 0.05, 0.001), {1: 3}),
    },
}
N_CELLS = {None: 1000, "s1": 500, "s2": 400}
ONE = ("poisson_gaussian", ONE_SAMPLE)
TWO = ("gaussian", TWO_SAMPLES)


@pytest.fixture(autouse=True)
def _figures_are_closed():
    before = set(plt.get_fignums())
    yield
    leaked = set(plt.get_fignums()) - before
    for number in leaked:
        plt.close(number)
    assert not leaked, f"Figures left open: {sorted(leaked)}"


def _store(n_guides: int) -> SimpleNamespace:
    """A store whose loader returns calls of a library of ``n_guides`` guides.

    Guide g is called in g + 1 cells, so guides rank in reverse order. The
    first cell carries guide 0 and also guide 1, so it is the one cell with two
    called guides.
    """
    guide_ids = np.asarray([f"g{index}" for index in range(n_guides)], dtype=object)
    codes = np.repeat(np.arange(n_guides), np.arange(1, n_guides + 1))
    cells = np.arange(len(codes))
    arrays = guide_call_arrays(
        np.concatenate([cells, [0]]),
        np.concatenate([codes, [1]]),
        np.full(len(codes) + 1, 20),
        n_cells=len(codes),
        guide_ids=guide_ids,
    )
    frames = guide_calls_frames(
        np.asarray([f"cell{index}" for index in cells]),
        values=arrays.values,
        n_guides=arrays.n_guides,
        top_guide=guide_ids[codes],
        top_umi=np.full(len(codes), 20),
        top_fraction=np.where(arrays.n_guides > 1, 0.5, 1.0),
        total_umi=np.where(arrays.n_guides > 1, 40, 20),
        guide_codes=arrays.guide_codes,
        guide_umis=arrays.guide_umis,
        guide_ids=guide_ids,
    )
    loaded = GuideCalls(
        cells=frames[0],
        calls=frames[1],
        guides=pd.Index(guide_ids, name="guide"),
        source="import_guide_calls",
    )
    return SimpleNamespace(
        perturbation=SimpleNamespace(load_guide_calls=lambda ref: loaded)
    )


def test_a_large_library_ranks_guides_by_their_cells() -> None:
    result = guide_calls(_store(50), CALLS, show=False)

    try:
        ax = result.axes["cells_per_guide"]
        assert ax.get_xlabel() == "Guides, ranked by cells"
        lines = ax.get_lines()
        assert len(lines) == 2
        per_guide = result.tables["cells_per_guide"]
        np.testing.assert_array_equal(lines[0].get_ydata(), per_guide["cells"])
        np.testing.assert_array_equal(
            lines[1].get_ydata(), per_guide["one_guide_cells"]
        )
        assert per_guide["guide"].iloc[0] == "g49"
        # one_guide_cells, several_guide_cells, cells
        assert per_guide.set_index("guide").loc["g0"].tolist() == [0, 1, 1]
        assert per_guide.set_index("guide").loc["g1"].tolist() == [2, 1, 3]
    finally:
        result.close()


def test_a_small_library_labels_one_bar_per_guide() -> None:
    result = guide_calls(_store(5), CALLS, show=False)

    try:
        ax = result.axes["cells_per_guide"]
        labels = [label.get_text() for label in ax.get_yticklabels()]
        assert sorted(labels) == ["g0", "g1", "g2", "g3", "g4"]
        assert ax.get_lines() == []
    finally:
        result.close()


def test_cells_without_a_call_take_the_grey_of_the_threshold_panels() -> None:
    result = guide_calls(_store(5), CALLS, max_guides_per_cell=3, show=False)

    try:
        # The bars of no guide, one guide, two guides, and three or more.
        bars = result.axes["guides_per_cell"].patches
        several = to_rgba("#d1495b")
        assert [bar.get_facecolor() for bar in bars] == [
            NOT_CALLED,
            CALLED,
            several,
            several,
        ]
    finally:
        result.close()


def test_showing_the_plot_releases_its_figure() -> None:
    result = guide_calls(_store(5), CALLS)

    assert result.tables["guides_per_cell"]["cells"].sum() == 15


def test_guide_calls_without_a_cell_draw_nothing() -> None:
    loaded = _store(5).perturbation.load_guide_calls(CALLS)
    empty = GuideCalls(
        cells=loaded.cells.iloc[:0],
        calls=loaded.calls.iloc[:0],
        guides=loaded.guides,
        source=loaded.source,
    )
    store = SimpleNamespace(
        perturbation=SimpleNamespace(load_guide_calls=lambda ref: empty)
    )
    figures = set(plt.get_fignums())

    with pytest.raises(
        ValueError, match="^The guide calls hold no cell, so there is nothing to plot$"
    ):
        guide_calls(store, CALLS, show=False)
    assert set(plt.get_fignums()) == figures


def test_the_plot_refuses_settings_it_cannot_use() -> None:
    with pytest.raises(ValueError, match="^bins must be at least 1$"):
        guide_calls(_store(5), CALLS, bins=0, show=False)
    with pytest.raises(TypeError, match="^bins must be an integer$"):
        guide_calls(_store(5), CALLS, bins=2.0, show=False)
    with pytest.raises(ValueError, match="^max_guides_per_cell must be at least 2$"):
        guide_calls(_store(5), CALLS, max_guides_per_cell=1, show=False)
    with pytest.raises(TypeError, match="^max_guides_per_cell must be an integer$"):
        guide_calls(_store(5), CALLS, max_guides_per_cell=True, show=False)


def _threshold(method: str, parameters: dict[str, float], max_umi: int) -> int:
    """Return the smallest count up to ``max_umi`` whose posterior exceeds one half, or 0."""
    counts = np.arange(1, max_umi + 1)
    above = np.flatnonzero(POSTERIORS[method](counts, **parameters) > 0.5)
    return int(counts[above[0]]) if len(above) else 0


def _record(
    method: str,
    fits: dict[str | None, dict[str, tuple[Any, ...]]],
    *,
    min_umi: int | None,
    guides: Sequence[str] | None = None,
) -> GuideThresholds:
    """Build the record that the loader returns for ``fits``, read for ``guides``."""
    samples = list(fits)
    every_guide = list(fits[samples[0]])
    chosen = every_guide if guides is None else list(guides)
    names = METHOD_PARAMETERS[method]
    shape = (len(samples), len(chosen))
    status = np.zeros(shape, dtype=np.int8)
    threshold = np.zeros(shape, dtype=np.int64)
    n_umi_cells = np.zeros(shape, dtype=np.int64)
    max_umi = np.zeros(shape, dtype=np.int64)
    iterations = np.zeros(shape, dtype=np.int64)
    parameters = {name: np.full(shape, np.nan) for name in names}
    sizes: list[int] = []
    counts: list[int] = []
    cells: list[int] = []
    for row, sample in enumerate(samples):
        for column, guide in enumerate(chosen):
            name, values, histogram = fits[sample][guide]
            umis = sorted(histogram)
            status[row, column] = FIT_STATUSES.index(name)
            n_umi_cells[row, column] = sum(histogram.values())
            max_umi[row, column] = max(umis, default=0)
            sizes.append(len(umis))
            counts.extend(umis)
            cells.extend(histogram[umi] for umi in umis)
            if values is None:
                continue
            for parameter, value in zip(names, (*values, -1.5), strict=True):
                parameters[parameter][row, column] = value
            iterations[row, column] = 7
            threshold[row, column] = _threshold(
                method, dict(zip(names[:4], values, strict=True)), umis[-1]
            )
            # Keep the test data honest: a fitted guide calls a count.
            assert (threshold[row, column] > 0) == (name == "fitted")
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    table, histograms = guide_threshold_frames(
        samples=samples,
        guides=np.asarray(chosen, dtype=object),
        n_cells=np.asarray([N_CELLS[sample] for sample in samples]),
        status=status,
        threshold=threshold,
        n_umi_cells=n_umi_cells,
        max_umi=max_umi,
        iterations=iterations,
        parameters=parameters,
        histogram_offsets=offsets,
        histogram_counts=np.asarray(counts, dtype=np.int64),
        histogram_cells=np.asarray(cells, dtype=np.int64),
        min_umi=min_umi,
    )
    return GuideThresholds(
        table=table,
        histograms=histograms,
        guides=pd.Index(every_guide, name="guide"),
        method=method,
        sample_column=None if samples == [None] else "sample",
        min_umi=min_umi,
    )


def _threshold_store(
    method: str,
    fits: dict[str | None, dict[str, tuple[Any, ...]]],
    *,
    min_umi: int | None,
    requests: list[tuple[ArtifactRef, list[str] | None]] | None = None,
) -> SimpleNamespace:
    """A store whose loader reads ``fits`` like ``load_guide_thresholds`` does."""

    def load(ref: ArtifactRef, *, guides: list[str] | None = None) -> GuideThresholds:
        if requests is not None:
            requests.append((ref, guides))
        return _record(method, fits, min_umi=min_umi, guides=guides)

    return SimpleNamespace(perturbation=SimpleNamespace(load_guide_thresholds=load))


def _rows(frame: pd.DataFrame, sample: str | None, guide: str) -> pd.DataFrame:
    return frame[frame["sample"].isin([sample]) & (frame["guide"] == guide)]


def _check_panel(result: Any, key: Any, row: dict[str, Any]) -> None:
    """Check that a panel draws its rows of the tables."""
    ax = result.axes[key]
    histogram = _rows(result.tables["histograms"], row["sample"], row["guide"])
    threshold = int(row["threshold"])
    limit = threshold if pd.isna(row["call_threshold"]) else int(row["call_threshold"])
    expected_bars = [
        (float(umis), float(cells), CALLED if 0 < limit <= umis else NOT_CALLED)
        for umis, cells in zip(histogram["umis"], histogram["cells"], strict=True)
    ]
    drawn_bars = sorted(
        (
            patch.get_x() + patch.get_width() / 2,
            patch.get_height(),
            tuple(patch.get_facecolor()),
        )
        for patch in ax.patches
    )
    assert drawn_bars == expected_bars
    expected_lines = ([(threshold, "--")] if threshold else []) + (
        [(limit, ":")] if limit != threshold else []
    )
    assert [
        (line.get_xdata()[0], line.get_linestyle()) for line in ax.get_lines()
    ] == expected_lines
    texts = [text.get_text() for text in ax.texts]
    status = row["status"]
    assert texts == ([] if status == "fitted" else [f"{status}, no threshold"])
    assert ax.get_xscale() == "function"
    assert ax.get_yscale() == "log"
    assert ax.get_xlabel() == "UMIs of the guide in a cell"
    assert ax.get_ylabel() == "Cells"
    assert ax.get_title() == (
        row["guide"] if row["sample"] is None else f"{row['guide']} in {row['sample']}"
    )
    posterior = _rows(result.tables["posterior"], row["sample"], row["guide"])
    if status not in ("fitted", "no_positive"):
        assert (key, "posterior") not in result.axes
        assert posterior.empty
        return
    names = METHOD_PARAMETERS[result.provenance.extras["method"]][:4]
    expected = POSTERIORS[result.provenance.extras["method"]](
        posterior["umis"].to_numpy(), **{name: row[name] for name in names}
    )
    np.testing.assert_allclose(posterior["posterior"], expected)
    assert posterior["umis"].iloc[0] == 1
    assert posterior["umis"].iloc[-1] == row["max_umi"]
    assert (np.diff(posterior["umis"]) > 0).all()
    twin = result.axes[(key, "posterior")]
    (line,) = twin.get_lines()
    np.testing.assert_array_equal(line.get_xdata(), posterior["umis"])
    np.testing.assert_array_equal(line.get_ydata(), posterior["posterior"])
    assert twin.get_ylim() == (0.0, 1.0)
    assert twin.get_ylabel() == "Posterior of the guide population"
    assert twin.spines["right"].get_visible()


def test_threshold_panels_of_one_sample_draw_the_record() -> None:
    requests: list[tuple[ArtifactRef, list[str] | None]] = []
    store = _threshold_store(
        "poisson_gaussian", ONE_SAMPLE, min_umi=None, requests=requests
    )
    record = _record("poisson_gaussian", ONE_SAMPLE, min_umi=None)

    result = guide_thresholds(
        store, THRESHOLDS, guides=["g1", "g2", "g3", "g1"], show=False
    )

    try:
        assert requests == [(THRESHOLDS, ["g1", "g2", "g3"])]
        pd.testing.assert_frame_equal(result.tables["thresholds"], record.table)
        pd.testing.assert_frame_equal(result.tables["histograms"], record.histograms)
        assert list(result.axes) == [
            "g1",
            "g2",
            "g3",
            ("g1", "posterior"),
            ("g3", "posterior"),
        ]
        for key, row in zip(
            ["g1", "g2", "g3"],
            result.tables["thresholds"].to_dict("records"),
            strict=True,
        ):
            _check_panel(result, key, row)
        posterior = result.tables["posterior"]
        assert posterior["sample"].isna().all()
        assert posterior["guide"].unique().tolist() == ["g1", "g3"]
        # g1 reaches 1,500 UMIs, so its posterior is drawn at 1,000 counts or
        # fewer spread on the log scale; g3 gets every count up to 40.
        assert len(_rows(posterior, None, "g1")) <= 1000
        assert _rows(posterior, None, "g3")["umis"].tolist() == list(range(1, 41))
        (legend,) = result.figure.legends
        assert [text.get_text() for text in legend.get_texts()] == [
            "Cells called",
            "Cells not called",
            "Posterior of the guide population",
            "Threshold of the model",
        ]
        assert result.legends[1].kind == "line"
        provenance = result.provenance
        assert provenance.assay == "CRISPR"
        assert provenance.n_cells == 1000
        assert provenance.n_samples is None
        assert provenance.extras["guides"] == ["g1", "g2", "g3"]
        assert provenance.extras["samples"] is None
        assert provenance.extras["min_umi"] is None
        assert provenance.extras["thresholds"] == THRESHOLDS.to_dict()
        # The panels draw bars, which are never rasterized.
        assert provenance.extras["rasterized"] is False
    finally:
        result.close()


def test_threshold_panels_put_the_samples_of_a_guide_side_by_side() -> None:
    store = _threshold_store("gaussian", TWO_SAMPLES, min_umi=5)
    record = _record("gaussian", TWO_SAMPLES, min_umi=5)
    keys = [("s1", "g3"), ("s2", "g3"), ("s1", "g1"), ("s2", "g1")]

    result = guide_thresholds(store, THRESHOLDS, guides=["g3", "g1"], show=False)

    try:
        expected = pd.concat(
            [_rows(record.table, *key) for key in keys], ignore_index=True
        )
        pd.testing.assert_frame_equal(result.tables["thresholds"], expected)
        pd.testing.assert_frame_equal(
            result.tables["histograms"],
            pd.concat([_rows(record.histograms, *key) for key in keys]).reset_index(
                drop=True
            ),
        )
        assert list(result.axes)[:4] == keys
        for key, row in zip(
            keys, result.tables["thresholds"].to_dict("records"), strict=True
        ):
            _check_panel(result, key, row)
            spec = result.axes[key].get_subplotspec()
            assert (spec.rowspan.start, spec.colspan.start) == (
                keys.index(key) // 2,
                keys.index(key) % 2,
            )
        # min_umi=5 raises the threshold of g1 in s1 from 2, and that of g3 in
        # s2 beyond its one count, whose posterior is a single point.
        assert expected["threshold"].tolist() == [0, 1, 2, 6]
        assert expected["call_threshold"].tolist() == [0, 5, 5, 6]
        (point,) = result.axes[(("s2", "g3"), "posterior")].get_lines()
        assert point.get_marker() == "o"
        assert result.axes[("s2", "g3")].get_xlim()[1] >= 10
        (legend,) = result.figure.legends
        assert [text.get_text() for text in legend.get_texts()][-1] == (
            "Threshold raised to min_umi"
        )
        assert result.provenance.n_cells == 900
        assert result.provenance.n_samples == 2
        assert result.provenance.extras["samples"] == ["s1", "s2"]
    finally:
        result.close()


def test_threshold_panels_keep_the_chosen_samples_in_their_order() -> None:
    store = _threshold_store("gaussian", TWO_SAMPLES, min_umi=5)

    result = guide_thresholds(
        store, THRESHOLDS, guides=["g1"], samples=["s2", "s1"], show=False
    )
    single = guide_thresholds(
        store, THRESHOLDS, guides=["g1"], samples=["s2"], show=False
    )

    try:
        assert list(result.axes)[:2] == [("s2", "g1"), ("s1", "g1")]
        assert result.tables["thresholds"]["sample"].tolist() == ["s2", "s1"]
        assert list(single.axes) == [("s2", "g1"), (("s2", "g1"), "posterior")]
        assert single.tables["histograms"]["sample"].unique().tolist() == ["s2"]
        assert single.provenance.n_cells == 400
        assert single.provenance.n_samples == 1
    finally:
        result.close()
        single.close()


@pytest.mark.parametrize(("case", "samples"), [(ONE, None), (TWO, None), (TWO, ["s2"])])
def test_the_threshold_overview_draws_every_fit(
    case: tuple[str, dict[str | None, dict[str, tuple[Any, ...]]]],
    samples: list[str] | None,
) -> None:
    record = _record(*case, min_umi=5)
    kept = samples or list(case[1])

    result = guide_thresholds(
        _threshold_store(*case, min_umi=5), THRESHOLDS, samples=samples, show=False
    )

    try:
        table = record.table[record.table["sample"].isin(kept)]
        table = table.reset_index(drop=True)
        pd.testing.assert_frame_equal(result.tables["thresholds"], table)
        pd.testing.assert_frame_equal(
            result.tables["histograms"],
            record.histograms[record.histograms["sample"].isin(kept)].reset_index(
                drop=True
            ),
        )
        assert set(result.tables) == {"thresholds", "histograms"}
        assert list(result.axes) == ["thresholds", "statuses"]
        ax = result.axes["thresholds"]
        fitted = table["status"] == "fitted"
        drawn = {
            tuple(collection.get_facecolor()[0][:3]): collection.get_offsets()
            for collection in ax.collections
        }
        for chosen, color in ((fitted, "#2b6cb0"), (~fitted, "#9e9e9e")):
            np.testing.assert_array_equal(
                drawn[to_rgba(color)[:3]],
                table.loc[chosen, ["n_umi_cells", "threshold"]].to_numpy(),
            )
        assert ax.get_xscale() == ax.get_yscale() == "function"
        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            "fitted",
            "other statuses",
        ]
        bars = result.axes["statuses"]
        fits_by_status = table["status"].value_counts()
        assert [patch.get_width() for patch in bars.patches] == [
            fits_by_status.get(status, 0) for status in FIT_STATUSES
        ]
        labels = {
            label.get_text(): label.get_position()[1]
            for label in bars.get_yticklabels()
        }
        assert sorted(labels, key=labels.__getitem__, reverse=True) == list(
            FIT_STATUSES
        )
        assert result.legends[0].label == "Status"
        assert result.provenance.extras["guides"] is None
        assert result.provenance.extras["rasterized"] is False
        assert not any(collection.get_rasterized() for collection in ax.collections)
    finally:
        result.close()


def test_the_threshold_overview_rasterizes_crowded_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The overview of one sample draws three fits.
    monkeypatch.setattr("cytearc.plotting.perturbation.DEFAULT_RASTERIZE_THRESHOLD", 2)
    store = _threshold_store("poisson_gaussian", ONE_SAMPLE, min_umi=None)

    overview = guide_thresholds(store, THRESHOLDS, show=False)
    panels = guide_thresholds(store, THRESHOLDS, guides=["g1"], show=False)

    try:
        collections = overview.axes["thresholds"].collections
        assert collections and all(c.get_rasterized() for c in collections)
        assert overview.provenance.extras["rasterized"] is True
        assert panels.provenance.extras["rasterized"] is False
    finally:
        overview.close()
        panels.close()


def test_threshold_panels_draw_into_caller_axes() -> None:
    figure, axes = plt.subplots(1, 2)
    store = _threshold_store("poisson_gaussian", ONE_SAMPLE, min_umi=None)

    try:
        result = guide_thresholds(
            store, THRESHOLDS, guides=["g2", "g1"], target=list(axes), show=False
        )
        assert result.owns_figure is False
        assert result.figure is figure
        assert result.figure.legends == []
        assert [text.get_text() for text in result.axes["g2"].get_legend().get_texts()][
            :2
        ] == ["Cells called", "Cells not called"]
        assert result.axes[("g1", "posterior")].get_legend() is None
    finally:
        plt.close(figure)


def test_threshold_panels_follow_the_theme_and_can_drop_the_legend() -> None:
    store = _threshold_store("gaussian", TWO_SAMPLES, min_umi=5)

    result = guide_thresholds(
        store, THRESHOLDS, guides=["g1"], theme="dark", show_legend=False, show=False
    )
    overview = guide_thresholds(store, THRESHOLDS, show_legend=False, show=False)

    try:
        assert result.figure.legends == []
        (line,) = result.axes[(("s1", "g1"), "posterior")].get_lines()
        assert to_rgba(line.get_color()) == to_rgba(foreground_color("dark"))
        assert overview.axes["thresholds"].get_legend() is None
    finally:
        result.close()
        overview.close()


def test_showing_the_threshold_plot_releases_its_figure() -> None:
    store = _threshold_store("poisson_gaussian", ONE_SAMPLE, min_umi=None)

    result = guide_thresholds(store, THRESHOLDS, guides=["g1"])

    assert result.tables["thresholds"]["guide"].tolist() == ["g1"]


def test_the_threshold_plot_takes_the_guides_index_of_a_record() -> None:
    store = _threshold_store("poisson_gaussian", ONE_SAMPLE, min_umi=None)

    for guides in (pd.Index(["g2", "g1"], name="guide"), np.asarray(["g2", "g1"])):
        result = guide_thresholds(store, THRESHOLDS, guides=guides, show=False)
        try:
            assert result.tables["thresholds"]["guide"].tolist() == ["g2", "g1"]
        finally:
            result.close()


@pytest.mark.parametrize(
    ("case", "arguments", "error", "message"),
    [
        (ONE, {"max_panels": 0}, ValueError, "^max_panels must be at least 1$"),
        (ONE, {"max_panels": True}, TypeError, "^max_panels must be an integer$"),
        (ONE, {"max_panels": 2.0}, TypeError, "^max_panels must be an integer$"),
        (
            ONE,
            {"guides": "g1"},
            TypeError,
            "guides must be a sequence of guide IDs or None",
        ),
        (
            ONE,
            {"guides": [1]},
            TypeError,
            "guides must be a sequence of guide IDs or None",
        ),
        (ONE, {"guides": []}, ValueError, "guides must name at least one"),
        # Bytes are not a sequence of labels, even without a byte.
        (
            ONE,
            {"guides": b""},
            TypeError,
            "^guides must be a sequence of guide IDs or None$",
        ),
        (
            TWO,
            {"samples": "s1"},
            TypeError,
            "samples must be a sequence of sample labels or None",
        ),
        (TWO, {"samples": []}, ValueError, "samples must name at least one"),
        (
            TWO,
            {"samples": ["s1", "s9", "s8"]},
            ValueError,
            "^The fits have no sample named 's9', 's8'$",
        ),
        (
            ONE,
            {"samples": ["s1"]},
            ValueError,
            "^The fits have no sample named 's1'$",
        ),
        (
            TWO,
            {"guides": ["g1", "g2", "g3"], "max_panels": 5},
            ValueError,
            r"guides and samples select 6 panels, more than max_panels=5",
        ),
    ],
)
def test_the_threshold_plot_refuses_settings_it_cannot_use(
    case: tuple[str, dict[str | None, dict[str, tuple[Any, ...]]]],
    arguments: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    store = _threshold_store(*case, min_umi=5)

    with pytest.raises(error, match=message):
        guide_thresholds(store, THRESHOLDS, show=False, **arguments)


@pytest.mark.parametrize("guides", [None, ["g1"]])
def test_the_threshold_plot_without_a_guide_draws_nothing(
    guides: list[str] | None,
) -> None:
    def load(ref: ArtifactRef, *, guides: list[str] | None = None) -> GuideThresholds:
        record = _record("poisson_gaussian", ONE_SAMPLE, min_umi=None)
        return GuideThresholds(
            table=record.table.iloc[:0],
            histograms=record.histograms.iloc[:0],
            guides=record.guides,
            method=record.method,
            sample_column=record.sample_column,
            min_umi=record.min_umi,
        )

    store = SimpleNamespace(perturbation=SimpleNamespace(load_guide_thresholds=load))
    figures = set(plt.get_fignums())

    with pytest.raises(
        ValueError, match="^The fits hold no guide, so there is nothing to plot$"
    ):
        guide_thresholds(store, THRESHOLDS, guides=guides, show=False)
    assert set(plt.get_fignums()) == figures


# The settings that the efficiency and E-distance plots refuse alike.
ROW_REFUSALS = [
    ({"max_rows": 0}, ValueError, "^max_rows must be at least 1$"),
    ({"max_rows": True}, TypeError, "^max_rows must be an integer$"),
    ({"max_rows": 2.0}, TypeError, "^max_rows must be an integer$"),
    ({"alpha": 0}, ValueError, "^alpha must be above 0 and below 1$"),
    ({"alpha": 1.0}, ValueError, "^alpha must be above 0 and below 1$"),
    ({"alpha": True}, TypeError, "^alpha must be a real number$"),
    ({"alpha": "0.05"}, TypeError, "^alpha must be a real number$"),
    ({"alpha": float("nan")}, ValueError, "^alpha must be finite$"),
]
EFFICIENCY = ArtifactRef(
    scope="assay", assay="RNA", kind="perturbation_efficiency", artifact_id="c" * 64
)
SIGNIFICANT = to_rgba("#2b6cb0")
NOT_SIGNIFICANT = to_rgba("#9e9e9e")

# Each row is a name, a status, the perturbed cells, the log2 fold change and
# the adjusted p-value. A guide's name starts with its target. Rows without a
# fold change or p-value hold NaN, as the record does, and a row without
# control expression keeps its p-value.
TARGET_ROWS = [
    ("MYC", "tested", 40, -2.5, 1e-6),
    ("TP53", "tested", 12, -np.inf, 2e-4),
    ("GATA1", "tested", 25, -0.4, 0.3),
    ("KLF1", "tested", 30, 1.2, 0.01),
    ("CD19", "no_feature", 18, np.nan, np.nan),
    ("PAX5", "no_cells", 0, np.nan, np.nan),
    ("SPI1", "no_control_expression", 9, np.nan, 1e-8),
]
GUIDE_ROWS = [
    ("MYC_1", "tested", 20, -2.0, 0.001),
    ("MYC_2", "tested", 15, -0.1, 0.8),
    ("GATA1_1", "tested", 8, -0.1, 0.004),
    ("TP53_1", "tested", 6, -np.inf, 0.02),
    ("TP53_2", "no_cells", 0, np.nan, np.nan),
    ("CD19_1", "no_feature", 4, np.nan, np.nan),
]
# A and B tie on their adjusted p-value; F has the smallest but was not tested.
MANY_ROWS = [
    ("A", "tested", 10, 0.5, 0.01),
    ("B", "tested", 30, -1.0, 0.01),
    ("C", "tested", 20, -3.0, 0.2),
    ("D", "tested", 5, -0.2, 0.001),
    ("E", "tested", 10, -np.inf, 0.5),
    ("F", "no_control_expression", 5, np.nan, 1e-9),
]


def _efficiency_record(
    rows: list[tuple[str, str, int, float, float]], *, by: str = "target"
) -> PerturbationEfficiency:
    """Build the record that ``load_efficiency`` returns for ``rows``.

    The control cells hold 4 UMIs of each tested gene on average, and the
    perturbed cells hold that times the fold change. A row without control
    expression holds UMIs only in its perturbed cells.
    """
    names = [row[0] for row in rows]
    targets = [name.split("_")[0] for name in names]
    status = np.asarray([row[1] for row in rows], dtype=object)
    fold = np.asarray([row[3] for row in rows], dtype=np.float64)
    adjusted = np.asarray([row[4] for row in rows], dtype=np.float64)
    unexpressed = status == "no_control_expression"
    control = np.where(status == "tested", 4.0, np.where(unexpressed, 0.0, np.nan))
    mean = np.where(unexpressed, 1.5, control * np.exp2(fold))
    columns: dict[str, Any] = {"guide": np.asarray(names)} if by == "guide" else {}
    columns.update(
        target=np.asarray(targets),
        feature=pd.Series(
            [
                None if row_status == "no_feature" else f"ENSG_{row_target}"
                for row_status, row_target in zip(status, targets, strict=True)
            ],
            dtype=object,
        ),
        status=status,
        n_cells=np.asarray([row[2] for row in rows], dtype=np.int64),
        n_control_cells=np.full(len(rows), 50, dtype=np.int64),
        mean_umi=mean,
        control_mean_umi=control,
        mean_expression=1.1 * mean,
        control_mean_expression=1.1 * control,
        remaining_fraction=np.exp2(fold),
        log2_fold_change=fold,
        p_value=adjusted / 2,
        adjusted_p_value=adjusted,
    )
    return PerturbationEfficiency(table=pd.DataFrame(columns), by=by, from_assay="RNA")


# The store's loader of each kind of record.
RECORD_LOADERS = {
    PerturbationEfficiency: "load_efficiency",
    PerturbationEDistance: "load_edistance",
    MixscapeResult: "load_mixscape",
    PerturbationAssociations: "load_associations",
    PerturbationClusters: "load_clusters",
}


def _record_store(
    record: Any, requests: list[ArtifactRef] | None = None
) -> SimpleNamespace:
    """A store whose loader returns ``record`` and notes each reference it reads."""

    def load(ref: ArtifactRef) -> Any:
        if requests is not None:
            requests.append(ref)
        return record

    loader = RECORD_LOADERS[type(record)]
    return SimpleNamespace(perturbation=SimpleNamespace(**{loader: load}))


def _row_labels(ax: Any) -> list[str]:
    """Return the row labels from the top of the plot to the bottom."""
    ticks = zip(ax.get_yticks(), ax.get_yticklabels(), strict=True)
    return [label.get_text() for _, label in sorted(ticks, key=lambda tick: -tick[0])]


def _check_efficiency_rows(result: Any, *, alpha: float) -> None:
    """Check that each row of the table is drawn at its height, colour and area."""
    ax = result.axes["efficiency"]
    table = result.tables["efficiency"]
    fold = table["log2_fold_change"].to_numpy()
    n_cells = table["n_cells"].to_numpy()
    no_counts = np.isneginf(fold)
    # The first row is drawn at the top.
    heights = np.arange(len(table))[::-1]
    groups = [
        (chosen, marker)
        for chosen, marker in ((~no_counts, "o"), (no_counts, "<"))
        if chosen.any()
    ]
    assert len(ax.collections) == len(groups)
    x = np.full(len(table), np.nan)
    colors = np.zeros((len(table), 4))
    sizes = np.zeros(len(table))
    for collection, (chosen, marker) in zip(ax.collections, groups, strict=True):
        style = MarkerStyle(marker)
        shape = style.get_path().transformed(style.get_transform())
        np.testing.assert_array_equal(
            collection.get_paths()[0].vertices, shape.vertices
        )
        offsets = np.asarray(collection.get_offsets())
        np.testing.assert_array_equal(offsets[:, 1], heights[chosen])
        x[chosen] = offsets[:, 0]
        colors[chosen] = collection.get_facecolors()
        sizes[chosen] = collection.get_sizes()
    np.testing.assert_array_equal(x[~no_counts], fold[~no_counts])
    left, right = ax.get_xlim()
    assert left < 0.0 < right
    assert ((left < x) & (x < right)).all()
    if no_counts.any():
        # The rows without counts share one position at the left edge.
        edge = x[no_counts]
        assert (edge == edge[0]).all()
        assert edge[0] < fold[~no_counts].min(initial=0.0)
        assert edge[0] - left < 0.05 * (right - left)
    significant = table["adjusted_p_value"].to_numpy() <= alpha
    expected_colors = np.asarray(
        [SIGNIFICANT if below else NOT_SIGNIFICANT for below in significant]
    ).reshape(-1, 4)
    np.testing.assert_array_equal(colors, expected_colors)
    # The areas rank as the perturbed cells do.
    np.testing.assert_array_equal(
        np.sign(np.subtract.outer(sizes, sizes)),
        np.sign(np.subtract.outer(n_cells, n_cells)),
    )
    names = table[result.provenance.extras["by"]].tolist()
    assert _row_labels(ax) == [
        f"{name} (no counts)" if missing else name
        for name, missing in zip(names, no_counts, strict=True)
    ]
    (line,) = ax.get_lines()
    assert list(line.get_xdata()) == [0.0, 0.0]


def test_efficiency_draws_each_tested_target_from_knockdown_to_activation() -> None:
    requests: list[ArtifactRef] = []
    record = _efficiency_record(TARGET_ROWS)

    result = efficiency(_record_store(record, requests), EFFICIENCY, show=False)

    try:
        assert requests == [EFFICIENCY]
        # TP53 has no counts, so its fold change of minus infinity is first.
        pd.testing.assert_frame_equal(
            result.tables["efficiency"],
            record.table.iloc[[1, 0, 2, 3]].reset_index(drop=True),
        )
        assert list(result.axes) == ["efficiency"]
        _check_efficiency_rows(result, alpha=0.05)
        ax = result.axes["efficiency"]
        assert _row_labels(ax) == ["TP53 (no counts)", "MYC", "GATA1", "KLF1"]
        assert ax.get_xlabel() == "Log2 fold change against control cells"
        assert ax.get_title().splitlines() == [
            "Perturbation efficiency by target",
            "Not plotted: 1 no_feature, 1 no_cells, 1 no_control_expression",
        ]
        (legend,) = result.figure.legends
        assert [text.get_text() for text in legend.get_texts()] == [
            "Significant at FDR 0.05",
            "Not significant",
            "No counts in the perturbed cells",
            "12 perturbed cells",
            "30 perturbed cells",
            "40 perturbed cells",
        ]
        handles = legend.legend_handles
        assert [handle.get_marker() for handle in handles] == ["o", "o", "<"] + [
            "o"
        ] * 3
        sizes = [handle.get_markersize() for handle in handles[3:]]
        assert sizes == sorted(sizes) and len(set(sizes)) == 3
        assert [spec.kind for spec in result.legends] == ["categorical", "size"]
        provenance = result.provenance
        assert provenance.assay == "RNA"
        # The perturbed cells of the four drawn targets and the control cells.
        assert provenance.n_cells == 12 + 40 + 25 + 30 + 50
        assert provenance.notes == ("perturbation_efficiency",)
        assert provenance.extras == {
            "efficiency": EFFICIENCY.to_dict(),
            "by": "target",
            "from_assay": "RNA",
            "max_rows": 40,
            "alpha": 0.05,
            "n_tested": 4,
            "statuses": {
                "tested": 4,
                "no_feature": 1,
                "no_cells": 1,
                "no_control_expression": 1,
            },
        }
    finally:
        result.close()


def test_efficiency_by_guide_draws_each_tested_guide() -> None:
    record = _efficiency_record(GUIDE_ROWS, by="guide")

    result = efficiency(
        _record_store(record), EFFICIENCY, alpha=np.float64(0.01), show=False
    )

    try:
        table = result.tables["efficiency"]
        # GATA1_1 and MYC_2 share a fold change, so their names order them.
        pd.testing.assert_frame_equal(
            table, record.table.iloc[[3, 0, 2, 1]].reset_index(drop=True)
        )
        assert list(table.columns)[:2] == ["guide", "target"]
        _check_efficiency_rows(result, alpha=0.01)
        ax = result.axes["efficiency"]
        assert _row_labels(ax) == [
            "TP53_1 (no counts)",
            "MYC_1",
            "GATA1_1",
            "MYC_2",
        ]
        assert ax.get_title().splitlines() == [
            "Perturbation efficiency by guide",
            "Not plotted: 1 no_feature, 1 no_cells",
        ]
        (legend,) = result.figure.legends
        assert [text.get_text() for text in legend.get_texts()] == [
            "Significant at FDR 0.01",
            "Not significant",
            "No counts in the perturbed cells",
            "6 perturbed cells",
            "15 perturbed cells",
            "20 perturbed cells",
        ]
        assert result.provenance.extras["by"] == "guide"
        assert result.provenance.extras["alpha"] == 0.01
        assert type(result.provenance.extras["alpha"]) is float
    finally:
        result.close()


def test_efficiency_keeps_the_rows_with_the_smallest_adjusted_p_values() -> None:
    record = _efficiency_record(MANY_ROWS)
    store = _record_store(record)

    kept = efficiency(store, EFFICIENCY, max_rows=np.int64(2), show=False)
    every = efficiency(store, EFFICIENCY, max_rows=5, show=False)

    try:
        # D has the smallest adjusted p-value of the tested rows, and A ties
        # with B but comes first by name.
        pd.testing.assert_frame_equal(
            kept.tables["efficiency"],
            record.table.iloc[[3, 0]].reset_index(drop=True),
        )
        _check_efficiency_rows(kept, alpha=0.05)
        assert kept.axes["efficiency"].get_title().splitlines() == [
            "Perturbation efficiency by target",
            "The 2 of 5 tested targets with the smallest adjusted p-values",
            "Not plotted: 1 no_control_expression",
        ]
        assert kept.provenance.extras["max_rows"] == 2
        assert type(kept.provenance.extras["max_rows"]) is int
        assert kept.provenance.extras["n_tested"] == 5
        assert every.tables["efficiency"]["target"].tolist() == [
            "E",
            "C",
            "B",
            "D",
            "A",
        ]
        _check_efficiency_rows(every, alpha=0.05)
        assert every.axes["efficiency"].get_title().splitlines() == [
            "Perturbation efficiency by target",
            "Not plotted: 1 no_control_expression",
        ]
    finally:
        kept.close()
        every.close()


def test_efficiency_without_a_finite_fold_change_still_shows_zero() -> None:
    lost = efficiency(
        _record_store(_efficiency_record([("A", "tested", 1, -np.inf, 0.5)])),
        EFFICIENCY,
        show=False,
    )
    try:
        _check_efficiency_rows(lost, alpha=0.05)
        (legend,) = lost.figure.legends
        assert [text.get_text() for text in legend.get_texts()] == [
            "Not significant",
            "No counts in the perturbed cells",
            "1 perturbed cell",
        ]
    finally:
        lost.close()


def test_efficiency_without_a_tested_row_draws_nothing() -> None:
    untested = _efficiency_record(
        [("B", "no_feature", 3, np.nan, np.nan), ("C", "no_cells", 0, np.nan, np.nan)]
    )
    figures = set(plt.get_fignums())

    with pytest.raises(
        ValueError,
        match=r"^No target was tested \(1 no_feature, 1 no_cells\), so there is "
        "no fold change to plot$",
    ):
        efficiency(_record_store(untested), EFFICIENCY, show=False)
    with pytest.raises(
        ValueError, match="^No guide was tested, so there is no fold change to plot$"
    ):
        efficiency(
            _record_store(_efficiency_record([], by="guide")),
            EFFICIENCY,
            show=False,
        )
    assert set(plt.get_fignums()) == figures


def test_efficiency_draws_into_caller_axes_in_the_theme() -> None:
    figure, ax = plt.subplots()
    store = _record_store(_efficiency_record(TARGET_ROWS))

    try:
        result = efficiency(store, EFFICIENCY, target=ax, theme="dark", show=False)
        assert result.owns_figure is False
        assert result.figure is figure
        assert result.axes == {"efficiency": ax}
        assert figure.legends == []
        assert [text.get_text() for text in ax.get_legend().get_texts()][:3] == [
            "Significant at FDR 0.05",
            "Not significant",
            "No counts in the perturbed cells",
        ]
        _check_efficiency_rows(result, alpha=0.05)
        (line,) = ax.get_lines()
        assert to_rgba(line.get_color()) == to_rgba(foreground_color("dark"))
    finally:
        plt.close(figure)


def test_efficiency_can_drop_the_legend() -> None:
    store = _record_store(_efficiency_record(TARGET_ROWS))

    plain = efficiency(store, EFFICIENCY, show_legend=False, show=False)
    legended = efficiency(store, EFFICIENCY, show=False)

    try:
        assert plain.figure.legends == []
        assert plain.axes["efficiency"].get_legend() is None
        # Only a figure with a legend leaves room for it at the right.
        assert plain.figure.get_size_inches()[0] < legended.figure.get_size_inches()[0]
    finally:
        plain.close()
        legended.close()


def test_showing_the_efficiency_plot_releases_its_figure() -> None:
    store = _record_store(_efficiency_record(TARGET_ROWS))

    result = efficiency(store, EFFICIENCY)

    assert result.tables["efficiency"]["target"].tolist() == [
        "TP53",
        "MYC",
        "GATA1",
        "KLF1",
    ]


@pytest.mark.parametrize(("arguments", "error", "message"), ROW_REFUSALS)
def test_the_efficiency_plot_refuses_settings_it_cannot_use(
    arguments: dict[str, Any], error: type[Exception], message: str
) -> None:
    requests: list[ArtifactRef] = []
    store = _record_store(_efficiency_record(TARGET_ROWS), requests)

    with pytest.raises(error, match=message):
        efficiency(store, EFFICIENCY, show=False, **arguments)
    # The settings are checked before the record is read.
    assert requests == []


EDISTANCE = ArtifactRef(
    scope="assay", assay="RNA", kind="perturbation_edistance", artifact_id="d" * 64
)

# Each row is a name, a status, the perturbed cells, the E-distance and the
# adjusted p-value. A guide's name starts with its target. A row with too few
# cells holds NaN, as the record does, and an E-distance can fall below 0.
EDISTANCE_TARGETS = [
    ("TP53", "tested", 12, -0.3, 0.9),
    ("MYC", "tested", 40, 6.5, 0.001),
    ("GATA1", "tested", 25, 2.0, 0.2),
    ("CD19", "too_few_cells", 1, np.nan, np.nan),
    ("KLF1", "tested", 30, 4.0, 0.01),
]
# MYC_2 and GATA1_1 share an E-distance, and MYC_2 comes first in the record.
EDISTANCE_GUIDES = [
    ("MYC_2", "tested", 15, 1.5, 0.04),
    ("TP53_1", "tested", 6, -0.2, 0.7),
    ("TP53_2", "too_few_cells", 1, np.nan, np.nan),
    ("MYC_1", "tested", 20, 5.0, 0.002),
    ("GATA1_1", "tested", 8, 1.5, 0.3),
    ("CD19_1", "too_few_cells", 0, np.nan, np.nan),
]
# B and A share an E-distance, and B comes first in the record.
EDISTANCE_MANY = [
    ("B", "tested", 30, 3.0, 0.01),
    ("A", "tested", 10, 3.0, 0.01),
    ("C", "tested", 20, 5.0, 0.2),
    ("D", "tested", 5, 0.5, 0.001),
    ("E", "tested", 10, -1.0, 0.5),
    ("F", "too_few_cells", 1, np.nan, np.nan),
]


def _edistance_record(
    rows: list[tuple[str, str, int, float, float]],
    *,
    by: str = "target",
    n_permutations: int = 1000,
) -> PerturbationEDistance:
    """Build the record that ``load_edistance`` returns for ``rows``.

    Every row is compared with the same 50 control cells, and the permutations
    drew from seed 7. Without permutations no row has a p-value.
    """
    names = [row[0] for row in rows]
    adjusted = np.asarray([row[4] for row in rows], dtype=np.float64)
    if not n_permutations:
        adjusted = np.full(len(rows), np.nan)
    columns: dict[str, Any] = {"guide": np.asarray(names)} if by == "guide" else {}
    columns.update(
        target=np.asarray([name.split("_")[0] for name in names]),
        status=np.asarray([row[1] for row in rows], dtype=object),
        n_cells=np.asarray([row[2] for row in rows], dtype=np.int64),
        n_control_cells=np.full(len(rows), 50, dtype=np.int64),
        edistance=np.asarray([row[3] for row in rows], dtype=np.float64),
        p_value=adjusted / 2,
        adjusted_p_value=adjusted,
    )
    return PerturbationEDistance(
        table=pd.DataFrame(columns),
        by=by,
        from_assay="RNA",
        n_permutations=n_permutations,
        random_seed=7,
    )


def _check_edistance_rows(result: Any, *, alpha: float) -> None:
    """Check that each row of the table is drawn at its height, colour and area."""
    ax = result.axes["edistance"]
    table = result.tables["edistance"]
    distances = table["edistance"].to_numpy()
    n_cells = table["n_cells"].to_numpy()
    adjusted = table["adjusted_p_value"].to_numpy()
    names = table[result.provenance.extras["by"]].tolist()
    assert _row_labels(ax) == names
    # The rows run from the largest E-distance down, ties broken by name.
    ranks = list(zip((-distances).tolist(), names, strict=True))
    assert ranks == sorted(ranks)
    if len(table):
        (points,) = ax.collections
        style = MarkerStyle("o")
        shape = style.get_path().transformed(style.get_transform())
        np.testing.assert_array_equal(points.get_paths()[0].vertices, shape.vertices)
        offsets = np.asarray(points.get_offsets())
        # The first row is drawn at the top.
        np.testing.assert_array_equal(offsets[:, 1], np.arange(len(table))[::-1])
        np.testing.assert_array_equal(offsets[:, 0], distances)
        # Every point is blue when no row has a p-value.
        significant = (
            adjusted <= alpha
            if np.isfinite(adjusted).any()
            else np.ones(len(table), dtype=bool)
        )
        colors = np.zeros((len(table), 4))
        colors[:] = points.get_facecolors()
        np.testing.assert_array_equal(
            colors,
            np.asarray(
                [SIGNIFICANT if below else NOT_SIGNIFICANT for below in significant]
            ),
        )
        # The areas rank as the perturbed cells do.
        sizes = points.get_sizes()
        np.testing.assert_array_equal(
            np.sign(np.subtract.outer(sizes, sizes)),
            np.sign(np.subtract.outer(n_cells, n_cells)),
        )
    else:
        assert len(ax.collections) == 0
    # The axis holds 0 and every E-distance with a margin of 5% of their range.
    low = distances.min(initial=0.0)
    high = distances.max(initial=0.0)
    if low == high:
        low, high = -1.0, 1.0
    margin = 0.05 * (high - low)
    assert ax.get_xlim() == pytest.approx((low - margin, high + margin))
    (line,) = ax.get_lines()
    assert list(line.get_xdata()) == [0.0, 0.0]


def test_edistance_draws_each_tested_target_from_the_largest_distance_down() -> None:
    requests: list[ArtifactRef] = []
    record = _edistance_record(EDISTANCE_TARGETS)

    result = edistance(_record_store(record, requests), EDISTANCE, show=False)

    try:
        assert requests == [EDISTANCE]
        # TP53 looks like the control cells, so its E-distance falls below 0.
        pd.testing.assert_frame_equal(
            result.tables["edistance"],
            record.table.iloc[[1, 4, 2, 0]].reset_index(drop=True),
        )
        assert list(result.axes) == ["edistance"]
        _check_edistance_rows(result, alpha=0.05)
        ax = result.axes["edistance"]
        assert _row_labels(ax) == ["MYC", "KLF1", "GATA1", "TP53"]
        assert ax.get_xlabel() == "E-distance to control cells"
        assert ax.get_xlim() == pytest.approx((-0.64, 6.84))
        assert ax.get_title().splitlines() == [
            "E-distance by target",
            "Not plotted: 1 too_few_cells",
        ]
        (legend,) = result.figure.legends
        assert [text.get_text() for text in legend.get_texts()] == [
            "Significant at FDR 0.05",
            "Not significant",
            "12 perturbed cells",
            "30 perturbed cells",
            "40 perturbed cells",
        ]
        handles = legend.legend_handles
        assert [handle.get_marker() for handle in handles] == ["o"] * 5
        sizes = [handle.get_markersize() for handle in handles[2:]]
        assert sizes == sorted(sizes) and len(set(sizes)) == 3
        assert result.legends == (
            LegendSpec(kind="categorical", label="Adjusted p-value"),
            LegendSpec(kind="size", label="Perturbed cells"),
        )
        provenance = result.provenance
        assert provenance.assay == "RNA"
        # The perturbed cells of the four drawn targets and the control cells.
        assert provenance.n_cells == 40 + 30 + 25 + 12 + 50
        assert provenance.notes == ("perturbation_edistance",)
        assert provenance.extras == {
            "edistance": EDISTANCE.to_dict(),
            "by": "target",
            "n_permutations": 1000,
            "random_seed": 7,
            "max_rows": 40,
            "alpha": 0.05,
            "n_tested": 4,
            "statuses": {"tested": 4, "too_few_cells": 1},
        }
    finally:
        result.close()


def test_edistance_by_guide_draws_each_tested_guide() -> None:
    record = _edistance_record(EDISTANCE_GUIDES, by="guide")

    result = edistance(
        _record_store(record), EDISTANCE, alpha=np.float64(0.01), show=False
    )

    try:
        table = result.tables["edistance"]
        # GATA1_1 and MYC_2 share an E-distance, so their names order them.
        pd.testing.assert_frame_equal(
            table, record.table.iloc[[3, 4, 0, 1]].reset_index(drop=True)
        )
        assert list(table.columns)[:2] == ["guide", "target"]
        _check_edistance_rows(result, alpha=0.01)
        ax = result.axes["edistance"]
        assert _row_labels(ax) == ["MYC_1", "GATA1_1", "MYC_2", "TP53_1"]
        assert ax.get_title().splitlines() == [
            "E-distance by guide",
            "Not plotted: 2 too_few_cells",
        ]
        (legend,) = result.figure.legends
        assert [text.get_text() for text in legend.get_texts()] == [
            "Significant at FDR 0.01",
            "Not significant",
            "6 perturbed cells",
            "15 perturbed cells",
            "20 perturbed cells",
        ]
        assert result.provenance.extras["by"] == "guide"
        assert result.provenance.extras["alpha"] == 0.01
        assert type(result.provenance.extras["alpha"]) is float
        assert result.provenance.extras["statuses"] == {
            "tested": 4,
            "too_few_cells": 2,
        }
    finally:
        result.close()


def test_edistance_keeps_the_rows_with_the_largest_e_distances() -> None:
    record = _edistance_record(EDISTANCE_MANY)
    store = _record_store(record)

    kept = edistance(store, EDISTANCE, max_rows=np.int64(2), show=False)
    every = edistance(store, EDISTANCE, max_rows=5, show=False)

    try:
        # C has the largest E-distance, and A ties with B but comes first by
        # name.
        pd.testing.assert_frame_equal(
            kept.tables["edistance"],
            record.table.iloc[[2, 1]].reset_index(drop=True),
        )
        _check_edistance_rows(kept, alpha=0.05)
        assert kept.axes["edistance"].get_title().splitlines() == [
            "E-distance by target",
            "The 2 of 5 tested targets with the largest E-distances",
            "Not plotted: 1 too_few_cells",
        ]
        assert kept.provenance.extras["max_rows"] == 2
        assert type(kept.provenance.extras["max_rows"]) is int
        assert kept.provenance.extras["n_tested"] == 5
        # The perturbed cells of C and A and the control cells.
        assert kept.provenance.n_cells == 20 + 10 + 50
        assert every.tables["edistance"]["target"].tolist() == [
            "C",
            "A",
            "B",
            "D",
            "E",
        ]
        _check_edistance_rows(every, alpha=0.05)
        assert every.axes["edistance"].get_title().splitlines() == [
            "E-distance by target",
            "Not plotted: 1 too_few_cells",
        ]
    finally:
        kept.close()
        every.close()


def test_edistance_without_permutation_tests_draws_every_point_blue() -> None:
    record = _edistance_record(EDISTANCE_TARGETS, n_permutations=0)
    store = _record_store(record)

    every = edistance(store, EDISTANCE, show=False)
    kept = edistance(store, EDISTANCE, max_rows=3, show=False)

    try:
        assert record.table[["p_value", "adjusted_p_value"]].isna().all().all()
        _check_edistance_rows(every, alpha=0.05)
        (points,) = every.axes["edistance"].collections
        np.testing.assert_array_equal(
            points.get_facecolors(), np.tile(SIGNIFICANT, (4, 1))
        )
        assert every.axes["edistance"].get_title().splitlines() == [
            "E-distance by target",
            "Without permutation tests",
            "Not plotted: 1 too_few_cells",
        ]
        (legend,) = every.figure.legends
        # The legend gives only the point areas.
        assert [text.get_text() for text in legend.get_texts()] == [
            "12 perturbed cells",
            "30 perturbed cells",
            "40 perturbed cells",
        ]
        assert every.provenance.extras["n_permutations"] == 0
        _check_edistance_rows(kept, alpha=0.05)
        assert _row_labels(kept.axes["edistance"]) == ["MYC", "KLF1", "GATA1"]
        assert kept.axes["edistance"].get_title().splitlines() == [
            "E-distance by target",
            "The 3 of 4 tested targets with the largest E-distances",
            "Without permutation tests",
            "Not plotted: 1 too_few_cells",
        ]
    finally:
        every.close()
        kept.close()


def test_edistance_without_a_tested_row_draws_nothing() -> None:
    record = _edistance_record(
        [
            ("A", "too_few_cells", 1, np.nan, np.nan),
            ("B", "too_few_cells", 0, np.nan, np.nan),
        ]
    )
    figures = set(plt.get_fignums())

    with pytest.raises(
        ValueError,
        match=r"^No target was tested \(2 too_few_cells\), so there is no "
        "E-distance to plot$",
    ):
        edistance(_record_store(record), EDISTANCE, show=False)
    assert set(plt.get_fignums()) == figures


def test_edistance_without_a_nonzero_distance_still_shows_zero() -> None:
    zero = edistance(
        _record_store(_edistance_record([("C", "tested", 4, 0.0, 1.0)])),
        EDISTANCE,
        show=False,
    )

    try:
        _check_edistance_rows(zero, alpha=0.05)
        assert zero.axes["edistance"].get_xlim() == pytest.approx((-1.1, 1.1))
        assert zero.axes["edistance"].get_title() == "E-distance by target"
        (legend,) = zero.figure.legends
        assert [text.get_text() for text in legend.get_texts()] == [
            "Not significant",
            "4 perturbed cells",
        ]
    finally:
        zero.close()


def test_edistance_draws_into_caller_axes_in_the_theme() -> None:
    figure, ax = plt.subplots()
    store = _record_store(_edistance_record(EDISTANCE_TARGETS))

    try:
        result = edistance(store, EDISTANCE, target=ax, theme="dark", show=False)
        assert result.owns_figure is False
        assert result.figure is figure
        assert result.axes == {"edistance": ax}
        assert figure.legends == []
        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            "Significant at FDR 0.05",
            "Not significant",
            "12 perturbed cells",
            "30 perturbed cells",
            "40 perturbed cells",
        ]
        _check_edistance_rows(result, alpha=0.05)
        (line,) = ax.get_lines()
        assert to_rgba(line.get_color()) == to_rgba(foreground_color("dark"))
    finally:
        plt.close(figure)


def test_edistance_can_drop_the_legend() -> None:
    store = _record_store(_edistance_record(EDISTANCE_TARGETS))

    plain = edistance(store, EDISTANCE, show_legend=False, show=False)
    legended = edistance(store, EDISTANCE, show=False)

    try:
        assert plain.figure.legends == []
        assert plain.axes["edistance"].get_legend() is None
        # Only a figure with a legend leaves room for it at the right.
        assert plain.figure.get_size_inches()[0] < legended.figure.get_size_inches()[0]
    finally:
        plain.close()
        legended.close()


def test_showing_the_edistance_plot_releases_its_figure() -> None:
    store = _record_store(_edistance_record(EDISTANCE_TARGETS))

    result = edistance(store, EDISTANCE)

    assert result.tables["edistance"]["target"].tolist() == [
        "MYC",
        "KLF1",
        "GATA1",
        "TP53",
    ]


@pytest.mark.parametrize(("arguments", "error", "message"), ROW_REFUSALS)
def test_the_edistance_plot_refuses_settings_it_cannot_use(
    arguments: dict[str, Any], error: type[Exception], message: str
) -> None:
    requests: list[ArtifactRef] = []
    store = _record_store(_edistance_record(EDISTANCE_TARGETS), requests)

    with pytest.raises(error, match=message):
        edistance(store, EDISTANCE, show=False, **arguments)
    # The settings are checked before the record is read.
    assert requests == []


MIXSCAPE = ArtifactRef(
    scope="assay", assay="RNA", kind="mixscape_classes", artifact_id="e" * 64
)
KNOCKED_OUT = to_rgba("#2b6cb0")
ESCAPED = to_rgba("#9e9e9e")
PERTURBED_CURVE = to_rgba("#d55e00")
MIXSCAPE_LEGEND = [
    "KO (perturbed)",
    "NP (not perturbed)",
    "Fitted control component",
    "Fitted perturbed component",
]
MixscapeRow = tuple[
    str, str | None, str, list[float], list[float], tuple[float, ...] | None
]

# Each row is a target, its split, its status, the scores of its KO cells and
# of its NP cells, and its fit: the mean and standard deviation of the control
# component, then of the perturbed component. A row without a fit has only NP
# cells, without scores.
MIXSCAPE_TARGETS: list[MixscapeRow] = [
    (
        "MYC",
        None,
        "fitted",
        [0.83, 0.91, 1.02, 1.04, 1.12, 1.29],
        [-0.23, 0.04, 0.11, 0.37],
        (0.0, 0.2, 1.0, 0.15),
    ),
    (
        "TP53",
        None,
        "fitted",
        [0.62, 1.18],
        [-0.12, 0.02, 0.03, 0.21, 0.43, 0.16],
        (0.05, 0.25, 0.9, 0.3),
    ),
    ("GATA1", None, "few_de_genes", [], [np.nan] * 5, None),
    ("KLF1", None, "fitted", [1.52, 2.03, 1.81], [0.22], (0.1, 0.3, 1.7, 0.2)),
    ("CD19", None, "few_cells", [], [np.nan] * 2, None),
]
# B and A share their cells, B and C their KO share, and E and D theirs.
MIXSCAPE_MANY: list[MixscapeRow] = [
    ("B", None, "fitted", [1.01, 1.13], [0.02, 0.12], (0.0, 0.2, 1.0, 0.2)),
    ("A", None, "fitted", [0.97], [0.01, 0.13, -0.11], (0.0, 0.2, 1.0, 0.2)),
    (
        "C",
        None,
        "fitted",
        [0.93, 1.04, 1.16],
        [0.03, 0.22, 0.14],
        (0.0, 0.2, 1.0, 0.2),
    ),
    ("E", None, "no_fit", [], [np.nan] * 7, None),
    ("D", None, "few_de_genes", [], [np.nan] * 3, None),
]
# MYC is fitted in run1 only, TP53 in neither run, with a different status in
# each, and KLF1 in both, with different fits.
MIXSCAPE_SPLITS: list[MixscapeRow] = [
    ("MYC", "run1", "fitted", [0.92, 1.03, 1.21], [0.02, 0.13], (0.0, 0.2, 1.0, 0.2)),
    ("MYC", "run2", "few_cells", [], [np.nan] * 3, None),
    ("TP53", "run1", "few_de_genes", [], [np.nan] * 2, None),
    ("TP53", "run2", "few_cells", [], [np.nan] * 2, None),
    ("KLF1", "run1", "fitted", [1.41, 1.62], [0.11, 0.24], (0.1, 0.3, 1.5, 0.2)),
    (
        "KLF1",
        "run2",
        "fitted",
        [1.12, 1.33, 1.24],
        [-0.07],
        (0.0, 0.25, 1.2, 0.15),
    ),
]


def _mixscape_record(
    rows: list[MixscapeRow],
    *,
    perturbation_type: str = "KO",
    split_column: str | None = None,
) -> MixscapeResult:
    """Build the record that ``load_mixscape`` returns for ``rows``.

    The cells of each row come in its order, KO cells first, and then three
    control cells and a cell with two targets. A cell with a score has the
    posterior 0.9 when it is KO and 0.1 when it is NP.
    """
    cells: list[tuple[str, str, float, float, str | None]] = []
    targets: list[dict[str, Any]] = []
    for name, split, status, ko_scores, np_scores, fit in rows:
        # Keep the test data honest: only a fitted row has KO cells or scores.
        assert (fit is not None) == (status == "fitted")
        assert fit is not None or (not ko_scores and np.isnan(np_scores).all())
        for scores, word, chance in (
            (ko_scores, perturbation_type, 0.9),
            (np_scores, "NP", 0.1),
        ):
            cells.extend(
                (
                    f"{name} {word}",
                    word,
                    np.nan if fit is None else chance,
                    score,
                    split,
                )
                for score in scores
            )
        control_mean, control_sd, perturbed_mean, perturbed_sd = fit or (np.nan,) * 4
        targets.append(
            {
                "target": name,
                "split": split,
                "status": status,
                "n_cells": len(ko_scores) + len(np_scores),
                "n_ko": len(ko_scores),
                "n_de_genes": 2 if status == "few_de_genes" else 12,
                "iterations": 0 if status == "few_de_genes" else 3,
                "converged": fit is not None,
                "control_mean": control_mean,
                "control_sd": control_sd,
                "perturbed_mean": perturbed_mean,
                "perturbed_sd": perturbed_sd,
                "perturbed_weight": np.nan if fit is None else 0.4,
            }
        )
    split = rows[0][1] if rows else None
    cells.extend(
        (label, label, np.nan, np.nan, split)
        for label in ["control"] * 3 + ["Multiple"]
    )
    frame = pd.DataFrame(
        cells,
        columns=["mixscape_class", "global_class", "posterior", "score", "split"],
        index=pd.Index([f"cell{index}" for index in range(len(cells))], name="ids"),
    )
    table = pd.DataFrame(
        targets,
        columns=[
            "target",
            "split",
            "status",
            "n_cells",
            "n_ko",
            "n_de_genes",
            "iterations",
            "converged",
            "control_mean",
            "control_sd",
            "perturbed_mean",
            "perturbed_sd",
            "perturbed_weight",
        ],
    ).astype(
        {
            "n_cells": np.int64,
            "n_ko": np.int64,
            "n_de_genes": np.int64,
            "iterations": np.int64,
            "converged": bool,
        }
    )
    return MixscapeResult(
        cells=frame,
        targets=table,
        de_genes=pd.DataFrame({"target": [], "split": [], "feature": []}),
        from_assay="RNA",
        perturbation_type=perturbation_type,
        split_column=split_column,
    )


def _drawn_cells(record: MixscapeResult, names: list[str]) -> pd.DataFrame:
    """Return the cells of ``names`` that have scores, target by target."""
    cells = record.cells
    frames = []
    for name in names:
        classes = [f"{name} {record.perturbation_type}", f"{name} NP"]
        chosen = cells["mixscape_class"].isin(classes) & cells["score"].notna()
        frames.append(cells[chosen].assign(target=name))
    return pd.concat(frames)[["target", "split", "score", "global_class", "posterior"]]


def _check_class_bars(result: Any) -> None:
    """Check that each drawn target's bar splits at its KO share, first on top."""
    ax = result.axes["classes"]
    table = result.tables["classes"]
    assert list(table.columns) == ["target", "n_cells", "n_ko", "n_np", "ko_share"]
    assert (table["n_np"] == table["n_cells"] - table["n_ko"]).all()
    shares = table["ko_share"].to_numpy()
    np.testing.assert_allclose(shares, table["n_ko"] / table["n_cells"])
    n_rows = len(table)
    assert len(ax.patches) == 2 * n_rows
    heights = np.arange(n_rows)[::-1]
    for bars, left, width, color in (
        (ax.patches[:n_rows], np.zeros(n_rows), shares, KNOCKED_OUT),
        (ax.patches[n_rows:], shares, 1.0 - shares, ESCAPED),
    ):
        np.testing.assert_allclose([bar.get_x() for bar in bars], left)
        np.testing.assert_allclose([bar.get_width() for bar in bars], width)
        np.testing.assert_allclose(
            [bar.get_y() + bar.get_height() / 2 for bar in bars], heights
        )
        assert [bar.get_facecolor() for bar in bars] == [color] * n_rows
    # Each bar fills the axis, and its label names the target and its cells.
    assert ax.get_xlim() == (0.0, 1.0)
    assert ax.xaxis.get_major_formatter().format_ticks(ax.get_xticks()) == [
        "0%",
        "25%",
        "50%",
        "75%",
        "100%",
    ]
    assert ax.get_xlabel() == "Share of the target's cells"
    assert _row_labels(ax) == [
        f"{name} ({count:,} cell{'' if count == 1 else 's'})"
        for name, count in zip(table["target"], table["n_cells"], strict=True)
    ]


def _check_score_panel(
    result: Any, name: str, fit: tuple[float, ...], *, theme: str = "notebook"
) -> None:
    """Check that a panel draws its target's scores and the normals of ``fit``."""
    ax = result.axes[name]
    table = result.tables["scores"]
    cells = table[table["target"] == name]
    scores = cells["score"].to_numpy()
    ko = (
        cells["global_class"] == result.provenance.extras["perturbation_type"]
    ).to_numpy()
    # 30 bins of equal width span the scores.
    low, high = scores.min(), scores.max()
    width = (high - low) / 30
    edges = low + width * np.arange(31)
    # Keep the test data honest: no score sits on an inner edge.
    assert np.abs(scores[:, None] - edges[1:-1]).min() > 1e-6
    bins = np.minimum(np.floor((scores - low) / width).astype(int), 29)
    ko_bars = [bar for bar in ax.patches if bar.get_facecolor() == KNOCKED_OUT]
    np_bars = [bar for bar in ax.patches if bar.get_facecolor() == ESCAPED]
    assert len(ko_bars) + len(np_bars) == len(ax.patches)
    for bars, chosen in ((ko_bars, ko), (np_bars, ~ko)):
        if not chosen.any():
            assert bars == []
            continue
        np.testing.assert_allclose([bar.get_x() for bar in bars], edges[:-1])
        np.testing.assert_allclose([bar.get_width() for bar in bars], width)
        # Each bar counts the cells of its class whose scores fall in its bin.
        assert [bar.get_height() for bar in bars] == np.bincount(
            bins[chosen], minlength=30
        ).tolist()
    # The NP bars stand on the KO bars.
    assert [bar.get_y() for bar in ko_bars] == [0.0] * len(ko_bars)
    if ko_bars and np_bars:
        assert [bar.get_y() for bar in np_bars] == [bar.get_height() for bar in ko_bars]
    assert (np.asarray(ax.get_yticks()) % 1 == 0).all()
    control_mean, control_sd, perturbed_mean, perturbed_sd = fit
    reach = [
        low,
        high,
        control_mean - 3 * control_sd,
        control_mean + 3 * control_sd,
        perturbed_mean - 3 * perturbed_sd,
        perturbed_mean + 3 * perturbed_sd,
    ]
    control, perturbed = ax.get_lines()
    assert to_rgba(control.get_color()) == to_rgba(foreground_color(theme))
    assert to_rgba(perturbed.get_color()) == PERTURBED_CURVE
    for line, mean, sd, n_class in (
        (control, control_mean, control_sd, np.count_nonzero(~ko)),
        (perturbed, perturbed_mean, perturbed_sd, np.count_nonzero(ko)),
    ):
        values = line.get_xdata()
        density = line.get_ydata()
        np.testing.assert_allclose(values, np.linspace(min(reach), max(reach), 200))
        np.testing.assert_allclose(
            density, n_class * width * norm.pdf(values, mean, sd)
        )
        # Each curve covers as many cells as the bars of its class.
        assert np.trapezoid(density, values) / width == pytest.approx(n_class, rel=0.01)
    assert ax.get_xlabel() == "Perturbation score"
    assert ax.get_ylabel() == "Cells"


def test_mixscape_classes_draw_each_target_from_the_largest_ko_share_down() -> None:
    requests: list[ArtifactRef] = []
    record = _mixscape_record(MIXSCAPE_TARGETS)

    result = mixscape(_record_store(record, requests), MIXSCAPE, show=False)

    try:
        assert requests == [MIXSCAPE]
        assert list(result.tables) == ["classes"]
        table = result.tables["classes"]
        assert table["target"].tolist() == ["KLF1", "MYC", "TP53", "CD19", "GATA1"]
        assert table["n_cells"].tolist() == [4, 10, 8, 2, 5]
        assert table["n_ko"].tolist() == [3, 6, 2, 0, 0]
        assert table["ko_share"].tolist() == pytest.approx([0.75, 0.6, 0.25, 0, 0])
        assert list(result.axes) == ["classes"]
        _check_class_bars(result)
        ax = result.axes["classes"]
        # CD19 and GATA1 were not fitted, so all their cells are NP, and their
        # names order them.
        assert _row_labels(ax) == [
            "KLF1 (4 cells)",
            "MYC (10 cells)",
            "TP53 (8 cells)",
            "CD19 (2 cells)",
            "GATA1 (5 cells)",
        ]
        assert ax.get_title().splitlines() == [
            "Mixscape classes by target",
            "Not fitted, so all NP: 1 few_de_genes, 1 few_cells",
        ]
        (legend,) = result.figure.legends
        assert [text.get_text() for text in legend.get_texts()] == MIXSCAPE_LEGEND[:2]
        assert result.legends == (LegendSpec(kind="categorical", label="Class"),)
        provenance = result.provenance
        assert provenance.assay == "RNA"
        # The cells of the drawn targets.
        assert provenance.n_cells == 4 + 10 + 8 + 2 + 5
        assert provenance.notes == ("mixscape_classes",)
        assert provenance.extras == {
            "classes": MIXSCAPE.to_dict(),
            "panel": "classes",
            "perturbation_type": "KO",
            "split_column": None,
            "targets": None,
            "max_targets": 40,
            "statuses": {"fitted": 3, "few_de_genes": 1, "few_cells": 1, "no_fit": 0},
        }
    finally:
        result.close()


def test_mixscape_classes_keep_the_targets_with_the_most_cells() -> None:
    store = _record_store(_mixscape_record(MIXSCAPE_MANY))

    kept = mixscape(store, MIXSCAPE, max_targets=np.int64(3), show=False)
    every = mixscape(store, MIXSCAPE, max_targets=5, show=False)

    try:
        # E and C have the most cells, and A ties with B but comes first by
        # name.
        assert kept.tables["classes"]["target"].tolist() == ["C", "A", "E"]
        _check_class_bars(kept)
        assert kept.axes["classes"].get_title().splitlines() == [
            "Mixscape classes by target",
            "The 3 of 5 targets with the most cells",
            "Not fitted, so all NP: 1 no_fit",
        ]
        assert kept.provenance.extras["max_targets"] == 3
        assert type(kept.provenance.extras["max_targets"]) is int
        assert kept.provenance.n_cells == 6 + 4 + 7
        # B and C share a KO share, and so do D and E, so names order them.
        assert every.tables["classes"]["target"].tolist() == ["B", "C", "A", "D", "E"]
        _check_class_bars(every)
        assert every.axes["classes"].get_title().splitlines() == [
            "Mixscape classes by target",
            "Not fitted, so all NP: 1 few_de_genes, 1 no_fit",
        ]
    finally:
        kept.close()
        every.close()


def test_mixscape_classes_keep_the_chosen_targets_in_their_order() -> None:
    store = _record_store(_mixscape_record(MIXSCAPE_TARGETS))

    result = mixscape(
        store,
        MIXSCAPE,
        targets=["GATA1", "MYC", "KLF1", "MYC"],
        max_targets=3,
        show=False,
    )

    try:
        assert result.tables["classes"]["target"].tolist() == ["GATA1", "MYC", "KLF1"]
        _check_class_bars(result)
        assert result.axes["classes"].get_title().splitlines() == [
            "Mixscape classes by target",
            "Not fitted, so all NP: 1 few_de_genes",
        ]
        assert result.provenance.extras["targets"] == ["GATA1", "MYC", "KLF1"]
        assert result.provenance.n_cells == 5 + 10 + 4
    finally:
        result.close()


def test_mixscape_classes_pool_the_splits_of_each_target() -> None:
    record = _mixscape_record(
        MIXSCAPE_SPLITS, perturbation_type="KD", split_column="run"
    )

    result = mixscape(_record_store(record), MIXSCAPE, show=False)

    try:
        table = result.tables["classes"]
        assert table["target"].tolist() == ["KLF1", "MYC", "TP53"]
        assert table["n_cells"].tolist() == [8, 8, 4]
        assert table["n_ko"].tolist() == [5, 3, 0]
        _check_class_bars(result)
        # MYC was fitted in run1, so only TP53 counts, under both its statuses.
        assert result.axes["classes"].get_title().splitlines() == [
            "Mixscape classes by target",
            "Not fitted, so all NP: 1 few_de_genes, 1 few_cells",
        ]
        (legend,) = result.figure.legends
        assert [text.get_text() for text in legend.get_texts()] == [
            "KD (perturbed)",
            "NP (not perturbed)",
        ]
        extras = result.provenance.extras
        assert extras["perturbation_type"] == "KD"
        assert extras["split_column"] == "run"
        assert extras["statuses"] == {
            "fitted": 3,
            "few_de_genes": 1,
            "few_cells": 2,
            "no_fit": 0,
        }
    finally:
        result.close()


@pytest.mark.parametrize("panel", ["classes", "scores"])
def test_mixscape_classes_without_a_target_draw_nothing(panel: str) -> None:
    figures = set(plt.get_fignums())

    with pytest.raises(
        ValueError,
        match="^The Mixscape classes hold no target, so there is nothing to plot$",
    ):
        mixscape(_record_store(_mixscape_record([])), MIXSCAPE, panel=panel, show=False)
    assert set(plt.get_fignums()) == figures


def test_mixscape_classes_without_a_fitted_target_still_draw() -> None:
    unfitted = mixscape(
        _record_store(_mixscape_record(MIXSCAPE_MANY[3:])), MIXSCAPE, show=False
    )

    try:
        # Every cell is NP, so the legend names only NP.
        _check_class_bars(unfitted)
        (legend,) = unfitted.figure.legends
        assert [text.get_text() for text in legend.get_texts()] == [
            "NP (not perturbed)"
        ]
    finally:
        unfitted.close()


def test_mixscape_classes_draw_into_caller_axes_in_the_theme() -> None:
    figure, ax = plt.subplots()
    store = _record_store(_mixscape_record(MIXSCAPE_TARGETS))

    try:
        result = mixscape(store, MIXSCAPE, target=ax, theme="dark", show=False)
        assert result.owns_figure is False
        assert result.figure is figure
        assert result.axes == {"classes": ax}
        assert figure.legends == []
        legend = ax.get_legend()
        assert [text.get_text() for text in legend.get_texts()] == MIXSCAPE_LEGEND[:2]
        # Every bar spans the axes, so the legend stands beside them.
        figure.canvas.draw()
        assert legend.get_window_extent().x0 > ax.get_window_extent().x1
        _check_class_bars(result)
        assert to_rgba(ax.title.get_color()) == to_rgba(foreground_color("dark"))
    finally:
        plt.close(figure)


def test_mixscape_scores_draw_one_panel_per_fitted_target() -> None:
    requests: list[ArtifactRef] = []
    record = _mixscape_record(MIXSCAPE_TARGETS)

    result = mixscape(
        _record_store(record, requests), MIXSCAPE, panel="scores", show=False
    )

    try:
        assert requests == [MIXSCAPE]
        # The fitted targets, from the largest KO share to the smallest.
        names = ["KLF1", "MYC", "TP53"]
        assert list(result.axes) == names
        assert list(result.tables) == ["scores"]
        pd.testing.assert_frame_equal(
            result.tables["scores"], _drawn_cells(record, names)
        )
        fits = {row[0]: row[5] for row in MIXSCAPE_TARGETS}
        for name in names:
            _check_score_panel(result, name, fits[name])
            assert result.axes[name].get_title() == name
        assert result.figure.get_suptitle().splitlines() == [
            "Mixscape scores by target",
            "Not fitted, so not drawn: 1 few_de_genes, 1 few_cells",
        ]
        (legend,) = result.figure.legends
        assert [text.get_text() for text in legend.get_texts()] == MIXSCAPE_LEGEND
        assert result.legends == (
            LegendSpec(kind="categorical", label="Class"),
            LegendSpec(kind="line", label="Fitted component"),
        )
        provenance = result.provenance
        # The drawn cells.
        assert provenance.n_cells == 4 + 10 + 8
        assert provenance.notes == ("mixscape_classes",)
        assert provenance.extras == {
            "classes": MIXSCAPE.to_dict(),
            "panel": "scores",
            "perturbation_type": "KO",
            "split_column": None,
            "targets": None,
            "max_targets": 40,
            "statuses": {"fitted": 3, "few_de_genes": 1, "few_cells": 1, "no_fit": 0},
        }
    finally:
        result.close()


def test_mixscape_scores_keep_the_fitted_targets_with_the_most_cells() -> None:
    record = _mixscape_record(MIXSCAPE_MANY)

    result = mixscape(
        _record_store(record), MIXSCAPE, panel="scores", max_targets=2, show=False
    )

    try:
        # C has the most cells of the fitted targets, and A ties with B but
        # comes first by name.
        assert list(result.axes) == ["C", "A"]
        pd.testing.assert_frame_equal(
            result.tables["scores"], _drawn_cells(record, ["C", "A"])
        )
        for name in ["C", "A"]:
            _check_score_panel(result, name, (0.0, 0.2, 1.0, 0.2))
        assert result.figure.get_suptitle().splitlines() == [
            "Mixscape scores by target",
            "The 2 of 3 fitted targets with the most cells",
            "Not fitted, so not drawn: 1 few_de_genes, 1 no_fit",
        ]
        assert result.provenance.n_cells == 6 + 4
    finally:
        result.close()


def test_mixscape_scores_fill_rows_of_four_panels() -> None:
    rows: list[MixscapeRow] = [
        (
            f"T{index}",
            None,
            "fitted",
            [0.93 + 0.01 * index, 1.07],
            [0.02, 0.11],
            (0.0, 0.2, 1.0, 0.2),
        )
        for index in range(5)
    ]

    result = mixscape(
        _record_store(_mixscape_record(rows)), MIXSCAPE, panel="scores", show=False
    )

    try:
        # The targets share their cells and their KO share, so names order them.
        names = [f"T{index}" for index in range(5)]
        assert list(result.axes) == names
        positions = [
            (spec.rowspan.start, spec.colspan.start)
            for spec in (result.axes[name].get_subplotspec() for name in names)
        ]
        assert positions == [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)]
        assert len(result.figure.axes) == 5
        assert result.figure.get_suptitle() == "Mixscape scores by target"
    finally:
        result.close()


def test_mixscape_scores_skip_chosen_targets_without_a_fit() -> None:
    record = _mixscape_record(MIXSCAPE_TARGETS)

    result = mixscape(
        _record_store(record),
        MIXSCAPE,
        panel="scores",
        targets=["TP53", "GATA1", "MYC"],
        show=False,
    )

    try:
        assert list(result.axes) == ["TP53", "MYC"]
        pd.testing.assert_frame_equal(
            result.tables["scores"], _drawn_cells(record, ["TP53", "MYC"])
        )
        assert result.figure.get_suptitle().splitlines() == [
            "Mixscape scores by target",
            "Not fitted, so not drawn: 1 few_de_genes",
        ]
        assert result.provenance.extras["targets"] == ["TP53", "GATA1", "MYC"]
    finally:
        result.close()


def test_mixscape_scores_pool_the_splits_and_draw_the_first_fit() -> None:
    record = _mixscape_record(
        MIXSCAPE_SPLITS, perturbation_type="KD", split_column="run"
    )

    result = mixscape(_record_store(record), MIXSCAPE, panel="scores", show=False)

    try:
        assert list(result.axes) == ["KLF1", "MYC"]
        table = result.tables["scores"]
        pd.testing.assert_frame_equal(table, _drawn_cells(record, ["KLF1", "MYC"]))
        # KLF1 pools both runs, and the cells of MYC in run2 have no score.
        assert table["split"].tolist() == ["run1"] * 4 + ["run2"] * 4 + ["run1"] * 5
        # The curves are the fit of each target's first fitted run.
        _check_score_panel(result, "KLF1", (0.1, 0.3, 1.5, 0.2))
        _check_score_panel(result, "MYC", (0.0, 0.2, 1.0, 0.2))
        assert result.axes["KLF1"].get_title() == "KLF1, fit in run1"
        assert result.axes["MYC"].get_title() == "MYC, fit in run1"
        assert result.figure.get_suptitle().splitlines() == [
            "Mixscape scores by target",
            "Not fitted, so not drawn: 1 few_de_genes, 1 few_cells",
        ]
        (legend,) = result.figure.legends
        assert [text.get_text() for text in legend.get_texts()] == [
            "KD (perturbed)",
            *MIXSCAPE_LEGEND[1:],
        ]
    finally:
        result.close()


def test_mixscape_scores_draw_into_caller_axes_in_the_theme() -> None:
    figure, axes = plt.subplots(1, 3)
    figure.suptitle("The caller's figure")
    store = _record_store(_mixscape_record(MIXSCAPE_TARGETS))

    try:
        result = mixscape(
            store,
            MIXSCAPE,
            panel="scores",
            targets=["TP53", "GATA1", "MYC"],
            target={"MYC": axes[0], "TP53": axes[2]},
            theme="dark",
            show=False,
        )
        assert result.owns_figure is False
        assert result.figure is figure
        assert result.axes == {"TP53": axes[2], "MYC": axes[0]}
        # The caller's figure keeps its title and has no legend: the first
        # panel holds the legend, and the plot's title above its own.
        assert figure.get_suptitle() == "The caller's figure"
        assert figure.legends == []
        assert [
            text.get_text() for text in axes[2].get_legend().get_texts()
        ] == MIXSCAPE_LEGEND
        assert axes[0].get_legend() is None
        assert axes[2].get_title().splitlines() == [
            "Mixscape scores by target",
            "Not fitted, so not drawn: 1 few_de_genes",
            "TP53",
        ]
        assert axes[0].get_title() == "MYC"
        _check_score_panel(result, "TP53", (0.05, 0.25, 0.9, 0.3), theme="dark")
        _check_score_panel(result, "MYC", (0.0, 0.2, 1.0, 0.15), theme="dark")
        assert not axes[1].has_data()
    finally:
        plt.close(figure)


def test_mixscape_can_drop_the_legend() -> None:
    store = _record_store(_mixscape_record(MIXSCAPE_TARGETS))

    results = {
        (panel, legend): mixscape(
            store, MIXSCAPE, panel=panel, show_legend=legend, show=False
        )
        for panel in ("classes", "scores")
        for legend in (False, True)
    }

    try:
        for panel in ("classes", "scores"):
            plain, legended = results[panel, False], results[panel, True]
            assert plain.figure.legends == []
            assert all(ax.get_legend() is None for ax in plain.axes.values())
            # Only a figure with a legend leaves room for it at the right.
            assert (
                plain.figure.get_size_inches()[0] < legended.figure.get_size_inches()[0]
            )
    finally:
        for result in results.values():
            result.close()


def test_showing_the_mixscape_plot_releases_its_figure() -> None:
    store = _record_store(_mixscape_record(MIXSCAPE_TARGETS))

    classes = mixscape(store, MIXSCAPE)
    scores = mixscape(store, MIXSCAPE, panel="scores")

    assert classes.tables["classes"]["target"].tolist() == [
        "KLF1",
        "MYC",
        "TP53",
        "CD19",
        "GATA1",
    ]
    assert scores.tables["scores"]["target"].unique().tolist() == [
        "KLF1",
        "MYC",
        "TP53",
    ]


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"panel": "violin"}, ValueError, "panel must be 'classes' or 'scores'"),
        ({"panel": None}, ValueError, "panel must be 'classes' or 'scores'"),
        ({"max_targets": 0}, ValueError, "^max_targets must be at least 1$"),
        ({"max_targets": True}, TypeError, "^max_targets must be an integer$"),
        ({"max_targets": 2.0}, TypeError, "^max_targets must be an integer$"),
        ({"targets": "MYC"}, TypeError, "targets must be a sequence of targets"),
        ({"targets": [1]}, TypeError, "targets must be a sequence of targets"),
        ({"targets": []}, ValueError, "targets must name at least one target"),
        (
            {"targets": ["MYC", "TP53", "KLF1"], "max_targets": 2},
            ValueError,
            "targets names 3 targets, more than max_targets=2",
        ),
        (
            {"panel": "scores", "targets": ["MYC", "TP53"], "max_targets": 1},
            ValueError,
            "targets names 2 targets, more than max_targets=1",
        ),
    ],
)
def test_the_mixscape_plot_refuses_settings_it_cannot_use(
    arguments: dict[str, Any], error: type[Exception], message: str
) -> None:
    requests: list[ArtifactRef] = []
    store = _record_store(_mixscape_record(MIXSCAPE_TARGETS), requests)

    with pytest.raises(error, match=message):
        mixscape(store, MIXSCAPE, show=False, **arguments)
    # The settings are checked before the record is read.
    assert requests == []


@pytest.mark.parametrize("panel", ["classes", "scores"])
def test_the_mixscape_plot_names_a_target_the_classes_lack(panel: str) -> None:
    store = _record_store(_mixscape_record(MIXSCAPE_TARGETS))

    with pytest.raises(
        ValueError, match="The Mixscape classes have no target named 'SOX2'"
    ):
        mixscape(
            store, MIXSCAPE, panel=panel, targets=["MYC", "SOX2", "PAX5"], show=False
        )


@pytest.mark.parametrize(
    ("rows", "targets"),
    [(MIXSCAPE_TARGETS, ["GATA1", "CD19"]), (MIXSCAPE_MANY[3:], None)],
)
def test_the_mixscape_scores_need_a_fitted_target(
    rows: list[MixscapeRow], targets: list[str] | None
) -> None:
    store = _record_store(_mixscape_record(rows))

    with pytest.raises(ValueError, match="None of the targets has a fitted split"):
        mixscape(store, MIXSCAPE, panel="scores", targets=targets, show=False)


ASSOCIATIONS = ArtifactRef(
    scope="assay", assay="RNA", kind="perturbation_associations", artifact_id="f" * 64
)
TARGETING = to_rgba("#2b6cb0")
NEGATIVE_CONTROL = to_rgba("#d55e00")
CUT = to_rgba("#d55e00")
NAN = np.nan
AssociationRow = tuple[str, str, float, float, float]

# Each pair is a target, a gene, its p-value, log2 fold change and z; a pair
# with a NaN p-value failed QC. At FDR 0.1 the six smallest p-values are
# significant, so the Benjamini-Hochberg cut is 0.03. KLF1 and HBG1 tie on
# their p-value, and KLF1 has the larger z.
DISCOVERY_ROWS: list[AssociationRow] = [
    ("MYC", "NPM1", 1e-12, -1.8, -7.1),
    ("MYC", "NCL", 3e-5, -0.9, -4.2),
    ("MYC", "CD19", 0.4, 0.1, 0.8),
    ("MYC", "ACTB", NAN, NAN, NAN),
    ("TP53", "CDKN1A", 2e-8, -2.6, -5.6),
    ("TP53", "BAX", 0.002, 0.7, 3.1),
    ("TP53", "GAPDH", 0.9, 0.02, 0.1),
    ("GATA1", "HBG1", 0.03, 1.2, 2.2),
    ("GATA1", "KLF1", 0.03, -0.5, -2.6),
    ("GATA1", "TAL1", NAN, NAN, NAN),
    ("GATA1", "LMO2", 1.0, 0.0, 0.0),
    ("CD4", "IL2", 0.5, 0.2, 0.67),
    ("CD4", "IL7R", 0.7, -0.1, -0.39),
]
# The tested pairs of DISCOVERY_ROWS from the smallest p-value up.
VOLCANO_ORDER = [
    "MYC:NPM1",
    "TP53:CDKN1A",
    "MYC:NCL",
    "TP53:BAX",
    "GATA1:KLF1",
    "GATA1:HBG1",
    "MYC:CD19",
    "CD4:IL2",
    "CD4:IL7R",
    "TP53:GAPDH",
    "GATA1:LMO2",
]
# A p-value of 0, or below 1e-300, has no height and sits at the top edge,
# unlike one of 1e-300. MYC's cells hold no counts of MYC or MAX.
EDGE_ROWS: list[AssociationRow] = [
    ("TP53", "CDKN1A", 0.0, -2.6, -40.0),
    ("TP53", "MDM2", 1e-310, 2.1, 38.0),
    ("TP53", "BAX", 1e-300, 0.7, 37.0),
    ("MYC", "MYC", 1e-6, -np.inf, -5.0),
    ("MYC", "MAX", 0.0, -np.inf, -41.0),
    ("MYC", "NCL", 0.5, 0.3, 0.7),
]
EDGE_LEGEND = [
    "Significant at FDR 0.1",
    "Not significant",
    "p-value below 1e-300",
    "No counts in the perturbed cells",
    "Benjamini-Hochberg cut",
]
# Groups of non-targeting guides against the other control cells. At FDR 0.1
# the two smallest p-values are significant: two false discoveries.
CALIBRATION_ROWS: list[AssociationRow] = [
    ("NT1", "NPM1", 1e-5, 0.21, 4.4),
    ("NT1", "NCL", 2e-5, -0.18, -4.3),
    ("NT1", "CD19", 0.03, 0.05, 2.2),
    ("NT1", "ACTB", 0.95, -0.02, -0.06),
    ("NT1", "BAX", 0.31, 0.1, 1.0),
    ("NT2", "GAPDH", 0.62, -0.08, -0.5),
    ("NT2", "HBG1", 0.11, 0.0, 1.6),
    ("NT2", "KLF1", 0.78, 0.03, 0.28),
    ("NT2", "TAL1", NAN, NAN, NAN),
    ("NT2", "LMO2", 0.5, -0.04, -0.67),
    ("NT3", "IL2", 0.23, 0.07, 1.2),
    ("NT3", "IL7R", 0.87, -0.01, -0.16),
    ("NT3", "MDM2", 0.04, 0.12, 2.05),
    ("NT3", "MAX", 0.69, -0.06, -0.4),
    ("NT3", "CDKN1A", 0.41, 0.02, 0.82),
]
# Each target with its own gene; the p-value of MYC underflows to 0.
POSITIVE_ROWS: list[AssociationRow] = [
    ("MYC", "MYC", 0.0, -3.1, -30.0),
    ("TP53", "TP53", 1e-40, -2.4, -13.0),
    ("GATA1", "GATA1", 1e-9, -1.5, -6.0),
    ("CD4", "CD4", NAN, NAN, NAN),
]


def _grade(n_false: int) -> str:
    """Return the SCEPTRE book's grade of a calibration with ``n_false`` false discoveries."""
    for most, grade in ((1, "excellent"), (6, "mild"), (14, "moderate")):
        if n_false <= most:
            return grade
    return "severe"


def _pair_table(
    target: Sequence[str] | np.ndarray,
    feature: Sequence[str] | np.ndarray,
    p_value: np.ndarray,
    fold_change: np.ndarray,
    z: np.ndarray,
    *,
    alpha: float = 0.1,
    group_cells: int | None = None,
) -> pd.DataFrame:
    """Build a table of pairs as ``load_associations`` returns it.

    Each gene is named by its ID. A pair with a NaN p-value failed QC and
    holds NaN in every test column.
    Benjamini-Hochberg adjusts the p-values of the other pairs, and those
    adjusted to at most ``alpha`` are significant. There are 300 control
    cells. Each target has 20 cells and 5 more for each target before it, or,
    with ``group_cells``, each target is a group of that many control cells
    tested against the others.
    """
    p = np.asarray(p_value, dtype=np.float64)
    passed = ~np.isnan(p)
    adjusted = np.full(len(p), np.nan)
    if passed.any():
        adjusted[passed] = false_discovery_control(p[passed])
    names = np.asarray(target, dtype=object)
    rank = {name: index for index, name in enumerate(dict.fromkeys(names.tolist()))}
    if group_cells is None:
        n_cells = np.asarray([20 + 5 * rank[name] for name in names], dtype=np.int64)
    else:
        n_cells = np.full(len(p), group_cells, dtype=np.int64)
    second = passed & (p <= 0.01)
    return pd.DataFrame(
        {
            "target": names,
            "feature": np.asarray(feature, dtype=object),
            "feature_name": np.asarray(feature, dtype=object),
            "n_cells": n_cells,
            "n_control_cells": 300 - (0 if group_cells is None else n_cells),
            "n_nonzero_cells": np.where(passed, 12, 3),
            "n_nonzero_control_cells": np.where(passed, 80, 4),
            "pass_qc": passed,
            "z": np.where(passed, z, np.nan),
            "p_value": p,
            "adjusted_p_value": adjusted,
            "significant": passed & (adjusted <= alpha),
            "log2_fold_change": np.where(passed, fold_change, np.nan),
            "se_log2_fold_change": np.where(passed, 0.3, np.nan),
            "stage": np.where(passed, np.where(second, 2, 1), 0),
            "skew_location": np.where(second, 0.1, np.nan),
            "skew_scale": np.where(second, 1.1, np.nan),
            "skew_shape": np.where(second, 0.2, np.nan),
        }
    )


def _pairs(
    rows: list[AssociationRow], *, alpha: float = 0.1, group_cells: int | None = None
) -> pd.DataFrame:
    """Build the table of ``rows``; negative-control groups name their guides."""
    columns = [list(column) for column in zip(*rows, strict=True)] or [[]] * 5
    target, feature, p_value, fold_change, z = columns
    table = _pair_table(
        target,
        feature,
        np.asarray(p_value, dtype=np.float64),
        np.asarray(fold_change, dtype=np.float64),
        np.asarray(z, dtype=np.float64),
        alpha=alpha,
        group_cells=group_cells,
    )
    if group_cells is not None:
        table["control_guides"] = [f"{name}_a+{name}_b" for name in target]
    return table


def _associations_record(
    discovery: list[AssociationRow] | pd.DataFrame = DISCOVERY_ROWS,
    calibration: list[AssociationRow] = CALIBRATION_ROWS,
    positive: list[AssociationRow] = POSITIVE_ROWS,
    *,
    side: str = "both",
    moi: str = "low",
) -> PerturbationAssociations:
    """Build the record that ``load_associations`` returns for these pairs.

    The negative controls are groups of 30 control cells, and their false
    discoveries and grade follow from their significant pairs. At high MOI
    every group is compared with all other cells of the 300.
    """
    negative = _pairs(calibration, group_cells=30)
    n_false = int(negative["significant"].sum())
    tables = [
        discovery if isinstance(discovery, pd.DataFrame) else _pairs(discovery),
        _pairs(positive),
    ]
    if moi == "high":
        for table in tables:
            table["n_control_cells"] = 300 - table["n_cells"]
    return PerturbationAssociations(
        discovery=tables[0],
        calibration=negative,
        positive=tables[1],
        alpha=0.1,
        side=side,
        integration="union",
        moi=moi,
        resampling="conditional" if moi == "high" else "permutation",
        from_assay="RNA",
        n_false_discoveries=n_false,
        calibration_grade=_grade(n_false),
    )


def _tested(pairs: pd.DataFrame) -> pd.DataFrame:
    """Return the pairs that passed QC, from the smallest p-value up."""
    return pairs[pairs["pass_qc"]].sort_values("p_value", kind="stable")


def _marker_vertices(marker: str) -> np.ndarray:
    style = MarkerStyle(marker)
    vertices: np.ndarray = style.get_path().transformed(style.get_transform()).vertices
    return vertices


def _point_groups(ax: Any) -> list[Any]:
    """Return the scatters of a panel, without a QQ band."""
    return [
        collection
        for collection in ax.collections
        if not isinstance(collection, PolyCollection)
    ]


def _check_points(
    collections: list[Any],
    groups: list[tuple[np.ndarray, str, tuple[float, ...]]],
    *,
    rasterized: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Check each scatter's marker, colour and rasterization against ``groups``.

    Each group is the rows of a scatter, its marker and its colour. Returns
    each row's drawn position.
    """
    assert len(collections) == len(groups)
    n_rows = len(groups[0][0])
    x = np.full(n_rows, np.nan)
    y = np.full(n_rows, np.nan)
    for collection, (chosen, marker, color) in zip(collections, groups, strict=True):
        np.testing.assert_array_equal(
            collection.get_paths()[0].vertices, _marker_vertices(marker)
        )
        np.testing.assert_allclose(collection.get_facecolors(), [to_rgba(color, 0.75)])
        assert collection.get_rasterized() is rasterized
        offsets = np.asarray(collection.get_offsets())
        x[chosen] = offsets[:, 0]
        y[chosen] = offsets[:, 1]
    assert not np.isnan(x).any()
    return x, y


def _check_top_edge(
    ax: Any, y: np.ndarray, at_top: np.ndarray, heights: np.ndarray
) -> None:
    """Check that the rows at the top edge share a height above the others."""
    bottom, top = ax.get_ylim()
    assert ((bottom < y) & (y < top)).all()
    np.testing.assert_allclose(y[~at_top], heights[~at_top])
    if at_top.any():
        edge = y[at_top]
        assert (edge == edge[0]).all()
        assert edge[0] > heights[~at_top].max(initial=0.0)
        assert top - edge[0] < 0.05 * (top - bottom)


def _check_volcano(result: Any) -> tuple[np.ndarray, np.ndarray]:
    """Check that each drawn pair is a point at its fold change and p-value.

    Returns the drawn position of each row of the table.
    """
    ax = result.axes["volcano"]
    table = result.tables["volcano"]
    fold = table["log2_fold_change"].to_numpy()
    p = table["p_value"].to_numpy()
    significant = table["significant"].to_numpy(dtype=bool)
    no_counts = np.isneginf(fold)
    at_top = p < 1e-300
    shapes = np.where(no_counts, "<", np.where(at_top, "^", "o"))
    # The pairs that are not significant come first, each shape in turn.
    groups = [
        (chosen, shape, color)
        for flag, color in ((False, NOT_SIGNIFICANT), (True, SIGNIFICANT))
        for shape in ("o", "^", "<")
        if (chosen := (significant == flag) & (shapes == shape)).any()
    ]
    x, y = _check_points(
        ax.collections, groups, rasterized=result.provenance.extras["rasterized"]
    )
    np.testing.assert_array_equal(x[~no_counts], fold[~no_counts])
    left, right = ax.get_xlim()
    assert ((left < x) & (x < right)).all()
    if no_counts.any():
        # A fold change of minus infinity sits at the left edge.
        edge = x[no_counts]
        assert (edge == edge[0]).all()
        assert edge[0] < fold[~no_counts].min(initial=0.0)
        assert edge[0] - left < 0.05 * (right - left)
    heights = np.full(len(p), np.inf)
    heights[~at_top] = -np.log10(p[~at_top])
    _check_top_edge(ax, y, at_top, heights)
    return x, y


def _label_text(table: pd.DataFrame) -> list[str]:
    return (table["target"] + ":" + table["feature_name"]).tolist()


def _check_labels_apart(result: Any) -> None:
    """Check that the labels lie inside the axes and that no two overlap."""
    ax = result.axes["volcano"]
    result.figure.draw_without_rendering()
    frame = ax.get_window_extent()
    boxes = [Text.get_window_extent(label) for label in ax.texts]
    # Half a pixel of slack absorbs rounding.
    for index, box in enumerate(boxes):
        assert frame.x0 - 0.5 <= box.x0 and box.x1 <= frame.x1 + 0.5
        assert frame.y0 - 0.5 <= box.y0 and box.y1 <= frame.y1 + 0.5
        for other in boxes[index + 1 :]:
            assert not (
                box.x0 < other.x1 - 0.5
                and other.x0 < box.x1 - 0.5
                and box.y0 < other.y1 - 0.5
                and other.y0 < box.y1 - 0.5
            )


def test_association_volcano_draws_each_tested_discovery_pair() -> None:
    requests: list[ArtifactRef] = []
    record = _associations_record()

    result = association_volcano(
        _record_store(record, requests), ASSOCIATIONS, show=False
    )

    try:
        assert requests == [ASSOCIATIONS]
        assert list(result.axes) == ["volcano"]
        table = result.tables["volcano"]
        assert _label_text(table) == VOLCANO_ORDER
        pd.testing.assert_frame_equal(
            table,
            record.discovery.iloc[[0, 4, 1, 5, 8, 7, 2, 11, 12, 6, 10]].reset_index(
                drop=True
            ),
        )
        x, y = _check_volcano(result)
        ax = result.axes["volcano"]
        zero, cut = ax.get_lines()
        assert list(zero.get_xdata()) == [0.0, 0.0]
        assert to_rgba(zero.get_color()) == to_rgba(foreground_color("notebook"))
        # The cut is the largest p-value of a significant pair.
        assert cut.get_linestyle() == "--"
        np.testing.assert_allclose(cut.get_ydata(), [-np.log10(0.03)] * 2)
        assert to_rgba(cut.get_color()) == CUT
        assert ax.get_xlabel() == "Log2 fold change against control cells"
        assert ax.get_ylabel() == "-log10 p-value"
        assert ax.get_title().splitlines() == [
            "Association tests of the discovery pairs",
            "6 of 11 tested pairs significant at FDR 0.1",
            "Not plotted: 2 untested pairs",
        ]
        # The ten smallest p-values are labelled at their points.
        labels = ax.texts
        assert [label.get_text() for label in labels] == VOLCANO_ORDER[:10]
        for row, label in enumerate(labels):
            assert label.xy == (x[row], y[row])
            assert to_rgba(label.get_color()) == to_rgba(foreground_color("notebook"))
        _check_labels_apart(result)
        (legend,) = result.figure.legends
        assert [text.get_text() for text in legend.get_texts()] == [
            "Significant at FDR 0.1",
            "Not significant",
            "Benjamini-Hochberg cut",
        ]
        assert result.legends == (
            LegendSpec(kind="categorical", label="Significance"),
            LegendSpec(kind="line", label="Benjamini-Hochberg cut"),
        )
        provenance = result.provenance
        assert provenance.assay == "RNA"
        # The cells of MYC, TP53, GATA1 and CD4, and the control cells.
        assert provenance.n_cells == 20 + 25 + 30 + 35 + 300
        assert provenance.notes == ("perturbation_associations",)
        assert provenance.extras == {
            "tests": ASSOCIATIONS.to_dict(),
            "side": "both",
            "integration": "union",
            "moi": "low",
            "from_assay": "RNA",
            "alpha": 0.1,
            "targets": None,
            "label_top": 10,
            "n_tested": 11,
            "n_significant": 6,
            "largest_significant_p_value": 0.03,
            "rasterized": False,
        }
    finally:
        result.close()


def test_association_volcano_breaks_ties_by_z_and_then_by_name() -> None:
    rows: list[AssociationRow] = [
        ("B", "G2", 0.01, 0.5, 2.6),
        ("A", "G9", 0.01, -0.5, -2.6),
        ("C", "G1", 0.01, 0.6, -3.0),
        ("A", "G1", 0.01, 0.4, 2.6),
    ]

    result = association_volcano(
        _record_store(_associations_record(rows)),
        ASSOCIATIONS,
        label_top=2,
        show=False,
    )

    try:
        assert _label_text(result.tables["volcano"]) == ["C:G1", "A:G1", "A:G9", "B:G2"]
        texts = [label.get_text() for label in result.axes["volcano"].texts]
        assert texts == ["C:G1", "A:G1"]
        _check_volcano(result)
    finally:
        result.close()


def test_association_volcano_keeps_the_chosen_targets() -> None:
    store = _record_store(_associations_record())

    kept = association_volcano(
        store, ASSOCIATIONS, targets=["CD4", "GATA1", "CD4"], show=False
    )
    quiet = association_volcano(store, ASSOCIATIONS, targets=["CD4"], show=False)

    try:
        assert _label_text(kept.tables["volcano"]) == [
            "GATA1:KLF1",
            "GATA1:HBG1",
            "CD4:IL2",
            "CD4:IL7R",
            "GATA1:LMO2",
        ]
        _check_volcano(kept)
        ax = kept.axes["volcano"]
        # The cut comes from the pairs of every target.
        _, cut = ax.get_lines()
        np.testing.assert_allclose(cut.get_ydata(), [-np.log10(0.03)] * 2)
        assert ax.get_title().splitlines() == [
            "Association tests of the discovery pairs",
            "2 of 5 tested pairs significant at FDR 0.1",
            "Not plotted: 1 untested pair",
        ]
        extras = kept.provenance.extras
        assert extras["targets"] == ["CD4", "GATA1"]
        assert extras["largest_significant_p_value"] == 0.03
        assert kept.provenance.n_cells == 30 + 35 + 300
        # No pair of CD4 is significant, and the axis still holds the cut.
        _check_volcano(quiet)
        ax = quiet.axes["volcano"]
        _, cut = ax.get_lines()
        assert cut.get_ydata()[0] == pytest.approx(-np.log10(0.03))
        assert cut.get_ydata()[0] < ax.get_ylim()[1]
        assert ax.get_title().splitlines()[1:] == [
            "0 of 2 tested pairs significant at FDR 0.1"
        ]
        (legend,) = quiet.figure.legends
        assert [text.get_text() for text in legend.get_texts()] == [
            "Not significant",
            "Benjamini-Hochberg cut",
        ]
    finally:
        kept.close()
        quiet.close()


def test_association_volcano_names_the_targets_it_lacks() -> None:
    store = _record_store(_associations_record())

    with pytest.raises(
        ValueError, match="The discovery pairs have no target named 'SOX2', 'PAX5'"
    ):
        association_volcano(
            store, ASSOCIATIONS, targets=["MYC", "SOX2", "PAX5"], show=False
        )


def test_association_volcano_draws_extreme_pairs_at_the_edges() -> None:
    record = _associations_record(EDGE_ROWS)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = association_volcano(_record_store(record), ASSOCIATIONS, show=False)

    try:
        table = result.tables["volcano"]
        # The p-values of 0 tie, so the larger z comes first.
        assert _label_text(table) == [
            "MYC:MAX",
            "TP53:CDKN1A",
            "TP53:MDM2",
            "TP53:BAX",
            "MYC:MYC",
            "MYC:NCL",
        ]
        x, y = _check_volcano(result)
        # A p-value of 1e-300 keeps its height, and MAX sits at both edges.
        assert y[3] == pytest.approx(300.0)
        assert (x[0], y[0]) == (x[4], y[1])
        ax = result.axes["volcano"]
        _, cut = ax.get_lines()
        np.testing.assert_allclose(cut.get_ydata(), [6.0, 6.0])
        (legend,) = result.figure.legends
        assert [text.get_text() for text in legend.get_texts()] == EDGE_LEGEND
        markers = [handle.get_marker() for handle in legend.legend_handles[:4]]
        assert markers == ["o", "o", "^", "<"]
        # The markers of the edges are hollow.
        assert [
            handle.get_markerfacecolor() for handle in legend.legend_handles[2:4]
        ] == [
            "none",
            "none",
        ]
        _check_labels_apart(result)
    finally:
        result.close()


def test_association_volcano_draws_a_cut_below_1e_300_at_the_top_edge() -> None:
    rows: list[AssociationRow] = [
        ("A", "G1", 0.0, -2.0, -40.0),
        ("A", "G2", 0.0, 2.0, 41.0),
        ("A", "G3", 0.6, 0.1, 0.5),
        ("A", "G4", 0.9, 0.0, 0.1),
    ]

    result = association_volcano(
        _record_store(_associations_record(rows)), ASSOCIATIONS, show=False
    )

    try:
        _, y = _check_volcano(result)
        ax = result.axes["volcano"]
        _, cut = ax.get_lines()
        assert list(cut.get_ydata()) == [y[0], y[0]]
        assert result.provenance.extras["largest_significant_p_value"] == 0.0
    finally:
        result.close()


def test_association_volcano_without_a_significant_pair_draws_no_cut() -> None:
    rows: list[AssociationRow] = [
        ("A", "G1", 0.3, 0.1, 1.0),
        ("A", "G2", 0.8, -0.1, -0.25),
    ]

    result = association_volcano(
        _record_store(_associations_record(rows)), ASSOCIATIONS, show=False
    )

    try:
        _check_volcano(result)
        ax = result.axes["volcano"]
        (zero,) = ax.get_lines()
        assert list(zero.get_xdata()) == [0.0, 0.0]
        (legend,) = result.figure.legends
        assert [text.get_text() for text in legend.get_texts()] == ["Not significant"]
        assert result.provenance.extras["largest_significant_p_value"] is None
        assert ax.get_title().splitlines() == [
            "Association tests of the discovery pairs",
            "0 of 2 tested pairs significant at FDR 0.1",
        ]
    finally:
        result.close()


def test_association_volcano_without_a_finite_height_still_shows_zero() -> None:
    ones: list[AssociationRow] = [
        ("A", "G1", 1.0, 0.1, 0.0),
        ("A", "G2", 1.0, -0.1, 0.0),
    ]
    zeros: list[AssociationRow] = [
        ("A", "G1", 0.0, 2.0, 40.0),
        ("A", "G2", 0.0, -2.0, -39.0),
    ]

    flat = association_volcano(
        _record_store(_associations_record(ones)), ASSOCIATIONS, show=False
    )
    edged = association_volcano(
        _record_store(_associations_record(zeros)), ASSOCIATIONS, show=False
    )

    try:
        # The p-values of 1 sit at 0, on an axis that runs to 1.
        _, y = _check_volcano(flat)
        assert list(y) == [0.0, 0.0]
        assert flat.axes["volcano"].get_ylim() == pytest.approx((-0.05, 1.05))
        # The p-values of 0, and the cut among them, sit at the top edge of an
        # axis that runs to 1 below it.
        _, y = _check_volcano(edged)
        assert list(y) == pytest.approx([1.15, 1.15])
        ax = edged.axes["volcano"]
        assert ax.get_ylim() == pytest.approx((-0.05, 1.2))
        _, cut = ax.get_lines()
        assert list(cut.get_ydata()) == pytest.approx([1.15, 1.15])
    finally:
        flat.close()
        edged.close()


def test_association_volcano_labels_the_chosen_number_of_pairs() -> None:
    store = _record_store(_associations_record())

    three = association_volcano(store, ASSOCIATIONS, label_top=np.int64(3), show=False)
    none = association_volcano(store, ASSOCIATIONS, label_top=0, show=False)
    every = association_volcano(store, ASSOCIATIONS, label_top=50, show=False)

    try:
        texts = [label.get_text() for label in three.axes["volcano"].texts]
        assert texts == VOLCANO_ORDER[:3]
        assert three.provenance.extras["label_top"] == 3
        assert type(three.provenance.extras["label_top"]) is int
        assert len(none.axes["volcano"].texts) == 0
        texts = [label.get_text() for label in every.axes["volcano"].texts]
        assert texts == VOLCANO_ORDER
        _check_labels_apart(every)
    finally:
        three.close()
        none.close()
        every.close()


def test_association_volcano_labels_genes_by_name() -> None:
    discovery = _pairs(DISCOVERY_ROWS)
    discovery["feature_name"] = "name-" + discovery["feature"]
    store = _record_store(_associations_record(discovery))

    plot = association_volcano(store, ASSOCIATIONS, label_top=3, show=False)

    try:
        texts = [label.get_text() for label in plot.axes["volcano"].texts]
        assert texts == [
            "{}:name-{}".format(*label.split(":")) for label in VOLCANO_ORDER[:3]
        ]
    finally:
        plot.close()


def test_association_volcano_needs_the_gene_names_of_the_record() -> None:
    # The loader always adds the names, so the plot never labels genes by ID.
    discovery = _pairs(DISCOVERY_ROWS).drop(columns="feature_name")
    store = _record_store(_associations_record(discovery))

    with pytest.raises(KeyError, match="feature_name"):
        association_volcano(store, ASSOCIATIONS, show=False)


def test_association_volcano_moves_crowded_labels_apart() -> None:
    # Five pairs share almost one place, near the top right of the volcano.
    rows: list[AssociationRow] = [
        ("A", f"GENE{index}", 1e-20 * (1 + index / 100), 2.0 + index / 1000, 9.0)
        for index in range(5)
    ]
    rows.append(("A", "OTHER", 0.5, -1.0, -0.7))

    result = association_volcano(
        _record_store(_associations_record(rows)), ASSOCIATIONS, show=False
    )

    try:
        labels = result.axes["volcano"].texts
        assert [label.get_text() for label in labels] == [
            *(f"A:GENE{index}" for index in range(5)),
            "A:OTHER",
        ]
        _check_labels_apart(result)
        # A label starts beside its point, towards the middle; the others move
        # aside, and those above or below their points have a line to them.
        first = labels[0]
        assert first.xyann == (-5.0, 0.0)
        assert first.get_horizontalalignment() == "right"
        other = labels[5]
        assert other.xyann == (5.0, 0.0)
        assert other.get_horizontalalignment() == "left"
        assert any(label.xyann[1] != 0 for label in labels[1:5])
        for label in labels:
            assert label.arrow_patch.get_visible() == (label.xyann[1] != 0)
            side = "left" if label.xyann[0] > 0 else "right"
            assert label.get_horizontalalignment() == side
    finally:
        result.close()


def test_association_volcano_rasterizes_more_than_50000_points() -> None:
    n_pairs = 50_000
    discovery = _pair_table(
        ["A"] * n_pairs + ["B"],
        [f"G{index}" for index in range(n_pairs + 1)],
        np.linspace(0.05, 1.0, n_pairs + 1),
        np.linspace(-1.0, 1.0, n_pairs + 1),
        np.linspace(2.0, 0.0, n_pairs + 1),
    )
    store = _record_store(_associations_record(discovery))

    every = association_volcano(store, ASSOCIATIONS, label_top=0, show=False)
    most = association_volcano(
        store, ASSOCIATIONS, targets=["A"], label_top=0, show=False
    )

    try:
        assert len(every.tables["volcano"]) == n_pairs + 1
        assert every.provenance.extras["rasterized"] is True
        collections = every.axes["volcano"].collections
        assert collections and all(c.get_rasterized() for c in collections)
        assert len(most.tables["volcano"]) == n_pairs
        assert most.provenance.extras["rasterized"] is False
        assert not any(c.get_rasterized() for c in most.axes["volcano"].collections)
    finally:
        every.close()
        most.close()


@pytest.mark.parametrize(
    ("side", "words"),
    [("both", ""), ("left", " for a decrease"), ("right", " for an increase")],
)
def test_the_association_plots_name_the_side_tested(side: str, words: str) -> None:
    store = _record_store(_associations_record(side=side))

    volcano = association_volcano(store, ASSOCIATIONS, show=False)
    calibration = association_calibration(store, ASSOCIATIONS, show=False)

    try:
        assert volcano.axes["volcano"].get_ylabel() == f"-log10 p-value{words}"
        qq_label = calibration.axes["qq"].get_ylabel()
        assert qq_label == f"Observed -log10 p-value{words}"
        controls_label = calibration.axes["controls"].get_ylabel()
        assert controls_label == f"-log10 p-value{words}"
        assert volcano.provenance.extras["side"] == side
        assert calibration.provenance.extras["side"] == side
    finally:
        volcano.close()
        calibration.close()


def test_association_volcano_draws_into_caller_axes_in_the_theme() -> None:
    figure, ax = plt.subplots()
    store = _record_store(_associations_record(EDGE_ROWS))

    try:
        result = association_volcano(
            store, ASSOCIATIONS, target=ax, theme="dark", show=False
        )
        assert result.owns_figure is False
        assert result.figure is figure
        assert result.axes == {"volcano": ax}
        assert figure.legends == []
        texts = [text.get_text() for text in ax.get_legend().get_texts()]
        assert texts == EDGE_LEGEND
        _check_volcano(result)
        dark = to_rgba(foreground_color("dark"))
        zero, _ = ax.get_lines()
        assert to_rgba(zero.get_color()) == dark
        assert all(to_rgba(label.get_color()) == dark for label in ax.texts)
        assert to_rgba(ax.title.get_color()) == dark
        _check_labels_apart(result)
    finally:
        plt.close(figure)


def test_association_volcano_can_drop_the_legend() -> None:
    store = _record_store(_associations_record())

    plain = association_volcano(store, ASSOCIATIONS, show_legend=False, show=False)
    legended = association_volcano(store, ASSOCIATIONS, show=False)

    try:
        assert plain.figure.legends == []
        assert plain.axes["volcano"].get_legend() is None
        # Only a figure with a legend leaves room for it at the right.
        assert plain.figure.get_size_inches()[0] < legended.figure.get_size_inches()[0]
    finally:
        plain.close()
        legended.close()


def test_showing_the_association_plots_releases_their_figures() -> None:
    store = _record_store(_associations_record())

    volcano = association_volcano(store, ASSOCIATIONS)
    calibration = association_calibration(store, ASSOCIATIONS)

    assert _label_text(volcano.tables["volcano"]) == VOLCANO_ORDER
    assert calibration.provenance.extras["n_false_discoveries"] == 2


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"label_top": -1}, ValueError, "^label_top must be at least 0$"),
        ({"label_top": True}, TypeError, "^label_top must be an integer$"),
        ({"label_top": 2.0}, TypeError, "^label_top must be an integer$"),
        ({"label_top": "3"}, TypeError, "^label_top must be an integer$"),
        ({"targets": "MYC"}, TypeError, "targets must be a sequence of targets"),
        ({"targets": [1]}, TypeError, "targets must be a sequence of targets"),
        ({"targets": []}, ValueError, "targets must name at least one target"),
    ],
)
def test_the_association_volcano_refuses_settings_it_cannot_use(
    arguments: dict[str, Any], error: type[Exception], message: str
) -> None:
    requests: list[ArtifactRef] = []
    store = _record_store(_associations_record(), requests)

    with pytest.raises(error, match=message):
        association_volcano(store, ASSOCIATIONS, show=False, **arguments)
    # The settings are checked before the record is read.
    assert requests == []


@pytest.mark.parametrize(
    ("rows", "targets", "message"),
    [
        (
            [("MYC", "NPM1", NAN, NAN, NAN)],
            None,
            "None of the discovery pairs has a p-value",
        ),
        ([], None, "None of the discovery pairs has a p-value to plot"),
        (
            [*DISCOVERY_ROWS, ("CD8", "GZMB", NAN, NAN, NAN)],
            ["CD8"],
            "None of the discovery pairs of these targets has a p-value",
        ),
    ],
)
def test_the_association_volcano_needs_a_tested_discovery_pair(
    rows: list[AssociationRow], targets: list[str] | None, message: str
) -> None:
    store = _record_store(_associations_record(rows))

    with pytest.raises(ValueError, match=message):
        association_volcano(store, ASSOCIATIONS, targets=targets, show=False)


def _check_qq(
    result: Any, record: PerturbationAssociations, *, theme: str = "notebook"
) -> None:
    """Check the QQ panel against the tested p-values and the Beta band."""
    ax = result.axes["qq"]
    points = result.tables["qq"]
    # Each set from its smallest p-value up, against the means of as many
    # ordered uniform p-values.
    sets = {
        "calibration": _tested(record.calibration),
        "discovery": _tested(record.discovery),
    }
    expected = pd.DataFrame(
        {
            "set": np.repeat(
                np.asarray(list(sets), dtype=object),
                [len(frame) for frame in sets.values()],
            ),
            **{
                column: np.concatenate(
                    [frame[column].to_numpy(dtype=object) for frame in sets.values()]
                )
                for column in ("target", "feature")
            },
            "p_value": np.concatenate(
                [frame["p_value"].to_numpy() for frame in sets.values()]
            ),
            "expected": np.concatenate(
                [
                    np.arange(1, len(frame) + 1) / (len(frame) + 1)
                    for frame in sets.values()
                ]
            ),
        }
    )
    pd.testing.assert_frame_equal(points, expected)
    # The band holds the 2.5% and 97.5% quantiles of the i-th smallest of n
    # uniform p-values, at every rank of these few p-values.
    n = len(_tested(record.calibration))
    band = result.tables["band"]
    ranks = np.arange(1, n + 1)
    assert band["rank"].tolist() == ranks.tolist()
    np.testing.assert_allclose(band["expected"], ranks / (n + 1))
    np.testing.assert_allclose(band["lower"], beta.ppf(0.025, ranks, n - ranks + 1))
    np.testing.assert_allclose(band["upper"], beta.ppf(0.975, ranks, n - ranks + 1))
    (fill,) = [c for c in ax.collections if isinstance(c, PolyCollection)]
    vertices = fill.get_paths()[0].vertices
    for x, low, high in zip(
        -np.log10(band["expected"]),
        -np.log10(band["upper"]),
        -np.log10(band["lower"]),
        strict=True,
    ):
        assert np.isclose(vertices, [x, low]).all(axis=1).any()
        assert np.isclose(vertices, [x, high]).all(axis=1).any()
    foreground = to_rgba(foreground_color(theme))
    np.testing.assert_allclose(fill.get_facecolor(), [to_rgba(foreground, 0.15)])
    (identity,) = ax.get_lines()
    assert identity.get_xy1() == (0.0, 0.0)
    assert identity.get_slope() == 1.0
    assert to_rgba(identity.get_color()) == foreground
    # The axis holds the identity line, the band and the negative controls,
    # and the discovery p-values above them, like p-values below 1e-300, sit
    # at the top edge.
    names = points["set"].to_numpy(dtype=object)
    p = points["p_value"].to_numpy()
    heights = np.full(len(p), np.inf)
    shown = p >= 1e-300
    heights[shown] = -np.log10(p[shown])
    high = max(
        np.log10(n + 1),
        -np.log10(beta.ppf(0.025, 1, n)),
        heights[(names == "calibration") & shown].max(),
        np.log10(np.count_nonzero(names == "discovery") + 1),
    )
    beyond = ~(heights <= high)
    groups = [
        (chosen, shape, color)
        for name, color in (
            ("calibration", NEGATIVE_CONTROL),
            ("discovery", TARGETING),
        )
        for shape, edge in (("o", False), ("^", True))
        if (chosen := (names == name) & (beyond == edge)).any()
    ]
    x, y = _check_points(_point_groups(ax), groups, rasterized=False)
    np.testing.assert_allclose(x, -np.log10(points["expected"].to_numpy()))
    _check_top_edge(ax, y, beyond, np.where(beyond, np.inf, heights))
    assert ax.get_xlabel() == "Expected -log10 p-value"
    assert ax.get_title() == "Negative-control p-values"


def _check_fold_changes(
    result: Any,
    record: PerturbationAssociations,
    compared: str = "the other control cells",
) -> None:
    """Check the histogram of the negative-control fold changes and its mean."""
    ax = result.axes["fold_changes"]
    folds = _tested(record.calibration)["log2_fold_change"].to_numpy()
    values = folds[np.isfinite(folds)]
    counts, edges = np.histogram(values, bins=30)
    histogram = result.tables["fold_changes"]
    np.testing.assert_allclose(histogram["bin_start"], edges[:-1])
    np.testing.assert_allclose(histogram["bin_stop"], edges[1:])
    assert histogram["pairs"].tolist() == counts.tolist()
    bars = ax.patches
    assert len(bars) == 30
    np.testing.assert_allclose([bar.get_x() for bar in bars], edges[:-1])
    np.testing.assert_allclose([bar.get_width() for bar in bars], np.diff(edges))
    assert [bar.get_height() for bar in bars] == counts.tolist()
    assert {bar.get_facecolor() for bar in bars} == {NEGATIVE_CONTROL}
    (mean,) = ax.get_lines()
    assert mean.get_linestyle() == "--"
    np.testing.assert_allclose(mean.get_xdata(), [values.mean()] * 2)
    assert result.provenance.extras["mean_log2_fold_change"] == pytest.approx(
        values.mean()
    )
    assert (np.asarray(ax.get_yticks()) % 1 == 0).all()
    assert ax.get_xlabel() == f"Log2 fold change against {compared}"
    assert ax.get_ylabel() == "Negative-control pairs"


def _check_controls(result: Any, record: PerturbationAssociations) -> None:
    """Check the strips of the positive-control and negative-control p-values."""
    ax = result.axes["controls"]
    controls = result.tables["controls"]
    positive = _tested(record.positive).sort_index()
    calibration = _tested(record.calibration).sort_index()
    expected = pd.DataFrame(
        {
            "set": np.repeat(
                np.asarray(["positive", "calibration"], dtype=object),
                [len(positive), len(calibration)],
            ),
            "target": np.concatenate(
                [positive["target"].to_numpy(), calibration["target"].to_numpy()]
            ).astype(object),
            "feature": np.concatenate(
                [positive["feature"].to_numpy(), calibration["feature"].to_numpy()]
            ).astype(object),
            "p_value": np.concatenate(
                [positive["p_value"].to_numpy(), calibration["p_value"].to_numpy()]
            ),
        }
    )
    pd.testing.assert_frame_equal(controls, expected)
    names = controls["set"].to_numpy(dtype=object)
    p = controls["p_value"].to_numpy()
    at_top = p < 1e-300
    groups = [
        (chosen, shape, color)
        for name, color in (("positive", TARGETING), ("calibration", NEGATIVE_CONTROL))
        for shape, edge in (("o", False), ("^", True))
        if (chosen := (names == name) & (at_top == edge)).any()
    ]
    x, y = _check_points(_point_groups(ax), groups, rasterized=False)
    # Each strip spreads its points around its position.
    positions = np.where(names == "positive", 0.0, 1.0)
    assert (np.abs(x - positions) <= 0.25).all()
    assert len(np.unique(x)) == len(x)
    heights = np.full(len(p), np.inf)
    heights[~at_top] = -np.log10(p[~at_top])
    _check_top_edge(ax, y, at_top, heights)
    assert list(ax.get_xticks()) == [0.0, 1.0]
    assert [label.get_text() for label in ax.get_xticklabels()] == [
        f"Positive controls\n{len(positive)} pair{'' if len(positive) == 1 else 's'}",
        f"Negative controls\n{len(calibration)} pairs",
    ]


def test_association_calibration_draws_the_three_panels() -> None:
    requests: list[ArtifactRef] = []
    record = _associations_record()

    result = association_calibration(
        _record_store(record, requests), ASSOCIATIONS, show=False
    )

    try:
        assert requests == [ASSOCIATIONS]
        assert list(result.axes) == ["qq", "fold_changes", "controls"]
        positions = [
            (spec.rowspan.start, spec.colspan.start)
            for spec in (ax.get_subplotspec() for ax in result.axes.values())
        ]
        assert positions == [(0, 0), (0, 1), (0, 2)]
        assert list(result.tables) == ["qq", "band", "fold_changes", "controls"]
        _check_qq(result, record)
        _check_fold_changes(result, record)
        _check_controls(result, record)
        qq, fold_changes, controls = result.axes.values()
        assert [text.get_text() for text in qq.get_legend().get_texts()] == [
            "Negative controls",
            "Discovery pairs",
            "Null expectation",
            "95% null band",
            "Beyond the axis",
        ]
        mean = _tested(record.calibration)["log2_fold_change"].mean()
        assert fold_changes.get_title().splitlines() == [
            "Negative-control fold changes",
            f"Dashed line at the mean, {mean:.3g}",
        ]
        assert fold_changes.get_legend() is None
        assert controls.get_title().splitlines() == [
            "Positive and negative controls",
            "2 false discoveries at FDR 0.1 (mild)",
        ]
        assert [text.get_text() for text in controls.get_legend().get_texts()] == [
            "p-value below 1e-300"
        ]
        assert result.figure.legends == []
        assert result.legends == (
            LegendSpec(kind="categorical", label="Pairs"),
            LegendSpec(kind="line", label="Expected under the null"),
        )
        provenance = result.provenance
        assert provenance.assay == "RNA"
        # The control cells, which each negative-control pair splits.
        assert provenance.n_cells == 300
        assert provenance.notes == ("perturbation_associations",)
        assert provenance.extras == {
            "tests": ASSOCIATIONS.to_dict(),
            "side": "both",
            "integration": "union",
            "moi": "low",
            "from_assay": "RNA",
            "alpha": 0.1,
            "n_false_discoveries": 2,
            "calibration_grade": "mild",
            "mean_log2_fold_change": pytest.approx(mean),
            "n_tested_pairs": {"discovery": 11, "calibration": 14, "positive": 3},
            "rasterized": False,
        }
    finally:
        result.close()


def test_association_plots_compare_with_all_other_cells_at_high_moi() -> None:
    record = _associations_record(moi="high")
    store = _record_store(record)

    volcano = association_volcano(store, ASSOCIATIONS, show=False)
    calibration = association_calibration(store, ASSOCIATIONS, show=False)

    try:
        # Every pair's cells are all 300 cells, which its group splits.
        assert volcano.provenance.n_cells == 300
        assert calibration.provenance.n_cells == 300
        assert volcano.provenance.extras["moi"] == "high"
        assert calibration.provenance.extras["moi"] == "high"
        assert (
            volcano.axes["volcano"].get_xlabel()
            == "Log2 fold change against all other cells"
        )
        _check_fold_changes(calibration, record, compared="all other cells")
        # The tables are those of low MOI.
        low = association_volcano(
            _record_store(_associations_record()), ASSOCIATIONS, show=False
        )
        try:
            assert low.provenance.n_cells == 20 + 25 + 30 + 35 + 300
            pd.testing.assert_frame_equal(
                volcano.tables["volcano"].drop(columns="n_control_cells"),
                low.tables["volcano"].drop(columns="n_control_cells"),
            )
        finally:
            low.close()
    finally:
        volcano.close()
        calibration.close()


def test_association_calibration_spreads_a_large_band_on_the_log_scale() -> None:
    n_pairs = 5000
    rows: list[AssociationRow] = [
        ("NT1", f"G{index}", (index + 0.5) / n_pairs, 0.0, 0.0)
        for index in range(n_pairs)
    ]
    record = _associations_record(calibration=rows)

    result = association_calibration(_record_store(record), ASSOCIATIONS, show=False)

    try:
        band = result.tables["band"]
        ranks = band["rank"].to_numpy()
        # 1,000 ranks spread on the log scale, without repeats.
        expected = np.unique(np.rint(np.geomspace(1, n_pairs, 1000)).astype(int))
        assert ranks.tolist() == expected.tolist()
        assert ranks[0] == 1 and ranks[-1] == n_pairs
        assert len(ranks) < 1000
        np.testing.assert_allclose(
            band["lower"], beta.ppf(0.025, ranks, n_pairs - ranks + 1)
        )
        np.testing.assert_allclose(
            band["upper"], beta.ppf(0.975, ranks, n_pairs - ranks + 1)
        )
        # Every negative control is drawn.
        assert np.count_nonzero(result.tables["qq"]["set"] == "calibration") == n_pairs
        assert result.axes["controls"].get_title().splitlines()[1] == (
            "0 false discoveries at FDR 0.1 (excellent)"
        )
    finally:
        result.close()


@pytest.mark.parametrize(
    ("positive", "message"),
    [
        ([], "No positive controls"),
        ([("CD4", "CD4", NAN, NAN, NAN)], "No positive control has a statistic"),
    ],
)
def test_association_calibration_says_when_no_positive_control_was_tested(
    positive: list[AssociationRow], message: str
) -> None:
    record = _associations_record(positive=positive)

    result = association_calibration(_record_store(record), ASSOCIATIONS, show=False)

    try:
        _check_controls(result, record)
        ax = result.axes["controls"]
        (note,) = ax.texts
        assert note.get_text() == message
        assert note.get_position()[0] == 0.0
        assert (result.tables["controls"]["set"] == "calibration").all()
        # Without a p-value below 1e-300 the panel needs no legend.
        assert ax.get_legend() is None
        assert result.provenance.extras["n_tested_pairs"]["positive"] == 0
    finally:
        result.close()


def test_association_calibration_without_tested_discovery_pairs_draws_no_overlay() -> (
    None
):
    record = _associations_record([("MYC", "NPM1", NAN, NAN, NAN)])

    result = association_calibration(_record_store(record), ASSOCIATIONS, show=False)

    try:
        _check_qq(result, record)
        assert set(result.tables["qq"]["set"]) == {"calibration"}
        texts = result.axes["qq"].get_legend().get_texts()
        assert [text.get_text() for text in texts] == [
            "Negative controls",
            "Null expectation",
            "95% null band",
        ]
    finally:
        result.close()


def test_association_calibration_draws_negative_controls_below_1e_300_at_the_top() -> (
    None
):
    rows = [*CALIBRATION_ROWS, ("NT4", "ACTB", 0.0, 0.3, 39.0)]
    record = _associations_record(calibration=rows)

    result = association_calibration(_record_store(record), ASSOCIATIONS, show=False)

    try:
        _check_qq(result, record)
        _check_controls(result, record)
        assert result.axes["controls"].get_title().splitlines()[1] == (
            "3 false discoveries at FDR 0.1 (mild)"
        )
    finally:
        result.close()


def test_association_calibration_leaves_out_infinite_fold_changes() -> None:
    some = _associations_record(
        calibration=[*CALIBRATION_ROWS, ("NT4", "ACTB", 0.7, -np.inf, -0.4)]
    )
    every = _associations_record(
        calibration=[
            ("NT1", "NPM1", 0.3, -np.inf, -1.0),
            ("NT1", "NCL", 0.8, -np.inf, -0.2),
        ]
    )

    partly = association_calibration(_record_store(some), ASSOCIATIONS, show=False)
    none = association_calibration(_record_store(every), ASSOCIATIONS, show=False)

    try:
        _check_fold_changes(partly, some)
        folds = _tested(some.calibration)["log2_fold_change"].to_numpy()
        mean = folds[np.isfinite(folds)].mean()
        assert partly.axes["fold_changes"].get_title().splitlines() == [
            "Negative-control fold changes",
            f"Dashed line at the mean, {mean:.3g}",
            "Not drawn: 1 without a finite fold change",
        ]
        ax = none.axes["fold_changes"]
        assert len(ax.patches) == 0
        assert len(ax.get_lines()) == 0
        assert none.tables["fold_changes"].empty
        assert list(none.tables["fold_changes"].columns) == [
            "bin_start",
            "bin_stop",
            "pairs",
        ]
        assert ax.get_title().splitlines() == [
            "Negative-control fold changes",
            "Not drawn: 2 without a finite fold change",
        ]
        assert none.provenance.extras["mean_log2_fold_change"] is None
    finally:
        partly.close()
        none.close()


@pytest.mark.parametrize(
    ("threshold", "discovery", "crowded"),
    [
        # The QQ panel draws 14 negative controls and 11 discovery pairs, and
        # the controls panel 3 positive and 14 negative controls.
        (20, DISCOVERY_ROWS, "qq"),
        # Without discovery pairs, the QQ panel draws the 14 negative controls.
        (15, [], "controls"),
    ],
)
def test_association_calibration_rasterizes_crowded_panels(
    monkeypatch: pytest.MonkeyPatch,
    threshold: int,
    discovery: list[AssociationRow],
    crowded: str,
) -> None:
    monkeypatch.setattr(
        "cytearc.plotting.perturbation.DEFAULT_RASTERIZE_THRESHOLD", threshold
    )
    store = _record_store(_associations_record(discovery))

    result = association_calibration(store, ASSOCIATIONS, show=False)

    try:
        for panel in ("qq", "controls"):
            points = _point_groups(result.axes[panel])
            assert points
            assert all(c.get_rasterized() is (panel == crowded) for c in points)
        # The figure records that one of its panels is rasterized.
        assert result.provenance.extras["rasterized"] is True
    finally:
        result.close()


def test_association_calibration_draws_into_caller_axes_in_the_theme() -> None:
    figure, axes = plt.subplots(1, 3)
    record = _associations_record()

    try:
        result = association_calibration(
            _record_store(record),
            ASSOCIATIONS,
            target={"qq": axes[2], "fold_changes": axes[0], "controls": axes[1]},
            theme="dark",
            show=False,
        )
        assert result.owns_figure is False
        assert result.figure is figure
        assert result.axes == {
            "qq": axes[2],
            "fold_changes": axes[0],
            "controls": axes[1],
        }
        _check_qq(result, record, theme="dark")
        _check_fold_changes(result, record)
        _check_controls(result, record)
        dark = to_rgba(foreground_color("dark"))
        (mean,) = axes[0].get_lines()
        assert to_rgba(mean.get_color()) == dark
        assert all(to_rgba(ax.title.get_color()) == dark for ax in axes)
        assert figure.legends == []
    finally:
        plt.close(figure)


def test_association_calibration_takes_a_sequence_of_axes_but_not_one_axes() -> None:
    figure, axes = plt.subplots(1, 3)
    store = _record_store(_associations_record())

    try:
        result = association_calibration(
            store, ASSOCIATIONS, target=list(axes), show=False
        )
        assert list(result.axes.values()) == list(axes)
        with pytest.raises(TypeError, match="only valid for one-panel plots"):
            association_calibration(store, ASSOCIATIONS, target=axes[0], show=False)
    finally:
        plt.close(figure)


def test_association_calibration_can_drop_the_legends() -> None:
    store = _record_store(_associations_record())

    result = association_calibration(store, ASSOCIATIONS, show_legend=False, show=False)

    try:
        assert result.figure.legends == []
        assert all(ax.get_legend() is None for ax in result.axes.values())
    finally:
        result.close()


@pytest.mark.parametrize(
    "calibration",
    [[], [("NT1", "NPM1", NAN, NAN, NAN), ("NT2", "NCL", NAN, NAN, NAN)]],
)
def test_the_association_calibration_needs_a_tested_negative_control(
    calibration: list[AssociationRow],
) -> None:
    store = _record_store(_associations_record(calibration=calibration))

    with pytest.raises(
        ValueError, match="None of the negative-control pairs passed QC"
    ):
        association_calibration(store, ASSOCIATIONS, show=False)


CLUSTERS = ArtifactRef(
    scope="assay", assay="RNA", kind="perturbation_clusters", artifact_id="9" * 64
)
CLUSTER_HEXES = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#2E8B57",
    "#A05195",
)
UNASSIGNED = to_rgba("#d4d4d4")
DARK_UNASSIGNED = to_rgba("#4d4d4d")
ClusterRow = tuple[str, int, int, float, float]

# Each row is a perturbation, its cluster, its cells and its place in the
# layout, in the order of the profiles. Cluster 0 lies in a row with its
# median at MED13, and the clusters lie far apart.
CLUSTER_ROWS: list[ClusterRow] = [
    ("MED12", 0, 40, 0.0, 0.0),
    ("MED13", 0, 35, 0.3, 0.0),
    ("MED14", 0, 30, 0.6, 0.0),
    ("RPL5", 1, 80, 10.0, 0.0),
    ("RPL7", 1, 75, 10.4, 0.5),
    ("RPL9", 1, 60, 9.8, 0.3),
    ("RPS3", 1, 70, 10.2, -0.4),
    ("MYC", -1, 25, 5.0, 5.0),
    ("TP53", -1, 20, 4.0, -3.0),
    ("CCT2", 2, 50, 0.0, 9.0),
    ("CCT3", 2, 45, 0.4, 9.5),
    ("CCT4", 2, 40, -0.3, 9.2),
]
# The leaf order runs through cluster 2 and then cluster 0, and splits
# cluster 1 around MYC, so that its longest run holds three perturbations.
CLUSTER_ORDER = [
    "CCT3",
    "CCT2",
    "CCT4",
    "MED13",
    "MED12",
    "MED14",
    "RPL5",
    "RPL9",
    "RPS3",
    "MYC",
    "RPL7",
    "TP53",
]


def _clusters_record(
    rows: list[ClusterRow] = CLUSTER_ROWS,
    order: list[str] = CLUSTER_ORDER,
    *,
    metric: str = "correlation",
) -> PerturbationClusters:
    """Build the record that ``load_clusters`` returns for ``rows``.

    The distances are those of random profiles over 30 features, as the
    clustering measures them.
    """
    index = pd.Index([row[0] for row in rows], name="perturbation")
    profiles = np.random.default_rng(0).normal(size=(len(rows), 30))
    return PerturbationClusters(
        table=pd.DataFrame(
            {
                "cluster": np.asarray([row[1] for row in rows], dtype=np.int64),
                "probability": np.asarray([0.0 if row[1] < 0 else 0.9 for row in rows]),
                "n_cells": np.asarray([row[2] for row in rows], dtype=np.int64),
                "x": np.asarray([row[3] for row in rows], dtype=np.float32),
                "y": np.asarray([row[4] for row in rows], dtype=np.float32),
            },
            index=index,
        ),
        order=np.asarray(order),
        distances=pd.DataFrame(
            profile_distances(profiles, metric), index=index, columns=index.copy()
        ),
        metric=metric,
        min_cluster_size=3,
        min_samples=2,
        selection="eom",
        seed=7,
        from_assay="RNA",
    )


def _cluster_hexes(record: PerturbationClusters) -> dict[int, str]:
    """Each cluster's colour, by the order in which it first appears in leaf order."""
    in_order = record.table["cluster"].reindex(record.order).tolist()
    first = list(dict.fromkeys(code for code in in_order if code >= 0))
    return {code: CLUSTER_HEXES[rank % 8] for rank, code in enumerate(first)}


def _drawn_colors(
    result: Any, record: PerturbationClusters, theme: str = "notebook"
) -> np.ndarray:
    """Each drawn perturbation's colour, grey when it is unassigned."""
    hexes = _cluster_hexes(record)
    grey = DARK_UNASSIGNED if theme == "dark" else UNASSIGNED
    return np.asarray(
        [
            to_rgba(hexes[code]) if code >= 0 else grey
            for code in result.tables["perturbations"]["cluster"].tolist()
        ]
    )


def _longest_runs(codes: list[int]) -> dict[int, tuple[float, int]]:
    """Each cluster's longest run of rows, ties to the first: its middle and length."""
    runs: dict[int, tuple[float, int]] = {}
    start = 0
    for stop in range(1, len(codes) + 1):
        if stop == len(codes) or codes[stop] != codes[start]:
            length = stop - start
            if codes[start] >= 0 and length > runs.get(codes[start], (0.0, 0))[1]:
                runs[codes[start]] = ((start + stop - 1) / 2, length)
            start = stop
    return runs


def _check_heatmap(
    result: Any, record: PerturbationClusters, *, theme: str = "notebook"
) -> None:
    """Check that the matrix, the bar and its numbers follow the drawn rows."""
    ax = result.axes["heatmap"]
    bar = result.axes["cluster_bar"]
    drawn = result.tables["perturbations"]
    names = drawn.index.tolist()
    n_drawn = len(names)
    # The drawn perturbations run in leaf order, with their rows of the record.
    assert names == [name for name in record.order.tolist() if name in set(names)]
    pd.testing.assert_frame_equal(drawn, record.table.loc[names])
    values = record.distances.loc[names, names].to_numpy()
    if record.metric == "correlation":
        values = 1.0 - values
    (image,) = ax.images
    np.testing.assert_array_equal(np.asarray(image.get_array()), values)
    pd.testing.assert_frame_equal(
        result.tables["matrix"],
        pd.DataFrame(values, index=drawn.index, columns=drawn.index),
    )
    assert list(image.get_extent()) == [-0.5, n_drawn - 0.5, n_drawn - 0.5, -0.5]
    (strip,) = bar.images
    np.testing.assert_array_equal(
        np.asarray(strip.get_array())[:, 0, :], _drawn_colors(result, record, theme)
    )
    assert bar.get_ylim() == ax.get_ylim() == (n_drawn - 0.5, -0.5)
    # Both images are rasterized, as the record says.
    assert image.get_rasterized() is strip.get_rasterized() is True
    assert result.provenance.extras["rasterized"] is True
    # The bar stands left of the heatmap and as high as it.
    result.figure.draw_without_rendering()
    box = ax.get_window_extent()
    bar_box = bar.get_window_extent()
    assert bar_box.x1 < box.x0
    assert (bar_box.y0, bar_box.y1) == pytest.approx((box.y0, box.y1))
    codes = drawn["cluster"].tolist()
    runs = _longest_runs(codes)
    clusters = result.tables["clusters"]
    assert clusters["cluster"].tolist() == sorted(runs)
    assert clusters["row"].tolist() == [runs[code][0] for code in sorted(runs)]
    assert clusters["n_perturbations"].tolist() == [
        codes.count(code) for code in sorted(runs)
    ]
    # Each kept number sits beside the middle of its cluster's longest run.
    kept = clusters[clusters["numbered"]]
    np.testing.assert_array_equal(bar.get_yticks(), kept["row"])
    assert [label.get_text() for label in bar.get_yticklabels()] == [
        str(code) for code in kept["cluster"]
    ]


def _check_map(
    result: Any, record: PerturbationClusters, *, theme: str = "notebook"
) -> None:
    """Check that each drawn perturbation is a point at its place and colour."""
    ax = result.axes["embedding"]
    drawn = result.tables["perturbations"]
    names = drawn.index.tolist()
    assert names == [name for name in record.order.tolist() if name in set(names)]
    pd.testing.assert_frame_equal(drawn, record.table.loc[names])
    x = drawn["x"].to_numpy(dtype=np.float64)
    y = drawn["y"].to_numpy(dtype=np.float64)
    colors = _drawn_colors(result, record, theme)
    assigned = drawn["cluster"].to_numpy() >= 0
    # The unassigned perturbations are drawn first, beneath the clusters.
    groups = [
        (chosen, order)
        for chosen, order in ((~assigned, 2.0), (assigned, 3.0))
        if chosen.any()
    ]
    size = default_point_size(len(drawn), size_min=6.0, size_max=30.0)
    for collection, (chosen, order) in zip(ax.collections, groups, strict=False):
        np.testing.assert_array_equal(
            np.asarray(collection.get_offsets()), np.column_stack([x, y])[chosen]
        )
        np.testing.assert_array_equal(collection.get_facecolors(), colors[chosen])
        np.testing.assert_array_equal(collection.get_sizes(), [size])
        assert collection.get_zorder() == order
        assert collection.get_rasterized() is result.provenance.extras["rasterized"]
    # The window is square and holds every point.
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    assert right - left == pytest.approx(top - bottom)
    assert ((left < x) & (x < right) & (bottom < y) & (y < top)).all()
    members = drawn[assigned].groupby("cluster")
    clusters = result.tables["clusters"]
    assert clusters["cluster"].tolist() == sorted(members.groups)
    np.testing.assert_array_equal(clusters["n_perturbations"], members.size())
    np.testing.assert_array_equal(
        clusters[["x", "y"]].to_numpy(), members[["x", "y"]].median().to_numpy()
    )


def _check_names_apart(result: Any) -> None:
    """Check that names lie inside the axes, apart and clear of every number."""
    ax = result.axes["embedding"]
    figure = result.figure
    figure.draw_without_rendering()
    frame = ax.get_window_extent()
    gap = 2.0 * figure.dpi / 72.0
    names = [Text.get_window_extent(t) for t in ax.texts if isinstance(t, Annotation)]
    numbers = [t.get_window_extent() for t in ax.texts if not isinstance(t, Annotation)]

    def apart(box: Any, other: Any, padding: float) -> bool:
        # Half a pixel of slack absorbs rounding.
        return (
            box.x1 <= other.x0 - padding + 0.5
            or other.x1 + padding <= box.x0 + 0.5
            or box.y1 <= other.y0 - padding + 0.5
            or other.y1 + padding <= box.y0 + 0.5
        )

    for index, box in enumerate(names):
        assert frame.x0 - 0.5 <= box.x0 and box.x1 <= frame.x1 + 0.5
        assert frame.y0 - 0.5 <= box.y0 and box.y1 <= frame.y1 + 0.5
        assert all(apart(box, other, 0.0) for other in names[index + 1 :])
        assert all(apart(box, number, gap) for number in numbers)


def test_cluster_heatmap_draws_the_correlations_in_leaf_order() -> None:
    requests: list[ArtifactRef] = []
    record = _clusters_record()

    result = perturbation_clusters(
        _record_store(record, requests), CLUSTERS, show=False
    )

    try:
        assert requests == [CLUSTERS]
        assert list(result.axes) == ["heatmap", "cluster_bar", "colorbar"]
        _check_heatmap(result, record)
        ax = result.axes["heatmap"]
        (image,) = ax.images
        assert image.get_cmap().name == "RdBu_r"
        assert image.get_clim() == (-1.0, 1.0)
        # Twelve perturbations are few enough to name, beside the rows on the
        # right and below the columns.
        assert [label.get_text() for label in ax.get_yticklabels()] == CLUSTER_ORDER
        assert [label.get_text() for label in ax.get_xticklabels()] == CLUSTER_ORDER
        assert ax.yaxis.get_ticks_position() == "right"
        assert {label.get_rotation() for label in ax.get_xticklabels()} == {90.0}
        # Cluster 2 comes first in the leaf order and takes the first colour.
        # Cluster 1's number sits beside its run of three, not beside RPL7.
        assert result.tables["clusters"].to_dict("list") == {
            "cluster": [0, 1, 2],
            "n_perturbations": [3, 4, 3],
            "row": [4.0, 7.0, 1.0],
            "numbered": [True, True, True],
        }
        colorbar = result.axes["colorbar"]
        assert colorbar.get_ylabel() == "Pearson correlation"
        np.testing.assert_allclose(colorbar.get_yticks(), [-1.0, -0.5, 0.0, 0.5, 1.0])
        assert ax.get_title().splitlines() == [
            "Correlation between perturbation profiles",
            "12 perturbations: 3 clusters, 2 unassigned in grey",
        ]
        # The plot's own figure makes the heatmap's box square.
        box = ax.get_window_extent()
        assert box.width == pytest.approx(box.height, abs=2.0)
        assert result.legends == (
            LegendSpec(
                kind="colorbar",
                label="Pearson correlation",
                extras={"vmin": -1.0, "vmax": 1.0},
            ),
            LegendSpec(kind="categorical", label="Cluster"),
        )
        assert result.scales == (
            ColorScale(cmap="RdBu_r", vmin=-1.0, vmax=1.0, vcenter=0.0),
            CategoricalScale(
                order=(0, 1, 2),
                palette={0: "#E69F00", 1: "#009E73", 2: "#0072B2"},
                missing_color="#d4d4d4",
                missing_label="Unassigned",
            ),
        )
        provenance = result.provenance
        assert provenance.assay == "RNA"
        assert provenance.n_cells == sum(row[2] for row in CLUSTER_ROWS)
        assert provenance.notes == ("perturbation_clusters",)
        assert provenance.extras == {
            "clusters": CLUSTERS.to_dict(),
            "panel": "heatmap",
            "metric": "correlation",
            "min_cluster_size": 3,
            "min_samples": 2,
            "selection": "eom",
            "seed": 7,
            "from_assay": "RNA",
            "perturbations": None,
            "cluster_ids": None,
            "label": None,
            "n_perturbations": 12,
            "n_clustered": 12,
            "n_clusters": 3,
            "n_unassigned": 2,
            "named": True,
            "color_limits": [-1.0, 1.0],
            "rasterized": True,
        }
    finally:
        result.close()


def test_cluster_heatmap_keeps_the_chosen_perturbations_or_clusters() -> None:
    record = _clusters_record()
    store = _record_store(record)

    named = perturbation_clusters(
        store, CLUSTERS, perturbations=["MYC", "CCT2", "RPL7", "CCT2"], show=False
    )
    chosen = perturbation_clusters(
        store, CLUSTERS, cluster_ids=[np.int64(1), -1, 1], show=False
    )

    try:
        # The named perturbations run in leaf order, once each, and keep the
        # colours of their clusters.
        assert named.tables["perturbations"].index.tolist() == ["CCT2", "MYC", "RPL7"]
        _check_heatmap(named, record)
        assert named.axes["heatmap"].get_title().splitlines()[1] == (
            "3 of 12 perturbations: 2 clusters, 1 unassigned in grey"
        )
        assert named.provenance.extras["perturbations"] == ["MYC", "CCT2", "RPL7"]
        assert named.provenance.extras["n_perturbations"] == 3
        assert named.provenance.n_cells == 50 + 25 + 75
        assert chosen.tables["perturbations"].index.tolist() == [
            "RPL5",
            "RPL9",
            "RPS3",
            "MYC",
            "RPL7",
            "TP53",
        ]
        _check_heatmap(chosen, record)
        assert chosen.tables["clusters"]["row"].tolist() == [1.0]
        assert chosen.axes["heatmap"].get_title().splitlines()[1] == (
            "6 of 12 perturbations: 1 cluster, 2 unassigned in grey"
        )
        assert chosen.provenance.extras["cluster_ids"] == [1, -1]
        assert all(
            type(code) is int for code in chosen.provenance.extras["cluster_ids"]
        )
    finally:
        named.close()
        chosen.close()


def test_cluster_heatmap_by_euclidean_distance() -> None:
    record = _clusters_record(metric="euclidean")
    store = _record_store(record)

    every = perturbation_clusters(store, CLUSTERS, show=False)
    alone = perturbation_clusters(store, CLUSTERS, perturbations=["MYC"], show=False)

    try:
        _check_heatmap(every, record)
        (image,) = every.axes["heatmap"].images
        largest = float(record.distances.to_numpy().max())
        assert image.get_cmap().name == "Blues_r"
        assert image.get_clim() == (0.0, largest)
        assert every.axes["colorbar"].get_ylabel() == "Euclidean distance"
        assert every.axes["heatmap"].get_title().splitlines()[0] == (
            "Euclidean distance between perturbation profiles"
        )
        assert every.scales[0] == ColorScale(cmap="Blues_r", vmin=0.0, vmax=largest)
        assert every.provenance.extras["metric"] == "euclidean"
        assert every.provenance.extras["color_limits"] == [0.0, largest]
        # A lone perturbation sits at distance 0 from itself, so the scale
        # runs to 1, and an unassigned one has no cluster to number.
        _check_heatmap(alone, record)
        assert alone.axes["heatmap"].images[0].get_clim() == (0.0, 1.0)
        assert alone.tables["clusters"].empty
        assert alone.axes["heatmap"].get_title().splitlines()[1] == (
            "1 of 12 perturbations: 0 clusters, 1 unassigned in grey"
        )
    finally:
        every.close()
        alone.close()


@pytest.mark.parametrize(("n_drawn", "named"), [(50, True), (51, False)])
def test_cluster_heatmap_names_at_most_50_perturbations(
    n_drawn: int, named: bool
) -> None:
    rows: list[ClusterRow] = [
        (f"P{index:02d}", index // 10, 30, float(index), 0.0)
        for index in range(n_drawn)
    ]
    record = _clusters_record(rows, [row[0] for row in rows])

    result = perturbation_clusters(_record_store(record), CLUSTERS, show=False)

    try:
        _check_heatmap(result, record)
        ax = result.axes["heatmap"]
        names = [row[0] for row in rows] if named else []
        assert [label.get_text() for label in ax.get_yticklabels()] == names
        assert [label.get_text() for label in ax.get_xticklabels()] == names
        assert result.provenance.extras["named"] is named
        box = ax.get_window_extent()
        assert box.width == pytest.approx(box.height, abs=2.0)
    finally:
        result.close()


def test_cluster_heatmap_numbers_longer_runs_first_without_overlaps() -> None:
    # Eighty clusters of 2 to 8 perturbations in 400 rows, too many to number
    # each.
    codes = np.repeat(np.arange(80), np.arange(80) % 7 + 2)
    rows: list[ClusterRow] = [
        (f"P{index:03d}", int(code), 30, float(index), 0.0)
        for index, code in enumerate(codes)
    ]
    record = _clusters_record(rows, [row[0] for row in rows])

    result = perturbation_clusters(_record_store(record), CLUSTERS, show=False)

    try:
        _check_heatmap(result, record)
        clusters = result.tables["clusters"]
        numbered = clusters["numbered"].to_numpy()
        assert 0 < numbered.sum() < len(clusters)
        bar = result.axes["cluster_bar"]
        boxes = [Text.get_window_extent(label) for label in bar.get_yticklabels()]
        height = max(box.height for box in boxes)
        # No two numbers overlap.
        tops = sorted((box.y0, box.y1) for box in boxes)
        assert all(
            low[1] <= high[0] + 0.5 for low, high in zip(tops, tops[1:], strict=False)
        )
        # A number is left out only where it would overlap the number of a
        # longer run, or of a run as long and higher up.
        rows_drawn = clusters["row"].to_numpy()
        lengths = clusters["n_perturbations"].to_numpy()
        middles = bar.transData.transform(
            np.column_stack([np.zeros(len(rows_drawn)), rows_drawn])
        )[:, 1]
        for index in np.flatnonzero(~numbered):
            assert any(
                abs(middles[other] - middles[index]) < height
                and (lengths[other], -rows_drawn[other])
                > (lengths[index], -rows_drawn[index])
                for other in np.flatnonzero(numbered)
            )
    finally:
        result.close()


def test_cluster_heatmap_keeps_a_chosen_figure_size() -> None:
    record = _clusters_record()

    result = perturbation_clusters(
        _record_store(record), CLUSTERS, figsize=(8.0, 5.0), show=False
    )

    try:
        assert tuple(result.figure.get_size_inches()) == (8.0, 5.0)
        _check_heatmap(result, record)
    finally:
        result.close()


def test_clusters_take_eight_colours_in_leaf_order_and_keep_them() -> None:
    # Ten clusters of two, which first appear in the leaf order from 9 down.
    rows: list[ClusterRow] = [
        (f"P{index:02d}", index // 2, 30, float(index), 0.0) for index in range(20)
    ]
    record = _clusters_record(rows, [row[0] for row in reversed(rows)])
    store = _record_store(record)

    every = perturbation_clusters(store, CLUSTERS, show=False)
    some = perturbation_clusters(store, CLUSTERS, cluster_ids=[0, 1], show=False)

    try:
        _check_heatmap(every, record)
        _check_heatmap(some, record)
        # The ninth cluster to appear, 1, takes the first colour again.
        assert every.scales[1].palette == {
            9: "#0072B2",
            8: "#E69F00",
            7: "#009E73",
            6: "#D55E00",
            5: "#CC79A7",
            4: "#56B4E9",
            3: "#2E8B57",
            2: "#A05195",
            1: "#0072B2",
            0: "#E69F00",
        }
        # The drawn clusters keep their colours.
        assert some.scales[1].palette == {0: "#E69F00", 1: "#0072B2"}
    finally:
        every.close()
        some.close()


def test_cluster_map_draws_each_perturbation_coloured_by_cluster() -> None:
    requests: list[ArtifactRef] = []
    record = _clusters_record()

    result = perturbation_clusters(
        _record_store(record, requests), CLUSTERS, panel="embedding", show=False
    )

    try:
        assert requests == [CLUSTERS]
        assert list(result.axes) == ["embedding"]
        _check_map(result, record)
        ax = result.axes["embedding"]
        assert len(ax.collections) == 2
        assert result.tables["clusters"].to_dict("list") == {
            "cluster": [0, 1, 2],
            "n_perturbations": [3, 4, 3],
            "x": pytest.approx([0.3, 10.1, 0.0]),
            "y": pytest.approx([0.0, 0.15, 9.2]),
        }
        # Each cluster is numbered at its median, which lie far apart.
        numbers = list(ax.texts)
        assert [number.get_text() for number in numbers] == ["0", "1", "2"]
        for number, (x, y) in zip(
            numbers, [(0.3, 0.0), (10.1, 0.15), (0.0, 9.2)], strict=True
        ):
            assert number.get_position() == pytest.approx((x, y), abs=1e-6)
            assert to_rgba(number.get_color()) == to_rgba("#222222")
            assert number.get_path_effects()
        assert ax.get_title().splitlines() == [
            "UMAP of perturbation profiles",
            "12 perturbations: 3 clusters, 2 unassigned in grey",
        ]
        assert ax.get_xticks().size == 0 and ax.get_yticks().size == 0
        assert result.legends == (LegendSpec(kind="categorical", label="Cluster"),)
        assert result.scales == (
            CategoricalScale(
                order=(0, 1, 2),
                palette={0: "#E69F00", 1: "#009E73", 2: "#0072B2"},
                missing_color="#d4d4d4",
                missing_label="Unassigned",
            ),
        )
        extras = result.provenance.extras
        assert extras["panel"] == "embedding"
        assert extras["rasterized"] is False
        assert "named" not in extras
    finally:
        result.close()


def test_cluster_map_names_chosen_perturbations_clear_of_the_numbers() -> None:
    record = _clusters_record()

    result = perturbation_clusters(
        _record_store(record),
        CLUSTERS,
        panel="embedding",
        label=["MED12", "TP53", "RPL7", "MED12"],
        show=False,
    )

    try:
        _check_map(result, record)
        ax = result.axes["embedding"]
        named = ["MED12", "TP53", "RPL7"]
        places = record.table.loc[named, ["x", "y"]].to_numpy(dtype=np.float64)
        # Each named perturbation gets a ring around its point.
        assert len(ax.collections) == 3
        ring = ax.collections[2]
        np.testing.assert_array_equal(np.asarray(ring.get_offsets()), places)
        assert len(ring.get_facecolors()) == 0
        np.testing.assert_array_equal(
            ring.get_edgecolors(), [to_rgba(foreground_color("notebook"))]
        )
        size = default_point_size(12, size_min=6.0, size_max=30.0)
        np.testing.assert_array_equal(ring.get_sizes(), [4.0 * size])
        labels = [text for text in ax.texts if isinstance(text, Annotation)]
        assert [label.get_text() for label in labels] == named
        for label, place in zip(labels, places.tolist(), strict=True):
            assert label.xy == pytest.approx(place)
            side = "left" if label.xyann[0] > 0 else "right"
            assert label.get_horizontalalignment() == side
            assert label.arrow_patch.get_visible() == (label.xyann[1] != 0)
        # MED12's name would start right of it, over the number of cluster 0
        # at MED13, so it moves.
        assert labels[0].xyann != (5.0, 0.0)
        _check_names_apart(result)
        assert result.provenance.extras["label"] == named
    finally:
        result.close()


@pytest.mark.parametrize(
    ("cluster_ids", "drawn", "n_texts"),
    [([2], ["CCT3", "CCT2", "CCT4"], 1), ([-1], ["MYC", "TP53"], 0)],
)
def test_cluster_map_keeps_the_chosen_clusters(
    cluster_ids: list[int], drawn: list[str], n_texts: int
) -> None:
    record = _clusters_record()

    result = perturbation_clusters(
        _record_store(record),
        CLUSTERS,
        panel="embedding",
        cluster_ids=cluster_ids,
        show=False,
    )

    try:
        assert result.tables["perturbations"].index.tolist() == drawn
        _check_map(result, record)
        ax = result.axes["embedding"]
        assert len(ax.collections) == 1
        assert len(ax.texts) == n_texts
    finally:
        result.close()


@pytest.mark.parametrize("panel", ["heatmap", "embedding"])
def test_cluster_plots_without_a_cluster_draw_every_perturbation_grey(
    panel: str,
) -> None:
    loners: list[ClusterRow] = [
        (name, -1, cells, x, y) for name, _, cells, x, y in CLUSTER_ROWS
    ]
    record = _clusters_record(loners)

    result = perturbation_clusters(
        _record_store(record), CLUSTERS, panel=panel, show=False
    )

    try:
        ax = result.axes[panel]
        assert ax.get_title().splitlines()[1] == (
            "12 perturbations: 0 clusters, 12 unassigned in grey"
        )
        assert result.tables["clusters"].empty
        assert result.scales[-1].order == ()
        if panel == "heatmap":
            _check_heatmap(result, record)
            assert result.axes["cluster_bar"].get_yticks().size == 0
        else:
            _check_map(result, record)
            assert len(ax.texts) == 0
    finally:
        result.close()


@pytest.mark.parametrize("panel", ["heatmap", "embedding"])
def test_cluster_plots_draw_into_caller_axes_in_the_theme(panel: str) -> None:
    figure, ax = plt.subplots(figsize=(5.0, 4.0))
    record = _clusters_record()

    try:
        result = perturbation_clusters(
            _record_store(record),
            CLUSTERS,
            panel=panel,
            label=["MYC"] if panel == "embedding" else None,
            target=ax,
            theme="dark",
            show=False,
        )
        assert result.owns_figure is False
        assert result.figure is figure
        assert result.axes[panel] is ax
        # The plot keeps the caller's figure as it is.
        assert tuple(figure.get_size_inches()) == (5.0, 4.0)
        dark = to_rgba(foreground_color("dark"))
        assert to_rgba(ax.title.get_color()) == dark
        if panel == "heatmap":
            assert list(result.axes) == ["heatmap", "cluster_bar", "colorbar"]
            _check_heatmap(result, record, theme="dark")
        else:
            _check_map(result, record, theme="dark")
            colors = {to_rgba(text.get_color()) for text in ax.texts}
            assert colors == {to_rgba("#f5f5f5"), dark}
            np.testing.assert_array_equal(ax.collections[-1].get_edgecolors(), [dark])
    finally:
        plt.close(figure)


@pytest.mark.parametrize("panel", ["heatmap", "embedding"])
def test_cluster_plots_without_a_perturbation_draw_nothing(panel: str) -> None:
    record = _clusters_record()
    empty = PerturbationClusters(
        table=record.table.iloc[:0],
        order=record.order[:0],
        distances=record.distances.iloc[:0, :0],
        metric=record.metric,
        min_cluster_size=record.min_cluster_size,
        min_samples=record.min_samples,
        selection=record.selection,
        seed=record.seed,
        from_assay=record.from_assay,
    )
    figures = set(plt.get_fignums())

    with pytest.raises(
        ValueError,
        match="^The clusters hold no perturbation, so there is nothing to plot$",
    ):
        perturbation_clusters(_record_store(empty), CLUSTERS, panel=panel, show=False)
    assert set(plt.get_fignums()) == figures


@pytest.mark.parametrize("panel", ["heatmap", "embedding"])
def test_cluster_plots_can_drop_the_legend(panel: str) -> None:
    store = _record_store(_clusters_record())

    plain = perturbation_clusters(
        store, CLUSTERS, panel=panel, show_legend=False, show=False
    )
    legended = perturbation_clusters(store, CLUSTERS, panel=panel, show=False)

    try:
        if panel == "heatmap":
            assert list(plain.axes) == ["heatmap", "cluster_bar"]
            # The bar is a child of the heatmap's axes, and no colorbar is
            # drawn.
            assert plain.figure.axes == [plain.axes["heatmap"]]
            assert plain.axes["heatmap"].child_axes == [plain.axes["cluster_bar"]]
            assert plain.axes["cluster_bar"].get_yticks().size == 0
            assert not plain.tables["clusters"]["numbered"].any()
            # Only a figure with a colorbar leaves room for it at the right.
            plain_width = plain.figure.get_size_inches()[0]
            assert plain_width < legended.figure.get_size_inches()[0]
        else:
            assert len(plain.axes["embedding"].texts) == 0
            assert len(legended.axes["embedding"].texts) == 3
    finally:
        plain.close()
        legended.close()


def test_cluster_map_rasterizes_more_than_50000_points() -> None:
    n_perturbations = 50_001
    names = [f"P{index}" for index in range(n_perturbations)]
    angles = np.linspace(0.0, 2.0 * np.pi, n_perturbations)
    index = pd.Index(names, name="perturbation")
    # The map reads no distances, so this record holds none.
    record = PerturbationClusters(
        table=pd.DataFrame(
            {
                "cluster": np.arange(n_perturbations) % 3 - 1,
                "probability": np.ones(n_perturbations),
                "n_cells": np.full(n_perturbations, 30),
                "x": np.cos(angles).astype(np.float32),
                "y": np.sin(angles).astype(np.float32),
            },
            index=index,
        ),
        order=np.asarray(names),
        distances=pd.DataFrame(),
        metric="correlation",
        min_cluster_size=3,
        min_samples=2,
        selection="eom",
        seed=0,
        from_assay="RNA",
    )
    store = _record_store(record)

    every = perturbation_clusters(store, CLUSTERS, panel="embedding", show=False)
    fewer = perturbation_clusters(
        store, CLUSTERS, panel="embedding", perturbations=names[1:], show=False
    )

    try:
        assert every.provenance.extras["rasterized"] is True
        collections = every.axes["embedding"].collections
        assert len(collections) == 2 and all(c.get_rasterized() for c in collections)
        assert fewer.provenance.extras["rasterized"] is False
        assert not any(c.get_rasterized() for c in fewer.axes["embedding"].collections)
    finally:
        every.close()
        fewer.close()


def test_showing_the_cluster_plots_releases_their_figures() -> None:
    store = _record_store(_clusters_record())

    heatmap = perturbation_clusters(store, CLUSTERS)
    embedding = perturbation_clusters(store, CLUSTERS, panel="embedding")

    assert heatmap.tables["perturbations"].index.tolist() == CLUSTER_ORDER
    assert embedding.tables["clusters"]["cluster"].tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"panel": "scores"}, ValueError, "panel must be 'heatmap' or 'embedding'"),
        ({"panel": None}, ValueError, "panel must be 'heatmap' or 'embedding'"),
        (
            {"perturbations": "MYC"},
            TypeError,
            "perturbations must be a sequence of perturbations",
        ),
        (
            {"perturbations": [1]},
            TypeError,
            "perturbations must be a sequence of perturbations",
        ),
        (
            {"perturbations": []},
            ValueError,
            "perturbations must name at least one perturbation",
        ),
        ({"cluster_ids": "0"}, TypeError, "cluster_ids must be a sequence of cluster"),
        ({"cluster_ids": b"\x00"}, TypeError, "cluster_ids must be a sequence"),
        ({"cluster_ids": [True]}, TypeError, "cluster_ids must be a sequence"),
        ({"cluster_ids": [1.0]}, TypeError, "cluster_ids must be a sequence"),
        ({"cluster_ids": np.array([0])}, TypeError, "cluster_ids must be a sequence"),
        (
            {"cluster_ids": []},
            ValueError,
            "^cluster_ids must name at least one cluster number$",
        ),
        (
            {"perturbations": ["MYC"], "cluster_ids": [0]},
            ValueError,
            "Pass perturbations or cluster_ids, not both",
        ),
        (
            {"label": ["MYC"]},
            ValueError,
            "label names perturbations in the embedding; the heatmap names every "
            "perturbation when it draws at most 50",
        ),
        (
            {"panel": "embedding", "label": "MYC"},
            TypeError,
            "label must be a sequence of perturbations",
        ),
        (
            {"panel": "embedding", "label": []},
            ValueError,
            "label must name at least one perturbation",
        ),
    ],
)
def test_the_cluster_plot_refuses_settings_it_cannot_use(
    arguments: dict[str, Any], error: type[Exception], message: str
) -> None:
    requests: list[ArtifactRef] = []
    store = _record_store(_clusters_record(), requests)

    with pytest.raises(error, match=message):
        perturbation_clusters(store, CLUSTERS, show=False, **arguments)
    # The settings are checked before the record is read.
    assert requests == []


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {"perturbations": ["MYC", "GATA1"]},
            "^The clusters have no perturbation named 'GATA1'$",
        ),
        (
            {"perturbations": [f"G{index}" for index in range(7)]},
            "named 'G0', 'G1', 'G2', 'G3', 'G4' and 2 more$",
        ),
        ({"panel": "embedding", "label": ["GATA1"]}, "no perturbation named 'GATA1'"),
        (
            {"cluster_ids": [7]},
            "^No perturbation is in cluster 7; the clusters run from 0 to 2, with -1 "
            "for unassigned perturbations$",
        ),
        ({"cluster_ids": [0, 7, 9]}, "^No perturbation is in clusters 7, 9;"),
        (
            {"panel": "embedding", "cluster_ids": [0], "label": ["MYC", "RPL5"]},
            "^label names perturbations that the plot does not draw: 'MYC', 'RPL5'$",
        ),
    ],
)
def test_the_cluster_plot_names_what_the_clusters_lack(
    arguments: dict[str, Any], message: str
) -> None:
    store = _record_store(_clusters_record())

    with pytest.raises(ValueError, match=message):
        perturbation_clusters(store, CLUSTERS, show=False, **arguments)


def test_the_cluster_plot_names_the_clusters_of_a_record_without_some() -> None:
    assigned = [row for row in CLUSTER_ROWS if row[1] >= 0]
    order = [name for name in CLUSTER_ORDER if name not in ("MYC", "TP53")]
    loners: list[ClusterRow] = [
        (name, -1, cells, x, y) for name, _, cells, x, y in CLUSTER_ROWS
    ]

    with pytest.raises(
        ValueError, match="^No perturbation is in cluster -1; the clusters run from 0"
    ):
        perturbation_clusters(
            _record_store(_clusters_record(assigned, order)),
            CLUSTERS,
            cluster_ids=[-1],
            show=False,
        )
    with pytest.raises(
        ValueError,
        match="^No perturbation is in cluster 0; every perturbation is unassigned, "
        "in cluster -1$",
    ):
        perturbation_clusters(
            _record_store(_clusters_record(loners)),
            CLUSTERS,
            cluster_ids=[0],
            show=False,
        )


# Each screen figure, a record it plots, and the loader that reads the record.
SCREEN_LOADERS = [
    (guide_calls, CALLS, "load_guide_calls"),
    (guide_thresholds, THRESHOLDS, "load_guide_thresholds"),
    (efficiency, EFFICIENCY, "load_efficiency"),
    (edistance, EDISTANCE, "load_edistance"),
    (mixscape, MIXSCAPE, "load_mixscape"),
    (association_volcano, ASSOCIATIONS, "load_associations"),
    (association_calibration, ASSOCIATIONS, "load_associations"),
    (perturbation_clusters, CLUSTERS, "load_clusters"),
]


@pytest.mark.parametrize(("plot", "ref", "loader"), SCREEN_LOADERS)
def test_screen_figures_need_a_store_with_their_loader(
    plot: Any, ref: ArtifactRef, loader: str
) -> None:
    # Every other figure's loader, which a figure that asks for the wrong
    # loader would call.
    others = {name: lambda ref: ref for _, _, name in SCREEN_LOADERS if name != loader}
    stores = [
        SimpleNamespace(),
        SimpleNamespace(perturbation=SimpleNamespace(**others)),
        SimpleNamespace(perturbation=SimpleNamespace(**{loader: "not a loader"})),
    ]

    for store in stores:
        with pytest.raises(
            TypeError, match=rf"^store does not provide perturbation\.{loader}$"
        ):
            plot(store, ref, show=False)


@pytest.mark.parametrize("plot", [plot for plot, _, _ in SCREEN_LOADERS])
def test_the_store_accessor_summarizes_each_screen_figure_as_it_does(
    plot: Any,
) -> None:
    from cytearc.datastore.plot_accessor import DataStorePlotAccessor

    method = getattr(DataStorePlotAccessor, plot.__name__)

    assert inspect.getdoc(method) == inspect.getdoc(plot).splitlines()[0]


@pytest.mark.parametrize("panel", ["efficiency", "edistance"])
def test_a_row_whose_adjusted_p_value_is_alpha_is_significant(panel: str) -> None:
    # KLF1's adjusted p-value is 0.01, so it is significant at an FDR of 0.01,
    # as a pair of the association tests would be.
    result = (
        efficiency(
            _record_store(_efficiency_record(TARGET_ROWS)),
            EFFICIENCY,
            alpha=0.01,
            show=False,
        )
        if panel == "efficiency"
        else edistance(
            _record_store(_edistance_record(EDISTANCE_TARGETS)),
            EDISTANCE,
            alpha=0.01,
            show=False,
        )
    )

    try:
        table = result.tables[panel]
        heights = {
            float(y): to_rgba(face)
            for collection in result.axes[panel].collections
            for (_, y), face in zip(
                np.asarray(collection.get_offsets()),
                np.broadcast_to(
                    collection.get_facecolors(), (len(collection.get_offsets()), 4)
                ),
                strict=True,
            )
        }
        row = int(np.flatnonzero(table["target"].to_numpy() == "KLF1")[0])
        assert table["adjusted_p_value"].iloc[row] == 0.01
        assert heights[float(len(table) - 1 - row)] == SIGNIFICANT
        texts = [text.get_text() for text in result.figure.legends[0].get_texts()]
        assert texts[:2] == ["Significant at FDR 0.01", "Not significant"]
    finally:
        result.close()


def test_screen_figures_take_numpy_numbers() -> None:
    results = {
        "max_guides_per_cell": guide_calls(
            _store(5),
            CALLS,
            max_guides_per_cell=np.int64(3),
            bins=np.int32(5),
            show=False,
        ),
        "max_panels": guide_thresholds(
            _threshold_store("poisson_gaussian", ONE_SAMPLE, min_umi=None),
            THRESHOLDS,
            guides=["g1"],
            max_panels=np.int64(12),
            show=False,
        ),
        "max_rows": efficiency(
            _record_store(_efficiency_record(TARGET_ROWS)),
            EFFICIENCY,
            max_rows=np.int64(2),
            alpha=np.float64(0.01),
            show=False,
        ),
        "alpha": edistance(
            _record_store(_edistance_record(EDISTANCE_TARGETS)),
            EDISTANCE,
            max_rows=np.int16(3),
            alpha=np.float64(0.01),
            show=False,
        ),
        "max_targets": mixscape(
            _record_store(_mixscape_record(MIXSCAPE_TARGETS)),
            MIXSCAPE,
            max_targets=np.int64(2),
            show=False,
        ),
        "label_top": association_volcano(
            _record_store(_associations_record()),
            ASSOCIATIONS,
            label_top=np.int64(2),
            show=False,
        ),
    }

    try:
        # Each figure records its settings as Python numbers.
        recorded = {
            name: result.provenance.extras[name] for name, result in results.items()
        }
        assert recorded == {
            "max_guides_per_cell": 3,
            "max_panels": 12,
            "max_rows": 2,
            "alpha": 0.01,
            "max_targets": 2,
            "label_top": 2,
        }
        assert {type(value) for value in recorded.values()} == {int, float}
        assert len(results["max_rows"].tables["efficiency"]) == 2
        assert len(results["alpha"].tables["edistance"]) == 3
        calls = results["max_guides_per_cell"]
        guides = calls.tables["guides_per_cell"]["guides"].tolist()
        assert guides == ["0", "1", "2", "3+"]
        # The guide calls also record their bins, which split the top guide's
        # share into five.
        assert len(calls.tables["top_fraction"]) == 5
        bins = calls.provenance.extras["bins"]
        assert bins == 5 and type(bins) is int
    finally:
        for result in results.values():
            result.close()


# The legends that each screen figure draws into caller axes, by the position
# of their axes, which a composed figure keeps.
COMPOSED_LEGENDS: dict[str, dict[int, list[str]]] = {
    "guide_calls": {
        1: ["Only guide called", "One of several called"],
        2: ["One guide called", "Several guides called"],
    },
    "guide_threshold_panels": {
        0: [
            "Cells called",
            "Cells not called",
            "Posterior of the guide population",
            "Threshold of the model",
        ]
    },
    "guide_threshold_overview": {0: ["fitted", "other statuses"]},
    "efficiency": {
        0: [
            "Significant at FDR 0.05",
            "Not significant",
            "No counts in the perturbed cells",
            "12 perturbed cells",
            "30 perturbed cells",
            "40 perturbed cells",
        ]
    },
    "edistance": {
        0: [
            "Significant at FDR 0.05",
            "Not significant",
            "12 perturbed cells",
            "30 perturbed cells",
            "40 perturbed cells",
        ]
    },
    "mixscape_classes": {0: MIXSCAPE_LEGEND[:2]},
    "mixscape_scores": {1: MIXSCAPE_LEGEND},
    "association_volcano": {0: EDGE_LEGEND},
    "association_calibration": {
        0: [
            "Negative controls",
            "Discovery pairs",
            "Null expectation",
            "95% null band",
            "Beyond the axis",
        ],
        2: ["p-value below 1e-300"],
    },
}


def _draw_screen_figure(name: str, axes: list[Any]) -> Any:
    """Draw the screen figure ``name`` into the caller's ``axes``."""
    if name == "guide_calls":
        return guide_calls(_store(5), CALLS, target=axes, show=False)
    if name.startswith("guide_threshold"):
        store = _threshold_store("poisson_gaussian", ONE_SAMPLE, min_umi=None)
        if name == "guide_threshold_overview":
            return guide_thresholds(store, THRESHOLDS, target=axes[:2], show=False)
        return guide_thresholds(
            store, THRESHOLDS, guides=["g2", "g1"], target=axes[:2], show=False
        )
    if name == "efficiency":
        store = _record_store(_efficiency_record(TARGET_ROWS))
        return efficiency(store, EFFICIENCY, target=axes[0], show=False)
    if name == "edistance":
        store = _record_store(_edistance_record(EDISTANCE_TARGETS))
        return edistance(store, EDISTANCE, target=axes[0], show=False)
    if name.startswith("mixscape"):
        store = _record_store(_mixscape_record(MIXSCAPE_TARGETS))
        if name == "mixscape_classes":
            return mixscape(store, MIXSCAPE, target=axes[0], show=False)
        return mixscape(
            store,
            MIXSCAPE,
            panel="scores",
            targets=["TP53", "MYC"],
            target={"TP53": axes[1], "MYC": axes[2]},
            show=False,
        )
    if name == "association_volcano":
        store = _record_store(_associations_record(EDGE_ROWS))
        return association_volcano(store, ASSOCIATIONS, target=axes[0], show=False)
    return association_calibration(
        _record_store(_associations_record()),
        ASSOCIATIONS,
        target=axes,
        show=False,
    )


@pytest.mark.parametrize("name", list(COMPOSED_LEGENDS))
def test_composed_screen_figures_keep_their_legends(name: str) -> None:
    figure, row = plt.subplots(1, 3, figsize=(12.0, 4.0), layout="constrained")
    axes = list(row)

    try:
        result = _draw_screen_figure(name, axes)
        composite = compose_results(figure, {name: result})

        legends = {
            position: [text.get_text() for text in axis.get_legend().get_texts()]
            for position, axis in enumerate(axes)
            if axis.get_legend() is not None
        }
        assert legends == COMPOSED_LEGENDS[name]
        # The legends stay in the panels, and none is drawn again beside them.
        assert figure.legends == []
        assert composite.legends == result.legends
    finally:
        plt.close(figure)


@pytest.mark.parametrize("panel", ["heatmap", "embedding"])
def test_composed_cluster_plots_share_their_cluster_legend(panel: str) -> None:
    figure, ax = plt.subplots(figsize=(6.0, 5.0), layout="constrained")

    try:
        result = perturbation_clusters(
            _record_store(_clusters_record()),
            CLUSTERS,
            panel=panel,
            target=ax,
            show=False,
        )
        child_colorbar = result.axes.get("colorbar")
        compose_results(figure, [result], panel_labels=False)

        # Every legend of the clusters has a scale, so the composed figure
        # draws the clusters beside the panel and the colorbar below it.
        assert [
            (
                legend.get_title().get_text(),
                [text.get_text() for text in legend.get_texts()],
            )
            for legend in figure.legends
        ] == [("Cluster", ["0", "1", "2"])]
        colorbars = [axis for axis in figure.axes if axis.get_label() == "<colorbar>"]
        if panel == "heatmap":
            assert child_colorbar not in figure.axes
            assert [axis.get_xlabel() for axis in colorbars] == ["Pearson correlation"]
        else:
            assert colorbars == []
    finally:
        plt.close(figure)
