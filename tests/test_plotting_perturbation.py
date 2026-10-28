from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from matplotlib.markers import MarkerStyle

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
from cytearc.perturbation.mixtures import (
    FIT_STATUSES,
    gaussian_posterior,
    poisson_gaussian_posterior,
)
from cytearc.plotting._style import foreground_color
from cytearc.plotting.perturbation import efficiency, guide_calls, guide_thresholds
from cytearc.storage.refs import ArtifactRef

CALLS = ArtifactRef(
    scope="assay", assay="CRISPR", kind="guide_calls", artifact_id="a" * 64
)
THRESHOLDS = ArtifactRef(
    scope="assay", assay="CRISPR", kind="guide_thresholds", artifact_id="b" * 64
)
CALLED = to_rgba("#2b6cb0")
NOT_CALLED = to_rgba("#bdbdbd")
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


def test_showing_the_plot_releases_its_figure() -> None:
    result = guide_calls(_store(5), CALLS)

    assert result.tables["guides_per_cell"]["cells"].sum() == 15


def test_the_plot_refuses_settings_and_stores_it_cannot_use() -> None:
    with pytest.raises(ValueError, match="bins must be a positive integer"):
        guide_calls(_store(5), CALLS, bins=0, show=False)
    with pytest.raises(ValueError, match="max_guides_per_cell"):
        guide_calls(_store(5), CALLS, max_guides_per_cell=True, show=False)
    with pytest.raises(TypeError, match="store does not provide guide calls"):
        guide_calls(SimpleNamespace(), CALLS, show=False)


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
    finally:
        result.close()


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


@pytest.mark.parametrize(
    ("case", "arguments", "error", "message"),
    [
        (ONE, {"max_panels": 0}, ValueError, "max_panels must be a positive"),
        (ONE, {"max_panels": True}, ValueError, "max_panels must be a positive"),
        (ONE, {"max_panels": 2.0}, ValueError, "max_panels must be a positive"),
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
            "The fits have no samples named 's9', 's8'",
        ),
        (
            ONE,
            {"samples": ["s1"]},
            ValueError,
            "The fits have no samples named 's1'",
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


def test_the_threshold_plot_needs_a_store_with_guide_thresholds() -> None:
    with pytest.raises(TypeError, match="store does not provide guide thresholds"):
        guide_thresholds(SimpleNamespace(), THRESHOLDS, show=False)


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


def _efficiency_store(
    record: PerturbationEfficiency, requests: list[ArtifactRef] | None = None
) -> SimpleNamespace:
    """A store whose loader returns ``record`` and notes each reference it reads."""

    def load(ref: ArtifactRef) -> PerturbationEfficiency:
        if requests is not None:
            requests.append(ref)
        return record

    return SimpleNamespace(perturbation=SimpleNamespace(load_efficiency=load))


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
    no_umi = np.isneginf(fold)
    # The first row is drawn at the top.
    heights = np.arange(len(table))[::-1]
    groups = [
        (chosen, marker)
        for chosen, marker in ((~no_umi, "o"), (no_umi, "<"))
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
    np.testing.assert_array_equal(x[~no_umi], fold[~no_umi])
    left, right = ax.get_xlim()
    assert left < 0.0 < right
    assert ((left < x) & (x < right)).all()
    if no_umi.any():
        # The rows where no UMI remains share one position at the left edge.
        edge = x[no_umi]
        assert (edge == edge[0]).all()
        assert edge[0] < fold[~no_umi].min(initial=0.0)
        assert edge[0] - left < 0.05 * (right - left)
    significant = table["adjusted_p_value"].to_numpy() < alpha
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
        f"{name} (no UMI remains)" if missing else name
        for name, missing in zip(names, no_umi, strict=True)
    ]
    (line,) = ax.get_lines()
    assert list(line.get_xdata()) == [0.0, 0.0]


def test_efficiency_draws_each_tested_target_from_knockdown_to_activation() -> None:
    requests: list[ArtifactRef] = []
    record = _efficiency_record(TARGET_ROWS)

    result = efficiency(_efficiency_store(record, requests), EFFICIENCY, show=False)

    try:
        assert requests == [EFFICIENCY]
        # No UMI of TP53 remains, so its fold change of minus infinity is first.
        pd.testing.assert_frame_equal(
            result.tables["efficiency"],
            record.table.iloc[[1, 0, 2, 3]].reset_index(drop=True),
        )
        assert list(result.axes) == ["efficiency"]
        _check_efficiency_rows(result, alpha=0.05)
        ax = result.axes["efficiency"]
        assert _row_labels(ax) == ["TP53 (no UMI remains)", "MYC", "GATA1", "KLF1"]
        assert ax.get_xlabel() == "Log2 fold change against control cells"
        assert ax.get_title().splitlines() == [
            "Perturbation efficiency by target",
            "Not plotted: 1 no_feature, 1 no_cells, 1 no_control_expression",
        ]
        (legend,) = result.figure.legends
        assert [text.get_text() for text in legend.get_texts()] == [
            "Adjusted p-value below 0.05",
            "Adjusted p-value of 0.05 or more",
            "No UMI remains in the perturbed cells",
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
        _efficiency_store(record), EFFICIENCY, alpha=np.float64(0.01), show=False
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
            "TP53_1 (no UMI remains)",
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
            "Adjusted p-value below 0.01",
            "Adjusted p-value of 0.01 or more",
            "No UMI remains in the perturbed cells",
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
    store = _efficiency_store(record)

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
        _efficiency_store(_efficiency_record([("A", "tested", 1, -np.inf, 0.5)])),
        EFFICIENCY,
        show=False,
    )
    record = _efficiency_record(
        [("B", "no_feature", 3, np.nan, np.nan), ("C", "no_cells", 0, np.nan, np.nan)]
    )
    empty = efficiency(_efficiency_store(record), EFFICIENCY, show=False)

    try:
        _check_efficiency_rows(lost, alpha=0.05)
        (legend,) = lost.figure.legends
        assert [text.get_text() for text in legend.get_texts()] == [
            "Adjusted p-value of 0.05 or more",
            "No UMI remains in the perturbed cells",
            "1 perturbed cell",
        ]
        _check_efficiency_rows(empty, alpha=0.05)
        ax = empty.axes["efficiency"]
        assert len(ax.collections) == 0
        assert list(ax.get_yticks()) == []
        assert ax.get_title().splitlines() == [
            "Perturbation efficiency by target",
            "Not plotted: 1 no_feature, 1 no_cells",
        ]
        assert empty.figure.legends == []
        pd.testing.assert_frame_equal(
            empty.tables["efficiency"], record.table.iloc[[]].reset_index(drop=True)
        )
        assert empty.provenance.n_cells == 50
        assert empty.provenance.extras["n_tested"] == 0
    finally:
        lost.close()
        empty.close()


def test_efficiency_draws_into_caller_axes_in_the_theme() -> None:
    figure, ax = plt.subplots()
    store = _efficiency_store(_efficiency_record(TARGET_ROWS))

    try:
        result = efficiency(store, EFFICIENCY, target=ax, theme="dark", show=False)
        assert result.owns_figure is False
        assert result.figure is figure
        assert result.axes == {"efficiency": ax}
        assert figure.legends == []
        assert [text.get_text() for text in ax.get_legend().get_texts()][:3] == [
            "Adjusted p-value below 0.05",
            "Adjusted p-value of 0.05 or more",
            "No UMI remains in the perturbed cells",
        ]
        _check_efficiency_rows(result, alpha=0.05)
        (line,) = ax.get_lines()
        assert to_rgba(line.get_color()) == to_rgba(foreground_color("dark"))
    finally:
        plt.close(figure)


def test_efficiency_can_drop_the_legend() -> None:
    store = _efficiency_store(_efficiency_record(TARGET_ROWS))

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
    store = _efficiency_store(_efficiency_record(TARGET_ROWS))

    result = efficiency(store, EFFICIENCY)

    assert result.tables["efficiency"]["target"].tolist() == [
        "TP53",
        "MYC",
        "GATA1",
        "KLF1",
    ]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"max_rows": 0}, "max_rows must be a positive integer"),
        ({"max_rows": True}, "max_rows must be a positive integer"),
        ({"max_rows": 2.0}, "max_rows must be a positive integer"),
        ({"alpha": 0}, "alpha must be a number above 0 and below 1"),
        ({"alpha": 1.0}, "alpha must be a number above 0 and below 1"),
        ({"alpha": True}, "alpha must be a number above 0 and below 1"),
        ({"alpha": "0.05"}, "alpha must be a number above 0 and below 1"),
        ({"alpha": float("nan")}, "alpha must be a number above 0 and below 1"),
    ],
)
def test_the_efficiency_plot_refuses_settings_it_cannot_use(
    arguments: dict[str, Any], message: str
) -> None:
    requests: list[ArtifactRef] = []
    store = _efficiency_store(_efficiency_record(TARGET_ROWS), requests)

    with pytest.raises(ValueError, match=message):
        efficiency(store, EFFICIENCY, show=False, **arguments)
    # The settings are checked before the record is read.
    assert requests == []


@pytest.mark.parametrize(
    "store",
    [
        SimpleNamespace(),
        SimpleNamespace(perturbation=SimpleNamespace(load_guide_calls=lambda ref: ref)),
    ],
)
def test_the_efficiency_plot_needs_a_store_with_perturbation_efficiency(
    store: SimpleNamespace,
) -> None:
    with pytest.raises(
        TypeError, match="store does not provide perturbation efficiency"
    ):
        efficiency(store, EFFICIENCY, show=False)
