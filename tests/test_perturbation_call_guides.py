import warnings
from pathlib import Path

import numba
import numpy as np
import pandas as pd
import pytest

from cytearc.datastore._operations import perturbation as operations
from cytearc.datastore.datastore import DataStore
from cytearc.perturbation import MULTIPLE, NEGATIVE, GuideThresholds
from cytearc.perturbation.guide_thresholds import call_thresholds
from cytearc.perturbation.histograms import guide_count_histograms
from cytearc.perturbation.mixtures import (
    FIT_STATUSES,
    fit_gaussian,
    fit_poisson_gaussian,
)
from cytearc.storage.artifacts import artifact_group, inspect_artifact
from tests.storage_helpers import write_count_store

N_PER_LANE = 200
# Cells come in this lane order, so the fits must sort the lanes.
LANES = np.repeat(np.asarray(["L2", "L0", "L1"]), N_PER_LANE)
SORTED_LANES = ["L0", "L1", "L2"]


def _counts(seed: int = 3) -> np.ndarray:
    """Guide counts of a three-lane screen whose seven guides differ.

    CRISPR0 to CRISPR2 have ambient UMIs and carriers in every lane; CRISPR3
    has carriers outside lane L2 and no UMI in it; CRISPR4 is in one cell,
    CRISPR5 has single UMIs only, and CRISPR6 has ambient UMIs only.
    """
    rng = np.random.default_rng(seed)
    n_cells = len(LANES)
    counts = np.zeros((n_cells, 7), dtype=np.int64)
    counts[:, :3] = rng.poisson(0.3, size=(n_cells, 3))
    counts[:, 6] = rng.poisson(1.0, size=n_cells)
    carriers = rng.random(n_cells) < 0.6
    guide = rng.integers(0, 3, size=n_cells)
    signal = np.round(2.0 ** rng.normal(6.0, 0.7, size=n_cells)).astype(np.int64)
    counts[carriers, guide[carriers]] += signal[carriers] + 8
    outside = LANES != "L2"
    in_g3 = outside & (rng.random(n_cells) < 0.15)
    counts[in_g3, 3] = np.round(2.0 ** rng.normal(5.0, 0.5, size=in_g3.sum())) + 4
    counts[outside, 3] += rng.poisson(0.3, size=int(outside.sum()))
    counts[5, 4] = 50
    counts[rng.random(n_cells) < 0.2, 5] = 1
    return counts.astype(np.uint16)


COUNTS = _counts()


def _store(tmp_path: Path, counts: np.ndarray = COUNTS, *, lanes=LANES) -> DataStore:
    rna = np.random.default_rng(7).poisson(2.0, size=(len(counts), 10))
    path = tmp_path / "store.zarr"
    write_count_store(str(path), {"RNA": rna, "CRISPR": counts}, "uint16")
    store = DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, nthreads=2
    )
    if lanes is not None:
        store.cells.insert("lane", np.asarray(lanes, dtype=object))
    return store


def _cells(store: DataStore):
    return store.select_measured_cells("CRISPR")


def _call_sets(store: DataStore, ref) -> set[tuple[str, str, int]]:
    calls = store.perturbation.load_guide_calls(ref).calls
    return set(
        zip(
            calls["ids"].astype(str),
            calls["guide"].astype(str),
            calls["umis"].astype(int),
            strict=True,
        )
    )


def _expected_calls(counts: np.ndarray, codes: np.ndarray, table: np.ndarray):
    limits = np.where(table > 0, table, np.iinfo(np.int64).max)[codes]
    rows, guides = np.nonzero(counts.astype(np.int64) >= limits)
    return {
        (f"cell{row}", f"CRISPR{guide}", int(counts[row, guide]))
        for row, guide in zip(rows, guides, strict=True)
    }


def _lane_codes() -> np.ndarray:
    return np.searchsorted(SORTED_LANES, LANES).astype(np.int64)


@pytest.mark.parametrize(
    ("method", "fit", "min_umi"),
    [
        ("poisson_gaussian", fit_poisson_gaussian, 1),
        ("gaussian", fit_gaussian, 3),
    ],
)
def test_fitted_methods_call_from_each_lane_fit(tmp_path, method, fit, min_umi) -> None:
    store = _store(tmp_path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ref = store.perturbation.call_guides(
            _cells(store), method=method, sample_column="lane"
        )

    codes = _lane_codes()
    histograms = guide_count_histograms([COUNTS], codes, n_samples=3, n_guides=7)
    fits = fit(histograms)
    table = call_thresholds(fits.threshold, min_umi)
    assert _call_sets(store, ref) == _expected_calls(COUNTS, codes, table)
    loaded = store.perturbation.load_guide_calls(ref)
    assert loaded.source == "call_guides"
    status = inspect_artifact(store.zw, ref)
    assert status.parameters == {"method": method, "min_umi": min_umi}
    thresholds = store.perturbation.load_guide_thresholds(ref)
    assert isinstance(thresholds, GuideThresholds)
    assert thresholds.method == method
    assert thresholds.sample_column == "lane"
    assert thresholds.min_umi == min_umi
    table_frame = thresholds.table
    assert table_frame["sample"].tolist() == [
        lane for lane in SORTED_LANES for _ in range(7)
    ]
    np.testing.assert_array_equal(table_frame["threshold"], fits.threshold.reshape(-1))
    np.testing.assert_array_equal(table_frame["call_threshold"], table.reshape(-1))
    assert table_frame["status"].tolist() == [
        FIT_STATUSES[code] for code in fits.status.reshape(-1)
    ]
    assert table_frame["n_cells"].tolist() == [N_PER_LANE] * 21
    # CRISPR3 has no UMI in lane L2, and CRISPR5 has single UMIs only.
    by_fit = table_frame.set_index(["sample", "guide"])["status"]
    assert by_fit[("L2", "CRISPR3")] == "no_cells"
    assert by_fit[("L0", "CRISPR0")] == "fitted"
    if method == "poisson_gaussian":
        assert by_fit[("L0", "CRISPR5")] == "too_sparse"
    # A fit that calls nothing is counted in a warning that names an example.
    messages = [str(warning.message) for warning in caught]
    assert any("guide fits call no cell" in message for message in messages)
    assert any("'CRISPR3' in sample 'L2'" in message for message in messages)


def test_histograms_of_the_fits_hold_every_count_and_zero(tmp_path) -> None:
    store = _store(tmp_path)
    ref = store.perturbation.call_guides(_cells(store), sample_column="lane")
    thresholds = store.perturbation.load_guide_thresholds(ref, guides=["CRISPR0"])
    frame = thresholds.histograms
    for lane in SORTED_LANES:
        column = COUNTS[LANES == lane, 0].astype(np.int64)
        values, cells = np.unique(column, return_counts=True)
        rows = frame[(frame["sample"] == lane) & (frame["guide"] == "CRISPR0")]
        assert rows["umis"].tolist() == values.tolist()
        assert rows["cells"].tolist() == cells.tolist()
    assert thresholds.table["guide"].tolist() == ["CRISPR0"] * 3
    assert list(thresholds.guides) == [f"CRISPR{index}" for index in range(7)]
    everything = store.perturbation.load_guide_thresholds(
        artifact_group_input(store, ref)
    )
    assert everything.min_umi is None
    assert everything.table["call_threshold"].isna().all()
    assert len(everything.table) == 21


def artifact_group_input(store: DataStore, ref):
    """Return the guide_thresholds input of guide calls."""
    return inspect_artifact(store.zw, ref).input_ref("guide_thresholds")


def test_one_fit_without_a_sample_column(tmp_path) -> None:
    store = _store(tmp_path, lanes=None)
    ref = store.perturbation.call_guides(_cells(store), method="gaussian", min_umi=2)
    codes = np.zeros(len(COUNTS), dtype=np.int64)
    histograms = guide_count_histograms([COUNTS], codes, n_samples=1, n_guides=7)
    table = call_thresholds(fit_gaussian(histograms).threshold, 2)
    assert _call_sets(store, ref) == _expected_calls(COUNTS, codes, table)
    thresholds = store.perturbation.load_guide_thresholds(ref)
    assert thresholds.sample_column is None
    assert thresholds.table["sample"].isna().all()
    assert inspect_artifact(store.zw, artifact_group_input(store, ref)).parameters == {
        "method": "gaussian",
        "sample_columns": [],
    }


def test_fixed_calls_every_guide_from_min_umi(tmp_path) -> None:
    store = _store(tmp_path)
    ref = store.perturbation.call_guides(_cells(store), method="fixed", min_umi=4)
    table = np.full((1, 7), 4)
    codes = np.zeros(len(COUNTS), dtype=np.int64)
    assert _call_sets(store, ref) == _expected_calls(COUNTS, codes, table)
    status = inspect_artifact(store.zw, ref)
    assert status.inputs["guide_thresholds"] is None
    assert status.parameters == {"method": "fixed", "min_umi": 4}


def _top_guide_store(tmp_path: Path) -> DataStore:
    counts = np.asarray(
        [
            [4, 1, 0],  # a share of exactly 0.8: not above it
            [5, 1, 0],  # 5 of 6: called at 0.8
            [9, 0, 0],  # called
            [0, 0, 0],  # no guide UMIs
            [3, 3, 0],  # a tie: the first guide is the top guide
            [2, 0, 0],  # below the floor of 5
            [0, 1, 1],
        ],
        dtype=np.uint16,
    )
    return _store(tmp_path, counts, lanes=None)


def test_top_guide_calls_the_top_guide_above_the_fraction(tmp_path) -> None:
    store = _top_guide_store(tmp_path)
    cells = _cells(store)

    default = store.perturbation.call_guides(cells, method="top_guide")
    assert _call_sets(store, default) == {
        ("cell1", "CRISPR0", 5),
        ("cell2", "CRISPR0", 9),
    }
    status = inspect_artifact(store.zw, default)
    assert status.parameters == {
        "method": "top_guide",
        "min_umi": 5,
        "fraction_threshold": 0.8,
    }
    labels = store.perturbation.load_guide_calls(default).cells["call"].tolist()
    assert labels == [
        NEGATIVE,
        "CRISPR0",
        "CRISPR0",
        NEGATIVE,
        NEGATIVE,
        NEGATIVE,
        NEGATIVE,
    ]

    loose = store.perturbation.call_guides(
        cells, method="top_guide", min_umi=1, fraction_threshold=0.3
    )
    assert ("cell4", "CRISPR0", 3) in _call_sets(store, loose)
    assert ("cell0", "CRISPR0", 4) in _call_sets(store, loose)
    every = store.perturbation.call_guides(
        cells, method="top_guide", min_umi=1, fraction_threshold=0
    )
    assert {cell for cell, _, _ in _call_sets(store, every)} == {
        "cell0",
        "cell1",
        "cell2",
        "cell4",
        "cell5",
        "cell6",
    }
    assert (
        MULTIPLE
        not in store.perturbation.load_guide_calls(every).cells["call"].tolist()
    )


def test_identical_calls_are_reused_without_reading_counts(
    tmp_path, monkeypatch
) -> None:
    store = _store(tmp_path)
    cells = _cells(store)
    with warnings.catch_warnings(record=True) as first:
        warnings.simplefilter("always")
        ref = store.perturbation.call_guides(cells, sample_column="lane")

    def unreachable(*_args, **_kwargs):
        raise AssertionError("the counts were read again")

    monkeypatch.setattr(operations, "guide_count_histograms", unreachable)
    with warnings.catch_warnings(record=True) as again:
        warnings.simplefilter("always")
        assert store.perturbation.call_guides(cells, sample_column="lane") == ref
    # The warning about fits that call nothing comes again on reuse.
    assert [str(w.message) for w in again] == [str(w.message) for w in first]

    # Another floor reuses the fits and reads the counts once, to call.
    floor = store.perturbation.call_guides(cells, sample_column="lane", min_umi=40)
    assert floor != ref
    fits = inspect_artifact(store.zw, ref).input_ref("guide_thresholds")
    assert inspect_artifact(store.zw, floor).input_ref("guide_thresholds") == fits
    calls = store.perturbation.load_guide_calls(floor).calls
    assert (calls["umis"] >= 40).all()

    monkeypatch.undo()
    fresh = store.perturbation.call_guides(
        cells, sample_column="lane", invalidate_cache=True
    )
    assert _call_sets(store, fresh) == _call_sets(store, ref)


def test_read_only_store_reuses_but_cannot_call_again(tmp_path) -> None:
    store = _store(tmp_path)
    cells = _cells(store)
    ref = store.perturbation.call_guides(cells, sample_column="lane")
    reader = DataStore(
        str(tmp_path / "store.zarr"),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
        zarr_mode="r",
    )
    assert reader.perturbation.call_guides(cells, sample_column="lane") == ref
    with pytest.raises(PermissionError, match="call_guides"):
        reader.perturbation.call_guides(cells, sample_column="lane", min_umi=7)


def test_sample_labels_must_cover_every_selected_cell(tmp_path) -> None:
    store = _store(tmp_path)
    blank = LANES.astype(object).copy()
    blank[3] = " "
    store.cells.insert("blank_lane", blank)
    masked = pd.array(LANES.tolist(), dtype="string")
    masked[4] = pd.NA
    store.cells.insert("masked_lane", masked)
    for column in ("blank_lane", "masked_lane"):
        with pytest.raises(ValueError, match="has no label for 1 selected cells"):
            store.perturbation.call_guides(_cells(store), sample_column=column)
    assert not store.artifacts.list(kind="guide_thresholds")
    assert not store.artifacts.list(kind="guide_calls")


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"method": "maximum"}, ValueError, "method must be 'poisson_gaussian'"),
        ({"method": 3}, ValueError, "not 3"),
        ({"min_umi": 0}, ValueError, "min_umi must be at least 1"),
        ({"min_umi": 1.5}, TypeError, "min_umi must be an integer"),
        ({"method": "fixed"}, ValueError, "method='fixed' needs min_umi"),
        ({"fraction_threshold": 0.5}, ValueError, "applies only to method='top_guide'"),
        (
            {"method": "top_guide", "fraction_threshold": 1.0},
            ValueError,
            "at least 0 and below 1",
        ),
        (
            {"method": "top_guide", "fraction_threshold": "high"},
            TypeError,
            "fraction_threshold must be a real number",
        ),
        (
            {"method": "fixed", "min_umi": 3, "sample_column": "lane"},
            ValueError,
            "sample_column applies only to method='poisson_gaussian'",
        ),
        ({"sample_column": ""}, TypeError, "non-empty string"),
        ({"sample_column": "batch"}, ValueError, "'batch' not found"),
        ({"invalidate_cache": "yes"}, TypeError, "invalidate_cache must be a boolean"),
    ],
)
def test_call_guides_refuses_arguments_it_cannot_use(
    tmp_path, arguments, error, message
) -> None:
    store = _store(tmp_path)
    with pytest.raises(error, match=message):
        store.perturbation.call_guides(_cells(store), **arguments)


def test_call_guides_refuses_a_selection_of_no_cell(tmp_path) -> None:
    store = _store(tmp_path)
    with pytest.raises(TypeError, match="cell_selection must be an ArtifactRef"):
        store.perturbation.call_guides("cells")
    store.cells.insert("none", np.zeros(len(COUNTS), dtype=bool))
    with pytest.raises(ValueError, match="selects no cell"):
        store.perturbation.call_guides(store.snapshot_cell_selection("none"))


def test_call_guides_checks_the_memory_budget(tmp_path, monkeypatch) -> None:
    store = _store(tmp_path)
    cells = _cells(store)
    # Enough for 16 bytes per cell of the fit, too little for the histograms
    # and for the calls.
    monkeypatch.setattr(store, "memoryBytes", 20_000)
    with pytest.raises(
        MemoryError, match="Fitting guide thresholds for 600 cells in 3"
    ):
        store.perturbation.call_guides(cells, sample_column="lane")
    with pytest.raises(MemoryError, match="Calling guides makes"):
        store.perturbation.call_guides(cells, method="fixed", min_umi=2)
    # With the histograms counted as free, the counts stream within 50,000
    # bytes, but the calls of 600 cells do not fit.
    monkeypatch.setattr(operations, "histogram_bytes", lambda *_args: 0)
    monkeypatch.setattr(store, "memoryBytes", 50_000)
    with pytest.raises(MemoryError, match="Calling guides makes"):
        store.perturbation.call_guides(cells, method="gaussian")
    # Saved fits are read again only within the budget.
    monkeypatch.undo()
    store.perturbation.call_guides(cells, method="gaussian")
    monkeypatch.setattr(store, "memoryBytes", 100)
    with pytest.raises(MemoryError, match="Reading the saved guide fits"):
        store.perturbation.call_guides(cells, method="gaussian", min_umi=9)


def test_gaussian_warns_when_most_calls_carry_several_guides(tmp_path) -> None:
    # Every guide has ambient UMIs in a quarter of the cells and about one
    # carrier, so the two Gaussians split the cells without UMIs from those
    # with ambient ones, and most called cells get several guides.
    rng = np.random.default_rng(11)
    counts = rng.poisson(0.3, size=(1000, 15))
    carriers = rng.choice(1000, size=20, replace=False)
    counts[carriers, rng.integers(0, 15, size=20)] += 100
    store = _store(tmp_path, counts.astype(np.uint16), lanes=None)
    cells = _cells(store)
    with warnings.catch_warnings(record=True) as first:
        warnings.simplefilter("always")
        calls = store.perturbation.call_guides(cells, method="gaussian", min_umi=1)
    messages = [str(entry.message) for entry in first]
    assert any("of the cells with a call have several guides" in m for m in messages)

    # Saved calls warn again, alike.
    with warnings.catch_warnings(record=True) as again:
        warnings.simplefilter("always")
        reused = store.perturbation.call_guides(cells, method="gaussian", min_umi=1)
    assert reused == calls
    assert [str(entry.message) for entry in again] == messages


def test_fits_run_on_the_store_threads_and_restore_them(tmp_path, monkeypatch) -> None:
    store = _store(tmp_path)
    seen = []
    fit = operations.fit_poisson_gaussian

    def recording(histograms):
        seen.append(numba.get_num_threads())
        return fit(histograms)

    monkeypatch.setattr(operations, "fit_poisson_gaussian", recording)
    monkeypatch.setattr(operations, "MAX_ITERATIONS", 1)
    before = numba.get_num_threads()
    store.perturbation.call_guides(_cells(store))
    assert seen == [min(2, numba.config.NUMBA_NUM_THREADS)]
    assert numba.get_num_threads() == before


def test_loading_thresholds_refuses_calls_without_fits(tmp_path) -> None:
    store = _store(tmp_path)
    cells = _cells(store)
    calls_path = tmp_path / "calls.csv"
    calls_path.write_text(
        "cell_barcode,num_features,feature_call,num_umis\ncell0,1,CRISPR0,30\n"
    )
    imported = store.perturbation.import_guide_calls(calls_path, cell_selection=cells)
    fixed = store.perturbation.call_guides(cells, method="fixed", min_umi=3)
    top = store.perturbation.call_guides(cells, method="top_guide")
    fitted = store.perturbation.call_guides(cells, sample_column="lane")
    load = store.perturbation.load_guide_thresholds
    with pytest.raises(TypeError, match="thresholds must be an ArtifactRef"):
        load("calls")
    with pytest.raises(TypeError, match="guides must be a sequence"):
        load(fitted, guides="CRISPR0")
    with pytest.raises(TypeError, match="guides must be a sequence"):
        load(fitted, guides=[0])
    with pytest.raises(ValueError, match="imported from Cell Ranger"):
        load(imported)
    with pytest.raises(ValueError, match="method='fixed' fit no thresholds"):
        load(fixed)
    with pytest.raises(ValueError, match="method='top_guide' fit no thresholds"):
        load(top)
    with pytest.raises(ValueError, match="not a cell_selection artifact"):
        load(cells)
    with pytest.raises(ValueError, match="2 guides are not guides of the fits"):
        load(fitted, guides=["CRISPR0", "SEC61B-1", "TPR-2"])


def test_loading_thresholds_checks_the_budget_and_the_samples(
    tmp_path, monkeypatch
) -> None:
    store = _store(tmp_path)
    ref = store.perturbation.call_guides(_cells(store), sample_column="lane")
    fits = artifact_group_input(store, ref)
    group = artifact_group(store.zw, fits)
    group.attrs["samples"] = "L0"
    with pytest.raises(ValueError, match="no list of samples"):
        store.perturbation.load_guide_thresholds(fits)
    group.attrs["samples"] = SORTED_LANES
    monkeypatch.setattr(store, "memoryBytes", 100)
    with pytest.raises(MemoryError, match="Pass guides to read fewer"):
        store.perturbation.load_guide_thresholds(fits)
