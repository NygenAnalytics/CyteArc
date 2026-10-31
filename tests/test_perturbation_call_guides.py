"""Guide calls fitted to the UMI counts of a three-lane screen.

The module writes the screen and its measured cells once, as a template. A
test that writes copies the template, and the tests of refused calls share
one copy, which they check is left as it was; tests of other counts or assays
write their own stores.
"""

import re
import warnings
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numba
import numpy as np
import pandas as pd
import pytest

from cytearc.datastore._operations import perturbation_guides as operations
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
from cytearc.utils.logging import logger
from tests.perturbation_helpers import (
    check_budget,
    copy_store,
    open_store,
    record_cell_reads,
    write_screen,
    written,
)
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


def _store(path: Path, counts: np.ndarray = COUNTS, *, lanes=LANES) -> DataStore:
    rna = np.random.default_rng(7).poisson(2.0, size=(len(counts), 10))
    columns = {} if lanes is None else {"lane": np.asarray(lanes, dtype=object)}
    return write_screen(path, rna, counts, cell_columns=columns, nthreads=2)


def _cells(store: DataStore):
    return store.select_measured_cells("CRISPR")


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Iterator[SimpleNamespace]:
    """The screen and its measured cells, which most tests start from."""
    path = tmp_path_factory.mktemp("call_guides") / "store.zarr"
    store = _store(path)
    screen = SimpleNamespace(path=path, cells=_cells(store))
    saved = written(store)
    yield screen
    assert written(open_store(path)) == saved


@pytest.fixture
def store(template, tmp_path) -> DataStore:
    """A copy of the template, for a test that writes."""
    return copy_store(template.path, tmp_path, nthreads=2)


@pytest.fixture(scope="module")
def shared_store(template, tmp_path_factory) -> DataStore:
    """One writable copy of the template for the calls that must be refused."""
    folder = tmp_path_factory.mktemp("call_guides_refused")
    return copy_store(template.path, folder, nthreads=2)


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
def test_fitted_methods_call_from_each_lane_fit(
    store, template, method, fit, min_umi
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ref = store.perturbation.call_guides(
            template.cells, method=method, sample_column="lane"
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


def test_histograms_of_the_fits_hold_every_count_and_zero(store, template) -> None:
    ref = store.perturbation.call_guides(template.cells, sample_column="lane")
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


def test_one_fit_without_a_sample_column(store, template) -> None:
    ref = store.perturbation.call_guides(template.cells, method="gaussian", min_umi=2)
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


def test_fixed_calls_every_guide_from_min_umi(store, template) -> None:
    ref = store.perturbation.call_guides(template.cells, method="fixed", min_umi=4)
    table = np.full((1, 7), 4)
    codes = np.zeros(len(COUNTS), dtype=np.int64)
    assert _call_sets(store, ref) == _expected_calls(COUNTS, codes, table)
    status = inspect_artifact(store.zw, ref)
    assert status.inputs["guide_thresholds"] is None
    assert status.parameters == {"method": "fixed", "min_umi": 4}


def _top_guide_store(folder: Path) -> DataStore:
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
    return _store(folder / "store.zarr", counts, lanes=None)


def test_top_guide_calls_the_top_guide_above_the_fraction(tmp_path) -> None:
    store = _top_guide_store(tmp_path)
    cells = _cells(store)

    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(message.record["message"]))
    try:
        default = store.perturbation.call_guides(cells, method="top_guide")
        half = store.perturbation.call_guides(
            cells, method="top_guide", fraction_threshold=0.5
        )
    finally:
        logger.remove(sink)
    assert _call_sets(store, default) == {
        ("cell1", "CRISPR0", 5),
        ("cell2", "CRISPR0", 9),
    }
    # The 5 guide UMIs of cell0 reach the floor of 5, though its top guide's 4
    # do not, so its share of 0.8 calls it at a fraction of 0.5.
    assert _call_sets(store, half) == {
        ("cell0", "CRISPR0", 4),
        ("cell1", "CRISPR0", 5),
        ("cell2", "CRISPR0", 9),
    }
    assert [message for message in messages if "but no call" in message] == [
        "4 cells hold guide UMIs but no call: they hold fewer than 5 guide UMIs, "
        "or their top guide holds at most 0.8 of them",
        "3 cells hold guide UMIs but no call: they hold fewer than 5 guide UMIs, "
        "or their top guide holds at most 0.5 of them",
    ]
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
    store, template, monkeypatch
) -> None:
    cells = template.cells
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


def test_read_only_store_reuses_but_cannot_call_again(store, template) -> None:
    cells = template.cells
    ref = store.perturbation.call_guides(cells, sample_column="lane")
    reader = open_store(store.zarr_loc, zarr_mode="r")
    assert reader.perturbation.call_guides(cells, sample_column="lane") == ref
    with pytest.raises(PermissionError, match="call_guides"):
        reader.perturbation.call_guides(cells, sample_column="lane", min_umi=7)


def _guide_store(tmp_path: Path, guide_assays: dict[str, str]) -> DataStore:
    """A store of RNA and of these guide assays, each declared as its type."""
    rna = np.random.default_rng(7).poisson(2.0, size=(len(COUNTS), 10))
    path = tmp_path / "guides.zarr"
    write_count_store(
        str(path), {"RNA": rna, **{name: COUNTS for name in guide_assays}}, "uint16"
    )
    return open_store(path, assay_types=guide_assays)


def test_the_guide_assay_is_the_stores_only_crispr_assay_whatever_its_name(
    tmp_path,
) -> None:
    store = _guide_store(tmp_path, {"GUIDES": "CRISPR"})
    cells = store.select_measured_cells("GUIDES")

    calls = store.perturbation.call_guides(cells, method="fixed", min_umi=10)

    assert calls.assay == "GUIDES"
    loaded = store.perturbation.load_guide_calls(calls)
    assert len(loaded.calls) == int((COUNTS >= 10).sum())
    assert set(loaded.calls["guide"]) <= {f"GUIDES{guide}" for guide in range(7)}
    # Naming the assay gives the same calls.
    assert (
        store.perturbation.call_guides(
            cells, method="fixed", min_umi=10, from_assay="GUIDES"
        )
        == calls
    )
    path = tmp_path / "calls.csv"
    path.write_text(
        "cell_barcode,num_features,feature_call,num_umis\ncell5,1,GUIDES4,50\n"
    )
    imported = store.perturbation.import_guide_calls(path, cell_selection=cells)
    assert imported.assay == "GUIDES"


@pytest.mark.parametrize(
    ("guide_assays", "expected"),
    [
        (
            {},
            "No assay of the store is declared 'CRISPR', and guide calls read one; "
            "declare the guide assay's type when opening the store, such as "
            "DataStore(..., assay_types={'RNA': 'CRISPR'}). The store's assays "
            "are 'RNA', declared 'RNA'",
        ),
        (
            {"CRISPR": "CRISPR", "GUIDES": "CRISPR"},
            "2 assays of the store, 'CRISPR', 'GUIDES', are declared 'CRISPR', and "
            "guide calls read one; name the guide assay with from_assay=...",
        ),
    ],
)
def test_a_store_without_one_crispr_assay_needs_from_assay(
    tmp_path, guide_assays, expected
) -> None:
    store = _guide_store(tmp_path, guide_assays)
    cells = store.select_measured_cells("RNA")
    path = tmp_path / "calls.csv"
    path.write_text("cell_barcode,num_features,feature_call,num_umis\n")
    message = re.escape(expected)
    saved = written(store)

    with pytest.raises(ValueError, match=f"^{message}$"):
        store.perturbation.call_guides(cells)
    with pytest.raises(ValueError, match=f"^{message}$"):
        store.perturbation.import_guide_calls(path, cell_selection=cells)
    # Naming an assay of another type names the assays declared CRISPR.
    with pytest.raises(TypeError, match="'RNA' is declared as 'RNA'; ") as refused:
        store.perturbation.call_guides(cells, from_assay="RNA")
    if guide_assays:
        declared = ", ".join(repr(name) for name in guide_assays)
        assert str(refused.value).endswith(
            f"the store's assays declared 'CRISPR' are {declared}"
        )
    assert written(store) == saved


def test_guide_assay_messages_name_the_remedy_that_works(tmp_path) -> None:
    # A guide assay whose name is not a preset opens as a generic Assay.
    path = tmp_path / "guides.zarr"
    rna = np.random.default_rng(7).poisson(2.0, size=(len(COUNTS), 10))
    write_count_store(str(path), {"RNA": rna, "Guides": COUNTS}, "uint16")
    store = open_store(path)
    cells = store.select_measured_cells("Guides")
    saved = written(store)
    declare = (
        "declare the guide assay's type when opening the store, such as "
        "DataStore(..., assay_types={'Guides': 'CRISPR'})"
    )
    with pytest.raises(ValueError, match=re.escape(declare + ". The store's assays")):
        store.perturbation.call_guides(cells, method="fixed", min_umi=10)
    with pytest.raises(
        TypeError,
        match=re.escape(f"'Guides' is declared as 'Assay'; {declare}") + "$",
    ):
        store.perturbation.call_guides(
            cells, method="fixed", min_umi=10, from_assay="Guides"
        )
    # A misspelt name is not an assay, whatever its type.
    with pytest.raises(
        ValueError,
        match="^from_assay 'GUIDES' is not an assay of the store, whose assays are "
        "'Guides', 'RNA'$",
    ):
        store.perturbation.call_guides(
            cells, method="fixed", min_umi=10, from_assay="GUIDES"
        )
    assert written(store) == saved
    # The declaration that the messages name works.
    declared = open_store(path, assay_types={"Guides": "CRISPR"})
    calls = declared.perturbation.call_guides(
        declared.select_measured_cells("Guides"), method="fixed", min_umi=10
    )
    assert calls.assay == "Guides"
    # Once an assay is declared CRISPR, naming another names it instead.
    with pytest.raises(
        TypeError,
        match="'RNA' is declared as 'RNA'; the store's assay declared 'CRISPR' is "
        "'Guides'$",
    ):
        declared.perturbation.call_guides(
            cells, method="fixed", min_umi=10, from_assay="RNA"
        )


def test_loading_thresholds_takes_any_collection_of_guide_ids(store, template) -> None:
    ref = store.perturbation.call_guides(template.cells, sample_column="lane")
    loaded = store.perturbation.load_guide_thresholds(ref)
    expected = loaded.table[loaded.table["guide"].isin(["CRISPR0", "CRISPR1"])]
    # The guides index of a loaded record, a NumPy array and a generator all
    # read the guides that a list reads.
    for guides in (
        loaded.guides[:2],
        loaded.guides[:2].to_numpy(),
        (guide for guide in ["CRISPR0", "CRISPR1"]),
        ["CRISPR0", "CRISPR1"],
    ):
        table = store.perturbation.load_guide_thresholds(ref, guides=guides).table
        pd.testing.assert_frame_equal(
            table.reset_index(drop=True), expected.reset_index(drop=True)
        )
    with pytest.raises(TypeError, match="guides must be a sequence"):
        store.perturbation.load_guide_thresholds(ref, guides=7)
    with pytest.raises(TypeError, match="guides must be a sequence"):
        store.perturbation.load_guide_thresholds(ref, guides=b"CRISPR0")


def test_a_read_only_store_checks_the_budget_before_writing(
    template, monkeypatch
) -> None:
    cells = template.cells
    reader = open_store(template.path, zarr_mode="r")

    # Neither the fits nor the calls fit the budget, which is checked before
    # the store must be writable.
    monkeypatch.setattr(reader, "memoryBytes", 100)
    with pytest.raises(MemoryError, match="Fitting guide thresholds for 600"):
        reader.perturbation.call_guides(cells, sample_column="lane")
    with pytest.raises(MemoryError, match="Calling guides makes"):
        reader.perturbation.call_guides(cells, method="fixed", min_umi=2)
    monkeypatch.undo()
    for arguments in ({"sample_column": "lane"}, {"method": "fixed", "min_umi": 2}):
        with pytest.raises(PermissionError, match="call_guides"):
            reader.perturbation.call_guides(cells, **arguments)


def test_fixed_calls_of_one_lane_read_only_its_counts(store, monkeypatch) -> None:
    store.cells.insert("in_l0", LANES == "L0")
    lane = store.snapshot_cell_selection("in_l0")
    reads = record_cell_reads(monkeypatch, store, {"CRISPR_nFeatures"})

    ref = store.perturbation.call_guides(lane, method="fixed", min_umi=2)

    in_lane = np.flatnonzero(LANES == "L0")
    assert set(np.concatenate(reads["CRISPR_nFeatures"]).tolist()) == set(
        in_lane.tolist()
    )
    table = np.full((1, COUNTS.shape[1]), 2, dtype=np.int64)
    expected = _expected_calls(COUNTS, np.zeros(len(COUNTS), dtype=np.int64), table)
    assert _call_sets(store, ref) == {
        call for call in expected if int(call[0][4:]) in set(in_lane.tolist())
    }


def test_sample_labels_must_cover_every_selected_cell(store, template) -> None:
    blank = LANES.astype(object).copy()
    blank[3] = " "
    store.cells.insert("blank_lane", blank)
    masked = pd.array(LANES.tolist(), dtype="string")
    masked[4] = pd.NA
    store.cells.insert("masked_lane", masked)
    saved = written(store)
    for column in ("blank_lane", "masked_lane"):
        with pytest.raises(ValueError, match="has no label for 1 selected cells"):
            store.perturbation.call_guides(template.cells, sample_column=column)
    assert written(store) == saved


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
    shared_store, template, arguments, error, message
) -> None:
    saved = written(shared_store)
    with pytest.raises(error, match=message):
        shared_store.perturbation.call_guides(template.cells, **arguments)
    assert written(shared_store) == saved


def test_call_guides_refuses_a_selection_of_no_cell(store) -> None:
    store.cells.insert("none", np.zeros(len(COUNTS), dtype=bool))
    nobody = store.snapshot_cell_selection("none")
    saved = written(store)
    with pytest.raises(TypeError, match="cell_selection must be an ArtifactRef"):
        store.perturbation.call_guides("cells")
    with pytest.raises(ValueError, match="selects no cell"):
        store.perturbation.call_guides(nobody)
    assert written(store) == saved


def _wide_screen(folder: Path) -> Path:
    """A two-lane screen with long guide IDs, where one cell carries every guide.

    4,000 cells and 80 guides whose IDs are 60 characters long. Every cell
    holds ambient UMIs and carries one guide with about 2**6 UMIs, and cell0
    holds 30 UMIs of every guide, so a floor of 1 UMI calls about 17 guides
    in a cell and 80 in cell0.
    """
    n_cells, n_guides = 4_000, 80
    rng = np.random.default_rng(17)
    counts = rng.poisson(0.2, size=(n_cells, n_guides))
    carried = rng.integers(0, n_guides, n_cells)
    counts[np.arange(n_cells), carried] += (
        np.round(2.0 ** rng.normal(6.0, 0.7, n_cells)).astype(np.int64) + 8
    )
    counts[0] = 30
    ids = [
        f"GUIDE{guide:03d}_TARGETS_ENSG{guide:011d}_ACGTACGTACGTACGTACGTACGTAC"
        for guide in range(n_guides)
    ]
    path = folder / "wide.zarr"
    write_count_store(
        str(path),
        {"RNA": rng.poisson(1.0, size=(n_cells, 3)) + 1, "CRISPR": counts},
        "uint16",
        feature_ids={"CRISPR": ids},
    )
    store = open_store(path)
    store.cells.insert(
        "lane", np.where(np.arange(n_cells) % 2, "A", "B").astype(object)
    )
    store.cells.insert("every", np.ones(n_cells, dtype=bool))
    store.snapshot_cell_selection("every")
    return path


@pytest.mark.slow
@pytest.mark.parametrize(
    "options",
    [
        {"method": "fixed", "min_umi": 1},
        {"method": "top_guide"},
        {"method": "poisson_gaussian"},
        {"method": "gaussian", "sample_column": "lane"},
    ],
    ids=["fixed", "top_guide", "poisson_gaussian", "gaussian"],
)
def test_call_guides_runs_within_the_budget_that_its_refusals_name(
    tmp_path, options
) -> None:
    path = _wide_screen(tmp_path)
    cells = open_store(path).snapshot_cell_selection("every")

    budget, refusals, _ = check_budget(
        path,
        lambda store: store.perturbation.call_guides(
            cells, invalidate_cache=True, **options
        ),
    )

    # The first refusal comes before any count is read and names a budget
    # that runs, or for the fitted methods one that gets the fits past the
    # stream; the calls may then need more, which a second refusal names.
    assert 1 <= len(refusals) <= (2 if options["method"].endswith("gaussian") else 1)
    first = (
        "Fitting guide" if options["method"].endswith("gaussian") else "Calling guides"
    )
    assert refusals[0].startswith(first)
    if options["method"] != "top_guide":
        # The padded calls of cell0, 80 slots for each of the 4,000 cells, and
        # the 60 characters of the labels count.
        assert budget > 8 * 80 * 4_000


def _distinct_screen(folder: Path) -> Path:
    """A screen whose cells carry 20 of 200 guides at counts that few cells share.

    The histograms of 3,000 cells then hold about 60,000 distinct counts,
    20 per cell, which only reading the counts tells.
    """
    n_cells, n_guides, per_cell = 3_000, 200, 20
    rng = np.random.default_rng(23)
    counts = rng.poisson(0.2, size=(n_cells, n_guides))
    for cell in range(n_cells):
        chosen = rng.choice(n_guides, size=per_cell, replace=False)
        counts[cell, chosen] += 33 + rng.integers(0, 100_000, per_cell)
    path = folder / "distinct.zarr"
    write_count_store(
        str(path),
        {"RNA": rng.poisson(1.0, size=(n_cells, 2)) + 1, "CRISPR": counts},
        "uint32",
    )
    store = open_store(path)
    store.cells.insert("every", np.ones(n_cells, dtype=bool))
    store.snapshot_cell_selection("every")
    return path


@pytest.mark.slow
def test_histogram_refusals_name_a_budget_past_where_they_stopped(tmp_path) -> None:
    path = _distinct_screen(tmp_path)
    cells = open_store(path).snapshot_cell_selection("every")

    budget, refusals, _ = check_budget(
        path,
        lambda store: store.perturbation.call_guides(cells, invalidate_cache=True),
    )

    # Within the least that the fits hold, the histograms stop while counts
    # remain, and estimate what every row needs from the rows read.
    stopped = [refusal for refusal in refusals if "histograms" in refusal]
    assert stopped
    first = stopped[0]
    assert re.search(
        r"; \d+ of 3000 rows of counts remain to be read, and at the ", first
    )
    distinct = [
        int(re.search(r"for (\d+) distinct counts", refusal).group(1))
        for refusal in stopped
    ]
    # Each refusal names a budget that gets past the point where the one before
    # stopped, so the next stops later, with more distinct counts, or not at
    # all, and a few refusals reach the budget that runs.
    assert distinct == sorted(set(distinct))
    assert len(refusals) <= 5
    # The budget holds the counts beside histograms of every count.
    assert budget > 16 * 60_000


def test_gaussian_warns_when_most_calls_carry_several_guides(tmp_path) -> None:
    # Every guide has ambient UMIs in a quarter of the cells and about one
    # carrier, so the two Gaussians split the cells without UMIs from those
    # with ambient ones, and most called cells get several guides.
    rng = np.random.default_rng(11)
    counts = rng.poisson(0.3, size=(1000, 15))
    carriers = rng.choice(1000, size=20, replace=False)
    counts[carriers, rng.integers(0, 15, size=20)] += 100
    store = _store(tmp_path / "store.zarr", counts.astype(np.uint16), lanes=None)
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


def test_fits_run_on_the_store_threads_and_restore_them(
    store, template, monkeypatch
) -> None:
    seen = []
    fit = operations.fit_poisson_gaussian

    def recording(histograms):
        seen.append(numba.get_num_threads())
        return fit(histograms)

    monkeypatch.setattr(operations, "fit_poisson_gaussian", recording)
    monkeypatch.setattr(operations, "MAX_ITERATIONS", 1)
    before = numba.get_num_threads()
    store.perturbation.call_guides(template.cells)
    assert seen == [min(2, numba.config.NUMBA_NUM_THREADS)]
    assert numba.get_num_threads() == before


def test_loading_thresholds_refuses_calls_without_fits(
    store, template, tmp_path
) -> None:
    cells = template.cells
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
    store, template, monkeypatch
) -> None:
    ref = store.perturbation.call_guides(template.cells, sample_column="lane")
    fits = artifact_group_input(store, ref)
    group = artifact_group(store.zw, fits)
    group.attrs["samples"] = "L0"
    with pytest.raises(ValueError, match="no list of samples"):
        store.perturbation.load_guide_thresholds(fits)
    group.attrs["samples"] = SORTED_LANES
    monkeypatch.setattr(store, "memoryBytes", 100)
    with pytest.raises(MemoryError, match="Pass guides to read fewer"):
        store.perturbation.load_guide_thresholds(fits)


def test_loading_thresholds_needs_the_status_names_that_decode_them(
    store, template
) -> None:
    ref = store.perturbation.call_guides(template.cells, sample_column="lane")
    group = artifact_group(store.zw, artifact_group_input(store, ref))
    table = store.perturbation.load_guide_thresholds(ref).table
    assert set(table["status"]) <= set(FIT_STATUSES)
    # Status codes saved under other names would be read as the wrong
    # statuses, through the calls or on their own.
    for names in (list(reversed(FIT_STATUSES)), None):
        group.attrs["status_names"] = names
        for source in (ref, artifact_group_input(store, ref)):
            with pytest.raises(ValueError, match="names its statuses"):
                store.perturbation.load_guide_thresholds(source)
    group.attrs["status_names"] = list(FIT_STATUSES)
    pd.testing.assert_frame_equal(
        store.perturbation.load_guide_thresholds(ref).table, table
    )
