"""Scoring variants and selecting the informative ones from saved variant counts.

The statistics are checked against mgatk's definitions computed densely from
the synthetic truth's reads, which the code under test never sees.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cytearc.variants import VariantSelection
from tests.clone_helpers import every_cell, open_clone_store
from tests.perturbation_helpers import traced_peak
from tests.variant_helpers import (
    band_budget,
    copy_variant_store,
    expected_mito_statistics,
    mito_variant_labels,
    rcrs_store,
    write_rcrs_mgatk,
    write_variant_store,
)
from tests.variant_truth import (
    mito_clone_truth,
    variant_truth,
    write_cellsnp,
    write_mgatk,
)

TRUTH = mito_clone_truth()
SMALL = variant_truth()
STATISTICS = [
    "mean",
    "variance",
    "vmr",
    "n_cells_conf_detected",
    "strand_correlation",
    "n_cells_over_5",
    "n_cells_over_10",
    "n_cells_over_20",
    "n_cells_over_95",
    "max_heteroplasmy",
    "mean_coverage",
    "n_cells_covered",
]


def _import(store, folder, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return store.variants.import_counts(
            folder, cell_selection=every_cell(store), **options
        )


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Path:
    """A store of the clone truth's cells with its counts imported twice:
    four columns, and six columns with base qualities."""
    folder = tmp_path_factory.mktemp("informative")
    path = folder / "store.zarr"
    store = write_variant_store(path, TRUTH)
    _import(store, write_mgatk(TRUTH, folder / "four"))
    _import(store, write_mgatk(TRUTH, folder / "six", six_columns=True), format="mgatk")
    return path


@pytest.fixture
def store(template, tmp_path):
    return copy_variant_store(template, tmp_path)


def _counts(store, *, qualities: bool = False):
    refs = store.artifacts.list(kind="variant_counts", scope="datastore")
    for ref in refs:
        fields = store.artifacts.load(ref).attrs["count_fields"]
        if ("quality_forward" in fields) == qualities:
            return ref
    raise AssertionError("missing counts")


def test_statistics_follow_mgatk_definitions_over_every_listed_cell(store) -> None:
    counts = _counts(store, qualities=True)
    selection = store.variants.load_informative(
        store.variants.informative(counts, preset="atac")
    )
    assert isinstance(selection, VariantSelection)
    expected = expected_mito_statistics(TRUTH, list(range(TRUTH.n_cells)))
    loaded = selection.variants
    assert set(loaded.index) == set(expected.index)
    pd.testing.assert_frame_equal(
        loaded.loc[expected.index, STATISTICS].astype(float),
        expected[STATISTICS].astype(float),
        check_exact=False,
        check_index_type=False,
        rtol=1e-9,
    )
    # Base qualities are stored as float32.
    np.testing.assert_allclose(
        loaded.loc[expected.index, "mean_quality"], expected["mean_quality"], rtol=1e-6
    )
    assert selection.n_cells == TRUTH.n_cells
    assert selection.counts == counts and selection.cells is None
    assert selection.stranded is True and selection.qualities is True


def test_the_atac_preset_selects_exactly_the_clone_variants(store) -> None:
    counts = _counts(store)
    ref = store.variants.informative(counts, preset="atac")
    selection = store.variants.load_informative(ref)
    informative = set(selection.variants.index[selection.variants["informative"]])
    assert informative == mito_variant_labels(TRUTH)
    assert selection.preset == "atac"
    assert selection.thresholds == {
        "min_cells": 5.0,
        "min_strand_correlation": 0.65,
        "min_vmr": 0.01,
        "min_depth": None,
        "min_quality": None,
    }
    attributes = store.artifacts.load(ref).attrs
    assert attributes["n_informative"] == len(mito_variant_labels(TRUTH))
    assert attributes["n_variants"] == len(selection.variants)
    assert attributes["stranded"] is True and attributes["qualities"] is False
    assert selection.stranded is True and selection.qualities is False
    # Errors sit on one strand, so none passes the strand test.
    errors = selection.variants[~selection.variants["informative"]]
    assert (errors["n_cells_conf_detected"] < 5).all() | (
        errors["strand_correlation"].fillna(-1) < 0.65
    ).all()


def test_cells_restrict_the_statistics_to_their_cells(store) -> None:
    counts = _counts(store)
    store.cells.insert("in_d2", TRUTH.clones.samples == "d2")
    chosen = store.snapshot_cell_selection("in_d2")
    selection = store.variants.load_informative(
        store.variants.informative(counts, cells=chosen, min_vmr=0.01)
    )
    rows = np.flatnonzero(TRUTH.clones.samples == "d2").tolist()
    expected = expected_mito_statistics(TRUTH, rows)
    assert selection.n_cells == len(rows)
    assert selection.cells == chosen
    pd.testing.assert_frame_equal(
        selection.variants.loc[expected.index, STATISTICS].astype(float),
        expected[STATISTICS].astype(float),
        check_exact=False,
        check_index_type=False,
        rtol=1e-9,
    )


def _recorded_depth_reads(monkeypatch) -> list[int]:
    """Record the rows of each read of the stored depths."""
    from cytearc.datastore._operations import variants as variant_operations

    reads: list[int] = []
    as_zarr_array = variant_operations.as_zarr_array

    class Recorded:
        """The stored depths, keeping the length of each read."""

        def __init__(self, stored) -> None:
            self.stored = stored
            self.shape = stored.shape
            self.dtype = stored.dtype
            self.chunks = stored.chunks

        def __getitem__(self, selection) -> np.ndarray:
            values = np.asarray(self.stored[selection])
            reads.append(len(values))
            return values

    def recording(node, *, name=""):
        stored = as_zarr_array(node, name=name)
        return Recorded(stored) if name == "depth" else stored

    monkeypatch.setattr(variant_operations, "as_zarr_array", recording)
    return reads


def test_scattered_cells_are_read_through_short_gaps_within_the_budget(
    store, monkeypatch
) -> None:
    counts = _counts(store)
    store.cells.insert("in_d2", TRUTH.clones.samples == "d2")
    chosen = store.snapshot_cell_selection("in_d2")
    rows = np.flatnonzero(TRUTH.clones.samples == "d2")
    reads = _recorded_depth_reads(monkeypatch)
    # A budget of 1,500 rows per band, a few cells.
    max_rows = 1500
    store.memoryBytes = band_budget(store, counts, max_rows)
    store.variants.informative(counts, min_vmr=0.01)
    every_read = len(reads)
    reads.clear()
    selection = store.variants.load_informative(
        store.variants.informative(counts, cells=chosen, min_vmr=0.01)
    )
    expected = expected_mito_statistics(TRUTH, rows.tolist())
    pd.testing.assert_frame_equal(
        selection.variants.loc[expected.index, STATISTICS].astype(float),
        expected[STATISTICS].astype(float),
        check_exact=False,
        check_index_type=False,
        rtol=1e-9,
    )
    # The chosen cells form many runs, yet take no more reads than every
    # cell, and each read stays within a band.
    runs = 1 + int(np.count_nonzero(np.diff(rows) != 1))
    assert every_read < runs
    assert len(reads) <= every_read
    assert max(reads) <= max_rows


def test_cells_a_chunk_apart_are_read_alone(store, monkeypatch, tmp_path) -> None:
    from cytearc.datastore._operations import variants as variant_operations

    # Chunks of 64 rows, fewer than the rows between every tenth cell.
    monkeypatch.setattr(variant_operations, "_ROW_BAND", 64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        counts = store.variants.import_counts(
            write_mgatk(TRUTH, tmp_path / "final"),
            cell_selection=every_cell(store),
            invalidate_cache=True,
        )
    group = store.artifacts.load(counts)
    assert group["depth"].chunks == (64,)
    rows = np.arange(0, TRUTH.n_cells, 10)
    combined = group["depth_offsets"][:] + group["count_offsets"][:]
    assert np.all(combined[rows[1:]] - combined[rows[:-1] + 1] >= 64)
    store.cells.insert("tenth", np.isin(np.arange(TRUTH.n_cells), rows))
    reads = _recorded_depth_reads(monkeypatch)
    selection = store.variants.load_informative(
        store.variants.informative(
            counts, cells=store.snapshot_cell_selection("tenth"), min_vmr=0.01
        )
    )
    expected = expected_mito_statistics(TRUTH, rows.tolist())
    pd.testing.assert_frame_equal(
        selection.variants.loc[expected.index, STATISTICS].astype(float),
        expected[STATISTICS].astype(float),
        check_exact=False,
        check_index_type=False,
        rtol=1e-9,
    )
    # A large budget would read every cell in one band; each chosen cell is
    # read alone instead, its own rows and no others.
    depth_rows = np.diff(group["depth_offsets"][:])
    assert len(reads) == len(rows)
    assert sum(reads) == int(depth_rows[rows].sum())


def test_bands_hold_at_most_max_cells_however_few_rows_they_have(tmp_path) -> None:
    import zarr

    from cytearc.datastore._operations import variants as variant_operations

    # Twelve cells, of which only cells 3 and 9 hold rows, as barcodes that a
    # cellsnp-lite list names without entries.
    root = zarr.open_group(str(tmp_path / "bands.zarr"), mode="w")
    depth_rows = np.zeros(12, dtype=np.int64)
    depth_rows[[3, 9]] = [2, 3]
    count_rows = np.zeros(12, dtype=np.int64)
    count_rows[[3, 9]] = [1, 2]
    for name, rows in (("depth", depth_rows), ("count", count_rows)):
        root.create_array(
            f"{name}_offsets", data=np.concatenate([[0], np.cumsum(rows)])
        )
    root.create_array("depth_sites", data=np.arange(5, dtype=np.int32), chunks=(2,))
    root.create_array("counts", data=np.arange(3, dtype=np.uint32), chunks=(2,))
    tables = {"depth": ("depth_sites",), "count": ("counts",)}
    positions = np.arange(12)
    bands = list(
        variant_operations._variant_bands(root, tables, positions, 100, max_cells=4)
    )
    assert [band.tolist() for band, _rows in bands] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ]
    depth_cells = np.concatenate([rows["depth"][0] for _band, rows in bands])
    sites = np.concatenate([rows["depth"][1]["depth_sites"] for _band, rows in bands])
    assert depth_cells.tolist() == [3, 3, 9, 9, 9]
    assert sites.tolist() == [0, 1, 2, 3, 4]
    count_cells = np.concatenate([rows["count"][0] for _band, rows in bands])
    assert count_cells.tolist() == [3, 9, 9]
    # Without the bound, the twelve cells' five rows make one band.
    assert (
        len(list(variant_operations._variant_bands(root, tables, positions, 100))) == 1
    )


def test_cells_that_no_file_names_are_left_out(tmp_path) -> None:
    store = write_variant_store(tmp_path / "store.zarr", SMALL)
    # The files name every cell but the first five.
    listed = list(range(5, SMALL.n_cells))
    counts = _import(store, write_mgatk(SMALL, tmp_path / "final", rows=listed))
    selection = store.variants.load_informative(
        store.variants.informative(counts, min_vmr=0.0)
    )
    expected = expected_mito_statistics(SMALL, listed)
    assert selection.n_cells == len(listed)
    pd.testing.assert_frame_equal(
        selection.variants.loc[expected.index, STATISTICS].astype(float),
        expected[STATISTICS].astype(float),
        check_exact=False,
        check_index_type=False,
        rtol=1e-9,
    )
    store.cells.insert("first", np.arange(SMALL.n_cells) < 5)
    with pytest.raises(ValueError, match="No chosen cell is named"):
        store.variants.informative(
            counts, cells=store.snapshot_cell_selection("first"), min_vmr=0.0
        )


def test_the_rna_preset_tests_depth_quality_and_vmr(store) -> None:
    counts = _counts(store, qualities=True)
    selection = store.variants.load_informative(
        store.variants.informative(counts, preset="rna")
    )
    expected = expected_mito_statistics(TRUTH, list(range(TRUTH.n_cells)))
    passed = (
        (expected["mean_coverage"] > 20)
        & (expected["mean_quality"] > 30)
        & (expected["vmr"] > 0.01)
    )
    informative = selection.variants["informative"]
    assert set(informative.index[informative]) == set(passed.index[passed])
    assert selection.thresholds["min_quality"] == 30.0


def test_thresholds_given_by_name_replace_the_presets(store) -> None:
    counts = _counts(store)
    selection = store.variants.load_informative(
        store.variants.informative(counts, preset="atac", min_cells=0, min_vmr=0.5)
    )
    frame = selection.variants
    expected = (
        (frame["n_cells_conf_detected"] >= 0)
        & (frame["strand_correlation"] >= 0.65)
        & (frame["vmr"] > 0.5)
    )
    assert frame["informative"].tolist() == expected.fillna(False).tolist()
    assert selection.thresholds["min_cells"] == 0.0


def test_tests_the_counts_cannot_make_fail_before_scoring(
    store, tmp_path, monkeypatch
) -> None:
    from cytearc.datastore._operations import variants as variant_operations

    small = write_variant_store(tmp_path / "small.zarr", SMALL)
    unstranded = _import(
        small, write_cellsnp(SMALL, tmp_path / "cellsnp"), count_unit="umis"
    )
    bands = variant_operations._variant_bands

    def no_rows(*args, **kwargs):
        raise AssertionError("a row was read before the thresholds were checked")

    # The refusals come before any band of rows is read.
    monkeypatch.setattr(variant_operations, "_variant_bands", no_rows)
    with pytest.raises(ValueError, match="no base qualities"):
        store.variants.informative(_counts(store), preset="rna")
    with pytest.raises(ValueError, match="no forward and reverse counts"):
        small.variants.informative(unstranded, preset="atac")
    monkeypatch.setattr(variant_operations, "_variant_bands", bands)
    selection = small.variants.load_informative(
        small.variants.informative(unstranded, min_vmr=0.0)
    )
    assert selection.variants["n_cells_conf_detected"].isna().all()
    assert selection.variants["strand_correlation"].isna().all()
    assert selection.stranded is False and selection.qualities is False


def test_masks_keep_listed_variants_from_being_informative(tmp_path) -> None:
    store = rcrs_store(tmp_path / "store.zarr")
    counts = _import(store, write_rcrs_mgatk(tmp_path / "final", (2, 4, 6)))
    selection = store.variants.load_informative(
        store.variants.informative(
            counts, min_vmr=0.0, mask=["numt", "artifact_prone", "rna_editing"]
        )
    )
    frame = selection.variants
    masked = dict(zip(frame["position"], frame["masked"], strict=True))
    # 310 is artifact-prone; 16293A>C and 16218C>T are NUMT false positives,
    # but 12684G>C is not (the NUMT reads A there); 2617A>G is an RNA-DNA
    # difference; 1000 is listed nowhere.
    assert masked == {
        310: True,
        1000: False,
        2617: True,
        12684: False,
        16218: True,
        16293: True,
    }
    assert frame["informative"].tolist() == (~frame["masked"]).tolist()
    assert selection.masks == ("artifact_prone", "numt", "rna_editing")
    one = store.variants.load_informative(
        store.variants.informative(counts, min_vmr=0.0, mask="artifact_prone")
    )
    assert one.variants.loc[one.variants["masked"], "position"].tolist() == [310]


def test_masks_refuse_counts_off_rcrs_and_unknown_names(store) -> None:
    counts = _counts(store)
    with pytest.raises(ValueError, match="a contig of 54 bases"):
        store.variants.informative(counts, preset="atac", mask="numt")
    with pytest.raises(ValueError, match="Unknown masks"):
        store.variants.informative(counts, preset="atac", mask=["numts"])
    with pytest.raises(TypeError, match="mask must be"):
        store.variants.informative(counts, preset="atac", mask=[3])


def test_saved_statistics_are_reused_and_identified_by_their_thresholds(store) -> None:
    counts = _counts(store)
    first = store.variants.informative(counts, preset="atac")
    assert store.variants.informative(counts, preset="atac") == first
    # The same resolved thresholds without the preset name are another selection.
    other = store.variants.informative(
        counts, min_cells=5, min_strand_correlation=0.65, min_vmr=0.01
    )
    assert other != first
    again = store.variants.informative(counts, preset="atac", invalidate_cache=True)
    assert again != first
    reader = open_clone_store(Path(store.zarr_loc), zarr_mode="r")
    assert reader.variants.informative(counts, preset="atac") in {first, again}
    with pytest.raises(PermissionError):
        reader.variants.informative(counts, preset="atac", min_vmr=0.2)


def test_a_selection_that_lost_an_array_is_not_reused(store) -> None:
    import zarr

    from cytearc.storage.artifacts import artifact_path

    counts = _counts(store)
    first = store.variants.informative(counts, preset="atac")

    def writable(ref) -> zarr.Group:
        return zarr.open_group(store.zarr_loc, mode="r+")[artifact_path(ref)]

    del writable(first)["vmr"]
    second = store.variants.informative(counts, preset="atac")
    assert second != first
    # An array of another length is not reused either.
    group = writable(second)
    del group["mean"]
    group.create_array("mean", data=np.zeros(3))
    third = store.variants.informative(counts, preset="atac")
    assert third not in (first, second)
    loaded = store.variants.load_informative(third).variants
    expected = expected_mito_statistics(TRUTH, list(range(TRUTH.n_cells)))
    np.testing.assert_allclose(
        loaded.loc[expected.index, "vmr"].astype(float), expected["vmr"], rtol=1e-9
    )
    # Counts with strands and qualities give their statistics too, so a
    # selection that lost one of them is not reused.
    rich = _counts(store, qualities=True)
    for name in ("strand_correlation", "n_cells_conf_detected", "mean_quality"):
        saved = store.variants.informative(rich, min_vmr=0.01)
        del writable(saved)[name]
        scored = store.variants.informative(rich, min_vmr=0.01)
        assert scored != saved
        assert store.variants.load_informative(scored).variants[name].notna().any()


def test_scoring_checks_its_arguments_and_memory(store) -> None:
    counts = _counts(store)
    with pytest.raises(TypeError, match="counts must be an ArtifactRef"):
        store.variants.informative("counts", preset="atac")
    with pytest.raises(TypeError, match="cells must be a cell selection"):
        store.variants.informative(counts, cells="d2", preset="atac")
    with pytest.raises(TypeError, match="invalidate_cache must be a boolean"):
        store.variants.informative(counts, preset="atac", invalidate_cache=1)
    with pytest.raises(ValueError, match="needs a preset"):
        store.variants.informative(counts)
    with pytest.raises(TypeError, match="preset must be one of atac, rna"):
        store.variants.informative(counts, preset=3)
    with pytest.raises(ValueError, match="preset must be one of atac, rna"):
        store.variants.informative(counts, preset="mtscatac")
    with pytest.raises(ValueError, match="must be a variant_counts artifact"):
        store.variants.informative(every_cell(store), preset="atac")
    store.memoryBytes = 5_000
    with pytest.raises(MemoryError, match="Raise mem_budget"):
        store.variants.informative(counts, preset="atac", min_vmr=0.3)


def test_load_informative_refuses_other_artifacts(store) -> None:
    with pytest.raises(ValueError, match="variant_selection artifact"):
        store.variants.load_informative(_counts(store))
    with pytest.raises(TypeError, match="variants must be an ArtifactRef"):
        store.variants.load_informative("selection")


class _Accepted(Exception):
    """Raised where scoring would start to read rows."""


def test_scoring_holds_no_more_than_a_budget_near_the_smallest_it_accepts(
    store, monkeypatch
) -> None:
    from cytearc.datastore._operations import variants as variant_operations

    # Counts with strands and qualities, read whole and for scattered cells:
    # a band's reads decompress whole stored chunks, which the budget holds.
    counts = _counts(store, qualities=True)
    store.cells.insert("odd", np.arange(TRUTH.n_cells) % 2 == 1)
    odd = store.snapshot_cell_selection("odd")
    bands = variant_operations._variant_bands

    def accepted(*args, **kwargs):
        raise _Accepted

    for cells in (None, odd):

        def call(budget: int, cells=cells) -> None:
            store.memoryBytes = budget
            store.variants.informative(
                counts, cells=cells, min_vmr=0.0, invalidate_cache=True
            )

        # The smallest budget that scoring accepts, found without reading.
        monkeypatch.setattr(variant_operations, "_variant_bands", accepted)
        low, high = 1, 1 << 32
        while high - low > 1:
            middle = (low + high) // 2
            try:
                call(middle)
            except MemoryError:
                low = middle
            except _Accepted:
                high = middle
        monkeypatch.setattr(variant_operations, "_variant_bands", bands)
        # Bands of 2,000 rows beyond it, a few cells each.
        budget = high + 200 * 2_000
        _result, peak = traced_peak(lambda: call(budget), budget)
        assert peak <= budget


def test_statistics_do_not_depend_on_the_band_size(store) -> None:
    counts = _counts(store, qualities=True)
    whole = store.variants.load_informative(
        store.variants.informative(counts, preset="atac")
    ).variants
    # A budget that reads a few cells per band.
    store.memoryBytes = band_budget(store, counts, 1500)
    banded = store.variants.load_informative(
        store.variants.informative(counts, preset="atac", invalidate_cache=True)
    ).variants
    pd.testing.assert_frame_equal(banded, whole, check_exact=False, rtol=1e-12)
