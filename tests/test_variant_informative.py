"""Scoring variants and selecting the informative ones from saved variant counts.

The statistics are checked against mgatk's definitions computed densely from
the synthetic truth's reads, which the code under test never sees.
"""

import gzip
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cytearc.variants import VariantSelection
from tests.clone_helpers import every_cell, open_clone_store
from tests.variant_helpers import (
    copy_variant_store,
    expected_mito_statistics,
    mito_variant_labels,
    write_variant_store,
)
from tests.variant_truth import (
    mito_clone_truth,
    variant_truth,
    write_cellsnp,
    write_mgatk,
)

DATA = Path(__file__).parent / "data" / "variants"
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


def test_tests_the_counts_cannot_make_fail_before_scoring(store, tmp_path) -> None:
    with pytest.raises(ValueError, match="no base qualities"):
        store.variants.informative(_counts(store), preset="rna")
    small = write_variant_store(tmp_path / "small.zarr", SMALL)
    unstranded = _import(
        small, write_cellsnp(SMALL, tmp_path / "cellsnp"), count_unit="umis"
    )
    with pytest.raises(ValueError, match="no forward and reverse counts"):
        small.variants.informative(unstranded, preset="atac")
    selection = small.variants.load_informative(
        small.variants.informative(unstranded, min_vmr=0.0)
    )
    assert selection.variants["n_cells_conf_detected"].isna().all()
    assert selection.variants["strand_correlation"].isna().all()


def _rcrs_folder(folder: Path) -> Path:
    """An mgatk folder on the rCRS reference: three cells reading variants at
    masked positions and at position 1000."""
    folder.mkdir(parents=True)
    with gzip.open(DATA / "mito_mtscatac" / "pL1000.refAllele.txt.gz", "rt") as handle:
        reference = {
            int(position): base.upper()
            for position, base in (line.split() for line in handle if line.strip())
        }
    (folder / "chrM_refAllele.txt").write_text(
        "".join(f"{position}\t{base}\n" for position, base in reference.items())
    )
    # Position: (reference reads, alternative base, alternative reads).
    reads = {
        310: (6, "C", 4),
        16293: (6, "C", 4),
        16218: (6, "T", 4),
        2617: (6, "G", 4),
        12684: (6, "C", 4),
        1000: (6, "C" if reference[1000] != "C" else "T", 4),
    }
    rows: dict[str, list[str]] = {base: [] for base in "ACGT"}
    for cell in ("cell0", "cell1", "cell2"):
        for position, (
            reference_reads,
            alternative,
            alternative_reads,
        ) in reads.items():
            ref = reference[position]
            rows[ref].append(f"{position},{cell},{reference_reads},{reference_reads}")
            rows[alternative].append(
                f"{position},{cell},{alternative_reads},{alternative_reads}"
            )
    for base, lines in rows.items():
        (folder / f"s.{base}.txt.gz").write_bytes(
            gzip.compress(("\n".join(lines) + "\n").encode())
        )
    (folder / "s.depthTable.txt").write_text("cell0\t1\ncell1\t1\ncell2\t1\n")
    return folder


def test_masks_keep_listed_variants_from_being_informative(tmp_path) -> None:
    from tests.storage_helpers import write_count_store
    from cytearc.datastore.datastore import DataStore

    write_count_store(
        str(tmp_path / "store.zarr"),
        {"RNA": np.ones((3, 2), dtype=int)},
        "uint16",
        cell_ids=["cell0", "cell1", "cell2"],
    )
    store = DataStore(
        str(tmp_path / "store.zarr"),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
    )
    store.cells.insert("every_cell", np.ones(3, dtype=bool))
    counts = _import(store, _rcrs_folder(tmp_path / "final"))
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


def test_statistics_do_not_depend_on_the_band_size(store) -> None:
    counts = _counts(store, qualities=True)
    whole = store.variants.load_informative(
        store.variants.informative(counts, preset="atac")
    ).variants
    # A budget that reads a few cells per band.
    store.memoryBytes = 240 * 60 * 4 + 16 * 60 + 200 * 1500
    banded = store.variants.load_informative(
        store.variants.informative(counts, preset="atac", invalidate_cache=True)
    ).variants
    pd.testing.assert_frame_equal(banded, whole, check_exact=False, rtol=1e-12)
