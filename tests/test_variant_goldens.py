"""Variant statistics and clones against the reference tools, run on the same inputs.

The goldens hold what mgatk 0.9.0's ``call_variants``, Signac 1.17.1's
``IdentifyVariants`` and ``FindClonotypes``, and MiTo 2.0.0 computed on
Signac's ``test_mgatk`` fixture, the synthetic truths written as mgatk files,
and MiTo's MAESTER and mtscATAC fixtures. The scripts that ran them on Modal
are kept with the stack's scratch work.
"""

import gzip
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from cytearc.datastore.datastore import DataStore
from tests.storage_helpers import open_store, write_count_store
from tests.network_helpers import download_or_skip
from tests.variant_truth import mito_clone_truth, variant_truth, write_mgatk

TESTS = Path(__file__).parent
DATA = TESTS / "data" / "variants"
MGATK = json.loads((TESTS / "mgatk_0_9_0_golden.json").read_text())
SIGNAC = json.loads((TESTS / "signac_1_17_1_golden.json").read_text())
MITO = json.loads((TESTS / "mito_2_0_0_golden.json").read_text())
MITO_DATA = (
    "https://raw.githubusercontent.com/andrecossa5/MiTo/"
    "05725c3e69ad2294bff03bb9dcbfcb8499600e10/docs/source/data/{}"
)
COUNTED = [
    "n_cells_conf_detected",
    "n_cells_over_5",
    "n_cells_over_10",
    "n_cells_over_20",
]


def _inputs(name: str, folder: Path) -> Path:
    """The mgatk folder that the reference tools read."""
    if name == "signac_test_mgatk":
        return DATA / "signac_test_mgatk"
    truth = mito_clone_truth() if name == "mito_clone_truth" else variant_truth()
    return write_mgatk(truth, folder, name="truth", rows=range(truth.n_cells))


def _store(folder: Path, cells: list[str]) -> DataStore:
    every = sorted(set(cells))
    write_count_store(
        str(folder / "store.zarr"),
        {"RNA": np.ones((len(every), 2), dtype=int)},
        "uint16",
        cell_ids=every,
    )
    store = open_store(folder / "store.zarr")
    store.cells.insert("every_cell", np.ones(len(every), dtype=bool))
    return store


def _scored(store: DataStore, mgatk_folder: Path, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        counts = store.variants.import_counts(
            mgatk_folder,
            cell_selection=store.snapshot_cell_selection("every_cell"),
            **options,
        )
    selection = store.variants.informative(counts, min_vmr=0.0)
    frame = store.variants.load_informative(selection).variants.copy()
    # The tools name a variant by position and change, without the contig.
    frame.index = [label.split(":", 1)[1] for label in frame.index]
    return counts, frame


def _golden_frame(variants: dict) -> pd.DataFrame:
    return pd.DataFrame.from_dict(variants, orient="index").astype(float)


@pytest.mark.parametrize(
    "name", ["signac_test_mgatk", "mito_clone_truth", "variant_truth"]
)
def test_statistics_equal_mgatk_0_9_0(name, tmp_path) -> None:
    golden = MGATK["inputs"][name]
    store = _store(tmp_path, golden["cells"])
    counts, frame = _scored(store, _inputs(name, tmp_path / "final"))
    expected = _golden_frame(golden["variants"])
    assert set(expected.index) <= set(frame.index)
    columns = [
        "mean",
        "variance",
        "vmr",
        "max_heteroplasmy",
        "mean_coverage",
        *COUNTED,
        "n_cells_over_95",
    ]
    pd.testing.assert_frame_equal(
        frame.loc[expected.index, columns].astype(float),
        expected[columns],
        check_exact=False,
        rtol=1e-9,
        atol=1e-12,
        check_index_type=False,
    )
    # mgatk writes the strand correlation rounded to three decimals.
    np.testing.assert_allclose(
        frame.loc[expected.index, "strand_correlation"].astype(float).round(3),
        expected["strand_correlation"],
        atol=1e-9,
    )
    # mgatk leaves out variants that one strand alone reads, summed over the
    # cells; CyteArc scores them too.
    loaded = store.variants.load_counts(counts)
    sums = loaded.counts.groupby(["site", "allele"], observed=True)[
        ["forward", "reverse"]
    ].sum()
    sums.index = [
        f"{str(site).split(':')[1]}{allele}" for site, allele in sums.index.tolist()
    ]
    for label in set(frame.index) - set(expected.index):
        position_and_reference, alternative = label.split(">")
        key = f"{position_and_reference[:-1]}{alternative}"
        assert (sums.loc[key] == 0).any(), label


@pytest.mark.integration
def test_statistics_equal_mgatk_on_mitos_mtscatac_fixture(tmp_path) -> None:
    # MiTo's 50 pL1000 cells, positions 1 to 3,000 (CC BY 4.0).
    golden = MGATK["inputs"]["mito_mtscatac"]
    folder = tmp_path / "final"
    folder.mkdir()
    for base in "ACGT":
        download_or_skip(
            MITO_DATA.format(f"mtscATAC_pL1000/pL1000.{base}.txt.gz"),
            folder / f"pL1000.{base}.txt.gz",
        )
    reference = gzip.decompress(
        (DATA / "mito_mtscatac" / "pL1000.refAllele.txt.gz").read_bytes()
    )
    (folder / "chrM_refAllele.txt").write_bytes(reference)
    store = _store(tmp_path, golden["cells"])
    _counts, frame = _scored(store, folder)
    expected = _golden_frame(golden["variants"])
    assert len(expected) == 1315
    for column in (
        "mean",
        "variance",
        "vmr",
        "max_heteroplasmy",
        "mean_coverage",
        *COUNTED,
    ):
        np.testing.assert_allclose(
            frame.loc[expected.index, column].astype(float),
            expected[column],
            rtol=1e-9,
            atol=1e-12,
            err_msg=column,
        )


@pytest.mark.parametrize(
    "name", ["signac_test_mgatk", "mito_clone_truth", "variant_truth"]
)
def test_statistics_agree_with_signac_1_17_1(name, tmp_path) -> None:
    golden = SIGNAC["inputs"][name]
    store = _store(tmp_path, golden["cells"])
    _counts, frame = _scored(store, _inputs(name, tmp_path / "final"))
    expected = _golden_frame(golden["variants"])
    # Signac scores every alternative base; those that some cell shows are
    # the variants CyteArc scores.
    assert set(expected.index) == set(frame.index)
    shown = frame.loc[expected.index]
    # Signac rounds the mean to seven decimals.
    np.testing.assert_allclose(shown["mean"], expected["mean"], atol=6e-8)
    for column in [*COUNTED, "mean_coverage"]:
        np.testing.assert_allclose(
            shown[column].astype(float), expected[column], rtol=1e-12, err_msg=column
        )
    # Signac adds an empty entry for the first cell at the reference's last
    # position to size its matrices, which its strand correlations there
    # count; elsewhere they are equal.
    last = int(max(int(label[:-3]) for label in expected.index))
    elsewhere = [label for label in expected.index if int(label[:-3]) != last]
    np.testing.assert_allclose(
        shown.loc[elsewhere, "strand_correlation"].astype(float),
        expected.loc[elsewhere, "strand_correlation"],
        rtol=1e-9,
    )
    # Where every cell is covered, the variance is Signac's without its
    # stabilisation; elsewhere Signac counts uncovered cells as 0.
    covered = shown["n_cells_covered"] == len(golden["cells"])
    np.testing.assert_allclose(
        shown.loc[covered, "variance"],
        expected.loc[covered, "variance_unstabilized"],
        rtol=1e-6,
        atol=1e-7,
    )
    informative = set(
        shown.index[
            (shown["n_cells_conf_detected"] >= 5)
            & (shown["strand_correlation"] >= 0.65)
            & (shown["vmr"] > 0.01)
        ]
    )
    if name == "signac_test_mgatk":
        # Eleven sparsely covered cells: Signac's zeros for uncovered cells
        # lower the VMR of variants that few covered cells carry.
        assert set(golden["high_conf"]) < informative
    else:
        assert informative == set(golden["high_conf"])


def test_clones_agree_with_signac_find_clonotypes(tmp_path) -> None:
    golden = SIGNAC["inputs"]["mito_clone_truth"]
    # Seurat counts each cell among its own ten neighbours, as CyteArc does;
    # annoy returns a tied copy instead of the cell for a few of them.
    assert golden["nn_neighbors"] == [10] and golden["nn_self"] >= 0.9 * 230
    truth = mito_clone_truth()
    store = _store(tmp_path, golden["cells"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        counts = store.variants.import_counts(
            _inputs("mito_clone_truth", tmp_path / "final"),
            cell_selection=store.snapshot_cell_selection("every_cell"),
        )
    selection = store.variants.informative(counts, preset="atac")
    called = store.clones.load_clones(store.variants.frequency_clones(selection)).cells[
        "clone"
    ]
    assigned = called.dropna()
    signac = pd.Series(golden["clones"])
    assert adjusted_rand_score(signac.loc[assigned.index], assigned.astype(str)) > 0.75
    labelled = pd.Series(truth.clones.clones, index=truth.clones.cell_ids)
    labelled = labelled[labelled >= 0]
    # Both over the labelled cells that CyteArc gives a clone.
    common = labelled.index.intersection(assigned.index)
    ours = adjusted_rand_score(labelled.loc[common], assigned.loc[common].astype(str))
    theirs = adjusted_rand_score(labelled.loc[common], signac.loc[common])
    assert ours > 0.95 and ours > theirs


@pytest.mark.integration
def test_maester_statistics_counts_and_clones_agree_with_mito_2_0_0(tmp_path) -> None:
    # MiTo's 50 MDA-MB-231 cells of five lentiviral clones (CC BY 4.0), read
    # as maegatk files with the hg38 reference beside them.
    folder = tmp_path / "maester"
    folder.mkdir()
    for base in "ACGT":
        download_or_skip(
            MITO_DATA.format(f"maester_MDA_clones/{base}.txt.gz"),
            folder / f"{base}.txt.gz",
        )
    meta = download_or_skip(
        MITO_DATA.format("maester_MDA_clones/cells_meta.csv"),
        tmp_path / "cells_meta.csv",
    )
    (folder / "chrM_refAllele.txt.gz").write_bytes(
        (DATA / "mito_maester" / "chrM_refAllele.txt.gz").read_bytes()
    )
    truth = pd.read_csv(meta, index_col=0)["GBC"]
    truth.index = [cell.split("_")[0] for cell in truth.index]
    store = _store(tmp_path, list(truth.index))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        counts = store.variants.import_counts(
            folder,
            cell_selection=store.snapshot_cell_selection("every_cell"),
            format="maegatk",
        )
    selection = store.variants.informative(counts, preset="rna")
    frame = store.variants.load_informative(selection).variants
    frame.index = [label.split(":", 1)[1] for label in frame.index]
    expected = pd.DataFrame.from_dict(MITO["variants"], orient="index")
    # MiTo scores the reference's N at 3107, which is no site of the counts.
    shared = expected.index.difference(["3107N>C"])
    assert set(shared) <= set(frame.index)
    np.testing.assert_allclose(
        frame.loc[shared, "mean_coverage"], expected.loc[shared, "mean_cov"], rtol=1e-6
    )
    np.testing.assert_allclose(
        frame.loc[shared, "mean_quality"], expected.loc[shared, "quality"], atol=0.05
    )
    loaded = store.variants.load_counts(counts)
    depth = {
        (str(cell), str(site).split(":")[1]): int(value)
        for cell, site, value in zip(
            loaded.depth["cell"],
            loaded.depth["site"],
            loaded.depth["depth"],
            strict=True,
        )
    }
    count = {
        (str(cell), str(site).split(":")[1], str(allele)): int(value)
        for cell, site, allele, value in zip(
            loaded.counts["cell"],
            loaded.counts["site"],
            loaded.counts["allele"],
            loaded.counts["count"],
            strict=True,
        )
    }
    for cell, variant, alternative_count, site_depth in MITO["entries"]:
        position = variant[:-3]
        assert depth.get((cell, position), 0) == site_depth
        assert count.get((cell, position, variant[-1]), 0) == alternative_count
    called = store.clones.load_clones(store.variants.frequency_clones(selection)).cells[
        "clone"
    ]
    assigned = called.dropna()
    ours = adjusted_rand_score(truth.loc[assigned.index], assigned.astype(str))
    mito = pd.Series(MITO["clones"])
    theirs = adjusted_rand_score(truth.loc[mito.index], mito)
    assert len(assigned) > len(mito) and ours > theirs
    # On the cells MiTo assigns, the clones are MiTo's.
    assert adjusted_rand_score(
        mito, assigned.loc[mito.index].astype(str)
    ) == pytest.approx(1.0)
