"""GIFT genotypes against giftwrap 0.6.2 and GIFT's own model code.

A script kept with the stack's scratch work ran, as one ephemeral Modal app,
giftwrap 0.6.2 (``read_h5_file``, and ``collect_counts`` on the per-UMI tables
that GEO ships) and the genotype functions of GIFT's reproducibility code
(``get_p``, ``get_cell_genotypes`` and ``assign_genotypes`` of
clareaulab/gift_figures_reproducibility at commit 35564bc, MIT) on giftwrap's
60-cell sample and on GEO GSM9714721, with gapfill likelihoods of 0.01 per
edit and either no swap probability or the published one of that library.
``tests/giftwrap_0_6_2_golden.json`` holds the results.
"""

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.storage_helpers import write_count_store
from tests.network_helpers import download_or_skip

GOLDEN = json.loads((Path(__file__).parent / "giftwrap_0_6_2_golden.json").read_text())
DATA = Path(__file__).parent / "data" / "variants"
CALL = {
    "w": "wild_type",
    "h": "heterozygous",
    "m": "homozygous_alternative",
    "f": "filtered",
}
_GEO = "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9714nnn/GSM9714721/suppl/{}"
_STEM = "GSM9714721_MPN_16plex_1_lane1_BC8"


def _store(path: Path, cell_ids: list[str]):
    from cytearc.datastore.datastore import DataStore

    write_count_store(
        str(path),
        {"RNA": np.ones((len(cell_ids), 1), dtype=np.uint16)},
        "uint16",
        cell_ids=cell_ids,
    )
    store = DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, nthreads=1
    )
    store.cells.insert("every_cell", np.ones(len(cell_ids), dtype=bool))
    return store


def _counts(store, path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return store.variants.import_counts(
            str(path), cell_selection=store.snapshot_cell_selection("every_cell")
        )


def _totals(loaded) -> dict[str, list[int]]:
    sums = loaded.counts.groupby("site", observed=True)[["count", "reads"]].sum()
    return {
        str(site): [int(row["count"]), int(row["reads"])]
        for site, row in sums.iterrows()
    }


def test_the_sample_reads_as_giftwrap_reads_it_and_its_genotypes_follow_gift(
    tmp_path,
) -> None:
    sample = GOLDEN["sample"]
    import h5py

    with h5py.File(DATA / "giftwrap_counts.1.filtered.h5", "r") as handle:
        cells = [value.decode() for value in handle["matrix/barcode"][:]]
    store = _store(tmp_path / "sample.zarr", cells)
    counts = _counts(store, DATA / "giftwrap_counts.1.filtered.h5")
    loaded = store.variants.load_counts(counts)
    entries = sorted(
        [str(row.cell), str(row.site), str(row.allele), int(row.count), int(row.reads)]
        for row in loaded.counts.itertuples(index=False)
    )
    text = "".join("\t".join(map(str, entry)) + "\n" for entry in entries)
    # Every entry: each cell's UMIs and reads of each probe and gapfill.
    assert len(entries) == sample["n_entries"]
    assert hashlib.sha256(text.encode()).hexdigest() == sample["entries_sha256"]
    assert _totals(loaded) == sample["probe_totals"]
    genotypes = store.variants.genotypes(
        counts,
        method="gift",
        sites=[sample["probe"]],
        alleles={sample["probe"]: tuple(sample["alleles"])},
    )
    calls = store.variants.load_genotypes(genotypes).calls.set_index("cell")
    for cell, posteriors, call in zip(
        sample["cells"], sample["posteriors"], sample["calls"], strict=True
    ):
        row = calls.loc[cell]
        np.testing.assert_allclose(
            [
                row["posterior_wild_type"],
                row["posterior_heterozygous"],
                row["posterior_homozygous_alternative"],
            ],
            posteriors,
            atol=1e-6,
        )
        assert str(row["call"]) == CALL[call], cell
    # The model calls every cell with UMIs of the probe, and no other.
    assert sorted(calls.index.astype(str)) == sorted(sample["cells"])


def _compare(store, genotypes, golden_calls, golden_posteriors, barcodes) -> int:
    """How many of the golden's calls the genotypes repeat.

    ``golden_posteriors`` holds posteriors that must be repeated, or None.
    """
    calls = store.variants.load_genotypes(genotypes).calls
    indexed = calls.set_index(["cell", "site"])
    assert len(calls) == sum(len(entry["cells"]) for entry in golden_calls.values())
    equal = 0
    for site, entry in golden_calls.items():
        for cell, letter in zip(entry["cells"], entry["calls"], strict=True):
            equal += str(indexed.loc[(barcodes[cell], site), "call"]) == CALL[letter]
    for site, cell, wild, het, hom, _confident in golden_posteriors or ():
        row = indexed.loc[(barcodes[cell], site)]
        np.testing.assert_allclose(
            [
                row["posterior_wild_type"],
                row["posterior_heterozygous"],
                row["posterior_homozygous_alternative"],
            ],
            [wild, het, hom],
            atol=1e-6,
        )
    return equal


@pytest.mark.integration
def test_gsm9714721_genotypes_follow_gifts_per_umi_model(tmp_path) -> None:
    golden = GOLDEN["gsm9714721"]
    folder = tmp_path / "BC8"
    folder.mkdir()
    for name, local in (
        (f"{_STEM}_GAP_probe_reads.tsv.gz", "probe_reads.tsv.gz"),
        (f"{_STEM}_GAP_barcodes.tsv.gz", "barcodes.tsv.gz"),
        (f"{_STEM}_BC16_GAP_manifest.tsv.gz", "manifest.tsv.gz"),
    ):
        download_or_skip(_GEO.format(name), folder / local)
    barcodes = (
        pd.read_csv(folder / "barcodes.tsv.gz", sep="\t")["barcode"]
        .astype(str)
        .tolist()
    )
    assert len(barcodes) == golden["n_cells"]
    store = _store(tmp_path / "store.zarr", barcodes)
    counts = _counts(store, folder)
    # giftwrap's collect_counts gives each probe the same UMIs and reads.
    assert _totals(store.variants.load_counts(counts)) == golden["probe_totals"]
    genotypes = store.variants.genotypes(counts, method="gift")
    sites = store.variants.load_genotypes(genotypes).sites
    assert {
        site: [row["reference"], row["alternative"]]
        for site, row in sites.iterrows()
        if site in golden["no_swap"]
    } == {site: golden["alleles"][site] for site in golden["no_swap"]}
    total = sum(len(entry["cells"]) for entry in golden["no_swap"].values())
    # Without swaps, every UMI of an entry counts alike, so the calls equal
    # GIFT's per-UMI calls exactly.
    assert (
        _compare(
            store, genotypes, golden["no_swap"], golden["no_swap_posteriors"], barcodes
        )
        == total
    )
    swapped = store.variants.genotypes(
        counts,
        method="gift",
        swap_probabilities=dict(map(tuple, golden["swap_probabilities"])),
    )
    # With swaps, each UMI takes its entry's reads per UMI where GIFT takes
    # its own: 9,962 of the 10,058 calls agree.
    assert total == 10_058
    assert _compare(store, swapped, golden["swap"], None, barcodes) == 9_962
