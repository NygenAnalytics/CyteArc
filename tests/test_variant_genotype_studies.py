"""Genotype calls of published GoT studies, reproduced from their own tables.

Single-cell_BPDCN's XV-seq analysis (commit 5c5f361, MIT) calls a cell
mutant with one mutant UMI or more and wild type otherwise; GEO GSE204845
states the same rule beside the calls its tables hold; and GoT-Multi's
example calls each channel of a target the same way. ``method="threshold"``
with ``min_mutant_umis=1`` applies the rule, and ``method="source"`` adopts
the calls that the tables hold.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cytearc.storage.artifacts import artifact_group
from tests.clone_helpers import every_cell, write_cells_store
from tests.variant_helpers import download_or_skip

DATA = Path(__file__).parent / "data" / "variants" / "got"
# Each XV-seq table's sample and allowlisted cells, and what
# 04_XV-seq/4.2_Genotyping_stats.txt records for it: the cells genotyped, the
# percentage of them mutant and the percentage of the allowlisted cells
# genotyped.
XV_SEQ = {
    "Patient1_TET2.2847": ("Pt1Dx", 2473, 25, 20.0, 1.0109179134654267),
    "Patient5_ZRSR2.500": ("Pt5Dx", 4145, 248, 14.516129032258066, 5.983112183353438),
}
CALL = {"WT": "wild_type", "MUT": "alternative_present", "Unprofiled": "filtered"}
_GEO = "https://ftp.ncbi.nlm.nih.gov/geo/samples/{}nnn/{}/suppl/{}"
_GOT_MULTI = (
    "https://raw.githubusercontent.com/AnnaNamLab/GoT-Multi-ML/"
    "c4a6721d64c0710e6fc2fa3356f9ad4f55e796e8/example_run/got_multi_ml_run/{}"
)


def _counts(store, paths):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return store.variants.import_counts(paths, cell_selection=every_cell(store))


def _calls(store, genotypes) -> dict[tuple[str, str], tuple[str, str]]:
    calls = store.variants.load_genotypes(genotypes).calls
    return {
        (str(row.cell), str(row.site)): (str(row.call), str(row.label))
        for row in calls.itertuples(index=False)
    }


@pytest.mark.parametrize("table", list(XV_SEQ))
def test_xv_seq_calls_give_the_published_genotyping_stats(tmp_path, table) -> None:
    sample, n_allowlisted, n_genotyped, mutant_share, efficiency = XV_SEQ[table]
    gene = table.split("_", 1)[1]
    listed = (DATA / f"{sample}_CellBarcode_Allowlist_in_{gene}.txt").read_text()
    named = listed.split()
    # The rest of the allowlist, which the table does not name.
    others = [f"unnamed{cell:05d}" for cell in range(n_allowlisted - len(named))]
    store = write_cells_store(tmp_path / "store.zarr", named + others)
    path = DATA / f"{table}_summTable.txt"
    counts = _counts(store, path)
    # The table also names cells that the allowlist leaves out.
    n_rows = len(path.read_text().splitlines()) - 1
    unmatched = artifact_group(store.zw, counts).attrs["n_unmatched_barcodes"]
    assert unmatched == n_rows - len(named)
    loaded = store.variants.load_genotypes(
        store.variants.genotypes(counts, method="threshold", min_mutant_umis=1)
    )
    calls = loaded.calls["call"].astype(str).value_counts().to_dict()
    assert set(calls) == {"wild_type", "alternative_present"}
    genotyped = calls["wild_type"] + calls["alternative_present"]
    assert genotyped == n_genotyped
    assert 100 * calls["alternative_present"] / genotyped == pytest.approx(
        mutant_share, rel=1e-12
    )
    called = np.count_nonzero(loaded.cells["n_called_sites"])
    assert 100 * called / len(loaded.cells) == pytest.approx(efficiency, rel=1e-12)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("sample", "name", "uncalled"),
    [
        ("GSM6198716", "GSM6198716_MDS04.SF3B1.tsv.gz", set()),
        # Two cells of 4 wild-type UMIs and 1 mutant UMI that MDS01's table
        # leaves without a call, which the rule calls mutant.
        (
            "GSM6198713",
            "GSM6198713_MDS01.SF3B1.tsv.gz",
            {"TAATCTCTCGGTGAAG", "ATTGGGTTCCTATGGA"},
        ),
    ],
)
def test_gse204845_calls_follow_the_rule_the_study_states(
    tmp_path, sample, name, uncalled
) -> None:
    folder = tmp_path / "geo"
    folder.mkdir()
    path = download_or_skip(_GEO.format(sample[:-3], sample, name), folder / name)
    # The tables name each cell in an unnamed first column.
    table = pd.read_csv(path, sep="\t")
    cells = table.index.astype(str).tolist()
    store = write_cells_store(tmp_path / "store.zarr", cells)
    counts = _counts(store, path)
    site = name.split(".tsv")[0]
    rule = _calls(
        store, store.variants.genotypes(counts, method="threshold", min_mutant_umis=1)
    )
    adopted = _calls(store, store.variants.genotypes(counts, method="source"))
    n_equal = 0
    for cell, wild, mutant, label in zip(
        cells,
        table["WT.calls"].tolist(),
        table["MUT.calls"].tolist(),
        table["Genotype"].tolist(),
        strict=True,
    ):
        key = (cell, site)
        if cell in uncalled:
            assert (wild, mutant) == (4, 1) and pd.isna(label)
            assert rule[key] == ("alternative_present", "")
            assert adopted[key][0] == "filtered"
            continue
        if pd.isna(label):
            # No UMI reads the cell, so neither method calls it.
            assert key not in rule and key not in adopted
        else:
            assert rule[key] == (CALL[label], "")
            assert adopted[key] == (CALL[label], label)
        n_equal += 1
    assert n_equal == len(cells) - len(uncalled)


@pytest.mark.integration
def test_got_multi_example_calls_each_channel_by_the_rule(tmp_path) -> None:
    path = download_or_skip(
        _GOT_MULTI.format("ironthrone_out_pp.csv"), tmp_path / "ironthrone_out_pp.csv"
    )
    table = pd.read_csv(path)
    store = write_cells_store(tmp_path / "store.zarr", sorted(set(table["BC"])))
    counts = _counts(store, path)
    rule = _calls(
        store, store.variants.genotypes(counts, method="threshold", min_mutant_umis=1)
    )
    adopted = _calls(store, store.variants.genotypes(counts, method="source"))
    for row in table.itertuples(index=False):
        for channel in ("geneseq", "probebc"):
            label = getattr(row, f"genotype_{channel}")
            # Unprofiled rows hold an ambiguous UMI at each channel, which
            # reads the cell but supports neither allele.
            assert rule[(row.BC, f"{row.target} ({channel})")] == (CALL[label], "")
        # The target holds the calls and its channels the UMIs, so an
        # Unprofiled call at the target is no evidence there.
        merged = row.genotype_merged
        call = "missing" if merged == "Unprofiled" else CALL[merged]
        assert adopted[(row.BC, row.target)] == (call, merged)
    # Every row is one cell at one target, called at both channels by the
    # rule and at the target by the source.
    assert len(rule) == 2 * len(table)
    assert len(adopted) == len(table)


@pytest.mark.integration
def test_got_multi_example_adopts_its_refined_calls(tmp_path) -> None:
    paths = [
        download_or_skip(_GOT_MULTI.format(name), tmp_path / Path(name).name)
        for name in ("ironthrone_out_pp.csv", "ml_denoise/got_multi_out_refined.csv")
    ]
    stage, refined = (pd.read_csv(path) for path in paths)
    cells = sorted(set(stage["BC"]) | set(refined["BC"]))
    store = write_cells_store(tmp_path / "store.zarr", cells)
    adopted = _calls(
        store, store.variants.genotypes(_counts(store, paths), method="source")
    )
    # The refined call takes precedence over the merged one; a mutant call
    # that the model rejected is filtered.
    final = {**CALL, "FalsePositive": "filtered"}
    expected = {
        (row.BC, row.target): (final[row.genotype_final], row.genotype_final)
        for row in refined.itertuples(index=False)
    }
    assert {key: adopted[key] for key in expected} == expected
    assert "FalsePositive" in set(refined["genotype_final"])
