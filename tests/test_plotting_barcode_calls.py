"""The barcode calls plot, on the synthetic truth's lineage barcodes.

Its tables are checked against counts taken with pandas from the truth's
molecules, which the import gives back unchanged.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cytearc import plotting
from tests.clone_helpers import every_cell, write_truth_store
from tests.clone_truth import (
    SAMPLES,
    barcode_truth,
    clone_truth,
    write_bartab_counts,
    write_trex_molecules,
)

pytestmark = pytest.mark.usefixtures("figures_closed")

TRUTH = clone_truth()
BARCODES = barcode_truth(TRUTH)
_PREFIXES = [f"{sample}__" for sample in SAMPLES]


@pytest.fixture(scope="module")
def imported(tmp_path_factory):
    """The truth's molecules from TREX's tables and its pairs from BARtab's."""
    folder = tmp_path_factory.mktemp("barcode_calls")
    store = write_truth_store(folder / "store.zarr", TRUTH)
    cells = every_cell(store)
    molecules = [
        write_trex_molecules(
            BARCODES, folder / f"{sample}_molecules.txt", sample=sample
        )
        for sample in SAMPLES
    ]
    counts = [
        write_bartab_counts(BARCODES, folder / f"{sample}.counts.tsv", sample=sample)
        for sample in SAMPLES
    ]
    with pytest.warns(UserWarning, match="match no cell of the store"):
        ref = store.clones.import_barcodes(
            [*molecules, *counts],
            cell_selection=cells,
            barcode_prefix=_PREFIXES * 2,
            barcode_suffix="-1",
            library=["trex", "trex", "bartab", "bartab"],
        )
    return store, ref


def test_barcode_calls_count_cells_pairs_and_molecules(imported) -> None:
    store, ref = imported
    result = store.plots.barcode_calls(ref, show=False)
    try:
        assert set(result.axes) == {
            "barcodes_per_cell",
            "umis_per_barcode",
            "reads_per_molecule",
        }
        molecules = BARCODES.expected_molecules(TRUTH)
        corrected = BARCODES.expected_molecules(TRUTH, corrected=True)
        # A cell's distinct barcodes span its libraries: TREX's sequenced
        # barcodes and BARtab's corrected ones.
        both = pd.concat([molecules, corrected])
        per_cell = both.groupby("cell")["barcode"].nunique()
        per_cell = per_cell.reindex(TRUTH.cell_ids, fill_value=0)
        expected = per_cell.value_counts().sort_index()
        assert result.tables["barcodes_per_cell"].to_dict("list") == {
            "n_barcodes": expected.index.tolist(),
            "n_cells": expected.tolist(),
        }
        umis = result.tables["umis_per_barcode"]
        for library, frame in (("trex", molecules), ("bartab", corrected)):
            pairs = (
                frame.groupby(["cell", "barcode"]).size().value_counts().sort_index()
            )
            rows = umis[umis["library"] == library]
            assert rows["umis"].tolist() == pairs.index.tolist()
            assert rows["n_pairs"].tolist() == pairs.tolist()
        reads = result.tables["reads_per_molecule"]
        # BARtab counts pairs, so only TREX's molecules have reads.
        assert set(reads["library"]) == {"trex"}
        observed = dict(zip(reads["reads"], reads["n_molecules"], strict=True))
        assert observed == molecules["reads"].value_counts().to_dict()
        assert result.provenance.extras["libraries"] == ["bartab", "trex"]
        bars = result.axes["barcodes_per_cell"].patches
        assert len(bars) == 11
        assert [bar.get_height() for bar in bars[: len(expected)]] == [
            float(per_cell.eq(count).sum()) for count in range(len(expected))
        ]
        assert result.axes["umis_per_barcode"].get_xscale() == "log"
    finally:
        plt.close(result.figure)


def test_cells_with_many_barcodes_share_the_last_bar(imported) -> None:
    store, ref = imported
    result = plotting.barcode_calls(store, ref, max_barcodes_per_cell=2, show=False)
    try:
        heights = [bar.get_height() for bar in result.axes["barcodes_per_cell"].patches]
        per_cell = store.clones.load_barcodes(ref).cells["n_barcodes"]
        assert heights == [
            float((per_cell == 0).sum()),
            float((per_cell == 1).sum()),
            float((per_cell >= 2).sum()),
        ]
        labels = [
            label.get_text()
            for label in result.axes["barcodes_per_cell"].get_xticklabels()
        ]
        assert labels == ["0", "1", "2+"]
    finally:
        plt.close(result.figure)


def test_a_library_of_pairs_alone_has_no_reads_panel(tmp_path) -> None:
    store = write_truth_store(tmp_path / "store.zarr", TRUTH)
    path = write_bartab_counts(BARCODES, tmp_path / "d4.counts.tsv", sample="d4")
    ref = store.clones.import_barcodes(
        path,
        cell_selection=every_cell(store),
        barcode_prefix="d4__",
        barcode_suffix="-1",
    )
    result = store.plots.barcode_calls(ref, show=False, show_legend=False)
    try:
        assert result.tables["reads_per_molecule"].empty
        assert list(result.tables["reads_per_molecule"].columns) == [
            "library",
            "reads",
            "n_molecules",
        ]
        texts = [text.get_text() for text in result.axes["reads_per_molecule"].texts]
        assert texts == ["No library counts reads per molecule"]
        # The hidden legend of the libraries is still reported, beside the
        # panel that draws them.
        assert [legend.label for legend in result.legends] == ["Library"]
        assert result.scales[0].order == ("d4",)
        assert all(ax.get_legend() is None for ax in result.axes.values())
    finally:
        plt.close(result.figure)


def test_a_library_without_umi_counts_has_no_umis_panel(tmp_path) -> None:
    store = write_truth_store(tmp_path / "store.zarr", TRUTH)
    path = tmp_path / "reads.tsv"
    pd.DataFrame(
        {"cell": [f"{TRUTH.barcodes[0]}-1"], "barcode": ["AC"], "reads": [3]}
    ).to_csv(path, sep="\t", index=False)
    ref = store.clones.import_barcodes(
        path,
        cell_selection=every_cell(store),
        barcode_prefix=f"{TRUTH.samples[0]}__",
        format="pairs",
        columns={"cell": "cell", "barcode": "barcode", "reads": "reads"},
    )
    result = store.plots.barcode_calls(ref, show=False)
    try:
        assert result.tables["umis_per_barcode"].empty
        assert [text.get_text() for text in result.axes["umis_per_barcode"].texts] == [
            "No library counts UMIs"
        ]
        assert result.tables["barcodes_per_cell"]["n_cells"].sum() == len(
            TRUTH.cell_ids
        )
    finally:
        plt.close(result.figure)


def test_barcode_calls_refuse_bad_arguments(imported) -> None:
    store, ref = imported
    with pytest.raises(ValueError, match="^max_barcodes_per_cell must be at least 1"):
        store.plots.barcode_calls(ref, max_barcodes_per_cell=0, show=False)
    with pytest.raises(TypeError, match="clones.load_barcodes"):
        plotting.barcode_calls(object(), ref, show=False)
    figure, axes = plt.subplots(1, 3)
    try:
        result = plotting.barcode_calls(store, ref, target=axes, show=False)
        assert result.figure is figure
        assert not np.isnan(axes[0].get_ylim()[1])
    finally:
        plt.close(figure)


def test_barcode_calls_show_their_figure_by_default(imported) -> None:
    store, ref = imported
    result = store.plots.barcode_calls(ref)
    plt.close(result.figure)
    assert result.tables["barcodes_per_cell"]["n_cells"].sum() == len(TRUTH.cell_ids)


def test_pairs_counted_by_reads_are_no_molecules_of_the_reads_panel(tmp_path) -> None:
    from tests.clone_truth import write_celltag_reads

    store = write_truth_store(tmp_path / "store.zarr", TRUTH)
    path = write_celltag_reads(
        BARCODES, tmp_path / "atac.txt.gz", sample="d4", assay="ATAC"
    )
    ref = store.clones.import_barcodes(
        path, cell_selection=every_cell(store), barcode_prefix="d4__"
    )
    result = store.plots.barcode_calls(ref, show=False)
    try:
        # ATAC's pairs count reads, but no molecule.
        assert result.tables["reads_per_molecule"].empty
        assert result.tables["umis_per_barcode"].empty
    finally:
        plt.close(result.figure)


def test_bins_of_barcodes_stand_on_a_ticked_axis_of_counts(imported) -> None:
    store, ref = imported
    per_cell = store.clones.load_barcodes(ref).cells["n_barcodes"]
    for cap in (10, 50):
        result = plotting.barcode_calls(
            store, ref, max_barcodes_per_cell=cap, show=False
        )
        try:
            ax = result.axes["barcodes_per_cell"]
            # One bar per count up to the cap, at that count, the last
            # holding the cells of the cap or more.
            bars = sorted(
                (patch.get_x() + patch.get_width() / 2, patch.get_height())
                for patch in ax.patches
            )
            assert [center for center, _height in bars] == pytest.approx(
                list(range(cap + 1))
            )
            assert bars[-1][1] == float((per_cell >= cap).sum())
            ticks = ax.get_xticks().tolist()
            labels = [label.get_text() for label in ax.get_xticklabels()]
            assert ticks[-1] == cap and labels[-1] == f"{cap}+"
            assert labels[:-1] == [str(int(tick)) for tick in ticks[:-1]]
            if cap == 10:
                # Every count up to 10 is named.
                assert labels == [*map(str, range(10)), "10+"]
            else:
                # Past 40 bins the axis still ticks, at equal steps.
                steps = set(np.diff(ticks[:-1]))
                assert len(steps) == 1 and len(ticks) > 5
                assert cap - ticks[-2] >= steps.pop() / 2
            assert all(float(tick).is_integer() for tick in ax.get_yticks())
        finally:
            result.close()
