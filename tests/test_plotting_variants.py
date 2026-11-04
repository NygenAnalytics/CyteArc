"""The variant plots, on counts whose depths and calls are known by hand."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from matplotlib.colors import to_hex

from cytearc.plotting import genotypes, variant_depth
from cytearc.plotting._style import (
    GENOTYPE_CALL_SCALE,
    MUTED_COLOR,
    NO_VALUE_COLOR,
)
from tests.clone_helpers import write_embedding
from tests.storage_helpers import open_store, write_count_store

pytestmark = pytest.mark.usefixtures("figures_closed")

CELLS = ["c0", "c1", "c2", "c3", "c4"]
GROUPS = ["x", "x", "y", "y", None]


@pytest.fixture(scope="module")
def plotted(tmp_path_factory):
    folder = tmp_path_factory.mktemp("variant_plot")
    write_count_store(
        str(folder / "store.zarr"),
        {"RNA": np.ones((5, 3), dtype=int)},
        "uint16",
        cell_ids=CELLS,
    )
    store = open_store(folder / "store.zarr")
    store.cells.insert("group", np.asarray(GROUPS, dtype=object))
    store.cells.insert("every_cell", np.ones(5, dtype=bool))
    cells = store.snapshot_cell_selection("every_cell")
    # Two GoT targets: c0 to c3 are covered at CALR, c0 at SF3B1 too; c4 is
    # not named.
    calr = folder / "CALR.tsv"
    calr.write_text("BC\twtUMIs\tmutUMIs\nc0\t3\t1\nc1\t2\t0\nc2\t0\t6\nc3\t0\t0\n")
    sf3b1 = folder / "SF3B1.tsv"
    sf3b1.write_text("BC\twtUMIs\tmutUMIs\nc0\t1\t1\n")
    targets = store.variants.import_counts([calr, sf3b1], cell_selection=cells)
    # A mitochondrial contig of four positions.
    mito = folder / "final"
    mito.mkdir()
    rows = {
        "A": "1,c0,2,2\n2,c0,1,0\n1,c1,10,10\n",
        "C": "3,c0,0,4\n",
        "G": "",
        "T": "4,c2,1,1\n",
    }
    import gzip

    for base, text in rows.items():
        (mito / f"s.{base}.txt.gz").write_bytes(gzip.compress(text.encode()))
    (mito / "chrM_refAllele.txt").write_text("1\tA\n2\tA\n3\tC\n4\tT\n")
    (mito / "s.depthTable.txt").write_text("c0\t2.25\nc1\t5\nc2\t0.5\nc3\t0\n")
    positions = store.variants.import_counts(mito, cell_selection=cells)
    return store, targets, positions


def test_the_plot_draws_each_sites_mean_depth_over_named_cells(plotted) -> None:
    store, _targets, positions = plotted

    result = variant_depth(store, positions, show=False)

    sites = result.tables["sites"]
    # Depths 4 + 20 at position 1, 1, 4 and 2, over the four named cells.
    assert sites["site"].tolist() == ["chrM:1", "chrM:2", "chrM:3", "chrM:4"]
    np.testing.assert_allclose(sites["mean_depth"], [24 / 4, 1 / 4, 4 / 4, 2 / 4])
    np.testing.assert_allclose(sites["covered_fraction"], [2 / 4, 1 / 4, 1 / 4, 1 / 4])
    cells = result.tables["cells"]
    assert cells["ids"].tolist() == ["c0", "c1", "c2", "c3"]
    np.testing.assert_allclose(cells["mean_depth"], [9 / 4, 20 / 4, 2 / 4, 0.0])
    assert cells["group"].tolist() == ["all cells"] * 4
    assert result.provenance.n_cells == 4
    assert result.provenance.extras["count_unit"] == "reads"
    assert result.axes["sites"].get_yscale() == "log"
    plt.close(result.figure)


def test_named_sites_are_bars_and_groups_split_the_cells(plotted) -> None:
    store, targets, _positions = plotted

    result = variant_depth(store, targets, group_by="group", show=False)

    sites = result.tables["sites"]
    assert sites["site"].tolist() == ["CALR", "SF3B1"]
    np.testing.assert_allclose(sites["mean_depth"], [12 / 4, 2 / 4])
    cells = result.tables["cells"]
    # c4 is not named, so it has no mean depth; every named cell has a group.
    assert cells["ids"].tolist() == ["c0", "c1", "c2", "c3"]
    assert cells["group"].tolist() == ["x", "x", "y", "y"]
    np.testing.assert_allclose(cells["mean_depth"], [6 / 2, 2 / 2, 6 / 2, 0.0])
    assert [tick.get_text() for tick in result.axes["sites"].get_xticklabels()] == [
        "CALR",
        "SF3B1",
    ]
    assert len(result.legends) == 1 and result.legends[0].label == "group"
    plt.close(result.figure)


def test_the_plot_refuses_what_it_cannot_draw(plotted, tmp_path) -> None:
    store, targets, _positions = plotted
    store.cells.insert(
        "only_c4", np.asarray([None, None, None, None, "z"], dtype=object)
    )
    with pytest.raises(ValueError, match="No cell that a file names has a group"):
        variant_depth(store, targets, group_by="only_c4", show=False)
    with pytest.raises(TypeError, match="groups must be an ArtifactRef"):
        variant_depth(store, targets, group_by="group", groups="group", show=False)
    with pytest.raises(ValueError, match="'nowhere', which is not a cell metadata"):
        variant_depth(store, targets, group_by=["group", "nowhere"], show=False)
    unnamed = tmp_path / "other.tsv"
    unnamed.write_text("BC\twtUMIs\tmutUMIs\nnobody\t1\t0\n")
    cells = store.snapshot_cell_selection("every_cell")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        empty = store.variants.import_counts(unnamed, cell_selection=cells)
    with pytest.raises(ValueError, match="No file names a cell of the counts"):
        variant_depth(store, empty, show=False)
    with pytest.raises(TypeError, match="does not provide variants.load_counts"):
        variant_depth(object(), targets, show=False)


def test_groups_of_several_columns_split_the_cells(plotted) -> None:
    store, targets, _positions = plotted
    store.cells.insert(
        "half", np.asarray(["a", "b", "a", "b", "a"], dtype=object), overwrite=True
    )
    result = variant_depth(store, targets, group_by=["group", "half"], show=False)
    # c4 has no group in the first column, so it has none.
    assert result.tables["cells"]["group"].tolist() == ["x, a", "x, b", "y, a", "y, b"]
    assert result.legends[0].label == "group, half"
    plt.close(result.figure)
    labels = store.snapshot_cluster_labels(
        "half", cell_selection=store.snapshot_cell_selection("every_cell")
    )
    # A column and a label artifact group the cells together.
    both = variant_depth(store, targets, group_by="group", groups=labels, show=False)
    assert both.tables["cells"]["group"].tolist() == ["x, a", "x, b", "y, a", "y, b"]
    assert both.provenance.extras["grouping"] == "group, cluster_labels"
    plt.close(both.figure)


def test_groups_are_drawn_in_the_natural_order_of_their_labels(plotted) -> None:
    store, targets, _positions = plotted
    # The first named cell is of s10, which comes after s2 in natural order.
    store.cells.insert(
        "lane",
        np.asarray(["s10", "s2", "s10", "s2", "s2"], dtype=object),
        overwrite=True,
    )
    result = variant_depth(store, targets, group_by="lane", show=False)
    legend = result.axes["cells"].get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["s2", "s10"]
    assert result.tables["cells"]["group"].tolist() == ["s10", "s2", "s10", "s2"]
    plt.close(result.figure)


def test_the_store_accessor_draws_the_same_plot(plotted) -> None:
    store, targets, _positions = plotted
    result = store.plots.variant_depth(targets, show=False)
    assert result.tables["sites"]["site"].tolist() == ["CALR", "SF3B1"]
    plt.close(result.figure)


def test_several_contigs_draw_one_line_each(tmp_path) -> None:
    import shutil
    from pathlib import Path

    from cytearc.readers._variant_files import read_barcodes

    folder = Path(__file__).parent / "data" / "variants" / "vartrix"
    cells = read_barcodes(str(folder / "barcodes.tsv")).tolist()
    write_count_store(
        str(tmp_path / "store.zarr"),
        {"RNA": np.ones((len(cells), 3), dtype=int)},
        "uint16",
        cell_ids=cells,
    )
    store = open_store(tmp_path / "store.zarr")
    store.cells.insert("every_cell", np.ones(len(cells), dtype=bool))
    shutil.copytree(folder, tmp_path / "vartrix")
    counts = store.variants.import_counts(
        tmp_path / "vartrix",
        cell_selection=store.snapshot_cell_selection("every_cell"),
        count_unit="reads",
    )
    result = variant_depth(store, counts, show=True)
    labels = [text.get_text() for text in result.axes["sites"].get_legend().get_texts()]
    assert labels == ["1", "2", "7", "17"]
    plt.close(result.figure)


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    """The clone truth's counts, their informative variants and clones."""
    import warnings

    from tests.clone_helpers import every_cell
    from tests.variant_helpers import write_variant_store
    from tests.variant_truth import mito_clone_truth, write_mgatk

    folder = tmp_path_factory.mktemp("scored")
    truth = mito_clone_truth()
    store = write_variant_store(folder / "store.zarr", truth)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        counts = store.variants.import_counts(
            write_mgatk(truth, folder / "final"), cell_selection=every_cell(store)
        )
    selection = store.variants.informative(counts, preset="atac", mask=None)
    clones = store.variants.frequency_clones(selection)
    return store, truth, counts, selection, clones


def test_variant_statistics_marks_informative_variants_and_thresholds(scored) -> None:
    from cytearc.plotting import variant_statistics

    store, _truth, _counts, selection, _clones = scored
    loaded = store.variants.load_informative(selection).variants
    result = variant_statistics(store, selection, show=False)
    table = result.tables["variants"]
    assert table["variant"].tolist() == loaded.index.tolist()
    assert table["informative"].tolist() == loaded["informative"].tolist()
    np.testing.assert_allclose(table["vmr"], loaded["vmr"])
    ax = result.axes["variants"]
    assert ax.get_yscale() == "log" and ax.get_xlabel() == "Strand correlation"
    # The VMR threshold is drawn across, the strand correlation threshold up.
    lines = {
        (round(line.get_xdata()[0], 6), round(line.get_ydata()[0], 6))
        for line in ax.get_lines()
    }
    assert (0.0, 0.01) in lines and (0.65, 0.0) in lines
    shown = np.isfinite(loaded["strand_correlation"].to_numpy(dtype=float))
    assert ax.get_title() == (
        f"{int((loaded['informative'].to_numpy() & shown).sum())} of "
        f"{int(shown.sum())} variants informative"
    )
    assert result.provenance.extras["x"] == "strand_correlation"
    plt.close(result.figure)


def test_variant_statistics_without_strands_draws_depth(tmp_path) -> None:
    import warnings

    from cytearc.plotting import variant_statistics
    from tests.clone_helpers import every_cell
    from tests.variant_helpers import write_variant_store
    from tests.variant_truth import variant_truth, write_cellsnp

    truth = variant_truth()
    store = write_variant_store(tmp_path / "store.zarr", truth)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        counts = store.variants.import_counts(
            write_cellsnp(truth, tmp_path / "cellsnp"),
            cell_selection=every_cell(store),
            count_unit="umis",
        )
    selection = store.variants.informative(counts, min_vmr=0.01, min_depth=5)
    result = variant_statistics(store, selection, show=False)
    ax = result.axes["variants"]
    assert ax.get_xlabel() == "Mean depth at the site" and ax.get_xscale() == "log"
    assert result.provenance.extras["x"] == "mean_coverage"
    assert result.tables["variants"]["strand_correlation"].isna().all()
    plt.close(result.figure)
    with pytest.raises(TypeError, match="does not provide variants.load_informative"):
        variant_statistics(object(), selection, show=False)


def _truth_frequency(truth, cell: str, variant: str) -> float:
    """A cell's allele frequency of a variant, from the truth's reads."""
    position = int(variant.split(":")[1][:-3])
    base = variant[-1]
    row = list(truth.clones.cell_ids).index(cell)
    reads = truth.mito[row, position - 1].sum(axis=1)
    depth = reads.sum()
    return np.nan if depth == 0 else reads["ACGT".index(base)] / depth


def test_allele_frequencies_order_cells_by_clone_and_keep_uncovered_apart(
    scored,
) -> None:
    from cytearc.plotting import allele_frequencies

    store, truth, _counts, selection, clones = scored
    result = allele_frequencies(store, selection, clones=clones, show=False)
    cells = result.tables["cells"]
    loaded = store.clones.load_clones(clones)
    with_clone = loaded.cells[loaded.cells["clone"].notna()]
    assert len(cells) == len(with_clone)
    # Cells follow the clones' ranks, largest clone first.
    order = [loaded.clones["clone"].tolist().index(label) for label in cells["clone"]]
    assert order == sorted(order)
    frequencies = result.tables["frequencies"]
    informative = store.variants.load_informative(selection).variants
    chosen = informative.index[informative["informative"]].tolist()
    assert set(frequencies["variant"]) == set(chosen)
    for row in frequencies.sample(200, random_state=0).itertuples():
        expected = _truth_frequency(truth, row.ids, row.variant)
        if np.isnan(expected):
            assert np.isnan(row.frequency)
        else:
            assert row.frequency == pytest.approx(expected)
    assert np.isnan(frequencies["frequency"]).any()
    assert "clones" in result.axes
    plt.close(result.figure)


def test_allele_frequencies_without_clones_draw_the_scored_cells(scored) -> None:
    from cytearc.plotting import allele_frequencies

    store, truth, _counts, selection, _clones = scored
    result = store.plots.allele_frequencies(selection, max_cells=40, show=False)
    cells = result.tables["cells"]
    assert len(cells) == 40 and cells["clone"].isna().all()
    ids = list(truth.clones.cell_ids)
    positions = [ids.index(cell) for cell in cells["ids"]]
    assert positions == sorted(positions)
    assert positions[0] == 0 and positions[-1] == len(ids) - 1
    assert "clones" not in result.axes
    plt.close(result.figure)
    nothing = store.variants.informative(_counts_of(store), min_vmr=1e9)
    with pytest.raises(ValueError, match="no informative variant"):
        allele_frequencies(store, nothing, show=False)


def _counts_of(store):
    return store.artifacts.list(kind="variant_counts", scope="datastore")[0]


def test_allele_frequencies_need_cells_with_a_clone(scored) -> None:
    from cytearc.plotting import allele_frequencies

    store, _truth, _counts, selection, _clones = scored
    store.cells.insert(
        "no_clone",
        np.full(len(store.cells.fetch_all("ids")), "", dtype=object),
        overwrite=True,
    )
    empty = store.clones.from_column(
        "no_clone", cell_selection=store.snapshot_cell_selection("every_cell")
    )
    with pytest.raises(ValueError, match="No cell has a clone"):
        allele_frequencies(store, selection, clones=empty, show=False)
    with pytest.raises(ValueError, match="max_cells"):
        allele_frequencies(store, selection, max_cells=0, show=False)


def _rcrs_counts(store, folder, alternative_reads):
    import warnings

    from tests.clone_helpers import every_cell
    from tests.variant_helpers import write_rcrs_mgatk

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return store.variants.import_counts(
            write_rcrs_mgatk(folder, alternative_reads),
            cell_selection=every_cell(store),
        )


def test_variant_statistics_mark_masked_variants_apart(tmp_path) -> None:
    from cytearc.plotting import variant_statistics
    from tests.variant_helpers import rcrs_store

    store = rcrs_store(tmp_path / "store.zarr")
    counts = _rcrs_counts(store, tmp_path / "final", (2, 4, 6))
    selection = store.variants.informative(counts, min_vmr=0.0, mask="artifact_prone")
    result = variant_statistics(store, selection, show=True)
    drawn = {
        collection.get_label(): np.asarray(collection.get_offsets())
        for collection in result.axes["variants"].collections
    }
    # 310 is the one artifact-prone position, and each cell reads as many
    # alternative reads on each strand, a strand correlation of 1; every
    # variant is informative or masked, so no other kind is drawn or listed.
    assert len(drawn["masked"]) == 1 and len(drawn["informative"]) == 5
    assert "other" not in drawn
    assert drawn["masked"][0, 0] == pytest.approx(1.0)
    legend = result.axes["variants"].get_legend()
    assert [text.get_text() for text in legend.get_texts()] == [
        "Informative",
        "Masked",
    ]
    assert result.scales[0].order == ("informative", "masked")
    table = result.tables["variants"]
    assert table.loc[table["masked"], "variant"].tolist() == ["chrM:310T>C"]
    plt.close(result.figure)


def test_variant_statistics_of_stranded_counts_never_draw_depth_instead(
    scored,
) -> None:
    from cytearc.plotting import variant_statistics

    store, truth, counts, _selection, _clones = scored
    # Scored over one cell, no variant has a strand correlation, though the
    # counts have strands and every variant a mean depth.
    store.cells.insert("only_first_cell", np.arange(truth.n_cells) == 0)
    selection = store.variants.informative(
        counts,
        cell_selection=store.snapshot_cell_selection("only_first_cell"),
        min_vmr=0.0,
    )
    loaded = store.variants.load_informative(selection)
    assert loaded.stranded and not loaded.variants.empty
    assert loaded.variants["strand_correlation"].isna().all()
    assert loaded.variants["mean_coverage"].notna().all()
    with pytest.raises(
        ValueError, match="^No variant has a strand correlation, so there is nothing"
    ):
        variant_statistics(store, selection, show=False)


def test_variant_statistics_need_a_variant_to_draw(tmp_path) -> None:
    from cytearc.plotting import variant_statistics
    from tests.variant_helpers import rcrs_store

    store = rcrs_store(tmp_path / "store.zarr")
    # Every read is a reference read, so no cell shows a variant.
    counts = _rcrs_counts(store, tmp_path / "final", (0, 0, 0))
    selection = store.variants.informative(counts, min_vmr=0.0)
    assert store.variants.load_informative(selection).variants.empty
    with pytest.raises(ValueError, match="No variant has a strand correlation"):
        variant_statistics(store, selection, show=False)


def test_allele_frequencies_show_and_need_a_store_with_clones(scored) -> None:
    from types import SimpleNamespace

    from cytearc.plotting import allele_frequencies

    store, _truth, _counts, selection, clones = scored
    result = allele_frequencies(
        store, selection, clones=clones, max_cells=30, show=True
    )
    assert len(result.tables["cells"]) == 30
    plt.close(result.figure)
    with pytest.raises(TypeError, match="does not provide clones.load_clones"):
        allele_frequencies(
            SimpleNamespace(variants=store.variants),
            selection,
            clones=clones,
            show=False,
        )


def _shares(result) -> dict[tuple[str, str], dict[str, float]]:
    """Each bar's share of each call that is not zero."""
    shares: dict[tuple[str, str], dict[str, float]] = {}
    for row in result.tables["calls"].itertuples(index=False):
        if row.share:
            shares.setdefault((row.site, row.group), {})[row.call] = row.share
    return shares


def test_genotypes_draw_each_sites_calls_by_group(plotted) -> None:
    store, targets, _positions = plotted
    called = store.variants.genotypes(targets, method="threshold", min_mutant_umis=1)
    # At CALR, c0 and c2 read a mutant UMI and c1 none; c3 reads no UMI and c4
    # is not named, so both are missing. At SF3B1 only c0 is read.
    result = genotypes(store, called, show=False)
    assert _shares(result) == {
        ("CALR", "all cells"): {
            "wild_type": 0.2,
            "alternative_present": 0.4,
            "missing": 0.4,
        },
        ("SF3B1", "all cells"): {"alternative_present": 0.2, "missing": 0.8},
    }
    calls = result.tables["calls"]
    assert calls.groupby(["site", "group"])["n_cells"].sum().tolist() == [5, 5]
    ax = result.axes["calls"]
    assert [label.get_text() for label in ax.get_yticklabels()] == ["CALR", "SF3B1"]
    # One segment per bar and call that some cell holds, named in words in
    # the vocabulary's order; the table keeps every call's token.
    assert len(ax.patches) == 2 * 3
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "Wild type",
        "Alternative present",
        "Missing",
    ]
    assert set(calls["call"]) == set(GENOTYPE_CALL_SCALE.order)
    # c4 has no group and is left out.
    grouped = genotypes(store, called, group_by="group", show=False)
    assert _shares(grouped) == {
        ("CALR", "x"): {"wild_type": 0.5, "alternative_present": 0.5},
        ("CALR", "y"): {"alternative_present": 0.5, "missing": 0.5},
        ("SF3B1", "x"): {"alternative_present": 0.5, "missing": 0.5},
        ("SF3B1", "y"): {"missing": 1.0},
    }
    assert grouped.provenance.n_cells == 4
    labels = [label.get_text() for label in grouped.axes["calls"].get_yticklabels()]
    assert labels == ["CALR, x", "CALR, y", "SF3B1, x", "SF3B1, y"]
    plt.close(result.figure)
    plt.close(grouped.figure)
    # Groups of two columns: c0 and c2 in a, c1 and c3 in b.
    store.cells.insert(
        "half", np.asarray(["a", "b", "a", "b", "a"], dtype=object), overwrite=True
    )
    crossed = genotypes(store, called, group_by=["group", "half"], show=False)
    assert _shares(crossed)[("CALR", "x, a")] == {"alternative_present": 1.0}
    assert _shares(crossed)[("CALR", "y, b")] == {"missing": 1.0}
    assert crossed.provenance.extras["grouping"] == "group, half"
    assert crossed.provenance.n_cells == 4
    plt.close(crossed.figure)
    # A label artifact of the labelled cells draws the groups of the column.
    labelled = np.asarray([group is not None for group in GROUPS])
    store.cells.insert("labelled", labelled, overwrite=True)
    labels = store.snapshot_cluster_labels(
        "group", cell_selection=store.snapshot_cell_selection("labelled")
    )
    by_labels = genotypes(store, called, groups=labels, show=False)
    assert _shares(by_labels) == _shares(grouped)
    assert by_labels.provenance.extras["grouping"] == "cluster_labels"
    plt.close(by_labels.figure)


def test_genotypes_colour_a_layout_by_one_sites_calls(plotted) -> None:
    store, targets, _positions = plotted
    called = store.variants.genotypes(targets, method="threshold", min_mutant_umis=1)
    layout = write_embedding(
        store, "every_cell", np.column_stack([np.arange(5), np.arange(5, 10)])
    )
    result = genotypes(store, called, layout=layout, sites=["CALR"], show=False)
    cells = result.tables["cells"]
    assert cells["ids"].tolist() == CELLS
    assert cells["x"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert cells["y"].tolist() == [5.0, 6.0, 7.0, 8.0, 9.0]
    assert cells["call"].tolist() == [
        "alternative_present",
        "wild_type",
        "alternative_present",
        "missing",
        "missing",
    ]
    panel = result.axes["layout"]
    assert panel.get_title() == "Calls at CALR"
    assert result.provenance.extras["layout"] == layout.to_dict()
    assert result.provenance.extras["site"] == "CALR"
    # One scatter of every cell in its call's colour, the cells without a
    # call, c3 and c4, drawn first so the others sit on top of them.
    (points,) = panel.collections
    drawn = [
        (tuple(offset), to_hex(color))
        for offset, color in zip(
            points.get_offsets(), points.get_facecolors(), strict=True
        )
    ]
    palette = GENOTYPE_CALL_SCALE.palette
    assert sorted(drawn[:2]) == [
        ((3.0, 8.0), palette["missing"]),
        ((4.0, 9.0), palette["missing"]),
    ]
    assert sorted(drawn[2:]) == [
        ((0.0, 5.0), palette["alternative_present"]),
        ((1.0, 6.0), palette["wild_type"]),
        ((2.0, 7.0), palette["alternative_present"]),
    ]
    # The one legend of the calls stands beside the layout, the last panel.
    assert result.axes["calls"].get_legend() is None
    assert panel.get_legend().get_title().get_text() == "Call"
    # With one site drawn, the layout takes its calls.
    alone = genotypes(store, called, sites=["SF3B1"], layout=layout, show=False)
    assert alone.tables["cells"]["call"].tolist() == [
        "alternative_present",
        *["missing"] * 4,
    ]
    shown = store.plots.genotypes(called, sites=["SF3B1"], layout=layout, show=False)
    assert shown.tables["cells"].equals(alone.tables["cells"])
    assert shown.tables["calls"].equals(alone.tables["calls"])
    for figure in (result.figure, alone.figure, shown.figure):
        plt.close(figure)


def test_genotypes_refuse_an_embedding_of_as_many_other_cells(
    plotted, tmp_path
) -> None:
    store, _targets, _positions = plotted
    store.cells.insert("first_four", np.arange(5) < 4, overwrite=True)
    store.cells.insert("last_four", np.arange(5) > 0, overwrite=True)
    calr = tmp_path / "CALR.tsv"
    calr.write_text("BC\twtUMIs\tmutUMIs\nc0\t3\t1\nc1\t2\t0\n")
    counts = store.variants.import_counts(
        calr, cell_selection=store.snapshot_cell_selection("first_four")
    )
    called = store.variants.genotypes(counts, method="threshold", min_mutant_umis=1)
    # Four cells of an embedding are not the genotypes' four cells.
    others = write_embedding(store, "last_four", np.zeros((4, 2)))
    with pytest.raises(ValueError, match="embedding of the genotypes' cells"):
        genotypes(store, called, layout=others, show=False)
    own = write_embedding(store, "first_four", np.arange(8).reshape(4, 2))
    drawn = genotypes(store, called, layout=own, show=False)
    cells = drawn.tables["cells"]
    assert cells["ids"].tolist() == ["c0", "c1", "c2", "c3"]
    assert cells["x"].tolist() == [0.0, 2.0, 4.0, 6.0]
    assert cells["call"].tolist() == [
        "alternative_present",
        "wild_type",
        "missing",
        "missing",
    ]
    plt.close(drawn.figure)


def test_genotypes_show_and_close_the_figure_they_own(plotted) -> None:
    store, targets, _positions = plotted
    called = store.variants.genotypes(targets, method="threshold", min_mutant_umis=1)
    result = genotypes(store, called, show=True)
    assert result.owns_figure
    assert result.figure.number not in plt.get_fignums()


def test_genotypes_refuse_what_they_cannot_draw(plotted) -> None:
    store, targets, _positions = plotted
    called = store.variants.genotypes(targets, method="threshold", min_mutant_umis=1)
    store.cells.insert("nobody", np.asarray([None] * 5, dtype=object))
    with pytest.raises(ValueError, match="No cell of the genotypes has a group"):
        genotypes(store, called, group_by="nobody", show=False)
    flat = write_embedding(store, "every_cell", np.zeros((5, 2)))
    with pytest.raises(ValueError, match="one site, and the plot draws 2; name one"):
        genotypes(store, called, layout=flat, show=False)
    # Sites are named one way, by sites.
    with pytest.raises(TypeError, match="unexpected keyword argument 'site'"):
        genotypes(store, called, sites=["CALR"], layout=flat, site="SF3B1", show=False)
    with pytest.raises(ValueError, match="have no site 'TP53'"):
        genotypes(store, called, sites=["TP53"], layout=flat, show=False)
    # The layout is an embedding of the genotypes' own cells.
    store.cells.insert("first_four", np.arange(5) < 4, overwrite=True)
    fewer = write_embedding(store, "first_four", np.zeros((4, 2)))
    with pytest.raises(ValueError, match="embedding of the genotypes' cells"):
        genotypes(store, called, sites=["CALR"], layout=fewer, show=False)
    with pytest.raises(ValueError, match="must identify an embedding artifact"):
        genotypes(store, called, sites=["CALR"], layout=targets, show=False)
    with pytest.raises(TypeError, match="layout must be an ArtifactRef"):
        genotypes(store, called, sites=["CALR"], layout="UMAP", show=False)
    with pytest.raises(ValueError, match="variant_genotypes artifact"):
        genotypes(store, targets, show=False)
    with pytest.raises(TypeError, match="does not provide variants.load_genotypes"):
        genotypes(object(), called, show=False)


def test_genotypes_refuse_a_layout_of_another_type_before_reading_the_store(
    plotted,
) -> None:
    store, targets, _positions = plotted
    called = store.variants.genotypes(targets, method="threshold", min_mutant_umis=1)
    # The text that layout_key took names no embedding: it is refused as the
    # argument it is, before the calls of two sites ask which site to draw.
    for layout in ("UMAP", 3):
        with pytest.raises(TypeError, match="^layout must be an ArtifactRef or None$"):
            genotypes(store, called, layout=layout, show=False)
    # Nothing is read from the store before the refusal.
    with pytest.raises(TypeError, match="^layout must be an ArtifactRef or None$"):
        genotypes(object(), called, layout="UMAP", show=False)


def test_genotypes_of_many_sites_and_groups_keep_a_drawable_height(tmp_path) -> None:
    import warnings

    from tests.clone_helpers import every_cell, write_cells_store

    # 120 sites of 400 cells in 8 groups draw 960 bars.
    rng = np.random.default_rng(3)
    ids = [f"cell{cell:03d}-1" for cell in range(400)]
    store = write_cells_store(tmp_path / "store.zarr", ids)
    store.cells.insert(
        "group", np.asarray([f"g{cell % 8}" for cell in range(400)], dtype=object)
    )
    paths = []
    for site in range(120):
        lines = ["BC\twtUMIs\tmutUMIs"] + [
            f"{ids[cell]}\t{int(rng.integers(0, 3))}\t{int(rng.integers(0, 2))}"
            for cell in range(400)
        ]
        path = tmp_path / f"S{site:03d}.tsv"
        path.write_text("\n".join(lines) + "\n")
        paths.append(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        counts = store.variants.import_counts(paths, cell_selection=every_cell(store))
    called = store.variants.genotypes(counts, method="threshold", min_mutant_umis=1)
    result = genotypes(store, called, group_by="group", show=False)
    shares = result.tables["calls"].groupby(["site", "group"])["share"].sum()
    assert len(shares) == 960
    np.testing.assert_allclose(shares.to_numpy(), 1.0)
    # Each bar is drawn; an axis names at most 40, and the panel stops
    # growing where its names stop: 1.2 inches and 0.22 for each of 40.
    ax = result.axes["calls"]
    assert len({round(patch.get_y(), 6) for patch in ax.patches}) == 960
    assert not ax.get_yticks().size
    assert result.figure.get_size_inches() == pytest.approx((6.0, 1.2 + 0.22 * 40))
    plt.close(result.figure)


def test_variant_depth_refuses_counts_without_depth_or_with_both_kinds_of_site(
    plotted, tmp_path
) -> None:
    from types import SimpleNamespace

    store, targets, _positions = plotted
    # Both cells are named at CALR without a UMI, so no site has depth.
    unread = tmp_path / "CALR.tsv"
    unread.write_text("BC\twtUMIs\tmutUMIs\nc0\t0\t0\nc1\t0\t0\n")
    counts = store.variants.import_counts(
        unread, cell_selection=store.snapshot_cell_selection("every_cell")
    )
    with pytest.raises(
        ValueError, match="^No site of the counts has depth, so there is nothing"
    ):
        variant_depth(store, counts, show=False)
    # One panel draws positions along a contig or bars of named sites, never
    # both: counts of a site with a position beside a named one are refused.
    real = store.variants.load_counts

    def mixed(artifact, **options):
        loaded = real(artifact, **options)
        sites = loaded.sites.assign(position=[7, -1], contig=["chr1", ""])
        return SimpleNamespace(
            sites=sites, cells=loaded.cells, count_unit=loaded.count_unit
        )

    both = SimpleNamespace(variants=SimpleNamespace(load_counts=mixed))
    figures = set(plt.get_fignums())
    with pytest.raises(ValueError, match="sites with coordinates and named sites"):
        variant_depth(both, targets, show=False)
    assert set(plt.get_fignums()) == figures


def test_a_depth_panel_of_groups_without_depth_says_so(plotted) -> None:
    store, targets, _positions = plotted
    # c3 is named at CALR without a UMI, so the one grouped cell has no depth,
    # while the sites hold the depth of every named cell.
    store.cells.insert(
        "only_c3", np.asarray([None, None, None, "z", None], dtype=object)
    )
    result = variant_depth(store, targets, group_by="only_c3", show=False)
    try:
        cells = result.axes["cells"]
        assert [text.get_text() for text in cells.texts] == [
            "No cell of a group has depth"
        ]
        assert not cells.patches and cells.get_legend() is None
        assert result.legends == () and result.scales == ()
        assert result.tables["cells"]["ids"].tolist() == ["c3"]
        assert len(result.axes["sites"].patches) == 2
    finally:
        result.close()


def test_variant_depth_stands_in_one_column_with_plain_log_labels(plotted) -> None:
    store, _targets, positions = plotted
    result = variant_depth(store, positions, group_by="group", show=False)
    try:
        assert tuple(result.figure.get_size_inches()) == (7.5, 5.2)
        sites, cells = result.axes["sites"], result.axes["cells"]
        assert sites.get_subplotspec().colspan == cells.get_subplotspec().colspan
        assert sites.get_xlabel() == "Position on chrM"
        # One contig draws one line in the theme's foreground, without a
        # legend; the groups' histograms stand under a legend of the grouping.
        (line,) = sites.get_lines()
        assert to_hex(line.get_color()) == "#333333"
        assert sites.get_legend() is None
        assert cells.get_legend().get_title().get_text() == "group"
        assert [legend.label for legend in result.legends] == ["group"]
        result.figure.canvas.draw()
        for axis in (sites.yaxis, cells.xaxis):
            labels = [
                tick.get_text()
                for tick in [*axis.get_ticklabels(), *axis.get_ticklabels(minor=True)]
                if tick.get_text()
            ]
            assert labels and all(float(label.replace(",", "")) > 0 for label in labels)
        assert result.provenance.notes == ("variant_depth",)
        assert result.provenance.extras["counts"] == positions.to_dict()
    finally:
        result.close()


def test_variant_statistics_mute_the_variants_that_carry_no_lineage(tmp_path) -> None:
    from cytearc.plotting import variant_statistics
    from tests.variant_helpers import rcrs_store

    store = rcrs_store(tmp_path / "store.zarr")
    counts = _rcrs_counts(store, tmp_path / "final", (2, 4, 6))
    # A threshold no variant passes leaves the variants not informative.
    for options, kinds in (
        ({"min_vmr": 0.0, "mask": "artifact_prone"}, {"informative", "masked"}),
        ({"min_vmr": 1e9}, {"other"}),
    ):
        selection = store.variants.informative(counts, **options)
        result = variant_statistics(store, selection, show=False)
        try:
            colors = {
                collection.get_label(): {
                    to_hex(color) for color in collection.get_facecolors()
                }
                for collection in result.axes["variants"].collections
            }
            assert set(colors) == kinds
            for kind in kinds - {"informative"}:
                assert colors[kind] == {MUTED_COLOR}
            if "informative" in kinds:
                assert MUTED_COLOR not in colors["informative"]
            scale = result.scales[0]
            assert set(scale.order) == kinds
            assert result.legends[0].label == "Variant"
        finally:
            result.close()


def test_allele_frequencies_say_how_many_cells_they_draw_and_name_their_clones(
    scored,
) -> None:
    from matplotlib.colors import to_rgba

    from cytearc.plotting import allele_frequencies

    store, _truth, _counts, selection, clones = scored
    loaded = store.clones.load_clones(clones)
    with_clone = int(loaded.cells["clone"].notna().sum())
    capped = allele_frequencies(
        store, selection, clones=clones, max_cells=30, show=False
    )
    whole = allele_frequencies(store, selection, clones=clones, show=False)
    try:
        assert capped.axes["frequencies"].get_xlabel() == f"Cells (30 of {with_clone})"
        assert whole.axes["frequencies"].get_xlabel() == "Cells"
        # The clones drawn, largest first, named under the clones' noun.
        strip = whole.axes["clones"]
        legend = strip.get_legend()
        assert legend.get_title().get_text() == "Clone"
        drawn = list(dict.fromkeys(whole.tables["cells"]["clone"]))
        assert [text.get_text() for text in legend.get_texts()] == drawn
        assert drawn == [
            clone for clone in loaded.clones["clone"] if clone in set(drawn)
        ]
        assert [tick.get_text() for tick in strip.get_yticklabels()] == ["Clone"]
        assert [legend.label for legend in whole.legends] == [
            "Allele frequency (grey: not covered)",
            "Clone",
        ]
        assert whole.legends[0].extras == {"vmin": 0.0, "vmax": 1.0}
        informative = store.variants.load_informative(selection).variants
        named = informative.index[informative["informative"]].tolist()
        ax = whole.axes["frequencies"]
        assert [tick.get_text() for tick in ax.get_yticklabels()] == named
        assert not ax.get_xticklabels()
        # Uncovered entries take the grey of no value, in a layer over the
        # frequencies.
        frequencies = whole.tables["frequencies"].pivot(
            index="variant", columns="ids", values="frequency"
        )
        frequencies = frequencies.loc[named, whole.tables["cells"]["ids"]]
        uncovered = frequencies.isna().to_numpy()
        assert uncovered.any()
        _image, layer = ax.images
        pixels = np.asarray(layer.get_array())
        assert np.allclose(pixels[uncovered], to_rgba(NO_VALUE_COLOR))
        assert np.allclose(pixels[~uncovered], 0.0)
    finally:
        capped.close()
        whole.close()


def test_genotype_calls_take_one_palette_with_greys_for_no_call(plotted) -> None:
    store, targets, _positions = plotted
    # Of the four UMIs or fewer that c0 to c3 read at CALR, c1's two and
    # c3's none fall short of three, so a rule does not call them; c4 is
    # not read, so it is missing.
    called = store.variants.genotypes(
        targets, method="threshold", min_umis=3, min_mutant_umis=1
    )
    result = genotypes(store, called, sites=["CALR"], show=False)
    try:
        table = result.tables["calls"].set_index("call")["n_cells"]
        held = [call for call in GENOTYPE_CALL_SCALE.order if table[call] > 0]
        assert {"missing", "filtered"} <= set(held)
        ax = result.axes["calls"]
        colors = [to_hex(patch.get_facecolor()) for patch in ax.patches]
        assert colors == [GENOTYPE_CALL_SCALE.palette[call] for call in held]
        assert GENOTYPE_CALL_SCALE.palette["missing"] == NO_VALUE_COLOR
        assert GENOTYPE_CALL_SCALE.palette["filtered"] == MUTED_COLOR
        # From the left, the calls in the legend's order at their fractions.
        widths = [patch.get_width() for patch in ax.patches]
        assert widths == pytest.approx([table[call] / 5 for call in held])
        lefts = [patch.get_x() for patch in ax.patches]
        assert lefts == pytest.approx(np.cumsum([0.0, *widths[:-1]]))
        assert ax.get_xlabel() == "Fraction of cells"
        assert result.scales[0].order == tuple(held)
    finally:
        result.close()


def test_allele_frequencies_of_many_variants_name_none(scored) -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from cytearc.plotting import allele_frequencies

    store, _truth, _counts, selection, _clones = scored
    loaders = store.variants

    def every_informative(ref):
        # Each of the 159 variants that a cell shows, as informative.
        loaded = loaders.load_informative(ref)
        return replace(loaded, variants=loaded.variants.assign(informative=True))

    every = SimpleNamespace(
        variants=SimpleNamespace(
            load_informative=every_informative, load_counts=loaders.load_counts
        )
    )
    result = allele_frequencies(every, selection, max_cells=50, show=False)
    try:
        ax = result.axes["frequencies"]
        n_variants = len(loaders.load_informative(selection).variants)
        assert n_variants > 40
        assert result.tables["frequencies"]["variant"].nunique() == n_variants
        assert result.provenance.extras["n_variants"] == n_variants
        # An axis names at most 40 items, and cells never.
        assert not ax.get_yticklabels() and not ax.get_xticklabels()
    finally:
        result.close()
