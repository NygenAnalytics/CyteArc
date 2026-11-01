"""The variant plots, on counts whose depths and calls are known by hand."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cytearc.datastore.datastore import DataStore
from cytearc.plotting import genotypes, variant_depth
from tests.storage_helpers import write_count_store

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
    store = DataStore(
        str(folder / "store.zarr"),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
    )
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
    assert cells["cell"].tolist() == ["c0", "c1", "c2", "c3"]
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
    assert cells["cell"].tolist() == ["c0", "c1", "c2", "c3"]
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
    with pytest.raises(ValueError, match="Pass group_by or groups, not both"):
        variant_depth(store, targets, group_by="group", groups=targets, show=False)
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
    store = DataStore(
        str(tmp_path / "store.zarr"),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
    )
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
    clones = store.variants.clones(counts, variants=selection)
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
        f"{int((loaded['informative'].to_numpy() & shown).sum())} informative of "
        f"{int(shown.sum())}"
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
        expected = _truth_frequency(truth, row.cell, row.variant)
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
    positions = [ids.index(cell) for cell in cells["cell"]]
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
    # alternative reads on each strand, a strand correlation of 1.
    assert len(drawn["masked"]) == 1 and len(drawn["informative"]) == 5
    assert len(drawn["other"]) == 0
    assert drawn["masked"][0, 0] == pytest.approx(1.0)
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
        counts, cells=store.snapshot_cell_selection("only_first_cell"), min_vmr=0.0
    )
    loaded = store.variants.load_informative(selection)
    assert loaded.stranded and not loaded.variants.empty
    assert loaded.variants["strand_correlation"].isna().all()
    assert loaded.variants["mean_coverage"].notna().all()
    with pytest.raises(ValueError, match="No variant has a strand correlation to plot"):
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
    assert calls.groupby(["site", "group"])["cells"].sum().tolist() == [5, 5]
    ax = result.axes["calls"]
    assert [label.get_text() for label in ax.get_yticklabels()] == ["CALR", "SF3B1"]
    # One segment per bar and call, in the vocabulary's order.
    assert len(ax.patches) == 2 * 6
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "wild_type",
        "heterozygous",
        "homozygous_alternative",
        "alternative_present",
        "missing",
        "filtered",
    ]
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


def test_genotypes_colour_a_layout_by_one_sites_calls(plotted) -> None:
    store, targets, _positions = plotted
    called = store.variants.genotypes(targets, method="threshold", min_mutant_umis=1)
    store.cells.insert("UMAP1", np.arange(5, dtype=np.float64))
    store.cells.insert("UMAP2", np.arange(5, 10, dtype=np.float64))
    result = genotypes(store, called, layout_key="UMAP", site="CALR", show=False)
    cells = result.tables["cells"]
    assert cells["cell"].tolist() == CELLS
    assert cells["x"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert cells["y"].tolist() == [5.0, 6.0, 7.0, 8.0, 9.0]
    assert cells["call"].tolist() == [
        "alternative_present",
        "wild_type",
        "alternative_present",
        "missing",
        "missing",
    ]
    layout = result.axes["layout"]
    assert layout.get_title() == "CALR"
    # One point set per call that a cell holds.
    offsets = sorted(
        tuple(map(tuple, collection.get_offsets())) for collection in layout.collections
    )
    assert offsets == [
        ((0.0, 5.0), (2.0, 7.0)),
        ((1.0, 6.0),),
        ((3.0, 8.0), (4.0, 9.0)),
    ]
    # With one site drawn, the layout takes its calls.
    alone = genotypes(store, called, sites=["SF3B1"], layout_key="UMAP", show=False)
    assert alone.tables["cells"]["call"].tolist() == [
        "alternative_present",
        *["missing"] * 4,
    ]
    shown = store.plots.genotypes(
        called, sites=["SF3B1"], layout_key="UMAP", show=False
    )
    assert shown.tables["cells"].equals(alone.tables["cells"])
    assert shown.tables["calls"].equals(alone.tables["calls"])
    for figure in (result.figure, alone.figure, shown.figure):
        plt.close(figure)


def test_genotypes_refuse_what_they_cannot_draw(plotted) -> None:
    store, targets, _positions = plotted
    called = store.variants.genotypes(targets, method="threshold", min_mutant_umis=1)
    store.cells.insert("nobody", np.asarray([None] * 5, dtype=object))
    with pytest.raises(ValueError, match="No cell of the genotypes has a group"):
        genotypes(store, called, group_by="nobody", show=False)
    store.cells.insert("XY1", np.zeros(5))
    store.cells.insert("XY2", np.zeros(5))
    with pytest.raises(ValueError, match="draws 2 sites; name the one"):
        genotypes(store, called, layout_key="XY", show=False)
    with pytest.raises(ValueError, match="site 'SF3B1' is not among the drawn"):
        genotypes(
            store, called, sites=["CALR"], layout_key="XY", site="SF3B1", show=False
        )
    with pytest.raises(ValueError, match="variant_genotypes artifact"):
        genotypes(store, targets, show=False)
    with pytest.raises(TypeError, match="does not provide variants.load_genotypes"):
        genotypes(object(), called, show=False)


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
    assert len(result.axes["calls"].get_yticks()) == 960
    assert result.figure.get_size_inches()[1] == 40.0
    plt.close(result.figure)
