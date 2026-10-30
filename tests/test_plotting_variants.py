"""The variant depth plot, on counts whose depths are known by hand."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cytearc.datastore.datastore import DataStore
from cytearc.plotting import variant_depth
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
