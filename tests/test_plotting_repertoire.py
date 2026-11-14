"""The receptor QC plot and clonotypes on an embedding, on eight cells' contigs."""

import hashlib
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

from cytearc import plotting
from cytearc.datastore.datastore import DataStore
from tests.clone_helpers import N_RECEPTOR_CELLS, bcr_file, receptor_store, tcr_file
from tests.storage_helpers import insert_nullable_cell_column

pytestmark = pytest.mark.usefixtures("figures_closed")

N_CELLS = N_RECEPTOR_CELLS


@pytest.fixture(scope="module")
def receptor_template(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("receptor_plot")
    store = receptor_store(folder)
    store.repertoire.import_receptors(
        [tcr_file(folder), bcr_file(folder)],
        cell_selection=store.snapshot_cell_selection(),
    )
    return folder


def _receptors(template: Path, tmp_path: Path) -> tuple[DataStore, object]:
    """A copy of the store with the TCR and BCR rows imported."""
    for name in ("store.zarr", "tcr.csv", "bcr.csv"):
        source = template / name
        if source.is_dir():
            shutil.copytree(source, tmp_path / name)
        else:
            shutil.copy(source, tmp_path / name)
    store = receptor_store(tmp_path)
    ref = store.repertoire.import_receptors(
        [tmp_path / "tcr.csv", tmp_path / "bcr.csv"],
        cell_selection=store.snapshot_cell_selection(),
    )
    return store, ref


@pytest.fixture
def receptors(receptor_template, tmp_path) -> tuple[DataStore, object]:
    return _receptors(receptor_template, tmp_path)


def _wide(table: pd.DataFrame, column: str) -> pd.DataFrame:
    """A table of cells per group and category, as categories by groups."""
    wide = table.pivot(index=column, columns="group", values="n_cells")
    return wide.reindex(
        index=pd.unique(table[column]), columns=pd.unique(table["group"])
    )


def test_receptor_qc_counts_cells_by_category(receptors) -> None:
    store, ref = receptors
    result = store.plots.receptor_qc(ref, show=False)
    try:
        assert set(result.axes) == {"receptor_type", "chain_pairing", "chains_per_cell"}
        # Every cell is in one group without a grouping, and each category
        # has a row, in the vocabulary's order.
        assert result.tables["receptor_type"].to_dict("list") == {
            "group": ["all cells"] * 5,
            "receptor_type": ["TCR", "BCR", "ambiguous", "multichain", "no IR"],
            "n_cells": [2, 1, 1, 1, 3],
        }
        pairing = result.tables["chain_pairing"]
        assert pairing.columns.tolist() == ["group", "chain_pairing", "n_cells"]
        assert pairing["n_cells"].sum() == N_CELLS
        chains = result.tables["chains_per_cell"]
        assert chains.index.tolist() == ["0", "1", "2", "3+"]
        assert chains.index.name == "n_chains"
        assert chains["VJ"].tolist() == [5, 2, 1, 0]
        assert chains["VDJ"].tolist() == [3, 3, 1, 1]
        # One bar per category with cells.
        assert len(result.axes["receptor_type"].patches) == 5
        assert result.axes["receptor_type"].get_xlabel() == "Cells"
        assert result.provenance.n_cells == N_CELLS
        assert result.provenance.extras["grouping"] is None
    finally:
        result.close()


def test_receptor_qc_stacks_groups_of_a_cell_column(receptors) -> None:
    store, ref = receptors
    store.cells.insert("sample", np.asarray(["a", "a", "b", "b", "a", "b", "a", "b"]))
    result = plotting.receptor_qc(store, ref, group_by="sample", show=False)
    try:
        long = result.tables["receptor_type"]
        assert long.columns.tolist() == ["group", "receptor_type", "n_cells"]
        # One row per group and receptor type, zeros included, by group.
        assert long["group"].tolist() == ["a"] * 5 + ["b"] * 5
        table = _wide(long, "receptor_type")
        assert table.columns.tolist() == ["a", "b"]
        assert table.loc["TCR"].tolist() == [1, 1]
        assert table.loc["no IR"].tolist() == [1, 2]
        assert table.sum().tolist() == [4, 4]
        assert result.axes["receptor_type"].get_ylabel() == "Fraction of cells"
        heights = [patch.get_height() for patch in result.axes["receptor_type"].patches]
        assert np.isclose(sum(heights), 2.0)
        assert result.provenance.extras["grouping"] == "sample"
        assert result.provenance.extras["normalize"] is True
    finally:
        result.close()
    counts = plotting.receptor_qc(
        store, ref, group_by="sample", normalize=False, show=False
    )
    try:
        assert counts.axes["chain_pairing"].get_ylabel() == "Cells"
    finally:
        counts.close()


def test_receptor_qc_groups_by_labels_of_fewer_cells(receptors) -> None:
    store, ref = receptors
    keep = np.zeros(N_CELLS, dtype=bool)
    keep[:4] = True
    store.cells.insert("first", keep)
    store.cells.insert("cluster", np.asarray(["x", "y", "x", "y", "x", "y", "x", "y"]))
    labels = store.snapshot_cluster_labels(
        "cluster", cell_selection=store.snapshot_cell_selection("first")
    )
    result = plotting.receptor_qc(store, ref, groups=labels, show=False)
    try:
        table = _wide(result.tables["chain_pairing"], "chain_pairing")
        assert table.sum().tolist() == [2, 2]
        assert result.provenance.n_cells == 4
        assert result.tables["chains_per_cell"].sum().tolist() == [4, 4]
        # The labels are titled by the column they copied.
        assert result.provenance.extras["grouping"] == "cluster"
    finally:
        result.close()


def _layout(store: DataStore) -> object:
    """A layout of every cell of the store, which the receptors share."""
    from cytearc.embeddings.imported import write_imported_embedding
    from cytearc.storage.artifacts import fingerprint_array

    coordinates = np.column_stack([np.arange(N_CELLS), np.zeros(N_CELLS)]).astype(
        np.float32
    )
    return write_imported_embedding(
        store.zw,
        assay="RNA",
        dimreduc_key="umap",
        role="umap",
        coordinates=coordinates,
        source_digest=hashlib.sha256(b"layout").digest(),
        payload_fingerprints={"values": fingerprint_array(coordinates)},
        source_cell_ids=np.asarray(store.cells.fetch_all("ids")).astype(str),
        cell_selection=store.snapshot_cell_selection(),
        block_rows=4,
    )


def _drawn_colors(result: plotting.PlotResult) -> list[str]:
    """The color of each cell, in stored order when the plot drew them so."""
    (points,) = result.axes[next(iter(result.axes))].collections
    return [to_hex(color) for color in points.get_facecolors()]


def test_embedding_draws_receptor_types_in_the_colors_of_receptor_qc(
    receptors,
) -> None:
    store, ref = receptors
    types = store.repertoire.load_receptors(ref).cells["receptor_type"].astype(str)
    assert types.nunique() == 5
    qc = store.plots.receptor_qc(ref, show=False)
    try:
        # Without a grouping, the bars of all cells stack one per category.
        categories = qc.tables["receptor_type"]["receptor_type"].astype(str)
        bars = [to_hex(bar.get_facecolor()) for bar in qc.axes["receptor_type"].patches]
        assert len(bars) == len(categories)
        qc_colors = dict(zip(categories, bars, strict=True))
    finally:
        qc.close()
    result = store.plots.embedding(
        layout=_layout(store), color_by=ref, seed=None, show=False
    )
    try:
        assert _drawn_colors(result) == [qc_colors[name] for name in types]
    finally:
        result.close()


def test_embedding_draws_receptor_types_in_a_given_scale(receptors) -> None:
    store, ref = receptors
    types = store.repertoire.load_receptors(ref).cells["receptor_type"].astype(str)
    palette = {
        "TCR": "#000000",
        "BCR": "#111111",
        "ambiguous": "#222222",
        "multichain": "#333333",
        "no IR": "#444444",
    }
    result = store.plots.embedding(
        layout=_layout(store),
        color_by=ref,
        categorical_scale=plotting.CategoricalScale(palette=palette),
        seed=None,
        show=False,
    )
    try:
        assert _drawn_colors(result) == [palette[name] for name in types]
    finally:
        result.close()


def test_embedding_draws_cells_without_a_clonotype_as_missing(receptors) -> None:
    store, ref = receptors
    layout = _layout(store)
    clonotypes = store.repertoire.clonotypes(ref)
    labels = store.artifacts.load_values(clonotypes).to_pandas()
    without = int((labels.astype(str) == "").sum())
    assert without > 0
    result = store.plots.embedding(layout=layout, color_by=clonotypes, show=False)
    try:
        scale = next(
            scale
            for scale in result.scales
            if isinstance(scale, plotting.CategoricalScale)
        )
        # Blank labels are missing: drawn in the missing color, not as a category.
        assert "" not in scale.palette
        assert set(scale.palette) == set(labels[labels.astype(str) != ""])
        drawn = _drawn_colors(result)
        assert drawn.count(to_hex(scale.missing_color)) == without
    finally:
        result.close()


def test_receptor_qc_groups_by_integer_labels_of_fewer_cells(receptors) -> None:
    # Integer labels, as clusterings hold them, align as objects; the cells
    # that the labels lack get no group.
    store, ref = receptors
    keep = np.zeros(N_CELLS, dtype=bool)
    keep[:4] = True
    store.cells.insert("first", keep)
    store.cells.insert("cluster", np.arange(N_CELLS) % 2)
    labels = store.snapshot_cluster_labels(
        "cluster", cell_selection=store.snapshot_cell_selection("first")
    )
    result = plotting.receptor_qc(store, ref, groups=labels, show=False)
    try:
        pairing = _wide(result.tables["chain_pairing"], "chain_pairing")
        assert pairing.sum().tolist() == [2, 2]
        assert result.provenance.n_cells == 4
    finally:
        result.close()


def test_receptor_qc_draws_on_caller_axes(receptors) -> None:
    store, ref = receptors
    figure, axes = plt.subplots(1, 3)
    try:
        result = plotting.receptor_qc(
            store,
            ref,
            target={
                "receptor_type": axes[0],
                "chain_pairing": axes[1],
                "chains_per_cell": axes[2],
            },
            show_legend=False,
            show=False,
        )
        assert result.figure is figure
        assert result.owns_figure is False
        assert axes[2].get_legend() is None
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"group_by": "sample", "groups": "x"}, TypeError, "groups must be an Artif"),
        ({"group_by": 3}, TypeError, "group_by must be a column name"),
        ({"group_by": ["sample", "sample"]}, ValueError, "names a column twice"),
        ({"groups": "clusters"}, TypeError, "groups must be an ArtifactRef"),
        ({"normalize": "yes"}, TypeError, "normalize must be a boolean"),
    ],
)
def test_receptor_qc_checks_its_arguments(receptors, kwargs, error, message) -> None:
    store, ref = receptors
    store.cells.insert("sample", np.asarray(list("aabbaabb")))
    with pytest.raises(error, match=message):
        plotting.receptor_qc(store, ref, show=False, **kwargs)


def test_receptor_qc_groups_cells_by_several_columns(receptors) -> None:
    store, ref = receptors
    sample = np.asarray(["a", "a", "b", "b", "a", "b", "a", "b"])
    lane = np.asarray(["p", "q", "p", "q", "q", "q", "p", "p"])
    store.cells.insert("sample", sample)
    store.cells.insert("lane", lane)
    types = store.repertoire.load_receptors(ref, fields=()).cells["receptor_type"]
    result = plotting.receptor_qc(
        store, ref, group_by=["sample", "lane"], normalize=False, show=False
    )
    try:
        table = _wide(result.tables["receptor_type"], "receptor_type")
        # One column per combination of sample and lane, its cells counted by
        # receptor type cell by cell.
        groups = [f"{s}, {lane_}" for s, lane_ in zip(sample, lane, strict=True)]
        assert table.columns.tolist() == sorted(set(groups))
        for group in table.columns:
            held = [str(t) for t, g in zip(types, groups, strict=True) if g == group]
            assert table[group].to_dict() == {
                name: held.count(name) for name in table.index
            }
        assert result.provenance.extras["grouping"] == "sample, lane"
        assert result.legends[0].label == "Receptor type"
    finally:
        result.close()
    keep = np.zeros(N_CELLS, dtype=bool)
    keep[:6] = True
    store.cells.insert("first_six", keep)
    labels = store.snapshot_cluster_labels(
        "lane", cell_selection=store.snapshot_cell_selection("first_six")
    )
    # A column and a label artifact of fewer cells group the cells together.
    both = plotting.receptor_qc(
        store, ref, group_by="sample", groups=labels, show=False
    )
    try:
        assert both.tables["chain_pairing"]["group"].drop_duplicates().tolist() == [
            "a, p",
            "a, q",
            "b, p",
            "b, q",
        ]
        assert both.provenance.n_cells == 6
        assert both.provenance.extras["grouping"] == "sample, lane"
        named = plotting.receptor_qc(
            store,
            ref,
            group_by="sample",
            groups=plotting.NamedCellArtifact("Lane", labels),
            show=False,
        )
        assert named.provenance.extras["grouping"] == "sample, Lane"
        named.close()
    finally:
        both.close()


@pytest.mark.parametrize(
    "label", ["n_cells", "group", "receptor_type", "chain_pairing", "all cells"]
)
def test_receptor_qc_draws_one_group_named_like_the_tables_columns(
    receptors, label
) -> None:
    store, ref = receptors
    drawn = {}
    for name in (label, "s1"):
        store.cells.insert("batch", np.asarray([name] * N_CELLS), overwrite=True)
        result = plotting.receptor_qc(store, ref, group_by="batch", show=False)
        try:
            ax = result.axes["receptor_type"]
            table = result.tables["receptor_type"]
            assert table["group"].tolist() == [name] * len(table)
            drawn[name] = {
                "table": table.drop(columns="group").to_numpy().tolist(),
                "ticks": [tick.get_text() for tick in ax.get_xticklabels()],
                "heights": sorted(round(bar.get_height(), 12) for bar in ax.patches),
                "legend": ax.get_legend() is not None,
                "axis": ax.get_ylabel(),
            }
        finally:
            result.close()
    # One group of every cell draws one stacked bar of fractions, whatever its
    # label, and a label named like a column of the tables changes nothing
    # else: the tables hold it as a value.
    assert drawn[label]["ticks"] == [label]
    assert drawn["s1"]["ticks"] == ["s1"]
    for part in ("table", "heights", "legend", "axis"):
        assert drawn[label][part] == drawn["s1"][part], part
    assert drawn[label]["legend"] and drawn[label]["axis"] == "Fraction of cells"
    assert sum(drawn[label]["heights"]) == pytest.approx(1.0)


def test_receptor_qc_needs_a_store_with_receptors(receptors) -> None:
    _store_unused, ref = receptors
    with pytest.raises(TypeError, match="does not provide immune receptors"):
        plotting.receptor_qc(object(), ref, show=False)


def test_receptor_qc_leaves_out_cells_with_a_missing_group(receptors) -> None:
    store, ref = receptors
    missing = np.zeros(N_CELLS, dtype=bool)
    missing[[0, 7]] = True
    insert_nullable_cell_column(store, "batch", np.asarray(list("pqpqpqpq")), missing)
    result = plotting.receptor_qc(store, ref, group_by="batch", show=False)
    try:
        assert _wide(
            result.tables["receptor_type"], "receptor_type"
        ).sum().tolist() == [
            3,
            3,
        ]
        assert result.provenance.n_cells == 6
    finally:
        result.close()
    # A blank label is a missing one.
    store.cells.insert("lane", np.asarray(["p", " ", "p", "q", "", "q", "p", "q"]))
    result = plotting.receptor_qc(store, ref, group_by="lane", show=False)
    try:
        assert result.tables["receptor_type"]["group"].drop_duplicates().tolist() == [
            "p",
            "q",
        ]
        assert result.provenance.n_cells == 6
    finally:
        result.close()


def test_receptor_qc_shows_fractions_of_all_cells_when_asked(receptors) -> None:
    store, ref = receptors
    result = plotting.receptor_qc(store, ref, normalize=True)
    assert result.axes["receptor_type"].get_xlabel() == "Fraction of cells"
    widths = [patch.get_width() for patch in result.axes["receptor_type"].patches]
    assert np.isclose(sum(widths), 1.0)
