import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex

from cytearc import plotting
from cytearc.datastore.datastore import DataStore
from tests.storage_helpers import insert_nullable_cell_column
from tests.test_repertoire_import import N_CELLS, _bcr, _store, _tcr


@pytest.fixture(autouse=True)
def _figures_are_closed():
    before = set(plt.get_fignums())
    yield
    leaked = set(plt.get_fignums()) - before
    for number in leaked:
        plt.close(number)
    assert not leaked, f"Figures left open: {sorted(leaked)}"


def _receptors(tmp_path: Path) -> tuple[DataStore, object]:
    store = _store(tmp_path)
    ref = store.repertoire.import_receptors(
        [_tcr(tmp_path), _bcr(tmp_path)], cell_selection=store.snapshot_cell_selection()
    )
    return store, ref


def test_receptor_qc_counts_cells_by_category(tmp_path) -> None:
    store, ref = _receptors(tmp_path)
    result = store.plots.receptor_qc(ref, show=False)
    try:
        assert set(result.axes) == {"receptor_type", "chain_pairing", "chains_per_cell"}
        receptor_type = result.tables["receptor_type"]["cells"].to_dict()
        assert receptor_type == {
            "TCR": 2,
            "BCR": 1,
            "ambiguous": 1,
            "multichain": 1,
            "no IR": 3,
        }
        assert result.tables["chain_pairing"]["cells"].sum() == N_CELLS
        chains = result.tables["chains_per_cell"]
        assert chains.index.tolist() == ["0", "1", "2", "3+"]
        assert chains["VJ"].tolist() == [5, 2, 1, 0]
        assert chains["VDJ"].tolist() == [3, 3, 1, 1]
        # One bar per category with cells.
        assert len(result.axes["receptor_type"].patches) == 5
        assert result.axes["receptor_type"].get_xlabel() == "Cells"
        assert result.provenance.n_cells == N_CELLS
        assert result.provenance.extras["grouping"] is None
    finally:
        result.close()


def test_receptor_qc_stacks_groups_of_a_cell_column(tmp_path) -> None:
    store, ref = _receptors(tmp_path)
    store.cells.insert("sample", np.asarray(["a", "a", "b", "b", "a", "b", "a", "b"]))
    result = plotting.receptor_qc(store, ref, group_by="sample", show=False)
    try:
        table = result.tables["receptor_type"]
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


def test_receptor_qc_groups_by_labels_of_fewer_cells(tmp_path) -> None:
    store, ref = _receptors(tmp_path)
    keep = np.zeros(N_CELLS, dtype=bool)
    keep[:4] = True
    store.cells.insert("first", keep)
    store.cells.insert("cluster", np.asarray(["x", "y", "x", "y", "x", "y", "x", "y"]))
    labels = store.snapshot_cluster_labels(
        "cluster", cell_selection=store.snapshot_cell_selection("first")
    )
    result = plotting.receptor_qc(store, ref, groups=labels, show=False)
    try:
        table = result.tables["chain_pairing"]
        assert table.sum().tolist() == [2, 2]
        assert result.provenance.n_cells == 4
        assert result.tables["chains_per_cell"].sum().tolist() == [4, 4]
        assert result.provenance.extras["grouping"] == "cluster_labels"
    finally:
        result.close()


def test_embedding_draws_cells_without_a_clonotype_as_missing(tmp_path) -> None:
    from cytearc.embeddings.imported import write_imported_embedding
    from cytearc.storage.artifacts import fingerprint_array

    store = _store(tmp_path)
    cells = store.snapshot_cell_selection()
    receptors = store.repertoire.import_receptors(
        [_tcr(tmp_path), _bcr(tmp_path)], cell_selection=cells
    )
    coordinates = np.column_stack([np.arange(N_CELLS), np.zeros(N_CELLS)]).astype(
        np.float32
    )
    layout = write_imported_embedding(
        store.zw,
        assay="RNA",
        dimreduc_key="umap",
        role="umap",
        coordinates=coordinates,
        source_digest=hashlib.sha256(b"layout").digest(),
        payload_fingerprints={"values": fingerprint_array(coordinates)},
        source_cell_ids=np.asarray(store.cells.fetch_all("ids")).astype(str),
        cell_selection=cells,
        block_rows=4,
    )
    clonotypes = store.repertoire.clonotypes(receptors)
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
        # Blank labels are missing: drawn in the missing colour, not as a category.
        assert "" not in scale.palette
        assert set(scale.palette) == set(labels[labels.astype(str) != ""])
        (points,) = result.axes[next(iter(result.axes))].collections
        drawn = [to_hex(color) for color in points.get_facecolors()]
        assert drawn.count(to_hex(scale.missing_color)) == without
    finally:
        result.close()


def test_receptor_qc_groups_by_integer_labels_of_fewer_cells(tmp_path) -> None:
    # Integer labels, as clusterings hold them, align as objects; the cells
    # that the labels lack get no group.
    store, ref = _receptors(tmp_path)
    keep = np.zeros(N_CELLS, dtype=bool)
    keep[:4] = True
    store.cells.insert("first", keep)
    store.cells.insert("cluster", np.arange(N_CELLS) % 2)
    labels = store.snapshot_cluster_labels(
        "cluster", cell_selection=store.snapshot_cell_selection("first")
    )
    result = plotting.receptor_qc(store, ref, groups=labels, show=False)
    try:
        assert result.tables["chain_pairing"].sum().tolist() == [2, 2]
        assert result.provenance.n_cells == 4
    finally:
        result.close()


def test_receptor_qc_draws_on_caller_axes(tmp_path) -> None:
    store, ref = _receptors(tmp_path)
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
        ({"group_by": "sample", "groups": "x"}, ValueError, "not both"),
        ({"group_by": 3}, TypeError, "group_by must be the name"),
        ({"groups": "clusters"}, TypeError, "groups must be an ArtifactRef"),
        ({"normalize": "yes"}, TypeError, "normalize must be a boolean"),
    ],
)
def test_receptor_qc_checks_its_arguments(tmp_path, kwargs, error, message) -> None:
    store, ref = _receptors(tmp_path)
    store.cells.insert("sample", np.asarray(list("aabbaabb")))
    with pytest.raises(error, match=message):
        plotting.receptor_qc(store, ref, show=False, **kwargs)


def test_receptor_qc_needs_a_store_with_receptors(tmp_path) -> None:
    _store_unused, ref = _receptors(tmp_path)
    with pytest.raises(TypeError, match="does not provide immune receptors"):
        plotting.receptor_qc(object(), ref, show=False)


def test_receptor_qc_leaves_out_cells_with_a_missing_group(tmp_path) -> None:
    store, ref = _receptors(tmp_path)
    missing = np.zeros(N_CELLS, dtype=bool)
    missing[[0, 7]] = True
    insert_nullable_cell_column(store, "batch", np.asarray(list("pqpqpqpq")), missing)
    result = plotting.receptor_qc(store, ref, group_by="batch", show=False)
    try:
        assert result.tables["receptor_type"].sum().tolist() == [3, 3]
        assert result.provenance.n_cells == 6
    finally:
        result.close()
    # A blank label is a missing one.
    store.cells.insert("lane", np.asarray(["p", " ", "p", "q", "", "q", "p", "q"]))
    result = plotting.receptor_qc(store, ref, group_by="lane", show=False)
    try:
        assert result.tables["receptor_type"].columns.tolist() == ["p", "q"]
        assert result.provenance.n_cells == 6
    finally:
        result.close()


def test_receptor_qc_shows_fractions_of_all_cells_when_asked(tmp_path) -> None:
    store, ref = _receptors(tmp_path)
    result = plotting.receptor_qc(store, ref, normalize=True)
    assert result.axes["receptor_type"].get_xlabel() == "Fraction of cells"
    widths = [patch.get_width() for patch in result.axes["receptor_type"].patches]
    assert np.isclose(sum(widths), 1.0)
