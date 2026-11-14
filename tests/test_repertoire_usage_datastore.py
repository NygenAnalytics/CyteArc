"""Gene usage and junction lengths of hand-written contigs, counted by hand."""

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import inspect
import re
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import zarr

import cytearc.datastore._operations.record_tables as record_tables
import cytearc.datastore._operations.repertoire as repertoire_operations
from cytearc import plotting
from cytearc.repertoire import SLOTS, GeneUsage, Spectratype
from cytearc.storage.artifacts import artifact_group, inspect_artifact
from cytearc.storage.refs import ArtifactRef
from tests.clone_helpers import (
    CELLRANGER_HEADER,
    CLONAL_ROWS,
    clonal_receptors,
    airr_file,
    every_cell,
    import_truth_receptors,
    receptor_store,
    write_cells_store,
    write_clone_contigs,
    write_lines,
    write_truth_store,
)
from cytearc.utils.memory_budget import MemoryBudgetError, MemoryLedger
from tests.memory_helpers import (
    named_budget,
)
from tests.clone_truth import clone_truth, receptor_truth

pytestmark = pytest.mark.usefixtures("figures_closed")

# CLONAL_ROWS: cell0 and cell1 pair TRAV1 with TRBV1, cell2 pairs TRAV2 with
# TRBV1, cell3 has TRBV1 alone, cell4 has TRAV1 and a second alpha chain
# TRAV3 beside TRBV1, and cell5 and cell6 pair IGHV1 with IGKV1.
SAMPLES = np.asarray(["s1", "s2", "s1", "s2", "s1", "s1", "s2", "s1"])


@pytest.fixture(scope="module")
def clonal_template(tmp_path_factory) -> Path:
    """A store of the eight cells with the clonal rows imported once."""
    folder = tmp_path_factory.mktemp("usage_template")
    store = receptor_store(folder)
    path = write_lines(folder, "clonal.csv", CELLRANGER_HEADER, CLONAL_ROWS)
    receptors = store.repertoire.import_receptors(
        path, cell_selection=store.snapshot_cell_selection()
    )
    store.cells.insert("sample", SAMPLES)
    store.repertoire.clonotypes(receptors)
    return folder / "store.zarr"


@pytest.fixture(autouse=True)
def _clonal_store_copy(clonal_template, tmp_path) -> None:
    """Give each test a copy of the template, which receptor_store opens."""
    shutil.copytree(clonal_template, tmp_path / "store.zarr")


def _sources(tmp_path):
    """The copied store, its receptors and their clonotypes.

    Importing the clonal rows and defining their clonotypes again reuses the
    copy's saved artifacts.
    """
    store = receptor_store(tmp_path)
    path = write_lines(tmp_path, "clonal.csv", CELLRANGER_HEADER, CLONAL_ROWS)
    receptors = store.repertoire.import_receptors(
        path, cell_selection=store.snapshot_cell_selection()
    )
    return store, receptors, store.repertoire.clonotypes(receptors)


def test_gene_usage_counts_each_cells_primary_chains(tmp_path) -> None:
    store, receptors, _clonotypes = _sources(tmp_path)
    saved = store.artifacts.list(scope="datastore")
    usage = store.repertoire.gene_usage(receptors)
    assert isinstance(usage, GeneUsage)
    assert usage.genes == ("v_call",) and usage.level == "gene"
    assert usage.slots == ("VJ_1", "VDJ_1") and usage.count == "cells"
    assert usage.source == receptors and usage.groups is None
    assert usage.table.to_dict("list") == {
        "group": ["all cells"] * 5,
        "locus": ["IGH", "IGK", "TRA", "TRA", "TRB"],
        "v_call": ["IGHV1", "IGKV1", "TRAV1", "TRAV2", "TRBV1"],
        "n_cells": [2, 2, 3, 1, 5],
        "fraction": [1.0, 1.0, 0.75, 0.25, 1.0],
    }
    # cell0 to cell6 hold a chain in the primary slots, cell7 none; one
    # group holds them, and each of their 13 chains has a V call.
    assert usage.diagnostics == {
        "cellsWithoutClone": 1,
        "cellsWithoutGroup": 0,
        "groups": 1,
        "chainsWithoutValue": 0,
    }
    # The rows give no D call, so each of those chains lacks the gene counted.
    without_d = store.repertoire.gene_usage(receptors, genes="d_call")
    assert (without_d.n_cells, len(without_d.table)) == (0, 0)
    assert without_d.diagnostics["chainsWithoutValue"] == 13
    second = store.repertoire.gene_usage(receptors, slots="VJ_2")
    assert second.table[["locus", "v_call", "n_cells"]].values.tolist() == [
        ["TRA", "TRAV3", 1]
    ]
    pairs = store.repertoire.gene_usage(
        receptors, genes=["v_call", "j_call"], slots=["VJ_1", "VJ_2"]
    )
    assert pairs.table.columns.tolist() == [
        "group",
        "locus",
        "v_call",
        "j_call",
        "n_cells",
        "fraction",
    ]
    tra = pairs.table[pairs.table["locus"] == "TRA"]
    assert tra[["v_call", "j_call", "n_cells"]].values.tolist() == [
        ["TRAV1", "TRAJ1", 3],
        ["TRAV2", "TRAJ1", 1],
        ["TRAV3", "TRAJ2", 1],
    ]
    assert store.artifacts.list(scope="datastore") == saved


def test_gene_usage_by_group_and_by_clonotype(tmp_path) -> None:
    store, receptors, clonotypes = _sources(tmp_path)
    usage = store.repertoire.gene_usage(receptors, group_by="sample")
    assert usage.table.columns[:3].tolist() == ["group", "locus", "v_call"]
    assert usage.group_keys.index.name == "group"
    assert usage.group_keys.columns.tolist() == ["sample"]
    by_sample = usage.table.join(usage.group_keys, on="group")
    assert (by_sample["group"] == by_sample["sample"]).all()
    s1 = by_sample[(by_sample["sample"] == "s1") & (by_sample["locus"] == "TRA")]
    assert s1[["v_call", "n_cells", "fraction"]].values.tolist() == [
        ["TRAV1", 2, 2 / 3],
        ["TRAV2", 1, 1 / 3],
    ]
    # Each clonotype counts once per gene: clonotype_1 for TRAV1 and
    # clonotype_4 for TRAV2; clonotype_3 has no alpha chain.
    once = store.repertoire.gene_usage(clonotypes, count="clonotypes").table
    tra = once[once["locus"] == "TRA"]
    assert tra[["v_call", "n_clonotypes", "fraction"]].values.tolist() == [
        ["TRAV1", 1, 0.5],
        ["TRAV2", 1, 0.5],
    ]
    assert once[once["locus"] == "TRB"]["n_clonotypes"].tolist() == [3]
    cells = store.repertoire.gene_usage(clonotypes).table
    assert cells[cells["locus"] == "TRB"]["n_cells"].tolist() == [5]
    t_cells = store.select_cells(receptors, include=["TCR"])
    selected = store.repertoire.gene_usage(receptors, cell_selection=t_cells)
    assert selected.cell_selection == t_cells
    assert set(selected.table["locus"]) == {"TRA", "TRB"}


def test_gene_calls_keep_their_alleles_or_families(tmp_path) -> None:
    store = receptor_store(tmp_path)
    receptors = store.repertoire.import_receptors(
        airr_file(tmp_path), cell_selection=store.snapshot_cell_selection()
    )
    alleles = store.repertoire.gene_usage(receptors, level="allele").table
    assert alleles["v_call"].tolist() == ["IGHV3-30*01", "IGLV1-40*01"]
    genes = store.repertoire.gene_usage(receptors).table
    assert genes["v_call"].tolist() == ["IGHV3-30", "IGLV1-40"]
    families = store.repertoire.gene_usage(receptors, genes="j_call", level="family")
    assert families.table["j_call"].tolist() == ["IGHJ4", "IGLJ2"]
    # Missing D calls leave nothing to count.
    assert store.repertoire.gene_usage(receptors, genes="d_call").table.empty


def test_spectratype_counts_junction_lengths(tmp_path) -> None:
    store, receptors, clonotypes = _sources(tmp_path)
    result = store.repertoire.spectratype(receptors)
    assert isinstance(result, Spectratype)
    assert result.sequence == "aa" and result.slots == ("VJ_1", "VDJ_1")
    assert result.table.to_dict("list") == {
        "group": ["all cells"] * 4,
        "locus": ["IGH", "IGK", "TRA", "TRB"],
        "length": [4, 4, 4, 5],
        "n_cells": [2, 2, 4, 5],
        "fraction": [1.0, 1.0, 1.0, 1.0],
    }
    nucleotides = store.repertoire.spectratype(
        clonotypes, sequence="nt", group_by="sample", count="clonotypes"
    )
    assert nucleotides.table.columns.tolist() == [
        "group",
        "locus",
        "length",
        "n_clonotypes",
        "fraction",
    ]
    assert nucleotides.group_keys.columns.tolist() == ["sample"]
    table = nucleotides.table.join(nucleotides.group_keys, on="group")
    tra = table[(table["sample"] == "s1") & (table["locus"] == "TRA")]
    # s1 holds clonotype_1 (cell0 and cell4) and clonotype_4 (cell2), both
    # with 12 alpha nucleotides.
    assert tra[["length", "n_clonotypes"]].values.tolist() == [[12, 2]]


def test_tables_follow_the_natural_order_of_the_groups(tmp_path) -> None:
    store, receptors, clonotypes = _sources(tmp_path)
    # "g10" sorts before "g2" as text, but after it as a label.
    store.cells.insert("batch", np.asarray(["g10", "g2"] * 4))
    spectratype = store.repertoire.spectratype(receptors, group_by="batch")
    assert spectratype.group_keys.index.tolist() == ["g2", "g10"]
    lengths = spectratype.table.join(spectratype.group_keys, on="group")
    assert list(dict.fromkeys(lengths["batch"])) == ["g2", "g10"]
    assert lengths["group"].tolist() == lengths["batch"].tolist()
    # Within a group, by locus and then by length.
    for _group, rows in lengths.groupby("group", sort=False):
        assert rows[["locus", "length"]].values.tolist() == sorted(
            rows[["locus", "length"]].values.tolist()
        )
    usage = store.repertoire.gene_usage(clonotypes, group_by="batch").table
    assert list(dict.fromkeys(usage["group"])) == ["g2", "g10"]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"genes": 3}, TypeError, "genes must be a gene field"),
        ({"genes": []}, TypeError, "genes must be a gene field"),
        ({"genes": "junction"}, ValueError, "genes must hold distinct fields"),
        (
            {"genes": ["v_call", "v_call"]},
            ValueError,
            "genes must hold distinct fields",
        ),
        ({"level": "segment"}, ValueError, "level must be one of"),
        ({"slots": 3}, TypeError, "slots must be a slot name"),
        ({"slots": []}, TypeError, "slots must be a slot name"),
        ({"slots": "VJ_3"}, ValueError, "slots must hold distinct slots"),
        ({"slots": ["VJ_1", "VJ_1"]}, ValueError, "slots must hold distinct slots"),
        ({"count": "chains"}, ValueError, "count must be"),
        ({"count": "clonotypes"}, ValueError, "needs a clonotypes artifact"),
    ],
)
def test_gene_usage_arguments_are_checked(tmp_path, kwargs, error, message) -> None:
    store, receptors, _clonotypes = _sources(tmp_path)
    with pytest.raises(error, match=message):
        store.repertoire.gene_usage(receptors, **kwargs)


def test_usage_refuses_other_sources_and_missing_fields(tmp_path) -> None:
    store, receptors, clonotypes = _sources(tmp_path)
    expansion = store.clones.expansion(clonotypes)
    with pytest.raises(ValueError, match="immune_receptors or clonotypes artifact"):
        store.repertoire.gene_usage(expansion)
    with pytest.raises(TypeError, match="source must be an ArtifactRef"):
        store.repertoire.spectratype("receptors")
    with pytest.raises(ValueError, match="sequence must be one of"):
        store.repertoire.spectratype(receptors, sequence="codons")
    bare = write_lines(
        tmp_path,
        "bare.tsv",
        "cell_id\tsequence_id\tlocus\tjunction_aa\tproductive",
        ["cell0\ts1\tTRB\tCASS\tT"],
    )
    sparse = store.repertoire.import_receptors(
        bare, cell_selection=store.snapshot_cell_selection()
    )
    # The message names what is counted rather than genes, which this call
    # leaves at its default.
    with pytest.raises(
        ValueError,
        match="^Counting gene usage reads each chain's v_call, which these "
        "receptors do not store",
    ):
        store.repertoire.gene_usage(sparse)
    with pytest.raises(
        ValueError,
        match="^sequence='nt' reads each chain's junction, which these receptors",
    ):
        store.repertoire.spectratype(sparse, sequence="nt")


def test_gene_usage_plot_draws_one_panel_per_locus(tmp_path) -> None:
    store, receptors, _clonotypes = _sources(tmp_path)
    result = store.plots.gene_usage(store.repertoire.gene_usage(receptors), show=False)
    try:
        assert list(result.axes) == ["IGH", "IGK", "TRA", "TRB"]
        # Without groups each locus draws bars.
        assert len(result.axes["TRA"].patches) == 2
        assert result.tables["usage"]["n_cells"].sum() == 13
        # Cells, each counted once however many of its chains count.
        usage = store.repertoire.gene_usage(receptors, group_by="sample")
        assert result.provenance.n_cells == usage.n_cells == 7
        assert usage.table["n_cells"].sum() == 13
    finally:
        result.close()
    grouped = plotting.gene_usage(
        store,
        store.repertoire.gene_usage(receptors, group_by="sample"),
        max_genes=1,
        show_legend=False,
        show=False,
    )
    try:
        ax = grouped.axes["TRA"]
        assert [tick.get_text() for tick in ax.get_yticklabels()] == ["TRAV1"]
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["s1", "s2"]
        assert len(grouped.figure.axes) == 4
    finally:
        grouped.close()
    j_genes = store.repertoire.gene_usage(receptors, genes="j_call", group_by="sample")
    shown = plotting.gene_usage(store, j_genes)
    assert shown.provenance.extras["gene"] == "j_call"
    assert shown.tables["usage"] is j_genes.table
    with pytest.raises(ValueError, match="^max_genes must be at least 1$"):
        plotting.gene_usage(store, j_genes, max_genes=0, show=False)
    for cap in (True, 2.0, "3"):
        with pytest.raises(TypeError, match="^max_genes must be an integer$"):
            plotting.gene_usage(store, j_genes, max_genes=cap, show=False)
    with pytest.raises(TypeError, match="^usage must be the GeneUsage"):
        plotting.gene_usage(store, receptors, show=False)
    # Cells need no name from the store, which the record makes unread.
    plotting.gene_usage(object(), j_genes, show=False).close()


def test_gene_usage_of_a_label_artifact_alone_draws_a_heatmap(tmp_path) -> None:
    store, receptors, _clonotypes = _sources(tmp_path)
    labels = store.snapshot_cluster_labels(
        "sample", cell_selection=store.snapshot_cell_selection()
    )
    usage = store.repertoire.gene_usage(receptors, groups=labels)
    assert not usage.group_by and usage.groups == labels
    result = store.plots.gene_usage(usage, show=False)
    try:
        # Groups of a label artifact are groups as those of a column are.
        ax = result.axes["TRA"]
        assert not ax.patches and len(ax.images) == 1
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["s1", "s2"]
        # The labels are titled by the column they copied.
        assert ax.get_xlabel() == "sample"
    finally:
        result.close()
    # A named artifact groups the summary as the bare one does, and titles
    # the figure; the spectratype takes the name too.
    named = plotting.NamedCellArtifact("Sample", labels)
    assert store.repertoire.gene_usage(receptors, groups=named).groups == labels
    renamed = store.plots.gene_usage(usage, groups=named, show=False)
    spectratype = store.repertoire.spectratype(receptors, groups=named)
    lengths = store.plots.spectratype(spectratype, groups=named, show=False)
    try:
        assert renamed.axes["TRA"].get_xlabel() == "Sample"
        assert lengths.provenance.extras["grouping"] == "Sample"
    finally:
        renamed.close()
        lengths.close()


def test_gene_pairing_plot_draws_v_against_j(tmp_path) -> None:
    store, receptors, _clonotypes = _sources(tmp_path)
    result = store.plots.gene_pairing(
        store.repertoire.gene_usage(
            receptors, genes=("v_call", "j_call"), slots=["VJ_1", "VJ_2"]
        ),
        show=False,
    )
    try:
        ax = result.axes["TRA"]
        assert [tick.get_text() for tick in ax.get_yticklabels()] == [
            "TRAV1",
            "TRAV2",
            "TRAV3",
        ]
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["TRAJ1", "TRAJ2"]
        assert result.tables["pairs"].columns[:4].tolist() == [
            "group",
            "locus",
            "v_call",
            "j_call",
        ]
    finally:
        result.close()
    pairs = store.repertoire.gene_usage(receptors, genes=("v_call", "j_call"))
    shown = plotting.gene_pairing(store, pairs, max_genes=1, show_legend=False)
    assert shown.provenance.extras["max_genes"] == 1
    # The V and J genes with the most chains; ties go to the earlier name.
    ax = shown.axes["TRA"]
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["TRAV1"]
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["TRAJ1"]
    with pytest.raises(TypeError, match="^max_genes must be an integer$"):
        plotting.gene_pairing(store, pairs, max_genes=True, show=False)
    with pytest.raises(TypeError, match="^usage must be the GeneUsage"):
        plotting.gene_pairing(store, receptors, show=False)
    # The pairs of the V and J genes, in either order, and no other fields.
    flipped = store.repertoire.gene_usage(receptors, genes=("j_call", "v_call"))
    found = plotting.gene_pairing(store, flipped, max_genes=1, show=False)
    try:
        assert [tick.get_text() for tick in found.axes["TRA"].get_yticklabels()] == [
            "TRAV1"
        ]
    finally:
        found.close()
    for genes in ("v_call", ("v_call", "d_call"), ("v_call", "j_call", "c_call")):
        with pytest.raises(ValueError, match="gene_pairing draws the pairs of v_call"):
            plotting.gene_pairing(
                store, store.repertoire.gene_usage(receptors, genes=genes), show=False
            )
    # Pairs counted within groups would be averaged over them, so they are
    # refused rather than drawn.
    grouped = store.repertoire.gene_usage(
        receptors, genes=("v_call", "j_call"), group_by="sample"
    )
    with pytest.raises(ValueError, match="pairs within groups of cells"):
        plotting.gene_pairing(store, grouped, show=False)


def test_the_pairing_plot_documents_the_error_of_an_unknown_theme(tmp_path) -> None:
    store, receptors, _clonotypes = _sources(tmp_path)
    pairs = store.repertoire.gene_usage(receptors, genes=("v_call", "j_call"))
    # The plot reads nothing from the store, so the theme is the one name it
    # looks up, and its section lists the class that lookup raises.
    with pytest.raises(KeyError, match="Unknown theme 'nope'"):
        plotting.gene_pairing(object(), pairs, theme="nope", show=False)
    doc = inspect.getdoc(inspect.unwrap(plotting.gene_pairing)) or ""
    raises = doc.split("Raises:\n", 1)[1]
    documented = set(re.findall(r"^\s+(\w+Error):", raises, re.M))
    assert {"TypeError", "ValueError", "KeyError"} <= documented


def test_spectratype_plot_draws_lines_per_group(tmp_path) -> None:
    store, receptors, _clonotypes = _sources(tmp_path)
    result = store.plots.spectratype(
        store.repertoire.spectratype(receptors, group_by="sample"), show=False
    )
    try:
        ax = result.axes["TRB"]
        assert len(ax.get_lines()) == 2
        # One legend of the groups, titled by the grouping, stands beside the
        # last panel of the first row of three.
        legend = result.axes["TRA"].get_legend()
        assert legend.get_title().get_text() == "sample"
        assert [text.get_text() for text in legend.get_texts()] == ["s1", "s2"]
        assert ax.get_legend() is None
        assert ax.get_xlabel() == "Junction length\n(residues)"
    finally:
        result.close()
    single = plotting.spectratype(
        store, store.repertoire.spectratype(receptors, sequence="nt"), show=False
    )
    try:
        assert len(single.axes["TRA"].patches) == 1
        assert single.axes["TRA"].get_xlabel() == "Junction length\n(nucleotides)"
        assert single.axes["TRB"].get_legend() is None
    finally:
        single.close()
    shown = plotting.spectratype(store, store.repertoire.spectratype(receptors))
    assert shown.provenance.extras["sequence"] == "aa"
    with pytest.raises(TypeError, match="^spectratype must be the Spectratype"):
        plotting.spectratype(store, receptors, show=False)


def _unsaved(ref: ArtifactRef) -> ArtifactRef:
    """A reference of the same kind to an artifact that no store holds."""
    return ArtifactRef(scope=ref.scope, kind=ref.kind, artifact_id="f" * 64)


def _counts(table, *columns: str) -> dict[tuple, int]:
    return {
        tuple(row[:-1]): row[-1]
        for row in table[[*columns, table.columns[-2]]].values.tolist()
    }


def _labelled(result) -> pd.DataFrame:
    """A record's table with its groups' labels after each row's group name."""
    table = result.table.copy()
    labels = result.group_keys.loc[table["group"]]
    for at, column in enumerate(labels.columns, start=1):
        table.insert(at, column, labels[column].to_numpy())
    return table


def test_a_clonotypes_source_counts_only_cells_with_a_clonotype(tmp_path) -> None:
    store, receptors, _clonotypes = _sources(tmp_path)
    # Under receptor_arms="VJ", cell3, with a beta chain alone, has no
    # clonotype, so its TRBV1 chain is not counted: four TRB cells, not five.
    alpha = store.repertoire.clonotypes(receptors, receptor_arms="VJ")
    usage = store.repertoire.gene_usage(alpha).table
    assert _counts(usage, "locus", "v_call") == {
        ("IGH", "IGHV1"): 2,
        ("IGK", "IGKV1"): 2,
        ("TRA", "TRAV1"): 3,
        ("TRA", "TRAV2"): 1,
        ("TRB", "TRBV1"): 4,
    }
    assert store.repertoire.gene_usage(alpha).n_cells == 6
    lengths = store.repertoire.spectratype(alpha).table
    assert _counts(lengths, "locus", "length")[("TRB", 5)] == 4


def test_cells_outside_a_selection_or_without_a_label_are_left_out(tmp_path) -> None:
    store, receptors, clonotypes = _sources(tmp_path)
    store.cells.insert("first_four", np.arange(8) < 4)
    first = store.snapshot_cell_selection("first_four")
    usage = store.repertoire.gene_usage(clonotypes, cell_selection=first)
    # cell0 to cell3: TRAV1 twice and TRAV2 once, TRBV1 four times.
    assert _counts(usage.table, "locus", "v_call") == {
        ("TRA", "TRAV1"): 2,
        ("TRA", "TRAV2"): 1,
        ("TRB", "TRBV1"): 4,
    }
    assert usage.n_cells == 4 and usage.cell_selection == first
    # Each of the four cells has a clonotype, and every chain a V call.
    assert usage.diagnostics == {
        "cellsWithoutClone": 0,
        "cellsWithoutGroup": 0,
        "groups": 1,
        "chainsWithoutValue": 0,
    }
    # A label artifact of the first four cells leaves the others ungrouped.
    labels = store.snapshot_cluster_labels("sample", cell_selection=first)
    spectratype = store.repertoire.spectratype(receptors, groups=labels)
    grouped = spectratype.table
    assert grouped.columns[:3].tolist() == ["group", "locus", "length"]
    assert spectratype.group_keys.to_dict("index") == {
        "s1": {labels.kind: "s1"},
        "s2": {labels.kind: "s2"},
    }
    assert _counts(grouped, "group", "locus", "length") == {
        ("s1", "TRA", 4): 2,
        ("s1", "TRB", 5): 2,
        ("s2", "TRA", 4): 1,
        ("s2", "TRB", 5): 2,
    }
    # cell4 to cell6 hold chains but no label, and cell7 no chain.
    assert spectratype.diagnostics == {
        "cellsWithoutClone": 1,
        "cellsWithoutGroup": 3,
        "groups": 2,
        "chainsWithoutValue": 0,
    }


def test_groups_of_two_columns_follow_the_natural_order_of_each(tmp_path) -> None:
    store, receptors, _clonotypes = _sources(tmp_path)
    store.cells.insert("batch", np.asarray(["b10", "b2"] * 4))
    usage = store.repertoire.gene_usage(receptors, group_by=["batch", "sample"])
    table = usage.table
    assert table.columns[:3].tolist() == ["group", "locus", "v_call"]
    assert list(dict.fromkeys(table["group"])) == [
        "b2, s1",
        "b2, s2",
        "b10, s1",
        "b10, s2",
    ]
    assert usage.group_keys.values.tolist() == [
        ["b2", "s1"],
        ["b2", "s2"],
        ["b10", "s1"],
        ["b10", "s2"],
    ]
    assert usage.group_by == ("batch", "sample")
    # Each group's shares of a locus sum to one.
    totals = table.groupby(["group", "locus"])["fraction"].sum()
    np.testing.assert_allclose(totals.to_numpy(), 1.0)


def test_group_columns_may_take_the_names_of_the_results_columns(tmp_path) -> None:
    store, receptors, clonotypes = _sources(tmp_path)
    measures = (store.repertoire.gene_usage, store.repertoire.spectratype)
    neutral = [measure(receptors, group_by="sample") for measure in measures]
    # Columns named like the results' own columns hold the samples' labels
    # and count what the samples do; their labels stay in the keys.
    for column in ("group", "locus", "v_call", "length", "n_cells", "fraction"):
        store.cells.insert(column, SAMPLES, overwrite=True)
        for measure, expected in zip(measures, neutral, strict=True):
            found = measure(receptors, group_by=column)
            pd.testing.assert_frame_equal(found.table, expected.table)
            assert found.group_keys.to_dict("index") == {
                sample: {column: sample} for sample in ("s1", "s2")
            }
            joined = found.table.join(found.group_keys, on="group", rsuffix="_key")
            label = f"{column}_key" if column in found.table.columns else column
            assert joined[label].tolist() == joined["group"].tolist()
    # Clonotypes as labels have the key column "clonotypes".
    by_clonotype = store.repertoire.gene_usage(receptors, groups=clonotypes)
    assert by_clonotype.group_keys.columns.tolist() == ["clonotypes"]


@pytest.mark.parametrize("measure", ["gene_usage", "spectratype"])
def test_missing_incomplete_and_other_sources_are_named(tmp_path, measure) -> None:
    store, receptors, clonotypes = _sources(tmp_path)
    runs = (getattr(store.repertoire, measure),)
    # The plots take the summary's record, so they read no source.
    with pytest.raises(TypeError, match="must be the"):
        getattr(store.plots, measure)(receptors, show=False)
    expansion = store.clones.expansion(clonotypes)
    for run in runs:
        with pytest.raises(ValueError, match="immune_receptors or clonotypes artifact"):
            run(expansion)
        for source in (receptors, clonotypes):
            with pytest.raises(KeyError, match="Artifact does not exist"):
                run(_unsaved(source))
    # An import that stopped before it finished leaves incomplete receptors,
    # which their clonotypes read too.
    for ref, sources in (
        (receptors, (receptors, clonotypes)),
        (clonotypes, (clonotypes,)),
    ):
        attributes = artifact_group(store.zw, ref).attrs
        attributes["complete"] = False
        try:
            for run in runs:
                for source in sources:
                    with pytest.raises(RuntimeError, match="Artifact is incomplete"):
                        run(source)
        finally:
            attributes["complete"] = True
    # Clonotypes whose receptors the store no longer holds.
    path = inspect_artifact(store.zw, receptors).path
    folder = tmp_path / "store.zarr" / path
    folder.rename(folder.with_name(folder.name + "_moved"))
    for run in runs:
        with pytest.raises(KeyError, match="Artifact does not exist"):
            run(clonotypes)


def test_every_slot_is_read_for_its_chains(tmp_path) -> None:
    store, receptors, _clonotypes = _sources(tmp_path)
    usage = store.repertoire.gene_usage(receptors, slots=SLOTS)
    # cell4's second alpha chain, TRAV3, joins the primary chains.
    assert _counts(usage.table, "locus", "v_call")[("TRA", "TRAV3")] == 1
    assert usage.slots == SLOTS and usage.n_cells == 7


def test_junctions_beyond_ascii_are_measured_in_characters(tmp_path) -> None:
    store = receptor_store(tmp_path)
    rows = [
        "cell0,true,cell0_contig_1,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,"
        "CAŚSF,TGTGCCAGCAGCTTT,200,8,ct1,ct1_1",
        "cell1,true,cell1_contig_1,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,"
        "C\U0001f600F,TGTTTT,200,8,ct2,ct2_1",
        "cell2,true,cell2_contig_1,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,"
        "CASSLF,TGTGCCAGCAGCCTGTTT,200,8,ct3,ct3_1",
    ]
    path = write_lines(tmp_path, "wide.csv", CELLRANGER_HEADER, rows)
    receptors = store.repertoire.import_receptors(
        path, cell_selection=store.snapshot_cell_selection()
    )
    lengths = store.repertoire.spectratype(receptors).table
    assert _counts(lengths, "length") == {(3,): 1, (5,): 1, (6,): 1}
    nucleotides = store.repertoire.spectratype(receptors, sequence="nt").table
    assert _counts(nucleotides, "length") == {(6,): 1, (15,): 1, (18,): 1}


def test_figures_name_what_they_count(tmp_path) -> None:
    store, receptors, clonotypes = _sources(tmp_path)
    clusters = store.repertoire.clonotypes(
        receptors, sequence="aa", metric="hamming", max_distance=1
    )
    for source, count, noun in (
        (receptors, "cells", "cells"),
        (clonotypes, "clonotypes", "clonotypes"),
        (clusters, "clonotypes", "clonotype clusters"),
    ):
        usage = store.plots.gene_usage(
            store.repertoire.gene_usage(source, count=count), show=False
        )
        lengths = store.plots.spectratype(
            store.repertoire.spectratype(source, count=count), show=False
        )
        try:
            assert usage.axes["TRB"].get_xlabel() == f"Fraction of {noun}"
            assert lengths.axes["TRB"].get_ylabel() == f"Fraction of {noun}"
        finally:
            usage.close()
            lengths.close()
    # Four loci stand three to a row, within the width of a page.
    figure = store.plots.gene_usage(store.repertoire.gene_usage(receptors), show=False)
    try:
        rows = {round(ax.get_position().y0, 3) for ax in figure.axes.values()}
        assert len(rows) == 2
        assert figure.figure.get_size_inches()[0] <= 7.5
    finally:
        figure.close()


def test_figures_with_nothing_to_draw_are_refused(tmp_path) -> None:
    store, receptors, _clonotypes = _sources(tmp_path)
    # No chain of the clonal rows has a D call.
    assert store.repertoire.gene_usage(receptors, genes="d_call").table.empty
    with pytest.raises(ValueError, match="has a d_call, so there is nothing to plot"):
        store.plots.gene_usage(
            store.repertoire.gene_usage(receptors, genes="d_call"), show=False
        )
    store.cells.insert("nobody", np.zeros(8, dtype=bool))
    nobody = store.snapshot_cell_selection("nobody")
    for plot, genes in (
        (store.plots.gene_usage, "v_call"),
        (store.plots.gene_pairing, ("v_call", "j_call")),
    ):
        with pytest.raises(ValueError, match="nothing to plot"):
            plot(
                store.repertoire.gene_usage(
                    receptors, genes=genes, cell_selection=nobody
                ),
                show=False,
            )
    with pytest.raises(ValueError, match="has a junction, so there is nothing"):
        store.plots.spectratype(
            store.repertoire.spectratype(receptors, cell_selection=nobody), show=False
        )


def _slot_genes(receptors, fields: tuple[str, ...]) -> dict[str, tuple]:
    """Each of the truth's chains, by its junction, as its locus and gene calls.

    The truth draws every chain's junction apart, so a slot's junction names
    the contig that fills it.
    """
    contigs = receptors.contigs.drop_duplicates("junction_aa")
    return {
        row[0]: tuple(row[1:])
        for row in contigs[["junction_aa", "locus", *fields]].values.tolist()
    }


def test_gene_usage_recovers_the_truths_genes_in_each_sample(tmp_path) -> None:
    truth = clone_truth()
    receptors_truth = receptor_truth(truth)
    folder = tmp_path / "truth"
    folder.mkdir()
    store = write_truth_store(folder / "store.zarr", truth)
    receptors = import_truth_receptors(store, receptors_truth, folder)
    clonotypes = store.repertoire.clonotypes(receptors, receptor_arms="any")
    genes = _slot_genes(receptors_truth, ("v_call", "j_call"))
    expected = receptors_truth.expected.loc[truth.cell_ids]
    by_cells: Counter = Counter()
    clones_of: dict[tuple, set] = {}
    for cell, row in enumerate(expected.itertuples()):
        for slot in ("VJ_1", "VDJ_1"):
            junction = getattr(row, f"{slot}_junction_aa")
            if pd.isna(junction):
                continue
            key = (str(truth.samples[cell]), *genes[junction])
            by_cells[key] += 1
            # Within its receptor type, each of the truth's clones is one
            # clonotype.
            clone = (int(truth.clones[cell]), row.receptor_type)
            clones_of.setdefault(key, set()).add(clone)
    usage = _labelled(
        store.repertoire.gene_usage(
            receptors, genes=("v_call", "j_call"), level="allele", group_by="sample"
        )
    )
    assert _counts(usage, "sample", "locus", "v_call", "j_call") == dict(by_cells)
    once = _labelled(
        store.repertoire.gene_usage(
            clonotypes,
            genes=("v_call", "j_call"),
            level="allele",
            group_by="sample",
            count="clonotypes",
        )
    )
    assert _counts(once, "sample", "locus", "v_call", "j_call") == {
        key: len(found) for key, found in clones_of.items()
    }
    # The fractions are each count's share of its sample's chains of the locus.
    totals = Counter()
    for (sample, locus, *_genes), cells in by_cells.items():
        totals[(sample, locus)] += cells
    for row in usage.itertuples():
        assert row.fraction == pytest.approx(
            row.n_cells / totals[(row.sample, row.locus)]
        )
    # At the gene level the alleles of the truth's calls fall away.
    genes_only = store.repertoire.gene_usage(receptors, group_by="sample").table
    assert not genes_only["v_call"].str.contains("*", regex=False).any()
    lengths = _labelled(store.repertoire.spectratype(receptors, group_by="sample"))
    expected_lengths: Counter = Counter()
    for cell, row in enumerate(expected.itertuples()):
        for slot in ("VJ_1", "VDJ_1"):
            junction = getattr(row, f"{slot}_junction_aa")
            if not pd.isna(junction):
                locus = genes[junction][0]
                expected_lengths[(str(truth.samples[cell]), locus, len(junction))] += 1
    assert _counts(lengths, "sample", "locus", "length") == dict(expected_lengths)
    # The cells without a chain in the primary slots are left out; every
    # chain of the truth has its genes and junction.
    without_chain = sum(
        all(pd.isna(getattr(row, f"{slot}_junction_aa")) for slot in ("VJ_1", "VDJ_1"))
        for row in expected.itertuples()
    )
    for record in (
        store.repertoire.gene_usage(
            receptors, genes=("v_call", "j_call"), group_by="sample"
        ),
        store.repertoire.spectratype(receptors, group_by="sample"),
    ):
        assert record.diagnostics == {
            "cellsWithoutClone": without_chain,
            "cellsWithoutGroup": 0,
            "groups": len(set(truth.samples.tolist())),
            "chainsWithoutValue": 0,
        }


def test_figures_of_clonotypes_need_the_store_to_name_them(tmp_path) -> None:
    store, _receptors, clonotypes = _sources(tmp_path)
    # A store that cannot say what its clonotypes are called.
    bare = SimpleNamespace()
    for plot, measure in (
        (plotting.gene_usage, store.repertoire.gene_usage),
        (plotting.spectratype, store.repertoire.spectratype),
    ):
        with pytest.raises(TypeError, match="what figures call clones"):
            plot(bare, measure(clonotypes, count="clonotypes"), show=False)
        # Cells need no name from the store.
        plot(bare, measure(clonotypes), show=False).close()


def test_a_read_only_store_counts_as_a_writable_one(tmp_path) -> None:
    store, receptors, clonotypes = _sources(tmp_path)
    expected = {
        "usage": store.repertoire.gene_usage(clonotypes, group_by="sample").table,
        "lengths": store.repertoire.spectratype(receptors, sequence="nt").table,
    }
    # Nothing is saved, so a store opened read-only gives the same tables.
    reader = receptor_store(tmp_path, zarr_mode="r")
    found = {
        "usage": reader.repertoire.gene_usage(clonotypes, group_by="sample").table,
        "lengths": reader.repertoire.spectratype(receptors, sequence="nt").table,
    }
    for name, table in expected.items():
        pd.testing.assert_frame_equal(found[name], table)
    reader.plots.gene_pairing(
        reader.repertoire.gene_usage(receptors, genes=("v_call", "j_call")), show=False
    ).close()


def test_the_lengths_legend_names_at_most_80_groups(tmp_path) -> None:
    ids = [f"cell{i}" for i in range(100)]
    folder = tmp_path / "many"
    folder.mkdir()
    store = write_cells_store(folder / "store.zarr", ids)
    contigs = write_clone_contigs(folder, "contigs.csv", ids, np.arange(100))
    receptors = store.repertoire.import_receptors(
        contigs, cell_selection=every_cell(store)
    )
    store.cells.insert("own", np.asarray([f"g{i:03d}" for i in range(100)]))
    result = store.plots.spectratype(
        store.repertoire.spectratype(receptors, group_by="own"), show=False
    )
    try:
        legend = result.axes["TRA"].get_legend()
        assert legend.get_title().get_text() == "own (80 of 100)"
        names = [text.get_text() for text in legend.get_texts()]
        assert names[:2] == ["g000", "g001"] and len(names) == 80
    finally:
        result.close()


def test_counting_is_checked_with_its_table_before_it_runs(tmp_path) -> None:
    store, _receptors, _clonotypes = _sources(tmp_path)
    keys = pd.DataFrame(
        {"sample": ["s1", "s2"]}, index=pd.Index(["s1", "s2"], name="group")
    )

    def needs(n_chains: int, widths: list[int], once: bool) -> int:
        with pytest.raises(MemoryBudgetError) as refused:
            store._check_count_budget(
                MemoryLedger(0, "Counting"), keys, n_chains, widths, once=once
            )
        return refused.value.needed_bytes

    # The table holds a row for each group, locus and value that the chains
    # hold, at most 2 * 3 * 5 = 30 for 1,000 chains, each with its group's
    # name, locus, value, count and fraction, 120 + 5 * 16 bytes; the sample's
    # label stays in the keys.
    assert needs(1_000, [3, 5], once=False) == 30 * (120 + 5 * 16)
    # At most one row per chain for 20 chains.
    assert needs(20, [3, 5], once=False) == 20 * (120 + 5 * 16)
    # Counting each clonotype, or each cell, once also holds each chain's row
    # of group, locus, value and clonotype or cell: 32 + 4 * 16 bytes.
    assert needs(1_000, [3, 5], once=True) == 30 * 200 + 1_000 * (32 + 4 * 16)
    # More key columns add nothing per row.
    wide = pd.DataFrame(
        {"sample": ["s1", "s2"], "batch": ["b1", "b1"], "site": ["x", "y"]},
        index=keys.index,
    )
    keys = wide
    assert needs(1_000, [3, 5], once=False) == 30 * (120 + 5 * 16)


def test_two_slots_of_one_arm_count_a_cell_once_per_value(tmp_path) -> None:
    # cell1's second alpha chain, with fewer UMIs, has the V gene and the
    # junction length of its first.
    rows = [
        *CLONAL_ROWS,
        "cell1,true,cell1_contig_3,true,500,TRA,TRAV1,,TRAJ2,TRAC,true,true,CAVZ,"
        "TGTGCTGTGTAT,20,2,ct1,ct1_3",
    ]
    store = receptor_store(tmp_path)
    path = write_lines(tmp_path, "second_alpha.csv", CELLRANGER_HEADER, rows)
    receptors = store.repertoire.import_receptors(
        path, cell_selection=store.snapshot_cell_selection()
    )
    alpha = ["VJ_1", "VJ_2"]

    def tra(table):
        return table[table["locus"] == "TRA"]

    usage = tra(store.repertoire.gene_usage(receptors, slots=alpha).table)
    # cell0, cell1 and cell4 carry TRAV1, cell2 TRAV2 and cell4 TRAV3.
    assert usage[["v_call", "n_cells"]].values.tolist() == [
        ["TRAV1", 3],
        ["TRAV2", 1],
        ["TRAV3", 1],
    ]
    assert usage["fraction"].tolist() == [3 / 5, 1 / 5, 1 / 5]
    # By pairs, cell1's TRAV1 chains are two, with TRAJ1 and TRAJ2.
    pairs = store.repertoire.gene_usage(
        receptors, genes=("v_call", "j_call"), slots=alpha
    )
    assert _counts(tra(pairs.table), "v_call", "j_call") == {
        ("TRAV1", "TRAJ1"): 3,
        ("TRAV1", "TRAJ2"): 1,
        ("TRAV2", "TRAJ1"): 1,
        ("TRAV3", "TRAJ2"): 1,
    }
    # Every alpha chain has 4 residues, so each of the four T cells with one
    # counts once, cell1 and cell4 with two alpha chains among them.
    lengths = tra(store.repertoire.spectratype(receptors, slots=alpha).table)
    assert lengths[["length", "n_cells"]].values.tolist() == [[4, 4]]


def test_grouped_gene_usage_names_the_group_of_a_locus_held_by_one(tmp_path) -> None:
    store, receptors, _clonotypes = _sources(tmp_path)
    # The B cells, cell5 and cell6, are all of group b, and the T cells of t.
    store.cells.insert("kind", np.asarray(["t"] * 5 + ["b", "b", "t"], dtype=object))
    result = store.plots.gene_usage(
        store.repertoire.gene_usage(receptors, group_by="kind"), show=False
    )
    try:
        for locus, group in (("IGH", "b"), ("IGK", "b"), ("TRA", "t"), ("TRB", "t")):
            ax = result.axes[locus]
            assert not ax.patches, locus
            assert [tick.get_text() for tick in ax.get_xticklabels()] == [group]
    finally:
        result.close()


def test_the_gene_usage_plot_draws_the_usage_of_one_gene_field(tmp_path) -> None:
    store, receptors, _clonotypes = _sources(tmp_path)
    for genes in (("v_call", "j_call"), ("d_call", "c_call")):
        pairs = store.repertoire.gene_usage(receptors, genes=genes)
        with pytest.raises(
            ValueError, match=f"^usage counts the combinations of {genes[0]}"
        ):
            store.plots.gene_usage(pairs, show=False)
    # A record of one field draws it, whichever field it is.
    calls = store.repertoire.gene_usage(receptors, genes=["c_call"])
    result = store.plots.gene_usage(calls, show=False)
    try:
        assert result.provenance.extras["gene"] == "c_call"
        assert result.tables["usage"].columns[2] == "c_call"
    finally:
        result.close()


def test_pairing_puts_j_genes_in_natural_order_and_v_genes_by_count(tmp_path) -> None:
    store = receptor_store(tmp_path)
    rows = [
        f"cell{cell},true,cell{cell}_contig_1,true,500,TRA,{v},,{j},TRAC,true,true,"
        f"CAV{letter},TGTGCTGTG{codon},90,9,ct{cell},ct{cell}_1"
        for cell, (v, j, letter, codon) in enumerate(
            [
                ("TRAV3", "TRAJ10", "A", "GCC"),
                ("TRAV3", "TRAJ10", "C", "TGT"),
                ("TRAV3", "TRAJ10", "D", "GAT"),
                ("TRAV12", "TRAJ2", "E", "GAA"),
            ]
        )
    ]
    path = write_lines(tmp_path, "pairs.csv", CELLRANGER_HEADER, rows)
    receptors = store.repertoire.import_receptors(
        path, cell_selection=store.snapshot_cell_selection()
    )
    result = store.plots.gene_pairing(
        store.repertoire.gene_usage(receptors, genes=("v_call", "j_call")), show=False
    )
    try:
        ax = result.axes["TRA"]
        # TRAJ2 comes before TRAJ10, though TRAJ10 holds three chains.
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["TRAJ2", "TRAJ10"]
        assert [tick.get_text() for tick in ax.get_yticklabels()] == ["TRAV3", "TRAV12"]
    finally:
        result.close()


def test_two_rows_of_panels_make_the_figure_twice_as_tall(tmp_path) -> None:
    store, receptors, _clonotypes = _sources(tmp_path)
    usage = store.plots.gene_usage(store.repertoire.gene_usage(receptors), show=False)
    pairing = store.plots.gene_pairing(
        store.repertoire.gene_usage(receptors, genes=("v_call", "j_call")),
        max_genes=4,
        show=False,
    )
    lengths = store.plots.spectratype(
        store.repertoire.spectratype(receptors), show=False
    )
    try:
        # Four loci stand in two rows, of three panels of bars or lines and of
        # two heatmaps, each row as tall as a panel of the two or three genes
        # drawn, at least 2.4 inches.
        for result, columns in ((usage, 3), (pairing, 2), (lengths, 3)):
            rows = [round(ax.get_position().y0, 3) for ax in result.axes.values()]
            assert len(set(rows)) == 2
            assert rows.count(max(rows)) == columns
            assert tuple(result.figure.get_size_inches()) == pytest.approx((7.5, 4.8))
    finally:
        for result in (usage, pairing, lengths):
            result.close()


def test_the_ranks_of_stored_values_are_checked_before_they_are_held(tmp_path) -> None:
    store, receptors, _clonotypes = _sources(tmp_path)
    group = artifact_group(store.zw, receptors)
    for name, measure in (
        ("v_call", store._value_names),
        ("junction_aa", store._value_measures),
    ):
        # Each stored value's presence and rank, before any value is read.
        n_values = group[f"contig_{name}_offsets"].shape[0] - 1
        needed = repertoire_operations._RANK_BYTES_PER_VALUE * n_values
        ledger = MemoryLedger(1_000 + needed - 1, "Counting")
        ledger.hold("what counting holds", 1_000)
        with pytest.raises(
            MemoryBudgetError,
            match=f"holds about {1_000 + needed} bytes for ranking the {name}",
        ):
            measure(ledger, group, name, np.asarray([0, -1]))


def test_junctions_that_span_stored_chunks_are_measured_whole(
    tmp_path, monkeypatch
) -> None:
    # Chunks of 7 bytes split the junctions, and the characters of two and
    # four bytes, between them.
    monkeypatch.setattr(record_tables, "_RECORD_CHUNK_BYTES", 7)
    junctions = ["CAŚSF", "C\U0001f600\U0001f600F", "CASSLGQAYEQYF", "Cé"]
    rows = [
        f"cell{cell},true,cell{cell}_contig_1,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,"
        f"{junction},TGTGCC,100,9,ct{cell},ct{cell}_1"
        for cell, junction in enumerate(junctions)
    ]
    store = receptor_store(tmp_path)
    path = write_lines(tmp_path, "spanning.csv", CELLRANGER_HEADER, rows)
    receptors = store.repertoire.import_receptors(
        path, cell_selection=store.snapshot_cell_selection()
    )
    packed = artifact_group(store.zw, receptors)["contig_junction_aa_values"]
    assert packed.chunks == (7,) and packed.shape[0] > 3 * 7
    lengths = store.repertoire.spectratype(receptors).table
    assert _counts(lengths, "length") == {(len(junction),): 1 for junction in junctions}
    # In the order of their text, CAŚSF and Cé lie between the junctions of
    # cell1 and cell2, so leaving them out leaves a chunk that no junction
    # measured touches.
    store.cells.insert("ends", np.isin(np.arange(8), [1, 2]))
    ends = store.repertoire.spectratype(
        receptors, cell_selection=store.snapshot_cell_selection("ends")
    ).table
    assert _counts(ends, "length") == {(4,): 1, (13,): 1}


def test_each_refusal_names_the_remedy_that_its_step_allows(
    tmp_path, monkeypatch
) -> None:
    store, receptors, _clonotypes = _sources(tmp_path)
    # Chains of 10 MB each make counting the 13 chains the costliest step.
    monkeypatch.setattr(repertoire_operations, "_CHAIN_BYTES", 10_000_000)

    def refusal(budget: int) -> str:
        monkeypatch.setattr(store, "memoryBytes", budget)
        with pytest.raises(MemoryError) as refused:
            store.repertoire.gene_usage(receptors)
        return str(refused.value)

    # Reading every cell's slots does not shrink with fewer cells, so 1 kB
    # asks for a larger budget alone; counting the chains does, so 100 MB
    # asks for smaller counts too.
    slots = refusal(1_000)
    assert "for each cell's chain slots" in slots
    assert slots.endswith(
        f"Raise mem_budget to at least {named_budget(MemoryError(slots))} bytes."
    )
    counting = refusal(100_000_000)
    assert "13 chains counted" in counting
    assert counting.endswith(
        ", or count fewer chains, of a smaller cell_selection or of fewer slots."
    )


def test_every_stored_array_is_read_through_the_memory_ledger(
    tmp_path, monkeypatch
) -> None:
    # Each whole read takes the concurrency its room allows, whose decoded
    # chunks its charge counts, and leaves Zarr's setting as it was.
    store, _receptors, clonotypes = _sources(tmp_path)
    read = repertoire_operations.read_whole
    names = []

    def recorded(ledger, group, name, *, item):
        names.append(name)
        return read(ledger, group, name, item=item)

    monkeypatch.setattr(repertoire_operations, "read_whole", recorded)
    before = zarr.config.get("async.concurrency")
    store.repertoire.gene_usage(clonotypes, genes=("v_call", "j_call"))
    store.repertoire.spectratype(clonotypes, sequence="nt")
    assert {"chain_slots", "codes", "contig_locus", "contig_v_call"} <= set(names)
    assert zarr.config.get("async.concurrency") == before


def test_chains_without_a_junction_are_counted_apart(tmp_path) -> None:
    # cell3's one chain, TRBV1, carries its junction's residues but not its
    # nucleotides, so the spectratype of nucleotides leaves it out and counts
    # it, while that of residues and gene usage count it.
    rows = [
        row.replace(",CASSA,TGTGCCAGCAGCGCC,30,", ",CASSA,,30,")
        if row.startswith("cell3,")
        else row
        for row in CLONAL_ROWS
    ]
    assert rows != CLONAL_ROWS
    folder = tmp_path / "without_junction"
    folder.mkdir()
    store, receptors = clonal_receptors(folder, rows)
    nucleotides = store.repertoire.spectratype(receptors, sequence="nt")
    assert nucleotides.diagnostics == {
        "cellsWithoutClone": 1,
        "cellsWithoutGroup": 0,
        "groups": 1,
        "chainsWithoutValue": 1,
    }
    # The other four TRB chains are counted, of 15 nucleotides each.
    assert _counts(nucleotides.table, "locus", "length")[("TRB", 15)] == 4
    residues = store.repertoire.spectratype(receptors)
    assert residues.diagnostics["chainsWithoutValue"] == 0
    assert _counts(residues.table, "locus", "length")[("TRB", 5)] == 5
    usage = store.repertoire.gene_usage(receptors)
    assert usage.diagnostics["chainsWithoutValue"] == 0
