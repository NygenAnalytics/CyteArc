import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cytearc import plotting
from cytearc.repertoire import GeneUsage, Spectratype
from tests.test_repertoire_clonotypes_datastore import CLONAL_ROWS, _receptors
from tests.test_repertoire_import import _airr, _store, _write

# CLONAL_ROWS: cell0 and cell1 pair TRAV1 with TRBV1, cell2 pairs TRAV2 with
# TRBV1, cell3 has TRBV1 alone, cell4 has TRAV1 and a second alpha chain
# TRAV3 beside TRBV1, and cell5 and cell6 pair IGHV1 with IGKV1.
SAMPLES = np.asarray(["s1", "s2", "s1", "s2", "s1", "s1", "s2", "s1"])


@pytest.fixture(autouse=True)
def _figures_are_closed():
    before = set(plt.get_fignums())
    yield
    leaked = set(plt.get_fignums()) - before
    for number in leaked:
        plt.close(number)
    assert not leaked, f"Figures left open: {sorted(leaked)}"


def _sources(tmp_path):
    store, receptors = _receptors(tmp_path, CLONAL_ROWS)
    store.cells.insert("sample", SAMPLES)
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
        "locus": ["IGH", "IGK", "TRA", "TRA", "TRB"],
        "v_call": ["IGHV1", "IGKV1", "TRAV1", "TRAV2", "TRBV1"],
        "cells": [2, 2, 3, 1, 5],
        "fraction": [1.0, 1.0, 0.75, 0.25, 1.0],
    }
    second = store.repertoire.gene_usage(receptors, slots="VJ_2")
    assert second.table[["locus", "v_call", "cells"]].values.tolist() == [
        ["TRA", "TRAV3", 1]
    ]
    pairs = store.repertoire.gene_usage(
        receptors, genes=["v_call", "j_call"], slots=["VJ_1", "VJ_2"]
    )
    assert pairs.table.columns.tolist() == [
        "locus",
        "v_call",
        "j_call",
        "cells",
        "fraction",
    ]
    tra = pairs.table[pairs.table["locus"] == "TRA"]
    assert tra[["v_call", "j_call", "cells"]].values.tolist() == [
        ["TRAV1", "TRAJ1", 3],
        ["TRAV2", "TRAJ1", 1],
        ["TRAV3", "TRAJ2", 1],
    ]
    assert store.artifacts.list(scope="datastore") == saved


def test_gene_usage_by_group_and_by_clonotype(tmp_path) -> None:
    store, receptors, clonotypes = _sources(tmp_path)
    by_sample = store.repertoire.gene_usage(receptors, group_by="sample").table
    assert by_sample.columns[:2].tolist() == ["sample", "locus"]
    s1 = by_sample[(by_sample["sample"] == "s1") & (by_sample["locus"] == "TRA")]
    assert s1[["v_call", "cells", "fraction"]].values.tolist() == [
        ["TRAV1", 2, 2 / 3],
        ["TRAV2", 1, 1 / 3],
    ]
    # Each clonotype counts once per gene: clonotype_1 for TRAV1 and
    # clonotype_4 for TRAV2; clonotype_3 has no alpha chain.
    once = store.repertoire.gene_usage(clonotypes, count="clonotypes").table
    tra = once[once["locus"] == "TRA"]
    assert tra[["v_call", "clonotypes", "fraction"]].values.tolist() == [
        ["TRAV1", 1, 0.5],
        ["TRAV2", 1, 0.5],
    ]
    assert once[once["locus"] == "TRB"]["clonotypes"].tolist() == [3]
    cells = store.repertoire.gene_usage(clonotypes).table
    assert cells[cells["locus"] == "TRB"]["cells"].tolist() == [5]
    t_cells = store.select_cells(receptors, include=["TCR"])
    selected = store.repertoire.gene_usage(receptors, cell_selection=t_cells)
    assert selected.cell_selection == t_cells
    assert set(selected.table["locus"]) == {"TRA", "TRB"}


def test_gene_calls_keep_their_alleles_or_families(tmp_path) -> None:
    store = _store(tmp_path)
    receptors = store.repertoire.import_receptors(
        _airr(tmp_path), cell_selection=store.snapshot_cell_selection()
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
        "locus": ["IGH", "IGK", "TRA", "TRB"],
        "length": [4, 4, 4, 5],
        "cells": [2, 2, 4, 5],
        "fraction": [1.0, 1.0, 1.0, 1.0],
    }
    nucleotides = store.repertoire.spectratype(
        clonotypes, sequence="nt", group_by="sample", count="clonotypes"
    )
    table = nucleotides.table
    assert table.columns.tolist() == [
        "sample",
        "locus",
        "length",
        "clonotypes",
        "fraction",
    ]
    tra = table[(table["sample"] == "s1") & (table["locus"] == "TRA")]
    # s1 holds clonotype_1 (cell0 and cell4) and clonotype_4 (cell2), both
    # with 12 alpha nucleotides.
    assert tra[["length", "clonotypes"]].values.tolist() == [[12, 2]]


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
    expansion = store.repertoire.clonal_expansion(clonotypes)
    with pytest.raises(ValueError, match="immune_receptors or clonotypes artifact"):
        store.repertoire.gene_usage(expansion)
    with pytest.raises(TypeError, match="source must be an ArtifactRef"):
        store.repertoire.spectratype("receptors")
    with pytest.raises(ValueError, match="sequence must be one of"):
        store.repertoire.spectratype(receptors, sequence="codons")
    bare = _write(
        tmp_path,
        "bare.tsv",
        "cell_id\tsequence_id\tlocus\tjunction_aa\tproductive",
        ["cell0\ts1\tTRB\tCASS\tT"],
    )
    sparse = store.repertoire.import_receptors(
        bare, cell_selection=store.snapshot_cell_selection()
    )
    with pytest.raises(ValueError, match="The receptors store no v_call"):
        store.repertoire.gene_usage(sparse)
    with pytest.raises(ValueError, match="The receptors store no junction"):
        store.repertoire.spectratype(sparse, sequence="nt")


def test_gene_usage_plot_draws_one_panel_per_locus(tmp_path) -> None:
    store, receptors, _clonotypes = _sources(tmp_path)
    result = store.plots.gene_usage(receptors, show=False)
    try:
        assert list(result.axes) == ["IGH", "IGK", "TRA", "TRB"]
        # Without groups each locus draws bars.
        assert len(result.axes["TRA"].patches) == 2
        assert result.tables["usage"]["cells"].sum() == 13
        # Cells, each counted once however many of its chains count.
        usage = store.repertoire.gene_usage(receptors, group_by="sample")
        assert result.provenance.n_cells == usage.n_cells == 7
        assert usage.table["cells"].sum() == 13
    finally:
        result.close()
    grouped = plotting.gene_usage(
        store, receptors, group_by="sample", top=1, show_legend=False, show=False
    )
    try:
        ax = grouped.axes["TRA"]
        assert [tick.get_text() for tick in ax.get_yticklabels()] == ["TRAV1"]
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["s1", "s2"]
        assert len(grouped.figure.axes) == 4
    finally:
        grouped.close()
    shown = plotting.gene_usage(store, receptors, gene="j_call", group_by="sample")
    assert shown.provenance.extras["gene"] == "j_call"
    with pytest.raises(ValueError, match="top must be a positive integer"):
        plotting.gene_usage(store, receptors, top=0, show=False)
    with pytest.raises(TypeError, match="does not provide gene_usage"):
        plotting.gene_usage(object(), receptors, show=False)


def test_gene_pairing_plot_draws_v_against_j(tmp_path) -> None:
    store, receptors, _clonotypes = _sources(tmp_path)
    result = store.plots.gene_pairing(receptors, slots=["VJ_1", "VJ_2"], show=False)
    try:
        ax = result.axes["TRA"]
        assert [tick.get_text() for tick in ax.get_yticklabels()] == [
            "TRAV1",
            "TRAV2",
            "TRAV3",
        ]
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["TRAJ1", "TRAJ2"]
        assert result.tables["pairs"].columns[:3].tolist() == [
            "locus",
            "v_call",
            "j_call",
        ]
    finally:
        result.close()
    shown = plotting.gene_pairing(store, receptors, top=1, show_legend=False)
    assert shown.provenance.extras["top"] == 1
    # The V and J genes with the most chains; ties go to the earlier name.
    ax = shown.axes["TRA"]
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["TRAV1"]
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["TRAJ1"]
    with pytest.raises(ValueError, match="top must be a positive integer"):
        plotting.gene_pairing(store, receptors, top=True, show=False)


def test_spectratype_plot_draws_lines_per_group(tmp_path) -> None:
    store, receptors, _clonotypes = _sources(tmp_path)
    result = store.plots.spectratype(receptors, group_by="sample", show=False)
    try:
        ax = result.axes["TRB"]
        assert len(ax.get_lines()) == 2
        legend = result.axes["TRB"].get_legend()
        assert [text.get_text() for text in legend.get_texts()] == ["s1", "s2"]
        assert ax.get_xlabel() == "Junction length (residues)"
    finally:
        result.close()
    single = plotting.spectratype(store, receptors, sequence="nt", show=False)
    try:
        assert len(single.axes["TRA"].patches) == 1
        assert single.axes["TRA"].get_xlabel() == "Junction length (nucleotides)"
        assert single.axes["TRB"].get_legend() is None
    finally:
        single.close()
    shown = plotting.spectratype(store, receptors)
    assert shown.provenance.extras["sequence"] == "aa"
    with pytest.raises(TypeError, match="does not provide spectratype"):
        plotting.spectratype(object(), receptors, show=False)
