"""Gene usage and junction lengths of hand-written contigs, counted by hand."""

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import re
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import zarr

import cytearc.datastore._operations.repertoire as repertoire_operations
from cytearc import plotting
from cytearc.datastore._operations.cell_groups import (
    GROUPING_BYTES_PER_CELL,
    GROUPING_BYTES_PER_COLUMN,
    GROUPING_BYTES_PER_LABEL_BYTE,
)
from cytearc.metadata.rows import array_row_selection_parts
from cytearc.repertoire import SLOTS, GeneUsage, Spectratype
from cytearc.storage.artifacts import artifact_group, inspect_artifact
from cytearc.storage.refs import ArtifactRef
from tests.clone_helpers import (
    CELLRANGER_HEADER,
    CLONAL_ROWS,
    airr_file,
    every_cell,
    import_truth_receptors,
    receptor_store,
    write_cells_store,
    write_clone_contigs,
    write_lines,
    write_truth_store,
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
        "group",
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
    assert by_sample.columns[:3].tolist() == ["group", "sample", "locus"]
    assert (by_sample["group"] == by_sample["sample"]).all()
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
        "cells": [2, 2, 4, 5],
        "fraction": [1.0, 1.0, 1.0, 1.0],
    }
    nucleotides = store.repertoire.spectratype(
        clonotypes, sequence="nt", group_by="sample", count="clonotypes"
    )
    table = nucleotides.table
    assert table.columns.tolist() == [
        "group",
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


def test_tables_follow_the_natural_order_of_the_groups(tmp_path) -> None:
    store, receptors, clonotypes = _sources(tmp_path)
    # "g10" sorts before "g2" as text, but after it as a label.
    store.cells.insert("batch", np.asarray(["g10", "g2"] * 4))
    lengths = store.repertoire.spectratype(receptors, group_by="batch").table
    assert list(dict.fromkeys(lengths["batch"])) == ["g2", "g10"]
    assert lengths["group"].tolist() == lengths["batch"].tolist()
    # Within a group, by locus and then by length.
    for _group, rows in lengths.groupby("group", sort=False):
        assert rows[["locus", "length"]].values.tolist() == sorted(
            rows[["locus", "length"]].values.tolist()
        )
    usage = store.repertoire.gene_usage(clonotypes, group_by="batch").table
    assert list(dict.fromkeys(usage["batch"])) == ["g2", "g10"]


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
        assert result.tables["pairs"].columns[:4].tolist() == [
            "group",
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


def _unsaved(ref: ArtifactRef) -> ArtifactRef:
    """A reference of the same kind to an artifact that no store holds."""
    return ArtifactRef(scope=ref.scope, kind=ref.kind, artifact_id="f" * 64)


def _counts(table, *columns: str) -> dict[tuple, int]:
    return {
        tuple(row[:-1]): row[-1]
        for row in table[[*columns, table.columns[-2]]].values.tolist()
    }


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
    # A label artifact of the first four cells leaves the others ungrouped.
    labels = store.snapshot_cluster_labels("sample", cell_selection=first)
    grouped = store.repertoire.spectratype(receptors, groups=labels).table
    assert grouped.columns[:3].tolist() == ["group", labels.kind, "locus"]
    assert _counts(grouped, "group", "locus", "length") == {
        ("s1", "TRA", 4): 2,
        ("s1", "TRB", 5): 2,
        ("s2", "TRA", 4): 1,
        ("s2", "TRB", 5): 2,
    }


def test_groups_of_two_columns_follow_the_natural_order_of_each(tmp_path) -> None:
    store, receptors, _clonotypes = _sources(tmp_path)
    store.cells.insert("batch", np.asarray(["b10", "b2"] * 4))
    usage = store.repertoire.gene_usage(receptors, group_by=["batch", "sample"])
    table = usage.table
    assert table.columns[:4].tolist() == ["group", "batch", "sample", "locus"]
    assert list(dict.fromkeys(table["group"])) == [
        "b2, s1",
        "b2, s2",
        "b10, s1",
        "b10, s2",
    ]
    assert usage.group_by == ("batch", "sample")
    # Each group's shares of a locus sum to one.
    totals = table.groupby(["group", "locus"])["fraction"].sum()
    np.testing.assert_allclose(totals.to_numpy(), 1.0)


def test_group_columns_may_not_share_a_name_with_the_result(tmp_path) -> None:
    store, receptors, clonotypes = _sources(tmp_path)
    for column in ("locus", "v_call", "length", "group"):
        store.cells.insert(column, SAMPLES)
    for measure, column in (
        (store.repertoire.gene_usage, "locus"),
        (store.repertoire.gene_usage, "v_call"),
        (store.repertoire.spectratype, "length"),
        (store.repertoire.spectratype, "group"),
    ):
        with pytest.raises(ValueError, match=f"'{column}', which is also a column"):
            measure(receptors, group_by=column)
    # Gene usage has no length column, nor junction lengths a gene column.
    assert len(store.repertoire.gene_usage(receptors, group_by="length").table)
    assert len(store.repertoire.spectratype(receptors, group_by="v_call").table)
    # Clonotypes as labels would name a key column after a count column.
    with pytest.raises(ValueError, match="key column 'clonotypes' is also a column"):
        store.repertoire.gene_usage(receptors, groups=clonotypes)


@pytest.mark.parametrize("measure", ["gene_usage", "spectratype"])
def test_missing_incomplete_and_other_sources_are_named(tmp_path, measure) -> None:
    store, receptors, clonotypes = _sources(tmp_path)
    runs = (
        getattr(store.repertoire, measure),
        lambda source: getattr(store.plots, measure)(source, show=False),
        *(
            (lambda source: store.plots.gene_pairing(source, show=False),)
            if measure == "gene_usage"
            else ()
        ),
    )
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


def _refusals(store, run) -> list[tuple[str, int]]:
    """Each refusal on the way from a budget of one byte to one that runs.

    The budget rises to each refusal's own estimate, so each later step's
    check shows once the earlier ones pass.
    """
    found: list[tuple[str, int]] = []
    budget, full = 1, store.memoryBytes
    try:
        for _attempt in range(100):
            store.memoryBytes = budget
            try:
                run()
                return found
            except MemoryError as error:
                matched = re.search(r"^(.*) holds about (\d+) bytes", str(error))
                assert matched is not None, str(error)
                estimate = int(matched.group(2))
                assert estimate > budget, str(error)
                found.append((matched.group(1), estimate))
                budget = estimate
    finally:
        store.memoryBytes = full
    raise AssertionError("the budget did not settle")


def _chunk(array) -> int:
    """The bytes of one chunk that reading an array holds, by the store's model."""
    return array_row_selection_parts(array)[0]


@pytest.mark.parametrize("source", ["receptors", "clonotypes"])
def test_each_step_is_checked_before_it_reads_or_allocates(tmp_path, source) -> None:
    store, receptors, clonotypes = _sources(tmp_path)
    group = artifact_group(store.zw, receptors)
    of_clonotypes = source == "clonotypes"
    # Reading the eight cells' slots: the stored int32 array whole and one
    # chunk of it, and for each cell its two chosen slots, whether each holds
    # a chain and whether either does; clonotypes also read each cell's code,
    # with its int64 copy and whether it has one.
    reading = _chunk(group["chain_slots"]) + 8 * 4 * 4 + 8 * (2 * 4 + 2 + 1)
    if of_clonotypes:
        codes = artifact_group(store.zw, clonotypes)["codes"]
        reading += _chunk(codes) + 8 * 4 + 8 * (8 + 1)
    # Grouping by the samples, labels of two characters stored in 8 bytes,
    # holds what the summaries' grouping holds for each cell and a stored
    # chunk of the column, beside the chosen slots, the cells with a chain
    # and the clonotypes' codes.
    held = 8 * 2 * 4 + 8 + (8 * 8 if of_clonotypes else 0)
    per_cell = GROUPING_BYTES_PER_CELL + GROUPING_BYTES_PER_COLUMN
    per_cell += GROUPING_BYTES_PER_LABEL_BYTE * 8
    grouping = held + 8 * per_cell + _chunk(store.cells._get_array("sample"))
    # Without a grouping, the 13 chains of VJ_1 and VDJ_1, with their locus
    # and V call, and the clonotype of each with clonotypes: beside each
    # cell's group and the one group's keys, a mask of the slots, and the 14
    # stored loci read with their int64 copy.
    held += 8 * 8 + repertoire_operations._KEY_BYTES_PER_GROUP - 8
    n_fields = 3 if of_clonotypes else 2
    per_chain = (
        repertoire_operations._CHAIN_BYTES
        + repertoire_operations._CHAIN_BYTES_PER_FIELD * n_fields
    )
    chains = held + 16 + _chunk(group["contig_locus"]) + 14 * (4 + 8)
    chains += 13 * per_chain
    ref = clonotypes if of_clonotypes else receptors
    for measure, what in (
        (store.repertoire.gene_usage, "genes"),
        (store.repertoire.spectratype, "junction lengths"),
    ):
        counting = f"Counting the {what} of the chains of 8 cells"
        assert _refusals(store, lambda measure=measure: measure(ref))[:2] == [
            (counting, reading),
            (counting, chains),
        ]
        assert _refusals(
            store, lambda measure=measure: measure(ref, group_by="sample")
        )[:2] == [(counting, reading), ("Grouping 8 cells by 1 columns", grouping)]


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
        receptors, sequence="aa", metric="hamming", cutoff=1
    )
    for source, count, noun in (
        (receptors, "cells", "cells"),
        (clonotypes, "clonotypes", "clonotypes"),
        (clusters, "clonotypes", "clonotype clusters"),
    ):
        usage = store.plots.gene_usage(source, count=count, show=False)
        lengths = store.plots.spectratype(source, count=count, show=False)
        try:
            assert usage.axes["TRB"].get_xlabel() == f"Fraction of {noun}"
            assert lengths.axes["TRB"].get_ylabel() == f"Fraction of {noun}"
        finally:
            usage.close()
            lengths.close()
    # Four loci stand two to a row, within the width of a page.
    figure = store.plots.gene_usage(receptors, show=False)
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
        store.plots.gene_usage(receptors, gene="d_call", show=False)
    store.cells.insert("nobody", np.zeros(8, dtype=bool))
    nobody = store.snapshot_cell_selection("nobody")
    for plot in (store.plots.gene_usage, store.plots.gene_pairing):
        with pytest.raises(ValueError, match="nothing to plot"):
            plot(receptors, cell_selection=nobody, show=False)
    with pytest.raises(ValueError, match="has a junction, so there is nothing"):
        store.plots.spectratype(receptors, cell_selection=nobody, show=False)


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
    usage = store.repertoire.gene_usage(
        receptors, genes=("v_call", "j_call"), level="allele", group_by="sample"
    ).table
    assert _counts(usage, "sample", "locus", "v_call", "j_call") == dict(by_cells)
    once = store.repertoire.gene_usage(
        clonotypes,
        genes=("v_call", "j_call"),
        level="allele",
        group_by="sample",
        count="clonotypes",
    ).table
    assert _counts(once, "sample", "locus", "v_call", "j_call") == {
        key: len(found) for key, found in clones_of.items()
    }
    # The fractions are each count's share of its sample's chains of the locus.
    totals = Counter()
    for (sample, locus, *_genes), cells in by_cells.items():
        totals[(sample, locus)] += cells
    for row in usage.itertuples():
        assert row.fraction == pytest.approx(
            row.cells / totals[(row.sample, row.locus)]
        )
    # At the gene level the alleles of the truth's calls fall away.
    genes_only = store.repertoire.gene_usage(receptors, group_by="sample").table
    assert not genes_only["v_call"].str.contains("*", regex=False).any()
    lengths = store.repertoire.spectratype(receptors, group_by="sample").table
    expected_lengths: Counter = Counter()
    for cell, row in enumerate(expected.itertuples()):
        for slot in ("VJ_1", "VDJ_1"):
            junction = getattr(row, f"{slot}_junction_aa")
            if not pd.isna(junction):
                locus = genes[junction][0]
                expected_lengths[(str(truth.samples[cell]), locus, len(junction))] += 1
    assert _counts(lengths, "sample", "locus", "length") == dict(expected_lengths)


def test_figures_of_clonotypes_need_the_store_to_name_them(tmp_path) -> None:
    store, _receptors, clonotypes = _sources(tmp_path)
    # A store that counts but cannot say what its clonotypes are called.
    bare = SimpleNamespace(
        repertoire=SimpleNamespace(
            gene_usage=store.repertoire.gene_usage,
            spectratype=store.repertoire.spectratype,
        )
    )
    for plot in (plotting.gene_usage, plotting.spectratype):
        with pytest.raises(TypeError, match="what figures call clonotypes"):
            plot(bare, clonotypes, count="clonotypes", show=False)
        # Cells need no name from the store.
        plot(bare, clonotypes, show=False).close()


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
    reader.plots.gene_pairing(receptors, show=False).close()


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
    result = store.plots.spectratype(receptors, group_by="own", show=False)
    try:
        legend = result.axes["TRB"].get_legend()
        assert legend.get_title().get_text() == "Group (80 of 100)"
        names = [text.get_text() for text in legend.get_texts()]
        assert names[:2] == ["g000", "g001"] and len(names) == 80
    finally:
        result.close()


def test_the_groups_keys_are_checked_before_their_frame_is_built(tmp_path) -> None:
    # 4,096 cells, each a group of its own by 24 key columns of two labels,
    # whose keys outweigh the grouping that names them.
    n_cells, n_columns = 4_096, 24
    ids = [f"c{i}" for i in range(n_cells)]
    folder = tmp_path / "keys"
    folder.mkdir()
    store = write_cells_store(folder / "store.zarr", ids)
    for column in range(n_columns):
        bits = (np.arange(n_cells) >> (column % 12)) & 1
        store.cells.insert(f"bit{column}", bits.astype(str).astype(object))
    contigs = write_clone_contigs(folder, "contigs.csv", ids, np.arange(n_cells))
    receptors = store.repertoire.import_receptors(
        contigs, cell_selection=every_cell(store)
    )
    columns = [f"bit{column}" for column in range(n_columns)]
    refusals = _refusals(
        store, lambda: store.repertoire.spectratype(receptors, group_by=columns)
    )
    last_grouping = max(
        at for at, (what, _bytes) in enumerate(refusals) if what.startswith("Grouping")
    )
    # Beside the two chosen int32 slots of each cell: each cell's group and,
    # for each of the 4,096 groups, its labels, name and row of the keys.
    keys = 4_096 * (
        repertoire_operations._KEY_BYTES_PER_GROUP
        + repertoire_operations._KEY_BYTES_PER_GROUP_COLUMN * n_columns
    )
    counting = "Counting the junction lengths of the chains of 4096 cells"
    held = n_cells * 2 * 4 + n_cells * 8 + keys
    assert refusals[last_grouping + 1] == (counting, held)
    # Then the mask of the chains of the grouped cells in the two slots:
    # whether each cell has a group, whether each slot holds a chain, and both.
    assert refusals[last_grouping + 2] == (counting, held + n_cells * (2 * 2 + 1))


def test_counting_is_checked_with_its_table_before_it_runs(tmp_path) -> None:
    store, _receptors, _clonotypes = _sources(tmp_path)
    keys = pd.DataFrame({"group": ["s1", "s2"], "sample": ["s1", "s2"]})

    def needs(n_chains: int, widths: list[int], once: bool) -> int:
        reads = repertoire_operations._Reads(
            limit=0, held=0, what="Counting", remedy=""
        )
        with pytest.raises(MemoryError) as refused:
            store._check_count_budget(reads, keys, n_chains, widths, once=once)
        found = re.search(r"holds about (\d+) bytes", str(refused.value))
        assert found is not None
        return int(found.group(1))

    # The table holds a row for each group, locus and value that the chains
    # hold, at most 2 * 3 * 5 = 30 for 1,000 chains, each with its group,
    # sample, locus, value, count and fraction: 120 + 6 * 16 bytes.
    assert needs(1_000, [3, 5], once=False) == 30 * (120 + 6 * 16)
    # At most one row per chain for 20 chains.
    assert needs(20, [3, 5], once=False) == 20 * (120 + 6 * 16)
    # Counting each clonotype, or each cell, once also holds each chain's row
    # of group, locus, value and clonotype or cell: 32 + 4 * 16 bytes.
    assert needs(1_000, [3, 5], once=True) == 30 * 216 + 1_000 * (32 + 4 * 16)


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
    assert usage[["v_call", "cells"]].values.tolist() == [
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
    assert lengths[["length", "cells"]].values.tolist() == [[4, 4]]


def test_grouped_gene_usage_names_the_group_of_a_locus_held_by_one(tmp_path) -> None:
    store, receptors, _clonotypes = _sources(tmp_path)
    # The B cells, cell5 and cell6, are all of group b, and the T cells of t.
    store.cells.insert("kind", np.asarray(["t"] * 5 + ["b", "b", "t"], dtype=object))
    result = store.plots.gene_usage(receptors, group_by="kind", show=False)
    try:
        for locus, group in (("IGH", "b"), ("IGK", "b"), ("TRA", "t"), ("TRB", "t")):
            ax = result.axes[locus]
            assert not ax.patches, locus
            assert [tick.get_text() for tick in ax.get_xticklabels()] == [group]
    finally:
        result.close()


def test_the_gene_usage_plot_checks_its_gene_before_counting(tmp_path) -> None:
    store, receptors, _clonotypes = _sources(tmp_path)
    for gene in (("v_call", "j_call"), "junction", 3):
        with pytest.raises(ValueError, match="gene must be one of v_call"):
            store.plots.gene_usage(receptors, gene=gene, show=False)


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
    result = store.plots.gene_pairing(receptors, show=False)
    try:
        ax = result.axes["TRA"]
        # TRAJ2 comes before TRAJ10, though TRAJ10 holds three chains.
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["TRAJ2", "TRAJ10"]
        assert [tick.get_text() for tick in ax.get_yticklabels()] == ["TRAV3", "TRAV12"]
    finally:
        result.close()


def test_two_rows_of_panels_make_the_figure_twice_as_tall(tmp_path) -> None:
    store, receptors, _clonotypes = _sources(tmp_path)
    usage = store.plots.gene_usage(receptors, show=False)
    pairing = store.plots.gene_pairing(receptors, top=4, show=False)
    lengths = store.plots.spectratype(receptors, show=False)
    try:
        # Four loci stand in two rows of two panels.
        for result, height in (
            (usage, 0.22 * 30 + 1.5),
            (pairing, 4.0),
            (lengths, 3.8),
        ):
            assert result.figure.get_size_inches()[1] == pytest.approx(2 * height)
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
        reads = repertoire_operations._Reads(
            limit=1_000 + needed - 1, held=1_000, what="Counting", remedy=""
        )
        with pytest.raises(MemoryError, match=f"holds about {1_000 + needed} bytes"):
            measure(reads, group, name, np.asarray([0, -1]))


def test_junctions_that_span_stored_chunks_are_measured_whole(
    tmp_path, monkeypatch
) -> None:
    # Chunks of 7 bytes split the junctions, and the characters of two and
    # four bytes, between them.
    monkeypatch.setattr(repertoire_operations, "_CONTIG_CHUNK_BYTES", 7)
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


def test_each_refusal_names_the_remedy_that_its_step_allows(tmp_path) -> None:
    store, receptors, _clonotypes = _sources(tmp_path)
    messages = []
    budget, full = 1, store.memoryBytes
    try:
        for _attempt in range(3):
            store.memoryBytes = budget
            with pytest.raises(MemoryError) as refused:
                store.repertoire.gene_usage(receptors)
            messages.append(str(refused.value))
            found = re.search(r"holds about (\d+) bytes", messages[-1])
            assert found is not None
            budget = int(found.group(1))
    finally:
        store.memoryBytes = full
    # Reading every cell's slots does not shrink with fewer cells; counting
    # the chains does.
    assert messages[0].endswith("Raise mem_budget.")
    assert messages[1].endswith(
        "Raise mem_budget, or count fewer chains, of a smaller cell_selection "
        "or of fewer slots."
    )


def test_every_stored_array_is_read_a_chunk_at_a_time(tmp_path, monkeypatch) -> None:
    # The checks count one decoded chunk of each array read, which zarr's
    # default concurrency would exceed for arrays of several chunks.
    store, _receptors, clonotypes = _sources(tmp_path)
    read = repertoire_operations._read
    limits = []

    def recorded(group, name):
        limits.append((name, zarr.config.get("async.concurrency")))
        return read(group, name)

    monkeypatch.setattr(repertoire_operations, "_read", recorded)
    store.repertoire.gene_usage(clonotypes, genes=("v_call", "j_call"))
    store.repertoire.spectratype(clonotypes, sequence="nt")
    names = {name for name, _limit in limits}
    assert {"chain_slots", "codes", "contig_locus", "contig_v_call"} <= names
    assert {limit for _name, limit in limits} == {1}
    assert zarr.config.get("async.concurrency") != 1
