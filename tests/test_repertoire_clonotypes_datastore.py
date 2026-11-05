"""Clonotypes of hand-written contigs, read back as clonotypes and as clones.

The clone summaries of ``ds.clones`` read clonotypes as clones and measure
each receptor type apart, so their expected values here are counted by hand
within the T cells and the B cells.
"""

import shutil
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import zarr

import cytearc.datastore._operations.repertoire as repertoire_operations
from cytearc import plotting
from cytearc.datastore._operations.record_tables import stored_text
from cytearc.datastore.datastore import DataStore
from cytearc.metadata.rows import (
    text_read_bytes,
)
from cytearc.repertoire import SLOTS, Clonotypes
from cytearc.repertoire.clonotypes import clonotype_nouns
from cytearc.storage.artifacts import artifact_group
from cytearc.storage.refs import ArtifactRef
from cytearc.utils.logging import logger
from cytearc.utils.memory_budget import MemoryBudgetError, MemoryLedger
from tests.clone_helpers import (
    CELLRANGER_HEADER,
    CLONAL_ROWS,
    N_RECEPTOR_CELLS,
    clonal_receptors,
    ids_read_bytes,
    import_truth_receptors,
    loaded_partition,
    receptor_store,
    write_lines,
    write_truth_store,
)
from tests.clone_truth import B_CELL_LIGHT_LOCI, clone_truth, receptor_truth
from tests.storage_helpers import (
    insert_nullable_cell_column,
    open_store,
    rechunk_column,
    write_count_store,
)

pytestmark = pytest.mark.usefixtures("figures_closed")

N_CELLS = N_RECEPTOR_CELLS
# The fields of each slot of a clonotype's representative chains.
FIELDS = ("junction_aa", "junction", "v_call", "j_call")


@pytest.fixture(scope="module")
def clonal_template(tmp_path_factory) -> Path:
    """A store of the eight cells with the clonal rows imported once."""
    folder = tmp_path_factory.mktemp("clonal_template")
    store = receptor_store(folder)
    path = write_lines(folder, "clonal.csv", CELLRANGER_HEADER, CLONAL_ROWS)
    store.repertoire.import_receptors(
        path, cell_selection=store.snapshot_cell_selection()
    )
    return folder / "store.zarr"


@pytest.fixture(autouse=True)
def _clonal_store_copy(clonal_template, tmp_path) -> None:
    """Give each test a copy of the template, which receptor_store opens.

    Importing the clonal rows again reuses the saved receptors of the copy.
    """
    shutil.copytree(clonal_template, tmp_path / "store.zarr")


def _members(table: Clonotypes) -> dict[str, list[str]]:
    cells = table.cells.dropna(subset=["clonotype"])
    return {
        str(name): sorted(group.index.tolist())
        for name, group in cells.groupby("clonotype")
    }


def test_clonotypes_follow_identical_chains(tmp_path) -> None:
    store, receptors = clonal_receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)

    assert ref.kind == "clonotypes" and ref.scope == "datastore"
    status = store.artifacts.inspect(ref)
    assert status.operation == "define_clonotypes"
    assert status.parameters == {
        "sequence": "nt",
        "receptor_arms": "all",
        "dual_ir": "any",
        "same_v_gene": False,
        "same_j_gene": False,
        "within": [],
        "within_receptor_type": True,
        "metric": "identity",
        "max_distance": 0,
    }
    assert status.input_ref("receptors") == receptors
    loaded = store.repertoire.load_clonotypes(ref)
    assert isinstance(loaded, Clonotypes)
    assert loaded.receptors == receptors and loaded.sequence == "nt"
    assert _members(loaded) == {
        "clonotype_1": ["cell0", "cell1", "cell4"],
        "clonotype_2": ["cell5", "cell6"],
        # Ties follow the sorted chains: cell3 lacks an alpha chain, which sorts
        # first.
        "clonotype_3": ["cell3"],
        "clonotype_4": ["cell2"],
    }
    assert loaded.cells.index.tolist() == [f"cell{index}" for index in range(N_CELLS)]
    assert pd.isna(loaded.cells.loc["cell7", "clonotype"])
    assert loaded.cells["clone_size"].tolist() == [3, 3, 1, 1, 3, 2, 2, 0]
    table = loaded.clonotypes
    assert table["clonotype"].tolist() == [
        "clonotype_1",
        "clonotype_2",
        "clonotype_3",
        "clonotype_4",
    ]
    assert table["n_cells"].tolist() == [3, 2, 1, 1]
    np.testing.assert_allclose(table["fraction"], [3 / 7, 2 / 7, 1 / 7, 1 / 7])
    # cell0 and cell1 share a configuration that cell4's second alpha chain
    # changes.
    assert table["n_configurations"].tolist() == [2, 1, 1, 1]
    assert table["receptor_type"].tolist() == ["TCR", "BCR", "TCR", "TCR"]
    first = table.iloc[0]
    assert first["VJ_1_junction_aa"] == "CAVA"
    assert first["VDJ_1_junction"] == "TGTGCCAGCAGCGCC"
    assert first["VDJ_1_v_call"] == "TRBV1"
    assert pd.isna(first["VJ_2_junction_aa"])
    assert table.iloc[1]["VDJ_1_junction_aa"] == "CARW"
    assert table.iloc[2]["VDJ_1_junction_aa"] == "CASSA"
    assert pd.isna(table.iloc[2]["VJ_1_junction_aa"])


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        (
            {"sequence": "aa"},
            [
                ["cell0", "cell1", "cell2", "cell4"],
                ["cell5", "cell6"],
                ["cell3"],
            ],
        ),
        (
            {"dual_ir": "all"},
            [
                ["cell0", "cell1"],
                ["cell5", "cell6"],
                ["cell2"],
                ["cell3"],
                ["cell4"],
            ],
        ),
        (
            {"receptor_arms": "VDJ"},
            [
                ["cell0", "cell1", "cell2", "cell3", "cell4"],
                ["cell5", "cell6"],
            ],
        ),
        (
            {"sequence": "aa", "same_v_gene": True},
            [
                ["cell0", "cell1", "cell4"],
                ["cell5", "cell6"],
                ["cell2"],
                ["cell3"],
            ],
        ),
    ],
)
def test_clonotype_options_change_the_links(tmp_path, options, expected) -> None:
    store, receptors = clonal_receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors, **options)
    partition = sorted(_members(store.repertoire.load_clonotypes(ref)).values())
    assert partition == sorted(expected)


def test_within_columns_keep_donors_apart(tmp_path) -> None:
    store, receptors = clonal_receptors(tmp_path)
    store.cells.insert(
        "donor", np.asarray(["d1", "d2", "d1", "d1", "d1", "d1", "d1", "d1"])
    )
    ref = store.repertoire.clonotypes(receptors, within="donor")
    loaded = store.repertoire.load_clonotypes(ref)
    assert loaded.within == ("donor",)
    first = loaded.cells.loc["cell0", "clonotype"]
    assert _members(loaded)[first] == ["cell0", "cell4"]
    assert loaded.cells.loc["cell1", "clonotype"] != first
    # The table names each clonotype's within group, whose labels sit in the
    # keys under the within columns' own names.
    assert "donor" not in loaded.clonotypes.columns
    assert loaded.group_keys.index.name == "group"
    assert loaded.group_keys.to_dict("index") == {
        "d1": {"donor": "d1"},
        "d2": {"donor": "d2"},
    }
    labelled = loaded.clonotypes.join(loaded.group_keys, on="group")
    assert labelled.set_index("clonotype").loc[first, "donor"] == "d1"
    donors = set(labelled["donor"])
    assert donors == {"d1", "d2"}
    status = store.artifacts.inspect(ref)
    assert status.parameters["within"] == ["donor"]
    assert status.inputs["within_fingerprint"] is not None


def test_within_columns_need_a_label_for_every_cell_with_a_receptor(tmp_path) -> None:
    store, receptors = clonal_receptors(tmp_path)
    missing = np.zeros(N_CELLS, dtype=bool)
    missing[7] = True  # cell7 has no receptor, so it needs no label
    insert_nullable_cell_column(store, "donor", np.asarray(["d1"] * N_CELLS), missing)
    assert store.repertoire.clonotypes(receptors, within=["donor"])
    missing[0] = True
    insert_nullable_cell_column(store, "batch", np.asarray(["b1"] * N_CELLS), missing)
    with pytest.raises(ValueError, match="needs a label in the within columns"):
        store.repertoire.clonotypes(receptors, within="batch")


def test_receptor_types_need_not_keep_clonotypes_apart(tmp_path) -> None:
    rows = CLONAL_ROWS + [
        # cell7 pairs a beta chain of the T cell clone with a heavy chain.
        "cell7,true,cell7_contig_1,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,CASSA,TGTGCCAGCAGCGCC,30,3,ct9,ct9_1",
        "cell7,true,cell7_contig_2,true,500,IGH,IGHV1,,IGHJ1,IGHM,true,true,CARW,TGTGCGAGATGG,20,2,ct9,ct9_2",
    ]
    store, receptors = clonal_receptors(tmp_path, rows)
    apart = store.repertoire.load_clonotypes(
        store.repertoire.clonotypes(receptors, receptor_arms="VDJ")
    )
    assert apart.cells.loc["cell7", "clone_size"] == 1
    together = store.repertoire.load_clonotypes(
        store.repertoire.clonotypes(
            receptors, receptor_arms="VDJ", within_receptor_type=False
        )
    )
    # Under "any" chain, its beta chain joins the T cells and its heavy chain
    # the B cells.
    assert together.cells.loc["cell7", "clone_size"] == 8
    assert together.within_receptor_type is False


def test_clonotypes_are_reused_and_selected_like_labels(tmp_path) -> None:
    store, receptors = clonal_receptors(tmp_path)
    first = store.repertoire.clonotypes(receptors)
    assert store.repertoire.clonotypes(receptors) == first
    assert store.repertoire.clonotypes(receptors, invalidate_cache=True) != first
    chosen = store.select_cells(first, include=["clonotype_1"])
    selected = store.artifacts.load_values(first, cell_selection=chosen)
    assert selected.cell_ids.tolist() == ["cell0", "cell1", "cell4"]
    expanded = store.select_cells(first, value="clone_size", low=2, keep_bounds=True)
    sizes = store.artifacts.load_values(
        first, value="clone_size", cell_selection=expanded
    )
    assert sizes.values.tolist() == [3, 3, 3, 2, 2]


def test_nucleotide_clonotypes_need_nucleotide_junctions(tmp_path) -> None:
    store = receptor_store(tmp_path)
    path = write_lines(
        tmp_path,
        "no_junction.tsv",
        "cell_id\tsequence_id\tlocus\tjunction_aa\tproductive",
        ["cell0\ts1\tTRB\tCASS\tT", "cell1\ts2\tTRB\tCASS\tT"],
    )
    receptors = store.repertoire.import_receptors(
        path, cell_selection=store.snapshot_cell_selection()
    )
    with pytest.raises(ValueError, match="store no junction"):
        store.repertoire.clonotypes(receptors)
    with pytest.raises(
        ValueError, match="^same_v_gene compares each chain's v_call, which these"
    ):
        store.repertoire.clonotypes(receptors, sequence="aa", same_v_gene=True)
    loaded = store.repertoire.load_clonotypes(
        store.repertoire.clonotypes(receptors, sequence="aa")
    )
    assert sorted(_members(loaded).values()) == [["cell0", "cell1"]]
    assert "VDJ_1_junction" not in loaded.clonotypes.columns


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"sequence": "dna"}, ValueError, "sequence must be one of"),
        ({"receptor_arms": "both"}, ValueError, "receptor_arms must be one of"),
        ({"dual_ir": "some"}, ValueError, "dual_ir must be one of"),
        ({"same_v_gene": "yes"}, TypeError, "same_v_gene must be a boolean"),
        ({"within": 3}, TypeError, "within must be"),
        ({"within": ["donor", "donor"]}, ValueError, "within names a column twice"),
        ({"within": ["donor", 3]}, TypeError, "within must be"),
        ({"within": ["absent"]}, ValueError, "'absent', which is not a cell metadata"),
    ],
)
def test_clonotype_arguments_are_checked(tmp_path, kwargs, error, message) -> None:
    store, receptors = clonal_receptors(tmp_path)
    store.cells.insert("donor", np.asarray(["d1"] * N_CELLS))
    with pytest.raises(error, match=message):
        store.repertoire.clonotypes(receptors, **kwargs)


def test_clonotypes_refuse_other_artifacts(tmp_path) -> None:
    store, receptors = clonal_receptors(tmp_path)
    with pytest.raises(ValueError, match="immune_receptors artifact"):
        store.repertoire.clonotypes(store.snapshot_cell_selection())
    with pytest.raises(TypeError, match="receptors must be an ArtifactRef"):
        store.repertoire.clonotypes("receptors")
    with pytest.raises(ValueError, match="clonotypes artifact"):
        store.repertoire.load_clonotypes(receptors)
    with pytest.raises(TypeError, match="clonotypes must be an ArtifactRef"):
        store.repertoire.load_clonotypes("clonotypes")


def test_a_read_only_store_refuses_new_clonotypes(tmp_path) -> None:
    store, receptors = clonal_receptors(tmp_path)
    saved = store.repertoire.clonotypes(receptors)
    reader = receptor_store(tmp_path, zarr_mode="r")
    assert reader.repertoire.clonotypes(receptors) == saved
    with pytest.raises(PermissionError, match="define_clonotypes"):
        reader.repertoire.clonotypes(receptors, sequence="aa")


def test_clone_abundance_counts_the_clonotypes_of_each_receptor_type(
    tmp_path,
) -> None:
    store, receptors = clonal_receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    result = store.plots.clone_abundance(ref, max_clones=3, show=False)
    try:
        sizes = result.tables["sizes"]
        # One clonotype of 2 B cells; T cell clonotypes of 1, 1 and 3 cells.
        assert sizes.to_dict("list") == {
            "group": ["BCR", "TCR", "TCR"],
            "clone_size": [2, 1, 3],
            "n_clones": [1, 2, 1],
        }
        assert sizes.groupby("clone_size")["n_clones"].sum().to_dict() == {
            1: 2,
            2: 1,
            3: 1,
        }
        top = result.tables["top"]
        assert top.to_dict("list") == {
            "clone": ["clonotype_1"] * 2 + ["clonotype_2"] * 2 + ["clonotype_3"] * 2,
            # Each clonotype is named by its representative's junction, and
            # clonotype_1 and clonotype_3, which share their beta chain, by
            # their numbers too.
            "label": ["CASSA 1"] * 2 + ["CARW"] * 2 + ["CASSA 3"] * 2,
            "group": ["BCR", "TCR"] * 3,
            "n_cells": [0, 3, 2, 0, 0, 1],
        }
        ticks = [tick.get_text() for tick in result.axes["top"].get_yticklabels()]
        assert ticks == top["label"].drop_duplicates().tolist()
        # The title names clonotypes once, so their labels need not.
        assert result.axes["top"].get_title() == "Largest 3 clonotypes"
        assert result.legends[0].label == "receptor_type"
        assert len(result.axes["sizes"].get_lines()) == 2
        assert result.provenance.n_cells == 7
        # Plain numbers label the log axes, not 2x10^0.
        ax = result.axes["sizes"]
        result.figure.canvas.draw()
        labels = [tick.get_text() for tick in ax.get_xticklabels(minor=True)]
        assert "2" in labels and "3" in labels
    finally:
        result.close()


def test_clone_abundance_splits_clonotypes_by_group_and_receptor_type(
    tmp_path,
) -> None:
    store, receptors = clonal_receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    store.cells.insert(
        "sample", np.asarray(["s1", "s2", "s1", "s2", "s1", "s1", "s2", "s1"])
    )
    result = plotting.clone_abundance(store, ref, group_by="sample", show=False)
    try:
        sizes = result.tables["sizes"]
        assert sizes.values.tolist() == [
            ["s1, BCR", 1, 1],
            ["s1, TCR", 1, 1],
            ["s1, TCR", 2, 1],
            ["s2, BCR", 1, 1],
            ["s2, TCR", 1, 2],
        ]
        in_s1 = sizes[sizes["group"].str.startswith("s1")]
        assert in_s1.groupby("clone_size")["n_clones"].sum().to_dict() == {1: 2, 2: 1}
        top = result.tables["top"].set_index(["clone", "group"])["n_cells"]
        assert top.loc[("clonotype_1", ["s1, TCR", "s2, TCR"])].tolist() == [2, 1]
        assert top.loc[("clonotype_1", ["s1, BCR", "s2, BCR"])].tolist() == [0, 0]
        assert result.legends[0].label == "sample, receptor_type"
        assert len(result.axes["sizes"].get_lines()) == 4
    finally:
        result.close()
    with pytest.raises(ValueError, match="max_clones must be at least 1"):
        plotting.clone_abundance(store, ref, max_clones=0, show=False)
    # Shown by default: on Agg the owned figure is closed.
    shown = plotting.clone_abundance(store, ref, max_clones=2)
    drawn = shown.tables["top"].drop_duplicates("clone")
    assert drawn["clone"].tolist() == ["clonotype_1", "clonotype_2"]
    # clonotype_3 is not drawn, so clonotype_1 shares its junction with no
    # clonotype of the figure.
    assert drawn["label"].tolist() == ["CASSA", "CARW"]
    with pytest.raises(TypeError, match="does not provide clones.load_clones"):
        plotting.clone_abundance(object(), ref, show=False)


def test_clonotypes_without_a_beta_chain_are_named_by_their_alpha_chain(
    tmp_path,
) -> None:
    rows = CLONAL_ROWS + [
        "cell7,true,cell7_contig_1,true,500,TRA,TRAV9,,TRAJ9,TRAC,true,true,CAVZ,TGTGCTGTGTAA,10,1,ct9,ct9_1",
    ]
    store, receptors = clonal_receptors(tmp_path, rows)
    ref = store.repertoire.clonotypes(receptors)
    result = plotting.clone_abundance(store, ref, show=False)
    try:
        labels = dict(zip(result.tables["top"]["clone"], result.tables["top"]["label"]))
        clonotype_of = store.repertoire.load_clonotypes(ref).cells["clonotype"]
        alone = {cell: str(clonotype_of[cell]) for cell in ("cell2", "cell3", "cell7")}
        # cell7's alpha chain alone names its clonotype; cell2 and cell3 carry
        # the beta chain CASSA of clonotype_1, so the three show their numbers.
        assert labels == {
            "clonotype_1": "CASSA 1",
            "clonotype_2": "CARW",
            alone["cell2"]: f"CASSA {alone['cell2'].removeprefix('clonotype_')}",
            alone["cell3"]: f"CASSA {alone['cell3'].removeprefix('clonotype_')}",
            alone["cell7"]: "CAVZ",
        }
    finally:
        result.close()


def test_figure_labels_name_clonotypes_by_their_representatives_junction(
    tmp_path,
) -> None:
    store, receptors = clonal_receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    # In any order and repeated; clonotype_4 is cell2's, of alpha CAVA and
    # beta CASSA. An ID given twice is one clonotype, so CASSA names one.
    labels = store._clone_figure_labels(
        ref, ["clonotype_4", "clonotype_2", "clonotype_4"]
    )
    assert labels.tolist() == ["CASSA", "CARW", "CASSA"]
    # clonotype_1, clonotype_3 (cell3's beta chain alone) and clonotype_4 all
    # carry CASSA, so each shows its number, wherever it is listed.
    assert store._clone_figure_labels(
        ref, ["clonotype_4", "clonotype_2", "clonotype_1", "clonotype_3"]
    ).tolist() == ["CASSA 4", "CARW", "CASSA 1", "CASSA 3"]
    assert store._clone_figure_labels(ref, []).tolist() == []
    with pytest.raises(ValueError, match=r"No clonotype is labelled \['clonotype_9'\]"):
        store._clone_figure_labels(ref, ["clonotype_1", "clonotype_9"])
    # Imported clones are labelled by their IDs alone.
    path = tmp_path / "clonotypes.csv"
    store.clones.export_clones(ref, path)
    imported = store.clones.import_clones(
        path, cell_selection=store.snapshot_cell_selection()
    )
    assert store._clone_figure_labels(
        imported, ["clonotype_2", "clonotype_1", "clonotype_3"]
    ).tolist() == ["clonotype_2", "clonotype_1", "clonotype_3"]
    assert store._clone_figure_labels(imported, []).tolist() == []
    shown = plotting.clone_abundance(store, imported, max_clones=3, show=False)
    try:
        assert shown.tables["top"]["label"].tolist() == [
            "clonotype_1",
            "clonotype_2",
            "clonotype_3",
        ]
        assert shown.axes["top"].get_title() == "Largest 3 clones"
    finally:
        shown.close()


def test_figure_labels_refuse_ids_that_only_resemble_a_clonotype_id(tmp_path) -> None:
    store, receptors = clonal_receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    # The IDs are clonotype_1 to clonotype_4, stored 11 characters wide: an ID
    # that extends one past that width, a prefix of one, another case, a
    # space or a leading zero names no clonotype, and the first five missing
    # are named in the order given.
    near = [
        "clonotype_12",
        "clonotype_",
        "Clonotype_1",
        "clonotype_1 ",
        " clonotype_2",
        "clonotype_04",
    ]
    with pytest.raises(ValueError) as refused:
        store._clone_figure_labels(ref, ["clonotype_2", *near, "clonotype_2"])
    assert str(refused.value) == f"No clonotype is labelled {near[:5]}"
    # IDs given as numbers are their text, which names no clonotype either.
    with pytest.raises(ValueError, match=r"No clonotype is labelled \['1'\]"):
        store._clone_figure_labels(ref, [1])  # type: ignore[list-item]
    # Nor does an ID that ends in a NUL character, which a comparison of
    # fixed-width NumPy text ignores.
    with pytest.raises(ValueError, match=r"labelled \['clonotype_1\\x00'\]"):
        store._clone_figure_labels(ref, ["clonotype_1\x00"])


def test_figure_labels_name_a_clone_given_many_times_by_its_long_junction(
    tmp_path, monkeypatch
) -> None:
    # cell7's beta chain of 20,000 residues names its own clonotype; 1 kB
    # cannot read that junction.
    junction = "C" + "A" * 20_000 + "F"
    rows = CLONAL_ROWS + [
        "cell7,true,cell7_contig_1,true,500,TRB,TRBV9,,TRBJ9,TRBC1,true,true,"
        f"{junction},TGT{'GCT' * 20_000}TTT,10,1,ct9,ct9_1",
    ]
    store, receptors = clonal_receptors(tmp_path, rows)
    ref = store.repertoire.clonotypes(receptors)
    clonotype = str(store.repertoire.load_clonotypes(ref).cells["clonotype"]["cell7"])
    assert store._clone_figure_labels(ref, [clonotype] * 20).tolist() == [junction] * 20
    monkeypatch.setattr(store, "memoryBytes", 1_000)
    with pytest.raises(
        MemoryError, match="^Labelling clonotypes by their junctions holds about"
    ):
        store._clone_figure_labels(ref, [clonotype])


def test_clonotypes_of_one_arm_are_named_by_the_chains_they_share(tmp_path) -> None:
    store, receptors = clonal_receptors(tmp_path)
    ids = ["clonotype_1", "clonotype_2", "clonotype_3"]
    # By alpha and light chains: cell0, cell1 and cell4 share CAVA, the B
    # cells CQQY, and cell2 has CAVA from other nucleotides; their beta and
    # heavy chains would name them CASSA and CARW.
    by_alpha = store.repertoire.clonotypes(receptors, receptor_arms="VJ")
    assert _members(store.repertoire.load_clonotypes(by_alpha))["clonotype_1"] == [
        "cell0",
        "cell1",
        "cell4",
    ]
    assert store._clone_figure_labels(by_alpha, ids).tolist() == [
        "CAVA 1",
        "CQQY",
        "CAVA 3",
    ]
    by_beta = store.repertoire.clonotypes(receptors, receptor_arms="VDJ")
    assert store._clone_figure_labels(by_beta, ids[:2]).tolist() == [
        "CASSA",
        "CARW",
    ]


def test_figure_labels_name_the_clonotypes_by_their_junctions(tmp_path) -> None:
    store, receptors = clonal_receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    labels = store._clone_figure_labels(ref, ["clonotype_1"])
    assert labels.tolist() == ["CASSA"]
    # The junctions of the representatives, read on the caller's ledger.
    ledger = MemoryLedger(None, "Labelling clonotypes by their junctions")
    ledger.hold("the clonotypes labelled", 400)
    junctions = store._representative_junctions(ref, np.asarray([0]), ledger=ledger)
    assert junctions.tolist() == ["CASSA"]


def test_a_representative_without_a_junction_is_refused(tmp_path) -> None:
    from cytearc.storage.artifacts import artifact_group

    store, receptors = clonal_receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    # Point clonotype_1's representative at cell7, which has no receptor.
    representatives = artifact_group(store.zw, ref)["representatives"]
    representatives[0] = 7
    with pytest.raises(ValueError, match="no junction amino-acid sequence"):
        store._clone_figure_labels(ref, ["clonotype_1"])


def test_figure_labels_name_missing_and_incomplete_clonotypes(tmp_path) -> None:
    store, receptors = clonal_receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    missing = ArtifactRef(
        scope="datastore", kind="clonotypes", artifact_id="0" * 64, assay=None
    )
    with pytest.raises(KeyError, match="Artifact does not exist"):
        store._clone_figure_labels(missing, ["clonotype_1"])
    with pytest.raises(KeyError, match="Artifact does not exist"):
        store._clone_figure_nouns(missing)
    for read in (
        lambda clones: store._clone_figure_labels(clones, ["clonotype_1"]),
        store._clone_figure_nouns,
    ):
        with pytest.raises(TypeError, match="clones must be an ArtifactRef"):
            read("clonotype_1")
        with pytest.raises(
            ValueError,
            match="clones must be a clones or clonotypes artifact, not "
            "'immune_receptors'",
        ):
            read(receptors)
    # A definition that stopped before it finished leaves an incomplete
    # artifact.
    artifact_group(store.zw, ref).attrs["complete"] = False
    with pytest.raises(RuntimeError, match="Artifact is incomplete"):
        store._clone_figure_labels(ref, ["clonotype_1"])
    with pytest.raises(RuntimeError, match="Artifact is incomplete"):
        store._clone_figure_nouns(ref)


def test_clone_abundance_needs_a_store_that_labels_clones(tmp_path) -> None:
    store, receptors = clonal_receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    loader = SimpleNamespace(
        clones=SimpleNamespace(load_clones=store.clones.load_clones)
    )
    # Clonotypes keep their receptor types apart, which the store groups.
    with pytest.raises(TypeError, match="does not provide the groups of figures"):
        plotting.clone_abundance(loader, ref, show=False)
    loader._figure_groups = store._figure_groups
    # The names of the clones drawn title the figure, and come first.
    with pytest.raises(TypeError, match="does not provide what figures call clones"):
        plotting.clone_abundance(loader, ref, show=False)
    loader._clone_figure_nouns = store._clone_figure_nouns
    with pytest.raises(TypeError, match="does not provide the figure labels"):
        plotting.clone_abundance(loader, ref, show=False)
    loader._clone_figure_labels = store._clone_figure_labels
    result = plotting.clone_abundance(loader, ref, show=False)
    try:
        assert result.axes["top"].get_title() == "Largest 4 clonotypes"
    finally:
        result.close()


# The cells' samples for following clonotypes: clonotype_1 holds cell0 and
# cell4 of s1 and cell1 of s2, clonotype_2 the B cells cell5 of s1 and cell6
# of s2, clonotype_3 cell3 of s2 and clonotype_4 cell2 of s1.
TRACKED_SAMPLES = np.asarray(["s1", "s2", "s1", "s2", "s1", "s1", "s2", "s1"])


class _NamesOnly:
    """A store that names and labels clones, but loads neither clones nor clonotypes."""

    def __init__(self, store: DataStore) -> None:
        self.clones = SimpleNamespace(load_clones=_loads)
        self.repertoire = SimpleNamespace(load_clonotypes=_loads)
        self._clone_figure_nouns = store._clone_figure_nouns
        self._clone_figure_labels = store._clone_figure_labels


def _loads(*_args, **_kwargs):
    raise AssertionError("the tracking plot loads the clones")


def _legend(result) -> list[str]:
    return [
        text.get_text() for text in result.axes["tracking"].get_legend().get_texts()
    ]


def test_the_tracking_plot_labels_clonotypes_by_their_junctions(tmp_path) -> None:
    store, receptors = clonal_receptors(tmp_path)
    store.cells.insert("sample", TRACKED_SAMPLES)
    ref = store.repertoire.clonotypes(receptors)
    # The largest clonotype of each sample's T cells is clonotype_1, which in
    # s2 ties cell3's clonotype_3 and wins as the larger one overall, and of
    # each sample's B cells clonotype_2. Their junctions differ, so the lines
    # show no numbers; the labels need neither the clones nor the clonotypes
    # loaded.
    tracking = store.clones.track(ref, group_by="sample", top_n=1)
    result = plotting.clone_tracking(_NamesOnly(store), tracking, show=False)
    try:
        assert result.provenance.extras["tracked"] == ["clonotype_1", "clonotype_2"]
        assert _legend(result) == ["CASSA", "CARW"]
        legend = result.axes["tracking"].get_legend()
        assert legend.get_title().get_text() == "Clonotype"
        assert result.axes["tracking"].get_title() == "Clonotypes across groups"
        # The table holds each clonotype's ID beside the label drawn.
        table = result.tables["tracking"]
        assert table[["clone", "label"]].drop_duplicates().values.tolist() == [
            ["clonotype_1", "CASSA"],
            ["clonotype_2", "CARW"],
        ]
    finally:
        result.close()
    unlabelled = _NamesOnly(store)
    del unlabelled._clone_figure_labels
    with pytest.raises(TypeError, match="does not provide the figure labels"):
        plotting.clone_tracking(unlabelled, tracking, show=False)


def test_the_tracking_plot_numbers_clonotypes_that_share_a_junction(
    tmp_path,
) -> None:
    store, receptors = clonal_receptors(tmp_path)
    store.cells.insert("sample", TRACKED_SAMPLES)
    ref = store.repertoire.clonotypes(receptors)
    # The two largest of each group add cell2's clonotype_4 in s1's T cells
    # and cell3's clonotype_3 in s2's, whose beta chains are clonotype_1's
    # CASSA, so the three show their numbers.
    result = store.plots.clone_tracking(
        store.clones.track(ref, group_by="sample", top_n=2), show=False
    )
    try:
        ax = result.axes["tracking"]
        assert result.provenance.extras["tracked"] == [
            "clonotype_1",
            "clonotype_2",
            "clonotype_3",
            "clonotype_4",
        ]
        assert _legend(result) == ["CASSA 1", "CARW", "CASSA 3", "CASSA 4"]
        # The T cells and B cells of a sample share its position. s1 holds
        # three T cells with a clonotype and one B cell, s2 two and one.
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["s1", "s2"]
        assert [line.get_ydata().tolist() for line in ax.get_lines()] == [
            pytest.approx([2 / 3, 1 / 2]),
            pytest.approx([1.0, 1.0]),
            pytest.approx([0.0, 1 / 2]),
            pytest.approx([1 / 3, 0.0]),
        ]
    finally:
        result.close()
    # A figure numbers only the clonotypes that share a junction among its
    # own lines, whatever order they are named in.
    for labels, expected in (
        (["clonotype_1"], ["CASSA"]),
        (["clonotype_3", "clonotype_1"], ["CASSA 1", "CASSA 3"]),
        (["clonotype_3", "clonotype_2"], ["CARW", "CASSA"]),
    ):
        named = plotting.clone_tracking(
            store,
            store.clones.track(ref, group_by="sample", labels=labels),
            show=False,
        )
        try:
            assert _legend(named) == expected, labels
        finally:
            named.close()
    # Imported clones keep their IDs.
    path = tmp_path / "clonotypes.csv"
    store.clones.export_clones(ref, path)
    imported = store.clones.import_clones(
        path, cell_selection=store.snapshot_cell_selection()
    )
    plain = plotting.clone_tracking(
        store, store.clones.track(imported, group_by="sample", top_n=1), show=False
    )
    try:
        assert _legend(plain) == ["clonotype_1"]
    finally:
        plain.close()


def test_the_fate_plot_labels_clonotypes_by_their_junctions(tmp_path) -> None:
    store, receptors = clonal_receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    # The even cells are early and the odd ones late, of one fate each.
    store.cells.insert("early", np.arange(N_CELLS) % 2 == 0)
    store.cells.insert("late", np.arange(N_CELLS) % 2 == 1)
    store.cells.insert("outcome", np.asarray(["x"] * N_CELLS))
    bias = store.clones.fate_bias(
        ref,
        group_by="outcome",
        early=store.snapshot_cell_selection("early"),
        late=store.snapshot_cell_selection("late"),
    )
    result = store.plots.clone_fate_bias(bias, show=False)
    try:
        table = result.tables["clones"]
        ids = table["clone"].tolist()
        # The late cells cell1, cell3 and cell5 hold clonotype_1, clonotype_3
        # and clonotype_2; the first two share the beta chain CASSA, so both
        # show their numbers, as in the other figures.
        assert sorted(ids) == ["clonotype_1", "clonotype_2", "clonotype_3"]
        named = {
            "clonotype_1": "CASSA 1",
            "clonotype_2": "CARW",
            "clonotype_3": "CASSA 3",
        }
        assert table["label"].tolist() == [named[clone] for clone in ids]
        ticks = [tick.get_text() for tick in result.axes["fates"].get_xticklabels()]
        assert ticks == table["label"].tolist()
        assert result.provenance.extras["clones"] == ref.to_dict()
        assert result.provenance.extras["bias"] == bias.to_dict()
        assert result.axes["fates"].get_xlabel() == "Clonotype"
        assert result.axes["early"].get_title() == (
            "Fates of the clonotypes of early cells"
        )
    finally:
        result.close()


def test_figures_name_clonotype_clusters_as_their_definition_does(tmp_path) -> None:
    store, receptors = clonal_receptors(tmp_path)
    store.cells.insert("sample", TRACKED_SAMPLES)
    identical = store.repertoire.clonotypes(receptors)
    # Within three substitutions cell2's alpha chain joins it to clonotype_1,
    # which then holds four T cells; clonotype_2 holds the B cells and
    # clonotype_3 cell3, whose beta chain alone is clonotype_1's CASSA.
    near = store.repertoire.clonotypes(receptors, metric="hamming", max_distance=3)
    assert store._clone_figure_nouns(identical) == ("clonotypes", "clonotype")
    assert store._clone_figure_nouns(near) == (
        "clonotype clusters",
        "clonotype cluster",
    )
    # A distance metric defines clusters even at a max_distance that links
    # only identical junctions, as the definition's log names them.
    exact = store.repertoire.clonotypes(receptors, metric="levenshtein", max_distance=0)
    assert store._clone_figure_nouns(exact) == (
        "clonotype clusters",
        "clonotype cluster",
    )
    path = tmp_path / "clusters.csv"
    store.clones.export_clones(near, path)
    imported = store.clones.import_clones(
        path, cell_selection=store.snapshot_cell_selection()
    )
    assert store._clone_figure_nouns(imported) == ("clones", "clone")
    result = plotting.clone_abundance(store, near, max_clones=3, show=False)
    try:
        top = result.tables["top"].drop_duplicates("clone")
        assert top["clone"].tolist() == ["clonotype_1", "clonotype_2", "clonotype_3"]
        assert top["label"].tolist() == ["CASSA 1", "CARW", "CASSA 3"]
        assert result.axes["top"].get_title() == "Largest 3 clonotype clusters"
    finally:
        result.close()
    two = plotting.clone_abundance(store, near, max_clones=2, show=False)
    try:
        assert two.axes["top"].get_title() == "Largest 2 clonotype clusters"
        assert two.tables["top"]["label"].drop_duplicates().tolist() == [
            "CASSA",
            "CARW",
        ]
    finally:
        two.close()
    # One clone drawn is named in the singular, of each kind.
    for ref, title in (
        (near, "Largest clonotype cluster"),
        (identical, "Largest clonotype"),
        (imported, "Largest clone"),
    ):
        one = plotting.clone_abundance(store, ref, max_clones=1, show=False)
        try:
            assert one.axes["top"].get_title() == title
        finally:
            one.close()
    tracked = store.plots.clone_tracking(
        store.clones.track(near, group_by="sample", top_n=2), show=False
    )
    try:
        assert _legend(tracked) == ["CASSA 1", "CARW", "CASSA 3"]
    finally:
        tracked.close()
    with pytest.raises(ValueError, match="metric must be one of"):
        clonotype_nouns("tcrdist")


def test_clonotypes_load_and_export_as_clones(tmp_path) -> None:
    store, receptors = clonal_receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    loaded = store.clones.load_clones(ref)
    assert loaded.source == "define_clonotypes"
    assert loaded.kept_apart == ("receptor_type",)
    cells = loaded.cells
    assert cells["clone"].astype(object).where(
        cells["clone"].notna(), "-"
    ).tolist() == [
        "clonotype_1",
        "clonotype_1",
        "clonotype_4",
        "clonotype_3",
        "clonotype_1",
        "clonotype_2",
        "clonotype_2",
        "-",
    ]
    assert cells["clone_size"].tolist() == [3, 3, 1, 1, 3, 2, 2, 0]
    assert cells["receptor_type"].astype(str).tolist() == [
        *["TCR"] * 5,
        "BCR",
        "BCR",
        "no IR",
    ]
    assert list(cells["receptor_type"].cat.categories) == ["BCR", "no IR", "TCR"]
    assert loaded.clones.to_dict("list") == {
        "clone": ["clonotype_1", "clonotype_2", "clonotype_3", "clonotype_4"],
        "n_cells": [3, 2, 1, 1],
    }
    path = tmp_path / "clonotypes.csv"
    assert store.clones.export_clones(ref, path).rows == 7
    assert path.read_text().splitlines() == [
        "Cell_ID,Clone_ID",
        "cell0,clonotype_1",
        "cell1,clonotype_1",
        "cell2,clonotype_4",
        "cell3,clonotype_3",
        "cell4,clonotype_1",
        "cell5,clonotype_2",
        "cell6,clonotype_2",
    ]
    imported = store.clones.import_clones(
        path, cell_selection=store.snapshot_cell_selection()
    )
    assert loaded_partition(store, imported) == loaded_partition(store, ref)
    assert store.clones.load_clones(imported).kept_apart == ()


def test_clone_summaries_measure_each_receptor_type_apart(tmp_path) -> None:
    store, receptors = clonal_receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    diversity = store.clones.diversity(ref)
    # The receptor types that clonotypes keep apart are their groups' key
    # column, beside the tables' own columns.
    assert diversity.group_keys.to_dict("index") == {
        "BCR": {"receptor_type": "BCR"},
        "TCR": {"receptor_type": "TCR"},
    }
    indices = diversity.indices.join(diversity.group_keys, on="group")
    assert indices[
        [
            "group",
            "receptor_type",
            "n_cells",
            "n_clones",
            "n_singletons",
            "n_doubletons",
        ]
    ].values.tolist() == [["BCR", "BCR", 2, 1, 0, 1], ["TCR", "TCR", 5, 3, 2, 0]]
    curves = store.clones.rarefaction(ref, orders=(0,))
    observed = curves.curves.join(curves.group_keys, on="group")
    observed = observed[observed["method"] == "observed"]
    assert observed[["receptor_type", "size", "diversity"]].values.tolist() == [
        ["BCR", 2, 1.0],
        ["TCR", 5, 3.0],
    ]
    expansion = store.clones.load_expansion(store.clones.expansion(ref))
    cells = expansion.cells
    assert cells["group_clone_size"].tolist() == [3, 3, 1, 1, 3, 2, 2, 0]
    # A cell without a clonotype has no group and no frequency.
    np.testing.assert_allclose(
        cells["group_clone_frequency"], [0.6, 0.6, 0.2, 0.2, 0.6, 1.0, 1.0, np.nan]
    )
    assert cells["group"].astype(object).where(
        cells["group"].notna(), "-"
    ).tolist() == [
        *["TCR"] * 5,
        "BCR",
        "BCR",
        "-",
    ]
    assert set(
        expansion.categories.join(expansion.group_keys, on="group")["receptor_type"]
    ) == {"BCR", "TCR"}


def test_summaries_keep_receptor_types_apart_within_one_clonotype(tmp_path) -> None:
    rows = CLONAL_ROWS + [
        # cell7 pairs a beta chain of the T cell clone with a heavy chain.
        "cell7,true,cell7_contig_1,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,CASSA,TGTGCCAGCAGCGCC,30,3,ct9,ct9_1",
        "cell7,true,cell7_contig_2,true,500,IGH,IGHV1,,IGHJ1,IGHM,true,true,CARW,TGTGCGAGATGG,20,2,ct9,ct9_2",
    ]
    store, receptors = clonal_receptors(tmp_path, rows)
    together = store.repertoire.clonotypes(
        receptors, receptor_arms="VDJ", within_receptor_type=False
    )
    # One clonotype holds every cell, of three receptor types.
    assert store.clones.load_clones(together).clones["n_cells"].tolist() == [8]
    expansion = store.clones.load_expansion(store.clones.expansion(together))
    assert expansion.cells["group_clone_size"].tolist() == [5, 5, 5, 5, 5, 2, 2, 1]
    diversity = store.clones.diversity(together)
    indices = diversity.indices.join(diversity.group_keys, on="group")
    assert indices[["receptor_type", "n_cells", "n_clones"]].values.tolist() == [
        ["ambiguous", 1, 1],
        ["BCR", 2, 1],
        ["TCR", 5, 1],
    ]
    # By alternate cells, a and b: its T cells are cell0, cell2 and cell4 in a
    # and cell1 and cell3 in b, its B cells cell6 in a and cell5 in b, and
    # cell7, in b, is ambiguous. Overlap and coupling count its cells of each
    # type as one clone of that type's groups.
    store.cells.insert("tp", np.asarray(["a", "b"] * (N_CELLS // 2)))
    overlap = store.clones.overlap(together, group_by="tp")
    assert overlap.pairs[["group_1", "group_2", "shared"]].values.tolist() == [
        ["a, BCR", "b, BCR", 1],
        ["a, TCR", "b, TCR", 1],
    ]
    coupling = store.clones.coupling(
        together, group_by="tp", method="abundance_cosine", n_permutations=0
    )
    assert coupling.group_sizes[["group", "n_cells", "n_clones"]].values.tolist() == [
        ["a, BCR", 1, 1],
        ["a, TCR", 3, 1],
        ["b, ambiguous", 1, 1],
        ["b, BCR", 1, 1],
        ["b, TCR", 2, 1],
    ]
    # Each pair shares its one clone, so the clone sizes are parallel.
    assert coupling.pairs[["group_1", "group_2", "coupling"]].values.tolist() == [
        ["a, BCR", "b, BCR", 1.0],
        ["a, TCR", "b, TCR", 1.0],
    ]


# Beta chains CASSA of cell0 to cell2 and CASSB of cell3 and cell4, heavy
# chains CARW of cell5 and cell6, and cell7, ambiguous, with CASSA and, of
# fewer UMIs, CARW: defined across receptor types, one clonotype holds the
# cells of CASSA, CARW and cell7, of three receptor types, and another those
# of CASSB.
SPANNING_ROWS = [
    f"cell{cell},true,cell{cell}_contig_{contig},true,500,{chain},{chain}V1,,"
    f"{chain}J1,{chain}C,true,true,{junction_aa},{junction},50,{umis},ct{cell},"
    f"ct{cell}_{contig}"
    for cell, contig, chain, junction_aa, junction, umis in (
        (0, 1, "TRB", "CASSA", "TGTGCCAGCAGCGCC", 5),
        (1, 1, "TRB", "CASSA", "TGTGCCAGCAGCGCC", 5),
        (2, 1, "TRB", "CASSA", "TGTGCCAGCAGCGCC", 5),
        (3, 1, "TRB", "CASSB", "TGTGCCAGCAGCTAC", 5),
        (4, 1, "TRB", "CASSB", "TGTGCCAGCAGCTAC", 5),
        (5, 1, "IGH", "CARW", "TGTGCGAGATGG", 5),
        (6, 1, "IGH", "CARW", "TGTGCGAGATGG", 5),
        (7, 1, "TRB", "CASSA", "TGTGCCAGCAGCGCC", 5),
        (7, 2, "IGH", "CARW", "TGTGCGAGATGG", 2),
    )
]


@pytest.fixture
def spanning(tmp_path) -> SimpleNamespace:
    """The clonotypes of ``SPANNING_ROWS`` across and within receptor types, with fates."""
    store, receptors = clonal_receptors(tmp_path, SPANNING_ROWS)
    store.cells.insert("early", np.isin(np.arange(N_CELLS), [0, 3, 5]))
    store.cells.insert("late", np.isin(np.arange(N_CELLS), [1, 2, 4, 6, 7]))
    store.cells.insert(
        "outcome", np.asarray(["x", "x", "x", "x", "y", "x", "x", "y"], dtype=object)
    )
    store.cells.insert("tp", np.asarray(["a", "b"] * (N_CELLS // 2), dtype=object))
    return SimpleNamespace(
        store=store,
        together=store.repertoire.clonotypes(
            receptors, receptor_arms="VDJ", within_receptor_type=False
        ),
        apart=store.repertoire.clonotypes(receptors, receptor_arms="VDJ"),
    )


def _fate_bias_of(spanning: SimpleNamespace, clonotypes: ArtifactRef) -> ArtifactRef:
    store = spanning.store
    return store.clones.fate_bias(
        clonotypes,
        group_by="outcome",
        early=store.snapshot_cell_selection("early"),
        late=store.snapshot_cell_selection("late"),
    )


def test_clonotypes_across_receptor_types_save_each_cells_clonotype_within_its_type(
    spanning, tmp_path
) -> None:
    store, together, apart = spanning.store, spanning.together, spanning.apart
    # Defined across types, the clonotypes also save each cell's clonotype as
    # within_receptor_type=True defines it, and those defined within types,
    # whose clonotypes they are, do not.
    across = artifact_group(store.zw, together)
    within = artifact_group(store.zw, apart)
    np.testing.assert_array_equal(across["codes_within_type"][:], within["codes"][:])
    assert "codes_within_type" not in within
    assert across["codes"][:].tolist() == [0, 0, 0, 1, 1, 0, 0, 0]
    # Loaded as clones, the clonotypes across types are those that they
    # define, beside each cell's clonotype within its type, labelled by its
    # clonotype and type, and by its clonotype alone where that holds no
    # other.
    cells = store.clones.load_clones(together).cells
    assert (
        cells["clone"].astype(str).tolist()
        == ["clonotype_1"] * 3 + ["clonotype_2"] * 2 + ["clonotype_1"] * 3
    )
    assert cells["within_type_clone"].astype(str).tolist() == [
        *["clonotype_1 (TCR)"] * 3,
        *["clonotype_2"] * 2,
        *["clonotype_1 (BCR)"] * 2,
        "clonotype_1 (ambiguous)",
    ]
    # Its categories follow the clonotypes within types by rank.
    assert cells["within_type_clone"].cat.categories.tolist() == [
        "clonotype_1 (TCR)",
        "clonotype_1 (BCR)",
        "clonotype_2",
        "clonotype_1 (ambiguous)",
    ]
    assert "within_type_clone" not in store.clones.load_clones(apart).cells
    # Their export writes each cell's clonotype that they define.
    path = tmp_path / "together.csv"
    store.clones.export_clones(together, path)
    written = pd.read_csv(path, dtype=str)
    assert written.shape[1] == 2
    assert sorted(map(tuple, written.values.tolist())) == sorted(
        zip(cells.index.astype(str), cells["clone"].astype(str), strict=True)
    )


def test_clonotypes_across_types_saved_without_their_clonotypes_within_types_are_refused(
    spanning, tmp_path
) -> None:
    store, together = spanning.store, spanning.together
    # Clonotypes across types saved before each cell's clonotype within its
    # type was saved cannot be measured as the summaries measure them.
    group = artifact_group(store.zw, together)
    saved = group["codes_within_type"][:]
    del group["codes_within_type"]
    early = store.snapshot_cell_selection("early")
    late = store.snapshot_cell_selection("late")
    for measure in (
        lambda: store.clones.diversity(together),
        lambda: store.clones.track(together, top_n=1),
        lambda: store.clones.fate_bias(
            together, group_by="outcome", early=early, late=late
        ),
        lambda: store.repertoire.gene_usage(together, count="clonotypes"),
        lambda: store.clones.load_clones(together),
    ):
        with pytest.raises(ValueError, match="without each cell's clonotype within"):
            measure()
    # Their own export still reads the clonotypes that they define, and
    # defining them again does not reuse them but saves the codes beside them.
    store.clones.export_clones(together, tmp_path / "old.csv")
    receptors = store.artifacts.inspect(together).input_ref("receptors")
    again = store.repertoire.clonotypes(
        receptors, receptor_arms="VDJ", within_receptor_type=False
    )
    assert again != together
    np.testing.assert_array_equal(
        artifact_group(store.zw, again)["codes_within_type"][:], saved
    )
    assert store.clones.diversity(again).indices["n_clones"].tolist() == [1, 1, 2]


def test_fate_bias_tests_a_clonotype_across_receptor_types_within_each_type(
    spanning,
) -> None:
    # Every summary measures clonotypes across receptor types by each cell's
    # clonotype within its type, so fate bias follows the early cells of each
    # type to their clonotype's late cells of that type, and tests them among
    # the late cells of that type with a clonotype.
    store, together = spanning.store, spanning.together
    assert store.clones.load_clones(together).clones["n_cells"].tolist() == [6, 2]

    def measured(clonotypes: ArtifactRef):
        return store.clones.load_fate_bias(_fate_bias_of(spanning, clonotypes))

    bias = measured(together)
    # The late cells: cell1 and cell2 of clonotype_1 and cell4 of clonotype_2
    # are T cells of fates x, x and y, cell6 a B cell of x, cell7 ambiguous of
    # y.
    assert bias.fates[["fate", "n_late_cells"]].values.tolist() == [
        ["x, BCR", 1],
        ["x, TCR", 2],
        ["y, ambiguous", 1],
        ["y, TCR", 1],
    ]
    # cell0 follows clonotype_1's two late T cells, cell3 clonotype_2's late
    # T cell, and cell5 clonotype_1's late B cell.
    assert bias.cells.astype(object).values.tolist() == [
        ["clonotype_1 (TCR)", 2, "x, TCR"],
        ["clonotype_2", 1, "y, TCR"],
        ["clonotype_1 (BCR)", 1, "x, BCR"],
    ]
    assert bias.fractions.values.tolist() == [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
    ]
    # Among the three late T cells with a clonotype, two of fate x:
    # clonotype_1's two late T cells, both of x, have the chance C(2, 2) /
    # C(3, 2) = 1/3, and clonotype_2's one of y has 1/3 too; a clonotype of
    # one late cell holds its type's only late cell, a chance of 1.
    rows = bias.clone_fates.join(bias.group_keys, on="fate")
    assert rows[
        ["clone", "receptor_type", "fate", "n_early_cells", "n_late_cells", "n_cells"]
    ].astype(object).values.tolist() == [
        ["clonotype_1 (TCR)", "TCR", "x, TCR", 1, 2, 2],
        ["clonotype_1 (TCR)", "TCR", "y, TCR", 1, 2, 0],
        ["clonotype_1 (BCR)", "BCR", "x, BCR", 1, 1, 1],
        ["clonotype_2", "TCR", "x, TCR", 1, 1, 0],
        ["clonotype_2", "TCR", "y, TCR", 1, 1, 1],
        ["clonotype_1 (ambiguous)", "ambiguous", "y, ambiguous", 0, 1, 1],
    ]
    np.testing.assert_allclose(rows["p_value"], [1 / 3, 1, 1, 1, 1 / 3, 1], rtol=1e-12)
    # Within each fate of T, Benjamini and Hochberg's adjustment of 1/3 over
    # two clonotypes gives 2/3.
    np.testing.assert_allclose(rows["q_value"], [2 / 3, 1, 1, 1, 2 / 3, 1], rtol=1e-12)
    assert bias.diagnostics == {
        "cellsWithoutClone": 0,
        "cellsWithoutGroup": 0,
        "groups": 4,
        "earlyCellsWithoutLateCells": 0,
        "earlyCellsWithTiedFates": 0,
        "testedClones": 4,
    }
    # Defined within receptor types, every table is the same but for the
    # clonotypes' labels.
    within = measured(spanning.apart)
    for table in ("fates", "fractions", "group_keys"):
        pd.testing.assert_frame_equal(getattr(within, table), getattr(bias, table))
    for table in ("cells", "clone_fates"):
        pd.testing.assert_frame_equal(
            getattr(within, table).drop(columns="clone"),
            getattr(bias, table).drop(columns="clone"),
        )
    assert within.diagnostics == bias.diagnostics


def test_the_fates_plot_draws_a_bar_for_each_clonotype_within_a_type(
    spanning,
) -> None:
    from cytearc.plotting._clone_names import clone_names

    store = spanning.store
    result = store.plots.clone_fate_bias(
        _fate_bias_of(spanning, spanning.together), show=False
    )
    # clonotype_1 is tested in BCR, TCR and the ambiguous type, a bar for
    # each under its clonotype's junction and its type; by dominant fate in
    # the fates' order, they come before clonotype_2.
    table = result.tables["clones"]
    assert table.columns.tolist() == [
        "clone",
        "label",
        "fate",
        "n_early_cells",
        "n_late_cells",
    ]
    assert table.astype(object).values.tolist() == [
        ["clonotype_1 (BCR)", "CASSA (BCR)", "x, BCR", 1, 1],
        ["clonotype_1 (TCR)", "CASSA (TCR)", "x, TCR", 1, 2],
        ["clonotype_1 (ambiguous)", "CASSA (ambiguous)", "y, ambiguous", 0, 1],
        ["clonotype_2", "CASSB", "y, TCR", 1, 1],
    ]
    fates = result.tables["fates"]
    drawn = fates.pivot(index="label", columns="fate", values="n_cells")
    order = ["x, BCR", "x, TCR", "y, ambiguous", "y, TCR"]
    assert drawn.loc[table["label"], order].values.tolist() == [
        [1, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    ticks = [tick.get_text() for tick in result.axes["fates"].get_xticklabels()]
    assert ticks == table["label"].tolist()
    result.close()
    # Defined within receptor types, the same bars under other labels.
    within = store.plots.clone_fate_bias(
        _fate_bias_of(spanning, spanning.apart), show=False
    )
    measures = ["fate", "n_early_cells", "n_late_cells"]
    pd.testing.assert_frame_equal(within.tables["clones"][measures], table[measures])
    assert within.tables["clones"]["label"].tolist() == [
        "CARW",
        "CASSA 1",
        "CASSA 4",
        "CASSB",
    ]
    within.close()
    # A figure labels a clonotype within a type by its clonotype's junction
    # and the text its label adds, and refuses a label of no clonotype.
    names = clone_names(store, spanning.together)
    assert names.labels(["clonotype_1 (TCR)", "clonotype_1 (BCR)", "clonotype_2"]) == [
        "CASSA (TCR)",
        "CASSA (BCR)",
        "CASSB",
    ]
    with pytest.raises(ValueError, match="No clonotype is labelled"):
        names.labels(["clonotype_3 (TCR)"])
    # A text that names no receptor type adds nothing, and clonotypes within
    # types take none, so neither label names a clonotype.
    with pytest.raises(ValueError, match="No clonotype is labelled"):
        names.labels(["clonotype_1 (XYZ 9)"])
    with pytest.raises(ValueError, match="No clonotype is labelled"):
        clone_names(store, spanning.apart).labels(["clonotype_1 (TCR)"])


def test_the_largest_clonotypes_and_their_tracks_draw_each_clonotype_within_a_type(
    spanning,
) -> None:
    store = spanning.store
    # Without each cell's clonotype within its type, clonotype_1 would draw
    # one bar of six cells; its types are clonotypes of three, two and one
    # cells, and clonotype_2's two cells follow clonotype_1's B cells, which
    # rank before them within types.
    found = store.plots.clone_abundance(spanning.together, group_by="tp", show=False)
    expected = store.plots.clone_abundance(spanning.apart, group_by="tp", show=False)
    top = found.tables["top"]
    assert top["label"].drop_duplicates().tolist() == [
        "CASSA (TCR)",
        "CASSA (BCR)",
        "CASSB",
        "CASSA (ambiguous)",
    ]
    assert top.groupby("label", sort=False)["n_cells"].sum().tolist() == [3, 2, 2, 1]
    for name in ("sizes", "top"):
        columns = [
            column for column in found.tables[name] if column not in ("clone", "label")
        ]
        pd.testing.assert_frame_equal(
            found.tables[name][columns], expected.tables[name][columns]
        )
    found.close()
    expected.close()
    # Each clonotype within a type is followed through the groups of its type,
    # as a line of its own, by rank.
    tracked = store.clones.track(spanning.together, group_by="tp", top_n=2)
    assert tracked.tracked == (
        "clonotype_1 (TCR)",
        "clonotype_1 (BCR)",
        "clonotype_2",
        "clonotype_1 (ambiguous)",
    )
    lines = store.plots.clone_tracking(tracked, show=False)
    table = lines.tables["tracking"]
    assert table[["label", "group", "n_cells"]].values.tolist() == [
        ["CASSA (TCR)", "a, TCR", 2],
        ["CASSA (TCR)", "b, TCR", 1],
        ["CASSA (BCR)", "a, BCR", 1],
        ["CASSA (BCR)", "b, BCR", 1],
        ["CASSB", "a, TCR", 1],
        ["CASSB", "b, TCR", 1],
        ["CASSA (ambiguous)", "b, ambiguous", 1],
    ]
    assert len(lines.axes["tracking"].get_lines()) == 4
    lines.close()
    within = store.clones.track(spanning.apart, group_by="tp", top_n=2)
    pd.testing.assert_frame_equal(
        within.table.drop(columns="clone"), tracked.table.drop(columns="clone")
    )
    assert within.diagnostics == tracked.diagnostics
    # A clonotype's own label follows each of its clonotypes within types,
    # and a clonotype within a type's label that one alone. Among the B cells
    # of a, only clonotype_1's B cells have a group of their type.
    store.cells.insert("b_cells_of_a", np.isin(np.arange(N_CELLS), [6]))
    named = store.clones.track(
        spanning.together,
        group_by="tp",
        labels=["clonotype_1"],
        cell_selection=store.snapshot_cell_selection("b_cells_of_a"),
    )
    assert named.tracked == (
        "clonotype_1 (TCR)",
        "clonotype_1 (BCR)",
        "clonotype_1 (ambiguous)",
    )
    assert named.table[["clone", "group", "n_cells"]].values.tolist() == [
        ["clonotype_1 (BCR)", "a, BCR", 1]
    ]
    one = store.clones.track(
        spanning.together, group_by="tp", labels=["clonotype_1 (BCR)", "clonotype_2"]
    )
    assert one.tracked == ("clonotype_1 (BCR)", "clonotype_2")
    # Labels of both kinds together follow each clonotype within a type they
    # name, by rank.
    mixed = store.clones.track(
        spanning.together, group_by="tp", labels=["clonotype_2", "clonotype_1"]
    )
    assert mixed.tracked == (
        "clonotype_1 (TCR)",
        "clonotype_1 (BCR)",
        "clonotype_2",
        "clonotype_1 (ambiguous)",
    )
    with pytest.raises(ValueError, match="No clone is labelled"):
        store.clones.track(spanning.together, labels=["clonotype_3"])


def test_clonotype_counts_of_clonotypes_across_types_count_each_within_a_type(
    spanning,
) -> None:
    store = spanning.store
    # The primary beta chains, TRBV1, of clonotype_1's T cells, of
    # clonotype_2 and of the ambiguous cell7, whose beta chain has more UMIs
    # than its heavy chain, and the heavy chains, IGHV1, of clonotype_1's B
    # cells: three clonotypes within types use TRBV1 and one IGHV1.
    usage = store.repertoire.gene_usage(spanning.together, count="clonotypes")
    assert usage.table[
        ["locus", "v_call", "n_clonotypes", "fraction"]
    ].values.tolist() == [
        ["IGH", "IGHV1", 1, 1.0],
        ["TRB", "TRBV1", 3, 1.0],
    ]
    lengths = store.repertoire.spectratype(spanning.together, count="clonotypes")
    assert lengths.table[["locus", "length", "n_clonotypes"]].values.tolist() == [
        ["IGH", 4, 1],
        ["TRB", 5, 3],
    ]
    # Within receptor types, the clonotypes count alike, and cells count as
    # they do whatever their clonotypes.
    for count in ("clonotypes", "cells"):
        for measure in (store.repertoire.gene_usage, store.repertoire.spectratype):
            pd.testing.assert_frame_equal(
                measure(spanning.together, count=count, group_by="tp").table,
                measure(spanning.apart, count=count, group_by="tp").table,
            )
    cells = store.repertoire.gene_usage(spanning.together)
    assert cells.table[["locus", "v_call", "n_cells"]].values.tolist() == [
        ["IGH", "IGHV1", 2],
        ["TRB", "TRBV1", 6],
    ]


_BRIDGED_CHAINS = {
    "CAVA": ("TRA", "TGTGCTGTGGCC"),
    "CAVB": ("TRA", "TGTGCTGTGTAC"),
    "CASSA": ("TRB", "TGTGCCAGCAGCGCC"),
    "CASSB": ("TRB", "TGTGCCAGCAGCTAC"),
    "CQQY": ("IGK", "TGTCAGCAGTAT"),
    "CARW": ("IGH", "TGTGCGAGATGG"),
}
# T cells of clone A, cell0 and cell1, and of clone B, cell2 and cell3, a B
# cell, cell4, and two doublets, cell5 of a T cell of A and a B cell of
# cell4's clone and cell6 of a T cell of B and a B cell of that clone.
BRIDGED_ROWS = [
    f"cell{cell},true,cell{cell}_contig_{contig},true,500,{chain},{chain}V1,,"
    f"{chain}J1,{chain}C,true,true,{junction_aa},{junction},50,{umis},ct{cell},"
    f"ct{cell}_{contig}"
    for cell, chains in (
        (0, (("CAVA", 9), ("CASSA", 9))),
        (1, (("CAVA", 9), ("CASSA", 9))),
        (2, (("CAVB", 9), ("CASSB", 9))),
        (3, (("CAVB", 9), ("CASSB", 9))),
        (4, (("CQQY", 9), ("CARW", 9))),
        (5, (("CAVA", 8), ("CASSA", 8), ("CQQY", 5), ("CARW", 5))),
        (6, (("CAVB", 8), ("CASSB", 8), ("CQQY", 5), ("CARW", 5))),
    )
    for contig, (junction_aa, umis) in enumerate(chains, start=1)
    for chain, junction in [_BRIDGED_CHAINS[junction_aa]]
]


def test_doublets_that_share_a_b_cell_clone_leave_t_cell_clonotypes_apart(
    tmp_path,
) -> None:
    # Across receptor types, the doublets share an arm's chains with A, with
    # B and with the B cell, which joins all seven cells into one clonotype;
    # every summary measures each cell's clonotype within its type, so A and
    # B stay two clonotypes of T cells, as within types.
    store, receptors = clonal_receptors(tmp_path, BRIDGED_ROWS)
    apart = store.repertoire.clonotypes(receptors)
    together = store.repertoire.clonotypes(receptors, within_receptor_type=False)
    assert store.clones.load_clones(apart).clones["n_cells"].tolist() == [2, 2, 2, 1]
    loaded = store.clones.load_clones(together)
    assert loaded.clones["n_cells"].tolist() == [7]
    # clonotype_1's T cells split into two clonotypes of T cells, numbered by
    # rank, and its B cell and doublets are one each.
    labels = loaded.cells["within_type_clone"]
    assert labels.astype(object).where(labels.notna(), None).tolist() == [
        *["clonotype_1 (TCR 1)"] * 2,
        *["clonotype_1 (TCR 2)"] * 2,
        "clonotype_1 (BCR)",
        *["clonotype_1 (ambiguous)"] * 2,
        None,
    ]
    within = store.clones.diversity(apart).indices
    across = store.clones.diversity(together).indices
    pd.testing.assert_frame_equal(across, within)
    assert across.set_index("group").loc["TCR", ["n_cells", "n_clones"]].tolist() == [
        4,
        2,
    ]
    # The T cells' and doublets' primary beta chains, TRBV1, count as three
    # clonotypes either way, A, B and the doublets.
    for clonotypes in (apart, together):
        usage = store.repertoire.gene_usage(clonotypes, count="clonotypes").table
        assert usage.loc[usage["locus"] == "TRB", "n_clonotypes"].tolist() == [3]
    abundance = store.plots.clone_abundance(together, show=False)
    assert abundance.tables["top"]["label"].drop_duplicates().tolist() == [
        "CASSA (TCR 1)",
        "CASSA (TCR 2)",
        "CASSA (ambiguous)",
        "CASSA (BCR)",
    ]
    abundance.close()


def test_measuring_clonotypes_across_receptor_types_charges_each_step(
    spanning,
) -> None:
    from cytearc.utils.memory_budget import NUMPY_STEP_BYTES
    from tests.memory_helpers import charges

    store, together = spanning.store, spanning.together

    def charged(call) -> dict[str, int]:
        _result, taken = charges(call)
        return {item: nbytes for item, nbytes, _planned in taken}

    # Defining clonotypes across types holds their assignment across types,
    # each array's eight cells and their clonotypes and configurations, while
    # each cell's clonotype within its type is found, which it keeps.
    receptors = store.artifacts.inspect(together).input_ref("receptors")
    found = charged(
        lambda: store.repertoire.clonotypes(
            receptors,
            receptor_arms="VDJ",
            within_receptor_type=False,
            invalidate_cache=True,
        )
    )
    assert found["each cell's clonotype within its receptor type"] == 8 * 8
    assert "the clonotypes across receptor types" in found
    # Each cell's group among the within groups alone sits beside its group
    # among the types too, and the save writes each cell's clonotype within
    # its type.
    assert found["each cell's group"] == 16 * 8
    assert found[
        "writing the clonotypes"
    ] == repertoire_operations._clonotype_write_bytes(8, (), [], within_types=True)
    within = charged(
        lambda: store.repertoire.clonotypes(
            receptors, receptor_arms="VDJ", invalidate_cache=True
        )
    )
    assert within["each cell's group"] == 8 * 8
    assert within[
        "writing the clonotypes"
    ] == repertoire_operations._clonotype_write_bytes(8, (), [])
    # Tracking names its four clonotypes within types from the eight cells;
    # a clonotype's own label is found among its two clonotypes' labels.
    found = charged(
        lambda: store.clones.track(together, group_by="tp", labels=["clonotype_1"])
    )
    assert found["finding the clonotype and type of 4 clonotypes within types"] == (
        30 * 8 + 20 * 4 + NUMPY_STEP_BYTES
    )
    assert found["finding which of 4 clonotypes within types share a clonotype"] == (
        48 * 4 + NUMPY_STEP_BYTES
    )
    assert found[
        "numbering 3 clonotypes within types of clonotypes that span types"
    ] == (24 * 4 + 240 * 3 + NUMPY_STEP_BYTES)
    assert "finding the clonotypes named" in found
    # Choosing and finding the clones followed hold 40 bytes a count, of its
    # seven counts of a clonotype within a type in a group.
    assert found["choosing the clones to follow among 7 counts"] == (
        40 * 7 + NUMPY_STEP_BYTES
    )
    assert found["finding the counts of 3 clones followed"] == (
        40 * 7 + NUMPY_STEP_BYTES
    )
    # Named among the B cells of a, two of clonotype_1's three clonotypes
    # within types have no counted cell, whose first cells are found among
    # the eight cells' clones.
    store.cells.insert("b_cell_of_a", np.isin(np.arange(N_CELLS), [6]))
    found = charged(
        lambda: store.clones.track(
            together,
            group_by="tp",
            labels=["clonotype_1"],
            cell_selection=store.snapshot_cell_selection("b_cell_of_a"),
        )
    )
    assert found[
        "finding the first cell of each clone named without a counted cell"
    ] == (20 * 8 + NUMPY_STEP_BYTES)


def test_summaries_of_clonotypes_follow_a_cell_selection(tmp_path) -> None:
    store, receptors = clonal_receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    # cell0, cell3, cell5 and cell7.
    store.cells.insert("chosen", np.isin(np.arange(N_CELLS), [0, 3, 5, 7]))
    chosen = store.snapshot_cell_selection("chosen")
    expansion = store.clones.load_expansion(
        store.clones.expansion(ref, cell_selection=chosen)
    )
    assert expansion.cells.index.tolist() == ["cell0", "cell3", "cell5", "cell7"]
    assert expansion.cells["group_clone_size"].tolist() == [1, 1, 1, 0]
    groups = expansion.cells["group"].astype(object)
    assert groups.where(groups.notna(), "-").tolist() == ["TCR", "TCR", "BCR", "-"]
    diversity = store.clones.diversity(ref, cell_selection=chosen)
    indices = diversity.indices.join(diversity.group_keys, on="group")
    assert indices[["receptor_type", "n_cells", "n_clones"]].values.tolist() == [
        ["BCR", 1, 1],
        ["TCR", 2, 2],
    ]


def test_group_by_may_not_name_the_receptor_type_again(tmp_path) -> None:
    store, receptors = clonal_receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    store.cells.insert("receptor_type", np.asarray(["x"] * N_CELLS))
    with pytest.raises(ValueError, match="'receptor_type', whose labels the artifact"):
        store.clones.diversity(ref, group_by="receptor_type")
    with pytest.raises(ValueError, match="'receptor_type', whose labels the artifact"):
        plotting.clone_abundance(store, ref, group_by="receptor_type", show=False)
    # Among several columns too.
    store.cells.insert("cell_kind", np.asarray(["x"] * N_CELLS))
    with pytest.raises(ValueError, match="'receptor_type', whose labels the artifact"):
        plotting.clone_abundance(
            store, ref, group_by=["cell_kind", "receptor_type"], show=False
        )
    with pytest.raises(ValueError, match="clones must be a clones or clonotypes"):
        store.clones.diversity(receptors)


def _assert_stored_widths(store, receptors, ref) -> None:
    receptor_group = artifact_group(store.zw, receptors)
    assert receptor_group["values"].dtype == np.dtype("<U5")
    assert receptor_group["contig_junction"].dtype == np.int32
    assert receptor_group["contig_junction"].shape == (14,)
    group = artifact_group(store.zw, ref)
    assert group["codes"].dtype == np.int32
    assert group["clonotype_ids"].dtype == np.dtype("<U11")


def test_defining_clonotypes_refuses_a_small_budget_before_it_saves(
    tmp_path, monkeypatch
) -> None:
    store, receptors = clonal_receptors(tmp_path)
    monkeypatch.setattr(store, "memoryBytes", 1_000)
    with pytest.raises(
        MemoryError, match="^Defining the clonotypes of 8 cells holds about"
    ):
        store.repertoire.clonotypes(receptors)
    assert store.artifacts.list(kind="clonotypes", scope="datastore") == []
    monkeypatch.undo()
    _assert_stored_widths(store, receptors, store.repertoire.clonotypes(receptors))


def test_grouping_by_within_columns_is_checked_against_the_memory_budget(
    tmp_path, monkeypatch
) -> None:
    store, receptors = clonal_receptors(tmp_path)
    store.cells.insert("donor", np.asarray(["d1", "d2"] * 4))
    # The donors in a chunk of 100,000 rows, as a store written before cell
    # columns were chunked by their rows keeps them: grouping the cells by
    # donor reads that chunk, which 300 kB cannot hold.
    rechunk_column(store.cells._group, "donor")
    store = receptor_store(tmp_path)
    monkeypatch.setattr(store, "memoryBytes", 300_000)
    with pytest.raises(
        MemoryError,
        match="Defining the clonotypes of 8 cells holds about .* for the labels of "
        "'donor'",
    ):
        store.repertoire.clonotypes(receptors, within="donor")
    assert store.artifacts.list(kind="clonotypes", scope="datastore") == []


def test_reading_clonotypes_as_clones_keeps_their_receptor_types_apart(
    tmp_path,
) -> None:
    store, receptors = clonal_receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    _assert_stored_widths(store, receptors, ref)
    source = store._read_clone_input(
        ref, MemoryLedger(None, "Reading clonotypes"), with_ids=True
    )
    assert source.clone_ids is not None
    assert source.clone_ids.tolist() == [f"clonotype_{rank}" for rank in range(1, 5)]
    ((name, labels),) = source.kept_apart
    assert name == "receptor_type"
    assert labels.tolist() == ["TCR"] * 5 + ["BCR"] * 2 + ["no IR"]


def test_loaded_expansion_of_other_cells_matches_their_receptor_types(
    tmp_path,
) -> None:
    store, receptors = clonal_receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    store.cells.insert("first_half", np.arange(N_CELLS) < 4)
    measured = store.snapshot_cell_selection("first_half")
    saved = store.clones.expansion(ref, cell_selection=measured)
    # The four measured cells are T cells of one group.
    assert store.clones.load_expansion(saved).cells.shape[0] == 4


def test_loaded_clonotypes_are_checked_against_the_memory_budget(
    tmp_path, monkeypatch
) -> None:
    store, receptors = clonal_receptors(tmp_path)
    store.cells.insert("donor", np.asarray(["d1", "d2"] * 4))
    receptor_group = artifact_group(store.zw, receptors)
    # The four representative fields store these values and packed bytes.
    values, packed = [5, 6, 6, 5], [21, 75, 30, 25]
    assert [
        receptor_group[f"contig_{name}_offsets"].shape[0] - 1 for name in FIELDS
    ] == values
    assert [receptor_group[f"contig_{name}_values"].nbytes for name in FIELDS] == (
        packed
    )
    # Each field takes 25 bytes for each value it stores, 64 for each that a
    # representative holds, here every one, and 7 for each byte of its packed
    # text; one batch of decoding takes 160 bytes for each value of the widest.
    fields = sum(89 * n + 7 * size for n, size in zip(values, packed, strict=True))
    assert fields + 160 * 6 == 3_015 + 960
    for within, n_clonotypes in (((), 4), (("donor",), 6)):
        ref = store.repertoire.clonotypes(receptors, within=list(within))
        assert artifact_group(store.zw, ref)["clonotype_ids"].shape == (n_clonotypes,)
        # The tables take 300 bytes per cell and 700 per clonotype, beside the
        # cells' IDs as text, the fields, and the within groups' labels.
        needed = (
            8 * 300 + 700 * n_clonotypes + ids_read_bytes(store, 8) + fields + 160 * 6
        )
        if within:
            labels = artifact_group(store.zw, ref)["within_labels"]
            assert labels.dtype == np.dtype("<U2") and labels.shape == (2,)
            needed += text_read_bytes(labels, 2)
        monkeypatch.setattr(store, "memoryBytes", needed - 1)
        with pytest.raises(
            MemoryError, match=f"clonotypes of 8 cells holds about {needed} bytes"
        ):
            store.repertoire.load_clonotypes(ref)
        monkeypatch.setattr(store, "memoryBytes", needed)
        assert len(store.repertoire.load_clonotypes(ref).clonotypes) == n_clonotypes
        monkeypatch.undo()


def test_loaded_chains_do_not_depend_on_the_decoding_batch(
    tmp_path, monkeypatch
) -> None:
    store, receptors = clonal_receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    whole = store.repertoire.load_clonotypes(ref).clonotypes
    assert whole.set_index("clonotype").loc["clonotype_1", "VDJ_1_junction_aa"] == (
        "CASSA"
    )
    for batch in (1, 2, 3):
        monkeypatch.setattr(repertoire_operations, "_DECODE_BATCH_VALUES", batch)
        pd.testing.assert_frame_equal(
            store.repertoire.load_clonotypes(ref).clonotypes, whole
        )


def test_every_chain_column_holds_text(tmp_path) -> None:
    store, receptors = clonal_receptors(tmp_path)
    table = store.repertoire.load_clonotypes(
        store.repertoire.clonotypes(receptors)
    ).clonotypes
    columns = [f"{slot}_{name}" for slot in SLOTS for name in FIELDS]
    assert all(isinstance(table[column].dtype, pd.StringDtype) for column in columns)
    # No representative has a second chain of an arm, and the table holds
    # those slots as missing text all the same.
    assert table[[f"VJ_2_{name}" for name in FIELDS]].isna().all().all()
    assert table[[f"VDJ_2_{name}" for name in FIELDS]].isna().all().all()


def test_clonotype_numbers_do_not_follow_the_order_of_the_cells(tmp_path) -> None:
    # A T cell and a B cell clonotype of two cells each tie on size: BCR sorts
    # before TCR, so the B cells' clonotype is clonotype_1 in either store.
    rows = [
        "t1,true,t1_contig_1,true,500,TRA,TRAV1,,TRAJ1,TRAC,true,true,CAVA,TGTGCTGTGGCC,90,9,ct1,ct1_1",
        "t1,true,t1_contig_2,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,CASSA,TGTGCCAGCAGCGCC,80,8,ct1,ct1_2",
        "t2,true,t2_contig_1,true,500,TRA,TRAV1,,TRAJ1,TRAC,true,true,CAVA,TGTGCTGTGGCC,70,7,ct1,ct1_1",
        "t2,true,t2_contig_2,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,CASSA,TGTGCCAGCAGCGCC,60,6,ct1,ct1_2",
        "b1,true,b1_contig_1,true,500,IGH,IGHV1,,IGHJ1,IGHM,true,true,CARW,TGTGCGAGATGG,500,20,cb1,cb1_1",
        "b1,true,b1_contig_2,true,500,IGK,IGKV1,,IGKJ1,IGKC,true,true,CQQY,TGTCAGCAGTAT,400,15,cb1,cb1_2",
        "b2,true,b2_contig_1,true,500,IGH,IGHV1,,IGHJ1,IGHM,true,true,CARW,TGTGCGAGATGG,300,12,cb1,cb1_1",
        "b2,true,b2_contig_2,true,500,IGK,IGKV1,,IGKJ1,IGKC,true,true,CQQY,TGTCAGCAGTAT,200,10,cb1,cb1_2",
    ]
    found = []
    for order in (["t1", "t2", "b1", "b2"], ["b2", "b1", "t2", "t1"]):
        folder = tmp_path / "_".join(order)
        folder.mkdir()
        path = folder / "store.zarr"
        counts = np.ones((len(order), 2), dtype=np.uint16)
        write_count_store(str(path), {"RNA": counts}, "uint16", cell_ids=order)
        store = open_store(path)
        receptors = store.repertoire.import_receptors(
            write_lines(folder, "contigs.csv", CELLRANGER_HEADER, rows),
            cell_selection=store.snapshot_cell_selection(),
        )
        loaded = store.repertoire.load_clonotypes(
            store.repertoire.clonotypes(receptors)
        )
        found.append(loaded.cells["clonotype"].astype(str).to_dict())
    assert (
        found[0]
        == found[1]
        == {
            "b1": "clonotype_1",
            "b2": "clonotype_1",
            "t1": "clonotype_2",
            "t2": "clonotype_2",
        }
    )


def test_receptor_types_and_within_labels_never_share_a_group(tmp_path) -> None:
    # cell0, a T cell of donor d2, and cell1, a cell of donor d1 with T cell
    # and B cell chains, share a beta chain; comparing beta chains alone, their
    # receptor types and their donors each keep them apart.
    rows = [
        "cell0,true,cell0_contig_1,true,500,TRA,TRAV1,,TRAJ1,TRAC,true,true,CAVA,TGTGCTGTGGCC,90,9,ct1,ct1_1",
        "cell0,true,cell0_contig_2,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,CASSA,TGTGCCAGCAGCGCC,80,8,ct1,ct1_2",
        "cell1,true,cell1_contig_1,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,CASSA,TGTGCCAGCAGCGCC,80,8,ct2,ct2_1",
        "cell1,true,cell1_contig_2,true,500,IGK,IGKV1,,IGKJ1,IGKC,true,true,CQQY,TGTCAGCAGTAT,400,15,ct2,ct2_2",
    ]
    store, receptors = clonal_receptors(tmp_path, rows)
    store.cells.insert("donor", np.asarray(["d2"] + ["d1"] * (N_CELLS - 1)))
    types = store.artifacts.load_values(receptors).values
    assert list(types[:2]) == ["TCR", "ambiguous"]
    for options in (
        {},
        {"within": "donor"},
        {"within": "donor", "within_receptor_type": False},
    ):
        loaded = store.repertoire.load_clonotypes(
            store.repertoire.clonotypes(receptors, receptor_arms="VDJ", **options)
        )
        clonotype = loaded.cells["clonotype"]
        assert clonotype["cell0"] != clonotype["cell1"], options
    # Without either, the shared beta chain links them.
    together = store.repertoire.load_clonotypes(
        store.repertoire.clonotypes(
            receptors, receptor_arms="VDJ", within_receptor_type=False
        )
    )
    assert (
        together.cells.loc["cell0", "clonotype"]
        == (together.cells.loc["cell1", "clonotype"])
    )


def test_within_labels_are_an_array_that_opening_the_artifact_never_reads(
    tmp_path,
) -> None:
    store, receptors = clonal_receptors(tmp_path)
    store.cells.insert(
        "donor", np.asarray(["d1", "d2", "d1", "d1", "d1", "d1", "d1", "d1"])
    )
    store.cells.insert("site", np.asarray(["s10", "s2"] * 4))
    ref = store.repertoire.clonotypes(receptors, within=["donor", "site"])
    group = artifact_group(store.zw, ref)
    assert "within_labels" not in group.attrs
    # The cells with a receptor make three groups, in the natural order of
    # their labels, each stored as its donor and its site.
    assert group["within_labels"][:].tolist() == ["d1", "s2", "d1", "s10", "d2", "s2"]
    loaded = store.repertoire.load_clonotypes(ref)
    assert loaded.group_keys.values.tolist() == [
        ["d1", "s2"],
        ["d1", "s10"],
        ["d2", "s2"],
    ]
    assert loaded.group_keys.index.tolist() == ["d1, s2", "d1, s10", "d2, s2"]
    first = loaded.cells.loc["cell0", "clonotype"]
    assert _members(loaded)[first] == ["cell0", "cell4"]
    labelled = loaded.clonotypes.join(loaded.group_keys, on="group")
    row = labelled.set_index("clonotype").loc[first]
    assert row[["group", "donor", "site"]].tolist() == ["d1, s10", "d1", "s10"]
    cell1 = labelled.set_index("clonotype").loc[loaded.cells.loc["cell1", "clonotype"]]
    assert cell1[["group", "donor", "site"]].tolist() == ["d2, s2", "d2", "s2"]
    # Clonotypes without within columns store no labels.
    assert "within_labels" not in artifact_group(
        store.zw, store.repertoire.clonotypes(receptors)
    )


def test_renamed_within_labels_give_other_clonotypes(tmp_path) -> None:
    store, receptors = clonal_receptors(tmp_path)
    donors = np.asarray(["d1", "d2", "d1", "d1", "d1", "d1", "d1", "d1"])
    store.cells.insert("donor", donors)
    first = store.repertoire.clonotypes(receptors, within="donor")
    # The same groups under other names give other clonotypes, and the same
    # labels again reuse the saved ones.
    store.cells.insert("donor", np.where(donors == "d1", "x1", "x2"), overwrite=True)
    renamed = store.repertoire.clonotypes(receptors, within="donor")
    assert renamed != first
    loaded = store.repertoire.load_clonotypes(renamed)
    assert set(loaded.clonotypes["group"]) == {"x1", "x2"}
    assert loaded.group_keys["donor"].tolist() == ["x1", "x2"]
    store.cells.insert("donor", donors.copy(), overwrite=True)
    assert store.repertoire.clonotypes(receptors, within="donor") == first


def test_within_columns_may_take_the_names_of_the_tables_columns(tmp_path) -> None:
    store, receptors = clonal_receptors(tmp_path)
    donors = np.asarray(["d1", "d2", "d1", "d1", "d1", "d1", "d1", "d1"])
    store.cells.insert("donor", donors)
    neutral = store.repertoire.load_clonotypes(
        store.repertoire.clonotypes(receptors, within="donor")
    )
    # Columns named like the tables' own columns hold the donors' labels, and
    # keep the donors' clonotypes apart as "donor" does; their labels stay in
    # the keys.
    for column in (
        "clonotype",
        "clone_size",
        "n_cells",
        "fraction",
        "n_configurations",
        "receptor_type",
        "group",
        "VDJ_1_junction_aa",
        "VJ_2_j_call",
    ):
        store.cells.insert(column, donors, overwrite=True)
        loaded = store.repertoire.load_clonotypes(
            store.repertoire.clonotypes(receptors, within=column)
        )
        pd.testing.assert_frame_equal(loaded.cells, neutral.cells)
        pd.testing.assert_frame_equal(loaded.clonotypes, neutral.clonotypes)
        assert loaded.group_keys.to_dict("index") == {
            donor: {column: donor} for donor in ("d1", "d2")
        }
        joined = loaded.clonotypes.join(loaded.group_keys, on="group", rsuffix="_key")
        label = f"{column}_key" if column in loaded.clonotypes.columns else column
        assert joined[label].tolist() == joined["group"].tolist()


def test_regrouped_expansion_plot_keeps_receptor_types_apart(tmp_path) -> None:
    store, receptors = clonal_receptors(tmp_path)
    store.cells.insert(
        "sample", np.asarray(["s1", "s2", "s1", "s2", "s1", "s1", "s2", "s1"])
    )
    store.cells.insert("tissue", np.asarray(["blood"] * N_CELLS))
    expansion = store.clones.expansion(
        store.repertoire.clonotypes(receptors), group_by="sample"
    )
    result = store.plots.clonal_expansion(expansion, group_by="tissue", show=False)
    # The five T cells, in three clonotypes, and the two B cells, in one, are
    # ranked apart within the tissue.
    windows = result.tables["rank_windows"]
    held = windows[windows["n_cells"] > 0]
    assert held[["group", "window", "n_cells"]].values.tolist() == [
        ["blood, BCR", "1 to 10", 2],
        ["blood, TCR", "1 to 10", 5],
    ]
    totals = result.tables["categories"].groupby("group")["n_cells"].sum()
    assert totals.to_dict() == {"blood, BCR": 2, "blood, TCR": 5}
    result.close()
    # An empty grouping draws the expansion's own groups, as no grouping does.
    own = store.plots.clonal_expansion(expansion, show=False)
    empty = store.plots.clonal_expansion(expansion, group_by=[], show=False)
    for name, table in own.tables.items():
        pd.testing.assert_frame_equal(empty.tables[name], table)
    assert empty.provenance.extras["grouping"] == "sample, receptor_type"
    own.close()
    empty.close()
    store.cells.insert("receptor_type", np.asarray(["x"] * N_CELLS))
    with pytest.raises(ValueError, match="'receptor_type', whose labels the artifact"):
        store.plots.clonal_expansion(expansion, group_by="receptor_type", show=False)


def test_each_step_that_reads_counts_the_stores_codec_threads(
    tmp_path, monkeypatch
) -> None:
    # A finished read can leave a chunk on each of the operation's codec
    # threads, which every step that reads stored arrays counts from the
    # store's nthreads, whatever the machine.
    store, receptors = clonal_receptors(tmp_path)
    clonotypes = store.repertoire.clonotypes(receptors)
    monkeypatch.setattr(store, "nthreads", 3)
    made: list[tuple[str, int]] = []
    initialize = MemoryLedger.__init__

    def recording(self, *arguments, **options) -> None:
        initialize(self, *arguments, **options)
        made.append((self.doing, self.codec_threads))

    monkeypatch.setattr(MemoryLedger, "__init__", recording)
    store.repertoire.import_receptors(
        tmp_path / "clonal.csv",
        cell_selection=store.snapshot_cell_selection(),
        invalidate_cache=True,
    )
    store.repertoire.clonotypes(receptors, invalidate_cache=True)
    store.repertoire.clonotypes(
        receptors, metric="hamming", max_distance=1, invalidate_cache=True
    )
    store.repertoire.gene_usage(clonotypes)
    store.repertoire.spectratype(receptors)
    store.repertoire.export_airr(clonotypes, tmp_path / "contigs.tsv")
    store._clone_figure_labels(clonotypes, ["clonotype_1"])
    reading = [threads for doing, threads in made if not doing.startswith("Loading")]
    assert len(reading) >= 7 and set(reading) == {3}


def test_clonotype_clusters_need_room_for_their_search(tmp_path, monkeypatch) -> None:
    store, receptors = clonal_receptors(tmp_path)
    # 100 MB defines the clonotypes of 8 cells, but leaves no room for the
    # search's reservation of 512 MiB.
    monkeypatch.setattr(store, "memoryBytes", 100_000_000)
    assert store.repertoire.clonotypes(receptors).kind == "clonotypes"
    with pytest.raises(
        MemoryBudgetError, match="bytes for the search for near junctions"
    ):
        store.repertoire.clonotypes(receptors, metric="hamming", max_distance=1)


def test_clonotype_clusters_within_three_substitutions_join_their_cells(
    tmp_path,
) -> None:
    store, receptors = clonal_receptors(tmp_path)
    # Within three substitutions, cell0's alpha junction pairs with cell2's
    # (3 substitutions) and with cell4's second one (2).
    ref = store.repertoire.clonotypes(receptors, metric="hamming", max_distance=3)
    assert _members(store.repertoire.load_clonotypes(ref))["clonotype_1"] == [
        "cell0",
        "cell1",
        "cell2",
        "cell4",
    ]


def test_junctions_beyond_ascii_decode_as_their_text(tmp_path) -> None:
    junctions = ["CASSŁ", "CAS", "CÄT", "C\U0001d538"]
    packed = [value.encode() for value in junctions]
    offsets = np.cumsum([0, *map(len, packed)])
    group = zarr.open_group(tmp_path / "receptors.zarr", mode="w")
    group.create_array("contig_junction_offsets", data=offsets.astype(np.int64))
    group.create_array(
        "contig_junction_values", data=np.frombuffer(b"".join(packed), np.uint8)
    )
    text = repertoire_operations._decoded_values(
        MemoryLedger(None, "Decoding"), group, "junction", np.arange(len(junctions))
    )
    assert text.tolist() == junctions


def test_clonotype_clusters_link_similar_junctions(tmp_path) -> None:
    store, receptors = clonal_receptors(tmp_path)
    # cell2's alpha junction differs from cell0's in 3 of 12 nucleotides.
    identical = store.repertoire.clonotypes(receptors)
    near = store.repertoire.clonotypes(receptors, metric="hamming", max_distance=3)
    assert near != identical
    loaded = store.repertoire.load_clonotypes(near)
    assert loaded.metric == "hamming" and loaded.max_distance == 3
    assert _members(loaded)["clonotype_1"] == ["cell0", "cell1", "cell2", "cell4"]
    status = store.artifacts.inspect(near)
    assert (status.parameters["metric"], status.parameters["max_distance"]) == (
        "hamming",
        3,
    )
    two = store.repertoire.load_clonotypes(
        store.repertoire.clonotypes(receptors, metric="hamming", max_distance=2)
    )
    assert _members(two)["clonotype_1"] == ["cell0", "cell1", "cell4"]
    # 3 of 12 is 25 percent.
    for cutoff, joined in ((25, True), (24, False)):
        result = store.repertoire.load_clonotypes(
            store.repertoire.clonotypes(
                receptors, metric="normalized_hamming", max_distance=cutoff
            )
        )
        assert ("cell2" in _members(result)["clonotype_1"]) is joined
    edits = store.repertoire.load_clonotypes(
        store.repertoire.clonotypes(receptors, metric="levenshtein", max_distance=3)
    )
    assert _members(edits)["clonotype_1"] == ["cell0", "cell1", "cell2", "cell4"]
    # Residues: cell2's alpha chain is identical, so amino-acid clusters
    # equal amino-acid clonotypes.
    residues = store.repertoire.load_clonotypes(
        store.repertoire.clonotypes(
            receptors, sequence="aa", metric="levenshtein", max_distance=1
        )
    )
    assert _members(residues)["clonotype_1"] == ["cell0", "cell1", "cell2", "cell4"]


def test_clusters_ignore_the_junctions_of_contigs_that_are_not_chains(
    tmp_path,
) -> None:
    store, receptors = clonal_receptors(tmp_path)
    clusters = store.repertoire.clonotypes(receptors, metric="hamming", max_distance=3)
    expected = _members(store.repertoire.load_clonotypes(clusters))
    # Non-productive alpha contigs of cell3 and of cell7, which has no chain,
    # one substitution from cell0's alpha chain: the import keeps them, but no
    # slot holds them, so they link no cells.
    rows = CLONAL_ROWS + [
        "cell3,true,cell3_contig_9,true,500,TRA,TRAV1,,TRAJ1,TRAC,true,false,CAVA,TGTGCTGTGGCA,5,1,ct3,ct3_9",
        "cell7,true,cell7_contig_9,true,500,TRA,TRAV1,,TRAJ1,TRAC,true,false,CAVA,TGTGCTGTGGCC,5,1,ct7,ct7_9",
    ]
    other = tmp_path / "other"
    other.mkdir()
    shutil.copytree(tmp_path / "store.zarr", other / "store.zarr")
    store, receptors = clonal_receptors(other, rows)
    assert len(store.repertoire.load_receptors(receptors).chains) == 16
    near = store.repertoire.clonotypes(receptors, metric="hamming", max_distance=3)
    assert _members(store.repertoire.load_clonotypes(near)) == expected


def test_decoded_values_read_only_the_codes_given(tmp_path, monkeypatch) -> None:
    store, receptors = clonal_receptors(tmp_path)
    group = artifact_group(store.zw, receptors)
    every = stored_text(group, "contig", "junction").categories
    chunked: list[str] = []
    rows = repertoire_operations.read_rows

    def counted(ledger, array, wanted, *, item):
        chunked.append(str(array.path).rsplit("/", 1)[-1])
        return rows(ledger, array, wanted, item=item)

    monkeypatch.setattr(repertoire_operations, "read_rows", counted)
    row_bytes = repertoire_operations.row_read_bytes
    # The bounds of so few junctions are read whole, which holds less than a
    # read of their rows; a read of rows as cheap as a byte, as a few codes
    # of a large store are, reads them a chunk at a time instead.
    for cheap_rows in (False, True):
        if cheap_rows:
            monkeypatch.setattr(
                repertoire_operations, "row_read_bytes", lambda geometry, wanted: 1
            )
        for codes in (np.asarray([1]), np.asarray([0, 2]), np.arange(len(every))):
            chunked.clear()
            ledger = MemoryLedger(1 << 30, "Reading junctions")
            ledger.hold("what the caller holds", 7)
            decoded = repertoire_operations._decoded_values(
                ledger, group, "junction", codes
            )
            assert decoded.tolist() == every[codes].tolist()
            assert chunked == (["contig_junction_offsets"] if cheap_rows else [])
            # The decoded junctions stay charged as the strings they are, 41
            # bytes and a byte a base each with an object's slot, and their
            # bounds are let go.
            kept = sum(8 + 41 + len(value) for value in every[codes].tolist())
            assert ledger.charged == 7 + kept
    monkeypatch.setattr(repertoire_operations, "row_read_bytes", row_bytes)
    with pytest.raises(
        MemoryBudgetError, match=r"Reading junctions holds about \d+ bytes"
    ):
        repertoire_operations._decoded_values(
            MemoryLedger(64, "Reading junctions"), group, "junction", np.asarray([1])
        )


def test_the_log_names_clonotype_clusters_as_such(tmp_path) -> None:
    store, receptors = clonal_receptors(tmp_path)
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="INFO"
    )
    try:
        store.repertoire.clonotypes(receptors)
        store.repertoire.clonotypes(receptors, metric="hamming", max_distance=3)
    finally:
        logger.remove(sink)
    # Of the seven cells with a receptor, cell0, cell1 and cell4 share their
    # chains, as do the B cells, and cell2 and cell3 stand alone; within
    # three substitutions cell2's alpha chain joins it to cell0.
    assert [message for message in messages if message.startswith("Defined")] == [
        "Defined 4 clonotypes of 7 of 8 cells; the largest holds 3 cells",
        "Defined 3 clonotype clusters of 7 of 8 cells; the largest holds 4 cells",
    ]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"metric": "tcrdist"}, ValueError, "metric must be one of"),
        (
            {"metric": "identity", "max_distance": 1},
            ValueError,
            "takes no max_distance",
        ),
        (
            {"metric": "identity", "max_distance": False},
            ValueError,
            "takes no max_distance",
        ),
        (
            {"metric": "identity", "max_distance": 0.0},
            ValueError,
            "takes no max_distance",
        ),
        (
            {"metric": "identity", "max_distance": np.asarray([0])},
            ValueError,
            "takes no max_distance",
        ),
        ({"metric": "hamming"}, ValueError, "needs a max_distance"),
        (
            {"metric": "hamming", "max_distance": 1.5},
            TypeError,
            "^max_distance must be an integer",
        ),
        (
            {"metric": "hamming", "max_distance": 1.0},
            TypeError,
            "^max_distance must be an integer",
        ),
        (
            {"metric": "levenshtein", "max_distance": True},
            TypeError,
            "must be an integer",
        ),
        (
            {"metric": "hamming", "max_distance": -1},
            ValueError,
            "^max_distance must be at least 0",
        ),
        (
            {"metric": "normalized_hamming", "max_distance": 101},
            ValueError,
            "at most 100",
        ),
    ],
)
def test_clonotype_cluster_options_are_checked(
    tmp_path, kwargs, error, message
) -> None:
    store, receptors = clonal_receptors(tmp_path)
    with pytest.raises(error, match=message):
        store.repertoire.clonotypes(receptors, **kwargs)


def test_clonotypes_that_an_earlier_version_saved_are_refused(tmp_path) -> None:
    store, receptors = clonal_receptors(tmp_path)
    saved = store.repertoire.clonotypes(receptors, metric="hamming", max_distance=1)
    # An earlier version recorded the distance as cutoff.
    group = artifact_group(store.zw, saved)
    provenance = dict(group.attrs["provenance"])
    parameters = dict(provenance["parameters"])
    parameters["cutoff"] = parameters.pop("max_distance")
    group.attrs["provenance"] = {**provenance, "parameters": parameters}
    refusal = (
        "^clonotypes was saved by an earlier version of CyteArc, whose parameters "
        "this version names otherwise; define the clonotypes again with "
        "ds.repertoire.clonotypes"
    )
    with pytest.raises(ValueError, match=refusal):
        store.repertoire.load_clonotypes(saved)
    with pytest.raises(ValueError, match=refusal):
        store.repertoire.gene_usage(saved)
    with pytest.raises(ValueError, match=refusal):
        store.repertoire.export_airr(saved, tmp_path / "airr.tsv")
    # The call that the refusal names defines them anew.
    again = store.repertoire.clonotypes(receptors, metric="hamming", max_distance=1)
    assert again != saved
    assert store.repertoire.load_clonotypes(again).max_distance == 1


def test_the_bounds_of_max_distance_are_accepted(tmp_path) -> None:
    store, receptors = clonal_receptors(tmp_path)
    # Identity records a max_distance of 0, so 0 is the same request.
    identity = store.repertoire.clonotypes(receptors)
    assert store.repertoire.clonotypes(receptors, max_distance=0) == identity
    assert store.repertoire.clonotypes(receptors, max_distance=np.int64(0)) == identity
    # Within 100 percent, every chain is near every chain of its length: the
    # alpha chains of cell0, cell2 and cell4 join their T cells, and the B
    # cells stay apart in their own receptor type.
    whole = store.repertoire.load_clonotypes(
        store.repertoire.clonotypes(
            receptors, metric="normalized_hamming", max_distance=100
        )
    )
    assert whole.max_distance == 100
    assert _members(whole)["clonotype_1"] == ["cell0", "cell1", "cell2", "cell4"]
    assert _members(whole)["clonotype_2"] == ["cell5", "cell6"]


def test_clusters_keep_whole_the_b_cell_clones_that_mutations_split(tmp_path) -> None:
    truth = clone_truth()
    receptors_truth = receptor_truth(truth, somatic_mutations=True)
    folder = tmp_path / "truth"
    folder.mkdir()
    store = write_truth_store(folder / "store.zarr", truth)
    receptors = import_truth_receptors(store, receptors_truth, folder)
    expected = truth.partition(
        split_by=receptors_truth.expected["receptor_type"].to_numpy(dtype=object)
    )
    clusters = store.repertoire.clonotypes(
        receptors, receptor_arms="any", metric="hamming", max_distance=1
    )
    assert loaded_partition(store, clusters) == expected
    # Identical junctions split each B cell clone of several cells into its
    # first cell and its mutated cells.
    identity = loaded_partition(
        store, store.repertoire.clonotypes(receptors, receptor_arms="any")
    )
    ids = truth.cell_ids
    split = [
        frozenset(str(cell) for cell in ids[truth.clones == clone])
        for clone in B_CELL_LIGHT_LOCI
        if (truth.clones == clone).sum() > 1
    ]
    assert split and all(clone in expected for clone in split)
    assert identity == (expected - set(split)) | {
        frozenset([cell]) for clone in split for cell in clone
    }
