"""Clonotypes of hand-written contigs, read back as clonotypes and as clones.

The clone summaries of ``ds.clones`` read clonotypes as clones and measure
each receptor type apart, so their expected values here are counted by hand
within the T cells and the B cells.
"""

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from cytearc import plotting
from cytearc.repertoire import Clonotypes
from tests.clone_helpers import (
    CELLRANGER_HEADER,
    N_RECEPTOR_CELLS,
    loaded_partition,
    receptor_store,
    write_lines,
)
from tests.storage_helpers import insert_nullable_cell_column

pytestmark = pytest.mark.usefixtures("figures_closed")

N_CELLS = N_RECEPTOR_CELLS

# cell0 and cell1 share an alpha and a beta chain. cell2 has the same alpha
# residues from other nucleotides, cell3 the beta chain alone, and cell4 a
# second alpha chain beside the pair. cell5 and cell6 are B cells of one
# clone, and cell7 has no contig.
CLONAL_ROWS = [
    "cell0,true,cell0_contig_1,true,500,TRA,TRAV1,,TRAJ1,TRAC,true,true,CAVA,TGTGCTGTGGCC,90,9,ct1,ct1_1",
    "cell0,true,cell0_contig_2,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,CASSA,TGTGCCAGCAGCGCC,80,8,ct1,ct1_2",
    "cell1,true,cell1_contig_1,true,500,TRA,TRAV1,,TRAJ1,TRAC,true,true,CAVA,TGTGCTGTGGCC,70,7,ct1,ct1_1",
    "cell1,true,cell1_contig_2,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,CASSA,TGTGCCAGCAGCGCC,60,6,ct1,ct1_2",
    "cell2,true,cell2_contig_1,true,500,TRA,TRAV2,,TRAJ1,TRAC,true,true,CAVA,TGTGCAGTAGCA,50,5,ct2,ct2_1",
    "cell2,true,cell2_contig_2,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,CASSA,TGTGCCAGCAGCGCC,40,4,ct2,ct2_2",
    "cell3,true,cell3_contig_1,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,CASSA,TGTGCCAGCAGCGCC,30,3,ct3,ct3_1",
    "cell4,true,cell4_contig_1,true,500,TRA,TRAV1,,TRAJ1,TRAC,true,true,CAVA,TGTGCTGTGGCC,90,9,ct4,ct4_1",
    "cell4,true,cell4_contig_2,true,500,TRA,TRAV3,,TRAJ2,TRAC,true,true,CAVB,TGTGCTGTGTAC,20,2,ct4,ct4_2",
    "cell4,true,cell4_contig_3,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,CASSA,TGTGCCAGCAGCGCC,80,8,ct4,ct4_3",
    "cell5,true,cell5_contig_1,true,500,IGH,IGHV1,,IGHJ1,IGHM,true,true,CARW,TGTGCGAGATGG,500,20,cb1,cb1_1",
    "cell5,true,cell5_contig_2,true,500,IGK,IGKV1,,IGKJ1,IGKC,true,true,CQQY,TGTCAGCAGTAT,400,15,cb1,cb1_2",
    "cell6,true,cell6_contig_1,true,500,IGH,IGHV1,,IGHJ1,IGHG1,true,true,CARW,TGTGCGAGATGG,300,12,cb1,cb1_1",
    "cell6,true,cell6_contig_2,true,500,IGK,IGKV1,,IGKJ1,IGKC,true,true,CQQY,TGTCAGCAGTAT,200,10,cb1,cb1_2",
]


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


def _receptors(tmp_path: Path, rows: list[str] = CLONAL_ROWS):
    store = receptor_store(tmp_path)
    path = write_lines(tmp_path, "clonal.csv", CELLRANGER_HEADER, rows)
    receptors = store.repertoire.import_receptors(
        path, cell_selection=store.snapshot_cell_selection()
    )
    return store, receptors


def _members(table: Clonotypes) -> dict[str, list[str]]:
    cells = table.cells.dropna(subset=["clonotype"])
    return {
        str(name): sorted(group.index.tolist())
        for name, group in cells.groupby("clonotype")
    }


def test_clonotypes_follow_identical_chains(tmp_path) -> None:
    store, receptors = _receptors(tmp_path)
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
    store, receptors = _receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors, **options)
    partition = sorted(_members(store.repertoire.load_clonotypes(ref)).values())
    assert partition == sorted(expected)


def test_within_columns_keep_donors_apart(tmp_path) -> None:
    store, receptors = _receptors(tmp_path)
    store.cells.insert(
        "donor", np.asarray(["d1", "d2", "d1", "d1", "d1", "d1", "d1", "d1"])
    )
    ref = store.repertoire.clonotypes(receptors, within="donor")
    loaded = store.repertoire.load_clonotypes(ref)
    assert loaded.within == ("donor",)
    first = loaded.cells.loc["cell0", "clonotype"]
    assert _members(loaded)[first] == ["cell0", "cell4"]
    assert loaded.cells.loc["cell1", "clonotype"] != first
    assert loaded.clonotypes.set_index("clonotype").loc[first, "donor"] == "d1"
    donors = set(loaded.clonotypes["donor"])
    assert donors == {"d1", "d2"}
    status = store.artifacts.inspect(ref)
    assert status.parameters["within"] == ["donor"]
    assert status.inputs["within_fingerprint"] is not None


def test_within_columns_need_a_label_for_every_cell_with_a_receptor(tmp_path) -> None:
    store, receptors = _receptors(tmp_path)
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
    store, receptors = _receptors(tmp_path, rows)
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
    store, receptors = _receptors(tmp_path)
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
    with pytest.raises(ValueError, match="store no v_call"):
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
        ({"within": ["absent"]}, KeyError, "absent"),
    ],
)
def test_clonotype_arguments_are_checked(tmp_path, kwargs, error, message) -> None:
    store, receptors = _receptors(tmp_path)
    store.cells.insert("donor", np.asarray(["d1"] * N_CELLS))
    with pytest.raises(error, match=message):
        store.repertoire.clonotypes(receptors, **kwargs)


def test_clonotypes_refuse_other_artifacts(tmp_path) -> None:
    store, receptors = _receptors(tmp_path)
    with pytest.raises(ValueError, match="immune_receptors artifact"):
        store.repertoire.clonotypes(store.snapshot_cell_selection())
    with pytest.raises(TypeError, match="receptors must be an ArtifactRef"):
        store.repertoire.clonotypes("receptors")
    with pytest.raises(ValueError, match="clonotypes artifact"):
        store.repertoire.load_clonotypes(receptors)
    with pytest.raises(TypeError, match="clonotypes must be an ArtifactRef"):
        store.repertoire.load_clonotypes("clonotypes")


def test_a_read_only_store_refuses_new_clonotypes(tmp_path) -> None:
    store, receptors = _receptors(tmp_path)
    saved = store.repertoire.clonotypes(receptors)
    reader = receptor_store(tmp_path, zarr_mode="r")
    assert reader.repertoire.clonotypes(receptors) == saved
    with pytest.raises(PermissionError, match="define_clonotypes"):
        reader.repertoire.clonotypes(receptors, sequence="aa")


def test_clone_abundance_counts_the_clonotypes_of_each_receptor_type(
    tmp_path,
) -> None:
    store, receptors = _receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    result = store.plots.clone_abundance(ref, top=3, show=False)
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
            "clone": ["clonotype_1", "clonotype_2", "clonotype_3"],
            "BCR": [0, 2, 0],
            "TCR": [3, 0, 1],
        }
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
    store, receptors = _receptors(tmp_path)
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
        top = result.tables["top"].set_index("clone")
        assert top.loc["clonotype_1", ["s1, TCR", "s2, TCR"]].tolist() == [2, 1]
        assert top.loc["clonotype_1", ["s1, BCR", "s2, BCR"]].tolist() == [0, 0]
        assert result.legends[0].label == "sample, receptor_type"
        assert len(result.axes["sizes"].get_lines()) == 4
    finally:
        result.close()
    with pytest.raises(ValueError, match="top must be at least 1"):
        plotting.clone_abundance(store, ref, top=0, show=False)
    # Shown by default: on Agg the owned figure is closed.
    shown = plotting.clone_abundance(store, ref, top=2)
    assert shown.tables["top"]["clone"].tolist() == ["clonotype_1", "clonotype_2"]
    with pytest.raises(TypeError, match="does not provide clones.load_clones"):
        plotting.clone_abundance(object(), ref, show=False)


def test_clonotypes_load_and_export_as_clones(tmp_path) -> None:
    store, receptors = _receptors(tmp_path)
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
    assert store.clones.export_clones(ref, path) == 7
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
    store, receptors = _receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    indices = store.clones.diversity(ref).indices
    assert indices[
        ["group", "receptor_type", "cells", "clones", "singletons", "doubletons"]
    ].values.tolist() == [["BCR", "BCR", 2, 1, 0, 1], ["TCR", "TCR", 5, 3, 2, 0]]
    curves = store.clones.rarefaction(ref, orders=(0,))
    observed = curves.curves[curves.curves["method"] == "observed"]
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
    assert set(expansion.categories["receptor_type"]) == {"BCR", "TCR"}


def test_summaries_keep_receptor_types_apart_within_one_clonotype(tmp_path) -> None:
    rows = CLONAL_ROWS + [
        # cell7 pairs a beta chain of the T cell clone with a heavy chain.
        "cell7,true,cell7_contig_1,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,CASSA,TGTGCCAGCAGCGCC,30,3,ct9,ct9_1",
        "cell7,true,cell7_contig_2,true,500,IGH,IGHV1,,IGHJ1,IGHM,true,true,CARW,TGTGCGAGATGG,20,2,ct9,ct9_2",
    ]
    store, receptors = _receptors(tmp_path, rows)
    together = store.repertoire.clonotypes(
        receptors, receptor_arms="VDJ", within_receptor_type=False
    )
    # One clonotype holds every cell, of three receptor types.
    assert store.clones.load_clones(together).clones["n_cells"].tolist() == [8]
    expansion = store.clones.load_expansion(store.clones.expansion(together))
    assert expansion.cells["group_clone_size"].tolist() == [5, 5, 5, 5, 5, 2, 2, 1]
    indices = store.clones.diversity(together).indices
    assert indices[["receptor_type", "cells", "clones"]].values.tolist() == [
        ["ambiguous", 1, 1],
        ["BCR", 2, 1],
        ["TCR", 5, 1],
    ]


def test_summaries_of_clonotypes_follow_a_cell_selection(tmp_path) -> None:
    store, receptors = _receptors(tmp_path)
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
    indices = store.clones.diversity(ref, cell_selection=chosen).indices
    assert indices[["receptor_type", "cells", "clones"]].values.tolist() == [
        ["BCR", 1, 1],
        ["TCR", 2, 2],
    ]


def test_group_by_may_not_name_the_receptor_type_again(tmp_path) -> None:
    store, receptors = _receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    store.cells.insert("receptor_type", np.asarray(["x"] * N_CELLS))
    with pytest.raises(ValueError, match="'receptor_type', whose labels the artifact"):
        store.clones.diversity(ref, group_by="receptor_type")
    with pytest.raises(ValueError, match="'receptor_type', whose labels the clones"):
        plotting.clone_abundance(store, ref, group_by="receptor_type", show=False)
    with pytest.raises(ValueError, match="clones must be a clones or clonotypes"):
        store.clones.diversity(receptors)


def test_clonotypes_refuse_more_cells_than_the_memory_budget(
    tmp_path, monkeypatch
) -> None:
    store, receptors = _receptors(tmp_path)
    monkeypatch.setattr(store, "memoryBytes", 1_000)
    with pytest.raises(MemoryError, match="Raise mem_budget"):
        store.repertoire.clonotypes(receptors)
    assert store.artifacts.list(kind="clonotypes", scope="datastore") == []


def test_loaded_clonotypes_are_checked_against_the_memory_budget(
    tmp_path, monkeypatch
) -> None:
    store, receptors = _receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    # The tables of eight cells take about 8,000 bytes.
    monkeypatch.setattr(store, "memoryBytes", 7_999)
    with pytest.raises(MemoryError, match="Loading the clonotypes of 8 cells"):
        store.repertoire.load_clonotypes(ref)
    monkeypatch.setattr(store, "memoryBytes", 8_000)
    assert len(store.repertoire.load_clonotypes(ref).clonotypes) == 4


def test_within_columns_may_not_share_a_name_with_the_result(tmp_path) -> None:
    store, receptors = _receptors(tmp_path)
    store.cells.insert("clone_size", np.arange(N_CELLS).astype(str))
    with pytest.raises(ValueError, match="'clone_size', which is also a column"):
        store.repertoire.clonotypes(receptors, within="clone_size")
