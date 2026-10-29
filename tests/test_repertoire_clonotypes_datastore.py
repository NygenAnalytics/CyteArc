from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cytearc import plotting
from cytearc.repertoire import Clonotypes
from tests.storage_helpers import insert_nullable_cell_column
from tests.test_repertoire_import import CELLRANGER_HEADER, N_CELLS, _store, _write

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


@pytest.fixture(autouse=True)
def _figures_are_closed():
    before = set(plt.get_fignums())
    yield
    leaked = set(plt.get_fignums()) - before
    for number in leaked:
        plt.close(number)
    assert not leaked, f"Figures left open: {sorted(leaked)}"


def _receptors(tmp_path: Path, rows: list[str] = CLONAL_ROWS):
    store = _store(tmp_path)
    path = _write(tmp_path, "clonal.csv", CELLRANGER_HEADER, rows)
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
    store = _store(tmp_path)
    path = _write(
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
    reader = _store(tmp_path, zarr_mode="r")
    assert reader.repertoire.clonotypes(receptors) == saved
    with pytest.raises(PermissionError, match="define_clonotypes"):
        reader.repertoire.clonotypes(receptors, sequence="aa")


def test_clonotype_abundance_counts_sizes_and_draws_the_largest(tmp_path) -> None:
    store, receptors = _receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    result = store.plots.clonotype_abundance(ref, top=3, show=False)
    try:
        sizes = result.tables["sizes"]
        assert sizes.to_dict("list") == {
            "group": ["all cells"] * 3,
            "clone_size": [1, 2, 3],
            "n_clonotypes": [2, 1, 1],
        }
        top = result.tables["top"]
        assert top["clonotype"].tolist() == [
            "clonotype_1",
            "clonotype_2",
            "clonotype_3",
        ]
        assert top["label"].tolist() == [
            "clonotype_1 (CASSA)",
            "clonotype_2 (CARW)",
            "clonotype_3 (CASSA)",
        ]
        assert top["cells"].tolist() == [3, 2, 1]
        assert result.provenance.n_cells == 7
    finally:
        result.close()


def test_clonotype_abundance_splits_clonotypes_by_group(tmp_path) -> None:
    store, receptors = _receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    store.cells.insert(
        "sample", np.asarray(["s1", "s2", "s1", "s2", "s1", "s1", "s2", "s1"])
    )
    result = plotting.clonotype_abundance(store, ref, group_by="sample", show=False)
    try:
        sizes = result.tables["sizes"]
        assert sizes.query("group == 's1'")[
            ["clone_size", "n_clonotypes"]
        ].values.tolist() == [
            [1, 2],
            [2, 1],
        ]
        top = result.tables["top"].set_index("clonotype")
        assert top.loc["clonotype_1", ["s1", "s2"]].tolist() == [2, 1]
        assert len(result.axes["sizes"].get_lines()) == 2
    finally:
        result.close()
    with pytest.raises(ValueError, match="top must be a positive integer"):
        plotting.clonotype_abundance(store, ref, top=0, show=False)
    # Shown by default: on Agg the owned figure is closed.
    shown = plotting.clonotype_abundance(store, ref, top=2)
    assert shown.tables["top"]["clonotype"].tolist() == ["clonotype_1", "clonotype_2"]
    with pytest.raises(TypeError, match="does not provide clonotypes"):
        plotting.clonotype_abundance(object(), ref, show=False)


def test_clonotypes_without_a_beta_chain_are_named_by_their_alpha_chain(
    tmp_path,
) -> None:
    rows = CLONAL_ROWS + [
        "cell7,true,cell7_contig_1,true,500,TRA,TRAV9,,TRAJ9,TRAC,true,true,CAVZ,TGTGCTGTGTAA,10,1,ct9,ct9_1",
    ]
    store, receptors = _receptors(tmp_path, rows)
    ref = store.repertoire.clonotypes(receptors)
    result = plotting.clonotype_abundance(store, ref, show=False)
    try:
        labels = dict(
            zip(result.tables["top"]["clonotype"], result.tables["top"]["label"])
        )
        orphan = store.repertoire.load_clonotypes(ref).cells.loc["cell7", "clonotype"]
        assert labels[orphan] == f"{orphan} (CAVZ)"
    finally:
        result.close()


def test_clonotypes_refuse_more_cells_than_the_memory_budget(
    tmp_path, monkeypatch
) -> None:
    store, receptors = _receptors(tmp_path)
    monkeypatch.setattr(store, "memoryBytes", 1_000)
    with pytest.raises(MemoryError, match="Raise mem_budget"):
        store.repertoire.clonotypes(receptors)
    assert store.artifacts.list(kind="clonotypes") == []
