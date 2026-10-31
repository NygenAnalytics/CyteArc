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

import cytearc.datastore._operations.repertoire as repertoire_operations
from cytearc import plotting
from cytearc.datastore.datastore import DataStore
from cytearc.metadata.rows import (
    READ_OBJECT_BYTES,
    array_row_selection_parts,
    text_read_bytes,
)
from cytearc.repertoire import SLOTS, Clonotypes
from cytearc.storage.artifacts import artifact_group
from tests.clone_helpers import (
    CELLRANGER_HEADER,
    N_RECEPTOR_CELLS,
    ids_read_bytes,
    import_truth_receptors,
    loaded_partition,
    receptor_store,
    write_lines,
    write_truth_store,
)
from tests.clone_truth import B_CELL_LIGHT_LOCI, clone_truth, receptor_truth
from tests.storage_helpers import insert_nullable_cell_column, write_count_store

pytestmark = pytest.mark.usefixtures("figures_closed")

N_CELLS = N_RECEPTOR_CELLS
# The fields of each slot of a clonotype's representative chains.
FIELDS = ("junction_aa", "junction", "v_call", "j_call")

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
        "metric": "identity",
        "cutoff": 0,
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
        ({"within": ["absent"]}, ValueError, "'absent', which is not a cell metadata"),
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
            # Each clonotype is named by its representative's junction:
            # clonotype_1 and clonotype_3 share their beta chain.
            "label": [
                "clonotype_1 (CASSA)",
                "clonotype_2 (CARW)",
                "clonotype_3 (CASSA)",
            ],
            "BCR": [0, 2, 0],
            "TCR": [3, 0, 1],
        }
        ticks = [tick.get_text() for tick in result.axes["top"].get_yticklabels()]
        assert ticks == top["label"].tolist()
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


def test_clonotypes_without_a_beta_chain_are_named_by_their_alpha_chain(
    tmp_path,
) -> None:
    rows = CLONAL_ROWS + [
        "cell7,true,cell7_contig_1,true,500,TRA,TRAV9,,TRAJ9,TRAC,true,true,CAVZ,TGTGCTGTGTAA,10,1,ct9,ct9_1",
    ]
    store, receptors = _receptors(tmp_path, rows)
    ref = store.repertoire.clonotypes(receptors)
    result = plotting.clone_abundance(store, ref, show=False)
    try:
        labels = dict(zip(result.tables["top"]["clone"], result.tables["top"]["label"]))
        orphan = store.repertoire.load_clonotypes(ref).cells.loc["cell7", "clonotype"]
        assert labels[orphan] == f"{orphan} (CAVZ)"
    finally:
        result.close()


def test_figure_labels_name_clonotypes_by_their_representatives_junction(
    tmp_path,
) -> None:
    store, receptors = _receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    # In any order and repeated; clonotype_4 is cell2's, of alpha CAVA and
    # beta CASSA.
    labels = store._clone_figure_labels(
        ref, ["clonotype_4", "clonotype_2", "clonotype_4"]
    )
    assert labels.tolist() == [
        "clonotype_4 (CASSA)",
        "clonotype_2 (CARW)",
        "clonotype_4 (CASSA)",
    ]
    assert store._clone_figure_labels(ref, []).tolist() == []
    with pytest.raises(ValueError, match=r"No clonotype is labelled \['clonotype_9'\]"):
        store._clone_figure_labels(ref, ["clonotype_1", "clonotype_9"])
    # Imported clones are labelled by their IDs alone.
    path = tmp_path / "clonotypes.csv"
    store.clones.export_clones(ref, path)
    imported = store.clones.import_clones(
        path, cell_selection=store.snapshot_cell_selection()
    )
    assert store._clone_figure_labels(imported, ["clonotype_2"]).tolist() == [
        "clonotype_2"
    ]


def test_clonotypes_of_one_arm_are_named_by_the_chains_they_share(tmp_path) -> None:
    store, receptors = _receptors(tmp_path)
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
        "clonotype_1 (CAVA)",
        "clonotype_2 (CQQY)",
        "clonotype_3 (CAVA)",
    ]
    by_beta = store.repertoire.clonotypes(receptors, receptor_arms="VDJ")
    assert store._clone_figure_labels(by_beta, ids[:2]).tolist() == [
        "clonotype_1 (CASSA)",
        "clonotype_2 (CARW)",
    ]


def test_figure_labels_read_the_clonotype_ids_within_the_budget(
    tmp_path, monkeypatch
) -> None:
    store, receptors = _receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    # Four IDs of 11 characters, read as fixed-width text of 4 bytes per
    # character: 176 bytes, three times with the buffers of the read, and 2
    # bytes each for a comparison.
    monkeypatch.setattr(store, "memoryBytes", 3 * 176 + 2 * 4 - 1)
    with pytest.raises(MemoryError, match="Labelling clonotypes reads the IDs of 4"):
        store._clone_figure_labels(ref, ["clonotype_1"])
    monkeypatch.setattr(store, "memoryBytes", 3 * 176 + 2 * 4)
    assert store._clone_figure_labels(ref, ["clonotype_1"]).tolist() == [
        "clonotype_1 (CASSA)"
    ]


def test_a_representative_without_a_junction_is_refused(tmp_path) -> None:
    from cytearc.storage.artifacts import artifact_group

    store, receptors = _receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    # Point clonotype_1's representative at cell7, which has no receptor.
    representatives = artifact_group(store.zw, ref)["representatives"]
    representatives[0] = 7
    with pytest.raises(ValueError, match="no junction amino-acid sequence"):
        store._clone_figure_labels(ref, ["clonotype_1"])


def test_clone_abundance_needs_a_store_that_labels_clones(tmp_path) -> None:
    store, receptors = _receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    loader = SimpleNamespace(
        clones=SimpleNamespace(load_clones=store.clones.load_clones)
    )
    with pytest.raises(TypeError, match="does not provide the figure labels"):
        plotting.clone_abundance(loader, ref, show=False)


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
        together, group_by="tp", method="abundance_cosine", permutations=0
    )
    assert coupling.group_sizes[["group", "cells", "clones"]].values.tolist() == [
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
    # Fate bias tests a clonotype in the fates of one receptor type, so it
    # refuses one whose early and late cells span types, and tests it among
    # the cells of one type.
    store.cells.insert("early", np.arange(N_CELLS) % 2 == 0)
    store.cells.insert("late", np.arange(N_CELLS) % 2 == 1)
    store.cells.insert("outcome", np.asarray(["x", "x", "y", "y"] * (N_CELLS // 4)))
    with pytest.raises(ValueError, match="two labels in 'receptor_type'"):
        store.clones.fate_bias(
            together,
            group_by="outcome",
            early=store.snapshot_cell_selection("early"),
            late=store.snapshot_cell_selection("late"),
        )
    t_cells = np.arange(N_CELLS) < 5
    store.cells.insert("early_t", t_cells & (np.arange(N_CELLS) % 2 == 0))
    store.cells.insert("late_t", t_cells & (np.arange(N_CELLS) % 2 == 1))
    bias = store.clones.load_fate_bias(
        store.clones.fate_bias(
            together,
            group_by="outcome",
            early=store.snapshot_cell_selection("early_t"),
            late=store.snapshot_cell_selection("late_t"),
        )
    )
    # The late T cells, cell1 and cell3, lie in fates x and y.
    assert bias.fates[["fate", "late_cells"]].values.tolist() == [
        ["x, TCR", 1],
        ["y, TCR", 1],
    ]
    assert bias.cells[["x, TCR", "y, TCR"]].values.tolist() == [[0.5, 0.5]] * 3


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


# Reading the eight cells' clonotypes holds each cell's stored int32 code and
# its int64 copy, its receptor type, stored in 20 bytes as "no IR" is five
# characters, and 100 more while the types are coded, and an 8-byte label; and
# with their labels, each of the four clonotypes' 44 stored bytes, eleven
# characters, three times and 100 more.
CLONOTYPES_READ = 8 * (4 + 8) + 8 * (20 + 100) + 8 * 8
CLONOTYPE_LABELS_READ = 4 * (100 + 3 * 44)


def _assert_stored_widths(store, receptors, ref) -> None:
    receptor_group = artifact_group(store.zw, receptors)
    assert receptor_group["values"].dtype == np.dtype("<U5")
    assert receptor_group["contig_junction"].dtype == np.int32
    assert receptor_group["contig_junction"].shape == (14,)
    group = artifact_group(store.zw, ref)
    assert group["codes"].dtype == np.int32
    assert group["clonotype_ids"].dtype == np.dtype("<U11")


def test_defining_clonotypes_is_checked_against_the_memory_budget(
    tmp_path, monkeypatch
) -> None:
    store, receptors = _receptors(tmp_path)
    # Reading the receptor types takes each cell's 20 stored bytes and 100
    # more while they are coded.
    monkeypatch.setattr(store, "memoryBytes", 8 * 120 - 1)
    with pytest.raises(MemoryError, match="types of 8 cells holds about 960 bytes"):
        store.repertoire.clonotypes(receptors)
    # Defining takes 1,000 bytes per cell, and each contig's stored junction
    # code and its int64 copy.
    needed = 8 * 1_000 + 14 * (4 + 8)
    monkeypatch.setattr(store, "memoryBytes", needed - 1)
    with pytest.raises(MemoryError, match=f"of 8 cells holds about {needed} bytes"):
        store.repertoire.clonotypes(receptors)
    assert store.artifacts.list(kind="clonotypes", scope="datastore") == []
    monkeypatch.setattr(store, "memoryBytes", needed)
    ref = store.repertoire.clonotypes(receptors)
    assert store.artifacts.list(kind="clonotypes", scope="datastore") == [ref]
    _assert_stored_widths(store, receptors, ref)


def test_grouping_by_within_columns_is_checked_against_the_memory_budget(
    tmp_path, monkeypatch
) -> None:
    store, receptors = _receptors(tmp_path)
    store.cells.insert("donor", np.asarray(["d1", "d2"] * 4))
    # The definition alone fits, but grouping the cells by donor reads a chunk
    # of 100,000 rows of the column, which does not.
    monkeypatch.setattr(store, "memoryBytes", 8 * 1_000 + 14 * 12)
    assert store.repertoire.clonotypes(receptors)
    with pytest.raises(MemoryError, match="Grouping 8 cells by 1 columns holds"):
        store.repertoire.clonotypes(receptors, within="donor")
    assert len(store.artifacts.list(kind="clonotypes", scope="datastore")) == 1


def test_reading_clonotypes_as_clones_counts_their_receptor_types(
    tmp_path, monkeypatch
) -> None:
    store, receptors = _receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    _assert_stored_widths(store, receptors, ref)
    for with_ids, needed in (
        (False, CLONOTYPES_READ),
        (True, CLONOTYPES_READ + CLONOTYPE_LABELS_READ),
    ):
        monkeypatch.setattr(store, "memoryBytes", needed - 1)
        with pytest.raises(MemoryError, match=f"8 cells holds about {needed} bytes"):
            store._read_clone_input(ref, with_ids=with_ids)
        monkeypatch.setattr(store, "memoryBytes", needed)
        source = store._read_clone_input(ref, with_ids=with_ids)
    assert source.clone_ids is not None
    assert source.clone_ids.tolist() == [f"clonotype_{rank}" for rank in range(1, 5)]
    ((name, labels),) = source.kept_apart
    assert name == "receptor_type"
    assert labels.tolist() == ["TCR"] * 5 + ["BCR"] * 2 + ["no IR"]


def test_clonotypes_load_and_export_as_clones_within_the_memory_budget(
    tmp_path, monkeypatch
) -> None:
    store, receptors = _receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    # Beside 250 bytes per cell for the tables, or 200 for the file, both read
    # the clonotypes and count their labels and the cells' IDs as text, the
    # four labels once each, as their stored chunk is read.
    labels = artifact_group(store.zw, ref)["clonotype_ids"]
    assert labels.shape == (4,)
    read = CLONOTYPES_READ + text_read_bytes(labels, 4) + ids_read_bytes(store, 8)
    assert text_read_bytes(labels, 4) > CLONOTYPE_LABELS_READ
    for verb, per_cell, run in (
        ("Loading", 250, lambda: store.clones.load_clones(ref)),
        ("Exporting", 200, lambda: store.clones.export_clones(ref, tmp_path / "x.csv")),
    ):
        needed = 8 * per_cell + read
        monkeypatch.setattr(store, "memoryBytes", needed - 1)
        with pytest.raises(MemoryError, match=f"{verb} the clones of 8 cells holds"):
            run()
        with pytest.raises(MemoryError, match=f"about {needed} bytes"):
            run()
        monkeypatch.setattr(store, "memoryBytes", needed)
        run()


def test_expansion_of_clonotypes_counts_each_cells_receptor_type(
    tmp_path, monkeypatch
) -> None:
    store, receptors = _receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    # Measuring takes 300 bytes per cell and 8 for its receptor type, and
    # 2,000 for each group, the T cells and the B cells.
    needed = 8 * (300 + 8) + 2 * 2_000
    monkeypatch.setattr(store, "memoryBytes", needed - 1)
    with pytest.raises(MemoryError, match=f"of 8 cells holds about {needed} bytes"):
        store.clones.expansion(ref)
    monkeypatch.setattr(store, "memoryBytes", needed)
    assert store.clones.expansion(ref).kind == "clonal_expansion"


def test_loaded_expansion_of_other_cells_matches_their_receptor_types(
    tmp_path, monkeypatch
) -> None:
    store, receptors = _receptors(tmp_path)
    ref = store.repertoire.clonotypes(receptors)
    store.cells.insert("first_half", np.arange(N_CELLS) < 4)
    measured = store.snapshot_cell_selection("first_half")
    saved = store.clones.expansion(ref, cell_selection=measured)
    # The four measured cells are T cells of one group: 500 bytes each and
    # 2,000 for the group, their IDs as text, the clonotypes with their
    # labels, and the clonotypes' eight cells matched to the measured ones,
    # 56 bytes each, and 32 for each measured cell and 8 for its receptor type,
    # with the read before: the largest stored chunk of the two selections and
    # the clonotypes' codes, and Zarr's objects.
    before = max(
        array_row_selection_parts(artifact_group(store.zw, artifact)[name])[0]
        for artifact, name in (
            (measured, "values"),
            (store.artifacts.inspect(ref).input_ref("cell_selection"), "values"),
            (ref, "codes"),
        )
    )
    needed = (
        4 * 500
        + 2_000
        + ids_read_bytes(store, 4)
        + CLONOTYPES_READ
        + CLONOTYPE_LABELS_READ
        + 8 * 56
        + 4 * (32 + 8)
        + before
        + READ_OBJECT_BYTES
    )
    monkeypatch.setattr(store, "memoryBytes", needed - 1)
    with pytest.raises(MemoryError, match=f"of 4 cells holds about {needed} bytes"):
        store.clones.load_expansion(saved)
    monkeypatch.setattr(store, "memoryBytes", needed)
    assert store.clones.load_expansion(saved).cells.shape[0] == 4


def test_loaded_clonotypes_are_checked_against_the_memory_budget(
    tmp_path, monkeypatch
) -> None:
    store, receptors = _receptors(tmp_path)
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
    store, receptors = _receptors(tmp_path)
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
    store, receptors = _receptors(tmp_path)
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
        store = DataStore(
            str(path), default_assay="RNA", min_features_per_cell=0, nthreads=1
        )
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
    store, receptors = _receptors(tmp_path, rows)
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
    store, receptors = _receptors(tmp_path)
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
    first = loaded.cells.loc["cell0", "clonotype"]
    assert _members(loaded)[first] == ["cell0", "cell4"]
    row = loaded.clonotypes.set_index("clonotype").loc[first]
    assert row[["donor", "site"]].tolist() == ["d1", "s10"]
    cell1 = loaded.clonotypes.set_index("clonotype").loc[
        loaded.cells.loc["cell1", "clonotype"]
    ]
    assert cell1[["donor", "site"]].tolist() == ["d2", "s2"]
    # Clonotypes without within columns store no labels.
    assert "within_labels" not in artifact_group(
        store.zw, store.repertoire.clonotypes(receptors)
    )


def test_renamed_within_labels_give_other_clonotypes(tmp_path) -> None:
    store, receptors = _receptors(tmp_path)
    donors = np.asarray(["d1", "d2", "d1", "d1", "d1", "d1", "d1", "d1"])
    store.cells.insert("donor", donors)
    first = store.repertoire.clonotypes(receptors, within="donor")
    # The same groups under other names give other clonotypes, and the same
    # labels again reuse the saved ones.
    store.cells.insert("donor", np.where(donors == "d1", "x1", "x2"), overwrite=True)
    renamed = store.repertoire.clonotypes(receptors, within="donor")
    assert renamed != first
    assert set(store.repertoire.load_clonotypes(renamed).clonotypes["donor"]) == {
        "x1",
        "x2",
    }
    store.cells.insert("donor", donors.copy(), overwrite=True)
    assert store.repertoire.clonotypes(receptors, within="donor") == first


def test_within_columns_may_not_share_a_name_with_the_result(tmp_path) -> None:
    store, receptors = _receptors(tmp_path)
    for column in ("clone_size", "VDJ_1_junction_aa", "VJ_2_j_call"):
        store.cells.insert(column, np.arange(N_CELLS).astype(str))
        with pytest.raises(ValueError, match=f"'{column}', which is also a column"):
            store.repertoire.clonotypes(receptors, within=column)


def test_regrouped_expansion_plot_keeps_receptor_types_apart(tmp_path) -> None:
    store, receptors = _receptors(tmp_path)
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
    held = windows[windows["cells"] > 0]
    assert held[["group", "window", "cells"]].values.tolist() == [
        ["blood, BCR", "1 to 10", 2],
        ["blood, TCR", "1 to 10", 5],
    ]
    totals = result.tables["categories"].groupby("group")["cells"].sum()
    assert totals.to_dict() == {"blood, BCR": 2, "blood, TCR": 5}
    result.close()
    store.cells.insert("receptor_type", np.asarray(["x"] * N_CELLS))
    with pytest.raises(ValueError, match="'receptor_type', whose labels the clones"):
        store.plots.clonal_expansion(expansion, group_by="receptor_type", show=False)


def test_clonotype_clusters_need_room_for_their_search(tmp_path, monkeypatch) -> None:
    store, receptors = _receptors(tmp_path)
    monkeypatch.setattr(store, "memoryBytes", 1_000_000)
    assert store.repertoire.clonotypes(receptors).kind == "clonotypes"
    with pytest.raises(MemoryError, match="Raise mem_budget"):
        store.repertoire.clonotypes(receptors, metric="hamming", cutoff=1)


def test_clonotype_clusters_get_what_the_budget_leaves_for_their_pairs(
    tmp_path, monkeypatch
) -> None:
    store, receptors = _receptors(tmp_path)
    # Within three substitutions, cell0's alpha junction pairs with cell2's
    # (3 substitutions) and with cell4's second one (2): two pairs of chains
    # of 32 bytes each beyond the 1,000 bytes per cell and the search. The T
    # cells with both arms hold these chains in one group, so linking them
    # adds two links of 32 bytes each.
    reserved = 1_000 * N_CELLS + 512 * 2**20
    for budget, held in ((2 * 32 - 1, "the 2 pairs"), (4 * 32 - 1, "the 2 links")):
        monkeypatch.setattr(store, "memoryBytes", reserved + budget)
        with pytest.raises(MemoryError, match=f"bytes for {held}"):
            store.repertoire.clonotypes(receptors, metric="hamming", cutoff=3)
    assert store.artifacts.list(kind="clonotypes", scope="datastore") == []
    monkeypatch.setattr(store, "memoryBytes", reserved + 4 * 32)
    assert store.repertoire.clonotypes(receptors, metric="hamming", cutoff=3)


def test_clonotype_clusters_link_similar_junctions(tmp_path) -> None:
    store, receptors = _receptors(tmp_path)
    # cell2's alpha junction differs from cell0's in 3 of 12 nucleotides.
    identical = store.repertoire.clonotypes(receptors)
    near = store.repertoire.clonotypes(receptors, metric="hamming", cutoff=3)
    assert near != identical
    loaded = store.repertoire.load_clonotypes(near)
    assert loaded.metric == "hamming" and loaded.cutoff == 3
    assert _members(loaded)["clonotype_1"] == ["cell0", "cell1", "cell2", "cell4"]
    status = store.artifacts.inspect(near)
    assert (status.parameters["metric"], status.parameters["cutoff"]) == ("hamming", 3)
    two = store.repertoire.load_clonotypes(
        store.repertoire.clonotypes(receptors, metric="hamming", cutoff=2)
    )
    assert _members(two)["clonotype_1"] == ["cell0", "cell1", "cell4"]
    # 3 of 12 is 25 percent.
    for cutoff, joined in ((25, True), (24, False)):
        result = store.repertoire.load_clonotypes(
            store.repertoire.clonotypes(
                receptors, metric="normalized_hamming", cutoff=cutoff
            )
        )
        assert ("cell2" in _members(result)["clonotype_1"]) is joined
    edits = store.repertoire.load_clonotypes(
        store.repertoire.clonotypes(receptors, metric="levenshtein", cutoff=3)
    )
    assert _members(edits)["clonotype_1"] == ["cell0", "cell1", "cell2", "cell4"]
    # Residues: cell2's alpha chain is identical, so amino-acid clusters
    # equal amino-acid clonotypes.
    residues = store.repertoire.load_clonotypes(
        store.repertoire.clonotypes(
            receptors, sequence="aa", metric="levenshtein", cutoff=1
        )
    )
    assert _members(residues)["clonotype_1"] == ["cell0", "cell1", "cell2", "cell4"]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"metric": "tcrdist"}, ValueError, "metric must be one of"),
        ({"metric": "identity", "cutoff": 1}, ValueError, "takes no cutoff"),
        ({"metric": "identity", "cutoff": False}, ValueError, "takes no cutoff"),
        ({"metric": "identity", "cutoff": 0.0}, ValueError, "takes no cutoff"),
        (
            {"metric": "identity", "cutoff": np.asarray([0])},
            ValueError,
            "takes no cutoff",
        ),
        ({"metric": "hamming"}, ValueError, "needs a cutoff"),
        ({"metric": "hamming", "cutoff": 1.5}, TypeError, "cutoff must be an integer"),
        ({"metric": "hamming", "cutoff": 1.0}, TypeError, "cutoff must be an integer"),
        ({"metric": "levenshtein", "cutoff": True}, TypeError, "must be an integer"),
        ({"metric": "hamming", "cutoff": -1}, ValueError, "cutoff must be at least 0"),
        ({"metric": "normalized_hamming", "cutoff": 101}, ValueError, "at most 100"),
    ],
)
def test_clonotype_cluster_options_are_checked(
    tmp_path, kwargs, error, message
) -> None:
    store, receptors = _receptors(tmp_path)
    with pytest.raises(error, match=message):
        store.repertoire.clonotypes(receptors, **kwargs)


def test_the_bounds_of_the_cutoff_are_accepted(tmp_path) -> None:
    store, receptors = _receptors(tmp_path)
    # Identity records a cutoff of 0, so a cutoff of 0 is the same request.
    identity = store.repertoire.clonotypes(receptors)
    assert store.repertoire.clonotypes(receptors, cutoff=0) == identity
    assert store.repertoire.clonotypes(receptors, cutoff=np.int64(0)) == identity
    # Within 100 percent, every chain is near every chain of its length: the
    # alpha chains of cell0, cell2 and cell4 join their T cells, and the B
    # cells stay apart in their own receptor type.
    whole = store.repertoire.load_clonotypes(
        store.repertoire.clonotypes(receptors, metric="normalized_hamming", cutoff=100)
    )
    assert whole.cutoff == 100
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
        receptors, receptor_arms="any", metric="hamming", cutoff=1
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
