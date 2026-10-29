import gzip
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cytearc.datastore.datastore import DataStore
from cytearc.repertoire import ImmuneReceptors
from tests.storage_helpers import write_count_store

N_CELLS = 8
CELLRANGER_HEADER = (
    "barcode,is_cell,contig_id,high_confidence,length,chain,v_gene,d_gene,j_gene,"
    "c_gene,full_length,productive,cdr3,cdr3_nt,reads,umis,raw_clonotype_id,"
    "raw_consensus_id"
)
# T cell contigs of cells cell0 to cell5 and of a barcode outside the store.
TCR_ROWS = [
    # cell0: one alpha and one beta chain.
    "cell0,true,cell0_contig_1,true,500,TRA,TRAV1,,TRAJ1,TRAC,true,true,CAVF,TGTGCTGTGTTT,100,5,ct1,ct1_1",
    "cell0,true,cell0_contig_2,true,500,TRB,TRBV1,TRBD1,TRBJ1,TRBC1,true,true,CASSF,TGTGCCAGCAGCTTT,200,8,ct1,ct1_2",
    # cell1: a second alpha chain with fewer UMIs.
    "cell1,true,cell1_contig_1,true,500,TRA,TRAV2,,TRAJ2,TRAC,true,true,CAVG,TGTGCTGTGGGT,90,6,ct2,ct2_1",
    "cell1,true,cell1_contig_2,true,500,TRA,TRAV3,,TRAJ3,TRAC,true,true,CAVH,TGTGCTGTGCAT,50,2,ct2,ct2_2",
    "cell1,true,cell1_contig_3,true,500,TRB,TRBV2,,TRBJ2,TRBC2,true,true,CASSG,TGTGCCAGCAGCGGT,300,9,ct2,ct2_3",
    # cell2: a beta chain and a non-productive alpha chain.
    "cell2,true,cell2_contig_1,true,500,TRB,TRBV3,,TRBJ3,TRBC1,true,true,CASSH,TGTGCCAGCAGCCAT,80,4,ct3,ct3_1",
    "cell2,true,cell2_contig_2,true,500,TRA,TRAV4,,TRAJ4,TRAC,true,false,CAVI,TGTGCTGTGATT,30,3,ct3,ct3_2",
    # cell3: three beta chains.
    "cell3,true,cell3_contig_1,true,500,TRB,TRBV4,,TRBJ4,TRBC1,true,true,CASSI,TGTGCCAGCAGCATT,10,3,ct4,ct4_1",
    "cell3,true,cell3_contig_2,true,500,TRB,TRBV5,,TRBJ5,TRBC1,true,true,CASSK,TGTGCCAGCAGCAAA,10,2,ct4,ct4_2",
    "cell3,true,cell3_contig_3,true,500,TRB,TRBV6,,TRBJ6,TRBC1,true,true,CASSL,TGTGCCAGCAGCCTT,10,1,ct4,ct4_3",
    # cell5: a low-confidence contig outside a called cell.
    "cell5,false,cell5_contig_1,false,300,TRB,TRBV7,,TRBJ7,TRBC1,true,true,CASSM,TGTGCCAGCAGCATG,5,1,None,None",
    # A barcode that is not a cell of the store.
    "outside-1,true,outside-1_contig_1,true,500,TRB,TRBV8,,TRBJ8,TRBC1,true,true,CASSN,TGTGCCAGCAGCAAT,40,3,ct5,ct5_1",
]
# B cell contigs: cell4 is a B cell and cell0 also has a heavy chain.
BCR_ROWS = [
    "cell4,true,cell4_contig_1,true,500,IGH,IGHV1,IGHD1,IGHJ1,IGHM,true,true,CARW,TGTGCGAGATGG,500,20,cb1,cb1_1",
    "cell4,true,cell4_contig_2,true,500,IGK,IGKV1,,IGKJ1,IGKC,true,true,CQQY,TGTCAGCAGTAT,400,15,cb1,cb1_2",
    "cell0,true,cell0_contig_1,true,500,IGH,IGHV2,,IGHJ2,IGHG1,true,true,CARX,TGTGCGAGATAA,50,2,cb2,cb2_1",
]
AIRR_HEADER = (
    "cell_id\tsequence_id\tv_call\td_call\tj_call\tc_call\tjunction\tjunction_aa"
    "\tproductive\tduplicate_count\tconsensus_count\tv_identity\tsequence"
    "\tsequence_alignment"
)
AIRR_ROWS = [
    "cell6\tcell6_contig_1\tIGHV3-30*01\t\tIGHJ4*02\tIGHG1\tTGTGCG\tCARW\tT\t12\t300\t0.95\tACGTACGT\tACGT-ACGT",
    "cell6\tcell6_contig_2\tIGLV1-40*01\t\tIGLJ2*01\tIGLC2\tTGTCAG\tCQSY\tT\t9\t200\t\tGGCC\t",
]


@pytest.fixture(autouse=True)
def _figures_are_closed():
    before = set(plt.get_fignums())
    yield
    leaked = set(plt.get_fignums()) - before
    for number in leaked:
        plt.close(number)
    assert not leaked, f"Figures left open: {sorted(leaked)}"


def _store(tmp_path: Path, **options) -> DataStore:
    path = tmp_path / "store.zarr"
    if not path.exists():
        counts = np.random.default_rng(5).poisson(2.0, size=(N_CELLS, 6)) + 1
        write_count_store(str(path), {"RNA": counts}, "uint16")
    return DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, nthreads=1, **options
    )


def _write(tmp_path: Path, name: str, header: str, rows: list[str]) -> Path:
    path = tmp_path / name
    path.write_text("\n".join([header, *rows]) + "\n")
    return path


def _tcr(tmp_path: Path, rows: list[str] = TCR_ROWS, name: str = "tcr.csv") -> Path:
    return _write(tmp_path, name, CELLRANGER_HEADER, rows)


def _bcr(tmp_path: Path) -> Path:
    return _write(tmp_path, "bcr.csv", CELLRANGER_HEADER, BCR_ROWS)


def _airr(tmp_path: Path) -> Path:
    return _write(tmp_path, "bcr.tsv", AIRR_HEADER, AIRR_ROWS)


def test_import_classifies_each_cells_chains(tmp_path, caplog) -> None:
    store = _store(tmp_path)
    cells = store.snapshot_cell_selection()

    ref = store.repertoire.import_receptors(
        [_tcr(tmp_path), _bcr(tmp_path), _airr(tmp_path)], cell_selection=cells
    )

    assert ref.kind == "immune_receptors" and ref.scope == "datastore"
    status = store.artifacts.inspect(ref)
    assert status.operation == "import_immune_receptors"
    assert status.parameters == {"productive_only": True}
    sources = status.inputs["sources"]
    assert [source["format"] for source in sources] == sorted(
        [source["format"] for source in sources],
        key=lambda value: [s["format"] for s in sources].index(value),
    )
    assert {source["format"] for source in sources} == {"cellranger", "airr"}
    assert sources == sorted(sources, key=lambda source: source["sha256"])

    loaded = store.repertoire.load_receptors(ref)
    assert isinstance(loaded, ImmuneReceptors)
    assert loaded.productive_only is True
    table = loaded.cells
    assert table.index.tolist() == [f"cell{index}" for index in range(N_CELLS)]
    assert table["receptor_type"].tolist() == [
        "ambiguous",  # alpha and beta chains with a heavy chain
        "TCR",
        "TCR",
        "multichain",
        "BCR",
        "no IR",  # its one contig is outside a called cell
        "BCR",
        "no IR",
    ]
    assert table["receptor_subtype"].tolist() == [
        "ambiguous",
        "TRA+TRB",
        "TRA+TRB",
        "multichain",
        "IGH+IGK",
        "no IR",
        "IGH+IGL",
        "no IR",
    ]
    assert table["chain_pairing"].tolist() == [
        "ambiguous",
        "extra VJ",
        "orphan VDJ",
        "multichain",
        "single pair",
        "no IR",
        "single pair",
        "no IR",
    ]
    assert table["n_vj"].tolist() == [1, 2, 0, 0, 1, 0, 1, 0]
    assert table["n_vdj"].tolist() == [2, 1, 1, 3, 1, 0, 1, 0]
    assert table["n_contigs"].tolist() == [3, 3, 2, 3, 2, 1, 2, 0]
    assert table["multichain"].tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    # The more expressed alpha chain of cell1 fills VJ_1.
    assert table.loc["cell1", "VJ_1_junction_aa"] == "CAVG"
    assert table.loc["cell1", "VJ_2_junction_aa"] == "CAVH"
    # cell0's beta chain has more UMIs than its heavy chain.
    assert table.loc["cell0", "VDJ_1_junction_aa"] == "CASSF"
    assert table.loc["cell0", "VDJ_2_locus"] == "IGH"
    assert table.loc["cell6", "VJ_1_v_call"] == "IGLV1-40*01"
    assert table.loc["cell6", "VJ_1_locus"] == "IGL"  # inferred from its calls
    assert pd.isna(table.loc["cell2", "VJ_1_junction_aa"])
    assert table.loc["cell0", "VDJ_1_umi_count"] == 8
    assert pd.isna(table.loc["cell7", "VDJ_1_umi_count"])
    assert "VJ_1_d_call" not in table.columns and "VDJ_1_d_call" in table.columns

    chains = loaded.chains
    assert len(chains) == 16
    assert chains["ids"].tolist()[:3] == ["cell0", "cell0", "cell0"]
    assert set(chains.columns) >= {
        "ids",
        "slot",
        "chain",
        "sequence",
        "sequence_alignment",
    }
    by_id = chains.set_index("sequence_id")
    assert by_id.loc["cell2_contig_2", "chain"] == np.False_  # non-productive
    assert by_id.loc["cell3_contig_3", "chain"] == np.True_  # a chain in no slot
    assert by_id.loc["cell3_contig_3", "slot"] == ""
    assert by_id.loc["cell5_contig_1", "is_cell"] == np.False_
    assert by_id.loc["cell6_contig_1", "sequence_alignment"] == "ACGT-ACGT"
    assert pd.isna(by_id.loc["cell6_contig_2", "sequence_alignment"])
    assert by_id.loc["cell6_contig_1", "duplicate_count"] == 12
    assert by_id.loc["cell6_contig_1", "v_identity"] == 0.95
    assert np.isnan(by_id.loc["cell6_contig_2", "v_identity"])

    group = store.artifacts.load(ref)
    assert group.attrs["n_file_contigs"] == 17
    assert group.attrs["n_unmatched_contigs"] == 1
    assert group.attrs["n_unmatched_barcodes"] == 1
    assert group.attrs["unmatched_barcode_examples"] == ["outside-1"]


def test_unmatched_barcodes_are_named_in_a_warning(tmp_path) -> None:
    from cytearc.utils.logging import logger

    messages: list[str] = []
    handler = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        store = _store(tmp_path)
        store.repertoire.import_receptors(
            _tcr(tmp_path), cell_selection=store.snapshot_cell_selection()
        )
    finally:
        logger.remove(handler)
    assert any(
        "1 of 12 contigs" in message and "'outside-1'" in message
        for message in messages
    )


def test_unmatched_barcodes_count_once_per_sample(tmp_path) -> None:
    store = _store(tmp_path)
    path = _tcr(tmp_path)
    # Under a second prefix no barcode of the file names a cell, so its
    # outside-1 is a barcode of its own beside the first file's.
    ref = store.repertoire.import_receptors(
        [path, path],
        cell_selection=store.snapshot_cell_selection(),
        barcode_prefix=["", "s2__"],
    )
    attributes = store.artifacts.load(ref).attrs
    barcodes = {row.split(",")[0] for row in TCR_ROWS}
    assert attributes["n_unmatched_barcodes"] == len(barcodes) + 1
    assert attributes["n_unmatched_contigs"] == len(TCR_ROWS) + 1


def test_import_is_reused_by_file_contents_in_any_order(tmp_path) -> None:
    store = _store(tmp_path)
    cells = store.snapshot_cell_selection()
    first = store.repertoire.import_receptors(
        [_tcr(tmp_path), _bcr(tmp_path)], cell_selection=cells
    )
    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.copy(tmp_path / "tcr.csv", moved / "renamed.csv")
    shutil.copy(tmp_path / "bcr.csv", moved / "bcr.csv")
    assert (
        store.repertoire.import_receptors(
            [moved / "bcr.csv", moved / "renamed.csv"], cell_selection=cells
        )
        == first
    )
    edited = _tcr(
        tmp_path, [row.replace("CASSF,", "CASSW,") for row in TCR_ROWS], "edited.csv"
    )
    changed = store.repertoire.import_receptors(
        [edited, _bcr(tmp_path)], cell_selection=cells
    )
    assert changed != first
    assert (
        store.repertoire.load_receptors(changed).cells.loc["cell0", "VDJ_1_junction_aa"]
        == "CASSW"
    )
    again = store.repertoire.import_receptors(
        [_tcr(tmp_path), _bcr(tmp_path)], cell_selection=cells, invalidate_cache=True
    )
    assert again != first


def test_full_ties_go_to_the_contig_id_in_any_row_order(tmp_path) -> None:
    store = _store(tmp_path)
    cells = store.snapshot_cell_selection()
    # Two alpha chains that tie on counts and both junctions.
    first = "cell0,true,cell0_contig_1,true,500,TRA,TRAV1,,TRAJ1,TRAC,true,true,CAVF,TGTGCTGTGTTT,100,5,ct1,ct1_1"
    second = "cell0,true,cell0_contig_2,true,500,TRA,TRAV2,,TRAJ1,TRAC,true,true,CAVF,TGTGCTGTGTTT,100,5,ct1,ct1_1"
    for name, rows in (("ab.csv", [first, second]), ("ba.csv", [second, first])):
        receptors = store.repertoire.import_receptors(
            _write(tmp_path, name, CELLRANGER_HEADER, rows), cell_selection=cells
        )
        table = store.repertoire.load_receptors(receptors).cells
        assert table.loc["cell0", ["VJ_1_v_call", "VJ_2_v_call"]].tolist() == [
            "TRAV1",
            "TRAV2",
        ]


def test_compressed_files_read_whatever_their_name(tmp_path) -> None:
    store = _store(tmp_path)
    cells = store.snapshot_cell_selection()
    # Gzip-compressed Cell Ranger and AIRR files without the .gz suffix.
    for plain in (_tcr(tmp_path), _airr(tmp_path)):
        packed = plain.with_name(f"packed_{plain.name}")
        packed.write_bytes(gzip.compress(plain.read_bytes()))
        expected, found = (
            store.repertoire.load_receptors(
                store.repertoire.import_receptors(path, cell_selection=cells)
            ).cells
            for path in (plain, packed)
        )
        pd.testing.assert_frame_equal(found, expected)


def test_non_productive_chains_count_when_asked(tmp_path) -> None:
    store = _store(tmp_path)
    ref = store.repertoire.import_receptors(
        _tcr(tmp_path),
        cell_selection=store.snapshot_cell_selection(),
        productive_only=False,
    )
    loaded = store.repertoire.load_receptors(ref)
    assert loaded.productive_only is False
    assert loaded.cells.loc["cell2", "chain_pairing"] == "single pair"
    assert loaded.cells.loc["cell2", "VJ_1_junction_aa"] == "CAVI"


def test_barcode_prefixes_join_the_cells_of_merged_stores(tmp_path) -> None:
    store = _store(tmp_path)
    cells = store.snapshot_cell_selection()
    # cell ids are "cell0"...; strip the "cell" of the barcodes and prefix it back.
    rows = [row.replace("cell", "", 1) for row in TCR_ROWS]
    path = _tcr(tmp_path, rows, "prefixed.csv")
    ref = store.repertoire.import_receptors(
        path, cell_selection=cells, barcode_prefix="cell"
    )
    loaded = store.repertoire.load_receptors(ref)
    assert loaded.cells.loc["cell0", "VDJ_1_junction_aa"] == "CASSF"
    assert store.artifacts.inspect(ref).inputs["sources"][0]["barcode_prefix"] == "cell"
    both = store.repertoire.import_receptors(
        [path, _bcr(tmp_path)], cell_selection=cells, barcode_prefix=["cell", None]
    )
    assert (
        store.repertoire.load_receptors(both).cells.loc["cell4", "receptor_type"]
        == "BCR"
    )
    with pytest.raises(ValueError, match="lists 1 prefixes for 2 contig files"):
        store.repertoire.import_receptors(
            [path, _bcr(tmp_path)], cell_selection=cells, barcode_prefix=["cell"]
        )


def test_a_cell_selection_limits_the_imported_cells(tmp_path) -> None:
    store = _store(tmp_path)
    keep = np.zeros(N_CELLS, dtype=bool)
    keep[[0, 4]] = True
    store.cells.insert("chosen", keep)
    ref = store.repertoire.import_receptors(
        [_tcr(tmp_path), _bcr(tmp_path)],
        cell_selection=store.snapshot_cell_selection("chosen"),
    )
    loaded = store.repertoire.load_receptors(ref)
    assert loaded.cells.index.tolist() == ["cell0", "cell4"]
    assert store.artifacts.load(ref).attrs["n_unmatched_contigs"] == 10
    values = store.artifacts.load_values(ref, value="chain_pairing")
    assert values.to_pandas().tolist() == ["ambiguous", "single pair"]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (TCR_ROWS + [TCR_ROWS[0]], "Contig ID 'cell0_contig_1' repeats"),
        ([TCR_ROWS[0].replace("cell0_contig_1", "")], "have no sequence_id"),
    ],
)
def test_contig_ids_are_required_and_unique_within_a_file(
    tmp_path, rows, message
) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match=message):
        store.repertoire.import_receptors(
            _tcr(tmp_path, rows), cell_selection=store.snapshot_cell_selection()
        )
    assert store.artifacts.list(kind="immune_receptors", scope="datastore") == []


def test_one_contig_id_may_appear_in_two_files(tmp_path) -> None:
    store = _store(tmp_path)
    # cell0_contig_1 is the ID of an alpha chain and of a heavy chain.
    ref = store.repertoire.import_receptors(
        [_tcr(tmp_path), _bcr(tmp_path)], cell_selection=store.snapshot_cell_selection()
    )
    chains = store.repertoire.load_receptors(
        ref, fields=["sequence_id", "locus"]
    ).chains
    assert (chains["sequence_id"] == "cell0_contig_1").sum() == 2


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"paths": 3}, TypeError, "paths must be a path"),
        ({"paths": []}, ValueError, "at least one contig file"),
        ({"paths": [3]}, TypeError, "paths must be a path"),
        ({"barcode_prefix": 3}, TypeError, "barcode_prefix must be text"),
        ({"barcode_prefix": [3]}, TypeError, "barcode_prefix must be text"),
        ({"productive_only": 1}, TypeError, "productive_only must be a boolean"),
        ({"invalidate_cache": "yes"}, TypeError, "invalidate_cache must be a boolean"),
        ({"cell_selection": "I"}, TypeError, "cell_selection must be an ArtifactRef"),
    ],
)
def test_import_arguments_are_checked(tmp_path, kwargs, error, message) -> None:
    store = _store(tmp_path)
    arguments = {
        "paths": _tcr(tmp_path),
        "cell_selection": store.snapshot_cell_selection(),
    }
    arguments.update(kwargs)
    with pytest.raises(error, match=message):
        store.repertoire.import_receptors(arguments.pop("paths"), **arguments)


def test_a_file_listed_twice_is_refused(tmp_path) -> None:
    store = _store(tmp_path)
    path = _tcr(tmp_path)
    with pytest.raises(ValueError, match="listed twice"):
        store.repertoire.import_receptors(
            [path, path], cell_selection=store.snapshot_cell_selection()
        )


def test_an_unknown_file_format_is_refused(tmp_path) -> None:
    store = _store(tmp_path)
    other = _write(tmp_path, "other.csv", "a,b", ["1,2"])
    with pytest.raises(ValueError, match="neither an AIRR Rearrangement file"):
        store.repertoire.import_receptors(
            other, cell_selection=store.snapshot_cell_selection()
        )


def test_import_refuses_an_empty_cell_selection(tmp_path) -> None:
    store = _store(tmp_path)
    store.cells.insert("nobody", np.zeros(N_CELLS, dtype=bool))
    with pytest.raises(ValueError, match="selects no cell"):
        store.repertoire.import_receptors(
            _tcr(tmp_path), cell_selection=store.snapshot_cell_selection("nobody")
        )


def test_import_refuses_contigs_larger_than_the_memory_budget(
    tmp_path, monkeypatch
) -> None:
    store = _store(tmp_path)
    cells = store.snapshot_cell_selection()
    monkeypatch.setattr(store, "memoryBytes", 1_000)
    with pytest.raises(MemoryError, match="Raise mem_budget"):
        store.repertoire.import_receptors(_tcr(tmp_path), cell_selection=cells)
    assert store.artifacts.list(kind="immune_receptors", scope="datastore") == []


def test_a_read_only_store_refuses_to_import_new_receptors(tmp_path) -> None:
    store = _store(tmp_path)
    cells = store.snapshot_cell_selection()
    saved = store.repertoire.import_receptors(_tcr(tmp_path), cell_selection=cells)
    reader = _store(tmp_path, zarr_mode="r")
    assert (
        reader.repertoire.import_receptors(_tcr(tmp_path), cell_selection=cells)
        == saved
    )
    with pytest.raises(PermissionError, match="import_immune_receptors"):
        reader.repertoire.import_receptors(_bcr(tmp_path), cell_selection=cells)


def test_loading_reads_the_requested_fields(tmp_path) -> None:
    store = _store(tmp_path)
    ref = store.repertoire.import_receptors(
        _tcr(tmp_path), cell_selection=store.snapshot_cell_selection()
    )
    loaded = store.repertoire.load_receptors(ref, fields=["junction_aa", "umi_count"])
    assert list(loaded.chains.columns) == [
        "ids",
        "slot",
        "chain",
        "junction_aa",
        "umi_count",
    ]
    # The table of cells holds the loaded slot fields.
    assert [name for name in loaded.cells.columns if name.startswith("VDJ_1")] == [
        "VDJ_1_junction_aa",
        "VDJ_1_umi_count",
    ]
    assert "raw_clonotype_id" in loaded.fields
    # Text has the string dtype, also in a field that no contig holds.
    without_d = store.repertoire.import_receptors(
        _tcr(tmp_path, [row.replace("TRBD1", "") for row in TCR_ROWS], "no_d.csv"),
        cell_selection=store.snapshot_cell_selection(),
    )
    calls = store.repertoire.load_receptors(
        without_d, fields=["v_call", "d_call"]
    ).chains
    assert calls["v_call"].dtype == "string" and calls["d_call"].dtype == "string"
    assert calls["d_call"].isna().all()
    empty = store.repertoire.load_receptors(ref, fields=[])
    assert list(empty.chains.columns) == ["ids", "slot", "chain"]
    assert list(empty.cells.columns) == [
        "receptor_type",
        "receptor_subtype",
        "chain_pairing",
        "multichain",
        "n_vj",
        "n_vdj",
        "n_contigs",
    ]


@pytest.mark.parametrize(
    ("fields", "error", "message"),
    [
        (["sequence"], ValueError, "store no field 'sequence'"),
        ("junction_aa", TypeError, "fields must be a sequence"),
    ],
)
def test_loading_refuses_fields_that_are_not_stored(
    tmp_path, fields, error, message
) -> None:
    store = _store(tmp_path)
    ref = store.repertoire.import_receptors(
        _tcr(tmp_path), cell_selection=store.snapshot_cell_selection()
    )
    with pytest.raises(error, match=message):
        store.repertoire.load_receptors(ref, fields=fields)


def test_loading_refuses_tables_larger_than_the_memory_budget(
    tmp_path, monkeypatch
) -> None:
    store = _store(tmp_path)
    ref = store.repertoire.import_receptors(
        _tcr(tmp_path), cell_selection=store.snapshot_cell_selection()
    )
    monkeypatch.setattr(store, "memoryBytes", 100)
    with pytest.raises(MemoryError, match="Pass fields="):
        store.repertoire.load_receptors(ref)


def test_loading_refuses_other_artifact_kinds(tmp_path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="immune_receptors artifact"):
        store.repertoire.load_receptors(store.snapshot_cell_selection())
    with pytest.raises(TypeError, match="ArtifactRef"):
        store.repertoire.load_receptors("receptors")


def test_receptors_select_cells_like_labels(tmp_path) -> None:
    store = _store(tmp_path)
    ref = store.repertoire.import_receptors(
        [_tcr(tmp_path), _bcr(tmp_path)], cell_selection=store.snapshot_cell_selection()
    )
    t_cells = store.select_cells(ref, include=["TCR"])
    selected = store.artifacts.load_values(ref, cell_selection=t_cells)
    assert selected.cell_ids.tolist() == ["cell1", "cell2"]
    paired = store.select_cells(ref, value="chain_pairing", include=["single pair"])
    assert store.artifacts.load_values(
        ref, cell_selection=paired
    ).cell_ids.tolist() == ["cell4"]
    many = store.select_cells(ref, value="n_vdj", low=2, keep_bounds=True)
    assert store.artifacts.load_values(ref, cell_selection=many).cell_ids.tolist() == [
        "cell0",
        "cell3",
    ]


def test_loading_refuses_a_corrupt_list_of_fields(tmp_path) -> None:
    store = _store(tmp_path)
    ref = store.repertoire.import_receptors(
        _tcr(tmp_path), cell_selection=store.snapshot_cell_selection()
    )
    from cytearc.storage.artifacts import artifact_group

    artifact_group(store.zw, ref).attrs["fields"] = ["not_a_field"]
    with pytest.raises(ValueError, match="no valid list of fields"):
        store.repertoire.load_receptors(ref)
