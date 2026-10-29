import gzip
import io

import numpy as np
import pandas as pd
import pytest

import cytearc.datastore._operations.repertoire as repertoire_operations
from cytearc.repertoire.contigs import encode_text
from cytearc.writers import write_airr_rearrangements
from cytearc.writers.airr import (
    AIRR_REQUIRED_FIELDS,
    AirrRearrangementWriter,
    sequence_id_hashes,
    suffix_repeated_ids,
)
from tests.test_repertoire_clonotypes_datastore import CLONAL_ROWS, _receptors
from tests.test_repertoire_import import (
    BCR_ROWS,
    CELLRANGER_HEADER,
    TCR_ROWS,
    _store,
    _write,
)

# Contig IDs repeat across the three files: X takes _3 and then _4, since the
# TCR file names a contig X_2, and Z takes _3, or _2 when only chains are
# written, since the contig named Z_2 is not a chain. Values hold a tab, a
# double quote, a newline and text beyond ASCII, and some are missing.
EDGE_TCR = [
    'cell0,true,X,true,500,TRA,"TRAV1\tB",,TRAJ1,TRAC,true,true,CAVA,TGTGCTGTGGCC,90,9,"ct""1",ct1_1',
    "cell0,true,X_2,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,CASSA,TGTGCCAGCAGCGCC,80,8,ct1,ct1_2",
    'cell1,true,Y,true,500,TRB,TRBVé,,TRBJ2,"TR\nBC1",true,true,CASSB,TGTGCCAGCAGCGCA,80,8,ct2,ct2_1',
    "cell2,true,Z,true,500,TRA,TRAV2,,TRAJ2,,true,true,CAVB,TGTGCTGTGGCA,60,6,ct3,ct3_1",
]
EDGE_BCR = [
    "cell0,true,X,true,500,IGH,IGHV1,,IGHJ1,IGHM,true,true,CARW,TGTGCGAGATGG,500,20,cb1,cb1_1",
    "cell3,true,Z,true,500,IGH,IGHV2,,IGHJ2,IGHM,true,true,CARX,TGTGCGAGATGA,400,,cb2,cb2_1",
    "cell4,true,Z_2,true,500,IGK,IGKV1,,IGKJ1,IGKC,true,false,CQQY,TGTCAGCAGTAT,400,15,cb3,cb3_1",
]
EDGE_AIRR_HEADER = (
    "cell_id\tsequence_id\tv_call\td_call\tj_call\tc_call\tjunction\tjunction_aa"
    "\tproductive\trev_comp\tumi_count\tv_identity\td_identity\tj_identity"
    "\tsequence\tv_cigar"
)
EDGE_AIRR = [
    "cell5\tcell5_contig_1\tIGHV3*01\t\tIGHJ4*02\tIGHG1\tTGTGCG\tCARW\tT\tF\t12\t1\t0.25\t\tACGTACGT\t5S10M",
    "cell5\tcell5_contig_2\tIGLV1*01\t\tIGLJ2*01\tIGLC2\tTGTCAG\tCQSY\tT\t\t\t0\t\t\t\t",
    "cell6\tX\tTRBV5*01\t\tTRBJ1*01\tTRBC1\tTGTGCC\tCASS\tT\tT\t3\t1\t0.5\t\tGG\t",
]
EDGE_FIELDS = [
    "c_call",
    "umi_count",
    "rev_comp",
    "v_identity",
    "d_identity",
    "j_identity",
    "sequence",
    "raw_clonotype_id",
]
# The edge receptors' files as the export wrote them before it streamed its
# contigs, which loaded every contig into one table and wrote it with
# DataFrame.to_csv. v_identity holds whole numbers only, so it is written as
# integers.
EXPECTED_EDGE = (
    "sequence_id\tsequence\trev_comp\tproductive\tv_call\td_call\tj_call"
    "\tsequence_alignment\tgermline_alignment\tjunction\tjunction_aa\tv_cigar"
    "\td_cigar\tj_cigar\tcell_id\tc_call\tumi_count\tv_identity\td_identity"
    "\tj_identity\traw_clonotype_id\n"
    "X\t\t\t\t\t\t\t\t\t\t\t\t\t\tcell0\tIGHM\t20\t\t\t\tcb1\n"
    'X_3\t\t\t\t\t\t\t\t\t\t\t\t\t\tcell0\tTRAC\t9\t\t\t\t"ct""1"\n'
    "X_2\t\t\t\t\t\t\t\t\t\t\t\t\t\tcell0\tTRBC1\t8\t\t\t\tct1\n"
    'Y\t\t\t\t\t\t\t\t\t\t\t\t\t\tcell1\t"TR\nBC1"\t8\t\t\t\tct2\n'
    "Z\t\t\t\t\t\t\t\t\t\t\t\t\t\tcell2\t\t6\t\t\t\tct3\n"
    "Z_3\t\t\t\t\t\t\t\t\t\t\t\t\t\tcell3\tIGHM\t\t\t\t\tcb2\n"
    "Z_2\t\t\t\t\t\t\t\t\t\t\t\t\t\tcell4\tIGKC\t15\t\t\t\tcb3\n"
    "cell5_contig_1\tACGTACGT\tF\t\t\t\t\t\t\t\t\t\t\t\tcell5\tIGHG1\t12\t1\t0.25\t\t\n"
    "cell5_contig_2\t\t\t\t\t\t\t\t\t\t\t\t\t\tcell5\tIGLC2\t\t0\t\t\t\n"
    "X_4\tGG\tT\t\t\t\t\t\t\t\t\t\t\t\tcell6\tTRBC1\t3\t1\t0.5\t\t\n"
)
EXPECTED_EDGE_CHAINS = (
    "sequence_id\tsequence\trev_comp\tproductive\tv_call\td_call\tj_call"
    "\tsequence_alignment\tgermline_alignment\tjunction\tjunction_aa\tv_cigar"
    "\td_cigar\tj_cigar\tcell_id\tc_call\tumi_count\tv_identity\td_identity"
    "\tj_identity\traw_clonotype_id\n"
    "X\t\t\t\t\t\t\t\t\t\t\t\t\t\tcell0\tIGHM\t20\t\t\t\tcb1\n"
    'X_3\t\t\t\t\t\t\t\t\t\t\t\t\t\tcell0\tTRAC\t9\t\t\t\t"ct""1"\n'
    "X_2\t\t\t\t\t\t\t\t\t\t\t\t\t\tcell0\tTRBC1\t8\t\t\t\tct1\n"
    'Y\t\t\t\t\t\t\t\t\t\t\t\t\t\tcell1\t"TR\nBC1"\t8\t\t\t\tct2\n'
    "Z\t\t\t\t\t\t\t\t\t\t\t\t\t\tcell2\t\t6\t\t\t\tct3\n"
    "Z_2\t\t\t\t\t\t\t\t\t\t\t\t\t\tcell3\tIGHM\t\t\t\t\tcb2\n"
    "cell5_contig_1\tACGTACGT\tF\t\t\t\t\t\t\t\t\t\t\t\tcell5\tIGHG1\t12\t1\t0.25\t\t\n"
    "cell5_contig_2\t\t\t\t\t\t\t\t\t\t\t\t\t\tcell5\tIGLC2\t\t0\t\t\t\n"
    "X_4\tGG\tT\t\t\t\t\t\t\t\t\t\t\t\tcell6\tTRBC1\t3\t1\t0.5\t\t\n"
)
# Each {cell} is that cell's clonotype label, empty for cell4, which has none.
EXPECTED_EDGE_CLONOTYPES = (
    "sequence_id\tsequence\trev_comp\tproductive\tv_call\td_call\tj_call"
    "\tsequence_alignment\tgermline_alignment\tjunction\tjunction_aa\tv_cigar"
    "\td_cigar\tj_cigar\tcell_id\tclone_id\n"
    "X\t\t\t\tIGHV1\t\t\t\t\t\t\t\t\t\tcell0\t{cell0}\n"
    'X_3\t\t\t\t"TRAV1\tB"\t\t\t\t\t\t\t\t\t\tcell0\t{cell0}\n'
    "X_2\t\t\t\tTRBV1\t\t\t\t\t\t\t\t\t\tcell0\t{cell0}\n"
    "Y\t\t\t\tTRBVé\t\t\t\t\t\t\t\t\t\tcell1\t{cell1}\n"
    "Z\t\t\t\tTRAV2\t\t\t\t\t\t\t\t\t\tcell2\t{cell2}\n"
    "Z_3\t\t\t\tIGHV2\t\t\t\t\t\t\t\t\t\tcell3\t{cell3}\n"
    "Z_2\t\t\t\tIGKV1\t\t\t\t\t\t\t\t\t\tcell4\t\n"
    "cell5_contig_1\t\t\t\tIGHV3*01\t\t\t\t\t\t\t\t\t\tcell5\t{cell5}\n"
    "cell5_contig_2\t\t\t\tIGLV1*01\t\t\t\t\t\t\t\t\t\tcell5\t{cell5}\n"
    "X_4\t\t\t\tTRBV5*01\t\t\t\t\t\t\t\t\t\tcell6\t{cell6}\n"
)
EXPECTED_EDGE_ALL = (
    "sequence_id\tsequence\trev_comp\tproductive\tv_call\td_call\tj_call"
    "\tsequence_alignment\tgermline_alignment\tjunction\tjunction_aa\tv_cigar"
    "\td_cigar\tj_cigar\tcell_id\tlocus\tc_call\tumi_count\tconsensus_count"
    "\tis_cell\thigh_confidence\tfull_length\tv_identity\td_identity\tj_identity"
    "\traw_clonotype_id\traw_consensus_id\n"
    "X\t\t\tT\tIGHV1\t\tIGHJ1\t\t\tTGTGCGAGATGG\tCARW\t\t\t\tcell0\tIGH\tIGHM"
    "\t20\t500\tT\tT\tT\t\t\t\tcb1\tcb1_1\n"
    'X_3\t\t\tT\t"TRAV1\tB"\t\tTRAJ1\t\t\tTGTGCTGTGGCC\tCAVA\t\t\t\tcell0\tTRA'
    '\tTRAC\t9\t90\tT\tT\tT\t\t\t\t"ct""1"\tct1_1\n'
    "X_2\t\t\tT\tTRBV1\t\tTRBJ1\t\t\tTGTGCCAGCAGCGCC\tCASSA\t\t\t\tcell0\tTRB"
    "\tTRBC1\t8\t80\tT\tT\tT\t\t\t\tct1\tct1_2\n"
    "Y\t\t\tT\tTRBVé\t\tTRBJ2\t\t\tTGTGCCAGCAGCGCA\tCASSB\t\t\t\tcell1\tTRB"
    '\t"TR\nBC1"\t8\t80\tT\tT\tT\t\t\t\tct2\tct2_1\n'
    "Z\t\t\tT\tTRAV2\t\tTRAJ2\t\t\tTGTGCTGTGGCA\tCAVB\t\t\t\tcell2\tTRA\t\t6"
    "\t60\tT\tT\tT\t\t\t\tct3\tct3_1\n"
    "Z_3\t\t\tT\tIGHV2\t\tIGHJ2\t\t\tTGTGCGAGATGA\tCARX\t\t\t\tcell3\tIGH\tIGHM"
    "\t\t400\tT\tT\tT\t\t\t\tcb2\tcb2_1\n"
    "Z_2\t\t\tF\tIGKV1\t\tIGKJ1\t\t\tTGTCAGCAGTAT\tCQQY\t\t\t\tcell4\tIGK\tIGKC"
    "\t15\t400\tT\tT\tT\t\t\t\tcb3\tcb3_1\n"
    "cell5_contig_1\tACGTACGT\tF\tT\tIGHV3*01\t\tIGHJ4*02\t\t\tTGTGCG\tCARW"
    "\t5S10M\t\t\tcell5\tIGH\tIGHG1\t12\t\t\t\t\t1\t0.25\t\t\t\n"
    "cell5_contig_2\t\t\tT\tIGLV1*01\t\tIGLJ2*01\t\t\tTGTCAG\tCQSY\t\t\t\tcell5"
    "\tIGL\tIGLC2\t\t\t\t\t\t0\t\t\t\t\n"
    "X_4\tGG\tT\tT\tTRBV5*01\t\tTRBJ1*01\t\t\tTGTGCC\tCASS\t\t\t\tcell6\tTRB"
    "\tTRBC1\t3\t\t\t\t\t1\t0.5\t\t\t\n"
)


def _edge_receptors(tmp_path):
    store = _store(tmp_path)
    files = [
        _write(tmp_path, "tcr.csv", CELLRANGER_HEADER, EDGE_TCR),
        _write(tmp_path, "bcr.csv", CELLRANGER_HEADER, EDGE_BCR),
        _write(tmp_path, "airr.tsv", EDGE_AIRR_HEADER, EDGE_AIRR),
    ]
    receptors = store.repertoire.import_receptors(
        files, cell_selection=store.snapshot_cell_selection()
    )
    return store, receptors


def _text(path) -> str:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as handle:
        return handle.read().decode("utf-8")


def _read(path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def test_the_writer_puts_required_fields_first_and_renders_values(tmp_path) -> None:
    table = pd.DataFrame(
        {
            "cell_id": ["c1", "c2"],
            "sequence_id": ["s1", "s2"],
            "productive": pd.array([True, None], dtype="boolean"),
            "umi_count": pd.array([3, None], dtype="Int64"),
            "v_identity": [0.95, np.nan],
            "junction_aa": pd.array(["CASSF", None], dtype="string"),
        }
    )
    path = tmp_path / "out.tsv"
    assert write_airr_rearrangements(table, path) == 2
    written = _read(path)
    assert written.columns.tolist() == [
        *AIRR_REQUIRED_FIELDS,
        "cell_id",
        "umi_count",
        "v_identity",
    ]
    assert written["productive"].tolist() == ["T", ""]
    assert written["umi_count"].tolist() == ["3", ""]
    assert written["v_identity"].tolist() == ["0.95", ""]
    assert written["junction_aa"].tolist() == ["CASSF", ""]
    assert written["sequence"].tolist() == ["", ""]
    chosen = tmp_path / "chosen.tsv.gz"
    assert (
        write_airr_rearrangements(table, chosen, columns=["umi_count", "junction_aa"])
        == 2
    )
    with gzip.open(chosen, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
    assert header == [*AIRR_REQUIRED_FIELDS, "umi_count"]
    with pytest.raises(TypeError, match="path must be a path"):
        write_airr_rearrangements(table, 3)
    with pytest.raises(ValueError, match="need a sequence_id"):
        write_airr_rearrangements(table.drop(columns="sequence_id"), path)
    with pytest.raises(ValueError, match="have no column"):
        write_airr_rearrangements(table, path, columns=["c_call"])


def test_the_writer_reads_logical_text_and_whole_counts(tmp_path) -> None:
    # A table as pandas reads an AIRR file: logical fields as text, and counts
    # as real numbers when one is missing.
    table = pd.read_csv(
        io.StringIO(
            "sequence_id\tproductive\trev_comp\tumi_count\tv_identity\n"
            "c1\tT\tFALSE\t3\t0.95\n"
            "c2\tF\ttrue\t\t1\n"
            "c3\t\t0\t12\t1\n"
        ),
        sep="\t",
    )
    path = tmp_path / "read.tsv"
    write_airr_rearrangements(table, path)
    written = _read(path)
    assert written["productive"].tolist() == ["T", "F", ""]
    assert written["rev_comp"].tolist() == ["F", "T", "F"]
    assert written["umi_count"].tolist() == ["3", "", "12"]
    assert written["v_identity"].tolist() == ["0.95", "1.0", "1.0"]
    numbers = pd.DataFrame(
        {"sequence_id": ["c1", "c2"], "productive": [1, 0], "is_cell": [1.0, np.nan]}
    )
    write_airr_rearrangements(numbers, path)
    assert _read(path)[["productive", "is_cell"]].values.tolist() == [
        ["T", "T"],
        ["F", ""],
    ]
    with pytest.raises(ValueError, match="productive must hold logical values"):
        write_airr_rearrangements(table.assign(productive=["T", "maybe", "F"]), path)
    with pytest.raises(ValueError, match="productive must hold logical values"):
        write_airr_rearrangements(numbers.assign(productive=[2, 0]), path)


def test_export_writes_each_contig_with_its_cell_and_clonotype(tmp_path) -> None:
    store, receptors = _receptors(tmp_path, CLONAL_ROWS)
    clonotypes = store.repertoire.clonotypes(receptors)
    path = tmp_path / "export.tsv"
    assert store.repertoire.export_airr(clonotypes, path) == len(CLONAL_ROWS)
    written = _read(path)
    assert written.columns[: len(AIRR_REQUIRED_FIELDS)].tolist() == list(
        AIRR_REQUIRED_FIELDS
    )
    assert written.columns[
        len(AIRR_REQUIRED_FIELDS) : len(AIRR_REQUIRED_FIELDS) + 5
    ].tolist() == [
        "cell_id",
        "clone_id",
        "locus",
        "c_call",
        "umi_count",
    ]
    by_contig = written.set_index("sequence_id")
    assert by_contig.loc["cell0_contig_1", "cell_id"] == "cell0"
    assert by_contig.loc["cell0_contig_1", "clone_id"] == "clonotype_1"
    assert by_contig.loc["cell5_contig_2", "clone_id"] == "clonotype_2"
    assert by_contig.loc["cell0_contig_1", "productive"] == "T"
    assert by_contig.loc["cell0_contig_1", "junction"] == "TGTGCTGTGGCC"
    plain = store.repertoire.export_airr(
        receptors, tmp_path / "plain.tsv", fields=["junction_aa", "v_call"]
    )
    assert plain == len(CLONAL_ROWS)
    columns = _read(tmp_path / "plain.tsv").columns.tolist()
    assert "clone_id" not in columns and "umi_count" not in columns
    assert columns[len(AIRR_REQUIRED_FIELDS) :] == ["cell_id"]


def test_export_repeats_round_trip_through_an_import(tmp_path) -> None:
    store = _store(tmp_path)
    cells = store.snapshot_cell_selection()
    tcr = _write(tmp_path, "tcr.csv", CELLRANGER_HEADER, TCR_ROWS)
    bcr = _write(tmp_path, "bcr.csv", CELLRANGER_HEADER, BCR_ROWS)
    receptors = store.repertoire.import_receptors([tcr, bcr], cell_selection=cells)
    path = tmp_path / "both.tsv.gz"
    written = store.repertoire.export_airr(receptors, path)
    exported = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    assert written == len(exported)
    # Contig IDs that the TCR and BCR files share get a suffix, so every
    # sequence_id is unique.
    assert exported["sequence_id"].is_unique
    again = store.repertoire.import_receptors(path, cell_selection=cells)
    first = store.repertoire.load_receptors(receptors).cells
    second = store.repertoire.load_receptors(again).cells
    for column in (
        "receptor_type",
        "receptor_subtype",
        "chain_pairing",
        "VDJ_1_junction_aa",
    ):
        pd.testing.assert_series_equal(first[column], second[column], check_names=False)


def test_export_suffixes_never_repeat_an_id(tmp_path) -> None:
    store = _store(tmp_path)
    # The TCR file already names a contig X_2, the suffix that the repeated X
    # of the BCR file would take.
    tcr = _write(
        tmp_path,
        "tcr.csv",
        CELLRANGER_HEADER,
        [
            "cell0,true,X,true,500,TRA,TRAV1,,TRAJ1,TRAC,true,true,CAVA,TGTGCTGTGGCC,90,9,ct1,ct1_1",
            "cell0,true,X_2,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,CASSA,TGTGCCAGCAGCGCC,80,8,ct1,ct1_2",
        ],
    )
    bcr = _write(
        tmp_path,
        "bcr.csv",
        CELLRANGER_HEADER,
        [
            "cell1,true,X,true,500,IGH,IGHV1,,IGHJ1,IGHM,true,true,CARW,TGTGCGAGATGG,500,20,cb1,cb1_1",
        ],
    )
    receptors = store.repertoire.import_receptors(
        [tcr, bcr], cell_selection=store.snapshot_cell_selection()
    )
    path = tmp_path / "ids.tsv"
    assert store.repertoire.export_airr(receptors, path) == 3
    assert sorted(_read(path)["sequence_id"]) == ["X", "X_2", "X_3"]


def test_export_keeps_only_chains_on_request(tmp_path) -> None:
    rows = CLONAL_ROWS + [
        "cell7,true,cell7_contig_1,true,500,TRB,TRBV9,,TRBJ9,TRBC1,true,false,CASSQ,TGTGCCAGCAGCCAG,10,1,ct9,ct9_1",
    ]
    store, receptors = _receptors(tmp_path, rows)
    everything = store.repertoire.export_airr(receptors, tmp_path / "all.tsv")
    chains = store.repertoire.export_airr(
        receptors, tmp_path / "chains.tsv", chains_only=True
    )
    assert everything == chains + 1
    assert "cell7_contig_1" not in set(_read(tmp_path / "chains.tsv")["sequence_id"])


def test_export_refuses_other_sources_and_arguments(tmp_path) -> None:
    store, receptors = _receptors(tmp_path, CLONAL_ROWS)
    clonotypes = store.repertoire.clonotypes(receptors)
    expansion = store.repertoire.clonal_expansion(clonotypes)
    with pytest.raises(ValueError, match="immune_receptors or clonotypes artifact"):
        store.repertoire.export_airr(expansion, tmp_path / "x.tsv")
    with pytest.raises(TypeError, match="source must be an ArtifactRef"):
        store.repertoire.export_airr("receptors", tmp_path / "x.tsv")
    with pytest.raises(TypeError, match="path must be a path"):
        store.repertoire.export_airr(receptors, 5)
    with pytest.raises(TypeError, match="chains_only must be a boolean"):
        store.repertoire.export_airr(receptors, tmp_path / "x.tsv", chains_only=1)
    with pytest.raises(ValueError, match="store no field"):
        store.repertoire.export_airr(receptors, tmp_path / "x.tsv", fields=["sequence"])
    with pytest.raises(TypeError, match="fields must be a sequence"):
        store.repertoire.export_airr(receptors, tmp_path / "x.tsv", fields="v_call")


def test_the_writer_writes_a_file_one_block_at_a_time(tmp_path) -> None:
    path = tmp_path / "blocks.tsv.gz"
    # Text stored once per distinct value: c1 and c2, coded 0 and 1.
    cells = encode_text(["c2", "c1", None])
    with AirrRearrangementWriter(
        path, ["cell_id", "umi_count", "v_identity"], whole={"v_identity"}
    ) as writer:
        assert writer.columns == (
            *AIRR_REQUIRED_FIELDS,
            "cell_id",
            "umi_count",
            "v_identity",
        )
        frame = pd.DataFrame(
            {
                "sequence_id": ["a"],
                "cell_id": ["c0"],
                "umi_count": pd.array([3], dtype="Int64"),
                "v_identity": [2.0],
                "junction": ["TGT"],
            }
        )
        assert writer.write(frame) == 1
        block = {
            "sequence_id": np.array(["b", "c", "d"], dtype=object),
            "cell_id": cells,
            "umi_count": pd.array([None, -7, 2**63 - 1], dtype="Int64"),
            "v_identity": np.array([np.nan, 1.0, -3.0]),
            "productive": pd.array([True, None, False], dtype="boolean"),
        }
        assert writer.write(block) == 3
        empty = {name: np.empty(0) for name in ("sequence_id", *writer.columns[-3:])}
        assert writer.write(empty) == 0
        assert writer.n_rows == 4 and not writer.closed
    assert writer.closed
    writer.close()
    with pytest.raises(ValueError, match="writer is closed"):
        writer.write(block)
    written = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    assert written.columns.tolist() == list(writer.columns)
    assert written["sequence_id"].tolist() == ["a", "b", "c", "d"]
    assert written["cell_id"].tolist() == ["c0", "c2", "c1", ""]
    assert written["umi_count"].tolist() == ["3", "", "-7", "9223372036854775807"]
    assert written["v_identity"].tolist() == ["2", "", "1", "-3"]
    assert written["productive"].tolist() == ["", "T", "", "F"]
    assert written["junction"].tolist() == ["TGT", "", "", ""]
    assert written["sequence"].tolist() == ["", "", "", ""]


def test_the_writer_refuses_blocks_without_their_columns(tmp_path) -> None:
    with pytest.raises(TypeError, match="path must be a path"):
        AirrRearrangementWriter(3)
    path = tmp_path / "refused.tsv"
    with AirrRearrangementWriter(path, ["cell_id"]) as writer:
        with pytest.raises(ValueError, match="need a sequence_id"):
            writer.write({"cell_id": ["c1"]})
        with pytest.raises(ValueError, match=r"have no column \['cell_id'\]"):
            writer.write({"sequence_id": ["s1"]})
        with pytest.raises(ValueError, match="differ in length"):
            writer.write({"sequence_id": ["s1", "s2"], "cell_id": ["c1"]})
    assert path.read_text() == "\t".join([*AIRR_REQUIRED_FIELDS, "cell_id"]) + "\n"


def test_the_writer_quotes_values_as_pandas_to_csv_does(tmp_path) -> None:
    path = tmp_path / "quoted.tsv"
    ids = np.array(["a\tb", 'q"x', "l\nm", "r\rs", "é"], dtype=object)
    with AirrRearrangementWriter(path, ["cell_id", "c_call"]) as writer:
        writer.write(
            {
                "sequence_id": ids,
                "cell_id": encode_text(["c1", None, "c1", "c2", "c2"]),
                "c_call": ids,
            }
        )
    empty = "\t" * (len(AIRR_REQUIRED_FIELDS) - 1)
    rows = path.read_bytes().decode("utf-8").split("\n", 1)[1]
    assert rows == (
        f'"a\tb"{empty}\tc1\t"a\tb"\n'
        f'"q""x"{empty}\t\t"q""x"\n'
        f'"l\nm"{empty}\tc1\t"l\nm"\n'
        f'"r\rs"{empty}\tc2\t"r\rs"\n'
        f"é{empty}\tc2\té\n"
    )
    table = pd.DataFrame(
        {
            "sequence_id": ids,
            **{name: "" for name in AIRR_REQUIRED_FIELDS[1:]},
            "cell_id": ["c1", "", "c1", "c2", "c2"],
            "c_call": ids,
        }
    )
    assert rows == table.to_csv(
        sep="\t", index=False, header=False, lineterminator="\n"
    )


def test_the_writer_renders_each_kind_of_column(tmp_path) -> None:
    path = tmp_path / "kinds.tsv"
    columns = ["a", "b", "c", "d", "e", "f", "g", "is_cell", "stop_codon"]
    with AirrRearrangementWriter(path, columns) as writer:
        writer.write(
            {
                "sequence_id": np.array(["s1", "s2", "s3"]),
                "a": np.array([-(2**63), 0, 9], dtype=np.int64),
                "b": np.array([1, 200, 65535], dtype=np.uint16),
                "c": np.array(["x", None, np.nan], dtype=object),
                "d": pd.Categorical(["p", None, "q"]),
                "e": np.array([0.5, np.nan, 1.0], dtype=np.float32),
                "f": np.array([True, False, True]),
                "g": np.array([2**64 - 1, 0, 1], dtype=np.uint64),
                "is_cell": np.array([True, False, True]),
                "stop_codon": encode_text(["TRUE", None, "0"]),
                "productive": pd.Series(["t", "F", ""]),
            }
        )
    written = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    assert written["a"].tolist() == ["-9223372036854775808", "0", "9"]
    assert written["b"].tolist() == ["1", "200", "65535"]
    assert written["c"].tolist() == ["x", "", ""]
    assert written["d"].tolist() == ["p", "", "q"]
    assert written["e"].tolist() == ["0.5", "", "1.0"]
    assert written["f"].tolist() == ["True", "False", "True"]
    assert written["g"].tolist() == ["18446744073709551615", "0", "1"]
    assert written["is_cell"].tolist() == ["T", "F", "T"]
    assert written["stop_codon"].tolist() == ["T", "", "F"]
    assert written["productive"].tolist() == ["T", "F", ""]


def test_repeated_ids_take_the_next_free_suffix() -> None:
    ids = np.array(["A_2", "A", "B", "A", "A", "B", "A_3"], dtype=object)
    asked = []

    def ids_at(rows):
        asked.append(rows.tolist())
        return ids[rows]

    rows, names = suffix_repeated_ids(sequence_id_hashes(ids), ids_at)
    # A_2 and A_3 name other contigs, and A_4 the third A.
    assert rows.tolist() == [3, 4, 5]
    assert names.tolist() == ["A_4", "A_5", "B_2"]
    # The repeated IDs are read, then the IDs that a suffixed ID's hash names.
    assert asked == [[1, 2, 3, 4, 5], [0, 6]]

    def unread(rows):
        raise AssertionError("no ID repeats")

    unique = suffix_repeated_ids(sequence_id_hashes(["a", "b"]), unread)
    assert unique[0].tolist() == [] and unique[1].tolist() == []
    # IDs whose hashes collide stay apart.
    same = np.zeros(3, dtype=np.uint64)
    rows, names = suffix_repeated_ids(
        same, lambda rows: np.array(["a", "b", "a"])[rows]
    )
    assert rows.tolist() == [2] and names.tolist() == ["a_2"]
    rows, names = suffix_repeated_ids(same[:2], lambda rows: np.array(["a", "b"])[rows])
    assert rows.tolist() == [] and names.tolist() == []


@pytest.mark.parametrize("block_bytes", [None, 1])
def test_streamed_exports_match_the_exports_of_one_table(
    tmp_path, monkeypatch, block_bytes
) -> None:
    store, receptors = _edge_receptors(tmp_path)
    clonotypes = store.repertoire.clonotypes(receptors)
    if block_bytes is not None:
        # One contig per block, so the repeats of each ID fall in other blocks.
        monkeypatch.setattr(repertoire_operations, "_EXPORT_BLOCK_BYTES", block_bytes)
    labels = store.repertoire.load_clonotypes(clonotypes).cells["clonotype"]
    cases = [
        ("edge.tsv", receptors, {"fields": EDGE_FIELDS}, EXPECTED_EDGE, 10),
        (
            "chains.tsv.gz",
            receptors,
            {"fields": EDGE_FIELDS, "chains_only": True},
            EXPECTED_EDGE_CHAINS,
            9,
        ),
        (
            "clonotypes.tsv",
            clonotypes,
            {"fields": ["v_call"]},
            EXPECTED_EDGE_CLONOTYPES.format(**labels.dropna().to_dict()),
            10,
        ),
        ("all.tsv.gz", receptors, {}, EXPECTED_EDGE_ALL, 10),
    ]
    for name, source, options, expected, n_contigs in cases:
        path = tmp_path / name
        assert store.repertoire.export_airr(source, path, **options) == n_contigs
        assert _text(path) == expected


def test_export_refuses_more_than_the_memory_budget(tmp_path, monkeypatch) -> None:
    store, receptors = _edge_receptors(tmp_path)
    path = tmp_path / "refused.tsv"
    monkeypatch.setattr(store, "memoryBytes", 1_000)
    with pytest.raises(MemoryError, match="Raise mem_budget"):
        store.repertoire.export_airr(receptors, path)
    assert not path.exists()
    # Within the budget of the blocks, the repeated IDs may not fit.
    monkeypatch.setattr(store, "memoryBytes", 10**9)
    monkeypatch.setattr(repertoire_operations, "_ID_BYTES", 10**9)
    with pytest.raises(MemoryError, match="Naming 5 repeated contig IDs"):
        store.repertoire.export_airr(receptors, path)
    assert not path.exists()
