import gzip
import io
import zlib

import numpy as np
import pandas as pd
import pytest
import zarr

import cytearc.datastore._operations.repertoire as repertoire_operations
import cytearc.writers.airr as airr_writer
from cytearc.metadata.records import EncodedText, encode_text
from cytearc.storage.artifacts import artifact_group, inspect_artifact
from cytearc.storage.refs import ArtifactRef
from cytearc.writers import write_airr_rearrangements
from cytearc.writers.airr import (
    AIRR_REQUIRED_FIELDS,
    AirrRearrangementWriter,
    sequence_id_hashes,
    suffix_repeated_ids,
)
from tests.clone_helpers import (
    BCR_ROWS,
    CELLRANGER_HEADER,
    CLONAL_ROWS,
    TCR_ROWS,
    budget_refusals,
    receptor_store,
    traced_peak,
    write_lines,
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


def _receptors(tmp_path, rows):
    store = receptor_store(tmp_path)
    path = write_lines(tmp_path, "clonal.csv", CELLRANGER_HEADER, rows)
    receptors = store.repertoire.import_receptors(
        path, cell_selection=store.snapshot_cell_selection()
    )
    return store, receptors


def _edge_receptors(tmp_path):
    store = receptor_store(tmp_path)
    files = [
        write_lines(tmp_path, "tcr.csv", CELLRANGER_HEADER, EDGE_TCR),
        write_lines(tmp_path, "bcr.csv", CELLRANGER_HEADER, EDGE_BCR),
        write_lines(tmp_path, "airr.tsv", EDGE_AIRR_HEADER, EDGE_AIRR),
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
    store = receptor_store(tmp_path)
    cells = store.snapshot_cell_selection()
    tcr = write_lines(tmp_path, "tcr.csv", CELLRANGER_HEADER, TCR_ROWS)
    bcr = write_lines(tmp_path, "bcr.csv", CELLRANGER_HEADER, BCR_ROWS)
    # The TCR rows name a barcode outside the store, about which the shared
    # join of barcodes to cells warns.
    with pytest.warns(UserWarning, match="match no selected cell"):
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
    store = receptor_store(tmp_path)
    # The TCR file already names a contig X_2, the suffix that the repeated X
    # of the BCR file would take.
    tcr = write_lines(
        tmp_path,
        "tcr.csv",
        CELLRANGER_HEADER,
        [
            "cell0,true,X,true,500,TRA,TRAV1,,TRAJ1,TRAC,true,true,CAVA,TGTGCTGTGGCC,90,9,ct1,ct1_1",
            "cell0,true,X_2,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,CASSA,TGTGCCAGCAGCGCC,80,8,ct1,ct1_2",
        ],
    )
    bcr = write_lines(
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
    expansion = store.clones.expansion(clonotypes)
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


def test_export_names_missing_and_incomplete_sources(tmp_path) -> None:
    store, receptors = _receptors(tmp_path, CLONAL_ROWS)
    clonotypes = store.repertoire.clonotypes(receptors)
    path = tmp_path / "refused.tsv"
    for source in (receptors, clonotypes):
        unsaved = ArtifactRef(
            scope=source.scope, kind=source.kind, artifact_id="f" * 64
        )
        with pytest.raises(KeyError, match="Artifact does not exist"):
            store.repertoire.export_airr(unsaved, path)
    # An import that stopped before it finished leaves incomplete receptors,
    # which the clonotypes read too.
    attributes = artifact_group(store.zw, receptors).attrs
    attributes["complete"] = False
    try:
        for source in (receptors, clonotypes):
            with pytest.raises(RuntimeError, match="Artifact is incomplete"):
                store.repertoire.export_airr(source, path)
    finally:
        attributes["complete"] = True
    attributes = artifact_group(store.zw, clonotypes).attrs
    attributes["complete"] = False
    try:
        with pytest.raises(RuntimeError, match="Artifact is incomplete"):
            store.repertoire.export_airr(clonotypes, path)
    finally:
        attributes["complete"] = True
    # Clonotypes whose receptors the store no longer holds.
    folder = tmp_path / "store.zarr" / inspect_artifact(store.zw, receptors).path
    folder.rename(folder.with_name(folder.name + "_moved"))
    with pytest.raises(KeyError, match="Artifact does not exist"):
        store.repertoire.export_airr(clonotypes, path)
    assert not path.exists()


@pytest.mark.parametrize("source", ["receptors", "clonotypes"])
def test_every_refusal_comes_before_the_file_is_written(tmp_path, source) -> None:
    store, receptors = _edge_receptors(tmp_path)
    ref = (
        store.repertoire.clonotypes(receptors) if source == "clonotypes" else receptors
    )
    n_fields = len(artifact_group(store.zw, receptors).attrs["fields"])
    attempts = []

    def export():
        path = tmp_path / f"attempt_{len(attempts)}.tsv.gz"
        attempts.append(path)
        return store.repertoire.export_airr(ref, path, chains_only=True)

    refusals = budget_refusals(store, export)
    assert len(refusals) > 1
    assert {what for what, _estimate in refusals} == {
        f"Exporting the 10 contigs of 8 cells with {n_fields} fields"
    }
    # Only the export that ran created its file, which holds the 9 chains.
    assert [path.exists() for path in attempts] == [False] * len(refusals) + [True]
    assert len(_read(attempts[-1])) == 9


def test_the_first_check_counts_one_chunk_of_the_first_text_values(tmp_path) -> None:
    store, receptors = _edge_receptors(tmp_path)
    group = artifact_group(store.zw, receptors)
    fields = group.attrs["fields"]
    first = next(
        name for name in fields if repertoire_operations.FIELDS[name] == "text"
    )
    values = group[f"contig_{first}_values"]
    # The Python objects of the arrays that the export opens, then the first
    # text field's distinct values, read whole: one stored chunk and its
    # compressed bytes beside the values.
    expected = (
        repertoire_operations._EXPORT_OBJECT_BYTES
        + repertoire_operations._EXPORT_OBJECT_BYTES_PER_FIELD * len(fields)
        + repertoire_operations.array_row_selection_parts(values)[0]
        + values.nbytes
    )
    refusals = budget_refusals(
        store, lambda: store.repertoire.export_airr(receptors, tmp_path / "x.tsv")
    )
    assert refusals[0] == (
        f"Exporting the 10 contigs of 8 cells with {len(fields)} fields",
        expected,
    )


def test_measuring_finds_the_longest_value_of_each_field(tmp_path) -> None:
    store = receptor_store(tmp_path)
    header = "cell_id\tsequence_id\tlocus\tv_call\tj_call\tjunction_aa\tproductive\tumi_count\tv_identity\td_identity\tsequence"
    rows = [
        # A chain of an 8-base sequence and whole identities.
        "cell0\tc0_1\tIGH\tIGHV1\tIGHJ1\tCARW\tT\t3\t1\t123456\tACGTACGT",
        # A sequence of 50 bases beyond ASCII in a contig that is no chain,
        # and an identity that is not whole.
        "cell1\tc1_1\tIGH\tIGHV1\tIGHJ1\tCARW\tF\t2\t0.5\t\t" + "\u00e9" * 25,
        "cell2\tc2_1\tIGK\tIGKV1\tIGKJ1\tCQQY\tT\t1\t-2\t-7\tGG",
    ]
    path = write_lines(tmp_path, "airr.tsv", header, rows)
    receptors = store.repertoire.import_receptors(
        path, cell_selection=store.snapshot_cell_selection()
    )
    group = artifact_group(store.zw, receptors)
    fields = tuple(group.attrs["fields"])
    widths = {}
    for chains_only in (False, True):
        contigs = repertoire_operations._ExportContigs(
            group,
            fields,
            chains_only=chains_only,
            reads=repertoire_operations._Reads(
                limit=10**9, held=0, what="Measuring", remedy=""
            ),
        )
        contigs.measure(10**9)
        widths[chains_only] = {
            name: contigs.width(name)
            for name in ("sequence_id", "sequence", "umi_count", "productive")
            + ("v_identity", "d_identity")
        } | {"whole": contigs.whole}
    # The 25 two-byte characters are 50 bytes; without them, 8. Whole numbers
    # are written as integers with a sign: 123456 takes 7 bytes and 1 or -2
    # takes 2, and other numbers as repr writes them, in at most 24 bytes.
    assert widths[False] == {
        "sequence_id": 4,
        "sequence": 50,
        "umi_count": 19,
        "productive": 1,
        "v_identity": 24,
        "d_identity": 7,
        "whole": {"d_identity"},
    }
    assert widths[True] == {
        "sequence_id": 4,
        "sequence": 8,
        "umi_count": 19,
        "productive": 1,
        "v_identity": 2,
        "d_identity": 7,
        "whole": {"v_identity", "d_identity"},
    }
    # Only the chains' identities are whole, so they are written as integers.
    store.repertoire.export_airr(receptors, tmp_path / "chains.tsv", chains_only=True)
    store.repertoire.export_airr(receptors, tmp_path / "all.tsv")
    assert _read(tmp_path / "chains.tsv")["v_identity"].tolist() == ["1", "-2"]
    assert _read(tmp_path / "all.tsv")["v_identity"].tolist() == ["1.0", "0.5", "-2.0"]


def test_the_longest_value_sizes_the_blocks(tmp_path) -> None:
    store = receptor_store(tmp_path)
    header = (
        "cell_id\tsequence_id\tlocus\tv_call\tj_call\tjunction_aa\tproductive\tsequence"
    )
    rows = [
        f"cell{cell % 8}\tc{index}\tIGH\tIGHV1\tIGHJ1\tCARW\tT\tACGT"
        for index, cell in enumerate(range(40))
    ]
    plain = store.repertoire.import_receptors(
        write_lines(tmp_path, "short.tsv", header, rows),
        cell_selection=store.snapshot_cell_selection(),
    )
    rows[17] = rows[17].replace("\tACGT", "\t" + "A" * 20_000)
    long = store.repertoire.import_receptors(
        write_lines(tmp_path, "long.tsv", header, rows),
        cell_selection=store.snapshot_cell_selection(),
    )
    planned = {}
    for name, receptors in (("short", plain), ("long", long)):
        group = artifact_group(store.zw, receptors)
        contigs = repertoire_operations._ExportContigs(
            group,
            tuple(group.attrs["fields"]),
            chains_only=False,
            reads=repertoire_operations._Reads(
                limit=10**9, held=0, what="Planning", remedy=""
            ),
        )
        contigs.hold_text()
        contigs.measure(10**9)
        planned[name] = contigs.plan_blocks(
            n_columns=20, cell_width=5, clone_width=None, block_bytes=100_000
        )
    # One sequence of 20,000 bytes, read and rendered, leaves room for at
    # most two rows of 40,000 bytes in 100,000 bytes; short rows all fit.
    assert planned["short"] == 40 and planned["long"] <= 2


def test_quoted_rows_render_in_slices_as_pandas_quotes_them(tmp_path) -> None:
    # Values of double quotes double in width, so the quoted rows take more
    # than the bytes of the first rendering, and are rendered in slices.
    rng = np.random.default_rng(3)
    letters = np.asarray(list('ab"\t\n\r\u00e9'))
    ids = np.asarray(
        ["".join(rng.choice(letters, rng.integers(0, 12))) for _ in range(300)],
        dtype=object,
    )
    ids[7] = '"' * 40
    labels = ["c1", 'q"', None, "t\tu"]
    codes = rng.integers(-1, len(labels) - 1, size=len(ids))
    cells = EncodedText.from_categories(codes, [label or "" for label in labels[:-1]])
    path = tmp_path / "quoted.tsv"
    block = {"sequence_id": ids, "cell_id": cells, "c_call": ids}
    with AirrRearrangementWriter(path, ["cell_id", "c_call"]) as writer:
        writer.write(block)
        columns = [
            airr_writer._bytes_column(block[name], name, False)
            if name in block
            else None
            for name in writer.columns
        ]
    expected = pd.DataFrame(
        {
            "sequence_id": ids,
            **{name: "" for name in AIRR_REQUIRED_FIELDS[1:]},
            "cell_id": [labels[code] or "" if code >= 0 else "" for code in codes],
            "c_call": ids,
        }
    ).to_csv(sep="\t", index=False, header=False, lineterminator="\n")
    assert path.read_bytes().decode("utf-8").split("\n", 1)[1] == expected
    pieces = [bytes(piece) for piece in airr_writer._rows(columns, len(ids))]
    assert len(pieces) > 1 and b"".join(pieces).decode("utf-8") == expected
    # One row whose quoted value takes more than the row's first rendering.
    one = [airr_writer._packed(np.asarray(['"' * 9], dtype=object))]
    assert [bytes(piece) for piece in airr_writer._rows(one, 1)] == [b'"' * 20 + b"\n"]


def test_rendered_rows_reach_the_file_in_bounded_writes(tmp_path, monkeypatch) -> None:
    # Compressing a write holds what it compresses, so the rows reach the
    # file a few bytes at a time.
    store, receptors = _edge_receptors(tmp_path)
    store.repertoire.export_airr(receptors, tmp_path / "whole.tsv")
    monkeypatch.setattr(airr_writer, "_WRITE_BYTES", 7)
    write = gzip.GzipFile.write
    sizes = []

    def recorded(handle, data):
        sizes.append(memoryview(data).nbytes)
        return write(handle, data)

    monkeypatch.setattr(gzip.GzipFile, "write", recorded)
    store.repertoire.export_airr(receptors, tmp_path / "pieces.tsv.gz")
    monkeypatch.undo()
    # After the header, no write holds more than 7 bytes.
    assert len(sizes) > 10 and max(sizes[1:]) == 7
    assert _text(tmp_path / "pieces.tsv.gz") == _text(tmp_path / "whole.tsv")


def test_repeated_ids_match_a_count_of_every_id(monkeypatch) -> None:
    # Hashes of one byte collide often, so most IDs are read as text.
    rng = np.random.default_rng(11)
    ids = np.asarray(
        [f"X{value}" for value in rng.integers(0, 40, size=2_000)]
        + [f"X{value}_2" for value in rng.integers(0, 40, size=50)],
        dtype=object,
    )
    rng.shuffle(ids)

    def weak(texts) -> np.ndarray:
        return np.asarray(
            [zlib.crc32(str(text).encode()) % 251 for text in texts], dtype=np.uint64
        )

    monkeypatch.setattr(airr_writer, "sequence_id_hashes", weak)
    rows, names = suffix_repeated_ids(weak(ids), lambda positions: ids[positions])
    # Each later occurrence takes the next suffix that no row holds and no
    # earlier occurrence took.
    expected_rows, expected_names = [], []
    taken = set(ids.tolist())
    seen: dict[str, int] = {}
    for row, text in enumerate(ids.tolist()):
        seen[text] = seen.get(text, 0) + 1
        if seen[text] == 1:
            continue
        number = seen[text]
        while f"{text}_{number}" in taken:
            number += 1
        taken.add(f"{text}_{number}")
        expected_rows.append(row)
        expected_names.append(f"{text}_{number}")
    assert rows.tolist() == expected_rows
    assert names.tolist() == expected_names


def test_every_export_read_decodes_one_chunk_at_a_time(tmp_path, monkeypatch) -> None:
    # The checks count one decoded chunk of each array read, which zarr's
    # default concurrency would exceed for arrays of several chunks.
    store, receptors = _edge_receptors(tmp_path)
    clonotypes = store.repertoire.clonotypes(receptors)
    getitem = zarr.Array.__getitem__
    limits = []

    def recorded(array, selection):
        limits.append(
            (array.path.rsplit("/", 1)[-1], zarr.config.get("async.concurrency"))
        )
        return getitem(array, selection)

    monkeypatch.setattr(zarr.Array, "__getitem__", recorded)
    store.repertoire.export_airr(clonotypes, tmp_path / "x.tsv", chains_only=True)
    monkeypatch.undo()
    names = {name for name, _limit in limits}
    assert {"codes", "clonotype_ids", "contig_cells", "contig_chain"} <= names
    assert {"contig_sequence_id", "contig_v_call_values", "contig_umi_count"} <= names
    assert {
        limit
        for name, limit in limits
        if name.startswith(("contig_", "codes", "clonotype_ids"))
    } == {1}
    assert zarr.config.get("async.concurrency") != 1


def test_categorical_logical_values_are_written_as_flags(tmp_path) -> None:
    # A table read with categories may hold booleans as categories.
    path = tmp_path / "categories.tsv"
    table = pd.DataFrame(
        {
            "sequence_id": ["s1", "s2", "s3"],
            "productive": pd.Categorical([True, None, False]),
            "rev_comp": pd.Categorical(["T", "f", None]),
        }
    )
    assert write_airr_rearrangements(table, path) == 3
    written = _read(path)
    assert written["productive"].tolist() == ["T", "", "F"]
    assert written["rev_comp"].tolist() == ["T", "F", ""]


_LONG = 20_000
_LONG_HEADER = (
    "cell_id\tsequence_id\tlocus\tv_call\tj_call\tjunction_aa\tproductive\tsequence"
)


def _long_files(folder, kind: str) -> list:
    """200 contigs of the eight cells, each holding one value of 20,000 bytes.

    Long IDs come in two files of 100 contigs that name them alike, as the
    TCR and BCR files of one cell do, so every ID of the second is named again.
    """
    rows = []
    for index in range(200):
        sequence_id = f"c{index}"
        junction, sequence = "CARW", "ACGT"
        if kind == "text":
            # One distinct value, stored once and written in every row.
            junction = "C" + "A" * _LONG + "F"
        elif kind == "ragged":
            sequence = "ACGT" * (_LONG // 4)
        else:
            sequence_id = f"{index % 100:05d}" + "x" * _LONG
        rows.append(
            f"cell{index % 8}\t{sequence_id}\tIGH\tIGHV1\tIGHJ1\t{junction}\tT\t{sequence}"
        )
    if kind != "ids":
        return [write_lines(folder, "long.tsv", _LONG_HEADER, rows)]
    return [
        write_lines(folder, "first.tsv", _LONG_HEADER, rows[:100]),
        write_lines(folder, "second.tsv", _LONG_HEADER, rows[100:]),
    ]


@pytest.mark.parametrize("kind", ["text", "ragged", "ids"])
def test_exports_of_long_values_fit_the_smallest_budget(tmp_path, kind) -> None:
    store = receptor_store(tmp_path)
    receptors = store.repertoire.import_receptors(
        _long_files(tmp_path, kind), cell_selection=store.snapshot_cell_selection()
    )
    path = tmp_path / "long_export.tsv"

    def export():
        return store.repertoire.export_airr(receptors, path)

    export()
    refusals = budget_refusals(store, export)
    budget = refusals[-1][1]
    store.memoryBytes = budget
    peak = traced_peak(export)
    assert peak <= budget, (peak, budget)
    written = _read(path)
    assert len(written) == 200 and written["sequence_id"].is_unique


def test_receptors_without_contigs_export_their_header(tmp_path) -> None:
    store = receptor_store(tmp_path)
    elsewhere = [
        "elsewhere,true,e_contig_1,true,500,TRA,TRAV1,,TRAJ1,TRAC,true,true,CAVA,TGTGCTGTGGCC,90,9,ct1,ct1_1"
    ]
    with pytest.warns(UserWarning, match="match no selected cell"):
        receptors = store.repertoire.import_receptors(
            write_lines(tmp_path, "none.csv", CELLRANGER_HEADER, elsewhere),
            cell_selection=store.snapshot_cell_selection(),
        )
    clonotypes = store.repertoire.clonotypes(receptors)
    for source, name in ((receptors, "none.tsv.gz"), (clonotypes, "none.tsv")):
        assert (
            store.repertoire.export_airr(source, tmp_path / name, chains_only=True) == 0
        )
        header = _text(tmp_path / name)
        assert header.endswith("\n") and header.count("\n") == 1
        assert header.split("\t")[: len(AIRR_REQUIRED_FIELDS)] == list(
            AIRR_REQUIRED_FIELDS
        )
