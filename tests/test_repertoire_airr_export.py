import asyncio
import csv
import dataclasses
import gc
import gzip
import io
import re
import tracemalloc
import weakref
import zlib

import numpy as np
import pandas as pd
import pytest
import zarr

import cytearc.datastore._operations.repertoire as repertoire_operations
import cytearc.metadata.rows as metadata_rows
import cytearc.writers.airr as airr_writer
from cytearc.metadata.records import EncodedText, encode_text
from cytearc.storage.artifacts import artifact_group, inspect_artifact
from cytearc.storage.refs import ArtifactRef
from cytearc.utils.logging import logger
from cytearc.writers import write_airr_rearrangements
from cytearc.writers.airr import (
    AIRR_REQUIRED_FIELDS,
    AirrRearrangementWriter,
    sequence_id_hashes,
    suffix_repeated_ids,
)
from tests.clone_helpers import (
    clonal_receptors,
    BCR_ROWS,
    CELLRANGER_HEADER,
    CLONAL_ROWS,
    TCR_ROWS,
    budget_refusals,
    receptor_store,
    traced_peak,
    write_cells_store,
    write_lines,
)

# Contig IDs repeat across the three files: X takes _3 and then _4, since the
# TCR file names a contig X_2, and Z takes _3, or _2 when only chains are
# written, since the contig named Z_2 is not a chain. Values hold @, double
# quotes, one at the start of a constant gene call, ', # and text beyond
# ASCII, which are written as they are, and some are missing.
EDGE_TCR = [
    'cell0,true,X,true,500,TRA,TRAV1@B,,TRAJ1,TRAC,true,true,CAVA,TGTGCTGTGGCC,90,9,"ct""1",ct1_1',
    "cell0,true,X_2,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,CASSA,TGTGCCAGCAGCGCC,80,8,ct1,ct1_2",
    """cell1,true,Y,true,500,TRB,TRBVé,,TRBJ2,\"\"\"TR'BC#1",true,true,CASSB,TGTGCCAGCAGCGCA,80,8,ct2,ct2_1""",
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
# DataFrame.to_csv, but with every value as it is, unquoted, as the AIRR file
# format asks. v_identity holds whole numbers only, so it is written as
# integers.
EXPECTED_EDGE = (
    "sequence_id\tsequence\trev_comp\tproductive\tv_call\td_call\tj_call"
    "\tsequence_alignment\tgermline_alignment\tjunction\tjunction_aa\tv_cigar"
    "\td_cigar\tj_cigar\tcell_id\tc_call\tumi_count\tv_identity\td_identity"
    "\tj_identity\traw_clonotype_id\n"
    "X\t\t\t\t\t\t\t\t\t\t\t\t\t\tcell0\tIGHM\t20\t\t\t\tcb1\n"
    'X_3\t\t\t\t\t\t\t\t\t\t\t\t\t\tcell0\tTRAC\t9\t\t\t\tct"1\n'
    "X_2\t\t\t\t\t\t\t\t\t\t\t\t\t\tcell0\tTRBC1\t8\t\t\t\tct1\n"
    "Y\t\t\t\t\t\t\t\t\t\t\t\t\t\tcell1\t\"TR'BC#1\t8\t\t\t\tct2\n"
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
    'X_3\t\t\t\t\t\t\t\t\t\t\t\t\t\tcell0\tTRAC\t9\t\t\t\tct"1\n'
    "X_2\t\t\t\t\t\t\t\t\t\t\t\t\t\tcell0\tTRBC1\t8\t\t\t\tct1\n"
    "Y\t\t\t\t\t\t\t\t\t\t\t\t\t\tcell1\t\"TR'BC#1\t8\t\t\t\tct2\n"
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
    "X_3\t\t\t\tTRAV1@B\t\t\t\t\t\t\t\t\t\tcell0\t{cell0}\n"
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
    "X_3\t\t\tT\tTRAV1@B\t\tTRAJ1\t\t\tTGTGCTGTGGCC\tCAVA\t\t\t\tcell0\tTRA"
    '\tTRAC\t9\t90\tT\tT\tT\t\t\t\tct"1\tct1_1\n'
    "X_2\t\t\tT\tTRBV1\t\tTRBJ1\t\t\tTGTGCCAGCAGCGCC\tCASSA\t\t\t\tcell0\tTRB"
    "\tTRBC1\t8\t80\tT\tT\tT\t\t\t\tct1\tct1_2\n"
    "Y\t\t\tT\tTRBVé\t\tTRBJ2\t\t\tTGTGCCAGCAGCGCA\tCASSB\t\t\t\tcell1\tTRB"
    "\t\"TR'BC#1\t8\t80\tT\tT\tT\t\t\t\tct2\tct2_1\n"
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


# The IDs of the receptor store's cells, packed as the export packs them.
_CELLS = EncodedText.from_categories(
    np.empty(0, dtype=np.int32), [f"cell{k}" for k in range(8)]
)


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
    # The AIRR file format quotes no value, so quotes are read as text.
    return pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE
    )


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
    store, receptors = clonal_receptors(tmp_path, CLONAL_ROWS)
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
    store, receptors = clonal_receptors(tmp_path, rows)
    everything = store.repertoire.export_airr(receptors, tmp_path / "all.tsv")
    chains = store.repertoire.export_airr(
        receptors, tmp_path / "chains.tsv", chains_only=True
    )
    assert everything == chains + 1
    assert "cell7_contig_1" not in set(_read(tmp_path / "chains.tsv")["sequence_id"])


def test_export_refuses_other_sources_and_arguments(tmp_path) -> None:
    store, receptors = clonal_receptors(tmp_path, CLONAL_ROWS)
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


def test_the_writer_writes_values_as_they_are(tmp_path) -> None:
    path = tmp_path / "unquoted.tsv"
    # Double quotes at the start, within and at the end, ', @, # and text
    # beyond ASCII: the AIRR file format quotes no value.
    ids = np.array(['"a', 'q"x', 'b""', "o'k", "@c#", "é"], dtype=object)
    cells = encode_text(['"c1', None, "c1", "c'2", "c'2", "c1"])
    with AirrRearrangementWriter(path, ["cell_id", "c_call"]) as writer:
        writer.write({"sequence_id": ids, "cell_id": cells, "c_call": ids})
    empty = "\t" * (len(AIRR_REQUIRED_FIELDS) - 1)
    rows = path.read_bytes().decode("utf-8").split("\n", 1)[1]
    assert rows == (
        f'"a{empty}\t"c1\t"a\n'
        f'q"x{empty}\t\tq"x\n'
        f'b""{empty}\tc1\tb""\n'
        f"o'k{empty}\tc'2\to'k\n"
        f"@c#{empty}\tc'2\t@c#\n"
        f"é{empty}\tc1\té\n"
    )
    # Read without quoting, as the format asks, each value is as written.
    written = _read(path)
    assert written["sequence_id"].tolist() == ids.tolist()
    assert written["c_call"].tolist() == ids.tolist()
    assert written["cell_id"].tolist() == ['"c1', "", "c1", "c'2", "c'2", "c1"]
    # Five IDs, the same five calls and three cells hold such characters.
    assert writer.n_discouraged == 13


def test_values_the_format_advises_against_warn_outside_phred_scores(
    tmp_path,
) -> None:
    table = pd.DataFrame(
        {
            "sequence_id": ["s1", "s2", "s3"],
            "v_call": ["IGHV1@2", "IGHV1", "IGHV#3"],
            "junction_aa": ["CAR'W", 'CAR"W', "CARW"],
            # Phred scores, ASCII 33 to 126, hold these characters by the
            # format's own encoding.
            "quality": ["II#@\"'", "IIII", "@@@@"],
            "quality_alignment": ["#'\"@", "", "IIII"],
        }
    )
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        assert write_airr_rearrangements(table, tmp_path / "phred.tsv") == 3
        with AirrRearrangementWriter(tmp_path / "scores.tsv", ["quality"]) as writer:
            writer.write({"sequence_id": ["s9"], "quality": ['@"#']})
            writer.write({"sequence_id": ["t9"], "quality": ["'"], "d_call": ["D'1"]})
    finally:
        logger.remove(sink)
    assert messages == [
        "4 values of v_call and junction_aa hold @, #, a double quote or ', which "
        "the AIRR format advises against since readers may take them differently; "
        "they are written as they are",
        "1 value of d_call holds @, #, a double quote or ', which the AIRR format "
        "advises against since readers may take them differently; they are "
        "written as they are",
    ]
    assert writer.n_discouraged == 1
    written = _read(tmp_path / "phred.tsv")
    for name in table.columns:
        assert written[name].tolist() == table[name].tolist()
    # A tab is no Phred score, so quality is refused for it as any field is.
    with pytest.raises(ValueError, match="1 value of quality, in contig 's2'"):
        write_airr_rearrangements(
            table.assign(quality=["II", "I\tI", "II"]), tmp_path / "tab.tsv"
        )


@pytest.mark.parametrize("character", ["\t", "\n", "\r"])
def test_tables_of_values_that_hold_a_tab_or_a_line_break_are_refused(
    tmp_path, character
) -> None:
    path = tmp_path / "refused.tsv"
    table = pd.DataFrame(
        {
            "sequence_id": ["s0", "s1", "s2", "s3"],
            "v_call": ["V0", f"V{character}1", "V2", f"V{character}{character}3"],
            "c_call": pd.Categorical(["C0", "C1", f"C{character}2", "C3"]),
            "umi_count": [1, 2, 3, 4],
        }
    )
    with pytest.raises(ValueError) as refused:
        write_airr_rearrangements(table, path)
    assert str(refused.value) == (
        "AIRR values must not hold a tab, a newline or a carriage return, which "
        "these values of the table do: 2 values of v_call, the first in contig "
        "'s1'; 1 value of c_call, in contig 's2'"
    )
    # An ID that holds one names itself, and a contig without an ID is named
    # by its empty ID, as the file would write it.
    ids = table.assign(sequence_id=["s0", "s1", f"s{character}2", "s3"])
    ids = ids.assign(v_call="V", c_call="C")
    with pytest.raises(ValueError) as refused:
        write_airr_rearrangements(ids, path)
    assert str(refused.value).endswith(
        f"1 value of sequence_id, in contig {'s' + character + '2'!r}"
    )
    unnamed = table.assign(sequence_id=["s0", None, "s2", "s3"], c_call="C")
    with pytest.raises(ValueError, match="2 values of v_call, the first in contig ''$"):
        write_airr_rearrangements(unnamed, path)
    # The fields come in the order of their first contigs, not of the file.
    later = table.assign(c_call=[f"C{character}0", "C1", "C2", "C3"])
    with pytest.raises(ValueError) as refused:
        write_airr_rearrangements(later, path)
    assert str(refused.value).endswith(
        "1 value of c_call, in contig 's0'; 2 values of v_call, the first in "
        "contig 's1'"
    )
    assert not path.exists()


def test_a_table_refusal_names_the_first_contig_of_a_later_chunk(
    tmp_path, monkeypatch
) -> None:
    # A table is checked three rows at a time, and the first value that holds
    # a tab is in the third chunk.
    monkeypatch.setattr(airr_writer, "_CHUNK_ROWS", 3)
    table = pd.DataFrame(
        {
            "sequence_id": [f"s{k}" for k in range(9)],
            "v_call": ["V"] * 7 + ["V\t7", "V\t8"],
        }
    )
    path = tmp_path / "chunks.tsv"
    with pytest.raises(
        ValueError, match="2 values of v_call, the first in contig 's7'$"
    ):
        write_airr_rearrangements(table, path)
    assert not path.exists()


def test_the_writer_refuses_a_block_before_writing_any_of_it(tmp_path) -> None:
    path = tmp_path / "blocks.tsv"
    with AirrRearrangementWriter(path, ["cell_id"]) as writer:
        writer.write({"sequence_id": ["a"], "cell_id": ["c1"]})
        with pytest.raises(ValueError) as refused:
            writer.write(
                {
                    "sequence_id": np.array(["b", "c", "d"], dtype=object),
                    "cell_id": encode_text(["c1", "c\r2", "c\r2"]),
                }
            )
        writer.write({"sequence_id": ["e"], "cell_id": ["c3"]})
    assert str(refused.value) == (
        "AIRR values must not hold a tab, a newline or a carriage return, which "
        "these values of the block do: 2 values of cell_id, the first in contig 'c'"
    )
    empty = "\t" * (len(AIRR_REQUIRED_FIELDS) - 1)
    header = "\t".join([*AIRR_REQUIRED_FIELDS, "cell_id"])
    assert path.read_bytes().decode("utf-8") == (
        f"{header}\na{empty}\tc1\ne{empty}\tc3\n"
    )
    assert writer.n_rows == 2
    # The fields come in the order of their first contigs, not of the file.
    with AirrRearrangementWriter(tmp_path / "ordered.tsv", ["cell_id"]) as writer:
        with pytest.raises(ValueError) as refused:
            writer.write(
                {
                    "sequence_id": ["g", "h"],
                    "v_call": ["V", "V\n1"],
                    "cell_id": ["c\r6", "c7"],
                }
            )
    assert str(refused.value).endswith(
        "1 value of cell_id, in contig 'g'; 1 value of v_call, in contig 'h'"
    )
    # A contig without an ID is named by its empty ID.
    with AirrRearrangementWriter(tmp_path / "unnamed.tsv", ["cell_id"]) as writer:
        with pytest.raises(ValueError, match="1 value of cell_id, in contig ''$"):
            writer.write(
                {"sequence_id": encode_text(["f", None]), "cell_id": ["c4", "c\t5"]}
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


@pytest.mark.parametrize(
    ("block_bytes", "chunk_bytes"), [(None, None), (1, None), (None, 16)]
)
def test_streamed_exports_match_the_exports_of_one_table(
    tmp_path, monkeypatch, block_bytes, chunk_bytes
) -> None:
    if chunk_bytes is not None:
        # Cell positions stored four to a chunk and offsets two, so blocks of
        # four contigs read their windows across stored chunks.
        monkeypatch.setattr(repertoire_operations, "_CONTIG_CHUNK_BYTES", chunk_bytes)
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
    store, receptors = clonal_receptors(tmp_path, CLONAL_ROWS)
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
    # text field's distinct values, read whole: a stored chunk and its
    # compressed bytes as they are decoded, and as much again for a chunk
    # that a read before may still hold, beside the values.
    expected = (
        repertoire_operations._EXPORT_OBJECT_BYTES
        + repertoire_operations._EXPORT_OBJECT_BYTES_PER_FIELD * len(fields)
        + 2 * repertoire_operations.array_row_selection_parts(values)[0]
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
        contigs.hold_text()
        contigs.measure(10**9, cells=_CELLS, clones=None)
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
        contigs.measure(10**9, cells=_CELLS, clones=None)
        planned[name] = contigs.plan_blocks(
            n_columns=20,
            cell_width=5,
            clone_width=None,
            block_bytes=100_000,
            compressed=False,
        )
    # One sequence of 20,000 bytes, read and rendered, leaves room for at
    # most two rows of 40,000 bytes in 100,000 bytes; short rows all fit.
    assert planned["short"] == 40 and planned["long"] <= 2


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


def test_the_cells_ids_are_checked_before_each_read(tmp_path) -> None:
    store, receptors = _edge_receptors(tmp_path)
    ids = store.cells._get_array("ids")
    fixed = repertoire_operations.text_row_selection_parts(ids)[0]
    selection = inspect_artifact(store.zw, receptors).input_ref("cell_selection")
    mask = _chunk(artifact_group(store.zw, selection)["values"])
    refusals = []
    limit = 1
    while True:
        reads = repertoire_operations._Reads(
            limit=limit, held=0, what="Reading", remedy=""
        )
        try:
            packed, longest = store._export_cell_ids(reads, receptors, 8, 1)
            break
        except MemoryError as error:
            found = re.search(r"holds about (\d+) bytes", str(error))
            assert found is not None and int(found.group(1)) > limit
            refusals.append(int(found.group(1)))
            limit = refusals[-1]
    # The selection's positions, 32 bytes a cell as read and 8 as kept, with
    # two chunks of its mask, one read and one held from a read before; the
    # offsets of the eight IDs; then each slice of one ID of 5 characters,
    # stored in 20 bytes: two stored chunks, one decoded and one held from
    # the read before, the row as text_read_bytes counts it, and its
    # packing, three copies of the stored bytes and 24 more, beside the IDs
    # packed before it, 5 bytes each.
    per_row = (3 * 20 + 16 * 8) + 64 + 4 * 5 + 3 * 20 + 24
    held = 8 * 8 + 8 * 9
    assert refusals == [32 * 8 + 2 * mask] + [
        held + 5 * k + 2 * fixed + per_row for k in range(8)
    ]
    assert bytes(packed.data) == b"".join(f"cell{k}".encode() for k in range(8))
    assert packed.offsets.tolist() == list(range(0, 45, 5)) and longest == 5
    # The packed IDs and their offsets stay counted; the positions do not.
    assert reads.held == 8 * 9 + 40


def test_packing_text_counts_each_slice_and_their_join() -> None:
    # 30 values of 1,000 characters beyond ASCII, 1,997 bytes each, in
    # chunks of two rows, so joining the packed slices outweighs any slice.
    values = np.asarray([f"{k:03d}" + "é" * 997 for k in range(30)])
    array = zarr.create_array(
        store=zarr.storage.MemoryStore(), shape=(30,), chunks=(2,), dtype=values.dtype
    )
    array[:] = values
    fixed = repertoire_operations.text_row_selection_parts(array)[0]
    refusals = []
    limit = 1
    while True:
        reads = repertoire_operations._Reads(
            limit=limit, held=0, what="Packing", remedy=""
        )
        try:
            packed, longest = repertoire_operations._packed_text(reads, array, None, 1)
            break
        except MemoryError as error:
            found = re.search(r"holds about (\d+) bytes", str(error))
            assert found is not None and int(found.group(1)) > limit
            refusals.append(int(found.group(1)))
            limit = refusals[-1]
    # The offsets; each slice of one row stored in 4,000 bytes, with two
    # stored chunks, one decoded and one held from the read before, beside
    # the slices packed before it; and the join, a second copy of every
    # slice beside the chunk read last.
    per_row = (3 * 4_000 + 16 * 8) + 64 + 4 * 1_000 + 3 * 4_000 + 24
    offsets = 8 * 31
    assert refusals == [
        offsets,
        *(offsets + 1_997 * k + 2 * fixed + per_row for k in range(30)),
        offsets + 2 * 1_997 * 30 + fixed,
    ]
    assert bytes(packed.data).decode("utf-8") == "".join(values.tolist())
    assert longest == 1_997 and reads.held == offsets + 1_997 * 30


def _walk(run) -> list[int]:
    """Each refusal's estimate, from a limit of one byte to one that runs."""
    refusals = []
    limit = 1
    while True:
        try:
            run(limit)
            return refusals
        except MemoryError as error:
            found = re.search(r"holds about (\d+) bytes", str(error))
            assert found is not None and int(found.group(1)) > limit
            refusals.append(int(found.group(1)))
            limit = refusals[-1]


def _edge_contigs(store, receptors, limit, fields=("sequence_id", "v_call")):
    """The edge receptors' contigs of some fields, read within ``limit``."""
    return repertoire_operations._ExportContigs(
        artifact_group(store.zw, receptors),
        fields,
        chains_only=False,
        reads=repertoire_operations._Reads(limit=limit, held=0, what="", remedy=""),
    )


def _chunk(array) -> int:
    """The bytes of one chunk that reading an array holds, by the store's model."""
    return repertoire_operations.array_row_selection_parts(array)[0]


@pytest.mark.parametrize("wider", [0, 2**20])
def test_measuring_is_checked_with_a_span_of_contigs(
    tmp_path, monkeypatch, wider
) -> None:
    store, receptors = _edge_receptors(tmp_path)
    group = artifact_group(store.zw, receptors)
    parts = repertoire_operations.array_row_selection_parts

    def model(array):
        # The bytes of the ragged fields, which measuring reads to check
        # each value, may come in stored chunks larger than their offsets'.
        name = str(array.path).rsplit("/", 1)[-1]
        fixed, per_row = parts(array)
        if name in ("contig_sequence_id", "contig_sequence"):
            fixed += wider
        return fixed, per_row

    monkeypatch.setattr(repertoire_operations, "array_row_selection_parts", model)
    fields = ("sequence_id", "sequence", "v_identity", "d_identity")
    refusals = _walk(
        lambda limit: _edge_contigs(store, receptors, limit, fields).measure(
            10**9, cells=_CELLS, clones=None
        )
    )
    # Two stored chunks of the arrays read, one decoded and one held from
    # the read before; a stored chunk of a ragged field's bytes as read, here
    # all 40 bytes of the ten IDs, X, X_2, Y and Z, X, Z and Z_2, two of 14
    # and X, against the 10 of the sequences; and 64 bytes for each of the
    # ten contigs of the span.
    reading = max(
        model(group[name])[0]
        for name in (
            "contig_sequence_id_offsets",
            "contig_sequence_offsets",
            "contig_sequence_id",
            "contig_sequence",
            "contig_v_identity",
            "contig_d_identity",
        )
    )
    if wider:
        assert reading == parts(group["contig_sequence_id"])[0] + wider
    assert refusals == [2 * reading + 40 + 64 * 10]


def test_hashing_the_ids_is_checked_with_the_hashes(tmp_path) -> None:
    store, receptors = _edge_receptors(tmp_path)
    group = artifact_group(store.zw, receptors)

    def scan(limit):
        contigs = _edge_contigs(store, receptors, 10**9, ("sequence_id",))
        contigs.measure(10**9, cells=_CELLS, clones=None)
        contigs.reads.limit = limit
        return contigs.scan(10**9)

    # The hashes of the ten contigs; a block of all ten IDs, 300 bytes and 6
    # per byte of the longest, cell5_contig_1, as they are decoded and hashed;
    # the window of the eleven offsets; and two stored chunks, one decoded
    # and one held from the read before.
    reading = max(
        _chunk(group[name])
        for name in ("contig_sequence_id_offsets", "contig_sequence_id")
    )
    assert _walk(scan) == [8 * 10 + (300 + 6 * 14) * 10 + 11 * 8 + 2 * reading]


def test_naming_repeated_ids_keeps_the_ids_read_before(tmp_path, monkeypatch) -> None:
    store, receptors = _edge_receptors(tmp_path)
    # Naming outweighs every other step, so its checks refuse last.
    monkeypatch.setattr(repertoire_operations, "_ID_BYTES", 10**8)
    refusals = budget_refusals(
        store, lambda: store.repertoire.export_airr(receptors, tmp_path / "named.tsv")
    )
    (five, first), (seven, second) = refusals[-2:]
    # The five X and Z are read, then X_2 and Z_2, whose hashes the suffixed
    # names take, while the five stay held: each 10**8 bytes and 4 per byte
    # of the longest ID.
    assert (five, seven) == (
        "Naming 5 repeated contig IDs",
        "Naming 7 repeated contig IDs",
    )
    assert second - first == 2 * (10**8 + 4 * 14)
    # Finding the repeated hashes holds, for each of the ten contigs, the
    # sorted hashes and a search's positions and matches, beside the naming.
    monkeypatch.setattr(repertoire_operations, "_NAMING_BYTES_PER_ROW", 10**6)
    widened = budget_refusals(
        store, lambda: store.repertoire.export_airr(receptors, tmp_path / "named.tsv")
    )
    assert widened[-2] == (five, first + 10 * (10**6 - 40))


@pytest.mark.parametrize("chunk_bytes", [None, 16])
def test_writing_a_block_counts_each_row_at_its_widest(
    tmp_path, monkeypatch, chunk_bytes
) -> None:
    if chunk_bytes is not None:
        monkeypatch.setattr(repertoire_operations, "_CONTIG_CHUNK_BYTES", chunk_bytes)
    store, receptors = _edge_receptors(tmp_path)
    group = artifact_group(store.zw, receptors)
    fields = ("sequence_id", "v_call", "v_identity")

    def plan(limit):
        contigs = _edge_contigs(store, receptors, limit, fields)
        contigs.hold_text()
        contigs.measure(10**9, cells=_CELLS, clones=None)
        return contigs.plan_blocks(
            n_columns=16,
            cell_width=5,
            clone_width=None,
            block_bytes=32 * 2**20,
            compressed=False,
        )

    refusals = _walk(plan)
    held = group["contig_v_call_values"].nbytes + group["contig_v_call_offsets"].nbytes
    # Each row at its widest: 16 separators, a cell ID of 5 bytes, an ID of
    # 14, a V call of 8 and a V identity of 2, the whole numbers 0 and 1 with
    # a sign, rendered; the ID and the identity read and rendered beside it;
    # 96 bytes for the row, 24 for the ID's offsets, 12 for the V call's
    # codes and 16 for the identity's offsets; and rendering the identity,
    # 136 bytes and 4 per byte of its widest text.
    rendered = 16 + 5 + 14 + 8 + 2
    per_row = rendered + (14 + 2) + (96 + 24 + 12 + 16) + (136 + 4 * 2)
    reading = max(
        _chunk(group[name])
        for name in (
            "contig_cells",
            "contig_sequence_id_offsets",
            "contig_v_call",
            "contig_v_identity",
            "contig_sequence_id",
        )
    )
    if chunk_bytes is None:
        # One block of the ten contigs; windows of their ten int32 positions,
        # eleven int64 offsets, ten int32 codes and ten float64 identities,
        # each in one chunk.
        rows, windows = 10, 10 * 4 + 11 * 8 + 10 * 4 + 10 * 8
    else:
        # Blocks of four contigs, a stored chunk of positions; the windows of
        # offsets and of identities span three chunks of two, and the codes
        # one chunk of four.
        rows, windows = 4, 4 * 4 + 6 * 8 + 4 * 4 + 6 * 8
    # Two stored chunks, one decoded and one held from the read before.
    assert refusals[-1] == held + per_row * rows + windows + 2 * reading


def test_renamed_ids_stay_held_while_the_blocks_are_written(
    tmp_path, monkeypatch
) -> None:
    store, receptors = _edge_receptors(tmp_path)
    plan = repertoire_operations._ExportContigs.plan_blocks
    held = []

    def recorded(contigs, **options):
        held.append(contigs.reads.held)
        return plan(contigs, **options)

    monkeypatch.setattr(repertoire_operations._ExportContigs, "plan_blocks", recorded)
    store.repertoire.export_airr(receptors, tmp_path / "named.tsv")
    monkeypatch.setattr(repertoire_operations, "_ID_BYTES", 1_200)
    store.repertoire.export_airr(receptors, tmp_path / "renamed.tsv")
    # X_3, X_4 and Z_3 stay held with their new names, 1,000 bytes more each.
    assert held[1] - held[0] == 3 * 1_000


class _StoredRows:
    """An array of six rows in chunks of two, which notes what it held at each read."""

    chunks = (2,)
    shape = (6,)

    def __init__(self) -> None:
        self.values = np.arange(6)
        self.previous: weakref.ref | None = None
        self.held_at_read: list[bool] = []

    def __getitem__(self, selection):
        out = self.values[selection].copy()
        if len(out):
            self.held_at_read.append(
                self.previous is not None and self.previous() is not None
            )
            self.previous = weakref.ref(out)
        return out


def test_a_window_lets_go_of_its_chunk_before_reading_the_next() -> None:
    stored = _StoredRows()
    window = repertoire_operations._Window(stored)
    assert [window.read(start, start + 2).tolist() for start in (0, 2, 4)] == [
        [0, 1],
        [2, 3],
        [4, 5],
    ]
    # Each chunk is read once, and the one before is gone by then.
    assert stored.held_at_read == [False, False, False]


def test_a_block_is_let_go_before_the_next_is_read(tmp_path, monkeypatch) -> None:
    # Contigs of small chunks, one per block, so every block reads its windows.
    monkeypatch.setattr(repertoire_operations, "_CONTIG_CHUNK_BYTES", 16)
    monkeypatch.setattr(repertoire_operations, "_EXPORT_BLOCK_BYTES", 1)
    store, receptors = _edge_receptors(tmp_path)
    written: list[weakref.ref] = []
    alive: list[bool] = []
    write = airr_writer.AirrRearrangementWriter.write
    read = repertoire_operations._Window.read

    def recorded_write(writer, block):
        written.append(weakref.ref(block["sequence_id"].data))
        return write(writer, block)

    def recorded_read(window, start, stop):
        if written:
            alive.append(written[-1]() is not None)
        return read(window, start, stop)

    monkeypatch.setattr(airr_writer.AirrRearrangementWriter, "write", recorded_write)
    monkeypatch.setattr(repertoire_operations._Window, "read", recorded_read)
    assert store.repertoire.export_airr(receptors, tmp_path / "blocks.tsv") == 10
    # Ten blocks were written, and no block was held while the next was read.
    assert len(written) == 10 and len(alive) > 10 and not any(alive)


def test_clonotypes_hold_their_codes_and_labels_while_blocks_are_written(
    tmp_path, monkeypatch
) -> None:
    store, receptors = _edge_receptors(tmp_path)
    clonotypes = store.repertoire.clonotypes(receptors)
    plan = repertoire_operations._ExportContigs.plan_blocks
    held = []

    def recorded(contigs, **options):
        held.append(contigs.reads.held)
        return plan(contigs, **options)

    monkeypatch.setattr(repertoire_operations._ExportContigs, "plan_blocks", recorded)
    store.repertoire.export_airr(receptors, tmp_path / "receptors.tsv")
    store.repertoire.export_airr(clonotypes, tmp_path / "clonotypes.tsv")
    labels = store.repertoire.load_clonotypes(clonotypes).clonotypes["clonotype"]
    # Each of the eight cells' int32 code, and the labels as packed bytes
    # with their offsets.
    packed = sum(len(str(label).encode()) for label in labels)
    assert held[1] - held[0] == 4 * 8 + 8 * (len(labels) + 1) + packed


@pytest.mark.parametrize(
    ("source", "suffix", "large"),
    [
        ("receptors", ".tsv", "every"),
        ("clonotypes", ".tsv", "every"),
        ("receptors", ".tsv.gz", "every"),
        ("clonotypes", ".tsv", "contig_"),
        ("receptors", ".tsv", "contig_j_identity"),
    ],
)
def test_each_step_counts_a_chunk_that_the_read_before_may_still_hold(
    tmp_path, monkeypatch, source, suffix, large
) -> None:
    # Zarr's decoding thread may hold a read's decoded chunk a moment after
    # the read returns. Each read here holds a chunk of 50 MiB, as the
    # store's model counts it, until after the next read, so every step's
    # traced peak must stay within the check made before it, also while a
    # compressed file's deflate state is held. Unless ``large`` is "every",
    # only the arrays whose names start with it hold chunks of 50 MiB, and
    # the others keep the store's own model, a few MiB at most: a step then
    # reads chunks larger than the one left behind before it, or, with the J
    # identities, the last field that measuring reads and the blocks next,
    # smaller ones.
    # Zarr decodes inline here, so the chunks that this test holds are the
    # only ones left behind.
    async def inline(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", inline)
    store, receptors = _edge_receptors(tmp_path)
    ref = (
        store.repertoire.clonotypes(receptors) if source == "clonotypes" else receptors
    )
    parts = metadata_rows.array_row_selection_parts

    def size(array) -> int:
        name = str(getattr(array, "path", "")).rsplit("/", 1)[-1]
        if large == "every" or name.startswith(large):
            return 50 * 2**20
        return parts(array)[0]

    def model(array):
        return size(array), parts(array)[1]

    monkeypatch.setattr(metadata_rows, "array_row_selection_parts", model)
    monkeypatch.setattr(repertoire_operations, "array_row_selection_parts", model)
    held: list[np.ndarray] = []
    getitem = zarr.Array.__getitem__

    def read(array, selection):
        values = getitem(array, selection)
        if np.asarray(values).size:
            # The new chunk is held before the one before it is let go.
            held[:] = [np.ones(size(array), dtype=np.uint8)]
        return values

    monkeypatch.setattr(zarr.Array, "__getitem__", read)
    reserve = repertoire_operations._Reads.reserve
    steps: list[tuple[int, int]] = []
    estimate = 0

    def recorded(reads, needed):
        nonlocal estimate
        reserve(reads, needed)
        steps.append((estimate, tracemalloc.get_traced_memory()[1]))
        tracemalloc.reset_peak()
        # The bytes that the check counted, from its refusal of a copy of
        # the reads with no budget.
        with pytest.raises(MemoryError) as refused:
            reserve(dataclasses.replace(reads, limit=-1), needed)
        found = re.search(r"holds about (-?\d+) bytes", str(refused.value))
        assert found is not None
        estimate = int(found.group(1))

    monkeypatch.setattr(repertoire_operations._Reads, "reserve", recorded)
    gc.collect()
    tracemalloc.start()
    try:
        store.repertoire.export_airr(ref, tmp_path / f"lingering{suffix}")
        steps.append((estimate, tracemalloc.get_traced_memory()[1]))
    finally:
        tracemalloc.stop()
        held.clear()
    # The first record closes what ran before the first check.
    over = [
        (index, peak - bound)
        for index, (bound, peak) in enumerate(steps[1:], start=1)
        if peak > bound
    ]
    assert len(steps) > 10 and not over, over


def test_the_export_warns_of_discouraged_characters_and_repeated_ids(
    tmp_path, monkeypatch
) -> None:
    store, receptors = _edge_receptors(tmp_path)
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        store.repertoire.export_airr(receptors, tmp_path / "all.tsv")
        # One contig per block names the same fields in the file's order.
        monkeypatch.setattr(repertoire_operations, "_EXPORT_BLOCK_BYTES", 1)
        store.repertoire.export_airr(receptors, tmp_path / "blocks.tsv")
    finally:
        logger.remove(sink)
    # The second and third X and the second Z take suffixes; a V call holds
    # @, a constant gene call a double quote, ' and #, and a clonotype ID a
    # double quote.
    assert (
        messages.count(
            "3 contig IDs repeat; their later occurrences are written with a "
            "suffix such as _2"
        )
        == 2
    )
    discouraged = [message for message in messages if "advises against" in message]
    assert discouraged == 2 * [
        "3 values of v_call, c_call and raw_clonotype_id hold @, #, a double quote "
        "or ', which the AIRR format advises against since readers may take them "
        "differently; they are written as they are"
    ]


# T cell rows with tabs in three V calls, two of one value and one of two
# tabs, a newline in two constant gene calls and a carriage return in one ID,
# all of which the AIRR format forbids within a value; the contig with the
# carriage return is not a chain. Stored
# four contigs to a chunk of cells and 16 bytes to a chunk of IDs, the first
# span's IDs take three pieces, the carriage return in the third, and the V
# calls hold a tab in both spans.
FORBIDDEN_TCR = [
    'cell0,true,x_contig_1,true,500,TRA,"TRAV1\tB",,TRAJ1,TRAC,true,true,CAVA,TGTGCTGTGGCC,90,9,ct1,ct1_1',
    "cell0,true,x_contig_2,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,CASSA,TGTGCCAGCAGCGCC,80,8,ct1,ct1_2",
    'cell1,true,y_contig_1,true,500,TRB,TRBV2,,TRBJ2,"TR\nBC1",true,true,CASSB,TGTGCCAGCAGCGCA,80,8,ct2,ct2_1',
    'cell2,true,"z_con\rtig_1",true,500,TRA,"TRAV1\tB",,TRAJ2,"TR\nBC1",true,false,CAVB,TGTGCTGTGGCA,60,6,ct3,ct3_1',
    "cell3,true,w_contig_1,true,500,TRB,TRBV3,,TRBJ3,TRBC1,true,true,CASSC,TGTGCCAGCAGCTGT,70,7,ct4,ct4_1",
    'cell4,true,v_contig_1,true,500,TRA,"TRAV4\t\tD",,TRAJ4,TRAC,true,true,CAVD,TGTGCTGTGGAT,50,5,ct5,ct5_1',
]
_FORBIDDEN = (
    "AIRR values must not hold a tab, a newline or a carriage return, which these "
    "values of the contigs do: "
)


def _forbidden_receptors(tmp_path):
    store = receptor_store(tmp_path)
    receptors = store.repertoire.import_receptors(
        write_lines(tmp_path, "tcr.csv", CELLRANGER_HEADER, FORBIDDEN_TCR),
        cell_selection=store.snapshot_cell_selection(),
    )
    return store, receptors


@pytest.mark.parametrize(
    ("source", "chunk_bytes"),
    [("receptors", None), ("clonotypes", None), ("receptors", 16)],
)
def test_export_refuses_values_the_format_forbids_before_the_file(
    tmp_path, monkeypatch, source, chunk_bytes
) -> None:
    if chunk_bytes is not None:
        # Contigs stored four to a chunk of cells and IDs 16 bytes to a chunk,
        # so measuring reads spans of four contigs and IDs in pieces.
        monkeypatch.setattr(repertoire_operations, "_CONTIG_CHUNK_BYTES", chunk_bytes)
    store, receptors = _forbidden_receptors(tmp_path)
    ref = (
        store.repertoire.clonotypes(receptors) if source == "clonotypes" else receptors
    )
    path = tmp_path / "refused.tsv.gz"
    cases = [
        (
            {},
            "3 values of v_call, the first in contig 'x_contig_1'; 2 values of "
            "c_call, the first in contig 'y_contig_1'; 1 value of sequence_id, in "
            "contig 'z_con\\rtig_1'",
        ),
        # Only the chains are written, and z_con\rtig_1 is no chain.
        (
            {"chains_only": True},
            "2 values of v_call, the first in contig 'x_contig_1'; 1 value of "
            "c_call, in contig 'y_contig_1'",
        ),
        (
            {"fields": ["umi_count"]},
            "1 value of sequence_id, in contig 'z_con\\rtig_1'",
        ),
    ]
    for options, found in cases:
        with pytest.raises(ValueError) as refused:
            store.repertoire.export_airr(ref, path, **options)
        assert str(refused.value) == _FORBIDDEN + found
        assert not path.exists()
    # Without the fields that hold them, the five chains are written.
    assert (
        store.repertoire.export_airr(
            ref, path, chains_only=True, fields=["umi_count", "j_call"]
        )
        == 5
    )


def test_a_chains_only_refusal_names_the_first_chain_that_holds_the_value(
    tmp_path,
) -> None:
    # a_1 is no chain, so the first chain with a V call of a tab is a_2.
    rows = [
        "cell0,true,a_1,true,500,TRA,TRAV2,,TRAJ1,TRAC,true,false,CAVA,TGTGCT,80,8,ct1,ct1_1",
        'cell0,true,a_2,true,500,TRB,"TRBV1\tB",,TRBJ1,TRBC1,true,true,CASSA,TGTGCC,80,8,ct1,ct1_1',
    ]
    store = receptor_store(tmp_path)
    receptors = store.repertoire.import_receptors(
        write_lines(tmp_path, "tcr.csv", CELLRANGER_HEADER, rows),
        cell_selection=store.snapshot_cell_selection(),
    )
    path = tmp_path / "chains.tsv"
    with pytest.raises(ValueError) as refused:
        store.repertoire.export_airr(receptors, path, chains_only=True)
    assert str(refused.value) == _FORBIDDEN + "1 value of v_call, in contig 'a_2'"
    assert not path.exists()


def test_ragged_values_of_discouraged_characters_are_written_and_counted(
    tmp_path,
) -> None:
    # IDs and sequences, which each contig stores as its own bytes, may hold
    # @, # and quotes, which are written as they are, not refused.
    store = receptor_store(tmp_path)
    header = (
        "cell_id\tsequence_id\tlocus\tv_call\tj_call\tjunction_aa\tproductive\tsequence"
    )
    rows = [
        "cell0\tc0#1\tIGH\tIGHV1\tIGHJ1\tCARW\tT\tAC@GT",
        "cell1\tc1_1\tIGH\tIGHV1\tIGHJ1\tCARW\tT\tACGT",
        "cell2\t\"c2'1\tIGK\tIGKV1\tIGKJ1\tCQQY\tT\tGG",
    ]
    receptors = store.repertoire.import_receptors(
        write_lines(tmp_path, "airr.tsv", header, rows),
        cell_selection=store.snapshot_cell_selection(),
    )
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        written = store.repertoire.export_airr(receptors, tmp_path / "ragged.tsv")
    finally:
        logger.remove(sink)
    assert written == 3
    exported = _read(tmp_path / "ragged.tsv")
    assert exported["sequence_id"].tolist() == ["c0#1", "c1_1", "\"c2'1"]
    assert exported["sequence"].tolist() == ["AC@GT", "ACGT", "GG"]
    assert messages == [
        "3 values of sequence_id and sequence hold @, #, a double quote or ', "
        "which the AIRR format advises against since readers may take them "
        "differently; they are written as they are"
    ]


def test_export_refuses_cell_ids_and_labels_the_format_forbids(tmp_path) -> None:
    # Cells whose IDs hold a tab within them, as the store holds them.
    store = write_cells_store(tmp_path / "store.zarr", ["ce\tll0", "cell1"])
    rows = [
        '"ce\tll0",true,a_1,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,CASSA,TGTGCC,80,8,ct1,ct1_1',
        "cell1,true,b_1,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,CASSA,TGTGCC,80,8,ct1,ct1_1",
        '"ce\tll0",true,a_2,true,500,TRA,TRAV1,,TRAJ1,TRAC,true,true,CAVA,TGTGCT,80,8,ct1,ct1_1',
    ]
    receptors = store.repertoire.import_receptors(
        write_lines(tmp_path, "tcr.csv", CELLRANGER_HEADER, rows),
        cell_selection=store.snapshot_cell_selection(),
    )
    path = tmp_path / "cells.tsv"
    with pytest.raises(ValueError) as refused:
        store.repertoire.export_airr(receptors, path)
    assert str(refused.value) == (
        _FORBIDDEN + "2 values of cell_id, the first in contig 'a_1'"
    )
    assert not path.exists()
    # A clonotype label that holds a newline, through each cell's code: cell1
    # and cell2 of the edge receptors hold it, so do their contigs Y and Z.
    edge, edge_receptors = _edge_receptors(tmp_path / "edge")
    group = artifact_group(edge.zw, edge_receptors)
    clones = EncodedText.from_categories(
        np.asarray([0, 1, 1, -1, 0, 2, 2, -1]), ["a", "b\nc", "d"]
    )
    contigs = repertoire_operations._ExportContigs(
        group,
        tuple(group.attrs["fields"]),
        chains_only=False,
        reads=repertoire_operations._Reads(limit=10**9, held=0, what="", remedy=""),
    )
    contigs.hold_text()
    with pytest.raises(ValueError) as refused:
        contigs.measure(10**9, cells=_CELLS, clones=clones)
    assert str(refused.value) == (
        _FORBIDDEN + "2 values of clone_id, the first in contig 'Y'"
    )


def test_measuring_counts_the_codes_of_values_that_hold_forbidden_characters(
    tmp_path,
) -> None:
    store, receptors = _forbidden_receptors(tmp_path)
    group = artifact_group(store.zw, receptors)
    fields = ("sequence_id", "v_call")
    base = []

    def contigs_within(limit):
        contigs = _edge_contigs(store, receptors, 10**9, fields)
        contigs.hold_text()
        base[:] = [contigs.reads.held, contigs.reads.lingering]
        contigs.reads.limit = limit
        return contigs

    def run(limit):
        with pytest.raises(ValueError, match="3 values of v_call"):
            contigs_within(limit).measure(10**9, cells=_CELLS, clones=None)

    refusals = _walk(run)
    held, lingering = base
    reading = max(
        _chunk(group[name])
        for name in (
            "contig_sequence_id_offsets",
            "contig_sequence_id",
            "contig_v_call",
        )
    )
    # The codes of the two V calls that hold a tab, one of them two, 8 bytes
    # each, stay held; then the span, as without such values, but with a
    # chunk of the V calls' codes, and a stored chunk of IDs, all 61 bytes of
    # the six, five of 10 bytes and z_con\rtig_1 of 11, and 64 bytes for each
    # of the six contigs.
    assert refusals == [
        held + lingering + 16,
        held + 16 + lingering + reading + max(0, reading - lingering) + 61 + 64 * 6,
    ]
    # Reading the ID of the V call's first contig, x_contig_1, is checked too:
    # a stored chunk of the IDs' arrays beside the span's last, 16 bytes of
    # offsets and the 11 of the longest ID.
    contigs = contigs_within(10**9)
    with pytest.raises(ValueError):
        contigs.measure(10**9, cells=_CELLS, clones=None)
    contigs.reads.limit = 1
    ids = max(
        _chunk(group[name])
        for name in ("contig_sequence_id_offsets", "contig_sequence_id")
    )
    reading_ids = held + 16 + reading + ids + max(0, ids - reading) + 16 + 11
    with pytest.raises(MemoryError, match=f"holds about {reading_ids} bytes"):
        contigs._contig_id(0)


def test_rows_are_marked_by_the_value_that_holds_a_forbidden_byte() -> None:
    # Five values: one that starts with a tab, an empty one, as a missing
    # value is, one that ends with a newline, one without, and one of a
    # carriage return and a tab within it.
    data = np.frombuffer(b"\tabcd\nef\rg\th", dtype=np.uint8)
    bounds = np.asarray([0, 3, 3, 6, 7, 12], dtype=np.int64)
    # Pieces of each length, as stored chunks of each length give them.
    for step in range(1, 13):
        marks = np.zeros(5, dtype=np.bool_)
        for start in range(0, 12, step):
            piece = data[start : start + step].copy()
            airr_writer._mark_forbidden_rows(piece, start, bounds, marks)
        assert marks.tolist() == [True, False, True, False, True], step


def test_exported_values_read_back_through_an_import_as_they_were(tmp_path) -> None:
    store, receptors = _edge_receptors(tmp_path)
    path = tmp_path / "all.tsv"
    store.repertoire.export_airr(receptors, path)
    again = store.repertoire.import_receptors(
        path, cell_selection=store.snapshot_cell_selection()
    )
    store.repertoire.export_airr(again, tmp_path / "again.tsv")
    # The import reads each value as it was written, the constant gene call
    # that starts with a double quote too, so the export of what it read is
    # the first export.
    assert _text(tmp_path / "again.tsv") == _text(path) == EXPECTED_EDGE_ALL
    cells = store.repertoire.load_receptors(again).cells
    assert cells.loc["cell1", "VDJ_1_c_call"] == "\"TR'BC#1"
    assert cells.loc["cell0", "VJ_1_v_call"] == "TRAV1@B"


def test_integers_of_every_length_are_written_whole(tmp_path) -> None:
    values = [0, 9, 10, 99, 100, *(10**k for k in range(3, 19)), -10, -100]
    values += [-(10**18), 2**63 - 1]
    table = pd.DataFrame(
        {
            "sequence_id": [f"s{k}" for k in range(len(values))],
            "np1_length": np.asarray(values, dtype=np.int64),
        }
    )
    write_airr_rearrangements(table, tmp_path / "digits.tsv")
    written = _read(tmp_path / "digits.tsv")["np1_length"].tolist()
    assert written == [str(value) for value in values]


def test_zero_counts_are_written_as_zero(tmp_path) -> None:
    store = receptor_store(tmp_path)
    header = (
        "cell_id\tsequence_id\tlocus\tv_call\tj_call\tjunction_aa\tproductive"
        "\tumi_count\tduplicate_count\tnp1_length"
    )
    rows = [
        "cell0\tc0\tIGH\tIGHV1\tIGHJ1\tCARW\tT\t0\t0\t0",
        "cell1\tc1\tIGH\tIGHV1\tIGHJ1\tCARW\tF\t5\t\t2",
    ]
    receptors = store.repertoire.import_receptors(
        write_lines(tmp_path, "zeros.tsv", header, rows),
        cell_selection=store.snapshot_cell_selection(),
    )
    store.repertoire.export_airr(receptors, tmp_path / "zeros_export.tsv")
    written = _read(tmp_path / "zeros_export.tsv")
    assert written["umi_count"].tolist() == ["0", "5"]
    assert written["duplicate_count"].tolist() == ["0", ""]
    assert written["np1_length"].tolist() == ["0", "2"]
    assert written["productive"].tolist() == ["T", "F"]


def test_clonotypes_replace_an_imported_clone_id(tmp_path) -> None:
    store = receptor_store(tmp_path)
    header = (
        "cell_id\tsequence_id\tlocus\tv_call\tj_call\tjunction\tjunction_aa"
        "\tproductive\tclone_id"
    )
    rows = [
        "cell0\tc0\tTRB\tTRBV1\tTRBJ1\tTGTGCCAGC\tCAS\tT\timported_a",
        "cell1\tc1\tTRB\tTRBV1\tTRBJ1\tTGTGCCAGC\tCAS\tT\timported_b",
        "cell2\tc2\tTRB\tTRBV2\tTRBJ1\tTGTGCCTTT\tCAF\tT\timported_c",
    ]
    receptors = store.repertoire.import_receptors(
        write_lines(tmp_path, "cloned.tsv", header, rows),
        cell_selection=store.snapshot_cell_selection(),
    )
    clonotypes = store.repertoire.clonotypes(receptors)
    store.repertoire.export_airr(receptors, tmp_path / "imported.tsv")
    store.repertoire.export_airr(clonotypes, tmp_path / "defined.tsv")
    imported = _read(tmp_path / "imported.tsv")
    assert imported["clone_id"].tolist() == ["imported_a", "imported_b", "imported_c"]
    defined = _read(tmp_path / "defined.tsv")
    labels = store.repertoire.load_clonotypes(clonotypes).cells["clonotype"]
    assert defined.columns.tolist().count("clone_id") == 1
    assert defined["clone_id"].tolist() == [labels[f"cell{k}"] for k in range(3)]
    # cell0 and cell1 share their junction, so their clonotype.
    clones = defined["clone_id"].tolist()
    assert clones[0] == clones[1] != clones[2]


def test_a_selection_of_some_cells_exports_their_ids(tmp_path) -> None:
    store = receptor_store(tmp_path)
    store.cells.insert(
        "chosen", np.asarray([False, False, True, False, True, True, False, True])
    )
    rows = [
        f"cell{k},true,cell{k}_contig_1,true,500,TRB,TRBV{k},,TRBJ1,TRBC1,true,true,"
        f"CASS{k},TGTGCCAGCAGC,100,5,ct{k},ct{k}_1"
        for k in (0, 2, 4, 5, 7)
    ]
    with pytest.warns(UserWarning, match="match no selected cell"):
        receptors = store.repertoire.import_receptors(
            write_lines(tmp_path, "some.csv", CELLRANGER_HEADER, rows),
            cell_selection=store.snapshot_cell_selection("chosen"),
        )
    assert store.repertoire.export_airr(receptors, tmp_path / "some.tsv") == 4
    written = _read(tmp_path / "some.tsv")
    assert written["cell_id"].tolist() == ["cell2", "cell4", "cell5", "cell7"]
    assert written["sequence_id"].tolist() == [
        f"cell{k}_contig_1" for k in (2, 4, 5, 7)
    ]


def test_blocks_hold_a_sixteenth_of_the_budget(tmp_path, monkeypatch) -> None:
    store, receptors = _edge_receptors(tmp_path)
    plan = repertoire_operations._ExportContigs.plan_blocks
    sizes = []

    def recorded(contigs, **options):
        sizes.append(options["block_bytes"])
        return plan(contigs, **options)

    monkeypatch.setattr(repertoire_operations._ExportContigs, "plan_blocks", recorded)
    for budget in (160 * 2**20, 2**30):
        monkeypatch.setattr(store, "memoryBytes", budget)
        store.repertoire.export_airr(receptors, tmp_path / "sized.tsv")
    # A sixteenth of 160 MiB, and at most 32 MiB.
    assert sizes == [10 * 2**20, 32 * 2**20]


def test_compressed_exports_are_reproducible(tmp_path) -> None:
    store, receptors = _edge_receptors(tmp_path)
    first, second = tmp_path / "first.tsv.gz", tmp_path / "second.tsv.gz"
    store.repertoire.export_airr(receptors, first)
    store.repertoire.export_airr(receptors, second)
    data = first.read_bytes()
    assert data == second.read_bytes()
    # The header names no file and no time, and its extra flags, 0, mark a
    # level neither the fastest nor the best.
    assert data[:10] == b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff"
    assert gzip.decompress(data) == _text(first).encode("utf-8")


def test_tables_and_columns_are_checked_before_writing(tmp_path) -> None:
    path = tmp_path / "checked.tsv"
    table = pd.DataFrame({"sequence_id": ["a", "b"], "umi_count": [1, 2]})
    with pytest.raises(TypeError, match="table must be a pandas DataFrame"):
        write_airr_rearrangements({"sequence_id": ["a"]}, path)
    with pytest.raises(TypeError, match="columns must be a sequence"):
        write_airr_rearrangements(table, path, columns="umi_count")
    with pytest.raises(ValueError, match=r"name \['umi_count'\] more than once"):
        write_airr_rearrangements(table, path, columns=["umi_count", "umi_count"])
    with pytest.raises(ValueError, match="sequence_id 'a' repeats"):
        write_airr_rearrangements(pd.DataFrame({"sequence_id": ["a", "b", "a"]}), path)
    with pytest.raises(TypeError, match="columns must be a sequence"):
        AirrRearrangementWriter(path, "cell_id")
    with pytest.raises(TypeError, match="columns must be a sequence of column names"):
        AirrRearrangementWriter(path, ["cell_id", 3])
    with pytest.raises(ValueError, match=r"name \['cell_id'\] more than once"):
        AirrRearrangementWriter(path, ["cell_id", "cell_id"])
    # A column's name is a field of the header row.
    with pytest.raises(
        ValueError, match=r"Column names must not hold .* \['my\\tfield'\]"
    ):
        write_airr_rearrangements(
            pd.DataFrame({"sequence_id": ["a"], "my\tfield": ["x"]}), path
        )
    for name in ("cell\nid", "cell\rid"):
        with pytest.raises(ValueError, match="Column names must not hold"):
            AirrRearrangementWriter(path, [name])
    assert not path.exists()


def test_fields_may_name_the_columns_that_are_always_written(tmp_path) -> None:
    store, receptors = clonal_receptors(tmp_path)
    clonotypes = store.repertoire.clonotypes(receptors)
    plain = store.repertoire.export_airr(
        receptors, tmp_path / "cells.tsv", fields=["cell_id", "v_call"]
    )
    labelled = store.repertoire.export_airr(
        clonotypes, tmp_path / "clones.tsv", fields=["clone_id", "cell_id"]
    )
    assert plain == labelled == len(CLONAL_ROWS)
    columns = _read(tmp_path / "cells.tsv").columns.tolist()
    assert columns.count("cell_id") == 1 and "clone_id" not in columns
    columns = _read(tmp_path / "clones.tsv").columns.tolist()
    assert columns[len(AIRR_REQUIRED_FIELDS) :] == ["cell_id", "clone_id"]
    # Receptors from Cell Ranger CSVs store no clone_id.
    with pytest.raises(ValueError, match="store no field 'clone_id'"):
        store.repertoire.export_airr(receptors, tmp_path / "x.tsv", fields=["clone_id"])
