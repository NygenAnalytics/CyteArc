import gzip

import numpy as np
import pytest

from cytearc.readers.airr import (
    airr_columns,
    contig_file_format,
    iter_airr_rearrangements,
)
from cytearc.readers.cellranger import iter_vdj_contigs, vdj_contig_columns

AIRR_HEADER = (
    "cell_id\tsequence_id\tlocus\tv_call\tj_call\tc_call\tjunction\tjunction_aa"
    "\tproductive\tumi_count\tconsensus_count\tv_identity\tsequence\textra"
)
AIRR_ROWS = [
    "c1\tc1_1\tTRB\tTRBV1*01\tTRBJ1*01\tTRBC1\tTGTGCC\tCASS\tT\t4\t100\t0.98\tACGTAC\tx",
    "c1\tc1_2\tTRA\tTRAV1\tTRAJ1\t\tTGTGCT\tCAVF\tF\t\t12\t\t\ty",
    "c2\tc2_1\tIGH\tIGHV3-30\tIGHJ4\tIGHM\tTGTGCG\tCARW\tT\t7\t70\t0.91\tGGG\tz",
]
CELLRANGER_HEADER = (
    "barcode,is_cell,contig_id,high_confidence,length,chain,v_gene,d_gene,j_gene,"
    "c_gene,full_length,productive,fwr1,fwr1_nt,cdr1,cdr1_nt,fwr2,fwr2_nt,cdr2,"
    "cdr2_nt,fwr3,fwr3_nt,cdr3,cdr3_nt,fwr4,fwr4_nt,reads,umis,raw_clonotype_id,"
    "raw_consensus_id,exact_subclonotype_id"
)
CELLRANGER_ROWS = [
    "AAAC-1,true,AAAC-1_contig_1,true,500,TRA,TRAV23/DV6,None,TRAJ22,TRAC,true,true,"
    "QQV,CAGCAG,TSG,ACCAGT,LYW,CTGTAC,IYS,ATCTAC,LSQ,CTGAGC,CAASKF,TGTGCAGCA,FGK,TTCGGC,"
    "2699,11,clonotype1,clonotype1_consensus_2,1",
    "AAAC-1,true,AAAC-1_contig_2,true,510,TRB,TRBV6-5,TRBD1,TRBJ2-3,TRBC2,true,false,"
    ",,,,,,,,,,CASSYF,TGTGCCAGC,,,3202,9,clonotype1,clonotype1_consensus_1,1",
    "AAAG-1,false,AAAG-1_contig_1,false,300,Multi,None,None,None,None,false,None,"
    ",,,,,,,,,,None,None,,,12,1,None,None,",
]


def _write(path, lines, compress=False):
    text = "\n".join(lines) + "\n"
    if compress:
        with gzip.open(path, "wt") as handle:
            handle.write(text)
    else:
        path.write_text(text)
    return path


@pytest.mark.parametrize("compress", [False, True])
def test_formats_are_told_apart_by_their_headers(tmp_path, compress) -> None:
    airr = _write(tmp_path / "rearrangements.tsv", [AIRR_HEADER], compress)
    cellranger = _write(tmp_path / "contigs.csv", [CELLRANGER_HEADER], compress)
    assert contig_file_format(airr) == "airr"
    assert contig_file_format(cellranger) == "cellranger"


def test_an_unknown_header_is_refused(tmp_path) -> None:
    other = _write(tmp_path / "other.csv", ["a,b,c"])
    with pytest.raises(ValueError, match="neither an AIRR Rearrangement file"):
        contig_file_format(other)


def test_files_without_a_header_are_refused(tmp_path) -> None:
    empty = tmp_path / "empty.tsv"
    empty.write_text("")
    with pytest.raises(ValueError, match="has no header line"):
        airr_columns(empty)
    with pytest.raises(ValueError, match="has no header line"):
        vdj_contig_columns(empty)


def test_airr_rows_are_parsed_by_field_kind(tmp_path) -> None:
    path = _write(tmp_path / "rearrangements.tsv", [AIRR_HEADER, *AIRR_ROWS])
    chunks = list(iter_airr_rearrangements(path))
    assert len(chunks) == 1
    first_line, chunk = chunks[0]
    assert first_line == 2
    assert list(chunk.columns) == [
        "cell_id",
        "sequence_id",
        "locus",
        "v_call",
        "j_call",
        "c_call",
        "junction",
        "junction_aa",
        "productive",
        "umi_count",
        "consensus_count",
        "v_identity",
        "sequence",
    ]
    assert chunk["cell_id"].tolist() == ["c1", "c1", "c2"]
    assert chunk["c_call"].tolist() == ["TRBC1", None, "IGHM"]
    assert chunk["productive"].tolist() == [1, 0, 1]
    assert chunk["umi_count"].tolist() == [4, -1, 7]
    np.testing.assert_array_equal(chunk["v_identity"], [0.98, np.nan, 0.91])
    assert chunk["sequence"].tolist() == ["ACGTAC", None, "GGG"]


def test_airr_files_are_read_in_chunks_with_their_lines(tmp_path) -> None:
    path = _write(tmp_path / "rearrangements.tsv.gz", [AIRR_HEADER, *AIRR_ROWS], True)
    chunks = list(iter_airr_rearrangements(path, chunk_rows=2))
    assert [first_line for first_line, _chunk in chunks] == [2, 4]
    assert [len(chunk) for _first_line, chunk in chunks] == [2, 1]


def test_airr_files_without_a_locus_have_it_inferred(tmp_path) -> None:
    header = "cell_id\tsequence_id\tv_call\tj_call\tjunction_aa\tproductive"
    rows = ["c1\ts1\tTRBV1\tTRBJ1\tCASS\tT", "c1\ts2\tTRAV29/DV5\tTRAJ1\tCAV\tT"]
    path = _write(tmp_path / "rearrangements.tsv", [header, *rows])
    ((_line, chunk),) = iter_airr_rearrangements(path)
    assert chunk["locus"].tolist() == ["TRB", "TRA"]


def test_a_constant_gene_call_alone_names_the_locus(tmp_path) -> None:
    header = "cell_id\tsequence_id\tv_call\tc_call\tjunction_aa\tproductive"
    rows = ["c1\ts1\t\tIGHM\tCAR\tT", "c1\ts2\tIGKV1\tIGHG1\tCQQ\tT"]
    path = _write(tmp_path / "rearrangements.tsv", [header, *rows])
    ((_line, chunk),) = iter_airr_rearrangements(path)
    # The second contig's calls name two loci, so it has none.
    assert chunk["locus"].tolist() == ["IGH", None]
    only_constant = _write(
        tmp_path / "constant.tsv",
        [
            "cell_id\tsequence_id\tc_call\tjunction_aa\tproductive",
            "c1\ts1\tIGHM\tCAR\tT",
        ],
    )
    ((_line, chunk),) = iter_airr_rearrangements(only_constant)
    assert chunk["locus"].tolist() == ["IGH"]


@pytest.mark.parametrize("compress", [False, True])
def test_a_byte_order_mark_is_not_part_of_the_header(tmp_path, compress) -> None:
    airr = _write(tmp_path / "bom.tsv", ["\ufeff" + AIRR_HEADER, *AIRR_ROWS], compress)
    cellranger = _write(
        tmp_path / "bom.csv", ["\ufeff" + CELLRANGER_HEADER, *CELLRANGER_ROWS], compress
    )
    assert contig_file_format(airr) == "airr"
    assert contig_file_format(cellranger) == "cellranger"
    assert airr_columns(airr)[0] == "cell_id"
    assert vdj_contig_columns(cellranger)[0] == "barcode"
    ((_line, airr_chunk),) = iter_airr_rearrangements(airr)
    ((_line, cellranger_chunk),) = iter_vdj_contigs(cellranger)
    assert airr_chunk["cell_id"].tolist() == ["c1", "c1", "c2"]
    assert cellranger_chunk["cell_id"].tolist() == ["AAAC-1", "AAAC-1", "AAAG-1"]


def test_a_separator_ending_every_row_leaves_the_columns_in_place(tmp_path) -> None:
    airr = _write(
        tmp_path / "trailing.tsv", [AIRR_HEADER, *(row + "\t" for row in AIRR_ROWS)]
    )
    ((_line, chunk),) = iter_airr_rearrangements(airr)
    assert chunk["cell_id"].tolist() == ["c1", "c1", "c2"]
    assert chunk["productive"].tolist() == [1, 0, 1]
    cellranger = _write(
        tmp_path / "trailing.csv",
        [CELLRANGER_HEADER, *(row + "," for row in CELLRANGER_ROWS)],
    )
    ((_line, chunk),) = iter_vdj_contigs(cellranger)
    assert chunk["sequence_id"].tolist() == [
        "AAAC-1_contig_1",
        "AAAC-1_contig_2",
        "AAAG-1_contig_1",
    ]


@pytest.mark.parametrize("chunk_rows", [1, 2, 50_000])
def test_blank_lines_hold_no_contig_and_keep_the_line_numbers(
    tmp_path, chunk_rows
) -> None:
    airr = _write(
        tmp_path / "blank.tsv",
        [AIRR_HEADER, AIRR_ROWS[0], "", AIRR_ROWS[1], "", "", AIRR_ROWS[2], ""],
    )
    chunks = list(iter_airr_rearrangements(airr, chunk_rows=chunk_rows))
    assert [value for _line, chunk in chunks for value in chunk["sequence_id"]] == [
        "c1_1",
        "c1_2",
        "c2_1",
    ]
    # A row of empty fields, as spreadsheets end a table, is a blank line too.
    cellranger = _write(
        tmp_path / "blank.csv",
        [CELLRANGER_HEADER, CELLRANGER_ROWS[0], "", CELLRANGER_ROWS[1], ",," * 15],
    )
    chunks = list(iter_vdj_contigs(cellranger, chunk_rows=chunk_rows))
    assert [value for _line, chunk in chunks for value in chunk["sequence_id"]] == [
        "AAAC-1_contig_1",
        "AAAC-1_contig_2",
    ]
    # Line 5 holds the bad flag: the header, a row, a blank line, a row.
    bad = AIRR_ROWS[1].replace("\tF\t", "\tmaybe\t")
    with pytest.raises(ValueError, match="holds 'maybe' on line 5$"):
        list(
            iter_airr_rearrangements(
                _write(
                    tmp_path / "bad.tsv",
                    [AIRR_HEADER, AIRR_ROWS[0], "", AIRR_ROWS[2], bad],
                ),
                chunk_rows=chunk_rows,
            )
        )
    bad = CELLRANGER_ROWS[0].replace("true,true,QQV", "true,maybe,QQV")
    with pytest.raises(ValueError, match="holds 'maybe' on line 5$"):
        list(
            iter_vdj_contigs(
                _write(
                    tmp_path / "bad.csv",
                    [
                        CELLRANGER_HEADER,
                        CELLRANGER_ROWS[1],
                        "",
                        CELLRANGER_ROWS[2],
                        bad,
                    ],
                ),
                chunk_rows=chunk_rows,
            )
        )
    # A row without a cell ID beside other values is not blank.
    with pytest.raises(ValueError, match="cell_id is missing on line 4"):
        list(
            iter_airr_rearrangements(
                _write(
                    tmp_path / "no_cell.tsv",
                    [
                        AIRR_HEADER,
                        AIRR_ROWS[0],
                        "",
                        "\t" + AIRR_ROWS[1].split("\t", 1)[1],
                    ],
                ),
                chunk_rows=chunk_rows,
            )
        )


def test_header_names_are_stripped(tmp_path) -> None:
    header = " cell_id \t sequence_id\tlocus \tjunction_aa\tproductive"
    path = _write(tmp_path / "rearrangements.tsv", [header, "c1\ts1\tTRB\tCASS\tT"])
    ((_line, chunk),) = iter_airr_rearrangements(path)
    assert list(chunk.columns) == [
        "cell_id",
        "sequence_id",
        "locus",
        "junction_aa",
        "productive",
    ]


@pytest.mark.parametrize(
    ("header", "row", "message"),
    [
        (
            "sequence_id\tlocus\tjunction_aa\tproductive",
            "s1\tTRB\tCASS\tT",
            "no cell_id field",
        ),
        (
            "cell_id\tsequence_id\tlocus\tproductive",
            "c1\ts1\tTRB\tT",
            "lacks the fields 'junction_aa'",
        ),
        (
            "cell_id\tsequence_id\tjunction_aa\tproductive",
            "c1\ts1\tCASS\tT",
            "lacks the fields 'locus'",
        ),
        (
            "cell_id\tsequence_id\tlocus\tlocus\tjunction_aa\tproductive",
            "c1\ts1\tTRB\tTRB\tCASS\tT",
            "repeats the fields 'locus'",
        ),
        (
            "cell_id\tsequence_id\tlocus\tjunction_aa\tproductive",
            "\ts1\tTRB\tCASS\tT",
            "cell_id is missing on line 2",
        ),
        (
            "cell_id\tsequence_id\tlocus\tjunction_aa\tproductive",
            "c1\ts1\tTRB\tCASS\tyes",
            "productive must be true or false",
        ),
    ],
)
def test_malformed_airr_files_are_refused(tmp_path, header, row, message) -> None:
    path = _write(tmp_path / "rearrangements.tsv", [header, row])
    with pytest.raises(ValueError, match=message):
        list(iter_airr_rearrangements(path))


def test_cellranger_columns_take_their_airr_names(tmp_path) -> None:
    path = _write(tmp_path / "contigs.csv", [CELLRANGER_HEADER, *CELLRANGER_ROWS])
    ((first_line, chunk),) = iter_vdj_contigs(path)
    assert first_line == 2
    # Fields come in storage order, whatever the file's column order.
    assert list(chunk.columns) == [
        "cell_id",
        "sequence_id",
        "locus",
        "v_call",
        "d_call",
        "j_call",
        "c_call",
        "junction",
        "junction_aa",
        "productive",
        "umi_count",
        "consensus_count",
        "is_cell",
        "high_confidence",
        "full_length",
        "fwr1",
        "fwr1_aa",
        "cdr1",
        "cdr1_aa",
        "fwr2",
        "fwr2_aa",
        "cdr2",
        "cdr2_aa",
        "fwr3",
        "fwr3_aa",
        "fwr4",
        "fwr4_aa",
        "raw_clonotype_id",
        "raw_consensus_id",
        "exact_subclonotype_id",
    ]
    assert chunk["junction_aa"].tolist() == ["CAASKF", "CASSYF", None]
    assert chunk["junction"].tolist() == ["TGTGCAGCA", "TGTGCCAGC", None]
    assert chunk["d_call"].tolist() == [None, "TRBD1", None]
    assert chunk["productive"].tolist() == [1, 0, -1]
    assert chunk["is_cell"].tolist() == [1, 1, 0]
    assert chunk["umi_count"].tolist() == [11, 9, 1]
    assert chunk["consensus_count"].tolist() == [2699, 3202, 12]
    # Each region's amino acids and nucleotides, as Cell Ranger names them.
    regions = {
        "fwr1_aa": "QQV", "fwr1": "CAGCAG", "cdr1_aa": "TSG", "cdr1": "ACCAGT",
        "fwr2_aa": "LYW", "fwr2": "CTGTAC", "cdr2_aa": "IYS", "cdr2": "ATCTAC",
        "fwr3_aa": "LSQ", "fwr3": "CTGAGC", "fwr4_aa": "FGK", "fwr4": "TTCGGC",
    }  # fmt: skip
    assert {name: chunk[name].tolist() for name in regions} == {
        name: [value, None, None] for name, value in regions.items()
    }
    assert chunk["locus"].tolist() == ["TRA", "TRB", "Multi"]
    assert chunk["exact_subclonotype_id"].tolist() == ["1", "1", None]


def test_cellranger_files_are_read_in_chunks(tmp_path) -> None:
    path = _write(
        tmp_path / "contigs.csv.gz", [CELLRANGER_HEADER, *CELLRANGER_ROWS], True
    )
    chunks = list(iter_vdj_contigs(path, chunk_rows=2))
    assert [(first_line, len(chunk)) for first_line, chunk in chunks] == [
        (2, 2),
        (4, 1),
    ]


@pytest.mark.parametrize(
    ("header", "message"),
    [
        ("barcode,contig_id,chain,productive", "lacks the columns 'cdr3'"),
        ("barcode,contig_id,chain,cdr3,productive,cdr3", "repeats the columns 'cdr3'"),
    ],
)
def test_malformed_cellranger_files_are_refused(tmp_path, header, message) -> None:
    path = _write(tmp_path / "contigs.csv", [header])
    with pytest.raises(ValueError, match=message):
        list(iter_vdj_contigs(path))
