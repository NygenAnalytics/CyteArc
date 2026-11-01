"""Reading barcode files: formats recognised from their contents, and their rows.

Each tool's header is recognised, as are pycashier's and FateMap's tables
without one; tables of other tools are read by the columns that ``columns``
names. Cell Ranger's ``molecule_info.h5`` is read in its documented layout,
with UMIs decoded as Cell Ranger packs them.
"""

import gzip
import json
import sys
from dataclasses import replace
from itertools import islice
from pathlib import Path

import h5py
import numpy as np
import pytest

from cytearc.readers.cellranger import (
    decode_umis,
    is_molecule_info,
    iter_molecule_info,
    molecule_info_library,
)
from cytearc.readers.lineage_barcodes import barcode_layout, iter_barcode_rows
from tests.clone_truth import (
    barcode_truth,
    clone_truth,
    pack_umis,
    write_molecule_info,
    write_quicat_h5ad,
)

TRUTH = clone_truth()
BARCODES = barcode_truth(TRUTH)


def _file(folder: Path, name: str, text: str) -> Path:
    path = folder / name
    path.write_text(text)
    return path


def _rows(layout, chunk_rows: int = 2):
    chunks = list(iter_barcode_rows(layout, chunk_rows))
    return chunks


# Each case: a file's text, and the format, level and columns it is read by.
_HEADERS = {
    "trex_current": (
        "#cell_id\tumi\tclone_id\tread_count\nAAAC\tGG\tTTT-0\t3\n",
        "trex",
        "molecules",
        {
            "cell": ["cell_id"],
            "barcode": ["clone_id"],
            "umi": "umi",
            "reads": "read_count",
        },
    ),
    "trex_corrected": (
        "#cell_id\tumi\toriginal_clone_id\tclone_id\tread_count\nAAAC\tGG\tTTA\tTTT\t3\n",
        "trex",
        "molecules",
        {
            "cell": ["cell_id"],
            "barcode": ["clone_id"],
            "umi": "umi",
            "reads": "read_count",
        },
    ),
    "trex_2022": (
        "#cell_id\tumi\tclone_id\nAAAC\tGG\tTTT\n",
        "trex",
        "molecules",
        {"cell": ["cell_id"], "barcode": ["clone_id"], "umi": "umi", "reads": None},
    ),
    "trex_smartseq2": (
        "#cell_id\tclone_id\nAAAC\tTTT\n",
        "trex",
        "reads",
        {"cell": ["cell_id"], "barcode": ["clone_id"], "umi": None, "reads": None},
    ),
    "pycashier": (
        "info\tumi\tcell\tbarcode\n@r1\tGG\tAAAC\tTTT\n",
        "pycashier",
        "reads",
        {"cell": ["cell"], "barcode": ["barcode"], "umi": "umi", "reads": None},
    ),
    "celltag_atac": (
        "Cell.BC\tCell.Tag\nAAAC-1\tTTT\n",
        "celltag",
        "reads",
        {"cell": ["Cell.BC"], "barcode": ["Cell.Tag"], "umi": None, "reads": None},
    ),
    "fatemap": (
        "cellID\tUMI\tOriginalBarcode\tBC50StarcodeD8\tSampleNum\nAAAC\tGG\tTTA\tTTT\t1\n",
        "fatemap",
        "molecules",
        {
            "cell": ["cellID"],
            "barcode": ["BC50StarcodeD8"],
            "umi": "UMI",
            "reads": None,
        },
    ),
    "quicat": (
        "sample,cell,barcode,counts,replicate\ns1,AAAC-1,TTT,4,r1\n",
        "quicat",
        "pairs",
        {"cell": ["cell"], "barcode": ["barcode"], "umis": "counts"},
    ),
    "quicat_wide": (
        "sample,cell,TTT,GGG\ns1,AAAC-1_s1,0,4\n",
        "quicat_wide",
        "pairs",
        {"cell": ["cell"], "barcode": []},
    ),
    "bartab": (
        "gene\tcell\tcount\nLB1\tAAAC\t4\n",
        "bartab",
        "pairs",
        {"cell": ["cell"], "barcode": ["gene"], "umis": "count"},
    ),
    "bartab_annotation": (
        "barcode\tbc.umi.count\nAAAC\tLB1;LB2\t4;1\n",
        "bartab_annotation",
        "pairs",
        {"cell": [], "barcode": ["barcode"], "umis": "bc.umi.count", "separator": ";,"},
    ),
}


@pytest.mark.parametrize("case", sorted(_HEADERS))
def test_each_tool_is_recognised_by_its_header(tmp_path, case) -> None:
    text, format, level, columns = _HEADERS[case]
    path = _file(tmp_path, "file.txt", text)

    layout = barcode_layout(path)

    assert (layout.format, layout.level) == (format, level)
    recorded = layout.record()
    assert {role: recorded["columns"][role] for role in columns} == columns
    assert recorded["library"] == "file"
    # The format named is read alike.
    assert barcode_layout(path, format=format) == layout


@pytest.mark.parametrize(
    ("text", "format", "names"),
    [
        ("@read1\tGGAT\tAAACCC\tTTTGG\n@read2\tGGAT\tAAACCC\tTTTGG\n", "pycashier", 4),
        (
            "AAAC\tGGAT\tTTA\tTTT\tTT\tT\tT\t1\nAAAC\tGGCC\tTTA\tTTT\tTT\tT\tT\t1\n",
            "fatemap",
            8,
        ),
    ],
)
def test_tables_without_a_header_are_recognised_by_their_rows(
    tmp_path, text, format, names
) -> None:
    path = _file(tmp_path, "rows.txt", text)

    layout = barcode_layout(path)

    assert layout.format == format
    (chunk,) = _rows(layout, 10)
    assert chunk.first_line == 1
    assert len(chunk.cells) == 2
    assert barcode_layout(path, format=format) == layout


def test_a_named_format_without_its_header_needs_its_columns_in_order(tmp_path) -> None:
    path = _file(tmp_path, "rows.txt", "@read1\tGGAT\tAAAC\n")
    with pytest.raises(
        ValueError, match="must hold the 4 columns of format='pycashier'"
    ):
        barcode_layout(path, format="pycashier")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("clone_nr\tcell_id\n1\tAAAC\n", "import it with ds.clones.import_clones"),
        ("a,b\n1,2\n", "format='reads', 'molecules' or 'pairs'"),
        (
            "#cell_id\tumi\tclone_id\tinfo\tcell\tbarcode\nA\tG\tT\t@r\tA\tT\n",
            "the columns of several barcode tables: trex, pycashier",
        ),
        ("\n\n", "is empty; a barcode table needs a header line"),
        ("barcode\tbc.umi.count\nLB1\t4\n", "R's row-name layout"),
        ("sample,cell,replicate\ns,A,r\n", "has no barcode columns"),
    ],
)
def test_files_that_name_no_known_table_are_refused(tmp_path, text, message) -> None:
    path = _file(tmp_path, "file.txt", text)
    format = None
    if "bc.umi.count" in text:
        format = "bartab_annotation"
    elif "replicate" in text:
        format = "quicat_wide"
    with pytest.raises(ValueError, match=message):
        barcode_layout(path, format=format)


@pytest.mark.parametrize(
    ("format", "columns", "error", "message"),
    [
        ("pairs", {"barcode": "b"}, ValueError, "has no row names, so name its cell"),
        ("pairs", {"cell": "c"}, ValueError, "needs the barcode column"),
        (
            "molecules",
            {"cell": "c", "barcode": "b"},
            ValueError,
            "needs each molecule's UMI",
        ),
        (
            "reads",
            {"cell": "c", "barcode": "b", "reads": "n"},
            ValueError,
            "reads no column for 'reads'",
        ),
        (
            "pairs",
            {"cell": "c", "barcode": "b", "umi": "u"},
            ValueError,
            "reads no column for 'umi'",
        ),
        ("pairs", {"cell": "c", "barcode": "x"}, ValueError, "has no column 'x'"),
        (
            "pairs",
            {"cell": "c", "barcode": ["b", "n"], "separator": ";"},
            ValueError,
            "must name one column",
        ),
        (
            "pairs",
            {"cell": "c", "barcode": "b", "separator": "/"},
            ValueError,
            "must be one of ';'",
        ),
        (
            "pairs",
            {"cell": 3, "barcode": "b"},
            TypeError,
            "a column name or a sequence",
        ),
        (
            "pairs",
            {"cell": "c", "barcode": "b", "umis": ["n"]},
            TypeError,
            "must be a column name",
        ),
        ("pairs", {"cell": "", "barcode": "b"}, ValueError, "names an empty column"),
        ("pairs", ["cell"], TypeError, "columns must map roles"),
        ("trex", {"cell": "c"}, ValueError, "format='trex' reads fixed columns"),
    ],
)
def test_columns_that_a_level_cannot_read_are_refused(
    tmp_path, format, columns, error, message
) -> None:
    path = _file(tmp_path, "table.tsv", "c\tb\tn\tu\nAAAC\tTT\t3\tGG\n")
    with pytest.raises(error, match=message):
        barcode_layout(path, format=format, columns=columns)


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"format": "larry"}, ValueError, "format must be one of"),
        ({"library": 3}, TypeError, "library must be text"),
        ({"library": ""}, ValueError, "must name a library"),
        (
            {
                "format": "pairs",
                "columns": {"cell": "c", "barcode": "b", "library": "u"},
                "library": "x",
            },
            ValueError,
            "library= would name no row",
        ),
        ({"format": "h5ad"}, ValueError, "is a text table, not format='h5ad'"),
    ],
)
def test_options_that_do_not_fit_the_file_are_refused(
    tmp_path, options, error, message
) -> None:
    path = _file(tmp_path, "table.tsv", "c\tb\tn\tu\nAAAC\tTT\t3\tGG\n")
    with pytest.raises(error, match=message):
        barcode_layout(path, **options)
    with pytest.raises(TypeError, match="path must be a path"):
        barcode_layout(3)


def test_a_file_without_a_name_before_its_extensions_needs_a_library(tmp_path) -> None:
    path = _file(tmp_path, ".tsv", "gene\tcell\tcount\nLB\tA\t1\n")
    with pytest.raises(ValueError, match="give it a library"):
        barcode_layout(path)
    assert barcode_layout(path, library="lib").library == "lib"


def test_rows_name_their_lines_when_a_value_is_wrong(tmp_path) -> None:
    missing_cell = _file(tmp_path, "a.tsv", "gene\tcell\tcount\nLB\tA\t1\nLB\t\t2\n")
    with pytest.raises(ValueError, match="without a cell barcode on lines 3"):
        _rows(barcode_layout(missing_cell))
    bad_count = _file(tmp_path, "b.tsv", "gene\tcell\tcount\nLB\tA\t1\nLB\tB\t2.5\n")
    with pytest.raises(ValueError, match="count must be a whole number .* line 3"):
        _rows(barcode_layout(bad_count))
    missing_umi = _file(
        tmp_path, "c.tsv", "#cell_id\tumi\tclone_id\nA\tGG\tT\nB\tNA\tT\n"
    )
    with pytest.raises(ValueError, match="without a UMI \\(umi\\) on line 3"):
        _rows(barcode_layout(missing_umi))
    missing_library = _file(tmp_path, "d.tsv", "c\tb\tl\nA\tT\tx\nB\tT\t\n")
    with pytest.raises(ValueError, match="without a library \\(l\\) on line 3"):
        _rows(
            barcode_layout(
                missing_library,
                format="pairs",
                columns={"cell": "c", "barcode": "b", "library": "l"},
            )
        )
    shifted = _file(
        tmp_path, "e.tsv", "info\tumi\tcell\tbarcode\n@r\tGG\tAA\tT\n@r_2\tx\tAA\tT\n"
    )
    with pytest.raises(ValueError, match="holds 'AA' and 'x' on line 3"):
        _rows(barcode_layout(shifted))


# Each case: a table whose bad row follows a row and two blank lines, or a
# quoted field that holds a line break, the options that read it, and the
# error, which names the file line where counting rows names line 3, or line
# 2 in a table without a header.
_PAST_BLANK_LINES = {
    "a row without a cell": (
        "gene\tcell\tcount\nLB\tA\t1\n\n\nLB\t\t2\n",
        {},
        "without a cell barcode on lines 5",
    ),
    "a count that is not whole": (
        "gene\tcell\tcount\nLB\tA\t1\n\n\nLB\tB\t2.5\n",
        {},
        r"count must be a whole number .* on line 5",
    ),
    "a molecule without a UMI": (
        "#cell_id\tumi\tclone_id\nA\tGG\tT\n\n\nB\tNA\tT\n",
        {},
        r"without a UMI \(umi\) on line 5",
    ),
    "a TREX molecule cut short": (
        "#cell_id\tumi\tclone_id\tread_count\nAAAC\tGG\tTTT\t3\n\n\nAAAC\tGC\tTTT\n",
        {},
        "without read_count on line 5",
    ),
    "a pycashier row shifted by its read name": (
        "info\tumi\tcell\tbarcode\n@r\tGG\tAA\tT\n\n\n@r_2\tx\tAA\tT\n",
        {},
        "holds 'AA' and 'x' on line 5",
    ),
    "a pycashier row of a table without a header": (
        "@r1\tGGAT\tAAAC\tTTTG\n\n\n@r_2\tx\tAAAC\tTTTG\n",
        {},
        "holds 'AAAC' and 'x' on line 4",
    ),
    "a pair without its library": (
        "c\tb\tl\nA\tT\tx\n\n\nB\tT\t\n",
        {
            "format": "pairs",
            "columns": {"cell": "c", "barcode": "b", "library": "l"},
        },
        r"without a library \(l\) on line 5",
    ),
    "lists whose counts do not align": (
        "cell\tbarcodes\tumis\nA\tx;y\t3;4\n\n\nC\tx;y\t3\n",
        {
            "format": "pairs",
            "columns": {
                "cell": "cell",
                "barcode": "barcodes",
                "umis": "umis",
                "separator": ";",
            },
        },
        "another number of umis than of barcodes on line 5",
    ),
    "a count after a quoted line break": (
        'c\tb\tn\n"A\nZ"\tT\t3\nB\tT\tx\n',
        {
            "format": "pairs",
            "columns": {"cell": "c", "barcode": "b", "umis": "n"},
        },
        "n must be a whole number of at least 0, but holds 'x' on line 4",
    ),
}


def test_a_row_of_two_bad_list_items_is_named_once(tmp_path) -> None:
    path = _file(tmp_path, "lists.tsv", "cell\tbarcodes\tumis\nA\tx;y\tz;w\n")
    columns = {"cell": "cell", "barcode": "barcodes", "umis": "umis", "separator": ";"}
    with pytest.raises(ValueError, match=r"holds 'z' on line 2$"):
        _rows(barcode_layout(path, format="pairs", columns=columns), 10)


def test_a_cell_barcode_joined_from_columns_needs_every_part(tmp_path) -> None:
    columns = {"cell": ["s", "c"], "barcode": "b"}
    whole = _file(tmp_path, "whole.tsv", "s\tc\tb\ns1\tAA\tT\ns2\tCC\tG\n")
    chunks = _rows(barcode_layout(whole, format="pairs", columns=columns), 10)
    assert chunks[0].cells.tolist() == ["s1__AA", "s2__CC"]
    for name, row in (("cell.tsv", "s2\t\tG"), ("sample.tsv", " \tCC\tG")):
        path = _file(tmp_path, name, f"s\tc\tb\ns1\tAA\tT\n\n{row}\n")
        with pytest.raises(ValueError, match="without a cell barcode on lines 4"):
            _rows(barcode_layout(path, format="pairs", columns=columns), 10)


@pytest.mark.parametrize("case", list(_PAST_BLANK_LINES))
def test_text_tables_name_the_file_line_of_a_row_past_blank_lines(
    tmp_path, case
) -> None:
    text, options, message = _PAST_BLANK_LINES[case]
    path = _file(tmp_path, "table.tsv", text)
    with pytest.raises(ValueError, match=message):
        _rows(barcode_layout(path, **options), 10)


def test_fatemap_tables_of_several_samples_are_refused(tmp_path) -> None:
    path = _file(
        tmp_path,
        "fm.txt",
        "cellID\tUMI\tBC50StarcodeD8\tSampleNum\nAAAC\tGG\tTT\t1\nAAAC\tGC\tTT\t1\n"
        "AAAC\tGA\tTT\t2\n",
    )
    with pytest.raises(ValueError, match="holds the samples 1, 2"):
        _rows(barcode_layout(path))


def test_two_barcode_columns_join_and_a_missing_half_names_no_barcode(tmp_path) -> None:
    path = _file(tmp_path, "xp.csv", "cell,bc14,bc30,n\nA,AC,GT,3\nB,AC,NA,2\n")
    layout = barcode_layout(
        path,
        format="pairs",
        columns={"cell": "cell", "barcode": ["bc14", "bc30"], "umis": "n"},
    )
    (chunk,) = _rows(layout, 10)
    assert chunk.barcodes.tolist() == ["AC__GT", None]
    assert chunk.umi_counts.tolist() == [3, 2]


def test_list_fields_split_with_their_counts_and_keep_cells_without_any(
    tmp_path,
) -> None:
    path = _file(
        tmp_path,
        "lists.tsv",
        "cell\tbarcodes\tumis\treads\nA\tx|y\t3|4\t30|40\nB\tNA\tNA\tNA\nC\tz|\t5|\t50|\n",
    )
    layout = barcode_layout(
        path,
        format="pairs",
        columns={
            "cell": "cell",
            "barcode": "barcodes",
            "umis": "umis",
            "reads": "reads",
            "separator": "|",
        },
    )
    (chunk,) = _rows(layout, 10)
    assert chunk.cells.tolist() == ["A", "A", "B", "C", "C"]
    assert chunk.barcodes.tolist() == ["x", "y", None, "z", None]
    assert chunk.umi_counts.tolist() == [3, 4, -1, 5, -1]
    assert chunk.reads.tolist() == [30, 40, -1, 50, -1]
    bad = _file(tmp_path, "bad.tsv", "cell\tbarcodes\tumis\nA\tx|y\t3|4\nB\tx|y\t3|z\n")
    with pytest.raises(
        ValueError,
        match="umis must be a whole number of at least 0, but holds 'z' on line 3",
    ):
        _rows(
            barcode_layout(
                bad,
                format="pairs",
                columns={
                    "cell": "cell",
                    "barcode": "barcodes",
                    "umis": "umis",
                    "separator": "|",
                },
            ),
            10,
        )


# The text that pandas reads as missing by default, and "none": each means no
# value in a barcode table, as in every record table of metadata/records.py.
_MISSING_TOKENS = (
    "",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "None",
    "n/a",
    "nan",
    "none",
    "null",
)


def test_missing_value_tokens_name_no_barcode_and_no_count(tmp_path) -> None:
    n_tokens = len(_MISSING_TOKENS)
    rows = [f"A{at}\t{token}\t5\t9" for at, token in enumerate(_MISSING_TOKENS)]
    rows += [
        f"B{at}\tbc{at}\t{token}\t{token}" for at, token in enumerate(_MISSING_TOKENS)
    ]
    # Text that pandas reads as a value stays a barcode, and a cell barcode
    # that spells a token is a cell barcode like any other.
    rows += ["C0\tNAN\t1\t1", "C1\tna\t1\t1", "NA\tNULLS\t2\t2"]
    path = _file(
        tmp_path, "pairs.tsv", "cell\tbarcode\tumis\treads\n" + "\n".join(rows) + "\n"
    )
    layout = barcode_layout(
        path,
        format="pairs",
        columns={
            "cell": "cell",
            "barcode": "barcode",
            "umis": "umis",
            "reads": "reads",
        },
    )
    (chunk,) = _rows(layout, 1_000)
    assert chunk.cells.tolist() == (
        [f"A{at}" for at in range(n_tokens)]
        + [f"B{at}" for at in range(n_tokens)]
        + ["C0", "C1", "NA"]
    )
    assert chunk.barcodes.tolist() == (
        [None] * n_tokens
        + [f"bc{at}" for at in range(n_tokens)]
        + ["NAN", "na", "NULLS"]
    )
    assert chunk.umi_counts.tolist() == [5] * n_tokens + [-1] * n_tokens + [1, 1, 2]
    assert chunk.reads.tolist() == [9] * n_tokens + [-1] * n_tokens + [1, 1, 2]
    # In lists, a token is an item without a barcode or without a count.
    listed = _file(
        tmp_path,
        "lists.tsv",
        "cell\tbarcodes\tumis\n"
        + "".join(
            f"A{at}\tx|{token}\t{token}|4\n" for at, token in enumerate(_MISSING_TOKENS)
        ),
    )
    (chunk,) = _rows(
        barcode_layout(
            listed,
            format="pairs",
            columns={
                "cell": "cell",
                "barcode": "barcodes",
                "umis": "umis",
                "separator": "|",
            },
        ),
        1_000,
    )
    assert chunk.barcodes.tolist() == ["x", None] * n_tokens
    assert chunk.umi_counts.tolist() == [-1, 4] * n_tokens
    # A molecule needs its UMI, so a token in its place is refused.
    for token in _MISSING_TOKENS:
        molecules = _file(
            tmp_path,
            "molecules.tsv",
            f"#cell_id\tumi\tclone_id\nA\tGG\tT\nB\t{token}\tT\n",
        )
        with pytest.raises(ValueError, match=r"without a UMI \(umi\) on line 3"):
            _rows(barcode_layout(molecules))


def test_tool_formats_refuse_a_missing_value_where_the_tool_writes_one(
    tmp_path,
) -> None:
    # TREX fills clone_id, and BARtab gene, in every row, so a token there is
    # a row cut short.
    trex = _file(
        tmp_path,
        "molecules.txt",
        "#cell_id\tumi\tclone_id\nAAAC\tGG\tNA\nAAAG\tGT\tT\n",
    )
    with pytest.raises(ValueError, match="without clone_id on line 2.*cut short"):
        _rows(barcode_layout(trex))
    bartab = _file(
        tmp_path, "x.counts.tsv", "gene\tcell\tcount\nN/A\tAAAC\t2\nLB\tAAAG\t3\n"
    )
    with pytest.raises(ValueError, match="without gene on line 2.*cut short"):
        _rows(barcode_layout(bartab))
    # Barcodes that head the columns of QuiCAT's merged table, and the names
    # of an AnnData file, are read as written.
    wide = _file(
        tmp_path,
        "barcodes_output.csv",
        "sample,cell,NA,ACGT\ns1,AAAC,2,1\ns1,AAAG,0,3\n",
    )
    chunks = _rows(barcode_layout(wide), 100)
    assert sorted(
        (cell, barcode)
        for chunk in chunks
        for cell, barcode in zip(chunk.cells, chunk.barcodes, strict=True)
    ) == [("AAAC", "ACGT"), ("AAAC", "NA"), ("AAAG", "ACGT")]
    import anndata
    import pandas as pd

    named = tmp_path / "named.h5ad"
    anndata.AnnData(
        X=np.asarray([[2.0, 0.0], [0.0, 3.0]]),
        obs=pd.DataFrame(index=["NA", "AAAG"]),
        var=pd.DataFrame(index=["NA", "ACGT"]),
    ).write_h5ad(named)
    chunks = _rows(barcode_layout(named), 100)
    assert sorted(
        (cell, barcode)
        for chunk in chunks
        for cell, barcode in zip(chunk.cells, chunk.barcodes, strict=True)
    ) == [("AAAG", "ACGT"), ("NA", "NA")]


def test_bartab_annotations_split_both_list_separators(tmp_path) -> None:
    current = _file(
        tmp_path,
        "current.tsv",
        "barcode\tbc.umi.count\nAAAC\tLB1;LB2\t15;12\nCCCG\tLB3\t1\n",
    )
    legacy = _file(
        tmp_path,
        "legacy.tsv",
        "barcode bc.umi.count\nAAAC-1 LB1,LB2 15,12\nCCCG-1 LB3 1\n",
    )
    for path, suffix in ((current, ""), (legacy, "-1")):
        (chunk,) = _rows(barcode_layout(path), 10)
        assert chunk.cells.tolist() == [f"AAAC{suffix}"] * 2 + [f"CCCG{suffix}"]
        assert chunk.barcodes.tolist() == ["LB1", "LB2", "LB3"]
        assert chunk.umi_counts.tolist() == [15, 12, 1]


def test_quicats_wide_table_gives_its_nonzero_counts(tmp_path) -> None:
    path = _file(
        tmp_path,
        "barcodes_output.csv",
        "sample,cell,condition,TTT,GGG\ns1,A_s1,c,0,4\ns1,B_s1,c,2,0\ns1,C_s1,c,0,0\n",
    )
    chunks = _rows(barcode_layout(path), 2)
    assert [chunk.first_line for chunk in chunks] == [2, 4]
    cells = np.concatenate([chunk.cells for chunk in chunks]).tolist()
    barcodes = np.concatenate([chunk.barcodes for chunk in chunks]).tolist()
    counts = np.concatenate([chunk.umi_counts for chunk in chunks]).tolist()
    assert list(zip(cells, barcodes, counts, strict=True)) == [
        ("A_s1", "GGG", 4),
        ("B_s1", "TTT", 2),
    ]


def test_gzip_compressed_tables_read_whatever_their_names(tmp_path) -> None:
    path = tmp_path / "counts.tsv"
    path.write_bytes(gzip.compress(b"gene\tcell\tcount\nLB1\tAAAC\t4\n"))
    (chunk,) = _rows(barcode_layout(path), 10)
    assert chunk.cells.tolist() == ["AAAC"]


def _quicat_h5ad(
    tmp_path: Path, layout: str = "csr", *, with_anndata: bool = False
) -> Path:
    return write_quicat_h5ad(
        BARCODES, tmp_path / f"{layout}.h5ad", layout=layout, with_anndata=with_anndata
    )


@pytest.mark.parametrize("layout", ["csr", "csc", "dense"])
def test_files_that_current_anndata_writes_give_the_same_pairs(
    tmp_path, layout
) -> None:
    written = tmp_path / "written"
    written.mkdir()
    by_anndata = _quicat_h5ad(tmp_path, layout, with_anndata=True)
    with h5py.File(by_anndata, "r") as handle:
        # Each index is a nullable string array, a group of values and a mask.
        assert isinstance(handle["obs/_index"], h5py.Group)
        assert isinstance(handle["var/_index"], h5py.Group)
    expected = _rows(barcode_layout(_quicat_h5ad(written, layout)), 7)
    found = _rows(barcode_layout(by_anndata), 7)

    def pairs(chunks) -> list:
        return sorted(
            (cell, barcode, int(count))
            for chunk in chunks
            for cell, barcode, count in zip(
                chunk.cells, chunk.barcodes, chunk.umi_counts, strict=True
            )
        )

    assert pairs(found) == pairs(expected) != []


def test_a_missing_name_and_an_index_of_another_form_are_refused(tmp_path) -> None:
    path = _quicat_h5ad(tmp_path, with_anndata=True)
    with h5py.File(path, "r+") as handle:
        mask = handle["var/_index/mask"]
        flags = np.asarray(mask[:], dtype=bool)
        flags[2] = True
        mask[...] = flags
    with pytest.raises(ValueError, match="name missing from the index .* row 3"):
        _rows(barcode_layout(path), 7)
    # An index that is neither a dataset of names nor a nullable string array.
    folder = tmp_path / "odd"
    folder.mkdir()
    odd = _quicat_h5ad(folder)
    with h5py.File(odd, "r+") as handle:
        names = handle["obs/_index"][:]
        del handle["obs/_index"]
        handle["obs"].create_dataset("_index", data=np.stack([names, names], axis=1))
    with pytest.raises(ValueError, match="stores its cell names in a form CyteArc"):
        barcode_layout(odd)


@pytest.mark.parametrize("layout", ["csr", "csc", "dense"])
def test_anndata_counts_are_pairs_of_their_cells_and_barcodes(tmp_path, layout) -> None:
    path = _quicat_h5ad(tmp_path, layout)
    with h5py.File(path, "r") as handle:
        cells = [value.decode() for value in handle["obs/_index"][:]]
        names = [value.decode() for value in handle["var/_index"][:]]
    read = barcode_layout(path)
    assert (read.format, read.level, read.umis) == ("h5ad", "pairs", "X")
    chunks = _rows(read, 7)
    pairs = sorted(
        (cell, barcode, int(count))
        for chunk in chunks
        for cell, barcode, count in zip(
            chunk.cells, chunk.barcodes, chunk.umi_counts, strict=True
        )
    )
    expected = []
    for sample in ("d2", "d4"):
        molecules = BARCODES.sample_molecules(sample)
        counted = molecules.groupby(["cell_barcode", "barcode"]).size()
        expected += [
            (f"{cell}-1_{sample}", barcode, int(count))
            for (cell, barcode), count in counted.items()
        ]
    assert pairs == sorted(expected)
    assert set(names) >= {barcode for _cell, barcode, _count in expected}
    assert len(cells) == len({cell for cell, _barcode, _count in expected})


def test_anndata_counts_may_be_reads_and_libraries_an_obs_column(tmp_path) -> None:
    path = _quicat_h5ad(tmp_path)
    layout = barcode_layout(
        path, format="h5ad", columns={"reads": "X", "library": "sample"}
    )
    (chunk,) = _rows(layout, 10_000)
    assert chunk.umi_counts is None
    assert (chunk.reads > 0).all()
    assert sorted(set(chunk.libraries)) == ["d2", "d4"]
    assert all(
        cell.endswith(library)
        for cell, library in zip(chunk.cells, chunk.libraries, strict=True)
    )


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        ({"cell": "x"}, "columns takes only 'umis' or 'reads'"),
        ({"umis": "X", "reads": "X"}, "name one of them"),
        ({"umis": "layers/counts"}, "must name 'X'"),
        ({"library": "absent"}, "has no obs column 'absent'"),
    ],
)
def test_anndata_columns_that_it_cannot_read_are_refused(
    tmp_path, columns, message
) -> None:
    path = _quicat_h5ad(tmp_path)
    with pytest.raises(ValueError, match=message):
        barcode_layout(path, format="h5ad", columns=columns)


def test_anndata_files_without_their_tables_or_with_wrong_counts_are_refused(
    tmp_path,
) -> None:
    path = _quicat_h5ad(tmp_path, "dense")
    with h5py.File(path, "a") as handle:
        handle["X"][0, 0] = 1.5
    with pytest.raises(ValueError, match="must be whole numbers"):
        _rows(barcode_layout(path), 1_000)
    with h5py.File(path, "a") as handle:
        del handle["X"]
        handle.create_dataset("X", data=np.ones((2, 2)))
    with pytest.raises(ValueError, match="is 2 by 2, but the file names"):
        _rows(barcode_layout(path), 1_000)
    with h5py.File(path, "a") as handle:
        del handle["X"]
    with pytest.raises(ValueError, match="has no X matrix"):
        barcode_layout(path)
    with h5py.File(path, "a") as handle:
        del handle["var"]
    with pytest.raises(ValueError, match="no AnnData var table"):
        barcode_layout(path)
    with h5py.File(path, "a") as handle:
        del handle["obs"]
    with pytest.raises(ValueError, match="no AnnData obs table"):
        barcode_layout(path)
    with pytest.raises(ValueError, match="is an HDF5 file, not format='trex'"):
        barcode_layout(path, format="trex")


def test_umis_decode_as_cell_ranger_packs_them() -> None:
    umis = [
        "AGTACTCCGAAA",
        "TAGATTCACGCT",
        "ATTTATTAATGT",
        "AAAAAAAAAAAA",
        "TTTTTTTTTTTT",
    ]
    # The values that Cell Ranger's molecule_info.h5 of two 10x Genomics
    # datasets (CC BY 4.0) store for these UMIs.
    assert pack_umis(umis[:3]).tolist() == [2913664, 13169767, 4144187]
    assert decode_umis(pack_umis(umis), 12).tolist() == umis
    assert decode_umis(np.asarray([0, 3], dtype=np.uint32), 2).tolist() == ["AA", "AT"]
    assert decode_umis(np.empty(0, dtype=np.uint32), 10).tolist() == []


def test_molecule_info_gives_the_molecules_of_its_lineage_library(tmp_path) -> None:
    path = write_molecule_info(BARCODES, tmp_path / "molecule_info.h5", sample="d4")
    assert is_molecule_info(path)
    library = molecule_info_library(path)
    assert (library.library_type, library.library_indices, library.umi_length) == (
        "Custom",
        (1,),
        12,
    )
    rows = list(iter_molecule_info(library, chunk_rows=1))
    molecules = BARCODES.sample_molecules("d4")
    barcodes = np.concatenate([chunk.barcodes for chunk in rows])
    umis = np.concatenate([chunk.umis for chunk in rows])
    reads = np.concatenate([chunk.reads for chunk in rows])
    features = np.concatenate([chunk.features for chunk in rows])
    names = {
        barcode: f"LB{index}"
        for index, barcode in enumerate(sorted(molecules["true_barcode"].unique()))
    }
    assert sorted(zip(barcodes, umis, features, reads, strict=True)) == sorted(
        (f"{row.cell_barcode}-1", row.umi, names[row.true_barcode], row.reads)
        for row in molecules.itertuples(index=False)
    )
    layout = barcode_layout(path)
    assert (layout.format, layout.library, layout.level) == (
        "molecule_info",
        "Custom",
        "molecules",
    )


def test_molecule_info_names_the_library_to_read(tmp_path) -> None:
    path = write_molecule_info(BARCODES, tmp_path / "molecule_info.h5", sample="d4")
    assert molecule_info_library(path, "Custom").library_type == "Custom"
    with pytest.raises(ValueError, match="holds no library of type 'Antibody Capture'"):
        molecule_info_library(path, "Antibody Capture")
    libraries = [
        {"library_id": 0, "library_type": "Gene Expression", "gem_group": 1},
        {"library_id": 1, "library_type": "Custom", "gem_group": 1},
        {"library_id": 2, "library_type": "Antibody Capture", "gem_group": 1},
    ]
    with h5py.File(path, "a") as handle:
        del handle["library_info"]
        handle.create_dataset(
            "library_info", data=np.asarray([json.dumps(libraries)], dtype="S4096")
        )
    with pytest.raises(ValueError, match="'Antibody Capture', 'Custom'; name the"):
        molecule_info_library(path)
    with h5py.File(path, "a") as handle:
        del handle["library_info"]
        handle.create_dataset(
            "library_info", data=np.asarray([json.dumps(libraries[:1])], dtype="S4096")
        )
    with pytest.raises(ValueError, match="holds no feature barcode library"):
        molecule_info_library(path)


def test_molecule_info_without_a_recorded_umi_length_is_refused(tmp_path) -> None:
    path = write_molecule_info(
        BARCODES, tmp_path / "molecule_info.h5", sample="d2", umi_length=None
    )
    with pytest.raises(ValueError, match="records no UMI length"):
        molecule_info_library(path)
    with h5py.File(path, "a") as handle:
        del handle["metrics_json"]
        handle.create_dataset(
            "metrics_json",
            data=json.dumps({"chemistry_defs": {"Custom": {"umi": [{"length": 18}]}}}),
            dtype=h5py.string_dtype(),
        )
    with pytest.raises(ValueError, match="a 32-bit packed UMI holds 1 to 16 bases"):
        molecule_info_library(path)
    with h5py.File(path, "a") as handle:
        del handle["metrics_json"]
        handle.create_dataset(
            "metrics_json",
            data=json.dumps({"chemistry_defs": {"Custom": {"umi": {"length": 10}}}}),
            dtype=h5py.string_dtype(),
        )
    assert molecule_info_library(path).umi_length == 10


def test_molecule_info_files_that_break_the_layout_are_refused(tmp_path) -> None:
    path = write_molecule_info(BARCODES, tmp_path / "molecule_info.h5", sample="d2")
    with h5py.File(path, "a") as handle:
        # A molecule of the lineage library names a barcode the file lacks.
        lineage = int(np.flatnonzero(handle["library_idx"][:] == 1)[0])
        handle["barcode_idx"][lineage] = 10**6
    with pytest.raises(ValueError, match="barcodes or features that it does not hold"):
        list(iter_molecule_info(molecule_info_library(path)))
    with h5py.File(path, "a") as handle:
        del handle["count"]
        handle.create_dataset("count", data=np.ones(3, dtype=np.uint32))
    with pytest.raises(ValueError, match="differ in length"):
        molecule_info_library(path)
    with h5py.File(path, "a") as handle:
        del handle["umi"]
    with pytest.raises(ValueError, match="lacks umi"):
        molecule_info_library(path)
    assert not is_molecule_info(path)
    with h5py.File(path, "a") as handle:
        del handle["library_info"]
        handle.create_dataset("library_info", data=np.asarray([b"[1, 2]"]))
        handle.create_dataset("umi", data=np.ones(3, dtype=np.uint32))
    with pytest.raises(ValueError, match="lists no library types"):
        molecule_info_library(path)


def test_a_table_in_rs_row_name_layout_takes_its_cells_from_the_row_names(
    tmp_path,
) -> None:
    # Watermelon's GEO metadata: R row names, and comma lists of barcodes.
    path = _file(
        tmp_path,
        "metaData_with_lineage.txt",
        '"sample_id" "lineage_barcode"\n"AAAC-1" "1" "TTT,GGG"\n"CCCG-1" "1" NA\n',
    )
    layout = barcode_layout(
        path,
        format="pairs",
        columns={"barcode": "lineage_barcode", "separator": ","},
    )
    assert layout.cell == ()
    (chunk,) = _rows(layout, 10)
    assert chunk.cells.tolist() == ["AAAC-1", "AAAC-1", "CCCG-1"]
    assert chunk.barcodes.tolist() == ["TTT", "GGG", None]
    assert chunk.umi_counts is None


def test_anndata_counts_skip_blocks_without_counts_and_refuse_odd_matrices(
    tmp_path,
) -> None:
    path = _quicat_h5ad(tmp_path, "dense")
    with h5py.File(path, "a") as handle:
        values = handle["X"][:]
        values[0] = 0
        del handle["X"]
        handle.create_dataset("X", data=values)
    # The first row holds no count, so the first chunk starts on the second.
    chunks = _rows(barcode_layout(path), values.shape[1])
    assert chunks[0].first_line == 2
    assert max(len(chunk.cells) for chunk in chunks) <= values.shape[1]
    assert sum(len(chunk.cells) for chunk in chunks) == np.count_nonzero(values)
    with h5py.File(path, "a") as handle:
        del handle["X"]
        group = handle.create_group("X")
        group.attrs["shape"] = np.asarray(values.shape)
    with pytest.raises(ValueError, match="neither a dense nor a sparse matrix"):
        _rows(barcode_layout(path), 10)


def test_an_obs_column_that_lacks_a_cells_library_is_refused(tmp_path) -> None:
    path = _quicat_h5ad(tmp_path)
    with h5py.File(path, "a") as handle:
        codes = handle["obs/sample/codes"][:]
        codes[0] = -1
        handle["obs/sample/codes"][:] = codes
    layout = barcode_layout(path, format="h5ad", columns={"library": "sample"})
    with pytest.raises(ValueError, match="lacks the library of some cells"):
        _rows(layout, 10_000)


def test_molecule_info_skips_rows_of_other_libraries(tmp_path, monkeypatch) -> None:
    from cytearc.readers import cellranger

    path = write_molecule_info(BARCODES, tmp_path / "molecule_info.h5", sample="d4")
    library = molecule_info_library(path)
    monkeypatch.setattr(cellranger, "MIN_MOLECULE_ROWS", 1)
    chunks = list(iter_molecule_info(library, chunk_rows=1))
    with h5py.File(path, "r") as handle:
        lineage = np.flatnonzero(handle["library_idx"][:] == 1)
    assert [chunk.first_row for chunk in chunks] == lineage.tolist()


def test_quicats_wide_table_parses_each_chunk_in_one_pass(
    tmp_path, monkeypatch
) -> None:
    from cytearc.readers import lineage_barcodes as reader

    names = [f"B{index}" for index in range(300)]
    rows = [
        f"s1,C{row}_s1,"
        + ",".join("2" if column == row else "0" for column in range(300))
        for row in range(40)
    ]
    path = _file(
        tmp_path,
        "barcodes_output.csv",
        "sample,cell," + ",".join(names) + "\n" + "\n".join(rows) + "\n",
    )
    calls = []
    parse = reader._parsed_counts

    def counted(*arguments, **options):
        calls.append(1)
        return parse(*arguments, **options)

    # The parser of a text table's counts, which a column at a time would call
    # once for each barcode.
    monkeypatch.setattr(reader, "_parsed_counts", counted)
    chunks = _rows(barcode_layout(path), 10)
    # Parsing one column at a time made a wide table's cost grow with the
    # square of its barcodes.
    assert len(calls) <= len(chunks) == 4
    pairs = sorted(
        (cell, barcode, int(count))
        for chunk in chunks
        for cell, barcode, count in zip(
            chunk.cells, chunk.barcodes, chunk.umi_counts, strict=True
        )
    )
    assert pairs == sorted((f"C{row}_s1", f"B{row}", 2) for row in range(40))


def _python_calls(layout, chunk_rows: int) -> int:
    calls = 0

    def count(frame, event, argument) -> None:
        nonlocal calls
        calls += event == "call"

    sys.setprofile(count)
    try:
        _rows(layout, chunk_rows)
    finally:
        sys.setprofile(None)
    return calls


def test_quicats_wide_table_does_python_work_per_row_not_per_barcode(
    tmp_path,
) -> None:
    def wide(n_barcodes: int) -> Path:
        rows = [
            f"s1,C{row}_s1,"
            + ",".join("3" if column == row else "0" for column in range(n_barcodes))
            for row in range(30)
        ]
        header = "sample,cell," + ",".join(f"B{index}" for index in range(n_barcodes))
        return _file(
            tmp_path, f"wide_{n_barcodes}.csv", header + "\n" + "\n".join(rows) + "\n"
        )

    narrow, broad = (barcode_layout(wide(n_barcodes)) for n_barcodes in (300, 1_500))
    # Parsing each column of each chunk made a merged table's cost grow with
    # its rows times the square of its barcodes, since more barcodes also
    # mean smaller chunks.
    assert _python_calls(broad, 10) - _python_calls(narrow, 10) < 100
    (chunk,) = _rows(broad, 30)
    assert chunk.cells.tolist() == [f"C{row}_s1" for row in range(30)]
    assert chunk.barcodes.tolist() == [f"B{row}" for row in range(30)]
    assert chunk.umi_counts.tolist() == [3] * 30


def test_quicats_wide_table_names_the_lines_of_blank_lines_and_quotes(
    tmp_path,
) -> None:
    path = _file(
        tmp_path,
        "barcodes_output.csv",
        'sample,cell,TTT,GGG\ns1,"A,1",0,4\n\n  \ns1, B ,"2",0\ns1,C,1.5,0\n',
    )
    # Counting rows instead of lines named line 4.
    with pytest.raises(
        ValueError, match=r"holds '1\.5' as the count of barcode 'TTT' on line 6;"
    ):
        _rows(barcode_layout(path), 10)
    chunks = list(islice(iter_barcode_rows(barcode_layout(path), 1), 2))
    assert [chunk.first_line for chunk in chunks] == [2, 5]
    assert [chunk.cells.tolist() for chunk in chunks] == [["A,1"], ["B"]]
    assert [chunk.barcodes.tolist() for chunk in chunks] == [["GGG"], ["TTT"]]
    assert [chunk.umi_counts.tolist() for chunk in chunks] == [[4], [2]]


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("s1,B_s1,0", "holds 3 fields on line 3, where its header names 4 columns"),
        ("s1,B_s1,0,1,5", "holds 5 fields on line 3, where its header names 4 columns"),
        ("s1,,0,1", "rows without a cell barcode on lines 3"),
        ("s1,B_s1,inf,1", "holds 'inf' as the count of barcode 'TTT' on line 3"),
        ("s1,B_s1,1,nan", "holds 'nan' as the count of barcode 'GGG' on line 3"),
    ],
)
def test_quicats_wide_table_refuses_rows_of_another_width_or_without_counts(
    tmp_path, row, message
) -> None:
    path = _file(
        tmp_path, "barcodes_output.csv", f"sample,cell,TTT,GGG\ns1,A_s1,0,4\n{row}\n"
    )
    with pytest.raises(ValueError, match=message):
        _rows(barcode_layout(path), 10)


def test_quicats_wide_table_reads_a_gzip_file_and_names_one_cut_short(
    tmp_path,
) -> None:
    text = b"sample,cell,TTT,GGG\n" + b"".join(
        b"s1,C%d_s1,%d,0\n" % (row, row % 3) for row in range(2_000)
    )
    path = tmp_path / "barcodes_output.csv"
    path.write_bytes(gzip.compress(text))
    chunks = _rows(barcode_layout(path), 500)
    assert sum(len(chunk.cells) for chunk in chunks) == sum(
        1 for row in range(2_000) if row % 3
    )
    path.write_bytes(gzip.compress(text)[:-40])
    with pytest.raises(ValueError, match="barcodes_output.csv is cut short or corrupt"):
        _rows(barcode_layout(path), 500)


@pytest.mark.parametrize("entry", ["", "NA", "x", "1.5", "-1"])
def test_quicats_wide_table_refuses_entries_that_are_not_counts(
    tmp_path, entry
) -> None:
    path = _file(
        tmp_path,
        "barcodes_output.csv",
        f"sample,cell,TTT,GGG\ns1,A_s1,0,4\ns1,B_s1,{entry},0\n",
    )
    with pytest.raises(ValueError, match="on line 3"):
        _rows(barcode_layout(path), 10)


def test_quicats_wide_table_refuses_an_index_column(tmp_path) -> None:
    # A merged table saved again by pandas gains an unnamed index column.
    path = _file(
        tmp_path,
        "barcodes_output.csv",
        ",sample,cell,TTT\n0,s1,A_s1,3\n1,s1,B_s1,0\n",
    )
    with pytest.raises(ValueError, match="unnamed column"):
        barcode_layout(path, format="quicat_wide")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        # A pycashier read cut short before its barcode.
        (
            "info\tumi\tcell\tbarcode\n@r1\tGGAT\tAAAC\tTTTG\n@r2\tGGAT\tAAAC\n",
            "without barcode on line 3",
        ),
        # A TREX molecule without its reads.
        (
            "#cell_id\tumi\tclone_id\tread_count\nAAAC\tGG\tTTT\t3\nAAAC\tGC\tTTT\n",
            "without read_count on line 3",
        ),
        # A BARtab pair without its UMIs.
        ("gene\tcell\tcount\nLB1\tAAAC\t4\nLB2\tAAAC\n", "without count on line 3"),
        # A CellTag read without its tag.
        (
            "Cell.BC\tUMI\tCell.Tag\nAAAC-1\tGG\tTTTT\nAAAC-1\tGG\n",
            "without Cell.Tag on line 3",
        ),
        # A QuiCAT pair with a barcode and no count.
        (
            "sample,cell,barcode,counts\ns1,AAAC-1,TTT,4\ns1,CCCG-1,GGG,\n",
            "without counts on line 3",
        ),
        # A FateMap molecule cut after its barcode, without its sample.
        (
            "cellID\tUMI\tOriginalBarcode\tBC50StarcodeD8\tSampleNum\n"
            "AAAC\tGG\tTTTA\tTTTT\t1\nCCCG\tGC\tTTTA\tTTTT\n",
            "without SampleNum on line 3",
        ),
        # A BARtab annotation without its UMI list.
        (
            "barcode\tbc.umi.count\nAAAC\tLB1\t4\nCCCG\tLB2\n",
            "without bc.umi.count on line 3",
        ),
    ],
)
def test_tool_tables_refuse_rows_cut_short(tmp_path, text, message) -> None:
    path = _file(tmp_path, "table.txt", text)
    with pytest.raises(ValueError, match=message):
        _rows(barcode_layout(path), 10)


def test_quicats_cells_without_barcodes_hold_no_count(tmp_path) -> None:
    path = _file(
        tmp_path,
        "sample_output.csv",
        "sample,cell,barcode,counts\ns1,AAAC-1,TTT,4\ns1,CCCG-1,,\n",
    )
    (chunk,) = _rows(barcode_layout(path), 10)
    assert chunk.barcodes.tolist() == ["TTT", None]
    assert chunk.umi_counts.tolist() == [4, -1]


def test_dense_anndata_reads_its_rows_within_the_chunk(tmp_path) -> None:
    import tracemalloc

    # 400 cells by 20,000 barcodes of float32 counts: 32 MB in memory.
    n_cells, n_barcodes = 400, 20_000
    path = tmp_path / "dense.h5ad"
    with h5py.File(path, "w") as handle:
        from tests.clone_truth import _obs_frame

        _obs_frame(handle, [f"C{index}" for index in range(n_cells)])
        var = handle.create_group("var")
        var.attrs["_index"] = "_index"
        var.create_dataset(
            "_index",
            data=np.asarray(
                [f"B{index}" for index in range(n_barcodes)], dtype=h5py.string_dtype()
            ),
        )
        values = handle.create_dataset(
            "X",
            shape=(n_cells, n_barcodes),
            dtype=np.float32,
            chunks=(10, n_barcodes),
            compression="gzip",
        )
        for row in range(n_cells):
            line = np.zeros(n_barcodes, dtype=np.float32)
            line[row] = 3
            values[row] = line
    layout = barcode_layout(path)
    entries = layout.chunk_rows(2 * 2**20)
    tracemalloc.start()
    chunks = _rows(layout, entries)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert sum(len(chunk.cells) for chunk in chunks) == n_cells
    assert peak < 8 * 2**20


def test_two_cell_columns_join_a_samples_number_to_its_barcodes(tmp_path) -> None:
    # FateMap's table of two samples, read by both cell columns.
    path = _file(
        tmp_path,
        "fm.txt",
        "cellID\tUMI\tBC50StarcodeD8\tSampleNum\nAAAC\tGG\tTT\t1\nAAAC\tGC\tTT\t2\n",
    )
    layout = barcode_layout(
        path,
        format="molecules",
        columns={
            "cell": ["SampleNum", "cellID"],
            "umi": "UMI",
            "barcode": "BC50StarcodeD8",
        },
    )
    (chunk,) = _rows(layout, 10)
    assert chunk.cells.tolist() == ["1__AAAC", "2__AAAC"]


def test_cell_barcodes_are_stripped_of_padding(tmp_path) -> None:
    path = _file(
        tmp_path, "pairs.csv", 'cell,barcode,n\n" AAAC-1 ",TT,3\nCCCG-1  ,GG,2\n'
    )
    layout = barcode_layout(
        path,
        format="pairs",
        columns={"cell": "cell", "barcode": "barcode", "umis": "n"},
    )
    (chunk,) = _rows(layout, 10)
    assert chunk.cells.tolist() == ["AAAC-1", "CCCG-1"]


def test_molecule_info_names_cells_by_their_gem_well(tmp_path) -> None:
    path = write_molecule_info(BARCODES, tmp_path / "molecule_info.h5", sample="d4")
    with h5py.File(path, "a") as handle:
        wells = handle["gem_group"][:]
        wells[::2] = 2
        handle["gem_group"][:] = wells
    rows = list(iter_molecule_info(molecule_info_library(path)))
    with h5py.File(path, "r") as handle:
        lineage = handle["library_idx"][:] == 1
        expected_wells = handle["gem_group"][:][lineage]
    barcodes = np.concatenate([chunk.barcodes for chunk in rows])
    assert [barcode.rsplit("-", 1)[1] for barcode in barcodes] == [
        str(well) for well in expected_wells
    ]
    assert set(expected_wells.tolist()) == {1, 2}


def test_trexs_early_clone_tables_point_to_import_clones(tmp_path) -> None:
    path = _file(tmp_path, "clones.txt", "#clone_id,cell_id\n1,AAAC\n1,CCCG\n")
    with pytest.raises(ValueError, match="import it with ds.clones.import_clones"):
        barcode_layout(path)


def test_a_bartab_annotation_without_cells_reads_no_pair(tmp_path) -> None:
    path = _file(tmp_path, "annotation.tsv", "barcode\tbc.umi.count\n")
    layout = barcode_layout(path)
    assert layout.format == "bartab_annotation"
    assert _rows(layout, 10) == []


def test_corrupt_gzip_files_are_refused_naming_the_file(tmp_path) -> None:
    whole = gzip.compress(b"gene\tcell\tcount\n" + b"LB1\tAAAC\t4\n" * 50_000)
    path = tmp_path / "counts.tsv.gz"
    path.write_bytes(whole[: len(whole) // 2])
    with pytest.raises(ValueError, match="counts.tsv.gz is cut short or corrupt"):
        _rows(barcode_layout(path), 1_000)


def test_molecule_info_reads_each_block_of_barcodes_once() -> None:
    from cytearc.readers.cellranger import BarcodeBlocks

    reads = []

    class Listed:
        shape = (11,)
        chunks = (4,)

        def __getitem__(self, selection):
            reads.append((selection.start, selection.stop))
            return np.asarray([f"B{index}".encode() for index in range(11)])[selection]

    blocks = BarcodeBlocks(Listed(), block_rows=4)
    assert blocks.texts(np.asarray([0, 1, 1, 3])).tolist() == ["B0", "B1", "B1", "B3"]
    assert blocks.texts(np.asarray([2, 3])).tolist() == ["B2", "B3"]
    # The last block holds three barcodes, so its offset is no multiple of
    # its length.
    assert blocks.texts(np.asarray([3, 4, 8, 10])).tolist() == ["B3", "B4", "B8", "B10"]
    # Each block of four barcodes is read once, however many chunks name it.
    assert reads == [(0, 4), (4, 8), (8, 11)]


def _matrix_h5ad(
    path: Path,
    values: np.ndarray,
    *,
    layout: str,
    chunks=None,
    compression=None,
    libraries=None,
) -> Path:
    """An AnnData file of the cells by barcodes ``values``, dense or sparse.

    ``libraries`` labels each cell in the ``obs`` column ``library``.
    """
    from tests.clone_truth import _obs_frame, write_categorical

    n_cells, n_barcodes = values.shape
    with h5py.File(path, "w") as handle:
        obs = _obs_frame(handle, [f"C{index}" for index in range(n_cells)])
        if libraries is not None:
            write_categorical(obs, "library", list(libraries))
        var = handle.create_group("var")
        var.attrs["_index"] = "_index"
        var.create_dataset(
            "_index",
            data=np.asarray(
                [f"B{index}" for index in range(n_barcodes)], dtype=h5py.string_dtype()
            ),
        )
        if layout == "dense":
            handle.create_dataset(
                "X", data=values, chunks=chunks, compression=compression
            )
            return path
        major = values if layout == "csr" else values.T
        group = handle.create_group("X")
        group.attrs["encoding-type"] = f"{layout}_matrix"
        group.attrs["encoding-version"] = "0.1.0"
        group.attrs["shape"] = np.asarray(values.shape)
        group.create_dataset("data", data=major[major != 0])
        group.create_dataset(
            "indices",
            data=np.concatenate([np.flatnonzero(line) for line in major]).astype(
                np.int32
            ),
        )
        group.create_dataset(
            "indptr",
            data=np.concatenate([[0], np.cumsum((major != 0).sum(axis=1))]).astype(
                np.int64
            ),
        )
    return path


def _entries(chunks) -> list[tuple[str, str, int]]:
    return sorted(
        (cell, barcode, int(count))
        for chunk in chunks
        for cell, barcode, count in zip(
            chunk.cells, chunk.barcodes, chunk.umi_counts, strict=True
        )
    )


def _nonzero_entries(values: np.ndarray) -> list[tuple[str, str, int]]:
    rows, columns = np.nonzero(values)
    return sorted(
        (f"C{row}", f"B{column}", int(values[row, column]))
        for row, column in zip(rows.tolist(), columns.tolist(), strict=True)
    )


@pytest.mark.parametrize(
    ("entries", "tile"),
    [
        # Fewer entries than a chunk holds: one chunk of 8 by 16 at a time.
        (10, (8, 16)),
        # 300 entries: two chunks side by side, 8 by 32, hold 256 of them.
        (300, (8, 32)),
        # 2,000 entries: all 70 columns, and 24 rows, three chunks of 8, hold
        # 1,680 of them.
        (2_000, (24, 70)),
    ],
)
def test_dense_anndata_is_read_in_tiles_of_whole_chunks(
    tmp_path, monkeypatch, entries, tile
) -> None:
    rng = np.random.default_rng(3)
    values = (rng.random((50, 70)) < 0.2) * rng.integers(1, 9, (50, 70))
    path = _matrix_h5ad(
        tmp_path / "dense.h5ad",
        values.astype(np.float32),
        layout="dense",
        chunks=(8, 16),
        compression="gzip",
    )
    reads = []
    read = h5py.Dataset.__getitem__

    def recorded(self, selection):
        if self.name == "/X":
            reads.append(selection)
        return read(self, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", recorded)
    chunks = _rows(barcode_layout(path), entries)
    assert _entries(chunks) == _nonzero_entries(values)
    assert max(len(chunk.cells) for chunk in chunks) <= entries
    height, width = tile
    covered = np.zeros(values.shape, dtype=np.int64)
    for rows, columns in reads:
        # Each read starts on a chunk's edge and spans whole chunks, so that
        # no chunk is decompressed twice.
        assert rows.start % 8 == 0 and columns.start % 16 == 0
        assert (
            rows.stop - rows.start <= height and columns.stop - columns.start <= width
        )
        covered[rows, columns] += 1
    assert (covered == 1).all()
    assert len(reads) == -(-50 // height) * -(-70 // width)


def test_a_dense_matrix_without_chunks_is_read_in_whole_rows_when_they_fit(
    tmp_path, monkeypatch
) -> None:
    reads = []
    read = h5py.Dataset.__getitem__

    def recorded(self, selection):
        if self.name == "/X":
            reads.append(tuple((part.start, part.stop) for part in selection))
        return read(self, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", recorded)
    values = np.zeros((30, 6), dtype=np.int64)
    values[np.arange(30), np.arange(30) % 6] = np.arange(1, 31)
    path = _matrix_h5ad(tmp_path / "dense.h5ad", values, layout="dense")
    # Four rows of six barcodes fit 25 entries; the last read holds two rows.
    chunks = _rows(barcode_layout(path), 25)
    assert reads == [((start, min(start + 4, 30)), (0, 6)) for start in range(0, 30, 4)]
    assert _entries(chunks) == _nonzero_entries(values)
    # Entries gather across reads: the six reads' 24 entries, then six more.
    assert [len(chunk.cells) for chunk in chunks] == [24, 6]
    # A row of more barcodes than the entries is read in parts.
    reads.clear()
    wide = np.ones((2, 9), dtype=np.int64)
    path = _matrix_h5ad(tmp_path / "wide.h5ad", wide, layout="dense")
    chunks = _rows(barcode_layout(path), 4)
    assert reads == [
        ((row, row + 1), columns)
        for row in (0, 1)
        for columns in ((0, 4), (4, 8), (8, 9))
    ]
    assert [len(chunk.cells) for chunk in chunks] == [4, 4, 1, 4, 4, 1]


@pytest.mark.parametrize("layout", ["csr", "csc"])
def test_sparse_anndata_reads_a_barcode_of_many_cells_in_parts(
    tmp_path, layout
) -> None:
    # One barcode in each of 50 cells beside barcodes of one cell each.
    values = np.zeros((50, 6), dtype=np.int64)
    values[:, 2] = np.arange(1, 51)
    values[np.arange(0, 50, 7), 4] = 3
    path = _matrix_h5ad(tmp_path / f"{layout}.h5ad", values, layout=layout)
    chunks = _rows(barcode_layout(path), 7)
    assert _entries(chunks) == _nonzero_entries(values)
    # Reading a barcode's cells at once held all 50 of them.
    assert max(len(chunk.cells) for chunk in chunks) == 7
    assert sum(len(chunk.cells) for chunk in chunks) == np.count_nonzero(values)


@pytest.mark.parametrize(
    ("dataset", "change", "message"),
    [
        ("indptr", lambda pointers: pointers[:-1], "pointers that do not fit"),
        (
            "indptr",
            lambda pointers: np.concatenate([[0, 3, 2], pointers[3:]]),
            "pointers that do not fit",
        ),
        ("indptr", lambda pointers: pointers + 1, "pointers that do not fit"),
        # Only the first pointer is wrong: its entry would go to the last cell.
        (
            "indptr",
            lambda pointers: np.concatenate([[1], pointers[1:]]),
            "pointers that do not fit",
        ),
        # Only the last pointer is wrong: it names an entry that is not stored.
        (
            "indptr",
            lambda pointers: np.concatenate([pointers[:-1], [pointers[-1] + 1]]),
            "pointers that do not fit",
        ),
        ("indices", lambda indices: indices + 6, "entries beyond its shape"),
        ("indices", lambda indices: indices - 7, "entries beyond its shape"),
    ],
)
def test_sparse_anndata_whose_pointers_or_entries_break_its_shape_is_refused(
    tmp_path, dataset, change, message
) -> None:
    values = np.eye(6, dtype=np.int64)
    path = _matrix_h5ad(tmp_path / "csr.h5ad", values, layout="csr")
    with h5py.File(path, "a") as handle:
        stored = handle[f"X/{dataset}"][:]
        del handle[f"X/{dataset}"]
        handle[f"X/{dataset}"] = change(stored)
    with pytest.raises(ValueError, match=message):
        _rows(barcode_layout(path), 10)


def test_quicats_wide_table_is_parsed_once_per_chunk(tmp_path, monkeypatch) -> None:
    rows = [f"s1,C{row}_s1," + ",".join(["0", "2", "0"]) for row in range(25)]
    path = _file(
        tmp_path, "barcodes_output.csv", "sample,cell,A,B,C\n" + "\n".join(rows) + "\n"
    )
    calls = []
    parse = np.loadtxt

    def counted(*arguments, **options):
        calls.append(1)
        return parse(*arguments, **options)

    monkeypatch.setattr(np, "loadtxt", counted)
    chunks = _rows(barcode_layout(path), 10)
    assert len(calls) == len(chunks) == 3
    assert [chunk.cells.tolist() for chunk in chunks][2] == [
        f"C{row}_s1" for row in range(20, 25)
    ]


def test_required_columns_are_found_in_time_linear_in_the_header(tmp_path) -> None:
    from cytearc.readers._delimited import TextLayout, require_columns

    class Name(str):
        comparisons = 0

        def __eq__(self, other):
            Name.comparisons += 1
            return str.__eq__(self, other)

        __hash__ = str.__hash__

    names = tuple(Name(f"B{index}") for index in range(2_000))
    text = TextLayout(
        path="wide.csv", names=names, delimiter=",", compression=None, row_names=False
    )
    require_columns(text, list(names))
    # Checking each name against the whole header compared 2,000 names about
    # 2,000 times each.
    assert Name.comparisons < 10 * len(names)
    with pytest.raises(ValueError, match="names the column 'B1' more than once"):
        require_columns(replace(text, names=(*names, Name("B1"))), ["B1", "B2"])
    with pytest.raises(ValueError, match="has no column 'B9999'"):
        require_columns(text, ["B9999"])


def _peak_while_reading(layout, chunk_rows: int) -> int:
    """The bytes held at the peak of reading a file chunk by chunk."""
    import tracemalloc

    tracemalloc.start()
    try:
        for _chunk in iter_barcode_rows(layout, chunk_rows):
            pass
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return int(peak)


@pytest.mark.parametrize(
    "kind", ["wide", "dense", "csr", "csc", "many_barcodes", "libraries"]
)
def test_reading_holds_no_more_than_the_layout_estimates(tmp_path, kind) -> None:
    rng = np.random.default_rng(5)
    values = (rng.random((300, 400)) < 0.05) * rng.integers(1, 9, (300, 400))
    columns = None
    libraries = None
    if kind == "many_barcodes":
        # One count per cell among 20,000 barcodes: their names outweigh
        # the counts.
        values = np.zeros((300, 20_000), dtype=np.int64)
        values[np.arange(300), rng.choice(20_000, 300, replace=False)] = 2
        kind = "csr"
    if kind == "libraries":
        # Each cell's library, read from an obs column for every cell.
        values = np.zeros((20_000, 3), dtype=np.int64)
        values[::50, 1] = 4
        libraries = [f"lane{index % 4}" for index in range(20_000)]
        columns = {"library": "library"}
        kind = "csr"
    if kind == "wide":
        lines = ["sample,cell," + ",".join(f"B{index}" for index in range(400))] + [
            f"s1,C{row}," + ",".join(map(str, values[row].tolist()))
            for row in range(300)
        ]
        path = _file(tmp_path, "barcodes_output.csv", "\n".join(lines) + "\n")
    else:
        path = _matrix_h5ad(
            tmp_path / f"{kind}.h5ad",
            values.astype(np.float32),
            layout=kind,
            chunks=(16, 64) if kind == "dense" else None,
            compression="gzip" if kind == "dense" else None,
            libraries=libraries,
        )
    layout = barcode_layout(
        path, format=None if columns is None else "h5ad", columns=columns
    )
    chunk_rows = layout.chunk_rows(400_000)
    assert _entries(_rows(layout, chunk_rows)) == _nonzero_entries(values)
    assert _peak_while_reading(layout, chunk_rows) <= layout.held_bytes() + (
        layout.chunk_bytes(chunk_rows)
    )


@pytest.mark.parametrize(
    ("name", "text", "message"),
    [
        (
            "annotation.tsv",
            "barcode\tbc.umi.count\nAAAC\tLB1;LB2\t4;99999999999999999999\n",
            "umis holds a number too large for int64, "
            "'99999999999999999999', on line 2",
        ),
        (
            "barcodes_output.csv",
            "sample,cell,A,B\ns1,C1,0,9223372036854775808\n",
            "holds '9223372036854775808' as the count of barcode 'B' on line 2",
        ),
    ],
)
def test_counts_beyond_int64_are_refused_naming_their_line(
    tmp_path, name, text, message
) -> None:
    # Cast to int64, they became -9223372036854775808.
    path = _file(tmp_path, name, text)
    with pytest.raises(ValueError, match=message):
        _rows(barcode_layout(path), 10)


@pytest.mark.parametrize("layout", ["dense", "csr"])
def test_anndata_counts_beyond_int64_are_refused(tmp_path, layout) -> None:
    values = np.asarray([[0.0, 2.0**63], [1.0, 0.0]])
    path = _matrix_h5ad(tmp_path / f"{layout}.h5ad", values, layout=layout)
    with pytest.raises(ValueError, match="whole numbers of at least 0 and below"):
        _rows(barcode_layout(path), 10)
    # The largest float64 below 2**63 is read exactly.
    values[0, 1] = np.nextafter(2.0**63, 0)
    path = _matrix_h5ad(tmp_path / f"{layout}_largest.h5ad", values, layout=layout)
    assert _entries(_rows(barcode_layout(path), 10)) == [
        ("C0", "B1", 2**63 - 1024),
        ("C1", "B0", 1),
    ]


def test_text_that_cannot_be_read_is_refused_naming_the_file(tmp_path) -> None:
    # A gzip header followed by bytes that do not inflate.
    corrupt = tmp_path / "counts.tsv.gz"
    corrupt.write_bytes(gzip.compress(b"gene\tcell\tcount\n")[:10] + b"\xff" * 64)
    with pytest.raises(ValueError, match="counts.tsv.gz is cut short or corrupt"):
        barcode_layout(corrupt)
    # A merged table whose line beyond the header's block of text is not UTF-8.
    path = tmp_path / "barcodes_output.csv"
    rows = b"".join(b"s1,C%d,2\n" % row for row in range(5_000))
    path.write_bytes(b"sample,cell,A\n" + rows + b"s1,C\xff2,1\n")
    with pytest.raises(ValueError, match="is not a text file that CyteArc can read"):
        _rows(barcode_layout(path), 10)


def test_a_merged_table_that_numpy_cannot_parse_is_refused_as_malformed(
    tmp_path, monkeypatch
) -> None:
    path = _file(tmp_path, "barcodes_output.csv", "sample,cell,A,B\ns1,C1,0,2\n")
    parse = np.loadtxt

    def refused(*arguments, **options):
        parse(*arguments, **options)
        raise ValueError("unbalanced quotes")

    monkeypatch.setattr(np, "loadtxt", refused)
    # Fields that are a cell and counts, which numpy refuses anyway.
    with pytest.raises(ValueError, match="is malformed: unbalanced quotes"):
        _rows(barcode_layout(path), 10)


def test_a_sparse_dense_matrix_yields_its_entries_in_few_chunks(tmp_path) -> None:
    values = np.zeros((50, 70), dtype=np.int64)
    values[np.arange(50), (7 * np.arange(50)) % 70] = 3
    path = _matrix_h5ad(tmp_path / "dense.h5ad", values, layout="dense", chunks=(8, 16))
    chunks = _rows(barcode_layout(path), 10)
    assert _entries(chunks) == _nonzero_entries(values)
    assert max(len(chunk.cells) for chunk in chunks) <= 10
    # A chunk per part of a tile held one or two entries each: 50 chunks.
    assert len(chunks) <= 2 * 50 // 10


def test_zeros_stored_in_a_sparse_matrix_name_no_barcode(tmp_path) -> None:
    path = _matrix_h5ad(tmp_path / "csr.h5ad", np.eye(3, dtype=np.int64), layout="csr")
    # The first three stored entries are zeros, so the first window holds none.
    with h5py.File(path, "a") as handle:
        for name, values in (
            ("data", [0, 0, 0, 2, 5]),
            ("indices", [0, 1, 2, 1, 2]),
            ("indptr", [0, 3, 4, 5]),
        ):
            del handle[f"X/{name}"]
            handle[f"X/{name}"] = np.asarray(values)
    chunks = _rows(barcode_layout(path), 3)
    assert _entries(chunks) == [("C1", "B1", 2), ("C2", "B2", 5)]
    assert len(chunks) == 1


def _named_h5ad(path: Path, *, n_cells: int, name_length: int, libraries: str) -> Path:
    """A CSR AnnData file of one count per cell, whose cell names are long.

    ``libraries`` is ``"none"``, ``"categorical"`` or ``"strings"``: the
    kind of ``obs`` column ``library`` that labels each cell's lane.
    """
    from tests.clone_truth import write_categorical

    names = [f"C{index:0{name_length - 1}d}" for index in range(n_cells)]
    lanes = [f"lane{index % 4}" for index in range(n_cells)]
    with h5py.File(path, "w") as handle:
        obs = handle.create_group("obs")
        obs.attrs["_index"] = "_index"
        obs.attrs["encoding-type"] = "dataframe"
        obs.attrs["encoding-version"] = "0.2.0"
        obs.attrs["column-order"] = np.asarray([], dtype=h5py.string_dtype())
        obs.create_dataset("_index", data=np.asarray(names, dtype=h5py.string_dtype()))
        if libraries == "categorical":
            write_categorical(obs, "library", lanes)
        elif libraries == "strings":
            column = obs.create_dataset(
                "library", data=np.asarray(lanes, dtype=h5py.string_dtype())
            )
            column.attrs["encoding-type"] = "string-array"
            column.attrs["encoding-version"] = "0.2.0"
        var = handle.create_group("var")
        var.attrs["_index"] = "_index"
        var.create_dataset(
            "_index", data=np.asarray(["B0", "B1", "B2"], dtype=h5py.string_dtype())
        )
        group = handle.create_group("X")
        group.attrs["encoding-type"] = "csr_matrix"
        group.attrs["encoding-version"] = "0.1.0"
        group.attrs["shape"] = np.asarray([n_cells, 3])
        group.create_dataset("data", data=np.full(n_cells, 2.0, dtype=np.float32))
        group.create_dataset("indices", data=(np.arange(n_cells) % 3).astype(np.int32))
        group.create_dataset("indptr", data=np.arange(n_cells + 1, dtype=np.int64))
    return path


@pytest.mark.slow
@pytest.mark.parametrize(
    ("name_length", "libraries"),
    [(18, "none"), (34, "none"), (100, "none"), (34, "categorical"), (34, "strings")],
)
def test_the_names_of_many_cells_are_held_within_the_estimate(
    tmp_path, monkeypatch, name_length, libraries
) -> None:
    from cytearc.readers import lineage_barcodes as reader

    # Blocks of 2,048 names leave little room beside the names, so decoding
    # them at once shows.
    monkeypatch.setattr(reader, "_NAME_BLOCK", 2_048)
    path = _named_h5ad(
        tmp_path / "barcodes_output.h5ad",
        n_cells=100_000,
        name_length=name_length,
        libraries=libraries,
    )
    columns = None if libraries == "none" else {"library": "library"}
    layout = barcode_layout(path, format="h5ad", columns=columns)
    chunk_rows = layout.chunk_rows(400_000)
    # Decoded at once and counted at 100 bytes a name, 34 characters held
    # about 166 bytes a cell at their peak.
    assert _peak_while_reading(layout, chunk_rows) <= layout.held_bytes() + (
        layout.chunk_bytes(chunk_rows)
    )


def test_a_fatemap_table_without_a_header_names_its_short_row(tmp_path) -> None:
    # The first row of a table without a header is no header of 8 columns.
    path = _file(
        tmp_path,
        "stepThreeStarcodeShavedReads.txt",
        "AAAC\tGGTA\tTTTA\tTTTT\tTTT\tTT\tTA\t1\nCCCG\tGCTA\tTTTA\n",
    )
    layout = barcode_layout(path)
    assert (layout.format, layout.level) == ("fatemap", "molecules")
    with pytest.raises(ValueError, match="without BC50StarcodeD8 on line 2"):
        _rows(layout, 10)


def test_a_gzip_file_of_no_text_is_refused_as_empty(tmp_path) -> None:
    # Its bytes are a gzip header, but it holds no line to recognise.
    path = tmp_path / "molecules.txt.gz"
    path.write_bytes(gzip.compress(b""))
    with pytest.raises(ValueError, match="is empty; a barcode table needs a header"):
        barcode_layout(path)
