"""The clone file reader: formats, layout quirks and the errors of broken files."""

import gzip
from pathlib import Path

import h5py
import numpy as np
import pytest

from cytearc.readers.clone_tables import (
    CELL_ID_SEPARATOR,
    CLONE_FORMATS,
    clone_layout,
    iter_clone_rows,
    matrix_cells,
    matrix_clone_ids,
    matrix_entries,
)
from tests.clone_truth import (
    clone_truth,
    write_categorical,
    write_cospar_h5ad,
    write_mito_h5ad,
)

DATA = Path(__file__).parent / "data" / "clones"
TRUTH = clone_truth()


def _rows(path: Path, chunk_rows: int = 2, **options) -> tuple[list, list, list]:
    layout = clone_layout(path, **options)
    barcodes, clones, probabilities = [], [], []
    for chunk in iter_clone_rows(layout, chunk_rows):
        barcodes += chunk.barcodes.tolist()
        clones += chunk.clones.tolist()
        if chunk.probabilities is not None:
            probabilities += chunk.probabilities.tolist()
    return barcodes, clones, probabilities


def _write(path: Path, text: str, *, compress: bool = False) -> Path:
    data = text.encode("utf-8")
    path.write_bytes(gzip.compress(data, mtime=0) if compress else data)
    return path


def test_the_formats_are_the_documented_ones() -> None:
    assert CLONE_FORMATS == (
        "cospar",
        "trex",
        "celltag_multi",
        "celltagr",
        "sticr",
        "cassiopeia",
        "numbat",
        "table",
        "matrix",
        "cospar_h5ad",
        "mito",
        "h5ad",
    )
    assert CELL_ID_SEPARATOR == "__"


@pytest.mark.parametrize(
    ("text", "format", "cells", "clones"),
    [
        # TREX before 2023 started its CSV header with "#".
        ("#clone_id,cell_id\n1,AAA\n2,CCC\n", "trex", ["AAA", "CCC"], ["1", "2"]),
        ("clone#\tcell_id\n1\tAAA\n", "trex", ["AAA"], ["1"]),
        ("clone_nr\tcell_id\n1\tAAA\n2\tCCC", "trex", ["AAA", "CCC"], ["1", "2"]),
        (
            "Cell_ID,Clone_ID\r\nAAA,7\r\nCCC,7\r\n",
            "cospar",
            ["AAA", "CCC"],
            ["7", "7"],
        ),
        ('"clone.id","cell.barcode"\n3,"AAA-1"\n', "celltagr", ["AAA-1"], ["3"]),
        (
            ",clone_id,cell.bc\n10,r1_1,AAA\n11,r2_1,CCC\n",
            "celltag_multi",
            ["AAA", "CCC"],
            ["r1_1", "r2_1"],
        ),
        ("clone.id,cell.bc,edge.den\n4,AAA,1.0\n", "celltag_multi", ["AAA"], ["4"]),
        (
            "cell_Id\tVBC\tn\nAAA\t \tx\nCCC\tGW_1\t  3\n",
            "sticr",
            ["AAA", "CCC"],
            [None, "GW_1"],
        ),
        (
            "cellBC\tlineageGrp\nAAA\t0\nCCC\t4.0\nGGG\t0.0\nTTT\tNA\nCGA\t4\n",
            "cassiopeia",
            ["AAA", "CCC", "GGG", "TTT", "CGA"],
            [None, "4", None, None, "4"],
        ),
        (
            "﻿Cell_ID,Clone_ID\nAAA,NaN\nCCC,nan\nGGG,\n",
            "cospar",
            ["AAA", "CCC", "GGG"],
            [None, None, None],
        ),
    ],
)
def test_each_tools_header_names_its_format(
    tmp_path, text, format, cells, clones
) -> None:
    path = _write(tmp_path / "clones.txt", text)

    layout = clone_layout(path)
    barcodes, labels, _ = _rows(path)

    assert layout.format == format
    assert barcodes == cells
    assert labels == clones


def test_gzip_is_read_from_the_contents_not_the_name(tmp_path) -> None:
    path = _write(tmp_path / "clones.txt", "Cell_ID,Clone_ID\nAAA,1\n", compress=True)
    assert _rows(path)[:2] == (["AAA"], ["1"])


def test_r_row_names_hold_the_cells_of_a_wide_table(tmp_path) -> None:
    # The Biddy et al. workflow's layout: space-separated, quoted, NA, one
    # clone column per round and a header one field shorter than the rows.
    path = _write(
        tmp_path / "celltag_data.txt",
        '"CellTagD0" "CellTagD3"\n"AAA-1" 16 NA\n"CCC-1" NA 2\n"GGG-1" 16 2',
    )

    layout = clone_layout(path, columns={"clone": "CellTagD3"})

    assert layout.format == "table"
    assert layout.cell == ()
    assert _rows(path, columns={"clone": "CellTagD3"})[:2] == (
        ["AAA-1", "CCC-1", "GGG-1"],
        [None, "2", "2"],
    )
    assert _rows(path, columns={"clone": "CellTagD0"})[1] == ["16", None, "16"]


def test_r_write_csv_row_names_hold_the_cells(tmp_path) -> None:
    # R's write.csv writes its row names under an empty first name.
    path = _write(tmp_path / "clones.csv", '"","clone"\n"AAA-1","3"\n"CCC-1",NA\n')

    layout = clone_layout(path, columns={"clone": "clone"})

    assert layout.cell == ()
    assert _rows(path, columns={"clone": "clone"})[:2] == (
        ["AAA-1", "CCC-1"],
        ["3", None],
    )


@pytest.mark.parametrize("quote", ["", '"'])
def test_r_row_names_whose_first_row_ends_in_an_empty_value(tmp_path, quote) -> None:
    # R's write.table leaves the row names out of its header. Its first row
    # here holds no clone, written as quoted empty text by default and as
    # nothing with quote=FALSE, which a delimiter after each row's last field
    # would give too; the later rows' clones show the row names.
    q = quote
    path = _write(
        tmp_path / "clones.txt",
        f"{q}sample{q}\t{q}clone{q}\n{q}AAA-1{q}\t{q}s1{q}\t{q}{q}\n"
        f"{q}CCC-1{q}\t{q}s1{q}\t{q}c2{q}\n{q}GGG-1{q}\t{q}s1{q}\t{q}c2{q}\n",
    )

    assert clone_layout(path, columns={"clone": "clone"}).cell == ()
    assert _rows(path, columns={"clone": "clone"})[:2] == (
        ["AAA-1", "CCC-1", "GGG-1"],
        [None, "c2", "c2"],
    )


def test_rows_past_the_first_ones_can_show_r_row_names(tmp_path) -> None:
    # The first 150 cells hold no clone; the 151st shows that the field
    # after the named columns holds values.
    rows = [f"cell{index}\ts1\t" for index in range(150)] + ["cell150\ts1\tc1"]
    path = _write(tmp_path / "clones.txt", "sample\tclone\n" + "\n".join(rows) + "\n")

    barcodes, clones, _ = _rows(path, chunk_rows=40, columns={"clone": "clone"})

    assert barcodes == [f"cell{index}" for index in range(151)]
    assert clones == [None] * 150 + ["c1"]


def test_space_separated_r_row_names_take_rows_that_end_in_spaces(tmp_path) -> None:
    # Spaces at the end of a space-separated row end no field, so the row
    # still holds a row name beside the header's names.
    path = _write(
        tmp_path / "celltag_data.txt",
        '"CellTagD0" "CellTagD3"\n"AAA-1" 16 NA \n"CCC-1" NA 2  \n',
    )

    assert _rows(path, columns={"clone": "CellTagD3"})[:2] == (
        ["AAA-1", "CCC-1"],
        [None, "2"],
    )


def test_spaces_after_the_last_delimiter_end_no_value(tmp_path) -> None:
    # A row whose field after the named columns holds only spaces ends in its
    # delimiter, so the table is refused as ambiguous rather than read as R's
    # row names, which would shift every column.
    path = _write(tmp_path / "clones.csv", "cell,clone\nAAAC,1, \nCCCG,2,\t\n")
    with pytest.raises(ValueError, match="all end in a delimiter with nothing after"):
        clone_layout(path)


def test_a_quote_that_never_closes_is_named_by_its_line(tmp_path) -> None:
    # The quote on line 6 swallows the 20,000 rows after it, which each line's
    # quotes, counted once, make quick to find.
    rows = [f"cell{index},{index}" for index in range(20_000)]
    rows[4] = 'cell4,"4'
    path = _write(tmp_path / "clones.csv", "Cell_ID,Clone_ID\n" + "\n".join(rows))
    with pytest.raises(ValueError, match="the quote opened on line 6 is never closed"):
        clone_layout(path)
    # Past the sampled rows, the search for R's row names meets it too.
    ended = [f"cell{index},{index}," for index in range(150)] + ['cell150,"150,']
    late = _write(tmp_path / "late.csv", "cell,clone\n" + "\n".join(ended) + "\n")
    with pytest.raises(
        ValueError, match="the quote opened on line 152 is never closed"
    ):
        clone_layout(late)


@pytest.mark.parametrize("delimiter", [",", "\t"])
def test_rows_that_each_end_in_an_empty_field_need_a_header_that_says_why(
    tmp_path, delimiter
) -> None:
    # One field more than the header, empty in every row, is R's row names
    # with an empty last column, or a delimiter after each row's last field:
    # the reader cannot tell which, so it refuses rather than shift every
    # column.
    d = delimiter
    path = _write(
        tmp_path / "clones.txt", f"clone_nr{d}cell_id\n1{d}AAA{d}\n2{d}CCC{d}\n"
    )
    with pytest.raises(
        ValueError,
        match=r"names 2 columns, and its rows of 3 fields all end in a delimiter "
        r"with nothing after it: R's row names .* or a delimiter after each row's "
        r"last field. Name "
        r"the row names in the header, as R's write.table\(\.\.\., col.names=NA\) "
        r"writes them, or end the header with the delimiter too",
    ):
        clone_layout(path)
    # A header that ends in the delimiter too names the empty last field.
    ended = _write(
        tmp_path / "ended.txt", f"clone_nr{d}cell_id{d}\n1{d}AAA{d}\n2{d}CCC{d}\n"
    )
    assert clone_layout(ended).cell == ("cell_id",)
    assert _rows(ended)[:2] == (["AAA", "CCC"], ["1", "2"])
    # A header that names the row names under an empty name, as
    # write.table(..., col.names=NA) writes them, makes them the cells.
    named = _write(
        tmp_path / "named.txt",
        f"{d}clone_nr{d}cell_id\nAAA{d}1{d}\nCCC{d}2{d}\n",
    )
    assert _rows(named, columns={"clone": "clone_nr"})[:2] == (
        ["AAA", "CCC"],
        ["1", "2"],
    )


def test_blank_lines_hold_no_rows(tmp_path) -> None:
    path = _write(tmp_path / "clones.csv", "Cell_ID,Clone_ID\n\nAAA,1\n\n\nCCC,2\n\n")
    assert clone_layout(path).cell == ("Cell_ID",)
    assert _rows(path)[:2] == (["AAA", "CCC"], ["1", "2"])
    header_only = _write(tmp_path / "header.csv", "Cell_ID,Clone_ID\n\n")
    assert _rows(header_only) == ([], [], [])


@pytest.mark.parametrize("chunk_rows", [1, 2, 10])
def test_errors_name_the_file_line_of_a_row_after_blank_lines(
    tmp_path, chunk_rows
) -> None:
    path = _write(tmp_path / "clones.csv", "Cell_ID,Clone_ID\nA,1\n\n\nB,2\n,3\n")
    with pytest.raises(ValueError, match="without a cell barcode on lines 6$"):
        _rows(path, chunk_rows=chunk_rows)
    # R's space-separated layout, where a row's name may be quoted empty text.
    spaced = _write(
        tmp_path / "celltag_data.txt",
        '"CellTagD0" "CellTagD3"\n"AAA-1" 16 NA\n\n"" NA 2\n',
    )
    with pytest.raises(ValueError, match="without a cell barcode on lines 4$"):
        _rows(spaced, chunk_rows=chunk_rows, columns={"clone": "CellTagD3"})
    # A quoted field that holds a line break spans two lines of the file.
    quoted = _write(
        tmp_path / "clone_post_2.tsv",
        'cell\tclone_opt\tp_opt\nA\t"1\n2"\t0.5\n\nB\t1\t3\n',
    )
    with pytest.raises(ValueError, match="or NA, on lines 5$"):
        _rows(quoted, chunk_rows=chunk_rows)


def test_a_gzip_file_cut_short_is_refused_by_name(tmp_path) -> None:
    rows = "".join(f"C{index},{index}\n" for index in range(5_000))
    data = gzip.compress(("Cell_ID,Clone_ID\n" + rows).encode(), mtime=0)
    table = tmp_path / "clones.csv.gz"
    table.write_bytes(data[: len(data) // 2])
    with pytest.raises(ValueError, match="clones.csv.gz is cut short or corrupt"):
        _rows(table, chunk_rows=100)
    # The rows of a matrix's cell list are counted before they are read.
    folder = tmp_path / "matrix"
    folder.mkdir()
    cells = gzip.compress(
        ("cell\n" + "".join(f"C{index}\n" for index in range(5_000))).encode(), mtime=0
    )
    (folder / "cells.tsv.gz").write_bytes(cells[: len(cells) // 2])
    _write(
        folder / "m.mtx",
        "%%MatrixMarket matrix coordinate pattern general\n5000 1 0\n",
    )
    with pytest.raises(ValueError, match="cells.tsv.gz is cut short or corrupt"):
        clone_layout(folder / "m.mtx")


def test_columns_join_several_cell_columns_and_read_probabilities(tmp_path) -> None:
    path = _write(
        tmp_path / "metadata.tsv",
        "Library\tCell barcode\tgroup\tp\nd2\tAC-GT\tx\t0.25\nd4\tAC-GT\tx\tNA\n",
    )
    columns = {
        "cell": ["Library", "Cell barcode"],
        "clone": "group",
        "probability": "p",
    }

    barcodes, clones, probabilities = _rows(path, columns=columns)

    assert barcodes == ["d2__AC-GT", "d4__AC-GT"]
    assert clones == ["x", "x"]
    assert probabilities[0] == 0.25
    assert np.isnan(probabilities[1])
    assert clone_layout(path, columns=columns).record() == {
        "format": "table",
        "cell": ["Library", "Cell barcode"],
        "clone": "group",
        "probability": "p",
    }


def test_numbat_reads_the_posterior_of_the_chosen_clone(tmp_path) -> None:
    path = _write(
        tmp_path / "clone_post_2.tsv",
        'cell\tclone_opt\tGT_opt\tp_opt\nA-1\t1\t""\t0.999\nB-1\t2\t11b,5b\t1\n',
    )
    barcodes, clones, probabilities = _rows(path)
    assert clone_layout(path).format == "numbat"
    assert (barcodes, clones, probabilities) == (
        ["A-1", "B-1"],
        ["1", "2"],
        [0.999, 1.0],
    )


@pytest.mark.parametrize("value", ["1.5", "-0.1", "high"])
def test_probabilities_outside_zero_to_one_name_their_lines(tmp_path, value) -> None:
    path = _write(
        tmp_path / "clone_post_2.tsv",
        f"cell\tclone_opt\tp_opt\nA\t1\t0.5\nB\t1\t{value}\n",
    )
    with pytest.raises(ValueError, match="numbers from 0 to 1, or NA, on lines 3"):
        _rows(path)


def test_rows_without_a_barcode_name_their_lines(tmp_path) -> None:
    path = _write(tmp_path / "clones.csv", "Cell_ID,Clone_ID\nA,1\n,2\nB,3\n ,4\n")
    with pytest.raises(ValueError, match="without a cell barcode on lines 3, 5"):
        _rows(path, chunk_rows=10)


def test_a_header_alone_holds_no_rows(tmp_path) -> None:
    path = _write(tmp_path / "clones.txt", "clone_nr\tcell_id\n")
    assert _rows(path) == ([], [], [])


@pytest.mark.parametrize(
    ("text", "options", "message"),
    [
        ("", {}, "is empty"),
        ("\n\n", {}, "is empty; a clone table needs a header line"),
        ("a,b\n1,2,3,4\n", {}, "names 2 columns, but its first row holds 4"),
        ("barcode\tclone\nA\t1\n", {}, "match no clone table that CyteArc knows"),
        (
            "Cell_ID,Clone_ID,cell_id,clone_nr\nA,1,A,1\n",
            {},
            "columns of several clone tables: cospar, trex",
        ),
        (
            "Cell_ID,Clone_ID\nA,1\n",
            {"format": "trex"},
            "has no column 'cell_id', 'clone_nr'",
        ),
        (
            "Cell_ID,Clone_ID\nA,1\n",
            {"format": "cospar", "columns": {"clone": "x"}},
            "reads fixed columns",
        ),
        ("Cell_ID,Clone_ID\nA,1\n", {"format": "table"}, "needs the clone column"),
        (
            "Cell_ID,Clone_ID,Clone_ID\nA,1,2\n",
            {},
            "names the column 'Clone_ID' more than once",
        ),
        (
            "Cell_ID,Clone_ID\nA,1\n",
            {"format": "mito"},
            "is a text table, not format='mito'",
        ),
        (
            "Cell_ID,Clone_ID\nA,1\n",
            {"columns": {"cell": "Cell_ID", "clone": "Clone_ID", "probability": "p"}},
            "has no column 'p'",
        ),
        (
            "Cell_ID,Clone_ID\nA,1\n",
            {"columns": {"cell": "", "clone": "Clone_ID"}},
            "names an empty column",
        ),
        ("Cell_ID,Clone_ID\nA,1\nB,2,3\n", {}, "is malformed"),
        # An unclosed quote that Python's csv module reads as two fields.
        ('Cell_ID,Clone_ID\nA,"1\nB,2\n', {}, "is malformed"),
    ],
)
def test_broken_tables_raise_errors_that_say_what_is_wrong(
    tmp_path, text, options, message
) -> None:
    path = _write(tmp_path / "clones.csv", text)
    with pytest.raises(ValueError, match=message):
        _rows(path, **options)


def test_arguments_of_the_wrong_type_are_refused(tmp_path) -> None:
    path = _write(tmp_path / "clones.csv", "Cell_ID,Clone_ID\nA,1\n")
    with pytest.raises(TypeError, match="path must be a path"):
        clone_layout(3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="format must be one of"):
        clone_layout(path, format="larry")
    with pytest.raises(TypeError, match="columns must map"):
        clone_layout(path, columns=["Cell_ID"])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"columns\['clone'\] must be a column name"):
        clone_layout(path, columns={"clone": ["Clone_ID"]})
    with pytest.raises(TypeError, match=r"columns\['cell'\] must be a column name or"):
        clone_layout(path, columns={"cell": [], "clone": "Clone_ID"})


def test_text_that_is_not_utf8_is_refused(tmp_path) -> None:
    path = tmp_path / "clones.bin"
    path.write_bytes(b"\xff\xfe\x00binary\x00\n\x80\x81\n")
    with pytest.raises(ValueError, match="not a text file"):
        clone_layout(path)


def test_quoted_barcodes_that_hold_delimiters_stay_one_field(tmp_path) -> None:
    folder = tmp_path / "matrix"
    folder.mkdir()
    _write(folder / "cells.tsv", '"cell"\n"a, b"\n"c d"\n"e\tf"\n')
    _write(
        folder / "clones.mtx",
        "%%MatrixMarket matrix coordinate pattern general\n3 1 3\n1 1\n2 1\n3 1\n",
    )

    layout = clone_layout(folder / "clones.mtx")
    cells = np.concatenate(list(matrix_cells(layout, 2))).tolist()

    assert cells == ["a, b", "c d", "e\tf"]
    assert matrix_clone_ids(layout).tolist() == ["0"]


def _larry(folder: Path, matrix: str, metadata: str) -> Path:
    _write(folder / "x_metadata.txt.gz", metadata, compress=True)
    return _write(folder / "x_clone_matrix.mtx.gz", matrix, compress=True)


_METADATA = "Library\tCell barcode\nd\tA\nd\tB\nd\tC"


def test_the_matrix_axis_whose_length_matches_the_cell_list_holds_the_cells(
    tmp_path,
) -> None:
    cells_by_clones = _larry(
        tmp_path,
        "%%MatrixMarket matrix coordinate integer general\n%\n3 2 2\n1 2 1\n3 1 1\n",
        _METADATA,
    )
    layout = clone_layout(
        cells_by_clones, columns={"cell": ["Library", "Cell barcode"]}
    )
    entries = list(matrix_entries(layout, 1))

    assert layout.related_paths == (str(tmp_path / "x_metadata.txt.gz"),)
    assert np.concatenate(list(matrix_cells(layout, 2))).tolist() == [
        "d__A",
        "d__B",
        "d__C",
    ]
    assert np.concatenate([e.cells for e in entries]).tolist() == [0, 2]
    assert np.concatenate([e.clones for e in entries]).tolist() == [1, 0]

    clones_by_cells = _larry(
        tmp_path,
        "%%MatrixMarket matrix coordinate real general\n2 3 3\n2 1 1.0\n1 3 2.5\n1 2 0\n",
        _METADATA,
    )
    layout = clone_layout(
        clones_by_cells, columns={"cell": ["Library", "Cell barcode"]}
    )
    entries = list(matrix_entries(layout, 10))
    # The stored zero is no membership.
    assert np.concatenate([e.cells for e in entries]).tolist() == [0, 2]
    assert np.concatenate([e.clones for e in entries]).tolist() == [1, 0]
    assert matrix_clone_ids(layout).tolist() == ["0", "1"]


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (
            "%%MatrixMarket matrix coordinate integer general\n4 5 0\n",
            "is 4 by 5, but its cell list",
        ),
        (
            "%%MatrixMarket matrix array integer general\n3 2\n",
            "a clone matrix must be a coordinate matrix",
        ),
        (
            "%%MatrixMarket matrix coordinate integer symmetric\n3 2 0\n",
            "holds integer symmetric values",
        ),
        ("%%MatrixMarket matrix coordinate integer general\n3 2\n", "size line of"),
        ("%%MatrixMarket matrix coordinate integer general\n%\n", "has no size line"),
        (
            "%%MatrixMarket matrix coordinate integer general\n3 2 1\n4 1 1\n",
            "entries outside its 3 by 2 shape",
        ),
        (
            "%%MatrixMarket matrix coordinate integer general\n3 2 1\n1 0 1\n",
            "entries outside its 3 by 2 shape",
        ),
        # Matrix Market counts from 1, so a row of 0 would wrap to the last cell.
        (
            "%%MatrixMarket matrix coordinate integer general\n3 2 1\n0 1 1\n",
            "entries outside its 3 by 2 shape",
        ),
        (
            "%%MatrixMarket matrix coordinate integer general\n3 2 1\n1.5 1 1\n",
            "entries outside its 3 by 2 shape",
        ),
        (
            "%%MatrixMarket matrix coordinate real general\n3 2 1\n1 1 inf\n",
            "not finite",
        ),
        (
            "%%MatrixMarket matrix coordinate integer general\n3 2 2\n1 1 1\n",
            "declares 2 entries but holds 1",
        ),
        (
            "%%MatrixMarket matrix coordinate integer general\n3 2 1\n1 x 1\n",
            "is malformed",
        ),
        ("%%MatrixMarket\n3 2 1\n", "is not a Matrix Market file"),
    ],
)
def test_broken_clone_matrices_are_refused(tmp_path, matrix, message) -> None:
    path = _larry(tmp_path, matrix, _METADATA)
    with pytest.raises(ValueError, match=message):
        layout = clone_layout(path, columns={"cell": ["Library", "Cell barcode"]})
        list(matrix_entries(layout, 10))


def test_a_matrix_that_fits_both_ways_holds_its_cells_in_rows(tmp_path) -> None:
    _write(tmp_path / "cells.tsv", "cell\nA\nB\n")
    _write(tmp_path / "clones.tsv", "clone\nx\ny\n")
    path = _write(
        tmp_path / "m.mtx",
        "%%MatrixMarket matrix coordinate integer general\n2 2 2\n1 2 1\n2 1 1\n",
    )

    layout = clone_layout(path)
    entries = list(matrix_entries(layout, 5))

    assert layout.clones_path == str(tmp_path / "clones.tsv")
    assert matrix_clone_ids(layout).tolist() == ["x", "y"]
    assert np.concatenate([e.cells for e in entries]).tolist() == [0, 1]
    assert np.concatenate([e.clones for e in entries]).tolist() == [1, 0]
    _write(tmp_path / "clones.tsv", "clone\nx\nx\n")
    with pytest.raises(ValueError, match="must give each clone its own label"):
        matrix_clone_ids(clone_layout(path))
    _write(tmp_path / "clones.tsv", "clone\nx\ny\nz\n")
    with pytest.raises(ValueError, match="holds 2 cells and its clone list 3 clones"):
        clone_layout(path)


def test_a_matrix_needs_one_cell_list_beside_it(tmp_path) -> None:
    matrix = "%%MatrixMarket matrix coordinate integer general\n1 1 0\n"
    alone = tmp_path / "alone"
    alone.mkdir()
    path = _write(alone / "clones.mtx", matrix)
    with pytest.raises(FileNotFoundError, match="needs its cell list beside it"):
        clone_layout(path)
    both = tmp_path / "both"
    both.mkdir()
    _write(both / "cells.tsv", "cell\nA\n")
    path = _larry(both, matrix, "Library\tCell barcode\nd\tA")
    with pytest.raises(ValueError, match="several cell lists beside it"):
        clone_layout(path)
    lists = tmp_path / "lists"
    lists.mkdir()
    _write(lists / "cells.tsv", "cell\nA\n")
    _write(lists / "clones.tsv", "clone\nx\n")
    _write(lists / "clones.tsv.gz", "clone\nx\n", compress=True)
    with pytest.raises(ValueError, match="two clone lists beside it"):
        clone_layout(_write(lists / "m.mtx", matrix))


def test_a_cell_list_of_several_columns_needs_the_cell_columns(tmp_path) -> None:
    path = _larry(
        tmp_path, "%%MatrixMarket matrix coordinate integer general\n3 1 0\n", _METADATA
    )
    with pytest.raises(ValueError, match="name the columns of each cell's barcode"):
        clone_layout(path)
    with pytest.raises(ValueError, match="takes only columns=\\{'cell'"):
        clone_layout(path, columns={"clone": "Library"})
    with pytest.raises(ValueError, match="has no column 'Barcode'"):
        clone_layout(path, columns={"cell": ["Library", "Barcode"]})
    with pytest.raises(ValueError, match="is a Matrix Market file, not format='table'"):
        clone_layout(path, format="table")


def test_cell_lists_name_cells_by_a_cell_column_or_row_names(tmp_path) -> None:
    matrix = "%%MatrixMarket matrix coordinate pattern general\n2 1 1\n2 1\n"
    named = tmp_path / "named"
    named.mkdir()
    _write(named / "cells.tsv", "cell\tsample\nA\ts1\nB\ts2\n")
    layout = clone_layout(_write(named / "m.mtx", matrix))
    assert layout.cell == ("cell",)
    rows = tmp_path / "rows"
    rows.mkdir()
    _write(rows / "cells.tsv", "sample\nA\ts1\nB\ts2\n")
    layout = clone_layout(_write(rows / "m.mtx", matrix))
    assert layout.cell == ()
    assert np.concatenate(list(matrix_cells(layout, 5))).tolist() == ["A", "B"]
    _write(rows / "cells.tsv", "sample\nA\ts1\n\ts2\n")
    with pytest.raises(ValueError, match="cells without a barcode on lines 3"):
        list(matrix_cells(clone_layout(rows / "m.mtx"), 5))


def test_readers_of_one_kind_refuse_layouts_of_another(tmp_path) -> None:
    table = clone_layout(_write(tmp_path / "t.csv", "Cell_ID,Clone_ID\nA,1\n"))
    with pytest.raises(ValueError, match="is not a matrix"):
        list(matrix_cells(table, 5))
    with pytest.raises(ValueError, match="is not a matrix"):
        matrix_clone_ids(table)
    with pytest.raises(ValueError, match="is not a matrix"):
        list(matrix_entries(table, 5))
    h5ad = clone_layout(write_cospar_h5ad(TRUTH, tmp_path / "a.h5ad"))
    with pytest.raises(ValueError, match="is a matrix; read it by entries"):
        list(iter_clone_rows(h5ad, 5))


@pytest.mark.parametrize("layout", ["csr", "csc", "dense"])
def test_cospars_clone_matrix_reads_in_any_storage(tmp_path, layout) -> None:
    path = write_cospar_h5ad(TRUTH, tmp_path / "a.h5ad", layout=layout)

    read = clone_layout(path)
    cells = np.concatenate(list(matrix_cells(read, 7)))
    entries = list(matrix_entries(read, 7))
    rows = np.concatenate([e.cells for e in entries])
    clones = np.concatenate([e.clones for e in entries])

    assert read.format == "cospar_h5ad"
    assert cells[: len(TRUTH.clones)].tolist() == TRUTH.cell_ids.tolist()
    order = np.argsort(rows, kind="stable")
    in_clone = np.flatnonzero(TRUTH.clones >= 0)
    assert rows[order][: len(in_clone)].tolist() == in_clone.tolist()
    assert clones[order][: len(in_clone)].tolist() == TRUTH.clones[in_clone].tolist()
    assert matrix_clone_ids(read).tolist() == [
        str(clone) for clone in range(TRUTH.n_clones)
    ]


def test_a_dense_clone_matrix_is_read_in_blocks_of_bounded_size(tmp_path) -> None:
    import tracemalloc

    n_cells, n_clones = 2_000, 2_000
    path = tmp_path / "dense.h5ad"
    with h5py.File(path, "w") as handle:
        obs = handle.create_group("obs")
        obs.attrs["_index"] = "_index"
        obs.create_dataset(
            "_index",
            data=np.asarray([f"C{index}" for index in range(n_cells)], dtype="S"),
        )
        dense = np.zeros((n_cells, n_clones), dtype=np.float32)
        dense[np.arange(n_cells), np.arange(n_cells) % n_clones] = 1
        handle.create_group("obsm").create_dataset("X_clone", data=dense)
    layout = clone_layout(path)

    tracemalloc.start()
    try:
        n_entries = sum(len(entries.cells) for entries in matrix_entries(layout, 1_000))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert n_entries == n_cells
    # A block of 1,000 rows would hold 8 MB of matrix values.
    assert peak < 2_000_000


def test_mito_clones_leave_unassigned_cells_without_a_clone(tmp_path) -> None:
    path = write_mito_h5ad(TRUTH, tmp_path / "afm.h5ad")
    barcodes, clones, _ = _rows(path, chunk_rows=10)
    assert clone_layout(path).format == "mito"
    assert barcodes[: len(TRUTH.clones)] == TRUTH.cell_ids.tolist()
    assert clones[: len(TRUTH.clones)] == [
        f"MT_{clone + 1}" if clone >= 0 else None for clone in TRUTH.clones
    ]


def _h5ad(path: Path, *, x_clone: bool = False, mito: bool = False) -> Path:
    with h5py.File(path, "w") as handle:
        obs = handle.create_group("obs")
        obs.attrs["_index"] = "_index"
        obs.create_dataset(
            "_index", data=np.asarray(["A", "B", "C"], dtype=h5py.string_dtype())
        )
        obs.create_dataset("lineage", data=np.asarray([3, 4, 3]))
        obs.create_dataset("score", data=np.asarray([0.5, np.nan, 1.0]))
        obs.create_dataset("bad_score", data=np.asarray([0.5, 2.0, 1.0]))
        obs.create_dataset(
            "label_score", data=np.asarray(["a", "b", "c"], dtype=h5py.string_dtype())
        )
        write_categorical(obs, "named", ["x", None, "y"])
        if mito:
            write_categorical(obs, "MiTo_clone", ["MT_1", "unassigned", "MT_1"])
        obsm = handle.create_group("obsm")
        if x_clone:
            obsm.create_dataset("X_clone", data=np.eye(3, 2))
    return path


def test_any_obs_column_of_an_anndata_file_reads_with_columns(tmp_path) -> None:
    path = _h5ad(tmp_path / "a.h5ad")

    assert _rows(path, columns={"clone": "named"})[:2] == (
        ["A", "B", "C"],
        ["x", None, "y"],
    )
    barcodes, clones, probabilities = _rows(
        path, columns={"clone": "lineage", "probability": "score"}
    )
    assert clones == ["3", "4", "3"]
    assert probabilities[0] == 0.5 and np.isnan(probabilities[1])
    with pytest.raises(ValueError, match="must hold probabilities from 0 to 1"):
        _rows(path, columns={"clone": "lineage", "probability": "bad_score"})
    with pytest.raises(ValueError, match="must hold numbers"):
        _rows(path, columns={"clone": "lineage", "probability": "label_score"})


@pytest.mark.parametrize(
    ("options", "contents", "message"),
    [
        ({}, {}, "holds neither obsm\\['X_clone'\\] nor obs\\['MiTo_clone'\\]"),
        ({}, {"x_clone": True, "mito": True}, "holds both"),
        ({"format": "cospar_h5ad"}, {}, "has no obsm\\['X_clone'\\]"),
        ({"format": "mito"}, {}, "has no obs\\['MiTo_clone'\\]"),
        (
            {"format": "mito", "columns": {"clone": "named"}},
            {"mito": True},
            "reads fixed columns",
        ),
        (
            {"columns": {"cell": "named", "clone": "named"}},
            {},
            "takes only 'clone' and 'probability'",
        ),
        ({"format": "h5ad"}, {}, "needs the clone column"),
        ({"columns": {"clone": "absent"}}, {}, "has no obs column 'absent'"),
        ({"format": "trex"}, {}, "is an AnnData file, not format='trex'"),
    ],
)
def test_anndata_files_that_leave_the_clones_open_are_refused(
    tmp_path, options, contents, message
) -> None:
    path = _h5ad(tmp_path / "a.h5ad", **contents)
    with pytest.raises(ValueError, match=message):
        clone_layout(path, **options)


def test_cell_names_must_be_a_dataset_of_the_obs_table(tmp_path) -> None:
    path = _h5ad(tmp_path / "a.h5ad", mito=True)
    with h5py.File(path, "a") as handle:
        handle["obs"].attrs["_index"] = "named"
    with pytest.raises(ValueError, match="stores its cell names in a form"):
        _rows(path)


def test_anndata_files_must_be_readable_as_cells_by_clones(tmp_path) -> None:
    path = tmp_path / "no_index.h5ad"
    with h5py.File(path, "w") as handle:
        handle.create_group("obs")
    with pytest.raises(ValueError, match="no AnnData obs table with cell names"):
        clone_layout(path)
    rows = _h5ad(tmp_path / "rows.h5ad")
    with h5py.File(rows, "a") as handle:
        handle["obsm"].create_dataset("X_clone", data=np.eye(4, 2))
    with pytest.raises(ValueError, match="has 4 rows for 3 cells"):
        list(matrix_entries(clone_layout(rows), 5))
    vector = _h5ad(tmp_path / "vector.h5ad")
    with h5py.File(vector, "a") as handle:
        handle["obsm"].create_dataset("X_clone", data=np.ones(3))
    with pytest.raises(ValueError, match="is not a matrix"):
        matrix_clone_ids(clone_layout(vector))
    group = _h5ad(tmp_path / "group.h5ad")
    with h5py.File(group, "a") as handle:
        handle["obsm"].create_group("X_clone")
    with pytest.raises(ValueError, match="neither a dense matrix nor a CSR or CSC"):
        matrix_clone_ids(clone_layout(group))


def test_trex_fixture_holds_its_published_clones() -> None:
    # A subset of TREX's tests/expected/clones.txt (MIT): every cell of its
    # three small clones and 30 of the 686 cells of its large one.
    barcodes, clones, _ = _rows(DATA / "trex_clones.txt", chunk_rows=7)
    assert clone_layout(DATA / "trex_clones.txt").format == "trex"
    assert len(barcodes) == len(set(barcodes)) == 41
    assert {label: clones.count(label) for label in set(clones)} == {
        "1": 2,
        "2": 1,
        "3": 8,
        "4": 30,
    }
    assert all(len(barcode) == 16 for barcode in barcodes)


def test_celltag_multi_fixture_holds_its_published_clones() -> None:
    # Six clones of the iEP clone table on Zenodo 23442375 (CC BY 4.0), with
    # its unnamed index column and repeated per-clone annotations.
    barcodes, clones, _ = _rows(DATA / "celltag_multi_iep_clones.csv", chunk_rows=5)
    assert clone_layout(DATA / "celltag_multi_iep_clones.csv").format == "celltag_multi"
    assert len(barcodes) == len(set(barcodes)) == 24
    assert {label: clones.count(label) for label in set(clones)} == {
        "r1_1001": 2,
        "r1_1002": 2,
        "r1_101": 6,
        "r1_1036": 3,
        "r2_1817": 7,
        "r2_1818": 4,
    }
    assert all(
        barcode.startswith("B4D") and barcode.endswith("-1") for barcode in barcodes
    )
