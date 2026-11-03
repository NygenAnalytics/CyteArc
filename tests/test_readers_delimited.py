"""The one reader of delimited text: rows, their fields and their lines.

Every table of clones, lineage barcodes, recorder alleles, contigs and
variant callers is read by ``readers/_delimited.py``, so these cases pin what
each of those readers inherits: a row of another number of fields than its
header is refused, naming its line, whatever size the chunks are, and every
row comes with its line in the file.
"""

import gzip
from pathlib import Path

import numpy as np
import pytest

from cytearc.readers._delimited import (
    count_rows,
    first_lines,
    split_fields,
    text_chunks,
    text_layout,
)

# Chunk sizes that end a chunk before, at and after each row of the tables
# below, as memory budgets of every size choose them.
CHUNK_ROWS = (1, 2, 3, 4, 1000)


def _write(path: Path, text: str, *, compress: bool = False) -> str:
    data = text.encode("utf-8")
    path.write_bytes(gzip.compress(data, mtime=0) if compress else data)
    return str(path)


def _read(path: str, chunk_rows: int, **options: object) -> tuple[list, list]:
    """Every row's line and fields, and the error that ended the rows, if any."""
    rows: list = []
    try:
        for lines, chunk in text_chunks(text_layout(path, **options), chunk_rows):
            rows += list(zip(lines.tolist(), chunk.values.tolist(), strict=True))
    except ValueError as error:
        return rows, [str(error).removeprefix(path)]
    return rows, []


@pytest.mark.parametrize("chunk_rows", CHUNK_ROWS)
@pytest.mark.parametrize(
    ("text", "line", "n_fields", "names"),
    [
        # A row longer than its header, which pandas drops a field of where a
        # chunk starts and refuses elsewhere.
        ("a\tb\n1\t2\n3\t4\n5\t6\t7\n8\t9\n", 4, 3, 2),
        # A row shorter than its header, which pandas pads with empty text.
        ("a\tb\n1\t2\n3\t4\n5\n8\t9\n", 4, 1, 2),
        ("a,b\n1,2\n3,4\n5,6,\n8,9\n", 4, 3, 2),
        ("a b\n1 2\n3   4\n5\n8 9\n", 4, 1, 2),
        # Past blank lines and a quoted field that holds a line break.
        ('a,b\n1,2\n\n"x\ny",4\n\n5\n8,9\n', 7, 1, 2),
    ],
)
def test_a_row_of_another_width_is_refused_wherever_the_chunks_end(
    tmp_path, chunk_rows, text, line, n_fields, names
) -> None:
    path = _write(tmp_path / "t.txt", text)
    rows, error = _read(path, chunk_rows)
    # The rows before it come first, whatever the chunks, so a caller meets
    # its own errors in them before this one.
    assert [found for found, _fields in rows][-1] < line
    assert len(rows) == 2
    assert error == [
        f" is malformed: line {line} holds {n_fields} "
        f"field{'' if n_fields == 1 else 's'}, but its header names {names} columns"
    ]


@pytest.mark.parametrize("chunk_rows", CHUNK_ROWS)
def test_r_row_names_and_tables_without_a_header_count_fields_too(
    tmp_path, chunk_rows
) -> None:
    named = _write(tmp_path / "r.txt", '"x" "y"\n"r1" 1 2\n"r2" 3 4\n"r3" 5\n')
    rows, error = _read(named, chunk_rows)
    assert [line for line, _fields in rows] == [2, 3]
    assert rows[0][1] == ["r1", "1", "2"]
    assert error == [
        " is malformed: line 4 holds 2 fields, but its rows hold 3: a row name "
        "and the 2 columns that its header names"
    ]


@pytest.mark.parametrize("chunk_rows", CHUNK_ROWS)
def test_rows_keep_their_lines_past_blank_lines_and_quoted_breaks(
    tmp_path, chunk_rows
) -> None:
    path = _write(
        tmp_path / "t.csv",
        'cell,clone\nA,1\n\n   \n"B\nC",2\n,\n"D""E",3\nF,4',
    )
    rows, error = _read(path, chunk_rows)
    assert error == []
    # A line of whitespace holds no row, a line of delimiters holds empty
    # fields, and the last row needs no line end.
    assert rows == [
        (2, ["A", "1"]),
        (5, ["B\nC", "2"]),
        (7, ["", ""]),
        (8, ['D"E', "3"]),
        (9, ["F", "4"]),
    ]


def test_a_quote_within_an_unquoted_field_is_text_and_joins_no_lines(tmp_path) -> None:
    path = _write(tmp_path / "t.tsv", 'cell\tallele\nA\tGC"A\nB\tT"\nC\tx\n')
    rows, error = _read(path, 2)
    assert error == []
    assert rows == [(2, ["A", 'GC"A']), (3, ["B", 'T"']), (4, ["C", "x"])]


def test_a_quote_that_never_closes_is_refused_after_the_rows_before_it(
    tmp_path,
) -> None:
    # The quote swallows more than the csv module's longest field, 131,072
    # characters, which the reader has no limit of.
    rows = [f"cell{index},{index}" for index in range(30_000)]
    rows[200] = 'cell200,"200'
    path = _write(tmp_path / "t.csv", "cell,clone\n" + "\n".join(rows) + "\n")
    # The layout samples the first 100 rows, so the rows read meet it.
    found, error = _read(path, 64)
    assert len(found) == 200 and found[-1][0] == 201
    assert error == [" is malformed: the quote opened on line 202 is never closed"]


def test_text_that_is_not_utf8_past_the_first_rows_is_refused_alike_by_any_chunks(
    tmp_path,
) -> None:
    # The rows of the reads of text before the bad bytes come first, and the
    # reads do not follow the chunks' size.
    rows = "".join(f"c{index}\t{index}\n" for index in range(200_000))
    path = tmp_path / "t.tsv"
    path.write_bytes(b"cell\tclone\n" + rows.encode() + b"\xff\xfe\tbad\n")
    found, error = _read(str(path), 5000)
    assert 0 < len(found) < 200_000
    assert error == [" is not a text file that CyteArc can read"]
    assert _read(str(path), 7) == (found, error)


def test_a_gzip_file_cut_short_is_refused_after_the_rows_it_holds(tmp_path) -> None:
    rows = "".join(f"c{index}\t{index}\n" for index in range(50_000))
    data = gzip.compress(("cell\tclone\n" + rows).encode(), mtime=0)
    path = tmp_path / "t.tsv.gz"
    path.write_bytes(data[: len(data) // 2])
    found, error = _read(str(path), 5000)
    assert 0 < len(found) < 50_000
    assert error[0].startswith(" is cut short or corrupt")


@pytest.mark.parametrize("compress", [False, True])
def test_a_byte_order_mark_is_no_part_of_the_header_or_a_first_row(
    tmp_path, compress
) -> None:
    path = _write(tmp_path / "t.tsv", "\ufeffcell\tclone\nA\t1\n", compress=compress)
    assert text_layout(path).names == ("cell", "clone")
    headerless = _write(tmp_path / "h.tsv", "\ufeffA\t1\n", compress=compress)
    assert first_lines(headerless, 1) == ["A\t1"]


@pytest.mark.parametrize(
    ("name", "compress"),
    [("plain.tsv.gz", False), ("compressed.tsv", True), ("c.tsv.gz", True)],
)
def test_a_files_bytes_not_its_name_say_whether_it_is_compressed(
    tmp_path, name, compress
) -> None:
    path = _write(tmp_path / name, "cell\tclone\nA\t1\nB\t2\n", compress=compress)
    rows, error = _read(path, 10)
    assert (rows, error) == ([(2, ["A", "1"]), (3, ["B", "2"])], [])


def test_a_row_longer_than_a_read_of_text_keeps_its_fields(tmp_path) -> None:
    long = "x" * 300_000
    path = _write(tmp_path / "t.tsv", f"cell\tclone\nA\t{long}\nB\t2\n")
    rows, error = _read(path, 1)
    assert error == [] and rows[0] == (2, ["A", long]) and rows[1] == (3, ["B", "2"])


def test_counting_rows_checks_every_rows_fields(tmp_path) -> None:
    path = _write(tmp_path / "cells.tsv", 'cell\n"a, b"\n\nc\n')
    assert count_rows(text_layout(path)) == 2
    wide = _write(tmp_path / "wide.tsv", "cell\tsample\nA\ts1\nB\n")
    with pytest.raises(ValueError, match="line 3 holds 1 field, but its header"):
        count_rows(text_layout(wide))


def test_quoted_and_unquoted_tables_split_fields_as_their_readers_do() -> None:
    assert split_fields('"a\tb"\tc', "\t") == ["a\tb", "c"]
    assert split_fields('"a\tb"\tc', "\t", quoted=False) == ['"a', 'b"', "c"]
    assert split_fields('  "x y"   z ', " ") == ["x y", "z"]
    assert split_fields("  x\t y  ", " ") == ["x", "y"]


def test_unquoted_tables_read_quotes_as_part_of_their_values(tmp_path) -> None:
    path = _write(tmp_path / "t.tsv", 'a\tb\n"x\ty"\n"z\t1\n')
    text = text_layout(path, quoted=False)
    lines, chunk = next(text_chunks(text, 10))
    assert lines.tolist() == [2, 3]
    assert chunk.values.tolist() == [['"x', 'y"'], ['"z', "1"]]


def test_columns_parse_only_the_positions_asked_but_count_every_field(
    tmp_path,
) -> None:
    path = _write(tmp_path / "t.tsv", "a\tb\tc\n1\t2\t3\n4\t5\n")
    text = text_layout(path)
    chunks = text_chunks(text, 1, columns=[2])
    lines, chunk = next(chunks)
    assert list(chunk.columns) == [2] and chunk[2].tolist() == ["3"]
    with pytest.raises(ValueError, match="line 3 holds 2 fields"):
        next(chunks)
    assert np.array_equal(lines, [2])


def test_quoted_rows_read_text_after_a_closing_quote_and_empty_fields(tmp_path) -> None:
    # Text after a closing quote joins its field, and two delimiters in a
    # row hold an empty field, in a row that the quote machine reads.
    path = _write(tmp_path / "t.tsv", 'a\tb\tc\n"x"y\t\t"z"\n"p""q"\t""\tr\n')
    rows, error = _read(path, 10)
    assert error == []
    assert rows == [(2, ["xy", "", "z"]), (3, ['p"q', "", "r"])]
    wide = _write(tmp_path / "w.tsv", 'a\tb\nx\ty\n"x"y\t\t\n')
    assert _read(wide, 10)[1] == [
        " is malformed: line 3 holds 3 fields, but its header names 2 columns"
    ]
