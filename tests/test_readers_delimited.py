"""The one reader of delimited text: rows, their fields and their lines.

Every table of clones, lineage barcodes, recorder alleles, contigs, GoT
and giftwrap is read by ``readers/_delimited.py``, so these cases pin what
each of those readers inherits: a row of another number of fields than its
header is refused, naming its line, whatever size the chunks are, and every
row comes with its line in the file.
"""

import csv
import gzip
import io
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cytearc.readers._delimited as delimited
from cytearc.readers._delimited import (
    TextLayout,
    count_rows,
    first_lines,
    split_fields,
    text_chunks,
    text_layout,
)
from tests.memory_helpers import traced_peak

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
    rows = "".join(f"c{index}\t{index}\n" for index in range(30_000))
    path = tmp_path / "t.tsv"
    path.write_bytes(b"cell\tclone\n" + rows.encode() + b"\xff\xfe\tbad\n")
    found, error = _read(str(path), 5000)
    assert 0 < len(found) < 30_000
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


@pytest.mark.parametrize("chunk_rows", CHUNK_ROWS)
@pytest.mark.parametrize("quoted", [False, True])
def test_space_separated_fields_are_split_by_spaces_and_tabs_alone(
    tmp_path, chunk_rows, quoted
) -> None:
    # Python's str.split splits on more whitespace than pandas separates
    # fields by, such as a no-break space, a form feed or an em space, which
    # are text in a field.
    text = "cell clone\nAAAC-1 clone\xa0one\nAAAG-1\tc\x0c2\nAAAT-1 \u2003\n"
    rows, error = _read(_write(tmp_path / "t.txt", text), chunk_rows, quoted=quoted)
    assert error == []
    assert rows == [
        (2, ["AAAC-1", "clone\xa0one"]),
        (3, ["AAAG-1", "c\x0c2"]),
        (4, ["AAAT-1", "\u2003"]),
    ]
    # Two fields that such whitespace alone parts are one field, too few.
    one = _write(tmp_path / "one.txt", "cell clone\nAAAC-1 c1\nAAAG-1\xa0c2\n")
    assert _read(one, chunk_rows, quoted=quoted)[1] == [
        " is malformed: line 3 holds 1 field, but its header names 2 columns"
    ]
    assert split_fields("A\xa0c2 x\t", " ", quoted=quoted) == ["A\xa0c2", "x"]


@pytest.mark.parametrize("chunk_rows", CHUNK_ROWS)
def test_quoted_space_separated_fields_keep_other_whitespace_and_split_at_tabs(
    tmp_path, chunk_rows
) -> None:
    path = _write(
        tmp_path / "t.txt",
        '"cell" "clone"\n"A"\t"1"\n"B" \t "x\xa0y"\t\n  "C"  "3"\nD\xa0E "4"\n',
    )
    rows, error = _read(path, chunk_rows)
    assert error == []
    assert rows == [
        (2, ["A", "1"]),
        (3, ["B", "x\xa0y"]),
        (4, ["C", "3"]),
        (5, ["D\xa0E", "4"]),
    ]


@pytest.mark.parametrize("chunk_rows", CHUNK_ROWS)
def test_quotes_after_a_closing_quote_and_doubled_before_a_delimiter(
    tmp_path, chunk_rows
) -> None:
    # Text after a closing quote joins its field, a quote in that text
    # included, and a doubled quote within a quoted field is one quote.
    path = _write(tmp_path / "t.csv", 'a,b\n"x"y"z,b\n"p""q,r",s\n"",""\n"t""",u\n')
    rows, error = _read(path, chunk_rows)
    assert error == []
    assert rows == [
        (2, ['xy"z', "b"]),
        (3, ['p"q,r', "s"]),
        (4, ["", ""]),
        (5, ['t"', "u"]),
    ]
    assert split_fields('"x"y"z,b', ",") == ['xy"z', "b"]
    assert split_fields('"p""q,r",s', ",") == ['p"q,r', "s"]


@pytest.mark.parametrize("chunk_rows", CHUNK_ROWS)
@pytest.mark.parametrize("first", ["", '"q"'])
def test_a_line_of_tabs_is_a_row_of_empty_fields(tmp_path, chunk_rows, first) -> None:
    path = _write(
        tmp_path / "t.tsv", f"a\tb\tc\n\t\t\nx\t\t\n \t \t \n{first}\t1\t2\n\t\n"
    )
    rows, error = _read(path, chunk_rows)
    assert rows == [
        (2, ["", "", ""]),
        (3, ["x", "", ""]),
        (4, [" ", " ", " "]),
        (5, [first.strip('"'), "1", "2"]),
    ]
    assert error == [
        " is malformed: line 6 holds 2 fields, but its header names 3 columns"
    ]


@pytest.mark.parametrize("read_characters", [1, 64, delimited._READ_CHARACTERS])
@pytest.mark.parametrize("chunk_rows", [1, 7, 1000])
def test_quoted_rows_that_straddle_reads_of_text_keep_their_lines_and_fields(
    tmp_path, monkeypatch, read_characters, chunk_rows
) -> None:
    # Every 37th row's quoted field holds two line breaks, so reads of text
    # of every size end within some of them, and every row holds a quote.
    monkeypatch.setattr(delimited, "_READ_CHARACTERS", read_characters)
    lines = ["cell,note"]
    expected = []
    for index in range(2_000):
        note = f"n{index}\nmore\nend" if index % 37 == 5 else f"n{index}"
        expected.append((len(lines) + 1, [f"c{index}", note]))
        lines += f'c{index},"{note}"'.split("\n")
    path = _write(tmp_path / "t.csv", "\n".join(lines) + "\n")
    rows, error = _read(path, chunk_rows)
    assert (rows, error) == (expected, [])


def test_a_read_of_text_that_holds_only_blank_lines_ends_no_rows(tmp_path) -> None:
    # 50,000 blank lines fill reads of text without a row.
    path = _write(tmp_path / "t.tsv", "cell\tclone\nA\t1\n" + "\n" * 50_000 + "B\t2\n")
    assert _read(path, 10) == ([(2, ["A", "1"]), (50_003, ["B", "2"])], [])
    assert count_rows(text_layout(path)) == 2


@pytest.mark.parametrize("chunk_rows", CHUNK_ROWS)
def test_lone_carriage_returns_end_lines_as_line_feeds_do(tmp_path, chunk_rows) -> None:
    path = _write(tmp_path / "t.csv", 'cell,clone\rA,1\r\r"B\rC",2\rD,3\r\nE,4')
    rows, error = _read(path, chunk_rows)
    assert error == []
    assert rows == [
        (2, ["A", "1"]),
        (4, ["B\rC", "2"]),
        (6, ["D", "3"]),
        (7, ["E", "4"]),
    ]


def test_a_quoted_field_longer_than_the_csv_modules_limit_reads_in_any_row(
    tmp_path,
) -> None:
    # The csv module refuses a field of more than 131,072 characters.
    long = "x" * 200_000
    path = _write(tmp_path / "t.csv", f'cell,"{long}"\n"A,{long}",1\n"B",2\n')
    assert text_layout(path).names == ("cell", long)
    assert _read(path, 1) == ([(2, [f"A,{long}", "1"]), (3, ["B", "2"])], [])
    assert split_fields(f'"{long}" y', " ") == [long, "y"]
    trailing = _write(tmp_path / "r.csv", f'cell\n"{long}",\n"B",\n')
    with pytest.raises(ValueError, match="end the header with the delimiter too"):
        text_layout(trailing)


def test_reading_a_long_table_holds_a_read_of_its_text_not_more(tmp_path) -> None:
    # 300,000 rows of nine characters: reads of a mebibyte of text at a time
    # held 12 MB of lines while their rows were counted.
    rows = "".join(f"c{index:07d}\n" for index in range(300_000))
    text = text_layout(_write(tmp_path / "cells.tsv", "cell\n" + rows))
    n_rows, peak = traced_peak(lambda: count_rows(text), 500_000)
    assert n_rows == 300_000
    assert peak < 500_000
    n_rows, peak = traced_peak(
        lambda: sum(len(chunk) for _lines, chunk in text_chunks(text, 1_000)),
        1_000_000,
    )
    assert n_rows == 300_000
    assert peak < 1_000_000


# Text that a table's random rows are made of, per delimiter.
_ATOMS = ["a", "b", '"', '"', "D", "D", " ", "\n", "\r\n", "\t", ",", '""']
# Lines end in \n, \r\n or \r and keep their ends.
_LINE = re.compile(r"[^\r\n]*(?:\r\n|\n|\r|$)")


def _blank(content: str, delimiter: str) -> bool:
    return not content.strip(" \t") and (delimiter == " " or delimiter not in content)


def _lines_of(text: str) -> list[str]:
    return [match.group(0) for match in _LINE.finditer(text) if match.group(0)]


def _marked(header: str, body: str, delimiter: str) -> str:
    """The text with ``Z<line>`` at the end of each line that holds text.

    pandas' last field of each row then ends in the number of its last line,
    which no padding hides, and the field shows how many fields the row
    holds.
    """
    out = [header]
    for number, line in enumerate(_lines_of(body), start=2):
        content = line.rstrip("\r\n")
        end = line[len(content) :]
        if _blank(content, delimiter):
            out.append(line)
        elif delimiter == " ":
            core = content.rstrip(" \t")
            out.append(core + f"Z{number}" + content[len(core) :] + end)
        else:
            out.append(content + f"Z{number}" + end)
    return "".join(out)


def _pandas_rows(text: str, delimiter: str) -> list | Exception:
    """Each row's first line, fields and first three values, as pandas reads them."""
    try:
        frame = pd.read_csv(
            io.StringIO(text),
            sep=r"\s+" if delimiter == " " else delimiter,
            header=None,
            skiprows=1,
            names=list(range(40)),
            dtype=str,
            keep_default_na=False,
            skip_blank_lines=True,
            quotechar='"',
            index_col=False,
        )
    except pd.errors.EmptyDataError:
        return []
    except pd.errors.ParserError as error:
        return error
    lines = _lines_of(text)
    rows, previous_end = [], 1
    for values in frame.itertuples(index=False):
        fields = list(values)
        n_fields = max(
            (position + 1 for position, value in enumerate(fields) if value != ""),
            default=0,
        )
        end = int(re.findall(r"Z(\d+)", fields[n_fields - 1])[-1])
        start = previous_end + 1
        while _blank(lines[start - 1].rstrip("\r\n"), delimiter):
            start += 1
        rows.append((start, n_fields, fields[:3]))
        previous_end = end
    return rows


def test_rows_lines_and_fields_match_pandas_reading_the_text_itself(
    tmp_path, monkeypatch
) -> None:
    """Random tables of quotes, doubled quotes, delimiters, spaces and line
    ends read alike through the rows, at every chunk and read size, and as
    pandas reads their text whole: each row's first line, its values, and
    the first row of another number of fields than three, which is refused."""
    rng = random.Random(11)
    path = tmp_path / "t.txt"
    checked = 0
    for _case in range(200):
        delimiter = rng.choice(["\t", ",", " "])
        atoms = [delimiter if atom == "D" else atom for atom in _ATOMS]
        body = "".join(rng.choice(atoms) for _atom in range(rng.randint(0, 60)))
        text = _marked(delimiter.join(["c0", "c1", "c2"]) + "\n", body, delimiter)
        path.write_text(text, encoding="utf-8", newline="")
        layout = TextLayout(
            path=str(path),
            names=("c0", "c1", "c2"),
            delimiter=delimiter,
            row_names=False,
        )
        results = []
        for read_characters in (1, delimited._READ_CHARACTERS):
            monkeypatch.setattr(delimited, "_READ_CHARACTERS", read_characters)
            for chunk_rows in (1, 2, 1000):
                rows, error = [], None
                try:
                    for lines, chunk in text_chunks(layout, chunk_rows):
                        rows += zip(lines.tolist(), chunk.values.tolist(), strict=True)
                except ValueError as refusal:
                    error = str(refusal).removeprefix(f"{path} is malformed: ")
                results.append((rows, error))
        assert all(result == results[0] for result in results), text
        expected = _pandas_rows(text, delimiter)
        rows, error = results[0]
        if isinstance(expected, Exception):
            # pandas refuses a row of more fields than it names, as the rows do.
            assert error is not None, text
            continue
        checked += 1
        wrong = next(
            (index for index, row in enumerate(expected) if row[1] != 3), len(expected)
        )
        assert rows == [(start, values) for start, _n, values in expected[:wrong]], text
        if wrong == len(expected):
            assert error is None, text
        else:
            start, n_fields, _values = expected[wrong]
            assert error is not None and error.startswith(
                f"line {start} holds {n_fields} field"
            ), text
    # Most tables are checked against pandas' own rows.
    assert checked > 100


def test_the_tokenizer_splits_rows_and_joins_their_lines_as_the_csv_module_does() -> (
    None
):
    """The reader's own tokenizer reads a field longer than the csv module
    reads, 131,072 characters, and the csv module reads every other row, so
    the two must split rows and join lines alike."""
    rng = random.Random(5)
    for _case in range(3_000):
        delimiter = rng.choice(["\t", ",", " "])
        atoms = ["a", "b", '"', '"', '""', delimiter, delimiter, " ", "\t", ",", "\n"]
        text = "".join(rng.choice(atoms) for _atom in range(rng.randint(1, 30)))
        lines = [line + "\n" for line in text.split("\n")]
        if '"' not in lines[0]:
            continue
        dialect = delimited._dialect(delimiter)
        reader = csv.reader(
            (delimited._for_csv(line, delimiter) for line in lines), **dialect
        )
        record = next(reader)
        taken, n_fields, _cut = delimited._quoted_record(
            lines[0], iter(lines[1:]), delimiter
        )
        assert (n_fields, len(taken)) == (len(record), reader.line_num), text
        # A row's fields, as split from its line without the line end.
        line = lines[0].rstrip("\n")
        record = next(csv.reader([delimited._for_csv(line, delimiter)], **dialect))
        _state, fields = delimited._tokenize(
            delimited._for_scan(line, delimiter), delimiter
        )
        assert split_fields(line, delimiter) == fields, text
        # The csv module reads space-separated text with its tabs made spaces,
        # which splits it into as many fields.
        if delimiter == " ":
            fields = [field.replace("\t", " ") for field in fields]
        assert fields == record, text


def test_pandas_errors_of_a_parse_name_the_file(tmp_path) -> None:
    # The rows hand pandas whole rows of the header's width, so these errors
    # reach a caller of parsed_chunks that hands pandas the file itself.
    options = {"header": None, "names": [0, 1], "chunksize": 10}
    path = _write(tmp_path / "t.csv", 'a,b\n"c,d\n')
    with pytest.raises(ValueError, match=r"t\.csv is malformed: .*EOF inside string"):
        list(delimited.parsed_chunks(path, path, **options))
    data = gzip.compress(("a,b\n" + "c,d\n" * 50_000).encode(), mtime=0)
    cut = tmp_path / "c.csv.gz"
    cut.write_bytes(data[: len(data) // 2])
    with (
        gzip.open(cut, "rt") as handle,
        pytest.raises(ValueError, match=r"c\.csv\.gz is cut short or corrupt"),
    ):
        list(delimited.parsed_chunks(str(cut), handle, **options))


def test_stripping_text_leaves_no_reference_cycle() -> None:
    import gc
    import weakref

    from cytearc.readers._delimited import joined_columns, stripped_text

    # pandas 2 keeps the .str accessor on its Series, which the accessor
    # holds, so a chunk stripped through it would outlive its last reference
    # until a collection; stripped one value at a time, it goes at once.
    frame = pd.DataFrame({0: [" a ", "b\t", ""], 1: ["x", " y", "z"]}, dtype=object)
    column = frame[0]
    gone = (weakref.ref(frame), weakref.ref(column))
    collecting = gc.isenabled()
    gc.disable()
    try:
        assert stripped_text(column).tolist() == ["a", "b", ""]
        assert joined_columns(frame, [0, 1], "_").tolist() == ["a_x", "b_y", ""]
        del frame, column
        assert [reference() for reference in gone] == [None, None]
    finally:
        if collecting:
            gc.enable()
