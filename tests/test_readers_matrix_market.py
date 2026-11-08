"""The one reader of Matrix Market files: headers, entries and their checks.

Master's count matrices, clone matrices and the allele count matrices of
variant callers read their headers and entries through
``readers/_matrix_market.py``, so each case here holds for all three.
"""

import gzip
import zipfile

import pandas as pd
import pytest

from cytearc.readers._matrix_market import (
    matrix_market_chunk_copy_bytes,
    matrix_market_entries,
    matrix_market_header,
)

_BANNER = "%%MatrixMarket matrix coordinate {field} general\n"


def _write(tmp_path, text: str, name: str = "m.mtx", compress: bool = False):
    path = tmp_path / name
    data = text.encode()
    path.write_bytes(gzip.compress(data, mtime=0) if compress else data)
    return path


def _entries(path, **options) -> list:
    header = matrix_market_header(path, fields=("integer", "real", "pattern"))
    return [
        (rows.tolist(), columns.tolist(), None if values is None else values.tolist())
        for rows, columns, values in matrix_market_entries(header, 2, **options)
    ]


def test_entries_come_in_chunks_from_zero_with_the_fields_values(tmp_path) -> None:
    text = (
        _BANNER.format(field="integer") + "% a comment\n\n3 2 3\n1 1 5\n3 2 7\n2 1 1\n"
    )
    assert _entries(_write(tmp_path, text)) == [
        ([0, 2], [0, 1], [5, 7]),
        ([1], [0], [1]),
    ]
    pattern = _BANNER.format(field="pattern") + "2 2 2\n1 2\n2 1\n"
    assert _entries(_write(tmp_path, pattern, "p.mtx")) == [([0, 1], [1, 0], None)]
    # A real matrix keeps its numbers, as scipy writes counts in one.
    real = _BANNER.format(field="real") + "1 1 1\n1 1 3.71E2\n"
    assert _entries(_write(tmp_path, real, "r.mtx")) == [([0], [0], [371.0])]


@pytest.mark.parametrize(
    ("version", "per_value_entry", "per_pattern_entry"),
    [
        # Each field, and the float32 column past them, becomes a 36-byte int,
        # the size of the largest int64, behind an 8-byte reference.
        ("2.2.3", 176, 132),
        # The int64 fields and the float32 column are copied once more.
        ("2.3.0", 28, 20),
        ("2.3.3", 28, 20),
        ("3.0.0", 0, 0),
        ("3.1.1", 0, 0),
    ],
)
def test_pandas_2_charges_the_copies_it_makes_of_each_parsed_entry(
    tmp_path, monkeypatch, version, per_value_entry, per_pattern_entry
) -> None:
    monkeypatch.setattr(pd, "__version__", version)
    for field, expected in [
        ("integer", per_value_entry),
        ("real", per_value_entry),
        ("pattern", per_pattern_entry),
    ]:
        path = _write(tmp_path, _BANNER.format(field=field) + "1 1 0\n", f"{field}.mtx")
        header = matrix_market_header(path, fields=("integer", "real", "pattern"))
        assert matrix_market_chunk_copy_bytes(header) == expected


def test_masters_separator_and_index_offset_reach_the_entries(tmp_path) -> None:
    path = _write(tmp_path, _BANNER.format(field="integer") + "2 2 2\n0,1,4\n1,0,2\n")
    header = matrix_market_header(path)
    found = [
        (rows.tolist(), columns.tolist(), values.tolist())
        for rows, columns, values in matrix_market_entries(
            header, 10, separator=",", index_offset=0
        )
    ]
    assert found == [([0, 1], [1, 0], [4, 2])]
    # With the default offset the coordinates of 0 lie outside the matrix.
    with pytest.raises(ValueError, match="outside the declared dimensions, 2 by 2"):
        list(matrix_market_entries(header, 10, separator=","))


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("1 1 -1\n", "counts must be finite non-negative integers, but .* holds -1.0$"),
        ("1 1 nan\n", "counts must be finite non-negative integers, but .* holds nan$"),
        ("1 1 2.5\n", "counts must be finite non-negative integers, but .* holds 2.5$"),
    ],
)
def test_counts_must_be_whole_numbers_of_at_least_zero(tmp_path, body, message) -> None:
    path = _write(tmp_path, _BANNER.format(field="real") + "2 2 1\n" + body)
    header = matrix_market_header(path)
    with pytest.raises(ValueError, match=message):
        list(matrix_market_entries(header, 10, counts=True))
    # Without counts=True the values are read as they are.
    assert len(list(matrix_market_entries(header, 10))) == 1


def test_a_matrix_without_entries_holds_none_and_one_that_declares_none_holds_none(
    tmp_path,
) -> None:
    empty = _write(tmp_path, _BANNER.format(field="integer") + "2 2 0\n")
    assert _entries(empty) == []
    # Master's count reader once returned early for a matrix that declares
    # no entries; one reader refuses the entries that it holds anyway.
    held = _write(tmp_path, _BANNER.format(field="integer") + "2 2 0\n1 1 1\n", "h.mtx")
    with pytest.raises(ValueError, match="header declares 0 entries, but 1 were read"):
        _entries(held)


def test_files_that_are_no_text_or_cut_short_are_refused_by_name(tmp_path) -> None:
    binary = tmp_path / "binary.mtx"
    binary.write_bytes(b"\xff\xfe\x00 binary")
    with pytest.raises(ValueError, match="binary.mtx is not a text file that CyteArc"):
        matrix_market_header(binary)
    data = gzip.compress(
        (
            _BANNER.format(field="integer")
            + "50000 2 50000\n"
            + "".join(f"{row} 1 1\n" for row in range(1, 50_001))
        ).encode(),
        mtime=0,
    )
    head = tmp_path / "head.mtx.gz"
    head.write_bytes(data[:20])
    with pytest.raises(ValueError, match="head.mtx.gz is cut short or corrupt"):
        matrix_market_header(head)
    cut = tmp_path / "cut.mtx.gz"
    cut.write_bytes(data[: len(data) // 2])
    header = matrix_market_header(cut)
    with pytest.raises(ValueError, match="cut.mtx.gz is cut short or corrupt"):
        list(matrix_market_entries(header, 1000))


# Text of 10,000 characters from a header, and how a message quotes it: its
# first 37 and last 38 characters around "...", 80 characters with the quotes.
_LONG = "a" * 5_000 + "z" * 5_000
_QUOTED = "'" + "a" * 37 + "..." + "z" * 38 + "'"


@pytest.mark.parametrize(
    ("text", "shown"),
    [
        (_LONG + "\n2 2 0\n", f"begins {_QUOTED}"),
        (_BANNER.format(field=_LONG) + "2 2 0\n", f"holds {_QUOTED} values"),
        (
            "%%MatrixMarket matrix coordinate integer " + _LONG + "\n2 2 0\n",
            f"is {_QUOTED}",
        ),
        (
            _BANNER.format(field="integer") + "1 1 " + _LONG + "\n",
            "has '1 1 " + "a" * 33 + "..." + "z" * 38 + "'",
        ),
        # Python reads whole numbers of up to 4,300 digits from text.
        (
            _BANNER.format(field="integer") + "-1 1 " + "0" * 4_000 + "\n",
            "declares '-1 1 " + "0" * 32 + "..." + "0" * 38 + "'",
        ),
    ],
    ids=["banner", "field", "symmetry", "dimensions", "negative dimensions"],
)
def test_a_header_quotes_its_long_text_cut_to_80_characters(
    tmp_path, text, shown
) -> None:
    path = _write(tmp_path, text)
    with pytest.raises(ValueError) as refused:
        matrix_market_header(path, fields=("integer", "real", "pattern"))
    message = str(refused.value)
    assert shown in message
    # The message's own words take at most 91 characters beside the file.
    assert len(message) <= 91 + len(_QUOTED) + len(str(path))


def test_entries_of_another_shape_than_their_fields_are_refused(tmp_path) -> None:
    path = _write(tmp_path, _BANNER.format(field="integer") + "2 2 1\n1 1\n")
    with pytest.raises(ValueError, match="Could not parse Matrix Market coordinates"):
        list(matrix_market_entries(matrix_market_header(path), 10))


@pytest.mark.parametrize("chunk_rows", [1, 2, 1000])
@pytest.mark.parametrize(
    ("field", "body", "wider"),
    [
        # A pattern clone matrix whose entries carry a value.
        ("pattern", "1 2 1\n2 3 1\n4 1 1\n", 1),
        # An integer matrix whose entries carry a fourth field: the first,
        # one within the entries, and the last.
        ("integer", "1 2 1 1\n2 3 2 1\n3 1 3 1\n", 1),
        ("integer", "1 2 1\n2 3 2 9\n3 1 3\n", 2),
        ("integer", "1 2 1\n2 3 2\n3 1 3 1\n", 3),
    ],
)
def test_entries_of_more_fields_than_their_field_are_refused_wherever_chunks_start(
    tmp_path, field, body, wider, chunk_rows
) -> None:
    # pandas read the first field of a wider first entry as an index, so
    # every coordinate shifted, and dropped the last field of one that
    # started a chunk; neither was refused.
    path = _write(tmp_path, _BANNER.format(field=field) + "4 4 3\n" + body)
    header = matrix_market_header(path, fields=("integer", "real", "pattern"))
    n_fields = 2 if field == "pattern" else 3
    message = (
        f"Each entry of a Matrix Market {field} matrix holds {n_fields} fields, .*"
        f"but (entry {wider} of .*m.mtx holds more"
        f"|line {wider + 2} of .*m.mtx holds {n_fields + 1})$"
    )
    with pytest.raises(ValueError, match=message):
        list(matrix_market_entries(header, chunk_rows))


def test_entries_of_two_fields_more_are_refused_naming_their_line(tmp_path) -> None:
    path = _write(
        tmp_path, _BANNER.format(field="integer") + "4 4 3\n1 2 1\n2 3 2\n3 1 3 1 1\n"
    )
    header = matrix_market_header(path)
    with pytest.raises(ValueError, match="but line 5 of .*m.mtx holds 5$"):
        list(matrix_market_entries(header, 1000))
    # A chunk that starts at the entry keeps its fourth field.
    with pytest.raises(ValueError, match="but entry 3 of .*m.mtx holds more$"):
        list(matrix_market_entries(header, 1))


def test_counts_of_more_fields_are_refused_by_each_matrix_reader(tmp_path) -> None:
    from cytearc.readers import MtxReader, inspect_mtx
    from cytearc.readers.clone_tables import clone_layout, matrix_entries

    # A count matrix as master's reader reads it, whose entries carry a
    # fourth field: every coordinate was read shifted, and its counts were
    # the fourth fields.
    folder = tmp_path / "mex"
    folder.mkdir()
    _write(
        folder,
        _BANNER.format(field="integer") + "3 3 3\n1 2 1 1\n2 3 2 1\n3 1 3 1\n",
        "matrix.mtx",
    )
    (folder / "features.tsv").write_text("".join(f"f{i}\tg{i}\n" for i in range(3)))
    (folder / "barcodes.tsv").write_text("".join(f"c{i}\n" for i in range(3)))
    with pytest.raises(ValueError, match="but entry 1 of .*matrix.mtx holds more"):
        MtxReader(inspect_mtx(folder)[0], temp_dir=str(tmp_path))
    # A pattern clone matrix whose entries carry a value, as a clone matrix.
    clones = tmp_path / "clones"
    clones.mkdir()
    _write(clones, _BANNER.format(field="pattern") + "3 2 2\n1 2 1\n3 1 1\n", "c.mtx")
    (clones / "cells.tsv").write_text("cell\nAAAC-1\nAAAG-1\nAAAT-1\n")
    (clones / "clones.tsv").write_text("clone\nk1\nk2\n")
    layout = clone_layout(clones / "c.mtx")
    with pytest.raises(ValueError, match="but entry 1 of .*c.mtx holds more"):
        list(matrix_entries(layout, 1000))


def test_errors_of_an_archives_matrix_name_its_member_and_the_archive(
    tmp_path,
) -> None:
    from cytearc.readers import MtxReader, inspect_mtx

    archive_path = tmp_path / "sample_MEX.zip"
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr(
            "matrix.mtx",
            _BANNER.format(field="integer") + "2 2 3\n2 2 1\n1 1 1\n3 1 1\n",
        )
        archive.writestr("features.tsv", "f1\tg1\nf2\tg2\n")
        archive.writestr("barcodes.tsv", "c1\nc2\n")
    # The member is read from a temporary file, which is gone once the
    # reader fails, so errors name the member of the archive instead.
    with pytest.raises(
        ValueError,
        match=r"coordinate of matrix\.mtx in .*sample_MEX\.zip is outside the declared",
    ) as error:
        MtxReader(inspect_mtx(archive_path)[0], temp_dir=str(tmp_path))
    assert "cytearc-mtx-archive" not in str(error.value)
