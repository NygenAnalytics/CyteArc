"""The one reader of Matrix Market files: headers, entries and their checks.

Master's count matrices, clone matrices and the allele count matrices of
variant callers read their headers and entries through
``readers/_matrix_market.py``, so each case here holds for all three.
"""

import gzip

import pytest

from cytearc.readers._matrix_market import (
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


def test_entries_of_another_shape_than_their_fields_are_refused(tmp_path) -> None:
    path = _write(tmp_path, _BANNER.format(field="integer") + "2 2 1\n1 1\n")
    with pytest.raises(ValueError, match="Could not parse Matrix Market coordinates"):
        list(matrix_market_entries(matrix_market_header(path), 10))
