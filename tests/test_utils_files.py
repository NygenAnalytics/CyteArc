"""The one opener of plain and gzip-compressed files.

A file's first bytes, never its name, say whether it is gzip-compressed, and
text is UTF-8 with a byte-order mark at its start dropped.
"""

import gzip
import io
import zipfile

import pytest

from cytearc.utils.files import (
    GZIP_SIGNATURE,
    HDF5_SIGNATURE,
    file_signature,
    is_gzip,
    is_hdf5,
    open_binary,
    open_text,
)

TEXT = "\ufeffcell\tclone\r\nA\t1\rB\t2\nC\t3"


@pytest.mark.parametrize(
    ("name", "compress"),
    [("t.txt", False), ("t.txt.gz", False), ("t.txt", True), ("t.gz", True)],
)
def test_a_files_first_bytes_decide_whether_it_is_compressed(
    tmp_path, name, compress
) -> None:
    data = TEXT.encode()
    path = tmp_path / name
    path.write_bytes(gzip.compress(data, mtime=0) if compress else data)
    assert is_gzip(path) is compress
    with open_binary(path) as handle:
        assert handle.read() == data
    # The mark is dropped, and each line keeps its end, whichever it is.
    with open_text(path) as handle:
        assert handle.readlines() == ["cell\tclone\r\n", "A\t1\r", "B\t2\n", "C\t3"]


@pytest.mark.parametrize("compress", [False, True])
def test_a_stream_such_as_an_archive_member_reads_alike(tmp_path, compress) -> None:
    data = TEXT.encode()
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(
            "member.txt", gzip.compress(data, mtime=0) if compress else data
        )
    with zipfile.ZipFile(archive) as handle, handle.open("member.txt") as raw:
        with open_text(raw) as text:
            assert text.read() == TEXT.removeprefix("\ufeff")
    # A stream that cannot peek is read through a buffer.
    raw_stream = io.RawIOBase()
    raw_stream.readinto = io.BytesIO(gzip.compress(data, mtime=0)).readinto  # type: ignore[method-assign]
    raw_stream.readable = lambda: True  # type: ignore[method-assign]
    with open_binary(raw_stream) as handle:
        assert handle.read() == data


def test_signatures_of_short_and_empty_files(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    assert file_signature(empty) == b""
    assert not is_gzip(empty) and not is_hdf5(empty)
    short = tmp_path / "short"
    short.write_bytes(GZIP_SIGNATURE[:1])
    assert not is_gzip(short)
    hdf5 = tmp_path / "file.h5"
    hdf5.write_bytes(HDF5_SIGNATURE + b"rest")
    assert is_hdf5(hdf5) and file_signature(hdf5) == HDF5_SIGNATURE


def test_text_that_is_not_utf8_raises_as_it_is_read_not_as_it_opens(tmp_path) -> None:
    path = tmp_path / "t.txt"
    path.write_bytes(b"cell\n\xff\xfe\n")
    with open_text(path) as handle, pytest.raises(UnicodeDecodeError):
        handle.read()
