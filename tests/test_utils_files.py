"""The one opener of plain and gzip-compressed files.

A file's first bytes, never its name, say whether it is gzip-compressed, and
text is UTF-8 with a byte-order mark at its start dropped.
"""

import gzip
import io
import zipfile

import pytest

from cytearc.utils.files import (
    COMPRESSION_SIGNATURE_BYTES,
    GZIP_SIGNATURE,
    HDF5_SIGNATURE,
    TAR_SIGNATURE_BYTES,
    compression_of,
    file_signature,
    is_gzip,
    is_hdf5,
    is_tar,
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


@pytest.mark.parametrize("compress", [False, True])
def test_closing_what_a_stream_opens_closes_the_stream_compressed_or_not(
    compress,
) -> None:
    data = TEXT.encode()
    payload = gzip.compress(data, mtime=0) if compress else data
    stream = io.BytesIO(payload)
    with open_text(stream) as text:
        assert text.read() == TEXT.removeprefix("\ufeff")
    assert stream.closed
    stream = io.BytesIO(payload)
    with open_binary(stream) as handle:
        assert handle.read() == data
    assert stream.closed


def test_the_first_bytes_name_each_compression_that_pandas_reads() -> None:
    import bz2
    import lzma

    text = b"cell,clone\nA,1\n"
    heads = {
        "gzip": gzip.compress(text, mtime=0),
        "bz2": bz2.compress(text),
        "xz": lzma.compress(text),
        # Zstandard's magic number and frame header, as zstd writes them.
        "zstd": b"\x28\xb5\x2f\xfd\x24\x10\x81\x00\x00\x63",
    }
    # A skippable frame, of magic numbers 0x184D2A50 to 0x184D2A5F, may come
    # first.
    for first in (0x50, 0x5F):
        skippable = bytes([first]) + b"\x2a\x4d\x18\x04\x00\x00\x00meta"
        assert compression_of(skippable[:COMPRESSION_SIGNATURE_BYTES]) == "zstd"
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("counts.csv", text)
    heads["zip"] = archive.getvalue()
    for name, data in heads.items():
        assert compression_of(data[:COMPRESSION_SIGNATURE_BYTES]) == name
    # An empty bzip2 stream ends at once.
    assert compression_of(bz2.compress(b"")[:COMPRESSION_SIGNATURE_BYTES]) == "bz2"
    # Text that starts as a signature does, or is cut short, is plain.
    for head in (b"BZh9 reads", b"BZh", b"PK\x03", b"\x1f", b"", text):
        assert compression_of(head[:COMPRESSION_SIGNATURE_BYTES]) is None


def test_a_tar_archive_is_known_by_its_first_members_header() -> None:
    import tarfile

    for archive_format in (
        tarfile.USTAR_FORMAT,
        tarfile.GNU_FORMAT,
        tarfile.PAX_FORMAT,
    ):
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w", format=archive_format) as handle:
            member = tarfile.TarInfo("counts.csv")
            member.size = 4
            handle.addfile(member, io.BytesIO(b"a,b\n"))
        assert is_tar(archive.getvalue()[:TAR_SIGNATURE_BYTES])
    # Text, and text cut short of the header's mark, is no tar archive.
    assert not is_tar(b"cell,clone\n" * 30)
    assert not is_tar(b" " * 257 + b"usta")
