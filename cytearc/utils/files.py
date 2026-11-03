"""Open plain and gzip-compressed files, whatever their names.

Single-cell tools write their tables plain or gzip-compressed, and the files
are often renamed when deposited, so a file's first bytes decide whether it
is compressed, never its name: a gzip file without ``.gz`` and a plain file
named ``.gz`` read alike. Text is UTF-8, and a byte-order mark at its start,
which spreadsheet programs write, is not part of the text. Lines keep their
endings, and a line may end in ``\\n``, ``\\r\\n`` or ``\\r``. A reader that
hands a file to pandas tells pandas its compression by the same rule,
through ``compression_of``.
"""

import gzip
import io
import os
import re
from typing import IO, Any, cast

__all__ = [
    "COMPRESSION_SIGNATURE_BYTES",
    "GZIP_SIGNATURE",
    "HDF5_SIGNATURE",
    "TAR_SIGNATURE_BYTES",
    "compression_of",
    "file_signature",
    "is_gzip",
    "is_hdf5",
    "is_tar",
    "open_binary",
    "open_text",
]

#: The first bytes of a gzip stream.
GZIP_SIGNATURE = b"\x1f\x8b"
#: The first bytes of an HDF5 file.
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
#: The first bytes of a file that ``compression_of`` reads.
COMPRESSION_SIGNATURE_BYTES = 10
#: The first bytes of a file's decompressed text that ``is_tar`` reads.
TAR_SIGNATURE_BYTES = 262
# A bzip2 stream's first bytes: its header with the block size, then the
# magic number of its first block or of the end of an empty stream.
_BZIP2_SIGNATURE = re.compile(rb"BZh[1-9](?:1AY&SY|\x17rE8P\x90)")
# A Zstandard stream's first frame: a frame of data, or a skippable frame,
# whose magic numbers run from 0x184D2A50 to 0x184D2A5F, little-endian.
_ZSTD_SIGNATURE = re.compile(rb"\x28\xb5\x2f\xfd|[\x50-\x5f]\x2a\x4d\x18")
# The first bytes of the other compressions that pandas reads, by the names
# pandas gives them: xz's magic number and the header of a ZIP archive's
# first member.
_SIGNATURES = (
    ("xz", b"\xfd7zXZ\x00"),
    ("zip", b"PK\x03\x04"),
)


def compression_of(head: bytes) -> str | None:
    """Return the compression that a file's first bytes show, as pandas names it.

    Args:
        head: The file's first ``COMPRESSION_SIGNATURE_BYTES`` bytes, fewer
            for a shorter file.

    Returns:
        ``"gzip"``, ``"bz2"``, ``"xz"``, ``"zstd"`` or ``"zip"``, or None
        for a file whose first bytes show none of them, whatever its name.
    """
    if head.startswith(GZIP_SIGNATURE):
        return "gzip"
    if _BZIP2_SIGNATURE.match(head):
        return "bz2"
    if _ZSTD_SIGNATURE.match(head):
        return "zstd"
    return next(
        (name for name, signature in _SIGNATURES if head.startswith(signature)), None
    )


def is_tar(head: bytes) -> bool:
    """Return whether a stream's first bytes, decompressed, begin a tar archive.

    The header of a tar archive's first member holds ``ustar`` at byte 257,
    in POSIX and GNU archives alike.

    Args:
        head: The stream's first ``TAR_SIGNATURE_BYTES`` bytes, decompressed.
    """
    return head[257:TAR_SIGNATURE_BYTES] == b"ustar"


def file_signature(path: str | os.PathLike[str]) -> bytes:
    """Return the first eight bytes of a file, fewer for a shorter file."""
    with open(path, "rb") as handle:
        return handle.read(len(HDF5_SIGNATURE))


def is_gzip(path: str | os.PathLike[str]) -> bool:
    """Return whether a file is gzip-compressed, by its first bytes."""
    return file_signature(path).startswith(GZIP_SIGNATURE)


def is_hdf5(path: str | os.PathLike[str]) -> bool:
    """Return whether a file is an HDF5 file, by its first bytes."""
    return file_signature(path) == HDF5_SIGNATURE


class _GzipStream(gzip.GzipFile):
    """A gzip stream that closes the stream it decompresses when it closes,
    as a plain stream read in its place closes."""

    def __init__(self, source: io.BufferedIOBase) -> None:
        self._source = source
        super().__init__(fileobj=source, mode="rb")

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._source.close()


def open_binary(source: str | os.PathLike[str] | IO[bytes]) -> io.BufferedIOBase:
    """Open a file, or an open binary stream, for reading its bytes.

    A gzip-compressed file is decompressed. Closing the stream returned for
    a stream closes that stream too, compressed or not.

    Args:
        source: A path, or a binary stream such as a member of a ZIP archive.
    """
    if isinstance(source, str | os.PathLike):
        if is_gzip(source):
            return gzip.GzipFile(source, "rb")
        return open(source, "rb")
    stream = cast(
        io.BufferedIOBase,
        source if hasattr(source, "peek") else io.BufferedReader(cast(Any, source)),
    )
    head = cast(Any, stream).peek(len(GZIP_SIGNATURE))[: len(GZIP_SIGNATURE)]
    if head == GZIP_SIGNATURE:
        return _GzipStream(stream)
    return stream


def open_text(source: str | os.PathLike[str] | IO[bytes]) -> io.TextIOWrapper:
    """Open a file, or an open binary stream, for reading UTF-8 text.

    A gzip-compressed file is decompressed, a byte-order mark at the start is
    dropped, and lines keep their endings. Text that is not UTF-8 raises
    ``UnicodeDecodeError`` as it is read. Closing the text of a stream closes
    that stream too.

    Args:
        source: A path, or a binary stream such as a member of a ZIP archive.
    """
    return io.TextIOWrapper(
        cast(Any, open_binary(source)), encoding="utf-8-sig", newline=""
    )
