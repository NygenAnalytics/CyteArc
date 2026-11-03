"""Open plain and gzip-compressed files, whatever their names.

Single-cell tools write their tables plain or gzip-compressed, and the files
are often renamed when deposited, so a file's first bytes decide whether it
is compressed, never its name: a gzip file without ``.gz`` and a plain file
named ``.gz`` read alike. Text is UTF-8, and a byte-order mark at its start,
which spreadsheet programs write, is not part of the text. Lines keep their
endings, and a line may end in ``\\n``, ``\\r\\n`` or ``\\r``.
"""

import gzip
import io
import os
from typing import IO, Any, cast

__all__ = [
    "GZIP_SIGNATURE",
    "HDF5_SIGNATURE",
    "file_signature",
    "is_gzip",
    "is_hdf5",
    "open_binary",
    "open_text",
]

#: The first bytes of a gzip stream.
GZIP_SIGNATURE = b"\x1f\x8b"
#: The first bytes of an HDF5 file.
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


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
