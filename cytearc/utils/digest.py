"""Digests that identify files, compiled digests and totals of count rows, and SplitMix64.

Importers identify each file they read by the SHA-256 of its bytes, and a
folder of files, such as a Zarr store, by the paths and bytes of the files it
holds: :func:`file_sha256`, :func:`directory_sha256` and :func:`path_sha256`.

SplitMix64's finaliser (Steele, Lea and Flood 2014) mixes a 64-bit integer
into one whose bits each depend on all of its bits. The row digests and the
set hashes of clone calling mix with :func:`mix64`, and every counter-based
generator keys its streams through it or :func:`mix64_array`: Leiden's, the
Monte Carlo draws of cell calling, the rounding draws of ambient RNA removal,
the shuffles of fate coupling's null and the random sets of the association
tests.
"""

import hashlib
import io
import os
from collections.abc import Iterator

import numpy as np

from .numba import njit

_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_ODD = np.uint64(0xD6E8FEB86659FD93)
# The most bytes that a digest reads into its buffer at once, as many as
# hashlib.file_digest reads.
_READ_BYTES = 256 * 1024


def _file_blocks(handle: io.FileIO, size: int) -> Iterator[memoryview]:
    """Yield a file's bytes, read into one buffer until the file ends.

    The buffer holds ``_READ_BYTES``, or ``size``, the bytes the file held
    when opened, when that is fewer, so no read asks for more than the file
    holds. Each block is a view of the buffer, which the next read
    overwrites. A file whose size reads as 0 is empty, or a pipe or a file of
    ``/proc`` that holds bytes the system does not count: a first read of
    one byte tells them apart, so an empty file takes no buffer.
    """
    if size == 0:
        first = handle.read(1)
        if not first:
            return
        yield memoryview(first)
        size = _READ_BYTES
    view = memoryview(bytearray(min(size, _READ_BYTES)))
    while read := handle.readinto(view):
        yield view[:read]


def file_sha256(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 of a file's bytes, as hexadecimal text.

    The file is read into one buffer of 256 KiB, or of the file's size when
    the file is smaller, which is all that the digest holds beside its state.
    """
    digest = hashlib.sha256()
    with open(path, "rb", buffering=0) as handle:
        for block in _file_blocks(handle, os.fstat(handle.fileno()).st_size):
            digest.update(block)
    return digest.hexdigest()


def directory_sha256(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 of a directory's files, such as a Zarr store's.

    Each file adds its path below the directory, in POSIX form, and its bytes,
    each after its length and in the order of the paths, so no two directories
    of other files or paths hash alike. Empty directories add nothing. Each
    file is read as :func:`file_sha256` reads one.

    Raises:
        OSError: If the path is not a directory, or the directory or a folder
            below it cannot be listed, which would otherwise hash as if it
            held no file.
    """
    root = os.fspath(path)

    def refuse(error: OSError) -> None:
        raise error

    files = sorted(
        os.path.relpath(os.path.join(folder, name), root).replace(os.sep, "/")
        for folder, _folders, names in os.walk(root, onerror=refuse)
        for name in names
    )
    digest = hashlib.sha256()
    for relative in files:
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little") + encoded)
        location = os.path.join(root, *relative.split("/"))
        with open(location, "rb", buffering=0) as handle:
            size = os.fstat(handle.fileno()).st_size
            digest.update(size.to_bytes(8, "little"))
            for block in _file_blocks(handle, size):
                digest.update(block)
    return digest.hexdigest()


def path_sha256(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 that identifies a path.

    A directory, such as a Zarr store, is identified by its files, as
    :func:`directory_sha256` hashes them, and any other path by its bytes,
    as :func:`file_sha256` hashes them.
    """
    return directory_sha256(path) if os.path.isdir(path) else file_sha256(path)


@njit(cache=True, inline="always")
def mix64(value: np.uint64) -> np.uint64:
    """Return the SplitMix64 finaliser of a 64-bit unsigned integer."""
    value = (value ^ (value >> np.uint64(30))) * _M1
    value = (value ^ (value >> np.uint64(27))) * _M2
    return value ^ (value >> np.uint64(31))


def mix64_array(values: np.ndarray) -> np.ndarray:
    """Return the SplitMix64 finaliser of each unsigned 64-bit integer.

    Each value equals :func:`mix64` of it; array products wrap around without
    warnings, as the compiled products do.
    """
    mixed = (values ^ (values >> np.uint64(30))) * _M1
    mixed = (mixed ^ (mixed >> np.uint64(27))) * _M2
    finalised: np.ndarray = mixed ^ (mixed >> np.uint64(31))
    return finalised


@njit(cache=True, nogil=True)
def summarize_rows(
    values: np.ndarray,
    bits: np.ndarray,
    digests: np.ndarray,
    row_sums: np.ndarray,
    row_positive: np.ndarray,
    column_positive: np.ndarray,
) -> None:
    """Digest each row's nonzero column-value pairs and total its values.

    ``bits`` is the unsigned-integer view of ``values``. Zero entries are
    skipped, so digests depend only on each row's sparse content. Positive
    entries are counted per row and added to ``column_positive``.

    Raises:
        ValueError: If a value is NaN or infinite. Count matrices hold finite
            values; the check compiles away for integer values.
    """
    for row in range(values.shape[0]):
        first = np.uint64(0x243F6A8885A308D3)
        second = np.uint64(0x13198A2E03707344)
        nonzero = np.uint64(0)
        positive = 0
        total = 0.0
        for column in range(values.shape[1]):
            value = values[row, column]
            if value != 0:
                if not np.isfinite(value):
                    raise ValueError(
                        "Count matrices must hold finite values; found NaN or infinity"
                    )
                key = mix64(
                    ((np.uint64(column) + np.uint64(1)) * _GOLDEN)
                    ^ np.uint64(bits[row, column])
                )
                first = mix64(first ^ key)
                second = (second + key) * _ODD
                nonzero += np.uint64(1)
                total += value
                if value > 0:
                    positive += 1
                    column_positive[column] += 1
        digests[row, 0] = mix64(first ^ nonzero)
        digests[row, 1] = mix64(second ^ (nonzero * _GOLDEN))
        row_sums[row] = total
        row_positive[row] = positive
