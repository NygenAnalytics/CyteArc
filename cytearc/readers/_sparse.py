"""Write sparse rows to local disk in any order and read them back by destination row.

Converting a CSC matrix to rows, shuffling cells and restoring a source order
all receive rows in one order and need them in another. A
:class:`SparseRowStore` takes rows tagged with their destination row and
writes them to buckets, bands of consecutive destination rows. A read loads
one bucket and returns canonical CSR windows of destination rows.

Rows arrive in one of three forms. Whole rows (``append_rows``) arrive once
each, as in a shuffle, and a read only puts them in destination order. Loose
entries (``append_entries``) may give a row in pieces over many appends, as a
CSC conversion does, and a read joins each row's pieces in the order they
were appended, sorts its columns and sums its duplicate coordinates, or keeps
them sorted for a reader that sums them later. Ragged rows
(``append_ragged_rows``) are whole rows of bytes without columns, such as the
rows of a text file, and a read (``iter_ragged_rows``) returns each row's
bytes in destination order.

Several processes may write one store. The owner creates the directory and
writes part 0; each other writer opens the directory with
:meth:`SparseRowStore.open` and its own part number and calls
:meth:`~SparseRowStore.finish`. The owner then calls
:meth:`~SparseRowStore.finalize`, which also splits every bucket that would take
more than the split size to read, and any process may open the finalized store
by its path to read it.

On disk, every bucket is a directory holding one file per writer. A file is a
sequence of blocks, each a header and four Blosc payloads: the row of each
piece within its bucket, the entries of each piece, their column indices and
their values. Payloads are compressed with zstd at level 1 and byte shuffle,
except those of ragged rows, which are stored uncompressed, since compressing
text took half or more of the time that its reorder added; Blosc decodes
either from the frame's header. Values use the narrowest unsigned dtype that
holds every value of the block exactly, and the source dtype otherwise; column
indices are stored as gaps within each piece when they never decrease, and a
block of ragged rows stores none. The header records these codes, the payload
sizes and a CRC-32 of the payloads, because Blosc decodes corrupted and
truncated buffers without an error.
"""

import io
import json
import math
import os
import shutil
import struct
import tempfile
import zlib
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Self

import numpy as np
from numcodecs import Blosc, blosc
from scipy.sparse import coo_matrix, csr_matrix

from ..utils.arrays import canonicalize_sparse, checked_sparse_cast, cumulative_nnz
from ..utils.memory_budget import MemoryBudgetError

__all__ = ["ROW_BYTES", "SparseRowStore", "encoded_entry_bytes"]

_CODEC = Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE)
# Ragged rows are stored as they are, in Blosc frames that _CODEC decodes.
_RAGGED_CODEC = Blosc(cname="lz4", clevel=0, shuffle=Blosc.NOSHUFFLE)
_UNSIGNED = tuple(np.dtype(name) for name in ("u1", "u2", "u4", "u8"))
# The value code of a block whose values keep the source dtype.
_SOURCE_VALUES = 255
# The index code's flag for column indices stored as gaps within each piece.
_GAPS = 0x10
# The index code's flag for a block of ragged rows, whose values have no
# column indices.
_RAGGED = 0x20
# The block flag for pieces whose column indices strictly increase.
_CANONICAL = 0x01
_MAGIC = b"CRB1"
# Magic, flags, the row, length, index and value codes, the piece and entry
# counts, the four payload sizes and the CRC-32 of the header before it and of
# the payloads.
_HEADER = struct.Struct("<4sBBBBB3xQQQQQQI")
_CHECKSUM = struct.Struct("<I")
# The most that a block holds beside its payloads: its header and the
# header of the Blosc frame of each of its four payloads.
_BLOCK_OVERHEAD = _HEADER.size + 4 * int(blosc.MAX_OVERHEAD)
_FORMAT = 1

#: Bytes per destination row that a store holds: the entries appended to each
#: row, the row pointers built from them, the mark of each whole row and the
#: bucket boundaries.
ROW_BYTES = 32
# The most pending bytes that one block takes.
_BLOCK_BYTES = 4 * 1024 * 1024
# The read size for which from_entries sizes its buckets.
_BUCKET_TARGET_BYTES = 128 * 1024 * 1024
# The most bytes that from_entries keeps pending before it writes blocks.
_PENDING_LIMIT_BYTES = 256 * 1024 * 1024
# The most buckets that from_entries makes when it cannot estimate the entries.
_MAX_UNKNOWN_BUCKETS = 4096
# Bytes per destination row that a writer holds: the entries appended to
# each row and the mark of each whole row.
_WRITER_ROW_BYTES = 9
# Bytes per bucket that a writer holds: its first row, file bytes, pieces and
# whether its pieces are canonical.
_WRITER_BUCKET_BYTES = 25
# The free disk that a measured spill estimate asks for beyond the bytes it
# measured, since sample rows compress unlike the rest.
_ESTIMATE_MARGIN = 1.25

type _Piece = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _capitalized(text: str) -> str:
    """Return ``text`` as a sentence starts with it."""
    return text[:1].upper() + text[1:]


def _unsigned_code(top: int) -> int:
    """Return the code of the narrowest unsigned dtype that holds ``top``."""
    for code, dtype in enumerate(_UNSIGNED[:-1]):
        if top <= np.iinfo(dtype).max:
            return code
    return len(_UNSIGNED) - 1


def _value_code(values: np.ndarray) -> int:
    """Return the code of the narrowest unsigned dtype that holds every value exactly.

    Booleans and non-negative integers always narrow. Floating-point values
    narrow, up to uint32, when every value is a finite, non-negative whole
    number and none is a negative zero. Other values keep their dtype.
    """
    if values.size == 0:
        return 0
    kind = values.dtype.kind
    if kind == "b":
        return 0
    if kind in "iu":
        if kind == "i" and int(values.min()) < 0:
            return _SOURCE_VALUES
        return _unsigned_code(int(values.max()))
    if kind != "f":
        return _SOURCE_VALUES
    low, high = float(values.min()), float(values.max())
    # NaN compares false, so it never narrows.
    if not (low >= 0 and high <= np.iinfo(np.uint32).max):
        return _SOURCE_VALUES
    if np.signbit(values).any() or not np.array_equal(values, np.trunc(values)):
        return _SOURCE_VALUES
    return _unsigned_code(int(high))


def _stable_order(keys: np.ndarray) -> np.ndarray:
    """Return the stable sorting order of non-negative int64 ``keys``.

    NumPy sorts 16-bit keys stably by radix in linear time, so keys below
    2**32 sort by their low and then their high 16 bits.
    """
    top = int(keys.max(initial=0))
    if top >= 2**32:
        return np.argsort(keys, kind="stable")
    # The cast to uint16 keeps the low 16 bits.
    order = np.argsort(keys.astype(np.uint16), kind="stable")
    if top >= 2**16:
        high = (keys[order] >> 16).astype(np.uint16)
        order = order[np.argsort(high, kind="stable")]
    return order


def _piece_starts(lengths: np.ndarray) -> np.ndarray:
    starts = np.zeros(lengths.size, dtype=np.int64)
    np.cumsum(lengths[:-1], out=starts[1:])
    return starts


def _encode_payload(values: np.ndarray, codec: Blosc = _CODEC) -> bytes:
    # Blosc refuses to decode the buffer of an empty array.
    if values.size == 0:
        return b""
    return bytes(codec.encode(np.ascontiguousarray(values)))


def _encode_block(
    pieces: list[_Piece], *, n_columns: int, bucket_rows: int, ragged: bool = False
) -> tuple[bytes, bool]:
    """Encode pieces of rows as one block; return it and whether every piece is canonical.

    Each piece holds the rows of its pieces within the bucket, their entries,
    and the column indices and values of those entries in turn; pieces of
    ``ragged`` rows hold no column indices, and their block is stored
    uncompressed.
    """
    rows, lengths, indices, values = (
        np.concatenate([piece[k] for piece in pieces]) for k in range(4)
    )
    row_code = _unsigned_code(max(0, bucket_rows - 1))
    length_code = _unsigned_code(int(lengths.max(initial=0)))
    column_code = 0 if ragged else _unsigned_code(max(0, n_columns - 1))
    flags = _CANONICAL
    index_code = _RAGGED if ragged else column_code
    stored_indices: np.ndarray = indices
    if indices.size:
        gaps = np.empty(indices.size, dtype=np.int64)
        gaps[0] = indices[0]
        np.subtract(indices[1:], indices[:-1], out=gaps[1:], dtype=np.int64)
        firsts = _piece_starts(lengths)[lengths > 0]
        gaps[firsts] = indices[firsts]
        if int(gaps.min()) >= 0:
            index_code |= _GAPS
            stored_indices = gaps
            # A zero gap after a piece's first entry repeats a column.
            if np.count_nonzero(gaps == 0) > np.count_nonzero(indices[firsts] == 0):
                flags = 0
        else:
            flags = 0
    value_code = _value_code(values)
    stored_values = (
        values
        if value_code == _SOURCE_VALUES
        else values.astype(_UNSIGNED[value_code], copy=False)
    )
    codec = _RAGGED_CODEC if ragged else _CODEC
    payloads = (
        _encode_payload(rows.astype(_UNSIGNED[row_code], copy=False), codec),
        _encode_payload(lengths.astype(_UNSIGNED[length_code], copy=False), codec),
        _encode_payload(
            stored_indices.astype(_UNSIGNED[column_code], copy=False), codec
        ),
        _encode_payload(stored_values, codec),
    )
    header = _HEADER.pack(
        _MAGIC,
        flags,
        row_code,
        length_code,
        index_code,
        value_code,
        int(rows.size),
        # A piece of a sparse row has one column index for each value.
        int(values.size),
        *(len(payload) for payload in payloads),
        0,
    )[: -_CHECKSUM.size]
    # One join, so that encoding holds three copies of an uncompressed block:
    # the pieces joined, their payloads and the block.
    return (
        b"".join((header, _CHECKSUM.pack(_checksum(header, payloads)), *payloads)),
        bool(flags & _CANONICAL),
    )


def _checksum(header: bytes, payloads: Iterable[bytes]) -> int:
    """Return the CRC-32 of a block's header, without its checksum, and payloads."""
    checksum = zlib.crc32(header)
    for payload in payloads:
        checksum = zlib.crc32(payload, checksum)
    return checksum


def _archive_bytes(arrays: dict[str, tuple[Any, tuple[int, ...]]]) -> int:
    """Return the bytes of an ``np.savez`` archive of arrays of these dtypes and shapes.

    Beside each array's data, the archive holds a header and an entry that
    take the same bytes for any number of rows, up to archives of 4 GiB,
    so an archive of the arrays without rows measures them.
    """
    rowless: dict[str, Any] = {
        name: np.zeros((0, *shape[1:]), dtype=dtype)
        for name, (dtype, shape) in arrays.items()
    }
    headers = io.BytesIO()
    np.savez(headers, **rowless)
    data = sum(
        int(np.dtype(dtype).itemsize) * math.prod(shape)
        for dtype, shape in arrays.values()
    )
    return headers.tell() + data


def encoded_entry_bytes(matrix: csr_matrix) -> float:
    """Return the bytes per entry that whole rows like ``matrix``'s take on disk.

    The rows are encoded as one block of a bucket, as a store writes them,
    so the result reflects how their values narrow and their columns and
    values compress. Pass a sample of the rows that a store will hold, of
    about a million entries, as ``entry_bytes`` of :class:`SparseRowStore`.

    Raises:
        TypeError: If ``matrix`` is not CSR.
    """
    if getattr(matrix, "format", None) != "csr":
        raise TypeError(
            f"encoded_entry_bytes takes a CSR matrix, not {type(matrix).__name__}"
        )
    n_rows, n_columns = matrix.shape
    lengths = np.diff(matrix.indptr).astype(np.int64)
    block, _canonical = _encode_block(
        [
            (
                np.arange(n_rows, dtype=np.int64),
                lengths,
                np.asarray(matrix.indices),
                np.asarray(matrix.data),
            )
        ],
        n_columns=n_columns,
        bucket_rows=max(1, n_rows),
    )
    return len(block) / max(1, int(lengths.sum()))


@dataclass(frozen=True, slots=True)
class _Block:
    """The pieces of one block: rows within the bucket, entries, indices and values."""

    rows: np.ndarray
    lengths: np.ndarray
    indices: np.ndarray
    values: np.ndarray


def _decode_payload(
    payload: bytes, dtype: np.dtype[Any], count: int, where: str
) -> np.ndarray:
    decoded = (
        np.frombuffer(_CODEC.decode(payload), dtype=dtype)
        if payload
        else np.empty(0, dtype=dtype)
    )
    if decoded.size != count:
        raise RuntimeError(
            f"{where} decodes to {decoded.size} values where its header records "
            f"{count}; the file is corrupted"
        )
    return decoded


def _read_blocks(
    stream: BinaryIO,
    *,
    size: int,
    path: Path,
    source_dtype: np.dtype[Any],
    n_columns: int,
    bucket_rows: int,
    ragged: bool = False,
) -> Iterator[_Block]:
    """Yield the blocks of one bucket file in turn, checking every size and checksum.

    Blocks of ``ragged`` rows hold no column indices, and their ``indices``
    are empty.
    """
    number = position = 0
    while header := stream.read(_HEADER.size):
        where = f"Block {number} of {path}"
        if len(header) < _HEADER.size:
            raise RuntimeError(f"{where} is truncated: its header is incomplete")
        (
            magic,
            _flags,
            row_code,
            length_code,
            index_code,
            value_code,
            n_pieces,
            n_entries,
            *sizes,
            checksum,
        ) = _HEADER.unpack(header)
        if magic != _MAGIC:
            raise RuntimeError(f"{where} does not start a block; the file is corrupted")
        position += _HEADER.size
        if sum(sizes) > size - position:
            raise RuntimeError(
                f"{where} is truncated: its payloads take {sum(sizes)} bytes and "
                f"{size - position} remain"
            )
        payloads = [stream.read(length) for length in sizes]
        position += sum(sizes)
        if _checksum(header[: -_CHECKSUM.size], payloads) != checksum:
            raise RuntimeError(f"{where} fails its checksum; the file is corrupted")
        if max(row_code, length_code, index_code & ~(_GAPS | _RAGGED)) >= len(
            _UNSIGNED
        ) or (value_code >= len(_UNSIGNED) and value_code != _SOURCE_VALUES):
            raise RuntimeError(f"{where} names an unknown dtype; the file is corrupted")
        if bool(index_code & _RAGGED) != ragged:
            raise RuntimeError(
                f"{where} holds "
                + ("ragged" if index_code & _RAGGED else "sparse")
                + " rows in a store of "
                + ("ragged" if ragged else "sparse")
                + " rows; the file is corrupted"
            )
        rows = _decode_payload(payloads[0], _UNSIGNED[row_code], n_pieces, where)
        lengths = _decode_payload(
            payloads[1], _UNSIGNED[length_code], n_pieces, where
        ).astype(np.int64)
        stored = _decode_payload(
            payloads[2],
            _UNSIGNED[index_code & ~(_GAPS | _RAGGED)],
            0 if ragged else n_entries,
            where,
        )
        values = _decode_payload(
            payloads[3],
            source_dtype if value_code == _SOURCE_VALUES else _UNSIGNED[value_code],
            n_entries,
            where,
        )
        del payloads
        if int(lengths.sum()) != n_entries:
            raise RuntimeError(
                f"{where} holds pieces of {int(lengths.sum())} entries where its "
                f"header records {n_entries}; the file is corrupted"
            )
        if rows.size and int(rows.max()) >= bucket_rows:
            raise RuntimeError(
                f"{where} holds a row outside its bucket; the file is corrupted"
            )
        if index_code & _GAPS and stored.size:
            indices = np.cumsum(stored, dtype=np.int64)
            starts = _piece_starts(lengths)
            nonempty = lengths > 0
            offsets = np.zeros(lengths.size, dtype=np.int64)
            offsets[nonempty] = indices[starts[nonempty]] - stored[starts[nonempty]]
            indices -= np.repeat(offsets, lengths)
            del starts, offsets
        else:
            indices = stored.astype(np.int64)
        # Unsigned gaps within each piece add up to no negative index.
        if indices.size and int(indices.max()) >= n_columns:
            raise RuntimeError(
                f"{where} holds a column outside the matrix; the file is corrupted"
            )
        yield _Block(rows.astype(np.int64), lengths, indices, values)
        number += 1


def _entry_bytes(itemsize: int, sum_duplicates: bool, ragged: bool = False) -> int:
    """Return what loading a bucket holds for each entry.

    Its int64 column index and value as they are decoded, with one block's
    payloads and gaps beside them, then the pieces' CSR and its gather into
    row order; summing duplicates builds int64 coordinates and sums in at
    least 64 bits. A byte of a ragged row is held as decoded, as joined and
    as gathered into row order. ``itemsize`` is the wider of the source and
    read dtypes.
    """
    if ragged:
        return 3 * itemsize
    per_entry = 3 * itemsize + 32
    if sum_duplicates:
        per_entry += 2 * max(itemsize, 8) + 24
    return per_entry


def _bucket_bytes(
    entries: int,
    pieces: int,
    rows: int,
    itemsize: int,
    sum_duplicates: bool,
    ragged: bool = False,
) -> int:
    """Return the most that loading and assembling one bucket holds.

    Besides its entries, each piece holds its row and length as they are
    decoded, joined and sorted, and each row its pointers and counts. A
    piece of a ragged row also holds its start, end and flag, as found and
    as gathered, while the bytes are gathered into row order.
    """
    per_entry = _entry_bytes(itemsize, sum_duplicates, ragged)
    return int(entries * per_entry + pieces * _piece_bytes(ragged) + (rows + 1) * 32)


def _piece_bytes(ragged: bool) -> int:
    """Return what loading a bucket holds for each piece, as :func:`_bucket_bytes` counts it."""
    return 120 if ragged else 48


class _PendingBuckets:
    """Pieces waiting for their bucket's file, written in blocks beyond a limit.

    ``limit`` is the most bytes that pending pieces take; beyond it the bucket
    with the most pending bytes is written. A block takes pending pieces up
    to ``block_bytes``, and at least one piece.
    """

    def __init__(
        self,
        n_buckets: int,
        limit: int,
        block_bytes: int,
        path: Callable[[int], Path],
        encode: Callable[[list[_Piece]], tuple[bytes, bool]],
        check_disk: Callable[[int], None],
    ) -> None:
        self._limit = limit
        self._blockBytes = block_bytes
        self._path = path
        self._encode = encode
        self._checkDisk = check_disk
        self._pending: dict[int, list[_Piece]] = {}
        self._pendingBytes: dict[int, int] = {}
        self._buffered = 0
        self.fileBytes = np.zeros(n_buckets, dtype=np.int64)
        self.pieces = np.zeros(n_buckets, dtype=np.int64)
        self.mixed = np.zeros(n_buckets, dtype=bool)

    def add(self, bucket: int, piece: _Piece) -> None:
        size = sum(int(array.nbytes) for array in piece)
        self._pending.setdefault(bucket, []).append(piece)
        self._pendingBytes[bucket] = self._pendingBytes.get(bucket, 0) + size
        self._buffered += size
        while self._buffered > self._limit and self._pending:
            self.write(max(self._pendingBytes, key=self._pendingBytes.__getitem__))

    def write(self, bucket: int) -> None:
        """Encode a bucket's pending pieces in blocks and append them to its file."""
        pieces = self._pending.pop(bucket)
        self._buffered -= self._pendingBytes.pop(bucket)
        path = self._path(bucket)
        while pieces:
            taken = group = 0
            while group < len(pieces):
                size = sum(int(array.nbytes) for array in pieces[group])
                if group and taken + size > self._blockBytes:
                    break
                taken += size
                group += 1
            block, canonical = self._encode(pieces[:group])
            self._checkDisk(len(block))
            path.parent.mkdir(exist_ok=True)
            with path.open("ab") as stream:
                stream.write(block)
            self.fileBytes[bucket] += len(block)
            self.pieces[bucket] += sum(int(piece[0].size) for piece in pieces[:group])
            self.mixed[bucket] |= not canonical
            del pieces[:group], block

    def write_all(self) -> None:
        for bucket in list(self._pending):
            self.write(bucket)


class SparseRowStore:
    """Disk-backed sparse rows that arrive in any order, read by destination row.

    The owner creates a store for a ``shape`` of destination rows and columns,
    with buckets of ``bucket_rows`` rows. Rows then arrive as whole rows, as
    loose entries or as ragged rows, from this process as part 0 and from
    writers that other processes open by part number; :meth:`finalize` joins
    every part, and :meth:`read` and :meth:`iter_rows` return canonical CSR
    rows of ``dtype``, or :meth:`iter_ragged_rows` the bytes of ragged rows.
    Closing the owner removes its directory.

    Args:
        shape: Destination rows and columns; for ragged rows, the most bytes
            that one row holds.
        dtype: The dtype that reads return; uint8 for ragged rows.
        operation: What the store does, as its errors name it inside a
            sentence, such as ``"shuffled Matrix Market import"``.
        bucket_rows: Destination rows of each bucket as rows arrive.
        whole_rows: Whether rows arrive whole, each once, through
            :meth:`append_rows`; otherwise entries arrive through
            :meth:`append_entries`.
        ragged: Whether whole rows arrive as bytes without columns, through
            :meth:`append_ragged_rows`, and read back through
            :meth:`iter_ragged_rows`.
        buffer_bytes: The most that appending holds beside the rows passed to
            it: pending pieces and the encoding of one block. Every writer,
            and :meth:`finalize`, holds up to this much.
        bucket_bytes: The most that reading one bucket holds; :meth:`finalize`
            refuses a row that needs more.
        split_bytes: The read size above which :meth:`finalize` splits a
            bucket into bands of rows, each read within it unless one row
            needs more; ``bucket_bytes`` by default, and at most that.
        entries: The entries expected, from which the free disk that the
            store's files may need is checked before anything is written, or
            None. Beside the entries, the check counts a block for each of
            ``parts`` parts in each bucket, the settings, the record of each
            part and the finalized record.
        entry_bytes: The bytes that each entry is expected to take on disk,
            as :func:`encoded_entry_bytes` measures them on sample rows; the
            check then asks for a quarter more than ``entries`` take at that
            rate, and 16 bytes for the place and length of each row. None
            bounds each entry by its bytes before compression.
        parts: The parts that writers will finish, the owner's part 0 among
            them, whose records the check of the free disk counts; 1 when
            only the owner writes.
        source_dtype: The dtype in which values arrive and in which a row's
            duplicate coordinates are summed; ``dtype`` by default.
        sum_duplicates: Whether reads sum a row's duplicate coordinates, or
            keep them, their columns sorted, for a reader that sums them.
        max_nnz: The most entries that the rows may hold.
        nnz_error: The error raised beyond ``max_nnz``.
        temp_dir: The directory in which the store's directory is created.

    Raises:
        MemoryBudgetError: If ``bucket_bytes`` cannot hold one entry.
        OSError: If the disk lacks the bytes that the store's files may need.
        ValueError: If ``ragged`` rows do not arrive whole or are not uint8,
            or ``parts`` is below 1.
    """

    # The temporary directory that the owner removes when it closes.
    _directory: tempfile.TemporaryDirectory[str] | None
    _nnzError: type[ValueError]

    def __init__(
        self,
        shape: tuple[int, int],
        dtype: Any,
        *,
        operation: str,
        bucket_rows: int,
        whole_rows: bool,
        buffer_bytes: int,
        bucket_bytes: int,
        split_bytes: int | None = None,
        entries: int | None = None,
        entry_bytes: float | None = None,
        parts: int = 1,
        source_dtype: Any | None = None,
        sum_duplicates: bool = True,
        max_nnz: int = np.iinfo(np.int64).max,
        nnz_error: type[ValueError] = ValueError,
        temp_dir: str | Path | None = None,
        ragged: bool = False,
    ) -> None:
        n_rows, n_columns = (int(value) for value in shape)
        if n_rows < 0 or n_columns < 0:
            raise ValueError(
                f"A sparse row store's shape must not be negative, not {shape}"
            )
        if ragged and not (
            whole_rows
            and np.dtype(dtype) == np.uint8
            and (source_dtype is None or np.dtype(source_dtype) == np.uint8)
        ):
            raise ValueError(
                "Ragged rows are bytes that arrive whole; pass whole_rows=True and "
                "dtype uint8 with ragged=True"
            )
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError(
                "operation names what a sparse row store does, such as "
                f"'shuffled Matrix Market import'; found {operation!r}"
            )
        if int(bucket_rows) < 1:
            raise ValueError(f"bucket_rows must be at least 1, not {bucket_rows}")
        if int(parts) < 1:
            raise ValueError(f"parts must be at least 1, not {parts}")
        for name, value in (
            ("buffer_bytes", buffer_bytes),
            ("bucket_bytes", bucket_bytes),
        ):
            if int(value) < 0:
                raise ValueError(f"{name} must not be negative, not {value}")
        split = int(bucket_bytes if split_bytes is None else split_bytes)
        if not 0 <= split <= int(bucket_bytes):
            raise ValueError(
                f"split_bytes must lie between 0 and bucket_bytes={bucket_bytes}, "
                f"not {split_bytes}"
            )
        self._configure(
            {
                "format": _FORMAT,
                "operation": operation,
                "nRows": n_rows,
                "nColumns": n_columns,
                "dtype": np.dtype(dtype).newbyteorder("=").str,
                "sourceDtype": np.dtype(dtype if source_dtype is None else source_dtype)
                .newbyteorder("=")
                .str,
                "bucketRows": int(bucket_rows),
                "wholeRows": bool(whole_rows),
                "ragged": bool(ragged),
                "sumDuplicates": bool(sum_duplicates),
                "bucketBytes": int(bucket_bytes),
                "splitBytes": split,
                "maxNnz": int(max_nnz),
            }
        )
        least = self._bucket_cost(1, 1, 1)
        if self._bucketLimit < least:
            raise MemoryBudgetError(
                f"{_capitalized(self.operation)} exceeds the memory limit. Reading one entry "
                f"needs at least {least} bytes.",
                needed_bytes=least,
                budget_bytes=self._bucketLimit,
                item=f"reading one entry of the {self.operation}",
            )
        self._nnzError = nnz_error
        self._directory = tempfile.TemporaryDirectory(
            prefix="cytearc-sparse-", dir=temp_dir
        )
        self.directory = Path(self._directory.name)
        try:
            if entries is not None:
                self._check_disk(
                    self._spill_bound(int(entries), entry_bytes, int(parts))
                )
            (self.directory / "parts").mkdir()
            with (self.directory / "store.json").open("w") as stream:
                json.dump(self._settings, stream)
            self._start_writing(0, int(buffer_bytes))
        except BaseException:
            self._directory.cleanup()
            raise

    def _configure(self, settings: dict[str, Any]) -> None:
        if settings.get("format") != _FORMAT:
            raise RuntimeError(
                f"Sparse row store format {settings.get('format')} is not "
                f"format {_FORMAT}"
            )
        self._settings = settings
        #: What the store does, as its errors name it.
        self.operation = str(settings["operation"])
        self.shape = (int(settings["nRows"]), int(settings["nColumns"]))
        self.dtype: np.dtype[Any] = np.dtype(settings["dtype"])
        self._sourceDtype: np.dtype[Any] = np.dtype(settings["sourceDtype"])
        self._bucketRows = int(settings["bucketRows"])
        self._wholeRows = bool(settings["wholeRows"])
        self._ragged = bool(settings["ragged"])
        self._sumDuplicates = bool(settings["sumDuplicates"])
        self._bucketLimit = int(settings["bucketBytes"])
        self._splitLimit = int(settings["splitBytes"])
        self._maxNnz = int(settings["maxNnz"])
        self._nnzError = ValueError
        self._directory = None
        self._part: int | None = None
        self._bufferBytes = 0
        self._finalized = False
        self.indptr = np.zeros(0, dtype=np.int64)
        # The first row of every bucket and the matrix's row count, after it.
        self.bucket_starts = np.append(
            np.arange(0, self.shape[0], self._bucketRows, dtype=np.int64),
            self.shape[0],
        )
        self._cache: tuple[int, csr_matrix] | None = None
        # The most that loading one bucket holds, and one assembled bucket.
        self._readBytes: tuple[int, int] | None = None

    def _bucket_cost(self, entries: int, pieces: int, rows: int) -> int:
        itemsize = max(self._sourceDtype.itemsize, self.dtype.itemsize)
        return _bucket_bytes(
            entries, pieces, rows, itemsize, self._sumDuplicates, self._ragged
        )

    def _folder(self, first_row: int) -> Path:
        return self.directory / f"rows-{first_row:012d}"

    # Writing.

    def _start_writing(self, part: int, buffer_bytes: int) -> None:
        if int(part) < 0:
            raise ValueError(f"A writer's part must be at least 0, not {part}")
        marker = self.directory / "parts" / f"part-{int(part):04d}.open"
        try:
            marker.touch(exist_ok=False)
        except FileExistsError as error:
            raise RuntimeError(
                f"Part {part} of the sparse row store in {self.directory} is "
                "already being written"
            ) from error
        self._part = int(part)
        self._bufferBytes = max(0, int(buffer_bytes))
        self._buckets = self._pending_buckets(
            self.directory, self.bucket_starts, self._part
        )
        self._rowEntries = np.zeros(self.shape[0], dtype=np.int64)
        self._rowPieces = (
            np.zeros(self.shape[0], dtype=np.uint8) if self._wholeRows else None
        )
        self._appended = 0

    def _pending_buckets(
        self, folder: Path, starts: np.ndarray, part: int
    ) -> _PendingBuckets:
        """Return buffers for buckets with first rows ``starts``, written under ``folder``."""
        # Pending pieces take a quarter of the buffer: encoding a block holds
        # about three more copies of it.
        limit = self._bufferBytes // 4
        return _PendingBuckets(
            starts.size - 1,
            limit,
            max(1, min(_BLOCK_BYTES, limit)),
            lambda bucket: (
                folder / f"rows-{int(starts[bucket]):012d}" / f"part-{part:04d}.bin"
            ),
            lambda pieces: _encode_block(
                pieces,
                n_columns=self.shape[1],
                bucket_rows=self._bucketRows,
                ragged=self._ragged,
            ),
            self._check_disk,
        )

    def _require_writing(self, form: str) -> None:
        if self._part is None:
            raise RuntimeError(
                "This sparse row store takes no more rows: its part was finished, "
                "or it was opened for reading"
            )
        expected, kind = self._form
        if form != expected:
            raise TypeError(f"This sparse row store takes {kind}; call {expected}")

    @property
    def _form(self) -> tuple[str, str]:
        """The method that appends this store's rows, and what it takes."""
        if self._ragged:
            return "append_ragged_rows", "ragged rows"
        if self._wholeRows:
            return "append_rows", "whole rows"
        return "append_entries", "loose entries"

    def _count_entries(self, entries: int) -> None:
        self._appended += entries
        if self._appended > self._maxNnz:
            raise self._nnzError(
                f"{_capitalized(self.operation)} exceeds maxNnz={self._maxNnz}"
            )

    def _check_columns(self, indices: np.ndarray) -> None:
        if indices.size and (
            int(indices.min()) < 0 or int(indices.max()) >= self.shape[1]
        ):
            raise IndexError(
                f"Column indices must lie in [0, {self.shape[1]}); found "
                f"{int(indices.min())} to {int(indices.max())}"
            )

    def _column_indices(self, indices: np.ndarray) -> np.ndarray:
        """Return column indices as int32 while pending, when the columns allow."""
        dtype = np.int32 if self.shape[1] <= np.iinfo(np.int32).max else np.int64
        return np.asarray(indices).astype(dtype, copy=False)

    def _check_rows(self, rows: Any) -> np.ndarray:
        given = np.asarray(rows)
        if given.ndim != 1 or (given.size and given.dtype.kind not in "iu"):
            raise TypeError("Destination rows must be a one-dimensional integer array")
        checked: np.ndarray = given.astype(np.int64, copy=False)
        if checked.size and (
            int(checked.min()) < 0 or int(checked.max()) >= self.shape[0]
        ):
            raise IndexError(
                f"Destination rows must lie in [0, {self.shape[0]}); found "
                f"{int(checked.min())} to {int(checked.max())}"
            )
        return checked

    def append_rows(self, rows: Any, matrix: csr_matrix) -> None:
        """Append whole rows: row ``i`` of ``matrix`` is destination row ``rows[i]``.

        Raises:
            TypeError: If the store takes loose entries, ``rows`` are not
                integers, or ``matrix`` is not CSR.
            IndexError: If a destination row or a column is outside the store.
            ValueError: If a destination row arrives a second time, ``matrix``
                does not hold one row of the store's columns for each
                destination row, or the entries pass ``max_nnz``.
        """
        self._require_writing("append_rows")
        rows = self._check_rows(rows)
        if getattr(matrix, "format", None) != "csr":
            raise TypeError(
                f"append_rows takes a CSR matrix, not {type(matrix).__name__}"
            )
        if matrix.shape != (rows.size, self.shape[1]):
            raise ValueError(
                f"append_rows takes {rows.size} rows of {self.shape[1]} columns, "
                f"not a matrix of shape {matrix.shape}"
            )
        self._refuse_repeated_rows(rows)
        if rows.size == 0:
            return
        self._check_columns(matrix.indices)
        lengths = np.diff(matrix.indptr).astype(np.int64)
        self._count_entries(int(lengths.sum()))
        assert self._rowPieces is not None
        self._rowPieces[rows] = 1
        self._rowEntries[rows] += lengths
        buckets = rows // self._bucketRows
        order = np.argsort(buckets, kind="stable")
        gathered = matrix[order]
        self._add_pieces(
            buckets[order],
            rows[order],
            lengths[order],
            np.asarray(gathered.indptr, dtype=np.int64),
            self._column_indices(gathered.indices),
            checked_sparse_cast(np.asarray(gathered.data), self._sourceDtype),
        )

    def _refuse_repeated_rows(self, rows: np.ndarray) -> None:
        """Refuse whole rows that repeat a destination row, here or in an earlier append."""
        assert self._rowPieces is not None
        ordered = np.sort(rows)
        repeated = ordered[1:][ordered[1:] == ordered[:-1]]
        earlier = rows[self._rowPieces[rows] != 0]
        if repeated.size or earlier.size:
            twice = int(repeated[0] if repeated.size else earlier[0])
            raise ValueError(
                f"Destination row {twice} was appended more than once; whole rows "
                "arrive once each"
            )

    def append_ragged_rows(self, rows: Any, values: Any, offsets: Any) -> None:
        """Append ragged rows: ``values[offsets[i]:offsets[i + 1]]`` is destination row ``rows[i]``.

        Raises:
            TypeError: If the store takes another form of rows, ``rows`` are
                not integers, or ``values`` are not uint8.
            IndexError: If a destination row is outside the store.
            ValueError: If a destination row arrives a second time, the
                offsets do not rise from 0 to the bytes of ``values``, one
                more than the rows, a row holds more bytes than the store's
                columns, or the bytes pass ``max_nnz``.
        """
        from ..metadata.records import take_ragged

        self._require_writing("append_ragged_rows")
        rows = self._check_rows(rows)
        values = np.asarray(values)
        if values.dtype != np.uint8 or values.ndim != 1:
            raise TypeError(
                "append_ragged_rows takes one-dimensional uint8 bytes, not an array "
                f"of dtype {values.dtype} and shape {values.shape}"
            )
        bounds = np.asarray(offsets)
        if (
            bounds.shape != (rows.size + 1,)
            or bounds.dtype.kind not in "iu"
            or int(bounds[0]) != 0
            or int(bounds[-1]) != values.size
            or bool(np.any(np.diff(bounds.astype(np.int64)) < 0))
        ):
            raise ValueError(
                f"append_ragged_rows takes {rows.size + 1} integer offsets that rise "
                f"from 0 to the {values.size} bytes; found dtype {bounds.dtype} and "
                f"shape {bounds.shape}"
            )
        bounds = bounds.astype(np.int64, copy=False)
        self._refuse_repeated_rows(rows)
        if rows.size == 0:
            return
        lengths = np.diff(bounds)
        if int(lengths.max()) > self.shape[1]:
            raise ValueError(
                f"A ragged row of the {self.operation} holds at most {self.shape[1]} "
                f"bytes; found one of {int(lengths.max())}"
            )
        self._count_entries(int(lengths.sum()))
        assert self._rowPieces is not None
        self._rowPieces[rows] = 1
        self._rowEntries[rows] += lengths
        buckets = rows // self._bucketRows
        order = np.argsort(buckets, kind="stable")
        gathered, pointers = take_ragged(values, bounds, order)
        self._add_pieces(
            buckets[order],
            rows[order],
            lengths[order],
            pointers,
            np.zeros(0, dtype=np.int32),
            gathered,
        )

    def append_entries(self, rows: Any, columns: Any, values: Any) -> None:
        """Append ``values[i]`` at destination row ``rows[i]`` and column ``columns[i]``.

        A row may arrive in pieces over many appends; a read joins them in
        the order they were appended.

        Raises:
            TypeError: If the store takes whole rows, or rows are not integers.
            IndexError: If a row or column is outside the store.
            ValueError: If the three arrays differ in length, or the entries
                pass ``max_nnz``.
        """
        self._require_writing("append_entries")
        rows = self._check_rows(rows)
        columns = np.asarray(columns)
        values = np.asarray(values)
        if columns.shape != rows.shape or values.shape != rows.shape:
            raise ValueError(
                "append_entries takes rows, columns and values of one length, not "
                f"{rows.shape}, {columns.shape} and {values.shape}"
            )
        if rows.size == 0:
            return
        self._check_columns(columns)
        self._count_entries(rows.size)
        order = _stable_order(rows)
        ordered = rows[order]
        firsts = np.flatnonzero(np.concatenate(([True], ordered[1:] != ordered[:-1])))
        piece_rows = ordered[firsts]
        del ordered
        pointers = np.append(firsts, rows.size).astype(np.int64)
        lengths = np.diff(pointers)
        # Each row is one piece of this append.
        self._rowEntries[piece_rows] += lengths
        self._add_pieces(
            piece_rows // self._bucketRows,
            piece_rows,
            lengths,
            pointers,
            self._column_indices(columns[order]),
            checked_sparse_cast(values[order], self._sourceDtype),
        )

    def _add_pieces(
        self,
        buckets: np.ndarray,
        rows: np.ndarray,
        lengths: np.ndarray,
        pointers: np.ndarray,
        indices: np.ndarray,
        values: np.ndarray,
    ) -> None:
        """Buffer pieces sorted by bucket, one piece of arrays for each bucket."""
        cuts = np.flatnonzero(np.diff(buckets)) + 1
        for first, last in zip(
            np.concatenate(([0], cuts)).tolist(),
            np.concatenate((cuts, [buckets.size])).tolist(),
            strict=True,
        ):
            bucket = int(buckets[first])
            left, right = int(pointers[first]), int(pointers[last])
            self._buckets.add(
                bucket,
                (
                    rows[first:last] - int(self.bucket_starts[bucket]),
                    lengths[first:last].copy(),
                    indices[left:right].copy(),
                    values[left:right].copy(),
                ),
            )

    def _spill_bound(
        self, entries: int, entry_bytes: float | None = None, parts: int = 1
    ) -> int:
        """Return the bytes that the store's files may take once ``entries`` are written.

        Without ``entry_bytes``, the entries take at most their bytes before
        compression, and each piece its place and length in two codes of at
        most 8 bytes; with it, a quarter more than they take at that rate,
        and each row its place and length in two codes of at most 8 bytes.
        The rate holds those of the measured rows, spread over their values,
        so rows of fewer values would take more than it charges them.
        Beside them, it counts one block for each of ``parts`` parts in
        each bucket, the least that a writer's file there holds, and the
        settings and records that the store writes (:meth:`_record_bytes`).
        """
        blocks = (self.bucket_starts.size - 1) * parts * _BLOCK_OVERHEAD
        beside = blocks + self._record_bytes(parts)
        if entry_bytes is not None:
            return (
                math.ceil(entries * float(entry_bytes) * _ESTIMATE_MARGIN)
                + self.shape[0] * 16
                + beside
            )
        index_bytes = (
            0
            if self._ragged
            else _UNSIGNED[_unsigned_code(max(0, self.shape[1] - 1))].itemsize
        )
        pieces = self.shape[0] if self._wholeRows else entries
        return (
            entries * (self._sourceDtype.itemsize + index_bytes) + pieces * 16 + beside
        )

    def _record_bytes(self, parts: int) -> int:
        """Return what the settings, the records of ``parts`` parts and the finalized record take on disk.

        The record of each part holds its writer's count of entries and mark
        of each row, and the file bytes, pieces and flag of each bucket; the
        finalized record holds the row pointers, the parts' numbers, and the
        first row, each part's file bytes, the pieces and the flag of each
        bucket.
        """
        n_rows = self.shape[0]
        n_buckets = self.bucket_starts.size - 1
        part = _archive_bytes(
            {
                "rowEntries": (np.int64, (n_rows,)),
                "rowPieces": (np.uint8, (n_rows if self._wholeRows else 0,)),
                "bucketBytes": (np.int64, (n_buckets,)),
                "bucketPieces": (np.int64, (n_buckets,)),
                "bucketMixed": (np.bool_, (n_buckets,)),
            }
        )
        finalized = _archive_bytes(
            {
                "indptr": (np.int64, (n_rows + 1,)),
                "partIds": (np.int64, (parts,)),
                "bucketStarts": (np.int64, (n_buckets + 1,)),
                "fileBytes": (np.int64, (parts, n_buckets)),
                "bucketPieces": (np.int64, (n_buckets,)),
                "bucketMixed": (np.bool_, (n_buckets,)),
            }
        )
        return len(json.dumps(self._settings)) + parts * part + finalized

    def _check_disk(self, required: int) -> None:
        free = shutil.disk_usage(self.directory).free
        if required > free:
            raise OSError(
                f"{_capitalized(self.operation)} needs {required} temporary bytes in "
                f"{self.directory}, which has {free} free; pass temp_dir on a "
                "disk with more free space"
            )

    def finish(self) -> None:
        """Write every pending piece and record this writer's part for :meth:`finalize`."""
        if self._part is None:
            raise RuntimeError("This sparse row store has no part to finish")
        self._buckets.write_all()
        parts = self.directory / "parts"
        name = f"part-{self._part:04d}"
        with (parts / f"{name}.tmp").open("wb") as stream:
            np.savez(
                stream,
                rowEntries=self._rowEntries,
                rowPieces=(
                    np.zeros(0, dtype=np.uint8)
                    if self._rowPieces is None
                    else self._rowPieces
                ),
                bucketBytes=self._buckets.fileBytes,
                bucketPieces=self._buckets.pieces,
                bucketMixed=self._buckets.mixed,
            )
        os.replace(parts / f"{name}.tmp", parts / f"{name}.npz")
        self._part = None
        del self._buckets, self._rowEntries, self._rowPieces

    def finalize(self) -> None:
        """Join every writer's part, split buckets beyond the split size, and plan reads.

        Raises:
            RuntimeError: If a writer's part was opened and never finished.
            ValueError: If whole rows are missing or arrived more than once;
                ``nnz_error`` if the entries pass ``max_nnz``.
            MemoryBudgetError: If reading one row needs more than
                ``bucket_bytes``.
        """
        if self._directory is None or self._finalized:
            raise RuntimeError("Only the owner finalizes a sparse row store, once")
        if self._part is not None:
            self.finish()
        parts = self.directory / "parts"
        opened = sorted(path.stem for path in parts.glob("part-*.open"))
        unfinished = [name for name in opened if not (parts / f"{name}.npz").exists()]
        if unfinished:
            raise RuntimeError(
                f"The writers of {', '.join(unfinished)} of the sparse row store in "
                f"{self.directory} did not finish"
            )
        n_rows = self.shape[0]
        n_buckets = self.bucket_starts.size - 1
        row_entries = np.zeros(n_rows, dtype=np.int64)
        row_pieces = np.zeros(n_rows, dtype=np.uint8) if self._wholeRows else None
        file_bytes = []
        pieces = np.zeros(n_buckets, dtype=np.int64)
        mixed = np.zeros(n_buckets, dtype=bool)
        for name in opened:
            with np.load(parts / f"{name}.npz") as part:
                row_entries += part["rowEntries"]
                if row_pieces is not None:
                    row_pieces += part["rowPieces"]
                    if n_rows and int(row_pieces.max()) > 1:
                        twice = int(np.flatnonzero(row_pieces > 1)[0])
                        raise ValueError(
                            f"Destination row {twice} was appended by more than "
                            "one writer; whole rows arrive once each"
                        )
                file_bytes.append(part["bucketBytes"])
                pieces += part["bucketPieces"]
                mixed |= part["bucketMixed"]
        if row_pieces is not None and n_rows and int(row_pieces.min()) == 0:
            missing = np.flatnonzero(row_pieces == 0)
            raise ValueError(
                f"{missing.size} destination rows were never appended, the first "
                f"{int(missing[0])}; whole rows arrive once each"
            )
        del row_pieces
        if int(row_entries.sum()) > self._maxNnz:
            raise self._nnzError(
                f"{_capitalized(self.operation)} exceeds maxNnz={self._maxNnz}"
            )
        self.indptr = cumulative_nnz(row_entries)
        del row_entries
        self._partIds = np.asarray(
            [int(name.removeprefix("part-")) for name in opened], dtype=np.int64
        )
        self._fileBytes = np.vstack(file_bytes)
        self._bucketPieces = pieces
        self._bucketMixed = mixed
        widest = self.widest_row_bytes()
        if widest > self._bucketLimit:
            raise MemoryBudgetError(
                f"One row of the {self.operation} exceeds its memory limit. "
                f"Reading it needs at least {widest} bytes.",
                needed_bytes=widest,
                budget_bytes=self._bucketLimit,
                item=f"one row of the {self.operation}",
            )
        self._split_buckets()
        np.savez(
            self.directory / "finalized.npz",
            indptr=self.indptr,
            partIds=self._partIds,
            bucketStarts=self.bucket_starts,
            fileBytes=self._fileBytes,
            bucketPieces=self._bucketPieces,
            bucketMixed=self._bucketMixed,
        )
        self._finalized = True

    def widest_row_bytes(self) -> int:
        """Return what reading the row with the most entries holds, once the rows are joined."""
        entries = int(np.diff(self.indptr).max(initial=0))
        return self._bucket_cost(entries, 1 if self._wholeRows else entries, 1)

    def _bucket_pieces(self, bucket: int, entries: int) -> int:
        if self._wholeRows:
            return int(self.bucket_starts[bucket + 1] - self.bucket_starts[bucket])
        return min(int(self._bucketPieces[bucket]), entries)

    def _bands(self, bucket: int) -> list[int]:
        """Return the first rows of bands of a bucket, each read within the split size."""
        first, last = (int(row) for row in self.bucket_starts[bucket : bucket + 2])
        pieces = int(self._bucketPieces[bucket])
        bands = []
        start = first
        while start < last:
            stop, low, high = start + 1, start + 1, last
            while low <= high:
                middle = (low + high) // 2
                entries = int(self.indptr[middle] - self.indptr[start])
                band_pieces = (
                    middle - start if self._wholeRows else min(pieces, entries)
                )
                if (
                    self._bucket_cost(entries, band_pieces, middle - start)
                    <= self._splitLimit
                ):
                    stop, low = middle, middle + 1
                else:
                    high = middle - 1
            bands.append(start)
            start = stop
        return bands

    def _split_buckets(self) -> None:
        """Rewrite every bucket that needs more than the split size to read in bands.

        A bucket's blocks are read in the order they were written and their
        pieces sent to the band of their row, so each band keeps the order
        of its rows' pieces. The rewrite holds one block and the buffer.
        """
        starts = []
        file_bytes = []
        pieces = []
        mixed = []
        for bucket in range(self.bucket_starts.size - 1):
            first, last = (int(row) for row in self.bucket_starts[bucket : bucket + 2])
            entries = int(self.indptr[last] - self.indptr[first])
            if (
                self._bucket_cost(
                    entries, self._bucket_pieces(bucket, entries), last - first
                )
                <= self._splitLimit
            ):
                starts.append(first)
                file_bytes.append(self._fileBytes[:, bucket])
                pieces.append(self._bucketPieces[bucket])
                mixed.append(self._bucketMixed[bucket])
                continue
            bands = np.asarray(self._bands(bucket), dtype=np.int64)
            staging = self.directory / f"split-{first:012d}"
            staging.mkdir()
            writer = self._pending_buckets(
                staging, np.append(bands, last), int(self._partIds[0])
            )
            for block in self._blocks(bucket):
                band = np.searchsorted(bands, block.rows + first, side="right") - 1
                for target in np.unique(band).tolist():
                    keep = band == target
                    entry_keep = np.repeat(keep, block.lengths)
                    writer.add(
                        target,
                        (
                            block.rows[keep] + first - int(bands[target]),
                            block.lengths[keep],
                            # Ragged rows have no column indices.
                            block.indices
                            if self._ragged
                            else block.indices[entry_keep],
                            block.values[entry_keep],
                        ),
                    )
                del block
            writer.write_all()
            # A bucket that no writer gave an entry has no folder.
            if self._folder(first).exists():
                shutil.rmtree(self._folder(first))
            for index, band_first in enumerate(bands.tolist()):
                folder = staging / f"rows-{band_first:012d}"
                if folder.exists():
                    folder.rename(self._folder(band_first))
                band_bytes = np.zeros(self._partIds.size, dtype=np.int64)
                band_bytes[0] = writer.fileBytes[index]
                starts.append(band_first)
                file_bytes.append(band_bytes)
                pieces.append(writer.pieces[index])
                mixed.append(writer.mixed[index])
            staging.rmdir()
        self.bucket_starts = np.append(
            np.asarray(starts, dtype=np.int64), self.shape[0]
        )
        self._fileBytes = (
            np.column_stack(file_bytes)
            if file_bytes
            else np.zeros((self._partIds.size, 0), dtype=np.int64)
        )
        self._bucketPieces = np.asarray(pieces, dtype=np.int64)
        self._bucketMixed = np.asarray(mixed, dtype=bool)

    # Opening by path.

    @classmethod
    def open(
        cls,
        directory: str | Path,
        *,
        part: int | None = None,
        buffer_bytes: int = 0,
    ) -> Self:
        """Open a store by its directory, in this or another process.

        With ``part``, the store takes rows as that part's writer, holding at
        most ``buffer_bytes`` while appending, and :meth:`finish` records the
        part for the owner's :meth:`finalize`. Without it, the store must be
        finalized and opens for reading. Closing an opened store leaves its
        directory to the owner.

        Raises:
            RuntimeError: If the directory holds no store, or no finalized
                store when it opens for reading.
        """
        store = cls.__new__(cls)
        path = Path(directory)
        try:
            with (path / "store.json").open() as stream:
                settings = json.load(stream)
        except FileNotFoundError as error:
            raise RuntimeError(f"{path} holds no sparse row store") from error
        store._configure(settings)
        store.directory = path
        if part is not None:
            store._start_writing(int(part), int(buffer_bytes))
            return store
        try:
            finalized = np.load(path / "finalized.npz")
        except FileNotFoundError as error:
            raise RuntimeError(
                f"The sparse row store in {path} is not finalized; its owner "
                "finalizes it once every writer finishes"
            ) from error
        with finalized:
            store.indptr = finalized["indptr"]
            store._partIds = finalized["partIds"]
            store.bucket_starts = finalized["bucketStarts"]
            store._fileBytes = finalized["fileBytes"]
            store._bucketPieces = finalized["bucketPieces"]
            store._bucketMixed = finalized["bucketMixed"]
        store._finalized = True
        return store

    # Reading.

    @property
    def resident_bytes(self) -> int:
        """The bytes of the row pointers, which a finalized store keeps."""
        return int(self.indptr.nbytes)

    def _require_finalized(self) -> None:
        if not self._finalized:
            raise RuntimeError("A sparse row store is read once its owner finalizes it")

    def _bucket_rows(self, bucket: int) -> tuple[int, int]:
        return int(self.bucket_starts[bucket]), int(self.bucket_starts[bucket + 1])

    def read_bytes(self, *, read_ahead: bool = False) -> int:
        """Return the most that reading holds beside the row pointers.

        Reading holds the bucket boundaries and one bucket as it loads, and
        with ``read_ahead`` the assembled bucket before it as well. The store
        must be finalized.
        """
        self._require_finalized()
        if self._readBytes is None:
            rows = np.diff(self.bucket_starts)
            entries = np.diff(self.indptr[self.bucket_starts])
            pieces = (
                rows if self._wholeRows else np.minimum(self._bucketPieces, entries)
            )
            itemsize = max(self._sourceDtype.itemsize, self.dtype.itemsize)
            peak = (
                entries * _entry_bytes(itemsize, self._sumDuplicates, self._ragged)
                + pieces * _piece_bytes(self._ragged)
                + (rows + 1) * 32
            )
            # An assembled bucket holds its values, each with its int64
            # column index unless the rows are ragged, and its row pointers.
            index_bytes = 0 if self._ragged else 8
            held = entries * (self.dtype.itemsize + index_bytes) + (rows + 1) * 8
            self._readBytes = (int(peak.max(initial=0)), int(held.max(initial=0)))
        peak, held = self._readBytes
        return int(self.bucket_starts.nbytes) + peak + (held if read_ahead else 0)

    def _blocks(self, bucket: int) -> Iterator[_Block]:
        """Yield the blocks of a bucket from every part, checking each file."""
        first, last = self._bucket_rows(bucket)
        folder = self._folder(first)
        if folder.with_suffix(".read").exists():
            raise RuntimeError(
                f"The rows [{first}, {last}) of the sparse row store in "
                f"{self.directory} were deleted once a read had taken them"
            )
        for part, expected in zip(
            self._partIds.tolist(), self._fileBytes[:, bucket].tolist(), strict=True
        ):
            if not expected:
                continue
            path = folder / f"part-{part:04d}.bin"
            try:
                stream = path.open("rb")
            except FileNotFoundError as error:
                raise RuntimeError(
                    f"{path}, which holds {expected} bytes of rows [{first}, {last}), "
                    "is missing"
                ) from error
            with stream:
                size = os.fstat(stream.fileno()).st_size
                if size != expected:
                    raise RuntimeError(
                        f"{path} holds {size} bytes where {expected} were written; "
                        "the file was truncated or changed"
                    )
                yield from _read_blocks(
                    stream,
                    size=size,
                    path=path,
                    source_dtype=self._sourceDtype,
                    n_columns=self.shape[1],
                    bucket_rows=last - first,
                    ragged=self._ragged,
                )

    def _read_pieces(
        self, bucket: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Read and check the pieces of one bucket.

        Returns:
            Each piece's row within the bucket and entries, the column indices
            and values of every entry, empty indices for ragged rows, and the
            pieces' order by row, which keeps the pieces of a row in the order
            they were appended.
        """
        first, last = self._bucket_rows(bucket)
        n_rows = last - first
        entries = int(self.indptr[last] - self.indptr[first])
        indices = np.empty(0 if self._ragged else entries, dtype=np.int64)
        values = np.empty(entries, dtype=self._sourceDtype)
        piece_rows: list[np.ndarray] = []
        piece_lengths: list[np.ndarray] = []
        filled = 0
        for block in self._blocks(bucket):
            count = block.values.size
            if filled + count > entries:
                raise RuntimeError(
                    f"The files of rows [{first}, {last}) hold more than the "
                    f"{entries} entries written to them"
                )
            if not self._ragged:
                indices[filled : filled + count] = block.indices
            values[filled : filled + count] = block.values
            filled += count
            piece_rows.append(block.rows)
            piece_lengths.append(block.lengths)
            del block
        if filled != entries:
            raise RuntimeError(
                f"The files of rows [{first}, {last}) hold {filled} entries where "
                f"{entries} were written"
            )
        rows = np.concatenate(piece_rows) if piece_rows else np.zeros(0, dtype=np.int64)
        lengths = (
            np.concatenate(piece_lengths)
            if piece_lengths
            else np.zeros(0, dtype=np.int64)
        )
        del piece_rows, piece_lengths
        if self._wholeRows:
            counts = np.bincount(rows, minlength=n_rows)
            if counts.size != n_rows or (
                n_rows and (counts.min() != 1 or counts.max() != 1)
            ):
                raise RuntimeError(
                    f"The files of rows [{first}, {last}) do not hold each of those "
                    "rows exactly once"
                )
            del counts
            order = np.argsort(rows)
        else:
            order = np.argsort(rows, kind="stable")
        return rows, lengths, indices, values, order

    def _load_ragged_bucket(self, bucket: int) -> tuple[np.ndarray, np.ndarray]:
        """Read, check and put in row order the ragged rows of one bucket.

        Returns:
            The rows' bytes, and the offset of each row in them, the last
            their end.
        """
        from ..metadata.records import take_ragged

        _rows, lengths, _indices, values, order = self._read_pieces(bucket)
        return take_ragged(values, cumulative_nnz(lengths), order)

    def _load_bucket(self, bucket: int) -> csr_matrix:
        """Read, check and assemble the rows of one bucket."""
        first, last = self._bucket_rows(bucket)
        n_rows = last - first
        rows, lengths, indices, values, order = self._read_pieces(bucket)
        pieces = csr_matrix(
            (values, indices, cumulative_nnz(lengths)), shape=(rows.size, self.shape[1])
        )
        del values, indices
        gathered = pieces[order] if order.size else pieces
        del pieces
        if self._wholeRows:
            # Each row is one piece, so the pieces in row order are the rows.
            matrix = gathered
            row_lengths = lengths[order]
        else:
            row_lengths = np.bincount(rows, weights=lengths, minlength=n_rows).astype(
                np.int64
            )
            matrix = csr_matrix(
                (gathered.data, gathered.indices, cumulative_nnz(row_lengths)),
                shape=(n_rows, self.shape[1]),
            )
        del gathered
        canonical = (not bool(self._bucketMixed[bucket]) and self._wholeRows) or bool(
            matrix.has_canonical_format
        )
        if canonical:
            matrix.data = checked_sparse_cast(matrix.data, self.dtype)
            matrix.has_canonical_format = True
            return matrix
        row_ids = np.repeat(np.arange(n_rows, dtype=np.int64), row_lengths)
        if self._sumDuplicates:
            return canonicalize_sparse(
                coo_matrix(
                    (matrix.data, (row_ids, matrix.indices)), shape=matrix.shape
                ),
                self.dtype,
            ).tocsr()
        # The stable sort keeps the duplicates of a coordinate in the order
        # they were appended, for the reader that sums them.
        order = np.lexsort((matrix.indices, row_ids))
        del row_ids
        sorted_rows = csr_matrix(
            (
                checked_sparse_cast(matrix.data[order], self.dtype),
                matrix.indices[order],
                cumulative_nnz(row_lengths),
            ),
            shape=matrix.shape,
        )
        sorted_rows.has_sorted_indices = True
        return sorted_rows

    def _check_window(self, start: int, stop: int) -> tuple[int, int]:
        self._require_finalized()
        if start < 0 or stop < start or stop > self.shape[0]:
            raise IndexError(
                f"Sparse row window [{start}, {stop}) is outside the matrix of "
                f"{self.shape[0]} rows"
            )
        return int(start), int(stop)

    def _buckets_of(self, start: int, stop: int) -> range:
        if start == stop:
            return range(0)
        first = int(np.searchsorted(self.bucket_starts, start, side="right")) - 1
        last = int(np.searchsorted(self.bucket_starts, stop, side="left"))
        return range(first, last)

    def _slice(
        self, matrix: csr_matrix, bucket: int, start: int, stop: int
    ) -> csr_matrix:
        first, last = self._bucket_rows(bucket)
        return matrix[max(start, first) - first : min(stop, last) - first]

    def _join(self, pieces: list[csr_matrix], rows: int) -> csr_matrix:
        if len(pieces) == 1:
            return pieces[0]
        if not pieces:
            empty = csr_matrix((rows, self.shape[1]), dtype=self.dtype)
            empty.has_canonical_format = True
            return empty
        joined = csr_matrix(
            (
                np.concatenate([piece.data for piece in pieces]),
                np.concatenate([piece.indices for piece in pieces]),
                cumulative_nnz(
                    np.concatenate([np.diff(piece.indptr) for piece in pieces])
                ),
            ),
            shape=(rows, self.shape[1]),
        )
        joined.has_canonical_format = all(
            piece.has_canonical_format for piece in pieces
        )
        return joined

    def read(self, start: int, stop: int) -> csr_matrix:
        """Return destination rows ``[start, stop)`` as canonical CSR.

        The store keeps the last bucket it loaded, so windows that follow one
        another load each bucket once and reading holds :meth:`read_bytes`.

        Raises:
            IndexError: If the window is outside the matrix.
            RuntimeError: If the store is not finalized, or a bucket file is
                missing, truncated or corrupted.
            TypeError: If the store holds ragged rows.
        """
        self._require_sparse_rows()
        start, stop = self._check_window(start, stop)
        pieces = []
        for bucket in self._buckets_of(start, stop):
            if self._cache is None or self._cache[0] != bucket:
                self._cache = None
                self._cache = (bucket, self._load_bucket(bucket))
            pieces.append(self._slice(self._cache[1], bucket, start, stop))
        return self._join(pieces, stop - start)

    def _require_sparse_rows(self) -> None:
        if self._ragged:
            raise TypeError(
                "This sparse row store holds ragged rows; read them with "
                "iter_ragged_rows"
            )

    def _slice_ragged(
        self, rows: tuple[np.ndarray, np.ndarray], bucket: int, start: int, stop: int
    ) -> tuple[np.ndarray, np.ndarray]:
        first, last = self._bucket_rows(bucket)
        values, offsets = rows
        low, high = max(start, first) - first, min(stop, last) - first
        begin = int(offsets[low])
        return values[begin : int(offsets[high])], offsets[low : high + 1] - begin

    def _join_ragged(
        self, pieces: list[tuple[np.ndarray, np.ndarray]], rows: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Join the rows of a batch, which a batch of at least one row gives in one piece or more."""
        if len(pieces) == 1:
            return pieces[0]
        values = np.concatenate([piece for piece, _ in pieces])
        offsets = cumulative_nnz(
            np.concatenate([np.diff(bounds) for _, bounds in pieces])
        )
        return values, offsets

    def iter_rows(
        self,
        start: int,
        stop: int,
        batch_rows: int,
        *,
        read_ahead: bool = False,
        delete: bool = False,
    ) -> Iterator[csr_matrix]:
        """Yield destination rows ``[start, stop)`` in batches of ``batch_rows``.

        With ``read_ahead``, the next bucket loads on a background thread
        while the current one is read, and reading holds :meth:`read_bytes`
        with ``read_ahead``; an error there raises here. With ``delete``, the
        files of every bucket that lies wholly inside ``[start, stop)`` are
        deleted once the batches that hold its rows are taken, and a later
        read of those rows raises.

        Raises:
            ValueError: If ``batch_rows`` is below 1.
            TypeError: If the store holds ragged rows, which
                :meth:`iter_ragged_rows` reads.
        """
        self._require_sparse_rows()
        yield from self._batches(
            start,
            stop,
            batch_rows,
            read_ahead,
            delete,
            self._load_bucket,
            self._slice,
            self._join,
        )

    def iter_ragged_rows(
        self,
        start: int,
        stop: int,
        batch_rows: int,
        *,
        read_ahead: bool = False,
        delete: bool = False,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield the bytes of ragged destination rows ``[start, stop)`` in batches of ``batch_rows``.

        Each batch is the rows' bytes and the offset of each row in them, the
        last their end. ``read_ahead`` and ``delete`` work as they do for
        :meth:`iter_rows`.

        Raises:
            ValueError: If ``batch_rows`` is below 1.
            TypeError: If the store holds sparse rows, which :meth:`iter_rows`
                reads.
        """
        if not self._ragged:
            raise TypeError(
                "This sparse row store holds sparse rows; read them with iter_rows"
            )
        yield from self._batches(
            start,
            stop,
            batch_rows,
            read_ahead,
            delete,
            self._load_ragged_bucket,
            self._slice_ragged,
            self._join_ragged,
        )

    def _batches[Rows](
        self,
        start: int,
        stop: int,
        batch_rows: int,
        read_ahead: bool,
        delete: bool,
        load: Callable[[int], Rows],
        cut: Callable[[Rows, int, int, int], Rows],
        join: Callable[[list[Rows], int], Rows],
    ) -> Iterator[Rows]:
        """Yield batches of destination rows, each bucket loaded by ``load``, cut and joined into batches."""
        start, stop = self._check_window(start, stop)
        if int(batch_rows) < 1:
            raise ValueError(f"batch_rows must be at least 1, not {batch_rows}")
        buckets = self._buckets_of(start, stop)
        executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="cytearc-rows")
            if read_ahead and len(buckets) > 1
            else None
        )
        loaded: dict[int, Rows] = {}
        ahead: tuple[int, Future[Rows]] | None = None

        def bucket_rows(bucket: int) -> Rows:
            nonlocal ahead
            if bucket not in loaded:
                if ahead is not None and ahead[0] == bucket:
                    future, ahead = ahead[1], None
                    loaded[bucket] = future.result()
                else:
                    loaded[bucket] = load(bucket)
            if executor is not None and ahead is None and bucket + 1 in buckets:
                ahead = (bucket + 1, executor.submit(load, bucket + 1))
            return loaded[bucket]

        try:
            for offset in range(start, stop, int(batch_rows)):
                end = min(stop, offset + int(batch_rows))
                pieces = []
                done = []
                for bucket in self._buckets_of(offset, end):
                    pieces.append(cut(bucket_rows(bucket), bucket, offset, end))
                    if self.bucket_starts[bucket + 1] <= end:
                        del loaded[bucket]
                        done.append(bucket)
                yield join(pieces, end - offset)
                del pieces
                if delete:
                    for bucket in done:
                        self._delete(bucket, start, stop)
        finally:
            loaded.clear()
            if executor is not None:
                if ahead is not None:
                    ahead[1].cancel()
                executor.shutdown(wait=True, cancel_futures=True)

    def release_cache(self) -> None:
        """Drop the bucket that :meth:`read` keeps for the next window; a later read loads it again."""
        self._cache = None

    def _delete(self, bucket: int, start: int, stop: int) -> None:
        """Delete a bucket's files if it lies wholly within ``[start, stop)``."""
        first, last = self._bucket_rows(bucket)
        if first < start or last > stop:
            return
        folder = self._folder(first)
        folder.with_suffix(".read").touch()
        shutil.rmtree(folder, ignore_errors=True)

    def close(self) -> None:
        """Release the store; the owner also removes its directory."""
        self._cache = None
        if self._directory is not None:
            self._directory.cleanup()

    # Sizes for callers that plan a store.

    @staticmethod
    def writer_bytes(n_rows: int, bucket_rows: int) -> int:
        """Return what each writer of a store holds beside its ``buffer_bytes``.

        Every writer, the owner's part included, keeps a count of entries
        and a mark for each of the ``n_rows`` destination rows, and a few
        numbers for each bucket of ``bucket_rows`` rows.
        """
        n_buckets = -(-int(n_rows) // max(1, int(bucket_rows)))
        return int(n_rows) * _WRITER_ROW_BYTES + (n_buckets + 1) * _WRITER_BUCKET_BYTES

    @staticmethod
    def whole_row_read_bytes(entries: int, itemsize: int) -> int:
        """Return what reading one whole row of ``entries`` entries holds, its duplicates not summed.

        Values take ``itemsize`` bytes, the wider of the source and read
        dtypes. A store refuses, when it is finalized, a row that needs
        more than its ``bucket_bytes``.
        """
        return _bucket_bytes(int(entries), 1, 1, int(itemsize), False)

    @staticmethod
    def planned_read_bytes(
        indptr: np.ndarray,
        *,
        bucket_rows: int,
        split_bytes: int,
        dtype: Any,
        read_ahead: bool = False,
    ) -> int:
        """Return what :meth:`read_bytes` gives for a finalized store of whole rows planned with these buckets.

        ``indptr`` are the row pointers of the rows, in destination order, of
        ``dtype`` values whose duplicates are not summed. The buckets start
        every ``bucket_rows`` rows, and :meth:`finalize` splits one that
        needs more than ``split_bytes`` to read into bands of the most rows
        that read within it, as it splits a store's own buckets. The store
        need not exist: the bands follow from the row pointers alone.
        """
        pointers = np.asarray(indptr, dtype=np.int64)
        n_rows = int(pointers.size) - 1
        itemsize = np.dtype(dtype).itemsize
        per_entry = _entry_bytes(itemsize, False)

        def cost(entries: int, rows: int) -> int:
            return _bucket_bytes(entries, rows, rows, itemsize, False)

        starts: list[int] = []
        for first in range(0, n_rows, max(1, int(bucket_rows))):
            last = min(n_rows, first + max(1, int(bucket_rows)))
            if cost(int(pointers[last] - pointers[first]), last - first) <= split_bytes:
                starts.append(first)
                continue
            start = first
            while start < last:
                stop, low, high = start + 1, start + 1, last
                while low <= high:
                    middle = (low + high) // 2
                    entries = int(pointers[middle] - pointers[start])
                    if cost(entries, middle - start) <= split_bytes:
                        stop, low = middle, middle + 1
                    else:
                        high = middle - 1
                starts.append(start)
                start = stop
        bounds = np.append(np.asarray(starts, dtype=np.int64), n_rows)
        rows = np.diff(bounds)
        entries = np.diff(pointers[bounds])
        peak = int(
            (entries * per_entry + rows * _piece_bytes(False) + (rows + 1) * 32).max(
                initial=0
            )
        )
        held = int((entries * (itemsize + 8) + (rows + 1) * 8).max(initial=0))
        return int(bounds.nbytes) + peak + (held if read_ahead else 0)

    @staticmethod
    def whole_row_bucket_rows(
        n_rows: int, entries: int, read_bytes: int, itemsize: int
    ) -> int:
        """Return the rows of buckets of whole rows that each read in about ``read_bytes``.

        The rows hold ``entries`` in all, with values of ``itemsize`` bytes
        that reads keep, without summing duplicates; each bucket holds at
        least one row and at most every row.
        """
        per_entry = _entry_bytes(int(itemsize), False)
        mean_entries = int(entries) / max(1, int(n_rows))
        row_bytes = mean_entries * per_entry + 48 + 32
        rows = int(max(0, int(read_bytes)) // max(1.0, row_bytes))
        return max(1, min(max(1, int(n_rows)), rows))

    @staticmethod
    def ragged_row_read_bytes(row_bytes: int) -> int:
        """Return what reading one ragged row of ``row_bytes`` bytes holds.

        A store of ragged rows refuses, when it is finalized, a row that
        needs more than its ``bucket_bytes``.
        """
        return _bucket_bytes(int(row_bytes), 1, 1, 1, False, ragged=True)

    @staticmethod
    def ragged_bucket_rows(n_rows: int, row_bytes: float, read_bytes: int) -> int:
        """Return the rows of buckets of ragged rows of about ``row_bytes`` bytes that each read in about ``read_bytes``.

        Each bucket holds at least one row and at most every row.
        """
        per_row = (
            float(row_bytes) * _entry_bytes(1, False, ragged=True)
            + _piece_bytes(ragged=True)
            + 32
        )
        rows = int(max(0, int(read_bytes)) // per_row)
        return max(1, min(max(1, int(n_rows)), rows))

    # Conversion of loose entries.

    @staticmethod
    def conversion_bytes(
        n_rows: int,
        *,
        entry_bytes: int,
        row_entries: int = 1,
        itemsize: int = 8,
        sum_duplicates: bool = True,
    ) -> int:
        """Return the least ``max_bytes`` with which :meth:`from_entries` converts rows.

        Rows hold up to ``row_entries`` entries, whose values take
        ``itemsize`` bytes, and arrive in chunks that hold ``entry_bytes`` for
        each entry. Beside the row bytes, half of the rest reads one bucket,
        and the other half holds a chunk and the write buffers.
        """
        read = _bucket_bytes(row_entries, row_entries, 1, int(itemsize), sum_duplicates)
        return (int(n_rows) + 1) * ROW_BYTES + 2 * max(read, int(entry_bytes))

    @staticmethod
    def chunk_entries(max_bytes: int, n_rows: int, entry_bytes: int) -> int:
        """Return the most entries of a chunk that :meth:`from_entries` may take within ``max_bytes``."""
        available = int(max_bytes) - (int(n_rows) + 1) * ROW_BYTES
        return max(1, (available // 2) // max(1, int(entry_bytes)))

    @classmethod
    def from_entries(
        cls,
        chunks: Iterable[coo_matrix],
        shape: tuple[int, int],
        dtype: Any,
        *,
        operation: str,
        max_bytes: int,
        entry_bytes: int,
        entries: int | None,
        source_dtype: Any | None = None,
        sum_duplicates: bool = True,
        max_nnz: int = np.iinfo(np.int64).max,
        nnz_error: type[ValueError] = ValueError,
        temp_dir: str | Path | None = None,
    ) -> Self:
        """Convert chunks of loose entries into a finalized store within ``max_bytes``.

        ``max_bytes`` holds :data:`ROW_BYTES` for each row; half of the rest
        holds one chunk of up to :meth:`chunk_entries` entries of
        ``entry_bytes`` each and the write buffers, and the other half reads
        the widest row. Buckets are sized to read in a thirty-second of
        ``max_bytes``, at most 128 MiB, and split beyond twice that, except a
        bucket of one wider row, so that a reader holds little beside the
        writes that it feeds. They are sized from ``entries``, the entries
        expected, which also check the free disk; None sizes them for dense
        rows. ``operation`` names the conversion in errors, as it names the
        store's.

        Raises:
            MemoryBudgetError: If ``max_bytes`` holds less than the row bytes
                and one entry, or than reading the row with the most entries.
            ValueError: If a chunk has the wrong shape; ``nnz_error`` beyond
                ``max_nnz``.
            OSError: If the disk lacks the bytes that the store's files may need.
        """
        n_rows, n_columns = (int(value) for value in shape)
        source = np.dtype(dtype if source_dtype is None else source_dtype)
        itemsize = max(source.itemsize, np.dtype(dtype).itemsize)
        fixed = (n_rows + 1) * ROW_BYTES
        least = cls.conversion_bytes(
            n_rows,
            entry_bytes=entry_bytes,
            itemsize=itemsize,
            sum_duplicates=sum_duplicates,
        )
        if int(max_bytes) < least:
            raise MemoryBudgetError(
                f"{_capitalized(operation)} exceeds the memory "
                f"limit. It needs at least {least} bytes.",
                needed_bytes=least,
                budget_bytes=int(max_bytes),
                item=f"the row pointers and one entry of the {operation}",
            )
        half = (int(max_bytes) - fixed) // 2
        per_entry = _bucket_bytes(2, 2, 0, itemsize, sum_duplicates) - _bucket_bytes(
            1, 1, 0, itemsize, sum_duplicates
        )
        # Reads hold one bucket beside the writes they feed, so buckets are
        # sized to read in a thirty-second of the budget and split beyond
        # twice that, when rows arrive unevenly; only a row that half of the
        # budget cannot hold is refused.
        target_bytes = min(_BUCKET_TARGET_BYTES, int(max_bytes) // 32)
        split = min(half, 2 * target_bytes)
        target = max(1, target_bytes // per_entry)
        if entries is None:
            bucket_rows = max(
                -(-n_rows // _MAX_UNKNOWN_BUCKETS), target // max(1, n_columns)
            )
        else:
            bucket_rows = int(target * n_rows // max(1, int(entries)))
        store = cls(
            shape,
            dtype,
            operation=operation,
            bucket_rows=max(1, min(n_rows, bucket_rows)),
            whole_rows=False,
            buffer_bytes=min(half, 4 * _PENDING_LIMIT_BYTES),
            bucket_bytes=half,
            split_bytes=split,
            entries=entries,
            source_dtype=source,
            sum_duplicates=sum_duplicates,
            max_nnz=max_nnz,
            nnz_error=nnz_error,
            temp_dir=temp_dir,
        )
        try:
            for chunk in chunks:
                if chunk.shape != (n_rows, n_columns):
                    raise ValueError("Sparse conversion chunk has the wrong shape")
                store.append_entries(chunk.row, chunk.col, chunk.data)
                del chunk
            try:
                store.finalize()
            except MemoryBudgetError as error:
                # Finalizing refuses only a row that one read cannot hold.
                needed = fixed + 2 * error.needed_bytes
                raise MemoryBudgetError(
                    f"One row of the {store.operation} exceeds its memory "
                    f"limit. It needs at least {needed} bytes.",
                    needed_bytes=needed,
                    budget_bytes=int(max_bytes),
                    item=error.item,
                ) from error
        except BaseException:
            store.close()
            raise
        return store
