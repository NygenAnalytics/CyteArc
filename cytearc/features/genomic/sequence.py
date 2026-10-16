"""Random access to genome sequences and per-interval nucleotide composition."""

import mmap
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import numpy as np

__all__ = [
    "FastaIndexRecord",
    "IndexedFasta",
    "encode_sequence",
    "gc_content",
    "parse_coordinate_ids",
    "read_fasta_index",
]

#: Nucleotide codes used by encoded sequences: A, C, G, and T are 0 to 3, and
#: every other byte, such as N or an IUPAC ambiguity code, is 4.
NUCLEOTIDE_CODES = b"ACGT"
UNKNOWN_NUCLEOTIDE = 4

_CODE_TABLE = np.full(256, UNKNOWN_NUCLEOTIDE, dtype=np.uint8)
for _code, _base in enumerate(NUCLEOTIDE_CODES):
    _CODE_TABLE[_base] = _code
    _CODE_TABLE[ord(chr(_base).lower())] = _code
_CODE_TABLE.setflags(write=False)


@dataclass(frozen=True, slots=True)
class FastaIndexRecord:
    """One sequence entry of a samtools ``faidx`` index.

    Attributes:
        name: Sequence name, the first word of its FASTA header.
        length: Number of bases in the sequence.
        offset: Byte offset of the first base in the FASTA file.
        lineBases: Bases on each full line.
        lineWidth: Bytes on each full line, including the line terminator.
    """

    name: str
    length: int
    offset: int
    lineBases: int
    lineWidth: int


def read_fasta_index(path: str | os.PathLike[str]) -> dict[str, FastaIndexRecord]:
    """Read a samtools ``faidx`` index into records keyed by sequence name."""
    records: dict[str, FastaIndexRecord] = {}
    with open(path, encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.rstrip("\r\n")
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) < 5:
                raise ValueError(
                    f"Malformed FASTA index line {line_number}: expected 5 fields"
                )
            try:
                length, offset, line_bases, line_width = (
                    int(value) for value in fields[1:5]
                )
            except ValueError as exc:
                raise ValueError(
                    f"Malformed FASTA index line {line_number}: non-integer field"
                ) from exc
            if length < 0 or offset < 0 or line_bases <= 0 or line_width < line_bases:
                raise ValueError(f"Invalid FASTA index line {line_number}")
            name = fields[0]
            if name in records:
                raise ValueError(f"FASTA index repeats sequence {name!r}")
            records[name] = FastaIndexRecord(
                name, length, offset, line_bases, line_width
            )
    if not records:
        raise ValueError(f"FASTA index {os.fspath(path)!r} lists no sequences")
    return records


def _build_fasta_index(path: Path) -> dict[str, FastaIndexRecord]:
    """Index an uncompressed FASTA file the way ``samtools faidx`` does.

    Every line of a sequence except its last must hold the same number of
    bases, because random access computes byte offsets from that width.
    """
    records: dict[str, FastaIndexRecord] = {}
    name: str | None = None
    length = offset = line_bases = line_width = 0
    short_line_seen = False
    position = 0

    def finish() -> None:
        if name is None:
            return
        if name in records:
            raise ValueError(f"FASTA file repeats sequence {name!r}")
        records[name] = FastaIndexRecord(
            name, length, offset, max(line_bases, 1), max(line_width, 1)
        )

    with open(path, "rb") as handle:
        for raw in handle:
            if raw.startswith(b">"):
                finish()
                header = raw[1:].strip().split()
                if not header:
                    raise ValueError(f"FASTA header at byte {position} has no name")
                name = header[0].decode("utf-8")
                length = line_bases = line_width = 0
                short_line_seen = False
                offset = position + len(raw)
            elif name is None:
                if raw.strip():
                    raise ValueError("FASTA file does not start with a header")
            else:
                bases = len(raw.rstrip(b"\r\n"))
                if bases == 0:
                    short_line_seen = True
                elif line_bases == 0:
                    line_bases = bases
                    line_width = len(raw)
                elif short_line_seen or bases > line_bases:
                    raise ValueError(
                        f"Sequence {name!r} has lines of different widths, "
                        "so it cannot be indexed for random access"
                    )
                elif bases < line_bases or len(raw) != line_width:
                    short_line_seen = True
                length += bases
            position += len(raw)
    finish()
    if not records:
        raise ValueError(f"FASTA file {os.fspath(path)!r} contains no sequences")
    return records


class IndexedFasta:
    """Read intervals of an uncompressed FASTA file without loading it.

    The file is memory mapped and read through a samtools ``faidx`` index.
    The index is read from ``<path>.fai`` when that file exists, or from
    ``index_path``; otherwise the FASTA file is scanned once to build it in
    memory. Nothing is written next to the FASTA file.

    Coordinates are zero-based and half-open, as in BED files.

    Args:
        path: Uncompressed FASTA file. Compressed files are not supported.
        index_path: Optional samtools ``faidx`` index of ``path``.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        index_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.path = Path(path)
        if self.path.suffix in {".gz", ".bgz", ".bz2", ".xz", ".zst"}:
            raise ValueError(
                "IndexedFasta needs an uncompressed FASTA file; "
                f"decompress {self.path.name} first"
            )
        default_index = Path(f"{self.path}.fai")
        if index_path is not None:
            self.records = read_fasta_index(index_path)
        elif default_index.is_file():
            self.records = read_fasta_index(default_index)
        else:
            self.records = _build_fasta_index(self.path)
        self._handle = open(self.path, "rb")
        try:
            self._map = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self._handle.close()
            raise
        file_size = len(self._map)
        for record in self.records.values():
            if record.length == 0:
                continue
            last = self._byte_offset(record, record.length - 1)
            if last >= file_size:
                self.close()
                raise ValueError(
                    f"FASTA index entry {record.name!r} points past the end of "
                    f"{self.path.name}; the index does not describe this file"
                )

    def __enter__(self) -> "IndexedFasta":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release the memory map and file handle."""
        mapped = getattr(self, "_map", None)
        if mapped is not None and not mapped.closed:
            mapped.close()
        handle = getattr(self, "_handle", None)
        if handle is not None and not handle.closed:
            handle.close()

    @property
    def names(self) -> tuple[str, ...]:
        """Sequence names in index order."""
        return tuple(self.records)

    def length(self, name: str) -> int:
        """Number of bases in sequence ``name``."""
        return self._record(name).length

    def _record(self, name: str) -> FastaIndexRecord:
        try:
            return self.records[name]
        except KeyError:
            alternative = name[3:] if name.startswith("chr") else f"chr{name}"
            hint = (
                f"; the file names it {alternative!r}"
                if alternative in self.records
                else ""
            )
            raise KeyError(
                f"Sequence {name!r} is not in {self.path.name}{hint}"
            ) from None

    @staticmethod
    def _byte_offset(record: FastaIndexRecord, base: int) -> int:
        line, column = divmod(base, record.lineBases)
        return record.offset + line * record.lineWidth + column

    def fetch(self, name: str, start: int, end: int) -> bytes:
        """Return the upper-case bases of ``name`` in ``[start, end)``."""
        record = self._record(name)
        start = int(start)
        end = int(end)
        if start < 0 or end < start:
            raise ValueError(f"Invalid interval {name}:{start}-{end}")
        if end > record.length:
            raise ValueError(
                f"Interval {name}:{start}-{end} extends past the sequence end "
                f"({record.length} bases)"
            )
        if start == end:
            return b""
        first = self._byte_offset(record, start)
        last = self._byte_offset(record, end - 1)
        raw = self._map[first : last + 1]
        if record.lineWidth != record.lineBases:
            raw = raw.replace(b"\n", b"").replace(b"\r", b"")
        if len(raw) != end - start:
            raise ValueError(
                f"Read {len(raw)} bases for {name}:{start}-{end}; the FASTA "
                "index does not match the file"
            )
        return raw.upper()

    def encode(self, name: str, start: int, end: int) -> np.ndarray:
        """Return ``fetch(name, start, end)`` as nucleotide codes (see ``encode_sequence``)."""
        return encode_sequence(self.fetch(name, start, end))


def encode_sequence(sequence: bytes | str) -> np.ndarray:
    """Encode bases as uint8 codes: A, C, G, T are 0 to 3 and anything else is 4."""
    raw = sequence.encode("ascii") if isinstance(sequence, str) else sequence
    return _CODE_TABLE[np.frombuffer(raw, dtype=np.uint8)]


def parse_coordinate_ids(
    ids: Iterable[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse ``chrom:start-end`` identifiers in their given order.

    Coordinates are read as zero-based, half-open BED intervals, the
    convention of Cell Ranger peak identifiers.

    Returns:
        Chromosome names, int64 starts, and int64 ends aligned to ``ids``.

    Raises:
        ValueError: If an identifier is not ``chrom:start-end`` with
            non-negative integer bounds and ``start < end``.
    """
    chroms: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for position, value in enumerate(ids):
        text = str(value)
        chrom, colon, span = text.rpartition(":")
        start_text, hyphen, end_text = span.partition("-")
        try:
            start = int(start_text)
            end = int(end_text)
        except ValueError:
            start = end = -1
        if not colon or not chrom or not hyphen or start < 0 or end <= start:
            raise ValueError(
                f"Feature {position} ({text!r}) is not a chrom:start-end interval "
                "with start < end"
            )
        chroms.append(chrom)
        starts.append(start)
        ends.append(end)
    return (
        np.asarray(chroms, dtype=object),
        np.asarray(starts, dtype=np.int64),
        np.asarray(ends, dtype=np.int64),
    )


def gc_content(genome: IndexedFasta, regions: Iterable[str]) -> np.ndarray:
    """Return the fraction of G and C among the A, C, G, and T bases of each region.

    N and other ambiguity codes count in neither the numerator nor the
    denominator, as in chromVAR's ``addGCBias``. A region without any A, C,
    G, or T base has no defined content and gives NaN.

    Args:
        genome: Genome the regions refer to.
        regions: Region ids ``chrom:start-end``, such as ATAC peak ids.
    """
    chrom_values, start_values, end_values = parse_coordinate_ids(regions)
    result = np.full(len(chrom_values), np.nan, dtype=np.float64)
    for index, (chrom, start, end) in enumerate(
        zip(chrom_values, start_values, end_values, strict=True)
    ):
        codes = genome.encode(str(chrom), int(start), int(end))
        counts = np.bincount(codes, minlength=5)
        known = int(counts[:4].sum())
        if known:
            result[index] = (counts[1] + counts[2]) / known
    return result
