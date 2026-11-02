"""Files that the readers of per-cell variant counts share.

The variant tools write folders of files with fixed suffixes, such as
mgatk's ``<name>.A.txt.gz`` or cellsnp-lite's ``cellSNP.tag.AD.mtx``, often
renamed with a prefix or gzip-compressed when deposited. These helpers find a
folder's files by suffix, open text whether or not it is compressed, read
Matrix Market entries and VCF records in bounded chunks, and identify a
folder by the digests of the files it holds. ``folder_format`` and
``hdf5_format`` recognise the format of a folder or an HDF5 file.
"""

import gzip
import hashlib
import io
import json
import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd

__all__ = [
    "VARIANT_FORMATS",
    "MatrixHeader",
    "VcfRecords",
    "find_file",
    "first_line",
    "folder_format",
    "hdf5_format",
    "is_gzip",
    "is_hdf5",
    "matrix_entries",
    "matrix_header",
    "open_text",
    "read_barcodes",
    "read_vcf",
    "source_sha256",
]

#: The values of ``format`` that ``import_counts`` reads.
VARIANT_FORMATS: tuple[str, ...] = (
    "mgatk",
    "maegatk",
    "cellsnp_lite",
    "mquad",
    "vartrix",
    "mito",
    "giftwrap",
    "ironthrone",
    "got",
    "got_multi",
    "gotcha",
)
_GZIP_SIGNATURE = b"\x1f\x8b"
_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def _signature(path: str | os.PathLike[str]) -> bytes:
    with open(path, "rb") as handle:
        return handle.read(8)


def is_gzip(path: str | os.PathLike[str]) -> bool:
    return _signature(path).startswith(_GZIP_SIGNATURE)


def open_text(path: str | os.PathLike[str]) -> io.TextIOBase:
    """Open a text file for reading, gzip-compressed or not, whatever its name.

    A byte order mark, which Excel writes before UTF-8 text, is dropped.
    """
    if is_gzip(path):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8-sig", newline="")
    return open(path, encoding="utf-8-sig", newline="")


def first_line(path: str | os.PathLike[str]) -> str:
    """The first line of a text file, without its line end, or "" when empty.

    Raises:
        ValueError: If the file is not UTF-8 text.
    """
    try:
        with open_text(path) as handle:
            return handle.readline().rstrip("\r\n")
    except UnicodeDecodeError:
        raise ValueError(f"{os.fspath(path)} is not a text file") from None


def find_file(
    folder: str | os.PathLike[str],
    pattern: str,
    what: str,
    *,
    required: bool = True,
) -> str | None:
    """Find the one file of a folder whose name matches ``pattern``.

    ``pattern`` is a regular expression matched against the whole name,
    which may end in ``.gz``.

    Raises:
        FileNotFoundError: If ``required`` and no file matches.
        ValueError: If several files match.
    """
    compiled = re.compile(pattern)
    matches = sorted(
        entry.name
        for entry in os.scandir(folder)
        if entry.is_file() and compiled.fullmatch(entry.name)
    )
    if len(matches) > 1:
        raise ValueError(
            f"{os.fspath(folder)} holds several {what}: {', '.join(matches)}; keep "
            "one source's files in each folder"
        )
    if not matches:
        if required:
            raise FileNotFoundError(f"{os.fspath(folder)} holds no {what}")
        return None
    return os.path.join(folder, matches[0])


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha256(files: Sequence[tuple[str, str]]) -> tuple[str, dict[str, str]]:
    """Identify a source by the SHA-256 of the files it reads.

    A source of one file is identified by the SHA-256 of its bytes, as every
    importer identifies its files; a folder of several files by the SHA-256
    of their roles and digests.

    Returns:
        The source's SHA-256, and each role's digest.
    """
    digests = {role: _file_sha256(path) for role, path in files}
    if len(digests) == 1:
        return next(iter(digests.values())), digests
    payload = json.dumps(sorted(digests.items()), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest(), digests


@dataclass(frozen=True, slots=True)
class MatrixHeader:
    """The banner and size line of a Matrix Market file."""

    path: str
    n_rows: int
    n_columns: int
    n_entries: int
    field: str
    header_lines: int


def matrix_header(path: str) -> MatrixHeader:
    """Read a Matrix Market file's banner and size line.

    Raises:
        ValueError: If the file is not a general coordinate matrix of integer
            or real values, or its size line is malformed.
    """
    with open_text(path) as handle:
        banner = handle.readline().strip().split()
        if len(banner) < 5 or banner[0].lower() != "%%matrixmarket":
            raise ValueError(f"{path} is not a Matrix Market file")
        kind, layout, field, symmetry = (word.lower() for word in banner[1:5])
        if (
            kind != "matrix"
            or layout != "coordinate"
            or field not in ("integer", "real")
            or symmetry != "general"
        ):
            raise ValueError(
                f"{path} is a {layout} {field} {symmetry} Matrix Market {kind}; "
                "counts are a general coordinate matrix of integer or real values"
            )
        header_lines = 1
        for line in handle:
            header_lines += 1
            if line.startswith("%") or not line.strip():
                continue
            sizes = line.split()
            break
        else:
            raise ValueError(f"{path} has no size line after its comments")
    try:
        n_rows, n_columns, n_entries = (int(value) for value in sizes)
    except ValueError:
        raise ValueError(
            f"The size line of {path} must hold three whole numbers, not "
            f"{' '.join(sizes)!r}"
        ) from None
    return MatrixHeader(
        path=path,
        n_rows=n_rows,
        n_columns=n_columns,
        n_entries=n_entries,
        field=field,
        header_lines=header_lines,
    )


def matrix_entries(
    header: MatrixHeader, chunk_rows: int
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield a Matrix Market file's entries in chunks: rows, columns, values.

    Rows and columns are 0-based. Values are counts: whole numbers of 0 or
    more, which scipy writes as ``3.71E2`` in a real matrix, as MQuad does.

    Raises:
        ValueError: If an entry is not three numbers or lies outside the
            matrix, a value is not a whole number of 0 or more, or the file
            holds another number of entries than its size line says.
    """
    try:
        reader = pd.read_csv(
            header.path,
            sep=r"\s+",
            header=None,
            skiprows=header.header_lines,
            usecols=[0, 1, 2],
            dtype=np.float64,
            chunksize=chunk_rows,
            compression="gzip" if is_gzip(header.path) else None,
        )
    except pd.errors.EmptyDataError:
        reader = None
    chunks = iter(reader if reader is not None else ())
    seen = 0
    while True:
        try:
            chunk = next(chunks)
        except (StopIteration, pd.errors.EmptyDataError):
            # A matrix without entries ends at its size line.
            break
        except ValueError as error:
            raise ValueError(
                f"{header.path} holds an entry that is not three numbers: {error}"
            ) from None
        seen += len(chunk)
        coordinates = chunk[[0, 1]].to_numpy(dtype=np.float64)
        rows = coordinates[:, 0].astype(np.int64) - 1
        columns = coordinates[:, 1].astype(np.int64) - 1
        if (
            not np.all(coordinates == np.floor(coordinates))
            or np.any(rows < 0)
            or np.any(rows >= header.n_rows)
            or np.any(columns < 0)
            or np.any(columns >= header.n_columns)
        ):
            raise ValueError(
                f"{header.path} has entries outside its {header.n_rows} by "
                f"{header.n_columns} shape"
            )
        values = chunk[2].to_numpy(dtype=np.float64)
        whole = np.isfinite(values) & (values >= 0) & (values == np.floor(values))
        if not whole.all():
            raise ValueError(
                f"{header.path} holds the value {values[~whole][0]!r}; counts are "
                "whole numbers of 0 or more"
            )
        yield rows, columns, values.astype(np.int64)
    if seen != header.n_entries:
        raise ValueError(
            f"{header.path} declares {header.n_entries} entries but holds {seen}"
        )


@dataclass(frozen=True, slots=True)
class VcfRecords:
    """The variant records of a VCF file, in file order.

    Attributes:
        chrom: Each record's contig.
        pos: Each record's 1-based position.
        ref: Each record's reference allele, uppercased.
        alt: Each record's alternative alleles, uppercased and comma-joined.
        contig_lengths: The contig lengths that the header declares.
    """

    chrom: np.ndarray
    pos: np.ndarray
    ref: np.ndarray
    alt: np.ndarray
    contig_lengths: dict[str, int]


_CONTIG_LINE = re.compile(r"^##contig=<(?P<fields>.*)>\s*$")


def read_vcf(path: str) -> VcfRecords:
    """Read the CHROM, POS, REF and ALT columns of a VCF file.

    Raises:
        ValueError: If the file has no ``#CHROM`` header, a record has fewer
            than five columns or a position that is not a whole number of 1
            or more.
    """
    contig_lengths: dict[str, int] = {}
    header_lines = 0
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("##"):
                header_lines += 1
                match = _CONTIG_LINE.match(line.rstrip("\r\n"))
                if match is not None:
                    fields = dict(
                        part.split("=", 1)
                        for part in match.group("fields").split(",")
                        if "=" in part
                    )
                    if "ID" in fields and fields.get("length", "").isdigit():
                        contig_lengths[fields["ID"]] = int(fields["length"])
                continue
            if line.startswith("#CHROM"):
                header_lines += 1
                break
            raise ValueError(f"{path} has no #CHROM header line before its records")
        else:
            raise ValueError(f"{path} has no #CHROM header line")
    try:
        table = pd.read_csv(
            path,
            sep="\t",
            header=None,
            skiprows=header_lines,
            usecols=[0, 1, 3, 4],
            dtype=str,
            keep_default_na=False,
            compression="gzip" if is_gzip(path) else None,
        )
    except pd.errors.EmptyDataError:
        # A VCF without records lists no variants.
        table = pd.DataFrame({column: [] for column in (0, 1, 3, 4)}, dtype=str)
    except ValueError:
        raise ValueError(
            f"The records of {path} need the columns CHROM, POS, ID, REF and ALT"
        ) from None
    positions = pd.to_numeric(table[1], errors="coerce").to_numpy(dtype=np.float64)
    whole = (
        np.isfinite(positions) & (positions >= 1) & (positions == np.floor(positions))
    )
    if not whole.all():
        record = int(np.flatnonzero(~whole)[0])
        raise ValueError(
            f"Line {record + header_lines + 1} of {path} holds the position "
            f"{table[1].iloc[record]!r}; positions are whole numbers of 1 or more"
        )
    return VcfRecords(
        chrom=table[0].to_numpy(dtype=object),
        pos=positions.astype(np.int64),
        ref=table[3].str.upper().to_numpy(dtype=object),
        alt=table[4].str.upper().to_numpy(dtype=object),
        contig_lengths=contig_lengths,
    )


def read_barcodes(path: str) -> np.ndarray:
    """Read a list of barcodes, one per line, as text.

    The first column of each line is the barcode, so Cell Ranger's
    ``barcodes.tsv`` and lists with further columns read alike; blank lines
    are skipped.

    Raises:
        ValueError: If a barcode repeats.
    """
    with open_text(path) as handle:
        barcodes = [
            line.split("\t")[0].strip()
            for line in handle.read().splitlines()
            if line.strip()
        ]
    array = np.asarray(barcodes, dtype=object)
    repeated = pd.Index(array).duplicated()
    if repeated.any():
        raise ValueError(
            f"The barcode {array[repeated][0]!r} repeats in {path}; each column of "
            "the matrices needs its own barcode"
        )
    return array


_BASE_FILE = r"(.*\.)?A\.txt(\.gz)?"
_GIFT_READS = r"(.*_)?probe_reads\.tsv(\.gz)?"


def folder_format(folder: str) -> str:
    """Recognise the format of a folder by the names of its files.

    Raises:
        ValueError: If the folder holds no known files.
    """
    names = [entry.name for entry in os.scandir(folder) if entry.is_file()]
    if any(re.fullmatch(_BASE_FILE, name) for name in names):
        return "mgatk"
    if any(re.fullmatch(r"(.*\.)?tag\.AD\.mtx(\.gz)?", name) for name in names):
        return "cellsnp_lite"
    if any(re.fullmatch(r"(.*_)?passed_ad\.mtx(\.gz)?", name) for name in names):
        return "mquad"
    if any(re.fullmatch(_GIFT_READS, name) for name in names):
        return "giftwrap"
    if (
        any(re.fullmatch(r".*\.vcf(\.gz)?", name) for name in names)
        and sum(bool(re.fullmatch(r".*\.mtx(\.gz)?", name)) for name in names) >= 2
    ):
        return "vartrix"
    raise ValueError(
        f"{folder} holds none of the files of mgatk, maegatk, cellsnp-lite, MQuad, "
        "VarTrix or giftwrap; pass format to name its format"
    )


def is_hdf5(path: str | os.PathLike[str]) -> bool:
    return _signature(path) == _HDF5_SIGNATURE


def hdf5_format(path: str) -> str:
    """Recognise giftwrap's counts or MiTo's allele frequency matrix.

    Raises:
        ValueError: If the file is neither.
    """
    with h5py.File(path, "r") as handle:
        matrix = handle.get("matrix")
        if isinstance(matrix, h5py.Group) and "probe" in matrix:
            return "giftwrap"
        layers = handle.get("layers")
        if isinstance(layers, h5py.Group) and "AD" in layers and "DP" in layers:
            return "mito"
    raise ValueError(
        f"{path} is an HDF5 file of neither giftwrap's counts nor MiTo's allele "
        "frequency matrix"
    )
