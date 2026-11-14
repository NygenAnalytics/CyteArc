"""Files that the readers of per-cell variant counts share.

The variant tools write folders of files with fixed suffixes, such as
mgatk's ``<name>.A.txt.gz`` or cellsnp-lite's ``cellSNP.tag.AD.mtx``, often
renamed with a prefix or gzip-compressed when deposited. These helpers find a
folder's files by suffix, read VCF records and barcode lists, and identify a
folder by the digests of the files it holds. ``folder_format`` and
``hdf5_format`` recognize the format of a folder or an HDF5 file. Files open
through :mod:`cytearc.utils.files`, and Matrix Market entries are read by
:mod:`cytearc.readers._matrix_market`.
"""

import hashlib
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd
from ..utils.arguments import quoted_value

from ..utils.digest import file_sha256
from ..utils.files import is_gzip, open_text

__all__ = [
    "VARIANT_FORMATS",
    "VcfRecords",
    "find_file",
    "folder_format",
    "hdf5_format",
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
    "tapestri",
    "dab_seq",
)


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


def source_sha256(files: Sequence[tuple[str, str]]) -> tuple[str, dict[str, str]]:
    """Identify a source by the SHA-256 of the files it reads.

    A source of one file is identified by the SHA-256 of its bytes, as every
    importer identifies its files; a folder of several files by the SHA-256
    of their roles and digests.

    Returns:
        The source's SHA-256, and each role's digest.
    """
    digests = {role: file_sha256(path) for role, path in files}
    if len(digests) == 1:
        return next(iter(digests.values())), digests
    payload = json.dumps(sorted(digests.items()), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest(), digests


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
            f"{quoted_value(table[1].iloc[record])}; positions are whole numbers of 1 or more"
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
            f"The barcode {quoted_value(array[repeated][0])} repeats in {path}; each column of "
            "the matrices needs its own barcode"
        )
    return array


_BASE_FILE = r"(.*\.)?A\.txt(\.gz)?"
_GIFT_READS = r"(.*_)?probe_reads\.tsv(\.gz)?"


def folder_format(folder: str) -> str:
    """Recognize the format of a folder by the names of its files.

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


_DAB_SEQ_ARRAYS = ("AD", "RD", "DP", "GT", "VARIANTS", "CELL_BARCODES")


def hdf5_format(path: str) -> str:
    """Recognize giftwrap's counts, MiTo's allele frequency matrix, Tapestri's
    DNA variants or DAb-seq's genotypes.

    Raises:
        ValueError: If the file is none of them.
    """
    with h5py.File(path, "r") as handle:
        matrix = handle.get("matrix")
        if isinstance(matrix, h5py.Group) and "probe" in matrix:
            return "giftwrap"
        layers = handle.get("layers")
        if isinstance(layers, h5py.Group) and "AD" in layers and "DP" in layers:
            return "mito"
        if isinstance(handle.get("assays/dna_variants"), h5py.Group):
            return "tapestri"
        if all(isinstance(handle.get(name), h5py.Dataset) for name in _DAB_SEQ_ARRAYS):
            return "dab_seq"
    raise ValueError(
        f"{path} is an HDF5 file of none of giftwrap's counts, MiTo's allele "
        "frequency matrix, Tapestri's DNA variants and DAb-seq's genotypes"
    )
