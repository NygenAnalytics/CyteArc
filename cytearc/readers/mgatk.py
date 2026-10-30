"""Read the per-cell base counts of mgatk and maegatk.

mgatk (mtscATAC-seq) and maegatk (MAESTER) count, for every cell and every
position of the mitochondrial genome, the reads of each base on each strand.
A folder of their output holds one gzip CSV per base, ``<name>.A.txt.gz`` to
``<name>.T.txt.gz``, a reference file ``<contig>_refAllele.txt`` and, from
mgatk, a depth table ``<name>.depthTable.txt`` that lists every cell it
genotyped, those without reads included. Deposits often add prefixes or gzip
the other files, so files are matched by suffix, as Signac matches them.

A base file holds ``position,barcode,forward,reverse``, 1-based and only
where a count is above 0, or six columns ``position,barcode,forward,
forward quality,reverse,reverse quality`` with mgatk's
``--emit-base-qualities``. maegatk always writes six columns and counts
reverse-aligned reads as forward, as mgatk's ``call`` and ``bcall`` did from
0.4.0 to 0.5.8, so maegatk's strands are swapped back, and old mgatk files
need ``swap_strands``. The two six-column layouts look alike, so such a
folder needs an explicit format. mgatk counts reads and maegatk UMI
consensus reads, which are molecules.

The reference is uppercased, since mgatk's references keep soft-masked
lowercase bases, and positions of ``N``, such as 3107 of rCRS, are not
sites. The contig is named after the reference file, ``chrM`` for
``chrM_refAllele.txt``, and is unnamed for nf-MiTo's ``refAllele.txt``.
"""

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..variants.counts import BASES, MAX_COUNT, Contig, EntryBatch, SiteTable
from ._variant_files import find_file, first_line, is_gzip

__all__ = ["MGATK_FORMATS", "MgatkSource", "mgatk_source"]

#: The formats this module reads.
MGATK_FORMATS: tuple[str, ...] = ("mgatk", "maegatk")
_REFERENCE = r"(.*[._])?refAllele\.txt(\.gz)?"
_DEPTH_TABLE = r"(.*\.)?depthTable\.txt(\.gz)?"
_FOUR = ("position", "barcode", "forward", "reverse")
_SIX = (
    "position",
    "barcode",
    "forward",
    "quality_forward",
    "reverse",
    "quality_reverse",
)


def _base_pattern(base: str) -> str:
    return rf"(.*\.)?{base}\.txt(\.gz)?"


def _contig_name(path: str) -> str:
    name = os.path.basename(path)
    if name.endswith(".gz"):
        name = name[: -len(".gz")]
    stem = name[: -len("refAllele.txt")]
    # mgatk names the file <contig>_refAllele.txt; deposits add "<sample>.".
    if not stem.endswith("_"):
        return ""
    return stem[:-1].rsplit(".", 1)[-1]


@dataclass(frozen=True, slots=True)
class _Reference:
    contig: Contig
    bases: np.ndarray

    @property
    def length(self) -> int:
        return len(self.bases)


def _read_reference(path: str) -> _Reference:
    """Read a ``refAllele`` file: 1-based positions in order and their bases.

    Raises:
        ValueError: If a row is not a position and a base, positions do not
            run from 1 without gaps, or a base is not A, C, G, T or N.
    """
    table = pd.read_csv(
        path,
        sep="\t",
        header=None,
        dtype=str,
        keep_default_na=False,
        compression="gzip" if is_gzip(path) else None,
    )
    if table.shape[1] != 2:
        raise ValueError(
            f"{path} must hold a position and a base on each line, not "
            f"{table.shape[1]} fields"
        )
    expected = np.arange(1, len(table) + 1).astype(str)
    if not np.array_equal(table[0].str.strip().to_numpy(dtype=str), expected):
        raise ValueError(
            f"The positions of {path} must run from 1 in order without gaps"
        )
    bases = table[1].str.strip().str.upper().to_numpy(dtype=object)
    known = np.isin(bases, [*BASES, "N"])
    if not known.all():
        position = int(np.flatnonzero(~known)[0])
        raise ValueError(
            f"Position {position + 1} of {path} holds {bases[position]!r}; a "
            "reference base is A, C, G, T or N"
        )
    sequence = "".join(bases.tolist())
    return _Reference(
        contig=Contig(
            name=_contig_name(path),
            length=len(bases),
            sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
        ),
        bases=bases,
    )


def _columns(path: str) -> tuple[str, ...]:
    """The columns of a base file, from its first line.

    Raises:
        ValueError: If the line holds neither four nor six fields, or four
            whose last is a mean quality, as mgatk wrote before 0.4.7.
    """
    line = first_line(path)
    if not line:
        return _FOUR
    fields = line.split(",")
    if len(fields) == 6:
        return _SIX
    if len(fields) == 4:
        if "." in fields[3]:
            raise ValueError(
                f"{path} holds position, barcode, count and mean quality, as mgatk "
                "wrote before 0.4.7; CyteArc reads mgatk 0.4.7 and later"
            )
        return _FOUR
    raise ValueError(
        f"{path} holds {len(fields)} fields per row; mgatk writes four "
        "(position, barcode, forward, reverse) or six, with base qualities"
    )


@dataclass(frozen=True, slots=True)
class MgatkSource:
    """An mgatk or maegatk folder, read as a source of allele counts."""

    path: str
    format: str
    files: tuple[tuple[str, str], ...]
    count_unit: str | None
    depth_definition: str
    count_fields: tuple[str, ...]
    depth_roles: frozenset[str] | None
    reference_from_depth: bool
    open_alleles: bool
    notes: dict[str, Any]
    _reference: _Reference = field(repr=False)
    _columns: tuple[str, ...] = field(repr=False)
    _swap: bool = field(repr=False)
    _threads: int = field(default=1, repr=False)

    def sites(self) -> SiteTable:
        """Every position of the reference whose base is A, C, G or T, each
        with the four bases as alleles."""
        bases = self._reference.bases
        positions = np.flatnonzero(bases != "N") + 1
        n_sites = len(positions)
        references = bases[positions - 1]
        alleles = np.tile(np.asarray(BASES, dtype=object), n_sites)
        roles = np.where(
            alleles == np.repeat(references, len(BASES)), "reference", "alternative"
        ).astype(object)
        return SiteTable(
            contigs=(self._reference.contig,),
            site_contig=np.zeros(n_sites, dtype=np.int32),
            site_position=positions,
            site_name=np.full(n_sites, "", dtype=object),
            site_channel=np.full(n_sites, "", dtype=object),
            allele_site=np.repeat(np.arange(n_sites, dtype=np.int32), len(BASES)),
            allele=alleles,
            allele_role=roles,
        )

    def _site_of_position(self) -> np.ndarray:
        """Each position's site, -1 for a position of N; position 0 unused."""
        bases = self._reference.bases
        sites = np.full(len(bases) + 1, -1, dtype=np.int64)
        kept = np.flatnonzero(bases != "N")
        sites[kept + 1] = np.arange(len(kept))
        return sites

    def batches(self, chunk_rows: int) -> Iterator[EntryBatch]:
        """The depth table's cells, then the base files' counts in chunks.

        With several threads, the four files are parsed at once and their
        chunks taken in turn, A, C, G, T, A and so on, so the batches come in
        one order whatever the threads; each file has at most one chunk being
        parsed while its last one is used.
        """
        roles = dict(self.files)
        if "depth_table" in roles:
            yield self._listed_cells(roles["depth_table"])
        site_of = self._site_of_position()
        if self._threads < 2:
            for base_index, base in enumerate(BASES):
                for chunk in self._chunks(roles[base], chunk_rows):
                    yield self._counts(chunk, roles[base], base_index, site_of)
            return
        # Each file's parsed chunk and the one being parsed share the budget.
        rows = max(1, chunk_rows // (2 * len(BASES)))
        readers = [self._chunks(roles[base], rows) for base in BASES]
        done = object()
        with ThreadPoolExecutor(max_workers=min(self._threads, len(BASES))) as pool:
            pending = {
                index: pool.submit(next, reader, done)
                for index, reader in enumerate(readers)
            }
            try:
                while pending:
                    for index in list(pending):
                        chunk = pending[index].result()
                        if chunk is done:
                            del pending[index]
                            continue
                        pending[index] = pool.submit(next, readers[index], done)
                        base = BASES[index]
                        yield self._counts(chunk, roles[base], index, site_of)
            finally:
                for future in pending.values():
                    future.cancel()

    def _listed_cells(self, path: str) -> EntryBatch:
        table = pd.read_csv(
            path,
            sep="\t",
            header=None,
            dtype=str,
            keep_default_na=False,
            compression="gzip" if is_gzip(path) else None,
        )
        if table.shape[1] != 2:
            raise ValueError(
                f"{path} must hold a barcode and a mean depth on each line, not "
                f"{table.shape[1]} fields"
            )
        barcodes = pd.unique(table[0].str.strip().to_numpy(dtype=object))
        return EntryBatch(
            kind="cells",
            barcodes=np.asarray(barcodes, dtype=object),
            barcode_codes=np.arange(len(barcodes), dtype=np.int64),
        )

    def _chunks(self, path: str, chunk_rows: int) -> Iterator[pd.DataFrame]:
        if os.path.getsize(path) == 0 or not first_line(path):
            return
        dtypes: dict[str, Any] = {
            "position": np.int64,
            "barcode": "category",
            "forward": np.int64,
            "reverse": np.int64,
            "quality_forward": np.float64,
            "quality_reverse": np.float64,
        }
        reader = pd.read_csv(
            path,
            header=None,
            names=list(self._columns),
            dtype={name: dtypes[name] for name in self._columns},
            chunksize=chunk_rows,
            compression="gzip" if is_gzip(path) else None,
            na_filter=False,
        )
        iterator = iter(reader)
        while True:
            try:
                chunk = next(iterator)
            except StopIteration:
                return
            except (ValueError, pd.errors.ParserError) as error:
                raise ValueError(
                    f"{path} holds a row that is not {len(self._columns)} fields "
                    f"({', '.join(self._columns)}) with whole-number counts: {error}"
                ) from None
            yield chunk

    def _counts(
        self, chunk: pd.DataFrame, path: str, base_index: int, site_of: np.ndarray
    ) -> EntryBatch:
        positions = chunk["position"].to_numpy(dtype=np.int64)
        if len(positions) and (
            int(positions.min()) < 1 or int(positions.max()) > self._reference.length
        ):
            outside = positions[(positions < 1) | (positions > self._reference.length)]
            raise ValueError(
                f"{path} counts position {int(outside[0])}, outside the "
                f"{self._reference.length} positions of its reference; the base "
                "files and the reference come from different references"
            )
        forward = chunk["forward"].to_numpy(dtype=np.int64)
        reverse = chunk["reverse"].to_numpy(dtype=np.int64)
        if len(forward) and (int(forward.min()) < 0 or int(reverse.min()) < 0):
            raise ValueError(f"{path} holds a negative count")
        if len(forward) and max(int(forward.max()), int(reverse.max())) > MAX_COUNT:
            raise ValueError(f"{path} holds a count above {MAX_COUNT}")
        sites = site_of[positions]
        kept = sites >= 0
        fields: dict[str, np.ndarray] = {}
        if self._swap:
            forward, reverse = reverse, forward
        fields["forward"] = forward[kept]
        fields["reverse"] = reverse[kept]
        if "quality_forward" in self._columns:
            quality_forward = chunk["quality_forward"].to_numpy(dtype=np.float64)
            quality_reverse = chunk["quality_reverse"].to_numpy(dtype=np.float64)
            if self._swap:
                quality_forward, quality_reverse = quality_reverse, quality_forward
            # A strand without reads has no mean quality; the files write 0.
            fields["quality_forward"] = np.where(forward > 0, quality_forward, np.nan)[
                kept
            ]
            fields["quality_reverse"] = np.where(reverse > 0, quality_reverse, np.nan)[
                kept
            ]
        barcodes = chunk["barcode"]
        return EntryBatch(
            kind="counts",
            barcodes=np.asarray(barcodes.cat.categories, dtype=object),
            barcode_codes=barcodes.cat.codes.to_numpy(dtype=np.int64)[kept],
            keys=sites[kept] * len(BASES) + base_index,
            values=(forward + reverse)[kept],
            fields=fields,
        )


def mgatk_source(
    path: str | os.PathLike[str],
    *,
    format: str | None,
    swap_strands: bool,
    threads: int = 1,
) -> MgatkSource:
    """Describe an mgatk or maegatk folder as a source of allele counts.

    Args:
        path: The folder, such as mgatk's ``final``.
        format: ``"mgatk"`` or ``"maegatk"``, or None to take mgatk when the
            base files hold four columns.
        swap_strands: Swap the strands of mgatk files from ``call`` or
            ``bcall`` of mgatk 0.4.0 to 0.5.8.
        threads: Threads that parse the base files at once, up to four.

    Raises:
        FileNotFoundError: If a base file or the reference is missing.
        ValueError: If a file is malformed, the base files hold different
            columns, six-column files come without a format, or
            ``swap_strands`` is given with maegatk.
    """
    folder = os.fspath(path)
    base_files = [
        (base, find_file(folder, _base_pattern(base), f"{base} base file"))
        for base in BASES
    ]
    reference = find_file(folder, _REFERENCE, "reference file (<contig>_refAllele.txt)")
    depth_table = find_file(folder, _DEPTH_TABLE, "depth table", required=False)
    columns = {_columns(str(file)) for _base, file in base_files}
    if len(columns) > 1:
        raise ValueError(
            f"The base files of {folder} hold different columns; they must come "
            "from one run"
        )
    (layout,) = columns
    six = layout == _SIX
    if format is None:
        if six:
            raise ValueError(
                f"The base files of {folder} hold six columns, which mgatk with "
                "--emit-base-qualities and maegatk both write, counting strands "
                "the other way round; pass format='mgatk' or format='maegatk'"
            )
        format = "mgatk"
    if format not in MGATK_FORMATS:
        raise ValueError(f"format must be one of {', '.join(MGATK_FORMATS)}")
    if format == "maegatk" and swap_strands:
        raise ValueError(
            "maegatk's strands are swapped back on import; swap_strands applies to "
            "mgatk files from call or bcall of mgatk 0.4.0 to 0.5.8"
        )
    if format == "maegatk" and not six:
        raise ValueError(
            f"The base files of {folder} hold four columns, but maegatk always "
            "writes six"
        )
    files = [(base, str(file)) for base, file in base_files]
    files.append(("reference", str(reference)))
    if depth_table is not None:
        files.append(("depth_table", depth_table))
    swap = format == "maegatk" or swap_strands
    return MgatkSource(
        path=folder,
        format=format,
        files=tuple(files),
        count_unit="reads" if format == "mgatk" else "molecules",
        depth_definition="all",
        count_fields=("forward", "reverse")
        + (("quality_forward", "quality_reverse") if six else ()),
        depth_roles=None,
        reference_from_depth=False,
        open_alleles=False,
        notes={"columns": len(layout), "strands_swapped": swap},
        _reference=_read_reference(str(reference)),
        _columns=layout,
        _swap=swap,
        _threads=threads,
    )
