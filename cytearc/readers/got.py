"""Read the per-cell tables of genotyping of transcriptomes (GoT).

GoT amplifies a mutation's site from each cell's cDNA and counts, per cell,
the UMIs whose read shows the wild-type or the mutant sequence. The tools
that do so write one row per cell, or per cell and target:

- ``"ironthrone"``: IronThrone-GoT's ``<prefix>.summTable.txt``, one target
  per run: ``BC``, ``whitelist``, per-UMI lists such as ``call.in.dups``, and
  the UMIs of each call, ``WT.calls``, ``MUT.calls`` and ``amb.calls``. The
  per-UMI lists are not read; the file's SHA-256 keeps them traceable.
- ``"got"``: per-cell tables of one target, as studies deposit them on GEO
  or in their repositories: ``BC`` (or R's row names) with
  ``num.WT.call``/``num.MUT.call``, ``WT.calls``/``MUT.calls`` (with
  ``Total.calls``, checked to be their sum) or ``wtUMIs``/``mutUMIs``, and a
  ``Genotype`` call, or a call alone.
- ``"got_multi"``: GoT-Multi-ML's ``ironthrone_out_pp.csv``, one row per cell
  and target with the UMIs of each call in two evidence channels, the gene
  sequence (``geneseq``) and the probe barcode (``probebc``), the reads
  behind them, and ``genotype_geneseq``, ``genotype_probebc`` and
  ``genotype_merged``; and its ``got_multi_out_refined.csv``, whose
  ``genotype_final`` is read.
- ``"gotcha"``: Gotcha's (GoT-ChA) tables written to CSV: ``WTcount`` and
  ``MUTcount`` with ``genotype_pred``, or ``<gene>_WTcount`` and
  ``<gene>_MUTcount`` for several genes. Gotcha reads ATAC, which has no
  UMIs, so its counts are reads.

Each target is a site without coordinates, named after the table's
``target`` or gene, or after a one-target file's name without its
extensions and ``summTable``, with the alleles ``"WT"`` (the reference),
``"MUT"`` (the alternative) and, where counted, ``"ambiguous"`` (other).
GoT-Multi's channels are sites of their own, ``BCOR (geneseq)`` and
``BCOR (probebc)``, while its genotypes concern the target, ``BCOR``. Calls
are kept as the source wrote them, on the mutant allele, and ``NA`` or an
empty field means no call; empty or ``NA`` counts mean the cell was not
covered. These readers follow the tools' documentation and output files
only, since the tools carry no licence or a non-commercial one.
"""

import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..metadata.records import MISSING_TEXT, file_lines
from ..variants.counts import EntryBatch, SiteTable
from ._delimited import first_lines, text_chunks, text_layout
from ..utils.arguments import quoted_value

__all__ = ["GOT_FORMATS", "GotSource", "got_source", "got_table_format"]

#: The formats this module reads.
GOT_FORMATS: tuple[str, ...] = ("ironthrone", "got", "got_multi", "gotcha")
WILD_TYPE = "WT"
MUTANT = "MUT"
AMBIGUOUS = "ambiguous"
#: GoT-Multi's evidence channels.
CHANNELS: tuple[str, ...] = ("geneseq", "probebc")
# Rows of a GoT table parsed at once; a table holds one row per cell.
_CHUNK_ROWS = 1 << 16
# Columns that name each cell, by preference; R's row names come last.
_CELL_COLUMNS = ("BC", "CellBarcode", "WhiteListMatch", "cell", "barcode", "X")
_WILD_TYPE_COLUMNS = ("WT.calls", "num.WT.call", "wtUMIs")
_MUTANT_COLUMNS = ("MUT.calls", "num.MUT.call", "mutUMIs")
_CALL_COLUMNS = ("Genotype", "genotype")
_SUFFIXES = re.compile(r"(\.gz)?$")


@dataclass(frozen=True, slots=True)
class _Table:
    path: str
    frame: pd.DataFrame
    cells: np.ndarray


def _delimiter(line: str) -> str:
    return "\t" if "\t" in line else ","


def _read_table(path: str) -> _Table:
    """Read a GoT table as text, with its cells from a column or R's row names.

    The table is read as :mod:`~cytearc.readers._delimited` reads tables: a
    first row of one field more than the header holds R's row names, a table
    whose rows of one field more all end in a delimiter with nothing after it
    reads alike with a delimiter after each row's last field and is refused,
    and a row of fewer or more fields than the header is refused. A row whose
    every field is empty, as spreadsheets end a table, holds no cell.

    Raises:
        ValueError: If the file is empty, a row holds another number of fields
            than its header, its rows of one field more than the header all end
            in a delimiter with nothing after it, a column is named twice, no
            column names the cells, or a row has no cell barcode.
    """
    text = text_layout(path, what="a GoT table")
    names = list(text.names)
    parts = list(text_chunks(text, _CHUNK_ROWS))
    lines = np.concatenate(
        [part_lines for part_lines, _chunk in parts] or [np.empty(0, np.int64)]
    )
    columns = [
        np.concatenate(
            [chunk[position].str.strip().to_numpy(dtype=object) for _l, chunk in parts]
            or [np.empty(0, dtype=object)]
        )
        for position in range(len(names))
    ]
    filled = np.zeros(len(lines), dtype=bool)
    for values in columns:
        filled |= values != ""
    lines = lines[filled]
    frame = pd.DataFrame(
        {position: values[filled] for position, values in enumerate(columns)}
    )
    frame.columns = pd.Index(names, dtype=object)
    if frame.columns.duplicated().any():
        repeated = frame.columns[frame.columns.duplicated()][0]
        raise ValueError(f"{path} names the column {quoted_value(repeated)} twice")
    cell_column = next((name for name in _CELL_COLUMNS if name in frame.columns), None)
    if cell_column is None and "" in frame.columns:
        cell_column = ""
    if cell_column is None:
        raise ValueError(
            f"{path} has no column of cell barcodes; GoT tables name cells in "
            f"{', '.join(_CELL_COLUMNS)} or in R's row names"
        )
    cells = frame.iloc[:, names.index(cell_column)].to_numpy(dtype=object)
    empty = np.flatnonzero(cells == "")
    if len(empty):
        raise ValueError(
            f"{path} has rows without a cell barcode on {file_lines(lines[empty])}"
        )
    return _Table(path=path, frame=frame, cells=cells)


def _counts(table: _Table, column: str) -> np.ndarray:
    """A column of counts as numbers, NaN where the cell was not covered.

    Raises:
        ValueError: If a value is neither missing nor a whole number of 0 or
            more.
    """
    texts = table.frame[column].to_numpy(dtype=object)
    missing = np.asarray([text in MISSING_TEXT for text in texts.tolist()], dtype=bool)
    values = np.full(len(texts), np.nan)
    numbers = pd.to_numeric(pd.Series(texts[~missing]), errors="coerce").to_numpy(
        dtype=np.float64
    )
    whole = np.isfinite(numbers) & (numbers >= 0) & (numbers == np.floor(numbers))
    if not whole.all():
        bad = texts[~missing][~whole][0]
        raise ValueError(
            f"The column {quoted_value(column)} of {table.path} holds {quoted_value(bad)}; counts are whole "
            "numbers of 0 or more, or NA"
        )
    values[~missing] = numbers
    return values


def _calls(table: _Table, column: str) -> np.ndarray:
    texts = table.frame[column].to_numpy(dtype=object)
    return np.asarray(
        [None if text in MISSING_TEXT else text for text in texts.tolist()],
        dtype=object,
    )


def _target_name(path: str) -> str:
    """A one-target table's site: its file name without extensions and
    ``summTable``, such as ``Patient1_TET2.2847`` for
    ``Patient1_TET2.2847_summTable.txt``."""
    name = _SUFFIXES.sub("", os.path.basename(path))
    name = re.sub(r"\.(txt|tsv|csv)$", "", name)
    name = re.sub(r"[._]summTable$", "", name)
    if not name:
        raise ValueError(f"{path} has no name to name its target")
    return name


def got_table_format(path: str | os.PathLike[str]) -> str:
    """Recognise a GoT table by its columns.

    Raises:
        ValueError: If the columns fit no GoT format.
    """
    text = os.fspath(path)
    lines = first_lines(text, 1)
    header = lines[0] if lines else ""
    names = {name.strip().strip('"') for name in header.split(_delimiter(header))}
    if {"whitelist", "WT.calls", "call.in.dups"} <= names:
        return "ironthrone"
    if "target" in names and names & {"genotype_merged", "genotype_final"}:
        return "got_multi"
    if "WTcount" in names or any(name.endswith("_WTcount") for name in names):
        return "gotcha"
    if (
        names & set(_WILD_TYPE_COLUMNS) and names & set(_MUTANT_COLUMNS)
    ) or names & set(_CALL_COLUMNS):
        return "got"
    raise ValueError(
        f"{text} is not a table of mgatk, cellsnp-lite, VarTrix, giftwrap, MiTo or "
        "GoT counts that CyteArc recognises; pass format to name its format"
    )


@dataclass(frozen=True, slots=True)
class _Target:
    """One site of a GoT table and where its rows' values come from."""

    name: str
    channel: str
    alleles: tuple[str, ...]
    counts: dict[str, np.ndarray] = field(default_factory=dict)
    reads: dict[str, np.ndarray] = field(default_factory=dict)
    calls: dict[str, np.ndarray] = field(default_factory=dict)
    rows: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class GotSource:
    """A GoT table, read as a source of allele counts and calls."""

    path: str
    format: str
    files: tuple[tuple[str, str], ...]
    count_unit: str | None
    depth_definition: str
    count_fields: tuple[str, ...]
    call_fields: tuple[str, ...]
    depth_roles: frozenset[str] | None
    reference_from_depth: bool
    alleles_overlap: bool
    open_alleles: bool
    notes: dict[str, Any]
    _cells: np.ndarray = field(repr=False)
    _targets: tuple[_Target, ...] = field(repr=False)

    def sites(self) -> SiteTable:
        """Each target, or target and channel, with its call classes."""
        allele_site: list[int] = []
        alleles: list[str] = []
        roles: list[str] = []
        role_of = {WILD_TYPE: "reference", MUTANT: "alternative", AMBIGUOUS: "other"}
        for site, target in enumerate(self._targets):
            for allele in sorted(target.alleles):
                allele_site.append(site)
                alleles.append(allele)
                roles.append(role_of[allele])
        return SiteTable(
            contigs=(),
            site_contig=np.full(len(self._targets), -1, dtype=np.int32),
            site_position=np.full(len(self._targets), -1, dtype=np.int64),
            site_name=np.asarray(
                [target.name for target in self._targets], dtype=object
            ),
            site_channel=np.asarray(
                [target.channel for target in self._targets], dtype=object
            ),
            allele_site=np.asarray(allele_site, dtype=np.int32),
            allele=np.asarray(alleles, dtype=object),
            allele_role=np.asarray(roles, dtype=object),
        )

    def batches(self, chunk_rows: int) -> Iterator[EntryBatch]:
        """The listed cells, then each target's counts and calls."""
        distinct, codes = np.unique(self._cells.astype(str), return_inverse=True)
        barcodes = np.asarray(distinct, dtype=object)
        codes = codes.astype(np.int64)
        yield EntryBatch(
            kind="cells",
            barcodes=barcodes,
            barcode_codes=np.arange(len(barcodes), dtype=np.int64),
        )
        offset = 0
        for target in self._targets:
            # The source's alleles of a site follow in the order of their text.
            local = {
                allele: offset + position
                for position, allele in enumerate(sorted(target.alleles))
            }
            offset += len(target.alleles)
            rows = (
                np.arange(len(self._cells), dtype=np.int64)
                if target.rows is None
                else target.rows
            )
            for allele, values in target.counts.items():
                covered = np.isfinite(values)
                yield from _chunked(
                    EntryBatch(
                        kind="counts",
                        barcodes=barcodes,
                        barcode_codes=codes[rows[covered]],
                        keys=np.full(int(covered.sum()), local[allele], dtype=np.int64),
                        values=values[covered].astype(np.int64),
                        fields={"reads": target.reads[allele][covered].astype(np.int64)}
                        if allele in target.reads
                        else {},
                    ),
                    chunk_rows,
                )
            for source, calls in target.calls.items():
                called = np.asarray([call is not None for call in calls], dtype=bool)
                yield from _chunked(
                    EntryBatch(
                        kind="calls",
                        barcodes=barcodes,
                        barcode_codes=codes[rows[called]],
                        keys=np.full(int(called.sum()), local[MUTANT], dtype=np.int64),
                        call_source=source,
                        calls=calls[called],
                    ),
                    chunk_rows,
                )


def _chunked(batch: EntryBatch, chunk_rows: int) -> Iterator[EntryBatch]:
    for start in range(0, batch.n_rows, chunk_rows):
        stop = start + chunk_rows
        yield EntryBatch(
            kind=batch.kind,
            barcodes=batch.barcodes,
            barcode_codes=batch.barcode_codes[start:stop],
            keys=batch.keys[start:stop],
            values=batch.values[start:stop] if len(batch.values) else batch.values,
            fields={name: values[start:stop] for name, values in batch.fields.items()},
            call_source=batch.call_source,
            calls=batch.calls[start:stop] if len(batch.calls) else batch.calls,
        )


def _first(names: Sequence[str], table: _Table) -> str | None:
    return next((name for name in names if name in table.frame.columns), None)


def _ironthrone_targets(table: _Table) -> tuple[_Target, ...]:
    name = _target_name(table.path)
    return (
        _Target(
            name=name,
            channel="",
            alleles=(WILD_TYPE, MUTANT, AMBIGUOUS),
            counts={
                WILD_TYPE: _counts(table, "WT.calls"),
                MUTANT: _counts(table, "MUT.calls"),
                AMBIGUOUS: _counts(table, "amb.calls"),
            },
        ),
    )


def _got_targets(table: _Table) -> tuple[_Target, ...]:
    wild_type = _first(_WILD_TYPE_COLUMNS, table)
    mutant = _first(_MUTANT_COLUMNS, table)
    call = _first(_CALL_COLUMNS, table)
    if (wild_type is None) != (mutant is None):
        raise ValueError(
            f"{table.path} counts only one of the wild type and the mutant; GoT "
            "tables count both"
        )
    counts: dict[str, np.ndarray] = {}
    if wild_type is not None and mutant is not None:
        counts = {WILD_TYPE: _counts(table, wild_type), MUTANT: _counts(table, mutant)}
        if "Total.calls" in table.frame.columns:
            total = _counts(table, "Total.calls")
            both = np.isfinite(total)
            if not np.array_equal(
                total[both], counts[WILD_TYPE][both] + counts[MUTANT][both]
            ):
                raise ValueError(
                    f"Total.calls of {table.path} is not the sum of its WT and MUT "
                    "calls, so its depth is unclear"
                )
    calls = {call: _calls(table, call)} if call is not None else {}
    if not counts and not calls:
        raise ValueError(f"{table.path} holds neither GoT counts nor calls")
    return (
        _Target(
            name=_target_name(table.path),
            channel="",
            alleles=(WILD_TYPE, MUTANT),
            counts=counts,
            calls=calls,
        ),
    )


def _got_multi_targets(table: _Table) -> tuple[_Target, ...]:
    frame = table.frame
    targets = frame["target"].to_numpy(dtype=object)
    names = list(dict.fromkeys(targets.tolist()))
    if any(not name for name in names):
        raise ValueError(f"{table.path} has a row without a target")
    refined = "genotype_final" in frame.columns
    call_columns = (
        ("genotype_final",)
        if refined
        else tuple(
            name
            for name in ("genotype_geneseq", "genotype_probebc", "genotype_merged")
            if name in frame.columns
        )
    )
    channel_columns = {
        channel: {
            WILD_TYPE: f"wt_calls_{channel}",
            MUTANT: f"mut_calls_{channel}",
            AMBIGUOUS: f"amb_calls_{channel}",
        }
        for channel in CHANNELS
    }
    result: list[_Target] = []
    for name in names:
        rows = np.flatnonzero(targets == name)
        part = _Table(path=table.path, frame=frame.iloc[rows], cells=table.cells[rows])
        result.append(
            _Target(
                name=name,
                channel="",
                alleles=(WILD_TYPE, MUTANT),
                calls={column: _calls(part, column) for column in call_columns},
                rows=rows,
            )
        )
        if refined:
            continue
        for channel, columns in channel_columns.items():
            if not set(columns.values()) <= set(frame.columns):
                missing = sorted(set(columns.values()) - set(frame.columns))[0]
                raise ValueError(
                    f"{table.path} lacks the column {quoted_value(missing)}"
                )
            read_columns = {
                allele: f"{column.split('_calls_')[0]}_reads_per_umi_total_{channel}"
                for allele, column in columns.items()
            }
            present = [
                column for column in read_columns.values() if column in frame.columns
            ]
            if present and len(present) < len(read_columns):
                missing = sorted(set(read_columns.values()) - set(present))[0]
                raise ValueError(
                    f"{table.path} lacks the column {quoted_value(missing)}, which its other "
                    "reads_per_umi_total columns need"
                )
            counts = {
                allele: _counts(part, column) for allele, column in columns.items()
            }
            reads: dict[str, np.ndarray] = {}
            if present:
                for allele, reads_column in read_columns.items():
                    values = _counts(part, reads_column)
                    unknown = (counts[allele] > 0) & ~np.isfinite(values)
                    if unknown.any():
                        cell = part.cells[int(np.flatnonzero(unknown)[0])]
                        raise ValueError(
                            f"The column {quoted_value(reads_column)} of {table.path} is empty "
                            f"for the cell {quoted_value(cell)}, which has UMIs of that call"
                        )
                    # Reads are kept only beside UMIs; rows without UMIs are
                    # not stored.
                    reads[allele] = np.where(counts[allele] > 0, values, 0)
            result.append(
                _Target(
                    name=name,
                    channel=channel,
                    alleles=(WILD_TYPE, MUTANT, AMBIGUOUS),
                    counts=counts,
                    reads=reads,
                    rows=rows,
                )
            )
    return tuple(result)


def _gotcha_targets(table: _Table) -> tuple[_Target, ...]:
    columns = table.frame.columns
    if "WTcount" in columns:
        if "MUTcount" not in columns:
            raise ValueError(f"{table.path} has WTcount but no MUTcount")
        calls = (
            {"genotype_pred": _calls(table, "genotype_pred")}
            if ("genotype_pred" in columns)
            else {}
        )
        return (
            _Target(
                name=_target_name(table.path),
                channel="",
                alleles=(WILD_TYPE, MUTANT),
                counts={
                    WILD_TYPE: _counts(table, "WTcount"),
                    MUTANT: _counts(table, "MUTcount"),
                },
                calls=calls,
            ),
        )
    genes = [
        str(name)[: -len("_WTcount")]
        for name in columns
        if str(name).endswith("_WTcount")
    ]
    targets = []
    for gene in genes:
        if f"{gene}_MUTcount" not in columns:
            raise ValueError(f"{table.path} has {gene}_WTcount but no {gene}_MUTcount")
        targets.append(
            _Target(
                name=gene,
                channel="",
                alleles=(WILD_TYPE, MUTANT),
                counts={
                    WILD_TYPE: _counts(table, f"{gene}_WTcount"),
                    MUTANT: _counts(table, f"{gene}_MUTcount"),
                },
            )
        )
    return tuple(targets)


def got_source(path: str | os.PathLike[str], *, format: str) -> GotSource:
    """Describe a GoT table as a source of allele counts and calls.

    Args:
        path: The table, plain or gzip-compressed.
        format: One of ``GOT_FORMATS``.

    Raises:
        ValueError: If the table is malformed, lacks the columns of its
            format, or holds counts that are not whole numbers.
    """
    text = os.fspath(path)
    if format not in GOT_FORMATS:
        raise ValueError(
            f"format must be one of {', '.join(GOT_FORMATS)}, not {quoted_value(format)}"
        )
    table = _read_table(text)
    if format == "ironthrone":
        missing = [
            name
            for name in ("WT.calls", "MUT.calls", "amb.calls")
            if name not in table.frame
        ]
        if missing:
            raise ValueError(
                f"{text} lacks the summTable column {quoted_value(missing[0])}"
            )
        targets = _ironthrone_targets(table)
    elif format == "got":
        targets = _got_targets(table)
    elif format == "got_multi":
        if "target" not in table.frame.columns:
            raise ValueError(f"{text} lacks GoT-Multi's 'target' column")
        targets = _got_multi_targets(table)
    else:
        targets = _gotcha_targets(table)
        if not targets:
            raise ValueError(f"{text} holds no WTcount and MUTcount columns of Gotcha")
    ambiguous = format in ("ironthrone", "got_multi")
    reads = format == "got_multi" and any(target.reads for target in targets)
    return GotSource(
        path=text,
        format=format,
        files=(("table", text),),
        count_unit="reads" if format == "gotcha" else "umis",
        depth_definition="all" if ambiguous else "reference_and_alternative",
        count_fields=("reads",) if reads else (),
        call_fields=(),
        depth_roles=None,
        reference_from_depth=False,
        alleles_overlap=False,
        open_alleles=False,
        notes={},
        _cells=table.cells,
        _targets=targets,
    )
