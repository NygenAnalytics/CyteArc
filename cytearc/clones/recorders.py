"""Lineage recorder alleles: the edits that each cell's target arrays carry.

Lineage recorders such as DARLIN, PEtracer and DNA Typewriter write heritable
edits into target arrays in the genome, so cells that share an edit descend
from the cell in which it arose. Tools read each cell's arrays from
sequencing reads and write the edits as allele tables. An integration is one
copy of a target array, named as its tool names it: a DARLIN array (``CA``,
``TA`` or ``RA``), a Cassiopeia ``intBC``, a PEtracer ``intID``, a DNA
Typewriter ``TargetBC``. An integration's sites are of one of three kinds:

- ``"independent"``: sites edited independently, each with a fixed lookup
  of marks, as PEtracer writes one of eight insertions at each of its
  ``RNF2``, ``HEK3`` and ``EMX1`` sites; character matrices are read as
  independent sites too.
- ``"ordered"``: slots edited one after the other, as DNA Typewriter writes
  its tape: slot k is edited only after slots 1 to k-1, so an edited slot
  implies that every slot before it is edited too. Each integration has its
  own number of slots.
- ``"indel"``: the edits of a CRISPR target array as text, such as CARLIN's
  ``23_268del,55_248del`` over a whole DARLIN array, whose one site is named
  ``""``, or Cassiopeia's ``CGCCG[111:1I]AAAAAA`` at each cut site.

Each site of a cell is in one of five states, ``STATES``: unedited, a mark
(an edit the recorder names), an unknown mark (an edit that a lookup does not
name, or that the tool could not resolve), missing (not read) and not
applicable (a slot beyond a contracted array's length, or an integration
that a cell's embryo lacks). Each format writes the states with its own
tokens, which a :class:`SiteRules` gives.

An import keeps every site that the files state as a record of one long
table (:mod:`cytearc.metadata.records`), one per cell, integration, site and
allele row, ordered by cell, then file, then file row. A cell may hold
several allele rows of one integration, as the ``darlin`` CLI, Cassiopeia's
conflict mode and PEtracer write them. A record holds:

- ``integration``, ``site`` and ``allele`` as text stored once per distinct
  value; ``allele`` is the edit of a mark or an unknown mark as the file
  writes it, and missing for the other states, whose tokens are notation;
- ``state``, the position of its state in ``STATES``;
- ``umis`` and ``reads``, those of the allele row, which every site of the
  row shares, -1 where the file counts none;
- ``row``, the allele row's line in its file: in a table, the header is line
  1 and each row takes the next line, so blank lines and line breaks within
  quoted fields, which hold no row, do not count; in a CARLIN pair, the line
  of its allele, the same in both files; in a parquet table or a TreeData
  file, its position from 1.

These functions never touch storage.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..metadata.records import INTEGER, TEXT, EncodedText, FieldKind, RecordTable
from ..utils.arrays import sort_categories, sorted_unique

__all__ = [
    "FIELDS",
    "MARK",
    "MISSING",
    "NOT_APPLICABLE",
    "SITE_KINDS",
    "STATES",
    "UNEDITED",
    "UNKNOWN_MARK",
    "UNKNOWN_POLICIES",
    "AlleleRows",
    "CharacterMatrix",
    "LineageAlleles",
    "SiteKind",
    "SiteRules",
    "allele_chunk",
    "apply_prefix_rule",
    "cell_summaries",
    "character_matrix",
    "character_name",
    "lineage_allele_frames",
    "site_state_counts",
]

type SiteKind = Literal["independent", "ordered", "indel"]

#: The states of a site, by the code that records store.
STATES: tuple[str, ...] = (
    "unedited",
    "mark",
    "unknown_mark",
    "missing",
    "not_applicable",
)
UNEDITED, MARK, UNKNOWN_MARK, MISSING, NOT_APPLICABLE = range(len(STATES))
#: The kinds of site that recorders edit.
SITE_KINDS: tuple[SiteKind, ...] = ("independent", "ordered", "indel")
#: What an export does with unknown marks: give them states of their own, by
#: their text, or write them as missing.
UNKNOWN_POLICIES: tuple[str, ...] = ("state", "missing")
#: The fields of each allele record, with their kinds, in storage order.
FIELDS: Mapping[str, FieldKind] = MappingProxyType(
    {
        "integration": TEXT,
        "site": TEXT,
        "allele": TEXT,
        "state": INTEGER,
        "umis": INTEGER,
        "reads": INTEGER,
        "row": INTEGER,
    }
)
# The states in which a site was read.
_READ_STATES = (UNEDITED, MARK, UNKNOWN_MARK)


@dataclass(frozen=True, slots=True)
class SiteRules:
    """How a format writes each state of a site.

    A token is a site's text, stripped. It is checked against the sets in
    the order of the attributes, and a token that none names is a mark when
    it matches ``pattern`` and refused otherwise. With a lookup of marks, a
    token in the lookup is a mark and another that matches ``pattern`` an
    unknown mark.

    Attributes:
        kind: The kind of the sites, one of ``SITE_KINDS``.
        not_applicable: Tokens of a site that the integration lacks.
        missing: Tokens of a site that was not read.
        unedited: Tokens of an unedited site.
        unedited_at: Tokens of an unedited site at one site only, by site,
            as PEtracer's unedited ``EMX1`` reads ``CTTGGG``.
        unedited_within: Text that marks a token unedited wherever it
            appears in it, as Cassiopeia writes ``[None]`` with the
            sequence around an uncut site; None for no such text.
        unknown: Tokens of an unknown mark.
        ambiguous: A regular expression of tokens that are missing because
            they cannot be read, such as an insertion that holds ``N``, or
            None.
        marks: Tokens of the known marks at every site, or None.
        marks_at: Tokens of the known marks at each site, by site.
        pattern: A regular expression that every other token must match in
            full, or None for any text.
        absent: The state of a site whose field the file does not hold.

    Raises:
        ValueError: If ``kind`` is unknown, a token is in two sets of
            states, or ``pattern`` is not a regular expression.
    """

    kind: SiteKind
    not_applicable: frozenset[str] = frozenset()
    missing: frozenset[str] = frozenset()
    unedited: frozenset[str] = frozenset()
    unedited_at: Mapping[str, frozenset[str]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    unedited_within: str | None = None
    unknown: frozenset[str] = frozenset()
    ambiguous: str | None = None
    marks: frozenset[str] | None = None
    marks_at: Mapping[str, frozenset[str]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    pattern: str | None = None
    absent: int = MISSING

    def __post_init__(self) -> None:
        if self.kind not in SITE_KINDS:
            raise ValueError(
                f"kind must be one of {', '.join(map(repr, SITE_KINDS))}, not "
                f"{self.kind!r}"
            )
        named = {
            "not_applicable": self.not_applicable,
            "missing": self.missing,
            "unedited": self.unedited,
            "unknown": self.unknown,
        }
        seen: dict[str, str] = {}
        for state, tokens in named.items():
            for token in tokens:
                if token in seen:
                    raise ValueError(
                        f"The token {token!r} cannot mean both {seen[token]} and "
                        f"{state}"
                    )
                seen[token] = state
        for name in ("pattern", "ambiguous"):
            expression = getattr(self, name)
            if expression is None:
                continue
            try:
                re.compile(expression)
            except re.error as error:
                raise ValueError(
                    f"{name} {expression!r} is not a regular expression: {error}"
                ) from None
        if self.absent not in range(len(STATES)):
            raise ValueError("absent must be the code of a state")

    @property
    def has_lookup(self) -> bool:
        """Whether the rules name the known marks, so other edits are unknown."""
        return self.marks is not None or bool(self.marks_at)

    def state_of(self, token: str, site: str) -> int | None:
        """Return the state of one token at a site, or None for a refused token."""
        if token in self.not_applicable:
            return NOT_APPLICABLE
        if token in self.missing:
            return MISSING
        if token in self.unedited or token in self.unedited_at.get(site, ()):
            return UNEDITED
        if self.unedited_within is not None and self.unedited_within in token:
            return UNEDITED
        if token in self.unknown:
            return UNKNOWN_MARK
        if self.ambiguous is not None and re.fullmatch(self.ambiguous, token):
            return MISSING
        matches = self.pattern is None or re.fullmatch(self.pattern, token) is not None
        if self.has_lookup:
            known = (self.marks or frozenset()) | self.marks_at.get(site, frozenset())
            if token in known:
                return MARK
            return UNKNOWN_MARK if self.pattern is not None and matches else None
        return MARK if matches else None

    def classify(
        self, tokens: Any, site: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return each token's state, its allele text and whether it is refused.

        Each distinct token is classified once.

        Args:
            tokens: One token per row, stripped, None where the file holds
                no field.
            site: The site the tokens belong to.

        Returns:
            The state codes as int8, the allele of each mark and unknown mark
            (None for other states and refused tokens), and a mask of the
            refused tokens.
        """
        values = np.asarray(tokens, dtype=object).reshape(-1)
        codes, distinct = pd.factorize(values, use_na_sentinel=True)
        distinct = np.asarray(distinct, dtype=object)
        states = np.empty(len(distinct), dtype=np.int8)
        refused = np.zeros(len(distinct), dtype=bool)
        alleles = np.full(len(distinct), None, dtype=object)
        for position, token in enumerate(distinct.tolist()):
            state = self.state_of(str(token), site)
            if state is None:
                refused[position] = True
                states[position] = MISSING
                continue
            states[position] = state
            # A placeholder of an unknown mark, such as PEtracer's "!", names
            # no allele, so cells that hold it share none.
            if state == MARK or (state == UNKNOWN_MARK and token not in self.unknown):
                alleles[position] = str(token)
        present = codes >= 0
        row_states = np.full(len(values), self.absent, dtype=np.int8)
        row_states[present] = states[codes[present]]
        row_alleles = np.full(len(values), None, dtype=object)
        row_alleles[present] = alleles[codes[present]]
        row_refused = np.zeros(len(values), dtype=bool)
        row_refused[present] = refused[codes[present]]
        return row_states, row_alleles, row_refused


def apply_prefix_rule(
    states: np.ndarray, alleles: np.ndarray, lengths: np.ndarray
) -> int:
    """Apply the prefix rule of ordered slots to rows of slots, in place.

    A slot at or beyond its row's length is not applicable, whatever the
    file holds there. Before the last edited slot of a row, every slot is
    edited, so a slot that reads as unedited there was not read and is
    missing.

    Args:
        states: The state of each slot, rows by slots, as int8.
        alleles: The allele of each slot, rows by slots.
        lengths: The number of slots of each row's integration.

    Returns:
        The slots at or beyond their row's length that held an edit, which
        are not applicable now.
    """
    n_rows, n_slots = states.shape
    slots = np.arange(n_slots)
    beyond = slots[None, :] >= np.asarray(lengths, dtype=np.int64)[:, None]
    edited = (states == MARK) | (states == UNKNOWN_MARK)
    n_beyond = int(np.count_nonzero(edited & beyond))
    states[beyond] = NOT_APPLICABLE
    alleles[beyond] = None
    edited &= ~beyond
    has_edit = edited.any(axis=1)
    last = np.where(has_edit, n_slots - 1 - np.argmax(edited[:, ::-1], axis=1), -1)
    before = slots[None, :] < last[:, None]
    states[before & (states == UNEDITED)] = MISSING
    return n_beyond


@dataclass(frozen=True, slots=True)
class AlleleRows:
    """Rows of an allele file, with the token of each site of each row.

    A long table names each row's integration and holds a column per site;
    a wide matrix names the integration of each column.

    Attributes:
        barcodes: Each row's cell barcode, as the file writes it.
        rows: Each row's line in its file, as the records keep it: a
            table's header is line 1 and each row the next line, blank lines
            and line breaks within quoted fields uncounted; a CARLIN pair's
            line of the allele; or the position from 1 in a parquet table or
            a TreeData file.
        columns: The integration (None when rows name it) and the site of
            each column of ``tokens``.
        tokens: The stripped token of each row and column, None where the
            file holds no field, rows by columns.
        integrations: Each row's integration, None for a row that names
            none, or None when ``columns`` names them.
        umis: Each row's UMIs, -1 where unknown, or None.
        reads: Each row's reads, -1 where unknown, or None.
    """

    barcodes: np.ndarray
    rows: np.ndarray
    columns: tuple[tuple[str | None, str], ...]
    tokens: np.ndarray
    integrations: np.ndarray | None = None
    umis: np.ndarray | None = None
    reads: np.ndarray | None = None

    def __post_init__(self) -> None:
        n_rows = len(self.barcodes)
        if np.asarray(self.tokens).shape != (n_rows, len(self.columns)):
            raise ValueError("AlleleRows needs one token per row and column")
        named = [integration is not None for integration, _site in self.columns]
        if self.integrations is None and not all(named):
            raise ValueError("Rows that do not name integrations need columns that do")
        if self.integrations is not None and any(named):
            raise ValueError("Rows that name integrations need columns that do not")
        for name, values in (
            ("rows", self.rows),
            ("integrations", self.integrations),
            ("umis", self.umis),
            ("reads", self.reads),
        ):
            if values is not None and len(values) != n_rows:
                raise ValueError(f"AlleleRows needs one value of {name} per row")

    def __len__(self) -> int:
        return len(self.barcodes)


@dataclass(frozen=True, slots=True)
class ChunkStates:
    """The states of a chunk's sites, and what the rules found in them.

    Attributes:
        states: Each row's and column's state, as int8.
        alleles: Each row's and column's allele.
        n_beyond: Slots beyond their integration's length that held an edit.
    """

    states: np.ndarray
    alleles: np.ndarray
    n_beyond: int


def _column_groups(
    columns: Sequence[tuple[str | None, str]],
) -> list[tuple[str | None, np.ndarray]]:
    """The runs of columns that share an integration, with their positions."""
    groups: list[tuple[str | None, list[int]]] = []
    for position, (integration, _site) in enumerate(columns):
        if groups and groups[-1][0] == integration:
            groups[-1][1].append(position)
        else:
            groups.append((integration, [position]))
    return [(name, np.asarray(found, dtype=np.int64)) for name, found in groups]


class RefusedTokens(ValueError):
    """Tokens that the rules of a format do not read.

    Attributes:
        rows: The positions in the chunk of the rows that hold them.
        tokens: Up to five of the refused tokens, as their sites and texts.
    """

    def __init__(
        self, rows: Sequence[int], tokens: Sequence[tuple[str | None, str]]
    ) -> None:
        self.rows = tuple(int(row) for row in rows)
        self.tokens = tuple((site, token) for site, token in tokens)
        # The arguments rebuild the error when a pickle of it is loaded.
        super().__init__(self.rows, self.tokens)

    def __str__(self) -> str:
        return "tokens that the format does not read"


def chunk_states(
    rows: AlleleRows,
    rules: SiteRules,
    *,
    lengths: Mapping[str, int] | None = None,
    n_slots: int | None = None,
) -> ChunkStates:
    """Classify every token of a chunk and apply the prefix rule of ordered slots.

    Args:
        rows: The chunk.
        rules: The format's rules.
        lengths: For ordered slots, the number of slots of integrations
            shorter than ``n_slots``.
        n_slots: For ordered slots, the number of slots of the recorder's
            arrays.

    Raises:
        RefusedTokens: If a token is none that the rules read.
    """
    tokens = np.asarray(rows.tokens, dtype=object)
    n_rows, n_columns = tokens.shape
    states = np.empty((n_rows, n_columns), dtype=np.int8)
    alleles = np.empty((n_rows, n_columns), dtype=object)
    refused = np.zeros(n_rows, dtype=bool)
    examples: list[tuple[str, str]] = []
    for position, (_integration, site) in enumerate(rows.columns):
        column_states, column_alleles, column_refused = rules.classify(
            tokens[:, position], site
        )
        states[:, position] = column_states
        alleles[:, position] = column_alleles
        if column_refused.any():
            refused |= column_refused
            for token in pd.unique(tokens[column_refused, position]).tolist():
                if len(examples) < 5:
                    examples.append((site, str(token)))
    if refused.any():
        raise RefusedTokens(np.flatnonzero(refused).tolist(), examples)
    n_beyond = 0
    if rules.kind == "ordered":
        if n_slots is None:
            raise ValueError("Ordered slots need the number of slots of their arrays")
        shorter = {} if lengths is None else dict(lengths)
        for integration, positions in _column_groups(rows.columns):
            if integration is None:
                assert rows.integrations is not None
                row_lengths = np.asarray(
                    [
                        shorter.get(str(name), n_slots)
                        for name in rows.integrations.tolist()
                    ],
                    dtype=np.int64,
                )
            else:
                row_lengths = np.full(
                    n_rows, shorter.get(integration, n_slots), dtype=np.int64
                )
            block_states = states[:, positions]
            block_alleles = alleles[:, positions]
            n_beyond += apply_prefix_rule(block_states, block_alleles, row_lengths)
            states[:, positions] = block_states
            alleles[:, positions] = block_alleles
    return ChunkStates(states=states, alleles=alleles, n_beyond=n_beyond)


def allele_chunk(
    rows: AlleleRows, cells: np.ndarray, classified: ChunkStates
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the records of a chunk's rows of selected cells, for the encoder.

    Each kept row gives one record per column, in column order. A row that
    names no integration, None, is left out.

    Args:
        rows: The chunk.
        cells: Each row's cell among the selected cells, -1 for none.
        classified: The chunk's states, from :func:`chunk_states`.

    Returns:
        The records, with a column per field of ``FIELDS``, and each
        record's cell.
    """
    kept = np.asarray(cells) >= 0
    if rows.integrations is not None:
        kept &= ~np.asarray(pd.isna(rows.integrations), dtype=bool)
    keep = np.flatnonzero(kept)
    n_columns = len(rows.columns)
    n_kept = len(keep)
    if rows.integrations is not None:
        integrations = np.repeat(
            np.asarray(rows.integrations, dtype=object)[keep], n_columns
        )
    else:
        integrations = np.tile(
            np.asarray([name for name, _site in rows.columns], dtype=object), n_kept
        )
    sites = np.tile(
        np.asarray([site for _name, site in rows.columns], dtype=object), n_kept
    )

    def per_row(values: np.ndarray | None) -> np.ndarray:
        if values is None:
            return np.full(n_kept * n_columns, -1, dtype=np.int64)
        return np.repeat(np.asarray(values, dtype=np.int64)[keep], n_columns)

    records = pd.DataFrame(
        {
            "integration": integrations,
            "site": sites,
            "allele": classified.alleles[keep].reshape(-1),
            "state": classified.states[keep].reshape(-1).astype(np.int64),
            "umis": per_row(rows.umis),
            "reads": per_row(rows.reads),
            "row": per_row(rows.rows),
        }
    )
    return records, np.repeat(np.asarray(cells, dtype=np.int64)[keep], n_columns)


def records_in_line_order(table: RecordTable) -> RecordTable:
    """The records ordered by cell, then file, then file row.

    The encoder keeps the order in which each file's records arrive, and a
    table whose rows each hold one site gives them site by site within each
    chunk, so they are ordered again by their rows, whatever the chunks.
    """
    order = np.lexsort(
        (
            np.asarray(table.values["row"]),
            np.asarray(table.sources),
            np.asarray(table.cells),
        )
    )
    return RecordTable(
        cells=np.asarray(table.cells)[order],
        sources=np.asarray(table.sources)[order],
        kinds=table.kinds,
        text={
            name: EncodedText(
                codes=np.asarray(encoded.codes)[order],
                data=encoded.data,
                offsets=encoded.offsets,
            )
            for name, encoded in table.text.items()
        },
        ragged={},
        values={
            name: np.asarray(values)[order] for name, values in table.values.items()
        },
    )


def empty_allele_chunk() -> tuple[pd.DataFrame, np.ndarray]:
    """A chunk of no records that names every field, as ``allele_chunk`` returns one.

    An encoder that is given it returns a table of every field even when no
    file holds a row, as a file of a header alone does.
    """
    text = {
        name: np.empty(0, dtype=object) for name in ("integration", "site", "allele")
    }
    numbers = {
        name: np.empty(0, dtype=np.int64) for name in ("state", "umis", "reads", "row")
    }
    return pd.DataFrame({**text, **numbers}), np.empty(0, dtype=np.int64)


def cell_summaries(table: RecordTable, n_cells: int) -> dict[str, np.ndarray]:
    """Count each selected cell's edits, integrations read and records.

    Returns:
        ``n_edits``, the cell's sites that hold a mark or an unknown mark,
        counted over its allele rows; ``n_integrations``, its integrations
        with a site read; and ``n_records``.
    """
    cells = np.asarray(table.cells, dtype=np.int64)
    states = np.asarray(table.values["state"])
    edited = (states == MARK) | (states == UNKNOWN_MARK)
    read = np.isin(states, _READ_STATES)
    integrations = np.asarray(table.text["integration"].codes, dtype=np.int64)
    # Each cell and integration read is one key, counted once.
    n_integrations = max(1, len(table.text["integration"].offsets) - 1)
    keys = sorted_unique(cells[read] * n_integrations + integrations[read])
    return {
        "n_edits": np.bincount(cells[edited], minlength=n_cells).astype(np.int64),
        "n_integrations": np.bincount(keys // n_integrations, minlength=n_cells).astype(
            np.int64
        ),
        "n_records": np.bincount(cells, minlength=n_cells).astype(np.int64),
    }


def character_name(integration: str, site: str) -> str:
    """Name a character as Cassiopeia does, ``{integration}-{site}``.

    An integration whose one site has no name, such as a DARLIN array, is
    a character of its own name.
    """
    return integration if site == "" else f"{integration}-{site}"


def _nullable_counts(values: np.ndarray) -> pd.arrays.IntegerArray:
    counts = np.asarray(values, dtype=np.int64)
    missing = counts < 0
    return pd.arrays.IntegerArray(np.where(missing, 0, counts), missing)


@dataclass(frozen=True, slots=True)
class LineageAlleles:
    """The recorder alleles of a set of cells, from a ``lineage_alleles`` artifact.

    Attributes:
        cells: One row per selected cell, indexed by cell ``ids``:
            ``n_edits``, its sites that hold a mark or an unknown mark,
            ``n_integrations``, its integrations with a site read, and
            ``n_records``, its rows of ``records``.
        records: One row per cell, integration, site and allele row, by
            cell, file and file row: ``ids``, the cell, ``file``, ``integration``,
            ``site``, ``state`` (one of ``STATES``) and ``allele`` as
            categoricals, ``allele`` missing for states other than marks
            and unknown marks and for placeholders of unknown marks, such as
            PEtracer's ``!``, ``umis`` and ``reads`` as ``Int64``, missing
            where unknown, and ``row``, the allele row's line in its file,
            a table's blank lines and line breaks within quoted fields
            uncounted, or its position in a parquet table or TreeData file.
        files: One row per file, in canonical order: ``file`` (its
            position), ``format``, ``tool``, ``pipeline``, ``kind``,
            ``integration`` (the one integration of a file that holds one,
            else missing), ``n_rows`` and ``n_records``.
    """

    cells: pd.DataFrame = field(repr=False)
    records: pd.DataFrame = field(repr=False)
    files: pd.DataFrame = field(repr=False)

    def site_states(self) -> pd.DataFrame:
        """Count the cells in each state at each site.

        The cells counted are those with a record, the cells that the
        recorder reached. A cell counts once per site, in the first of these
        states that one of its records there holds: mark, unknown mark,
        unedited, missing, not applicable; a cell without a record at the
        site is missing there.

        Returns:
            ``integration``, ``site``, ``state`` and ``n_cells``, one row per
            integration, site and state.
        """
        records = self.records
        return _site_state_counts(
            records["ids"].cat.codes.to_numpy(),
            records["integration"].cat.codes.to_numpy(),
            records["site"].cat.codes.to_numpy(),
            records["state"].cat.codes.to_numpy(),
            np.asarray(records["integration"].cat.categories, dtype=object),
            np.asarray(records["site"].cat.categories, dtype=object),
        )


def lineage_allele_frames(
    cell_ids: Any,
    summaries: Mapping[str, np.ndarray],
    table: RecordTable,
    sources: Sequence[Mapping[str, Any]],
) -> LineageAlleles:
    """Build the tables of a ``LineageAlleles`` record.

    Args:
        cell_ids: The selected cells' IDs, in selection order.
        summaries: The per-cell arrays of :func:`cell_summaries`.
        table: The records.
        sources: The files, by position: ``format``, ``tool``,
            ``pipeline``, ``kind``, ``integration``, ``rows`` and
            ``records``.
    """
    ids = pd.Index(np.asarray(cell_ids, dtype=object), name="ids")
    cells = pd.DataFrame(
        {
            name: np.asarray(summaries[name], dtype=np.int64)
            for name in ("n_edits", "n_integrations", "n_records")
        },
        index=ids,
    )

    def text(name: str) -> pd.Categorical:
        encoded = table.text[name]
        return pd.Categorical.from_codes(
            np.asarray(encoded.codes, dtype=np.int64),
            categories=pd.Index(encoded.categories, dtype=object),
        )

    records = pd.DataFrame(
        {
            "ids": pd.Categorical.from_codes(
                np.asarray(table.cells, dtype=np.int64), categories=ids
            ),
            "file": pd.Categorical.from_codes(
                np.asarray(table.sources, dtype=np.int64),
                categories=pd.Index(range(len(sources)), dtype=np.int64),
            ),
            "integration": text("integration"),
            "site": text("site"),
            "state": pd.Categorical.from_codes(
                np.asarray(table.values["state"], dtype=np.int64),
                categories=pd.Index(STATES, dtype=object),
            ),
            "allele": text("allele"),
            "umis": _nullable_counts(table.values["umis"]),
            "reads": _nullable_counts(table.values["reads"]),
            "row": np.asarray(table.values["row"], dtype=np.int64),
        }
    )
    files = pd.DataFrame(
        {
            "file": pd.Series(range(len(sources)), dtype=np.int64),
            **{
                name: pd.Series([source.get(name) for source in sources], dtype=object)
                for name in ("format", "tool", "pipeline", "kind", "integration")
            },
            "n_rows": pd.Series(
                [int(source["rows"]) for source in sources], dtype=np.int64
            ),
            "n_records": pd.Series(
                [int(source["records"]) for source in sources], dtype=np.int64
            ),
        }
    )
    return LineageAlleles(cells=cells, records=records, files=files)


# Digits of a whole number that fits int64 whatever its digits, and the
# largest state that the matrix's int32 codes hold.
_MAX_NUMBER_DIGITS = 18
_MAX_STATE = 2**31 - 2


def _allele_ranks(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rank of each distinct allele, and its value as a state, or 0.

    Whole numbers from 1 that an int32 state holds rank first, by value, and
    are their own state; other text ranks after them by code point.
    """
    texts = np.asarray(values, dtype=object)
    # Only a number written as such, without leading zeros, is its own state,
    # so that 1 and 01 stay two alleles.
    numeric = np.asarray(
        [
            text.isdigit()
            and text.isascii()
            and not text.startswith("0")
            and len(text) <= _MAX_NUMBER_DIGITS
            for text in texts.tolist()
        ],
        dtype=bool,
    )
    numbers = np.zeros(len(texts), dtype=np.int64)
    numbers[numeric] = [int(text) for text in texts[numeric].tolist()]
    numeric &= (numbers >= 1) & (numbers <= _MAX_STATE)
    numbers[~numeric] = 0
    text_order = np.argsort(texts, kind="stable")
    text_ranks = np.empty(len(texts), dtype=np.int64)
    text_ranks[text_order] = np.arange(len(texts), dtype=np.int64)
    order = np.lexsort((text_ranks, numbers, ~numeric))
    ranks = np.empty(len(texts), dtype=np.int64)
    ranks[order] = np.arange(len(texts), dtype=np.int64)
    return ranks, numbers


def _integer_keys(*columns: tuple[np.ndarray, int]) -> np.ndarray:
    """One int64 key per row of integer columns, each below its bound.

    The keys sort as the rows do by their columns in turn, so one sort of the
    keys orders the rows, and dividing by the bounds recovers each column.

    Raises:
        ValueError: If the bounds' product passes the int64 range.
    """
    total = 1
    for _values, bound in columns:
        total *= max(1, int(bound))
    if total >= 2**63:
        raise ValueError(
            f"Columns of {' by '.join(str(bound) for _values, bound in columns)} "
            "values are too many to number as one integer key"
        )
    keys = np.zeros(len(columns[0][0]), dtype=np.int64)
    for values, bound in columns:
        keys = keys * max(1, int(bound)) + np.asarray(values, dtype=np.int64)
    return keys


# The order in which a cell's records at one site decide its state there, by
# state code: a mark first, then an unknown mark, unedited, missing and not
# applicable.
_STATE_PRIORITY = np.asarray([2, 0, 1, 3, 4], dtype=np.int64)


def _site_state_counts(
    cells: np.ndarray,
    integrations: np.ndarray,
    sites: np.ndarray,
    states: np.ndarray,
    integration_names: np.ndarray,
    site_names: np.ndarray,
) -> pd.DataFrame:
    """Count the cells in each state at each site from the codes of records.

    Each record's site, cell and rank form one integer key, so one sort of
    the keys finds each cell's lowest rank at each site.
    """
    cells = np.asarray(cells, dtype=np.int64)
    n_states = len(STATES)
    n_sites = max(1, len(site_names))
    n_pairs = max(1, len(integration_names)) * n_sites
    n_cells = int(cells.max()) + 1 if len(cells) else 1
    n_reached = int(np.count_nonzero(np.bincount(cells, minlength=1)))
    keys = sorted_unique(
        _integer_keys(
            (integrations, max(1, len(integration_names))),
            (sites, n_sites),
            (cells, n_cells),
            (_STATE_PRIORITY[np.asarray(states, dtype=np.int64)], n_states),
        )
    )
    # The first key of each cell at each site holds its lowest rank.
    groups = keys // n_states
    first = np.ones(len(keys), dtype=bool)
    first[1:] = groups[1:] != groups[:-1]
    chosen = keys[first]
    del keys, groups, first
    counts = np.bincount(
        (chosen // n_states // n_cells) * n_states + chosen % n_states,
        minlength=n_pairs * n_states,
    ).reshape(n_pairs, n_states)
    named = {
        (
            str(integration_names[pair // n_sites]),
            str(site_names[pair % n_sites]),
        ): counts[pair][_STATE_PRIORITY].copy()
        for pair in np.flatnonzero(counts.sum(axis=1)).tolist()
    }
    rows: list[tuple[str, str, str, int]] = []
    for integration, site in sort_categories(named):
        found = named[(integration, site)]
        # A cell that the recorder reached without a record at the site was
        # not read there.
        found[MISSING] += n_reached - int(found.sum())
        rows.extend(
            (integration, site, STATES[state], int(found[state]))
            for state in range(len(STATES))
        )
    return pd.DataFrame(
        {
            "integration": pd.Series([row[0] for row in rows], dtype=object),
            "site": pd.Series([row[1] for row in rows], dtype=object),
            "state": pd.Series([row[2] for row in rows], dtype=object),
            "n_cells": pd.Series([row[3] for row in rows], dtype=np.int64),
        }
    )


def site_state_counts(table: RecordTable) -> pd.DataFrame:
    """Count the cells in each state at each site.

    The cells counted are those with a record, the cells that the recorder
    reached. A cell counts once per site, in the first of these states that
    one of its records there holds: mark, unknown mark, unedited, missing,
    not applicable; a cell without a record at the site is missing there.

    Returns:
        One row per integration, site and state: ``integration``, ``site``,
        ``state`` and ``n_cells``, by integration and site in natural order,
        then state in ``STATES`` order.
    """
    return _site_state_counts(
        table.cells,
        table.text["integration"].codes,
        table.text["site"].codes,
        table.values["state"],
        table.text["integration"].categories,
        table.text["site"].categories,
    )


@dataclass(frozen=True, slots=True)
class CharacterMatrix:
    """Cells by characters of integer states, for tree builders.

    A character is one site of one integration. A state is 0 for an
    unedited site and -1 for a missing one. A mark written as a whole number
    from 1, such as PEtracer's marks ``1`` to ``8`` or the state of a
    character matrix, is its own state; the character's other marks, by
    their text, then the unknown marks that ``unknown="state"`` keeps, take
    the states after the largest such number. So each state names one
    allele of its character, in every cell, and a matrix's states survive an
    export that reads it back.

    Attributes:
        cells: The position of each exported cell among the selected cells,
            in ascending order.
        integrations: Each character's integration.
        sites: Each character's site.
        codes: The state of each cell and character, as int32.
        states: One row per character and edited state: ``character``,
            ``state``, ``allele``, ``kind`` (``"mark"`` or
            ``"unknown_mark"``) and ``n_cells``, the exported cells in it.
        n_resolved: Cells and integrations that several allele rows give,
            of which the export kept one.
        n_without_states: Cells with records whose every exported state is
            missing, which are left out.
    """

    cells: np.ndarray
    integrations: tuple[str, ...]
    sites: tuple[str, ...]
    codes: np.ndarray = field(repr=False)
    states: pd.DataFrame = field(repr=False)
    n_resolved: int
    n_without_states: int

    @property
    def characters(self) -> tuple[str, ...]:
        """Each character's name, as :func:`character_name` writes it."""
        return tuple(
            character_name(integration, site)
            for integration, site in zip(self.integrations, self.sites, strict=True)
        )

    def priors(self) -> pd.DataFrame:
        """Each edited state's share of the exported cells with an edit at its character.

        Returns:
            ``character`` (its position among the characters), ``state`` and
            ``probability``.
        """
        names = {name: position for position, name in enumerate(self.characters)}
        states = self.states
        totals = states.groupby("character", sort=False)["n_cells"].transform("sum")
        return pd.DataFrame(
            {
                "character": states["character"].map(names).astype(np.int64),
                "state": states["state"].astype(np.int64),
                "probability": states["n_cells"].to_numpy(dtype=np.float64)
                / totals.to_numpy(dtype=np.float64),
            }
        )


def _group_starts(*keys: np.ndarray) -> np.ndarray:
    """Whether each row of sorted keys starts a new run of equal keys."""
    n_rows = len(keys[0])
    starts = np.zeros(n_rows, dtype=bool)
    if n_rows:
        starts[0] = True
        for key in keys:
            starts[1:] |= key[1:] != key[:-1]
    return starts


def _state_codes(characters: np.ndarray, numbers: np.ndarray) -> np.ndarray:
    """The state of each distinct edit, sorted by character, kind and rank.

    An edit whose number is above 0 keeps it; the others of its character
    follow the largest such number, in order.
    """
    n_edits = len(characters)
    codes = np.asarray(numbers, dtype=np.int64).copy()
    if not n_edits:
        return codes
    starts = _group_starts(characters)
    runs = np.cumsum(starts) - 1
    largest = np.zeros(int(runs[-1]) + 1, dtype=np.int64)
    np.maximum.at(largest, runs, codes)
    unnumbered = codes == 0
    # The position of each unnumbered edit among its character's.
    counts = np.cumsum(unnumbered) - unnumbered
    first_counts = counts[np.flatnonzero(starts)]
    codes[unnumbered] = (largest[runs] + 1 + counts - first_counts[runs])[unnumbered]
    return codes


def character_matrix(
    table: RecordTable,
    *,
    unknown: str = "state",
    memory_bytes: int | None = None,
    held_bytes: int = 0,
) -> CharacterMatrix:
    """Turn allele records into a cells by characters matrix of integer states.

    Where the files give a cell several allele rows of one integration, the
    export keeps the row of most UMIs, then most reads, then most sites read,
    then the first in canonical file order, as Cassiopeia keeps each
    integration's most abundant allele. Missing and not applicable sites
    are missing, and so are unknown marks with ``unknown="missing"``.
    Characters that no exported cell reads, and cells that read no
    character, are left out.

    Args:
        table: The allele records.
        unknown: ``"state"`` gives each distinct unknown mark of a character
            a state of its own, after the marks; ``"missing"`` writes unknown
            marks as missing.
        memory_bytes: The memory budget, or None for no limit.
        held_bytes: The bytes that the caller holds beside the matrix's
            codes, 4 per cell and character.

    Raises:
        ValueError: If ``unknown`` is not one of ``UNKNOWN_POLICIES``.
        MemoryError: If the codes and ``held_bytes`` would pass
            ``memory_bytes``, before the codes are allocated.
    """
    if unknown not in UNKNOWN_POLICIES:
        raise ValueError(
            f"unknown must be one of {', '.join(map(repr, UNKNOWN_POLICIES))}, not "
            f"{unknown!r}"
        )
    cells = np.asarray(table.cells, dtype=np.int64)
    sources = np.asarray(table.sources, dtype=np.int64)
    integrations = np.asarray(table.text["integration"].codes, dtype=np.int64)
    sites = np.asarray(table.text["site"].codes, dtype=np.int64)
    alleles = np.asarray(table.text["allele"].codes, dtype=np.int64)
    states = np.asarray(table.values["state"], dtype=np.int64)
    rows = np.asarray(table.values["row"], dtype=np.int64)
    read = np.isin(states, _READ_STATES)

    # Each allele row of a cell and integration is a group of records.
    order = np.lexsort((rows, sources, integrations, cells))
    group_starts = _group_starts(
        cells[order], integrations[order], sources[order], rows[order]
    )
    groups = np.empty(len(order), dtype=np.int64)
    groups[order] = np.cumsum(group_starts) - 1
    n_groups = int(np.count_nonzero(group_starts))
    first = order[group_starts]
    group_umis = np.full(n_groups, -1, dtype=np.int64)
    np.maximum.at(group_umis, groups, np.asarray(table.values["umis"], np.int64))
    group_reads = np.full(n_groups, -1, dtype=np.int64)
    np.maximum.at(group_reads, groups, np.asarray(table.values["reads"], np.int64))
    group_read = np.bincount(groups[read], minlength=n_groups)
    # The kept row of each cell and integration ranks first.
    ranked = np.lexsort(
        (
            rows[first],
            sources[first],
            -group_read,
            -group_reads,
            -group_umis,
            integrations[first],
            cells[first],
        )
    )
    pair_starts = _group_starts(cells[first][ranked], integrations[first][ranked])
    kept_groups = np.zeros(n_groups, dtype=bool)
    kept_groups[ranked[pair_starts]] = True
    run_lengths = np.diff(np.flatnonzero(np.append(pair_starts, True)))
    n_resolved = int(np.count_nonzero(run_lengths > 1))
    kept = kept_groups[groups]

    edited = states == MARK
    if unknown == "state":
        # An unknown mark that names its allele, such as an insertion outside
        # a lookup, is a state; a placeholder names none and is missing.
        edited |= (states == UNKNOWN_MARK) & (alleles >= 0)
    exported = kept & (edited | (states == UNEDITED))
    # The characters that an exported record reads, in natural order.
    integration_names = table.text["integration"].categories
    site_names = table.text["site"].categories
    n_sites = max(len(site_names), 1)
    pairs = sorted_unique(
        _integer_keys(
            (integrations[exported], max(len(integration_names), 1)),
            (sites[exported], n_sites),
        )
    )
    pairs = np.stack([pairs // n_sites, pairs % n_sites], axis=1)
    named = [
        (str(integration_names[integration]), str(site_names[site]))
        for integration, site in pairs.tolist()
    ]
    ranked_names = sort_categories(named)
    rank_of = {pair: rank for rank, pair in enumerate(ranked_names)}
    character_of = np.full(
        (max(len(integration_names), 1), max(len(site_names), 1)), -1, dtype=np.int64
    )
    for (integration, site), pair in zip(pairs.tolist(), named, strict=True):
        character_of[integration, site] = rank_of[pair]
    record_characters = np.full(len(states), -1, dtype=np.int64)
    record_characters[exported] = character_of[integrations[exported], sites[exported]]

    # Each character's edits: a mark written as a whole number is its own
    # state; the other marks, by their text, then the unknown marks that are
    # kept, take the states after the largest such number.
    allele_names = table.text["allele"].categories
    allele_ranks, allele_numbers = _allele_ranks(allele_names)
    edits = exported & edited
    edit_alleles = alleles[edits]
    edit_kinds = (states[edits] == UNKNOWN_MARK).astype(np.int64)
    # Each edit's character, kind and allele rank as one key, ordered as
    # those columns are, so that one stable sort finds the distinct edits.
    n_ranks = max(len(allele_names), 1)
    edit_keys = _integer_keys(
        (record_characters[edits], len(ranked_names)),
        (edit_kinds, 2),
        (allele_ranks[edit_alleles], n_ranks),
    )
    edit_order = np.argsort(edit_keys, kind="stable")
    edit_starts = _group_starts(edit_keys[edit_order])
    distinct_keys = edit_keys[edit_order][edit_starts]
    distinct = np.stack(
        [
            distinct_keys // (2 * n_ranks),
            distinct_keys // n_ranks % 2,
            distinct_keys % n_ranks,
        ],
        axis=1,
    ).reshape(-1, 3)
    first_edit = edit_order[edit_starts]
    inverse = np.empty(len(edit_keys), dtype=np.int64)
    inverse[edit_order] = np.cumsum(edit_starts) - 1
    del edit_keys, edit_order, edit_starts, distinct_keys
    distinct_numbers = np.where(
        distinct[:, 1] == 0, allele_numbers[edit_alleles[first_edit]], 0
    )
    state_codes = _state_codes(distinct[:, 0], distinct_numbers)
    if len(state_codes) and int(state_codes.max()) > _MAX_STATE + 1:
        position = int(np.argmax(state_codes))
        raise ValueError(
            f"The states of the character {character_name(*ranked_names[int(distinct[position, 0])])!r} "
            f"pass {_MAX_STATE + 1}, the largest that an int32 matrix holds; its "
            "edits follow marks numbered up to it"
        )
    record_codes = np.full(len(states), -1, dtype=np.int64)
    record_codes[exported & (states == UNEDITED)] = 0
    record_codes[edits] = state_codes[inverse]

    # The cells that read at least one character, in ascending order; each
    # kept allele row names a site once, so a cell has one record per
    # character.
    with_state = np.flatnonzero(record_codes >= 0)
    exported_cells = sorted_unique(cells[with_state])
    held = int(held_bytes) + 4 * len(exported_cells) * len(ranked_names)
    if memory_bytes is not None and held > memory_bytes:
        raise MemoryError(
            f"Exporting the character matrix of {len(exported_cells)} cells by "
            f"{len(ranked_names)} characters holds about {held} bytes, more than "
            f"the memory budget of {memory_bytes} bytes. Raise mem_budget."
        )
    codes = np.full((len(exported_cells), len(ranked_names)), -1, dtype=np.int32)
    codes[
        np.searchsorted(exported_cells, cells[with_state]),
        record_characters[with_state],
    ] = record_codes[with_state]

    characters = [character_name(*pair) for pair in ranked_names]
    if len(set(characters)) != len(characters):
        seen: dict[str, tuple[str, str]] = {}
        for name, pair in zip(characters, ranked_names, strict=True):
            if name in seen:
                raise ValueError(
                    f"The integration {seen[name][0]!r} at site {seen[name][1]!r} and "
                    f"{pair[0]!r} at {pair[1]!r} are both the character {name!r}; "
                    "rename an integration so that each character has its own name"
                )
            seen[name] = pair
    allele_of_rank = np.empty(len(allele_names), dtype=object)
    allele_of_rank[allele_ranks] = allele_names
    state_table = pd.DataFrame(
        {
            "character": pd.Series(
                [characters[int(column)] for column in distinct[:, 0]], dtype=object
            ),
            "state": pd.Series(state_codes, dtype=np.int64),
            "allele": pd.Series(allele_of_rank[distinct[:, 2]], dtype=object),
            "kind": pd.Series(
                np.where(distinct[:, 1] == 0, "mark", "unknown_mark"), dtype=object
            ),
            "n_cells": pd.Series(
                np.bincount(inverse, minlength=len(distinct)), dtype=np.int64
            ),
        }
    )
    # By character, then state.
    table_order = np.lexsort((state_codes, distinct[:, 0]))
    state_table = state_table.iloc[table_order].reset_index(drop=True)
    return CharacterMatrix(
        cells=exported_cells,
        integrations=tuple(pair[0] for pair in ranked_names),
        sites=tuple(pair[1] for pair in ranked_names),
        codes=codes,
        states=state_table,
        n_resolved=n_resolved,
        n_without_states=int(len(sorted_unique(cells[kept])) - len(exported_cells)),
    )
