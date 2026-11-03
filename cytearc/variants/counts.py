"""Per-cell allele evidence at variant sites, encoded for storage.

A site is where alleles are read: a position of a contig, such as a base of
the mitochondrial genome, or a target without coordinates, such as a GoT
target or a GIFT probe. Each site has alleles: the four bases of an mgatk
position, the reference and alternative of a VCF record, the gapfills of a
GIFT probe, or a GoT target's wild type and mutant. A cell's evidence at a
site is its depth there and its count of each allele, in the unit the source
counts (reads, UMIs or molecules), with strand counts, base qualities, reads
or support where the source has them, and the calls the source made itself.

Depth 0 means the cell was not covered at the site, so an allele frequency
there is undefined rather than 0. Entries are stored only where the depth or
count is above 0, sorted by cell, so a reader takes a band of cells at a
time. Tools define depth two ways, recorded per import: every read or UMI at
the site (``"all"``), or those of the reference and alternative alleles only
(``"reference_and_alternative"``).

Readers describe each source file's sites with a :class:`SiteTable` and yield
its rows as :class:`EntryBatch` values; :func:`merge_sites` gives the sites of
several files one order, and :func:`finish_band` turns a band of sorted rows
into the stored depth and counts. These functions never touch storage.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd

from ..utils.arrays import has_duplicates, sort_categories

__all__ = [
    "ALLELE_ROLES",
    "BASES",
    "CALL_DTYPE",
    "COUNT_FIELDS",
    "COUNT_UNITS",
    "DEPTH_DEFINITIONS",
    "DEPTH_DTYPE",
    "MAX_COUNT",
    "Contig",
    "CountUnit",
    "EntryBatch",
    "SiteTable",
    "VariantCounts",
    "VariantSource",
    "count_dtype",
    "finish_band",
    "merge_sites",
    "variant_count_frames",
]

type CountUnit = Literal["reads", "umis", "molecules"]

#: The units that sources count alleles in.
COUNT_UNITS: tuple[str, ...] = ("reads", "umis", "molecules")
#: What a source's depth counts: every read or UMI at the site, or only those
#: of the reference and alternative alleles.
DEPTH_DEFINITIONS: tuple[str, ...] = ("all", "reference_and_alternative")
#: What an allele is at its site: the reference base or wild type, an
#: alternative base or mutant, or reads that match neither, such as
#: cellsnp-lite's other bases, GoT's ambiguous UMIs and the gapfills that a
#: GIFT manifest does not name.
ALLELE_ROLES: tuple[str, ...] = ("reference", "alternative", "other")
#: The bases that mgatk counts at each position.
BASES: tuple[str, ...] = ("A", "C", "G", "T")
#: The optional fields of an allele count, by source: strand counts and mean
#: base qualities (mgatk, maegatk), and the reads behind the UMIs and the
#: share of reads that support the allele (giftwrap, GoT-Multi).
COUNT_FIELDS: Mapping[str, np.dtype[Any]] = {
    "forward": np.dtype(np.uint32),
    "reverse": np.dtype(np.uint32),
    "quality_forward": np.dtype(np.float32),
    "quality_reverse": np.dtype(np.float32),
    "reads": np.dtype(np.uint32),
    "support": np.dtype(np.float32),
}
#: The largest depth or count that storage holds.
MAX_COUNT = int(np.iinfo(np.uint32).max)
#: A depth row: the cell, the site and the depth.
DEPTH_DTYPE = np.dtype([("cell", np.int32), ("key", np.int32), ("depth", np.uint32)])
#: A call row: the cell, the allele the call is about, the source of the call
#: (a column of the file, such as GoT-Multi's ``genotype_final``) and the
#: call, both as codes of text, and the call's quality, such as Tapestri's
#: genotype quality, NaN where the source gives none.
CALL_DTYPE = np.dtype(
    [
        ("cell", np.int32),
        ("key", np.int32),
        ("source", np.int32),
        ("value", np.int32),
        ("quality", np.float32),
    ]
)
_ROLE_ORDER = {role: position for position, role in enumerate(ALLELE_ROLES)}
# Examples named in the errors about rows.
_EXAMPLES = 3


def count_dtype(fields: Sequence[str]) -> np.dtype[Any]:
    """The dtype of a count row: the cell, the allele, the count and ``fields``.

    Raises:
        ValueError: If a field is not one of ``COUNT_FIELDS``.
    """
    unknown = [name for name in fields if name not in COUNT_FIELDS]
    if unknown:
        raise ValueError(
            f"Count fields must be among {', '.join(COUNT_FIELDS)}, not {unknown}"
        )
    return np.dtype(
        [
            ("cell", np.int32),
            ("key", np.int32),
            ("count", np.uint32),
            *((name, COUNT_FIELDS[name]) for name in COUNT_FIELDS if name in fields),
        ]
    )


@dataclass(frozen=True, slots=True)
class Contig:
    """A contig that sites lie on.

    Attributes:
        name: The contig's name, such as ``"chrM"`` or ``"MT"``, or empty when
            the files do not name it.
        length: Its length in bases, or None when the files do not say.
        sequence_sha256: The SHA-256 of its uppercased sequence, or None when
            the files do not hold it.
    """

    name: str
    length: int | None = None
    sequence_sha256: str | None = None

    def record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "length": self.length,
            "sequence_sha256": self.sequence_sha256,
        }


def _text_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    for value in array.tolist():
        if not isinstance(value, str):
            raise TypeError(f"{name} must hold text, not {type(value).__name__}")
    return array


@dataclass(frozen=True, slots=True)
class SiteTable:
    """Sites and their alleles.

    A site has coordinates (a contig and a 1-based position) and no name, or
    a name and no coordinates; ``channel`` keeps apart the evidence channels
    of one named target, such as GoT-Multi's gene-sequence and probe-barcode
    reads, and is empty otherwise. Alleles are text: a base, a sequence, a
    gapfill (empty for a probe's empty gapfill) or a call class such as
    ``"WT"``. A site's coordinates or its name and channel identify it, and
    its alleles are unique.

    Attributes:
        contigs: The contigs that sites lie on.
        site_contig: Each site's contig, as a position in ``contigs``, -1 for
            a named site.
        site_position: Each site's 1-based position, -1 for a named site.
        site_name: Each site's name, empty for a site with coordinates.
        site_channel: Each site's evidence channel, usually empty.
        allele_site: Each allele's site.
        allele: Each allele's text.
        allele_role: Each allele's role, one of ``ALLELE_ROLES``.

    Raises:
        TypeError: If names, channels, alleles or roles are not text.
        ValueError: If the arrays do not align, a site has both or neither of
            coordinates and a name, a contig or site is out of range, a role
            is unknown, or a site or one of a site's alleles repeats.
    """

    contigs: tuple[Contig, ...]
    site_contig: np.ndarray
    site_position: np.ndarray
    site_name: np.ndarray
    site_channel: np.ndarray
    allele_site: np.ndarray
    allele: np.ndarray
    allele_role: np.ndarray

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "contigs", tuple(self.contigs))
        set_(self, "site_contig", np.asarray(self.site_contig, dtype=np.int32))
        set_(self, "site_position", np.asarray(self.site_position, dtype=np.int64))
        set_(self, "site_name", _text_array(self.site_name, "site names"))
        set_(self, "site_channel", _text_array(self.site_channel, "site channels"))
        set_(self, "allele_site", np.asarray(self.allele_site, dtype=np.int32))
        set_(self, "allele", _text_array(self.allele, "alleles"))
        set_(self, "allele_role", _text_array(self.allele_role, "allele roles"))
        n_sites = len(self.site_contig)
        if not (
            len(self.site_position)
            == len(self.site_name)
            == len(self.site_channel)
            == n_sites
        ):
            raise ValueError("The arrays of the sites must align")
        if not len(self.allele) == len(self.allele_role) == len(self.allele_site):
            raise ValueError("The arrays of the alleles must align")
        coordinates = self.site_contig >= 0
        named = np.asarray([bool(name) for name in self.site_name], dtype=bool)
        if np.any(coordinates & named) or np.any(~coordinates & ~named):
            raise ValueError(
                "Each site needs coordinates or a name, and not both: "
                f"{self._first_site(coordinates == named)}"
            )
        if np.any(self.site_contig >= len(self.contigs)) or np.any(
            self.site_contig < -1
        ):
            raise ValueError("A site names a contig that the table does not hold")
        if np.any(coordinates & (self.site_position < 1)) or np.any(
            ~coordinates & (self.site_position != -1)
        ):
            raise ValueError(
                "A site's position must be 1 or more, and -1 for a named site"
            )
        if np.any(self.allele_site < 0) or np.any(self.allele_site >= n_sites):
            raise ValueError("An allele names a site that the table does not hold")
        unknown = set(self.allele_role.tolist()) - set(ALLELE_ROLES)
        if unknown:
            raise ValueError(
                f"Allele roles must be among {', '.join(ALLELE_ROLES)}, not "
                f"{sorted(unknown)}"
            )
        keys = pd.Series(list(zip(*self._site_key_columns(), strict=True)))
        if keys.duplicated().any():
            repeated = int(np.flatnonzero(keys.duplicated().to_numpy())[0])
            raise ValueError(f"The site {self.label(repeated)} is listed twice")
        alleles = pd.DataFrame({"site": self.allele_site, "allele": self.allele})
        repeated_alleles = alleles.duplicated().to_numpy()
        if repeated_alleles.any():
            first = int(np.flatnonzero(repeated_alleles)[0])
            raise ValueError(
                f"The allele {self.allele[first]!r} of the site "
                f"{self.label(int(self.allele_site[first]))} is listed twice"
            )

    def _first_site(self, mask: np.ndarray) -> str:
        position = int(np.flatnonzero(mask)[0])
        return (
            f"contig {int(self.site_contig[position])}, position "
            f"{int(self.site_position[position])}, name "
            f"{self.site_name[position]!r}"
        )

    def _site_key_columns(self) -> tuple[list[str], list[int], list[str], list[str]]:
        contig_names = [contig.name for contig in self.contigs]
        return (
            [
                contig_names[code] if code >= 0 else ""
                for code in self.site_contig.tolist()
            ],
            self.site_position.tolist(),
            self.site_name.tolist(),
            self.site_channel.tolist(),
        )

    @property
    def n_sites(self) -> int:
        return len(self.site_contig)

    @property
    def n_alleles(self) -> int:
        return len(self.allele)

    def label(self, site: int) -> str:
        """Name a site as people read it: ``chrM:310``, ``IDH2_Mut1``, or with
        its channel, ``BCOR (geneseq)``."""
        code = int(self.site_contig[site])
        if code >= 0:
            contig = self.contigs[code].name
            text = (
                f"{contig}:{int(self.site_position[site])}"
                if contig
                else str(int(self.site_position[site]))
            )
        else:
            text = str(self.site_name[site])
        channel = str(self.site_channel[site])
        return f"{text} ({channel})" if channel else text

    def labels(self) -> np.ndarray:
        """Each site's label, as :meth:`label` names it."""
        return np.asarray(
            [self.label(site) for site in range(self.n_sites)], dtype=object
        )

    def allele_offsets(self) -> np.ndarray:
        """The alleles of site ``s``, when sorted by site, are ``[o[s], o[s + 1])``.

        Raises:
            ValueError: If the alleles are not sorted by site.
        """
        if len(self.allele_site) and np.any(np.diff(self.allele_site) < 0):
            raise ValueError("The alleles are not sorted by site")
        offsets = np.zeros(self.n_sites + 1, dtype=np.int64)
        np.cumsum(
            np.bincount(self.allele_site, minlength=self.n_sites), out=offsets[1:]
        )
        return offsets

    def reference_alleles(self) -> np.ndarray:
        """Each site's reference allele, -1 for a site without one.

        Raises:
            ValueError: If a site has two reference alleles.
        """
        references = np.full(self.n_sites, -1, dtype=np.int64)
        positions = np.flatnonzero(self.allele_role == "reference")
        sites = self.allele_site[positions]
        if has_duplicates(sites):
            repeated = int(sites[np.flatnonzero(np.diff(np.sort(sites)) == 0)[0]])
            raise ValueError(
                f"The site {self.label(repeated)} has two reference alleles"
            )
        references[sites] = positions
        return references


def _natural_rank(values: Sequence[str]) -> dict[str, int]:
    return {value: rank for rank, value in enumerate(sort_categories(set(values)))}


def _merged_contigs(tables: Sequence[SiteTable]) -> tuple[Contig, ...]:
    """The contigs of all tables, one per name, in natural order of names.

    Raises:
        ValueError: If two tables give one contig name different lengths or
            sequences, or one sequence two names, which would make each of
            its positions two sites.
    """
    by_name: dict[str, Contig] = {}
    for table in tables:
        for contig in table.contigs:
            seen = by_name.get(contig.name)
            if seen is None:
                by_name[contig.name] = contig
                continue
            if (
                seen.length is not None
                and contig.length is not None
                and seen.length != contig.length
            ) or (
                seen.sequence_sha256 is not None
                and contig.sequence_sha256 is not None
                and seen.sequence_sha256 != contig.sequence_sha256
            ):
                described = repr(contig.name) if contig.name else "the unnamed contig"
                raise ValueError(
                    f"The files hold {described} with different references "
                    f"({seen.length} and {contig.length} bases, or other bases), "
                    "so their positions do not name the same sites; import them "
                    "separately"
                )
            by_name[contig.name] = Contig(
                name=contig.name,
                length=seen.length if seen.length is not None else contig.length,
                sequence_sha256=(
                    seen.sequence_sha256
                    if seen.sequence_sha256 is not None
                    else contig.sequence_sha256
                ),
            )
    names = sort_categories(by_name)
    if "" in by_name and len(by_name) > 1:
        raise ValueError(
            "Some files name their contig and others do not, so their positions "
            "cannot be matched; import them separately"
        )
    by_sequence: dict[str, str] = {}
    for name in names:
        digest = by_name[name].sequence_sha256
        if digest is None:
            continue
        other = by_sequence.setdefault(digest, name)
        if other != name:
            raise ValueError(
                f"The files hold one reference sequence under the contig names "
                f"{other!r} and {name!r}, so each of its positions would be two "
                "sites; give it one name in every file, such as by naming mgatk's "
                "reference files <contig>_refAllele.txt alike"
            )
    return tuple(by_name[name] for name in names)


def merge_sites(
    tables: Sequence[SiteTable], *, open_alleles: bool
) -> tuple[SiteTable, list[np.ndarray], list[np.ndarray]]:
    """Give the sites and alleles of several files one table in one order.

    Sites with coordinates come first, by contig name in natural order and
    position, then named sites by name and channel; a site's alleles follow
    in the order of their text. A site that two files share is one site.

    Args:
        tables: Each file's sites, in that file's own order.
        open_alleles: Whether files may list different alleles of a site, as
            GIFT files list the gapfills they saw; otherwise a shared site
            must have the same alleles in every file, since an allele such as
            cellsnp-lite's other bases means something else beside another
            alternative.

    Returns:
        The merged table, and for each file the merged position of each of
        its sites and of each of its alleles.

    Raises:
        ValueError: If the files hold one contig with different references or
            one reference sequence under two names, mix named and unnamed
            contigs, give an allele two roles or, without ``open_alleles``,
            give a shared site different alleles.
    """
    if not tables:
        raise ValueError("No site table to merge")
    contigs = _merged_contigs(tables)
    contig_rank = {contig.name: rank for rank, contig in enumerate(contigs)}
    site_keys: dict[tuple[int, int, str, str], int] = {}
    per_table_keys: list[list[tuple[int, int, str, str]]] = []
    for table in tables:
        names = [contig.name for contig in table.contigs]
        keys = [
            (
                contig_rank[names[code]] if code >= 0 else -1,
                int(position),
                str(name),
                str(channel),
            )
            for code, position, name, channel in zip(
                table.site_contig.tolist(),
                table.site_position.tolist(),
                table.site_name.tolist(),
                table.site_channel.tolist(),
                strict=True,
            )
        ]
        per_table_keys.append(keys)
        for key in keys:
            site_keys.setdefault(key, len(site_keys))
    name_rank = _natural_rank([key[2] for key in site_keys])
    channel_rank = _natural_rank([key[3] for key in site_keys])
    ordered = sorted(
        site_keys,
        key=lambda key: (
            key[0] < 0,
            key[0],
            key[1],
            name_rank[key[2]],
            channel_rank[key[3]],
        ),
    )
    site_index = {key: position for position, key in enumerate(ordered)}
    site_maps = [
        np.asarray([site_index[key] for key in keys], dtype=np.int64)
        for keys in per_table_keys
    ]
    # The alleles of each merged site, with their roles and the files that
    # list them.
    alleles: dict[tuple[int, str], str] = {}
    listed: dict[int, list[frozenset[str]]] = {}
    for table, site_map in zip(tables, site_maps, strict=True):
        merged_sites = site_map[table.allele_site]
        per_site: dict[int, set[str]] = {}
        for site, allele, role in zip(
            merged_sites.tolist(),
            table.allele.tolist(),
            table.allele_role.tolist(),
            strict=True,
        ):
            seen = alleles.setdefault((site, allele), role)
            if seen != role:
                label = _merged_label(ordered[site], contigs)
                raise ValueError(
                    f"The files call the allele {allele!r} of the site {label} "
                    f"{seen} and {role}; import them separately"
                )
            per_site.setdefault(site, set()).add(allele)
        for site, texts in per_site.items():
            listed.setdefault(site, []).append(frozenset(texts))
    if not open_alleles:
        for site, sets in listed.items():
            if len(set(sets)) > 1:
                label = _merged_label(ordered[site], contigs)
                described = " and ".join(
                    "{" + ", ".join(sorted(texts)) + "}"
                    for texts in sorted(set(sets), key=sorted)
                )
                raise ValueError(
                    f"The files give the site {label} different alleles, "
                    f"{described}; import them separately"
                )
    allele_order = sorted(alleles, key=lambda key: (key[0], key[1]))
    allele_index = {key: position for position, key in enumerate(allele_order)}
    allele_maps = [
        np.asarray(
            [
                allele_index[(int(site), str(allele))]
                for site, allele in zip(
                    site_map[table.allele_site].tolist(),
                    table.allele.tolist(),
                    strict=True,
                )
            ],
            dtype=np.int64,
        )
        for table, site_map in zip(tables, site_maps, strict=True)
    ]
    merged = SiteTable(
        contigs=contigs,
        site_contig=np.asarray([key[0] for key in ordered], dtype=np.int32),
        site_position=np.asarray([key[1] for key in ordered], dtype=np.int64),
        site_name=np.asarray([key[2] for key in ordered], dtype=object),
        site_channel=np.asarray([key[3] for key in ordered], dtype=object),
        allele_site=np.asarray([key[0] for key in allele_order], dtype=np.int32),
        allele=np.asarray([key[1] for key in allele_order], dtype=object),
        allele_role=np.asarray([alleles[key] for key in allele_order], dtype=object),
    )
    return merged, site_maps, allele_maps


def _merged_label(key: tuple[int, int, str, str], contigs: tuple[Contig, ...]) -> str:
    contig, position, name, channel = key
    if contig >= 0:
        contig_name = contigs[contig].name
        text = f"{contig_name}:{position}" if contig_name else str(position)
    else:
        text = name
    return f"{text} ({channel})" if channel else text


type BatchKind = Literal["cells", "counts", "depth", "calls"]


@dataclass(frozen=True, slots=True)
class EntryBatch:
    """Rows of one source file that name cells by barcode.

    A batch holds one kind of row. ``"cells"`` rows list the cells that the
    file genotyped, such as mgatk's depth table, whether or not the cell has
    counts; ``"counts"`` rows give an allele's count with ``fields`` such as
    the strand counts; ``"depth"`` rows give a site's depth, for sources that
    state depth apart from their counts; ``"calls"`` rows give the call that
    the source made about an allele, as text, with its quality where the
    source gives one.

    Attributes:
        kind: The kind of row.
        barcodes: The distinct barcodes of the rows.
        barcode_codes: Each row's barcode, as a position in ``barcodes``.
        keys: Each row's allele (counts and calls) or site (depth), as a
            position in the source's own site table; empty for cells.
        values: Each row's count (counts) or depth (depth); empty otherwise.
        fields: Further columns of count rows, named as ``COUNT_FIELDS``.
        call_source: The source of the calls of a calls batch, such as the
            column that holds them.
        calls: Each row's call, as text, for a calls batch.
        call_quality: Each call's quality, such as Tapestri's genotype
            quality, NaN where the source gives none for the call; empty when
            the source gives no quality at all.
    """

    kind: BatchKind
    barcodes: np.ndarray
    barcode_codes: np.ndarray
    keys: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    values: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    fields: Mapping[str, np.ndarray] = field(default_factory=dict)
    call_source: str = ""
    calls: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))
    call_quality: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )

    @property
    def n_rows(self) -> int:
        return len(self.barcode_codes)


class VariantSource(Protocol):
    """One source of allele evidence, such as an mgatk folder or a GoT table.

    Attributes:
        path: The file or folder that the source was read from.
        format: Its format, such as ``"mgatk"``.
        files: Every file that the source reads, as (role, path) pairs.
        count_unit: The unit of its counts, or None when its files do not
            say, as for cellsnp-lite.
        depth_definition: What its depth counts, one of
            ``DEPTH_DEFINITIONS``.
        count_fields: The optional fields of its counts.
        depth_roles: The roles of the alleles that its stated depth counts,
            such as the reference and alternative of cellsnp-lite's DP, or
            None when its depth is the sum of its counts at the site.
        reference_from_depth: Whether its reference count is the stated depth
            less the other counts it covers, as for cellsnp-lite.
        alleles_overlap: Whether reads may count toward several alleles of a
            site, as Tapestri's allele frequencies of the variants at one
            position may add up to more than 100%, so that the stated depth
            bounds each count rather than their sum.
        open_alleles: Whether its alleles are the ones it saw, as GIFT's
            gapfills are, so files may differ in them.
        notes: What the import should record about the source, such as
            whether strands were swapped.
    """

    @property
    def path(self) -> str: ...

    @property
    def format(self) -> str: ...

    @property
    def files(self) -> tuple[tuple[str, str], ...]: ...

    @property
    def count_unit(self) -> str | None: ...

    @property
    def depth_definition(self) -> str: ...

    @property
    def count_fields(self) -> tuple[str, ...]: ...

    @property
    def depth_roles(self) -> frozenset[str] | None: ...

    @property
    def reference_from_depth(self) -> bool: ...

    @property
    def alleles_overlap(self) -> bool: ...

    @property
    def open_alleles(self) -> bool: ...

    @property
    def notes(self) -> Mapping[str, Any]: ...

    def sites(self) -> SiteTable: ...

    def batches(self, chunk_rows: int) -> Iterator[EntryBatch]: ...


def _row_label(
    cell: int, site: int, name_cell: Callable[[int], str], sites: SiteTable
) -> str:
    return f"cell {name_cell(cell)!r} at the site {sites.label(site)}"


def _run_sums(keys: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The first row of each run of equal sorted keys, and the run's sum."""
    if not len(keys):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint64)
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    return starts, np.add.reduceat(values.astype(np.uint64), starts)


def finish_band(
    counts: np.ndarray,
    depth: np.ndarray,
    *,
    sites: SiteTable,
    depth_roles: frozenset[str] | None,
    reference_from_depth: bool,
    name_cell: Callable[[int], str],
    alleles_overlap: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a band of sorted rows into the depth and counts that are stored.

    Rows with a count or depth of 0 are dropped. Without ``depth_roles``, a
    cell's depth at a site is the sum of its counts there. With them, the
    source states a depth that counts the alleles of those roles, such as
    cellsnp-lite's DP of the reference and alternative: their counts at a
    site must not exceed it, ``reference_from_depth`` derives the reference
    count as the depth less the other counts it covers, and the counts of
    the other roles, such as cellsnp-lite's other bases, add to the stored
    depth, so a stored depth counts every stored allele. With
    ``alleles_overlap``, as for Tapestri, whose variants at one position each
    give their allele frequency over the position's depth, each count must
    not exceed the stated depth, while their sum may.

    Args:
        counts: Count rows sorted by cell and allele, of :func:`count_dtype`.
        depth: Depth rows sorted by cell and site, of ``DEPTH_DTYPE``.
        sites: The sites and alleles that the rows' keys name.
        depth_roles: The roles that the stated depth counts, or None.
        reference_from_depth: Whether to derive the reference count.
        name_cell: Names a cell in errors.
        alleles_overlap: Whether the stated depth bounds each count of the
            roles it counts rather than their sum.

    Returns:
        Depth rows sorted by cell and site, and count rows sorted by cell and
        allele.

    Raises:
        ValueError: If counts are given at a site where the stated depth is
            0, exceed the stated depth, or add up to more than ``MAX_COUNT``,
            a site without a reference allele needs one, or overlapping
            alleles are asked to derive the reference count or to count
            without a stated depth.
    """
    if alleles_overlap and (depth_roles is None or reference_from_depth):
        raise ValueError(
            "Overlapping alleles need a stated depth and cannot derive the "
            "reference count from it"
        )
    counts = counts[counts["count"] > 0]
    stride = sites.n_sites + 1
    count_sites = sites.allele_site[counts["key"]].astype(np.int64)
    count_keys = counts["cell"].astype(np.int64) * stride + count_sites
    if depth_roles is None:
        starts, sums = _run_sums(count_keys, counts["count"])
        if len(sums) and int(sums.max()) > MAX_COUNT:
            position = int(starts[int(np.argmax(sums))])
            raise ValueError(
                "The counts of "
                + _row_label(
                    int(counts["cell"][position]),
                    int(count_sites[position]),
                    name_cell,
                    sites,
                )
                + f" add up to more than {MAX_COUNT}"
            )
        derived = np.empty(len(starts), dtype=DEPTH_DTYPE)
        derived["cell"] = counts["cell"][starts]
        derived["key"] = count_sites[starts]
        derived["depth"] = sums
        return derived, counts
    stated = depth[depth["depth"] > 0]
    stated_keys = stated["cell"].astype(np.int64) * stride + stated["key"].astype(
        np.int64
    )
    covered = np.isin(sites.allele_role[counts["key"]], list(depth_roles))
    at = np.searchsorted(stated_keys, count_keys)
    found = at < len(stated_keys)
    found[found] = stated_keys[at[found]] == count_keys[found]
    missing = covered & ~found
    if missing.any():
        position = int(np.flatnonzero(missing)[0])
        raise ValueError(
            "An allele is counted where the stated depth is 0: "
            + _row_label(
                int(counts["cell"][position]),
                int(count_sites[position]),
                name_cell,
                sites,
            )
        )
    totals = np.zeros(len(stated), dtype=np.uint64)
    if alleles_overlap:
        # Each count is checked alone: the largest of a site's counts.
        np.maximum.at(totals, at[covered], counts["count"][covered].astype(np.uint64))
    else:
        np.add.at(totals, at[covered], counts["count"][covered].astype(np.uint64))
    excess = totals > stated["depth"].astype(np.uint64)
    if excess.any():
        position = int(np.flatnonzero(excess)[0])
        raise ValueError(
            ("A count of " if alleles_overlap else "The counts of ")
            + _row_label(
                int(stated["cell"][position]),
                int(stated["key"][position]),
                name_cell,
                sites,
            )
            + (" is " if alleles_overlap else " add up to ")
            + f"{int(totals[position])}, more than its depth "
            f"{int(stated['depth'][position])}"
        )
    rows = [counts]
    if reference_from_depth:
        references = sites.reference_alleles()
        reference = references[stated["key"]]
        if np.any(reference < 0):
            position = int(np.flatnonzero(reference < 0)[0])
            raise ValueError(
                f"The site {sites.label(int(stated['key'][position]))} has no "
                "reference allele to count"
            )
        counted = np.isin(counts["key"], references[references >= 0])
        if counted.any():
            position = int(np.flatnonzero(counted)[0])
            raise ValueError(
                "The reference allele is counted and also derived from the depth: "
                + _row_label(
                    int(counts["cell"][position]),
                    int(count_sites[position]),
                    name_cell,
                    sites,
                )
            )
        remainder = stated["depth"].astype(np.uint64) - totals
        keep = remainder > 0
        derived = np.zeros(int(np.count_nonzero(keep)), dtype=counts.dtype)
        derived["cell"] = stated["cell"][keep]
        derived["key"] = reference[keep]
        derived["count"] = remainder[keep]
        for name in derived.dtype.names or ():
            if derived.dtype[name].kind == "f":
                derived[name] = np.nan
        rows.append(derived)
    stored_counts = np.concatenate(rows)
    stored_counts = stored_counts[
        np.lexsort((stored_counts["key"], stored_counts["cell"]))
    ]
    # The stored depth adds the counts that the stated depth leaves out.
    outside = ~covered
    extra_keys = np.concatenate([stated_keys, count_keys[outside]])
    extra_values = np.concatenate(
        [
            stated["depth"].astype(np.uint64),
            counts["count"][outside].astype(np.uint64),
        ]
    )
    order = np.argsort(extra_keys, kind="stable")
    starts, sums = _run_sums(extra_keys[order], extra_values[order])
    if len(sums) and int(sums.max()) > MAX_COUNT:
        raise ValueError(f"A depth adds up to more than {MAX_COUNT}")
    keys = extra_keys[order][starts]
    stored = np.empty(len(keys), dtype=DEPTH_DTYPE)
    stored["cell"] = keys // stride
    stored["key"] = keys % stride
    stored["depth"] = sums
    return stored, stored_counts


@dataclass(frozen=True, slots=True)
class VariantCounts:
    """Per-cell allele evidence at variant sites, read from a ``variant_counts``
    artifact.

    Long tables hold one row per stored entry, so a cell that a table does
    not list at a site was not covered there (``depth``), did not show the
    allele (``counts``) or has no call from the source (``calls``). A
    frequency is a count over the depth at its cell and site, and is
    undefined where the depth is 0.

    Attributes:
        sites: One row per site, indexed by label (such as ``chrM:310`` or
            ``IDH2_Mut1``): ``contig``, ``position`` (-1 for a named site),
            ``name``, ``channel``, ``reference`` (its reference allele, or
            None), ``depth`` (summed over all of the artifact's cells) and
            ``n_cells`` (cells covered there).
        alleles: One row per allele: ``site``, ``allele`` and ``role``.
        cells: One row per cell, indexed by cell ``ids``: ``listed`` (whether
            a file names the cell), ``mean_depth`` (over the sites, NaN for a
            cell no file names) and ``n_covered_sites``.
        depth: ``ids``, ``site`` and ``depth``, one row per covered cell and
            site.
        counts: ``ids``, ``site``, ``allele`` and ``count``, then the fields
            the source has: ``forward`` and ``reverse`` (strand counts),
            ``quality_forward`` and ``quality_reverse`` (mean base quality,
            NaN where the strand count is 0), ``reads`` and ``support``.
        calls: ``ids``, ``site``, ``allele``, ``source`` (the column that
            held the call), ``call``, as the source wrote it, and
            ``quality``, the call's quality where the source gives one, such
            as Tapestri's genotype quality ``GQ``, and NaN elsewhere.
        contigs: One row per contig: ``name``, ``length`` and
            ``sequence_sha256``.
        format: The format of the imported files, such as ``"mgatk"``.
        count_unit: ``"reads"``, ``"umis"`` or ``"molecules"``.
        depth_definition: ``"all"`` or ``"reference_and_alternative"``.
    """

    sites: pd.DataFrame = field(repr=False)
    alleles: pd.DataFrame = field(repr=False)
    cells: pd.DataFrame = field(repr=False)
    depth: pd.DataFrame = field(repr=False)
    counts: pd.DataFrame = field(repr=False)
    calls: pd.DataFrame = field(repr=False)
    contigs: pd.DataFrame = field(repr=False)
    format: str
    count_unit: str
    depth_definition: str


def variant_count_frames(
    *,
    sites: SiteTable,
    site_depth: np.ndarray,
    site_cells: np.ndarray,
    kept_sites: np.ndarray,
    cell_ids: np.ndarray,
    listed: np.ndarray,
    mean_depth: np.ndarray,
    n_covered_sites: np.ndarray,
    depth: tuple[np.ndarray, np.ndarray, np.ndarray],
    counts: tuple[np.ndarray, np.ndarray, Mapping[str, np.ndarray]],
    calls: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    call_sources: Sequence[str],
    call_values: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Return the tables of a :class:`VariantCounts` record.

    Args:
        sites: All of the artifact's sites and alleles.
        site_depth: Each site's depth summed over the artifact's cells.
        site_cells: Each site's covered cells.
        kept_sites: The sites to show, as positions in ``sites``, ascending.
        cell_ids: The IDs of the cells to show.
        listed: Whether a file names each cell.
        mean_depth: Each cell's mean depth.
        n_covered_sites: Each cell's covered sites.
        depth: Depth rows: cells (positions in ``cell_ids``), sites, depths.
        counts: Count rows: cells, alleles and the count and its fields.
        calls: Call rows: cells, alleles, sources and values as codes, and
            each call's quality.
        call_sources: The text of the source codes.
        call_values: The text of the value codes.
    """
    labels = sites.labels()
    shown = labels[kept_sites]
    site_categories = pd.CategoricalDtype(pd.Index(shown, dtype=object), ordered=True)
    cell_categories = pd.CategoricalDtype(
        pd.Index(np.asarray(cell_ids, dtype=object), dtype=object), ordered=True
    )
    site_rank = np.full(sites.n_sites, -1, dtype=np.int64)
    site_rank[kept_sites] = np.arange(len(kept_sites))

    def site_column(site_positions: np.ndarray) -> pd.Categorical:
        return pd.Categorical.from_codes(
            site_rank[np.asarray(site_positions, dtype=np.int64)],
            dtype=site_categories,
        )

    def cell_column(cell_positions: np.ndarray) -> pd.Categorical:
        return pd.Categorical.from_codes(
            np.asarray(cell_positions, dtype=np.int64), dtype=cell_categories
        )

    references = sites.reference_alleles()
    contig_names = [contig.name for contig in sites.contigs]
    sites_frame = pd.DataFrame(
        {
            "contig": [
                contig_names[code] if code >= 0 else None
                for code in sites.site_contig[kept_sites].tolist()
            ],
            "position": sites.site_position[kept_sites],
            "name": sites.site_name[kept_sites],
            "channel": sites.site_channel[kept_sites],
            "reference": [
                str(sites.allele[allele]) if allele >= 0 else None
                for allele in references[kept_sites].tolist()
            ],
            "depth": np.asarray(site_depth, dtype=np.int64)[kept_sites],
            "n_cells": np.asarray(site_cells, dtype=np.int64)[kept_sites],
        },
        index=pd.Index(shown, name="site", dtype=object),
    )
    allele_rows = np.flatnonzero(np.isin(sites.allele_site, kept_sites))
    alleles_frame = pd.DataFrame(
        {
            "site": site_column(sites.allele_site[allele_rows]),
            "allele": sites.allele[allele_rows],
            "role": sites.allele_role[allele_rows],
        }
    )
    cells_frame = pd.DataFrame(
        {
            "listed": np.asarray(listed, dtype=bool),
            "mean_depth": np.asarray(mean_depth, dtype=np.float64),
            "n_covered_sites": np.asarray(n_covered_sites, dtype=np.int64),
        },
        index=pd.Index(np.asarray(cell_ids, dtype=object), name="ids", dtype=object),
    )
    depth_cells, depth_sites, depth_values = depth
    depth_frame = pd.DataFrame(
        {
            "ids": cell_column(depth_cells),
            "site": site_column(depth_sites),
            "depth": np.asarray(depth_values, dtype=np.int64),
        }
    )
    count_cells, count_alleles, count_values = counts
    count_columns: dict[str, Any] = {
        "ids": cell_column(count_cells),
        "site": site_column(sites.allele_site[count_alleles]),
        "allele": sites.allele[count_alleles],
    }
    for name, values in count_values.items():
        array = np.asarray(values)
        count_columns[name] = (
            array.astype(np.float64)
            if array.dtype.kind == "f"
            else array.astype(np.int64)
        )
    counts_frame = pd.DataFrame(count_columns)
    call_cells, call_alleles, call_source_codes, call_value_codes, call_quality = calls
    calls_frame = pd.DataFrame(
        {
            "ids": cell_column(call_cells),
            "site": site_column(sites.allele_site[call_alleles]),
            "allele": sites.allele[call_alleles],
            "source": np.asarray(list(call_sources), dtype=object)[
                np.asarray(call_source_codes, dtype=np.int64)
            ]
            if len(call_source_codes)
            else np.empty(0, dtype=object),
            "call": np.asarray(list(call_values), dtype=object)[
                np.asarray(call_value_codes, dtype=np.int64)
            ]
            if len(call_value_codes)
            else np.empty(0, dtype=object),
            "quality": np.asarray(call_quality, dtype=np.float64),
        }
    )
    contigs_frame = pd.DataFrame(
        {
            "name": [contig.name for contig in sites.contigs],
            "length": pd.array(
                [contig.length for contig in sites.contigs], dtype="Int64"
            ),
            "sequence_sha256": [contig.sequence_sha256 for contig in sites.contigs],
        }
    )
    return {
        "sites": sites_frame,
        "alleles": alleles_frame,
        "cells": cells_frame,
        "depth": depth_frame,
        "counts": counts_frame,
        "calls": calls_frame,
        "contigs": contigs_frame,
    }
