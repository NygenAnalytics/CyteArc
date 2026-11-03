"""Read MiTo's allele frequency matrix of mitochondrial variants.

MiTo assembles the counts of maegatk, mgatk or RedeemV into an AnnData file
of cells by variants: ``layers['AD']`` holds each cell's count of the
alternative base, sparse, and ``layers['DP']`` the coverage of the variant's
site, dense and defined for every cell; ``var`` holds each variant's
``pos``, ``ref`` and ``alt``, named ``<pos>_<ref>><alt>``, and cells are
named ``{barcode}_{sample}``. A position with several alternatives is one
site whose depth is the largest DP of its variants, since MiTo raises a
variant's DP to its AD where that exceeds the coverage. The reference base
is not counted. The file names no contig, so the contig is unnamed.

The counts are molecules when ``uns`` says so: MiTo 2 records
``uns['molecules']`` when it turned mtscATAC reads into molecules, and
``uns['pp_method']`` names maegatk, whose counts are UMI consensus reads, or
ReDeeM's pipelines, which count molecules. Otherwise, as for the mgatk reads
that MiTo 1 assembled, the file does not say, so the import takes
``count_unit``.
"""

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import h5py
import numpy as np

from ..variants.counts import ALLELE_ROLES, Contig, EntryBatch, SiteTable
from ._h5ad_columns import index_key, read_column, sparse_encoding, sparse_shape
from ._text import as_text
from ..utils.arguments import quoted_value

__all__ = ["MitoAfmSource", "mito_afm_source"]

# The pipelines and assays of MiTo's uns whose counts are molecules.
_MOLECULE_PIPELINES = frozenset({"maegatk", "redeem-v", "RedeemV", "RedeemR"})
_MOLECULE_SYSTEMS = frozenset({"ReDeeM", "RedeeM"})


def _texts(values: Any) -> np.ndarray:
    return np.asarray(
        [as_text(value) for value in np.asarray(values).tolist()], dtype=object
    )


def _column(table: h5py.Group, name: str, path: str) -> np.ndarray:
    if name not in table:
        raise ValueError(f"var of {path} lacks MiTo's {quoted_value(name)} column")
    values, missing = read_column(table[name])
    if np.any(missing):
        raise ValueError(
            f"The {quoted_value(name)} column of {path}'s var has missing values"
        )
    return np.asarray(values)


def _names(table: h5py.Group, path: str, what: str) -> np.ndarray:
    key = index_key(table)
    if key is None or key not in table:
        raise ValueError(f"{what} of {path} has no index of names")
    values, _missing = read_column(table[key])
    return _texts(values)


@dataclass(frozen=True, slots=True)
class MitoAfmSource:
    """MiTo's allele frequency matrix, read as a source of allele counts."""

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
    _sites: SiteTable = field(repr=False)
    _variant_sites: np.ndarray = field(repr=False)
    _variant_alleles: np.ndarray = field(repr=False)
    _barcodes: np.ndarray = field(repr=False)

    def sites(self) -> SiteTable:
        """Each position, with its reference and its alternatives."""
        return self._sites

    def batches(self, chunk_rows: int) -> Iterator[EntryBatch]:
        """The listed cells, then each block of cells' AD and site depth."""
        barcodes = self._barcodes
        n_cells = len(barcodes)
        n_variants = len(self._variant_sites)
        yield EntryBatch(
            kind="cells",
            barcodes=barcodes,
            barcode_codes=np.arange(n_cells, dtype=np.int64),
        )
        block = max(1, chunk_rows // max(n_variants, 1))
        with h5py.File(self.path, "r") as handle:
            ad = handle["layers/AD"]
            dp = handle["layers/DP"]
            for first in range(0, n_cells, block):
                stop = min(n_cells, first + block)
                alt = _dense_rows(ad, first, stop, n_variants, self.path)
                depth = _dense_rows(dp, first, stop, n_variants, self.path)
                for name, values in (("AD", alt), ("DP", depth)):
                    if np.any(values < 0) or np.any(values != np.floor(values)):
                        raise ValueError(
                            f"layers/{name} of {self.path} holds a value that is not "
                            "a whole number of 0 or more"
                        )
                rows, columns = np.nonzero(alt)
                yield EntryBatch(
                    kind="counts",
                    barcodes=barcodes,
                    barcode_codes=rows.astype(np.int64) + first,
                    keys=self._variant_alleles[columns],
                    values=alt[rows, columns].astype(np.int64),
                )
                # A site's depth is the largest DP of its variants, whose
                # columns sorted by site fall in one run per site.
                order = np.argsort(self._variant_sites, kind="stable")
                sorted_sites = self._variant_sites[order]
                starts = np.flatnonzero(
                    np.r_[True, sorted_sites[1:] != sorted_sites[:-1]]
                )
                site_depth = np.maximum.reduceat(depth[:, order], starts, axis=1)
                rows, sites = np.nonzero(site_depth)
                yield EntryBatch(
                    kind="depth",
                    barcodes=barcodes,
                    barcode_codes=rows.astype(np.int64) + first,
                    keys=sites.astype(np.int64),
                    values=site_depth[rows, sites].astype(np.int64),
                )


def _dense_rows(
    node: h5py.Group | h5py.Dataset, first: int, stop: int, n_columns: int, path: str
) -> np.ndarray:
    """Rows ``[first, stop)`` of a dense or CSR layer, as floats."""
    if isinstance(node, h5py.Dataset):
        return np.asarray(node[first:stop], dtype=np.float64)
    if sparse_encoding(node) != "csr":
        raise ValueError(
            f"A layer of {path} is stored as {sparse_encoding(node)}; MiTo writes CSR "
            "or dense layers"
        )
    indptr = np.asarray(node["indptr"][first : stop + 1], dtype=np.int64)
    start, end = int(indptr[0]), int(indptr[-1])
    indices = np.asarray(node["indices"][start:end], dtype=np.int64)
    data = np.asarray(node["data"][start:end], dtype=np.float64)
    out = np.zeros((stop - first, n_columns), dtype=np.float64)
    rows = np.repeat(np.arange(stop - first), np.diff(indptr))
    np.add.at(out, (rows, indices), data)
    return out


def mito_afm_source(path: str | os.PathLike[str]) -> MitoAfmSource:
    """Describe MiTo's allele frequency matrix as a source of allele counts.

    Raises:
        ValueError: If the file lacks MiTo's layers or ``var`` columns, a layer
            has another shape, or a position's variants give it two reference
            bases.
    """
    text = os.fspath(path)
    with h5py.File(text, "r") as handle:
        for name in ("obs", "var", "layers/AD", "layers/DP"):
            if name not in handle:
                raise ValueError(f"{text} lacks MiTo's {name}")
        obs = handle["obs"]
        var = handle["var"]
        barcodes = _names(obs, text, "obs")
        variants = _names(var, text, "var")
        positions = _column(var, "pos", text).astype(np.int64)
        references = np.char.upper(
            _texts(_column(var, "ref", text)).astype(str)
        ).astype(object)
        alternatives = np.char.upper(
            _texts(_column(var, "alt", text)).astype(str)
        ).astype(object)
        shape = (len(barcodes), len(variants))
        for name in ("AD", "DP"):
            node = handle["layers"][name]
            layer_shape = (
                tuple(node.shape)
                if isinstance(node, h5py.Dataset)
                else sparse_shape(node)
            )
            if layer_shape != shape:
                raise ValueError(
                    f"layers/{name} of {text} is {layer_shape}, not {shape} cells by "
                    "variants"
                )
        uns = handle.get("uns")
        notes = {
            key: as_text(uns[key][()])
            for key in ("scLT_system", "pp_method")
            if isinstance(uns, h5py.Group) and key in uns
        }
        converted = isinstance(uns, h5py.Group) and "molecules" in uns
    molecules = (
        converted
        or notes.get("pp_method") in _MOLECULE_PIPELINES
        or notes.get("scLT_system") in _MOLECULE_SYSTEMS
    )
    if np.any(positions < 1):
        raise ValueError(f"var of {text} holds a position below 1")
    site_positions = list(dict.fromkeys(positions.tolist()))
    site_of = {position: site for site, position in enumerate(site_positions)}
    variant_sites: list[int] = [site_of[position] for position in positions.tolist()]
    alternative_texts: list[str] = [
        str(alternative) for alternative in alternatives.tolist()
    ]
    reference_of: dict[int, str] = {}
    for position, reference in zip(
        positions.tolist(), references.tolist(), strict=True
    ):
        if reference_of.setdefault(position, reference) != reference:
            raise ValueError(
                f"{text} gives position {position} the reference bases "
                f"{quoted_value(reference_of[position])} and {quoted_value(reference)}"
            )
    pairs: dict[tuple[int, str], str] = {}
    for position, reference in reference_of.items():
        pairs[(site_of[position], reference)] = "reference"
    for site, alternative in zip(variant_sites, alternative_texts, strict=True):
        if pairs.setdefault((site, alternative), "alternative") != "alternative":
            raise ValueError(
                f"{text} names a variant whose alternative is its reference base"
            )
    ordered = sorted(pairs)
    allele_of = {pair: position for position, pair in enumerate(ordered)}
    variant_alleles = np.asarray(
        [
            allele_of[(site, alternative)]
            for site, alternative in zip(variant_sites, alternative_texts, strict=True)
        ],
        dtype=np.int64,
    )
    n_sites = len(site_positions)
    sites = SiteTable(
        contigs=(Contig(name=""),),
        site_contig=np.zeros(n_sites, dtype=np.int32),
        site_position=np.asarray(site_positions, dtype=np.int64),
        site_name=np.full(n_sites, "", dtype=object),
        site_channel=np.full(n_sites, "", dtype=object),
        allele_site=np.asarray([site for site, _ in ordered], dtype=np.int32),
        allele=np.asarray([allele for _, allele in ordered], dtype=object),
        allele_role=np.asarray([pairs[pair] for pair in ordered], dtype=object),
    )
    return MitoAfmSource(
        path=text,
        format="mito",
        files=(("afm", text),),
        count_unit="molecules" if molecules else None,
        depth_definition="all",
        count_fields=(),
        call_fields=(),
        depth_roles=frozenset(ALLELE_ROLES),
        reference_from_depth=False,
        alleles_overlap=False,
        open_alleles=False,
        notes=notes,
        _sites=sites,
        _variant_sites=np.asarray(variant_sites, dtype=np.int64),
        _variant_alleles=variant_alleles,
        _barcodes=barcodes,
    )
