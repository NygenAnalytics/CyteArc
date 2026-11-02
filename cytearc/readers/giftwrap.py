"""Read the per-cell gapfill counts of giftwrap (GIFT-seq).

GIFT genotypes fixed cells by filling the gap between two probe arms with the
cell's own sequence. giftwrap counts, for each cell, the UMIs of each probe
and gapfill, and writes them as ``counts.N.h5`` (``counts.N.filtered.h5``
after filtering): a CSR matrix of cells by (probe, gapfill) pairs in
``matrix/data``, the reads behind those UMIs in ``matrix/total_reads``, and
in ``matrix/percent_supporting`` the mean share of each UMI's reads that
match its gapfill, from 0 to 1, whose missing entries are the matrix's zeros.
``probe_metadata`` names each probe's expected mutant gapfill
(``gap_probe_sequence``) and wild-type gapfill (``original_sequence``), often
as ``nan`` or ``NA``. giftwrap 0.3.0 added ``matrix/cell_index``,
``matrix/probe_index`` and ``pcr_thresholded_counts``, which the reader does
not need.

GEO ships GIFT data as per-UMI tables instead: ``probe_reads.tsv.gz`` with
one row per UMI (``cell_idx``, ``probe_idx``, ``gapfill``,
``pcr_duplicate_count`` reads and ``percent_supporting``), the cells in
``barcodes.tsv.gz`` and the probes in ``manifest.tsv``, whose
``original_gap_probe_sequence`` is the wild type. A folder of one sample's
three files reads as the HDF5 file does: UMIs counted, reads summed and the
share of supporting reads averaged per cell, probe and gapfill.

Each probe is a site without coordinates and each gapfill an allele, in the
probe's orientation, the reverse complement of the transcript; the set is
open, since cells show gapfills that no manifest names. The wild type is the
reference, the expected mutant an alternative, and other gapfills, among them
the empty gapfill, have the role ``"other"``. Some manifests name one gapfill
as both the wild type and the mutant; it is read as the reference, and the
source record lists those probes as ``same_gapfills``. A site's depth is the
sum of its gapfills' UMIs.
"""

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import h5py
import numpy as np
import pandas as pd

from ..utils.arrays import has_duplicates
from ..variants.counts import EntryBatch, SiteTable
from ._variant_files import find_file, is_gzip, open_text

__all__ = ["GiftwrapSource", "giftwrap_source"]

# Text that giftwrap writes for a missing manifest value.
_MISSING = frozenset({"", "NA", "nan", "NaN", "None", "na"})
_MATRICES = ("data", "total_reads", "percent_supporting")


def _texts(values: Any) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes | np.bytes_) else str(value)
        for value in np.asarray(values).tolist()
    ]


def _named(value: str) -> str | None:
    return None if value.strip() in _MISSING else value.strip()


@dataclass(frozen=True, slots=True)
class _Probes:
    """The probes, their gapfills, and which allele each (probe, gapfill) is."""

    names: list[str]
    sites: SiteTable
    alleles: dict[tuple[str, str], int]
    same_gapfills: tuple[str, ...]


def _probes(
    manifest: list[tuple[str, str | None, str | None]],
    observed: list[tuple[str, str]],
    path: str,
) -> _Probes:
    """Every probe of the manifest or the counts, with its gapfills.

    Args:
        manifest: Each probe's name, wild-type and expected mutant gapfill.
        observed: The (probe, gapfill) pairs that the counts hold.
        path: The file, for errors.
    """
    names: list[str] = []
    roles: dict[tuple[str, str], str] = {}
    seen: set[str] = set()
    same: list[str] = []
    for name, wild_type, mutant in manifest:
        if name in seen:
            raise ValueError(f"{path} lists the probe {name!r} twice")
        seen.add(name)
        names.append(name)
        if mutant is not None and mutant != wild_type:
            roles[(name, mutant)] = "alternative"
        elif mutant is not None:
            # A manifest that names one gapfill as both cannot tell them apart.
            same.append(name)
        if wild_type is not None:
            roles[(name, wild_type)] = "reference"
    for name, gapfill in observed:
        if name not in seen:
            seen.add(name)
            names.append(name)
        roles.setdefault((name, gapfill), "other")
    site_of = {name: position for position, name in enumerate(names)}
    pairs = sorted(roles, key=lambda pair: (site_of[pair[0]], pair[1]))
    sites = SiteTable(
        contigs=(),
        site_contig=np.full(len(names), -1, dtype=np.int32),
        site_position=np.full(len(names), -1, dtype=np.int64),
        site_name=np.asarray(names, dtype=object),
        site_channel=np.full(len(names), "", dtype=object),
        allele_site=np.asarray([site_of[name] for name, _ in pairs], dtype=np.int32),
        allele=np.asarray([gapfill for _, gapfill in pairs], dtype=object),
        allele_role=np.asarray([roles[pair] for pair in pairs], dtype=object),
    )
    return _Probes(
        names=names,
        sites=sites,
        alleles={pair: position for position, pair in enumerate(pairs)},
        same_gapfills=tuple(same),
    )


@dataclass(frozen=True, slots=True)
class GiftwrapSource:
    """giftwrap's counts, from its HDF5 file or GEO's per-UMI tables."""

    path: str
    format: str
    files: tuple[tuple[str, str], ...]
    count_unit: str | None
    depth_definition: str
    count_fields: tuple[str, ...]
    depth_roles: frozenset[str] | None
    reference_from_depth: bool
    alleles_overlap: bool
    open_alleles: bool
    notes: dict[str, Any]
    _probes: _Probes = field(repr=False)
    _barcodes: np.ndarray = field(repr=False)
    _feature_alleles: np.ndarray | None = field(default=None, repr=False)
    _umis: pd.DataFrame | None = field(default=None, repr=False)

    def sites(self) -> SiteTable:
        """Each probe, with the gapfills that the counts or the manifest name."""
        return self._probes.sites

    def batches(self, chunk_rows: int) -> Iterator[EntryBatch]:
        """The listed cells, then each cell's UMIs, reads and support."""
        yield EntryBatch(
            kind="cells",
            barcodes=self._barcodes,
            barcode_codes=np.arange(len(self._barcodes), dtype=np.int64),
        )
        if self._umis is not None:
            yield from self._table_batches(chunk_rows)
        else:
            yield from self._h5_batches(chunk_rows)

    def _table_batches(self, chunk_rows: int) -> Iterator[EntryBatch]:
        table = self._umis
        assert table is not None
        for start in range(0, len(table), chunk_rows):
            part = table.iloc[start : start + chunk_rows]
            yield EntryBatch(
                kind="counts",
                barcodes=self._barcodes,
                barcode_codes=part["cell_idx"].to_numpy(dtype=np.int64),
                keys=part["allele"].to_numpy(dtype=np.int64),
                values=part["umis"].to_numpy(dtype=np.int64),
                fields={
                    "reads": part["reads"].to_numpy(dtype=np.int64),
                    "support": part["support"].to_numpy(dtype=np.float64),
                },
            )

    def _h5_batches(self, chunk_rows: int) -> Iterator[EntryBatch]:
        assert self._feature_alleles is not None
        with h5py.File(self.path, "r") as handle:
            matrix = handle["matrix"]
            indptr = np.asarray(matrix["data/indptr"][:], dtype=np.int64)
            support_indptr = np.asarray(
                matrix["percent_supporting/indptr"][:], dtype=np.int64
            )
            n_cells = len(indptr) - 1
            n_features = len(self._feature_alleles)
            first = 0
            while first < n_cells:
                stop = int(
                    np.searchsorted(indptr, indptr[first] + chunk_rows, side="right")
                    - 1
                )
                stop = min(n_cells, max(first + 1, stop))
                start, end = int(indptr[first]), int(indptr[stop])
                columns = np.asarray(matrix["data/indices"][start:end], dtype=np.int64)
                umis = np.asarray(matrix["data/data"][start:end], dtype=np.int64)
                reads = np.asarray(
                    matrix["total_reads/data"][start:end], dtype=np.int64
                )
                if np.any(columns < 0) or np.any(columns >= n_features):
                    raise ValueError(
                        f"{self.path} has entries outside its {n_features} probe and "
                        "gapfill columns"
                    )
                cells = np.repeat(
                    np.arange(first, stop, dtype=np.int64),
                    np.diff(indptr[first : stop + 1]),
                )
                keys = cells * n_features + columns
                support_start = int(support_indptr[first])
                support_end = int(support_indptr[stop])
                support_cells = np.repeat(
                    np.arange(first, stop, dtype=np.int64),
                    np.diff(support_indptr[first : stop + 1]),
                )
                support_keys = support_cells * n_features + np.asarray(
                    matrix["percent_supporting/indices"][support_start:support_end],
                    dtype=np.int64,
                )
                support_values = np.asarray(
                    matrix["percent_supporting/data"][support_start:support_end],
                    dtype=np.float64,
                )
                support = np.zeros(len(keys), dtype=np.float64)
                order = np.argsort(support_keys, kind="stable")
                at = np.searchsorted(support_keys[order], keys)
                found = at < len(order)
                found[found] = support_keys[order][at[found]] == keys[found]
                support[found] = support_values[order][at[found]]
                yield EntryBatch(
                    kind="counts",
                    barcodes=self._barcodes,
                    barcode_codes=cells,
                    keys=self._feature_alleles[columns],
                    values=umis,
                    fields={"reads": reads, "support": support},
                )
                first = stop


def _h5_source(path: str) -> GiftwrapSource:
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        missing = [
            name
            for name in ("barcode", "probe", *(f"{part}/indptr" for part in _MATRICES))
            if name not in matrix
        ]
        if missing:
            raise ValueError(
                f"{path} lacks matrix/{missing[0]}; it is not a giftwrap counts file"
            )
        barcodes = np.asarray(_texts(matrix["barcode"][:]), dtype=object)
        features = np.asarray(matrix["probe"][:])
        if features.ndim != 2 or features.shape[1] != 2:
            raise ValueError(
                f"matrix/probe of {path} must hold a probe and a gapfill per column"
            )
        probe_names = _texts(features[:, 0])
        gapfills = _texts(features[:, 1])
        shape = tuple(int(value) for value in matrix["data"].attrs["shape"])
        if shape != (len(barcodes), len(probe_names)):
            raise ValueError(
                f"The counts of {path} are {shape[0]} by {shape[1]}, but the file "
                f"names {len(barcodes)} cells and {len(probe_names)} probe and "
                "gapfill pairs"
            )
        for part in _MATRICES:
            if len(matrix[f"{part}/indptr"]) != len(barcodes) + 1:
                raise ValueError(f"matrix/{part} of {path} has another number of cells")
        if not np.array_equal(
            matrix["data/indptr"][:], matrix["total_reads/indptr"][:]
        ) or not np.array_equal(
            matrix["data/indices"][:], matrix["total_reads/indices"][:]
        ):
            raise ValueError(
                f"matrix/total_reads of {path} must hold the entries of matrix/data"
            )
        manifest: list[tuple[str, str | None, str | None]] = []
        metadata = handle.get("probe_metadata")
        if isinstance(metadata, h5py.Group) and "name" in metadata:
            names = _texts(metadata["name"][:])
            wild = (
                _texts(metadata["original_sequence"][:])
                if "original_sequence" in metadata
                else [""] * len(names)
            )
            mutant = (
                _texts(metadata["gap_probe_sequence"][:])
                if "gap_probe_sequence" in metadata
                else [""] * len(names)
            )
            manifest = [
                (name, _named(wt), _named(mt))
                for name, wt, mt in zip(names, wild, mutant, strict=True)
            ]
        layout = "0.3.0" if "cell_index" in matrix else "before 0.3.0"
    probes = _probes(manifest, list(zip(probe_names, gapfills, strict=True)), path)
    feature_alleles = np.asarray(
        [
            probes.alleles[(name, gapfill)]
            for name, gapfill in zip(probe_names, gapfills, strict=True)
        ],
        dtype=np.int64,
    )
    return GiftwrapSource(
        path=path,
        format="giftwrap",
        files=(("counts", path),),
        count_unit="umis",
        depth_definition="all",
        count_fields=("reads", "support"),
        depth_roles=None,
        reference_from_depth=False,
        alleles_overlap=False,
        open_alleles=True,
        notes={"layout": layout, "same_gapfills": list(probes.same_gapfills)},
        _probes=probes,
        _barcodes=barcodes,
        _feature_alleles=feature_alleles,
    )


def _read_table(path: str, **options: Any) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        comment=None,
        compression="gzip" if is_gzip(path) else None,
        **options,
    )


def _manifest_rows(
    path: str,
) -> tuple[pd.DataFrame, list[tuple[str, str | None, str | None]]]:
    with open_text(path) as handle:
        lines = [
            line for line in handle.read().splitlines() if not line.startswith("#")
        ]
    if not lines:
        raise ValueError(f"{path} holds no probes")
    header = lines[0].split("\t")
    rows = [line.split("\t") for line in lines[1:] if line.strip()]
    table = pd.DataFrame(rows, columns=header) if rows else pd.DataFrame(columns=header)
    needed = ("name", "gap_probe_sequence", "original_gap_probe_sequence", "index")
    missing = [name for name in needed if name not in table.columns]
    if missing:
        raise ValueError(f"{path} lacks the manifest column {missing[0]!r}")
    manifest = [
        (str(name).strip(), _named(str(wild)), _named(str(mutant)))
        for name, wild, mutant in zip(
            table["name"],
            table["original_gap_probe_sequence"],
            table["gap_probe_sequence"],
            strict=True,
        )
    ]
    return table, manifest


def _table_source(folder: str) -> GiftwrapSource:
    reads_path = find_file(
        folder, r"(.*_)?probe_reads\.tsv(\.gz)?", "probe_reads.tsv.gz"
    )
    barcode_path = find_file(folder, r"(.*_)?barcodes\.tsv(\.gz)?", "barcodes.tsv.gz")
    manifest_path = find_file(folder, r"(.*_)?manifest\.tsv(\.gz)?", "manifest.tsv")
    assert reads_path is not None and barcode_path is not None
    assert manifest_path is not None
    barcode_table = _read_table(barcode_path)
    if "barcode" not in barcode_table.columns:
        raise ValueError(f"{barcode_path} lacks its 'barcode' column")
    barcodes = barcode_table["barcode"].str.strip().to_numpy(dtype=object)
    probe_table, manifest = _manifest_rows(manifest_path)
    index = pd.to_numeric(probe_table["index"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if np.any(~np.isfinite(index)) or has_duplicates(index):
        raise ValueError(
            f"The index column of {manifest_path} must hold distinct numbers"
        )
    probe_of_index = dict(
        zip(
            index.astype(np.int64).tolist(),
            probe_table["name"].str.strip(),
            strict=True,
        )
    )
    table = _read_table(
        reads_path,
        usecols=[
            "cell_idx",
            "probe_idx",
            "gapfill",
            "pcr_duplicate_count",
            "percent_supporting",
        ],
    )
    cells = pd.to_numeric(table["cell_idx"], errors="coerce").to_numpy(dtype=np.float64)
    probes = pd.to_numeric(table["probe_idx"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    reads = pd.to_numeric(table["pcr_duplicate_count"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    support = pd.to_numeric(table["percent_supporting"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    for name, values, low, high in (
        ("cell_idx", cells, 0, len(barcodes) - 1),
        ("probe_idx", probes, 0, np.inf),
        ("pcr_duplicate_count", reads, 1, np.inf),
    ):
        whole = np.isfinite(values) & (values == np.floor(values))
        if not whole.all() or np.any(values < low) or np.any(values > high):
            raise ValueError(
                f"{reads_path} holds a {name} that is not a whole number from {low} "
                f"to {high}"
            )
    unknown = ~np.isin(probes, list(probe_of_index))
    if unknown.any():
        raise ValueError(
            f"{reads_path} names the probe index {int(probes[unknown][0])}, which "
            f"{manifest_path} does not list"
        )
    if np.any(~np.isfinite(support)) or np.any(support < 0) or np.any(support > 1):
        raise ValueError(f"{reads_path} holds a percent_supporting outside 0 to 1")
    umis = pd.DataFrame(
        {
            "cell_idx": cells.astype(np.int64),
            "probe": [probe_of_index[int(value)] for value in probes.tolist()],
            "gapfill": table["gapfill"].str.strip().to_numpy(dtype=object),
            "reads": reads.astype(np.int64),
            "support": support,
        }
    )
    grouped = (
        umis.groupby(["cell_idx", "probe", "gapfill"], sort=True)
        .agg(
            umis=("reads", "size"), reads=("reads", "sum"), support=("support", "mean")
        )
        .reset_index()
    )
    observed = list(
        dict.fromkeys(
            zip(grouped["probe"].tolist(), grouped["gapfill"].tolist(), strict=True)
        )
    )
    probes_table = _probes(manifest, observed, manifest_path)
    grouped["allele"] = [
        probes_table.alleles[(probe, gapfill)]
        for probe, gapfill in zip(grouped["probe"], grouped["gapfill"], strict=True)
    ]
    return GiftwrapSource(
        path=folder,
        format="giftwrap",
        files=(
            ("probe_reads", reads_path),
            ("barcodes", barcode_path),
            ("manifest", manifest_path),
        ),
        count_unit="umis",
        depth_definition="all",
        count_fields=("reads", "support"),
        depth_roles=None,
        reference_from_depth=False,
        alleles_overlap=False,
        open_alleles=True,
        notes={
            "layout": "per-UMI table",
            "same_gapfills": list(probes_table.same_gapfills),
        },
        _probes=probes_table,
        _barcodes=barcodes,
        _umis=grouped,
    )


def giftwrap_source(path: str | os.PathLike[str]) -> GiftwrapSource:
    """Describe giftwrap's counts file, or a folder of GEO's per-UMI tables.

    Raises:
        FileNotFoundError: If a folder lacks one of its three files.
        ValueError: If a file is malformed, a matrix's shape differs from its
            cells and probes, or a table names a probe the manifest lacks.
    """
    text = os.fspath(path)
    if os.path.isdir(text):
        return _table_source(text)
    return _h5_source(text)
