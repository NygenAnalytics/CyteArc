"""The variant profile of the synthetic lineage tracing experiment, in every format.

``variant_truth`` draws, for the cells and clones of ``clone_truth``,
mitochondrial base counts on a short synthetic reference, with heteroplasmic
variants that each clone shares, sequencing errors, uncovered positions and
one cell without any coverage; GoT counts of two targets; and GIFT UMIs of
three probes. Writers emit them as each tool writes its files, with the
quirks an importer must handle: mgatk's soft-masked reference with an ``N``,
maegatk's swapped strands, cellsnp-lite's sorted barcodes, scipy's real
Matrix Market values, VarTrix's explicit zeros and unsorted entries, giftwrap's
sparse support and both HDF5 layouts, and GoT tables with R row names, ``NA``
and quotes. Files also name barcodes of droplets outside the store. Tests
check that an import recovers the truth, which this module fixes and the
code under test never computes.
"""

import gzip
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from tests.clone_truth import CloneTruth, clone_truth

BASES = ("A", "C", "G", "T")
CONTIG = "chrM"
# A synthetic mitochondrial reference: soft-masked lowercase bases, which
# mgatk's references keep, and an N, which is not a site.
REFERENCE_TEXT = "GATCACAGGTctatcACCCTATTAACCACTCACGGGAGCTNTCCATGCATTTGG"
REFERENCE = REFERENCE_TEXT.upper()
N_POSITIONS = len(REFERENCE)
# 1-based positions that are sites: every base but the N.
SITE_POSITIONS = tuple(
    position + 1 for position, base in enumerate(REFERENCE) if base != "N"
)
TARGETS = ("CALR", "SF3B1")
# Probes: name, wild-type gapfill, expected mutant gapfill (None where the
# manifest has none) and the gapfills cells show.
PROBES = (
    ("JAK2 c.1849G>T", "CAC", "AAC", ("AAC", "CAA", "CAC")),
    ("ASXL1 c.1772dup", None, "GTAAG", ("GTAAG", "GTAG")),
    ("JAK2 0bp", None, None, ("",)),
)
# The cell (by position among the truth's cells) that no read covers.
UNCOVERED_CELL = 5


@dataclass(frozen=True, slots=True)
class VariantTruth:
    """Allele evidence of the synthetic experiment's cells.

    Rows are the truth's cells, then droplets outside the store.

    Attributes:
        clones: The experiment's cells and clones.
        mito: Forward and reverse reads of each base at each position,
            shaped (rows, positions, bases, strands); a position of N holds
            reads too, which no site counts.
        quality: The mean base quality of each base and strand, 0 where the
            strand has no reads, as mgatk and maegatk write it.
        variants: Each clone's heteroplasmic variants, as (position, base).
        got: WT, MUT and ambiguous UMIs at each target, shaped (rows,
            targets, 3), -1 where the cell was not covered.
        got_reads: The reads behind those UMIs.
        gift: Each cell's UMIs of each probe and gapfill, as a mapping from
            (row, probe, gapfill) to a list of (reads, supporting share) per
            UMI.
    """

    clones: CloneTruth
    mito: np.ndarray
    quality: np.ndarray
    variants: dict[int, tuple[tuple[int, str], ...]]
    got: np.ndarray
    got_reads: np.ndarray
    gift: dict[tuple[int, int, str], list[tuple[int, float]]]

    @property
    def row_ids(self) -> list[str]:
        """Each row's barcode in the files: the cell's ID, then the droplets'."""
        return [str(cell) for cell in self.clones.cell_ids] + [
            f"d2__{barcode}-1" for barcode in self.clones.outside
        ]

    @property
    def n_cells(self) -> int:
        return len(self.clones.cell_ids)

    def depth(self) -> np.ndarray:
        """Reads at each position, shaped (rows, positions)."""
        return self.mito.sum(axis=(2, 3))

    def variant_sites(self) -> list[tuple[int, str, str]]:
        """Every clone variant, as (position, reference base, alternative base)."""
        found = {variant for variants in self.variants.values() for variant in variants}
        return sorted(
            (position, REFERENCE[position - 1], base) for position, base in found
        )

    def got_call(self, row: int, target: int) -> str:
        """The call a GoT table writes: MUT with a mutant UMI, else WT."""
        wild_type, mutant, _ambiguous = self.got[row, target]
        if wild_type < 0:
            return "NA"
        return "MUT" if mutant > 0 else "WT"

    def gift_counts(self) -> dict[tuple[int, int, str], tuple[int, int, float]]:
        """UMIs, reads and mean supporting share of each cell, probe and gapfill."""
        return {
            key: (
                len(umis),
                sum(reads for reads, _share in umis),
                float(np.mean([share for _reads, share in umis])),
            )
            for key, umis in self.gift.items()
        }


def variant_truth(seed: int = 0) -> VariantTruth:
    """Draw the variant evidence of ``clone_truth(seed)``'s cells."""
    clones = clone_truth(seed)
    rng = np.random.default_rng(seed + 100)
    n_rows = len(clones.cell_ids) + len(clones.outside)
    variants: dict[int, tuple[tuple[int, str], ...]] = {}
    # Two variants per clone at distinct positions, so a site has one
    # alternative, as cellsnp-lite counts it.
    drawn = rng.choice(SITE_POSITIONS, size=2 * clones.n_clones, replace=False)
    for clone in range(clones.n_clones):
        positions = sorted(drawn[2 * clone : 2 * clone + 2].tolist())
        variants[clone] = tuple(
            (
                int(position),
                str(rng.choice([b for b in BASES if b != REFERENCE[position - 1]])),
            )
            for position in positions
        )
    clone_of = np.concatenate([clones.clones, np.full(len(clones.outside), -1)])
    mito = np.zeros((n_rows, N_POSITIONS, len(BASES), 2), dtype=np.int64)
    for row in range(n_rows):
        if row == UNCOVERED_CELL:
            continue
        heteroplasmy = {
            position: (base, float(rng.uniform(0.2, 0.9)))
            for position, base in variants.get(int(clone_of[row]), ())
        }
        for position in range(1, N_POSITIONS + 1):
            if rng.random() < 0.15:
                continue
            depth = int(rng.poisson(25)) + 1
            reference = REFERENCE[position - 1]
            shares = np.zeros(len(BASES))
            if reference != "N":
                shares[BASES.index(reference)] = 1.0
            else:
                shares[:] = 0.25
            if position in heteroplasmy:
                base, level = heteroplasmy[position]
                shares *= 1 - level
                shares[BASES.index(base)] += level
            reads = rng.multinomial(depth, shares / shares.sum())
            if rng.random() < 0.2:
                reads[int(rng.integers(len(BASES)))] += 1
            forward = rng.binomial(reads, 0.5)
            mito[row, position - 1, :, 0] = forward
            mito[row, position - 1, :, 1] = reads - forward
    quality = np.where(mito > 0, np.round(rng.uniform(20, 38, mito.shape), 1), 0.0)
    got = np.full((n_rows, len(TARGETS), 3), -1, dtype=np.int64)
    got_reads = np.zeros_like(got)
    mutant_clones = {
        target: set(rng.choice(clones.n_clones, size=3, replace=False).tolist())
        for target in range(len(TARGETS))
    }
    for row in range(n_rows):
        for target in range(len(TARGETS)):
            if row == UNCOVERED_CELL or rng.random() < 0.3:
                continue
            mutant = int(clone_of[row]) in mutant_clones[target]
            umis = [
                int(rng.poisson(3 if not mutant else 1)),
                int(rng.poisson(2)) if mutant else 0,
                int(rng.poisson(0.4)),
            ]
            got[row, target] = umis
            got_reads[row, target] = [
                umis[call] + int(rng.poisson(2 * umis[call])) for call in range(3)
            ]
    gift: dict[tuple[int, int, str], list[tuple[int, float]]] = {}
    for row in range(n_rows):
        if row == UNCOVERED_CELL:
            continue
        for probe, (_name, _wild, _mutant, gapfills) in enumerate(PROBES):
            for gapfill in gapfills:
                n_umis = int(rng.poisson(1.5))
                if n_umis:
                    gift[(row, probe, gapfill)] = [
                        (
                            int(rng.integers(1, 30)),
                            float(rng.choice([1.0, 0.5, 0.0, 0.75])),
                        )
                        for _ in range(n_umis)
                    ]
    return VariantTruth(
        clones=clones,
        mito=mito,
        quality=quality,
        variants=variants,
        got=got,
        got_reads=got_reads,
        gift=gift,
    )


def clone_variant_truth(clones: CloneTruth, seed: int = 0) -> VariantTruth:
    """Mitochondrial reads of ``clones``' cells, from which their clones follow.

    Each clone carries two variants of its own, which each of its cells holds
    at a frequency drawn from 0.5 to 0.9; cells without a clone carry none.
    Every cell and droplet is read at every position, at a depth around 60,
    without read errors, so a cell shows a variant exactly when its clone
    carries it, and the variants that two cells or more show join exactly the
    cells of each clone of two cells or more. The experiment has no GoT or
    GIFT counts.
    """
    rng = np.random.default_rng(seed + 300)
    n_rows = len(clones.cell_ids) + len(clones.outside)
    drawn = rng.choice(SITE_POSITIONS, size=2 * clones.n_clones, replace=False)
    variants: dict[int, tuple[tuple[int, str], ...]] = {}
    for clone in range(clones.n_clones):
        positions = sorted(drawn[2 * clone : 2 * clone + 2].tolist())
        variants[clone] = tuple(
            (
                int(position),
                str(rng.choice([b for b in BASES if b != REFERENCE[position - 1]])),
            )
            for position in positions
        )
    clone_of = np.concatenate([clones.clones, np.full(len(clones.outside), -1)])
    mito = np.zeros((n_rows, N_POSITIONS, len(BASES), 2), dtype=np.int64)
    for row in range(n_rows):
        heteroplasmy = {
            position: (base, float(rng.uniform(0.5, 0.9)))
            for position, base in variants.get(int(clone_of[row]), ())
        }
        for position in range(1, N_POSITIONS + 1):
            reference = REFERENCE[position - 1]
            shares = np.zeros(len(BASES))
            if reference != "N":
                shares[BASES.index(reference)] = 1.0
            else:
                shares[:] = 0.25
            if position in heteroplasmy:
                base, level = heteroplasmy[position]
                shares *= 1 - level
                shares[BASES.index(base)] += level
            reads = rng.multinomial(int(rng.poisson(40)) + 20, shares / shares.sum())
            forward = rng.binomial(reads, 0.5)
            mito[row, position - 1, :, 0] = forward
            mito[row, position - 1, :, 1] = reads - forward
    quality = np.where(mito > 0, np.round(rng.uniform(20, 38, mito.shape), 1), 0.0)
    return VariantTruth(
        clones=clones,
        mito=mito,
        quality=quality,
        variants=variants,
        got=np.full((n_rows, len(TARGETS), 3), -1, dtype=np.int64),
        got_reads=np.zeros((n_rows, len(TARGETS), 3), dtype=np.int64),
        gift={},
    )


def genotype_clone_truth(clones: CloneTruth, seed: int = 0) -> np.ndarray:
    """GoT UMIs of ``clones``' cells at one target per clone, from which the clones follow.

    Each clone carries a mutation at a target of its own, which every cell of
    the clone is read at, with one mutant UMI or more beside a few wild-type
    ones, as one mutant copy of two gives. Every other target reads wild type
    in a cell, or is not read at all in about a fifth of the cells, which
    leaves it unknown rather than wild type. Cells without a clone read wild
    type wherever they are read, so genotype clones of the mutant calls are
    exactly the truth's clones, its clone of one cell among them.

    Returns:
        The wild-type and mutant UMIs of each cell at each target, shaped
        ``(cells, clones, 2)``, -1 where the cell was not read.
    """
    rng = np.random.default_rng(seed + 500)
    n_cells = len(clones.cell_ids)
    umis = np.full((n_cells, clones.n_clones, 2), -1, dtype=np.int64)
    for cell in range(n_cells):
        own = int(clones.clones[cell])
        for target in range(clones.n_clones):
            if target != own and rng.random() < 0.2:
                continue
            if target == own:
                umis[cell, target] = (int(rng.poisson(1)), 1 + int(rng.poisson(1)))
            else:
                umis[cell, target] = (1 + int(rng.poisson(2)), 0)
    return umis


def write_genotype_tables(
    umis: np.ndarray, cell_ids: Sequence[str], folder: Path
) -> list[Path]:
    """One GoT table (``BC wtUMIs mutUMIs``) per target of :func:`genotype_clone_truth`.

    The tables are named ``T1.tsv`` and on, so their targets are ``T1`` and on,
    and list only the cells read at the target.
    """
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for target in range(umis.shape[1]):
        lines = ["BC\twtUMIs\tmutUMIs"] + [
            f"{cell}\t{umis[row, target, 0]}\t{umis[row, target, 1]}"
            for row, cell in enumerate(cell_ids)
            if umis[row, target, 0] >= 0
        ]
        paths.append(_write(folder / f"T{target + 1}.tsv", lines))
    return paths


#: The clones of :func:`mito_clone_truth`, largest first.
MITO_CLONE_SIZES = (60, 45, 35, 30, 20, 15)


def mito_clone_truth(
    seed: int = 0,
    *,
    clone_sizes: Sequence[int] = MITO_CLONE_SIZES,
    n_without_clone: int = 25,
    n_shallow: int = 10,
    depth: float = 40.0,
) -> VariantTruth:
    """Draw a larger experiment whose clones its mitochondrial variants define.

    Each clone carries two variants of its own, which each of its cells
    holds at a frequency drawn from 0.3 to 0.9; cells without a clone carry
    none. Reads cover each position at a depth drawn around ``depth``, except
    10% of positions per cell left uncovered and ``n_shallow`` cells read at
    a depth around 3. A quarter of the covered positions gain one erroneous
    read of another base on one strand, so errors fail strand tests while the
    clones' variants pass them. The experiment has no GoT or GIFT counts.
    """
    rng = np.random.default_rng(seed + 200)
    sizes = [int(size) for size in clone_sizes]
    n_cells = sum(sizes) + n_without_clone
    n_outside = 3
    barcodes: set[str] = set()
    while len(barcodes) < n_cells + n_outside:
        barcodes.add("".join(rng.choice(list(BASES), size=16)))
    drawn = np.asarray(sorted(barcodes))[rng.permutation(n_cells + n_outside)]
    clone_of_cell = np.full(n_cells, -1, dtype=np.int64)
    members = rng.permutation(n_cells)
    start = 0
    for clone, size in enumerate(sizes):
        clone_of_cell[members[start : start + size]] = clone
        start += size
    clones = CloneTruth(
        barcodes=drawn[:n_cells],
        samples=rng.choice(["d2", "d4"], size=n_cells),
        cell_types=rng.choice(["Undifferentiated", "Neutrophil"], size=n_cells),
        clones=clone_of_cell,
        probabilities=np.ones(n_cells),
        outside=drawn[n_cells:],
    )
    positions = rng.choice(SITE_POSITIONS, size=2 * len(sizes), replace=False)
    variants: dict[int, tuple[tuple[int, str], ...]] = {}
    for clone in range(len(sizes)):
        chosen = sorted(positions[2 * clone : 2 * clone + 2].tolist())
        variants[clone] = tuple(
            (
                int(position),
                str(rng.choice([b for b in BASES if b != REFERENCE[position - 1]])),
            )
            for position in chosen
        )
    clone_of_row = np.concatenate([clone_of_cell, np.full(n_outside, -1)])
    shallow = set(rng.choice(n_cells, size=n_shallow, replace=False).tolist())
    n_rows = n_cells + n_outside
    mito = np.zeros((n_rows, N_POSITIONS, len(BASES), 2), dtype=np.int64)
    for row in range(n_rows):
        heteroplasmy = {
            position: (base, float(rng.uniform(0.3, 0.9)))
            for position, base in variants.get(int(clone_of_row[row]), ())
        }
        mean_depth = 3.0 if row in shallow else depth
        for position in range(1, N_POSITIONS + 1):
            if rng.random() < 0.1:
                continue
            reads_at = int(rng.poisson(mean_depth)) + 1
            reference = REFERENCE[position - 1]
            shares = np.zeros(len(BASES))
            if reference != "N":
                shares[BASES.index(reference)] = 1.0
            else:
                shares[:] = 0.25
            if position in heteroplasmy:
                base, level = heteroplasmy[position]
                shares *= 1 - level
                shares[BASES.index(base)] += level
            reads = rng.multinomial(reads_at, shares / shares.sum())
            forward = rng.binomial(reads, 0.5)
            mito[row, position - 1, :, 0] = forward
            mito[row, position - 1, :, 1] = reads - forward
            if rng.random() < 0.25:
                others = [b for b in range(len(BASES)) if BASES[b] != reference]
                mito[
                    row, position - 1, int(rng.choice(others)), int(rng.integers(2))
                ] += 1
    quality = np.where(mito > 0, np.round(rng.uniform(20, 38, mito.shape), 1), 0.0)
    return VariantTruth(
        clones=clones,
        mito=mito,
        quality=quality,
        variants=variants,
        got=np.full((n_rows, len(TARGETS), 3), -1, dtype=np.int64),
        got_reads=np.zeros((n_rows, len(TARGETS), 3), dtype=np.int64),
        gift={},
    )


def _write(path: Path, lines: Sequence[str], *, compress: bool = False) -> Path:
    data = ("\n".join(lines) + "\n").encode("utf-8")
    if compress:
        path.write_bytes(gzip.compress(data, mtime=0))
    else:
        path.write_bytes(data)
    return path


def write_reference(path: Path, *, compress: bool = False) -> Path:
    """mgatk's ``<contig>_refAllele.txt``: 1-based positions and bases as given."""
    return _write(
        path,
        [f"{position}\t{base}" for position, base in enumerate(REFERENCE_TEXT, 1)],
        compress=compress,
    )


def write_mgatk(
    truth: VariantTruth,
    folder: Path,
    *,
    name: str = "sample",
    six_columns: bool = False,
    swapped: bool = False,
    depth_table: bool = True,
    rows: Sequence[int] | None = None,
    cell_order: Sequence[int] | None = None,
) -> Path:
    """An mgatk ``final`` folder: four base files, the depth table and the reference.

    Each base file groups rows by cell, in ``cell_order``, with rising
    positions, and holds a row only where a strand count is above 0. With
    ``six_columns`` the rows hold each strand's mean quality, and with
    ``swapped`` the reverse reads sit in the forward columns, as maegatk and
    mgatk 0.4.0 to 0.5.8 wrote them.
    """
    folder.mkdir(parents=True, exist_ok=True)
    ids = truth.row_ids
    chosen = list(range(len(ids))) if rows is None else list(rows)
    order = chosen if cell_order is None else list(cell_order)
    for base_index, base in enumerate(BASES):
        lines = []
        for row in order:
            for position in range(1, N_POSITIONS + 1):
                forward, reverse = truth.mito[row, position - 1, base_index]
                if forward == 0 and reverse == 0:
                    continue
                q_forward, q_reverse = truth.quality[row, position - 1, base_index]
                if swapped:
                    forward, reverse = reverse, forward
                    q_forward, q_reverse = q_reverse, q_forward
                if six_columns:
                    lines.append(
                        f"{position},{ids[row]},{forward},{q_forward:.1f},{reverse},"
                        f"{q_reverse:.1f}"
                    )
                else:
                    lines.append(f"{position},{ids[row]},{forward},{reverse}")
        _write(folder / f"{name}.{base}.txt.gz", lines, compress=True)
    if depth_table:
        depth = truth.depth()
        _write(
            folder / f"{name}.depthTable.txt",
            [f"{ids[row]}\t{depth[row].sum() / N_POSITIONS:.2f}" for row in chosen],
        )
    write_reference(folder / f"{CONTIG}_refAllele.txt")
    return folder


def _mtx(
    path: Path,
    entries: Sequence[tuple[int, int, float]],
    shape: tuple[int, int],
    *,
    real: bool = False,
    separator: str = "\t",
) -> Path:
    field = "real" if real else "integer"
    lines = [f"%%MatrixMarket matrix coordinate {field} general", "%"]
    lines.append(f"{shape[0]} {shape[1]} {len(entries)}")
    for row, column, value in entries:
        text = f"{value:.15e}" if real else str(int(value))
        lines.append(f"{row + 1}{separator}{column + 1}{separator}{text}")
    return _write(path, lines)


def _variant_counts(truth: VariantTruth) -> list[dict[str, np.ndarray]]:
    """Each variant site's reference, alternative and other reads per row."""
    counts = truth.mito.sum(axis=3)
    sites = []
    for position, reference, alternative in truth.variant_sites():
        per_base = counts[:, position - 1, :]
        ref = per_base[:, BASES.index(reference)]
        alt = per_base[:, BASES.index(alternative)]
        sites.append(
            {"ref": ref, "alt": alt, "other": per_base.sum(axis=1) - ref - alt}
        )
    return sites


def cellsnp_rows(truth: VariantTruth) -> list[int]:
    """The rows that cellsnp-lite lists: every row, by sorted barcode."""
    ids = truth.row_ids
    return sorted(range(len(ids)), key=lambda row: ids[row])


def write_cellsnp(
    truth: VariantTruth,
    folder: Path,
    *,
    other: bool = True,
    compress: bool = False,
    lowercase: bool = False,
) -> Path:
    """A cellsnp-lite folder: the base VCF, the sorted cell list, AD, DP and OTH.

    With ``lowercase`` the VCF writes its alleles in lowercase, as a VCF
    called against a soft-masked reference can.
    """
    folder.mkdir(parents=True, exist_ok=True)
    rows = cellsnp_rows(truth)
    ids = truth.row_ids
    sites = truth.variant_sites()
    counts = _variant_counts(truth)

    def case(text: str) -> str:
        return text.lower() if lowercase else text

    _write(
        folder / ("cellSNP.base.vcf.gz" if compress else "cellSNP.base.vcf"),
        [
            "##fileformat=VCFv4.2",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
            *(
                f"{CONTIG}\t{position}\t.\t{case(ref)}\t{case(alt)}\t.\tPASS\t"
                "AD=0;DP=0;OTH=0"
                for position, ref, alt in sites
            ),
        ],
        compress=compress,
    )
    _write(folder / "cellSNP.samples.tsv", [ids[row] for row in rows])
    shape = (len(sites), len(rows))
    for tag, value in (
        ("AD", lambda site: site["alt"]),
        ("DP", lambda site: site["ref"] + site["alt"]),
        ("OTH", lambda site: site["other"]),
    ):
        if tag == "OTH" and not other:
            continue
        entries = [
            (site_index, column, int(value(site)[row]))
            for site_index, site in enumerate(counts)
            for column, row in enumerate(rows)
            if value(site)[row] > 0
        ]
        _mtx(folder / f"cellSNP.tag.{tag}.mtx", entries, shape)
    return folder


def write_mquad(truth: VariantTruth, folder: Path, *, bic: bool = True) -> Path:
    """An MQuad folder: passed AD and DP as scipy writes them, the names, the
    BIC table and the upstream cell list."""
    folder.mkdir(parents=True, exist_ok=True)
    rows = cellsnp_rows(truth)
    ids = truth.row_ids
    sites = truth.variant_sites()
    counts = _variant_counts(truth)
    # MQuad lists the passed variants by decreasing deltaBIC, not by position.
    passed = list(range(len(sites)))[::-1]
    shape = (len(passed), len(rows))
    for name, value in (
        ("passed_ad.mtx", lambda site: site["alt"]),
        ("passed_dp.mtx", lambda site: site["ref"] + site["alt"]),
    ):
        entries = [
            (rank, column, float(value(counts[site])[row]))
            for rank, site in enumerate(passed)
            for column, row in enumerate(rows)
            if value(counts[site])[row] > 0
        ]
        _mtx(folder / name, entries, shape, real=True, separator=" ")
    names = [f"{sites[site][0]}{sites[site][1]}>{sites[site][2]}" for site in passed]
    (folder / "passed_variant_names.txt").write_text("\n".join(names))
    if bic:
        _write(
            folder / "BIC_params.csv",
            [
                "num_cells,deltaBIC,variant_name",
                *(
                    f"{len(rows)},{100.0 - rank},{CONTIG}_{sites[site][0]}_"
                    f"{sites[site][1]}_{sites[site][2]}"
                    for rank, site in enumerate(passed)
                ),
                f"{len(rows)},1.0,{CONTIG}_9_A_C",
            ],
        )
    _write(folder / "cellSNP.samples.tsv", [ids[row] for row in rows])
    return folder


def write_vartrix(truth: VariantTruth, folder: Path) -> Path:
    """A VarTrix folder: the VCF with a multi-allelic record, the barcodes and
    the two coverage matrices, with explicit zeros, entries in reverse order."""
    folder.mkdir(parents=True, exist_ok=True)
    ids = truth.row_ids
    sites = truth.variant_sites()
    counts = _variant_counts(truth)
    records = [f"{CONTIG}\t{position}\t.\t{ref}\t{alt}" for position, ref, alt in sites]
    records.insert(1, f"{CONTIG}\t2\t.\tA\tC,G")
    _write(
        folder / "variants.vcf",
        [
            "##fileformat=VCFv4.2",
            f"##contig=<ID={CONTIG},length={N_POSITIONS}>",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
            *(record + "\t.\t.\t." for record in records),
        ],
    )
    _write(folder / "barcodes.tsv", ids)
    record_of = [index if index < 1 else index + 1 for index in range(len(sites))]
    shape = (len(records), len(ids))
    for name, key in (("alt.mtx", "alt"), ("ref_matrix.mtx", "ref")):
        entries = []
        for site, site_counts in enumerate(counts):
            covered = site_counts["ref"] + site_counts["alt"] > 0
            for row in np.flatnonzero(covered).tolist():
                entries.append((record_of[site], row, float(site_counts[key][row])))
        _mtx(folder / name, entries[::-1], shape, real=True, separator=" ")
    return folder


def write_mito_afm(
    truth: VariantTruth,
    path: Path,
    *,
    system: str | None = "MAESTER",
    pipeline: str | None = "maegatk",
    molecules: bool = False,
) -> Path:
    """MiTo's allele frequency matrix of the clone variants: CSR ``AD`` and
    dense ``DP`` layers, ``pos``, ``ref`` and ``alt`` in ``var``.

    ``uns`` holds ``scLT_system`` and ``pp_method`` where given, and with
    ``molecules`` the record that MiTo 2 writes when it turns reads into
    molecules; without any of them the file has no ``uns``.
    """
    ids = truth.row_ids
    sites = truth.variant_sites()
    counts = _variant_counts(truth)
    depth = truth.depth()
    alt = np.stack([site["alt"] for site in counts], axis=1)
    dp = np.stack([depth[:, position - 1] for position, _r, _a in sites], axis=1)
    string = h5py.string_dtype()
    with h5py.File(path, "w") as handle:
        for name, index in (
            ("obs", ids),
            ("var", [f"{p}_{r}>{a}" for p, r, a in sites]),
        ):
            group = handle.create_group(name)
            group.attrs["_index"] = "_index"
            group.attrs["encoding-type"] = "dataframe"
            group.attrs["encoding-version"] = "0.2.0"
            group.create_dataset("_index", data=np.asarray(index, dtype=string))
        var = handle["var"]
        var.attrs["column-order"] = np.asarray(["pos", "ref", "alt"], dtype=string)
        var.create_dataset("pos", data=np.asarray([p for p, _r, _a in sites]))
        var.create_dataset(
            "ref", data=np.asarray([r for _p, r, _a in sites], dtype=string)
        )
        var.create_dataset(
            "alt", data=np.asarray([a for _p, _r, a in sites], dtype=string)
        )
        layers = handle.create_group("layers")
        rows, columns = np.nonzero(alt)
        group = layers.create_group("AD")
        group.attrs["encoding-type"] = "csr_matrix"
        group.attrs["encoding-version"] = "0.1.0"
        group.attrs["shape"] = np.asarray(alt.shape)
        group.create_dataset("data", data=alt[rows, columns].astype(np.int16))
        group.create_dataset("indices", data=columns.astype(np.int32))
        group.create_dataset(
            "indptr",
            data=np.concatenate(
                [[0], np.cumsum(np.bincount(rows, minlength=len(ids)))]
            ),
        )
        layers.create_dataset("DP", data=dp.astype(np.int16))
        if system is None and pipeline is None and not molecules:
            return path
        uns = handle.create_group("uns")
        for key, value in (("scLT_system", system), ("pp_method", pipeline)):
            if value is not None:
                uns.create_dataset(key, data=value, dtype=string)
        if molecules:
            record = uns.create_group("molecules")
            record.create_dataset("reads_per_molecule", data=1.66)
            record.create_dataset("frac_both_mates", data=0.66)
            record.create_dataset("alt_rule", data="both_strands", dtype=string)
    return path


def _gift_features(truth: VariantTruth) -> list[tuple[str, str]]:
    observed = {(PROBES[probe][0], gapfill) for (_row, probe, gapfill) in truth.gift}
    return sorted(observed, key=lambda pair: (pair[0].encode(), pair[1].encode()))


def write_giftwrap_h5(
    truth: VariantTruth,
    path: Path,
    *,
    layout: str = "before 0.3.0",
    unsorted_support: bool = False,
) -> Path:
    """giftwrap's ``counts.N.h5``: CSR UMIs, reads and support of cells by
    probe and gapfill, whose zero support entries are left out, and the
    probe metadata with ``nan`` for the gapfills it does not name.

    With ``unsorted_support`` each cell's support entries come in reverse
    column order, which a CSR matrix allows.
    """
    ids = truth.row_ids
    features = _gift_features(truth)
    column_of = {feature: column for column, feature in enumerate(features)}
    counts = truth.gift_counts()
    per_row: dict[int, list[tuple[int, tuple[int, int, float]]]] = {}
    for (row, probe, gapfill), values in counts.items():
        per_row.setdefault(row, []).append(
            (column_of[(PROBES[probe][0], gapfill)], values)
        )
    bytes_dtype = "S64"
    with h5py.File(path, "w") as handle:
        handle.attrs["n_cells"] = len(ids)
        handle.attrs["plex"] = 1
        matrix = handle.create_group("matrix")
        matrix.create_dataset("barcode", data=np.asarray(ids, dtype=bytes_dtype))
        matrix.create_dataset(
            "probe",
            data=np.asarray([list(feature) for feature in features], dtype=bytes_dtype),
        )
        parts: dict[str, tuple[list[float], list[int], list[int]]] = {
            name: ([], [], [0])
            for name in ("data", "total_reads", "percent_supporting")
        }
        for row in range(len(ids)):
            entries = sorted(per_row.get(row, []))
            for name, position in (
                ("data", 0),
                ("total_reads", 1),
                ("percent_supporting", 2),
            ):
                kept = [
                    (column, values[position])
                    for column, values in entries
                    if values[position] != 0
                ]
                if unsorted_support and name == "percent_supporting":
                    kept = kept[::-1]
                for column, value in kept:
                    parts[name][0].append(value)
                    parts[name][1].append(column)
                parts[name][2].append(len(parts[name][0]))
        for name, (data, indices, indptr) in parts.items():
            group = matrix.create_group(name)
            group.attrs["shape"] = np.asarray([len(ids), len(features)])
            dtype = np.float32 if name == "percent_supporting" else np.uint32
            group.create_dataset("data", data=np.asarray(data, dtype=dtype))
            group.create_dataset("indices", data=np.asarray(indices, dtype=np.int32))
            group.create_dataset("indptr", data=np.asarray(indptr, dtype=np.int32))
        metadata = handle.create_group("probe_metadata")
        metadata.create_dataset(
            "name", data=np.asarray([p[0] for p in PROBES], dtype=bytes_dtype)
        )
        metadata.create_dataset(
            "original_sequence",
            data=np.asarray(
                ["nan" if p[1] is None else p[1] for p in PROBES], dtype=bytes_dtype
            ),
        )
        metadata.create_dataset(
            "gap_probe_sequence",
            data=np.asarray(
                ["nan" if p[2] is None else p[2] for p in PROBES], dtype=bytes_dtype
            ),
        )
        metadata.create_dataset("index", data=np.arange(len(PROBES), dtype=np.uint32))
        if layout == "0.3.0":
            matrix.create_dataset("cell_index", data=np.arange(len(ids)))
            matrix.create_dataset("probe_index", data=np.arange(len(features)))
            thresholded = handle.create_group("pcr_thresholded_counts")
            thresholded.create_group("pcr1")
    return path


def write_giftwrap_tables(truth: VariantTruth, folder: Path) -> Path:
    """GEO's GIFT files: one row per UMI, the cells and the manifest."""
    folder.mkdir(parents=True, exist_ok=True)
    ids = truth.row_ids
    lines = [
        "cell_idx\tprobe_idx\tprobe_barcode\tumi\tgapfill\tpcr_duplicate_count\tpercent_supporting"
    ]
    for (row, probe, gapfill), umis in sorted(truth.gift.items()):
        for number, (reads, share) in enumerate(umis):
            lines.append(
                f"{row}\t{probe}\t1\tUMI{number:06d}\t{gapfill}\t{reads}\t{share}"
            )
    _write(folder / "GSM0_GAP_probe_reads.tsv.gz", lines, compress=True)
    _write(
        folder / "GSM0_GAP_barcodes.tsv.gz",
        ["barcode\tplex", *(f"{cell}\t1" for cell in ids)],
        compress=True,
    )
    _write(
        folder / "GSM0_GAP_manifest.tsv.gz",
        [
            '# key={"panel": "synthetic"}',
            "name\tlhs_probe\trhs_probe\tgap_probe_sequence\toriginal_gap_probe_sequence\tgene\tindex",
            *(
                f"{name}\tAAAA\tTTTT\t{mutant or 'NA'}\t{wild or 'NA'}\t\t{index}"
                for index, (name, wild, mutant, _gapfills) in enumerate(PROBES)
            ),
        ],
        compress=True,
    )
    return folder


def write_got(truth: VariantTruth, path: Path, *, style: str, target: int = 0) -> Path:
    """One target's per-cell GoT table, in a style that a tool or study wrote.

    Styles: ``"ironthrone"`` (IronThrone's summTable with per-UMI lists),
    ``"xv"`` (``BC wtUMIs mutUMIs``), ``"geo_calls"`` (``BC num.WT.call
    num.MUT.call``), ``"geo_total"`` (R row names with ``WT.calls``,
    ``MUT.calls``, ``Total.calls`` and ``Genotype``, ``NA`` where not
    covered) and ``"geo_call_only"`` (R's quoted ``"","Genotype"``).
    """
    ids = truth.row_ids
    lines: list[str] = []
    if style == "ironthrone":
        lines.append(
            "BC\twhitelist\tUMI\tnum.WT.in.dups\tnum.MUT.in.dups\tnum.amb.in.dups\t"
            "call.in.dups\tWT.calls\tMUT.calls\tamb.calls"
        )
    elif style == "xv":
        lines.append("BC\twtUMIs\tmutUMIs")
    elif style == "geo_calls":
        lines.append("BC\tnum.WT.call\tnum.MUT.call")
    elif style == "geo_total":
        lines.append("WT.calls\tMUT.calls\tTotal.calls\tGenotype\tCell.Assignment")
    elif style == "geo_call_only":
        lines.append('"","Genotype"')
    else:
        raise ValueError(f"Unknown GoT style {style!r}")
    for row, cell in enumerate(ids):
        wild_type, mutant, ambiguous = truth.got[row, target].tolist()
        call = truth.got_call(row, target)
        if style == "geo_total":
            if wild_type < 0:
                lines.append(f"{cell}\tNA\tNA\tNA\tNA\tHSPC")
            else:
                lines.append(
                    f"{cell}\t{wild_type}\t{mutant}\t{wild_type + mutant}\t{call}\tHSPC"
                )
            continue
        if style == "geo_call_only":
            lines.append(
                f'"{cell}",{call if call == "NA" else chr(34) + call + chr(34)}'
            )
            continue
        if wild_type < 0:
            continue
        if style == "ironthrone":
            umis = wild_type + mutant + ambiguous
            calls = ";".join(
                ["WT"] * wild_type + ["MUT"] * mutant + ["amb"] * ambiguous
            )
            lines.append(
                f"{cell}\tY\t{';'.join('U' * (index + 1) for index in range(umis))}\t"
                f"{';'.join('1' * umis)}\t{';'.join('0' * umis)}\t{';'.join('0' * umis)}\t"
                f"{calls}\t{wild_type}\t{mutant}\t{ambiguous}"
            )
        else:
            lines.append(f"{cell}\t{wild_type}\t{mutant}")
    return _write(path, lines, compress=path.suffix == ".gz")


def write_got_multi(truth: VariantTruth, path: Path, *, refined: bool = False) -> Path:
    """GoT-Multi-ML's tables: per channel UMIs and reads with three genotypes,
    or the refined table's ``genotype_final``. The probe-barcode channel
    counts one WT UMI more than the gene-sequence channel."""
    ids = truth.row_ids
    if refined:
        lines = ["BC,target,sample,genotype_merged,genotype_final"]
    else:
        columns = ["BC", "target"]
        for call in ("wt", "mut", "amb"):
            for channel in ("geneseq", "probebc"):
                columns.append(f"{call}_reads_per_umi_total_{channel}")
        for channel in ("geneseq", "probebc"):
            columns += [f"{call}_calls_{channel}" for call in ("wt", "mut", "amb")]
        columns += ["genotype_geneseq", "genotype_probebc", "genotype_merged"]
        lines = [",".join(columns)]
    for row, cell in enumerate(ids):
        for target, name in enumerate(TARGETS):
            umis = truth.got[row, target].tolist()
            if umis[0] < 0:
                continue
            call = truth.got_call(row, target)
            if refined:
                final = "FalsePositive" if call == "MUT" and row % 4 == 0 else call
                lines.append(f"{cell},{name},s1,{call},{final}")
                continue
            reads = truth.got_reads[row, target].tolist()
            probebc = [umis[0] + 1, umis[1], umis[2]]
            fields = [cell, name]
            for index in range(3):
                fields += [str(reads[index]), str(reads[index] + (index == 0))]
            fields += [str(value) for value in umis] + [str(value) for value in probebc]
            fields += [call, call, call]
            lines.append(",".join(fields))
    return _write(path, lines)


def write_gotcha(truth: VariantTruth, path: Path, *, style: str = "labeling") -> Path:
    """Gotcha's tables written with R's ``write.csv``: ``GotchaLabeling``'s
    output (row names, ``WTcount``, ``MUTcount``, ``genotype_pred``) or the
    merged metadata of several genes (``<gene>_WTcount``)."""
    ids = truth.row_ids
    if style == "labeling":
        lines = ['"","WTcount","MUTcount","quadrant_class","genotype_pred"']
        for row, cell in enumerate(ids):
            wild_type, mutant, _ambiguous = truth.got[row, 0].tolist()
            if wild_type < 0:
                lines.append(f'"{cell}",NA,NA,NA,NA')
                continue
            call = truth.got_call(row, 0)
            lines.append(f'"{cell}",{wild_type},{mutant},"{call}","{call}"')
    elif style == "genes":
        header = ['"X"', '"Sample"']
        for name in TARGETS:
            header += [f'"{name}_WTcount"', f'"{name}_MUTcount"']
        lines = [",".join(header)]
        for row, cell in enumerate(ids):
            fields = [f'"{cell}"', '"s1"']
            for target in range(len(TARGETS)):
                wild_type, mutant, _ambiguous = truth.got[row, target].tolist()
                fields += (
                    ["NA", "NA"] if wild_type < 0 else [str(wild_type), str(mutant)]
                )
            lines.append(",".join(fields))
    else:
        raise ValueError(f"Unknown Gotcha style {style!r}")
    return _write(path, lines)


@dataclass(frozen=True, slots=True)
class TapestriVariant:
    """A candidate variant of the synthetic Tapestri panel.

    ``chrom`` is written as Tapestri's ``CHROM`` column writes it, without
    ``chr``, which its variant IDs add.
    """

    gene: str
    chrom: str
    position: int
    reference: str
    alternative: str
    passing: bool

    @property
    def name(self) -> str:
        """The variant's ``id`` in a Tapestri file."""
        return f"chr{self.chrom}:{self.position}:{self.reference}/{self.alternative}"

    @property
    def dabseq_name(self) -> str:
        """The variant's name in a DAb-seq file: its gene, then its ``id``."""
        return f"{self.gene}:{self.name}"


#: The panel's candidate variants, as a Tapestri file lists them: two
#: alternatives of one position that both pass; a deletion and a SNV of one
#: position, whose references differ; an alternative that fails beside one
#: that passes; empty ALTs, one of them passing; and alleles that are not
#: bases.
TAPESTRI_VARIANTS = (
    TapestriVariant("NRAS", "1", 1000, "G", "A", True),
    TapestriVariant("NRAS", "1", 1000, "G", "T", True),
    TapestriVariant("NRAS", "1", 1500, "C", "", False),
    TapestriVariant("DNMT3A", "1", 2000, "TC", "T", True),
    TapestriVariant("DNMT3A", "1", 2000, "T", "G", True),
    TapestriVariant("IDH1", "2", 500, "A", "G", True),
    TapestriVariant("IDH1", "2", 500, "A", "C", False),
    TapestriVariant("IDH1", "2", 800, "*", "C+T", False),
    TapestriVariant("EZH2", "7", 300, "A", "", True),
    TapestriVariant("EZH2", "7", 400, "G", "A", True),
)
#: Each clone's mutated variants, by position in ``TAPESTRI_VARIANTS``, so
#: every clone carries a combination of its own; the deletion is homozygous
#: and every other mutation heterozygous.
TAPESTRI_CLONE_VARIANTS = {
    0: (0,),
    1: (1,),
    2: (3,),
    3: (4,),
    4: (5,),
    5: (9,),
    6: (0, 9),
    7: (5, 9),
}
HOMOZYGOUS_VARIANTS = frozenset({3})
#: The amplicons: ID, CHROM, start and end.
TAPESTRI_AMPLICONS = (
    ("AML_NRAS_1", "1", 950, 1600),
    ("AML_DNMT3A_2", "1", 1950, 2150),
    ("AML_IDH1_3", "2", 450, 900),
    ("AML_EZH2_4", "7", 250, 450),
)
#: The antibodies: ID and tag sequence.
TAPESTRI_ANTIBODIES = (
    ("CD34", "CCTGGAATC"),
    ("CD3", "CCTCTATGC"),
    ("IgG1", "GCGGAACAT"),
)
TAPESTRI_METADATA = {
    "panel_name": "AML",
    "genome_version": "hg19",
    "pipeline_version": "2.0.2",
    "chemistry_version": "V2",
}
PROTEIN_METADATA = {"panel_name": "ab_barcodes_test", "pipeline_version": "2.0.1"}
#: DAb-seq's FLT3-ITD variant, which it adds from ITDseek with RD holding the
#: depth.
DABSEQ_ITD = "FLT3-ITD:chr13:28608251:T/TCTC"


@dataclass(frozen=True, slots=True)
class TapestriTruth:
    """Targeted DNA genotypes and protein reads of the synthetic experiment.

    Rows are the truth's cells, a twin cell of sample ``d4`` with the
    barcode of the first cell of sample ``d2``, and the droplets outside the
    store, of sample ``d2``. Columns of the variant arrays follow
    ``TAPESTRI_VARIANTS``; the variants of one position share their depth.

    Attributes:
        clones: The experiment's cells and clones.
        samples: Each row's sample.
        barcodes: Each row's barcode.
        depth: Each row's depth at each variant.
        alternative: Each row's alternative reads of each variant.
        genotypes: Each row's ``NGT`` of each variant, 0 to 3.
        quality: Each row's ``GQ`` of each variant.
        filtered: Whether the pipeline's filters removed each row.
        amplicon_reads: Each row's reads of each amplicon.
        protein_reads: Each row's reads of each antibody.
        overlap_row: The row whose two alternatives of ``chr2:500`` each
            show its one read, so their reads add up to more than its depth.
        no_call_row: The row with reads but no call at ``chr7:400``.
        uncovered_row: The row without reads at ``chr7:400``.
        wild_type_row: The row called wild type at ``chr1:1000`` without
            reads there, as Tapestri's files hold such calls.
    """

    clones: CloneTruth
    samples: np.ndarray
    barcodes: np.ndarray
    depth: np.ndarray
    alternative: np.ndarray
    genotypes: np.ndarray
    quality: np.ndarray
    filtered: np.ndarray
    amplicon_reads: np.ndarray
    protein_reads: np.ndarray
    overlap_row: int
    no_call_row: int
    uncovered_row: int
    wild_type_row: int

    @property
    def row_ids(self) -> list[str]:
        """Each row's cell ID, ``<sample>__<barcode>``."""
        return [
            f"{sample}__{barcode}"
            for sample, barcode in zip(self.samples, self.barcodes, strict=True)
        ]

    @property
    def n_rows(self) -> int:
        return len(self.samples)

    @property
    def twin_row(self) -> int:
        return len(self.clones.cell_ids)

    def frequency(self) -> np.ndarray:
        """Tapestri's ``AF``: the alternative reads over the depth, in percent."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(
                self.depth > 0, self.alternative / np.maximum(self.depth, 1) * 100, 0.0
            )


def tapestri_truth(seed: int = 0) -> TapestriTruth:
    """Draw the targeted genotypes and protein reads of ``clone_truth(seed)``.

    Every cell of a clone carries its clone's mutations, and every truth
    cell is read and called at every variant that passes, so the genotype
    clones of the mutant calls are the truth's clones. The cases that test
    the import fall on cells without a clone.
    """
    clones = clone_truth(seed)
    rng = np.random.default_rng(seed + 700)
    first_d2 = int(np.flatnonzero(clones.samples == "d2")[0])
    samples = np.concatenate(
        [clones.samples, ["d4"], np.full(len(clones.outside), "d2")]
    ).astype(object)
    barcodes = np.concatenate(
        [clones.barcodes, [clones.barcodes[first_d2]], clones.outside]
    ).astype(object)
    n_rows = len(samples)
    n_variants = len(TAPESTRI_VARIANTS)
    clone_of = np.concatenate([clones.clones, np.full(n_rows - len(clones.clones), -1)])
    positions = [(variant.chrom, variant.position) for variant in TAPESTRI_VARIANTS]
    distinct = list(dict.fromkeys(positions))
    depth = np.zeros((n_rows, n_variants), dtype=np.int64)
    alternative = np.zeros((n_rows, n_variants), dtype=np.int64)
    genotypes = np.zeros((n_rows, n_variants), dtype=np.int64)
    quality = np.zeros((n_rows, n_variants), dtype=np.int64)
    for row in range(n_rows):
        for key in distinct:
            columns = [index for index, at in enumerate(positions) if at == key]
            depth[row, columns] = int(rng.poisson(30)) + 8
        mutated = TAPESTRI_CLONE_VARIANTS.get(int(clone_of[row]), ())
        mutated_positions = {positions[column] for column in mutated}
        for column, variant in enumerate(TAPESTRI_VARIANTS):
            reads = int(depth[row, column])
            if column in mutated:
                if column in HOMOZYGOUS_VARIANTS:
                    alternative[row, column] = reads
                    genotypes[row, column] = 2
                else:
                    alternative[row, column] = int(
                        np.clip(rng.binomial(reads, 0.5), 2, reads - 2)
                    )
                    genotypes[row, column] = 1
            elif (
                variant.alternative
                and positions[column] not in mutated_positions
                and rng.random() < 0.1
            ):
                # A read error that the caller still calls wild type.
                alternative[row, column] = 1
            quality[row, column] = int(rng.integers(30, 100))
    unclonal = [row for row in range(len(clones.clones)) if int(clone_of[row]) < 0]
    overlap_row, no_call_row, uncovered_row, wild_type_row = unclonal[:4]
    # One read, which each alternative of chr2:500 claims.
    depth[overlap_row, [5, 6]] = 1
    alternative[overlap_row, [5, 6]] = 1
    genotypes[overlap_row, [5, 6]] = 3
    quality[overlap_row, [5, 6]] = 0
    genotypes[no_call_row, 9] = 3
    quality[no_call_row, 9] = 0
    depth[uncovered_row, 9] = 0
    alternative[uncovered_row, 9] = 0
    genotypes[uncovered_row, 9] = 3
    quality[uncovered_row, 9] = 0
    depth[wild_type_row, [0, 1]] = 0
    alternative[wild_type_row, [0, 1]] = 0
    quality[wild_type_row, [0, 1]] = 0
    filtered = np.zeros(n_rows, dtype=bool)
    filtered[len(clones.clones) + 1 :] = True
    filtered[unclonal[4]] = True
    amplicon_reads = rng.poisson(40, size=(n_rows, len(TAPESTRI_AMPLICONS)))
    amplicon_reads[unclonal[5], 1] = 0
    protein_reads = rng.poisson(15, size=(n_rows, len(TAPESTRI_ANTIBODIES)))
    protein_reads[:, 2] = rng.poisson(0.5, size=n_rows)
    return TapestriTruth(
        clones=clones,
        samples=samples,
        barcodes=barcodes,
        depth=depth,
        alternative=alternative,
        genotypes=genotypes,
        quality=quality,
        filtered=filtered,
        amplicon_reads=amplicon_reads.astype(np.int64),
        protein_reads=protein_reads.astype(np.int64),
        overlap_row=overlap_row,
        no_call_row=no_call_row,
        uncovered_row=uncovered_row,
        wild_type_row=wild_type_row,
    )


def _texts_dataset(group: h5py.Group, name: str, values: Sequence[str]) -> None:
    encoded = [str(value).encode("utf-8") for value in values]
    width = max([1, *(len(value) for value in encoded)])
    group.create_dataset(
        name, data=np.asarray(encoded, dtype=f"S{width}"), compression="gzip"
    )


def _metadata_group(group: h5py.Group, values: dict[str, object], samples) -> None:
    for name, value in values.items():
        if isinstance(value, str):
            group.create_dataset(name, data=np.bytes_(value.encode("utf-8")))
        else:
            group.create_dataset(name, data=value)
    group.create_dataset(
        "sample_name",
        data=np.asarray([[sample.encode()] for sample in samples], dtype="S8"),
    )


def _cell_rows(group: h5py.Group, truth: TapestriTruth, rows: np.ndarray) -> None:
    _texts_dataset(group, "barcode", truth.barcodes[rows].tolist())
    _texts_dataset(group, "sample_name", truth.samples[rows].tolist())


def write_tapestri(
    truth: TapestriTruth,
    path: Path,
    *,
    protein: bool = True,
    rows: Sequence[int] | None = None,
    chunks: tuple[int, int] = (8, 4),
) -> Path:
    """Write a Tapestri Pipeline 2.0 ``.dna+protein.h5`` file of the truth.

    The layout and dtypes follow Mission Bio's documented structure and the
    files it publishes: ``assays/dna_variants`` with the ``AF`` (float64),
    ``DP`` (int16), ``NGT`` and ``GQ`` (int8), ``RGQ`` and ``FILTER_MASK``
    layers, ``assays/dna_read_counts`` and, with ``protein``,
    ``assays/protein_read_counts``, each with ``ra``, ``ca`` and
    ``metadata``, ``all_barcodes`` of background barcodes, and the file's
    ``metadata``. Layers are chunked in ``chunks`` of cells by features.
    """
    chosen = np.arange(truth.n_rows) if rows is None else np.asarray(rows)
    samples = list(dict.fromkeys(truth.samples[chosen].tolist()))
    with h5py.File(path, "w") as handle:
        top = handle.create_group("metadata")
        top.create_dataset("date_created", data=np.bytes_(b"2026-01-01"))
        top.create_dataset("sdk_version", data=np.bytes_(b"3.1.2"))
        assays = handle.create_group("assays")
        variants = assays.create_group("dna_variants")
        columns = variants.create_group("ca")
        _texts_dataset(columns, "id", [variant.name for variant in TAPESTRI_VARIANTS])
        _texts_dataset(
            columns, "CHROM", [variant.chrom for variant in TAPESTRI_VARIANTS]
        )
        columns.create_dataset(
            "POS",
            data=np.asarray(
                [variant.position for variant in TAPESTRI_VARIANTS], np.int32
            ),
        )
        _texts_dataset(
            columns, "REF", [variant.reference for variant in TAPESTRI_VARIANTS]
        )
        _texts_dataset(
            columns, "ALT", [variant.alternative for variant in TAPESTRI_VARIANTS]
        )
        _texts_dataset(
            columns,
            "amplicon",
            [f"AML_{variant.gene}" for variant in TAPESTRI_VARIANTS],
        )
        columns.create_dataset(
            "filtered",
            data=np.asarray(
                [0 if variant.passing else 1 for variant in TAPESTRI_VARIANTS], np.uint8
            ),
        )
        columns.create_dataset(
            "QUAL", data=np.full(len(TAPESTRI_VARIANTS), 100.0, dtype=np.float32)
        )
        layers = variants.create_group("layers")
        layer_chunks = (min(chunks[0], len(chosen)), chunks[1])
        for name, values, dtype in (
            ("AF", truth.frequency()[chosen], np.float64),
            ("DP", truth.depth[chosen], np.int16),
            ("NGT", truth.genotypes[chosen], np.int8),
            ("GQ", truth.quality[chosen], np.int8),
            ("RGQ", np.zeros_like(truth.depth[chosen]), np.int32),
            (
                "FILTER_MASK",
                (truth.quality[chosen] < 30).astype(np.uint8),
                np.uint8,
            ),
        ):
            layers.create_dataset(
                name,
                data=np.asarray(values, dtype=dtype),
                chunks=layer_chunks,
                compression="gzip",
            )
        cell_rows = variants.create_group("ra")
        _cell_rows(cell_rows, truth, chosen)
        cell_rows.create_dataset(
            "filtered", data=truth.filtered[chosen].astype(np.uint8)
        )
        _metadata_group(
            variants.create_group("metadata"),
            {**TAPESTRI_METADATA, "n_passing_variants": 7, "gq_cutoff": 30},
            samples,
        )
        assay_layouts = [
            (
                "dna_read_counts",
                truth.amplicon_reads[chosen],
                {
                    "id": [amplicon[0] for amplicon in TAPESTRI_AMPLICONS],
                    "CHROM": [amplicon[1] for amplicon in TAPESTRI_AMPLICONS],
                    "start_pos": [amplicon[2] for amplicon in TAPESTRI_AMPLICONS],
                    "end_pos": [amplicon[3] for amplicon in TAPESTRI_AMPLICONS],
                },
                TAPESTRI_METADATA,
            )
        ]
        if protein:
            assay_layouts.append(
                (
                    "protein_read_counts",
                    truth.protein_reads[chosen],
                    {
                        "id": [antibody[0] for antibody in TAPESTRI_ANTIBODIES],
                        "antibody_sequence": [
                            antibody[1] for antibody in TAPESTRI_ANTIBODIES
                        ],
                        "antibody_id": ["" for _ in TAPESTRI_ANTIBODIES],
                    },
                    PROTEIN_METADATA,
                )
            )
        background = handle.create_group("all_barcodes")
        for name, reads, feature_columns, metadata in assay_layouts:
            assay = assays.create_group(name)
            feature_group = assay.create_group("ca")
            for key, values in feature_columns.items():
                if isinstance(values[0], str):
                    _texts_dataset(feature_group, key, values)
                else:
                    feature_group.create_dataset(key, data=np.asarray(values, np.int64))
            assay.create_group("layers").create_dataset(
                "read_counts",
                data=np.asarray(reads, dtype=np.int64),
                chunks=(min(chunks[0], len(chosen)), reads.shape[1]),
                compression="gzip",
            )
            _cell_rows(assay.create_group("ra"), truth, chosen)
            _metadata_group(assay.create_group("metadata"), dict(metadata), samples)
            # Background barcodes, with Mission Bio's -1 suffix.
            extra = background.create_group(name)
            _texts_dataset(extra.create_group("ca"), "id", feature_columns["id"])
            extra_rows = extra.create_group("ra")
            _texts_dataset(extra_rows, "barcode", ["ACGTACGTACGTACGTAA-1"])
            _texts_dataset(extra_rows, "sample_name", [samples[0]])
            extra.create_group("layers").create_dataset(
                "read_counts", data=np.ones((1, reads.shape[1]), dtype=np.int64)
            )
    return path


def write_dabseq(truth: TapestriTruth, path: Path) -> Path:
    """Write a DAb-seq ``.hdf5`` file of the truth's genotypes.

    The arrays follow the DAb-seq pipeline's layout: ``AD``, ``RD``, ``DP``
    (int16), ``GQ`` and ``GT`` (int8), cells by variants named
    ``<gene>:<chrom>:<pos>:<ref>/<alt>``, cells named
    ``<barcode>-1-<sample>``, and missing values as -1, as for the empty
    ALTs' reads and quality and one cell's depth. The variants of one
    position count the same reference reads, the reads of the position that
    no alternative claims; the reads that two alternatives of the overlap
    cell claim become one alternative's. FLT3-ITD is added with RD holding
    its depth and GT 0 without reads, as the pipeline adds it.
    """
    variants = list(TAPESTRI_VARIANTS)
    alternative = truth.alternative.copy()
    alternative[truth.overlap_row, 6] = 0
    depth = truth.depth.copy()
    genotypes = truth.genotypes.copy()
    quality = np.where(genotypes == 3, -1, truth.quality)
    reference = np.zeros_like(depth)
    positions = [(variant.chrom, variant.position) for variant in variants]
    for key in dict.fromkeys(positions):
        columns = [index for index, at in enumerate(positions) if at == key]
        claimed = alternative[:, columns].sum(axis=1)
        reference[:, columns] = (depth[:, columns[0]] - claimed)[:, None]
    empty = np.asarray([not variant.alternative for variant in variants])
    alternative[:, empty] = -1
    reference[:, empty] = -1
    # A missing depth, which a no-call without reads may hold.
    depth[truth.uncovered_row, 9] = -1
    n_rows = truth.n_rows
    itd_depth = np.zeros(n_rows, dtype=np.int64)
    itd_alternative = np.zeros(n_rows, dtype=np.int64)
    itd_genotype = np.zeros(n_rows, dtype=np.int64)
    itd_quality = np.zeros(n_rows, dtype=np.int64)
    itd_rows = np.flatnonzero(truth.clones.clones == 4)
    itd_depth[itd_rows] = 60
    itd_alternative[itd_rows] = 25
    itd_genotype[itd_rows] = 1
    itd_quality[itd_rows] = 100
    names = [variant.dabseq_name for variant in variants]
    # DAb-seq names a variant without a gene annotation with an empty gene.
    names = [
        name if variant.alternative else f":{variant.name}"
        for name, variant in zip(names, variants, strict=True)
    ]
    with h5py.File(path, "w") as handle:
        for name, values, own, dtype in (
            ("AD", alternative, itd_alternative, np.int16),
            ("RD", reference, itd_depth, np.int16),
            ("DP", depth, itd_depth, np.int16),
            ("GQ", quality, itd_quality, np.int8),
            ("GT", genotypes, itd_genotype, np.int8),
        ):
            handle.create_dataset(
                name,
                data=np.column_stack([values, own]).astype(dtype),
                chunks=(5, 4),
                compression="gzip",
            )
        _texts_dataset(handle, "VARIANTS", [*names, DABSEQ_ITD])
        _texts_dataset(
            handle,
            "CELL_BARCODES",
            [
                f"{barcode}-1-{sample}"
                for barcode, sample in zip(truth.barcodes, truth.samples, strict=True)
            ],
        )
        amplicons = handle.create_group("AMPLICONS")
        _texts_dataset(
            amplicons, "NAMES", [amplicon[0] for amplicon in TAPESTRI_AMPLICONS]
        )
        amplicons.create_dataset(
            "COUNTS", data=truth.amplicon_reads.astype(np.int32), compression="gzip"
        )
    return path
