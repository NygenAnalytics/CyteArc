"""Per-variant statistics of allele frequencies, gathered over bands of cells.

A variant is an alternative allele at a site, such as ``310T>C``: the
alleles that a ``variant_counts`` artifact gives the role ``"alternative"``.
A cell's allele frequency of a variant is its count of the allele over its
depth at the site, defined only where the depth is above 0. The statistics
are those of mgatk's ``variant_stats`` table, with mgatk's definitions:
``mean`` pools the counts of all cells over their depths, ``variance`` is
the sample variance of the allele frequencies of the covered cells, ``vmr``
their ratio, ``n_cells_conf_detected`` counts cells with at least 2 reads
of the allele on each strand, ``strand_correlation`` is the Pearson
correlation of the forward and reverse counts over the cells that show the
allele, and ``n_cells_over_5`` to ``n_cells_over_95`` count cells whose
frequency reaches 5% to 95%. ``mean_quality`` is MAESTER's mean base
quality: the mean over cells and strands with reads of the allele of their
mean quality.

Signac computes the same columns but sets the frequency of an uncovered cell
to 0, and by default replaces the frequency of a cell with depth below 10 by
the pooled mean before taking the variance; here an uncovered cell has no
frequency, as in mgatk.

:class:`VariantStatisticsAccumulator` takes the stored rows of one band of
cells at a time and keeps sums per variant, so the statistics need memory
for the variants, not the cells. These functions never touch storage.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from ..storage.refs import ArtifactRef
from .counts import SiteTable

__all__ = [
    "HETEROPLASMY_LEVELS",
    "INFORMATIVE_PRESETS",
    "THRESHOLDS",
    "VMR_OFFSET",
    "VariantSelection",
    "VariantStatistics",
    "VariantStatisticsAccumulator",
    "informative_variants",
    "resolve_thresholds",
    "variant_labels",
    "variant_selection_frame",
]

#: The columns that count cells whose allele frequency reaches a level, as
#: mgatk names them.
HETEROPLASMY_LEVELS: Mapping[str, float] = {
    "n_cells_over_5": 0.05,
    "n_cells_over_10": 0.10,
    "n_cells_over_20": 0.20,
    "n_cells_over_95": 0.95,
}
#: What mgatk and Signac add to the mean before dividing the variance by it.
VMR_OFFSET = 1e-11
# Reads of the allele on each strand that make a cell confidently detected.
_CONFIDENT_READS = 2
# Float64 holds every whole number below this exactly.
_EXACT_FLOAT = 1 << 53
# The largest strand count whose square stays below _EXACT_FLOAT, so that the
# products of strand counts are exact.
_MAX_STRAND_COUNT = 94_906_265


@dataclass(frozen=True, slots=True)
class VariantStatistics:
    """Per-variant statistics over a set of cells.

    Every array holds one value per variant, in the order of ``alleles``.

    Attributes:
        alleles: Each variant's allele, as a position in the site table.
        n_cells: The cells that the statistics cover.
        n_cells_covered: Cells with depth above 0 at the variant's site.
        mean: The pooled allele frequency: the counts of the allele over the
            depths at its site, summed over the cells.
        variance: The sample variance of the covered cells' allele
            frequencies, NaN with fewer than two covered cells.
        vmr: ``variance / (mean + 1e-11)``.
        max_heteroplasmy: The highest allele frequency of a cell.
        mean_coverage: The depth at the site summed over the cells and
            divided by their number, uncovered cells included.
        n_cells_over: For each column of ``HETEROPLASMY_LEVELS``, the cells
            whose allele frequency reaches its level.
        n_cells_conf_detected: Cells with at least 2 reads of the allele on
            each strand, or None for counts without strands.
        strand_correlation: The Pearson correlation of the forward and
            reverse counts of the cells that show the allele, NaN when fewer
            than two cells show it or a strand's counts do not vary, or None
            for counts without strands.
        mean_quality: The mean over the cells and strands that read the
            allele of their mean base quality, NaN without such reads, or
            None for counts without base qualities.
    """

    alleles: np.ndarray
    n_cells: int
    n_cells_covered: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    vmr: np.ndarray
    max_heteroplasmy: np.ndarray
    mean_coverage: np.ndarray
    n_cells_over: Mapping[str, np.ndarray] = field(default_factory=dict)
    n_cells_conf_detected: np.ndarray | None = None
    strand_correlation: np.ndarray | None = None
    mean_quality: np.ndarray | None = None

    @property
    def n_variants(self) -> int:
        return len(self.alleles)

    def arrays(self) -> dict[str, np.ndarray]:
        """The statistics by column name, without the alleles; columns that
        the counts cannot give are left out."""
        columns: dict[str, np.ndarray] = {
            "n_cells_covered": self.n_cells_covered,
            "mean": self.mean,
            "variance": self.variance,
            "vmr": self.vmr,
            **dict(self.n_cells_over),
            "max_heteroplasmy": self.max_heteroplasmy,
            "mean_coverage": self.mean_coverage,
        }
        if self.n_cells_conf_detected is not None:
            columns["n_cells_conf_detected"] = self.n_cells_conf_detected
        if self.strand_correlation is not None:
            columns["strand_correlation"] = self.strand_correlation
        if self.mean_quality is not None:
            columns["mean_quality"] = self.mean_quality
        return columns


def _exact_sums(positions: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    """Sum non-negative whole-number values per position exactly, as int64.

    ``np.bincount`` sums in float64, which holds whole numbers exactly below
    2**53; a sum of non-negative values passes through no larger partial
    sum, so a sum below 2**53 is exact.

    Raises:
        ValueError: If a sum reaches 2**53.
    """
    sums = np.bincount(positions, weights=values, minlength=size)
    if sums.size and float(sums.max()) >= _EXACT_FLOAT:
        raise ValueError(
            "A band's counts sum to 2**53 or more, beyond what is summed exactly; "
            "score fewer cells at once by lowering mem_budget"
        )
    return np.rint(sums).astype(np.int64)


class VariantStatisticsAccumulator:
    """Gather per-variant statistics from the stored rows of bands of cells.

    The variants are the alleles that ``sites`` gives the role
    ``"alternative"``. Each call of :meth:`add` takes every depth row and
    count row of a band of cells that no earlier band held, sorted by cell
    and then by site or allele, as a ``variant_counts`` artifact stores them;
    rows of cells outside the statistics must be left out. Depth and count
    sums are exact integers; the variance of the allele frequencies is merged
    band by band with Chan's update, so the bands' sizes change it only in
    the last bits.

    Args:
        sites: The sites and alleles of the counts.
        stranded: Whether count rows carry forward and reverse counts.
        qualities: Whether count rows carry each strand's mean base quality.
    """

    def __init__(self, sites: SiteTable, *, stranded: bool, qualities: bool) -> None:
        if qualities and not stranded:
            raise ValueError("Base qualities come with strand counts")
        self._n_sites = sites.n_sites
        candidates = np.flatnonzero(sites.allele_role == "alternative")
        self._alleles = candidates.astype(np.int64)
        self._variant_of_allele = np.full(sites.n_alleles, -1, dtype=np.int64)
        self._variant_of_allele[candidates] = np.arange(len(candidates))
        self._variant_site = sites.allele_site[candidates].astype(np.int64)
        n_variants = len(candidates)
        self._stranded = stranded
        self._qualities = qualities
        self._site_depth = np.zeros(sites.n_sites, dtype=np.int64)
        self._count = np.zeros(n_variants, dtype=np.int64)
        # The covered cells, mean and summed squared deviations of the allele
        # frequencies, merged band by band.
        self._covered = np.zeros(n_variants, dtype=np.int64)
        self._frequency_mean = np.zeros(n_variants, dtype=np.float64)
        self._frequency_m2 = np.zeros(n_variants, dtype=np.float64)
        self._max = np.full(n_variants, -np.inf, dtype=np.float64)
        self._over = {
            name: np.zeros(n_variants, dtype=np.int64) for name in HETEROPLASMY_LEVELS
        }
        if stranded:
            # Cells that show the allele, and the sums of the strand counts,
            # their squares and their product, for the Pearson correlation.
            self._shown = np.zeros(n_variants, dtype=np.int64)
            self._strand_sums = {
                name: np.zeros(n_variants, dtype=np.int64)
                for name in ("f", "r", "ff", "rr", "fr")
            }
            self._confident = np.zeros(n_variants, dtype=np.int64)
        if qualities:
            self._quality_sum = np.zeros(n_variants, dtype=np.float64)
            self._quality_entries = np.zeros(n_variants, dtype=np.int64)

    @property
    def n_variants(self) -> int:
        """The candidate variants: every alternative allele."""
        return len(self._alleles)

    def add(
        self,
        *,
        depth_cells: np.ndarray,
        depth_sites: np.ndarray,
        depth: np.ndarray,
        count_cells: np.ndarray,
        count_alleles: np.ndarray,
        counts: np.ndarray,
        forward: np.ndarray | None = None,
        reverse: np.ndarray | None = None,
        quality_forward: np.ndarray | None = None,
        quality_reverse: np.ndarray | None = None,
    ) -> None:
        """Add the depth and count rows of one band of cells.

        Args:
            depth_cells: Each depth row's cell, in ascending order.
            depth_sites: Each depth row's site, ascending within a cell.
            depth: Each depth row's depth, above 0.
            count_cells: Each count row's cell.
            count_alleles: Each count row's allele.
            counts: Each count row's count, above 0 and at most the depth.
            forward: Each count row's forward count, for stranded counts.
            reverse: Each count row's reverse count, for stranded counts.
            quality_forward: Each count row's mean forward base quality, NaN
                without forward reads, for counts with qualities.
            quality_reverse: The same for the reverse strand.

        Raises:
            ValueError: If the rows do not align, the depth rows are not
                sorted, an allele is counted where its site has no depth, or
                a count exceeds the depth.
        """
        depth_cells = np.asarray(depth_cells, dtype=np.int64)
        depth_sites = np.asarray(depth_sites, dtype=np.int64)
        depth = np.asarray(depth, dtype=np.int64)
        count_cells = np.asarray(count_cells, dtype=np.int64)
        count_alleles = np.asarray(count_alleles, dtype=np.int64)
        counts = np.asarray(counts, dtype=np.int64)
        if not len(depth_cells) == len(depth_sites) == len(depth):
            raise ValueError("The depth rows' cells, sites and depths must align")
        if not len(count_cells) == len(count_alleles) == len(counts):
            raise ValueError("The count rows' cells, alleles and counts must align")
        if self._stranded and (forward is None or reverse is None):
            raise ValueError("Stranded counts need forward and reverse counts")
        if self._qualities and (quality_forward is None or quality_reverse is None):
            raise ValueError("Counts with qualities need both strands' qualities")
        n_sites = self._n_sites
        self._site_depth += _exact_sums(depth_sites, depth, n_sites)
        band_covered = np.bincount(depth_sites, minlength=n_sites).astype(np.int64)
        variants = self._variant_of_allele[count_alleles]
        kept = variants >= 0
        variants = variants[kept]
        rows_cells = count_cells[kept]
        rows_counts = counts[kept]
        # Each variant row's depth: its cell's depth row at the variant's site.
        depth_keys = depth_cells * n_sites + depth_sites
        if len(depth_keys) > 1 and np.any(np.diff(depth_keys) <= 0):
            raise ValueError(
                "The depth rows must be sorted by cell and site, once each"
            )
        keys = rows_cells * n_sites + self._variant_site[variants]
        at = np.searchsorted(depth_keys, keys)
        found = at < len(depth_keys)
        found[found] = depth_keys[at[found]] == keys[found]
        if not found.all():
            raise ValueError(
                "An allele is counted in a cell that has no depth at its site"
            )
        row_depth = depth[at]
        if np.any(rows_counts > row_depth) or np.any(rows_counts <= 0):
            raise ValueError("A count lies outside 1 to its site's depth")
        frequency = rows_counts / row_depth
        n_variants = self.n_variants
        self._count += _exact_sums(variants, rows_counts, n_variants)
        self._merge_variance(variants, frequency, band_covered[self._variant_site])
        np.maximum.at(self._max, variants, frequency)
        for name, level in HETEROPLASMY_LEVELS.items():
            self._over[name] += np.bincount(
                variants[frequency >= level], minlength=n_variants
            )
        if self._stranded:
            assert forward is not None and reverse is not None
            f = np.asarray(forward, dtype=np.int64)[kept]
            r = np.asarray(reverse, dtype=np.int64)[kept]
            if len(f) and max(int(f.max()), int(r.max())) > _MAX_STRAND_COUNT:
                raise ValueError(
                    f"A strand count exceeds {_MAX_STRAND_COUNT} reads, beyond what "
                    "the strand correlation sums exactly"
                )
            self._shown += np.bincount(variants, minlength=n_variants)
            for name, values in (
                ("f", f),
                ("r", r),
                ("ff", f * f),
                ("rr", r * r),
                ("fr", f * r),
            ):
                self._strand_sums[name] += _exact_sums(variants, values, n_variants)
            confident = (f >= _CONFIDENT_READS) & (r >= _CONFIDENT_READS)
            self._confident += np.bincount(variants[confident], minlength=n_variants)
            if self._qualities:
                assert quality_forward is not None and quality_reverse is not None
                for strand_counts, quality in (
                    (f, np.asarray(quality_forward, dtype=np.float64)[kept]),
                    (r, np.asarray(quality_reverse, dtype=np.float64)[kept]),
                ):
                    read = (strand_counts > 0) & np.isfinite(quality)
                    self._quality_sum += np.bincount(
                        variants[read], weights=quality[read], minlength=n_variants
                    )
                    self._quality_entries += np.bincount(
                        variants[read], minlength=n_variants
                    )

    def _merge_variance(
        self, variants: np.ndarray, frequency: np.ndarray, band_covered: np.ndarray
    ) -> None:
        """Merge a band's allele frequencies into the running mean and squared
        deviations, with Chan's update.

        A covered cell without a count row has frequency 0, so the band's
        squared deviations add ``mean**2`` for each such cell.
        """
        n_variants = self.n_variants
        present = band_covered > 0
        if not present.any():
            return
        shown = np.bincount(variants, minlength=n_variants)
        totals = np.bincount(variants, weights=frequency, minlength=n_variants)
        band_mean = np.zeros(n_variants, dtype=np.float64)
        band_mean[present] = totals[present] / band_covered[present]
        deviations = np.bincount(
            variants,
            weights=(frequency - band_mean[variants]) ** 2,
            minlength=n_variants,
        )
        band_m2 = deviations + (band_covered - shown) * band_mean**2
        before = self._covered[present].astype(np.float64)
        added = band_covered[present].astype(np.float64)
        merged = before + added
        delta = band_mean[present] - self._frequency_mean[present]
        self._frequency_mean[present] += delta * added / merged
        self._frequency_m2[present] += (
            band_m2[present] + delta**2 * before * added / merged
        )
        self._covered[present] += band_covered[present]

    def finish(self, n_cells: int) -> VariantStatistics:
        """Return the statistics of the variants that some cell shows.

        Args:
            n_cells: The cells that the bands held, uncovered ones included.

        Raises:
            ValueError: If ``n_cells`` is below 1.
        """
        if n_cells < 1:
            raise ValueError("Statistics need at least one cell")
        shown = self._count > 0
        variant_site = self._variant_site[shown]
        site_depth = self._site_depth[variant_site]
        count = self._count[shown]
        covered = self._covered[shown]
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(site_depth > 0, count / site_depth, np.nan)
            variance = np.where(
                covered > 1, self._frequency_m2[shown] / (covered - 1), np.nan
            )
        vmr = variance / (mean + VMR_OFFSET)
        maximum = self._max[shown]
        statistics: dict[str, Any] = {}
        if self._stranded:
            statistics["n_cells_conf_detected"] = self._confident[shown]
            statistics["strand_correlation"] = _pearson(
                self._shown[shown],
                {name: sums[shown] for name, sums in self._strand_sums.items()},
            )
        if self._qualities:
            entries = self._quality_entries[shown]
            with np.errstate(divide="ignore", invalid="ignore"):
                statistics["mean_quality"] = np.where(
                    entries > 0, self._quality_sum[shown] / entries, np.nan
                )
        return VariantStatistics(
            alleles=self._alleles[shown],
            n_cells=int(n_cells),
            n_cells_covered=covered,
            mean=mean,
            variance=variance,
            vmr=vmr,
            max_heteroplasmy=np.where(np.isfinite(maximum), maximum, np.nan),
            mean_coverage=site_depth / float(n_cells),
            n_cells_over={name: values[shown] for name, values in self._over.items()},
            **statistics,
        )


def _pearson(n: np.ndarray, sums: Mapping[str, np.ndarray]) -> np.ndarray:
    """The Pearson correlation of forward and reverse counts from exact sums.

    The numerator and both variance terms are exact integers, computed with
    Python integers so that large sums cannot overflow; the correlation is
    NaN with fewer than two cells or a strand whose counts do not vary.
    """
    correlation = np.full(len(n), np.nan, dtype=np.float64)
    for position in np.flatnonzero(n > 1).tolist():
        cells = int(n[position])
        f = int(sums["f"][position])
        r = int(sums["r"][position])
        covariance = cells * int(sums["fr"][position]) - f * r
        variance_f = cells * int(sums["ff"][position]) - f * f
        variance_r = cells * int(sums["rr"][position]) - r * r
        if variance_f > 0 and variance_r > 0:
            correlation[position] = covariance / (
                float(variance_f) ** 0.5 * float(variance_r) ** 0.5
            )
    return correlation


#: The thresholds that select informative variants, each with the statistic
#: it tests and whether the statistic must exceed it (``">"``) or reach it
#: (``">="``), as the published rules compare them.
THRESHOLDS: Mapping[str, tuple[str, str]] = {
    "min_cells": ("n_cells_conf_detected", ">="),
    "min_strand_correlation": ("strand_correlation", ">="),
    "min_vmr": ("vmr", ">"),
    "min_depth": ("mean_coverage", ">"),
    "min_quality": ("mean_quality", ">"),
}
#: The published rules of each preset of ``informative``, by argument, read
#: only: ``"atac"`` is the rule of Lareau et al. (2020) for mtscATAC-seq as
#: Signac's vignette applies it: at least 5 cells with 2 reads on each strand,
#: a strand correlation of at least 0.65 and a VMR above 0.01. ``"rna"`` is
#: MAESTER's for RNA libraries, whose strands follow the transcript (Miller et
#: al. 2022): a mean depth above 20 and a mean base quality above 30, with
#: Signac's VMR above 0.01 for the variability that MAESTER tests by
#: percentiles of the allele frequencies.
INFORMATIVE_PRESETS: Mapping[str, Mapping[str, int | float]] = MappingProxyType(
    {
        "atac": MappingProxyType(
            {"min_cells": 5, "min_strand_correlation": 0.65, "min_vmr": 0.01}
        ),
        "rna": MappingProxyType(
            {"min_depth": 20.0, "min_quality": 30.0, "min_vmr": 0.01}
        ),
    }
)


def resolve_thresholds(
    preset: str | None, given: Mapping[str, float | int | None]
) -> dict[str, int | float | None]:
    """Combine a preset's thresholds with the thresholds given by name.

    A threshold given as a number replaces the preset's; one left as None
    keeps the preset's, or no test without a preset. ``min_cells`` resolves
    to an integer and the other thresholds to floats.

    Raises:
        TypeError: If the preset is not text or None, or a threshold is not a
            real number.
        ValueError: If the preset or a threshold name is unknown, a
            threshold is not finite, ``min_cells`` is not a whole number at
            least 0, or no threshold results.
    """
    if preset is not None and not isinstance(preset, str):
        raise TypeError(
            f"preset must be one of {', '.join(INFORMATIVE_PRESETS)}, or None, not "
            f"{preset!r}"
        )
    if preset is not None and preset not in INFORMATIVE_PRESETS:
        raise ValueError(
            f"preset must be one of {', '.join(INFORMATIVE_PRESETS)}, or None, not "
            f"{preset!r}"
        )
    unknown = sorted(set(given) - set(THRESHOLDS))
    if unknown:
        raise ValueError(f"Unknown thresholds {unknown}")
    resolved: dict[str, int | float | None] = {
        name: (None if preset is None else INFORMATIVE_PRESETS[preset].get(name))
        for name in THRESHOLDS
    }
    for name, value in given.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int | float | np.number):
            raise TypeError(f"{name} must be a number or None")
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"{name} must be finite")
        if name == "min_cells" and (number < 0 or number != int(number)):
            raise ValueError("min_cells must be a whole number of cells, 0 or more")
        resolved[name] = number
    resolved = {
        name: None
        if value is None
        else (int(value) if name == "min_cells" else float(value))
        for name, value in resolved.items()
    }
    if all(value is None for value in resolved.values()):
        raise ValueError(
            "Selecting informative variants needs a preset ('atac' or 'rna') or "
            f"at least one threshold of {', '.join(THRESHOLDS)}"
        )
    return resolved


def informative_variants(
    statistics: Mapping[str, np.ndarray],
    thresholds: Mapping[str, int | float | None],
    *,
    n_variants: int,
) -> np.ndarray:
    """Whether each variant passes every threshold that is set.

    A statistic that is NaN, such as the strand correlation of a variant
    that one cell shows, fails its test.

    Raises:
        ValueError: If a set threshold tests a statistic that the counts
            cannot give, such as a strand correlation without strand counts.
    """
    passed = np.ones(n_variants, dtype=bool)
    for name, value in thresholds.items():
        if value is None:
            continue
        column, comparison = THRESHOLDS[name]
        if column not in statistics:
            raise ValueError(
                f"{name} tests {column}, which these counts cannot give: "
                + (
                    "they hold no base qualities"
                    if column == "mean_quality"
                    else "they hold no forward and reverse counts"
                )
                + f"; pass {name}=None with a preset, or leave it out"
            )
        values = np.asarray(statistics[column], dtype=np.float64)
        with np.errstate(invalid="ignore"):
            test = values >= value if comparison == ">=" else values > value
        passed &= test & ~np.isnan(values)
    return passed


def variant_labels(sites: SiteTable, alleles: np.ndarray) -> np.ndarray:
    """Name variants as mgatk does, after their site: ``chrM:310T>C``.

    A site with coordinates on an unnamed contig gives mgatk's own name,
    ``310T>C``, and a named site its name, a space and the change, as in
    ``IDH2_Mut1 WT>MUT``. A site without a reference allele names the
    alternative alone.
    """
    alleles = np.asarray(alleles, dtype=np.int64)
    references = sites.reference_alleles()
    labels = np.empty(len(alleles), dtype=object)
    for row, allele in enumerate(alleles.tolist()):
        site = int(sites.allele_site[allele])
        reference = int(references[site])
        change = (
            f"{sites.allele[reference]}>{sites.allele[allele]}"
            if reference >= 0
            else f">{sites.allele[allele]}"
        )
        separator = " " if sites.site_contig[site] < 0 else ""
        labels[row] = f"{sites.label(site)}{separator}{change}"
    return labels


@dataclass(frozen=True, slots=True)
class VariantSelection:
    """Per-variant statistics and the informative variants they select.

    Attributes:
        variants: One row per variant that some cell shows, indexed by its
            label (such as ``chrM:310T>C``): ``site``, ``position`` (-1 for a
            named site), ``reference`` and ``alternative``; mgatk's
            ``variant_stats`` columns ``mean``, ``variance``, ``vmr``,
            ``n_cells_conf_detected`` and ``strand_correlation`` (NA without
            strand counts), ``n_cells_over_5`` to ``n_cells_over_95``,
            ``max_heteroplasmy`` and ``mean_coverage``; ``n_cells_covered``,
            ``mean_quality`` (NaN without base qualities), ``masked`` and
            ``informative``.
        counts: The ``variant_counts`` artifact that the statistics read.
        cell_selection: The cell selection that the statistics were
            restricted to, or None for every cell that the counts' files name.
        preset: The preset that set the thresholds, or None.
        thresholds: Each threshold as the selection applied it, None for a
            test not made, from the artifact's provenance parameters:
            ``min_cells`` an integer and the others floats.
        mask: The masks whose variants are never informative, as the
            selection's ``mask`` names them.
        n_cells: The cells that the statistics cover.
        stranded: Whether the counts give forward and reverse reads, and so
            ``n_cells_conf_detected`` and ``strand_correlation``.
        qualities: Whether the counts give base qualities, and so
            ``mean_quality``.
    """

    variants: pd.DataFrame = field(repr=False)
    counts: ArtifactRef
    cell_selection: ArtifactRef | None
    preset: str | None
    thresholds: Mapping[str, int | float | None]
    mask: tuple[str, ...]
    n_cells: int
    stranded: bool
    qualities: bool


# The columns of a selection's table after the variant's identity, in
# mgatk's order of its statistics, then this module's.
_FRAME_STATISTICS = (
    "mean",
    "variance",
    "vmr",
    "n_cells_conf_detected",
    "strand_correlation",
    *HETEROPLASMY_LEVELS,
    "max_heteroplasmy",
    "mean_coverage",
    "n_cells_covered",
    "mean_quality",
)
# Columns of whole numbers, which are NA rather than NaN where the counts
# cannot give them.
_COUNT_COLUMNS = frozenset(
    {"n_cells_conf_detected", "n_cells_covered", *HETEROPLASMY_LEVELS}
)


def variant_selection_frame(
    sites: SiteTable, alleles: np.ndarray, arrays: Mapping[str, np.ndarray]
) -> pd.DataFrame:
    """Return the table of a :class:`VariantSelection` from stored arrays.

    Args:
        sites: The sites and alleles of the counts.
        alleles: Each variant's allele.
        arrays: The stored statistics, ``masked`` and ``informative``, by
            name; a statistic that the counts cannot give is absent.
    """
    alleles = np.asarray(alleles, dtype=np.int64)
    n_variants = len(alleles)
    variant_sites = sites.allele_site[alleles].astype(np.int64)
    references = sites.reference_alleles()[variant_sites]
    labels = sites.labels()
    columns: dict[str, Any] = {
        "site": labels[variant_sites],
        "position": sites.site_position[variant_sites],
        "reference": [
            str(sites.allele[reference]) if reference >= 0 else None
            for reference in references.tolist()
        ],
        "alternative": sites.allele[alleles],
    }
    for name in _FRAME_STATISTICS:
        if name in arrays:
            values = np.asarray(arrays[name])
            columns[name] = (
                values.astype(np.int64)
                if name in _COUNT_COLUMNS
                else values.astype(np.float64)
            )
        elif name in _COUNT_COLUMNS:
            columns[name] = pd.array([pd.NA] * n_variants, dtype="Int64")
        else:
            columns[name] = np.full(n_variants, np.nan)
    columns["masked"] = np.asarray(arrays["masked"], dtype=bool)
    columns["informative"] = np.asarray(arrays["informative"], dtype=bool)
    return pd.DataFrame(
        columns,
        index=pd.Index(variant_labels(sites, alleles), name="variant", dtype=object),
    )
