"""Ambient RNA profiles, contamination fractions, and corrected counts.

These functions implement SoupX as Young and Behjati describe it in the paper
and its Supplementary Methods: Young MD, Behjati S (2020). SoupX removes ambient
RNA contamination from droplet-based single-cell RNA sequencing data.
GigaScience 9(12):giaa151, doi:10.1093/gigascience/giaa151. They follow the
paper's description; no SoupX code was read or adapted.

The counts ``n_gc`` of gene ``g`` in droplet ``c`` are the counts of its cell,
``m_gc``, plus counts of cell-free ambient RNA, the soup:
``n_gc = m_gc + rho_c N_c b_g``, where ``N_c`` is the droplet's total count,
``rho_c`` its contamination fraction, and ``b_g`` the share of the soup that
gene ``g`` contributes.

- The soup profile ``b_g`` is the share of gene ``g`` among the summed counts
  of droplets that hold no cell (equation S1).
- Genes that some cells cannot express, such as haemoglobin genes outside
  erythroid cells, measure contamination: in such a cell every count of those
  genes comes from the soup. A cell expresses a gene set when its counts of the
  set are significantly larger than the soup alone would give it, a Poisson
  test at the largest plausible contamination (equation S3), here with a
  Benjamini-Hochberg false discovery rate. With a clustering, every cell of a
  cluster that holds such a cell is excluded too. The contamination fraction
  is the ratio of the observed counts of the gene sets to the counts that pure
  soup would give, summed over the remaining cells (equations S2 and S4).
- Correction removes ``rho_c N_c`` counts from each cell. The most likely
  removal under a multinomial model with the soup probabilities, subject to
  removing no more of a gene than the cell holds, empties genes in order of
  ``n_gc / b_g`` and takes the rest from the other genes in proportion to
  ``b_g``. As in the paper, each gene's removal is rounded to whole counts at
  random, up with a probability equal to its fractional part, so that summed
  over cells every gene loses its share of the soup on average. The draws
  follow systematic sampling with probabilities proportional to size (Madow
  WG (1949), On the theory of systematic sampling, II. Annals of
  Mathematical Statistics 20(3):333-354, doi:10.1214/aoms/1177729988), one
  seeded draw per cell, so each cell also loses its removal rounded down or
  up.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, poisson

from ..utils.arguments import float_argument, integer_argument
from ..utils.arrays import sort_categories
from ..utils.warnings import warn
from .cell_calling import (
    _GOLDEN,
    _MIX_1,
    _MIX_2,
    _SHIFT_11,
    _SHIFT_27,
    _SHIFT_30,
    _SHIFT_31,
    _UNIT_INTERVAL,
)

__all__ = [
    "AmbientContamination",
    "ContaminationFit",
    "estimate_contamination",
    "estimation_cells",
    "expressing_cells",
    "gene_set_evidence",
    "remove_ambient_counts",
    "soup_fractions",
]

type ContaminationLevel = Literal["global", "cluster"]

# A Poisson count is 0 with probability exp(-mean), so a count of 0 puts the
# mean below -log(0.05), about 3, with 95% confidence.
_ZERO_COUNT_BOUND = -math.log(0.05)

# Running sums over a cell's genes are exact integers, so a cell's corrected
# counts do not depend on the other rows that share a call. Soup weights are
# counted in units of 2**-62 of the soup, and fractional parts of a count in
# units of 2**-40; a row of fewer than 2**23 genes keeps every sum below 2**63.
_SOUP_UNITS = 2.0**62
_FRACTION_BITS = 40
_FRACTION_UNITS = 2**_FRACTION_BITS
_MAX_GENES = 2**23

# The rounding draws are the first draws of counter-based SplitMix64 streams
# built as cell calling builds its streams, from its constants, keyed by the
# seed and each row's key. This salt, "AmbientR" in ASCII, keeps them apart
# from the streams of EmptyDrops.
_ROUNDING_SALT = np.uint64(0x416D6269656E7452)


@dataclass(frozen=True, slots=True)
class ContaminationFit:
    """Contamination fractions estimated from gene sets.

    Attributes:
        fraction: The global estimate over every gene set and cell used.
        set_fractions: The estimate from each gene set alone, NaN for a set
            that no used cell measures.
        cluster_labels: The distinct cluster labels, in natural order, when
            the estimate is made for each cluster; otherwise None.
        cluster_fractions: The estimate of each cluster in ``cluster_labels``,
            or None.
        cell_fractions: The fraction of each cell: the global estimate, or its
            cluster's estimate.
    """

    fraction: float
    set_fractions: np.ndarray
    cluster_labels: np.ndarray | None
    cluster_fractions: np.ndarray | None
    cell_fractions: np.ndarray


@dataclass(frozen=True, slots=True)
class AmbientContamination:
    """A saved ambient RNA contamination estimate and its evidence.

    Attributes:
        fractions: The contamination fraction of each cell, indexed by cell
            ``ids``. Correction removes this fraction of each cell's counts.
        estimate: The global estimate over every gene set and cell used, or
            the fraction that was supplied.
        method: ``"gene_sets"`` for an estimate from gene sets, ``"fixed"``
            for a supplied fraction.
        level: ``"global"`` when every cell has the global estimate,
            ``"cluster"`` when each cell has its cluster's estimate.
        set_estimates: The estimate from each gene set alone, None for a set
            that no used cell measures.
        cluster_estimates: The estimate of each cluster, keyed by the cluster
            label as text; empty unless ``level`` is ``"cluster"``.
        evidence: One row per cell and gene set, with the cell ``ids``, the
            ``gene_set``, its ``observed`` counts, the ``expected`` counts of a
            droplet of pure soup with the cell's total, the Poisson
            ``p_value`` of the observed counts at the largest plausible
            contamination, ``expressing``, whether the cell expresses the set
            at the false discovery rate, and ``used``, whether the cell
            measured the fraction. None for a supplied fraction.
    """

    fractions: pd.Series = field(repr=False)
    estimate: float
    method: Literal["gene_sets", "fixed"]
    level: ContaminationLevel
    set_estimates: dict[str, float | None]
    cluster_estimates: dict[str, float]
    evidence: pd.DataFrame | None = field(repr=False)


def soup_fractions(counts: Any) -> np.ndarray:
    """Return the soup profile ``b_g`` from the summed counts of empty droplets.

    This is equation S1 of Young and Behjati (2020): the share of each gene
    among all counts of the droplets that hold no cell.

    Args:
        counts: The summed count of each gene over the empty droplets.

    Returns:
        The float64 fraction of each gene; the fractions sum to 1.

    Raises:
        ValueError: If a count is negative or not finite, or the counts sum to 0.
    """
    values = np.asarray(counts, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Soup counts must be one-dimensional")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Soup counts must be finite and non-negative")
    total = float(values.sum())
    if total <= 0:
        raise ValueError(
            "The empty droplets hold no counts, so they define no soup profile"
        )
    fractions: np.ndarray = values / total
    return fractions


def gene_set_evidence(
    observed: Any,
    totals: Any,
    set_soup: Any,
    *,
    max_contamination: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Test each cell's gene-set counts against pure soup.

    ``expected[c, s]`` is the count of set ``s`` that a droplet of pure soup
    with the total ``N_c`` of cell ``c`` holds, ``N_c`` times the soup
    fraction of the set. The p-value is the Poisson probability of at least
    the observed count when the mean is ``max_contamination`` times the
    expected count, equation S3 of Young and Behjati (2020).

    Args:
        observed: Counts of each gene set in each cell, cells by sets.
        totals: Total count of each cell.
        set_soup: Soup fraction of each gene set, the sum of ``b_g`` over its
            genes.
        max_contamination: The largest plausible contamination fraction,
            ``rho_max`` of the paper, greater than 0 and at most 1.

    Returns:
        The expected counts and the p-values, both cells by sets.

    Raises:
        ValueError: If an argument has the wrong shape or an invalid value.
    """
    counts = np.asarray(observed, dtype=np.float64)
    cell_totals = np.asarray(totals, dtype=np.float64)
    fractions = np.asarray(set_soup, dtype=np.float64)
    rho_max = float_argument(max_contamination, "max_contamination")
    if not 0.0 < rho_max <= 1.0:
        raise ValueError("max_contamination must be greater than 0 and at most 1")
    if (
        counts.ndim != 2
        or cell_totals.shape != (counts.shape[0],)
        or fractions.shape != (counts.shape[1],)
    ):
        raise ValueError(
            "observed must be cells by gene sets, with one total per cell and "
            "one soup fraction per gene set"
        )
    for name, values in (
        ("observed", counts),
        ("totals", cell_totals),
        ("set_soup", fractions),
    ):
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name} must be finite and non-negative")
    expected = cell_totals[:, None] * fractions[None, :]
    p_values = np.asarray(poisson.sf(counts - 1, rho_max * expected), dtype=np.float64)
    return expected, p_values


def expressing_cells(p_values: Any, fdr: float = 0.05) -> np.ndarray:
    """Return which cells express each gene set.

    A cell expresses a set when its p-value, adjusted with the
    Benjamini-Hochberg procedure over the cells of that set, is below ``fdr``.

    Args:
        p_values: P-values from :func:`gene_set_evidence`, cells by sets.
        fdr: The false discovery rate, greater than 0 and less than 1.

    Returns:
        A boolean array of the shape of ``p_values``.

    Raises:
        ValueError: If ``fdr`` is out of range or ``p_values`` is not a
            two-dimensional array of probabilities.
    """
    rate = float_argument(fdr, "fdr")
    if not 0.0 < rate < 1.0:
        raise ValueError("fdr must be greater than 0 and less than 1")
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("p_values must be cells by gene sets")
    if not ((values >= 0) & (values <= 1)).all():
        raise ValueError("p_values must lie between 0 and 1")
    if values.shape[0] == 0:
        return np.zeros(values.shape, dtype=bool)
    adjusted = false_discovery_control(values, axis=0, method="bh")
    expressing: np.ndarray = adjusted < rate
    return expressing


def estimation_cells(expressing: Any, clusters: Any | None = None) -> np.ndarray:
    """Return the cells that estimate the contamination from each gene set.

    A cell is used for a set unless it expresses the set. With ``clusters``,
    every cell of a cluster that holds a cell expressing the set is left out
    too, as Young and Behjati (2020) recommend.

    Args:
        expressing: From :func:`expressing_cells`, cells by sets.
        clusters: Optional cluster label of each cell.

    Returns:
        A boolean array of the shape of ``expressing``.

    Raises:
        ValueError: If ``clusters`` does not have one label per cell.
    """
    flags = np.asarray(expressing, dtype=bool)
    if flags.ndim != 2:
        raise ValueError("expressing must be cells by gene sets")
    if clusters is None:
        used: np.ndarray = ~flags
        return used
    labels = np.asarray(clusters)
    if labels.shape != (flags.shape[0],):
        raise ValueError("clusters must hold one label per cell")
    _, codes = np.unique(labels, return_inverse=True)
    codes = np.asarray(codes).reshape(-1)
    n_clusters = int(codes.max()) + 1 if codes.size else 0
    flagged = np.zeros((n_clusters, flags.shape[1]), dtype=bool)
    np.logical_or.at(flagged, codes, flags)
    clustered: np.ndarray = ~flagged[codes]
    return clustered


def _ratio(observed: np.ndarray, expected: np.ndarray) -> float | None:
    """Return equation S4 over the given pairs, or None without evidence."""
    denominator = float(expected.sum())
    if denominator <= 0:
        return None
    return float(observed.sum()) / denominator


def estimate_contamination(
    observed: Any,
    expected: Any,
    used: Any,
    *,
    clusters: Any | None = None,
    level: ContaminationLevel = "global",
    set_names: Sequence[str] | None = None,
) -> ContaminationFit:
    """Estimate contamination fractions from gene sets, equation S4.

    The estimate is the observed count of the gene sets summed over the cells
    used for them, divided by the counts of the same cells and sets in a
    droplet of pure soup. The global estimate pools every set and cell. With
    ``level="cluster"`` each cluster also gets the estimate from its own
    cells, and its cells take that estimate.

    An estimate is exactly 0 when its cells hold no count of the gene sets
    they measure although pure soup would give them some. A count of 0 only
    bounds the fraction, below ``-log(0.05) / expected``, about ``3 /
    expected``, with 95% confidence, so each such estimate (global, of a gene
    set when there are several, or of a cluster) issues a warning that names
    it, its cells, the expected count, and that bound. The estimate itself is
    kept.

    Args:
        observed: Counts of each gene set in each cell, cells by sets.
        expected: Pure-soup counts from :func:`gene_set_evidence`.
        used: Cells used for each set, from :func:`estimation_cells`.
        clusters: Cluster label of each cell; required for ``level="cluster"``.
        level: ``"global"`` or ``"cluster"``.
        set_names: The name of each gene set, for warnings. Defaults to the
            sets' positions.

    Returns:
        The estimates and the fraction of each cell.

    Raises:
        ValueError: If an argument has the wrong shape, no used cell measures
            the contamination, or with ``level="cluster"``, a cluster has no
            such cell.
    """
    counts = np.asarray(observed, dtype=np.float64)
    soup = np.asarray(expected, dtype=np.float64)
    mask = np.asarray(used, dtype=bool)
    if counts.ndim != 2 or soup.shape != counts.shape or mask.shape != counts.shape:
        raise ValueError("observed, expected, and used must all be cells by gene sets")
    if level not in ("global", "cluster"):
        raise ValueError("level must be 'global' or 'cluster'")
    names = (
        [str(position) for position in range(counts.shape[1])]
        if set_names is None
        else [str(name) for name in set_names]
    )
    if len(names) != counts.shape[1]:
        raise ValueError("set_names must hold one name per gene set")
    fraction = _ratio(counts[mask], soup[mask])
    if fraction is None:
        raise ValueError(
            "No cell can measure the contamination: every cell expresses each "
            "gene set, belongs to a cluster that does, or holds none of the "
            "soup's counts of the set. Use gene sets that some clusters do not "
            "express, or supply a fraction."
        )
    set_fractions = np.full(counts.shape[1], np.nan, dtype=np.float64)
    for position in range(counts.shape[1]):
        cells = mask[:, position]
        value = _ratio(counts[cells, position], soup[cells, position])
        if value is not None:
            set_fractions[position] = value
    n_cells = counts.shape[0]
    # Each estimate of 0: what it is, its measuring cells, the genes they
    # measure it with, their pure-soup count, and what a correction with it does.
    zeros: list[tuple[str, int, str, float, str]] = []
    if fraction == 0.0:
        zeros.append(
            (
                "The global contamination estimate",
                int(mask.any(axis=1).sum()),
                "the measured gene sets",
                float(soup[mask].sum()),
                f"a correction with this estimate would leave {_cells(n_cells)} "
                "uncorrected",
            )
        )
    if len(names) > 1:
        for position, name in enumerate(names):
            if set_fractions[position] == 0.0:
                cells = mask[:, position]
                zeros.append(
                    (
                        f"The contamination estimate from gene set {name!r}",
                        int(cells.sum()),
                        "the set",
                        float(soup[cells, position].sum()),
                        "a correction with an estimate from this gene set alone "
                        f"would leave {_cells(n_cells)} uncorrected",
                    )
                )
    if level == "global":
        _warn_zero_estimates(zeros)
        return ContaminationFit(
            fraction=fraction,
            set_fractions=set_fractions,
            cluster_labels=None,
            cluster_fractions=None,
            cell_fractions=np.full(n_cells, fraction, dtype=np.float64),
        )
    if clusters is None:
        raise ValueError("level='cluster' needs the cluster of each cell")
    labels = np.asarray(clusters)
    if labels.shape != (n_cells,):
        raise ValueError("clusters must hold one label per cell")
    distinct = np.asarray(
        sort_categories(np.unique(labels).tolist()), dtype=labels.dtype
    )
    cluster_fractions = np.empty(len(distinct), dtype=np.float64)
    missing: list[str] = []
    cell_fractions = np.empty(n_cells, dtype=np.float64)
    for position, label in enumerate(distinct):
        members = labels == label
        pairs = mask & members[:, None]
        value = _ratio(counts[pairs], soup[pairs])
        if value is None:
            missing.append(str(label))
            continue
        cluster_fractions[position] = value
        cell_fractions[members] = value
        if value == 0.0:
            zeros.append(
                (
                    f"The contamination estimate of cluster {str(label)!r}",
                    int(pairs.any(axis=1).sum()),
                    "the measured gene sets",
                    float(soup[pairs].sum()),
                    "a correction at the cluster level would leave its "
                    f"{_cells(int(members.sum()))} uncorrected",
                )
            )
    if missing:
        raise ValueError(
            "These clusters have no cell that can measure the contamination, "
            f"because they express every gene set: {', '.join(missing)}. Add a "
            "gene set that they do not express, or use level='global'."
        )
    _warn_zero_estimates(zeros)
    return ContaminationFit(
        fraction=fraction,
        set_fractions=set_fractions,
        cluster_labels=distinct,
        cluster_fractions=cluster_fractions,
        cell_fractions=cell_fractions,
    )


def _cells(count: int) -> str:
    return f"{count} cell" if count == 1 else f"{count} cells"


def _warn_zero_estimates(zeros: list[tuple[str, int, str, float, str]]) -> None:
    """Warn about each estimate of 0 that pure soup would have given counts."""
    for subject, n_measuring, genes, expected, consequence in zeros:
        warn(
            f"{subject} is 0: no count of {genes} was observed in the "
            f"{_cells(n_measuring)} measuring it, where pure soup with their "
            f"totals would give {expected:.3g}. A count of 0 puts the fraction "
            f"below {_ZERO_COUNT_BOUND / expected:.3g} (the one-sided 95% "
            f"Poisson bound, {_ZERO_COUNT_BOUND:.3g} / {expected:.3g}), not at "
            f"0, so {consequence}. Check that the gene sets and the soup profile "
            "suit these cells."
        )


def remove_ambient_counts(
    counts: Any,
    fractions: Any,
    soup: Any,
    *,
    random_seed: int = 0,
    row_keys: Any | None = None,
) -> np.ndarray:
    """Return integer counts with each cell's soup counts removed.

    Each row is one cell. A cell with contamination fraction ``rho`` and total
    ``N`` loses ``T = min(rho * N, S)`` counts, where ``S`` is its count of the
    genes in the soup. The removal is the multinomial maximum-likelihood
    allocation of Young and Behjati (2020): it never removes more of a gene
    than the cell holds, empties genes in increasing order of their count
    divided by their soup fraction, and takes the rest from the remaining
    genes in proportion to their soup fraction. A gene that the soup does not
    hold is never reduced.

    Each gene's removal ``t`` is then rounded to whole counts at random, as the
    paper does, by systematic sampling with probabilities proportional to size
    (Madow WG (1949), On the theory of systematic sampling, II. Annals of
    Mathematical Statistics 20(3):333-354, doi:10.1214/aoms/1177729988). Take
    a cell's genes in gene order, let ``f_j`` be the fractional part of the
    removal ``t_j`` of gene ``j``, ``C_j`` the sum of ``f`` over the cell's
    genes up to and including ``j`` (``C_0 = 0``), and ``u`` one uniform draw
    on [0, 1) for the cell. Gene ``j`` loses
    ``floor(t_j) + floor(C_j + u) - floor(C_(j-1) + u)`` counts, so:

    - each gene is rounded up with a probability equal to its fractional part,
      and its expected removal is ``t``: summed over cells, every gene loses
      its share of the soup on average, however small its share in one cell;
    - each cell loses ``T`` rounded down or up, and exactly ``T`` when it is a
      whole number;
    - no gene loses more counts than the cell holds, and a gene that the
      removal empties loses all of them.

    The sums ``C`` are counted exactly in units of 2**-40 of a count, which
    sets the resolution of these probabilities; the units of a cell are
    aligned to sum to ``T`` less its whole parts. A cell's draw is the first
    draw of a counter-based SplitMix64 stream (Steele, Lea and Flood 2014,
    doi:10.1145/2714064.2660195) keyed by ``random_seed`` and the row's key,
    and every sum is taken over the row's own genes, so a seed and keys give
    the same counts however the rows are split into calls.

    Args:
        counts: Non-negative integer counts, cells by genes, with fewer than
            2**23 genes.
        fractions: The contamination fraction of each cell, at least 0.
        soup: The soup fraction ``b_g`` of each gene, at least 0. Only the
            ratios between genes matter.
        random_seed: Seed of the rounding draws, a non-negative integer.
        row_keys: One non-negative integer key per row, which with
            ``random_seed`` sets the row's draw; distinct keys give independent
            draws. Defaults to the row positions. To correct a matrix in row
            blocks, pass each row's position in the whole matrix.

    Returns:
        The corrected counts, in the dtype of ``counts``: never negative and
        never more than the counts.

    Raises:
        TypeError: If ``counts`` or ``row_keys`` does not hold integers, or
            ``random_seed`` is not an integer.
        ValueError: If an argument has the wrong shape or an invalid value.
    """
    values = np.asarray(counts)
    if values.dtype.kind not in "iu":
        raise TypeError("counts must hold integers")
    if values.ndim != 2:
        raise ValueError("counts must be cells by genes")
    if values.shape[1] >= _MAX_GENES:
        raise ValueError(f"counts must hold fewer than {_MAX_GENES} genes")
    if values.dtype.kind == "i" and values.size and int(values.min()) < 0:
        raise ValueError("counts must not be negative")
    rho = np.asarray(fractions, dtype=np.float64)
    if rho.shape != (values.shape[0],):
        raise ValueError("fractions must hold one fraction per cell")
    if not np.isfinite(rho).all() or (rho < 0).any():
        raise ValueError("fractions must be finite and non-negative")
    weights = np.asarray(soup, dtype=np.float64)
    if weights.shape != (values.shape[1],):
        raise ValueError("soup must hold one fraction per gene")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("soup must be finite and non-negative")
    seed = integer_argument(random_seed, "random_seed", minimum=0)
    keys = _row_keys(row_keys, values.shape[0])
    corrected = values.copy()
    rows, columns = np.nonzero(values)
    in_soup = weights[columns] > 0
    rows = rows[in_soup]
    columns = columns[in_soup]
    if rows.size == 0:
        return corrected
    n_rows = values.shape[0]
    held = values[rows, columns].astype(np.int64)
    totals = values.sum(axis=1, dtype=np.float64)
    removable = np.bincount(rows, weights=held, minlength=n_rows)
    targets = np.minimum(rho * totals, removable)
    removal = _continuous_removal(
        rows, columns, held, _soup_units(weights)[columns], targets, removable, n_rows
    )
    whole = np.floor(removal)
    # A row's fractional parts sum to its target less its whole parts.
    rest = targets - np.bincount(rows, weights=whole, minlength=n_rows)
    whole += _round_up(rows, removal - whole, rest, _row_uniforms(seed, keys))
    corrected[rows, columns] -= whole.astype(values.dtype)
    return corrected


def _row_keys(row_keys: Any | None, n_rows: int) -> np.ndarray:
    """Return the key of each row's rounding draw as unsigned integers."""
    if row_keys is None:
        return np.arange(n_rows, dtype=np.uint64)
    keys = np.asarray(row_keys)
    if keys.dtype.kind not in "iu":
        raise TypeError("row_keys must hold integers")
    if keys.shape != (n_rows,):
        raise ValueError("row_keys must hold one key per row")
    if keys.dtype.kind == "i" and keys.size and int(keys.min()) < 0:
        raise ValueError("row_keys must not be negative")
    return keys.astype(np.uint64)


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Return the SplitMix64 finaliser of each unsigned 64-bit integer.

    Cell calling compiles the same finaliser for one integer at a time; this
    form applies it to whole arrays, whose products wrap around as intended.
    """
    mixed = (values ^ (values >> _SHIFT_30)) * _MIX_1
    mixed = (mixed ^ (mixed >> _SHIFT_27)) * _MIX_2
    finalised: np.ndarray = mixed ^ (mixed >> _SHIFT_31)
    return finalised


def _row_uniforms(random_seed: int, keys: np.ndarray) -> np.ndarray:
    """Return each row's rounding draw, uniform on [0, 1).

    The draw is the first draw of a stream whose key mixes the seed, this
    module's salt, and the row key, as cell calling's ``_iteration_key`` mixes
    a seed and an iteration and its ``_uniform`` draws, for every row at once.
    """
    seed = _splitmix64(
        np.array([random_seed % 2**64], dtype=np.uint64) ^ _ROUNDING_SALT
    )
    streams = _splitmix64(seed ^ _splitmix64(keys + _GOLDEN))
    uniforms: np.ndarray = (_splitmix64(streams) >> _SHIFT_11) * _UNIT_INTERVAL
    return uniforms


def _soup_units(weights: np.ndarray) -> np.ndarray:
    """Return each gene's soup weight in whole units of 2**-62 of the soup.

    Integer weights make the running sums over a cell's genes exact. Each
    gene in the soup has at least one unit, so it keeps a positive weight.
    """
    share = weights / weights.max()
    share /= share.sum()
    units: np.ndarray = np.rint(share * _SOUP_UNITS).astype(np.int64)
    units[(units == 0) & (weights > 0)] = 1
    return units


def _segment_starts(rows: np.ndarray, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the first position and the length of each row's run of entries."""
    lengths = np.bincount(rows, minlength=n_rows)
    starts = np.zeros(n_rows, dtype=np.int64)
    np.cumsum(lengths[:-1], out=starts[1:])
    return starts, lengths


def _running_sums(values: np.ndarray, first: np.ndarray) -> np.ndarray:
    """Return each entry's sum with the earlier entries of its row.

    ``values`` are non-negative integers, consecutive within each row, and
    ``first`` is the position of the first entry of each entry's row. The
    running sum over all entries wraps around in unsigned 64-bit arithmetic,
    which leaves its differences exact while a row's own sum stays below
    2**63, so a row's sums do not depend on the rows before it.
    """
    running = np.cumsum(values, dtype=np.uint64)
    before = running[first] - values[first].astype(np.uint64)
    within: np.ndarray = (running - before).astype(np.int64)
    return within


def _continuous_removal(
    rows: np.ndarray,
    columns: np.ndarray,
    held: np.ndarray,
    units: np.ndarray,
    targets: np.ndarray,
    removable: np.ndarray,
    n_rows: int,
) -> np.ndarray:
    """Return the removal from each nonzero entry before rounding.

    The entries are the nonzero counts of genes in the soup, given in row
    order, with their soup weights in integer units. A row's removal at level
    ``t`` is ``sum(min(n_g, t w_g))``; the level that removes the row's target
    is found from the entries sorted by ``n_g / w_g``.
    """
    weights = units.astype(np.float64)
    ratio = held / weights
    order = np.lexsort((columns, ratio, rows))
    rows = rows[order]
    held = held[order]
    units = units[order]
    weights = weights[order]
    ratio = ratio[order]
    starts, lengths = _segment_starts(rows, n_rows)
    first = starts[rows]
    # Counts of the earlier entries of each row, and the soup weight of each
    # entry and those after it in its row.
    held_before = (_running_sums(held, first) - held).astype(np.float64)
    weight_through = _running_sums(units, first)
    row_weight = np.zeros(n_rows, dtype=np.int64)
    filled = lengths > 0
    row_weight[filled] = weight_through[starts[filled] + lengths[filled] - 1]
    weight_from = (row_weight[rows] - weight_through + units).astype(np.float64)
    # At the level of an entry's own ratio, the earlier entries are empty and
    # it and the later ones lose that level times their soup weight.
    removal_at = held_before + ratio * weight_from
    short = removal_at < targets[rows]
    stop = starts + np.bincount(rows[short], minlength=n_rows)
    level = np.zeros(n_rows, dtype=np.float64)
    active = (targets > 0) & (targets < removable) & (stop < starts + lengths)
    at = stop[active]
    level[active] = np.maximum(
        (targets[active] - held_before[at]) / weight_from[at], 0.0
    )
    level[(targets > 0) & ~active] = np.inf
    removal = np.empty(len(order), dtype=np.float64)
    removal[order] = np.minimum(held, level[rows] * weights)
    return removal


def _round_up(
    rows: np.ndarray, fractions: np.ndarray, totals: np.ndarray, uniforms: np.ndarray
) -> np.ndarray:
    """Return 1 for each entry that systematic sampling rounds up, else 0.

    ``rows`` holds the row of each entry, in non-decreasing order with a row's
    entries in gene order; ``fractions`` each entry's fractional part, in
    [0, 1); ``totals`` the sum of each row's fractional parts, which may differ
    from the sum of ``fractions`` by rounding; and ``uniforms`` one draw ``u``
    on [0, 1) for each row. Entry ``j`` of a row is rounded up when
    ``floor(C_j + u) > floor(C_(j-1) + u)``, where ``C_j`` is the sum of the
    row's fractional parts up to and including ``j`` and ``C_0 = 0``: with a
    probability equal to its fractional part. A row rounds up
    ``floor(total + u)`` entries, its total rounded down or up.

    The parts are counted in units of 2**-40, which makes the sums exact. When
    a row's units fall short of its total, the shortfall goes to its earliest
    entries with a fractional part, short of making any of them whole; when
    they exceed it, the excess comes off its last entries.
    """
    largest = _FRACTION_UNITS - 1
    units = np.minimum(np.rint(fractions * _FRACTION_UNITS), largest).astype(np.int64)
    room = np.where(fractions > 0, largest - units, 0)
    starts, lengths = _segment_starts(rows, len(uniforms))
    first = starts[rows]
    through = _running_sums(units, first)
    capacity = _running_sums(room, first)
    goal = np.rint(np.maximum(totals, 0.0) * _FRACTION_UNITS).astype(np.int64)
    filled = lengths > 0
    shortfall = np.zeros(len(uniforms), dtype=np.int64)
    shortfall[filled] = goal[filled] - through[starts[filled] + lengths[filled] - 1]
    np.maximum(shortfall, 0, out=shortfall)
    through = np.minimum(through + np.minimum(capacity, shortfall[rows]), goal[rows])
    before = np.empty_like(through)
    before[1:] = through[:-1]
    before[first == np.arange(len(rows))] = 0
    shift = np.floor(uniforms * _FRACTION_UNITS).astype(np.int64)[rows]
    up: np.ndarray = ((through + shift) >> _FRACTION_BITS) - (
        (before + shift) >> _FRACTION_BITS
    )
    return up
