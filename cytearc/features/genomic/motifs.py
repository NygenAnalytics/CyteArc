"""Position frequency matrices, log-odds scoring, and motif scanning."""

import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numba import njit, prange
from scipy.sparse import csr_matrix

from ...utils.arguments import integer_argument
from ...utils.arrays import read_only_copy

if TYPE_CHECKING:
    import pandas as pd

    from .sequence import IndexedFasta

__all__ = [
    "Motif",
    "log_odds_matrix",
    "match_motifs",
    "nucleotide_frequencies",
    "read_jaspar_motifs",
    "read_meme_motifs",
    "reverse_complement_matrix",
    "scan_motif_matches",
    "score_threshold",
]


@dataclass(frozen=True, slots=True, eq=False)
class Motif:
    """A DNA motif as a position frequency matrix.

    Attributes:
        motifId: Identifier, such as a JASPAR matrix ID.
        name: Display name, usually the transcription factor.
        counts: ``(width, 4)`` non-negative counts or probabilities in A, C,
            G, T column order.
    """

    motifId: str
    name: str
    counts: np.ndarray

    def __post_init__(self) -> None:
        counts = np.asarray(self.counts, dtype=np.float64)
        if counts.ndim != 2 or counts.shape[1] != 4 or counts.shape[0] == 0:
            raise ValueError(
                f"Motif {self.motifId!r} must be a non-empty (width, 4) matrix"
            )
        if not np.isfinite(counts).all() or np.any(counts < 0):
            raise ValueError(
                f"Motif {self.motifId!r} has negative or non-finite values"
            )
        if np.any(counts.sum(axis=1) <= 0):
            raise ValueError(f"Motif {self.motifId!r} has an empty position")
        if not self.motifId:
            raise ValueError("Motif identifiers must be non-empty")
        object.__setattr__(self, "counts", read_only_copy(counts, np.float64))

    @property
    def width(self) -> int:
        """Number of positions."""
        return int(self.counts.shape[0])


_JASPAR_ROW = re.compile(r"^\s*([ACGT])\s*\[?\s*([^\]]*)\]?\s*$")


def read_jaspar_motifs(path: str | os.PathLike[str]) -> list[Motif]:
    """Read count matrices from a JASPAR-format file.

    Each motif is a ``>ID NAME`` header followed by A, C, G, and T rows, with
    or without brackets, as in the JASPAR collection downloads.
    """
    motifs: list[Motif] = []
    header: tuple[str, str] | None = None
    rows: dict[str, list[float]] = {}

    def finish() -> None:
        if header is None:
            return
        if set(rows) != set("ACGT"):
            raise ValueError(f"JASPAR motif {header[0]!r} needs A, C, G, and T rows")
        widths = {len(values) for values in rows.values()}
        if len(widths) != 1:
            raise ValueError(
                f"JASPAR motif {header[0]!r} has rows of different lengths"
            )
        counts = np.column_stack([rows[base] for base in "ACGT"])
        motifs.append(Motif(header[0], header[1], counts))

    with open(path, encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish()
                fields = line[1:].split(None, 1)
                if not fields:
                    raise ValueError(f"JASPAR header on line {line_number} has no ID")
                motif_id = fields[0]
                name = fields[1].strip() if len(fields) > 1 else motif_id
                header = (motif_id, name)
                rows = {}
                continue
            match = _JASPAR_ROW.match(line)
            if header is None or match is None:
                raise ValueError(f"Malformed JASPAR line {line_number}: {line!r}")
            base = match.group(1)
            if base in rows:
                raise ValueError(
                    f"JASPAR motif {header[0]!r} repeats row {base} (line {line_number})"
                )
            try:
                rows[base] = [float(value) for value in match.group(2).split()]
            except ValueError as exc:
                raise ValueError(f"Malformed JASPAR line {line_number}") from exc
    finish()
    if not motifs:
        raise ValueError("The JASPAR file contains no motifs")
    return motifs


def read_meme_motifs(path: str | os.PathLike[str]) -> list[Motif]:
    """Read motifs from a MEME-format file as counts.

    Each letter-probability row is multiplied by its matrix's ``nsites`` (20
    when absent, the MEME format's default), so a pseudocount weighs as much
    as it does for count matrices such as JASPAR's.
    """
    motifs: list[Motif] = []
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    alphabet_line = next(
        (line for line in lines if line.strip().upper().startswith("ALPHABET")),
        None,
    )
    if alphabet_line is not None and "ACGT" not in alphabet_line.upper().replace(
        " ", ""
    ):
        raise ValueError("Only DNA (ACGT) MEME motif files are supported")
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line.startswith("MOTIF"):
            index += 1
            continue
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"MEME MOTIF line {index + 1} has no identifier")
        motif_id = fields[1]
        name = fields[2] if len(fields) > 2 else motif_id
        index += 1
        while index < len(lines) and not lines[index].strip().startswith(
            "letter-probability matrix"
        ):
            if lines[index].strip().startswith("MOTIF"):
                raise ValueError(f"MEME motif {motif_id!r} has no probability matrix")
            index += 1
        if index >= len(lines):
            raise ValueError(f"MEME motif {motif_id!r} has no probability matrix")
        width_match = re.search(r"\bw=\s*(\d+)", lines[index])
        sites_match = re.search(r"\bnsites=\s*([0-9.eE+-]+)", lines[index])
        sites = float(sites_match.group(1)) if sites_match else 20.0
        index += 1
        values: list[list[float]] = []
        while index < len(lines):
            row = lines[index].split()
            if not row:
                if values:
                    break
                index += 1
                continue
            try:
                numbers = [float(value) for value in row]
            except ValueError:
                break
            if len(numbers) != 4:
                raise ValueError(f"MEME motif {motif_id!r} has a row without 4 values")
            values.append(numbers)
            index += 1
        if width_match is not None and int(width_match.group(1)) != len(values):
            raise ValueError(
                f"MEME motif {motif_id!r} declares w={width_match.group(1)} but "
                f"has {len(values)} rows"
            )
        motifs.append(
            Motif(motif_id, name, np.asarray(values, dtype=np.float64) * sites)
        )
    if not motifs:
        raise ValueError("The MEME file contains no motifs")
    return motifs


def _background(background: Sequence[float] | np.ndarray) -> np.ndarray:
    """Validate A, C, G, T background frequencies; they are used as given."""
    values = np.asarray(background, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("background must hold four positive A, C, G, T frequencies")
    if abs(values.sum() - 1.0) > 1e-6:
        raise ValueError("background frequencies must sum to 1")
    return values


def log_odds_matrix(
    counts: np.ndarray,
    background: Sequence[float] | np.ndarray,
    *,
    pseudocount: float,
    log_base: float = 2.0,
) -> np.ndarray:
    """Convert counts to a log-odds matrix, as MOODS ``tools.log_odds`` does.

    Each position becomes
    ``(ln((n_b + pseudocount * bg_b) / column_sum) - ln(bg_b)) / ln(log_base)``,
    where ``column_sum`` adds ``n_b + pseudocount * bg_b`` over the four bases.
    With an even background and ``pseudocount=0.8`` in base 2 this is
    TFBSTools' ``toPWM``, which motifmatchr and chromVAR use; pychromvar uses
    the natural logarithm and ``pseudocount=1e-4``.
    """
    values = np.asarray(counts, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("counts must be a (width, 4) matrix")
    if not np.isfinite(pseudocount) or pseudocount < 0:
        raise ValueError("pseudocount must be finite and non-negative")
    if not log_base > 1:
        raise ValueError("log_base must be greater than 1")
    bg = _background(background)
    smoothed = values + pseudocount * bg
    # MOODS adds the four smoothed entries of a column in base order.
    column_sum = ((smoothed[:, 0] + smoothed[:, 1]) + smoothed[:, 2]) + smoothed[:, 3]
    with np.errstate(divide="ignore"):
        scores = np.log(smoothed / column_sum[:, None]) - np.log(bg)
    if log_base != np.e:
        scores = scores / np.log(log_base)
    if not np.isfinite(scores).all():
        raise ValueError(
            "A zero count with no pseudocount gives an infinite log-odds score"
        )
    return scores


def reverse_complement_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return the matrix scoring the reverse-complement strand."""
    values = np.asarray(matrix)
    return np.ascontiguousarray(values[::-1, ::-1])


def _min_delta(matrix: np.ndarray) -> float:
    """MOODS ``min_delta``: the smallest gap between a column's top two scores."""
    smallest = np.inf
    for column in matrix:
        best = -np.inf
        second = -np.inf
        for value in column:
            if value > best:
                second = best
                best = value
            elif best > value > second:
                second = value
        smallest = min(smallest, best - second)
    return float(smallest)


def score_threshold(
    matrix: np.ndarray,
    background: Sequence[float] | np.ndarray,
    p_value: float,
    *,
    precision: float = 2000.0,
) -> float:
    """Return the match threshold of a log-odds matrix at a p-value.

    This is MOODS ``tools.threshold_from_p``, which motifmatchr and pychromvar
    use. Scores are discretized to ``1 / precision`` units, rounding halves
    away from zero, the exact distribution of the discretized total under an
    independent background is computed by dynamic programming, and the
    threshold is the smallest discretized level whose upper-tail probability
    is at most ``p_value``. A site matches when its exact score is at least
    the threshold. When even the best score is more probable than
    ``p_value``, the threshold sits half the smallest column gap below it.
    """
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] == 0:
        raise ValueError("matrix must be a non-empty (width, 4) matrix")
    if not 0 < p_value < 1:
        raise ValueError("p_value must lie strictly between 0 and 1")
    if not precision > 0:
        raise ValueError("precision must be positive")
    bg = _background(background)
    scaled = precision * values
    grid = np.where(
        scaled > 0.0, np.trunc(scaled + 0.5), np.trunc(scaled - 0.5)
    ).astype(np.int64)
    width = grid.shape[0]
    max_total = int(grid.max(axis=1).sum())
    min_value = int(grid.min())
    span = max_total - width * min_value
    table = np.zeros(span + 1, dtype=np.float64)
    for base in range(4):
        table[grid[0, base] - min_value] += bg[base]
    for row in grid[1:]:
        updated = np.zeros_like(table)
        for base in range(4):
            shift = int(row[base]) - min_value
            updated[shift:] += bg[base] * table[: span + 1 - shift]
        table = updated
    tail = table[span]
    if tail > p_value:
        return float(values.max(axis=1).sum() - _min_delta(values) / 2)
    for level in range(span - 1, -1, -1):
        tail += table[level]
        if tail > p_value:
            return float((level + width * min_value + 1) / precision)
    return float(values.min(axis=1).sum() - 1.0)


def _visit_order(matrix: np.ndarray) -> np.ndarray:
    """Order positions so the most selective are scored first.

    A position whose best base scores far above its typical base rejects most
    windows early, so positions are visited by decreasing gap between their
    maximum and mean scores, as in MOODS lookahead filtration.
    """
    gap = matrix.max(axis=1) - matrix.mean(axis=1)
    return np.argsort(-gap, kind="stable").astype(np.int64)


@njit(cache=True)
def _suffix_maxima(
    matrix: np.ndarray, order: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    """Best achievable score of each motif after each visited position.

    ``order[offset + step]`` is the position visited at ``step``; the result
    at ``offset + step`` is the best total of the positions visited after it.
    """
    out = np.zeros(matrix.shape[0], dtype=np.float64)
    for motif in range(offsets.size - 1):
        start = offsets[motif]
        stop = offsets[motif + 1]
        remaining = 0.0
        for step in range(stop - 1, start - 1, -1):
            out[step] = remaining
            row = start + order[step]
            best = matrix[row, 0]
            for base in range(1, 4):
                if matrix[row, base] > best:
                    best = matrix[row, base]
            remaining += best
    return out


@njit(cache=True)
def _strand_hit(
    codes: np.ndarray,
    start: int,
    matrix: np.ndarray,
    order: np.ndarray,
    suffix: np.ndarray,
    offset: int,
    width: int,
    threshold: float,
) -> bool:
    score = 0.0
    for step in range(width):
        position = order[offset + step]
        code = codes[start + position]
        if code > 3:
            return False
        score += matrix[offset + position, code]
        if score + suffix[offset + step] < threshold:
            return False
    return score >= threshold


@njit(cache=True, parallel=True)
def _scan_block(
    codes: np.ndarray,
    sequence_offsets: np.ndarray,
    forward: np.ndarray,
    reverse: np.ndarray,
    forward_order: np.ndarray,
    reverse_order: np.ndarray,
    forward_suffix: np.ndarray,
    reverse_suffix: np.ndarray,
    motif_offsets: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    n_sequences = sequence_offsets.size - 1
    n_motifs = motif_offsets.size - 1
    hits = np.zeros((n_sequences, n_motifs), dtype=np.bool_)
    for sequence in prange(n_sequences):
        begin = sequence_offsets[sequence]
        length = sequence_offsets[sequence + 1] - begin
        for motif in range(n_motifs):
            offset = motif_offsets[motif]
            width = motif_offsets[motif + 1] - offset
            for position in range(length - width + 1):
                window = begin + position
                if _strand_hit(
                    codes,
                    window,
                    forward,
                    forward_order,
                    forward_suffix,
                    offset,
                    width,
                    thresholds[motif, 0],
                ) or _strand_hit(
                    codes,
                    window,
                    reverse,
                    reverse_order,
                    reverse_suffix,
                    offset,
                    width,
                    thresholds[motif, 1],
                ):
                    hits[sequence, motif] = True
                    break
    return hits


def scan_motif_matches(
    sequences: Iterable[np.ndarray],
    matrices: Sequence[np.ndarray],
    thresholds: Sequence[float] | np.ndarray,
    *,
    batch_size: int = 4096,
) -> csr_matrix:
    """Mark which sequences hold at least one match of each motif on either strand.

    Args:
        sequences: Encoded sequences (see ``encode_sequence``). A window that
            holds an unknown base (code 4) never matches.
        matrices: Forward-strand ``(width, 4)`` log-odds matrices.
        thresholds: Minimum score of a match for each matrix, as a
            ``(len(matrices), 2)`` array of forward-strand and reverse-strand
            thresholds: under a background whose A and T, or C and G,
            frequencies differ, the strands score differently.
        batch_size: Sequences scanned per parallel batch.

    Returns:
        A boolean sequences-by-motifs CSR matrix.
    """
    if not matrices:
        raise ValueError("At least one motif matrix is required")
    threshold_values = np.ascontiguousarray(thresholds, dtype=np.float64)
    if threshold_values.shape != (len(matrices), 2):
        raise ValueError("thresholds must hold two strand thresholds per matrix")
    if np.isnan(threshold_values).any():
        raise ValueError("thresholds must not be NaN")
    batch_size = integer_argument(batch_size, "batch_size", minimum=1)
    widths = []
    for matrix in matrices:
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] == 0:
            raise ValueError("Every motif matrix must be a non-empty (width, 4) matrix")
        if not np.isfinite(values).all():
            raise ValueError("Motif matrices must be finite")
        widths.append(values.shape[0])
    motif_offsets = np.zeros(len(matrices) + 1, dtype=np.int64)
    motif_offsets[1:] = np.cumsum(widths)
    forward_matrices = [np.asarray(matrix, dtype=np.float64) for matrix in matrices]
    reverse_matrices = [
        reverse_complement_matrix(matrix) for matrix in forward_matrices
    ]
    forward = np.ascontiguousarray(np.vstack(forward_matrices))
    reverse = np.ascontiguousarray(np.vstack(reverse_matrices))
    forward_order = np.concatenate(
        [_visit_order(matrix) for matrix in forward_matrices]
    )
    reverse_order = np.concatenate(
        [_visit_order(matrix) for matrix in reverse_matrices]
    )
    forward_suffix = _suffix_maxima(forward, forward_order, motif_offsets)
    reverse_suffix = _suffix_maxima(reverse, reverse_order, motif_offsets)

    blocks: list[csr_matrix] = []
    batch: list[np.ndarray] = []

    def flush() -> None:
        lengths = np.fromiter((len(values) for values in batch), dtype=np.int64)
        offsets = np.zeros(len(batch) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)
        codes = (
            np.concatenate(batch).astype(np.uint8, copy=False)
            if offsets[-1]
            else np.zeros(0, dtype=np.uint8)
        )
        hits = _scan_block(
            codes,
            offsets,
            forward,
            reverse,
            forward_order,
            reverse_order,
            forward_suffix,
            reverse_suffix,
            motif_offsets,
            threshold_values,
        )
        blocks.append(csr_matrix(hits))
        batch.clear()

    for values in sequences:
        encoded = np.asarray(values)
        if encoded.ndim != 1 or encoded.dtype != np.uint8:
            raise ValueError("Encoded sequences must be one-dimensional uint8 arrays")
        batch.append(encoded)
        if len(batch) == batch_size:
            flush()
    if batch:
        flush()
    if not blocks:
        return csr_matrix((0, len(matrices)), dtype=bool)
    from scipy.sparse import vstack

    return csr_matrix(vstack(blocks, format="csr"), dtype=bool)


def nucleotide_frequencies(sequences: Iterable[np.ndarray]) -> np.ndarray:
    """Pooled A, C, G, T frequencies of encoded sequences, ignoring other bases.

    This is motifmatchr's ``"subject"`` background, the frequencies of the
    scanned sequences themselves.
    """
    counts = np.zeros(5, dtype=np.int64)
    for values in sequences:
        counts += np.bincount(np.asarray(values, dtype=np.uint8), minlength=5)[:5]
    known = counts[:4].sum()
    if known == 0:
        raise ValueError("The sequences hold no A, C, G, or T base")
    return counts[:4] / known


def _strand_thresholds(
    matrices: Sequence[np.ndarray], background: np.ndarray, p_value: float
) -> np.ndarray:
    """Forward and reverse-strand thresholds at ``p_value`` of each matrix."""
    return np.asarray(
        [
            [
                score_threshold(matrix, background, p_value),
                score_threshold(reverse_complement_matrix(matrix), background, p_value),
            ]
            for matrix in matrices
        ],
        dtype=np.float64,
    )


def match_motifs(
    regions: Sequence[str] | np.ndarray,
    genome: "IndexedFasta",
    motifs: Sequence[Motif],
    *,
    p_value: float = 5e-5,
    background: Literal["subject", "even"] | Sequence[float] | np.ndarray = "subject",
    pseudocount: float = 0.8,
    log_base: float = 2.0,
    batch_size: int = 4096,
) -> "pd.DataFrame":
    """Find which regions contain a match of each motif on either strand.

    Counts become log-odds scores against the background (``log_odds_matrix``),
    each strand's threshold is the MOODS score threshold at ``p_value`` of that
    strand's matrix, and a region matches when any window on either strand
    scores at least its strand's threshold. Windows never span an N or other
    ambiguity code. These are the motifmatchr defaults that chromVAR uses,
    except that scores are proper log-odds against the ``"subject"``
    background, and that the reverse strand has its own threshold, as in
    MOODS: motifmatchr reuses the forward one, which misses ``p_value`` for a
    background whose A and T, or C and G, frequencies differ.

    Args:
        regions: Region ids ``chrom:start-end`` (zero-based, half-open), such
            as ATAC peak ids.
        genome: Genome the regions refer to.
        motifs: Motifs to scan, for example from ``read_jaspar_motifs``.
        p_value: Probability under the background of a window scoring at
            least the threshold.
        background: ``"subject"`` uses the pooled base frequencies of the
            regions, ``"even"`` uses 0.25 each, or pass four frequencies.
        pseudocount: Total pseudocount per motif position, split by the
            background.
        log_base: Base of the log-odds scores.
        batch_size: Regions scanned per parallel batch.

    Returns:
        A network with ``source`` (motif ID), ``target`` (region id), and
        ``motif_name`` columns, one row per matching motif and region, sorted
        by motif then region order.
    """
    import pandas as pd

    from .sequence import parse_coordinate_ids

    region_names = np.asarray(regions, dtype=object).astype(str)
    chrom_values, start_values, end_values = parse_coordinate_ids(region_names)
    if not motifs:
        raise ValueError("At least one motif is required")
    motif_ids = [motif.motifId for motif in motifs]
    if len(set(motif_ids)) != len(motif_ids):
        raise ValueError("Motif identifiers must be unique")
    sequences = [
        genome.encode(str(chrom), int(start), int(end))
        for chrom, start, end in zip(
            chrom_values, start_values, end_values, strict=True
        )
    ]
    if isinstance(background, str):
        if background == "subject":
            frequencies = nucleotide_frequencies(sequences)
        elif background == "even":
            frequencies = np.full(4, 0.25)
        else:
            raise ValueError("background must be 'subject', 'even', or frequencies")
    else:
        frequencies = _background(background)
    matrices = [
        log_odds_matrix(
            motif.counts, frequencies, pseudocount=pseudocount, log_base=log_base
        )
        for motif in motifs
    ]
    thresholds = _strand_thresholds(matrices, frequencies, p_value)
    hits = scan_motif_matches(sequences, matrices, thresholds, batch_size=batch_size)
    by_motif = hits.tocsc()
    rows = []
    for column, motif in enumerate(motifs):
        regions = by_motif.indices[
            by_motif.indptr[column] : by_motif.indptr[column + 1]
        ]
        for region in np.sort(regions):
            rows.append((motif.motifId, region_names[region], motif.name))
    return pd.DataFrame.from_records(rows, columns=["source", "target", "motif_name"])
