"""Pairs of junctions within a distance of each other.

Clonotype clusters link cells whose junctions are similar rather than
identical, as scirpy's ``define_clonotype_clusters`` does: T cells whose
receptors differ by a residue or two often recognize the same epitope, and B
cells of one clone differ by somatic mutations. The distances are scirpy's:

- ``hamming``: the substitutions between junctions of equal length;
- ``normalized_hamming``: those substitutions as a percentage of the length,
  rounded half up, between junctions of equal length;
- ``levenshtein``: the substitutions, insertions and deletions.

Every pair within the cutoff is found by comparing the junctions of each
length, and for ``levenshtein`` of each pair of lengths within the cutoff,
in compiled loops that stop at the first position where the cutoff is
exceeded. These functions work on text and never touch storage.
"""

from math import comb
from typing import Any

import numpy as np
from numba import njit

__all__ = ["DISTANCE_METRICS", "METRICS", "allowed_mismatches", "neighbor_pairs"]

#: The metrics that link cells by their junctions: ``identity`` links only
#: identical junctions.
METRICS = ("identity", "hamming", "normalized_hamming", "levenshtein")
#: The metrics with a cutoff.
DISTANCE_METRICS = ("hamming", "normalized_hamming", "levenshtein")


def allowed_mismatches(length: int, percent: int) -> int:
    """The most substitutions whose rounded percentage of ``length`` is within ``percent``.

    scirpy rounds ``100 * mismatches / length`` half up to whole percent.
    """
    mismatches = 0
    while mismatches < length and _rounded_percent(mismatches + 1, length) <= percent:
        mismatches += 1
    return mismatches


def _rounded_percent(mismatches: int, length: int) -> int:
    """``100 * mismatches / length`` rounded half up, in exact integer arithmetic."""
    return (200 * mismatches + length) // (2 * length)


@njit(cache=True)
def _hamming_counts(codes: np.ndarray, limit: int) -> np.ndarray:  # pragma: no cover
    """For each row, the later rows within ``limit`` substitutions."""
    rows, width = codes.shape
    counts = np.zeros(rows, dtype=np.int64)
    for first in range(rows):
        for second in range(first + 1, rows):
            mismatches = 0
            for position in range(width):
                if codes[first, position] != codes[second, position]:
                    mismatches += 1
                    if mismatches > limit:
                        break
            if mismatches <= limit:
                counts[first] += 1
    return counts


@njit(cache=True)
def _hamming_fill(
    codes: np.ndarray,
    limit: int,
    starts: np.ndarray,
    firsts: np.ndarray,
    seconds: np.ndarray,
) -> None:  # pragma: no cover
    """Write the pairs that :func:`_hamming_counts` counted."""
    rows, width = codes.shape
    for first in range(rows):
        at = starts[first]
        for second in range(first + 1, rows):
            mismatches = 0
            for position in range(width):
                if codes[first, position] != codes[second, position]:
                    mismatches += 1
                    if mismatches > limit:
                        break
            if mismatches <= limit:
                firsts[at] = first
                seconds[at] = second
                at += 1


@njit(cache=True)
def _within_edits(
    left: np.ndarray,
    left_length: int,
    right: np.ndarray,
    right_length: int,
    limit: int,
    rows: np.ndarray,
) -> bool:  # pragma: no cover
    """Whether two sequences are within ``limit`` edits.

    Dynamic programming over a band of ``limit`` around the diagonal: a cell
    further from it needs more than ``limit`` insertions or deletions, so it
    reads as ``limit + 1``. The loop stops once a whole row exceeds the
    limit. ``rows`` is scratch space of two rows of ``right_length + 1``.
    """
    if abs(left_length - right_length) > limit:
        return False
    big = limit + 1
    previous = 0
    for column in range(right_length + 1):
        rows[previous, column] = column if column <= limit else big
    for line in range(1, left_length + 1):
        current = 1 - previous
        low = max(0, line - limit)
        high = min(right_length, line + limit)
        if low == 0:
            rows[current, 0] = line
            best = line
            begin = 1
        else:
            rows[current, low - 1] = big
            best = big
            begin = low
        for column in range(begin, high + 1):
            value = rows[previous, column - 1]
            if left[line - 1] != right[column - 1]:
                value += 1
            above = rows[previous, column] + 1
            if above < value:
                value = above
            beside = rows[current, column - 1] + 1
            if beside < value:
                value = beside
            if value > big:
                value = big
            rows[current, column] = value
            if value < best:
                best = value
        if high < right_length:
            rows[current, high + 1] = big
        if best > limit:
            return False
        previous = current
    return bool(rows[previous, right_length] <= limit)


@njit(cache=True)
def _edit_pairs(
    left: np.ndarray,
    left_lengths: np.ndarray,
    right: np.ndarray,
    right_lengths: np.ndarray,
    limit: int,
    same: bool,
    fill: bool,
    starts: np.ndarray,
    firsts: np.ndarray,
    seconds: np.ndarray,
) -> np.ndarray:  # pragma: no cover
    """Count, or write, the pairs of a left and a right block within ``limit`` edits."""
    counts = np.zeros(left.shape[0], dtype=np.int64)
    rows = np.empty((2, right.shape[1] + 1), dtype=np.int64)
    for first in range(left.shape[0]):
        at = starts[first] if fill else 0
        begin = first + 1 if same else 0
        for second in range(begin, right.shape[0]):
            if _within_edits(
                left[first],
                left_lengths[first],
                right[second],
                right_lengths[second],
                limit,
                rows,
            ):
                if fill:
                    firsts[at] = first
                    seconds[at] = second
                    at += 1
                else:
                    counts[first] += 1
    return counts


# Use keys when a text yields at most this many of them; compare every pair
# of a length otherwise.
_MAX_KEYS = 1_200
_FNV_OFFSET = np.uint64(14695981039346656037)
_FNV_PRIME = np.uint64(1099511628211)
_WILDCARD = np.uint64(0x1FF)


@njit(cache=True)
def _variant_keys(
    codes: np.ndarray,
    lengths: np.ndarray,
    limit: int,
    deletions: bool,
    out_keys: np.ndarray,
    out_rows: np.ndarray,
) -> int:  # pragma: no cover
    """Hash each text's variants, writing keys and rows; return how many.

    With ``deletions``, the variants drop 0 to ``limit`` positions, so two
    texts within ``limit`` edits share one; otherwise they replace exactly
    ``limit`` positions by a wildcard, so two texts of one length within
    ``limit`` substitutions share one. Up to three positions take part.
    """
    at = 0
    for row in range(codes.shape[0]):
        length = lengths[row]
        low = 0 if deletions else limit
        for count in range(low, limit + 1):
            if count > length:
                break
            first_end = length if count >= 1 else 1
            for first in range(first_end):
                second_start = first + 1 if count >= 2 else 0
                second_end = length if count >= 2 else 1
                for second in range(second_start, second_end):
                    third_start = second + 1 if count >= 3 else 0
                    third_end = length if count >= 3 else 1
                    for third in range(third_start, third_end):
                        value = _FNV_OFFSET
                        for position in range(length):
                            dropped = (
                                (count >= 1 and position == first)
                                or (count >= 2 and position == second)
                                or (count >= 3 and position == third)
                            )
                            if dropped:
                                if deletions:
                                    continue
                                symbol = _WILDCARD
                            else:
                                symbol = np.uint64(codes[row, position])
                            value = (value ^ symbol) * _FNV_PRIME
                        value = (
                            value ^ np.uint64(length - (count if deletions else 0))
                        ) * _FNV_PRIME
                        out_keys[at] = value
                        out_rows[at] = row
                        at += 1
    return at


@njit(cache=True)
def _run_pairs(
    keys: np.ndarray,
    rows: np.ndarray,
    count_only: bool,
    firsts: np.ndarray,
    seconds: np.ndarray,
) -> int:  # pragma: no cover
    """Pair the rows of each run of equal keys, sorted by key."""
    at = 0
    start = 0
    n = len(keys)
    while start < n:
        end = start + 1
        while end < n and keys[end] == keys[start]:
            end += 1
        for left in range(start, end):
            for right in range(left + 1, end):
                if rows[left] != rows[right]:
                    if not count_only:
                        firsts[at] = rows[left]
                        seconds[at] = rows[right]
                    at += 1
        start = end
    return at


@njit(cache=True)
def _check_pairs(
    codes: np.ndarray,
    lengths: np.ndarray,
    firsts: np.ndarray,
    seconds: np.ndarray,
    limit: int,
    edits: bool,
) -> np.ndarray:  # pragma: no cover
    """Whether each candidate pair is within ``limit`` substitutions or edits."""
    keep = np.zeros(len(firsts), dtype=np.bool_)
    rows = np.empty((2, codes.shape[1] + 1), dtype=np.int64)
    for at in range(len(firsts)):
        first, second = firsts[at], seconds[at]
        if edits:
            keep[at] = _within_edits(
                codes[first],
                lengths[first],
                codes[second],
                lengths[second],
                limit,
                rows,
            )
        elif lengths[first] == lengths[second]:
            mismatches = 0
            for position in range(lengths[first]):
                if codes[first, position] != codes[second, position]:
                    mismatches += 1
            keep[at] = mismatches <= limit
    return keep


def _key_count(length: int, limit: int, deletions: bool) -> int:
    """The keys that one text of ``length`` yields."""
    if deletions:
        return sum(comb(length, count) for count in range(min(limit, length) + 1))
    return comb(length, limit) if limit <= length else 0


def _keyed_pairs(
    codes: np.ndarray, lengths: np.ndarray, limit: int, deletions: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Every pair within ``limit`` through shared variant keys, verified."""
    total = int(sum(_key_count(int(length), limit, deletions) for length in lengths))
    keys = np.empty(total, dtype=np.uint64)
    rows = np.empty(total, dtype=np.int64)
    written = _variant_keys(codes, lengths, limit, deletions, keys, rows)
    order = np.argsort(keys[:written], kind="stable")
    keys, rows = keys[order], rows[order]
    empty = np.empty(0, dtype=np.int64)
    n_pairs = _run_pairs(keys, rows, True, empty, empty)
    firsts = np.empty(n_pairs, dtype=np.int64)
    seconds = np.empty(n_pairs, dtype=np.int64)
    _run_pairs(keys, rows, False, firsts, seconds)
    low = np.minimum(firsts, seconds)
    high = np.maximum(firsts, seconds)
    width = max(int(len(lengths)), 1)
    unique = np.unique(low * width + high)
    low, high = unique // width, unique % width
    keep = _check_pairs(codes, lengths, low, high, limit, deletions)
    return low[keep], high[keep]


def _encoded(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Encode texts as a padded matrix of bytes and their lengths."""
    lengths = np.fromiter(
        (len(text) for text in texts), dtype=np.int64, count=len(texts)
    )
    width = int(lengths.max()) if len(texts) else 0
    matrix = np.zeros((len(texts), max(width, 1)), dtype=np.uint8)
    for row, text in enumerate(texts):
        encoded = text.encode("utf-8")
        matrix[row, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return matrix, lengths


def _hamming_block(codes: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    counts = _hamming_counts(codes, limit)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    total = int(counts.sum())
    firsts = np.empty(total, dtype=np.int64)
    seconds = np.empty(total, dtype=np.int64)
    if total:
        _hamming_fill(codes, limit, starts, firsts, seconds)
    return firsts, seconds


def _edit_block(
    left: np.ndarray,
    left_lengths: np.ndarray,
    right: np.ndarray,
    right_lengths: np.ndarray,
    limit: int,
    same: bool,
) -> tuple[np.ndarray, np.ndarray]:
    empty = np.empty(0, dtype=np.int64)
    counts = _edit_pairs(
        left,
        left_lengths,
        right,
        right_lengths,
        limit,
        same,
        False,
        empty,
        empty,
        empty,
    )
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    total = int(counts.sum())
    firsts = np.empty(total, dtype=np.int64)
    seconds = np.empty(total, dtype=np.int64)
    if total:
        _edit_pairs(
            left,
            left_lengths,
            right,
            right_lengths,
            limit,
            same,
            True,
            starts,
            firsts,
            seconds,
        )
    return firsts, seconds


def neighbor_pairs(
    texts: Any, metric: str, cutoff: int, blocks: Any = None
) -> tuple[np.ndarray, np.ndarray]:
    """Find every pair of distinct texts within ``cutoff`` of each other.

    Args:
        texts: The texts, such as distinct junctions.
        metric: ``"hamming"``, ``"normalized_hamming"`` or ``"levenshtein"``.
        cutoff: The largest distance that links two texts: substitutions,
            percent, or edits.
        blocks: Each text's block, such as its gene calls; only texts of one
            block are compared. None compares all.

    Returns:
        The positions of the pairs, the first below the second.

    Raises:
        ValueError: If the metric is unknown or the cutoff is negative.
    """
    if metric not in DISTANCE_METRICS:
        raise ValueError(f"metric must be one of {', '.join(DISTANCE_METRICS)}")
    if (
        isinstance(cutoff, bool)
        or not isinstance(cutoff, int | np.integer)
        or cutoff < 0
    ):
        raise ValueError("cutoff must be a non-negative integer")
    values = [str(text) for text in np.asarray(texts, dtype=object).tolist()]
    n_texts = len(values)
    block_codes = (
        np.zeros(n_texts, dtype=np.int64)
        if blocks is None
        else np.asarray(blocks, dtype=np.int64).reshape(-1)
    )
    if len(block_codes) != n_texts:
        raise ValueError("blocks must hold one block per text")
    matrix, lengths = _encoded(values)
    firsts: list[np.ndarray] = []
    seconds: list[np.ndarray] = []
    for block in np.unique(block_codes):
        in_block = np.flatnonzero(block_codes == block)
        block_lengths = lengths[in_block]
        if metric == "levenshtein":
            limit = int(cutoff)
            if (
                limit <= 3
                and _key_count(int(block_lengths.max()), limit, True) <= _MAX_KEYS
            ):
                first, second = _keyed_pairs(
                    matrix[in_block], block_lengths, limit, True
                )
                firsts.append(in_block[first])
                seconds.append(in_block[second])
                continue
            for length in np.unique(block_lengths):
                members = in_block[block_lengths == length]
                # Each pair of lengths within the cutoff, the shorter first.
                for other in np.unique(block_lengths):
                    if other < length or other - length > limit:
                        continue
                    partners = in_block[block_lengths == other]
                    first, second = _edit_block(
                        matrix[members],
                        lengths[members],
                        matrix[partners],
                        lengths[partners],
                        limit,
                        bool(other == length),
                    )
                    firsts.append(members[first])
                    seconds.append(partners[second])
            continue
        for length in np.unique(block_lengths):
            members = in_block[block_lengths == length]
            if len(members) < 2:
                continue
            limit = (
                int(cutoff)
                if metric == "hamming"
                else allowed_mismatches(int(length), int(cutoff))
            )
            limit = min(limit, int(length))
            if limit <= 3 and _key_count(int(length), limit, False) <= _MAX_KEYS:
                first, second = _keyed_pairs(
                    matrix[members, : int(length)], lengths[members], limit, False
                )
            else:
                first, second = _hamming_block(matrix[members, : int(length)], limit)
            firsts.append(members[first])
            seconds.append(members[second])
    if not firsts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    first_all = np.concatenate(firsts)
    second_all = np.concatenate(seconds)
    low = np.minimum(first_all, second_all)
    high = np.maximum(first_all, second_all)
    return low, high
