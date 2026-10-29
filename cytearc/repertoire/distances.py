"""Pairs of junctions within a distance of each other.

Clonotype clusters link cells whose junctions are similar rather than
identical, as scirpy's ``define_clonotype_clusters`` does: T cells whose
receptors differ by a residue or two often recognize the same epitope, and B
cells of one clone differ by somatic mutations. The distances are scirpy's:

- ``hamming``: the substitutions between junctions of equal length;
- ``normalized_hamming``: those substitutions as a percentage of the length,
  rounded half up, between junctions of equal length;
- ``levenshtein``: the substitutions, insertions and deletions.

Every pair within the cutoff is found exactly, in one of two ways, whichever
is estimated to take less time for the junctions of each length, or for
``levenshtein`` of each block. Compiled loops compare every pair of one
length, or of lengths within the cutoff, and stop at the first position where
the cutoff is exceeded. Or, for cutoffs up to 3, each junction yields variant
keys that any two junctions within the cutoff share, and only the junctions
that share a key are compared. These functions work on text and never touch
storage.
"""

from math import comb
from typing import Any

import numpy as np
from numba import njit

from ..utils.arrays import sorted_unique, unique_inverse

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
def _grown(values: np.ndarray) -> np.ndarray:  # pragma: no cover
    """A copy of ``values`` with room for twice as many."""
    grown = np.empty(2 * len(values), dtype=values.dtype)
    grown[: len(values)] = values
    return grown


def _words(codes: np.ndarray) -> np.ndarray:
    """The rows of a matrix of byte codes as 64-bit words, padded with zeros."""
    rows, width = codes.shape
    padded = np.zeros((rows, max(-(-width // 8), 1) * 8), dtype=np.uint8)
    padded[:, :width] = codes
    return padded.view(np.uint64)


@njit(cache=True)
def _hamming_pairs(
    words: np.ndarray, limit: int
) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
    """Every pair of rows within ``limit`` substitutions, the first below the second.

    The rows are texts of one length, eight codes to a word. The bytes that
    differ in two words are the nonzero bytes of their exclusive or, which
    folding each byte onto its lowest bit and summing those bits counts.
    """
    rows, n_words = words.shape
    ones = np.uint64(0x0101010101010101)
    firsts = np.empty(max(rows, 16), dtype=np.int64)
    seconds = np.empty(max(rows, 16), dtype=np.int64)
    at = 0
    for first in range(rows):
        for second in range(first + 1, rows):
            mismatches = 0
            for word in range(n_words):
                differ = words[first, word] ^ words[second, word]
                if differ:
                    differ |= differ >> np.uint64(4)
                    differ |= differ >> np.uint64(2)
                    differ |= differ >> np.uint64(1)
                    mismatches += int(((differ & ones) * ones) >> np.uint64(56))
                    if mismatches > limit:
                        break
            if mismatches <= limit:
                if at == len(firsts):
                    firsts = _grown(firsts)
                    seconds = _grown(seconds)
                firsts[at] = first
                seconds[at] = second
                at += 1
    return firsts[:at].copy(), seconds[:at].copy()


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
) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
    """Every pair of a left and a right row within ``limit`` edits.

    With ``same``, the blocks are one and the first row is below the second.
    """
    rows = np.empty((2, right.shape[1] + 1), dtype=np.int64)
    firsts = np.empty(max(left.shape[0], 16), dtype=np.int64)
    seconds = np.empty(max(left.shape[0], 16), dtype=np.int64)
    at = 0
    for first in range(left.shape[0]):
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
                if at == len(firsts):
                    firsts = _grown(firsts)
                    seconds = _grown(seconds)
                firsts[at] = first
                seconds[at] = second
                at += 1
    return firsts[:at].copy(), seconds[:at].copy()


# The search compares every pair or uses keys, whichever is estimated to take
# less time: m * (m - 1) / 2 comparisons of m texts of one length (for edits,
# of the texts of lengths within the cutoff), or the keys that the texts
# yield, each at the cost below in nanoseconds. A key costs its hash and its
# share of the sort, of pairing the texts that share it and of checking them:
# 25 to 50 ns, more as the keys grow from 100,000 to 50 million. A comparison
# of substitutions costs 2 to 5 ns; one of edits about 90 ns plus 70 ns for
# each edit of the cutoff, and several times more between similar texts, such
# as those of one B cell clone, which fill the whole band. Measured on one core
# on simulated junctions of T cells (amino acids of 14 and 18 residues) and B
# cells (nucleotides of 36 to 66 bases), 300 to 30,000 of a length, with
# cutoffs of 1 to 3. On 200,000 simulated cells, the choices took at most 5%
# longer than the faster way for each length within 1 to 3 substitutions.
_KEY_NS = 40.0
_PAIR_NS = 3.5
_EDIT_KEY_NS = 35.0
_EDIT_PAIR_NS = 90.0
_EDIT_CUTOFF_NS = 70.0
# Keys hashed and sorted in one pass, 8 bytes each, 128 MiB in all; more
# keys are split by their hash into passes of at most this many, counted in
# one pass over the texts. A pass also holds the pairs of texts that share
# its keys, 8 bytes each.
_KEYS_PER_PASS = 1 << 24
# The seed of the random codes that hash the keys, fixed so that the keys,
# and the time a search takes, never vary. The pairs never depend on them:
# texts that share a hash by chance are compared and left out.
_TABLE_SEED = 20_240_601
# The column of the table that codes a masked position.
_MASKED = 256


def _zobrist_table(width: int) -> np.ndarray:
    """Random 64-bit codes of each character at each position below ``width``.

    Column 256 codes a masked position.
    """
    table = np.random.PCG64(_TABLE_SEED).random_raw(width * (_MASKED + 1))
    return np.asarray(table, dtype=np.uint64).reshape(width, _MASKED + 1)


@njit(cache=True)
def _text_variants(
    text: np.ndarray,
    length: int,
    limit: int,
    deletions: bool,
    table: np.ndarray,
    shifted: np.ndarray,
    hashes: np.ndarray,
) -> int:  # pragma: no cover
    """Hash the variants of one text into ``hashes``; return how many.

    With ``deletions``, the variants drop 0 to ``limit`` positions, so two
    texts within ``limit`` edits share one; otherwise they mask exactly
    ``limit`` positions, so two texts of one length within ``limit``
    substitutions share one. Up to three positions take part.

    A variant's hash is the exclusive or of the table's codes of its
    characters at their positions, a masked position taking the code in
    column 256, so equal variants share their hash. Masking a further position
    changes a hash by two codes. A deletion moves the characters after it one
    position to the left, and ``shifted`` holds prefix sums of the text's
    codes moved by 0 to 3 positions, so a variant of deletions takes a few of
    them. Either way, a key costs a few operations however long the text.
    Deleting a character that equals the one before it, which stays, gives
    the variant of deleting that one instead, so it is skipped.
    """
    if not deletions:
        if limit > length:
            return 0
        value = np.uint64(0)
        for position in range(length):
            value ^= table[position, text[position]]
        if limit == 0:
            hashes[0] = value
            return 1
        n_hashes = 0
        for first in range(length):
            one = value ^ table[first, text[first]] ^ table[first, _MASKED]
            if limit == 1:
                hashes[n_hashes] = one
                n_hashes += 1
                continue
            for second in range(first + 1, length):
                two = one ^ table[second, text[second]] ^ table[second, _MASKED]
                if limit == 2:
                    hashes[n_hashes] = two
                    n_hashes += 1
                    continue
                for third in range(second + 1, length):
                    hashes[n_hashes] = (
                        two ^ table[third, text[third]] ^ table[third, _MASKED]
                    )
                    n_hashes += 1
        return n_hashes
    # shifted[shift, end]: the hash of the characters from position ``shift``
    # to ``end``, each moved ``shift`` positions to the left.
    for shift in range(min(limit, length) + 1):
        shifted[shift, shift] = 0
        for position in range(shift, length):
            shifted[shift, position + 1] = (
                shifted[shift, position] ^ table[position - shift, text[position]]
            )
    hashes[0] = shifted[0, length]
    n_hashes = 1
    if limit == 0:
        return n_hashes
    for first in range(length):
        if first > 0 and text[first - 1] == text[first]:
            continue
        # The characters before the first deletion keep their positions.
        kept = shifted[0, first]
        hashes[n_hashes] = kept ^ shifted[1, length] ^ shifted[1, first + 1]
        n_hashes += 1
        if limit == 1:
            continue
        for second in range(first + 1, length):
            if second > first + 1 and text[second - 1] == text[second]:
                continue
            kept_two = kept ^ shifted[1, second] ^ shifted[1, first + 1]
            hashes[n_hashes] = kept_two ^ shifted[2, length] ^ shifted[2, second + 1]
            n_hashes += 1
            if limit == 2:
                continue
            for third in range(second + 1, length):
                if third > second + 1 and text[third - 1] == text[third]:
                    continue
                kept_three = kept_two ^ shifted[2, third] ^ shifted[2, second + 1]
                hashes[n_hashes] = (
                    kept_three ^ shifted[3, length] ^ shifted[3, third + 1]
                )
                n_hashes += 1
    return n_hashes


@njit(cache=True)
def _variant_keys(
    codes: np.ndarray,
    lengths: np.ndarray,
    limit: int,
    deletions: bool,
    table: np.ndarray,
    hashes: np.ndarray,
    bits: int,
    n_parts: int,
    part: int,
    counts: np.ndarray,
    out: np.ndarray,
) -> int:  # pragma: no cover
    """Hash each text's variants, counting each part's keys or writing one part's.

    The top 32 bits of a hash choose its part among ``n_parts``. With
    ``counts``, each key adds one to the count of its part; otherwise the keys
    of ``part`` are written to ``out``, the low ``bits`` bits of each hash
    replaced by the text's row, and their number is returned. ``hashes`` is
    scratch space for the variants of one text.
    """
    high = ~np.uint64((1 << bits) - 1)
    parts = np.uint64(n_parts)
    chosen = np.uint64(part)
    top = np.uint64(32)
    shifted = np.zeros((4, codes.shape[1] + 1), dtype=np.uint64)
    at = 0
    for row in range(codes.shape[0]):
        n_hashes = _text_variants(
            codes[row], lengths[row], limit, deletions, table, shifted, hashes
        )
        if len(counts):
            for index in range(n_hashes):
                counts[((hashes[index] >> top) * parts) >> top] += 1
            continue
        tag = np.uint64(row)
        for index in range(n_hashes):
            value = hashes[index]
            if n_parts == 1 or ((value >> top) * parts) >> top == chosen:
                out[at] = (value & high) | tag
                at += 1
    return at


@njit(cache=True)
def _run_pairs(
    keys: np.ndarray, bits: int, n_rows: int, out: np.ndarray
) -> int:  # pragma: no cover
    """Pair the rows of each run of keys with equal hashes; return how many pairs.

    The keys are sorted, so the rows of a run ascend and a row that a run
    repeats is paired once. A pair is written as ``first * n_rows + second``;
    with an empty output, the pairs are only counted.
    """
    shift = np.uint64(bits)
    low = np.uint64((1 << bits) - 1)
    write = len(out) > 0
    at = 0
    start = 0
    n_keys = len(keys)
    while start < n_keys:
        value = keys[start] >> shift
        end = start + 1
        while end < n_keys and keys[end] >> shift == value:
            end += 1
        for left in range(start, end - 1):
            if left > start and keys[left] == keys[left - 1]:
                continue
            first = np.int64(keys[left] & low) * n_rows
            for right in range(left + 1, end):
                if keys[right] != keys[right - 1]:
                    if write:
                        out[at] = first + np.int64(keys[right] & low)
                    at += 1
        start = end
    return at


@njit(cache=True)
def _check_pairs(
    codes: np.ndarray,
    lengths: np.ndarray,
    pairs: np.ndarray,
    n_rows: int,
    limit: int,
    edits: bool,
) -> int:  # pragma: no cover
    """Keep each distinct pair within ``limit`` substitutions or edits.

    The pairs are sorted codes ``first * n_rows + second``; those kept move to
    the front, and their number is returned.
    """
    rows = np.empty((2, codes.shape[1] + 1), dtype=np.int64)
    kept = 0
    previous = -1
    for at in range(len(pairs)):
        pair = pairs[at]
        if pair == previous:
            continue
        previous = pair
        first = pair // n_rows
        second = pair - first * n_rows
        if edits:
            near = _within_edits(
                codes[first],
                lengths[first],
                codes[second],
                lengths[second],
                limit,
                rows,
            )
        elif lengths[first] != lengths[second]:
            near = False
        else:
            mismatches = 0
            for position in range(lengths[first]):
                if codes[first, position] != codes[second, position]:
                    mismatches += 1
                    if mismatches > limit:
                        break
            near = mismatches <= limit
        if near:
            pairs[kept] = pair
            kept += 1
    return kept


def _key_count(length: int, limit: int, deletions: bool) -> int:
    """The keys that one text of ``length`` yields, at most."""
    if deletions:
        return sum(comb(length, count) for count in range(min(limit, length) + 1))
    return comb(length, limit) if limit <= length else 0


def _keyed_pairs(
    codes: np.ndarray,
    lengths: np.ndarray,
    limit: int,
    deletions: bool,
    table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Every pair within ``limit`` through shared variant keys, verified.

    A key holds its hash with the row in the low bits, so one sort in place
    brings the rows of each hash together, ascending. Keys are split by their
    hash into passes of at most ``_KEYS_PER_PASS``, so the keys held at once
    stay bounded however many the texts yield: one pass over the texts counts
    the keys of every pass, and each pass hashes them again and keeps its own.
    """
    n_rows = len(lengths)
    bits = max(int(n_rows - 1).bit_length(), 1)
    per_length = np.bincount(lengths)
    total = sum(
        int(texts) * _key_count(length, limit, deletions)
        for length, texts in enumerate(per_length.tolist())
        if texts
    )
    n_parts = max(1, -(-total // _KEYS_PER_PASS))
    no_keys = np.empty(0, dtype=np.uint64)
    no_counts = np.empty(0, dtype=np.int64)
    hashes = np.empty(
        _key_count(len(per_length) - 1, limit, deletions), dtype=np.uint64
    )
    sizes = np.full(1, total, dtype=np.int64)
    if n_parts > 1:
        sizes = np.zeros(n_parts, dtype=np.int64)
        _variant_keys(
            codes,
            lengths,
            limit,
            deletions,
            table,
            hashes,
            bits,
            n_parts,
            0,
            sizes,
            no_keys,
        )
    found = []
    for part in range(n_parts):
        keys = np.empty(int(sizes[part]), dtype=np.uint64)
        written = _variant_keys(
            codes,
            lengths,
            limit,
            deletions,
            table,
            hashes,
            bits,
            n_parts,
            part,
            no_counts,
            keys,
        )
        keys = keys[:written]
        keys.sort()
        pairs = np.empty(_run_pairs(keys, bits, n_rows, no_counts), dtype=np.int64)
        _run_pairs(keys, bits, n_rows, pairs)
        del keys
        pairs.sort()
        kept = _check_pairs(codes, lengths, pairs, n_rows, limit, deletions)
        found.append(pairs[:kept].copy())
        del pairs
    if n_parts == 1:
        near = found[0]
    else:
        # A pair whose texts share keys in several passes is found in each.
        merged = np.concatenate(found)
        found.clear()
        near = sorted_unique(merged)
        del merged
    return np.divmod(near, n_rows)


def _encoded(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Encode texts as a padded matrix of character codes and their lengths.

    A character below 256, such as an ASCII letter, is coded by its code
    point. Texts with other characters code each distinct character by its
    rank instead, which allows 256 of them.
    """
    lengths = np.fromiter(map(len, texts), dtype=np.int64, count=len(texts))
    width = int(lengths.max()) if len(texts) else 0
    points = np.frombuffer("".join(texts).encode("utf-32-le"), dtype=np.uint32)
    if len(points) and int(points.max()) > 255:
        distinct, ranks = unique_inverse(points)
        if len(distinct) > 256:
            raise ValueError("texts can hold at most 256 distinct characters")
        points = ranks
    matrix = np.zeros((len(texts), max(width, 1)), dtype=np.uint8)
    matrix[np.arange(matrix.shape[1]) < lengths[:, None]] = points.astype(np.uint8)
    return matrix, lengths


def _ordered(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The positions of pairs, the lower of each first; ``second`` is reused."""
    low = np.minimum(first, second)
    np.maximum(first, second, out=second)
    return low, second


def _hamming_keyed(n_texts: int, length: int, limit: int) -> bool:
    """Whether keys find the pairs of texts of one length sooner."""
    if limit > 3:
        return False
    keys = n_texts * _key_count(length, limit, False)
    return keys * _KEY_NS < n_texts * (n_texts - 1) / 2 * _PAIR_NS


def _edits_keyed(sizes: np.ndarray, lengths: np.ndarray, limit: int) -> bool:
    """Whether keys find the pairs of a block sooner.

    ``sizes`` holds the block's texts of each length in ``lengths``, ascending.
    """
    if limit > 3:
        return False
    keys = sum(
        int(size) * _key_count(int(length), limit, True)
        for size, length in zip(sizes.tolist(), lengths.tolist(), strict=True)
    )
    # Each length with itself, and with each longer length within the cutoff.
    cumulative = np.concatenate([[0], np.cumsum(sizes)])
    reach = np.searchsorted(lengths, lengths + limit, side="right")
    longer = cumulative[reach] - cumulative[1:]
    pairs = float(np.sum(sizes * (sizes - 1) / 2 + sizes * longer))
    return keys * _EDIT_KEY_NS < pairs * (_EDIT_PAIR_NS + _EDIT_CUTOFF_NS * limit)


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
    empty = np.empty(0, dtype=np.int64)
    if n_texts < 2:
        return empty, empty.copy()
    table = _zobrist_table(matrix.shape[1])
    # The texts in order of block and length, so that the texts of a block,
    # and of each of its lengths, are slices.
    _distinct, ranks = unique_inverse(block_codes)
    group_keys = ranks * (int(lengths.max()) + 1) + lengths
    order = np.argsort(group_keys, kind="stable")
    ordered = group_keys[order]
    starts = np.flatnonzero(np.concatenate([[True], ordered[1:] != ordered[:-1]]))
    ends = np.append(starts[1:], n_texts)
    group_lengths = lengths[order[starts]]
    firsts: list[np.ndarray] = []
    seconds: list[np.ndarray] = []
    if metric == "levenshtein":
        limit = int(cutoff)
        group_ranks = ranks[order[starts]]
        block_starts = np.flatnonzero(
            np.concatenate([[True], group_ranks[1:] != group_ranks[:-1]])
        )
        block_ends = np.append(block_starts[1:], len(starts))
        for low, high in zip(block_starts.tolist(), block_ends.tolist(), strict=True):
            members = order[starts[low] : ends[high - 1]]
            if len(members) < 2:
                continue
            sizes = ends[low:high] - starts[low:high]
            if _edits_keyed(sizes, group_lengths[low:high], limit):
                first, second = _keyed_pairs(
                    matrix[members], lengths[members], limit, True, table
                )
                low_positions, high_positions = _ordered(
                    members[first], members[second]
                )
                firsts.append(low_positions)
                seconds.append(high_positions)
                continue
            # Each pair of lengths within the cutoff, the shorter first.
            for group in range(low, high):
                length = int(group_lengths[group])
                texts_of = order[starts[group] : ends[group]]
                for other in range(group, high):
                    other_length = int(group_lengths[other])
                    if other_length - length > limit:
                        break
                    partners = order[starts[other] : ends[other]]
                    first, second = _edit_pairs(
                        matrix[texts_of, :length],
                        lengths[texts_of],
                        matrix[partners, :other_length],
                        lengths[partners],
                        limit,
                        other == group,
                    )
                    low_positions, high_positions = _ordered(
                        texts_of[first], partners[second]
                    )
                    firsts.append(low_positions)
                    seconds.append(high_positions)
    else:
        for group in range(len(starts)):
            if ends[group] - starts[group] < 2:
                continue
            members = order[starts[group] : ends[group]]
            length = int(group_lengths[group])
            limit = (
                int(cutoff)
                if metric == "hamming"
                else allowed_mismatches(length, int(cutoff))
            )
            limit = min(limit, length)
            codes = matrix[members, :length]
            if _hamming_keyed(len(members), length, limit):
                first, second = _keyed_pairs(
                    codes, lengths[members], limit, False, table
                )
            else:
                first, second = _hamming_pairs(_words(codes), limit)
            firsts.append(members[first])
            seconds.append(members[second])
    if not firsts:
        return empty, empty.copy()
    if len(firsts) == 1:
        return firsts[0], seconds[0]
    return np.concatenate(firsts), np.concatenate(seconds)
