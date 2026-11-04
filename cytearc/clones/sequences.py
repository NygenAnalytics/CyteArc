"""Pairs of sequences within a distance of each other.

Clonotype clusters link receptor chains whose junctions are similar rather
than identical, and lineage barcode correction merges each barcode into a
more abundant one a few substitutions or edits away. Both need every pair of
texts within a distance, found exactly, under one of scirpy's distances:

- ``hamming``: the substitutions between texts of equal length;
- ``normalized_hamming``: those substitutions as a percentage of the length,
  rounded half up, between texts of equal length;
- ``levenshtein``: the substitutions, insertions and deletions.

:func:`neighbor_pairs` takes the texts, a metric, the largest distance and,
optionally, a block per text, such as its gene calls or its sample, so that
only the texts of one block are compared. It returns the positions of every
pair within the distance and charges what it holds while it searches to the
caller's memory ledger: the encoded texts, the variant keys and candidate
pairs of one pass, and the pairs it finds. A search that would need more than
the ledger's budget is refused before it allocates.

Every pair is found in one of two ways, whichever is estimated to take less
time for the texts of each length, or under ``levenshtein`` for those of each
block. Compiled loops compare every pair of one length, or of lengths within
the distance, and stop at the first position where the distance is exceeded.
Or each text yields variant keys that any two texts within the distance
share, and only the texts that share a key are compared. Under substitutions
a key masks as many runs of positions as the distance allows: one position
each, or, for long texts such as lineage barcodes, fewer and longer runs,
whose keys are far fewer, at the cost of texts that share one by chance,
which are compared and left out; a few more runs than the distance serve
any distance, such as TREX's 5. Under ``levenshtein``, keys delete up to 3
positions.

Texts in which some characters hold no base, such as TREX's partial
cloneIDs, compare only the positions where both hold one. They are grouped
by the positions where they hold none, so that the texts of two groups
compare the same positions, and each pair of groups is searched on those
positions in the same two ways. These functions work on text and never touch
storage.

:func:`expected_chance_pairs` says how many of the pairs that the search
compares would fall within ``max_distance`` by chance, if texts of different
components, such as the barcodes that correction links, agreed at each
position independently, as often as one text per component of each does: an
expectation that grows with the square of the texts compared, which lineage
barcode correction records beside the barcodes it merges.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from math import comb
from typing import Any

import numpy as np

from ..utils.arguments import integer_argument
from ..utils.arrays import sorted_unique, unique_inverse
from ..utils.memory_budget import NUMPY_STEP_BYTES, MemoryLedger
from ..utils.numba import njit

__all__ = [
    "DISTANCE_METRICS",
    "MAX_CHANCE_EDITS",
    "allowed_mismatches",
    "expected_chance_pairs",
    "neighbor_pairs",
]

#: The metrics of :func:`neighbor_pairs`.
DISTANCE_METRICS: tuple[str, ...] = ("hamming", "normalized_hamming", "levenshtein")
#: The most edits within which :func:`expected_chance_pairs` counts the pairs
#: expected by chance under ``levenshtein``, whose cost grows about eightfold
#: with every two edits more.
MAX_CHANCE_EDITS = 8
# The bytes of an array's object beside its values, for the arrays that each
# length or block, and each pass of keys, keeps until the search returns.
_ARRAY_BYTES = 192
# The bytes of each group of texts of one length and block: its start, end
# and length, and under levenshtein its block and its block's start and end.
_GROUP_BYTES = 48
# The bytes that each pair found holds until the search returns: its two
# positions, and their copy when the pairs of all lengths are joined.
_FOUND_PAIR_BYTES = 32
# The bytes of each text beside its characters while the search runs: its
# place in the list of texts, its length, block, rank among the blocks, group
# key and position in the order of blocks and lengths, and the sorted keys,
# with two copies of the lengths of the texts compared at once.
_TEXT_BYTES = 72
# The bytes of each text of a length while texts with characters of no base
# are grouped by the positions they hold none at: its count of bases, its
# place among the texts that hold enough, its place in their order and a sort
# of them, its place among the texts of two groups, and its flag of a new
# group; two copies of its positions take two 64-bit words per 64 positions.
_MASK_TEXT_BYTES = 48
# The bytes of a pair while a compiled loop collects it: two buffers that
# double as they fill, and the copies returned.
_LOOP_PAIR_BYTES = 32
# The bytes of a variant key, and of a candidate pair of texts that share one.
_KEY_BYTES = 8
_CANDIDATE_BYTES = 8
# The pairs that a pass of variant keys keeps hold their codes until the
# passes are joined, which then needs this many bytes per pair beside them,
# 32 in all: the joined codes, their sorted copy and the distinct ones, and
# then the two positions of each distinct pair.
_KEPT_PAIR_BYTES = 8
_JOINED_PAIR_BYTES = 24
# Variant keys take at most this share of the bytes a search has left for a
# pass, so that the candidate pairs of the pass fit beside them, but a pass
# holds at least _MIN_KEYS_PER_PASS keys, 8 MiB: a budget too small for them
# fails rather than hashing every text again in thousands of passes.
_KEY_SHARE = 4
_MIN_KEYS_PER_PASS = 1 << 20
# The room of a search without a limit: more bytes than any machine holds.
_UNLIMITED = 2**62
#: What a refusal of a search suggests beside a larger budget.
_REMEDY = "lower max_distance, or compare fewer texts at once or within smaller blocks"
# The pairs expected by chance hold, for each position of the texts of one
# length in a block, the frequency of each of 256 codes, and their chance of
# agreeing: under levenshtein for each pair of positions of two lengths, and
# with characters of no base for each position.
_FREQUENCY_BYTES = 8 * 256
_AGREEMENT_BYTES = 8
# Counting the codes at one position holds each text's code and its copy
# widened to 8 bytes.
_COUNTED_CODE_BYTES = 9
# Each state of the edit distances of chance pairs while one cell expands:
# its key and chance, those of its two children, their order, sorted copies
# and flags, and the states kept, at most 146 bytes as traced at 8 edits;
# and the small arrays of the cells, up to 25 KiB at 4 edits.
_CHANCE_STATE_BYTES = 160
_CHANCE_FIXED_BYTES = 1 << 15
# Choosing one text per component holds, for each text, its component and
# priority, their order, the sorted runs and the texts chosen; taking each
# position's character of one text per component, with characters of no base,
# also holds each text's run and, for one position, the texts that hold a
# base there.
_REPRESENTATIVE_ROW_BYTES = 48
_LIBRARY_ROW_BYTES = 64


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


def _ledger(ledger: MemoryLedger | None) -> MemoryLedger:
    """The caller's ledger, or one without a budget for a search without one."""
    if ledger is None:
        return MemoryLedger(None, "Finding the pairs of texts within max_distance")
    if not isinstance(ledger, MemoryLedger):
        raise TypeError("ledger must be a MemoryLedger or None")
    return ledger


def _room(ledger: MemoryLedger) -> int:
    """The bytes the search may still allocate, more than any machine holds without a budget."""
    room = ledger.room
    return _UNLIMITED if room is None else room


def _check(ledger: MemoryLedger, needed: int, what: str) -> None:
    """Refuse ``needed`` more bytes for ``what`` that would pass the ledger's budget."""
    ledger.peak(what, needed, remedy=_REMEDY)


def _found_bytes(n_pairs: int) -> int:
    """What the pairs of a length or block hold until the search returns: ``_FOUND_PAIR_BYTES`` each and their two arrays, nothing for none."""
    return _FOUND_PAIR_BYTES * n_pairs + 2 * _ARRAY_BYTES if n_pairs else 0


def _hold(ledger: MemoryLedger, nbytes: int, what: str) -> None:
    """Charge ``nbytes`` that the search keeps for ``what``."""
    ledger.hold(what, nbytes, remedy=_REMEDY)


def _charge_to(ledger: MemoryLedger, charged: int, what: str) -> None:
    """Hold or let go of what the search keeps, so that the ledger's charges come to ``charged``."""
    if charged >= ledger.charged:
        _hold(ledger, charged - ledger.charged, what)
    else:
        ledger.release(ledger.charged - charged)


@njit(cache=True)
def _grown(values: np.ndarray, most: int) -> np.ndarray:
    """A copy of ``values`` with room for twice as many, and at most ``most``."""
    grown = np.empty(min(2 * len(values), most), dtype=values.dtype)
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
    words: np.ndarray, limit: int, most: int, split: int
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Every pair of rows within ``limit`` substitutions, the first below the second.

    The rows are texts of one length, eight codes to a word. With a
    ``split`` above 0, only the rows below it pair, with those at or above
    it. The bytes that differ in two words are the nonzero bytes of their
    exclusive or, which folding each byte onto its lowest bit and summing
    those bits counts. Returns False, with no pairs, once more than ``most``
    pairs are found.
    """
    rows, n_words = words.shape
    ones = np.uint64(0x0101010101010101)
    start = min(max(rows, 16), most)
    firsts = np.empty(start, dtype=np.int64)
    seconds = np.empty(start, dtype=np.int64)
    at = 0
    for first in range(split if split else rows):
        for second in range(split if split else first + 1, rows):
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
                    if at == most:
                        return firsts[:0].copy(), seconds[:0].copy(), False
                    firsts = _grown(firsts, most)
                    seconds = _grown(seconds, most)
                firsts[at] = first
                seconds[at] = second
                at += 1
    return firsts[:at].copy(), seconds[:at].copy(), True


@njit(cache=True)
def _unknown_masks(
    matrix: np.ndarray, rows: np.ndarray, length: int, unknown: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The positions where each row holds a character that ``unknown`` flags.

    Returns the positions of each row as the bits of 64-bit words, and how
    many of its first ``length`` positions hold another character.
    """
    n_words = max((length + 63) // 64, 1)
    masks = np.zeros((len(rows), n_words), dtype=np.uint64)
    known = np.empty(len(rows), dtype=np.int64)
    for at in range(len(rows)):
        row = rows[at]
        held = 0
        for position in range(length):
            if unknown[matrix[row, position]]:
                masks[at, position >> 6] |= np.uint64(1) << np.uint64(position & 63)
            else:
                held += 1
        known[at] = held
    return masks, known


@njit(cache=True)
def _within_edits(
    left: np.ndarray,
    left_length: int,
    right: np.ndarray,
    right_length: int,
    limit: int,
    rows: np.ndarray,
) -> bool:
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
    most: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Every pair of a left and a right row within ``limit`` edits.

    With ``same``, the blocks are one and the first row is below the second.
    Returns False, with no pairs, once more than ``most`` pairs are found.
    """
    rows = np.empty((2, right.shape[1] + 1), dtype=np.int64)
    start = min(max(left.shape[0], 16), most)
    firsts = np.empty(start, dtype=np.int64)
    seconds = np.empty(start, dtype=np.int64)
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
                    if at == most:
                        return firsts[:0].copy(), seconds[:0].copy(), False
                    firsts = _grown(firsts, most)
                    seconds = _grown(seconds, most)
                firsts[at] = first
                seconds[at] = second
                at += 1
    return firsts[:at].copy(), seconds[:at].copy(), True


# The search compares every pair or uses keys, whichever is estimated to take
# less time: m * (m - 1) / 2 comparisons of m texts of one length (for edits,
# of the texts of lengths within max_distance), or the keys that the texts
# yield, each at the cost below in nanoseconds. A key costs its hash and its
# share of the sort, of pairing the texts that share it and of checking them:
# 25 to 50 ns, more as the keys grow from 100,000 to 50 million. A comparison
# of substitutions costs 2 to 5 ns; one of edits about 90 ns plus 70 ns for
# each edit of max_distance, and several times more between similar texts, such
# as those of one B cell clone, which fill the whole band. Measured on one core
# on simulated junctions of T cells (amino acids of 14 and 18 residues) and B
# cells (nucleotides of 36 to 66 bases), 300 to 30,000 of a length, with
# distances of 1 to 3. On 200,000 simulated cells, the choices took at most 5%
# longer than the faster way for each length within 1 to 3 substitutions.
_KEY_NS = 40.0
_PAIR_NS = 3.5
# A candidate pair of texts that share a key by chance costs its place in the
# pairs of its run, its share of their sort and its comparison.
_CANDIDATE_NS = 30.0
# The segments of masked keys tried beyond the fewest that the limit allows,
# besides one segment per position.
_MAX_EXTRA_SEGMENTS = 12
_EDIT_KEY_NS = 35.0
_EDIT_PAIR_NS = 90.0
_EDIT_CUTOFF_NS = 70.0
# Keys hashed and sorted in one pass, 8 bytes each, 128 MiB in all, or fewer
# when the memory left is smaller; more keys are split by their hash into
# passes of at most this many, counted in one pass over the texts. A pass also
# holds the pairs of texts that share its keys, 8 bytes each.
_KEYS_PER_PASS = 1 << 24
# The seed of the random codes that hash the keys, fixed so that the keys,
# and the time a search takes, never vary. The pairs never depend on them:
# texts that share a hash by chance are compared and left out.
_TABLE_SEED = 20_240_601
# The column of the table that codes a masked position.
_MASKED = 256


def _table_bytes(width: int) -> int:
    """The bytes of the codes that hash the keys of texts up to ``width`` long."""
    return 8 * width * (_MASKED + 1)


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
    bounds: np.ndarray,
) -> int:
    """Hash the variants of one text into ``hashes``; return how many.

    With ``deletions``, the variants drop 0 to ``limit`` positions, so two
    texts within ``limit`` edits share one. Otherwise they mask exactly
    ``limit`` of the segments that ``bounds`` delimits, each one or more
    positions in a row, so two texts of one length within ``limit``
    substitutions share the variant that masks the segments of their
    substitutions, or any segments where those are fewer. Masking ``limit``
    segments keeps the others, and the variants are found by whichever
    takes fewer loops, the segments masked or those kept, up to three of
    them.

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
        value = np.uint64(0)
        for position in range(length):
            value ^= table[position, text[position]]
        if limit == 0:
            hashes[0] = value
            return 1
        # Masking a segment changes the hash by the codes of its characters
        # and of its masked positions, held in the first row of ``shifted``.
        n_segments = len(bounds) - 1
        for segment in range(n_segments):
            change = np.uint64(0)
            for position in range(bounds[segment], bounds[segment + 1]):
                change ^= table[position, text[position]] ^ table[position, _MASKED]
            shifted[0, segment] = change
        # Keeping a set of segments masks the others: the hash of every
        # position masked, changed back by the segments kept.
        kept = n_segments - limit
        chosen = limit
        if kept < limit:
            chosen = kept
            for segment in range(n_segments):
                value ^= shifted[0, segment]
        if chosen == 0:
            hashes[0] = value
            return 1
        n_hashes = 0
        for first in range(n_segments):
            one = value ^ shifted[0, first]
            if chosen == 1:
                hashes[n_hashes] = one
                n_hashes += 1
                continue
            for second in range(first + 1, n_segments):
                two = one ^ shifted[0, second]
                if chosen == 2:
                    hashes[n_hashes] = two
                    n_hashes += 1
                    continue
                for third in range(second + 1, n_segments):
                    hashes[n_hashes] = two ^ shifted[0, third]
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
    bounds: np.ndarray,
) -> int:
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
            codes[row], lengths[row], limit, deletions, table, shifted, hashes, bounds
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
    keys: np.ndarray, bits: int, n_rows: int, out: np.ndarray, split: int
) -> int:
    """Pair the rows of each run of keys with equal hashes; return how many pairs.

    The keys are sorted, so the rows of a run ascend and a row that a run
    repeats is paired once. With a ``split`` above 0, only the rows below it
    pair, with those at or above it. A pair is written as
    ``first * n_rows + second``; with an empty output, the pairs are only
    counted.
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
        # The rows of the run at or above the split follow those below it.
        middle = end - 1
        if split:
            middle = start
            while middle < end and np.int64(keys[middle] & low) < split:
                middle += 1
        for left in range(start, middle):
            if left > start and keys[left] == keys[left - 1]:
                continue
            first = np.int64(keys[left] & low) * n_rows
            for right in range(middle if split else left + 1, end):
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
) -> int:
    """Keep each distinct pair within ``limit`` substitutions or edits.

    The pairs are sorted codes ``first * n_rows + second``; those kept move to
    the front, and their number is returned. Under substitutions, the texts
    have one length.
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


def _key_count(
    length: int, limit: int, deletions: bool, segments: int | None = None
) -> int:
    """The keys that one text of ``length`` yields, at most.

    Under substitutions a text yields one key for each way of masking
    ``limit`` of its ``segments``, one per position unless given.
    """
    if deletions:
        return sum(comb(length, count) for count in range(min(limit, length) + 1))
    parts = length if segments is None else segments
    return comb(parts, limit) if limit <= parts else 0


def _segment_bounds(length: int, segments: int) -> np.ndarray:
    """The bounds of ``segments`` runs of positions that split ``length`` evenly."""
    return (np.arange(segments + 1, dtype=np.int64) * length) // segments


def _keyed_pairs(
    codes: np.ndarray,
    lengths: np.ndarray,
    limit: int,
    deletions: bool,
    table: np.ndarray,
    ledger: MemoryLedger,
    segments: int = 0,
    split: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Every pair within ``limit`` through shared variant keys, verified.

    Under substitutions the texts share one length, split into ``segments``
    runs of positions, or one per position for 0, of which each key masks
    ``limit``. With a ``split`` above 0, only the rows below it pair, with
    those at or above it.

    A key holds its hash with the row in the low bits, so one sort in place
    brings the rows of each hash together, ascending. Keys are split by their
    hash into passes of at most ``_KEYS_PER_PASS``, or fewer when the memory
    left is smaller, so the keys held at once stay bounded however many the
    texts yield: one pass over the texts counts the keys of every pass, and
    each pass hashes them again and keeps its own. The keys and candidate
    pairs of a pass, and the pairs kept, are charged to ``ledger`` before
    they are allocated, and the pairs returned stay charged to it as
    :class:`_Found` keeps them.
    """
    n_rows = len(lengths)
    bits = max(int(n_rows - 1).bit_length(), 1)
    per_length = np.bincount(lengths)
    if deletions:
        parts = None
        bounds = np.zeros(1, dtype=np.int64)
    else:
        length = len(per_length) - 1
        parts = segments or length
        bounds = _segment_bounds(length, parts)
    total = sum(
        int(texts) * _key_count(length, limit, deletions, parts)
        for length, texts in enumerate(per_length.tolist())
        if texts
    )
    keys_per_pass = min(
        _KEYS_PER_PASS,
        max(_MIN_KEYS_PER_PASS, _room(ledger) // (_KEY_SHARE * _KEY_BYTES)),
    )
    n_parts = max(1, -(-total // keys_per_pass))
    no_keys = np.empty(0, dtype=np.uint64)
    no_counts = np.empty(0, dtype=np.int64)
    entry = ledger.charged
    # The scratch space of one text's keys, held while the search runs.
    scratch = _key_count(len(per_length) - 1, limit, deletions, parts)
    _check(ledger, _KEY_BYTES * scratch, "the variant keys of one text")
    hashes = np.empty(scratch, dtype=np.uint64)
    _hold(ledger, _KEY_BYTES * scratch, "the variant keys of one text")
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
            bounds,
        )
    held = ledger.charged
    found = []
    n_kept = 0
    for part in range(n_parts):
        size = int(sizes[part])
        _check(ledger, _KEY_BYTES * size, "the variant keys of one pass")
        keys = np.empty(size, dtype=np.uint64)
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
            bounds,
        )
        keys = keys[:written]
        keys.sort()
        n_candidates = _run_pairs(keys, bits, n_rows, no_counts, split)
        # The slice of the keys written keeps the whole buffer of the pass.
        _check(
            ledger,
            _KEY_BYTES * size + _CANDIDATE_BYTES * n_candidates,
            "the candidate pairs of texts that share variant keys",
        )
        pairs = np.empty(n_candidates, dtype=np.int64)
        _run_pairs(keys, bits, n_rows, pairs, split)
        del keys
        pairs.sort()
        kept = _check_pairs(codes, lengths, pairs, n_rows, limit, deletions)
        # The pairs kept are copied out while the candidates are still held.
        _check(
            ledger,
            _CANDIDATE_BYTES * n_candidates + _KEPT_PAIR_BYTES * kept + _ARRAY_BYTES,
            "the pairs found",
        )
        found.append(pairs[:kept].copy())
        del pairs
        n_kept += kept
        _charge_to(
            ledger,
            held + _KEPT_PAIR_BYTES * n_kept + _ARRAY_BYTES * len(found),
            "the pairs found",
        )
    # The two arrays of the pairs are then held until the search returns.
    _check(ledger, _JOINED_PAIR_BYTES * n_kept + 2 * _ARRAY_BYTES, "the pairs found")
    if n_parts == 1:
        near = found[0]
    else:
        # A pair whose texts share keys in several passes is found in each.
        merged = np.concatenate(found)
        found.clear()
        near = sorted_unique(merged)
        del merged
    # The pairs returned stay charged as the pairs found.
    first, second = np.divmod(near, n_rows)
    del near
    _charge_to(ledger, entry + _found_bytes(len(first)), "the pairs found")
    return first, second


def _texts(texts: Any) -> list[str]:
    """The texts as a list of str, refusing anything that is not text."""
    if isinstance(texts, str | bytes):
        raise TypeError("texts must be a sequence of texts, not one text")
    array = np.asarray(texts, dtype=object)
    if array.ndim != 1:
        raise TypeError("texts must be a one-dimensional sequence of texts")
    values: list[str] = array.tolist()
    if not all(isinstance(text, str) for text in values):
        raise TypeError(
            "texts must hold only text (str), such as junctions or barcodes"
        )
    return values


def _blocks(blocks: Any, n_texts: int) -> np.ndarray:
    """Each text's block as int64, every text in block 0 without blocks."""
    if blocks is None:
        return np.zeros(n_texts, dtype=np.int64)
    array = np.asarray(blocks)
    if array.ndim != 1 or (len(array) and array.dtype.kind not in "biu"):
        raise TypeError("blocks must hold one integer code per text, or be None")
    if len(array) != n_texts:
        raise ValueError("blocks must hold one block per text")
    return array.astype(np.int64)


def _encoded(
    texts: list[str], ledger: MemoryLedger
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Encode texts as a padded matrix of character codes and their lengths.

    A character below 256, such as an ASCII letter, is coded by its code
    point. Texts with other characters code each distinct character by its
    rank instead, which allows 256 of them, and the sorted code points of
    those characters are returned, None for texts coded by code point. The
    memory the encoding needs is charged before it is allocated, and the
    matrix and the per-text arrays of the search are held.
    """
    lengths = np.fromiter(map(len, texts), dtype=np.int64, count=len(texts))
    width = max(int(lengths.max()) if len(texts) else 0, 1)
    characters = int(lengths.sum())
    # The matrix, and two copies of the rows of one length or block while
    # they are searched, such as the rows and their words or the rows of two
    # lengths, and the codes that hash the keys, 257 for each position, held
    # until the search returns; fewer than two texts are never compared.
    held = 3 * len(texts) * width + _TEXT_BYTES * len(texts)
    if len(texts) > 1:
        held += _table_bytes(width) + NUMPY_STEP_BYTES
    what = "the encoded texts"
    # Joined, the texts take a byte per character, and their codes another.
    _check(ledger, held + 2 * characters, what)
    joined = "".join(texts)
    coding = None
    try:
        points = np.frombuffer(joined.encode("latin-1"), dtype=np.uint8)
    except UnicodeEncodeError:
        # Four bytes per character joined and as code points, and about 32
        # while the characters are ranked. A lone surrogate is a character
        # like any other.
        _check(ledger, held + 40 * characters, what)
        wide = np.frombuffer(
            joined.encode("utf-32-le", "surrogatepass"), dtype=np.uint32
        )
        distinct, ranks = unique_inverse(wide)
        if len(distinct) > 256:
            raise ValueError("texts can hold at most 256 distinct characters") from None
        points = ranks.astype(np.uint8)
        coding = distinct
    del joined
    matrix = np.zeros((len(texts), width), dtype=np.uint8)
    matrix[np.arange(width) < lengths[:, None]] = points
    _hold(ledger, held, what)
    return matrix, lengths, coding


def _unknown_table(unknown: str, coding: np.ndarray | None) -> np.ndarray:
    """Flag the codes of the characters that hold no base, by code."""
    table = np.zeros(256, dtype=np.bool_)
    for character in unknown:
        point = ord(character)
        if coding is None:
            if point < 256:
                table[point] = True
            continue
        rank = int(np.searchsorted(coding, point))
        if rank < len(coding) and int(coding[rank]) == point:
            table[rank] = True
    return table


def _agreement(codes: np.ndarray, split: int = 0) -> np.ndarray:
    """The chance that two rows hold one character, at each position.

    Two rows drawn from the frequencies of each position's characters agree
    there with the chance that the frequencies give: the sum of their
    squares, or with a ``split`` above 0, the sum of the products of the
    frequencies below the split and at or above it, since only such rows
    pair. Rows across a split may never agree at a position.
    """
    n_texts, length = codes.shape
    agree = np.empty(length, dtype=np.float64)
    for position in range(length):
        if split:
            below = np.bincount(codes[:split, position], minlength=256) / split
            above = np.bincount(codes[split:, position], minlength=256)
            agree[position] = float(np.sum(below * above / (n_texts - split)))
        else:
            counts = np.bincount(codes[:, position], minlength=256)
            agree[position] = float(np.sum((counts / n_texts) ** 2))
    return agree


def _component_agreement(codes: np.ndarray) -> np.ndarray:
    """The chance that the rows of two components agree, at each position.

    Each row stands for one component, so the chance is the share of pairs
    of two rows that hold one character there: the sum over the characters
    of each one's count times one less, over the rows times one less. A row
    never pairs with itself, which would raise the chance by about its
    complement over the rows, and the rows must be two or more.
    """
    n_rows, length = codes.shape
    agree = np.empty(length, dtype=np.float64)
    for position in range(length):
        counts = np.bincount(codes[:, position], minlength=256)
        agree[position] = int(counts @ (counts - 1)) / (n_rows * (n_rows - 1))
    return agree


@njit(cache=True)
def _mismatch_chances(disagree: np.ndarray, limit: int) -> np.ndarray:
    """The chances of 0 to ``limit`` mismatches between two texts.

    Each position disagrees independently, with its chance in ``disagree``,
    so the mismatches follow a Poisson binomial distribution, whose chances
    up to ``limit`` are built one position at a time.
    """
    chances = np.zeros(limit + 1)
    chances[0] = 1.0
    for chance in disagree:
        for count in range(limit, 0, -1):
            chances[count] = (
                chances[count] * (1.0 - chance) + chances[count - 1] * chance
            )
        chances[0] *= 1.0 - chance
    return chances


def _chance_pairs(n_texts: int, split: int, chances: np.ndarray) -> float:
    """The pairs of texts expected within ``max_distance``, from each pair's chances.

    ``chances`` holds the chance of each number of mismatches within the
    max_distance. With a ``split`` above 0, each text below it pairs with each at
    or above it; otherwise the texts are distinct and differ somewhere, so a
    pair's chance leaves out agreeing at every position and is scaled to the
    pairs that do not.
    """
    within = float(chances.sum())
    if split:
        return split * (n_texts - split) * within
    identical = float(chances[0])
    if identical >= 1.0:
        return 0.0
    return n_texts * (n_texts - 1) / 2 * (within - identical) / (1.0 - identical)


def _edit_chance(agree: np.ndarray, limit: int, ledger: MemoryLedger) -> float:
    """The chance that two texts are within ``limit`` edits.

    The texts have the lengths of ``agree``'s rows and columns, and the
    characters at positions ``i`` and ``j`` agree with chance
    ``agree[i, j]``, independently of the other pairs of positions. The
    chance follows the rows of the edit distance's dynamic programming one
    cell at a time, carrying the chance of each state of the cells that can
    still end within ``limit``: a path through the cell of offset ``o``
    from the diagonal costs at least ``|o| + |k - o|`` for texts ``k``
    characters apart in length, so only offsets where that stays within
    ``limit`` are kept, and each cell's distance is capped where it can no
    longer end within ``limit``, which keeps the distance of every cell that
    can. Each state is a number whose digits are the cells' capped
    distances.
    """
    n_rows, n_columns = agree.shape
    apart = n_columns - n_rows
    if abs(apart) > limit:
        return 0.0
    if limit >= max(n_rows, n_columns):
        # Two texts are never more edits apart than the longer is long.
        return 1.0
    spread = (limit - abs(apart)) // 2
    offsets = np.arange(min(0, apart) - spread, max(0, apart) + spread + 1)
    width = len(offsets)
    caps = limit - np.abs(apart - offsets) + 1
    radix = caps + 1
    powers = np.ones(width, dtype=np.int64)
    for cell in range(1, width):
        powers[cell] = powers[cell - 1] * radix[cell - 1]
    # Distances of at most MAX_CHANCE_EDITS keep the digits within 64 bits.
    assert float(np.prod(radix.astype(np.float64))) < 2.0**62
    first_row = np.where((offsets >= 0) & (offsets <= n_columns), offsets, caps)
    keys = np.asarray([int(np.sum(np.minimum(first_row, caps) * powers))], np.int64)
    chances = np.ones(1)
    for row in range(1, n_rows + 1):
        for cell in range(width):
            column = row + int(offsets[cell])
            digit = (keys // powers[cell]) % radix[cell]
            if column < 0 or column > n_columns:
                keys = keys + (caps[cell] - digit) * powers[cell]
                continue
            if column == 0:
                keys = keys + (min(row, int(caps[cell])) - digit) * powers[cell]
                continue
            _check(
                ledger,
                _CHANCE_STATE_BYTES * len(keys) + _CHANCE_FIXED_BYTES,
                "the edit distances of chance pairs",
            )
            # The cells beside, in the row above and to the left in this row,
            # bound the cell; a cell outside the kept offsets bounds it no
            # lower than its cap.
            bound = np.full(len(keys), caps[cell])
            if cell + 1 < width:
                above = (keys // powers[cell + 1]) % radix[cell + 1] + 1
                bound = np.minimum(bound, above)
            if cell:
                left = (keys // powers[cell - 1]) % radix[cell - 1] + 1
                bound = np.minimum(bound, left)
            matched = np.minimum(digit, bound)
            missed = np.minimum(digit + 1, bound)
            chance = float(agree[row - 1, column - 1])
            same = matched == missed
            child_keys = np.concatenate(
                [
                    keys + (matched - digit) * powers[cell],
                    (keys + (missed - digit) * powers[cell])[~same],
                ]
            )
            child_chances = np.concatenate(
                [
                    np.where(same, chances, chances * chance),
                    chances[~same] * (1.0 - chance),
                ]
            )
            order = np.argsort(child_keys, kind="stable")
            ordered = child_keys[order]
            starts = np.flatnonzero(
                np.concatenate([[True], ordered[1:] != ordered[:-1]])
            )
            keys = ordered[starts]
            chances = np.add.reduceat(child_chances[order], starts)
        # A state whose every cell is capped cannot end within the limit, and
        # one that no pattern of agreeing positions reaches adds nothing, such
        # as a mismatch where every text holds one base.
        alive = np.zeros(len(keys), dtype=bool)
        for cell in range(width):
            alive |= (keys // powers[cell]) % radix[cell] < caps[cell]
        alive &= chances > 0.0
        keys, chances = keys[alive], chances[alive]
        if not len(keys):
            return 0.0
    end = int(np.flatnonzero(offsets == apart)[0])
    within = (keys // powers[end]) % radix[end] <= limit
    return float(chances[within].sum())


def _edit_chance_pairs(
    matrix: np.ndarray,
    groups: list[np.ndarray],
    chosen: list[np.ndarray],
    lengths: list[int],
    limit: int,
    components: np.ndarray,
    ledger: MemoryLedger,
) -> float:
    """The pairs of a block's texts expected within ``limit`` edits by chance.

    ``groups`` holds the rows of each length of the block, ``chosen`` the rows
    of one text per component of that length, and ``lengths`` that length,
    ascending. Each pair of positions agrees with the share of pairs of chosen
    texts of two components that hold one character there; a pair of lengths
    whose chosen texts are of one component alone expects nothing. Each pair
    of lengths within ``limit`` pairs every text of one with every text of
    the other, and two texts of one length are distinct, so pairs that would
    be identical are left out as :func:`_chance_pairs` leaves them out.
    """
    longest, most = max(lengths), max(map(len, chosen))
    held = _FREQUENCY_BYTES * sum(lengths) + _AGREEMENT_BYTES * longest**2
    # Counting one position's codes copies them, widened to 8 bytes each;
    # the chosen texts of one component in two lengths are copied, and one
    # position of one compared with every position of the other.
    counting = _COUNTED_CODE_BYTES * most + _FREQUENCY_BYTES + 3 * most * longest
    _check(ledger, held + counting, "the frequencies of the texts' characters")
    _hold(ledger, held, "the frequencies of the texts' characters")
    counts = []
    for rows, length in zip(chosen, lengths, strict=True):
        tally = np.zeros((length, 256), dtype=np.float64)
        for position in range(length):
            tally[position] = np.bincount(matrix[rows, position], minlength=256)
        counts.append(tally)
    expected = 0.0
    for first in range(len(groups)):
        for second in range(first, len(groups)):
            if lengths[second] - lengths[first] > limit:
                break
            # The chosen texts of one component in both lengths, which never
            # pair by chance.
            rows_first, rows_second = chosen[first], chosen[second]
            if first == second:
                shared_first = shared_second = rows_first
            else:
                _, at_first, at_second = np.intersect1d(
                    components[rows_first],
                    components[rows_second],
                    assume_unique=True,
                    return_indices=True,
                )
                shared_first = rows_first[at_first]
                shared_second = rows_second[at_second]
            apart = len(rows_first) * len(rows_second) - len(shared_first)
            if apart < 1:
                continue
            agree = counts[first] @ counts[second].T
            codes_first = matrix[shared_first, : lengths[first]]
            codes_second = matrix[shared_second, : lengths[second]]
            for position in range(lengths[first]):
                agree[position] -= np.sum(
                    codes_first[:, position, None] == codes_second, axis=0
                )
            del codes_first, codes_second
            agree /= apart
            chance = _edit_chance(agree, limit, ledger)
            n_first, n_second = len(groups[first]), len(groups[second])
            if first != second:
                expected += n_first * n_second * chance
                continue
            # Identical texts take one path of the chance within max_distance,
            # summed apart from it, so rounding may leave them a hair above.
            identical = float(np.prod(np.diag(agree)))
            if n_first > 1 and identical < 1.0:
                expected += (
                    n_first
                    * (n_first - 1)
                    / 2
                    * max(0.0, chance - identical)
                    / (1.0 - identical)
                )
    ledger.release(held)
    return expected


def _hamming_segments(codes: np.ndarray, limit: int, room: int, split: int = 0) -> int:
    """The segments whose masked keys find the pairs of texts of one length soonest.

    A key masks ``limit`` of ``m`` segments of positions in a row, so a text
    yields ``comb(m, limit)`` keys, and two texts share a key when they agree
    on the segments it keeps, which two texts within ``limit`` substitutions
    always do. One segment per position gives the most keys and the fewest
    texts that share one by chance; fewer, longer segments give fewer keys
    and more chance candidates, each compared. Each choice is timed by its
    keys, hashed once for each pass, whose size follows ``room`` as the search
    sizes it, and its candidate pairs, estimated from how often two texts
    agree at each position by the frequencies of its characters, as
    :func:`_agreement` gives them; a choice whose candidates of one pass would
    not fit ``room`` is left out. With a ``split`` above 0, only the rows
    below it pair, with those at or above it.

    Returns:
        The segments of the fastest keyed search, or 0 when comparing every
        pair is faster.
    """
    n_texts, length = codes.shape
    pairs = split * (n_texts - split) if split else n_texts * (n_texts - 1) / 2
    best_time = pairs * _PAIR_NS
    best = 0
    if length == 0 or n_texts < 2 or limit >= length:
        return best
    # The fewest keys, of the fewest segments, without a chance candidate,
    # cost more than comparing every pair of a few texts.
    if n_texts * (limit + 1) * _KEY_NS >= best_time:
        return best
    # The logarithms of the chance that two texts agree at each position,
    # summed along the positions; a position where they never agree counts
    # as a tiny chance.
    agree = _agreement(codes, split)
    summed = np.concatenate([[0.0], np.cumsum(np.log(np.maximum(agree, 1e-300)))])
    keys_per_pass = min(
        _KEYS_PER_PASS, max(_MIN_KEYS_PER_PASS, room // (_KEY_SHARE * _KEY_BYTES))
    )
    # Splits into a few more segments than the limit keep the keys few, and
    # one segment per position the chance candidates fewest; the splits
    # between take more keys for fewer candidates than either.
    # A key masks or keeps at most three segments.
    choices = sorted(
        segments
        for segments in {
            *range(limit + 1, min(length, limit + _MAX_EXTRA_SEGMENTS) + 1),
            length,
        }
        if min(limit, segments - limit) <= 3
    )
    for segments in choices:
        bounds = _segment_bounds(length, segments)
        kept = segments - limit
        # The chance that two texts agree on each segment, and on the kept
        # segments of some key: the elementary symmetric sum of order kept.
        agreements = np.exp(summed[bounds[1:]] - summed[bounds[:-1]]).tolist()
        sums = [1.0] + [0.0] * kept
        for value in agreements:
            for order in range(kept, 0, -1):
                sums[order] += sums[order - 1] * value
        candidates = pairs * sums[kept]
        keys = n_texts * comb(segments, limit)
        # Passes split the keys, and the candidates with them, by hash.
        passes = -(-keys // keys_per_pass)
        if _CANDIDATE_BYTES * candidates / passes > room:
            continue
        time = keys * passes * _KEY_NS + candidates * _CANDIDATE_NS
        if time < best_time:
            best_time = time
            best = segments
    return best


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
    # Each length with itself, and with each longer length within max_distance.
    cumulative = np.concatenate([[0], np.cumsum(sizes)])
    reach = np.searchsorted(lengths, lengths + limit, side="right")
    longer = cumulative[reach] - cumulative[1:]
    pairs = float(np.sum(sizes * (sizes - 1) / 2 + sizes * longer))
    return keys * _EDIT_KEY_NS < pairs * (_EDIT_PAIR_NS + _EDIT_CUTOFF_NS * limit)


@dataclass(slots=True)
class _Found:
    """The pairs found so far, each holding ``_FOUND_PAIR_BYTES`` of ``ledger``.

    The pairs of each length or block that has any are kept as two arrays,
    which hold ``_ARRAY_BYTES`` each beside their values.
    """

    ledger: MemoryLedger
    firsts: list[np.ndarray]
    seconds: list[np.ndarray]

    def most(self) -> int:
        """The most pairs that a compiled loop may collect in the memory left."""
        return max(0, _room(self.ledger) - 2 * _ARRAY_BYTES) // _LOOP_PAIR_BYTES

    def add(self, first: np.ndarray, second: np.ndarray) -> None:
        """Keep the pairs of a length or block, if it has any, which the search that found them charged."""
        if not len(first):
            return
        self.firsts.append(first)
        self.seconds.append(second)

    def looped(
        self, found: tuple[np.ndarray, np.ndarray, bool]
    ) -> tuple[np.ndarray, np.ndarray]:
        """The pairs of a compiled loop, refused if they outgrew the memory left."""
        first, second, complete = found
        if not complete:
            # A loop stops short only within a budget, which refuses the pairs.
            _check(
                self.ledger,
                _LOOP_PAIR_BYTES * (self.most() + 1) + 2 * _ARRAY_BYTES,
                "the pairs found",
            )
            raise AssertionError("A search without a budget stopped collecting pairs")
        _hold(self.ledger, _found_bytes(len(first)), "the pairs found")
        return first, second


def _substitution_pairs(
    codes: np.ndarray,
    limit: int,
    table: np.ndarray,
    ledger: MemoryLedger,
    found: _Found,
    split: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Every pair of rows within ``limit`` substitutions, by keys or by loops.

    The rows are texts of one length; with a ``split`` above 0, only the rows
    below it pair, with those at or above it.
    """
    limit = min(limit, codes.shape[1])
    if segments := _hamming_segments(codes, limit, _room(ledger), split):
        lengths = np.full(len(codes), codes.shape[1], dtype=np.int64)
        return _keyed_pairs(
            codes, lengths, limit, False, table, ledger, segments, split
        )
    return found.looped(_hamming_pairs(_words(codes), limit, found.most(), split))


def _mask_groups(
    matrix: np.ndarray,
    members: np.ndarray,
    length: int,
    unknown: np.ndarray,
    min_compared: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group the texts of one length by the positions where they hold no base.

    Only texts that hold at least ``min_compared`` bases are grouped, since
    the others pair with nothing.

    Returns:
        The positions among ``members`` of the grouped texts, each group's in
        a row; each group's start and end in that order; and each group's
        positions of no base, as bits of 64-bit words.
    """
    masks, known = _unknown_masks(matrix, members, length, unknown)
    linkable = np.flatnonzero(known >= min_compared)
    del known
    order = linkable[np.lexsort(masks[linkable].T)]
    del linkable
    ordered_masks = masks[order]
    del masks
    starts = np.flatnonzero(
        np.concatenate(
            [[True], np.any(ordered_masks[1:] != ordered_masks[:-1], axis=1)]
        )
    )[: len(order)]
    ends = np.append(starts[1:], len(order))
    return order, starts, ends, ordered_masks[starts]


def _group_pairs(
    group_masks: np.ndarray, length: int, min_compared: int
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Each pair of groups that compares ``min_compared`` positions or more.

    A group pairs with itself and with each later group, on the positions
    where neither holds a character of no base.

    Yields:
        The first group, the second group and the positions compared.
    """
    positions = np.arange(length)
    words = positions >> 6
    shifts = (positions & 63).astype(np.uint64)
    for first_group in range(len(group_masks)):
        unions = group_masks[first_group] | group_masks[first_group:]
        compared = length - np.bitwise_count(unions).sum(axis=1)
        for offset in np.flatnonzero(compared >= min_compared).tolist():
            union = unions[offset]
            columns = np.flatnonzero(((union[words] >> shifts) & np.uint64(1)) == 0)
            yield first_group, first_group + offset, columns


def _partial_pairs(
    matrix: np.ndarray,
    members: np.ndarray,
    length: int,
    limit: int,
    min_compared: int,
    unknown: np.ndarray,
    table: np.ndarray,
    ledger: MemoryLedger,
    found: _Found,
) -> None:
    """Find the pairs of texts of one length that may hold characters of no base.

    Two texts compare the positions where both hold a base. The texts are
    grouped by the positions where they hold none, so that any two texts of
    two groups compare the same positions, where they hold only bases. Each
    pair of groups that compares at least ``min_compared`` positions is then
    searched on those positions as texts without unknown characters are, by
    keys or by loops, the texts of two groups paired across them; a text
    that holds fewer bases than that has no pairs.
    """
    n_words = max(-(-length // 64), 1)
    held = len(members) * (_MASK_TEXT_BYTES + 16 * n_words)
    _hold(ledger, held, "the positions where the texts hold no base")
    order, starts, ends, group_masks = _mask_groups(
        matrix, members, length, unknown, min_compared
    )
    for first_group, second_group, columns in _group_pairs(
        group_masks, length, min_compared
    ):
        rows_below = order[starts[first_group] : ends[first_group]]
        split = 0
        rows = members[rows_below]
        if second_group != first_group:
            split = len(rows_below)
            rows_above = order[starts[second_group] : ends[second_group]]
            rows = np.concatenate([rows, members[rows_above]])
        elif len(rows) < 2:
            continue
        codes = matrix[np.ix_(rows, columns)]
        first, second = _substitution_pairs(codes, limit, table, ledger, found, split)
        del codes
        first, second = _ordered(rows[first], rows[second])
        found.add(first, second)
    ledger.release(held)


def _search_arguments(
    metric: str,
    max_distance: Any,
    unknown: Any,
    min_compared: Any,
) -> tuple[int, int]:
    """Check the arguments that a search and its expectation share.

    Returns:
        The max_distance and the fewest positions compared.
    """
    if metric not in DISTANCE_METRICS:
        raise ValueError(f"metric must be one of {', '.join(DISTANCE_METRICS)}")
    max_distance = integer_argument(max_distance, "max_distance", minimum=0)
    if not isinstance(unknown, str):
        raise TypeError("unknown must be text, the characters that hold no base")
    min_compared = integer_argument(min_compared, "min_compared", minimum=0)
    if (unknown or min_compared) and metric != "hamming":
        raise ValueError(
            "unknown and min_compared apply to metric='hamming' only, which "
            "compares texts position by position"
        )
    return max_distance, min_compared


def _length_groups(
    block_codes: np.ndarray, lengths: np.ndarray, ledger: MemoryLedger
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Order the texts by block and length, so that each group's are a slice.

    Each group of one length and block holds ``_GROUP_BYTES`` of ``ledger``.

    Returns:
        The order of the texts, each group's start, end and length in it, and
        each text's rank among the blocks.
    """
    _distinct, ranks = unique_inverse(block_codes)
    group_keys = ranks * (int(lengths.max()) + 1) + lengths
    order = np.argsort(group_keys, kind="stable")
    ordered = group_keys[order]
    first_of_group = np.concatenate([[True], ordered[1:] != ordered[:-1]])
    n_groups = int(np.count_nonzero(first_of_group))
    _check(ledger, _GROUP_BYTES * n_groups, "the lengths and blocks of the texts")
    _hold(ledger, _GROUP_BYTES * n_groups, "the lengths and blocks of the texts")
    starts = np.flatnonzero(first_of_group)
    del first_of_group
    ends = np.append(starts[1:], len(order))
    return order, starts, ends, lengths[order[starts]], ranks


def _block_spans(
    ranks: np.ndarray, order: np.ndarray, starts: np.ndarray
) -> list[tuple[int, int]]:
    """Each block's first group and the group after its last, in group order."""
    group_ranks = ranks[order[starts]]
    block_starts = np.flatnonzero(
        np.concatenate([[True], group_ranks[1:] != group_ranks[:-1]])
    )
    block_ends = np.append(block_starts[1:], len(starts))
    return list(zip(block_starts.tolist(), block_ends.tolist(), strict=True))


def neighbor_pairs(
    texts: Any,
    metric: str,
    max_distance: int,
    blocks: Any = None,
    *,
    ledger: MemoryLedger | None = None,
    unknown: str = "",
    min_compared: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Find every pair of texts within ``max_distance`` of each other.

    Two texts at positions ``i < j`` form a pair when their distance by
    ``metric`` is at most ``max_distance``; equal texts at two positions form a
    pair at distance 0. Only texts of one block are compared when ``blocks``
    is given.

    Under ``hamming``, ``unknown`` names characters that hold no base, such
    as the ``-`` of a base that no read covered and the ``0`` of a deletion
    in TREX's cloneIDs: a position where either text holds one is not
    compared, so it is never a substitution, and two texts form a pair only
    when at least ``min_compared`` positions are compared, as TREX links
    partial cloneIDs.

    :func:`expected_chance_pairs` gives the pairs that the search would find
    by chance among the pairs it compares.

    Args:
        texts: The texts, a sequence of str, such as the distinct junctions
            of receptor chains or the distinct barcodes of a library.
        metric: ``"hamming"``, ``"normalized_hamming"`` or ``"levenshtein"``.
        max_distance: The largest distance that links two texts: substitutions,
            percent of the length, or edits. 0 links only equal texts, as
            scirpy compares them, even where one substitution is less than
            half a percent of the length.
        blocks: Each text's block as an integer code, such as its gene calls
            or its sample; only texts of one block are compared. None
            compares all.
        ledger: The caller's memory ledger, which the search charges before
            it allocates, and leaves the pairs returned charged to, 16 bytes
            each, for the caller to release: the encoded texts,
            about three times their longest length and 72 bytes more per
            text, with 2 KiB of hash codes for each position of the longest
            and ``NUMPY_STEP_BYTES`` for its sorts and smaller arrays; 48 bytes for each
            length of each block; the variant keys and candidate pairs of one
            pass, 8 bytes each, and 8 more for each candidate within the
            max_distance; and 32 bytes for each pair it finds, counted once for
            each pass of keys that finds it, with 384 for the arrays of each
            length or block that has pairs. With ``unknown`` or
            ``min_compared``, each text of the length searched holds 48 bytes
            more and 16 for each 64 positions. None sets no limit.
        unknown: Characters that hold no base, under ``hamming`` only.
        min_compared: The fewest positions that two texts must compare,
            under ``hamming`` only.

    Returns:
        The positions of the pairs as int64, the first below the second,
        each pair once.

    Raises:
        TypeError: If ``texts`` is not a one-dimensional sequence of str,
            ``blocks`` not one of integer codes, ``max_distance`` not an
            integer, or ``ledger`` not a memory ledger.
        ValueError: If the metric is unknown, ``max_distance`` or
            ``min_compared`` is negative, ``blocks`` does not hold one block
            per text, the texts hold more than 256 distinct characters, or
            ``unknown`` or ``min_compared`` is given with a metric other than
            ``hamming``.
        MemoryBudgetError: If the search would pass the ledger's budget: the
            encoded texts, the keys or candidate pairs of one pass, or the
            pairs within ``max_distance`` need more.
    """
    max_distance, min_compared = _search_arguments(
        metric, max_distance, unknown, min_compared
    )
    values = _texts(texts)
    n_texts = len(values)
    block_codes = _blocks(blocks, n_texts)
    ledger = _ledger(ledger)
    entry = ledger.charged
    matrix, lengths, coding = _encoded(values, ledger)
    partial = bool(unknown) or min_compared > 0
    unknown_codes = _unknown_table(unknown, coding)
    empty = np.empty(0, dtype=np.int64)
    if n_texts < 2:
        del matrix, lengths, coding
        ledger.release(ledger.charged - entry)
        return empty, empty.copy()
    table = _zobrist_table(matrix.shape[1])
    order, starts, ends, group_lengths, ranks = _length_groups(
        block_codes, lengths, ledger
    )
    found = _Found(ledger, [], [])
    # The positions of the pairs among the members of a length or block are
    # replaced in turn by their positions among all texts, so that no copy of
    # them outlives the pairs found.
    if metric == "levenshtein":
        # Texts are at most as many edits apart as the longer is long, so a
        # larger distance links every pair.
        limit = min(max_distance, int(lengths.max()))
        for low, high in _block_spans(ranks, order, starts):
            members = order[starts[low] : ends[high - 1]]
            if len(members) < 2:
                continue
            sizes = ends[low:high] - starts[low:high]
            if _edits_keyed(sizes, group_lengths[low:high], limit):
                first, second = _keyed_pairs(
                    matrix[members], lengths[members], limit, True, table, ledger
                )
                first = members[first]
                second = members[second]
                first, second = _ordered(first, second)
                found.add(first, second)
                continue
            # Each pair of lengths within max_distance, the shorter first.
            for group in range(low, high):
                length = int(group_lengths[group])
                texts_of = order[starts[group] : ends[group]]
                for other in range(group, high):
                    other_length = int(group_lengths[other])
                    if other_length - length > limit:
                        break
                    partners = order[starts[other] : ends[other]]
                    first, second = found.looped(
                        _edit_pairs(
                            matrix[texts_of, :length],
                            lengths[texts_of],
                            matrix[partners, :other_length],
                            lengths[partners],
                            limit,
                            other == group,
                            found.most(),
                        )
                    )
                    first = texts_of[first]
                    second = partners[second]
                    first, second = _ordered(first, second)
                    found.add(first, second)
    else:
        for group in range(len(starts)):
            if ends[group] - starts[group] < 2:
                continue
            members = order[starts[group] : ends[group]]
            length = int(group_lengths[group])
            # A max_distance of 0 links only equal texts, as scirpy's identity.
            limit = (
                max_distance
                if metric == "hamming" or not max_distance
                else allowed_mismatches(length, max_distance)
            )
            limit = min(limit, length)
            if partial:
                if length >= min_compared:
                    _partial_pairs(
                        matrix,
                        members,
                        length,
                        limit,
                        min_compared,
                        unknown_codes,
                        table,
                        ledger,
                        found,
                    )
                continue
            first, second = _substitution_pairs(
                matrix[members, :length], limit, table, ledger, found
            )
            first = members[first]
            second = members[second]
            found.add(first, second)
    if not found.firsts:
        first, second = empty, empty.copy()
    elif len(found.firsts) == 1:
        first, second = found.firsts[0], found.seconds[0]
    else:
        first, second = np.concatenate(found.firsts), np.concatenate(found.seconds)
    del found, matrix, lengths, coding, table, order, starts, ends
    del group_lengths, ranks, unknown_codes, block_codes
    # The texts and the members of the last length or block, a view that
    # keeps the order alive, go before what the search held is let go.
    values = []
    members = texts_of = partners = empty
    # The pairs returned stay charged, for the caller to keep or let go.
    _charge_to(ledger, entry + first.nbytes + second.nbytes, "the pairs found")
    return first, second


def _component_arguments(
    components: Any, priority: Any, n_texts: int
) -> tuple[np.ndarray, np.ndarray]:
    """Each text's component and priority, every text its own without them."""
    if components is None:
        if priority is not None:
            raise ValueError(
                "priority ranks the texts of each component; pass components too"
            )
        return np.arange(n_texts, dtype=np.int64), np.zeros(n_texts, dtype=np.int64)
    codes = np.asarray(components)
    if codes.ndim != 1 or (len(codes) and codes.dtype.kind not in "iu"):
        raise TypeError("components must hold one integer code per text")
    if len(codes) != n_texts:
        raise ValueError("components must hold one component per text")
    if priority is None:
        return codes.astype(np.int64), np.zeros(n_texts, dtype=np.int64)
    ranks = np.asarray(priority)
    if ranks.ndim != 1 or (len(ranks) and ranks.dtype.kind not in "iuf"):
        raise TypeError("priority must hold one number per text")
    if len(ranks) != n_texts:
        raise ValueError("priority must hold one number per text")
    if ranks.dtype.kind == "f" and not np.isfinite(ranks).all():
        raise ValueError("priority must hold finite numbers, not NaN or infinity")
    return codes.astype(np.int64), ranks


def _representatives(
    rows: np.ndarray, components: np.ndarray, priority: np.ndarray
) -> np.ndarray:
    """One row per component among ``rows``: its row of highest priority.

    Of rows of equal priority, the later one represents the component.
    """
    order = rows[np.lexsort((rows, priority[rows], components[rows]))]
    runs = components[order]
    last = np.ones(len(order), dtype=bool)
    last[:-1] = runs[1:] != runs[:-1]
    return order[last]


def _library_counts(
    matrix: np.ndarray,
    rows: np.ndarray,
    length: int,
    components: np.ndarray,
    priority: np.ndarray,
    unknown: np.ndarray,
) -> np.ndarray:
    """Each position's count of each character among one text per component.

    At each position, each component of ``rows`` gives the character of its
    text of highest priority that holds a base there, so a component whose
    text of highest priority was cut short still counts where others hold
    bases.
    """
    # Components in a row, the text of highest priority first and of equal
    # priorities the later.
    order = rows[np.lexsort((rows, priority[rows], components[rows]))][::-1]
    runs = components[order]
    first_of_run = np.ones(len(order), dtype=bool)
    first_of_run[1:] = runs[1:] != runs[:-1]
    run_of = np.cumsum(first_of_run) - 1
    del runs, first_of_run
    counts = np.zeros((length, 256), dtype=np.float64)
    for position in range(length):
        codes = matrix[order, position]
        holding = np.flatnonzero(~unknown[codes])
        if len(holding):
            first = np.ones(len(holding), dtype=bool)
            first[1:] = run_of[holding][1:] != run_of[holding][:-1]
            counts[position] = np.bincount(codes[holding[first]], minlength=256)
    return counts


def _partial_chance_pairs(
    matrix: np.ndarray,
    members: np.ndarray,
    length: int,
    limit: int,
    min_compared: int,
    unknown: np.ndarray,
    components: np.ndarray,
    priority: np.ndarray,
    ledger: MemoryLedger,
) -> float:
    """The pairs of texts of one length expected within ``limit`` by chance.

    The texts are grouped as :func:`_partial_pairs` groups them, and each
    pair of groups that compares ``min_compared`` positions or more adds its
    pairs times their chance on those positions. Each position agrees with
    the share of pairs of components that hold one character there, of one
    text per component of the grouped texts; where fewer than two
    components hold a base, every text that holds one is of one component,
    so a pair of groups that compares that position expects nothing.
    """
    n_words = max(-(-length // 64), 1)
    held = len(members) * (_MASK_TEXT_BYTES + 16 * n_words)
    _hold(ledger, held, "the positions where the texts hold no base")
    order, starts, ends, group_masks = _mask_groups(
        matrix, members, length, unknown, min_compared
    )
    _check(
        ledger,
        _LIBRARY_ROW_BYTES * len(order) + 2 * _FREQUENCY_BYTES * length,
        "the characters of one text per component",
    )
    counts = _library_counts(
        matrix, members[order], length, components, priority, unknown
    )
    # Each position's chance that the texts of two components agree there,
    # held for every pair of groups with the positions that two or more
    # components hold, once the counts are let go.
    holding = counts.sum(axis=1)
    held_apart = holding >= 2
    agreement = np.zeros(length, dtype=np.float64)
    for position in np.flatnonzero(held_apart).tolist():
        tally = counts[position]
        agreement[position] = float(tally @ (tally - 1.0)) / (
            holding[position] * (holding[position] - 1.0)
        )
    del counts, holding
    _hold(ledger, (_AGREEMENT_BYTES + 1) * length, "the agreement of the positions")
    expected = 0.0
    for first_group, second_group, columns in _group_pairs(
        group_masks, length, min_compared
    ):
        if not held_apart[columns].all():
            continue
        n_first = int(ends[first_group] - starts[first_group])
        if second_group == first_group:
            if n_first < 2:
                continue
            split, n_texts = 0, n_first
        else:
            split = n_first
            n_texts = n_first + int(ends[second_group] - starts[second_group])
        chances = _mismatch_chances(1.0 - agreement[columns], min(limit, len(columns)))
        expected += _chance_pairs(n_texts, split, chances)
    ledger.release(held + (_AGREEMENT_BYTES + 1) * length)
    return expected


def expected_chance_pairs(
    texts: Any,
    metric: str,
    max_distance: int,
    blocks: Any = None,
    *,
    components: Any = None,
    priority: Any = None,
    ledger: MemoryLedger | None = None,
    unknown: str = "",
    min_compared: int = 0,
) -> float:
    """The pairs of texts expected within ``max_distance`` of each other by chance.

    Of the pairs that :func:`neighbor_pairs` compares with the same texts,
    metric, max_distance, blocks, ``unknown`` and ``min_compared``, these are the
    pairs expected within ``max_distance`` if each text's character at each
    position were drawn independently from the characters that one text per
    component holds there. ``components`` gives each text's component, such
    as the barcodes that correction linked, and ``priority`` ranks the texts
    of a component, so that its text of highest priority represents it, such
    as the barcode that it merges into; without them, every text is its own
    component. Error variants of one barcode are not independent draws, so
    one text per component keeps a few abundant barcodes and their errors
    from weighting the frequencies, while the representatives keep the real
    biases of each position, such as positions that every barcode fixes.

    Two texts agree at a position with the chance that the representatives
    of two components do: the share of pairs of them, among the texts of
    each length of a block, that hold one character there. A text never
    pairs with itself here, which would raise the chance by about its
    complement over the components. With ``unknown``, each position takes
    the character of each component's text of highest priority that holds a
    base there. Where fewer than two components give a set of texts its
    characters, no two of its texts are of two components, and the set
    expects nothing: the texts of one length and block, a pair of groups of
    texts that compares a position that fewer than two components hold, or
    under ``levenshtein`` a pair of lengths whose representatives are of one
    component. Pairs are counted over every text compared: each pair of the
    texts of one length and block or, with ``unknown``, of two groups of
    texts on the positions both hold, as the search compares them. Two
    texts of one length are distinct, so a pair of them that would agree
    everywhere counts as one that does not.

    Under ``hamming`` and ``normalized_hamming``, the chance of each number of
    mismatches follows exactly from the chance that two texts agree at each
    position. Under ``levenshtein``, each pair of lengths of a block follows
    the edit
    distance's dynamic programming over the chance that each pair of
    positions agrees, treating the pairs of positions as independent, an
    approximation that comes within a tenth of exact counts on random texts
    of 6 to 20 characters of about equal frequency and overstates them where
    one character dominates, more so for longer texts and larger distances.

    Args:
        texts: The texts, a sequence of str, as :func:`neighbor_pairs` takes
            them.
        metric: ``"hamming"``, ``"normalized_hamming"`` or ``"levenshtein"``.
        max_distance: The largest distance that links two texts.
        blocks: Each text's block as an integer code, or None.
        components: Each text's component as an integer code, or None.
        priority: Each text's rank within its component, a finite number,
            higher first, of equal ranks the later text first, or None for
            ranks by position.
        ledger: The caller's memory ledger, which the expectation charges
            before it allocates and lets go of on return: the encoded texts, as
            :func:`neighbor_pairs` counts them, and 48 bytes for each length
            of each block; while one text per component of a length is
            chosen, 57 bytes and its length for each text, with 4 KiB; with
            ``unknown``, the positions of no base as the search counts them,
            64 bytes for each text that holds enough bases and 4 KiB for
            each position while the characters are counted, and 9 bytes for
            each position after; under ``levenshtein``, 48 bytes for each
            text of a block while one per component is chosen, 2 KiB for
            each position of each length of the block, 8 bytes for each pair
            of positions of its longest length, for each text chosen of its
            most numerous length 9 bytes and 3 for each position of its
            longest, with 2 KiB, and 160 bytes for each state of the edit
            distances followed, with 32 KiB more. None sets no limit.
        unknown: Characters that hold no base, under ``hamming`` only.
        min_compared: The fewest positions that two texts must compare,
            under ``hamming`` only.

    Returns:
        The expected pairs, a float.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range as :func:`neighbor_pairs`
            refuses it, ``components`` or ``priority`` does not hold one value
            per text, ``priority`` holds a number that is not finite or comes
            without ``components``, or ``max_distance`` under ``levenshtein`` is
            above ``MAX_CHANCE_EDITS``.
        MemoryBudgetError: If it would pass the ledger's budget.
    """
    max_distance, min_compared = _search_arguments(
        metric, max_distance, unknown, min_compared
    )
    if metric == "levenshtein" and max_distance > MAX_CHANCE_EDITS:
        raise ValueError(
            f"The pairs expected by chance within more than {MAX_CHANCE_EDITS} "
            "edits take too long to count, since their cost grows about eightfold "
            f"with every two edits; pass a max_distance of at most {MAX_CHANCE_EDITS} "
            "with metric='levenshtein', or use metric='hamming', whose chance "
            "pairs are counted for any max_distance"
        )
    values = _texts(texts)
    n_texts = len(values)
    block_codes = _blocks(blocks, n_texts)
    component_codes, ranks_of = _component_arguments(components, priority, n_texts)
    ledger = _ledger(ledger)
    entry = ledger.charged
    try:
        return _expected_pairs(
            values,
            metric,
            max_distance,
            block_codes,
            component_codes,
            ranks_of,
            ledger,
            unknown,
            min_compared,
        )
    finally:
        # What the expectation held is let go with its frames.
        ledger.release(ledger.charged - entry)


def _expected_pairs(
    values: list[str],
    metric: str,
    max_distance: int,
    block_codes: np.ndarray,
    component_codes: np.ndarray,
    ranks_of: np.ndarray,
    ledger: MemoryLedger,
    unknown: str,
    min_compared: int,
) -> float:
    """The pairs expected by chance, of :func:`expected_chance_pairs`'s checked arguments."""
    n_texts = len(values)
    matrix, lengths, coding = _encoded(values, ledger)
    if n_texts < 2:
        return 0.0
    unknown_codes = _unknown_table(unknown, coding)
    partial = bool(unknown) or min_compared > 0
    order, starts, ends, group_lengths, block_ranks = _length_groups(
        block_codes, lengths, ledger
    )
    expected = 0.0
    if metric == "levenshtein":
        limit = min(max_distance, int(lengths.max()))
        for low, high in _block_spans(block_ranks, order, starts):
            members = order[starts[low] : ends[high - 1]]
            if len(members) < 2:
                continue
            groups = [order[starts[group] : ends[group]] for group in range(low, high)]
            _check(
                ledger,
                _REPRESENTATIVE_ROW_BYTES * len(members),
                "the representatives of the components",
            )
            chosen = [
                _representatives(rows, component_codes, ranks_of) for rows in groups
            ]
            expected += _edit_chance_pairs(
                matrix,
                groups,
                chosen,
                group_lengths[low:high].tolist(),
                limit,
                component_codes,
                ledger,
            )
        return expected
    for group in range(len(starts)):
        if ends[group] - starts[group] < 2:
            continue
        members = order[starts[group] : ends[group]]
        length = int(group_lengths[group])
        limit = (
            max_distance
            if metric == "hamming" or not max_distance
            else allowed_mismatches(length, max_distance)
        )
        limit = min(limit, length)
        if partial:
            if length >= min_compared:
                expected += _partial_chance_pairs(
                    matrix,
                    members,
                    length,
                    limit,
                    min_compared,
                    unknown_codes,
                    component_codes,
                    ranks_of,
                    ledger,
                )
            continue
        # The representatives, their characters and the count of one position
        # with its copy less one.
        _check(
            ledger,
            (_REPRESENTATIVE_ROW_BYTES + length + _COUNTED_CODE_BYTES) * len(members)
            + 2 * _FREQUENCY_BYTES,
            "the representatives of the components",
        )
        representatives = _representatives(members, component_codes, ranks_of)
        if len(representatives) < 2:
            continue
        agree = _component_agreement(matrix[representatives, :length])
        expected += _chance_pairs(
            len(members), 0, _mismatch_chances(1.0 - agree, limit)
        )
    return expected


def _ordered(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The positions of pairs, the lower of each first; ``second`` is reused.

    The caller rebinds both names to the result, so that ``first`` is freed.
    """
    low = np.minimum(first, second)
    np.maximum(first, second, out=second)
    return low, second
