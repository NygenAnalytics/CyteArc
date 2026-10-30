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
pair within the distance and holds at most ``max_bytes`` while it searches:
the encoded texts, the variant keys and candidate pairs of one pass, and the
pairs it finds. A search that would need more raises :class:`MemoryError`
before it allocates.

Every pair is found in one of two ways, whichever is estimated to take less
time for the texts of each length, or under ``levenshtein`` for those of each
block. Compiled loops compare every pair of one length, or of lengths within
the distance, and stop at the first position where the distance is exceeded.
Or, for distances up to 3, each text yields variant keys that any two texts
within the distance share, and only the texts that share a key are compared.
These functions work on text and never touch storage.
"""

from dataclasses import dataclass
from math import comb
from typing import Any

import numpy as np

from ..utils.arguments import integer_argument
from ..utils.arrays import sorted_unique, unique_inverse
from ..utils.numba import njit

__all__ = ["DISTANCE_METRICS", "allowed_mismatches", "neighbor_pairs"]

#: The metrics of :func:`neighbor_pairs`.
DISTANCE_METRICS: tuple[str, ...] = ("hamming", "normalized_hamming", "levenshtein")
# The bytes that each pair found holds until the search returns: its two
# positions, and their copy when the pairs of all lengths are joined.
_FOUND_PAIR_BYTES = 32
# The bytes of each text beside its characters while the search runs: its
# place in the list of texts, its length, block, rank among the blocks, group
# key and position in the order of blocks and lengths, and the sorted keys.
_TEXT_BYTES = 64
# The bytes of a pair while a compiled loop collects it: two buffers that
# double as they fill, and the copies returned.
_LOOP_PAIR_BYTES = 32
# The bytes of a variant key, and of a candidate pair of texts that share one.
_KEY_BYTES = 8
_CANDIDATE_BYTES = 8
# The pairs that a pass of variant keys keeps hold their codes until the
# passes are joined, which then hold about this many bytes per pair: the
# joined codes, their sorted copy and the two positions of each pair.
_KEPT_PAIR_BYTES = 8
_JOINED_PAIR_BYTES = 40
# Variant keys take at most this share of the bytes a search has left for a
# pass, so that the candidate pairs of the pass fit beside them, but a pass
# holds at least _MIN_KEYS_PER_PASS keys, 8 MiB: a budget too small for them
# fails rather than hashing every text again in thousands of passes.
_KEY_SHARE = 4
_MIN_KEYS_PER_PASS = 1 << 20
# The room of a search without a limit: more bytes than any machine holds.
_UNLIMITED = 2**62


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


@dataclass(slots=True)
class _Memory:
    """The bytes that a search holds and the most it may hold, None for no limit."""

    limit: int | None
    held: int = 0

    @property
    def room(self) -> int:
        """The bytes the search may still allocate."""
        if self.limit is None:
            return _UNLIMITED
        return max(0, self.limit - self.held)

    def check(self, needed: int, what: str) -> None:
        """Raise MemoryError if ``needed`` more bytes would exceed the limit."""
        if self.limit is not None and self.held + needed > self.limit:
            self.exceeded(needed, what)

    def exceeded(self, needed: int, what: str) -> None:
        """Raise MemoryError for ``needed`` more bytes than the search may hold."""
        allowed = (
            "more than one search can count"
            if self.limit is None
            else f"more than the {self.limit} bytes it may hold"
        )
        raise MemoryError(
            "Finding the pairs of texts within the cutoff needs about "
            f"{self.held + needed} bytes for {what}, {allowed}. Lower the "
            "cutoff, compare fewer texts at once or within smaller blocks, or "
            "raise the memory budget."
        )


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
    words: np.ndarray, limit: int, most: int
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Every pair of rows within ``limit`` substitutions, the first below the second.

    The rows are texts of one length, eight codes to a word. The bytes that
    differ in two words are the nonzero bytes of their exclusive or, which
    folding each byte onto its lowest bit and summing those bits counts.
    Returns False, with no pairs, once more than ``most`` pairs are found.
    """
    rows, n_words = words.shape
    ones = np.uint64(0x0101010101010101)
    start = min(max(rows, 16), most)
    firsts = np.empty(start, dtype=np.int64)
    seconds = np.empty(start, dtype=np.int64)
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
                    if at == most:
                        return firsts[:0].copy(), seconds[:0].copy(), False
                    firsts = _grown(firsts, most)
                    seconds = _grown(seconds, most)
                firsts[at] = first
                seconds[at] = second
                at += 1
    return firsts[:at].copy(), seconds[:at].copy(), True


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
) -> int:
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
def _run_pairs(keys: np.ndarray, bits: int, n_rows: int, out: np.ndarray) -> int:
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
    memory: _Memory,
) -> tuple[np.ndarray, np.ndarray]:
    """Every pair within ``limit`` through shared variant keys, verified.

    A key holds its hash with the row in the low bits, so one sort in place
    brings the rows of each hash together, ascending. Keys are split by their
    hash into passes of at most ``_KEYS_PER_PASS``, or fewer when the memory
    left is smaller, so the keys held at once stay bounded however many the
    texts yield: one pass over the texts counts the keys of every pass, and
    each pass hashes them again and keeps its own. The keys and candidate
    pairs of a pass, and the pairs kept, are checked against ``memory``
    before they are allocated.
    """
    n_rows = len(lengths)
    bits = max(int(n_rows - 1).bit_length(), 1)
    per_length = np.bincount(lengths)
    total = sum(
        int(texts) * _key_count(length, limit, deletions)
        for length, texts in enumerate(per_length.tolist())
        if texts
    )
    keys_per_pass = min(
        _KEYS_PER_PASS,
        max(_MIN_KEYS_PER_PASS, memory.room // (_KEY_SHARE * _KEY_BYTES)),
    )
    n_parts = max(1, -(-total // keys_per_pass))
    no_keys = np.empty(0, dtype=np.uint64)
    no_counts = np.empty(0, dtype=np.int64)
    entry = memory.held
    # The scratch space of one text's keys, held while the search runs.
    scratch = _key_count(len(per_length) - 1, limit, deletions)
    memory.check(_KEY_BYTES * scratch, "the variant keys of one text")
    hashes = np.empty(scratch, dtype=np.uint64)
    memory.held += _KEY_BYTES * scratch
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
    held = memory.held
    found = []
    n_kept = 0
    for part in range(n_parts):
        size = int(sizes[part])
        memory.check(_KEY_BYTES * size, "the variant keys of one pass")
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
        )
        keys = keys[:written]
        keys.sort()
        n_candidates = _run_pairs(keys, bits, n_rows, no_counts)
        memory.check(
            _KEY_BYTES * written + _CANDIDATE_BYTES * n_candidates,
            "the candidate pairs of texts that share variant keys",
        )
        pairs = np.empty(n_candidates, dtype=np.int64)
        _run_pairs(keys, bits, n_rows, pairs)
        del keys
        pairs.sort()
        kept = _check_pairs(codes, lengths, pairs, n_rows, limit, deletions)
        found.append(pairs[:kept].copy())
        del pairs
        n_kept += kept
        memory.held = held + _KEPT_PAIR_BYTES * n_kept
    memory.check(_JOINED_PAIR_BYTES * n_kept, "the pairs found")
    if n_parts == 1:
        near = found[0]
    else:
        # A pair whose texts share keys in several passes is found in each.
        merged = np.concatenate(found)
        found.clear()
        near = sorted_unique(merged)
        del merged
    memory.held = entry
    return np.divmod(near, n_rows)


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


def _encoded(texts: list[str], memory: _Memory) -> tuple[np.ndarray, np.ndarray]:
    """Encode texts as a padded matrix of character codes and their lengths.

    A character below 256, such as an ASCII letter, is coded by its code
    point. Texts with other characters code each distinct character by its
    rank instead, which allows 256 of them. The memory the encoding needs is
    checked before it is allocated, and the matrix and the per-text arrays of
    the search are held.
    """
    lengths = np.fromiter(map(len, texts), dtype=np.int64, count=len(texts))
    width = max(int(lengths.max()) if len(texts) else 0, 1)
    characters = int(lengths.sum())
    # The matrix, and two copies of the rows of one length or block while
    # they are searched, such as the rows and their words or the rows of two
    # lengths, and the codes that hash the keys, 257 for each position, held
    # until the search returns.
    held = 3 * len(texts) * width + _TEXT_BYTES * len(texts) + _table_bytes(width)
    what = "the encoded texts"
    # Joined, the texts take a byte per character, and their codes another.
    memory.check(held + 2 * characters, what)
    joined = "".join(texts)
    try:
        points = np.frombuffer(joined.encode("latin-1"), dtype=np.uint8)
    except UnicodeEncodeError:
        # Four bytes per character joined and as code points, and about 32
        # while the characters are ranked.
        memory.check(held + 40 * characters, what)
        wide = np.frombuffer(joined.encode("utf-32-le"), dtype=np.uint32)
        distinct, ranks = unique_inverse(wide)
        if len(distinct) > 256:
            raise ValueError("texts can hold at most 256 distinct characters") from None
        points = ranks.astype(np.uint8)
    del joined
    matrix = np.zeros((len(texts), width), dtype=np.uint8)
    matrix[np.arange(width) < lengths[:, None]] = points
    memory.held += held
    return matrix, lengths


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


@dataclass(slots=True)
class _Found:
    """The pairs found so far, each holding ``_FOUND_PAIR_BYTES`` of ``memory``."""

    memory: _Memory
    firsts: list[np.ndarray]
    seconds: list[np.ndarray]

    def most(self) -> int:
        """The most pairs that a compiled loop may collect in the memory left."""
        return self.memory.room // _LOOP_PAIR_BYTES

    def add(self, first: np.ndarray, second: np.ndarray) -> None:
        self.memory.held += _FOUND_PAIR_BYTES * len(first)
        self.firsts.append(first)
        self.seconds.append(second)

    def looped(
        self, found: tuple[np.ndarray, np.ndarray, bool]
    ) -> tuple[np.ndarray, np.ndarray]:
        """The pairs of a compiled loop; MemoryError if they outgrew the memory left."""
        first, second, complete = found
        if not complete:
            self.memory.exceeded(
                _LOOP_PAIR_BYTES * (self.most() + 1), "the pairs found"
            )
        return first, second


def neighbor_pairs(
    texts: Any,
    metric: str,
    cutoff: int,
    blocks: Any = None,
    *,
    max_bytes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Find every pair of texts within ``cutoff`` of each other.

    Two texts at positions ``i < j`` form a pair when their distance by
    ``metric`` is at most ``cutoff``; equal texts at two positions form a
    pair at distance 0. Only texts of one block are compared when ``blocks``
    is given.

    Args:
        texts: The texts, a sequence of str, such as the distinct junctions
            of receptor chains or the distinct barcodes of a library.
        metric: ``"hamming"``, ``"normalized_hamming"`` or ``"levenshtein"``.
        cutoff: The largest distance that links two texts: substitutions,
            percent of the length, or edits.
        blocks: Each text's block as an integer code, such as its gene calls
            or its sample; only texts of one block are compared. None
            compares all.
        max_bytes: The most bytes the search may hold: the encoded texts,
            about three times their longest length and 64 bytes more per
            text, with 2 KiB of hash codes for each position of the longest,
            the variant keys and candidate pairs of one pass, 8 bytes each,
            and 32 bytes for each pair it finds. None sets no limit.

    Returns:
        The positions of the pairs as int64, the first below the second,
        each pair once.

    Raises:
        TypeError: If ``texts`` is not a one-dimensional sequence of str,
            ``blocks`` not one of integer codes, or ``max_bytes`` not an
            integer.
        ValueError: If the metric is unknown, the cutoff or ``max_bytes`` is
            negative, ``blocks`` does not hold one block per text, or the
            texts hold more than 256 distinct characters.
        MemoryError: If the search would hold more than ``max_bytes``: the
            encoded texts, the keys or candidate pairs of one pass, or the
            pairs within the cutoff need more.
    """
    if metric not in DISTANCE_METRICS:
        raise ValueError(f"metric must be one of {', '.join(DISTANCE_METRICS)}")
    if (
        isinstance(cutoff, bool)
        or not isinstance(cutoff, int | np.integer)
        or cutoff < 0
    ):
        raise ValueError("cutoff must be a non-negative integer")
    limit_bytes = (
        None
        if max_bytes is None
        else integer_argument(max_bytes, "max_bytes", minimum=0)
    )
    values = _texts(texts)
    n_texts = len(values)
    block_codes = _blocks(blocks, n_texts)
    memory = _Memory(limit_bytes)
    matrix, lengths = _encoded(values, memory)
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
    found = _Found(memory, [], [])
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
                    matrix[members], lengths[members], limit, True, table, memory
                )
                found.add(*_ordered(members[first], members[second]))
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
                    found.add(*_ordered(texts_of[first], partners[second]))
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
                    codes, lengths[members], limit, False, table, memory
                )
            else:
                first, second = found.looped(
                    _hamming_pairs(_words(codes), limit, found.most())
                )
            found.add(members[first], members[second])
    if not found.firsts:
        return empty, empty.copy()
    if len(found.firsts) == 1:
        return found.firsts[0], found.seconds[0]
    return np.concatenate(found.firsts), np.concatenate(found.seconds)


def _ordered(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The positions of pairs, the lower of each first; ``second`` is reused."""
    low = np.minimum(first, second)
    np.maximum(first, second, out=second)
    return low, second
