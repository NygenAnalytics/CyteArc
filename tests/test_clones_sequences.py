"""Pairs of texts within a distance, checked against every pair compared directly."""

import itertools
import warnings

import numpy as np
import pytest

from cytearc.clones import sequences
from cytearc.clones.sequences import (
    DISTANCE_METRICS,
    allowed_mismatches,
    expected_chance_pairs,
    neighbor_pairs,
)
from cytearc.repertoire.clonotypes import METRICS
from tests.clone_helpers import levenshtein_distance as _levenshtein
from tests.clone_helpers import link_count_variance, smallest_budget
from tests.perturbation_helpers import traced_peak
from tests.clone_helpers import within_distance as _within


def _pairs(first: np.ndarray, second: np.ndarray) -> set[tuple[int, int]]:
    """The pairs that a search returned, checked to be distinct and ordered."""
    assert first.dtype == second.dtype == np.int64
    assert (first < second).all()
    pairs = set(zip(first.tolist(), second.tolist(), strict=True))
    assert len(pairs) == len(first)
    return pairs


def _expected(texts, metric, cutoff, blocks=None) -> set[tuple[int, int]]:
    return {
        (i, j)
        for i, j in itertools.combinations(range(len(texts)), 2)
        if (blocks is None or blocks[i] == blocks[j])
        and _within(texts[i], texts[j], metric, cutoff)
    }


def _force(monkeypatch, path: str) -> None:
    """Make the search use keys, or compare pairs, wherever it can.

    Forced keys under substitutions mask single positions, the keys whose
    costs the tests of the budget count, up to 3 substitutions; beyond them,
    they keep one of one more segment than the substitutions.
    """
    names = (
        ("_KEY_NS", "_EDIT_KEY_NS")
        if path == "keyed"
        else ("_PAIR_NS", "_EDIT_PAIR_NS", "_EDIT_CUTOFF_NS")
    )
    for name in names:
        monkeypatch.setattr(sequences, name, 0.0)
    if path == "keyed":
        monkeypatch.setattr(
            sequences,
            "_hamming_segments",
            lambda codes, limit, room, split=0: (
                max(codes.shape[1], 1)
                if limit <= 3
                else (limit + 1 if limit < codes.shape[1] else 0)
            ),
        )


def _families(rng, alphabet: str, n_bases: int, high: int) -> list[str]:
    """Texts of up to ``high`` letters, each with variants by one to three edits."""
    letters = list(alphabet)
    texts = []
    for _ in range(n_bases):
        base = "".join(rng.choice(letters, size=int(rng.integers(0, high + 1))))
        texts.append(base)
        for _ in range(int(rng.integers(0, 4))):
            text = list(base)
            for _ in range(int(rng.integers(1, 4))):
                position = int(rng.integers(0, len(text) + 1))
                letter = str(rng.choice(letters))
                kind = int(rng.integers(0, 3))
                if kind == 0 and position < len(text):
                    text[position] = letter
                elif kind == 1:
                    text.insert(position, letter)
                elif position < len(text):
                    del text[position]
            texts.append("".join(text))
    return texts


@pytest.mark.parametrize(
    ("metric", "cutoff"),
    [
        ("hamming", 0),
        ("hamming", 1),
        ("hamming", 2),
        ("hamming", 3),
        ("hamming", 5),
        ("levenshtein", 1),
        ("levenshtein", 2),
        ("levenshtein", 3),
        ("levenshtein", 4),
        ("normalized_hamming", 20),
        ("normalized_hamming", 40),
    ],
)
def test_neighbor_pairs_match_every_pair(metric, cutoff) -> None:
    rng = np.random.default_rng(cutoff)
    texts = sorted(
        {
            "".join(rng.choice(list("ACG"), size=int(rng.integers(1, 10))))
            for _ in range(150)
        }
    )
    blocks = rng.integers(0, 2, size=len(texts))
    for chosen in (None, blocks):
        first, second = neighbor_pairs(texts, metric, cutoff, blocks=chosen)
        assert (first < second).all()
        got = set(zip(first.tolist(), second.tolist(), strict=True))
        want = {
            (i, j)
            for i, j in itertools.combinations(range(len(texts)), 2)
            if (chosen is None or chosen[i] == chosen[j])
            and _within(texts[i], texts[j], metric, cutoff)
        }
        assert got == want


def test_long_texts_compare_every_pair_of_a_length() -> None:
    # Texts whose variant keys would be too many are compared pair by pair.
    rng = np.random.default_rng(7)
    base = "".join(rng.choice(list("ACGT"), size=60))
    texts = [base]
    for count in (5, 6, 7, 12):
        changed = list(base)
        for position in rng.choice(60, size=count, replace=False):
            changed[position] = "A" if changed[position] != "A" else "C"
        texts.append("".join(changed))
    first, second = neighbor_pairs(texts, "normalized_hamming", 10)
    pairs = set(zip(first.tolist(), second.tolist(), strict=True))
    assert {(0, 1), (0, 2)} <= pairs and (0, 4) not in pairs
    first, second = neighbor_pairs(texts[:2] + [base[:-2] + "GG"], "levenshtein", 6)
    assert (0, 1) in set(zip(first.tolist(), second.tolist(), strict=True))


def test_allowed_mismatches_round_half_up() -> None:
    assert [allowed_mismatches(length, 15) for length in (30, 33, 36, 45, 60)] == [
        4,
        5,
        5,
        6,
        9,
    ]
    # 1 of 50 is exactly 2%; 1 of 49 rounds to 2%.
    assert [allowed_mismatches(length, 2) for length in (10, 49, 50, 75)] == [
        0,
        1,
        1,
        1,
    ]
    assert allowed_mismatches(4, 100) == 4
    # 1 of 40 is 2.5% and 1 of 8 is 12.5%, which round up to 3% and 13%, so
    # no substitution is allowed; rounding half to even would allow one.
    assert allowed_mismatches(40, 2) == 0
    assert allowed_mismatches(8, 12) == 0
    # 3 of 8 is 37.5%, which rounds up to 38%.
    assert allowed_mismatches(8, 37) == 2
    assert allowed_mismatches(8, 38) == 3


@pytest.mark.parametrize("path", ["keyed", "pairwise"])
def test_a_percentage_of_one_half_rounds_up(monkeypatch, path) -> None:
    # 1 of 8 letters is 12.5% and 1 of 40 is 2.5%: within 13% and 3%, but
    # not within 12% and 2%, as rounding half to even would have them.
    _force(monkeypatch, path)
    for length, below in ((8, 12), (40, 2)):
        texts = ["A" * length, "A" * (length - 1) + "C"]
        found = neighbor_pairs(texts, "normalized_hamming", below)
        assert _pairs(*found) == set(), length
        found = neighbor_pairs(texts, "normalized_hamming", below + 1)
        assert _pairs(*found) == {(0, 1)}, length


@pytest.mark.parametrize("path", ["keyed", "pairwise"])
def test_a_cutoff_of_zero_links_only_equal_texts(monkeypatch, path) -> None:
    # One substitution in 201 letters is 0.498%, which rounds to 0%, but
    # scirpy compares identity at a cutoff of 0.
    long = "A" * 201
    texts = [long, long[:-1] + "C", long]
    found = neighbor_pairs(texts, "normalized_hamming", 1)
    assert _pairs(*found) == {(0, 1), (0, 2), (1, 2)}
    _force(monkeypatch, path)
    assert _pairs(*neighbor_pairs(texts, "normalized_hamming", 0)) == {(0, 2)}


def test_neighbor_pairs_check_their_arguments() -> None:
    with pytest.raises(ValueError, match="metric must be one of"):
        neighbor_pairs(["A"], "identity", 0)
    with pytest.raises(ValueError, match="cutoff must be at least 0"):
        neighbor_pairs(["A"], "hamming", -1)
    for cutoff in (1.5, 1.0, True, "1"):
        with pytest.raises(TypeError, match="cutoff must be an integer"):
            neighbor_pairs(["A"], "hamming", cutoff)
    with pytest.raises(ValueError, match="blocks must hold one block per text"):
        neighbor_pairs(["A", "C"], "hamming", 1, blocks=[0])
    first, second = neighbor_pairs([], "levenshtein", 1)
    assert len(first) == len(second) == 0
    assert METRICS[0] == "identity"


def test_a_cutoff_beyond_every_length_links_every_pair() -> None:
    # Two texts are at most as many edits apart as the longer is long, and
    # texts of one length differ in at most as many letters, whatever the
    # cutoff beyond that, even past 64 bits.
    texts = ["CASSF", "", "GG", "CAGSFW", "CASSF"]
    every = set(itertools.combinations(range(len(texts)), 2))
    for cutoff in (6, 2**40, 2**63 - 1, 2**63, 2**64):
        found = neighbor_pairs(texts, "levenshtein", cutoff)
        assert _pairs(*found) == every
        found = neighbor_pairs(texts, "hamming", cutoff)
        assert _pairs(*found) == {(0, 4)}


@pytest.mark.parametrize(
    ("metric", "cutoff"), [("hamming", 1), ("hamming", 2), ("levenshtein", 2)]
)
def test_neighbor_pairs_split_keys_into_passes(monkeypatch, metric, cutoff) -> None:
    rng = np.random.default_rng(3)
    texts = sorted(
        {
            "".join(rng.choice(list("ACG"), size=int(rng.integers(3, 9))))
            for _ in range(120)
        }
    )
    # A small pass splits the keys by their hash into many passes.
    _force(monkeypatch, "keyed")
    monkeypatch.setattr(sequences, "_KEYS_PER_PASS", 97)
    first, second = neighbor_pairs(texts, metric, cutoff)
    assert _pairs(first, second) == _expected(texts, metric, cutoff)


@pytest.mark.parametrize("path", ["keyed", "pairwise"])
@pytest.mark.parametrize("alphabet", ["AC", "ACGT", "ACDEFGHIKLMNPQRSTVWY"])
def test_keys_and_pairwise_comparisons_find_every_pair(
    monkeypatch, path, alphabet
) -> None:
    # Either search, wherever it applies, finds every pair within each cutoff:
    # of texts of 0 to 12 letters and their variants, with runs of one letter
    # whose deletions coincide, a repeated text, and blocks.
    _force(monkeypatch, path)
    rng = np.random.default_rng(len(alphabet))
    texts = _families(rng, alphabet, 20, 12) + ["AAAA", "AAAA", "AAA", "AACAA", "A"]
    blocks = rng.integers(0, 2, size=len(texts))
    pairs = list(itertools.combinations(range(len(texts)), 2))
    edits = {(i, j): _levenshtein(texts[i], texts[j]) for i, j in pairs}
    same_length = [(i, j) for i, j in pairs if len(texts[i]) == len(texts[j])]
    for metric, cutoffs in (
        ("hamming", (0, 1, 2, 3)),
        ("levenshtein", (0, 1, 2, 3)),
        ("normalized_hamming", (0, 25, 50)),
    ):
        for cutoff in cutoffs:
            if metric == "levenshtein":
                near = {pair for pair, edit in edits.items() if edit <= cutoff}
            else:
                near = {
                    (i, j)
                    for i, j in same_length
                    if _within(texts[i], texts[j], metric, cutoff)
                }
            for chosen in (None, blocks):
                first, second = neighbor_pairs(texts, metric, cutoff, blocks=chosen)
                want = {
                    (i, j) for i, j in near if chosen is None or chosen[i] == chosen[j]
                }
                assert _pairs(first, second) == want, (metric, cutoff, chosen)


@pytest.mark.parametrize("codes", ["zero", "four values"])
@pytest.mark.parametrize(
    ("metric", "cutoff"), [("hamming", 1), ("hamming", 3), ("levenshtein", 2)]
)
def test_texts_whose_keys_share_a_hash_by_chance_are_compared(
    monkeypatch, codes, metric, cutoff
) -> None:
    # With every code zero, or drawn from four values, most keys share a hash
    # with keys of other variants; those texts are compared and left out, and
    # most passes stay empty.
    values = np.random.default_rng(0).integers(0, 2**63, size=4).astype(np.uint64)

    def table(width: int) -> np.ndarray:
        if codes == "zero":
            return np.zeros((width, 257), dtype=np.uint64)
        return values[np.random.default_rng(1).integers(0, 4, size=(width, 257))]

    monkeypatch.setattr(sequences, "_zobrist_table", table)
    monkeypatch.setattr(sequences, "_KEYS_PER_PASS", 500)
    _force(monkeypatch, "keyed")
    rng = np.random.default_rng(4)
    texts = sorted(set(_families(rng, "ACG", 30, 8)))
    first, second = neighbor_pairs(texts, metric, cutoff)
    assert _pairs(first, second) == _expected(texts, metric, cutoff)


def test_the_search_compares_pairs_or_uses_keys_by_their_estimated_time(
    monkeypatch,
) -> None:
    calls: dict[str, int] = {}
    for name in ("_keyed_pairs", "_hamming_pairs", "_edit_pairs"):

        def spy(*args, _original=getattr(sequences, name), _name=name):
            calls[_name] = calls.get(_name, 0) + 1
            return _original(*args)

        monkeypatch.setattr(sequences, name, spy)
    rng = np.random.default_rng(8)
    # Many short texts yield few keys each, so keys find their pairs sooner;
    # a few long texts yield many keys each, so comparing them does.
    short = sorted({"".join(rng.choice(list("ACGT"), size=6)) for _ in range(1500)})
    long = [
        base[:position] + "A" + base[position + 1 :]
        for base in ("".join(rng.choice(list("ACGT"), size=30)) for _ in range(8))
        for position in (0, 7, 29)
    ]
    texts = short + long
    first, second = neighbor_pairs(texts, "hamming", 1)
    assert calls == {"_keyed_pairs": 1, "_hamming_pairs": 1}
    codes = np.asarray([[ord(letter) for letter in text] for text in short])
    mismatches = (codes[:, None, :] != codes[None, :, :]).sum(axis=2)
    within = np.argwhere(np.triu(mismatches <= 1, k=1))
    want = set(map(tuple, within.tolist())) | {
        (len(short) + i, len(short) + j)
        for i, j in itertools.combinations(range(len(long)), 2)
        if _within(long[i], long[j], "hamming", 1)
    }
    assert _pairs(first, second) == want
    # Edits search each block their own way: keys for many short texts,
    # comparisons for a few long ones.
    calls.clear()
    edited = short[:150] + long[:6]
    blocks = np.repeat([0, 1], [150, 6])
    first, second = neighbor_pairs(edited, "levenshtein", 2, blocks=blocks)
    assert calls["_keyed_pairs"] == 1 and calls["_edit_pairs"] >= 1
    assert "_hamming_pairs" not in calls
    assert _pairs(first, second) == _expected(edited, "levenshtein", 2, blocks)


def test_texts_compare_by_character_beyond_ascii() -> None:
    # Latin-1 characters keep their code points; other characters are coded
    # by rank, so each counts as one substitution or edit.
    for texts in (["éa", "ea", "éb", "ébc"], ["αβγ", "αβδ", "ωβγ", "αβ", "éβγ"]):
        for metric, cutoff in (("hamming", 1), ("levenshtein", 1)):
            first, second = neighbor_pairs(texts, metric, cutoff)
            assert _pairs(first, second) == _expected(texts, metric, cutoff)
    # Alpha is U+03B1 and the plus-minus sign U+00B1: coded by their low
    # byte, they would be one character. A lone surrogate is a character.
    for texts in (["α", "±"], ["α±", "±α"], ["\ud800a", "\udc00a", "\ud800b"]):
        for metric in ("hamming", "levenshtein"):
            for cutoff in (0, 1):
                first, second = neighbor_pairs(texts, metric, cutoff)
                assert _pairs(first, second) == _expected(texts, metric, cutoff)
    assert _pairs(*neighbor_pairs(["α", "±"], "hamming", 0)) == set()
    with pytest.raises(ValueError, match="at most 256 distinct characters"):
        neighbor_pairs([chr(0x100 + index) for index in range(257)], "hamming", 1)


def test_few_unmatched_or_empty_texts() -> None:
    for texts, metric, cutoff, blocks in (
        (["CASSF"], "levenshtein", 1, None),
        (["CAF", "CASF", "CASSF"], "hamming", 3, None),
        (["CAF", "CASF", "CASSF"], "normalized_hamming", 50, None),
        (["CASSF", "CASSY"], "levenshtein", 1, [0, 1]),
    ):
        first, second = neighbor_pairs(texts, metric, cutoff, blocks=blocks)
        assert _pairs(first, second) == set()
    # Empty texts are identical, and one edit from a letter.
    for metric, cutoff in (
        ("hamming", 1),
        ("normalized_hamming", 0),
        ("levenshtein", 1),
    ):
        first, second = neighbor_pairs(["", "", "A"], metric, cutoff)
        assert _pairs(first, second) == _expected(["", "", "A"], metric, cutoff)


def test_the_metrics_are_scirpys_distances() -> None:
    assert DISTANCE_METRICS == ("hamming", "normalized_hamming", "levenshtein")
    assert METRICS == ("identity", *DISTANCE_METRICS)


# A family of texts of 12 letters whose variants at one position are all within
# one substitution of each other: 4 positions times 9 other letters, and the
# base, so every two variants at one position, and each variant and the base,
# are pairs.
_BASE = "ACDEFGHIKLMN"
_VARIANTS = [_BASE] + [
    _BASE[:position] + letter + _BASE[position + 1 :]
    for position in (0, 3, 6, 9)
    for letter in "PQRSTVWYC"
    if letter != _BASE[position]
]


def _encoding_bytes(texts: list[str]) -> int:
    """The bytes the encoded texts hold, as the search documents them.

    Three bytes for each letter of the longest text and 72 for each text,
    257 codes of 8 bytes for each position of the longest, and 64 KiB for the
    sorts and smaller arrays.
    """
    width = max(map(len, texts))
    return len(texts) * (3 * width + 72) + 8 * 257 * width + 2**16


def _group_bytes(texts: list[str]) -> int:
    """The 48 bytes that the search holds for each length of the texts."""
    return 48 * len(set(map(len, texts)))


# The two arrays that each length or block with pairs keeps, 192 bytes each.
_PAIR_ARRAYS = 2 * 192


@pytest.mark.parametrize("metric", ["hamming", "levenshtein"])
@pytest.mark.parametrize("path", ["keyed", "passes", "pairwise"])
def test_the_search_holds_no_more_than_the_smallest_budget_it_accepts(
    monkeypatch, path, metric
) -> None:
    # Copies of two texts of two lengths, the longer last, pair with each
    # other: their 30,450 pairs dominate what the search holds, the shorter
    # copies' pairs are held while the longer copies' are found, and each
    # pass of keys keeps more pairs than it holds keys. Small passes split
    # the keys of each length into several.
    texts = ["CASSF"] * 60 + ["CASSLGQAYEQYF"] * 240
    _force(monkeypatch, "pairwise" if path == "pairwise" else "keyed")
    if path == "passes":
        monkeypatch.setattr(sequences, "_KEYS_PER_PASS", 800)
        monkeypatch.setattr(sequences, "_MIN_KEYS_PER_PASS", 1)

    def search(budget: int | None) -> tuple[np.ndarray, np.ndarray]:
        return neighbor_pairs(texts, metric, 1, max_bytes=budget)

    assert _pairs(*search(None)) == _expected(texts, metric, 1)
    budget = smallest_budget(search, 1 << 23)
    _, peak = traced_peak(lambda: search(budget), budget)
    assert peak <= budget


@pytest.mark.parametrize("metric", ["hamming", "levenshtein"])
def test_keys_of_distinct_texts_hold_no_more_than_the_smallest_budget(
    monkeypatch, metric
) -> None:
    # 200 families of 20 distinct texts of 12 residues, the texts of each
    # differing at one position: every pair of a family is one substitution
    # or edit apart and shares the one key that masks or deletes that
    # position. The 38,000 pairs, each a candidate once, outnumber the keys
    # and take the most once joined and placed among all texts.
    rng = np.random.default_rng(6)
    letters = list("ACDEFGHIKLMNPQRSTVWY")
    texts = []
    for family in range(200):
        text = list(rng.choice(letters, size=12))
        for letter in letters:
            text[family % 12] = letter
            texts.append("".join(text))
    _force(monkeypatch, "keyed")

    def search(budget: int | None) -> tuple[np.ndarray, np.ndarray]:
        return neighbor_pairs(texts, metric, 1, max_bytes=budget)

    assert len(search(None)[0]) == 200 * 20 * 19 // 2
    budget = smallest_budget(search, 1 << 23)
    _, peak = traced_peak(lambda: search(budget), budget)
    assert peak <= budget


@pytest.mark.parametrize("metric", ["hamming", "levenshtein"])
@pytest.mark.parametrize("near", [True, False])
def test_many_blocks_hold_no_more_than_the_smallest_budget_they_accept(
    metric, near
) -> None:
    # Each of 1,500 blocks holds two texts, within one substitution of each
    # other or not: what each block keeps, its pairs and their arrays or
    # nothing, adds up over the blocks.
    other = "CASSLGQAYEQYW" if near else "GGGGGGGGGGGGG"
    texts = ["CASSLGQAYEQYF", other] * 1_500
    blocks = np.repeat(np.arange(1_500), 2)

    def search(budget: int | None) -> tuple[np.ndarray, np.ndarray]:
        return neighbor_pairs(texts, metric, 1, blocks=blocks, max_bytes=budget)

    assert len(search(None)[0]) == (1_500 if near else 0)
    budget = smallest_budget(search, 1 << 23)
    _, peak = traced_peak(lambda: search(budget), budget)
    assert peak <= budget


@pytest.mark.parametrize("metric", ["hamming", "levenshtein"])
@pytest.mark.parametrize("path", ["keyed", "pairwise"])
def test_a_budget_below_the_pairs_found_raises_memory_error(
    monkeypatch, path, metric
) -> None:
    _force(monkeypatch, path)
    expected = _expected(_VARIANTS, metric, 1)
    # The base pairs with each of its 36 variants, and the 9 variants at each
    # position pair with each other; variants at two positions are two
    # substitutions or edits apart.
    assert len(expected) == 36 + 4 * 9 * 8 // 2
    needed = (
        _encoding_bytes(_VARIANTS)
        + _group_bytes(_VARIANTS)
        + 32 * len(expected)
        + _PAIR_ARRAYS
    )
    with pytest.raises(MemoryError, match="for the pairs found, more than the"):
        neighbor_pairs(_VARIANTS, metric, 1, max_bytes=needed - 1)
    roomy = neighbor_pairs(_VARIANTS, metric, 1, max_bytes=needed + 2**20)
    assert _pairs(*roomy) == expected


@pytest.mark.parametrize("metric", ["hamming", "levenshtein"])
def test_pairs_found_hold_memory_while_later_lengths_are_searched(
    monkeypatch, metric
) -> None:
    _force(monkeypatch, "pairwise")
    # Two families of near variants, of 8 and 12 letters, more than one edit
    # apart: the shorter family's 36 pairs are found first and hold 32 bytes
    # each, and their arrays 384, while the longer family's 180 pairs are
    # collected.
    short = [letter + "CDEFGHIK"[1:] for letter in "PQRSTVWYC"]
    texts = short + _VARIANTS
    expected = _expected(texts, metric, 1)
    assert len(expected) == 36 + 180
    needed = (
        _encoding_bytes(texts)
        + _group_bytes(texts)
        + 32 * len(expected)
        + 2 * _PAIR_ARRAYS
    )
    with pytest.raises(MemoryError, match="for the pairs found"):
        neighbor_pairs(texts, metric, 1, max_bytes=needed - 1)
    assert _pairs(*neighbor_pairs(texts, metric, 1, max_bytes=needed)) == expected


def test_the_hash_codes_of_long_texts_count_against_the_budget() -> None:
    # Two texts of 2,000 letters need 2,000 positions of 257 codes of 8 bytes,
    # about 4 MB, beside a few kilobytes of encoded text.
    texts = ["A" * 2_000, "A" * 1_999 + "C"]
    with pytest.raises(MemoryError, match="for the encoded texts"):
        neighbor_pairs(texts, "hamming", 1, max_bytes=8 * 257 * 2_000 - 1)
    found = neighbor_pairs(
        texts, "hamming", 1, max_bytes=_encoding_bytes(texts) + 2**20
    )
    assert _pairs(*found) == {(0, 1)}


def test_the_encoded_texts_must_fit_the_budget() -> None:
    texts = ["CASSLGF", "CASSLGY", "CAVF"]
    with pytest.raises(MemoryError, match="for the encoded texts"):
        neighbor_pairs(texts, "levenshtein", 1, max_bytes=_encoding_bytes(texts))
    with pytest.raises(MemoryError, match="for the encoded texts"):
        neighbor_pairs(["A"], "hamming", 1, max_bytes=0)
    # No text, or one, is never compared, so the codes that hash keys are
    # neither built nor counted.
    assert len(neighbor_pairs([], "hamming", 1, max_bytes=0)[0]) == 0
    one = ["CASSLGQAYEQYF"]
    # Three bytes per letter and 72 for the text, and 2 per letter joined.
    needed = 3 * len(one[0]) + 72 + 2 * len(one[0])
    assert len(neighbor_pairs(one, "hamming", 1, max_bytes=needed)[0]) == 0
    # Texts beyond Latin-1 take more bytes per character while they are coded.
    wide = ["αβγ", "αβδ"]
    enough_for_latin = _encoding_bytes(wide) + 2 * 6
    with pytest.raises(MemoryError, match="for the encoded texts"):
        neighbor_pairs(wide, "hamming", 1, max_bytes=enough_for_latin)


def test_the_keys_and_candidate_pairs_of_a_pass_must_fit_the_budget(
    monkeypatch,
) -> None:
    _force(monkeypatch, "keyed")
    # Every variant at position 0 shares the key that masks it: 9 texts, 36
    # candidate pairs, all within one substitution. The keys of the 9 texts
    # number 9 * 12, beside the 12 of one text's scratch space; the
    # candidates need 8 bytes each beside them.
    texts = [letter + _BASE[1:] for letter in "PQRSTVWYC"]
    held = _encoding_bytes(texts) + _group_bytes(texts) + 8 * len(_BASE)
    keys = 8 * len(texts) * len(_BASE)
    with pytest.raises(MemoryError, match="the variant keys of one pass"):
        neighbor_pairs(texts, "hamming", 1, max_bytes=held + keys - 1)
    with pytest.raises(MemoryError, match="the candidate pairs of texts"):
        neighbor_pairs(texts, "hamming", 1, max_bytes=held + keys + 8 * 36 - 1)
    # Joined, the 36 pairs need 32 bytes each and their arrays 384, beside
    # the 192 of the array the pass kept: more than the keys and candidates.
    budget = held + 32 * 36 + 192 + _PAIR_ARRAYS
    with pytest.raises(MemoryError, match="the pairs found"):
        neighbor_pairs(texts, "hamming", 1, max_bytes=budget - 1)
    found = neighbor_pairs(texts, "hamming", 1, max_bytes=budget)
    assert _pairs(*found) == _expected(texts, "hamming", 1)
    # Of two texts, whose 24 letters joined fit beside the encoded texts, the
    # scratch space of one text's 12 keys is checked first.
    two = texts[:2]
    held = _encoding_bytes(two) + _group_bytes(two)
    with pytest.raises(MemoryError, match="the variant keys of one text"):
        neighbor_pairs(two, "hamming", 1, max_bytes=held + 8 * len(_BASE) - 1)


@pytest.mark.parametrize(
    ("text", "binding"),
    [
        # Copies of a text of 5 letters share its 5 keys: 10 copies yield 50
        # keys and 5 * 45 candidate pairs, of 45 distinct pairs. Copying the
        # pairs kept, and their array, beside the candidates takes the most.
        ("CASSF", 8 * 225 + 8 * 45 + 192),
        # Of 2 letters: 20 keys, 90 candidates and 45 pairs, which take the
        # most once joined, 32 bytes each, with the array the pass kept and
        # the two arrays of the pairs.
        ("CA", 32 * 45 + 192 + 2 * 192),
    ],
)
def test_the_pairs_kept_from_a_pass_must_fit_the_budget(
    monkeypatch, text, binding
) -> None:
    _force(monkeypatch, "keyed")
    texts = [text] * 10
    budget = _encoding_bytes(texts) + _group_bytes(texts) + 8 * len(text) + binding
    with pytest.raises(MemoryError, match="for the pairs found"):
        neighbor_pairs(texts, "hamming", 1, max_bytes=budget - 1)
    found = neighbor_pairs(texts, "hamming", 1, max_bytes=budget)
    assert _pairs(*found) == set(itertools.combinations(range(10), 2))


@pytest.mark.parametrize("metric", ["hamming", "levenshtein"])
def test_the_keys_of_a_pass_follow_the_memory_left(monkeypatch, metric) -> None:
    _force(monkeypatch, "keyed")
    # Keys may take a ten-thousandth of the bytes left, and a pass as few keys
    # as that allows, so a budget of about 1 MB holds 13 keys at a time.
    monkeypatch.setattr(sequences, "_KEY_SHARE", 10_000)
    monkeypatch.setattr(sequences, "_MIN_KEYS_PER_PASS", 1)
    passes: list[int] = []
    original = sequences._variant_keys

    def counted(*args):
        passes.append(args[7])
        return original(*args)

    monkeypatch.setattr(sequences, "_variant_keys", counted)
    rng = np.random.default_rng(12)
    texts = sorted(set(_families(rng, "ACGT", 40, 10)))
    expected = _expected(texts, metric, 1)
    assert _pairs(*neighbor_pairs(texts, metric, 1)) == expected
    assert set(passes) == {1}
    passes.clear()
    budget = _encoding_bytes(texts) + 2**20
    limited = neighbor_pairs(texts, metric, 1, max_bytes=budget)
    assert _pairs(*limited) == expected
    # The texts of a length, or of a block, yield many times 13 keys.
    assert max(passes) >= 5


def test_the_budget_and_the_texts_are_checked() -> None:
    with pytest.raises(TypeError, match="max_bytes must be an integer"):
        neighbor_pairs(["A", "C"], "hamming", 1, max_bytes=1.5)
    with pytest.raises(TypeError, match="max_bytes must be an integer"):
        neighbor_pairs(["A", "C"], "hamming", 1, max_bytes=True)
    with pytest.raises(ValueError, match="max_bytes must be at least 0"):
        neighbor_pairs(["A", "C"], "hamming", 1, max_bytes=-1)
    with pytest.raises(TypeError, match="not one text"):
        neighbor_pairs("ACGT", "hamming", 1)
    for texts in (["A", None], ["A", b"C"], ["A", 3]):
        with pytest.raises(TypeError, match="texts must hold only text"):
            neighbor_pairs(texts, "hamming", 1)
    with pytest.raises(TypeError, match="one-dimensional sequence of texts"):
        neighbor_pairs([["CASSF", "CASSY"]], "hamming", 1)
    for blocks in ([0.5, 1.0], [[0, 1]], ["a", "b"]):
        with pytest.raises(TypeError, match="one integer code per text"):
            neighbor_pairs(["CASSF", "CASSY"], "hamming", 1, blocks=blocks)
    # Boolean blocks are two blocks.
    found = neighbor_pairs(
        ["CASSF", "CASSY", "CASSW"], "hamming", 1, [True, True, False]
    )
    assert _pairs(*found) == {(0, 1)}
    # An array of fixed-width text is text, read as Python strings.
    array = np.asarray(["CASSF", "CASSY", "CAF"])
    assert _pairs(*neighbor_pairs(array, "hamming", 1)) == {(0, 1)}


# Texts with characters that hold no base, as TREX's partial cloneIDs.


def _known_within(first: str, second: str, cutoff: int, min_compared: int) -> bool:
    """TREX's test of two cloneIDs: mismatches where both hold a base."""
    if len(first) != len(second):
        return False
    compared = mismatches = 0
    for left, right in zip(first, second, strict=True):
        if left in "-0" or right in "-0":
            continue
        compared += 1
        mismatches += left != right
    return compared >= min_compared and mismatches <= cutoff


@pytest.mark.parametrize("cutoff", [0, 1, 3, 5])
@pytest.mark.parametrize("min_compared", [0, 4, 8])
def test_unknown_bases_are_not_compared_and_pairs_need_compared_positions(
    cutoff, min_compared
) -> None:
    rng = np.random.default_rng([cutoff, min_compared])
    letters = list("ACGT-0")
    texts = []
    for _ in range(120):
        base = "".join(rng.choice(letters[:4], size=10))
        texts.append(base)
        # A partial copy, as reads that cover only part of a cloneID give.
        cut = int(rng.integers(0, 10))
        texts.append(base[:cut] + "-" * (10 - cut))
        texts.append("".join(rng.choice(letters, size=10)))
    texts.append("".join(rng.choice(letters, size=9)))
    blocks = rng.integers(0, 3, size=len(texts))
    found = _pairs(
        *neighbor_pairs(
            texts, "hamming", cutoff, blocks, unknown="-0", min_compared=min_compared
        )
    )
    expected = {
        (i, j)
        for i, j in itertools.combinations(range(len(texts)), 2)
        if blocks[i] == blocks[j]
        and _known_within(texts[i], texts[j], cutoff, min_compared)
    }
    assert found == expected and expected


def test_unknown_characters_beyond_latin_one_are_coded_by_their_rank() -> None:
    texts = ["αβ-γ", "αβδγ", "αεδγ", "----"]
    found = _pairs(*neighbor_pairs(texts, "hamming", 0, unknown="-", min_compared=3))
    # The first matches the second where both hold a letter; the third
    # differs from both, and the last compares no position.
    assert found == {(0, 1)}
    found = _pairs(*neighbor_pairs(texts, "hamming", 1, unknown="-", min_compared=3))
    assert found == {(0, 1), (0, 2), (1, 2)}
    # A character that no text holds changes nothing.
    assert _pairs(
        *neighbor_pairs(texts, "hamming", 0, unknown="-ω", min_compared=3)
    ) == {(0, 1)}


def test_without_unknown_characters_the_search_is_unchanged() -> None:
    rng = np.random.default_rng(4)
    texts = ["".join(rng.choice(list("ACGT"), size=12)) for _ in range(300)]
    texts += [text[:5] + "A" + text[6:] for text in texts[:100]]
    for cutoff in (0, 1, 2):
        plain = _pairs(*neighbor_pairs(texts, "hamming", cutoff))
        known = _pairs(*neighbor_pairs(texts, "hamming", cutoff, unknown="-0"))
        assert plain == known == _expected(texts, "hamming", cutoff)


def test_unknown_characters_apply_to_hamming_only() -> None:
    for metric in ("levenshtein", "normalized_hamming"):
        with pytest.raises(ValueError, match="'hamming' only"):
            neighbor_pairs(["AC", "AG"], metric, 1, unknown="-")
        with pytest.raises(ValueError, match="'hamming' only"):
            neighbor_pairs(["AC", "AG"], metric, 1, min_compared=1)
    with pytest.raises(TypeError, match="unknown must be text"):
        neighbor_pairs(["AC", "AG"], "hamming", 1, unknown=["-"])
    with pytest.raises(ValueError, match="min_compared must be at least 0"):
        neighbor_pairs(["AC", "AG"], "hamming", 1, min_compared=-1)


def test_partial_pairs_found_count_against_the_budget() -> None:
    # Every pair of 400 texts with nothing known but a shared base matches.
    texts = ["A" + "-" * 9] * 400
    needed = len(texts) * (len(texts) - 1) // 2
    with pytest.raises(MemoryError, match="the pairs found"):
        neighbor_pairs(texts, "hamming", 0, unknown="-", max_bytes=200_000)
    found = neighbor_pairs(texts, "hamming", 0, unknown="-", max_bytes=40 * needed)
    assert len(found[0]) == needed


# Keys that mask segments of positions, for long texts.


def _barcodes(rng, n_texts: int, length: int, most: int = 3) -> list[str]:
    """Random DNA barcodes with a variant 1 to ``most`` substitutions from each tenth."""
    letters = np.asarray(list("ACGT"))
    texts = ["".join(rng.choice(letters, size=length)) for _ in range(n_texts)]
    for base in texts[: n_texts // 10]:
        variant = list(base)
        changed = int(rng.integers(1, most + 1))
        for position in rng.choice(length, size=changed, replace=False):
            variant[position] = "ACGT"[("ACGT".index(variant[position]) + 1) % 4]
        texts.append("".join(variant))
    return sorted(set(texts))


@pytest.mark.parametrize("cutoff", [1, 2, 3])
def test_segment_keys_find_every_pair_of_long_barcodes(cutoff, monkeypatch) -> None:
    rng = np.random.default_rng(cutoff)
    texts = _barcodes(rng, 2_000, 28)
    chosen: list[int] = []
    original = sequences._keyed_pairs

    def spy(*args):
        chosen.append(args[6])
        return original(*args)

    monkeypatch.setattr(sequences, "_keyed_pairs", spy)
    found = _pairs(*neighbor_pairs(texts, "hamming", cutoff))
    assert found == _expected(texts, "hamming", cutoff) and found
    # Fewer, longer segments than positions take their keys.
    assert len(chosen) == 1 and cutoff < chosen[0] < 28


@pytest.mark.parametrize(
    ("cutoff", "segments"),
    # Keys that mask 3 of 9 or 28 segments, keep 1 or 2 of 4 or 5, or keep 1
    # to 3 of 6 to 8 under 5 substitutions.
    [(3, 4), (3, 5), (3, 9), (3, 28), (5, 6), (5, 7), (5, 8)],
)
def test_every_split_into_segments_finds_the_same_pairs(
    cutoff, segments, monkeypatch
) -> None:
    rng = np.random.default_rng(segments)
    texts = _barcodes(rng, 600, 28, most=cutoff + 1)
    monkeypatch.setattr(
        sequences,
        "_hamming_segments",
        lambda codes, limit, room, split=0: segments,
    )
    expected = _expected(texts, "hamming", cutoff)
    assert _pairs(*neighbor_pairs(texts, "hamming", cutoff)) == expected
    assert len(expected) >= 30


def test_the_segments_cut_the_keys_of_long_barcodes() -> None:
    rng = np.random.default_rng(31)
    codes = rng.integers(0, 4, size=(100_000, 28)).astype(np.uint8)
    segments = sequences._hamming_segments(codes, 3, 1 << 34)
    # Five segments of five or six bases make ten keys per barcode, and two
    # random barcodes rarely agree on the eleven bases a key keeps; masking
    # positions would make 3,276.
    assert 4 <= segments <= 8
    assert sequences._key_count(28, 3, False, segments) < 100
    # Texts that agree almost everywhere share keys by chance, so their
    # segments stay short or they are compared pair by pair.
    alike = np.zeros((2_000, 28), dtype=np.uint8)
    alike[:, -1] = rng.integers(0, 4, size=2_000)
    assert sequences._hamming_segments(alike, 1, 1 << 34) in (0, 28)
    # Candidates that would not fit the room leave a choice out.
    assert sequences._hamming_segments(codes, 3, 0) == 0
    # Within 5 substitutions, keys keep 1 to 3 of 6 to 8 segments: at most 56
    # keys a barcode, where masking positions would make 98,280.
    segments = sequences._hamming_segments(codes, 5, 1 << 34)
    assert 6 <= segments <= 8
    assert sequences._key_count(28, 5, False, segments) <= 56


def test_keys_find_every_pair_within_four_or_five_substitutions(monkeypatch) -> None:
    rng = np.random.default_rng(45)
    texts = _barcodes(rng, 3_000, 30, most=6)
    codes = np.asarray([list(map(ord, text)) for text in texts], dtype=np.uint8)
    differ = (codes[:, None, :] != codes[None, :, :]).sum(axis=2)
    chosen: list[int] = []
    original = sequences._keyed_pairs

    def spy(*args):
        chosen.append(args[6])
        return original(*args)

    monkeypatch.setattr(sequences, "_keyed_pairs", spy)
    for cutoff in (4, 5):
        chosen.clear()
        first, second = np.nonzero(np.triu(differ <= cutoff, 1))
        expected = set(zip(first.tolist(), second.tolist(), strict=True))
        assert _pairs(*neighbor_pairs(texts, "hamming", cutoff)) == expected
        assert len(expected) > 100
        # Keys of a few more segments than the cutoff found them.
        assert len(chosen) == 1 and cutoff < chosen[0] <= cutoff + 3


def test_texts_across_a_split_that_never_agree_at_a_position_choose_keys() -> None:
    rng = np.random.default_rng(8)
    codes = rng.integers(0, 4, size=(4_000, 24)).astype(np.uint8)
    # The rows below the split hold 0 at the first position, the rest never.
    codes[:2_000, 0] = 0
    codes[2_000:, 0] = rng.integers(1, 4, size=2_000)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        across = sequences._hamming_segments(codes, 2, 1 << 34, split=2_000)
        within = sequences._hamming_segments(codes, 2, 1 << 34)
    assert across > 2 and within > 2


# TREX's partial cloneIDs, grouped by the positions where they hold no base.


def _cloneids(rng, n_bases: int, length: int = 30) -> list[str]:
    """CloneIDs with variants of up to 6 substitutions, some cut or with a deletion.

    A cut variant lost up to 13 bases at one end, as a read that covers part
    of the barcode gives, marked ``-``; a deletion marks 3 bases ``0``.
    """
    letters = np.asarray(list("ACGT"))
    texts = []
    for _ in range(n_bases):
        base = list(rng.choice(letters, size=length))
        texts.append("".join(base))
        for _ in range(int(rng.integers(0, 5))):
            variant = list(base)
            changed = int(rng.integers(0, 7))
            for position in rng.choice(length, size=changed, replace=False):
                variant[position] = str(rng.choice(letters))
            kind = int(rng.integers(0, 4))
            cut = int(rng.integers(1, 14))
            if kind == 1:
                variant[:cut] = "-" * cut
            elif kind == 2:
                variant[-cut:] = "-" * cut
            elif kind == 3:
                at = int(rng.integers(0, length - 3))
                variant[at : at + 3] = "000"
            texts.append("".join(variant))
    return texts


@pytest.mark.parametrize("path", ["chosen", "keyed", "pairwise"])
@pytest.mark.parametrize(
    ("cutoff", "min_compared"), [(5, 20), (3, 20), (5, 0), (0, 25), (4, 10)]
)
def test_partial_cloneids_find_every_pair_by_their_unknown_positions(
    monkeypatch, path, cutoff, min_compared
) -> None:
    if path != "chosen":
        _force(monkeypatch, path)
    rng = np.random.default_rng([cutoff, min_compared])
    texts = _cloneids(rng, 60)
    blocks = rng.integers(0, 2, size=len(texts))
    found = _pairs(
        *neighbor_pairs(
            texts, "hamming", cutoff, blocks, unknown="-0", min_compared=min_compared
        )
    )
    expected = {
        (i, j)
        for i, j in itertools.combinations(range(len(texts)), 2)
        if blocks[i] == blocks[j]
        and _known_within(texts[i], texts[j], cutoff, min_compared)
    }
    assert found == expected
    # Pairs of a cut or deleted cloneID with a full one, and of two cut ones
    # where a pair compares few enough positions.
    partial = {i for i, text in enumerate(texts) if "-" in text or "0" in text}
    assert any((i in partial) != (j in partial) for i, j in expected)
    if min_compared <= 20:
        assert any(i in partial and j in partial for i, j in expected)


@pytest.mark.parametrize("path", ["keyed", "pairwise"])
def test_two_partial_texts_compare_the_positions_both_hold(monkeypatch, path) -> None:
    _force(monkeypatch, path)
    texts = ["AAAAAAAAAA", "AAAAA-----", "-----CCCCC", "AAAAACCCCC"]
    # The cut texts match the full texts on their halves; they share no
    # position, so they pair only when a pair needs none.
    assert _pairs(
        *neighbor_pairs(texts, "hamming", 0, unknown="-", min_compared=5)
    ) == {(0, 1), (1, 3), (2, 3)}
    assert _pairs(*neighbor_pairs(texts, "hamming", 0, unknown="-")) == {
        (0, 1),
        (1, 2),
        (1, 3),
        (2, 3),
    }
    # Five substitutions on the second half link the first full text with
    # the last and with the text that holds that half only.
    assert _pairs(
        *neighbor_pairs(texts, "hamming", 5, unknown="-", min_compared=5)
    ) == {(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)}


def test_partial_cloneids_search_their_groups_by_keys(monkeypatch) -> None:
    # 20,000 cloneIDs, three in ten cut at one end, as TREX's reads give:
    # each group of a cut, and the full cloneIDs, find their pairs through
    # keys, and only small groups compare every pair.
    rng = np.random.default_rng(20)
    codes = rng.integers(0, 4, size=(20_000, 30))
    cut = rng.random(20_000) < 0.3
    lost = rng.integers(1, 11, size=20_000)
    texts = []
    for row, values in enumerate(codes.tolist()):
        text = "".join("ACGT"[value] for value in values)
        if cut[row]:
            text = "-" * lost[row] + text[lost[row] :]
        texts.append(text)
    looped: list[int] = []
    keyed: list[int] = []
    original_loops = sequences._hamming_pairs
    original_keys = sequences._keyed_pairs

    def loops(words, limit, most, split):
        looped.append(len(words))
        return original_loops(words, limit, most, split)

    def keys(*args):
        keyed.append(len(args[0]))
        return original_keys(*args)

    monkeypatch.setattr(sequences, "_hamming_pairs", loops)
    monkeypatch.setattr(sequences, "_keyed_pairs", keys)
    neighbor_pairs(texts, "hamming", 5, unknown="-", min_compared=20)
    assert max(keyed) > 14_000
    assert not looped or max(looped) < 2_000


@pytest.mark.parametrize("path", ["keyed", "pairwise"])
@pytest.mark.parametrize("near", [True, False])
def test_partial_texts_hold_no_more_than_the_smallest_budget_they_accept(
    monkeypatch, path, near
) -> None:
    # Copies of cut and full texts that all pair, whose pairs take the most;
    # or distinct random cloneIDs cut in many ways, which pair with nothing,
    # so that their positions and groups take the most.
    _force(monkeypatch, path)
    if near:
        texts = (
            ["A" * 20 + "-" * 10] * 120 + ["A" * 30] * 120 + ["-" * 10 + "A" * 20] * 120
        )
    else:
        rng = np.random.default_rng(9)
        texts = _cloneids(rng, 1_500)

    def search(budget: int | None) -> tuple[np.ndarray, np.ndarray]:
        return neighbor_pairs(
            texts, "hamming", 0, unknown="-0", min_compared=10, max_bytes=budget
        )

    if near:
        assert len(search(None)[0]) == 360 * 359 // 2
    budget = smallest_budget(search, 1 << 24)
    assert traced_peak(lambda: search(budget)) <= budget


def test_the_positions_of_partial_texts_count_against_the_budget() -> None:
    # 600 cut texts that pair with nothing hold their positions, 64 bytes
    # each, while their groups are searched.
    rng = np.random.default_rng(10)
    texts = ["".join(rng.choice(list("ACGT"), size=20)) + "-" * 10 for _ in range(600)]
    held = _encoding_bytes(texts) + _group_bytes(texts)
    with pytest.raises(MemoryError, match="where the texts hold no base"):
        neighbor_pairs(texts, "hamming", 0, unknown="-", max_bytes=held + 64 * 600 - 1)
    found = neighbor_pairs(
        texts, "hamming", 0, unknown="-", max_bytes=held + 64 * 600 + 2**20
    )
    assert len(found[0]) == 0


# The pairs that the search expects by chance, against counts of random texts.


def _random_texts(
    rng, n_texts: int, lengths, probabilities, fixed: dict[int, str] | None = None
) -> list[str]:
    """Distinct texts whose characters are drawn independently, position by position.

    Each text takes one of ``lengths`` at random, and ``fixed`` gives the
    letter that every text holds at some positions, as designed barcodes do.
    """
    letters = np.asarray(list("ACGT"))
    texts = set()
    for length in rng.choice(np.asarray(lengths), size=n_texts):
        text = letters[rng.choice(4, size=int(length), p=probabilities)]
        for position, letter in (fixed or {}).items():
            text[position] = letter
        texts.add("".join(text))
    return sorted(texts)


def _substitution_links(
    texts: list[str], cutoff: int, unknown: str = "", min_compared: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """The pairs of texts of one length within ``cutoff`` substitutions, by brute force.

    Positions where either text holds an ``unknown`` character are not
    compared, and a pair needs ``min_compared`` positions compared.
    """
    firsts, seconds = [], []
    lengths = np.fromiter(map(len, texts), dtype=np.int64)
    for length in np.unique(lengths).tolist():
        members = np.flatnonzero(lengths == length)
        codes = np.asarray([list(texts[member]) for member in members])
        known = ~np.isin(codes, list(unknown))
        for row in range(len(members) - 1):
            both = known[row] & known[row + 1 :]
            differ = ((codes[row + 1 :] != codes[row]) & both).sum(axis=1)
            linked = np.flatnonzero(
                (differ <= cutoff) & (both.sum(axis=1) >= min_compared)
            )
            firsts.append(np.full(len(linked), members[row]))
            seconds.append(members[row + 1 + linked])
    return np.concatenate(firsts), np.concatenate(seconds)


def _edit_links(texts: list[str], cutoff: int) -> tuple[np.ndarray, np.ndarray]:
    """The pairs of texts within ``cutoff`` edits, the dynamic programming of every pair."""
    n_texts = len(texts)
    lengths = np.fromiter(map(len, texts), dtype=np.int64)
    width = int(lengths.max())
    codes = np.full((n_texts, width), -1, dtype=np.int64)
    for row, text in enumerate(texts):
        codes[row, : len(text)] = [ord(letter) for letter in text]
    first, second = np.triu_indices(n_texts, 1)
    near = np.abs(lengths[first] - lengths[second]) <= cutoff
    first, second = first[near], second[near]
    within = np.zeros(len(first), dtype=bool)
    for start in range(0, len(first), 50_000):
        rows = first[start : start + 50_000]
        columns = second[start : start + 50_000]
        left, right = codes[rows], codes[columns]
        ends = lengths[columns]
        previous = np.tile(np.arange(width + 1), (len(rows), 1))
        distance = np.where(lengths[rows] == 0, ends, 0)
        for row in range(1, width + 1):
            current = np.empty_like(previous)
            current[:, 0] = row
            for column in range(1, width + 1):
                current[:, column] = np.minimum(
                    np.minimum(
                        previous[:, column - 1]
                        + (left[:, row - 1] != right[:, column - 1]),
                        previous[:, column] + 1,
                    ),
                    current[:, column - 1] + 1,
                )
            done = lengths[rows] == row
            distance[done] = current[done, ends[done]]
            previous = current
        within[start : start + 50_000] = distance <= cutoff
    return first[within], second[within]


def _assert_expected(count: int, variance: float, expected: float, slack: float = 0.0):
    """The count lies within four standard deviations of the expectation.

    ``slack`` widens the tolerance by that share of the count, for an
    approximate expectation. The tolerance is tight enough that an
    expectation off by a factor of 2 lies outside it.
    """
    tolerance = 4 * variance**0.5 + slack * count
    assert abs(count - expected) <= tolerance, (count, expected, tolerance)
    for wrong in (2 * expected, expected / 2):
        assert abs(count - wrong) > tolerance, (count, wrong, tolerance)


@pytest.mark.parametrize(
    ("length", "probabilities", "cutoff", "fixed"),
    [
        (11, (0.25, 0.25, 0.25, 0.25), 2, None),
        (16, (0.55, 0.25, 0.15, 0.05), 2, None),
        # Pairs of short texts that would agree everywhere are a large share
        # of those within one substitution, and distinct texts never do.
        (8, (0.4, 0.3, 0.2, 0.1), 1, None),
        (13, (0.25, 0.25, 0.25, 0.25), 2, {3: "A", 9: "T"}),
    ],
)
def test_the_pairs_expected_by_chance_match_a_count_of_random_texts(
    length, probabilities, cutoff, fixed
) -> None:
    rng = np.random.default_rng(length * 10 + cutoff)
    texts = _random_texts(rng, 3_000, [length], probabilities, fixed)
    first, second = _substitution_links(texts, cutoff)
    count, variance = link_count_variance(len(texts), first, second)
    found, _second = neighbor_pairs(texts, "hamming", cutoff)
    expected = expected_chance_pairs(texts, "hamming", cutoff)
    assert len(found) == count
    _assert_expected(count, variance, expected)


def test_texts_of_two_blocks_expect_only_the_pairs_of_one_block() -> None:
    rng = np.random.default_rng(21)
    texts = _random_texts(rng, 3_000, [11], (0.25, 0.25, 0.25, 0.25))
    # Blocks drawn at random: sorted texts that share a prefix lie together,
    # so blocks taken in turn would split near texts apart.
    blocks = rng.integers(0, 2, size=len(texts))
    first, second = _substitution_links(texts, 2)
    same = blocks[first] == blocks[second]
    count, variance = link_count_variance(len(texts), first[same], second[same])
    expected = expected_chance_pairs(texts, "hamming", 2, blocks=blocks)
    _assert_expected(count, variance, expected)
    # Every pair of every block: the pairs across blocks are not compared.
    alone = sum(
        expected_chance_pairs(
            [text for text, block in zip(texts, blocks, strict=True) if block == b],
            "hamming",
            2,
        )
        for b in (0, 1)
    )
    assert expected == pytest.approx(alone, rel=1e-12)


def test_partial_texts_expect_pairs_on_the_positions_both_hold() -> None:
    rng = np.random.default_rng(14)
    texts = []
    for text in _random_texts(rng, 3_000, [14], (0.4, 0.3, 0.2, 0.1)):
        cut = int(rng.integers(0, 6))
        if cut and rng.random() < 0.5:
            text = (
                "-" * cut + text[cut:]
                if rng.random() < 0.5
                else text[:-cut] + "-" * cut
            )
        texts.append(text)
    texts = sorted(set(texts))
    first, second = _substitution_links(texts, 2, unknown="-", min_compared=9)
    count, variance = link_count_variance(len(texts), first, second)
    found, _second = neighbor_pairs(texts, "hamming", 2, unknown="-", min_compared=9)
    expected = expected_chance_pairs(texts, "hamming", 2, unknown="-", min_compared=9)
    assert len(found) == count
    _assert_expected(count, variance, expected)


@pytest.mark.parametrize(
    ("lengths", "cutoff", "n_texts"),
    [((6, 7), 1, 1_500), ((8, 9), 2, 700), ((10, 11), 3, 500)],
)
def test_edits_expect_pairs_within_a_tenth_of_a_count_of_random_texts(
    lengths, cutoff, n_texts
) -> None:
    rng = np.random.default_rng(cutoff)
    texts = _random_texts(rng, n_texts, lengths, (0.25, 0.25, 0.25, 0.25))
    first, second = _edit_links(texts, cutoff)
    count, variance = link_count_variance(len(texts), first, second)
    found, _second = neighbor_pairs(texts, "levenshtein", cutoff)
    expected = expected_chance_pairs(texts, "levenshtein", cutoff)
    assert len(found) == count
    # The approximation treats the pairs of positions as independent.
    _assert_expected(count, variance, expected, slack=0.1)


def _pattern_edit_distance(matches: np.ndarray) -> int:
    """The edit distance of two texts whose positions agree where ``matches`` holds."""
    rows, columns = matches.shape
    previous = list(range(columns + 1))
    for row in range(1, rows + 1):
        current = [row]
        for column in range(1, columns + 1):
            current.append(
                min(
                    previous[column - 1] + (not matches[row - 1, column - 1]),
                    previous[column] + 1,
                    current[column - 1] + 1,
                )
            )
        previous = current
    return previous[-1]


@pytest.mark.parametrize(
    ("rows", "columns", "limit"),
    [(3, 3, 1), (3, 4, 1), (4, 3, 2), (2, 4, 2), (4, 4, 2), (3, 5, 2), (4, 4, 3)],
)
def test_the_chance_of_edits_follows_every_pattern_of_agreeing_positions(
    rows, columns, limit
) -> None:
    # Every pattern of agreeing pairs of positions, weighted by its chance
    # when each pair agrees independently, gives the chance exactly.
    rng = np.random.default_rng(rows * 100 + columns * 10 + limit)
    agree = rng.uniform(0.05, 0.95, size=(rows, columns))
    exact = 0.0
    for pattern in itertools.product((False, True), repeat=rows * columns):
        matches = np.asarray(pattern).reshape(rows, columns)
        if _pattern_edit_distance(matches) <= limit:
            exact += float(np.prod(np.where(matches, agree, 1.0 - agree)))
    chance = sequences._edit_chance(agree, limit, sequences._Memory(None))
    assert chance == pytest.approx(exact, rel=1e-12, abs=1e-15)


def test_the_edit_distances_of_chance_pairs_keep_only_the_states_that_can_end_within(
    monkeypatch,
) -> None:
    # Capping each cell where it can no longer end within the cutoff keeps
    # 534 states of 30 positions within 6 edits, with the same chance as the
    # 2,213 states that a cap of one more than the cutoff in every cell keeps.
    held: list[int] = []
    check = sequences._Memory.check

    def recorded(self, needed, what):
        if what == "the edit distances of chance pairs":
            held.append(needed - sequences._CHANCE_FIXED_BYTES)
        return check(self, needed, what)

    monkeypatch.setattr(sequences._Memory, "check", recorded)
    sequences._edit_chance(np.full((30, 30), 0.25), 6, sequences._Memory(None))
    assert max(held) // sequences._CHANCE_STATE_BYTES <= 600


@pytest.mark.parametrize(
    ("cutoff", "neighbours"), [(1, 12), (2, 12 + 54), (3, 12 + 54 + 108), (4, 255)]
)
def test_every_text_of_a_length_expects_exactly_the_pairs_it_holds(
    cutoff, neighbours
) -> None:
    # The 256 texts of four bases hold each base at each position as often,
    # so they are the independent draws of the expectation, without repeats:
    # each text has 4 * 3 others one substitution away, 6 * 3 * 3 two away,
    # 4 * 3 * 3 * 3 three away, and every other within four.
    texts = ["".join(bases) for bases in itertools.product("ACGT", repeat=4)]
    found, _second = neighbor_pairs(texts, "hamming", cutoff)
    expected = expected_chance_pairs(texts, "hamming", cutoff)
    assert len(found) == 256 * neighbours // 2
    assert expected == pytest.approx(256 * neighbours / 2, rel=1e-12)


def test_texts_that_cannot_pair_by_chance_expect_nothing() -> None:
    # Two cut texts of one group agree wherever both hold a base, so the
    # search finds them, while texts of one group that agree everywhere they
    # are compared are none of the distinct texts that chance pairs.
    texts = ["ACGT-", "ACGT0"]
    found, _second = neighbor_pairs(texts, "hamming", 0, unknown="-0")
    expected = expected_chance_pairs(texts, "hamming", 0, unknown="-0")
    assert len(found) == 1 and expected == 0.0
    memory = sequences._Memory(None)
    # Lengths more edits apart than the cutoff, and positions that never
    # agree, are never within it.
    assert sequences._edit_chance(np.full((3, 6), 0.5), 2, memory) == 0.0
    assert sequences._edit_chance(np.zeros((6, 6)), 1, memory) == 0.0
    # Texts of 5 and of 9 characters are 4 edits apart at least, so within 2
    # each length expects only its own pairs.
    rng = np.random.default_rng(13)
    short = _random_texts(rng, 300, [5], (0.25, 0.25, 0.25, 0.25))
    long = _random_texts(rng, 300, [9], (0.25, 0.25, 0.25, 0.25))
    alone = [expected_chance_pairs(texts, "levenshtein", 2) for texts in (short, long)]
    both = expected_chance_pairs(short + long, "levenshtein", 2)
    assert both == pytest.approx(sum(alone), rel=1e-12)


def test_texts_of_one_length_expect_the_same_pairs_within_one_edit_or_substitution() -> (
    None
):
    # Texts of one length are within one edit only through one substitution,
    # so the edit distance's chance is the substitutions' exactly, distinct
    # texts leaving out the same identical pairs, on skewed composition.
    rng = np.random.default_rng(23)
    texts = _random_texts(rng, 2_000, [8], (0.55, 0.25, 0.15, 0.05))
    substitutions = expected_chance_pairs(texts, "hamming", 1)
    edits = expected_chance_pairs(texts, "levenshtein", 1)
    assert substitutions > 100
    assert edits == pytest.approx(substitutions, rel=1e-9)


def test_lengths_of_texts_that_share_no_character_expect_no_pair_across() -> None:
    # Texts of 6 bases of A and C, and of 7 of G and T, agree at no pair of
    # positions, so the pairs across lengths add nothing to each length's own.
    rng = np.random.default_rng(29)
    letters = np.asarray(list("ACGT"))
    short = sorted({"".join(letters[rng.integers(0, 2, size=6)]) for _ in range(60)})
    long = sorted({"".join(letters[rng.integers(2, 4, size=7)]) for _ in range(120)})
    alone = [expected_chance_pairs(texts, "levenshtein", 2) for texts in (short, long)]
    both = expected_chance_pairs(short + long, "levenshtein", 2)
    assert both == pytest.approx(sum(alone), rel=1e-12)


def test_the_frequencies_of_one_block_are_let_go_before_the_next() -> None:
    # Each block's frequencies are held while its pairs of lengths are
    # followed, and the search holds as much as before once they are done.
    rng = np.random.default_rng(31)
    codes = rng.integers(0, 4, size=(90, 9)).astype(np.uint8)
    lengths = [7, 8, 9]
    groups = [np.arange(0, 30), np.arange(30, 60), np.arange(60, 90)]
    memory = sequences._Memory(None, held=12_345)
    expected = sequences._edit_chance_pairs(codes, groups, groups, lengths, 2, memory)
    assert expected > 0 and memory.held == 12_345


def test_one_text_represents_each_component_by_its_priority() -> None:
    # Rows 0 to 5 in components 7, 3, 7, 3, 9, 7: the row of highest
    # priority represents each component, and of equal priorities the later.
    components = np.asarray([7, 3, 7, 3, 9, 7])
    priority = np.asarray([2, 5, 8, 1, 0, 8])
    rows = np.arange(6)
    chosen = sequences._representatives(rows, components, priority)
    assert sorted(chosen.tolist()) == [1, 4, 5]
    assert sorted(
        sequences._representatives(rows[:3], components, priority).tolist()
    ) == [1, 2]


def test_each_position_counts_the_best_text_of_each_component_that_holds_a_base() -> (
    None
):
    # Component 0 holds "AC-" ranked first and "GTA" second; component 1
    # holds "C-G". Where the first text of component 0 has no base, its
    # second text gives its character.
    texts = ["AC-", "GTA", "C-G"]
    matrix, _lengths, coding = sequences._encoded(texts, sequences._Memory(None))
    unknown = sequences._unknown_table("-", coding)
    frequencies = sequences._library_frequencies(
        matrix,
        np.arange(3),
        3,
        np.asarray([0, 0, 1]),
        np.asarray([2, 1, 0]),
        unknown,
    )

    def share(position: int, letter: str) -> float:
        return float(frequencies[position, ord(letter)])

    assert share(0, "A") == 0.5 and share(0, "C") == 0.5
    assert share(1, "C") == 1.0
    assert share(2, "A") == 0.5 and share(2, "G") == 0.5


def test_texts_that_are_their_own_components_expect_as_without_components() -> None:
    rng = np.random.default_rng(41)
    texts = _random_texts(rng, 1_500, [10], (0.4, 0.3, 0.2, 0.1))
    # Every third text cut by two bases at its end.
    cut = sorted(
        {
            text[:-2] + "--" if index % 3 == 0 else text
            for index, text in enumerate(texts)
        }
    )
    for metric, values, search in (
        ("hamming", texts, {}),
        ("levenshtein", texts, {}),
        ("hamming", cut, {"unknown": "-", "min_compared": 8}),
    ):
        alone = np.arange(len(values))
        assert expected_chance_pairs(
            values, metric, 2, components=alone, priority=alone, **search
        ) == pytest.approx(
            expected_chance_pairs(values, metric, 2, **search), rel=1e-12
        )


def test_a_block_of_one_text_expects_no_pairs() -> None:
    # The third text is alone in its block, so only the first two can pair.
    texts = ["ACGTAC", "ACGTAA", "TTTTTT"]
    for metric in ("hamming", "levenshtein"):
        assert expected_chance_pairs(
            texts, metric, 1, blocks=[0, 0, 1]
        ) == pytest.approx(expected_chance_pairs(texts[:2], metric, 1), rel=1e-12)


def test_the_representatives_alone_give_the_frequencies() -> None:
    # "AAAA" and "CCCC" represent their components, so each position agrees
    # with chance 1/2: within one substitution, a pair of distinct texts
    # falls with chance (1/16 + 4/16 - 1/16) / (1 - 1/16) = 4/15, and the
    # four texts pair six times. Texts linked into one component would all
    # be its representative, never distinct, so they expect no pairs.
    texts = ["AAAT", "AAAA", "CCCG", "CCCC"]
    expected = expected_chance_pairs(
        texts, "hamming", 1, components=[0, 0, 1, 1], priority=[0, 1, 0, 1]
    )
    assert expected == pytest.approx(6 * 4 / 15, rel=1e-12)
    assert expected_chance_pairs(texts, "hamming", 1, components=[0, 0, 0, 0]) == 0.0


@pytest.mark.parametrize("metric", ["hamming", "levenshtein"])
def test_a_cutoff_beyond_every_length_expects_every_pair(metric) -> None:
    rng = np.random.default_rng(3)
    texts = _random_texts(rng, 40, [6], (0.25, 0.25, 0.25, 0.25))
    found, _second = neighbor_pairs(texts, metric, 6)
    expected = expected_chance_pairs(texts, metric, 6)
    assert len(found) == len(texts) * (len(texts) - 1) // 2
    assert expected == pytest.approx(len(found), rel=1e-12)


def test_the_expectation_checks_its_arguments_and_needs_few_edits() -> None:
    texts = ["ACGTACGT", "ACGTACGA", "TTTTACGT"]
    assert len(neighbor_pairs(texts, "hamming", 1)) == 2
    assert expected_chance_pairs(["ACGT"], "levenshtein", 2) == 0.0
    with pytest.raises(ValueError, match="at most 8 with metric='levenshtein'"):
        expected_chance_pairs(texts, "levenshtein", 9)
    with pytest.raises(ValueError, match="metric='hamming', whose chance pairs"):
        expected_chance_pairs(texts, "levenshtein", 9)
    # The pairs alone need no limit on the edits.
    assert len(neighbor_pairs(texts, "levenshtein", 9)[0]) == 3
    with pytest.raises(ValueError, match="pass components too"):
        expected_chance_pairs(texts, "hamming", 1, priority=[0, 1, 2])
    with pytest.raises(ValueError, match="one component per text"):
        expected_chance_pairs(texts, "hamming", 1, components=[0, 1])
    with pytest.raises(TypeError, match="integer code per text"):
        expected_chance_pairs(texts, "hamming", 1, components=["a", "b", "c"])
    with pytest.raises(TypeError, match="one number per text"):
        expected_chance_pairs(
            texts, "hamming", 1, components=[0, 0, 1], priority=["x", "y", "z"]
        )
    with pytest.raises(ValueError, match="one number per text"):
        expected_chance_pairs(texts, "hamming", 1, components=[0, 0, 1], priority=[1])
    with pytest.raises(ValueError, match="metric must be one of"):
        expected_chance_pairs(texts, "identity", 1)


def test_the_frequencies_of_edited_texts_count_against_the_budget() -> None:
    # Texts of lengths 7 and 8 in one block hold the frequency of each of 256
    # codes at each of their 15 positions, and the 64 chances of agreeing of
    # two lengths, while their expectation is found; counting one position
    # copies the codes of the larger length's texts, widened to 8 bytes, and
    # counts them in 256 bins.
    rng = np.random.default_rng(5)
    texts = _random_texts(rng, 200, [7, 8], (0.25, 0.25, 0.25, 0.25))
    held = _encoding_bytes(texts) + _group_bytes(texts)
    most = max(sum(len(text) == length for text in texts) for length in (7, 8))
    frequencies = 8 * 256 * 15 + 8 * 8**2 + 9 * most + 8 * 256
    with pytest.raises(MemoryError, match="frequencies of the texts' characters"):
        expected_chance_pairs(texts, "levenshtein", 1, max_bytes=held + frequencies - 1)
    with pytest.raises(MemoryError) as refused:
        expected_chance_pairs(texts, "levenshtein", 1, max_bytes=held + frequencies)
    assert "frequencies" not in str(refused.value)


@pytest.mark.parametrize(
    ("metric", "cutoff", "lengths", "n_texts", "options"),
    [
        ("levenshtein", 3, (9, 10, 11, 12), 600, {}),
        ("levenshtein", 8, (20, 22, 24), 60, {}),
        ("hamming", 2, (12,), 3_000, {}),
        ("hamming", 5, (30,), 3_000, {"unknown": "-", "min_compared": 20}),
    ],
)
def test_the_expectation_holds_no_more_than_the_smallest_budget_it_accepts(
    metric, cutoff, lengths, n_texts, options
) -> None:
    # Families of ten texts each, so that one text per component is chosen,
    # and with characters of no base, a third cut at either end.
    rng = np.random.default_rng(cutoff)
    texts = _random_texts(rng, n_texts, lengths, (0.25, 0.25, 0.25, 0.25))
    if options:
        texts = [
            text
            if index % 3
            else ("-" * 5 + text[5:] if index % 2 else text[:-5] + "-" * 5)
            for index, text in enumerate(texts)
        ]
    components = np.arange(len(texts)) // 10
    priority = np.arange(len(texts))

    def expect(budget: int | None) -> float:
        return expected_chance_pairs(
            texts,
            metric,
            cutoff,
            components=components,
            priority=priority,
            max_bytes=budget,
            **options,
        )

    budget = smallest_budget(expect, 1 << 26)
    assert traced_peak(lambda: expect(budget)) <= budget
