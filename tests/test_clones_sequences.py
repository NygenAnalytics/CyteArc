"""Pairs of texts within a distance, checked against every pair compared directly."""

import itertools

import numpy as np
import pytest

from cytearc.clones import sequences
from cytearc.clones.sequences import (
    DISTANCE_METRICS,
    allowed_mismatches,
    neighbor_pairs,
)
from cytearc.repertoire.clonotypes import METRICS
from tests.clone_helpers import levenshtein_distance as _levenshtein
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
    """Make the search use keys, or compare pairs, wherever it can."""
    names = (
        ("_KEY_NS", "_EDIT_KEY_NS")
        if path == "keyed"
        else ("_PAIR_NS", "_EDIT_PAIR_NS", "_EDIT_CUTOFF_NS")
    )
    for name in names:
        monkeypatch.setattr(sequences, name, 0.0)


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


def test_neighbor_pairs_check_their_arguments() -> None:
    with pytest.raises(ValueError, match="metric must be one of"):
        neighbor_pairs(["A"], "identity", 0)
    with pytest.raises(ValueError, match="cutoff must be a non-negative integer"):
        neighbor_pairs(["A"], "hamming", -1)
    with pytest.raises(ValueError, match="cutoff must be a non-negative integer"):
        neighbor_pairs(["A"], "hamming", 1.5)
    with pytest.raises(ValueError, match="blocks must hold one block per text"):
        neighbor_pairs(["A", "C"], "hamming", 1, blocks=[0])
    first, second = neighbor_pairs([], "levenshtein", 1)
    assert len(first) == len(second) == 0
    assert METRICS[0] == "identity"


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
    """The bytes the encoded texts hold, as the search documents them."""
    return len(texts) * (3 * max(map(len, texts)) + 64)


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
    needed = _encoding_bytes(_VARIANTS) + 32 * len(expected)
    with pytest.raises(MemoryError, match="for the pairs found, more than the"):
        neighbor_pairs(_VARIANTS, metric, 1, max_bytes=needed - 1)
    roomy = neighbor_pairs(_VARIANTS, metric, 1, max_bytes=needed + 2**20)
    assert _pairs(*roomy) == expected


def test_the_encoded_texts_must_fit_the_budget() -> None:
    texts = ["CASSLGF", "CASSLGY", "CAVF"]
    with pytest.raises(MemoryError, match="for the encoded texts"):
        neighbor_pairs(texts, "levenshtein", 1, max_bytes=_encoding_bytes(texts))
    with pytest.raises(MemoryError, match="for the encoded texts"):
        neighbor_pairs(["A"], "hamming", 1, max_bytes=0)
    # Texts beyond Latin-1 take more bytes per character while they are coded.
    wide = ["αβγ", "αβδ"]
    enough_for_latin = _encoding_bytes(wide) + 2 * 6
    with pytest.raises(MemoryError, match="for the encoded texts"):
        neighbor_pairs(wide, "hamming", 1, max_bytes=enough_for_latin)


def test_candidate_pairs_of_a_pass_must_fit_the_budget(monkeypatch) -> None:
    _force(monkeypatch, "keyed")
    # Every variant at position 0 shares the key that masks it: 9 texts, 36
    # candidate pairs, all within one substitution. The keys of the 9 texts
    # number 9 * 12; the candidates need 8 bytes each beside them.
    texts = [letter + _BASE[1:] for letter in "PQRSTVWYC"]
    keys = 8 * len(texts) * len(_BASE)
    encoded = _encoding_bytes(texts)
    with pytest.raises(MemoryError, match="the candidate pairs of texts"):
        neighbor_pairs(texts, "hamming", 1, max_bytes=encoded + keys + 8 * 36 - 1)
    with pytest.raises(MemoryError, match="the pairs found"):
        neighbor_pairs(texts, "hamming", 1, max_bytes=encoded + keys + 8 * 36)
    found = neighbor_pairs(texts, "hamming", 1, max_bytes=encoded + 48 * 36)
    assert _pairs(*found) == _expected(texts, "hamming", 1)


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
