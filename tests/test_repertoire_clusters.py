import itertools

import numpy as np
import pytest

from cytearc.repertoire.clonotypes import (
    DUAL_IR,
    RECEPTOR_ARMS,
    assign_clonotype_clusters,
    assign_clonotypes,
)
from cytearc.repertoire.distances import (
    METRICS,
    allowed_mismatches,
    neighbor_pairs,
)


def _levenshtein(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for row, left in enumerate(first, 1):
        current = [row]
        for column, right in enumerate(second, 1):
            current.append(
                min(
                    previous[column - 1] + (left != right),
                    previous[column] + 1,
                    current[column - 1] + 1,
                )
            )
        previous = current
    return previous[-1]


def _within(first: str, second: str, metric: str, cutoff: int) -> bool:
    if metric == "identity":
        return first == second
    if metric == "levenshtein":
        return _levenshtein(first, second) <= cutoff
    if len(first) != len(second):
        return False
    mismatches = sum(a != b for a, b in zip(first, second, strict=True))
    if metric == "hamming":
        return mismatches <= cutoff
    return not first or (200 * mismatches + len(first)) // (2 * len(first)) <= cutoff


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
    from cytearc.repertoire import distances

    names = (
        ("_KEY_NS", "_EDIT_KEY_NS")
        if path == "keyed"
        else ("_PAIR_NS", "_EDIT_PAIR_NS", "_EDIT_CUTOFF_NS")
    )
    for name in names:
        monkeypatch.setattr(distances, name, 0.0)


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


def _pairwise_partition(
    sequences, texts, groups, genes, receptor_arms, dual_ir, metric, cutoff
):
    """Clusters from the pairwise rules, checked for every pair of cells."""
    chains = [0] if dual_ir == "primary_only" else [0, 1]
    arms = {"VJ": list(chains), "VDJ": [2 + chain for chain in chains]}
    taking = ["VJ", "VDJ"] if receptor_arms in ("all", "any") else [receptor_arms]

    def elements(cell, arm):
        return [
            (texts[sequences[cell, slot]], int(genes[cell, slot]))
            for slot in arms[arm]
            if sequences[cell, slot] >= 0
        ]

    def near(x, y):
        return x[1] == y[1] and _within(x[0], y[0], metric, cutoff)

    def arm_match(first, second, arm):
        a, b = elements(first, arm), elements(second, arm)
        if dual_ir == "all":
            return (
                bool(a)
                and len(a) == len(b)
                and any(
                    all(near(x, y) for x, y in zip(a, order, strict=True))
                    for order in itertools.permutations(b)
                )
            )
        if dual_ir == "primary_only":
            return bool(a) and bool(b) and near(a[0], b[0])
        return any(near(x, y) for x in a for y in b)

    def primary_arms(cell):
        return sum(sequences[cell, arms[arm][0]] >= 0 for arm in ("VJ", "VDJ"))

    def match(first, second):
        if groups[first] != groups[second]:
            return False
        if receptor_arms in ("VJ", "VDJ"):
            return arm_match(first, second, receptor_arms)
        matched = [arm_match(first, second, arm) for arm in ("VJ", "VDJ")]
        if receptor_arms == "any":
            return any(matched)
        count = primary_arms(first)
        if count != primary_arms(second):
            return False
        return all(matched) if count == 2 else any(matched)

    cells = [
        cell
        for cell in range(len(groups))
        if groups[cell] >= 0 and any(elements(cell, arm) for arm in taking)
    ]
    parent = {cell: cell for cell in cells}

    def find(cell):
        while parent[cell] != cell:
            parent[cell] = parent[parent[cell]]
            cell = parent[cell]
        return cell

    for first, second in itertools.combinations(cells, 2):
        if match(first, second):
            parent[find(first)] = find(second)
    components = {}
    for cell in cells:
        components.setdefault(find(cell), set()).add(cell)
    return sorted(components.values(), key=min)


def _partition(codes):
    clusters = {}
    for cell, code in enumerate(codes.tolist()):
        if code >= 0:
            clusters.setdefault(code, set()).add(cell)
    return sorted(clusters.values(), key=min)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(
    ("metric", "cutoff"),
    [("hamming", 1), ("levenshtein", 1), ("normalized_hamming", 34)],
)
def test_clusters_link_exactly_the_cells_that_the_pairwise_rules_link(
    seed, metric, cutoff
) -> None:
    rng = np.random.default_rng(seed)
    n_cells = 70
    texts = sorted(
        {
            "".join(rng.choice(list("AC"), size=int(rng.integers(2, 5))))
            for _ in range(40)
        }
    )
    sequences = rng.integers(0, len(texts), size=(n_cells, 4))
    # Empty second chains, empty arms, and a cell without a primary chain.
    sequences[rng.random((n_cells, 4)) < 0.3] = -1
    sequences[sequences[:, 0] < 0, 1] = -1
    sequences[sequences[:, 2] < 0, 3] = -1
    groups = rng.integers(-1, 2, size=n_cells)
    genes = rng.integers(0, 2, size=(n_cells, 4))
    for receptor_arms in RECEPTOR_ARMS:
        for dual_ir in DUAL_IR:
            for with_genes in (False, True):
                assignment = assign_clonotype_clusters(
                    sequences,
                    texts,
                    groups=groups,
                    metric=metric,
                    cutoff=cutoff,
                    v_genes=genes if with_genes else None,
                    receptor_arms=receptor_arms,
                    dual_ir=dual_ir,
                )
                expected = _pairwise_partition(
                    sequences,
                    texts,
                    groups,
                    genes if with_genes else np.zeros_like(genes),
                    receptor_arms,
                    dual_ir,
                    metric,
                    cutoff,
                )
                assert _partition(assignment.codes) == expected, (
                    receptor_arms,
                    dual_ir,
                    with_genes,
                )
                assert (np.diff(assignment.sizes) <= 0).all()


def test_a_zero_cutoff_gives_the_identity_clonotypes() -> None:
    rng = np.random.default_rng(11)
    texts = [f"C{index}F" for index in range(12)]
    sequences = rng.integers(-1, len(texts), size=(60, 4))
    sequences[sequences[:, 0] < 0, 1] = -1
    sequences[sequences[:, 2] < 0, 3] = -1
    groups = rng.integers(0, 2, size=60)
    for receptor_arms in RECEPTOR_ARMS:
        for dual_ir in DUAL_IR:
            identity = assign_clonotypes(
                sequences, groups=groups, receptor_arms=receptor_arms, dual_ir=dual_ir
            )
            clusters = assign_clonotype_clusters(
                sequences,
                texts,
                groups=groups,
                metric="hamming",
                cutoff=0,
                receptor_arms=receptor_arms,
                dual_ir=dual_ir,
            )
            np.testing.assert_array_equal(identity.codes, clusters.codes)


def test_cluster_assignment_checks_its_arguments() -> None:
    sequences = np.array([[0, -1, 1, -1]])
    texts = ["CAVF", "CASSF"]
    groups = np.array([0])
    with pytest.raises(ValueError, match="metric must be one of"):
        assign_clonotype_clusters(
            sequences, texts, groups=groups, metric="identity", cutoff=0
        )
    with pytest.raises(ValueError, match="receptor_arms must be one of"):
        assign_clonotype_clusters(
            sequences,
            texts,
            groups=groups,
            metric="hamming",
            cutoff=1,
            receptor_arms="x",
        )
    with pytest.raises(ValueError, match="dual_ir must be one of"):
        assign_clonotype_clusters(
            sequences, texts, groups=groups, metric="hamming", cutoff=1, dual_ir="x"
        )
    with pytest.raises(ValueError, match="four slots"):
        assign_clonotype_clusters(
            sequences[:, :3], texts, groups=groups, metric="hamming", cutoff=1
        )
    with pytest.raises(ValueError, match="one group per cell"):
        assign_clonotype_clusters(
            sequences, texts, groups=np.array([0, 1]), metric="hamming", cutoff=1
        )
    with pytest.raises(ValueError, match="Gene codes must align"):
        assign_clonotype_clusters(
            sequences,
            texts,
            groups=groups,
            metric="hamming",
            cutoff=1,
            v_genes=np.zeros((1, 3)),
        )
    empty = assign_clonotype_clusters(
        sequences, texts, groups=np.array([-1]), metric="hamming", cutoff=1
    )
    assert empty.codes.tolist() == [-1] and empty.n_clonotypes == 0
    chainless = assign_clonotype_clusters(
        np.full((2, 4), -1),
        texts,
        groups=np.array([0, 0]),
        metric="levenshtein",
        cutoff=1,
    )
    assert chainless.codes.tolist() == [-1, -1]


@pytest.mark.parametrize(
    ("metric", "cutoff"), [("hamming", 1), ("hamming", 2), ("levenshtein", 2)]
)
def test_neighbor_pairs_split_keys_into_passes(monkeypatch, metric, cutoff) -> None:
    from cytearc.repertoire import distances

    rng = np.random.default_rng(3)
    texts = sorted(
        {
            "".join(rng.choice(list("ACG"), size=int(rng.integers(3, 9))))
            for _ in range(120)
        }
    )
    # A small pass splits the keys by their hash into many passes.
    _force(monkeypatch, "keyed")
    monkeypatch.setattr(distances, "_KEYS_PER_PASS", 97)
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
    from cytearc.repertoire import distances

    # With every code zero, or drawn from four values, most keys share a hash
    # with keys of other variants; those texts are compared and left out, and
    # most passes stay empty.
    values = np.random.default_rng(0).integers(0, 2**63, size=4).astype(np.uint64)

    def table(width: int) -> np.ndarray:
        if codes == "zero":
            return np.zeros((width, 257), dtype=np.uint64)
        return values[np.random.default_rng(1).integers(0, 4, size=(width, 257))]

    monkeypatch.setattr(distances, "_zobrist_table", table)
    monkeypatch.setattr(distances, "_KEYS_PER_PASS", 500)
    _force(monkeypatch, "keyed")
    rng = np.random.default_rng(4)
    texts = sorted(set(_families(rng, "ACG", 30, 8)))
    first, second = neighbor_pairs(texts, metric, cutoff)
    assert _pairs(first, second) == _expected(texts, metric, cutoff)


def test_the_search_compares_pairs_or_uses_keys_by_their_estimated_time(
    monkeypatch,
) -> None:
    from cytearc.repertoire import distances

    calls: dict[str, int] = {}
    for name in ("_keyed_pairs", "_hamming_pairs", "_edit_pairs"):

        def spy(*args, _original=getattr(distances, name), _name=name):
            calls[_name] = calls.get(_name, 0) + 1
            return _original(*args)

        monkeypatch.setattr(distances, name, spy)
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


def test_clusters_do_not_depend_on_the_chunks_of_pairs(monkeypatch) -> None:
    from cytearc.repertoire import clonotypes

    # Many cells share one of two near alpha chains, as invariant T cells do,
    # with second alpha chains and beta chains of their own.
    rng = np.random.default_rng(5)
    texts = ["CAVMDSNYQLIW", "CAVRDSNYQLIW"] + sorted(
        {"".join(rng.choice(list("AC"), size=5)) for _ in range(60)}
    )
    n_cells = 80
    sequences = np.full((n_cells, 4), -1, dtype=np.int64)
    sequences[:, 0] = rng.integers(0, 2, size=n_cells)
    sequences[:, 1] = np.where(
        rng.random(n_cells) < 0.4, rng.integers(2, len(texts), size=n_cells), -1
    )
    sequences[:, 2] = rng.integers(2, len(texts), size=n_cells)
    sequences[rng.random(n_cells) < 0.2, 2] = -1
    groups = rng.integers(0, 2, size=n_cells)
    whole = {}
    for receptor_arms in RECEPTOR_ARMS:
        for dual_ir in DUAL_IR:
            whole[receptor_arms, dual_ir] = assign_clonotype_clusters(
                sequences,
                texts,
                groups=groups,
                metric="hamming",
                cutoff=1,
                receptor_arms=receptor_arms,
                dual_ir=dual_ir,
            ).codes
    monkeypatch.setattr(clonotypes, "_PAIRS_PER_CHUNK", 3)
    for (receptor_arms, dual_ir), codes in whole.items():
        split = assign_clonotype_clusters(
            sequences,
            texts,
            groups=groups,
            metric="hamming",
            cutoff=1,
            receptor_arms=receptor_arms,
            dual_ir=dual_ir,
        )
        np.testing.assert_array_equal(split.codes, codes)
        expected = _pairwise_partition(
            sequences,
            texts,
            groups,
            np.zeros_like(sequences),
            receptor_arms,
            dual_ir,
            "hamming",
            1,
        )
        assert _partition(split.codes) == expected, (receptor_arms, dual_ir)
