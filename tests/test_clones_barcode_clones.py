"""The rules of barcode clone calling on hand-built records.

Records are written out molecule by molecule, and what each rule keeps,
merges or refuses is derived by hand from the rule's published statement.
"""

import numpy as np
import pytest

from cytearc.clones import barcode_clones
from cytearc.clones.barcode_clones import (
    BARCODE_PRESETS,
    BarcodeRecords,
    allowlist_texts,
    barcode_call_settings,
    barcode_clone_calls,
    clone_labels,
)
from cytearc.clones.sequences import expected_chance_pairs, neighbor_pairs
from cytearc.metadata.records import pack_text
from tests.clone_helpers import (
    clone_partition,
    levenshtein_distance,
    link_count_variance,
)
from tests.memory_helpers import smallest_budget, traced_peak


def _largest_share(called: "barcode_clones.BarcodeClones") -> float:
    """The share of the cells given a clone that the largest clone holds, as the clone writer saves it."""
    from cytearc.datastore._operations.clone_artifacts import clone_diagnostics

    return clone_diagnostics(called.calls.codes, called.calls.sizes)[
        "largestCloneShare"
    ]


A = "ACGTACGTACGTACGTACGTACGTACGT"
# One substitution from A, and one more.
A1 = "ACGTACGTACGTACGTACGTACGTACGA"
A2 = "ACGTACGTACGTACGTACGTACGTACCA"
B = "TTTTGGGGCCCCAAAATTTTGGGGCCCC"
C = "GACTGACTGACTGACTGACTGACTGACT"


def _records(rows, libraries=(("lib", "molecules"),)) -> BarcodeRecords:
    """Records from (cell, library, barcode, umi or None, umis, reads) rows."""
    barcodes = sorted({row[2] for row in rows})
    umis = sorted({row[3] for row in rows if row[3] is not None})
    data, offsets = pack_text(barcodes)
    return BarcodeRecords(
        cells=np.asarray([row[0] for row in rows], dtype=np.int64),
        sources=np.asarray([row[1] for row in rows], dtype=np.int64),
        barcodes=np.asarray([barcodes.index(row[2]) for row in rows], dtype=np.int64),
        umi_codes=np.asarray(
            [-1 if row[3] is None else umis.index(row[3]) for row in rows],
            dtype=np.int64,
        ),
        umis=np.asarray([row[4] for row in rows], dtype=np.int64),
        reads=np.asarray([row[5] for row in rows], dtype=np.int64),
        barcode_data=data,
        barcode_offsets=offsets,
        libraries=tuple(libraries),
    )


def _molecules(cell: int, barcode: str, n: int, reads: int = 20, library: int = 0):
    return [
        (cell, library, barcode, f"U{cell}_{barcode}_{k}", 1, reads) for k in range(n)
    ]


def _call(records, n_cells, groups=None, allowlist=None, **options):
    settings = barcode_call_settings(allowlist=allowlist, **options)
    return barcode_clone_calls(
        records,
        n_cells,
        groups=np.zeros(n_cells, dtype=np.int64) if groups is None else groups,
        settings=settings,
        allowlist=None if allowlist in (None, False) else allowlist_texts(allowlist),
    )


# The published rules of each preset, as INPUT_FORMATS states them.
PUBLISHED = {
    "larry": {
        "min_reads": 10,
        "max_distance": 3,
        "metric": "hamming",
        "min_umis": 3,
        "min_fraction": 0.0,
        "barcodes_per_cell": None,
        "method": "sets",
        "similarity": None,
        "inclusive": None,
        "prune": False,
        "doublets": False,
        "min_size": 1,
        "trex_rules": False,
    },
    "trex": {
        "min_reads": 0,
        "max_distance": 5,
        "metric": "hamming",
        "min_umis": 0,
        "min_fraction": 0.0,
        "barcodes_per_cell": None,
        "method": "jaccard",
        "similarity": 0.7,
        "inclusive": False,
        "prune": True,
        "doublets": True,
        "min_size": 1,
        "trex_rules": True,
    },
    "celltag": {
        "min_reads": 0,
        "max_distance": 0,
        "metric": None,
        "min_umis": 2,
        "min_fraction": 0.0,
        "barcodes_per_cell": [2, 20],
        "method": "jaccard",
        "similarity": 0.7,
        "inclusive": False,
        "prune": False,
        "doublets": False,
        "min_size": 3,
        "trex_rules": False,
    },
    "celltag_multi": {
        "min_reads": 2,
        "max_distance": 2,
        "metric": "levenshtein",
        "min_umis": 2,
        "min_fraction": 0.0,
        "barcodes_per_cell": [2, 25],
        "method": "jaccard",
        "similarity": [0.6, 0.5],
        "inclusive": False,
        "prune": False,
        "doublets": False,
        "min_size": 2,
        "trex_rules": False,
    },
}


@pytest.mark.parametrize("preset", list(BARCODE_PRESETS))
def test_each_preset_resolves_to_its_published_rules(preset) -> None:
    allowlist = [A] if BARCODE_PRESETS[preset].get("allowlist") else None
    settings = barcode_call_settings(preset, allowlist=allowlist)
    assert settings.record() == {"preset": preset, **PUBLISHED[preset]}
    assert settings.allowlist == (allowlist is not None)


def test_given_values_replace_a_presets_and_zero_turns_a_filter_off() -> None:
    settings = barcode_call_settings("larry", min_reads=0, min_umis=5, max_distance=0)
    record = settings.record()
    assert (record["min_reads"], record["min_umis"], record["max_distance"]) == (
        0,
        5,
        0,
    )
    # Without correction the preset's metric falls away.
    assert record["metric"] is None
    settings = barcode_call_settings(
        "larry", method="jaccard", similarity=0.9, inclusive=True
    )
    assert (settings.method, settings.similarity, settings.inclusive) == (
        "jaccard",
        0.9,
        True,
    )
    settings = barcode_call_settings("trex", method="sets")
    # A preset's graph settings fall away under sets.
    assert (settings.similarity, settings.prune, settings.doublets) == (
        None,
        False,
        False,
    )
    assert barcode_call_settings("celltag", allowlist=False).allowlist is False
    # NumPy's False is False.
    assert barcode_call_settings("celltag", allowlist=np.False_).allowlist is False


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({}, ValueError, "needs a preset or a method"),
        ({"method": "jaccard"}, ValueError, "needs similarity"),
        ({"method": "jaccard", "similarity": 0.7}, ValueError, "needs inclusive"),
        ({"method": "sets", "max_distance": 2}, ValueError, "needs a metric"),
        ({"method": "sets", "metric": "hamming"}, ValueError, "give a max_distance"),
        (
            {"preset": "celltag", "allowlist": False, "metric": "hamming"},
            ValueError,
            "give a max_distance",
        ),
        (
            {"preset": "larry", "max_distance": 0, "metric": "levenshtein"},
            ValueError,
            "give a max_distance",
        ),
        (
            {"method": "sets", "max_distance": 2, "metric": "jaro"},
            ValueError,
            "metric must be",
        ),
        ({"preset": "celltag"}, ValueError, "allowlist=False"),
        ({"preset": "celltag_multi"}, ValueError, "allowlist=False"),
        ({"preset": "trex", "metric": "levenshtein"}, ValueError, "metric='hamming'"),
        (
            {"preset": "celltag_multi", "allowlist": False, "trex_rules": True},
            ValueError,
            "TREX's rules compare cloneIDs .* metric='hamming'; pass trex_rules=False",
        ),
        (
            {"preset": "trex", "trex_rules": 1},
            TypeError,
            "trex_rules must be a boolean",
        ),
        (
            {"preset": "trex", "trex_rules": "yes"},
            TypeError,
            "trex_rules must be a boolean",
        ),
        (
            {"preset": "trex", "method": "sets", "prune": True},
            ValueError,
            "change nothing",
        ),
        ({"method": "sets", "similarity": 0.5}, ValueError, "change nothing"),
        ({"preset": "cospar"}, ValueError, "preset must be one of"),
        ({"method": "graph"}, ValueError, "method must be one of"),
        (
            {"method": "jaccard", "similarity": 1.2, "inclusive": True},
            ValueError,
            "from 0 to 1",
        ),
        (
            {"method": "jaccard", "similarity": (0.6,), "inclusive": True},
            ValueError,
            "or two",
        ),
        (
            {"method": "jaccard", "similarity": "0.6", "inclusive": True},
            TypeError,
            "number",
        ),
        (
            {"method": "jaccard", "similarity": 0.6, "inclusive": 1},
            TypeError,
            "boolean",
        ),
        ({"method": "sets", "min_reads": -1}, ValueError, "at least 0"),
        ({"method": "sets", "min_reads": 2.0}, TypeError, "integer"),
        ({"method": "sets", "min_fraction": 1.5}, ValueError, "from 0 to 1"),
        ({"method": "sets", "barcodes_per_cell": 2}, TypeError, "two numbers"),
        (
            {"method": "sets", "barcodes_per_cell": (3, 2)},
            ValueError,
            "more than its most",
        ),
        ({"method": "sets", "min_size": 0}, ValueError, "at least 1"),
        ({"method": "sets", "allowlist": True}, TypeError, "sequence of barcodes"),
        (
            {"method": "sets", "allowlist": np.True_},
            TypeError,
            "sequence of barcodes",
        ),
        ({"method": "sets", "allowlist": "ACGT"}, TypeError, "sequence of barcodes"),
        ({"method": "sets", "prune": 1}, ValueError, "change nothing"),
    ],
)
def test_settings_name_what_is_missing_or_wrong(arguments, error, message) -> None:
    preset = arguments.pop("preset", None)
    with pytest.raises(error, match=message):
        barcode_call_settings(preset, **arguments)


def test_edits_and_substitutions_correct_within_any_distance() -> None:
    # A diagnostic never refuses a call: edits beyond the eight within which
    # chance matches are counted correct as nearer ones do.
    assert barcode_call_settings(method="sets", max_distance=9, metric="levenshtein")
    assert barcode_call_settings(method="sets", max_distance=30, metric="hamming")


def test_edits_beyond_eight_merge_with_no_expectation_of_chance_matches() -> None:
    # Every third of A's first 27 bases replaced by its complement: nine
    # substitutions, which no shift of A's period shortens, so the two are
    # nine edits apart and merge within nine edits but not within eight.
    complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
    far = "".join(
        complement[base] if position % 3 == 0 and position < 27 else base
        for position, base in enumerate(A)
    )
    assert levenshtein_distance(A, far) == 9
    rows = _molecules(0, A, 5) + _molecules(1, far, 1)
    near = _call(_records(rows), 2, method="sets", max_distance=8, metric="levenshtein")
    assert near.diagnostics["mergedBarcodes"] == 0
    assert near.diagnostics["expectedChanceMatches"] is not None
    calls = _call(
        _records(rows), 2, method="sets", max_distance=9, metric="levenshtein"
    )
    assert calls.diagnostics["mergedBarcodes"] == 1
    assert calls.diagnostics["recordsCorrected"] == 1
    assert calls.diagnostics["distinctBarcodes"] == 2
    # Counting the chance matches within nine edits takes too long, so it is
    # not measured.
    assert calls.diagnostics["expectedChanceMatches"] is None
    assert calls.n_barcodes.tolist() == [1, 1]
    assert clone_partition(calls.calls.codes) == {frozenset({0, 1})}


def test_a_filter_of_every_cell_with_a_barcode_is_none() -> None:
    for bounds in ((0, None), (1, None)):
        assert (
            barcode_call_settings(
                method="sets", barcodes_per_cell=bounds
            ).barcodes_per_cell
            is None
        )
    assert barcode_call_settings(
        method="sets", barcodes_per_cell=(2, None)
    ).barcodes_per_cell == (2, None)


def test_an_allowlist_holds_distinct_text() -> None:
    assert allowlist_texts([B, A, B]).tolist() == [A, B]
    with pytest.raises(ValueError, match="holds no barcode"):
        allowlist_texts([])
    with pytest.raises(TypeError, match="as text"):
        allowlist_texts([A, 3])
    with pytest.raises(TypeError, match="sequence of barcodes"):
        allowlist_texts(A)


def test_clones_are_labelled_by_rank() -> None:
    assert clone_labels(3).tolist() == ["clone_1", "clone_2", "clone_3"]


# Molecule filters.


def test_molecules_of_fewer_reads_than_min_reads_are_dropped() -> None:
    rows = (
        _molecules(0, A, 3, reads=10)
        + _molecules(1, A, 3, reads=9)
        + _molecules(2, A, 2, reads=10)
        + _molecules(2, A, 1, reads=9)
    )
    calls = _call(_records(rows), 3, method="sets", min_reads=10)
    # Cell 1's molecules all fall short; cell 2 keeps two of three.
    assert calls.n_barcodes.tolist() == [1, 0, 1]
    assert calls.diagnostics["moleculesBelowMinReads"] == 4


def test_min_reads_refuses_molecules_without_reads_and_libraries_of_pairs() -> None:
    rows = _molecules(0, A, 2) + [(0, 0, B, "x", 1, -1)]
    with pytest.raises(ValueError, match="'lib' counts none for 1 of its molecules"):
        _call(_records(rows), 1, method="sets", min_reads=2)
    pairs = [(0, 0, A, None, 4, 10)]
    with pytest.raises(ValueError, match="every library of the barcodes holds pairs"):
        _call(_records(pairs, (("p", "pairs"),)), 1, method="sets", min_reads=2)
    # A library of pairs beside molecules passes the molecule filter whole.
    mixed = _molecules(0, A, 2, reads=1) + [(1, 1, B, None, 4, 1)]
    calls = _call(
        _records(mixed, (("rna", "molecules"), ("atac", "pairs"))),
        2,
        method="sets",
        min_reads=2,
    )
    assert calls.n_barcodes.tolist() == [0, 1]


# Correction.


def test_an_error_merges_into_the_more_abundant_barcode() -> None:
    rows = _molecules(0, A, 5) + _molecules(1, A1, 1) + _molecules(2, A, 2)
    calls = _call(_records(rows), 3, method="sets", max_distance=1, metric="hamming")
    assert clone_partition(calls.calls.codes) == {frozenset({0, 1, 2})}
    assert calls.diagnostics["mergedBarcodes"] == 1
    assert calls.diagnostics["recordsCorrected"] == 1


def test_equally_abundant_barcodes_merge_into_the_one_that_sorts_last() -> None:
    # Z sorts after A, so cells of either end with Z, as TREX picks the larger.
    later = "C" + A[1:]
    assert later > A
    rows = _molecules(0, A, 2) + _molecules(1, later, 2)
    records = _records(rows)
    calls = _call(
        records, 2, method="sets", max_distance=1, metric="hamming", allowlist=[later]
    )
    # Only the barcode that the tie kept passes an allowlist of it.
    assert calls.n_barcodes.tolist() == [1, 1]
    calls = _call(
        records, 2, method="sets", max_distance=1, metric="hamming", allowlist=[A]
    )
    assert calls.n_barcodes.tolist() == [0, 0]
    # One more molecule of A makes it the more abundant, whatever its order.
    rows += _molecules(2, A, 1)
    calls = _call(
        _records(rows),
        3,
        method="sets",
        max_distance=1,
        metric="hamming",
        allowlist=[A],
    )
    assert calls.n_barcodes.tolist() == [1, 1, 1]


def test_a_chain_of_close_barcodes_merges_into_one_component() -> None:
    # A and A2 are two substitutions apart, but each is one from A1, so
    # within one substitution the three are one component, merged into A.
    rows = _molecules(0, A, 3) + _molecules(1, A1, 1) + _molecules(2, A2, 2)
    calls = _call(_records(rows), 3, method="sets", max_distance=1, metric="hamming")
    assert clone_partition(calls.calls.codes) == {frozenset({0, 1, 2})}
    assert calls.diagnostics["mergedBarcodes"] == 2


def test_correction_counts_the_distinct_barcodes_of_each_block_and_those_it_merges() -> (
    None
):
    # A, its errors A1 and A2, and B in one library; A and C in another. A1
    # and A2 merge into A within the first library, while the second
    # library's A is a barcode of its own block.
    rows = (
        _molecules(0, A, 5)
        + _molecules(0, A1, 1)
        + _molecules(1, A, 3)
        + _molecules(1, A2, 1)
        + _molecules(1, B, 4)
        + _molecules(2, A, 4, library=1)
        + _molecules(2, C, 4, library=1)
    )
    libraries = (("lib", "molecules"), ("other", "molecules"))
    records = _records(rows, libraries)
    corrected = _call(records, 3, method="sets", max_distance=3, metric="hamming")
    assert corrected.diagnostics["distinctBarcodes"] == 6
    assert corrected.diagnostics["mergedBarcodes"] == 2
    assert corrected.diagnostics["mergedBarcodes"] == 2
    # Groups split the blocks too: A counts once in each group of the first
    # library, beside A1 in one and A2 and B in the other.
    within = _call(
        records,
        3,
        groups=np.asarray([0, 1, 0]),
        method="sets",
        max_distance=3,
        metric="hamming",
    )
    assert within.diagnostics["distinctBarcodes"] == 7
    assert within.diagnostics["mergedBarcodes"] == 2
    # Without correction, the same barcodes are counted, and none is merged
    # or expected to match.
    plain = _call(records, 3, method="sets")
    assert plain.diagnostics["distinctBarcodes"] == 6
    assert plain.diagnostics["mergedBarcodes"] == 0
    assert plain.diagnostics["expectedChanceMatches"] == 0.0
    # A barcode whose only molecule falls below min_reads is never compared.
    shallow = _records(rows + [(3, 0, C, "U3_C", 1, 1)], libraries)
    for options in ({}, {"max_distance": 3, "metric": "hamming"}):
        counted = _call(shallow, 4, method="sets", **options)
        filtered = _call(shallow, 4, method="sets", min_reads=2, **options)
        assert counted.diagnostics["distinctBarcodes"] == 7
        assert filtered.diagnostics["distinctBarcodes"] == 6


@pytest.mark.parametrize("distance", [None, 0])
def test_trex_counts_the_cloneids_it_compares(distance) -> None:
    # A cloneID of low complexity is dropped before correction, and one of
    # 19 bases is never compared, since TREX compares 20: neither counts.
    low = "A" * 29 + "C"
    short = T[:19] + "-" * 11
    rows = (
        _molecules(0, T, 3)
        + _molecules(1, low, 3)
        + _molecules(2, T, 2)
        + _molecules(3, short, 2)
    )
    calls = _call(_records(rows), 4, preset="trex", max_distance=distance)
    assert calls.diagnostics["moleculesOfLowComplexity"] == 3
    assert calls.diagnostics["distinctBarcodes"] == 1
    assert calls.diagnostics["mergedBarcodes"] == 0


def test_a_doublet_is_not_among_the_cells_given_a_clone() -> None:
    # Two clones of three cells each, and a cell that holds the barcodes of
    # both: it links them alone, so TREX's rule removes it as a doublet, and
    # the largest clone holds 3 of the 6 cells given a clone.
    rows = []
    for cell in range(3):
        rows += _molecules(cell, A, 3) + _molecules(cell, B, 3)
    for cell in range(3, 6):
        rows += _molecules(cell, C, 3) + _molecules(cell, T, 3)
    for barcode in (A, B, C, T):
        rows += _molecules(6, barcode, 3)
    calls = _call(
        _records(rows),
        7,
        method="jaccard",
        similarity=0.4,
        inclusive=False,
        prune=False,
        doublets=True,
    )
    assert calls.calls.doublets.tolist() == [False] * 6 + [True]
    assert _largest_share(calls) == 3 / 6


def test_the_largest_clone_shares_the_cells_given_a_clone() -> None:
    rows = (
        _molecules(0, A, 3)
        + _molecules(1, A, 3)
        + _molecules(2, A, 3)
        + _molecules(3, B, 3)
        + _molecules(4, B, 3)
        + _molecules(5, C, 3)
    )
    # A's clone holds 3 of the 5 cells given a clone; C's cell is in a clone
    # too small to keep, and cell 6 holds no barcode.
    calls = _call(_records(rows), 7, method="sets", min_size=2)
    assert _largest_share(calls) == 3 / 5
    # With no cell given a clone, the share is 0.
    none = _call(_records(rows), 7, method="sets", min_size=4)
    assert _largest_share(none) == 0.0
    assert _call(_records([]), 2, method="sets").diagnostics == {
        "distinctBarcodes": 0,
        "expectedChanceMatches": 0.0,
        "mergedBarcodes": 0,
    }


def _random_cloneids(rng, n_texts: int) -> list[str]:
    """Distinct cloneIDs of 24 bases from three letters, a third cut at one end."""
    texts = set()
    for _ in range(n_texts):
        text = "".join(rng.choice(list("ACG"), size=24))
        if rng.random() < 1 / 3:
            cut = int(rng.integers(1, 5))
            text = (
                "-" * cut + text[cut:]
                if rng.random() < 0.5
                else text[:-cut] + "-" * cut
            )
        texts.add(text)
    return sorted(texts)


def _linked(n_texts: int, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Each text's component of linked pairs, by union and find."""
    parent = list(range(n_texts))

    def root(text: int) -> int:
        while parent[text] != text:
            parent[text] = parent[parent[text]]
            text = parent[text]
        return text

    for one, other in zip(first.tolist(), second.tolist(), strict=True):
        parent[root(one)] = root(other)
    return np.asarray([root(text) for text in range(n_texts)])


@pytest.mark.parametrize("preset", ["trex", "celltag_multi"])
def test_the_expected_matches_are_those_of_each_library_as_correction_links_it(
    preset,
) -> None:
    # Each library's barcodes are compared alone, with TREX's characters of
    # no base and its 20 compared bases, or within 2 edits; the barcodes that
    # correction links are one component, which its barcode that sorts last
    # represents, since every barcode holds one molecule.
    rng = np.random.default_rng(12)
    if preset == "trex":
        texts = [_random_cloneids(rng, 300), _random_cloneids(rng, 300)]
        search = {
            "metric": "hamming",
            "max_distance": 5,
            "unknown": "-0",
            "min_compared": 20,
        }
    else:
        texts = [
            sorted({"".join(rng.choice(list("ACGT"), size=8)) for _ in range(300)})
            for _library in range(2)
        ]
        search = {"metric": "levenshtein", "max_distance": 2}
    rows = [
        (cell, library, text, f"U{cell}", 1, 20)
        for library, barcodes in enumerate(texts)
        for cell, text in enumerate(barcodes, start=1_000 * library)
    ]
    libraries = (("one", "molecules"), ("two", "molecules"))
    options = (
        {"preset": "trex"}
        if preset == "trex"
        else {"preset": "celltag_multi", "allowlist": False, "min_reads": 0}
    )
    calls = _call(_records(rows, libraries), 2_000, **options)

    def expect(barcodes: list[str]) -> float:
        first, second = neighbor_pairs(barcodes, **search)
        return expected_chance_pairs(
            barcodes,
            **search,
            components=_linked(len(barcodes), first, second),
            priority=np.arange(len(barcodes)),
        )

    expected = calls.diagnostics["expectedChanceMatches"]
    assert expected == pytest.approx(sum(map(expect, texts)), rel=1e-9)
    assert expected > 0
    assert calls.diagnostics["distinctBarcodes"] == sum(map(len, texts))
    # Compared together, the two libraries would expect more.
    assert expect(sorted(texts[0] + texts[1])) > 1.5 * expected


def test_barcodes_without_errors_expect_as_if_each_were_its_own_component() -> None:
    # Random barcodes of 24 bases never lie within 2 substitutions of each
    # other, so each is a component of its own, and their frequencies are
    # those of every barcode compared.
    rng = np.random.default_rng(15)
    texts = sorted({"".join(rng.choice(list("ACGT"), size=24)) for _ in range(800)})
    rows = [(cell, 0, text, f"U{cell}", 1, 20) for cell, text in enumerate(texts)]
    calls = _call(
        _records(rows), len(texts), method="sets", max_distance=2, metric="hamming"
    )
    assert calls.diagnostics["mergedBarcodes"] == 0
    assert calls.diagnostics["expectedChanceMatches"] == pytest.approx(
        expected_chance_pairs(texts, "hamming", 2), rel=1e-12
    )


def test_error_variants_weigh_no_more_than_their_parents_in_the_expectation() -> None:
    # 600 random parents of 12 bases, one with the variants of 400 draws of
    # one or two substitutions and every fifth with one more, each parent in
    # five molecules and each variant in one, so that correction merges each
    # family into its parent. The parents are independent random barcodes, so
    # the pairs of parents within 3 substitutions are chance matches; their
    # count, scaled from the pairs of parents to every pair compared, is what
    # the expectation must give, within four standard deviations of the count.
    rng = np.random.default_rng(17)
    letters = np.asarray(list("ACGT"))
    parents = sorted({"".join(rng.choice(letters, size=12)) for _ in range(600)})

    def variant(text: str, changes: int) -> str:
        bases = list(text)
        for position in rng.choice(12, size=changes, replace=False):
            bases[position] = str(rng.choice(letters[letters != bases[position]]))
        return "".join(bases)

    barcodes = set(parents)
    for _ in range(400):
        barcodes.add(variant(parents[0], int(rng.integers(1, 3))))
    for index in range(1, len(parents), 5):
        barcodes.add(variant(parents[index], 1))
    texts = sorted(barcodes)
    rows = []
    for cell, text in enumerate(texts):
        reads = 5 if text in parents else 1
        rows += [(cell, 0, text, f"U{cell}_{k}", 1, 20) for k in range(reads)]
    calls = _call(
        _records(rows), len(texts), method="sets", max_distance=3, metric="hamming"
    )
    expected = calls.diagnostics["expectedChanceMatches"]
    # The pairs of parents within 3 substitutions, by brute force.
    codes = np.asarray([list(text) for text in parents])
    firsts, seconds = [], []
    for row in range(len(parents) - 1):
        linked = np.flatnonzero((codes[row + 1 :] != codes[row]).sum(axis=1) <= 3)
        firsts.append(np.full(len(linked), row))
        seconds.append(row + 1 + linked)
    count, variance = link_count_variance(
        len(parents), np.concatenate(firsts), np.concatenate(seconds)
    )
    scale = len(texts) * (len(texts) - 1) / (len(parents) * (len(parents) - 1))
    reference = count * scale
    tolerance = 4 * variance**0.5 * scale
    assert count > 50
    assert abs(expected - reference) <= tolerance, (expected, reference, tolerance)
    # The frequencies of every barcode compared lean towards the parent of
    # many variants, and expect far more.
    every_barcode = expected_chance_pairs(texts, "hamming", 3)
    assert every_barcode - reference > tolerance, (every_barcode, reference, tolerance)


def test_the_most_abundant_barcode_represents_its_component_in_the_expectation() -> (
    None
):
    # "AATAG" in five molecules and its variant "TATAG" in one, which sorts
    # after it, beside "TATGC" and "TGTGA" in five each. The abundant barcode
    # represents its family, as correction merges into it: two components
    # agree at the first, second and fourth positions with chance 1/3, at
    # the third surely and at the last never, so a pair of distinct barcodes
    # falls within one substitution with chance 1/27, and the four pair six
    # times. "TATAG" would make the first position agree surely, and the
    # chance 1/9.
    texts = ["AATAG", "TATAG", "TATGC", "TGTGA"]
    molecules = [5, 1, 5, 5]
    rows = [
        (cell, 0, text, f"U{cell}_{k}", 1, 20)
        for cell, (text, count) in enumerate(zip(texts, molecules, strict=True))
        for k in range(count)
    ]
    calls = _call(
        _records(rows), len(texts), method="sets", max_distance=1, metric="hamming"
    )
    assert calls.diagnostics["mergedBarcodes"] == 1
    assert calls.diagnostics["expectedChanceMatches"] == pytest.approx(
        6 / 27, rel=1e-12
    )
    assert expected_chance_pairs(
        texts, "hamming", 1, components=[0, 0, 1, 2]
    ) == pytest.approx(6 / 9, rel=1e-12)


def test_the_expectation_may_hold_what_the_search_could(monkeypatch) -> None:
    # Correction holds its barcodes and their texts until the expectation is
    # counted and lets the pairs go once the components are found, so the
    # expectation gets the room that the search got.
    rooms = {}

    def recorded(name, function):
        def call(*args, max_bytes=None, **kwargs):
            rooms[name] = max_bytes
            return function(*args, max_bytes=max_bytes, **kwargs)

        return call

    monkeypatch.setattr(
        barcode_clones,
        "neighbor_pairs",
        recorded("search", barcode_clones.neighbor_pairs),
    )
    monkeypatch.setattr(
        barcode_clones,
        "expected_chance_pairs",
        recorded("expectation", barcode_clones.expected_chance_pairs),
    )
    rows = _molecules(0, A, 4) + _molecules(1, A1, 2) + _molecules(2, B, 4)
    budget = 10**7
    calls = barcode_clone_calls(
        _records(rows),
        3,
        groups=np.zeros(3, dtype=np.int64),
        settings=barcode_call_settings(method="sets", max_distance=1, metric="hamming"),
        max_bytes=budget,
    )
    assert calls.diagnostics["mergedBarcodes"] == 1
    assert rooms["expectation"] == rooms["search"] < budget


def _family_rows(rng, parent: str, clone: int, first_cell: int) -> list[tuple]:
    """A cloneID in three molecules in each of 5 cells, and its variants once in them.

    The variants are those of 30 draws of one or two substitutions and of 30
    cuts of 1 to 8 bases at either end with at most one substitution.
    """
    letters = np.asarray(list("ACGT"))

    def substituted(text: str, changes: int) -> str:
        bases = list(text)
        for position in rng.choice(len(text), size=changes, replace=False):
            bases[position] = str(rng.choice(letters[letters != bases[position]]))
        return "".join(bases)

    cells = list(range(first_cell, first_cell + 5))
    rows = [
        (cell, 0, parent, f"P{clone}_{cell}_{k}", 1, 20)
        for cell in cells
        for k in range(3)
    ]
    variants = {substituted(parent, int(rng.integers(1, 3))) for _ in range(30)}
    for draw in range(30):
        cut = int(rng.integers(1, 9))
        text = substituted(parent, int(rng.integers(0, 2)))
        variants.add("-" * cut + text[cut:] if draw % 2 else text[:-cut] + "-" * cut)
    variants.discard(parent)
    for k, text in enumerate(sorted(variants)):
        rows.append((cells[k % 5], 0, text, f"V{clone}_{k}", 1, 2))
    return rows


def test_a_group_of_one_clone_adds_nothing_to_the_expectation() -> None:
    # Under TREX's rules, 30 clones of a cloneID of 30 bases and its variants
    # fill one group of `within`, and one more such clone a second group.
    # The second group is one component, whose cloneIDs are of no two
    # components, so the call expects what the 30 clones expect alone.
    rng = np.random.default_rng(11)
    letters = np.asarray(list("ACGT"))
    rows = []
    for clone in range(30):
        parent = "".join(rng.choice(letters, size=30))
        rows += _family_rows(rng, parent, clone, 5 * clone)
    alone = _call(_records(rows), 150, preset="trex")
    parent = "".join(rng.choice(letters, size=30))
    rows += _family_rows(rng, parent, 30, 150)
    both = _call(_records(rows), 155, groups=np.repeat([0, 1], [150, 5]), preset="trex")
    assert both.diagnostics["mergedBarcodes"] > alone.diagnostics["mergedBarcodes"]
    assert alone.diagnostics["expectedChanceMatches"] > 0
    assert both.diagnostics["expectedChanceMatches"] == pytest.approx(
        alone.diagnostics["expectedChanceMatches"], rel=1e-12
    )


def test_barcodes_that_correction_chains_into_one_component_expect_nothing() -> None:
    # 766 random barcodes of 6 or 7 bases lie within 2 edits of so many
    # others that correction chains them all into one component, and every
    # cell into one clone. The expectation measures chance merging among
    # many components and expects nothing of one; the largest clone's share
    # shows the chaining.
    rng = np.random.default_rng(23)
    letters = np.asarray(list("ACGT"))
    texts = sorted(
        {"".join(rng.choice(letters, size=int(rng.integers(6, 8)))) for _ in range(800)}
    )
    rows = [(cell, 0, text, f"U{cell}", 1, 20) for cell, text in enumerate(texts)]
    calls = _call(
        _records(rows), len(texts), method="sets", max_distance=2, metric="levenshtein"
    )
    assert calls.diagnostics["mergedBarcodes"] == len(texts) - 1
    assert calls.diagnostics["expectedChanceMatches"] == 0
    assert _largest_share(calls) == 1.0


def test_counting_the_distinct_barcodes_without_correction_checks_the_budget() -> None:
    rows = _molecules(0, A, 4) + _molecules(1, B, 4) + _molecules(2, C, 4)
    records = _records(rows)
    settings = barcode_call_settings(method="sets")
    groups = np.zeros(3, dtype=np.int64)
    # Every rule holds 40 bytes per cell and 18 per record; counting the
    # distinct barcodes, 48 more per record kept.
    counting = 40 * 3 + 18 * 12 + 48 * 12
    with pytest.raises(MemoryError, match="counting the distinct barcodes of 12"):
        barcode_clone_calls(
            records, 3, groups=groups, settings=settings, max_bytes=counting - 1
        )
    with pytest.raises(MemoryError) as refused:
        barcode_clone_calls(
            records, 3, groups=groups, settings=settings, max_bytes=counting
        )
    assert "counting the distinct" not in str(refused.value)


def test_barcodes_of_two_groups_or_libraries_never_merge() -> None:
    rows = _molecules(0, A, 5) + _molecules(1, A1, 1)
    groups = np.asarray([0, 1])
    calls = _call(
        _records(rows),
        2,
        groups=groups,
        method="sets",
        max_distance=1,
        metric="hamming",
    )
    assert calls.diagnostics["mergedBarcodes"] == 0
    two = _molecules(0, A, 5) + _molecules(1, A1, 1, library=1)
    calls = _call(
        _records(two, (("a", "molecules"), ("b", "molecules"))),
        2,
        method="sets",
        max_distance=1,
        metric="hamming",
    )
    assert calls.diagnostics["mergedBarcodes"] == 0


def test_edits_merge_an_indel_that_substitutions_cannot() -> None:
    shifted = A[:5] + A[6:] + "G"
    rows = _molecules(0, A, 5) + _molecules(1, shifted, 1)
    for metric, merged in (("hamming", 0), ("levenshtein", 1)):
        calls = _call(_records(rows), 2, method="sets", max_distance=2, metric=metric)
        assert calls.diagnostics["mergedBarcodes"] == merged


def test_molecules_that_correction_gives_one_barcode_and_umi_are_one() -> None:
    # One UMI read under two barcodes is one molecule once both are A.
    rows = [(0, 0, A, "u", 1, 3), (0, 0, A1, "u", 1, 2)] + _molecules(1, A, 4)
    calls = _call(
        _records(rows), 2, method="sets", max_distance=1, metric="hamming", min_umis=2
    )
    # Cell 0 holds one molecule of A, below two UMIs.
    assert calls.n_barcodes.tolist() == [0, 1]
    calls = _call(_records(rows), 2, method="sets", max_distance=1, metric="hamming")
    assert calls.n_barcodes.tolist() == [1, 1]


def test_molecules_rank_by_their_umis_whatever_their_reads() -> None:
    # A has three molecules of one read; A1 one molecule of 100 reads. In a
    # library of molecules, each molecule is one UMI, so A is the more
    # abundant, and only an allowlist of A keeps the merged barcodes.
    rows = _molecules(0, A, 3, reads=1) + _molecules(1, A1, 1, reads=100)
    for allowed, kept in (([A], [1, 1]), ([A1], [0, 0])):
        calls = _call(
            _records(rows),
            2,
            method="sets",
            max_distance=1,
            metric="hamming",
            allowlist=allowed,
        )
        assert calls.n_barcodes.tolist() == kept
    assert calls.units == ("umis",)


def test_pairs_of_a_library_without_umis_rank_by_reads() -> None:
    # B has fewer pairs but more reads than A1 in a library counting reads.
    rows = [(0, 0, A1, None, -1, 3), (1, 0, A1, None, -1, 3), (2, 0, A, None, -1, 10)]
    calls = _call(
        _records(rows, (("atac", "pairs"),)),
        3,
        method="sets",
        max_distance=1,
        metric="hamming",
    )
    assert calls.units == ("reads",)
    assert calls.diagnostics["mergedBarcodes"] == 1
    assert clone_partition(calls.calls.codes) == {frozenset({0, 1, 2})}


def test_correction_refuses_a_library_whose_counts_are_partly_unknown() -> None:
    rows = [(0, 0, A, None, 4, 10), (1, 0, A1, None, -1, 3)]
    with pytest.raises(ValueError, match=r"'q' counts UMIs for 1 of its 2 pairs"):
        _call(
            _records(rows, (("q", "pairs"),)),
            2,
            method="sets",
            max_distance=1,
            metric="hamming",
        )
    # Without correction or count filters, the counts are never read.
    calls = _call(_records(rows, (("q", "pairs"),)), 2, method="sets")
    assert calls.units == (None,)
    assert calls.n_barcodes.tolist() == [1, 1]


# The allowlist.


def test_barcodes_outside_the_allowlist_are_dropped_after_correction() -> None:
    rows = _molecules(0, A, 5) + _molecules(1, A1, 1) + _molecules(2, B, 3)
    calls = _call(
        _records(rows),
        3,
        allowlist=[A],
        method="sets",
        max_distance=1,
        metric="hamming",
    )
    # A1 becomes A before the allowlist is read, so only B goes.
    assert calls.n_barcodes.tolist() == [1, 1, 0]
    assert calls.diagnostics["recordsOutsideAllowlist"] == 3


# Pair filters.


def test_pairs_of_fewer_umis_than_min_umis_are_dropped() -> None:
    rows = _molecules(0, A, 2) + _molecules(0, B, 1) + _molecules(1, B, 2)
    calls = _call(_records(rows), 2, method="sets", min_umis=2)
    assert calls.n_barcodes.tolist() == [1, 1]
    assert calls.diagnostics["pairsBelowMinUmis"] == 1
    # In a library that counts reads alone, the count is reads.
    pairs = [(0, 0, A, None, -1, 2), (0, 0, B, None, -1, 1)]
    calls = _call(_records(pairs, (("atac", "pairs"),)), 1, method="sets", min_umis=2)
    assert calls.n_barcodes.tolist() == [1]
    # A library that counts both is read in UMIs: A's one UMI of ten reads
    # goes, and B's three UMIs of three reads stay.
    both = [(0, 0, A, None, 1, 10), (1, 0, B, None, 3, 3)]
    calls = _call(_records(both, (("bartab", "pairs"),)), 2, method="sets", min_umis=2)
    assert calls.units == ("umis",)
    assert calls.n_barcodes.tolist() == [0, 1]


def test_min_fraction_keeps_pairs_at_bartabs_share_of_the_top_barcode() -> None:
    # BARtab keeps a barcode whose count divided by the fraction reaches the
    # cell's top count: 3 / 0.3 is 10.000000000000002, 2 / 0.3 is 6.67.
    rows = _molecules(0, A, 10) + _molecules(0, B, 3) + _molecules(0, C, 2)
    calls = _call(_records(rows), 1, method="sets", min_fraction=0.3)
    assert calls.n_barcodes.tolist() == [2]
    assert calls.diagnostics["pairsBelowMinFraction"] == 1
    # BARtab divides by the fraction: 17 / 0.34 is 49.99999999999999, below
    # a top of 50, so the barcode goes, though 0.34 * 50 is exactly 17.
    rows = _molecules(0, A, 50) + _molecules(0, B, 17)
    calls = _call(_records(rows), 1, method="sets", min_fraction=0.34)
    assert calls.n_barcodes.tolist() == [1]
    # Each library's top barcode is its own.
    two = _molecules(0, A, 10) + _molecules(0, B, 2, library=1)
    calls = _call(
        _records(two, (("a", "molecules"), ("b", "molecules"))),
        1,
        method="sets",
        min_fraction=0.3,
    )
    assert calls.n_barcodes.tolist() == [2]


def test_pair_filters_refuse_libraries_without_a_unit() -> None:
    rows = [(0, 0, A, None, 4, -1), (0, 0, B, None, -1, -1)]
    with pytest.raises(ValueError, match="min_umis counts each barcode"):
        _call(_records(rows, (("w", "pairs"),)), 1, method="sets", min_umis=2)
    with pytest.raises(ValueError, match="min_fraction counts each barcode"):
        _call(_records(rows, (("w", "pairs"),)), 1, method="sets", min_fraction=0.2)


# Cell filters.


def test_cells_outside_barcodes_per_cell_get_no_clone() -> None:
    rows = []
    for cell, count in enumerate((1, 2, 3, 4)):
        for barcode in (A, B, C, A1)[:count]:
            rows += _molecules(cell, barcode, 2)
    calls = _call(_records(rows), 4, method="sets", barcodes_per_cell=(2, 3))
    assert calls.n_barcodes.tolist() == [1, 2, 3, 4]
    assert (calls.calls.codes >= 0).tolist() == [False, True, True, False]
    assert calls.diagnostics["cellsOutsideBarcodesPerCell"] == 2


# TREX's rules.

T = "ACGTTGCAACGTTGCAACGTTGCAACGTTG"


def test_trex_drops_cloneids_of_low_complexity_and_few_bases() -> None:
    low = "A" * 29 + "G"
    # Two letters in equal parts are 1 bit exactly, which TREX keeps.
    even = "AC" * 15
    partial = T[:19] + "-" * 11
    rows = (
        _molecules(0, T, 2, reads=3)
        + _molecules(1, low, 2, reads=3)
        + _molecules(2, even, 2, reads=3)
        + _molecules(3, partial, 2, reads=3)
        + _molecules(4, T[:20] + "-" * 10, 2, reads=3)
    )
    calls = _call(_records(rows), 5, preset="trex", max_distance=0)
    assert calls.n_barcodes.tolist() == [1, 0, 1, 0, 1]
    assert calls.diagnostics["moleculesOfLowComplexity"] == 2
    assert calls.diagnostics["moleculesOfFewBases"] == 2


def test_trex_merges_partial_cloneids_over_the_bases_both_hold() -> None:
    # 24 shared bases match; 19 would be too few to compare.
    rows = _molecules(0, T, 5, reads=3) + _molecules(1, T[:24] + "-" * 6, 2, reads=3)
    calls = _call(_records(rows), 2, preset="trex")
    assert calls.diagnostics["mergedBarcodes"] == 1
    assert clone_partition(calls.calls.codes) == {frozenset({0, 1})}
    rows = _molecules(0, T, 5, reads=3) + _molecules(1, "-" * 11 + T[11:], 2, reads=3)
    calls = _call(_records(rows), 2, preset="trex")
    assert calls.diagnostics["mergedBarcodes"] == 0


def test_trex_drops_one_umi_cloneids_that_another_cell_holds_or_one_read_supports() -> (
    None
):
    U = "TGCATGCATGCAACGTACGTTTTTGGGGCC"
    V = "GGGGCCCCAAAATTTTGCGCGCGCATATAT"
    rows = (
        _molecules(0, T, 3, reads=3)
        # One UMI of T in cell 1: T is in cell 0, so it goes.
        + [(1, 0, T, "u1", 1, 5)]
        # One UMI of U, held nowhere else, read twice: kept.
        + [(2, 0, U, "u2", 1, 2)]
        # One UMI of V, held nowhere else, read once: dropped.
        + [(3, 0, V, "u3", 1, 1)]
    )
    calls = _call(_records(rows), 4, preset="trex", max_distance=0)
    assert calls.n_barcodes.tolist() == [1, 0, 1, 0]
    assert calls.diagnostics["pairsOfOneUmiShared"] == 1
    assert calls.diagnostics["pairsOfOneRead"] == 1


def test_trex_tests_a_one_umi_cloneid_against_the_cells_of_its_group_and_library() -> (
    None
):
    # Cell 1's one UMI of T is shared with cell 0 only when both are in one
    # group and one library.
    rows = _molecules(0, T, 3, reads=3) + [(1, 0, T, "u1", 1, 5)]
    shared = _call(_records(rows), 2, preset="trex", max_distance=0)
    assert shared.n_barcodes.tolist() == [1, 0]
    apart = _call(
        _records(rows), 2, groups=np.asarray([0, 1]), preset="trex", max_distance=0
    )
    assert apart.n_barcodes.tolist() == [1, 1]
    assert apart.diagnostics["pairsOfOneUmiShared"] == 0
    other_library = _molecules(0, T, 3, reads=3) + [(1, 1, T, "u1", 1, 5)]
    libraries = (("a", "molecules"), ("b", "molecules"))
    calls = _call(_records(other_library, libraries), 2, preset="trex", max_distance=0)
    assert calls.n_barcodes.tolist() == [1, 1]


def test_trex_rules_follow_the_preset_unless_given() -> None:
    # None follows the preset; True or False sets the rules in any call, and
    # the rest of the preset stays.
    trex = barcode_call_settings("trex")
    assert trex.trex_rules is True
    assert barcode_call_settings("trex", trex_rules=np.True_).trex_rules is True
    without = barcode_call_settings("trex", trex_rules=False)
    assert without.record() == {**trex.record(), "trex_rules": False}
    assert barcode_call_settings("larry").trex_rules is False
    assert barcode_call_settings("larry", trex_rules=True).record() == {
        **barcode_call_settings("larry").record(),
        "trex_rules": True,
    }
    assert barcode_call_settings(method="sets").trex_rules is False
    assert barcode_call_settings(method="sets", trex_rules=True).trex_rules is True


def test_trex_rules_turn_on_and_off_whatever_the_preset() -> None:
    # A cloneID of low complexity, a one-UMI cloneID held by another cell
    # and one supported by a single read: TREX's rules drop all three.
    low = "A" * 29 + "C"
    V = "GGGGCCCCAAAATTTTGCGCGCGCATATAT"
    rows = (
        _molecules(0, T, 3, reads=3)
        + _molecules(1, low, 3, reads=3)
        + [(2, 0, T, "u2", 1, 5)]
        + [(3, 0, V, "u3", 1, 1)]
    )
    with_rules = _call(_records(rows), 4, preset="trex", max_distance=0)
    assert with_rules.n_barcodes.tolist() == [1, 0, 0, 0]
    without = _call(_records(rows), 4, preset="trex", max_distance=0, trex_rules=False)
    assert without.n_barcodes.tolist() == [1, 1, 1, 1]
    # A rule that does not run counts nothing, and says so with None.
    assert without.diagnostics["moleculesOfLowComplexity"] is None
    assert with_rules.diagnostics["moleculesOfLowComplexity"] == 1
    # The rules alone, beside LARRY's other rules, drop the same barcodes.
    larry = _call(
        _records(rows),
        4,
        preset="larry",
        min_reads=0,
        min_umis=0,
        max_distance=0,
        trex_rules=True,
    )
    assert larry.n_barcodes.tolist() == [1, 0, 0, 0]
    assert larry.diagnostics["moleculesOfLowComplexity"] == 3
    assert larry.diagnostics["pairsOfOneUmiShared"] == 1
    assert larry.diagnostics["pairsOfOneRead"] == 1


def test_trex_rules_need_molecules_and_their_reads() -> None:
    pairs = [(0, 0, T, None, 3, 9)]
    with pytest.raises(ValueError, match="TREX's rules apply to molecules"):
        _call(_records(pairs, (("p", "pairs"),)), 1, preset="trex")
    # The rules need molecules whichever call turns them on.
    with pytest.raises(ValueError, match="TREX's rules apply to molecules"):
        _call(_records(pairs, (("p", "pairs"),)), 1, method="sets", trex_rules=True)
    assert (
        _call(
            _records(pairs, (("p", "pairs"),)), 1, preset="trex", trex_rules=False
        ).calls.n_clones
        == 1
    )
    # Two of the three molecules of the library count no reads; the route
    # from an earlier run's reads.txt is named.
    rows = _molecules(0, T, 2, reads=-1) + _molecules(1, T, 1, reads=4)
    with pytest.raises(
        ValueError, match=r"2 of the 3 molecules of library 'lib' count no reads"
    ) as refused:
        _call(_records(rows), 2, preset="trex")
    assert "format='reads'" in str(refused.value)
    assert "'barcode': 'clone_id'" in str(refused.value)


def test_the_settings_and_the_allowlist_go_together() -> None:
    settings = barcode_call_settings(method="sets")
    with pytest.raises(ValueError, match="exactly when the settings use one"):
        barcode_clone_calls(
            _records(_molecules(0, A, 1)),
            1,
            groups=np.zeros(1, dtype=np.int64),
            settings=settings,
            allowlist=allowlist_texts([A]),
        )
    with pytest.raises(ValueError, match="one code per cell"):
        barcode_clone_calls(
            _records(_molecules(0, A, 1)),
            1,
            groups=np.zeros(2, dtype=np.int64),
            settings=settings,
        )


def test_a_similarity_across_libraries_compares_cells_of_two_libraries() -> None:
    # Cell 0 (RNA) and cell 1 (ATAC) share 11 of 20 barcodes, 0.55.
    texts = [f"{index:028b}".replace("0", "A").replace("1", "C") for index in range(20)]
    rows = []
    for barcode in texts[:15]:
        rows += _molecules(0, barcode, 2)
    for barcode in texts[4:20]:
        rows += [(1, 1, barcode, None, -1, 5)]
    records = _records(rows, (("rna", "molecules"), ("atac", "pairs")))
    across = _call(records, 2, method="jaccard", similarity=(0.6, 0.5), inclusive=False)
    assert clone_partition(across.calls.codes) == {frozenset({0, 1})}
    within = _call(records, 2, method="jaccard", similarity=0.6, inclusive=False)
    assert clone_partition(within.calls.codes) == {frozenset({0}), frozenset({1})}


@pytest.mark.parametrize(
    "options",
    [
        {"preset": "larry"},
        {"preset": "trex"},
        {"preset": "celltag", "allowlist": False},
        {"preset": "celltag_multi", "allowlist": False},
        {"method": "sets", "min_fraction": 0.3},
    ],
)
def test_records_of_no_barcode_call_no_clone(options) -> None:
    # A molecule library without records of the selected cells passes the
    # read filter, since it holds molecules.
    records = _records([], (("lib", "molecules"),))
    calls = _call(records, 3, **options)
    assert calls.calls.n_clones == 0
    assert calls.n_barcodes.tolist() == [0, 0, 0]
    assert (
        calls.diagnostics["pairs"] == 0 and calls.diagnostics["cellsWithBarcodes"] == 0
    )


def _synthetic_records(
    n_cells: int, *, molecules: bool, seed: int = 0
) -> BarcodeRecords:
    """Clones of up to nine cells with five barcodes each, errors and ambient barcodes."""
    rng = np.random.default_rng(seed)
    letters = np.frombuffer(b"ACGT", dtype=np.uint8)

    def texts(count: int) -> list[str]:
        drawn = letters[rng.integers(0, 4, size=(count, 28))]
        return np.ascontiguousarray(drawn).view("S28").reshape(-1).astype(str).tolist()

    clone_of = np.repeat(np.arange(n_cells), rng.integers(1, 10, size=n_cells))[
        :n_cells
    ]
    clone_barcodes = np.asarray(texts(5 * (int(clone_of.max()) + 1)), dtype=object)
    clone_barcodes = clone_barcodes.reshape(-1, 5)
    rows = []
    for cell in range(n_cells):
        lost = int(rng.integers(0, 5)) if rng.random() < 0.3 else -1
        for slot, barcode in enumerate(clone_barcodes[clone_of[cell]]):
            if slot == lost:
                continue
            if molecules:
                rows += [
                    (
                        cell,
                        0,
                        barcode,
                        f"u{cell}_{slot}_{k}",
                        1,
                        int(rng.integers(10, 31)),
                    )
                    for k in range(3)
                ]
            else:
                rows.append((cell, 0, barcode, None, int(rng.integers(3, 10)), -1))
        if cell % 2 == 0:
            source = clone_barcodes[clone_of[cell], 0]
            error = source[:-1] + ("C" if source[-1] != "C" else "G")
            rows.append(
                (cell, 0, error, f"e{cell}", 1, 12)
                if molecules
                else (cell, 0, error, None, 1, -1)
            )
        if cell % 10 == 0:
            ambient = clone_barcodes[int(rng.integers(0, len(clone_barcodes))), 1]
            rows.append(
                (cell, 0, ambient, f"a{cell}", 1, 1)
                if molecules
                else (cell, 0, ambient, None, 1, -1)
            )
    library = ("lib", "molecules" if molecules else "pairs")
    return _records(rows, (library,))


@pytest.mark.slow
@pytest.mark.parametrize(
    ("molecules", "options"),
    [
        (True, {"preset": "larry"}),
        (True, {"preset": "trex"}),
        (True, {"preset": "trex", "max_distance": 0}),
        (False, {"preset": "celltag", "allowlist": False}),
        (False, {"preset": "celltag_multi", "allowlist": False, "min_reads": 0}),
        (False, {"method": "sets", "min_fraction": 0.3}),
    ],
)
def test_calls_hold_no_more_than_the_smallest_budget_they_accept(
    molecules, options
) -> None:
    records = _synthetic_records(4_000, molecules=molecules)
    settings = barcode_call_settings(**options)
    groups = np.zeros(4_000, dtype=np.int64)

    def call(budget: int) -> object:
        return barcode_clone_calls(
            records, 4_000, groups=groups, settings=settings, max_bytes=budget
        )

    budget = smallest_budget(call, 1 << 30)
    _, peak = traced_peak(lambda: call(budget), budget)
    assert peak <= budget
    with pytest.raises(MemoryError, match="(?i)raise"):
        call(budget - 1)


def test_a_similarity_across_libraries_tells_apart_at_most_62_libraries() -> None:
    libraries = tuple((f"lib{index}", "molecules") for index in range(63))
    rows = [(index, index, A, f"u{index}", 1, 20) for index in range(63)]
    with pytest.raises(ValueError, match="at most 62 libraries"):
        _call(
            _records(rows, libraries),
            63,
            method="jaccard",
            similarity=(0.6, 0.5),
            inclusive=False,
        )
    # One similarity compares any number of libraries.
    calls = _call(
        _records(rows, libraries), 63, method="jaccard", similarity=0.6, inclusive=False
    )
    assert calls.calls.n_clones == 1
