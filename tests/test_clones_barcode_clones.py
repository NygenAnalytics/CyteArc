"""The rules of barcode clone calling on hand-built records.

Records are written out molecule by molecule, and what each rule keeps,
merges or refuses is derived by hand from the rule's published statement.
"""

import numpy as np
import pytest

from cytearc.clones.barcode_clones import (
    PRESETS,
    BarcodeRecords,
    allowlist_texts,
    barcode_call_settings,
    barcode_clone_calls,
    clone_labels,
)
from cytearc.metadata.records import pack_text

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
        allowlist=None if allowlist is None else allowlist_texts(allowlist),
    )


def _partition(codes) -> set[frozenset[int]]:
    return {
        frozenset(np.flatnonzero(codes == clone).tolist())
        for clone in range(int(codes.max()) + 1)
    }


# The published rules of each preset, as INPUT_FORMATS states them.
PUBLISHED = {
    "larry": {
        "min_reads": 10,
        "distance": 3,
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
        "distance": 5,
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
        "distance": 0,
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
        "distance": 2,
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


@pytest.mark.parametrize("preset", list(PRESETS))
def test_each_preset_resolves_to_its_published_rules(preset) -> None:
    allowlist = [A] if PRESETS[preset].get("allowlist") else None
    settings = barcode_call_settings(preset, allowlist=allowlist)
    assert settings.record() == {"preset": preset, **PUBLISHED[preset]}
    assert settings.allowlist == (allowlist is not None)


def test_given_values_replace_a_presets_and_zero_turns_a_filter_off() -> None:
    settings = barcode_call_settings("larry", min_reads=0, min_umis=5, distance=0)
    record = settings.record()
    assert (record["min_reads"], record["min_umis"], record["distance"]) == (0, 5, 0)
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


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({}, ValueError, "needs a preset or a method"),
        ({"method": "jaccard"}, ValueError, "needs similarity"),
        ({"method": "jaccard", "similarity": 0.7}, ValueError, "needs inclusive"),
        ({"method": "sets", "distance": 2}, ValueError, "needs a metric"),
        ({"method": "sets", "metric": "hamming"}, ValueError, "give a distance"),
        (
            {"method": "sets", "distance": 2, "metric": "jaro"},
            ValueError,
            "metric must be",
        ),
        ({"preset": "celltag"}, ValueError, "allowlist=False"),
        ({"preset": "celltag_multi"}, ValueError, "allowlist=False"),
        ({"preset": "trex", "metric": "levenshtein"}, ValueError, "metric='hamming'"),
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
        ({"method": "sets", "allowlist": "ACGT"}, TypeError, "sequence of barcodes"),
        ({"method": "sets", "prune": 1}, ValueError, "change nothing"),
    ],
)
def test_settings_name_what_is_missing_or_wrong(arguments, error, message) -> None:
    preset = arguments.pop("preset", None)
    with pytest.raises(error, match=message):
        barcode_call_settings(preset, **arguments)


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
    assert calls.counts["molecules_below_min_reads"] == 4


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
    calls = _call(_records(rows), 3, method="sets", distance=1, metric="hamming")
    assert _partition(calls.calls.codes) == {frozenset({0, 1, 2})}
    assert calls.counts["barcodes_corrected"] == 1
    assert calls.counts["records_corrected"] == 1


def test_equally_abundant_barcodes_merge_into_the_one_that_sorts_last() -> None:
    # Z sorts after A, so cells of either end with Z, as TREX picks the larger.
    later = "C" + A[1:]
    assert later > A
    rows = _molecules(0, A, 2) + _molecules(1, later, 2)
    records = _records(rows)
    calls = _call(
        records, 2, method="sets", distance=1, metric="hamming", allowlist=[later]
    )
    # Only the barcode that the tie kept passes an allowlist of it.
    assert calls.n_barcodes.tolist() == [1, 1]
    calls = _call(
        records, 2, method="sets", distance=1, metric="hamming", allowlist=[A]
    )
    assert calls.n_barcodes.tolist() == [0, 0]
    # One more molecule of A makes it the more abundant, whatever its order.
    rows += _molecules(2, A, 1)
    calls = _call(
        _records(rows), 3, method="sets", distance=1, metric="hamming", allowlist=[A]
    )
    assert calls.n_barcodes.tolist() == [1, 1, 1]


def test_a_chain_of_close_barcodes_merges_into_one_component() -> None:
    # A and A2 are two substitutions apart, but each is one from A1, so
    # within one substitution the three are one component, merged into A.
    rows = _molecules(0, A, 3) + _molecules(1, A1, 1) + _molecules(2, A2, 2)
    calls = _call(_records(rows), 3, method="sets", distance=1, metric="hamming")
    assert _partition(calls.calls.codes) == {frozenset({0, 1, 2})}
    assert calls.counts["barcodes_corrected"] == 2


def test_barcodes_of_two_groups_or_libraries_never_merge() -> None:
    rows = _molecules(0, A, 5) + _molecules(1, A1, 1)
    groups = np.asarray([0, 1])
    calls = _call(
        _records(rows), 2, groups=groups, method="sets", distance=1, metric="hamming"
    )
    assert calls.counts["barcodes_corrected"] == 0
    two = _molecules(0, A, 5) + _molecules(1, A1, 1, library=1)
    calls = _call(
        _records(two, (("a", "molecules"), ("b", "molecules"))),
        2,
        method="sets",
        distance=1,
        metric="hamming",
    )
    assert calls.counts["barcodes_corrected"] == 0


def test_edits_merge_an_indel_that_substitutions_cannot() -> None:
    shifted = A[:5] + A[6:] + "G"
    rows = _molecules(0, A, 5) + _molecules(1, shifted, 1)
    for metric, merged in (("hamming", 0), ("levenshtein", 1)):
        calls = _call(_records(rows), 2, method="sets", distance=2, metric=metric)
        assert calls.counts["barcodes_corrected"] == merged


def test_molecules_that_correction_gives_one_barcode_and_umi_are_one() -> None:
    # One UMI read under two barcodes is one molecule once both are A.
    rows = [(0, 0, A, "u", 1, 3), (0, 0, A1, "u", 1, 2)] + _molecules(1, A, 4)
    calls = _call(
        _records(rows), 2, method="sets", distance=1, metric="hamming", min_umis=2
    )
    # Cell 0 holds one molecule of A, below two UMIs.
    assert calls.n_barcodes.tolist() == [0, 1]
    calls = _call(_records(rows), 2, method="sets", distance=1, metric="hamming")
    assert calls.n_barcodes.tolist() == [1, 1]


def test_pairs_of_a_library_without_umis_rank_by_reads() -> None:
    # B has fewer pairs but more reads than A1 in a library counting reads.
    rows = [(0, 0, A1, None, -1, 3), (1, 0, A1, None, -1, 3), (2, 0, A, None, -1, 10)]
    calls = _call(
        _records(rows, (("atac", "pairs"),)),
        3,
        method="sets",
        distance=1,
        metric="hamming",
    )
    assert calls.units == ("reads",)
    assert calls.counts["barcodes_corrected"] == 1
    assert _partition(calls.calls.codes) == {frozenset({0, 1, 2})}


def test_correction_refuses_a_library_whose_counts_are_partly_unknown() -> None:
    rows = [(0, 0, A, None, 4, 10), (1, 0, A1, None, -1, 3)]
    with pytest.raises(ValueError, match=r"'q' counts UMIs for 1 of its 2 pairs"):
        _call(
            _records(rows, (("q", "pairs"),)),
            2,
            method="sets",
            distance=1,
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
        _records(rows), 3, allowlist=[A], method="sets", distance=1, metric="hamming"
    )
    # A1 becomes A before the allowlist is read, so only B goes.
    assert calls.n_barcodes.tolist() == [1, 1, 0]
    assert calls.counts["records_outside_allowlist"] == 3


# Pair filters.


def test_pairs_of_fewer_umis_than_min_umis_are_dropped() -> None:
    rows = _molecules(0, A, 2) + _molecules(0, B, 1) + _molecules(1, B, 2)
    calls = _call(_records(rows), 2, method="sets", min_umis=2)
    assert calls.n_barcodes.tolist() == [1, 1]
    assert calls.counts["pairs_below_min_umis"] == 1
    # In a library that counts reads alone, the count is reads.
    pairs = [(0, 0, A, None, -1, 2), (0, 0, B, None, -1, 1)]
    calls = _call(_records(pairs, (("atac", "pairs"),)), 1, method="sets", min_umis=2)
    assert calls.n_barcodes.tolist() == [1]


def test_min_fraction_keeps_pairs_at_bartabs_share_of_the_top_barcode() -> None:
    # BARtab keeps a barcode whose count divided by the fraction reaches the
    # cell's top count: 3 / 0.3 is 10.000000000000002, 2 / 0.3 is 6.67.
    rows = _molecules(0, A, 10) + _molecules(0, B, 3) + _molecules(0, C, 2)
    calls = _call(_records(rows), 1, method="sets", min_fraction=0.3)
    assert calls.n_barcodes.tolist() == [2]
    assert calls.counts["pairs_below_min_fraction"] == 1
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
    assert calls.counts["cells_outside_barcodes_per_cell"] == 2


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
    calls = _call(_records(rows), 5, preset="trex", distance=0)
    assert calls.n_barcodes.tolist() == [1, 0, 1, 0, 1]
    assert calls.counts["molecules_of_low_complexity"] == 2
    assert calls.counts["molecules_of_few_bases"] == 2


def test_trex_merges_partial_cloneids_over_the_bases_both_hold() -> None:
    # 24 shared bases match; 19 would be too few to compare.
    rows = _molecules(0, T, 5, reads=3) + _molecules(1, T[:24] + "-" * 6, 2, reads=3)
    calls = _call(_records(rows), 2, preset="trex")
    assert calls.counts["barcodes_corrected"] == 1
    assert _partition(calls.calls.codes) == {frozenset({0, 1})}
    rows = _molecules(0, T, 5, reads=3) + _molecules(1, "-" * 11 + T[11:], 2, reads=3)
    calls = _call(_records(rows), 2, preset="trex")
    assert calls.counts["barcodes_corrected"] == 0


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
    calls = _call(_records(rows), 4, preset="trex", distance=0)
    assert calls.n_barcodes.tolist() == [1, 0, 1, 0]
    assert calls.counts["pairs_of_one_umi_shared"] == 1
    assert calls.counts["pairs_of_one_read"] == 1


def test_trex_rules_need_molecules_and_their_reads() -> None:
    pairs = [(0, 0, T, None, 3, 9)]
    with pytest.raises(ValueError, match="applies TREX's rules to molecules"):
        _call(_records(pairs, (("p", "pairs"),)), 1, preset="trex")
    rows = _molecules(0, T, 2, reads=-1)
    with pytest.raises(ValueError, match="read_count since September 2024"):
        _call(_records(rows), 1, preset="trex")


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
    assert _partition(across.calls.codes) == {frozenset({0, 1})}
    within = _call(records, 2, method="jaccard", similarity=0.6, inclusive=False)
    assert _partition(within.calls.codes) == {frozenset({0}), frozenset({1})}
