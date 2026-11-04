"""Clone labels, assignment and ranking, checked against hand-derived values."""

import sys

import numpy as np
import pandas as pd
import pytest

from cytearc.clones import MULTIPLE_POLICIES
from cytearc.clones.assignments import (
    CloneAssigner,
    CloneAssignment,
    clone_frames,
    clone_label_texts,
    ranked_clones,
)
from cytearc.utils.arrays import sort_categories
from cytearc.utils.memory_budget import MemoryBudgetError, MemoryLedger
from tests.memory_helpers import audited, charges, cost_bound, traced_peak

CELLS = np.asarray([f"c{index}" for index in range(6)])


def _labels(*values: object) -> np.ndarray:
    return np.asarray(values, dtype=object)


def test_labels_read_as_text_and_missing_values_mean_no_clone() -> None:
    assert clone_label_texts(
        _labels(" a ", "", None, "NA ", b"b", 5, True)
    ).tolist() == [
        "a",
        None,
        None,
        "NA",
        "b",
        "5",
        "True",
    ]
    assert clone_label_texts(np.asarray([3, 1])).tolist() == ["3", "1"]
    assert clone_label_texts(np.asarray([2.0, np.nan, -1.0])).tolist() == [
        "2",
        None,
        "-1",
    ]
    assert clone_label_texts(_labels(4.0, np.nan, pd.NA)).tolist() == ["4", None, None]
    assert clone_label_texts(
        np.asarray(["a", "b"]), missing=np.asarray([False, True])
    ).tolist() == ["a", None]


@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        (np.asarray([1.5]), TypeError, "text or whole numbers, not 1.5"),
        (_labels(2.25), TypeError, "text or whole numbers, not 2.25"),
        (_labels(object()), TypeError, "text or numbers, not object"),
        (
            np.asarray([None, ["a", "b"]], dtype=object)[1:],
            TypeError,
            "text or numbers, not list",
        ),
        (
            np.asarray(["2026-01-01"], dtype="datetime64[D]"),
            TypeError,
            "text or numbers",
        ),
        (np.zeros((2, 2)), ValueError, "one-dimensional"),
    ],
)
def test_labels_that_are_not_text_or_whole_numbers_are_refused(
    values, error, message
) -> None:
    with pytest.raises(error, match=message):
        clone_label_texts(values)


def test_a_missing_mask_must_align_with_its_labels() -> None:
    with pytest.raises(ValueError, match="must align"):
        clone_label_texts(np.asarray(["a", "b"]), missing=np.asarray([True]))


def test_clones_rank_by_size_then_by_label_in_natural_order() -> None:
    # Sizes: b10 1, b2 2, 07 2, 7 1, unused 0.
    codes = np.asarray([0, 1, 1, 2, 2, 3, -1])
    ranked, ids, sizes = ranked_clones(codes, ["b10", "b2", "07", "7", "unused"])
    assert ids.tolist() == ["07", "b2", "7", "b10"]
    assert sizes.tolist() == [2, 2, 1, 1]
    assert ranked.tolist() == [3, 1, 1, 0, 0, 2, -1]
    assert sizes.dtype == np.int64


def test_digit_labels_rank_as_integers_and_equal_values_by_text() -> None:
    labels = ["10", "9", "007", "7", "100"]
    _ranked, ids, _sizes = ranked_clones(np.arange(5), labels)
    assert ids.tolist() == ["007", "7", "9", "10", "100"]
    # Labels too long for integers rank as natural text.
    long_labels = ["2" * 30, "1" * 30]
    _ranked, ids, _sizes = ranked_clones(np.arange(2), long_labels)
    assert ids.tolist() == ["1" * 30, "2" * 30]


def test_numpy_booleans_read_as_their_text() -> None:
    assert clone_label_texts(np.asarray([True, False])).tolist() == ["True", "False"]


@pytest.mark.parametrize(
    "labels",
    [
        # Floats hold integers of up to 15 digits exactly; these two round to
        # one float, so natural order ranks them by text.
        ["09007199254740993", "9007199254740992"],
        # Digits outside ASCII, which int() and str.isdigit take differently.
        ["²", "3", "⑤", "٣"],
    ],
)
def test_digit_like_labels_rank_in_natural_order(labels) -> None:
    _ranked, ids, _sizes = ranked_clones(np.arange(len(labels)), labels)
    assert ids.tolist() == sort_categories(labels)


def test_ranking_holds_each_label_as_its_own_text() -> None:
    # One long label must not widen the other 29,999 to its length, which
    # as fixed-width text would take 240 MB.
    labels = [str(index) for index in range(30_000)]
    labels[0] = "X" * 2_000
    (_ranked, ids, _sizes), peak = traced_peak(
        lambda: ranked_clones(np.arange(len(labels)), labels), 24_000_000 - 1
    )
    # Numbers come first, then other text.
    assert ids.tolist()[:2] == ["1", "2"]
    assert ids.tolist()[-1] == "X" * 2_000
    assert peak < 24_000_000


def test_no_cell_in_a_clone_gives_no_clone() -> None:
    ranked, ids, sizes = ranked_clones(np.full(3, -1), ["a"])
    assert ranked.tolist() == [-1, -1, -1]
    assert ids.tolist() == []
    assert sizes.tolist() == []


def test_rows_assign_cells_and_repeated_rows_change_nothing() -> None:
    assigner = CloneAssigner(
        CELLS, multiple="error", ledger=MemoryLedger(None, "assigning")
    )
    assigner.add(np.asarray([0, 1, 1, -1, 3]), _labels("x", "y", "y", "x", None))
    assigner.add(np.asarray([2, 0, 3]), _labels("x", "x", "z"))

    result = assigner.finish()

    # x: c0 and c2; y: c1; z: c3; c4 and c5 have no clone.
    assert result.clone_ids.tolist() == ["x", "y", "z"]
    assert result.clone_sizes.tolist() == [2, 1, 1]
    assert result.codes.tolist() == [0, 1, 0, 2, -1, -1]
    assert result.values.tolist() == ["x", "y", "x", "z", "", ""]
    assert result.clone_size.tolist() == [2, 1, 2, 1, 0, 0]
    assert result.probabilities is None
    assert result.n_cells_in_several_clones == 0


def test_a_cell_in_two_clones_raises_within_one_chunk_or_across_two() -> None:
    within = CloneAssigner(
        CELLS, multiple="error", ledger=MemoryLedger(None, "assigning")
    )
    with pytest.raises(
        ValueError,
        match="1 cells are listed in two different clones, for example 'c2' in 'a' and 'b'",
    ):
        within.add(np.asarray([2, 2]), _labels("a", "b"))
    across = CloneAssigner(
        CELLS, multiple="error", ledger=MemoryLedger(None, "assigning")
    )
    across.add(np.asarray([4]), _labels("a"))
    with pytest.raises(ValueError, match="'c4' in 'a' and 'b'"):
        across.add(np.asarray([4]), _labels("b"))


def test_dropped_cells_stay_without_a_clone() -> None:
    assigner = CloneAssigner(
        CELLS, multiple="drop", ledger=MemoryLedger(None, "assigning")
    )
    assigner.add(np.asarray([0, 0, 1, 2]), _labels("a", "b", "a", "a"))
    assigner.add(np.asarray([1, 0]), _labels("c", "a"))

    result = assigner.finish()

    # c0 was given a and b, c1 a and c; c2 keeps a, and c0's later row does
    # not give it a clone again.
    assert result.codes.tolist() == [-1, -1, 0, -1, -1, -1]
    assert result.clone_ids.tolist() == ["a"]
    assert result.n_cells_in_several_clones == 2


def test_probabilities_follow_the_cells_and_must_agree() -> None:
    assigner = CloneAssigner(
        CELLS, multiple="error", ledger=MemoryLedger(None, "assigning")
    )
    assigner.add(
        np.asarray([0, 1, 1, 2]),
        _labels("a", "b", "b", "a"),
        probabilities=np.asarray([0.9, 0.5, 0.5, np.nan]),
    )
    assigner.add(np.asarray([2]), _labels("a"), probabilities=np.asarray([0.7]))
    result = assigner.finish()
    np.testing.assert_array_equal(
        result.probabilities, [0.9, 0.5, 0.7, np.nan, np.nan, np.nan]
    )
    clashing = CloneAssigner(
        CELLS, multiple="error", ledger=MemoryLedger(None, "assigning")
    )
    with pytest.raises(ValueError, match="'c1' is listed with two probabilities"):
        clashing.add(
            np.asarray([1, 1]), _labels("a", "a"), probabilities=np.asarray([0.2, 0.3])
        )
    later = CloneAssigner(
        CELLS, multiple="error", ledger=MemoryLedger(None, "assigning")
    )
    later.add(np.asarray([3]), _labels("a"), probabilities=np.asarray([0.2]))
    with pytest.raises(ValueError, match="'c3' is listed with two probabilities"):
        later.add(np.asarray([3]), _labels("a"), probabilities=np.asarray([0.4]))


@pytest.mark.parametrize(
    "chunks",
    [
        [([0, 0], [np.nan, 0.5])],
        [([0, 0], [0.5, np.nan])],
        [([0], [np.nan]), ([0], [0.5])],
        [([0], [0.5]), ([0], [np.nan])],
    ],
)
def test_a_row_without_a_probability_gives_none_in_any_chunk(chunks) -> None:
    # A row's NaN gives no probability, so the cell's other row decides it,
    # in one chunk or across two, in either order.
    assigner = CloneAssigner(
        CELLS, multiple="error", ledger=MemoryLedger(None, "assigning")
    )
    for cells, probabilities in chunks:
        assigner.add(
            np.asarray(cells),
            _labels(*["a"] * len(cells)),
            probabilities=np.asarray(probabilities),
        )
    assert assigner.finish().probabilities.tolist()[0] == 0.5


def test_the_assigner_counts_the_text_of_each_distinct_label_once() -> None:
    assigner = CloneAssigner(
        CELLS, multiple="error", ledger=MemoryLedger(None, "assigning")
    )
    long_label = "a" * 1_000
    assigner.add(np.asarray([0, 1, 2]), _labels(long_label, long_label, "b"))
    assigner.add(np.asarray([3]), _labels(long_label))
    assert assigner.label_bytes == sys.getsizeof(long_label) + sys.getsizeof("b")


def test_probabilities_of_dropped_cells_are_cleared() -> None:
    assigner = CloneAssigner(
        CELLS, multiple="drop", ledger=MemoryLedger(None, "assigning")
    )
    assigner.add(np.asarray([0]), _labels("a"), probabilities=np.asarray([0.8]))
    # c2's two clones come with two probabilities; it is dropped, not refused.
    assigner.add(
        np.asarray([2, 2]), _labels("a", "c"), probabilities=np.asarray([0.3, 0.4])
    )
    assigner.add(
        np.asarray([0, 1]), _labels("b", "b"), probabilities=np.asarray([0.6, 0.6])
    )
    result = assigner.finish()
    assert result.codes.tolist()[:3] == [-1, 0, -1]
    np.testing.assert_array_equal(result.probabilities[:3], [np.nan, 0.6, np.nan])
    assert result.n_cells_in_several_clones == 2


def test_a_label_from_two_sources_is_refused() -> None:
    assigner = CloneAssigner(
        CELLS, multiple="error", ledger=MemoryLedger(None, "assigning")
    )
    assigner.add(np.asarray([0]), _labels("x"), source=0)
    assigner.add(np.asarray([1]), _labels("x"), source=0)
    with pytest.raises(ValueError, match="'x' comes from two sources"):
        assigner.add(np.asarray([2]), _labels("x"), source=1)


def test_rows_of_unequal_lengths_and_unknown_policies_are_refused() -> None:
    with pytest.raises(ValueError, match="multiple must be one of error, drop"):
        CloneAssigner(CELLS, multiple="keep", ledger=MemoryLedger(None, "assigning"))  # type: ignore[arg-type]
    assigner = CloneAssigner(
        CELLS, multiple="error", ledger=MemoryLedger(None, "assigning")
    )
    with pytest.raises(ValueError, match="one cell and one label each"):
        assigner.add(np.asarray([0, 1]), _labels("a"))
    with pytest.raises(ValueError, match="one probability each"):
        assigner.add(
            np.asarray([0]), _labels("a"), probabilities=np.asarray([0.1, 0.2])
        )
    assert MULTIPLE_POLICIES == ("error", "drop")


def test_rows_that_name_no_cell_or_no_clone_leave_the_labels_unknown() -> None:
    assigner = CloneAssigner(
        CELLS, multiple="error", ledger=MemoryLedger(None, "assigning")
    )
    assigner.add(np.asarray([-1, 0]), _labels("x", None))
    assert assigner.n_labels == 0
    result = assigner.finish()
    assert result.clone_ids.tolist() == []
    assert result.codes.tolist() == [-1] * 6


def test_clone_frames_hold_cells_and_clones() -> None:
    cells, clones = clone_frames(
        np.asarray(["a", "b", "c"]),
        codes=np.asarray([1, -1, 0]),
        clone_size=np.asarray([1, 0, 1]),
        clone_ids=np.asarray(["k1", "k2"], dtype=object),
        clone_sizes=np.asarray([1, 1]),
        extra={"clone_probability": np.asarray([0.5, np.nan, 1.0])},
    )
    assert cells.index.tolist() == ["a", "b", "c"]
    assert cells.index.name == "ids"
    assert cells["clone"].astype(object).where(
        cells["clone"].notna(), None
    ).tolist() == [
        "k2",
        None,
        "k1",
    ]
    assert list(cells["clone"].cat.categories) == ["k1", "k2"]
    assert cells["clone_probability"].tolist()[2] == 1.0
    assert clones.to_dict("list") == {"clone": ["k1", "k2"], "n_cells": [1, 1]}


def test_an_assignment_reads_labels_and_sizes_per_cell() -> None:
    assignment = CloneAssignment(
        codes=np.asarray([1, -1, 0, 1]),
        clone_ids=np.asarray(["big", "small"], dtype=object),
        clone_sizes=np.asarray([1, 2]),
        probabilities=None,
        n_cells_in_several_clones=0,
    )
    assert assignment.values.tolist() == ["small", "", "big", "small"]
    assert assignment.clone_size.tolist() == [2, 0, 1, 2]


def _assigner(
    budget: int | None = None, **options
) -> tuple[CloneAssigner, MemoryLedger]:
    ledger = MemoryLedger(budget, "Assigning clones")
    return CloneAssigner(CELLS, multiple="error", ledger=ledger, **options), ledger


def test_the_assigner_charges_each_cell_each_new_label_and_the_probabilities() -> None:
    from cytearc.clones.assignments import _LABEL_ENTRY_BYTES

    assigner, ledger = _assigner()
    # Each cell's clone code, eight bytes, and whether it was given several
    # clones, one.
    assert ledger.charged == 9 * len(CELLS)
    assigner.add(np.asarray([0, 1, 2]), _labels("a", "b", "a"))
    labels = sys.getsizeof("a") + sys.getsizeof("b")
    assert ledger.charged == 9 * len(CELLS) + labels + 2 * _LABEL_ENTRY_BYTES
    # A label seen before costs nothing more; a new one its string and entry.
    assigner.add(np.asarray([3, 4]), _labels("b", "longer label"))
    labels += sys.getsizeof("longer label")
    assert ledger.charged == 9 * len(CELLS) + labels + 3 * _LABEL_ENTRY_BYTES
    # The first probabilities take eight bytes for each cell.
    assigner.add(np.asarray([5]), _labels("a"), probabilities=np.asarray([0.5]))
    assert ledger.charged == 17 * len(CELLS) + labels + 3 * _LABEL_ENTRY_BYTES
    assert assigner.label_bytes == labels


def test_a_label_beyond_the_budget_is_refused_before_it_is_kept() -> None:
    from cytearc.clones.assignments import _LABEL_ENTRY_BYTES

    held = 9 * len(CELLS) + sys.getsizeof("a") + _LABEL_ENTRY_BYTES
    assigner, ledger = _assigner(held + sys.getsizeof("b") + _LABEL_ENTRY_BYTES - 1)
    assigner.add(np.asarray([0]), _labels("a"))
    with pytest.raises(MemoryBudgetError) as refused:
        assigner.add(np.asarray([1]), _labels("b"))
    assert refused.value.item == "2 clone labels"
    assert refused.value.needed_bytes == held + sys.getsizeof("b") + _LABEL_ENTRY_BYTES
    # The refused label was not kept, so the cells keep their clones, which
    # ranking them without a budget shows.
    assert assigner.n_labels == 1
    assert ledger.charged == held
    ledger.budget = None
    assert assigner.finish().clone_ids.tolist() == ["a"]


def test_numbering_few_clones_of_many_cells_holds_what_ranking_charges() -> None:
    # 200,000 cells in 100 clones: numbering them by rank holds each cell's
    # code by rank, whether it has a clone and the codes gathered to rank
    # them, 25 bytes a cell, beside NumPy's objects, and outweighs ordering
    # the labels.
    n_cells = 200_000
    cells = np.asarray([f"c{index}" for index in range(n_cells)])
    ledger = MemoryLedger(None, "Assigning clones")
    assigner = CloneAssigner(cells, multiple="error", ledger=ledger)
    for start in range(0, n_cells, 50_000):
        rows = np.arange(start, start + 50_000)
        labels = np.asarray(
            [f"clone_{row % 100}" for row in rows.tolist()], dtype=object
        )
        assigner.add(rows, labels)
    held = ledger.held_bytes
    _result, taken = charges(assigner.finish)
    (ranking,) = [planned for item, _n, planned in taken if item.startswith("ranking")]
    assert ranking - held > 25 * n_cells
    cost_bound(ranking - held, assigner.finish, slack=1.3)


LABEL_KINDS = {
    "ASCII": lambda index: f"clone_{index:06d}",
    "numbers": str,
    "four-byte": lambda index: f"\U0001f600clone{index}",
}


@pytest.mark.parametrize("kind", list(LABEL_KINDS))
def test_gathering_many_labels_holds_no_more_than_their_charges(kind) -> None:
    # 40,000 cells, each of its own clone, given in chunks of 4,000 rows: the
    # labels the assigner keeps, and their ranking, beside each chunk, which
    # the caller charges as its labels' new strings, their references and the
    # rows, and the chunk's factorization, 64 bytes a row beside the UTF-8
    # copy that hashing caches with each label that is not ASCII.
    n_cells = 40_000
    name = LABEL_KINDS[kind]
    cells = np.asarray([f"c{index}" for index in range(n_cells)])
    label = name(n_cells)
    utf8 = 0 if label.isascii() else len(label.encode()) + 1
    chunk = 4_000 * (sys.getsizeof(label) + utf8 + 24 + 64)

    def gather() -> CloneAssignment:
        ledger = MemoryLedger(None, "Assigning clones")
        assigner = CloneAssigner(cells, multiple="error", ledger=ledger)
        for start in range(0, n_cells, 4_000):
            ledger.hold("a chunk of labels", chunk)
            rows = np.arange(start, start + 4_000)
            labels = np.asarray(
                [name(row) for row in range(start, start + 4_000)], dtype=object
            )
            assigner.add(rows, labels)
            del rows, labels
            ledger.release(chunk)
        return assigner.finish()

    gather()
    _result, taken = charges(gather)
    planned = max(planned for _item, _nbytes, planned in taken)
    _result, peak = traced_peak(gather, planned)
    assert peak <= planned + 64 * 1024, (peak, planned)
    assert planned <= 1.5 * peak, (peak, planned)
    _result, overruns = audited(gather)
    assert overruns == [], overruns
