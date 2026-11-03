"""Calling clones from rare recorder alleles, on edits written out by hand.

Each case builds a cell's edits as (cell, character, allele) rows, with -1
for a placeholder of an unknown mark, and the banks as DARLIN's tables. The
p-values of CARLIN's test are derived by hand from its formula, and the
Benjamini-Hochberg cases from the step-up rule; a random check compares the
clones with a separate reading of the rules on Python sets and networkx.
"""

import math

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from cytearc.clones.allele_clones import (
    AlleleObservations,
    allele_banks,
    allele_call_settings,
    allele_clone_calls,
    benjamini_hochberg,
    clonal_p_values,
)
from tests.clone_helpers import traced_steps
from tests.memory_helpers import smallest_budget, traced_peak

_TEXTS = ("a", "b", "c", "d", "e", "f", "g", "h")


def _observations(
    rows: list[tuple[int, int, int]], characters: tuple[str, ...] = ("CA", "TA", "RA")
) -> AlleleObservations:
    table = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
    return AlleleObservations(
        cells=table[:, 0].copy(),
        characters=table[:, 1].copy(),
        alleles=table[:, 2].copy(),
        character_names=characters,
        allele_texts=np.asarray(_TEXTS, dtype=object),
    )


def _bank(entries: dict[str, tuple[float, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "allele": list(entries),
            "normalized_count": [rate for rate, _samples in entries.values()],
            "sample_count": [samples for _rate, samples in entries.values()],
        }
    )


def _call(
    rows,
    n_cells,
    *,
    method="graph",
    banks=None,
    groups=None,
    characters=("CA", "TA", "RA"),
    **rules,
):
    checked = None if banks is None else allele_banks(banks)
    settings = allele_call_settings(method, banks=checked, **rules)
    return allele_clone_calls(
        _observations(rows, characters),
        n_cells,
        groups=None if groups is None else np.asarray(groups, dtype=np.int64),
        settings=settings,
        banks=checked,
    )


def _lacking(characters=("CA", "TA", "RA")) -> dict[str, pd.DataFrame]:
    """Banks that list none of the alleles, so that every allele is rare."""
    return {name: _bank({"zz": (0.5, 3)}) for name in characters}


def _partition(codes: np.ndarray) -> set[frozenset[int]]:
    found: dict[int, set[int]] = {}
    for cell, code in enumerate(codes.tolist()):
        if code >= 0:
            found.setdefault(code, set()).add(cell)
    return {frozenset(cells) for cells in found.values()}


# CARLIN's p_clonal ------------------------------------------------------------


def test_p_clonal_takes_the_values_its_formula_gives_by_hand() -> None:
    # With x = λM: at x = ln 2, e^(-x) = 1/2, so p = (1/2 - ln2/2) / (1/2).
    # At x = 1, p = (1 - 2/e) / (1 - 1/e) = (e - 2) / (e - 1); at x = 2,
    # p = (1 - 3/e²) / (1 - 1/e²) = (e² - 3) / (e² - 1).
    p_values = clonal_p_values([math.log(2) / 4, 0.25, 0.5, 2.0], [4, 4, 4, 1])
    expected = [
        1 - math.log(2),
        (math.e - 2) / (math.e - 1),
        (math.e**2 - 3) / (math.e**2 - 1),
        (math.e**2 - 3) / (math.e**2 - 1),
    ]
    np.testing.assert_allclose(p_values, expected, rtol=1e-13)
    # As x approaches 0, p approaches x / 2 - x² / 12, the leading terms of
    # the ratio of the series of its numerator and denominator.
    small = clonal_p_values([1e-8, 1e-12], [1, 1])
    np.testing.assert_allclose(small, [5e-9 - 1e-16 / 12, 5e-13], rtol=1e-12)
    # A rate of 0, an allele missing from the bank, gives 0; a large mean
    # makes an allele almost surely arisen more than once.
    assert clonal_p_values([0.0], [1_000])[0] == 0.0
    assert clonal_p_values([1.0], [60])[0] == pytest.approx(1.0, abs=1e-20)


def test_p_clonal_equals_carlins_formula_across_means() -> None:
    # Below about 1e-3 the formula as written loses digits to cancellation,
    # which the Poisson tails avoid; the series above checks those means.
    means = np.geomspace(1e-3, 40, 200)
    expected = [
        (1 - math.exp(-x) - x * math.exp(-x)) / (1 - math.exp(-x)) for x in means
    ]
    np.testing.assert_allclose(clonal_p_values(means, 1), expected, rtol=1e-8)


def test_p_clonal_of_the_smallest_means_is_their_series_and_never_nan() -> None:
    # Near the smallest floats the Poisson tails underflow, where the series
    # x / 2 - x² / 12 is exact; the two meet at 1e-8.
    means = [1e-310, 1e-200, 1e-12, np.nextafter(1e-8, 0.0), 1e-8]
    found = clonal_p_values(means, 1)
    assert not np.any(np.isnan(found))
    np.testing.assert_allclose(
        found, [x / 2 - x**2 / 12 for x in means], rtol=1e-12, atol=0
    )


@pytest.mark.parametrize(
    ("rates", "observations"),
    [([-0.1], [3]), ([np.nan], [3]), ([0.1], [np.inf]), ([0.1], [-1])],
)
def test_p_clonal_refuses_rates_and_observations_that_are_not_counts(
    rates, observations
) -> None:
    with pytest.raises(ValueError, match="finite and at least 0"):
        clonal_p_values(rates, observations)


# Benjamini-Hochberg --------------------------------------------------------------


def test_benjamini_hochberg_keeps_the_ranks_below_the_largest_that_passes() -> None:
    # Ranked 0.01, 0.03, 0.04, 0.2 against k·0.05/4: only the first passes.
    assert benjamini_hochberg([0.01, 0.04, 0.03, 0.2], [0] * 4, 0.05).tolist() == [
        True,
        False,
        False,
        False,
    ]
    # 0.026 fails its own rank (0.026 > 2·0.05/4) but 0.03 passes rank 3
    # (0.03 ≤ 3·0.05/4), so the step-up keeps all three.
    assert benjamini_hochberg([0.03, 0.01, 0.026, 0.2], [0] * 4, 0.05).tolist() == [
        True,
        True,
        True,
        False,
    ]
    # p·m equal to k·fdr passes: 0.25·2 = 2·0.25; a hair above fails.
    assert benjamini_hochberg([0.125, 0.25], [0, 0], 0.25).tolist() == [True, True]
    assert benjamini_hochberg(
        [0.125, np.nextafter(0.25, 1.0)], [0, 0], 0.25
    ).tolist() == [True, False]
    # Equal p-values take consecutive ranks: the last of three passes 3·0.05/3.
    assert benjamini_hochberg([0.05, 0.05, 0.05], [0] * 3, 0.05).all()
    assert benjamini_hochberg([], [], 0.05).tolist() == []


def test_benjamini_hochberg_runs_on_each_family_apart() -> None:
    # Alone, 0.04 passes its family of one; pooled with the other family's two
    # tests it would rank second of three, and 0.04·3 > 2·0.05.
    kept = benjamini_hochberg([0.01, 0.9, 0.04], [0, 0, 1], 0.05)
    assert kept.tolist() == [True, False, True]
    assert benjamini_hochberg([0.01, 0.9, 0.04], [0, 0, 0], 0.05).tolist() == [
        True,
        False,
        False,
    ]


def test_benjamini_hochberg_agrees_with_scipys_adjusted_p_values() -> None:
    rng = np.random.default_rng(3)
    for trial in range(300):
        n_tests = int(rng.integers(1, 40))
        p_values = rng.choice(
            [rng.random() * 0.1, rng.random(), 0.0], size=n_tests
        ) * rng.random(n_tests)
        families = rng.integers(0, 3, size=n_tests)
        fdr = float(rng.choice([0.01, 0.05, 0.2]))
        kept = benjamini_hochberg(p_values, families, fdr)
        for family in np.unique(families):
            chosen = families == family
            adjusted = stats.false_discovery_control(p_values[chosen], method="bh")
            assert kept[chosen].tolist() == (adjusted <= fdr).tolist(), trial


def test_benjamini_hochberg_refuses_tests_without_one_family_each() -> None:
    with pytest.raises(ValueError, match="one value per test"):
        benjamini_hochberg([0.1, 0.2], [0], 0.05)


# Banks ---------------------------------------------------------------------------


def test_bank_alleles_compare_without_surrounding_spaces_as_the_import_reads_them() -> (
    None
):
    bank = allele_banks(
        {"CA": pd.DataFrame({"allele": [" b", "a "], "normalized_count": [0.1, 0.2]})}
    )["CA"]
    assert bank.alleles.tolist() == ["a", "b"]
    with pytest.raises(ValueError, match="lists 1 alleles more than once, such as 'a'"):
        allele_banks(
            {
                "CA": pd.DataFrame(
                    {"allele": ["a", " a"], "normalized_count": [0.1, 0.2]}
                )
            }
        )


def test_banks_are_sorted_by_allele_and_look_alleles_up() -> None:
    table = pd.DataFrame(
        {
            "allele": pd.Series(["b", "[]", "a"], dtype="str"),
            "normalized_count": [0.25, 1.0, 1e-6],
            "sample_count": [2.0, 3.0, 1.0],
            "smoothed_homoplasy": [0.1, 0.2, 0.3],
        }
    )
    bank = allele_banks({"CA": table})["CA"]
    assert bank.alleles.tolist() == ["[]", "a", "b"]
    assert bank.frequencies.tolist() == [1.0, 1e-6, 0.25]
    assert bank.samples is not None
    assert bank.samples.tolist() == [3, 1, 2]
    assert bank.positions(np.asarray(["b", "zz", "a"], dtype=object)).tolist() == [
        2,
        -1,
        1,
    ]
    # A bank that counts no samples keeps None.
    plain = allele_banks({"CA": table[["allele", "normalized_count"]]})["CA"]
    assert plain.samples is None


@pytest.mark.parametrize(
    ("frequencies", "error", "match"),
    [
        ([("CA", None)], TypeError, "must map each character"),
        ({}, ValueError, "names no bank"),
        ({3: _bank({"a": (0.1, 1)})}, TypeError, "keyed by character names"),
        ({"": _bank({"a": (0.1, 1)})}, TypeError, "keyed by character names"),
        ({"CA": {"allele": ["a"]}}, TypeError, "must be a pandas DataFrame"),
        (
            {"CA": pd.DataFrame({"allele": ["a"]})},
            ValueError,
            "lacks the column 'normalized_count'",
        ),
        (
            {"CA": pd.DataFrame({"normalized_count": [0.1]})},
            ValueError,
            "lacks the column 'allele'",
        ),
        (
            {"CA": pd.DataFrame({"allele": [], "normalized_count": []})},
            ValueError,
            "holds no allele",
        ),
        (
            {
                "CA": pd.DataFrame(
                    {"allele": ["a", None], "normalized_count": [0.1, 0.2]}
                )
            },
            ValueError,
            "1 alleles that are not text",
        ),
        (
            {"CA": pd.DataFrame({"allele": [1, 2], "normalized_count": [0.1, 0.2]})},
            ValueError,
            "2 alleles that are not text",
        ),
        (
            {"CA": _bank({"a": (0.1, 1)}).iloc[[0, 0]]},
            ValueError,
            "lists 1 alleles more than once, such as 'a'",
        ),
        ({"CA": _bank({"a": (1.5, 1)})}, ValueError, "outside 0 to 1, such as 1.5"),
        ({"CA": _bank({"a": (-0.1, 1)})}, ValueError, "outside 0 to 1"),
        (
            {"CA": pd.DataFrame({"allele": ["a"], "normalized_count": [np.nan]})},
            ValueError,
            "missing or not finite",
        ),
        (
            {"CA": pd.DataFrame({"allele": ["a"], "normalized_count": ["often"]})},
            ValueError,
            "not numbers",
        ),
        (
            {"CA": _bank({"a": (0.1, -1)})},
            ValueError,
            "not whole numbers of at least 0",
        ),
        (
            {
                "CA": pd.DataFrame(
                    {"allele": ["a"], "normalized_count": [0.1], "sample_count": [1.5]}
                )
            },
            ValueError,
            "such as 1.5",
        ),
        (
            {
                "CA": pd.DataFrame(
                    {"allele": ["a"], "normalized_count": [0.1], "sample_count": [None]}
                )
            },
            ValueError,
            "sample_count values that are missing",
        ),
    ],
)
def test_banks_refuse_what_they_cannot_read(frequencies, error, match) -> None:
    with pytest.raises(error, match=match):
        allele_banks(frequencies)


# Settings ------------------------------------------------------------------------


def test_settings_record_the_rules_of_each_method_as_python_values() -> None:
    banks = allele_banks({"CA": _bank({"a": (0.1, 1)})})
    graph = allele_call_settings(
        "graph",
        max_frequency=np.float64(2e-5),
        max_samples=np.int64(1),
        max_alleles=np.int32(3),
        fdr=0.2,
        banks=banks,
    )
    assert graph.record() == {
        "method": "graph",
        "max_frequency": 2e-5,
        "max_samples": 1,
        "max_alleles": 3,
        "fdr": None,
    }
    assert all(
        type(value) in (str, float, int, type(None))
        for value in graph.record().values()
    )
    tested = allele_call_settings("p_clonal", fdr=np.float32(0.25), banks=banks)
    assert tested.record() == {
        "method": "p_clonal",
        "max_frequency": None,
        "max_samples": None,
        "max_alleles": None,
        "fdr": 0.25,
    }
    # Without a bank, the graph takes frequencies from the cells.
    assert allele_call_settings("graph", max_frequency=1).max_frequency == 1.0


@pytest.mark.parametrize(
    ("method", "rules", "error", "match"),
    [
        ("tree", {}, ValueError, "method must be one of 'graph', 'p_clonal'"),
        (None, {}, ValueError, "method must be one of"),
        ("graph", {}, ValueError, "needs max_frequency, max_samples or both"),
        ("graph", {"max_frequency": 0}, ValueError, "greater than 0 and at most 1"),
        ("graph", {"max_frequency": 1.5}, ValueError, "greater than 0 and at most 1"),
        ("graph", {"max_frequency": True}, TypeError, "max_frequency must be a real"),
        ("graph", {"max_frequency": "2e-5"}, TypeError, "max_frequency must be a real"),
        ("graph", {"max_frequency": 0.1, "max_alleles": 0}, ValueError, "at least 1"),
        ("graph", {"max_frequency": 0.1, "max_alleles": 2.0}, TypeError, "integer"),
        ("graph", {"max_samples": -1}, ValueError, "max_samples must be at least 0"),
        ("graph", {"max_samples": True}, TypeError, "max_samples must be an integer"),
        ("graph", {"max_samples": 1}, ValueError, "pass frequencies"),
        ("graph", {"max_frequency": 0.1, "fdr": 0}, ValueError, "fdr must be greater"),
        ("graph", {"max_frequency": 0.1, "fdr": 1}, ValueError, "less than 1"),
        ("graph", {"max_frequency": 0.1, "fdr": None}, TypeError, "fdr must be a real"),
        ("p_clonal", {}, ValueError, "needs frequencies"),
        ("p_clonal", {"max_frequency": 0.1}, ValueError, "tests each allele's p-value"),
    ],
)
def test_settings_refuse_missing_and_wrong_rules(method, rules, error, match) -> None:
    with pytest.raises(error, match=match):
        allele_call_settings(method, **rules)


def test_max_samples_needs_every_bank_to_count_samples() -> None:
    banks = allele_banks(
        {
            "CA": _bank({"a": (0.1, 1)}),
            "TA": pd.DataFrame({"allele": ["b"], "normalized_count": [0.1]}),
        }
    )
    with pytest.raises(ValueError, match="sample_count, which the bank of 'TA' lacks"):
        allele_call_settings("graph", max_samples=1, banks=banks)


# The graph ----------------------------------------------------------------------


def test_cells_that_share_a_rare_allele_at_any_character_are_one_clone() -> None:
    # Cell 0 shares a on CA with cell 1, which shares b on TA with cell 2; cell
    # 3's c on CA joins nobody, and cell 4 holds a placeholder on TA. Without a
    # bank, a holds 2 of CA's 3 edited cells and b 2 of TA's 3, below 0.9.
    rows = [(0, 0, 0), (1, 0, 0), (1, 1, 1), (2, 1, 1), (3, 0, 2), (4, 1, -1)]
    called = _call(rows, 6, max_frequency=0.9)
    assert _partition(called.codes) == {frozenset({0, 1, 2}), frozenset({3})}
    # The clones are numbered by size, so the clone of three is clone 0.
    assert called.codes.tolist() == [0, 0, 0, 1, -1, -1]
    assert called.sizes.tolist() == [3, 1]
    assert called.n_alleles.tolist() == [1, 2, 1, 1, 0, 0]
    assert called.clone_alleles.tolist() == [2, 1]
    # An allele that every edited cell of its character holds, at frequency 1,
    # is rare only at max_frequency=1: b alone on TA links cells 1 and 2 there,
    # and no cell one step below.
    alone = _call(rows[:5], 5, max_frequency=1.0)
    assert _partition(alone.codes) == {frozenset({0, 1, 2}), frozenset({3})}
    below = _call(rows[:5], 5, max_frequency=float(np.nextafter(1.0, 0.0)))
    assert _partition(below.codes) == {frozenset({0, 1}), frozenset({3})}


def test_placeholders_never_link_cells_and_the_same_text_on_two_characters_is_two_alleles() -> (
    None
):
    # Cells 0 and 1 hold CARLIN's X at CA, -1 here; cells 2 and 3 hold a, one
    # on CA and one on TA. No bank lists a, so it is rare on both.
    rows = [(0, 0, -1), (1, 0, -1), (2, 0, 0), (3, 1, 0)]
    called = _call(rows, 4, banks=_lacking(), max_frequency=1e-9)
    assert _partition(called.codes) == {frozenset({2}), frozenset({3})}
    assert called.n_alleles.tolist() == [0, 0, 1, 1]


def test_an_allele_listed_twice_for_a_cell_counts_once() -> None:
    # Cell 0 holds a twice at CA, as two allele rows of the darlin CLI give it;
    # without a bank, a's frequency is the share of the 2 edited cells that
    # hold it, 1/2, not 2/3.
    rows = [(0, 0, 0), (0, 0, 0), (1, 0, 1)]
    assert _call(rows, 2, max_frequency=0.5).counts["alleles_too_frequent"] == 0
    below = float(np.nextafter(0.5, 0.0))
    assert _call(rows, 2, max_frequency=below).counts["alleles_too_frequent"] == 2
    # At 2/3, a would be too frequent for any bound below it.
    assert _call(rows, 2, max_frequency=0.6).counts["alleles_too_frequent"] == 0


def test_a_frequency_at_max_frequency_is_rare() -> None:
    rows = [(0, 0, 0), (1, 0, 0), (2, 0, 1), (3, 0, 1)]
    banks = {"CA": _bank({"a": (2e-5, 1), "b": (np.nextafter(2e-5, 0.0), 1)})}
    called = _call(rows, 4, banks=banks, max_frequency=2e-5)
    # a at exactly 2e-5 reaches the bound, as b, one step below, does.
    assert _partition(called.codes) == {frozenset({0, 1}), frozenset({2, 3})}
    assert called.counts["alleles_too_frequent"] == 0
    assert called.counts["rare_alleles"] == 2
    # DARLIN's rule, a frequency below 2e-5, is the bound one step below it.
    darlin = _call(rows, 4, banks=banks, max_frequency=float(np.nextafter(2e-5, 0.0)))
    assert _partition(darlin.codes) == {frozenset({2, 3})}
    assert darlin.counts["alleles_too_frequent"] == 1
    assert darlin.counts["rare_alleles"] == 1


def test_max_samples_keeps_alleles_of_at_most_that_many_bank_samples() -> None:
    rows = [(0, 0, 0), (1, 0, 0), (2, 0, 1), (3, 0, 1), (4, 0, 2), (5, 0, 2)]
    banks = {"CA": _bank({"a": (1e-6, 1), "b": (1e-6, 2)})}
    # c is missing from the bank, so it counts as found in no sample.
    one = _call(rows, 6, banks=banks, max_samples=1)
    assert _partition(one.codes) == {frozenset({0, 1}), frozenset({4, 5})}
    assert one.counts["alleles_in_too_many_samples"] == 1
    two = _call(rows, 6, banks=banks, max_samples=2)
    assert len(_partition(two.codes)) == 3
    # max_samples=0 keeps only alleles that the bank lacks, as DARLIN's de novo
    # alleles are.
    none = _call(rows, 6, banks=banks, max_samples=0)
    assert _partition(none.codes) == {frozenset({4, 5})}


def test_alleles_missing_from_the_bank_count_as_rare() -> None:
    rows = [(0, 0, 0), (1, 0, 0), (2, 0, 1), (3, 0, 1)]
    banks = {"CA": _bank({"a": (0.3, 3)})}
    called = _call(rows, 4, banks=banks, max_frequency=1e-12, max_samples=0)
    assert _partition(called.codes) == {frozenset({2, 3})}
    assert called.coverage == [
        {
            "character": "CA",
            "alleles": 2,
            "alleles_in_bank": 1,
            "observations": 4,
            "observations_in_bank": 2,
        }
    ]


def test_max_alleles_leaves_clones_of_more_rare_alleles_without_one() -> None:
    # Clone {0, 1, 2} holds a, b and c; clone {3, 4} holds d and e, joined
    # through d, and f in cell 5 alone.
    rows = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 1),
        (2, 1, 1),
        (2, 2, 2),
        (3, 0, 3),
        (4, 0, 3),
        (4, 1, 4),
        (5, 0, 5),
    ]
    three = _call(rows, 6, banks=_lacking(), max_frequency=1e-9, max_alleles=3)
    assert _partition(three.codes) == {
        frozenset({0, 1, 2}),
        frozenset({3, 4}),
        frozenset({5}),
    }
    two = _call(rows, 6, banks=_lacking(), max_frequency=1e-9, max_alleles=2)
    # The clone of three alleles goes; the others keep their order by size.
    assert two.codes.tolist() == [-1, -1, -1, 0, 0, 1]
    assert two.sizes.tolist() == [2, 1]
    assert two.clone_alleles.tolist() == [2, 1]
    assert two.counts["clones_with_too_many_alleles"] == 1
    assert two.counts["cells_in_clones_with_too_many_alleles"] == 3
    # Cells left without a clone keep their rare alleles.
    assert two.n_alleles.tolist() == [1, 2, 2, 1, 2, 1]


def test_frequencies_from_the_cells_are_shares_of_each_groups_edited_cells() -> None:
    # Group 0 has four cells edited at CA, two of which hold a: 1/2. Group 1
    # has two, both holding a: 1. Below 0.75, a is rare in group 0 alone.
    rows = [(0, 0, 0), (1, 0, 0), (2, 0, 1), (3, 0, 2), (4, 0, 0), (5, 0, 0)]
    called = _call(rows, 6, max_frequency=0.75, groups=[0, 0, 0, 0, 1, 1])
    assert _partition(called.codes) == {
        frozenset({0, 1}),
        frozenset({2}),
        frozenset({3}),
    }
    # A placeholder is an edit, so it counts among a character's edited cells:
    # with cell 2's b turned into one, a holds 2 of 4 cells, still 1/2.
    rows[2] = (2, 0, -1)
    assert (
        _call(rows, 6, max_frequency=0.51, groups=[0, 0, 0, 0, 1, 1]).counts[
            "alleles_too_frequent"
        ]
        == 1
    )


def test_groups_never_share_a_clone() -> None:
    rows = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
    called = _call(rows, 4, banks=_lacking(), max_frequency=1e-9, groups=[0, 1, 0, 1])
    assert _partition(called.codes) == {frozenset({0, 2}), frozenset({1, 3})}


def test_p_clonal_counts_every_edited_cell_of_the_character_among_its_observations() -> (
    None
):
    # a, at rate 0.05, is held by cells 0 and 1. With M = 2 edited cells,
    # x = 0.1 and p = 0.04917 survives 0.05 alone; a placeholder in cell 2
    # makes M = 3, x = 0.15 and p = 0.0731, which does not.
    banks = {"CA": _bank({"a": (0.05, 1)})}
    alone = _call([(0, 0, 0), (1, 0, 0)], 3, method="p_clonal", banks=banks)
    assert _partition(alone.codes) == {frozenset({0, 1})}
    crowded = _call(
        [(0, 0, 0), (1, 0, 0), (2, 0, -1)], 3, method="p_clonal", banks=banks
    )
    assert crowded.n_clones == 0
    assert crowded.counts["alleles_not_significant"] == 1
    # The same edits within two groups: the placeholder's group is cell 2's
    # own, so a's group keeps M = 2.
    split = _call(
        [(0, 0, 0), (1, 0, 0), (2, 0, -1)],
        3,
        method="p_clonal",
        banks=banks,
        groups=[0, 0, 1],
    )
    assert _partition(split.codes) == {frozenset({0, 1})}


def test_p_clonal_corrects_for_the_alleles_of_each_group_apart() -> None:
    # In group 0, a at rate 0.006 among M = 9 edited cells gives x = 0.054,
    # e^(-x) = 0.94743 and p = (1 - 1.054·0.94743) / (1 - 0.94743) = 0.0268;
    # alone it survives 0.05, but beside b, at rate 0.5 (p near 0.94), BH
    # needs p·2 ≤ 0.05, which it does not meet. Group 1's c, missing
    # from the bank, has p = 0 and survives whatever group 0 holds.
    banks = {"CA": _bank({"a": (0.006, 1), "b": (0.5, 3)})}
    rows = [(0, 0, 0), (1, 0, 0)] + [(cell, 0, 1) for cell in range(2, 9)]
    rows += [(9, 0, 2), (10, 0, 2)]
    groups = [0] * 9 + [1, 1]
    called = _call(rows, 11, method="p_clonal", banks=banks, groups=groups)
    assert _partition(called.codes) == {frozenset({9, 10})}
    assert called.counts["alleles_not_significant"] == 2
    expected = clonal_p_values([0.006], [9])[0]
    assert expected == pytest.approx(0.0268, abs=1e-4)
    # Without b's cells, a alone in its group survives.
    alone = _call(
        rows[:2] + rows[-2:], 11, method="p_clonal", banks=banks, groups=groups
    )
    assert _partition(alone.codes) == {frozenset({0, 1}), frozenset({9, 10})}


def test_banks_must_name_the_characters_of_the_alleles() -> None:
    rows = [(0, 0, 0), (1, 1, 1)]
    with pytest.raises(ValueError, match="names 'XA', which is no character"):
        _call(rows, 2, banks={"XA": _bank({"a": (0.1, 1)})}, max_frequency=0.5)
    with pytest.raises(ValueError, match="holds no bank of 'TA'"):
        _call(rows, 2, banks={"CA": _bank({"a": (0.1, 1)})}, max_frequency=0.5)
    # A character that holds only placeholders needs no bank.
    called = _call(
        [(0, 0, 0), (1, 1, -1)],
        2,
        banks={"CA": _bank({"a": (0.1, 1)})},
        max_frequency=0.5,
    )
    assert called.n_clones == 1
    with pytest.raises(ValueError, match="share the name 'CA'"):
        _call(
            rows,
            2,
            banks={"CA": _bank({"a": (0.1, 1)})},
            max_frequency=0.5,
            characters=("CA", "CA", "RA"),
        )


def test_inconsistent_observations_are_refused() -> None:
    settings = allele_call_settings("graph", max_frequency=0.5)
    for rows, n_cells, match in [
        ([(3, 0, 0)], 3, "cells must be positions below 3"),
        ([(0, 5, 0)], 3, "characters must be positions"),
        ([(0, 0, 9)], 3, "alleles must be positions"),
        ([(0, 0, -2)], 3, "alleles must be positions"),
    ]:
        with pytest.raises(ValueError, match=match):
            allele_clone_calls(
                _observations(rows), n_cells, groups=None, settings=settings
            )
    with pytest.raises(ValueError, match="one integer code per cell"):
        allele_clone_calls(
            _observations([(0, 0, 0)]),
            2,
            groups=np.zeros(3, dtype=np.int64),
            settings=settings,
        )
    with pytest.raises(ValueError, match="at least 0"):
        allele_clone_calls(
            _observations([(0, 0, 0)]), 2, groups=np.array([0, -1]), settings=settings
        )
    broken = AlleleObservations(
        cells=np.zeros(2, dtype=np.int64),
        characters=np.zeros(1, dtype=np.int64),
        alleles=np.zeros(2, dtype=np.int64),
        character_names=("CA",),
        allele_texts=np.asarray(["a"], dtype=object),
    )
    with pytest.raises(ValueError, match="one cell, character and allele"):
        allele_clone_calls(broken, 2, groups=None, settings=settings)


def test_a_budget_too_small_is_refused_with_the_bytes_it_needs() -> None:
    observations = _observations([(0, 0, 0), (1, 0, 0)])
    settings = allele_call_settings("graph", max_frequency=1.0)
    with pytest.raises(MemoryError, match=r"holds about \d+ bytes for the cells, more"):
        allele_clone_calls(
            observations, 2, groups=None, settings=settings, max_bytes=100
        )


def test_no_edits_call_no_clone() -> None:
    called = _call([], 4, max_frequency=0.5)
    assert called.codes.tolist() == [-1] * 4
    assert called.n_clones == 0
    assert called.counts["alleles"] == 0


def _random_case(rng: np.random.Generator):
    n_cells = int(rng.integers(1, 60))
    n_characters = int(rng.integers(1, 4))
    n_rows = int(rng.integers(0, 150))
    rows = [
        (
            int(rng.integers(0, n_cells)),
            int(rng.integers(0, n_characters)),
            int(rng.integers(-1, len(_TEXTS))),
        )
        for _ in range(n_rows)
    ]
    names = ("CA", "TA", "RA")[:n_characters]
    groups = rng.integers(0, 3, size=n_cells)
    banks = None
    if rng.random() < 0.6:
        banks = {
            name: _bank(
                {
                    text: (
                        float(rng.choice([1e-6, 0.01, 0.2])),
                        int(rng.integers(0, 4)),
                    )
                    for text in _TEXTS
                    if rng.random() < 0.6
                }
                or {"z": (0.1, 1)}
            )
            for name in names
        }
    return n_cells, names, rows, groups, banks


def _expected(n_cells, names, rows, groups, banks, method, rules):
    """The rules read again on Python sets, and the clones by networkx."""
    edits = {(cell, character, allele) for cell, character, allele in rows}
    edited = {}
    holders = {}
    for cell, character, allele in edits:
        edited.setdefault((groups[cell], character), set()).add(cell)
        if allele >= 0:
            holders.setdefault((groups[cell], character, allele), set()).add(cell)
    rates, samples = {}, {}
    for key, cells in holders.items():
        group, character, allele = key
        if banks is None:
            rates[key] = len(cells) / len(edited[(group, character)])
            samples[key] = 0
        else:
            table = banks[names[character]].set_index("allele")
            text = _TEXTS[allele]
            listed = text in table.index
            rates[key] = float(table.loc[text, "normalized_count"]) if listed else 0.0
            samples[key] = int(table.loc[text, "sample_count"]) if listed else 0
    if method == "graph":
        rare = {
            key
            for key in holders
            if rules.get("max_frequency") is None
            or rates[key] <= rules["max_frequency"]
        }
    else:
        rare = set()
        by_group = {}
        for key in holders:
            by_group.setdefault(key[0], []).append(key)
        for keys in by_group.values():
            p_values = [
                0.0
                if rates[key] * len(edited[(key[0], key[1])]) == 0
                else (
                    lambda x: (1 - math.exp(-x) - x * math.exp(-x)) / (1 - math.exp(-x))
                )(rates[key] * len(edited[(key[0], key[1])]))
                for key in keys
            ]
            adjusted = stats.false_discovery_control(p_values, method="bh")
            rare |= {key for key, value in zip(keys, adjusted) if value <= 0.05}
    if rules.get("max_samples") is not None:
        rare = {key for key in rare if samples[key] <= rules["max_samples"]}
    graph = nx.Graph()
    cell_alleles = {cell: set() for cell in range(n_cells)}
    for key in rare:
        for cell in holders[key]:
            graph.add_edge(("cell", cell), ("allele", key))
            cell_alleles[cell].add(key)
    partition = set()
    for component in nx.connected_components(graph):
        cells = frozenset(node[1] for node in component if node[0] == "cell")
        alleles = {node[1] for node in component if node[0] == "allele"}
        if rules.get("max_alleles") is None or len(alleles) <= rules["max_alleles"]:
            partition.add(cells)
    return partition, [len(cell_alleles[cell]) for cell in range(n_cells)]


def test_random_edits_give_the_clones_of_the_rules_read_on_sets() -> None:
    rng = np.random.default_rng(20)
    for trial in range(250):
        n_cells, names, rows, groups, banks = _random_case(rng)
        method = "p_clonal" if banks is not None and rng.random() < 0.5 else "graph"
        rules = {}
        if method == "graph":
            rules["max_frequency"] = float(rng.choice([2e-5, 0.05, 0.5, 1.0]))
        if banks is not None and rng.random() < 0.5:
            rules["max_samples"] = int(rng.integers(0, 3))
        if rng.random() < 0.5:
            rules["max_alleles"] = int(rng.integers(1, 4))
        called = _call(
            rows,
            n_cells,
            method=method,
            banks=banks,
            groups=groups,
            characters=names,
            **rules,
        )
        partition, n_alleles = _expected(
            n_cells, names, rows, groups.tolist(), banks, method, rules
        )
        assert _partition(called.codes) == partition, trial
        assert called.n_alleles.tolist() == n_alleles, trial
        # Clones are numbered by size, then by their first cell.
        firsts = [
            min(np.flatnonzero(called.codes == code).tolist())
            for code in range(called.n_clones)
        ]
        keys = [(-int(size), first) for size, first in zip(called.sizes, firsts)]
        assert keys == sorted(keys), trial


def test_clones_do_not_follow_the_order_of_the_edits() -> None:
    rng = np.random.default_rng(7)
    for _trial in range(30):
        n_cells, names, rows, groups, banks = _random_case(rng)
        if not rows:
            continue
        rules = {"max_frequency": 0.5, "max_alleles": 3}
        first = _call(
            rows, n_cells, banks=banks, groups=groups, characters=names, **rules
        )
        shuffled = [rows[index] for index in rng.permutation(len(rows))]
        second = _call(
            shuffled, n_cells, banks=banks, groups=groups, characters=names, **rules
        )
        np.testing.assert_array_equal(first.codes, second.codes)
        np.testing.assert_array_equal(first.n_alleles, second.n_alleles)
        assert first.counts == second.counts
        assert first.coverage == second.coverage


# Memory ---------------------------------------------------------------------------


def _darlin_like(n_cells: int, seed: int = 0):
    """Three arrays read in four cells in five, alleles of a Zipf law, a bank of half."""
    rng = np.random.default_rng(seed)
    n_alleles = n_cells // 2
    cells, characters, alleles = [], [], []
    for array in range(3):
        read = np.flatnonzero(rng.random(n_cells) < 0.8)
        drawn = np.minimum(rng.zipf(1.3, size=len(read)) - 1, n_alleles - 1)
        drawn[rng.random(len(read)) < 0.02] = -1
        cells.append(read)
        characters.append(np.full(len(read), array))
        alleles.append(np.where(drawn >= 0, drawn + array * n_alleles, -1))
    texts = np.asarray(
        [f"{a % 200 + 20}_{a % 200 + 25 + a // 200}del" for a in range(3 * n_alleles)],
        dtype=object,
    )
    observations = AlleleObservations(
        cells=np.concatenate(cells).astype(np.int64),
        characters=np.concatenate(characters).astype(np.int64),
        alleles=np.concatenate(alleles).astype(np.int64),
        character_names=("CA", "TA", "RA"),
        allele_texts=texts,
    )
    banks = {}
    for array, name in enumerate(("CA", "TA", "RA")):
        listed = rng.choice(n_alleles, size=n_alleles // 2, replace=False)
        banks[name] = pd.DataFrame(
            {
                "allele": texts[listed + array * n_alleles],
                "normalized_count": rng.random(len(listed)) * 1e-4,
                "sample_count": rng.integers(0, 4, size=len(listed)),
            }
        )
    return observations, allele_banks(banks)


def _wide(n_cells: int, n_characters: int = 300, seed: int = 1) -> AlleleObservations:
    """Marks 1 to 8 at four sites in ten of PEtracer-like characters, a few placeholders."""
    rng = np.random.default_rng(seed)
    cells = np.repeat(np.arange(n_cells), n_characters)
    characters = np.tile(np.arange(n_characters), n_cells)
    kept = rng.random(len(cells)) < 0.4
    return AlleleObservations(
        cells=cells[kept].astype(np.int64),
        characters=characters[kept].astype(np.int64),
        alleles=rng.integers(-1, 8, size=int(kept.sum())).astype(np.int64),
        character_names=tuple(f"intID{c // 3}-s{c % 3}" for c in range(n_characters)),
        allele_texts=np.asarray([str(mark) for mark in range(1, 9)], dtype=object),
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("shape", "method", "rules", "n_groups"),
    [
        (
            "darlin",
            "graph",
            {"max_frequency": 2e-5, "max_samples": 1, "max_alleles": 3},
            1,
        ),
        ("darlin", "p_clonal", {"max_alleles": 3}, 1),
        ("darlin", "graph", {"max_frequency": 2e-5}, 40),
        ("wide", "graph", {"max_frequency": 0.2}, 1),
    ],
)
def test_calls_hold_no_more_than_the_smallest_budget_they_accept(
    shape, method, rules, n_groups
) -> None:
    if shape == "darlin":
        observations, banks = _darlin_like(20_000)
        n_cells = 20_000
    else:
        observations, banks = _wide(1_500), None
        n_cells = 1_500
    settings = allele_call_settings(method, banks=banks, **rules)
    groups = np.arange(n_cells, dtype=np.int64) % n_groups

    def call(budget: int) -> object:
        return allele_clone_calls(
            observations,
            n_cells,
            groups=groups,
            settings=settings,
            banks=banks,
            max_bytes=budget,
        )

    # Compiled kernels are warmed before anything is traced.
    call(1 << 40)
    budget = smallest_budget(call, 1 << 30)
    _, peak = traced_peak(lambda: call(budget), budget)
    assert peak <= budget
    with pytest.raises(MemoryError, match="Raise mem_budget"):
        call(budget - 1)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("method", "rules", "n_groups"),
    [
        ("graph", {"max_frequency": 2e-5, "max_samples": 1, "max_alleles": 3}, 1),
        ("p_clonal", {"max_alleles": 3}, 40),
    ],
)
def test_each_step_on_two_million_edits_holds_no_more_than_its_check_states(
    monkeypatch, method, rules, n_groups
) -> None:
    observations, banks = _darlin_like(900_000, seed=4)
    assert len(observations) > 2_000_000
    settings = allele_call_settings(method, banks=banks, **rules)
    groups = np.arange(900_000, dtype=np.int64) % n_groups
    budget = 1 << 34

    def call() -> object:
        return allele_clone_calls(
            observations,
            900_000,
            groups=groups,
            settings=settings,
            banks=banks,
            max_bytes=budget,
        )

    checks, peak = traced_steps(monkeypatch, call, budget)
    # Each step holds its shortfall from the largest check beside it.
    ceiling = max(stated for _what, stated in checks)
    assert peak <= ceiling, (peak - ceiling, checks)
    assert len(checks) > 10
