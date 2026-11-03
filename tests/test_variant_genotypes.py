"""The genotype call vocabulary and the rules that call genotypes.

Expected values are derived by hand from the rules' definitions: GoT's
mutant-UMI and share rules, GIFT's edit-distance assignment of gapfills and
its per-cell posterior under a uniform prior.
"""

from fractions import Fraction

import numpy as np
import pytest

from cytearc.variants.genotypes import (
    ALTERNATIVE_PRESENT,
    CALLS,
    FILTERED,
    HETEROZYGOUS,
    HIGH_CONFIDENCE_SWAP,
    HOMOZYGOUS_ALTERNATIVE,
    MISSING,
    MUTANT_CALLS,
    SOURCE_CALLS,
    SOURCE_PRECEDENCE,
    UNKNOWN_CALLS,
    WILD_TYPE,
    edit_distance,
    gapfill_wild_type_probabilities,
    genotype_rules,
    gift_calls,
    gift_posteriors,
    source_call_codes,
    swap_probability_table,
    threshold_calls,
    unit_swap_probabilities,
)


def test_the_vocabulary_names_six_calls_in_the_order_of_their_codes() -> None:
    assert CALLS == (
        "wild_type",
        "heterozygous",
        "homozygous_alternative",
        "alternative_present",
        "missing",
        "filtered",
    )
    codes = (
        WILD_TYPE,
        HETEROZYGOUS,
        HOMOZYGOUS_ALTERNATIVE,
        ALTERNATIVE_PRESENT,
        MISSING,
        FILTERED,
    )
    assert [CALLS[code] for code in codes] == list(CALLS)
    assert MUTANT_CALLS == {HETEROZYGOUS, HOMOZYGOUS_ALTERNATIVE, ALTERNATIVE_PRESENT}
    assert UNKNOWN_CALLS == {MISSING, FILTERED}
    assert set(SOURCE_PRECEDENCE) == set(SOURCE_CALLS)


@pytest.mark.parametrize(
    ("column", "labels", "covered", "expected"),
    [
        # GEO's GoT tables: MUT carries the mutation without a zygosity.
        ("Genotype", ["WT", "MUT"], [True, True], [WILD_TYPE, ALTERNATIVE_PRESENT]),
        # GoT-Multi: Unprofiled beside ambiguous UMIs is evidence without a
        # call, and without any evidence no call at all.
        (
            "genotype_merged",
            ["Unprofiled", "Unprofiled", "MUT"],
            [True, False, True],
            [FILTERED, MISSING, ALTERNATIVE_PRESENT],
        ),
        (
            "genotype_final",
            ["FalsePositive", "WT"],
            [True, True],
            [FILTERED, WILD_TYPE],
        ),
        # Gotcha tells heterozygotes apart.
        (
            "genotype_pred",
            ["HET", "MUT", "WT"],
            [True, True, True],
            [HETEROZYGOUS, ALTERNATIVE_PRESENT, WILD_TYPE],
        ),
        # Tapestri's NGT: 3 at a covered site is a call its quality filter
        # rejected, and at an uncovered one no evidence.
        (
            "NGT",
            ["0", "1", "2", "3", "3"],
            [True, True, True, True, False],
            [WILD_TYPE, HETEROZYGOUS, HOMOZYGOUS_ALTERNATIVE, FILTERED, MISSING],
        ),
        # DAb-seq's GT codes the genotypes as NGT does.
        (
            "GT",
            ["3", "2", "1", "0", "3"],
            [False, True, True, True, True],
            [MISSING, HOMOZYGOUS_ALTERNATIVE, HETEROZYGOUS, WILD_TYPE, FILTERED],
        ),
        # No labels give no calls.
        ("NGT", [], [], []),
    ],
)
def test_source_labels_map_to_calls(column, labels, covered, expected) -> None:
    calls = source_call_codes(
        column, np.asarray(labels, dtype=object), np.asarray(covered)
    )
    assert calls.dtype == np.int8
    assert calls.tolist() == expected


def test_unknown_source_columns_and_labels_are_refused() -> None:
    with pytest.raises(ValueError, match="'genotype_call' have no vocabulary"):
        source_call_codes(
            "genotype_call", np.asarray(["MUT"], dtype=object), np.ones(1, bool)
        )
    with pytest.raises(ValueError, match="holds the call 'HOM'"):
        source_call_codes("Genotype", np.asarray(["WT", "HOM"]), np.ones(2, bool))
    # GoT's tables do not write Unprofiled, so it is no label of theirs.
    with pytest.raises(ValueError, match="'Unprofiled'"):
        source_call_codes("Genotype", np.asarray(["Unprofiled"]), np.ones(1, bool))
    # The first unknown label in text order is named, whatever the rows' order.
    with pytest.raises(ValueError, match="holds the call '4'"):
        source_call_codes("NGT", np.asarray(["7", "0", "4"]), np.ones(3, bool))
    with pytest.raises(ValueError, match="'NGT' holds a call that is not text"):
        source_call_codes(
            "NGT", np.asarray(["0", None], dtype=object), np.ones(2, bool)
        )


def test_the_patient_rule_calls_one_mutant_umi_mutant() -> None:
    reference = np.asarray([3, 0, 2, 0, 0, 5])
    alternative = np.asarray([0, 1, 1, 0, 2, 0])
    calls = threshold_calls(
        reference, alternative, min_umis=1, min_mutant_umis=1, min_fraction=None
    )
    assert calls.tolist() == [
        WILD_TYPE,
        ALTERNATIVE_PRESENT,
        ALTERNATIVE_PRESENT,
        FILTERED,
        ALTERNATIVE_PRESENT,
        WILD_TYPE,
    ]
    # Two mutant UMIs: one leaves the cell neither mutant nor wild type.
    calls = threshold_calls(
        reference, alternative, min_umis=1, min_mutant_umis=2, min_fraction=None
    )
    assert calls.tolist()[1:5] == [FILTERED, FILTERED, FILTERED, ALTERNATIVE_PRESENT]
    # Three UMIs of the two alleles before any call.
    calls = threshold_calls(
        reference, alternative, min_umis=3, min_mutant_umis=1, min_fraction=None
    )
    assert calls.tolist() == [
        WILD_TYPE,
        FILTERED,
        ALTERNATIVE_PRESENT,
        FILTERED,
        FILTERED,
        WILD_TYPE,
    ]


def test_the_cell_line_rule_needs_a_share_above_the_fraction() -> None:
    # 9 of 10 is 90% exactly, which does not exceed 90%; 10 of 11 does.
    reference = np.asarray([1, 1, 10, 5, 0, 11])
    alternative = np.asarray([9, 10, 1, 5, 3, 1])
    calls = threshold_calls(
        reference, alternative, min_umis=1, min_mutant_umis=None, min_fraction=0.9
    )
    assert calls.tolist() == [
        FILTERED,
        ALTERNATIVE_PRESENT,
        WILD_TYPE,
        FILTERED,
        ALTERNATIVE_PRESENT,
        WILD_TYPE,
    ]
    # With both rules, a mutant call needs both.
    calls = threshold_calls(
        reference, alternative, min_umis=1, min_mutant_umis=4, min_fraction=0.9
    )
    assert calls.tolist()[1] == ALTERNATIVE_PRESENT
    assert calls.tolist()[4] == FILTERED


@pytest.mark.parametrize(
    ("first", "second", "distance"),
    [
        ("CAC", "AAC", 1),
        ("GTAAG", "GTAG", 1),
        ("", "ACG", 3),
        ("kitten", "sitting", 3),
        ("ATCT", "ATCTTGT", 3),
        ("GATTACA", "GATTACA", 0),
    ],
)
def test_edit_distances(first, second, distance) -> None:
    assert edit_distance(first, second) == distance
    assert edit_distance(second, first) == distance


def test_gapfills_go_to_the_allele_they_lie_closer_to() -> None:
    probabilities = gapfill_wild_type_probabilities(
        ["CAC", "AAC", "CAA", "", "AAA"],
        ["CAC"] * 5,
        ["AAC"] * 5,
    )
    # A match one edit from the other allele: 1 / (1 + 0.01); CAA lies one
    # edit from CAC and two from AAC; the empty gapfill three from both;
    # AAA two from CAC and one from AAC.
    expected = [
        Fraction(100, 101),
        Fraction(1, 101),
        Fraction(100, 101),
        Fraction(1, 2),
        Fraction(1, 101),
    ]
    np.testing.assert_allclose(probabilities, [float(x) for x in expected], rtol=1e-14)
    # A gapfill 400 edits from the wild type and 399 from the mutant goes to
    # the mutant as one a single edit closer does, rather than to an
    # undefined share of two vanishing likelihoods.
    far = gapfill_wild_type_probabilities(["T" * 400], ["A" * 400], ["A" * 399 + "T"])
    assert far[0] == pytest.approx(1 / 101, rel=1e-14)


def test_swap_tables_are_step_functions_of_reads_per_umi() -> None:
    table = swap_probability_table({1: 0.9, 2: 0.5, 5: 0.05})
    np.testing.assert_array_equal(table[0], [1, 2, 5])
    # An entry of 3 UMIs and 13 reads gives each UMI 4 reads, which take the
    # value of 2 reads; 5 reads and more take that of 5.
    probabilities = unit_swap_probabilities(
        np.asarray([1, 3, 1, 2]), np.asarray([1, 13, 5, 400]), table
    )
    assert probabilities.tolist() == [0.9, 0.5, 0.05, 0.05]
    # An entry whose UMIs hold no reads has no probability.
    with pytest.raises(ValueError, match="no probability for UMIs of 0 reads"):
        unit_swap_probabilities(np.asarray([2]), np.asarray([1]), table)


@pytest.mark.parametrize(
    ("table", "error", "match"),
    [
        ([0.5], TypeError, "must map reads per UMI"),
        ({}, ValueError, "at least one probability"),
        ({0: 0.5}, ValueError, "at least 1"),
        ({1: 1.5}, ValueError, "probabilities lie from 0 to 1"),
        ({1.5: 0.5}, TypeError, "whole numbers"),
        ({True: 0.5}, TypeError, "whole numbers"),
        ({1: "0.5"}, TypeError, "must be numbers"),
        ({2: 0.1, 5: 0.0}, ValueError, "starts at UMIs of 2 reads"),
    ],
)
def test_swap_tables_are_checked(table, error, match) -> None:
    with pytest.raises(error, match=match):
        swap_probability_table(table)


def _exact_posteriors(scores: list[tuple[Fraction, Fraction, Fraction]]) -> list[float]:
    likelihoods = [Fraction(1), Fraction(1), Fraction(1)]
    for score in scores:
        likelihoods = [
            likelihood * value
            for likelihood, value in zip(likelihoods, score, strict=True)
        ]
    total = sum(likelihoods)
    return [float(likelihood / total) for likelihood in likelihoods]


def test_gift_posteriors_follow_the_umis_of_each_cell() -> None:
    wild, mutant = Fraction(100, 101), Fraction(1, 101)
    # Two UMIs of the wild type and one of the mutant: the heterozygote
    # scores half of each UMI's allele, 50/101.
    of_wild = (wild, Fraction(50, 101), mutant)
    of_mutant = (mutant, Fraction(50, 101), wild)
    posteriors, confident = gift_posteriors(
        np.asarray([0, 0, 1]),
        2,
        np.asarray([2, 1, 3]),
        np.asarray([float(wild), float(mutant), float(mutant)]),
        np.zeros(3),
    )
    np.testing.assert_allclose(
        posteriors[0], _exact_posteriors([of_wild, of_wild, of_mutant]), rtol=1e-12
    )
    # 10,000, 125,000 and 100 over 101 cubed: the heterozygote at 0.925.
    assert posteriors[0, 1] == pytest.approx(125_000 / 135_100, rel=1e-12)
    np.testing.assert_allclose(
        posteriors[1], _exact_posteriors([of_mutant] * 3), rtol=1e-12
    )
    assert confident.tolist() == [3, 3]


def test_a_likely_swap_scores_every_genotype_alike() -> None:
    wild, mutant = Fraction(100, 101), Fraction(1, 101)
    swap = Fraction(3, 5)
    # The mutant UMI's swap probability of 0.6 lifts its scores of the wild
    # type and the heterozygote to 0.6, so it counts mostly for neither.
    posteriors, confident = gift_posteriors(
        np.zeros(2, dtype=np.int64),
        1,
        np.asarray([2, 1]),
        np.asarray([float(wild), float(mutant)]),
        np.asarray([0.0, float(swap)]),
    )
    expected = _exact_posteriors(
        [
            (wild, Fraction(50, 101), mutant),
            (wild, Fraction(50, 101), mutant),
            (swap, swap, wild),
        ]
    )
    np.testing.assert_allclose(posteriors[0], expected, rtol=1e-12)
    # Only the two wild-type UMIs lie below a swap probability of 0.1.
    assert confident.tolist() == [2]
    assert HIGH_CONFIDENCE_SWAP == 0.1


def test_a_umi_confident_below_a_swap_probability_of_one_tenth() -> None:
    # UMIs whose swap probability is exactly 0.1 are not confident; just
    # below it they are, as GIFT counts the UMIs of a genotyped cell.
    _posteriors, confident = gift_posteriors(
        np.asarray([0, 0, 1], dtype=np.int64),
        2,
        np.asarray([3, 2, 4]),
        np.asarray([0.99, 0.99, 0.99]),
        np.asarray([0.1, np.nextafter(0.1, 0.0), 0.1]),
    )
    assert confident.tolist() == [2, 0]


def test_many_umis_do_not_underflow() -> None:
    posteriors, _confident = gift_posteriors(
        np.zeros(1, dtype=np.int64),
        1,
        np.asarray([5000]),
        np.asarray([100 / 101]),
        np.zeros(1),
    )
    assert posteriors[0].tolist() == pytest.approx([1.0, 0.0, 0.0])


def test_gift_calls_need_a_posterior_above_the_bound_and_confident_umis() -> None:
    posteriors = np.asarray(
        [
            [0.7, 0.2, 0.1],
            [0.6, 0.3, 0.1],
            [0.1, 0.8, 0.1],
            [0.05, 0.05, 0.9],
            [0.9, 0.05, 0.05],
            [np.nan, np.nan, np.nan],
        ]
    )
    confident = np.asarray([1, 1, 2, 1, 0, 3])
    calls = gift_calls(posteriors, confident, min_posterior=0.6, min_umis=1)
    # 0.6 exactly does not exceed 0.6; the fifth row has no confident UMI.
    assert calls.tolist() == [
        WILD_TYPE,
        FILTERED,
        HETEROZYGOUS,
        HOMOZYGOUS_ALTERNATIVE,
        FILTERED,
        FILTERED,
    ]
    calls = gift_calls(posteriors, confident, min_posterior=0.6, min_umis=2)
    assert calls.tolist()[:3] == [FILTERED, FILTERED, HETEROZYGOUS]


@pytest.mark.parametrize(
    ("method", "options", "error", "match"),
    [
        ("vote", {}, ValueError, "method must be one of"),
        (None, {}, ValueError, "method must be one of"),
        ("source", {"min_umis": 2}, ValueError, "uses no min_umis"),
        ("source", {"alleles": {"x": ("A", "C")}}, ValueError, "uses no alleles"),
        ("source", {"min_posterior": 0.9}, ValueError, "uses no min_posterior"),
        ("threshold", {}, ValueError, "needs min_mutant_umis or min_fraction"),
        (
            "threshold",
            {"min_mutant_umis": 1, "min_posterior": 0.8},
            ValueError,
            "uses no min_posterior",
        ),
        ("threshold", {"min_fraction": 0.4}, ValueError, "from 0.5 to below 1"),
        ("threshold", {"min_fraction": 1.0}, ValueError, "from 0.5 to below 1"),
        ("threshold", {"min_fraction": True}, TypeError, "must be a number"),
        ("threshold", {"min_mutant_umis": 0}, ValueError, "at least 1"),
        ("threshold", {"min_mutant_umis": 1.5}, TypeError, "whole number"),
        ("threshold", {"min_mutant_umis": True}, TypeError, "whole number"),
        ("gift", {"min_mutant_umis": 1}, ValueError, "uses no min_mutant_umis"),
        ("gift", {"min_fraction": 0.9}, ValueError, "uses no min_fraction"),
        ("gift", {"min_posterior": 1.0}, ValueError, "from 0.0 to below 1.0"),
        ("gift", {"alleles": ["A", "C"]}, TypeError, "must map site labels"),
        ("gift", {"alleles": {"p": "AC"}}, TypeError, "a wild-type and an"),
        ("gift", {"alleles": {3: ("A", "C")}}, TypeError, "keyed by site labels"),
        ("gift", {"alleles": {"p": ("A", 3)}}, TypeError, "must be text"),
        ("gift", {"alleles": {"p": ("A", "A")}}, ValueError, "as both the wild"),
    ],
)
def test_genotype_rules_refuse_arguments_their_method_does_not_use(
    method, options, error, match
) -> None:
    with pytest.raises(error, match=match):
        genotype_rules(method, **options)


def test_genotype_rules_resolve_their_defaults() -> None:
    threshold = genotype_rules("threshold", min_mutant_umis=1)
    assert threshold.min_umis == 1
    assert threshold.min_posterior is None
    gift = genotype_rules("gift", swap_probabilities={1: 0.5, 3: 0.0})
    assert gift.min_posterior == 0.6 and gift.min_umis == 1
    assert gift.swap_record == ((1, 0.5), (3, 0.0))
    source = genotype_rules("source")
    assert source.min_umis is None and source.swap is None
    # NumPy's integers and floats read as Python's.
    rules = genotype_rules(
        "threshold", min_mutant_umis=np.int64(2), min_fraction=np.float64(0.75)
    )
    assert (rules.min_mutant_umis, rules.min_fraction) == (2, 0.75)
