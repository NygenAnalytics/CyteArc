"""Per-variant statistics, presets, labels and masks, against values derived by hand."""

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cytearc.variants.counts import Contig, SiteTable
from cytearc.variants.masks import RCRS_LENGTH, VARIANT_MASKS, masked_variants
from cytearc.variants.statistics import (
    INFORMATIVE_PRESETS,
    VariantStatisticsAccumulator,
    informative_variants,
    resolve_thresholds,
    variant_labels,
    variant_selection_frame,
)

DATA = Path(__file__).parent / "data" / "variants"


def _sites() -> SiteTable:
    """One site at position 5, reference A, and a named target."""
    return SiteTable(
        contigs=(Contig("chrM", 16569, None),),
        site_contig=np.asarray([0, -1]),
        site_position=np.asarray([5, -1]),
        site_name=np.asarray(["", "IDH2_Mut1"], dtype=object),
        site_channel=np.asarray(["", ""], dtype=object),
        allele_site=np.asarray([0, 0, 0, 0, 1, 1]),
        allele=np.asarray(["A", "C", "G", "T", "WT", "MUT"], dtype=object),
        allele_role=np.asarray(
            [
                "reference",
                "alternative",
                "alternative",
                "alternative",
                "reference",
                "alternative",
            ],
            dtype=object,
        ),
    )


# Four cells at the site: cell 0 has depth 10 with 6 A and 4 C (2 forward,
# 2 reverse); cell 1 depth 5, all A; cell 2 is not covered; cell 3 depth 20
# with 9 A, 10 C (6 and 4) and 1 G on the forward strand.
DEPTH = {"cells": [0, 1, 3], "sites": [0, 0, 0], "depth": [10, 5, 20]}
COUNTS = {
    "cells": [0, 0, 1, 3, 3, 3],
    "alleles": [0, 1, 0, 0, 1, 2],
    "counts": [6, 4, 5, 9, 10, 1],
    "forward": [3, 2, 3, 5, 6, 1],
    "reverse": [3, 2, 2, 4, 4, 0],
    "quality_forward": [30.0, 32.0, 31.0, 30.0, 34.0, 20.0],
    "quality_reverse": [30.0, 36.0, 31.0, 30.0, 38.0, np.nan],
}


def _rows(cells: list[int]) -> tuple[dict, dict]:
    """The depth and count rows of some cells, as add() takes them."""
    depth_keep = np.isin(DEPTH["cells"], cells)
    count_keep = np.isin(COUNTS["cells"], cells)
    depth = {
        "depth_cells": np.asarray(DEPTH["cells"])[depth_keep],
        "depth_sites": np.asarray(DEPTH["sites"])[depth_keep],
        "depth": np.asarray(DEPTH["depth"])[depth_keep],
    }
    counts = {
        "count_cells": np.asarray(COUNTS["cells"])[count_keep],
        "count_alleles": np.asarray(COUNTS["alleles"])[count_keep],
        "counts": np.asarray(COUNTS["counts"])[count_keep],
        "forward": np.asarray(COUNTS["forward"])[count_keep],
        "reverse": np.asarray(COUNTS["reverse"])[count_keep],
        "quality_forward": np.asarray(COUNTS["quality_forward"])[count_keep],
        "quality_reverse": np.asarray(COUNTS["quality_reverse"])[count_keep],
    }
    return depth, counts


def _statistics(bands: list[list[int]], **options):
    accumulator = VariantStatisticsAccumulator(_sites(), **options)
    for band in bands:
        depth, counts = _rows(band)
        if not options.get("stranded", False):
            for name in ("forward", "reverse", "quality_forward", "quality_reverse"):
                counts.pop(name)
        elif not options.get("qualities", False):
            counts.pop("quality_forward")
            counts.pop("quality_reverse")
        accumulator.add(**depth, **counts)
    return accumulator.finish(4)


def test_statistics_follow_mgatk_definitions_derived_by_hand() -> None:
    statistics = _statistics([[0, 1, 2, 3]], stranded=True, qualities=True)
    # Only C and G are shown; T and the named target's mutant are not.
    assert statistics.alleles.tolist() == [1, 2]
    columns = statistics.arrays()
    # C: frequencies 0.4, 0 and 0.5 in the covered cells; pooled 14 / 35.
    np.testing.assert_allclose(columns["mean"], [14 / 35, 1 / 35])
    # Sample variances: of (0.4, 0, 0.5) and of (0, 0, 0.05).
    variance_c = ((0.4 - 0.3) ** 2 + 0.3**2 + 0.2**2) / 2
    variance_g = (2 * (0.05 / 3) ** 2 + (0.05 - 0.05 / 3) ** 2) / 2
    np.testing.assert_allclose(columns["variance"], [variance_c, variance_g])
    np.testing.assert_allclose(
        columns["vmr"], [variance_c / (14 / 35 + 1e-11), variance_g / (1 / 35 + 1e-11)]
    )
    assert columns["n_cells_covered"].tolist() == [3, 3]
    # Cells 0 and 3 read C at least twice on each strand; G is read once.
    assert columns["n_cells_conf_detected"].tolist() == [2, 0]
    # Two cells show C, so their strands correlate perfectly; one shows G.
    np.testing.assert_allclose(columns["strand_correlation"][0], 1.0)
    assert np.isnan(columns["strand_correlation"][1])
    assert columns["n_cells_over_5"].tolist() == [2, 1]
    assert columns["n_cells_over_10"].tolist() == [2, 0]
    assert columns["n_cells_over_20"].tolist() == [2, 0]
    assert columns["n_cells_over_95"].tolist() == [0, 0]
    np.testing.assert_allclose(columns["max_heteroplasmy"], [0.5, 0.05])
    # Depth summed over the four cells, the uncovered one included.
    np.testing.assert_allclose(columns["mean_coverage"], [35 / 4, 35 / 4])
    # MAESTER's quality: the mean of the strands that read the allele.
    np.testing.assert_allclose(columns["mean_quality"], [(32 + 36 + 34 + 38) / 4, 20.0])


def test_statistics_do_not_depend_on_the_bands_of_cells() -> None:
    whole = _statistics([[0, 1, 2, 3]], stranded=True, qualities=True).arrays()
    banded = _statistics([[0], [1], [2], [3]], stranded=True, qualities=True)
    for name, values in whole.items():
        np.testing.assert_allclose(banded.arrays()[name], values, rtol=1e-12)


def test_counts_without_strands_give_no_strand_or_quality_statistics() -> None:
    statistics = _statistics([[0, 1, 2, 3]], stranded=False, qualities=False)
    columns = statistics.arrays()
    assert "n_cells_conf_detected" not in columns
    assert "strand_correlation" not in columns
    assert "mean_quality" not in columns
    np.testing.assert_allclose(columns["mean"], [14 / 35, 1 / 35])
    with pytest.raises(ValueError, match="Base qualities come with strand"):
        VariantStatisticsAccumulator(_sites(), stranded=False, qualities=True)


def _one_variant_rows(forward: np.ndarray, reverse: np.ndarray) -> dict:
    """Rows of cells that each read only allele 1 at site 0, with these
    strand counts."""
    cells = np.arange(len(forward))
    return {
        "depth_cells": cells,
        "depth_sites": np.zeros(len(cells), dtype=int),
        "depth": forward + reverse,
        "count_cells": cells,
        "count_alleles": np.ones(len(cells), dtype=int),
        "counts": forward + reverse,
        "forward": forward,
        "reverse": reverse,
    }


def test_large_strand_sums_correlate_without_overflow() -> None:
    # 4,000 cells, half with about 1.9 million reads per strand and half with
    # a few: every sum stays exact in float64, but the cells squared times a
    # strand's variance, the correlation's variance terms, reach 1.4e19, past
    # int64's 9.2e18, so they must be taken as Python integers.
    rng = np.random.default_rng(1)
    forward = np.where(np.arange(4_000) % 2 == 0, 1_900_000, 1) + rng.integers(
        0, 50_000, size=4_000
    )
    reverse = forward + rng.integers(0, 50_000, size=4_000)
    variance_term = 4_000 * int((forward * forward).sum()) - int(forward.sum()) ** 2
    assert variance_term > np.iinfo(np.int64).max
    assert int((reverse * reverse).sum()) < 2**53
    accumulator = VariantStatisticsAccumulator(_sites(), stranded=True, qualities=False)
    accumulator.add(**_one_variant_rows(forward, reverse))
    correlation = accumulator.finish(4_000).strand_correlation
    assert correlation is not None
    expected = np.corrcoef(forward.astype(float), reverse.astype(float))[0, 1]
    np.testing.assert_allclose(correlation[0], expected, rtol=1e-12)


def test_counts_too_large_to_sum_exactly_are_refused() -> None:
    accumulator = VariantStatisticsAccumulator(_sites(), stranded=True, qualities=False)
    # A strand count whose square reaches 2**53 cannot be summed exactly.
    big = np.asarray([94_906_266, 5])
    with pytest.raises(ValueError, match="strand count exceeds 94906265"):
        accumulator.add(**_one_variant_rows(big, np.asarray([1, 2])))
    # Strand counts within the bound whose squares sum to 2**53 or more.
    near = np.full(2, 94_906_265)
    with pytest.raises(ValueError, match=r"sum to 2\*\*53 or more"):
        accumulator.add(**_one_variant_rows(near, near))


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ({"depth_cells": [0, 1]}, "depth rows' cells, sites and depths must align"),
        ({"count_cells": [0]}, "count rows' cells, alleles and counts must align"),
        (
            {"depth_cells": [3, 1, 0], "depth": [20, 5, 10]},
            "sorted by cell and site",
        ),
        ({"count_cells": [0, 0, 1, 3, 3, 2]}, "no depth at its site"),
        ({"counts": [6, 40, 5, 9, 10, 1]}, "outside 1 to its site's depth"),
    ],
)
def test_malformed_rows_are_refused(rows, message) -> None:
    depth, counts = _rows([0, 1, 2, 3])
    for name in ("quality_forward", "quality_reverse"):
        counts.pop(name)
    merged = {**depth, **counts}
    merged.update({name: np.asarray(values) for name, values in rows.items()})
    accumulator = VariantStatisticsAccumulator(_sites(), stranded=True, qualities=False)
    with pytest.raises(ValueError, match=message):
        accumulator.add(**merged)


def test_stranded_rows_need_their_strand_counts_and_cells_need_a_count() -> None:
    depth, counts = _rows([0])
    accumulator = VariantStatisticsAccumulator(_sites(), stranded=True, qualities=True)
    with pytest.raises(ValueError, match="need forward and reverse"):
        accumulator.add(**depth, count_cells=counts["count_cells"], count_alleles=counts["count_alleles"], counts=counts["counts"])  # fmt: skip
    with pytest.raises(ValueError, match="both strands' qualities"):
        accumulator.add(**depth, **{k: v for k, v in counts.items() if "quality" not in k})  # fmt: skip
    with pytest.raises(ValueError, match="at least one cell"):
        accumulator.finish(0)


def test_presets_hold_the_published_thresholds() -> None:
    # Signac's vignette and MAESTER's paper, with VMR for RNA's variability.
    # A threshold is reached, so a rule that a statistic exceed a value holds
    # the next floating-point number above it, written out here.
    assert dict(INFORMATIVE_PRESETS["atac"]) == {
        "min_cells": 5,
        "min_strand_correlation": 0.65,
        "min_vmr": 0.010000000000000002,
    }
    assert dict(INFORMATIVE_PRESETS["rna"]) == {
        "min_depth": 20.000000000000004,
        "min_quality": 30.000000000000004,
        "min_vmr": 0.010000000000000002,
    }
    assert INFORMATIVE_PRESETS["atac"]["min_vmr"] > 0.01
    assert np.nextafter(INFORMATIVE_PRESETS["rna"]["min_depth"], 0.0) == 20.0
    # The table is public and read-only, at both levels.
    import cytearc.variants

    assert cytearc.variants.INFORMATIVE_PRESETS is INFORMATIVE_PRESETS
    with pytest.raises(TypeError):
        INFORMATIVE_PRESETS["atac"]["min_vmr"] = 0.5  # type: ignore[index]
    with pytest.raises(TypeError):
        INFORMATIVE_PRESETS["dna"] = {}  # type: ignore[index]
    assert not hasattr(cytearc.variants, "PRESETS")


def test_thresholds_combine_a_preset_with_overrides() -> None:
    resolved = resolve_thresholds("atac", {"min_vmr": 0.05, "min_cells": None})
    assert resolved == {
        "min_cells": 5,
        "min_strand_correlation": 0.65,
        "min_vmr": 0.05,
        "min_depth": None,
        "min_quality": None,
    }
    # Each threshold resolves in its argument's own type: a whole number of
    # cells, and floats for the statistics.
    assert type(resolved["min_cells"]) is int
    assert type(resolve_thresholds(None, {"min_cells": 2.0})["min_cells"]) is int
    assert resolve_thresholds(None, {"min_depth": 3})["min_depth"] == 3.0
    assert type(resolve_thresholds(None, {"min_depth": 3})["min_depth"]) is float
    with pytest.raises(ValueError, match="needs a preset"):
        resolve_thresholds(None, {"min_vmr": None})
    with pytest.raises(ValueError, match="preset must be one of"):
        resolve_thresholds("dna", {})
    with pytest.raises(ValueError, match="Unknown thresholds"):
        resolve_thresholds("atac", {"min_reads": 3})
    with pytest.raises(TypeError, match="min_vmr must be a number"):
        resolve_thresholds("atac", {"min_vmr": "0.1"})
    with pytest.raises(TypeError, match="min_cells must be a number"):
        resolve_thresholds("atac", {"min_cells": True})
    with pytest.raises(ValueError, match="must be finite"):
        resolve_thresholds("atac", {"min_vmr": float("nan")})
    with pytest.raises(ValueError, match="whole number of cells"):
        resolve_thresholds("atac", {"min_cells": 2.5})


def test_thresholds_compare_as_the_published_rules_do() -> None:
    statistics = {
        "n_cells_conf_detected": np.asarray([5, 4, 5, 5]),
        "strand_correlation": np.asarray([0.65, 0.9, np.nan, 0.9]),
        "vmr": np.asarray([0.02, 0.02, 0.02, 0.01]),
    }
    thresholds = resolve_thresholds("atac", {})
    # Cells and strand correlation reach theirs; VMR must exceed 0.01; an
    # undefined correlation fails.
    assert informative_variants(statistics, thresholds, n_variants=4).tolist() == [
        True,
        False,
        False,
        False,
    ]
    with pytest.raises(ValueError, match="no base qualities"):
        informative_variants(statistics, {"min_quality": 30.0}, n_variants=4)
    with pytest.raises(ValueError, match="no forward and reverse counts"):
        informative_variants({"vmr": np.ones(1)}, thresholds, n_variants=1)


@pytest.mark.parametrize(
    ("name", "column", "value", "preset", "published_exceeds"),
    [
        ("min_cells", "n_cells_conf_detected", 5, "atac", False),
        ("min_strand_correlation", "strand_correlation", 0.65, "atac", False),
        ("min_vmr", "vmr", 0.01, "atac", True),
        ("min_vmr", "vmr", 0.01, "rna", True),
        ("min_depth", "mean_coverage", 20.0, "rna", True),
        ("min_quality", "mean_quality", 30.0, "rna", True),
    ],
)
def test_a_statistic_at_a_threshold_passes_and_a_preset_keeps_its_published_rule(
    name, column, value, preset, published_exceeds
) -> None:
    # Three variants: at the value, one step below it and one step above.
    at = float(value)
    tested = np.asarray([at, np.nextafter(at, -np.inf), np.nextafter(at, np.inf)])
    # Every other statistic of the preset passes by far.
    statistics = {
        "n_cells_conf_detected": np.full(3, 100.0),
        "strand_correlation": np.full(3, 1.0),
        "vmr": np.full(3, 1.0),
        "mean_coverage": np.full(3, 1000.0),
        "mean_quality": np.full(3, 40.0),
    }
    statistics[column] = tested
    given = informative_variants(statistics, {name: value}, n_variants=3)
    assert given.tolist() == [True, False, True]
    # The preset keeps exactly what the published rule keeps: the value itself
    # only when the rule says "at least".
    kept = informative_variants(
        statistics, resolve_thresholds(preset, {}), n_variants=3
    )
    assert kept.tolist() == [not published_exceeds, False, True]


def test_variants_are_named_after_their_site_as_mgatk_names_them() -> None:
    sites = _sites()
    assert variant_labels(sites, np.asarray([1, 3, 5])).tolist() == [
        "chrM:5A>C",
        "chrM:5A>T",
        "IDH2_Mut1 WT>MUT",
    ]
    unnamed = SiteTable(
        contigs=(Contig(""),),
        site_contig=np.asarray([0]),
        site_position=np.asarray([310]),
        site_name=np.asarray([""], dtype=object),
        site_channel=np.asarray([""], dtype=object),
        allele_site=np.asarray([0]),
        allele=np.asarray(["C"], dtype=object),
        allele_role=np.asarray(["alternative"], dtype=object),
    )
    # mgatk's own name on an unnamed contig; no reference names the change
    # by its alternative alone.
    assert variant_labels(unnamed, np.asarray([0])).tolist() == ["310>C"]


def test_selection_table_marks_columns_the_counts_cannot_give() -> None:
    sites = _sites()
    statistics = _statistics([[0, 1, 2, 3]], stranded=False, qualities=False)
    arrays = {
        **statistics.arrays(),
        "masked": np.asarray([False, True]),
        "informative": np.asarray([True, False]),
    }
    frame = variant_selection_frame(sites, statistics.alleles, arrays)
    assert frame.index.tolist() == ["chrM:5A>C", "chrM:5A>G"]
    assert frame["site"].tolist() == ["chrM:5", "chrM:5"]
    assert frame["reference"].tolist() == ["A", "A"]
    assert frame["n_cells_conf_detected"].isna().all()
    assert str(frame["n_cells_conf_detected"].dtype) == "Int64"
    assert frame["strand_correlation"].isna().all()
    assert frame["mean_quality"].isna().all()
    assert frame["n_cells_over_5"].tolist() == [2, 1]
    assert frame["informative"].tolist() == [True, False]


def _rcrs() -> dict[int, str]:
    """The rCRS bases of MiTo's mtscATAC reference, a real mgatk reference."""
    with gzip.open(DATA / "mito_mtscatac" / "pL1000.refAllele.txt.gz", "rt") as handle:
        rows = [line.split() for line in handle if line.strip()]
    return {int(position): base.upper() for position, base in rows}


def test_mask_lists_name_rcrs_bases_at_their_positions() -> None:
    reference = _rcrs()
    assert len(reference) == RCRS_LENGTH
    for mask in VARIANT_MASKS.values():
        for position, base, alternative in mask.variants:
            assert reference[position] == base, (mask.name, position)
            assert alternative != base
    assert len(VARIANT_MASKS["numt"].variants) == 25
    assert sorted({p for p, _r, _a in VARIANT_MASKS["rna_editing"].variants}) == [
        295,
        2617,
        13710,
    ]
    assert [p for p, _r, _a in VARIANT_MASKS["artifact_prone"].variants] == [
        301,
        302,
        310,
        316,
        3107,
        16182,
    ]


def test_masks_match_position_and_alternative() -> None:
    masked = masked_variants(
        ["numt", "artifact_prone"],
        positions=np.asarray([16293, 16293, 310, 311, 2617]),
        alternatives=np.asarray(["C", "G", "C", "C", "G"], dtype=object),
        contig_lengths=np.full(5, RCRS_LENGTH),
    )
    # 16293A>C is a NUMT false positive and 16293A>G is not; every variant
    # at 310 is artifact-prone; 2617 is masked only by rna_editing.
    assert masked.tolist() == [True, False, True, False, False]
    assert masked_variants(
        ["rna_editing"],
        positions=np.asarray([2617, 2617, 295]),
        alternatives=np.asarray(["G", "C", "T"], dtype=object),
        contig_lengths=np.full(3, RCRS_LENGTH),
    ).tolist() == [True, False, True]
    assert masked_variants(
        [], positions=np.asarray([310]), alternatives=np.asarray(["C"]), contig_lengths=np.asarray([54])
    ).tolist() == [False]  # fmt: skip


@pytest.mark.parametrize(
    ("lengths", "positions", "message"),
    [
        ([16571], [310], "a contig of 16571 bases"),
        ([-1], [310], "a contig of unknown length"),
        ([-1], [-1], "a named site"),
    ],
)
def test_masks_refuse_coordinates_that_are_not_rcrs(
    lengths, positions, message
) -> None:
    with pytest.raises(ValueError, match=message):
        masked_variants(
            ["numt"],
            positions=np.asarray(positions),
            alternatives=np.asarray(["C"], dtype=object),
            contig_lengths=np.asarray(lengths),
        )
    with pytest.raises(ValueError, match="Unknown masks"):
        masked_variants(
            ["numts"],
            positions=np.asarray([310]),
            alternatives=np.asarray(["C"]),
            contig_lengths=np.asarray([RCRS_LENGTH]),
        )


def test_frame_column_order_follows_mgatk() -> None:
    sites = _sites()
    statistics = _statistics([[0, 1, 2, 3]], stranded=True, qualities=True)
    frame = variant_selection_frame(
        sites,
        statistics.alleles,
        {
            **statistics.arrays(),
            "masked": np.zeros(2, dtype=bool),
            "informative": np.zeros(2, dtype=bool),
        },
    )
    assert list(frame.columns) == [
        "site",
        "position",
        "reference",
        "alternative",
        "mean",
        "variance",
        "vmr",
        "n_cells_conf_detected",
        "strand_correlation",
        "n_cells_over_5",
        "n_cells_over_10",
        "n_cells_over_20",
        "n_cells_over_95",
        "max_heteroplasmy",
        "mean_coverage",
        "n_cells_covered",
        "mean_quality",
        "masked",
        "informative",
    ]
    assert isinstance(frame.index, pd.Index) and frame.index.name == "variant"
