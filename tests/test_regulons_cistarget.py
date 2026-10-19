import json
from pathlib import Path

import numpy as np
import pytest

from cytearc.regulons import cistarget
from cytearc.regulons.cistarget import (
    auc_rank_cutoff,
    enrich_signature,
    recovery_aucs,
    recovery_curves,
)

_RANKS = np.array([[0, 3], [2, 1]], dtype=np.int32)

# Outputs of ctxcore 0.2.0, run in a separate environment on the inputs stored
# beside them. The provenance names the call behind each output.
_CTXCORE = json.loads((Path(__file__).parent / "ctxcore_0_2_0_golden.json").read_text())


@pytest.mark.parametrize(
    "case", _CTXCORE["recovery"], ids=lambda case: f"seed{case['seed']}"
)
def test_aucs_and_curves_match_ctxcore(case):
    rankings = np.asarray(case["rankings"], dtype=np.int32)
    weights = np.asarray(case["weights"])
    total = case["totalGenes"]

    aucs = recovery_aucs(rankings, total, weights, case["aucThreshold"])
    curves = recovery_curves(rankings, weights, case["rankThreshold"])

    # The weights are multiples of 1/16, so every sum is exact and the
    # results equal ctxcore's bit for bit.
    assert auc_rank_cutoff(case["aucThreshold"], total) == case["rankCutoff"]
    np.testing.assert_array_equal(aucs, case["aucs"])
    np.testing.assert_array_equal(aucs, case["recoveryAucs"])
    np.testing.assert_array_equal(curves, case["curves"])


def test_python_kernels_match_compiled_kernels():
    # Ranks at or past the AUC cutoff (4) or the curve length (6) are skipped.
    ranks = np.array([[0, 2, 5, 9], [4, 1, 7, 3], [8, 6, 0, 2]], dtype=np.int32)
    weights = np.array([1.0, 0.5, 2.0, 0.25])
    curves = np.array(
        [
            [1.0, 1.0, 1.5, 1.5, 1.5, 3.5],
            [0.0, 0.5, 0.5, 0.75, 1.75, 1.75],
            [2.0, 2.0, 2.25, 2.25, 2.25, 2.25],
        ]
    )
    envelope = curves.mean(axis=0) + 2.0 * curves.std(axis=0)
    # The last motif's curve minus the envelope dips at ranks 2 and 3 and
    # peaks at rank 4.
    cases = [
        (cistarget._aucs, (ranks, weights, 4), [5.0, 1.75, 8.5]),
        (cistarget._curves, (ranks, weights, 6), curves),
        (cistarget._envelope, (ranks, weights, 6), envelope),
        (cistarget._rank_at_max, (ranks[2], weights, envelope), 4),
    ]

    for kernel, arguments, expected in cases:
        for implementation in (kernel, kernel.py_func):
            np.testing.assert_allclose(implementation(*arguments), expected)


def test_auc_rank_cutoff_rounds_halves_to_even():
    # 0.05 * 250 = 12.5 rounds to 12, then ctxcore subtracts one.
    assert auc_rank_cutoff(0.05, 250) == 11
    # 0.05 * 270 = 13.5 rounds to 14.
    assert auc_rank_cutoff(0.05, 270) == 13
    with pytest.raises(ValueError, match="leaves no ranks"):
        auc_rank_cutoff(0.001, 100)


@pytest.mark.parametrize(
    ("rankings", "weights", "message"),
    [
        (_RANKS[0], np.ones(2), "two-dimensional integer"),
        (_RANKS.astype(np.float64), np.ones(2), "two-dimensional integer"),
        (_RANKS, np.ones(3), "align with the ranking columns"),
        (-_RANKS, np.ones(2), "non-negative"),
    ],
)
def test_recovery_functions_reject_malformed_rankings(rankings, weights, message):
    with pytest.raises(ValueError, match=message):
        recovery_aucs(rankings, 10, weights, 0.5)
    with pytest.raises(ValueError, match=message):
        recovery_curves(rankings, weights, 5)


@pytest.mark.parametrize(
    ("weights", "auc_threshold", "message"),
    [
        (np.zeros(2), 0.5, "sum to a positive value"),
        (np.ones(2), 0.0, r"auc_threshold must be in \(0, 1\]"),
        (np.ones(2), 1.5, r"auc_threshold must be in \(0, 1\]"),
    ],
)
def test_recovery_aucs_rejects_invalid_weights_and_thresholds(
    weights, auc_threshold, message
):
    with pytest.raises(ValueError, match=message):
        recovery_aucs(_RANKS, 10, weights, auc_threshold)


def test_enrich_signature_matches_ctxcore_nes_and_leading_edges():
    case = _CTXCORE["enrichment"]
    assert _CTXCORE["provenance"]["packageVersion"] == "0.2.0"

    result = enrich_signature(
        np.asarray(case["rankings"], dtype=np.int32),
        np.asarray(case["motifs"]),
        np.asarray(case["genes"]),
        case["totalGenes"],
        weights=np.asarray(case["weights"]),
        rank_threshold=case["rankThreshold"],
        auc_threshold=case["aucThreshold"],
        nes_threshold=case["nesThreshold"],
    )

    # ctxcore's leading edges are listed by decreasing NES, the table's order;
    # they belong to the three motifs planted to recover the signature.
    expected = case["leadingEdges"]
    assert len(expected) == 3
    assert result.table["motif"].tolist() == [edge["motif"] for edge in expected]
    table = result.table.set_index("motif")
    aucs = dict(zip(case["motifs"], case["auc"], strict=True))
    nes = dict(zip(case["motifs"], case["nes"], strict=True))
    for edge in expected:
        motif = edge["motif"]
        assert table.loc[motif, "AUC"] == aucs[motif]
        np.testing.assert_allclose(table.loc[motif, "NES"], nes[motif], rtol=1e-12)
        assert table.loc[motif, "rank_at_max"] == edge["rankAtMax"]
        assert table.loc[motif, "n_targets"] == len(edge["genes"])
        leading = result.leading_edges[motif]
        assert leading["feature"].tolist() == edge["genes"]
        assert leading["weight"].tolist() == edge["weights"]


def test_enrich_signature_without_enriched_motifs_returns_empty_table():
    rng = np.random.default_rng(9)
    rankings = np.vstack([rng.permutation(500)[:10] for _ in range(30)])

    result = enrich_signature(
        rankings,
        np.array([f"m{index}" for index in range(30)]),
        np.array([f"g{index}" for index in range(10)]),
        500,
        rank_threshold=100,
        auc_threshold=0.05,
        nes_threshold=50.0,
    )

    assert result.table.empty
    assert result.leading_edges == {}
    assert result.n_signature == 10


@pytest.mark.parametrize(
    ("rankings", "motifs", "message"),
    [
        (_RANKS, ["m0", "m1", "m2"], "motifs by signature features"),
        (_RANKS[:1], ["m0"], "at least two motifs"),
    ],
)
def test_enrich_signature_rejects_misaligned_or_single_motif_rankings(
    rankings, motifs, message
):
    with pytest.raises(ValueError, match=message):
        enrich_signature(
            rankings,
            np.array(motifs),
            np.array(["g0", "g1"]),
            10,
            rank_threshold=5,
            auc_threshold=0.5,
            nes_threshold=3.0,
        )


def test_enrich_signature_gives_zero_nes_when_motifs_recover_equally():
    # Identical rankings leave no AUC spread to standardize by.
    result = enrich_signature(
        np.vstack([_RANKS[0], _RANKS[0]]),
        np.array(["m0", "m1"]),
        np.array(["g0", "g1"]),
        10,
        rank_threshold=5,
        auc_threshold=0.5,
        nes_threshold=0.0,
    )

    assert result.table["NES"].tolist() == [0.0, 0.0]
