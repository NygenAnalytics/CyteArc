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


def _ctxcore_recovery(rankings, total, weights, rank_threshold, auc_threshold):
    """A line-by-line transcription of ctxcore 0.2.0 ``recovery.recovery``."""
    rank_cutoff = int(round(auc_threshold * total)) - 1
    weights = np.insert(weights, len(weights), 0.0)
    rankings = np.append(
        rankings, np.full((rankings.shape[0], 1), fill_value=total), axis=1
    )
    rccs = np.empty((rankings.shape[0], rank_threshold))
    for row in range(rankings.shape[0]):
        rccs[row, :] = np.cumsum(
            np.bincount(rankings[row], weights=weights)[:rank_threshold]
        )
    maxauc = float((rank_cutoff + 1) * weights.sum())
    return rccs, rccs[:, :rank_cutoff].sum(axis=1) / maxauc


def _ctxcore_weighted_auc(ranking, weights, rank_cutoff, max_auc):
    """ctxcore ``weighted_auc1d``."""
    keep = ranking < rank_cutoff
    x = ranking[keep]
    y = weights[keep]
    order = np.argsort(x)
    x = np.concatenate((x[order], [rank_cutoff]))
    return np.sum(np.diff(x) * y[order].cumsum()) / max_auc


def _random_database(seed, n_motifs=60, total=400, n_signature=25):
    rng = np.random.default_rng(seed)
    rankings = np.vstack(
        [rng.permutation(total)[:n_signature] for _ in range(n_motifs)]
    )
    # Make a few motifs strongly recover the signature.
    for motif in range(3):
        rankings[motif] = rng.permutation(60)[:n_signature]
    weights = rng.uniform(0.1, 2.0, size=n_signature)
    return rankings.astype(np.int32), weights, total


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_aucs_and_curves_match_ctxcore(seed):
    rankings, weights, total = _random_database(seed)

    expected_curves, expected_aucs = _ctxcore_recovery(
        rankings, total, weights, 200, 0.05
    )

    np.testing.assert_allclose(
        recovery_aucs(rankings, total, weights, 0.05), expected_aucs
    )
    np.testing.assert_allclose(recovery_curves(rankings, weights, 200), expected_curves)
    cutoff = auc_rank_cutoff(0.05, total)
    max_auc = (cutoff + 1) * weights.sum()
    single = [_ctxcore_weighted_auc(row, weights, cutoff, max_auc) for row in rankings]
    np.testing.assert_allclose(recovery_aucs(rankings, total, weights, 0.05), single)


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
    rankings, weights, total = _random_database(4)
    motifs = np.array([f"m{index}" for index in range(rankings.shape[0])])
    genes = np.array([f"g{index}" for index in range(rankings.shape[1])])

    result = enrich_signature(
        rankings,
        motifs,
        genes,
        total,
        weights=weights,
        rank_threshold=200,
        auc_threshold=0.05,
        nes_threshold=3.0,
    )

    rccs, aucs = _ctxcore_recovery(rankings, total, weights, 200, 0.05)
    nes = (aucs - aucs.mean()) / aucs.std()
    enriched = set(motifs[nes >= 3.0])
    assert set(result.table["motif"]) == enriched
    assert enriched, "the planted motifs must be enriched"
    assert result.table["NES"].is_monotonic_decreasing
    envelope = rccs.mean(axis=0) + 2.0 * rccs.std(axis=0)
    for motif in enriched:
        row = int(motif[1:])
        rank_at_max = int(np.argmax(rccs[row] - envelope))
        order = np.argsort(rankings[row])
        expected = [
            (genes[index], weights[index])
            for index in order
            if rankings[row, index] <= rank_at_max
        ]
        edge = result.leading_edges[motif]
        assert list(zip(edge["feature"], edge["weight"])) == expected
        table_row = result.table.set_index("motif").loc[motif]
        assert table_row["rank_at_max"] == rank_at_max
        assert table_row["n_targets"] == len(expected)
        np.testing.assert_allclose(table_row["NES"], nes[row])


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
