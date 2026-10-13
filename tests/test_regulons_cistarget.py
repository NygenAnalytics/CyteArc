import numpy as np
import pytest

from cytearc.regulons.cistarget import (
    auc_rank_cutoff,
    enrich_signature,
    recovery_aucs,
    recovery_curves,
)


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


def test_auc_rank_cutoff_rounds_halves_to_even():
    # 0.05 * 250 = 12.5 rounds to 12, then ctxcore subtracts one.
    assert auc_rank_cutoff(0.05, 250) == 11
    # 0.05 * 270 = 13.5 rounds to 14.
    assert auc_rank_cutoff(0.05, 270) == 13
    with pytest.raises(ValueError, match="leaves no ranks"):
        auc_rank_cutoff(0.001, 100)


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
