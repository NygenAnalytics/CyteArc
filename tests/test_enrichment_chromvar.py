import numpy as np
import pytest
from scipy.sparse import csc_matrix

from cytearc.features.enrichment import chromvar
from cytearc.features.enrichment.chromvar import (
    background_peaks,
    build_chromvar_model,
    score_chromvar_block,
)


def _annotation(sets: list[list[int]], n_peaks: int) -> csc_matrix:
    rows = [peak for members in sets for peak in members]
    columns = [index for index, members in enumerate(sets) for _ in members]
    return csc_matrix((np.ones(len(rows)), (rows, columns)), shape=(n_peaks, len(sets)))


def test_deviations_match_the_chromvar_test_fixture():
    # chromVAR tests/testthat/test_compute_deviations.R, with the expected
    # values stored in inst/extdata/deviations_test (rows are peak sets).
    peaks_by_cells = np.array([[0, 1, 2, 0], [1, 1, 0, 1], [2, 0, 1, 1]])
    annotation = _annotation([[0, 1, 2], [0, 1], [1, 2]], 3)
    background = np.array([[2, 2, 2], [1, 1, 3], [2, 2, 1]]) - 1
    model = build_chromvar_model(annotation, peaks_by_cells.sum(axis=1), background)

    deviations, z = score_chromvar_block(peaks_by_cells.T, model)

    np.testing.assert_allclose(model.expectation, [0.3, 0.3, 0.4])
    np.testing.assert_allclose(
        deviations.T,
        [
            [
                0.17283950617283947,
                -0.4444444444444446,
                0.17283950617283947,
                -0.07407407407407414,
            ],
            [
                -0.29100529100529104,
                0.31746031746031744,
                0.21164021164021163,
                -0.19841269841269843,
            ],
            [
                0.7407407407407409,
                -0.6349206349206349,
                -0.7407407407407408,
                0.634920634920635,
            ],
        ],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        z.T,
        [
            [
                1.1547005383792515,
                -1.1547005383792517,
                1.1547005383792515,
                -1.1547005383792515,
            ],
            [
                -0.5773502691896257,
                0.5773502691896256,
                0.5773502691896257,
                -0.5773502691896257,
            ],
            [
                3.2331615074619044,
                -1.1547005383792512,
                -4.041451884327379,
                9.237604307034015,
            ],
        ],
        rtol=1e-12,
    )


def _brute_force(counts, sets, background, fragments):
    expectation = fragments / fragments.sum()
    n_cells = counts.shape[0]
    deviations = np.full((n_cells, len(sets)), np.nan)
    z = np.full((n_cells, len(sets)), np.nan)
    for cell in range(n_cells):
        depth = counts[cell].sum()
        for index, members in enumerate(sets):
            expected = expectation[members].sum() * depth
            if expected < 1:
                continue
            observed = counts[cell, members].sum()
            dev = (observed - expected) / expected
            sampled = []
            for iteration in range(background.shape[1]):
                peaks = background[members, iteration]
                exp_t = expectation[peaks].sum() * depth
                sampled.append((counts[cell, peaks].sum() - exp_t) / exp_t)
            sampled = np.asarray(sampled)
            deviations[cell, index] = dev - sampled.mean()
            z[cell, index] = (dev - sampled.mean()) / sampled.std(ddof=1)
    return deviations, z


def test_inverted_background_index_matches_brute_force():
    rng = np.random.default_rng(0)
    n_peaks, n_cells = 60, 25
    counts = rng.poisson(0.4, size=(n_cells, n_peaks))
    counts[:, :] += (rng.random((n_cells, n_peaks)) < 0.02).astype(int) * 3
    counts[0] = 0  # a cell without fragments has no values
    counts[1] += 1  # keep every peak's fragment total positive
    fragments = counts.sum(axis=0)
    sets = [
        sorted(rng.choice(n_peaks, size=size, replace=False).tolist())
        for size in (5, 12, 30)
    ]
    background = rng.integers(0, n_peaks, size=(n_peaks, 7))
    model = build_chromvar_model(_annotation(sets, n_peaks), fragments, background)

    deviations, z = score_chromvar_block(counts, model)

    expected_dev, expected_z = _brute_force(counts, sets, background, fragments)
    np.testing.assert_allclose(deviations, expected_dev, rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(z, expected_z, rtol=1e-12, equal_nan=True)
    assert np.isnan(z[0]).all()
    # Block boundaries do not change any cell.
    split = np.vstack(
        [
            score_chromvar_block(counts[:9], model)[1],
            score_chromvar_block(counts[9:], model)[1],
        ]
    )
    np.testing.assert_array_equal(split, z)


def test_background_peaks_match_gc_and_accessibility():
    rng = np.random.default_rng(1)
    n_peaks = 2000
    fragments = rng.lognormal(3.0, 1.0, size=n_peaks).round() + 1
    gc = np.clip(rng.normal(0.5, 0.08, size=n_peaks), 0.2, 0.8)

    first = background_peaks(fragments, gc, n_iterations=50, seed=7)
    again = background_peaks(fragments, gc, n_iterations=50, seed=7)
    other = background_peaks(fragments, gc, n_iterations=50, seed=8)

    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)
    assert first.shape == (n_peaks, 50)
    assert first.min() >= 0 and first.max() < n_peaks
    # Background peaks resemble their peak in GC and log accessibility far
    # more than random peaks do.
    gc_gap = np.abs(gc[first] - gc[:, None]).mean()
    random_gap = np.abs(
        gc[rng.integers(0, n_peaks, size=first.shape)] - gc[:, None]
    ).mean()
    assert gc_gap < 0.5 * random_gap
    count_gap = np.abs(np.log10(fragments[first]) - np.log10(fragments)[:, None]).mean()
    random_count_gap = np.abs(
        np.log10(fragments[rng.integers(0, n_peaks, size=first.shape)])
        - np.log10(fragments)[:, None]
    ).mean()
    assert count_gap < 0.5 * random_count_gap


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gc": np.array([0.4, 0.45])}, "aligned vectors"),
        (
            {"fragments_per_peak": np.array([1.0]), "gc": np.array([0.4])},
            "at least two peaks",
        ),
        ({"fragments_per_peak": np.array([0.0, 2.0, 3.0])}, "at least one fragment"),
        ({"gc": np.array([0.4, np.nan, 0.5])}, "GC content must be finite"),
        # Constant GC content leaves a singular covariance to whiten.
        ({"gc": np.full(3, 0.5)}, "collinear or constant"),
        ({"n_iterations": 1}, "n_iterations"),
        ({"kernel_width": 0.0}, "kernel_width"),
    ],
)
def test_background_peaks_reject_invalid_inputs(kwargs, message):
    arguments = {
        "fragments_per_peak": np.array([1.0, 2.0, 3.0]),
        "gc": np.array([0.4, 0.45, 0.5]),
        "seed": 0,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        background_peaks(**arguments)


def test_z_is_missing_when_background_deviations_do_not_vary():
    counts = np.array([[3, 1, 2], [1, 4, 2], [2, 2, 5]], dtype=np.float64)
    # Every peak draws only itself, so a set's background deviations are equal.
    background = np.repeat(np.arange(3)[:, None], 50, axis=1)
    annotation = csc_matrix(np.array([[1.0], [0.0], [0.0]]))

    deviations, z = score_chromvar_block(
        counts, build_chromvar_model(annotation, counts.sum(axis=0), background)
    )

    assert np.isfinite(deviations).all()
    assert np.isnan(z).all()


def _small_model_arguments() -> dict[str, object]:
    return {
        "annotation": _annotation([[0, 1], [1, 2]], 3),
        "fragments_per_peak": np.array([1.0, 2.0, 3.0]),
        "background_index": np.array([[1, 2], [0, 2], [0, 1]]),
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"background_index": np.zeros((2, 2), dtype=np.int64)}, "must align"),
        ({"fragments_per_peak": np.array([1.0, 0.0, 3.0])}, "at least one fragment"),
        ({"background_index": np.full((3, 2), 3)}, "out of range"),
        (
            {"annotation": csc_matrix(np.array([[0.5], [0.0], [1.0]]))},
            "positive integer multiplicities",
        ),
    ],
)
def test_build_chromvar_model_rejects_invalid_inputs(kwargs, message):
    arguments = _small_model_arguments()
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        build_chromvar_model(**arguments)


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        (np.ones((2, 4)), "cells by the model's peaks"),
        (np.array([["1", "2", "3"]]), "finite numbers"),
        (np.array([[1.0, np.inf, 2.0]]), "finite numbers"),
        (np.array([[1, -1, 2]]), "non-negative"),
    ],
    ids=["shape", "dtype", "infinite", "negative"],
)
def test_score_chromvar_block_rejects_invalid_counts(counts, message):
    model = build_chromvar_model(**_small_model_arguments())
    with pytest.raises(ValueError, match=message):
        score_chromvar_block(counts, model)


def test_python_kernels_match_the_compiled_kernels():
    # A repeated member, a zero count, a cell without fragments, and a set
    # whose only member draws itself, so its background deviations are equal.
    counts = np.array([[0, 0, 0, 0], [5, 0, 4, 3], [2, 6, 3, 0]], dtype=np.float64)
    annotation = csc_matrix(np.array([[2.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
    background = np.array([[1, 2, 3], [3, 0, 2], [2, 2, 2], [0, 1, 2]])
    model = build_chromvar_model(annotation, counts.sum(axis=0), background)
    sets = model.annotation
    scoring = (
        counts,
        sets.indptr,
        sets.indices,
        sets.data,
        model.invertedIndptr,
        model.invertedIndices,
        model.expectedSet,
        model.expectedBackground,
    )

    pointers, indices = chromvar._invert_background.py_func(
        sets.indptr.astype(np.int64),
        sets.indices.astype(np.int64),
        sets.data.astype(np.int64),
        model.backgroundIndex,
        model.n_sets,
    )
    deviations, z = chromvar._score_block.py_func(*scoring)

    assert (model.n_sets, model.n_iterations) == (2, 3)
    # The model holds the compiled inversion of the same inputs.
    np.testing.assert_array_equal(pointers, model.invertedIndptr)
    np.testing.assert_array_equal(indices, model.invertedIndices)
    compiled_deviations, compiled_z = chromvar._score_block(*scoring)
    np.testing.assert_allclose(deviations, compiled_deviations, rtol=1e-12)
    np.testing.assert_allclose(z, compiled_z, rtol=1e-12)
    assert np.isnan(deviations[0]).all()
    assert np.isfinite(z[1:, 0]).all()
    assert np.isfinite(deviations[1:, 1]).all() and np.isnan(z[1:, 1]).all()
