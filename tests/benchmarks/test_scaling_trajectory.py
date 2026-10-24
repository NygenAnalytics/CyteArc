"""Benchmarks of the kernels behind lineage-resolved trajectory analyses.

Each benchmark calls the function its operation calls, on cells shaped like
the recorded 1M-cell funnel with four lineages, and checks the result against
a dense reference, so it doubles as a test of that kernel at the smoke size.
Lineage markers and drivers stream normalized features in blocks; curves and
the cluster tree stream coordinates; every benchmark here covers one block or
one pass, which the operation repeats over the cells.
"""

from functools import lru_cache

import numpy as np
import pytest

from . import inputs
from .harness import Ladder

pytestmark = pytest.mark.benchmark

CELLS = Ladder(sizes=(4_000, 8_000, 16_000, 32_000), smoke=600)
N_LINEAGES = 4
N_FEATURES = 200
N_POINTS = 100


@lru_cache(maxsize=4)
def _lineage_inputs(n_cells: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pseudotime, fate-like weights, and log-normalized features.

    Weights are Dirichlet draws that tilt toward one lineage as pseudotime
    grows, so lineages share early cells and part late, as fate maps do.
    """
    rng = np.random.default_rng(inputs.SEED)
    ordering = rng.random(n_cells)
    favoured = rng.integers(0, N_LINEAGES, size=n_cells)
    concentration = np.full((n_cells, N_LINEAGES), 1.0)
    concentration[np.arange(n_cells), favoured] += 20.0 * ordering
    weights = np.vstack([rng.dirichlet(row) for row in concentration])
    counts, _labels = inputs.labelled_counts(n_cells)
    totals = np.maximum(counts.sum(axis=1, dtype=np.float64), 1.0)
    values = np.log1p(
        inputs.SIZE_FACTOR * counts[:, :N_FEATURES] / totals[:, np.newaxis]
    )
    return ordering, weights, np.asfortranarray(values)


def test_lineage_weighted_correlations(bench) -> None:
    from cytearc.trajectory.lineages import weighted_correlations

    def make(n_cells: int):
        ordering, weights, values = _lineage_inputs(n_cells)
        regressors = np.repeat(ordering[:, np.newaxis], N_LINEAGES, axis=1)
        return lambda: weighted_correlations(
            values,
            weights,
            regressors,
            min_mass=1.0,
            nthreads=1,
            feature_labels=list(range(N_FEATURES)),
        )

    def check(n_cells: int, result) -> None:
        ordering, weights, values = _lineage_inputs(n_cells)
        r_values, status = result
        for lineage in range(N_LINEAGES):
            w = weights[:, lineage]
            centered = values - (w[:, None] * values).sum(axis=0) / w.sum()
            order = ordering - (w * ordering).sum() / w.sum()
            # Genes without counts in the smoke sample divide 0 by 0; they
            # are untested and skipped below.
            with np.errstate(invalid="ignore", divide="ignore"):
                expected = (w[:, None] * centered * order[:, None]).sum(
                    axis=0
                ) / np.sqrt(
                    (w[:, None] * centered**2).sum(axis=0) * (w * order**2).sum()
                )
            tested = status[:, lineage] == 0
            np.testing.assert_allclose(
                r_values[tested, lineage], expected[tested], atol=1e-10
            )

    bench("lineage.weighted_correlations", make, CELLS, check=check)


def test_lineage_equal_mass_profiles(bench) -> None:
    from cytearc.trajectory.lineages import (
        bin_smoothing_matrix,
        equal_mass_bin_operator,
        lineage_profiles,
    )

    smoothing = bin_smoothing_matrix(50, 1.0)

    def make(n_cells: int):
        ordering, weights, values = _lineage_inputs(n_cells)

        def run():
            operator, _, _ = equal_mass_bin_operator(ordering, weights, 50)
            return lineage_profiles(values, operator, smoothing, N_LINEAGES)

        return run

    def check(n_cells: int, profiles) -> None:
        ordering, weights, values = _lineage_inputs(n_cells)
        assert profiles.shape == (N_FEATURES, N_LINEAGES, 50)
        # Smoothing and equal-mass bins keep each lineage's weighted mean.
        for lineage in range(N_LINEAGES):
            w = weights[:, lineage]
            np.testing.assert_allclose(
                profiles[:, lineage].mean(axis=1),
                (w[:, None] * values).sum(axis=0) / w.sum(),
                rtol=0.05,
                atol=1e-3,
            )

    bench("lineage.equal_mass_profiles", make, CELLS, check=check)


def _curve_inputs(n_cells: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordering, weights, _values = _lineage_inputs(n_cells)
    coordinates, _labels = inputs.labelled_coordinates(n_cells)
    return ordering, weights, np.asarray(coordinates, dtype=np.float64)


def test_lineage_curve_fit(bench) -> None:
    from cytearc.trajectory.curves import fit_lineage_curves, lineage_ranks

    def make(n_cells: int):
        ordering, weights, coordinates = _curve_inputs(n_cells)
        ranks = lineage_ranks(ordering, weights)
        blocks = [
            (start, coordinates[start : start + 4_096])
            for start in range(0, n_cells, 4_096)
        ]
        return lambda: fit_lineage_curves(
            blocks,
            ranks,
            weights,
            ordering,
            n_dims=inputs.N_DIMS,
            n_points=N_POINTS,
            degrees_of_freedom=10.0,
        )

    def check(n_cells: int, fit) -> None:
        assert fit.points.shape == (N_LINEAGES, N_POINTS, inputs.N_DIMS)
        assert np.isfinite(fit.points).all()
        assert (np.diff(fit.ordering, axis=1) >= 0.0).all()

    bench("lineage.curve_fit", make, CELLS, check=check)


def test_lineage_curve_projection(bench) -> None:
    from cytearc.trajectory.curves import project_onto_curve

    def make(n_cells: int):
        _ordering, _weights, coordinates = _curve_inputs(n_cells)
        rng = np.random.default_rng(inputs.SEED)
        curves = np.cumsum(
            rng.normal(size=(N_LINEAGES, N_POINTS, inputs.N_DIMS)), axis=1
        )
        progress = np.linspace(0.0, 1.0, N_POINTS)
        return lambda: [
            project_onto_curve(coordinates, curve, progress) for curve in curves
        ]

    def check(n_cells: int, projections) -> None:
        _ordering, _weights, coordinates = _curve_inputs(n_cells)
        rng = np.random.default_rng(inputs.SEED)
        curves = np.cumsum(
            rng.normal(size=(N_LINEAGES, N_POINTS, inputs.N_DIMS)), axis=1
        )
        dense = np.linspace(0.0, 1.0, 20)
        for curve, (position, distance) in zip(curves, projections, strict=True):
            assert ((position >= 0.0) & (position <= 1.0)).all()
            # No point sampled densely along the polyline is nearer than the
            # reported projection.
            samples = (
                curve[:-1, None, :] * (1.0 - dense[None, :, None])
                + curve[1:, None, :] * dense[None, :, None]
            ).reshape(-1, inputs.N_DIMS)
            nearest = np.sqrt(
                ((coordinates[:50, None, :] - samples[None]) ** 2).sum(axis=2)
            ).min(axis=1)
            assert (distance[:50] <= nearest + 1e-9).all()

    bench("lineage.curve_projection", make, CELLS, check=check)


def test_lineage_cluster_tree(bench) -> None:
    from cytearc.trajectory.lineage_tree import (
        cluster_distances,
        cluster_moments,
        spanning_tree,
    )

    def make(n_cells: int):
        coordinates, labels = inputs.labelled_coordinates(n_cells)
        values = np.asarray(coordinates, dtype=np.float64)
        blocks = [
            (start, values[start : start + 4_096]) for start in range(0, n_cells, 4_096)
        ]

        def run():
            counts, means, covariances = cluster_moments(
                blocks, labels, n_clusters=inputs.N_GROUPS, n_dims=inputs.N_DIMS
            )
            return (
                counts,
                means,
                spanning_tree(cluster_distances(means, covariances), root=0),
            )

        return run

    def check(n_cells: int, result) -> None:
        coordinates, labels = inputs.labelled_coordinates(n_cells)
        counts, means, tree = result
        np.testing.assert_array_equal(counts, np.bincount(labels))
        for cluster in range(inputs.N_GROUPS):
            np.testing.assert_allclose(
                means[cluster],
                coordinates[labels == cluster].astype(np.float64).mean(axis=0),
                rtol=1e-10,
                atol=1e-10,
            )
        assert tree.edges.shape == (inputs.N_GROUPS - 1, 2)

    bench("lineage.cluster_tree", make, CELLS, check=check)
