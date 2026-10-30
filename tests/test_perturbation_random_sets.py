from collections import Counter

import numpy as np
import pytest
from scipy import stats

from cytearc.perturbation import random_sets
from cytearc.perturbation.random_sets import N_BUCKETS, resampling_keys
from tests.perturbation_helpers import use_python_kernels


def _python_kernels(monkeypatch: pytest.MonkeyPatch, python_kernels: bool) -> None:
    """Run the kernels uncompiled, whose 64-bit products wrap as compiled ones do."""
    if python_kernels:
        use_python_kernels(monkeypatch, random_sets)
        np.seterr(over="ignore")


@pytest.fixture(autouse=True)
def _restore_floating_point_errors():
    previous = np.geterr()
    yield
    np.seterr(**previous)


def _buckets(probabilities: np.ndarray):
    order = np.empty(len(probabilities), dtype=np.int64)
    offsets = np.empty(N_BUCKETS + 1, dtype=np.int64)
    bounds = np.empty(N_BUCKETS)
    random_sets.probability_buckets(probabilities, order, offsets, bounds)
    return order, offsets, bounds


def _sets(key: np.uint64, probabilities: np.ndarray, indices) -> list[np.ndarray]:
    """The Bernoulli sets of ``indices`` under a group's key, each as drawn."""
    order, offsets, bounds = _buckets(probabilities)
    out = np.empty(len(probabilities), dtype=np.int64)
    drawn = []
    for d in indices:
        # A key returned to Python is a Python integer, which may not fit
        # int64; as uint64 it passes back unchanged.
        set_key = np.uint64(random_sets.set_key(key, d))
        count = random_sets.draw_bernoulli_set(
            set_key, probabilities, order, offsets, bounds, out
        )
        drawn.append(out[:count].copy())
    return drawn


def _spread_probabilities(n: int, seed: int) -> np.ndarray:
    """Probabilities from 1e-4 to 1 on a log scale, with exact 0, 1/2 and 1."""
    rng = np.random.default_rng(seed)
    probabilities = 10.0 ** rng.uniform(-4.0, 0.0, n)
    probabilities[:4] = [0.0, 0.5, 1.0, 0.25]
    return probabilities


@pytest.mark.parametrize("python_kernels", [False, True])
def test_buckets_hold_each_probability_below_a_bound_twice_as_large(
    monkeypatch, python_kernels
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    probabilities = np.r_[
        _spread_probabilities(300, 1), 2.0**-62, 2.0**-63, 2.0**-80, 5e-324, 0.0
    ]

    order, offsets, bounds = _buckets(probabilities)

    assert sorted(order.tolist()) == list(range(len(probabilities)))
    assert offsets[0] == 0 and offsets[-1] == len(probabilities)
    np.testing.assert_array_equal(bounds, 2.0 ** -np.arange(N_BUCKETS))
    for k in range(N_BUCKETS):
        cells = order[offsets[k] : offsets[k + 1]]
        assert (np.diff(cells) > 0).all()
        held = probabilities[cells]
        assert (held <= bounds[k]).all()
        if k == 0:
            assert (held >= 0.5).all()
        elif k < N_BUCKETS - 1:
            assert (held >= bounds[k] / 2).all() and (held < bounds[k]).all()
        else:
            assert (held < bounds[k]).all()
    # 2^-63 opens the second last bucket; the last holds every smaller one.
    assert probabilities[order[offsets[-3] : offsets[-2]]].tolist() == [2.0**-63]
    assert set(probabilities[order[offsets[-2] :]]) == {2.0**-80, 5e-324, 0.0}


def test_bernoulli_sets_take_each_cell_with_its_probability() -> None:
    probabilities = _spread_probabilities(3000, 2)
    n_sets = 4000

    sets = _sets(resampling_keys(5, np.array([3]), 1)[0], probabilities, range(n_sets))

    hits = np.bincount(np.concatenate(sets), minlength=len(probabilities))
    assert all(len(np.unique(cells)) == len(cells) for cells in sets)
    assert hits[0] == 0 and hits[2] == n_sets
    expected = n_sets * probabilities
    likely = expected >= 20
    z = (hits[likely] - expected[likely]) / np.sqrt(
        expected[likely] * (1 - probabilities[likely]) + 1e-300
    )
    z = z[probabilities[likely] < 1]
    assert abs(z.mean()) < 0.1 and 0.9 < z.std() < 1.1
    # The cells rarely drawn, together, at their expected rate.
    rare = (expected < 20) & (probabilities > 0)
    assert abs(hits[rare].sum() - expected[rare].sum()) < 4 * np.sqrt(
        expected[rare].sum()
    )
    sizes = np.array([len(cells) for cells in sets])
    mean = probabilities.sum()
    variance = (probabilities * (1 - probabilities)).sum()
    assert abs(sizes.mean() - mean) < 4 * np.sqrt(variance / n_sets)
    assert 0.9 < sizes.var() / variance < 1.1


@pytest.mark.parametrize("python_kernels", [False, True])
def test_bernoulli_sets_depend_only_on_their_key(monkeypatch, python_kernels) -> None:
    probabilities = _spread_probabilities(200, 3)
    key = resampling_keys(1, np.array([0, 1]), 2)
    # The compiled sets, against which the Python twins are checked.
    compiled = _sets(key[0], probabilities, range(30))
    _python_kernels(monkeypatch, python_kernels)

    forward = _sets(key[0], probabilities, range(30))
    backward = _sets(key[0], probabilities, range(29, -1, -1))[::-1]
    other = _sets(key[1], probabilities, range(30))

    for a, b, c in zip(compiled, forward, backward, strict=True):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, c)
    assert sum(not np.array_equal(a, b) for a, b in zip(forward, other)) > 25
    # Cells come bucket after bucket, so a set lists the likeliest first.
    order, offsets, _ = _buckets(probabilities)
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    assert all((np.diff(rank[cells]) > 0).all() for cells in forward)


def _conditional_chances(probabilities: np.ndarray) -> dict[tuple[int, ...], float]:
    """Each set of some cells but not all, with its chance given that condition.

    Every subset of the cells is listed with the product of its cells'
    chances, independently of the sampler.
    """
    n = len(probabilities)
    chances = {}
    for mask in range(1, 2**n - 1):
        inside = np.array([(mask >> j) & 1 for j in range(n)], dtype=bool)
        cells = tuple(int(j) for j in np.flatnonzero(inside))
        chances[cells] = np.prod(np.where(inside, probabilities, 1 - probabilities))
    total = sum(chances.values())
    return {cells: chance / total for cells, chance in chances.items()}


@pytest.mark.parametrize("python_kernels", [False, True])
@pytest.mark.parametrize(
    "probabilities",
    [[0.8, 0.5, 0.2], [0.3, 0.2, 0.1, 0.05, 0.02], [0.98, 0.95, 0.9, 0.7]],
    # Draws hold no cell or every cell 8% of the time each, no cell 47% of
    # the time, or every cell 59% of the time.
    ids=["none-or-all", "often-none", "often-all"],
)
def test_bernoulli_sets_hold_some_cells_but_not_all_with_their_conditional_chances(
    monkeypatch, python_kernels, probabilities
) -> None:
    probabilities = np.array(probabilities)
    n_sets = 40_000 if not python_kernels else 4_000
    key = resampling_keys(6, np.array([2]), 1)[0]
    # The compiled sets, against which the Python twins are checked.
    compiled = _sets(key, probabilities, range(100))
    _python_kernels(monkeypatch, python_kernels)

    sets = _sets(key, probabilities, range(n_sets))

    for a, b in zip(compiled, sets, strict=False):
        np.testing.assert_array_equal(a, b)
    found = Counter(tuple(sorted(cells.tolist())) for cells in sets)
    chances = _conditional_chances(probabilities)
    assert set(found) <= set(chances)
    observed = np.array([found[cells] for cells in chances])
    expected = n_sets * np.array(list(chances.values()))
    # Sets expected fewer than 5 times are pooled, as the test needs.
    rare = expected < 5
    if rare.any():
        observed = np.r_[observed[~rare], observed[rare].sum()]
        expected = np.r_[expected[~rare], expected[rare].sum()]
    assert stats.chisquare(observed, expected).pvalue > 1e-3


@pytest.mark.parametrize("python_kernels", [False, True])
def test_bernoulli_sets_keep_a_draw_of_some_cells_and_draw_others_again_from_the_key(
    monkeypatch, python_kernels
) -> None:
    probabilities = np.array([0.8, 0.5, 0.2])
    order, offsets, bounds = _buckets(probabilities)
    key = resampling_keys(3, np.array([7]), 2)[0]
    _python_kernels(monkeypatch, python_kernels)

    forward = _sets(key, probabilities, range(200))
    backward = _sets(key, probabilities, range(199, -1, -1))[::-1]

    out = np.empty(3, dtype=np.int64)
    redrawn = 0
    for d, cells in enumerate(forward):
        assert 0 < len(cells) < 3
        np.testing.assert_array_equal(cells, backward[d])
        set_key = np.uint64(random_sets.set_key(key, d))
        count, _ = random_sets._draw_bernoulli(
            set_key, 0, probabilities, order, offsets, bounds, out
        )
        if 0 < count < 3:
            np.testing.assert_array_equal(cells, out[:count])
        else:
            redrawn += 1
    # About 16% of the first draws hold no cell or every cell.
    assert 15 <= redrawn <= 50


@pytest.mark.parametrize("python_kernels", [False, True])
def test_cells_far_below_their_bucket_bound_end_its_skips(
    monkeypatch, python_kernels
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    # Skips at a bound of 2^-63 run far past any integer, so they end the
    # bucket at once; such cells are drawn about once in 1e15 sets.
    probabilities = np.full(500, 1e-25)
    probabilities[7] = 1.0

    sets = _sets(np.uint64(12345), probabilities, range(50))

    assert all(cells.tolist() == [7] for cells in sets)


@pytest.mark.parametrize("python_kernels", [False, True])
def test_uniform_numbers_lie_above_0_and_at_most_1(monkeypatch, python_kernels) -> None:
    _python_kernels(monkeypatch, python_kernels)
    n = 20_000 if not python_kernels else 2_000
    key = np.uint64(random_sets.set_key(resampling_keys(0, np.array([4]), 1)[0], 9))

    values = np.array([random_sets.unit_uniform(key, counter) for counter in range(n)])

    assert values.min() > 0 and values.max() <= 1
    assert (values * 2.0**53 == np.round(values * 2.0**53)).all()
    assert stats.kstest(values, "uniform").pvalue > 0.001
    assert len(np.unique(values)) == n
