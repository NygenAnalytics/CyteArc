"""HTO demultiplexing against planted identities and an exact NB2 quantile."""

import dataclasses
import json
import re
import warnings
from fractions import Fraction
from math import exp, lgamma, log
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

import cytearc.storage.operation_revisions as operation_revisions
from cytearc import DataStore
from cytearc.metadata.arguments import HtoIdentityArguments
from cytearc.quality_control import hto
from cytearc.quality_control.hto import (
    _background_clusters,
    _classify_hto_identities,
    _cluster_labels,
    _clr_normalize,
    _fit_negative_binomial_parameters,
    _hto_clr_diagnostics,
    _hto_demux_method,
    _hto_diagnostics,
    _negative_binomial_cutoff,
    _positive_hto_calls,
    _strict_background_cutoff,
    _validated_hto_counts,
    _warn_unseparated_htos,
    hto_demux,
)
from cytearc.storage.artifacts import ArtifactRef, make_provenance, provenance_hash
from cytearc.storage.operation_revisions import effective_revision
from cytearc.storage.selections import read_stored_selection_indices
from cytearc.utils.logging import logger
from tests.storage_helpers import write_count_store

HTOS = ["HTO_A", "HTO_B", "HTO_C"]


def planted_hto_counts(
    *,
    names: list[str] = HTOS,
    n_singlets: int = 16,
    n_doublets: int = 6,
    n_negatives: int = 6,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return HTO counts and the identity planted in each cell.

    Background counts cycle through 0 to 6 with mean 3 and variance 4, so the
    negative-binomial 99% cutoff of every background lies above all of them,
    and each tagged HTO adds 100 to 199 counts.
    """
    truth = [name for name in names for _ in range(n_singlets)]
    truth += ["Doublet"] * n_doublets + ["Negative"] * n_negatives
    rows = np.arange(len(truth))[:, None]
    counts = (rows * 5 + np.arange(len(names))[None, :] * 3) % 7
    rng = np.random.default_rng(seed)
    for row, label in enumerate(truth):
        if label in names:
            tagged = [names.index(label)]
        elif label == "Doublet":
            tagged = [row % 3, (row + 1) % 3]
        else:
            tagged = []
        for column in tagged:
            counts[row, column] += 100 + rng.integers(0, 100)
    frame = pd.DataFrame(
        counts, columns=names, index=[f"cell{index}" for index in range(len(truth))]
    )
    return frame, np.asarray(truth, dtype=object)


def poisson_background_counts(
    sizes: tuple[int, ...] = (230, 200, 160, 10),
    *,
    n_negatives: int = 0,
    n_doublets: int = 0,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return HTO counts whose untagged channels are Poisson(5) background.

    ``sizes`` holds the number of singlets of each HTO, whose own tag is
    Poisson(200). Negatives carry no tag, and each doublet carries two tags
    drawn from all HTOs but the last.
    """
    rng = np.random.default_rng(seed)
    names = [f"HTO_{index + 1}" for index in range(len(sizes))]
    tags = np.repeat(np.arange(len(sizes)), sizes)
    singlets = rng.poisson(5, size=(tags.size, len(sizes)))
    singlets[np.arange(tags.size), tags] = rng.poisson(200, size=tags.size)
    negatives = rng.poisson(5, size=(n_negatives, len(sizes)))
    doublets = rng.poisson(5, size=(n_doublets, len(sizes)))
    for row in range(n_doublets):
        pair = rng.choice(len(sizes) - 1, size=2, replace=False)
        doublets[row, pair] = rng.poisson(200, size=2)
    truth = np.concatenate(
        [
            np.asarray(names, dtype=object)[tags],
            np.full(n_negatives, "Negative", dtype=object),
            np.full(n_doublets, "Doublet", dtype=object),
        ]
    )
    frame = pd.DataFrame(
        np.vstack([singlets, negatives, doublets]),
        columns=names,
        index=[f"cell{index}" for index in range(truth.size)],
    )
    return frame, truth


def nb2_quantile(mu: float, alpha: float, quantile: float = 0.99) -> int:
    """Smallest count whose NB2 cumulative probability reaches ``quantile``."""
    size = 1 / alpha
    success = 1 / (1 + alpha * mu)
    cumulative = 0.0
    count = 0
    while True:
        cumulative += exp(
            lgamma(count + size)
            - lgamma(count + 1)
            - lgamma(size)
            + size * log(success)
            + count * log(1 - success)
        )
        if cumulative >= quantile:
            return count
        count += 1


def _replace_nb2_model(
    monkeypatch,
    *,
    params: object = (0.0, 1.0),
    retvals: object = None,
    error: Exception | None = None,
) -> None:
    """Replace the statsmodels NB2 model with one that returns a fixed fit."""

    class FitResult:
        pass

    if retvals is not None:
        FitResult.mle_retvals = retvals
    FitResult.params = params

    class Model:
        def __init__(self, *args, **kwargs):
            if error is not None:
                raise error

        def fit(self, **kwargs):
            return FitResult()

    monkeypatch.setattr("statsmodels.discrete.discrete_model.NegativeBinomial", Model)


@pytest.mark.parametrize(
    ("seed", "names"),
    [(0, HTOS), (1, HTOS), (2, ["cluster", "HTO_B", "HTO_C"])],
    ids=["seed0", "seed1", "hto_named_cluster"],
)
def test_hto_demux_recovers_planted_singlets_doublets_and_negatives(seed, names):
    counts, truth = planted_hto_counts(names=names, seed=seed)
    original = counts.copy(deep=True)

    identities = hto_demux(counts, random_seed=seed)

    np.testing.assert_array_equal(identities.to_numpy(), truth)
    assert identities.index.equals(counts.index)
    pd.testing.assert_frame_equal(counts, original)
    pd.testing.assert_series_equal(hto_demux(counts, random_seed=seed), identities)
    # Permuting cells permutes their identities.
    order = np.random.default_rng(seed).permutation(len(counts))
    permuted = hto_demux(counts.iloc[order], random_seed=seed)
    np.testing.assert_array_equal(permuted.to_numpy(), truth[order])


@pytest.mark.parametrize(
    ("mu", "alpha"),
    [(1.0, 1.0), (3.0, 0.1), (0.2, 4.0), (7.5, 0.05), (3.0, 1 / 9)],
)
def test_negative_binomial_cutoff_is_the_nb2_99th_percentile(mu, alpha):
    assert _negative_binomial_cutoff(mu, alpha) == nb2_quantile(mu, alpha)
    assert _negative_binomial_cutoff(mu, alpha, quantile=0.5) == nb2_quantile(
        mu, alpha, 0.5
    )


def test_positive_calls_exceed_the_unshifted_cutoff_strictly(monkeypatch):
    cutoff = nb2_quantile(1.0, 1.0)
    counts = pd.DataFrame({"HTO_A": [1, 2, cutoff, cutoff + 1]})
    fitted: list[tuple[list[int], str]] = []

    def fit(values, name):
        fitted.append((np.asarray(values).tolist(), name))
        return 1.0, 1.0

    monkeypatch.setattr(hto, "_fit_negative_binomial_parameters", fit)

    positive = _positive_hto_calls(counts, np.asarray([0, 0, 1, 1]))

    # The fit sees the cluster with the lower raw mean, and a count equal to
    # the 99th percentile itself is not positive.
    assert fitted == [([1, 2], "HTO_A")]
    assert positive["HTO_A"].tolist() == [False, False, False, True]


@pytest.mark.parametrize("quantile", [0.5, 0.99, 0.999, 0.9999])
def test_positive_calls_exceed_the_requested_quantile(monkeypatch, quantile):
    monkeypatch.setattr(
        hto, "_fit_negative_binomial_parameters", lambda values, name: (3.0, 0.1)
    )
    values = np.arange(30)
    counts = pd.DataFrame({"HTO_A": values})

    positive = _positive_hto_calls(
        counts, np.repeat([0, 1], 15), positive_quantile=quantile
    )

    cutoff = nb2_quantile(3.0, 0.1, quantile)
    assert positive["HTO_A"].tolist() == (values > cutoff).tolist()


def test_background_cluster_uses_raw_means():
    counts = pd.DataFrame({"HTO_A": [0, 100, 40, 40]})
    cluster_labels = np.asarray([0, 0, 1, 1])

    background = _background_clusters(counts, cluster_labels)
    normalized_means = _clr_normalize(counts).groupby(cluster_labels).mean()

    # Cluster 1 has the lower raw mean although cluster 0 has the lower CLR mean.
    assert background["HTO_A"] == 1
    assert normalized_means["HTO_A"].idxmin() == 0


def test_classification_assigns_singlets_their_positive_tag():
    index = ["negative", "singlet", "doublet", "tie", "second"]
    normalized = pd.DataFrame(
        {
            "HTO_A": [0.2, 0.2, 1.0, 0.5, 0.9],
            "HTO_B": [0.1, 1.5, 0.9, 0.5, 0.3],
        },
        index=index,
    )
    positive = pd.DataFrame(
        {
            "HTO_A": [False, True, True, True, False],
            "HTO_B": [False, False, True, False, True],
        },
        index=index,
    )

    identities = _classify_hto_identities(positive)

    # A singlet takes its one positive tag, even where another tag has the
    # higher CLR value, as for "singlet" and "second". The CLR argmax is a
    # diagnostic only.
    assert identities.to_dict() == {
        "negative": "Negative",
        "singlet": "HTO_A",
        "doublet": "Doublet",
        "tie": "HTO_A",
        "second": "HTO_B",
    }
    diagnostics = _hto_diagnostics(normalized)
    assert diagnostics["max_id"].to_dict() == {
        "negative": "HTO_A",
        "singlet": "HTO_B",
        "doublet": "HTO_A",
        "tie": "HTO_A",
        "second": "HTO_A",
    }
    assert diagnostics["second_id"].to_dict() == {
        "negative": "HTO_B",
        "singlet": "HTO_A",
        "doublet": "HTO_B",
        "tie": "HTO_B",
        "second": "HTO_B",
    }
    np.testing.assert_allclose(
        diagnostics["margin"].to_numpy(), [0.1, 1.3, 0.1, 0.0, 0.6], rtol=0, atol=1e-15
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((0.0, 1.0), "mean must be finite and greater than 0"),
        ((np.inf, 1.0), "mean must be finite and greater than 0"),
        ((1.0, 0.0), "dispersion must be finite and greater than 0"),
        ((1.0, np.nan), "dispersion must be finite and greater than 0"),
        ((1.0, 1.0, 1.0), "quantile must be between 0 and 1"),
        ((1.0, 1.0, 0.0), "quantile must be between 0 and 1"),
        ((1.0, 1.0, np.nan), "quantile must be between 0 and 1"),
        # The success probability underflows to zero.
        ((1e300, 1e300), "parameters produced an invalid distribution"),
        # The size 1 / alpha overflows.
        ((1.0, 1e-320), "parameters produced an invalid distribution"),
    ],
)
def test_negative_binomial_cutoff_rejects_invalid_parameters(arguments, message):
    with pytest.raises(ValueError, match=f"^Negative-binomial {message}$"):
        _negative_binomial_cutoff(*arguments)


@pytest.mark.parametrize("quantile", [np.inf, np.nan, 2.5])
def test_negative_binomial_cutoff_requires_a_finite_integer_quantile(
    monkeypatch, quantile
):
    monkeypatch.setattr("scipy.stats.nbinom.ppf", lambda *args, **kwargs: quantile)

    with pytest.raises(
        ValueError, match="^Negative-binomial cutoff must be a finite integer$"
    ):
        _negative_binomial_cutoff(mu=1, alpha=1)


def test_positive_calls_report_which_hto_has_an_invalid_cutoff(monkeypatch):
    counts = pd.DataFrame({"HTO_A": [1, 2, 60, 70], "HTO_B": [50, 60, 1, 2]})
    monkeypatch.setattr(
        hto,
        "_fit_negative_binomial_parameters",
        lambda values, name: (1e300, 1e300) if name == "HTO_B" else (1.0, 1.0),
    )

    with pytest.raises(
        ValueError, match="^Negative-binomial cutoff is invalid for HTO 'HTO_B'$"
    ) as caught:
        _positive_hto_calls(counts, np.asarray([0, 0, 1, 1]))
    assert "invalid distribution" in str(caught.value.__cause__)


@pytest.mark.parametrize(
    ("counts", "labels", "message"),
    [
        (
            pd.DataFrame({"HTO_A": [1, 10, 20]}),
            np.asarray([0, 1, 2]),
            "must contain at least two cells",
        ),
    ],
    ids=["single-cell"],
)
def test_positive_calls_reject_invalid_backgrounds(counts, labels, message):
    with pytest.raises(
        ValueError, match=f"^Background cluster for HTO 'HTO_A' {message}$"
    ):
        _positive_hto_calls(counts, labels)


def test_an_all_zero_background_calls_every_nonzero_count_positive(monkeypatch):
    # The all-zero background used to abort demultiplexing. Its cutoff is now
    # 0, with a warning, and the other HTO keeps its negative-binomial fit.
    counts = pd.DataFrame(
        {"HTO_A": [0, 0, 1, 10, 20], "HTO_B": [40, 60, 2, 1, 0]},
        index=[f"cell{index}" for index in range(5)],
    )
    labels = np.asarray([0, 0, 1, 1, 1])
    fitted: list[tuple[list[int], str]] = []

    def fit(values, name):
        fitted.append((np.asarray(values).tolist(), name))
        return 1.0, 1.0

    monkeypatch.setattr(hto, "_fit_negative_binomial_parameters", fit)

    with pytest.warns(UserWarning) as caught:
        positive = _positive_hto_calls(counts, labels)

    assert fitted == [([2, 1, 0], "HTO_B")]
    assert [str(warning.message) for warning in caught] == [
        "Background cluster for HTO 'HTO_A' contains only zero counts, so its "
        "cutoff is 0 and every nonzero count of this HTO is called positive, "
        "including a few background counts in cells outside that cluster."
    ]
    assert positive["HTO_A"].tolist() == [False, False, True, True, True]
    cutoff = nb2_quantile(1.0, 1.0)
    assert positive["HTO_B"].tolist() == (counts["HTO_B"] > cutoff).tolist()
    assert positive.index.equals(counts.index)


@pytest.mark.parametrize(
    "background",
    [
        np.tile(np.arange(7), 40),
        np.random.default_rng(4).negative_binomial(2, 0.3, size=300),
    ],
    ids=["cycled", "negative_binomial"],
)
def test_negative_binomial_fit_is_the_nb2_maximum_likelihood(background):
    from scipy.optimize import minimize_scalar
    from scipy.stats import nbinom

    mu, alpha = _fit_negative_binomial_parameters(background, "HTO_A")

    # The NB2 likelihood of an intercept-only model peaks at the sample mean,
    # and its dispersion maximizes the likelihood profiled at that mean.
    mean = float(np.mean(background))

    def negative_log_likelihood(log_alpha: float) -> float:
        dispersion = np.exp(log_alpha)
        return -float(
            nbinom.logpmf(background, 1 / dispersion, 1 / (1 + dispersion * mean)).sum()
        )

    profile = minimize_scalar(
        negative_log_likelihood,
        bounds=(-12, 5),
        method="bounded",
        options={"xatol": 1e-10},
    )
    assert mu == pytest.approx(mean, rel=1e-5)
    assert alpha == pytest.approx(np.exp(profile.x), rel=1e-4)


@pytest.mark.parametrize(
    ("fit", "message"),
    [
        ({"retvals": {"converged": False}}, "did not converge"),
        ({}, "did not converge"),
        ({"retvals": {"converged": "yes"}}, "did not converge"),
        (
            {"retvals": {"converged": True}, "params": {"intercept": 1.0}},
            "returned invalid parameters",
        ),
        (
            {"retvals": {"converged": True}, "params": np.asarray([0.5])},
            "returned invalid parameters",
        ),
        (
            {"retvals": {"converged": True}, "params": np.zeros(3)},
            "returned invalid parameters",
        ),
        (
            {"retvals": {"converged": True}, "params": np.asarray([np.inf, 1.0])},
            "returned invalid mean",
        ),
        (
            {"retvals": {"converged": True}, "params": np.asarray([np.nan, 1.0])},
            "returned invalid mean",
        ),
        (
            {"retvals": {"converged": True}, "params": np.asarray([0.0, 0.0])},
            "returned invalid dispersion",
        ),
        (
            {"retvals": {"converged": True}, "params": np.asarray([0.0, -0.5])},
            "returned invalid dispersion",
        ),
    ],
    ids=[
        "not_converged",
        "no_convergence_record",
        "convergence_not_boolean",
        "mapping",
        "one_parameter",
        "three_parameters",
        "infinite_mean",
        "nan_mean",
        "zero_dispersion",
        "negative_dispersion",
    ],
)
def test_negative_binomial_fit_rejects_unusable_results(monkeypatch, fit, message):
    _replace_nb2_model(monkeypatch, **fit)

    with pytest.raises(
        ValueError,
        match=f"^Negative-binomial background fit {message} for HTO 'HTO_A'$",
    ):
        _fit_negative_binomial_parameters(np.asarray([1, 2]), "HTO_A")


def test_negative_binomial_fit_wraps_optimizer_errors(monkeypatch):
    _replace_nb2_model(monkeypatch, error=RuntimeError("optimizer failed"))

    with pytest.raises(
        ValueError, match="^Negative-binomial background fit failed for HTO 'HTO_A'$"
    ) as caught:
        _fit_negative_binomial_parameters(np.asarray([1, 2]), "HTO_A")
    assert str(caught.value.__cause__) == "optimizer failed"


def test_cluster_labels_reject_collapsed_kmeans(monkeypatch):
    class CollapsedKMeans:
        def __init__(self, **kwargs):
            pass

        def fit_predict(self, values):
            return np.asarray([0, 0, 1])

    monkeypatch.setattr("sklearn.cluster.KMeans", CollapsedKMeans)
    normalized = pd.DataFrame(
        {
            "HTO_A": [0.0, 1.0, 2.0],
            "HTO_B": [2.0, 1.0, 0.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="^HTO clustering produced 2 occupied clusters; expected 3$",
    ):
        _cluster_labels(normalized, random_seed=0)


@pytest.mark.parametrize(
    ("hto_counts", "error", "message"),
    [
        (np.asarray([[1], [2]]), TypeError, "hto_counts must be a pandas DataFrame"),
        (
            pd.DataFrame(index=range(2)),
            ValueError,
            "hto_counts must contain at least one HTO",
        ),
        (
            pd.DataFrame(np.ones((3, 2)), columns=["HTO_A", "HTO_A"]),
            ValueError,
            "HTO IDs must be unique",
        ),
        (pd.DataFrame({" ": [1, 2]}), ValueError, "HTO IDs must be non-empty strings"),
        (
            pd.DataFrame({"Negative": [1, 2]}),
            ValueError,
            "HTO IDs conflict with reserved identity labels: 'Negative'",
        ),
        (
            pd.DataFrame({"HTO_A": [1, 2]}, index=["cell", "cell"]),
            ValueError,
            "hto_counts cell index must be unique",
        ),
        (
            pd.DataFrame({"HTO_A": ["1", "2"]}),
            TypeError,
            "hto_counts must contain only numeric raw counts",
        ),
        (
            pd.DataFrame({"HTO_A": [1 + 0j, 2 + 0j, 3 + 0j]}),
            TypeError,
            "hto_counts must contain only real numeric raw counts",
        ),
        (
            pd.DataFrame({"HTO_A": [1, np.nan]}),
            ValueError,
            "hto_counts must contain only finite raw counts",
        ),
        (
            pd.DataFrame({"HTO_A": [1, -1]}),
            ValueError,
            "hto_counts must contain only nonnegative raw counts",
        ),
        (
            pd.DataFrame({"HTO_A": [1, 1.5]}),
            ValueError,
            "hto_counts must contain integer-valued raw counts",
        ),
        (
            pd.DataFrame({"HTO_A": [1, 2], "HTO_B": [2, 1]}),
            ValueError,
            "HTO demultiplexing requires at least 3 selected cells",
        ),
        (
            pd.DataFrame({"HTO_A": [0, 0], "HTO_B": [1, 2], "HTO_C": [0, 0]}),
            ValueError,
            "HTO demultiplexing requires at least 4 selected cells",
        ),
        (
            pd.DataFrame({"HTO_A": [0, 0, 0], "HTO_B": [1, 2, 3]}),
            ValueError,
            "HTOs with no positive counts cannot be demultiplexed: 'HTO_A'",
        ),
        (
            pd.DataFrame({"HTO_A": [1, 1, 1], "HTO_B": [2, 2, 2]}),
            ValueError,
            "HTO demultiplexing requires at least 3 distinct normalized cell "
            "profiles; found 1",
        ),
    ],
    ids=[
        "not-dataframe",
        "no-htos",
        "duplicate-htos",
        "empty-hto",
        "reserved-hto",
        "duplicate-cells",
        "nonnumeric",
        "complex",
        "nonfinite",
        "negative",
        "fractional",
        "too-few-cells",
        "too-few-cells-for-three-htos",
        "all-zero-hto",
        "identical-profiles",
    ],
)
def test_hto_demux_rejects_invalid_input(hto_counts, error, message):
    with pytest.raises(error, match=f"^{re.escape(message)}$"):
        hto_demux(hto_counts)


@pytest.mark.parametrize("random_seed", [True, 1.5, "0", None])
def test_hto_demux_requires_an_integer_seed(random_seed):
    counts, _ = planted_hto_counts()

    with pytest.raises(TypeError, match="^random_seed must be an integer$"):
        hto_demux(counts, random_seed=random_seed)


_NOT_REAL = (TypeError, "positive_quantile must be a real number")
_NOT_FINITE = (ValueError, "positive_quantile must be finite")
_OUT_OF_RANGE = (ValueError, "positive_quantile must be greater than 0 and less than 1")
INVALID_QUANTILES = [
    (True, *_NOT_REAL),
    (np.False_, *_NOT_REAL),
    ("0.99", *_NOT_REAL),
    (None, *_NOT_REAL),
    (0.99 + 0j, *_NOT_REAL),
    (np.nan, *_NOT_FINITE),
    (-np.inf, *_NOT_FINITE),
    (0, *_OUT_OF_RANGE),
    (0.0, *_OUT_OF_RANGE),
    (1, *_OUT_OF_RANGE),
    (1.0, *_OUT_OF_RANGE),
    (-0.5, *_OUT_OF_RANGE),
    (99.0, *_OUT_OF_RANGE),
]


@pytest.mark.parametrize(("positive_quantile", "error", "message"), INVALID_QUANTILES)
def test_hto_demux_requires_a_quantile_between_0_and_1(
    positive_quantile, error, message
):
    counts, _ = planted_hto_counts()

    with pytest.raises(error, match=f"^{re.escape(message)}$"):
        hto_demux(counts, positive_quantile=positive_quantile)


def test_hto_demux_defaults_to_the_seurat_positive_quantile():
    counts, truth = planted_hto_counts()

    identities = hto_demux(counts)

    for quantile in (0.99, np.float64(0.99), Fraction(99, 100)):
        pd.testing.assert_series_equal(
            hto_demux(counts, positive_quantile=quantile), identities
        )
    np.testing.assert_array_equal(identities.to_numpy(), truth)


@pytest.mark.parametrize("seed", [0, 1])
def test_a_higher_positive_quantile_calls_fewer_false_doublets(seed):
    # Every cell carries one tag. Without a negative population, k-means
    # splits a large hashtag into two of its clusters, and background counts
    # above a tag's cutoff make false doublets.
    counts, truth = poisson_background_counts(seed=seed)

    # The k-means restarts over these few cells gain nothing from threads.
    with threadpool_limits(limits=1):
        calls = {
            quantile: hto_demux(counts, positive_quantile=quantile).to_numpy()
            for quantile in (0.99, 0.999, 0.9999)
        }

    doublets = {quantile: int(np.sum(c == "Doublet")) for quantile, c in calls.items()}
    assert doublets[0.99] > 0.05 * len(counts)
    assert doublets[0.999] < doublets[0.99] / 2
    assert doublets[0.9999] <= doublets[0.999]
    for called in calls.values():
        # A higher cutoff still finds every dominant tag, and singlets keep
        # the tag they carry.
        assert not np.any(called == "Negative")
        singlets = called != "Doublet"
        np.testing.assert_array_equal(called[singlets], truth[singlets])


def test_a_higher_positive_quantile_still_calls_doublets_and_negatives():
    counts, truth = poisson_background_counts(n_negatives=30, n_doublets=48, seed=1)

    with threadpool_limits(limits=1):
        default = hto_demux(counts).to_numpy()
        stricter = hto_demux(counts, positive_quantile=0.999).to_numpy()

    def false_doublets(called: np.ndarray) -> int:
        return int(np.sum((called == "Doublet") & (truth != "Doublet")))

    assert false_doublets(stricter) < false_doublets(default)
    for called in (default, stricter):
        np.testing.assert_array_equal(called[truth == "Doublet"], "Doublet")
        assert np.mean(called[truth == "Negative"] == "Negative") >= 0.9


def test_hto_identity_artifact_holds_the_planted_identities_of_selected_cells(
    tmp_path: Path,
):
    counts, truth = planted_hto_counts()
    n_cells = len(counts)
    rna = np.random.default_rng(3).poisson(2.0, size=(n_cells, 12))
    path = tmp_path / "store.zarr"
    write_count_store(str(path), {"RNA": rna, "HTO": counts.to_numpy()}, "uint16")
    store = DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, nthreads=1
    )
    keep = np.ones(n_cells, dtype=bool)
    keep[[2, 17, 40, 55]] = False
    store.cells.insert("tagged", keep)
    selection = store.snapshot_cell_selection("tagged")

    ref = store.qc.hto_demultiplexing(selection)

    status = store.artifacts.inspect(ref)
    assert ref.kind == "hto_identity"
    assert status.operation == "run_hto_demultiplexing"
    assert status.parameters["method"] == {
        "normalization": "clr_per_hto",
        "clustering": {
            "method": "kmeans",
            "init": "random",
            "n_starts": 100,
            "cluster_count": "n_htos_plus_one",
        },
        "background": "raw_mean",
        "cutoff": {
            "distribution": "negative_binomial_nb2",
            "quantile": 0.99,
            "location": 0,
            "comparison": "strictly_greater",
        },
        "singlet_assignment": "clr_argmax",
    }
    cells = read_stored_selection_indices(
        store.zw,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    np.testing.assert_array_equal(cells, np.flatnonzero(keep))
    # The HTO assay names its features HTO0, HTO1, and HTO2.
    renamed = {name: f"HTO{index}" for index, name in enumerate(HTOS)}
    expected = [renamed.get(label, label) for label in truth[keep]]
    np.testing.assert_array_equal(store.artifacts.load(ref)["values"][:], expected)
    assert store.qc.hto_demultiplexing(selection) == ref
    assert "HTO0" not in store.cells.columns


def test_default_hto_identity_is_the_one_cytearc_1_2_recorded():
    arguments = HtoIdentityArguments(
        cell_selection=ArtifactRef(
            scope="datastore",
            assay=None,
            kind="cell_selection",
            artifact_id="1" * 64,
        ),
        feature_ids_fingerprint="2" * 64,
        method=_hto_demux_method(0.99),
        random_seed=0,
        invalidate_cache=False,
    )

    def identity(arguments: HtoIdentityArguments) -> str:
        record = arguments.to_record()
        return provenance_hash(
            make_provenance(
                operation=arguments.operation,
                parameters=record.parameters,
                inputs=record.inputs,
            )
        )

    # CyteArc 1.2.0 recorded its fixed quantile, 0.99, as the cutoff quantile
    # of the method record, so stored artifacts of the default keep their
    # parameters and inputs: revision 2 supersedes them as matches that differ
    # only in their revision.
    assert identity(arguments) == (
        "eade487927ab18af43de927d0ff3d067ae58d3de6fde4f5d6b144cf4f0c6c80c"
    )
    record = arguments.to_record()
    assert (
        effective_revision(
            arguments.operation,
            arguments.artifact_kind,
            record.parameters,
            record.inputs,
        )
        == 2
    )
    stricter = dataclasses.replace(arguments, method=_hto_demux_method(0.999))
    assert stricter.to_record().parameters["method"]["cutoff"]["quantile"] == 0.999
    assert identity(stricter) != identity(arguments)


@pytest.mark.parametrize(("positive_quantile", "error", "message"), INVALID_QUANTILES)
def test_hto_demultiplexing_rejects_an_invalid_quantile_before_reading_counts(
    tmp_path: Path, monkeypatch, positive_quantile, error, message
):
    counts, _ = planted_hto_counts()
    store = _hto_store(tmp_path, counts)
    selection = store.snapshot_cell_selection("I")

    def unread(*args, **kwargs):
        raise AssertionError("the selection was read")

    monkeypatch.setattr(
        "cytearc.datastore._operations.quality_control.read_stored_selection_indices",
        unread,
    )
    with pytest.raises(error, match=f"^{re.escape(message)}$"):
        store.qc.hto_demultiplexing(selection, positive_quantile=positive_quantile)
    assert store.artifacts.list(kind="hto_identity", from_assay="HTO") == []


def test_each_positive_quantile_has_its_own_hto_identity_artifact(tmp_path: Path):
    counts, _ = poisson_background_counts(seed=0)
    store = _hto_store(tmp_path, counts)
    selection = store.snapshot_cell_selection("I")
    # The HTO assay names its features HTO0 to HTO3.
    renamed = counts.set_axis([f"HTO{index}" for index in range(4)], axis=1)

    # The datastore and the direct calls run their k-means restarts alike.
    with threadpool_limits(limits=1):
        default = store.qc.hto_demultiplexing(selection)
        stricter = store.qc.hto_demultiplexing(selection, positive_quantile=0.999)
        expected = {
            quantile: hto_demux(renamed, positive_quantile=quantile).to_numpy()
            for quantile in (0.99, 0.999)
        }

        assert stricter != default
        for ref, quantile in ((default, 0.99), (stricter, 0.999)):
            parameters = store.artifacts.inspect(ref).parameters
            assert parameters is not None
            assert parameters["method"]["cutoff"]["quantile"] == quantile
            np.testing.assert_array_equal(
                store.artifacts.load(ref)["values"][:], expected[quantile]
            )
        assert np.sum(expected[0.999] == "Doublet") < np.sum(
            expected[0.99] == "Doublet"
        )
        # Equal quantiles reuse their artifact, however they are spelled.
        assert store.qc.hto_demultiplexing(selection) == default
        assert store.qc.hto_demultiplexing(selection, positive_quantile=0.99) == default
        for spelling in (np.float64(0.999), Fraction(999, 1000)):
            assert (
                store.qc.hto_demultiplexing(selection, positive_quantile=spelling)
                == stricter
            )
    assert len(store.artifacts.list(kind="hto_identity", from_assay="HTO")) == 2


def _hto_store(tmp_path: Path, counts: pd.DataFrame) -> DataStore:
    """Write ``counts`` as the HTO assay of a new store beside a small RNA assay."""
    rna = np.random.default_rng(3).poisson(2.0, size=(len(counts), 12))
    path = tmp_path / "store.zarr"
    write_count_store(str(path), {"RNA": rna, "HTO": counts.to_numpy()}, "uint16")
    return DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, nthreads=1
    )


def unbalanced_hto_counts(
    n_cells: int = 800, seed: int = 0
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return counts of four HTOs pooled 70/10/10/10 and the planted tags.

    Every cell carries one tag of NB2(400, size 5) counts over NB2(20, size 2)
    background. The per-HTO CLR divides the dominant tag by a large geometric
    mean, so a weakly stained cell of it can have its highest CLR value on a
    tag that the cutoff calls negative.
    """
    rng = np.random.default_rng(seed)

    def nb(mean: float, size: float, shape) -> np.ndarray:
        return rng.negative_binomial(size, size / (size + mean), size=shape)

    counts = np.column_stack([nb(20, 2, n_cells) for _ in range(4)])
    tags = rng.choice(4, n_cells, p=[0.7, 0.1, 0.1, 0.1])
    counts[np.arange(n_cells), tags] += nb(400, 5, n_cells)
    names = [f"HTO_{index}" for index in range(4)]
    return (
        pd.DataFrame(counts, columns=names),
        np.asarray(names, dtype=object)[tags],
    )


def test_unbalanced_pools_assign_each_singlet_its_positive_tag():
    counts, truth = unbalanced_hto_counts()

    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error")
        identities = hto_demux(counts)
        validated = _validated_hto_counts(counts)
        labels = _cluster_labels(_clr_normalize(validated), 0)
        positive = _positive_hto_calls(validated, labels)

    identity = identities.to_numpy()
    assert identities.index.equals(counts.index)
    n_positive = positive.sum(axis=1).to_numpy()
    singlets = n_positive == 1
    assert singlets.sum() > 0.9 * len(counts)
    # Each singlet takes the one tag called positive, which is its planted tag.
    np.testing.assert_array_equal(
        identity[singlets], positive[singlets].idxmax(axis=1).to_numpy()
    )
    np.testing.assert_array_equal(identity[singlets], truth[singlets])
    np.testing.assert_array_equal(
        identity[~singlets],
        np.where(n_positive[~singlets] == 0, "Negative", "Doublet"),
    )
    # The CLR argmax, which named singlets before, is a tag called negative
    # for dozens of them here.
    max_id = _hto_clr_diagnostics(counts)["max_id"].to_numpy()
    relabelled = np.flatnonzero(singlets & (max_id != identity))
    assert len(relabelled) >= 30
    np.testing.assert_array_equal(
        max_id[relabelled] != truth[relabelled], np.ones(len(relabelled), bool)
    )
    columns = [counts.columns.get_loc(tag) for tag in max_id[relabelled]]
    assert not positive.to_numpy()[relabelled, columns].any()


def test_hto_diagnostics_name_the_two_highest_clr_tags():
    normalized = pd.DataFrame(
        {
            "HTO_A": [0.5, 2.0, -1.0],
            "HTO_B": [1.5, 2.0, 0.25],
            "HTO_C": [1.0, 0.0, 0.0],
        },
        index=["b_first", "tie", "negative_values"],
    )

    diagnostics = _hto_diagnostics(normalized)

    assert list(diagnostics.columns) == ["max_id", "second_id", "margin"]
    assert diagnostics.index.equals(normalized.index)
    assert diagnostics["max_id"].tolist() == ["HTO_B", "HTO_A", "HTO_B"]
    # Ties go to the earlier HTO, so the tie has a zero margin.
    assert diagnostics["second_id"].tolist() == ["HTO_C", "HTO_B", "HTO_C"]
    np.testing.assert_array_equal(diagnostics["margin"].to_numpy(), [0.5, 0.0, 0.25])

    one = _hto_diagnostics(pd.DataFrame({"HTO_A": [0.1, -0.2]}, index=["x", "y"]))

    assert one["max_id"].tolist() == ["HTO_A", "HTO_A"]
    assert one["second_id"].tolist() == ["", ""]
    assert np.isnan(one["margin"].to_numpy()).all()


@pytest.mark.parametrize(
    ("mu", "alpha", "quantile"),
    [(1.0, 1.0, 0.99), (3.0, 0.1, 0.99), (20.0, 0.5, 0.9), (7.5, 0.05, 0.999)],
)
def test_strict_background_cutoff_is_the_nb2_tail_quantile(mu, alpha, quantile):
    tail = (1 - quantile) ** 3
    size = 1 / alpha
    success = 1 / (1 + alpha * mu)

    def survival(count: int) -> float:
        # The probability of a count above ``count``: one minus the NB2 pmf
        # summed from 0 to ``count``.
        return 1.0 - sum(
            exp(
                lgamma(k + size)
                - lgamma(k + 1)
                - lgamma(size)
                + size * log(success)
                + k * log(1 - success)
            )
            for k in range(count + 1)
        )

    strict = _strict_background_cutoff(mu, alpha, quantile)

    # The smallest count whose upper tail probability is at most the tail.
    assert strict.is_integer()
    assert survival(int(strict)) <= tail * (1 + 1e-6)
    assert survival(int(strict) - 1) > tail * (1 - 1e-6)
    assert strict >= _negative_binomial_cutoff(mu, alpha, quantile)


def test_strict_background_cutoff_stays_finite_for_quantiles_near_one():
    # (1 - q) ** 3 underflows to zero; the tail is floored at the smallest
    # normal float instead.
    strict = _strict_background_cutoff(3.0, 0.1, 1 - 1e-120)

    assert np.isfinite(strict)
    assert strict > _strict_background_cutoff(3.0, 0.1, 0.999)


def test_unseparated_htos_are_named_in_one_warning():
    counts = pd.DataFrame(
        {
            "HTO_A": [0, 1, 50, 60, 70],
            "HTO_B": [0, 1, 12, 13, 40],
            "HTO_C": [5, 6, 0, 0, 0],
        }
    )
    positive = counts > pd.Series({"HTO_A": 10, "HTO_B": 10, "HTO_C": 10})
    strict = pd.Series({"HTO_A": 30.0, "HTO_B": 30.0, "HTO_C": 30.0})

    with pytest.warns(UserWarning) as caught:
        _warn_unseparated_htos(counts, positive, strict, 0.99)

    # HTO_A has 3 strong counts among 3 positive cells and HTO_C no positive
    # cell to judge; HTO_B has 1 strong count among 3 positive cells.
    assert [str(warning.message) for warning in caught] == [
        "Positive and background HTO counts overlap for 'HTO_B' (1 of 3 positive "
        "cells): fewer than half of the cells called positive have counts that "
        "the background fit exceeds with probability 1e-06, so these calls may "
        "be background, as in an unused tag or pure background. Check the HTO "
        "count distributions; a tag without a separated positive population "
        "cannot be demultiplexed."
    ]
    # Half of the positive cells above the strict cutoff counts as separated.
    halves = pd.DataFrame({"HTO_A": [11, 31]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_unseparated_htos(halves, halves > 10, pd.Series({"HTO_A": 30.0}), 0.99)


def test_pure_background_counts_warn_that_every_hto_overlaps_its_background():
    rng = np.random.default_rng(2)
    counts = pd.DataFrame(
        rng.negative_binomial(2, 2 / 12, size=(600, 4)),
        columns=[f"HTO_{index}" for index in range(4)],
    )

    with threadpool_limits(limits=1), pytest.warns(UserWarning) as caught:
        identities = hto_demux(counts)

    messages = [str(warning.message) for warning in caught]
    assert len(messages) == 1
    assert messages[0].startswith("Positive and background HTO counts overlap for ")
    for name in counts.columns:
        assert f"{name!r} (" in messages[0]
    # Background alone still yields calls; the warning is what flags them.
    assert set(identities) - {"Negative", "Doublet"}


def test_an_unused_hto_is_named_as_overlapping_its_background():
    rng = np.random.default_rng(2)
    counts = rng.negative_binomial(2, 2 / 12, size=(900, 4))
    tags = rng.integers(0, 3, 900)
    counts[np.arange(900), tags] += rng.negative_binomial(5, 5 / 305, size=900)
    frame = pd.DataFrame(counts, columns=[f"HTO_{index}" for index in range(4)])

    with threadpool_limits(limits=1), pytest.warns(UserWarning) as caught:
        identities = hto_demux(frame).to_numpy()

    messages = [str(warning.message) for warning in caught]
    assert len(messages) == 1
    assert messages[0].startswith(
        "Positive and background HTO counts overlap for 'HTO_3' ("
    )
    assert "HTO_0" not in messages[0]
    singlets = ~np.isin(identities, ["Negative", "Doublet"])
    np.testing.assert_array_equal(
        identities[singlets], np.asarray(frame.columns)[tags][singlets]
    )


def test_separated_hashing_data_raise_no_hto_warning():
    planted, truth = planted_hto_counts()
    fixture = json.loads(
        (Path(__file__).parent / "seurat_hto_5_5_1_golden.json").read_text()
    )
    golden = pd.DataFrame(
        fixture["inputs"]["rawCounts"], columns=fixture["inputs"]["htoNames"]
    )

    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error")
        np.testing.assert_array_equal(hto_demux(planted).to_numpy(), truth)
        hto_demux(golden)


def test_clean_hashing_with_zero_background_counts_is_demultiplexed():
    # Negatives and untagged channels hold no counts at all, so every
    # background cluster is all zero. This used to abort demultiplexing.
    rng = np.random.default_rng(4)
    tags = rng.integers(0, 3, 300)
    counts = np.zeros((300, 3), dtype=np.int64)
    counts[np.arange(300), tags] = rng.integers(50, 400, 300)
    negative = np.arange(300) % 10 == 0
    counts[negative] = 0
    frame = pd.DataFrame(counts, columns=["HTO_A", "HTO_B", "HTO_C"])
    truth = np.where(negative, "Negative", np.asarray(frame.columns)[tags])

    with threadpool_limits(limits=1), pytest.warns(UserWarning) as caught:
        identities = hto_demux(frame)

    np.testing.assert_array_equal(identities.to_numpy(), truth)
    assert sorted(str(warning.message) for warning in caught) == [
        f"Background cluster for HTO {name!r} contains only zero counts, so its "
        "cutoff is 0 and every nonzero count of this HTO is called positive, "
        "including a few background counts in cells outside that cluster."
        for name in ("HTO_A", "HTO_B", "HTO_C")
    ]


def test_hto_identity_artifact_stores_clr_diagnostics_as_revision_2(
    tmp_path: Path,
):
    counts, truth = planted_hto_counts()
    store = _hto_store(tmp_path, counts)
    selection = store.snapshot_cell_selection("I")
    renamed = counts.set_axis([f"HTO{index}" for index in range(3)], axis=1)

    with threadpool_limits(limits=1):
        ref = store.qc.hto_demultiplexing(selection)
        identities = hto_demux(renamed)
    expected = _hto_clr_diagnostics(renamed)

    status = store.artifacts.inspect(ref)
    assert status.revision == 2
    assert status.is_current
    payload = store.artifacts.load(ref)
    renaming = {name: f"HTO{index}" for index, name in enumerate(HTOS)}
    np.testing.assert_array_equal(
        payload["values"][:], [renaming.get(label, label) for label in truth]
    )
    np.testing.assert_array_equal(payload["values"][:], identities)
    np.testing.assert_array_equal(payload["max_id"][:], expected["max_id"])
    np.testing.assert_array_equal(payload["second_id"][:], expected["second_id"])
    # The store and the in-memory helper read the counts in different layouts,
    # which NumPy releases round differently in the last bits.
    np.testing.assert_allclose(
        payload["margin"][:], expected["margin"], rtol=0, atol=1e-12
    )
    normalized = _clr_normalize(_validated_hto_counts(renamed)).to_numpy()
    top_two = -np.sort(-normalized, axis=1)[:, :2]
    np.testing.assert_allclose(
        payload["margin"][:], top_two[:, 0] - top_two[:, 1], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        store.artifacts.load_values(ref, value="margin").values,
        expected["margin"],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        store.artifacts.load_values(ref, value="max_id").values, expected["max_id"]
    )


def test_hto_identity_of_revision_1_is_recomputed_not_reused(
    tmp_path: Path, monkeypatch
):
    counts, _ = planted_hto_counts()
    store = _hto_store(tmp_path, counts)
    selection = store.snapshot_cell_selection("I")
    monkeypatch.setattr(
        operation_revisions, "OPERATION_REVISIONS", MappingProxyType({})
    )
    with threadpool_limits(limits=1):
        legacy = store.qc.hto_demultiplexing(selection)
    assert store.artifacts.inspect(legacy).revision == 1
    monkeypatch.undo()

    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="INFO"
    )
    try:
        with threadpool_limits(limits=1):
            current = store.qc.hto_demultiplexing(selection)
    finally:
        logger.remove(sink)

    assert current != legacy
    assert store.artifacts.inspect(current).revision == 2
    legacy_status = store.artifacts.inspect(legacy)
    assert not legacy_status.is_current
    assert [entry.revision for entry in legacy_status.superseded_by] == [2]
    (line,) = [message for message in messages if "Recomputing" in message]
    assert line.startswith(
        f"Recomputing run_hto_demultiplexing: artifact {legacy.artifact_id[:12]} "
        "is revision 1, current 2: A singlet takes the one HTO"
    )
    assert store.qc.hto_demultiplexing(selection) == current
