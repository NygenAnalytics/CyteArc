import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from ..assay.normalization import clr_values
from ..utils.arguments import float_argument
from ..utils.warnings import warn

__all__ = ["hto_demux"]

_NORMALIZATION_METHOD = "clr_per_hto"
_CLUSTERING_METHOD = "kmeans"
_KMEANS_INIT = "random"
_KMEANS_N_STARTS = 100
_CLUSTER_COUNT_RULE = "n_htos_plus_one"
_BACKGROUND_STATISTIC = "raw_mean"
_CUTOFF_DISTRIBUTION = "negative_binomial_nb2"
# Seurat's default positive.quantile.
_DEFAULT_POSITIVE_QUANTILE = 0.99
_CUTOFF_LOCATION = 0
_CUTOFF_COMPARISON = "strictly_greater"
# The method record keeps the singlet rule that revision 1 recorded, so stored
# artifacts of revision 1 are superseded matches of revision 2, which assigns
# each singlet the one tag called positive (see run_hto_demultiplexing in
# storage/operation_revisions.py). The CLR argmax is now a stored diagnostic.
_SINGLET_ASSIGNMENT = "clr_argmax"
_RESERVED_IDENTITIES = frozenset({"Negative", "Singlet", "Doublet"})
# A positive population is separated from its background when at least this
# fraction of the cells called positive have counts that the background
# distribution exceeds with probability (1 - positive_quantile) ** 3, one in a
# million for the default quantile.
_SEPARATED_POSITIVE_FRACTION = 0.5
_STRICT_TAIL_POWER = 3


def _validated_positive_quantile(value: object) -> float:
    """Return ``positive_quantile`` as a Python float between 0 and 1.

    Raises:
        TypeError: If ``value`` is a boolean or not a real number.
        ValueError: If ``value`` is not finite or not strictly between 0 and 1.
    """
    quantile = float_argument(value, "positive_quantile")
    if not 0.0 < quantile < 1.0:
        raise ValueError("positive_quantile must be greater than 0 and less than 1")
    return quantile


def _hto_demux_method(positive_quantile: float) -> dict[str, object]:
    return {
        "normalization": _NORMALIZATION_METHOD,
        "clustering": {
            "method": _CLUSTERING_METHOD,
            "init": _KMEANS_INIT,
            "n_starts": _KMEANS_N_STARTS,
            "cluster_count": _CLUSTER_COUNT_RULE,
        },
        "background": _BACKGROUND_STATISTIC,
        "cutoff": {
            "distribution": _CUTOFF_DISTRIBUTION,
            # Artifacts recorded the fixed 0.99 here before the quantile could
            # be chosen, so artifacts of the default keep their identities.
            "quantile": positive_quantile,
            "location": _CUTOFF_LOCATION,
            "comparison": _CUTOFF_COMPARISON,
        },
        "singlet_assignment": _SINGLET_ASSIGNMENT,
    }


def _validated_hto_counts(hto_counts: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(hto_counts, pd.DataFrame):
        raise TypeError("hto_counts must be a pandas DataFrame")
    if hto_counts.shape[1] == 0:
        raise ValueError("hto_counts must contain at least one HTO")
    if not hto_counts.index.is_unique:
        raise ValueError("hto_counts cell index must be unique")
    if not hto_counts.columns.is_unique:
        raise ValueError("HTO IDs must be unique")

    hto_names = hto_counts.columns.tolist()
    if any(not isinstance(name, str) or not name.strip() for name in hto_names):
        raise ValueError("HTO IDs must be non-empty strings")
    reserved = sorted(_RESERVED_IDENTITIES.intersection(hto_names))
    if reserved:
        raise ValueError(
            "HTO IDs conflict with reserved identity labels: "
            + ", ".join(repr(name) for name in reserved)
        )
    if not all(is_numeric_dtype(dtype) for dtype in hto_counts.dtypes):
        raise TypeError("hto_counts must contain only numeric raw counts")

    raw_counts = hto_counts.to_numpy()
    if np.iscomplexobj(raw_counts):
        raise TypeError("hto_counts must contain only real numeric raw counts")
    counts = raw_counts.astype(float)
    if not np.all(np.isfinite(counts)):
        raise ValueError("hto_counts must contain only finite raw counts")
    if np.any(counts < 0):
        raise ValueError("hto_counts must contain only nonnegative raw counts")
    if np.any(counts != np.floor(counts)):
        raise ValueError("hto_counts must contain integer-valued raw counts")

    required_cells = hto_counts.shape[1] + 1
    if hto_counts.shape[0] < required_cells:
        raise ValueError(
            f"HTO demultiplexing requires at least {required_cells} selected cells"
        )

    empty_htos = [
        hto_names[index]
        for index in np.flatnonzero(np.all(counts == 0, axis=0)).tolist()
    ]
    if empty_htos:
        raise ValueError(
            "HTOs with no positive counts cannot be demultiplexed: "
            + ", ".join(repr(name) for name in empty_htos)
        )
    return pd.DataFrame(counts, index=hto_counts.index, columns=hto_counts.columns)


def _clr_normalize(hto_counts: pd.DataFrame) -> pd.DataFrame:
    normalized = pd.DataFrame(
        clr_values(hto_counts.to_numpy()),
        index=hto_counts.index,
        columns=hto_counts.columns,
    )
    return normalized


def _cluster_labels(normalized: pd.DataFrame, random_seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    n_centers = normalized.shape[1] + 1
    unique_profiles = np.unique(normalized.to_numpy(), axis=0).shape[0]
    if unique_profiles < n_centers:
        raise ValueError(
            "HTO demultiplexing requires at least "
            f"{n_centers} distinct normalized cell profiles; found {unique_profiles}"
        )
    labels = np.asarray(
        KMeans(
            n_clusters=n_centers,
            init=_KMEANS_INIT,
            n_init=_KMEANS_N_STARTS,
            random_state=random_seed,
        ).fit_predict(normalized)
    )
    occupied = np.unique(labels).size
    if occupied != n_centers:
        raise ValueError(
            f"HTO clustering produced {occupied} occupied clusters; "
            f"expected {n_centers}"
        )
    return labels


def _background_clusters(
    hto_counts: pd.DataFrame,
    cluster_labels: np.ndarray,
) -> pd.Series:
    cluster_means = hto_counts.groupby(cluster_labels, sort=True).mean()
    return cluster_means.idxmin(axis=0)


def _fit_negative_binomial_parameters(
    values: np.ndarray,
    hto_name: str,
) -> tuple[float, float]:
    from statsmodels.discrete.discrete_model import NegativeBinomial

    exog = np.ones((len(values), 1), dtype=float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = NegativeBinomial(
                values,
                exog,
                loglike_method="nb2",
            ).fit(
                start_params=np.asarray([1.0, 1.0]),
                disp=0,
                full_output=True,
            )
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Negative-binomial background fit failed for HTO {hto_name!r}"
        ) from exc

    fit_details = getattr(fit, "mle_retvals", None)
    convergence = (
        fit_details.get("converged") if isinstance(fit_details, Mapping) else None
    )
    if not isinstance(convergence, bool | np.bool_) or not bool(convergence):
        raise ValueError(
            f"Negative-binomial background fit did not converge for HTO {hto_name!r}"
        )
    try:
        parameters = np.asarray(fit.params, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Negative-binomial background fit returned invalid parameters "
            f"for HTO {hto_name!r}"
        ) from exc
    if parameters.size != 2:
        raise ValueError(
            f"Negative-binomial background fit returned invalid parameters "
            f"for HTO {hto_name!r}"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        mu = float(np.exp(parameters[0]))
    alpha = float(parameters[-1])
    if not np.isfinite(mu) or mu <= 0:
        raise ValueError(
            f"Negative-binomial background fit returned invalid mean "
            f"for HTO {hto_name!r}"
        )
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError(
            f"Negative-binomial background fit returned invalid dispersion "
            f"for HTO {hto_name!r}"
        )
    return mu, alpha


def _negative_binomial_cutoff(
    mu: float,
    alpha: float,
    quantile: float = _DEFAULT_POSITIVE_QUANTILE,
) -> int:
    from scipy.stats import nbinom

    if not np.isfinite(mu) or mu <= 0:
        raise ValueError("Negative-binomial mean must be finite and greater than 0")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError(
            "Negative-binomial dispersion must be finite and greater than 0"
        )
    if not np.isfinite(quantile) or not 0 < quantile < 1:
        raise ValueError("Negative-binomial quantile must be between 0 and 1")

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        size = 1 / alpha
        probability = 1 / (1 + alpha * mu)
    if (
        not np.isfinite(size)
        or size <= 0
        or not np.isfinite(probability)
        or not 0 < probability <= 1
    ):
        raise ValueError(
            "Negative-binomial parameters produced an invalid distribution"
        )

    cutoff = float(nbinom.ppf(quantile, n=size, p=probability))
    if not np.isfinite(cutoff) or not cutoff.is_integer():
        raise ValueError("Negative-binomial cutoff must be a finite integer")
    return int(cutoff)


def _strict_background_cutoff(mu: float, alpha: float, quantile: float) -> float:
    """Return the count that a valid background fit exceeds very rarely.

    The background exceeds it with probability ``(1 - quantile) ** 3``, one in
    a million for the default quantile. The tail is floored at the smallest
    normal float, so a quantile near 1 still gives a finite tail.
    """
    from scipy.stats import nbinom

    tail = max((1.0 - quantile) ** _STRICT_TAIL_POWER, np.finfo(float).tiny)
    return float(nbinom.isf(tail, n=1 / alpha, p=1 / (1 + alpha * mu)))


def _warn_unseparated_htos(
    hto_counts: pd.DataFrame,
    positive_calls: pd.DataFrame,
    strict_cutoffs: pd.Series,
    positive_quantile: float,
) -> None:
    """Warn about HTOs whose positive calls overlap their background.

    An HTO with a stained population has most of its positive cells far above
    the background, beyond the strict cutoff. When fewer than half of them
    are, its positive and background counts overlap: the calls follow the
    tail of the background, as in pure background or an unused tag.
    """
    n_positive = positive_calls.sum(axis=0)
    # Each strict cutoff is at least its cutoff, so strong counts are positive.
    n_strong = (hto_counts > strict_cutoffs).sum(axis=0)
    overlapping = [
        f"{name!r} ({int(n_strong[name])} of {int(n_positive[name])} positive cells)"
        for name in hto_counts.columns
        if n_positive[name] > 0
        and n_strong[name] < _SEPARATED_POSITIVE_FRACTION * n_positive[name]
    ]
    if overlapping:
        tail = (1.0 - positive_quantile) ** _STRICT_TAIL_POWER
        warn(
            "Positive and background HTO counts overlap for "
            + ", ".join(overlapping)
            + ": fewer than half of the cells called positive have counts that "
            f"the background fit exceeds with probability {tail:.3g}, so these "
            "calls may be background, as in an unused tag or pure background. "
            "Check the HTO count distributions; a tag without a separated "
            "positive population cannot be demultiplexed."
        )


def _positive_hto_calls(
    hto_counts: pd.DataFrame,
    cluster_labels: np.ndarray,
    *,
    positive_quantile: float = _DEFAULT_POSITIVE_QUANTILE,
) -> pd.DataFrame:
    """Call each count positive when it exceeds its HTO's background cutoff.

    A background cluster that holds only zero counts has no negative-binomial
    fit; its cutoff is 0, so every nonzero count of that HTO is positive, and
    a warning says so. A warning also names the HTOs whose positive calls
    overlap their background (see ``_warn_unseparated_htos``).
    """
    background_clusters = _background_clusters(hto_counts, cluster_labels)
    cutoffs: dict[str, int] = {}
    strict_cutoffs: dict[str, float] = {}
    for hto_name in hto_counts.columns:
        background = hto_counts.loc[
            cluster_labels == background_clusters[hto_name],
            hto_name,
        ].to_numpy()
        if len(background) < 2:
            raise ValueError(
                f"Background cluster for HTO {hto_name!r} must contain "
                "at least two cells"
            )
        if not np.any(background > 0):
            warn(
                f"Background cluster for HTO {hto_name!r} contains only zero "
                "counts, so its cutoff is 0 and every nonzero count of this HTO "
                "is called positive, including a few background counts in cells "
                "outside that cluster."
            )
            cutoffs[hto_name] = 0
            strict_cutoffs[hto_name] = 0.0
            continue
        mu, alpha = _fit_negative_binomial_parameters(background, hto_name)
        try:
            cutoffs[hto_name] = _negative_binomial_cutoff(mu, alpha, positive_quantile)
        except ValueError as exc:
            raise ValueError(
                f"Negative-binomial cutoff is invalid for HTO {hto_name!r}"
            ) from exc
        strict_cutoffs[hto_name] = _strict_background_cutoff(
            mu, alpha, positive_quantile
        )
    positive_calls = hto_counts > pd.Series(cutoffs)
    _warn_unseparated_htos(
        hto_counts, positive_calls, pd.Series(strict_cutoffs), positive_quantile
    )
    return positive_calls


def _classify_hto_identities(positive_calls: pd.DataFrame) -> pd.Series:
    """Return each cell's identity from its positive calls.

    A cell with no positive HTO is ``Negative``, one with several is
    ``Doublet``, and a singlet takes the one HTO called positive.
    """
    positive_count = positive_calls.sum(axis=1)
    identities = positive_count.map(
        lambda count: (
            "Negative" if count == 0 else "Singlet" if count == 1 else "Doublet"
        )
    )
    singlets = positive_count == 1
    identities.loc[singlets] = positive_calls.loc[singlets].idxmax(axis=1)
    return identities


def _hto_diagnostics(normalized: pd.DataFrame) -> pd.DataFrame:
    """Return each cell's two highest CLR tags and the margin between them.

    ``max_id`` and ``second_id`` name the HTOs with the highest and the second
    highest CLR value, ties going to the earlier HTO, and ``margin`` is the
    difference of their values, as Seurat reports ``maxID``, ``secondID``,
    and ``margin``. With one HTO, ``second_id`` is empty and ``margin`` NaN.
    """
    values = normalized.to_numpy()
    names = normalized.columns.to_numpy(dtype=object)
    rows = np.arange(values.shape[0])
    order = np.argsort(-values, axis=1, kind="stable")
    first = order[:, 0]
    if values.shape[1] > 1:
        second = order[:, 1]
        second_id = names[second]
        margin = values[rows, first] - values[rows, second]
    else:
        second_id = np.full(values.shape[0], "", dtype=object)
        margin = np.full(values.shape[0], np.nan)
    return pd.DataFrame(
        {"max_id": names[first], "second_id": second_id, "margin": margin},
        index=normalized.index,
    )


def _hto_clr_diagnostics(hto_counts: pd.DataFrame) -> pd.DataFrame:
    """Return the CLR diagnostics of ``_hto_diagnostics`` for raw HTO counts.

    The counts are validated and normalized as ``hto_demux`` does, and the
    diagnostics depend on nothing else, so they match its identities.
    """
    return _hto_diagnostics(_clr_normalize(_validated_hto_counts(hto_counts)))


def hto_demux(
    hto_counts: pd.DataFrame,
    *,
    random_seed: int = 0,
    positive_quantile: float = 0.99,
) -> pd.Series:
    """Assigns HTO identity to each cell based on the HTO count distribution.
    The algorithm is adapted from the Seurat package's HTOdemux function [Satija15]_.

    A count is positive when it exceeds the ``positive_quantile`` quantile of
    the negative-binomial distribution fitted to its HTO's background cluster.
    A cell with no positive HTO is ``Negative``, one with several is
    ``Doublet``, and a singlet takes the one HTO called positive. Seurat's
    ``HTODemux`` instead names a singlet after its highest normalized (CLR)
    value, which in unbalanced pools can be an HTO that the cutoff called
    negative.

    A background cluster that holds only zero counts gets a cutoff of 0, so
    every nonzero count of that HTO is positive, with a warning. A warning
    also names each HTO whose positive calls overlap its background: fewer
    than half of its positive cells have counts that the background fit
    exceeds with probability ``(1 - positive_quantile) ** 3``. Such calls,
    for example of an unused tag or of pure background, may be background.

    Args:
        hto_counts: A dataframe containing the raw HTO counts for each cell.
        random_seed: Seed of the k-means clustering.
        positive_quantile: Quantile of each HTO's background distribution that
            a count must exceed to be positive, Seurat's ``positive.quantile``.
            It must be greater than 0 and less than 1. A higher quantile, such
            as 0.999, calls fewer background counts positive, and so fewer
            false doublets, but can miss weakly stained tags.

    Returns:
        A series containing the HTO identity for each cell.

    Raises:
        TypeError: If ``random_seed`` is not an integer or ``positive_quantile``
            is not a real number.
        ValueError: If ``positive_quantile`` is not finite or not between 0 and
            1, or if the counts cannot be demultiplexed.
    """
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise TypeError("random_seed must be an integer")
    quantile = _validated_positive_quantile(positive_quantile)
    counts = _validated_hto_counts(hto_counts)
    normalized = _clr_normalize(counts)
    cluster_labels = _cluster_labels(normalized, random_seed)
    positive_calls = _positive_hto_calls(
        counts, cluster_labels, positive_quantile=quantile
    )
    return _classify_hto_identities(positive_calls)
