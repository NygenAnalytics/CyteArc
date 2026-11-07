import warnings
from collections.abc import Mapping, Sequence

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


def checked_hashtags(hashtags: object, hashtag_ids: Sequence[str]) -> list[str]:
    """Return the hashtags that ``hashtags`` names, in the order of ``hashtag_ids``.

    Raises:
        TypeError: If ``hashtags`` is not a sequence of texts.
        ValueError: If it names no hashtag, an empty or repeated one, or one
            that ``hashtag_ids`` lacks.
    """
    if isinstance(hashtags, str | bytes) or not isinstance(hashtags, Sequence):
        raise TypeError(
            "hashtags must be a list of hashtag IDs, such as ['HTO1', 'HTO2'], "
            f"not {type(hashtags).__name__}"
        )
    names = list(hashtags)
    if not names:
        raise ValueError("hashtags must name at least one hashtag")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("hashtags must name hashtags by non-empty IDs")
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"hashtags names {repeated} more than once")
    known = set(hashtag_ids)
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(
            f"hashtags names {unknown}, which the HTO assay does not hold; its "
            f"hashtags are {list(hashtag_ids)}"
        )
    chosen = set(names)
    return [name for name in hashtag_ids if name in chosen]


def checked_hashtag_mapping(
    hashtags: Mapping[object, object], hashtag_ids: Sequence[str]
) -> dict[str, list[str]]:
    """Return each sample's hashtags, by label, in the order of ``hashtag_ids``.

    Raises:
        TypeError: If a label is not text or a sample's hashtags are not a
            sequence of texts.
        ValueError: If a sample's hashtags are invalid as
            :func:`checked_hashtags` states; the error names the sample.
    """
    if any(not isinstance(label, str) for label in hashtags):
        raise TypeError(
            "hashtags must map each sample's label, as text, to its hashtag IDs, "
            "such as {'lane1': ['HTO1', 'HTO2']}"
        )
    checked = {}
    for label, chosen in hashtags.items():
        try:
            checked[str(label)] = checked_hashtags(chosen, hashtag_ids)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"For sample {label!r}, {exc}") from exc
    return checked


def checked_sample_hashtags(
    hashtags: object,
    hashtag_ids: Sequence[str],
    samples: Sequence[str | None],
    *,
    sample_column: str | None,
) -> tuple[list[list[str]], list[str] | dict[str, list[str]] | None]:
    """Return each sample's hashtags and the hashtags that a call records.

    ``hashtags`` is None for every hashtag of the assay, a list of hashtag IDs
    for every sample, or a mapping from each sample's label, as text, to its
    hashtag IDs, which needs ``sample_column``. Each sample's hashtags follow
    the order of ``hashtag_ids``; the record is None, the list, or the
    mapping in the order of ``samples``.

    Raises:
        TypeError: If ``hashtags`` is not a sequence of texts or a mapping of
            texts to them.
        ValueError: If a mapping comes without ``sample_column``, names
            samples that no selected cell has or leaves one out, or if a list
            is invalid as :func:`checked_hashtags` states.
    """
    if hashtags is None:
        return [list(hashtag_ids) for _ in samples], None
    if not isinstance(hashtags, Mapping):
        chosen = checked_hashtags(hashtags, hashtag_ids)
        return [list(chosen) for _ in samples], chosen
    if sample_column is None:
        raise ValueError(
            "hashtags maps samples to their hashtags, which needs sample_column, "
            "the cell metadata column of each cell's sample"
        )
    checked = checked_hashtag_mapping(hashtags, hashtag_ids)
    labels = [str(sample) for sample in samples]
    extra = sorted(set(checked) - set(labels))
    if extra:
        raise ValueError(
            f"hashtags names samples {extra} that no selected cell has in "
            f"{sample_column!r}; the selected cells' samples are {labels}"
        )
    missing = [label for label in labels if label not in checked]
    if missing:
        raise ValueError(
            f"hashtags gives no hashtags for samples {missing} of "
            f"{sample_column!r}; give each sample of the selected cells its "
            "hashtags"
        )
    chosen_by_sample = [checked[label] for label in labels]
    return chosen_by_sample, dict(zip(labels, chosen_by_sample, strict=True))


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


def _hto_cutoffs(
    hto_counts: pd.DataFrame,
    cluster_labels: np.ndarray,
    *,
    positive_quantile: float = _DEFAULT_POSITIVE_QUANTILE,
) -> tuple[dict[str, int], dict[str, float]]:
    """Return each HTO's cutoff and the strict cutoff of its background fit.

    A background cluster that holds only zero counts has no
    negative-binomial fit, and one whose fit has a tiny mean gives a cutoff
    of 0 too; a cutoff of 0 calls every cell with one count positive. That is
    right only for counts without ambient tags. Ambient tags reach most
    cells: most cells of a background cluster hold counts of its HTO, or most
    cells hold counts of more than one HTO, where without them only doublets
    do. When the counts carry ambient tags, an HTO with a cutoff of 0 is
    unused in the cells of its background, such as the cells of a pool that
    did not use it, and every such HTO is refused in one error.

    Raises:
        ValueError: If a background cluster holds fewer than two cells, a fit
            is invalid, or an HTO has a cutoff of 0 in counts that carry
            ambient tags.
    """
    background_clusters = _background_clusters(hto_counts, cluster_labels)
    cutoffs: dict[str, int] = {}
    strict_cutoffs: dict[str, float] = {}
    # The HTOs whose background cluster holds counts of them in most cells.
    ambient_backgrounds: list[str] = []
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
        if 2 * np.count_nonzero(background) > len(background):
            ambient_backgrounds.append(str(hto_name))
        if np.any(background > 0):
            mu, alpha = _fit_negative_binomial_parameters(background, hto_name)
            try:
                cutoff = _negative_binomial_cutoff(mu, alpha, positive_quantile)
            except ValueError as exc:
                raise ValueError(
                    f"Negative-binomial cutoff is invalid for HTO {hto_name!r}"
                ) from exc
            strict_cutoff = _strict_background_cutoff(mu, alpha, positive_quantile)
        else:
            cutoff, strict_cutoff = 0, 0.0
        cutoffs[hto_name] = cutoff
        strict_cutoffs[hto_name] = strict_cutoff
    unfit = [str(name) for name, cutoff in cutoffs.items() if cutoff == 0]
    if unfit:
        _refuse_zero_cutoffs_in_ambient_counts(hto_counts, unfit, ambient_backgrounds)
    return cutoffs, strict_cutoffs


def _refuse_zero_cutoffs_in_ambient_counts(
    hto_counts: pd.DataFrame, unfit: list[str], ambient_backgrounds: list[str]
) -> None:
    """Refuse the HTOs with a cutoff of 0 when the counts carry ambient tags.

    Ambient tags show as ``ambient_backgrounds``, the HTOs whose background
    cluster holds counts of them in most cells, or as most cells holding
    counts of more than one HTO. Without them, a cell holds counts of its own
    tags only, and doublets, the cells with two, are a minority.
    """
    counts = hto_counts.to_numpy()
    n_cells = counts.shape[0]
    several = int(np.count_nonzero(np.count_nonzero(counts > 0, axis=1) > 1))
    if not ambient_backgrounds and 2 * several <= n_cells:
        return
    evidence = []
    if ambient_backgrounds:
        names = ", ".join(repr(name) for name in ambient_backgrounds)
        evidence.append(
            f"most cells of the background cluster of {names} hold counts of it"
            if len(ambient_backgrounds) == 1
            else f"most cells of the background clusters of {names} hold counts "
            "of their hashtags"
        )
    if 2 * several > n_cells:
        evidence.append(
            f"{several} of the {n_cells} cells hold counts of more than one hashtag"
        )
    raise ValueError(
        "HTO demultiplexing cannot place a cutoff for "
        + ", ".join(repr(name) for name in unfit)
        + ": "
        + (
            "its background cluster gives"
            if len(unfit) == 1
            else "the background cluster of each gives"
        )
        + " a cutoff of 0, which calls every cell with one count of it positive. "
        "That is right only for counts without ambient tags, while here "
        + " and ".join(evidence)
        + ". Such a hashtag is unused, or nearly so, in the cells of its "
        "background, as in a pool of cells that did not use it. Leave these "
        "hashtags out, or demultiplex each pool of cells with its own hashtags."
    )


def _positive_hto_calls(
    hto_counts: pd.DataFrame,
    cluster_labels: np.ndarray,
    *,
    positive_quantile: float = _DEFAULT_POSITIVE_QUANTILE,
) -> pd.DataFrame:
    """Call each count positive when it exceeds its HTO's background cutoff.

    The cutoffs are those of :func:`_hto_cutoffs`. A warning names the HTOs
    whose positive calls overlap their background (see
    ``_warn_unseparated_htos``).
    """
    return _calls_and_cutoffs(
        hto_counts, cluster_labels, positive_quantile=positive_quantile
    )[0]


def _calls_and_cutoffs(
    hto_counts: pd.DataFrame,
    cluster_labels: np.ndarray,
    *,
    positive_quantile: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return the positive calls of :func:`_positive_hto_calls` and the cutoffs."""
    cutoffs, strict_cutoffs = _hto_cutoffs(
        hto_counts, cluster_labels, positive_quantile=positive_quantile
    )
    positive_calls = hto_counts > pd.Series(cutoffs)
    _warn_unseparated_htos(
        hto_counts, positive_calls, pd.Series(strict_cutoffs), positive_quantile
    )
    return positive_calls, cutoffs


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

    A background cluster that holds only zero counts, or whose fit gives a
    cutoff of 0, calls every nonzero count of that HTO positive. That is
    accepted only for counts without ambient tags, in which no background
    cluster holds counts of its HTO in most of its cells and at most half of
    the cells hold counts of more than one HTO, as doublets do. Otherwise the
    HTO is unused in the cells of its background, as in a pool that did not
    use it, and the call is refused, naming each such HTO. A warning also
    names each HTO whose positive calls overlap its background: fewer than
    half of its positive cells have counts that the background fit exceeds
    with probability ``(1 - positive_quantile) ** 3``. Such calls, for
    example of an unused tag or of pure background, may be background.

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
    return _hto_calls(
        _validated_hto_counts(hto_counts),
        random_seed=random_seed,
        positive_quantile=quantile,
    )[0]


def _hto_calls(
    counts: pd.DataFrame,
    *,
    random_seed: int,
    positive_quantile: float,
) -> tuple[pd.Series, dict[str, int]]:
    """Return the identities that :func:`hto_demux` gives validated counts, and each HTO's cutoff.

    ``counts`` are raw counts that ``_validated_hto_counts`` has checked.
    """
    cluster_labels = _cluster_labels(_clr_normalize(counts), random_seed)
    positive_calls, cutoffs = _calls_and_cutoffs(
        counts, cluster_labels, positive_quantile=positive_quantile
    )
    return _classify_hto_identities(positive_calls), cutoffs
