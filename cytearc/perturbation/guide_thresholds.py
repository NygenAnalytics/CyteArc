"""Fitted guide thresholds of pooled CRISPR screens, as tables.

``ds.perturbation.call_guides`` can fit a model to the UMI counts of every guide
in every sample and save the fits as a ``guide_thresholds`` artifact, beside
each guide's histogram of counts. These functions hold the methods' settings
and turn the stored arrays into tables; they never touch storage.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import pandas as pd

from cytearc.perturbation.histograms import MAX_COUNT
from cytearc.perturbation.mixtures import FIT_STATUSES
from cytearc.utils.arguments import (
    float_argument,
    integer_argument,
    optional_name_argument,
)

__all__ = [
    "DEFAULT_FRACTION_THRESHOLD",
    "DEFAULT_MIN_UMI",
    "FITTED_METHODS",
    "METHODS",
    "METHOD_PARAMETERS",
    "GuideThresholds",
    "call_settings",
    "call_thresholds",
    "guide_threshold_frames",
]

#: The guide-calling methods of ``ds.perturbation.call_guides``.
METHODS = ("poisson_gaussian", "gaussian", "fixed", "top_guide")
#: The methods that fit a threshold to each guide in each sample.
FITTED_METHODS = ("poisson_gaussian", "gaussian")
#: The UMI floor of each method when ``min_umi`` is not given; ``fixed``
#: needs one.
DEFAULT_MIN_UMI: Mapping[str, int] = MappingProxyType(
    {"poisson_gaussian": 1, "gaussian": 3, "top_guide": 5}
)
#: The share of a cell's guide UMIs that the top guide must exceed, by default.
DEFAULT_FRACTION_THRESHOLD = 0.8
#: The fitted parameters of each fitted method, as stored arrays.
METHOD_PARAMETERS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "poisson_gaussian": (
            "poisson_rate",
            "gaussian_mean",
            "gaussian_std",
            "gaussian_weight",
            "loss",
        ),
        "gaussian": ("low_mean", "high_mean", "std", "high_weight", "log_likelihood"),
    }
)


@dataclass(frozen=True, slots=True)
class GuideThresholds:
    """The fitted UMI thresholds of every guide in every sample.

    Attributes:
        table: One row per sample and guide, with ``sample`` (None without a
            sample column), ``guide``, ``status`` (one of ``"fitted"``,
            ``"no_cells"``, ``"too_sparse"``, ``"no_fit"`` and
            ``"no_positive"``), ``threshold`` (the smallest count the model
            calls, 0 where the guide calls nothing), ``call_threshold`` (the
            smallest count called once ``min_umi`` applies, missing when the
            thresholds were read without calls), ``n_cells`` (the selected
            cells of the sample), ``n_umi_cells`` (those with a UMI of the
            guide), ``max_umi``, ``iterations``, and the method's fitted
            parameters.
        histograms: One row per sample, guide and UMI count that some cell
            holds, zero included, with ``sample``, ``guide``, ``umis`` and
            ``cells``.
        guides: The ID of every guide of the assay, in assay order.
        method: ``"poisson_gaussian"`` or ``"gaussian"``.
        sample_column: The cell metadata column whose labels are the samples,
            or None for one sample of every selected cell.
        min_umi: The UMI floor of the calls the thresholds were read through,
            or None when they were read on their own.
    """

    table: pd.DataFrame = field(repr=False)
    histograms: pd.DataFrame = field(repr=False)
    guides: pd.Index = field(repr=False)
    method: str
    sample_column: str | None
    min_umi: int | None


def call_settings(
    method: object,
    min_umi: object,
    fraction_threshold: object,
    sample_column: object,
) -> tuple[int, float]:
    """Check the settings of guide calling, as ``call_guides`` takes them.

    The pipeline recipe checks the same settings before its run starts.

    Args:
        method: One of :data:`METHODS`.
        min_umi: The fewest UMIs of a called guide, or None for the method's
            default.
        fraction_threshold: The share of a cell's guide UMIs that its top
            guide must exceed, for ``"top_guide"`` only, or None.
        sample_column: The name of the column whose samples are fitted
            apart, for the fitted methods only, or None. Whether the column
            exists is not checked here.

    Returns:
        The UMI floor and the share that the calls use.

    Raises:
        TypeError: If a setting has the wrong type.
        ValueError: If a setting is out of range or does not apply to the
            method.
    """
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError(
            "method must be 'poisson_gaussian', 'gaussian', 'fixed' or "
            f"'top_guide', not {method!r}"
        )
    if min_umi is not None:
        resolved_min_umi = integer_argument(
            min_umi, "min_umi", minimum=1, maximum=MAX_COUNT
        )
    elif method == "fixed":
        raise ValueError(
            "method='fixed' needs min_umi, the UMI count from which every "
            "guide is called"
        )
    else:
        resolved_min_umi = DEFAULT_MIN_UMI[method]
    fraction = DEFAULT_FRACTION_THRESHOLD
    if method != "top_guide" and fraction_threshold is not None:
        raise ValueError("fraction_threshold applies only to method='top_guide'")
    if fraction_threshold is not None:
        fraction = float_argument(fraction_threshold, "fraction_threshold")
        if not 0.0 <= fraction < 1.0:
            raise ValueError("fraction_threshold must be at least 0 and below 1")
    if sample_column is not None and method not in FITTED_METHODS:
        raise ValueError(
            "sample_column applies only to method='poisson_gaussian' and "
            "method='gaussian', which fit each sample alone"
        )
    optional_name_argument(sample_column, "sample_column")
    return resolved_min_umi, fraction


def call_thresholds(threshold: np.ndarray, min_umi: int) -> np.ndarray:
    """Return the smallest called count of each guide once ``min_umi`` applies.

    A fitted threshold below ``min_umi`` is raised to it, and a guide whose
    threshold is 0, which calls nothing, keeps 0.
    """
    fitted = np.asarray(threshold, dtype=np.int64)
    called: np.ndarray = np.where(fitted > 0, np.maximum(fitted, min_umi), 0)
    return called


def guide_threshold_frames(
    *,
    samples: list[str | None],
    guides: np.ndarray,
    n_cells: np.ndarray,
    status: np.ndarray,
    threshold: np.ndarray,
    n_umi_cells: np.ndarray,
    max_umi: np.ndarray,
    iterations: np.ndarray,
    parameters: Mapping[str, np.ndarray],
    histogram_offsets: np.ndarray,
    histogram_counts: np.ndarray,
    histogram_cells: np.ndarray,
    min_umi: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the table of fits and the table of histograms.

    Every per-fit array is samples x guides, and histogram entry
    ``s * len(guides) + g`` holds the nonzero counts of guide ``g`` in sample
    ``s`` and the cells that hold them, in ascending order.
    """
    n_samples, n_guides = np.shape(status)
    sample_of = np.repeat(np.arange(n_samples), n_guides)
    guide_of = np.tile(np.arange(n_guides), n_samples)
    sample_labels = np.asarray(samples, dtype=object)
    guide_labels = np.asarray(guides, dtype=object)
    called = (
        pd.array(call_thresholds(threshold, min_umi).reshape(-1), dtype="Int64")
        if min_umi is not None
        else pd.array([pd.NA] * (n_samples * n_guides), dtype="Int64")
    )
    table = pd.DataFrame(
        {
            "sample": sample_labels[sample_of],
            "guide": guide_labels[guide_of],
            "status": np.asarray(FIT_STATUSES, dtype=object)[
                np.asarray(status).reshape(-1)
            ],
            "threshold": np.asarray(threshold, dtype=np.int64).reshape(-1),
            "call_threshold": called,
            "n_cells": np.asarray(n_cells, dtype=np.int64)[sample_of],
            "n_umi_cells": np.asarray(n_umi_cells, dtype=np.int64).reshape(-1),
            "max_umi": np.asarray(max_umi, dtype=np.int64).reshape(-1),
            "iterations": np.asarray(iterations, dtype=np.int64).reshape(-1),
            **{
                name: np.asarray(values, dtype=np.float64).reshape(-1)
                for name, values in parameters.items()
            },
        }
    )
    entries = np.repeat(np.arange(n_samples * n_guides), np.diff(histogram_offsets))
    zero_cells = table["n_cells"].to_numpy() - table["n_umi_cells"].to_numpy()
    zero_entries = np.flatnonzero(zero_cells > 0)
    all_entries = np.concatenate([zero_entries, entries])
    umis = np.concatenate(
        [np.zeros(len(zero_entries), dtype=np.int64), np.asarray(histogram_counts)]
    ).astype(np.int64)
    cells = np.concatenate(
        [zero_cells[zero_entries], np.asarray(histogram_cells)]
    ).astype(np.int64)
    order = np.lexsort((umis, all_entries))
    all_entries = all_entries[order]
    histograms = pd.DataFrame(
        {
            "sample": sample_labels[sample_of[all_entries]],
            "guide": guide_labels[guide_of[all_entries]],
            "umis": umis[order],
            "cells": cells[order],
        }
    )
    return table, histograms
