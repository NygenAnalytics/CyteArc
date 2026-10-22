from typing import Any

import numpy as np
import pandas as pd

__all__ = ["binned_sampling"]


def binned_sampling(
    values: pd.Series,
    feature_list: list[str],
    ctrl_size: int,
    n_bins: int,
    rand_seed: int,
) -> list[str]:
    """Sample the control features of a gene-set score by expression bin.

    A gene-set score is the mean expression of the set's features minus the
    mean of these controls. The binning and the draw are those of scanpy's
    ``score_genes``, from which this function is adapted; they are not
    Seurat's ``AddModuleScore``:

    - With ``N`` features, a bin holds ``k = round(N / (n_bins - 1))``
      features. Features are ranked by value, NaN counting as 0, with tied
      values sharing the lowest of their ranks, and a feature of rank ``r``
      (from 1) falls in bin ``r // k``. Bin 0 holds ``k - 1`` ranks and the
      top bin the remainder, so there are about ``n_bins`` bins.
    - Tied values share one bin, however many there are. Every feature with
      the lowest value, such as a mean of 0, shares the lowest bin, which on
      sparse data can hold thousands of features and leave the next bins
      empty.
    - For each bin that holds a feature of the set, in increasing order,
      ``ctrl_size`` features are drawn without replacement from all of the
      bin's features, or the whole bin when it holds no more. One
      ``RandomState(rand_seed)`` makes the draws, and drawn features of the
      set are then dropped. Controls are drawn once per bin, so a bin with
      several features of the set gives ``ctrl_size`` controls in all,
      whereas Tirosh et al. (2016) draw controls for each gene of the set.

    Args:
        values: The values for the features, such as their mean expression.
        feature_list: The features of the set; a repeated feature counts once.
        ctrl_size: Number of reference features to be sampled from each bin.
        n_bins: Number of bins for sampling.
        rand_seed: The seed to use for the random number generation.

    Returns:
        The control features, in the order of ``values``, without the
        features of the set.
    """
    if isinstance(ctrl_size, (bool, np.bool_)) or not isinstance(
        ctrl_size, (int, np.integer)
    ):
        raise TypeError("ctrl_size must be a positive integer")
    if ctrl_size < 1:
        raise ValueError("ctrl_size must be a positive integer")
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(
        n_bins, (int, np.integer)
    ):
        raise TypeError("n_bins must be an integer greater than one")
    if n_bins < 2:
        raise ValueError("n_bins must be greater than one")
    n_items = int(np.round(len(values) / (n_bins - 1)))
    if n_items < 1:
        raise ValueError("n_bins is too large for the number of available features")
    rng = np.random.RandomState(rand_seed)
    feature_set = set(feature_list)
    obs_cut: pd.Series = values.fillna(0).rank(method="min").divide(n_items).astype(int)

    control_genes: set[Any] = set()
    for cut in np.unique(obs_cut[list(feature_set)]):
        sub_obs = obs_cut[obs_cut == cut]
        r_genes = (
            sub_obs.sample(n=ctrl_size, random_state=rng).index
            if len(sub_obs) > ctrl_size
            else sub_obs.index
        )
        control_genes.update(set(r_genes))
    return list(values.index[values.index.isin(control_genes - feature_set)])
