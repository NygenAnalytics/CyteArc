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
    """Score a set of genes [Satija15]_. The score is the average expression of
    a set of genes subtracted with the average expression of a reference set of
    genes. The reference set is randomly sampled from the `gene_pool` for each
    binned expression value.

    This reproduces the approach in Seurat [Satija15]_ and has been implemented
    for Scanpy by Davide Cittaro.

    This function is adapted from Scanpy's `score_genes`.

    Args:
        values: The values for the features.
        feature_list: The list of features to use for score calculation.
        ctrl_size: Number of reference features to be sampled from each bin.
        n_bins: Number of bins for sampling.
        rand_seed: The seed to use for the random number generation.

    Returns:
        A list of sampled features.
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
        if len(sub_obs) == 0:
            continue
        r_genes = (
            sub_obs.sample(n=ctrl_size, random_state=rng).index
            if len(sub_obs) > ctrl_size
            else sub_obs.index
        )
        control_genes.update(set(r_genes))
    return list(values.index[values.index.isin(control_genes - feature_set)])
