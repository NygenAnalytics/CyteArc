"""Region-to-gene links scored by accessibility-expression association."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy as np
import pandas as pd

__all__ = ["region_gene_links"]

#: SCENIC+ region-to-gene regressor settings (``GBM_KWARGS``, seed 666).
SCENICPLUS_R2G_PARAMETERS = {
    "learning_rate": 0.01,
    "n_estimators": 500,
    "max_features": 0.1,
}


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks of each column, the ranking Spearman's rho uses."""
    from scipy.stats import rankdata

    return np.asarray(rankdata(values, axis=0), dtype=np.float64)


def _standardized(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0)
    norms = np.sqrt((centered * centered).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return centered / np.where(norms > 0, norms, np.nan)


def region_gene_links(
    accessibility: pd.DataFrame,
    expression: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    importance: Literal["gbm", "none"] = "gbm",
    seed: int = 666,
    nthreads: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Score candidate region-to-gene links over matched samples.

    For each gene, the accessibility of its candidate regions is compared with
    its expression over the same samples, such as metacells that aggregate
    both modalities of a multiome experiment. ``rho`` is Spearman's rank
    correlation of each region with the gene. With ``importance="gbm"``, a
    gradient boosting regressor with SCENIC+'s settings (500 stages, learning
    rate 0.01, a tenth of the regions per split, no subsampling) predicts the
    gene from all its candidate regions, and each region's normalized impurity
    importance is reported, as SCENIC+'s region-to-gene step does.

    Args:
        accessibility: Samples-by-regions accessibility named by region, such
            as the ATAC metacell profiles of ``ds.regulons.profiles``.
        expression: Samples-by-genes expression named by gene, over the same
            samples in the same order.
        candidates: ``region`` and ``gene`` columns, such as
            ``link_regions_to_genes`` returns.
        importance: ``"gbm"`` fits the regressor; ``"none"`` reports rho only.
        seed: Random state of each gene's regressor.
        nthreads: Genes fitted concurrently.
        progress: Optional ``callback(done, total)`` after each gene.

    Returns:
        ``region``, ``gene``, ``importance``, and ``rho`` for every candidate
        whose region and gene are both measured. Constant vectors give NaN rho.
    """
    if not accessibility.index.equals(expression.index):
        raise ValueError(
            "accessibility and expression must hold the same samples in the same order"
        )
    access = accessibility.to_numpy()
    express = expression.to_numpy()
    regions = accessibility.columns.astype(str)
    genes = expression.columns.astype(str)
    if not (regions.is_unique and genes.is_unique):
        raise ValueError("Region and gene names must be unique")
    if access.shape[0] < 3:
        raise ValueError("At least three samples are needed")
    if importance not in ("gbm", "none"):
        raise ValueError("importance must be 'gbm' or 'none'")
    if not {"region", "gene"} <= set(candidates.columns):
        raise ValueError("candidates needs region and gene columns")
    table = candidates.loc[:, ["region", "gene"]].astype(str).drop_duplicates()
    table = table.loc[table["region"].isin(regions) & table["gene"].isin(genes)]
    # Rank only the candidates' columns: ranking copies what it ranks.
    used_regions = pd.unique(table["region"])
    used_genes = pd.unique(table["gene"])
    access = access[:, regions.get_indexer(used_regions)].astype(np.float64)
    express = express[:, genes.get_indexer(used_genes)].astype(np.float64)
    region_position = {name: index for index, name in enumerate(used_regions)}
    gene_position = {name: index for index, name in enumerate(used_genes)}
    region_ranks = _standardized(_ranks(access))
    gene_ranks = _standardized(_ranks(express))
    groups = list(table.groupby("gene", sort=True))
    lock_free_total = len(groups)

    def score(item: tuple[str, pd.DataFrame]) -> pd.DataFrame:
        gene, group = item
        region_index = np.asarray([region_position[name] for name in group["region"]])
        target = gene_ranks[:, gene_position[gene]]
        rho = np.clip(region_ranks[:, region_index].T @ target, -1.0, 1.0)
        values = np.full(len(region_index), np.nan)
        if importance == "gbm":
            outcome = express[:, gene_position[gene]]
            if np.ptp(outcome) > 0:
                from sklearn.ensemble import GradientBoostingRegressor

                regressor = GradientBoostingRegressor(
                    random_state=seed, **SCENICPLUS_R2G_PARAMETERS
                )
                regressor.fit(access[:, region_index], outcome)
                values = np.asarray(regressor.feature_importances_, dtype=np.float64)
            else:
                values = np.zeros(len(region_index))
        return pd.DataFrame(
            {
                "region": group["region"].to_numpy(),
                "gene": gene,
                "importance": values,
                "rho": rho,
            }
        )

    frames: list[pd.DataFrame] = []
    if nthreads <= 1:
        for done, item in enumerate(groups, start=1):
            frames.append(score(item))
            if progress is not None:
                progress(done, lock_free_total)
    else:
        with ThreadPoolExecutor(max_workers=nthreads) as executor:
            for done, frame in enumerate(executor.map(score, groups), start=1):
                frames.append(frame)
                if progress is not None:
                    progress(done, lock_free_total)
    if not frames:
        return pd.DataFrame({"region": [], "gene": [], "importance": [], "rho": []})
    return pd.concat(frames, ignore_index=True)
