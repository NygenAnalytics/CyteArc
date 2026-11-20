from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..utils.arguments import (
    RANDOM_STATE_MAX,
    integer_argument,
    optional_bool_argument,
)
from .cell_cycle_genes import g2m_phase_genes, s_phase_genes

__all__ = ["CellCycleSettings", "assign_cell_cycle_phase", "cell_cycle_settings"]


@dataclass(frozen=True, slots=True)
class CellCycleSettings:
    """The settings of cell cycle scoring, as Python values.

    Each attribute is the ``scores.cell_cycle`` keyword of the same name,
    with the default gene lists and control size filled in. ``log_transform``
    stays None when not given, so that the assay's normalizer decides it.
    """

    s_genes: Sequence[str]
    g2m_genes: Sequence[str]
    ctrl_size: int
    log_transform: bool | None
    n_bins: int
    rand_seed: int


def cell_cycle_settings(
    *,
    s_genes: Any,
    g2m_genes: Any,
    ctrl_size: object,
    log_transform: object,
    n_bins: object,
    rand_seed: object,
) -> CellCycleSettings:
    """Check the settings of cell cycle scoring and return them as Python values.

    ``scores.cell_cycle`` and the cell cycle stage of a pipeline recipe check
    their settings with this function, so a recipe refuses a setting before
    its run starts, with the message that ``scores.cell_cycle`` gives.
    Without gene lists, the human S and G2M phase genes of
    ``cytearc.quality_control.cell_cycle_genes`` are scored, and without
    ``ctrl_size``, as many control genes as the shorter list holds.

    Raises:
        TypeError: If a gene list is not a sequence of gene names, or another
            setting has the wrong type.
        ValueError: If ``ctrl_size`` is below 1, ``n_bins`` below 2, or
            ``rand_seed`` outside 0 to 2**32 - 1.
    """
    if s_genes is None:
        s_genes = list(s_phase_genes)
    if g2m_genes is None:
        g2m_genes = list(g2m_phase_genes)
    for label, genes in (("s_genes", s_genes), ("g2m_genes", g2m_genes)):
        if isinstance(genes, str) or not all(isinstance(gene, str) for gene in genes):
            raise TypeError(f"{label} must be a sequence of gene names")
    control_size = integer_argument(
        min(len(s_genes), len(g2m_genes)) if ctrl_size is None else ctrl_size,
        "ctrl_size",
        minimum=1,
    )
    return CellCycleSettings(
        s_genes=s_genes,
        g2m_genes=g2m_genes,
        ctrl_size=control_size,
        log_transform=optional_bool_argument(log_transform, "log_transform"),
        n_bins=integer_argument(n_bins, "n_bins", minimum=2),
        # The seed seeds NumPy's legacy RandomState.
        rand_seed=integer_argument(
            rand_seed, "rand_seed", minimum=0, maximum=RANDOM_STATE_MAX
        ),
    )


def assign_cell_cycle_phase(
    s_score: np.ndarray,
    g2m_score: np.ndarray,
) -> np.ndarray:
    phase = np.full(np.asarray(s_score).shape, "S", dtype=object)
    phase[np.asarray(g2m_score) > np.asarray(s_score)] = "G2M"
    phase[(np.asarray(g2m_score) < 0) & (np.asarray(s_score) < 0)] = "G1"
    return phase
