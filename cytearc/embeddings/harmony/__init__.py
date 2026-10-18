"""Harmony2 batch correction of reduced cell embeddings.

Korsunsky I, et al. Fast, sensitive and accurate integration of single-cell data with Harmony. Nat Methods 16, 1289-1296 (2019). doi:10.1038/s41592-019-0619-0
Patikas N, Yao H, et al. Integration of large, complex single-cell datasets with Harmony2. bioRxiv (2026). doi:10.64898/2026.03.16.711825
"""

from .fit import fit_harmony
from .memory import harmony_peak_bytes
from .models import HarmonyResult
from .parameters import harmony_cluster_count, validate_harmony_parameters

__all__ = [
    "HarmonyResult",
    "fit_harmony",
    "harmony_cluster_count",
    "harmony_peak_bytes",
    "validate_harmony_parameters",
]

for _public_object in (
    HarmonyResult,
    fit_harmony,
    harmony_cluster_count,
    harmony_peak_bytes,
    validate_harmony_parameters,
):
    _public_object.__module__ = __name__
del _public_object
