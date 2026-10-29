from typing import TYPE_CHECKING

from .._facade import lazy_facade as _lazy_facade

if TYPE_CHECKING:
    from .chains import (
        CHAIN_PAIRINGS as CHAIN_PAIRINGS,
        LOCI as LOCI,
        NO_RECEPTOR as NO_RECEPTOR,
        RECEPTOR_SUBTYPES as RECEPTOR_SUBTYPES,
        RECEPTOR_TYPES as RECEPTOR_TYPES,
        SLOTS as SLOTS,
        VDJ_LOCI as VDJ_LOCI,
        VJ_LOCI as VJ_LOCI,
    )
    from .contigs import FIELDS as FIELDS
    from .receptors import ImmuneReceptors as ImmuneReceptors

__all__ = [
    "CHAIN_PAIRINGS",
    "FIELDS",
    "LOCI",
    "NO_RECEPTOR",
    "RECEPTOR_SUBTYPES",
    "RECEPTOR_TYPES",
    "SLOTS",
    "VDJ_LOCI",
    "VJ_LOCI",
    "ImmuneReceptors",
]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "CHAIN_PAIRINGS": ".chains",
        "FIELDS": ".contigs",
        "LOCI": ".chains",
        "NO_RECEPTOR": ".chains",
        "RECEPTOR_SUBTYPES": ".chains",
        "RECEPTOR_TYPES": ".chains",
        "SLOTS": ".chains",
        "VDJ_LOCI": ".chains",
        "VJ_LOCI": ".chains",
        "ImmuneReceptors": ".receptors",
    },
)
