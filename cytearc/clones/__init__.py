from typing import TYPE_CHECKING

from .._facade import lazy_facade as _lazy_facade

if TYPE_CHECKING:
    from .assignments import (
        MULTIPLE_POLICIES as MULTIPLE_POLICIES,
        Clones as Clones,
    )
    from .barcode_clones import BARCODE_PRESETS as BARCODE_PRESETS
    from .barcodes import LineageBarcodes as LineageBarcodes
    from .diversity import (
        ORDERS as ORDERS,
        ClonalDiversity as ClonalDiversity,
        Rarefaction as Rarefaction,
    )
    from .expansion import ClonalExpansion as ClonalExpansion
    from .fate import (
        COUPLING_METHODS as COUPLING_METHODS,
        CloneCoupling as CloneCoupling,
        CloneFateBias as CloneFateBias,
    )
    from .overlap import (
        OVERLAP_INDICES as OVERLAP_INDICES,
        CloneOverlap as CloneOverlap,
        CloneTracking as CloneTracking,
    )
    from .phylogeny import CellPhylogeny as CellPhylogeny
    from .recorders import LineageAlleles as LineageAlleles

__all__ = [
    "BARCODE_PRESETS",
    "COUPLING_METHODS",
    "MULTIPLE_POLICIES",
    "ORDERS",
    "OVERLAP_INDICES",
    "CellPhylogeny",
    "ClonalDiversity",
    "ClonalExpansion",
    "CloneCoupling",
    "CloneFateBias",
    "CloneOverlap",
    "CloneTracking",
    "Clones",
    "LineageAlleles",
    "LineageBarcodes",
    "Rarefaction",
]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "BARCODE_PRESETS": ".barcode_clones",
        "COUPLING_METHODS": ".fate",
        "MULTIPLE_POLICIES": ".assignments",
        "ORDERS": ".diversity",
        "OVERLAP_INDICES": ".overlap",
        "ClonalDiversity": ".diversity",
        "ClonalExpansion": ".expansion",
        "CloneCoupling": ".fate",
        "CloneFateBias": ".fate",
        "CellPhylogeny": ".phylogeny",
        "CloneOverlap": ".overlap",
        "CloneTracking": ".overlap",
        "Clones": ".assignments",
        "LineageAlleles": ".recorders",
        "LineageBarcodes": ".barcodes",
        "Rarefaction": ".diversity",
    },
)
