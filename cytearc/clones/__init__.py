from typing import TYPE_CHECKING

from .._facade import lazy_facade as _lazy_facade

if TYPE_CHECKING:
    from .assignments import (
        MULTIPLE_POLICIES as MULTIPLE_POLICIES,
        Clones as Clones,
    )
    from .diversity import (
        ORDERS as ORDERS,
        ClonalDiversity as ClonalDiversity,
        Rarefaction as Rarefaction,
    )
    from .expansion import ClonalExpansion as ClonalExpansion
    from .overlap import (
        OVERLAP_INDICES as OVERLAP_INDICES,
        CloneOverlap as CloneOverlap,
        CloneTracking as CloneTracking,
    )

__all__ = [
    "MULTIPLE_POLICIES",
    "ORDERS",
    "OVERLAP_INDICES",
    "ClonalDiversity",
    "ClonalExpansion",
    "CloneOverlap",
    "CloneTracking",
    "Clones",
    "Rarefaction",
]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "MULTIPLE_POLICIES": ".assignments",
        "ORDERS": ".diversity",
        "OVERLAP_INDICES": ".overlap",
        "ClonalDiversity": ".diversity",
        "ClonalExpansion": ".expansion",
        "CloneOverlap": ".overlap",
        "CloneTracking": ".overlap",
        "Clones": ".assignments",
        "Rarefaction": ".diversity",
    },
)
