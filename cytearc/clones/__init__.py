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

__all__ = [
    "MULTIPLE_POLICIES",
    "ORDERS",
    "ClonalDiversity",
    "ClonalExpansion",
    "Clones",
    "Rarefaction",
]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "MULTIPLE_POLICIES": ".assignments",
        "ORDERS": ".diversity",
        "ClonalDiversity": ".diversity",
        "ClonalExpansion": ".expansion",
        "Clones": ".assignments",
        "Rarefaction": ".diversity",
    },
)
