from typing import TYPE_CHECKING

from .._facade import lazy_facade as _lazy_facade

if TYPE_CHECKING:
    from .counts import (
        ALLELE_ROLES as ALLELE_ROLES,
        COUNT_UNITS as COUNT_UNITS,
        DEPTH_DEFINITIONS as DEPTH_DEFINITIONS,
        VariantCounts as VariantCounts,
    )

__all__ = [
    "ALLELE_ROLES",
    "COUNT_UNITS",
    "DEPTH_DEFINITIONS",
    "VariantCounts",
]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "ALLELE_ROLES": ".counts",
        "COUNT_UNITS": ".counts",
        "DEPTH_DEFINITIONS": ".counts",
        "VariantCounts": ".counts",
    },
)
