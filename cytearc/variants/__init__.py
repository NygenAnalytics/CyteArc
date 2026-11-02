from typing import TYPE_CHECKING

from .._facade import lazy_facade as _lazy_facade

if TYPE_CHECKING:
    from .counts import (
        ALLELE_ROLES as ALLELE_ROLES,
        COUNT_UNITS as COUNT_UNITS,
        DEPTH_DEFINITIONS as DEPTH_DEFINITIONS,
        VariantCounts as VariantCounts,
    )
    from .masks import VARIANT_MASKS as VARIANT_MASKS
    from .statistics import PRESETS as PRESETS, VariantSelection as VariantSelection

__all__ = [
    "ALLELE_ROLES",
    "COUNT_UNITS",
    "DEPTH_DEFINITIONS",
    "PRESETS",
    "VARIANT_MASKS",
    "VariantCounts",
    "VariantSelection",
]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "ALLELE_ROLES": ".counts",
        "COUNT_UNITS": ".counts",
        "DEPTH_DEFINITIONS": ".counts",
        "PRESETS": ".statistics",
        "VARIANT_MASKS": ".masks",
        "VariantCounts": ".counts",
        "VariantSelection": ".statistics",
    },
)
