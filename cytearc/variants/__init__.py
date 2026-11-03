from typing import TYPE_CHECKING

from .._facade import lazy_facade as _lazy_facade

if TYPE_CHECKING:
    from .counts import (
        ALLELE_ROLES as ALLELE_ROLES,
        COUNT_UNITS as COUNT_UNITS,
        DEPTH_DEFINITIONS as DEPTH_DEFINITIONS,
        VariantCounts as VariantCounts,
    )
    from .genotypes import (
        CALLS as CALLS,
        GENOTYPE_METHODS as GENOTYPE_METHODS,
        SOURCE_CALLS as SOURCE_CALLS,
        VariantGenotypes as VariantGenotypes,
    )
    from .masks import VARIANT_MASKS as VARIANT_MASKS
    from .statistics import (
        INFORMATIVE_PRESETS as INFORMATIVE_PRESETS,
        VariantSelection as VariantSelection,
    )

__all__ = [
    "ALLELE_ROLES",
    "CALLS",
    "COUNT_UNITS",
    "DEPTH_DEFINITIONS",
    "GENOTYPE_METHODS",
    "INFORMATIVE_PRESETS",
    "SOURCE_CALLS",
    "VARIANT_MASKS",
    "VariantCounts",
    "VariantGenotypes",
    "VariantSelection",
]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "ALLELE_ROLES": ".counts",
        "CALLS": ".genotypes",
        "COUNT_UNITS": ".counts",
        "DEPTH_DEFINITIONS": ".counts",
        "GENOTYPE_METHODS": ".genotypes",
        "INFORMATIVE_PRESETS": ".statistics",
        "SOURCE_CALLS": ".genotypes",
        "VARIANT_MASKS": ".masks",
        "VariantCounts": ".counts",
        "VariantGenotypes": ".genotypes",
        "VariantSelection": ".statistics",
    },
)
