"""Differential abundance of cell categories and neighbourhoods.

:func:`propeller` tests category proportions between groups of samples, and
:mod:`cytearc.abundance.milo` holds the storage-free steps of Milo's
neighbourhood test. Both were implemented from their papers and treat samples,
not cells, as replicates.
"""

from typing import TYPE_CHECKING

from .._facade import lazy_facade as _lazy_facade

if TYPE_CHECKING:
    from .propeller import (
        PropellerFit as PropellerFit,
        count_categories as count_categories,
        propeller as propeller,
        transformed_proportions as transformed_proportions,
    )

__all__ = [
    "PropellerFit",
    "count_categories",
    "propeller",
    "transformed_proportions",
]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "PropellerFit": ".propeller",
        "count_categories": ".propeller",
        "propeller": ".propeller",
        "transformed_proportions": ".propeller",
    },
)
