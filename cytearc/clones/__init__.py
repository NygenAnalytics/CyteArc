from typing import TYPE_CHECKING

from .._facade import lazy_facade as _lazy_facade

if TYPE_CHECKING:
    from .assignments import (
        MULTIPLE_POLICIES as MULTIPLE_POLICIES,
        Clones as Clones,
    )

__all__ = [
    "MULTIPLE_POLICIES",
    "Clones",
]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "MULTIPLE_POLICIES": ".assignments",
        "Clones": ".assignments",
    },
)
