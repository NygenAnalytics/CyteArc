from typing import TYPE_CHECKING

from .._facade import lazy_facade as _lazy_facade

if TYPE_CHECKING:
    from .guide_calls import (
        MULTIPLE as MULTIPLE,
        NEGATIVE as NEGATIVE,
        GuideCalls as GuideCalls,
    )

__all__ = [
    "MULTIPLE",
    "NEGATIVE",
    "GuideCalls",
]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "MULTIPLE": ".guide_calls",
        "NEGATIVE": ".guide_calls",
        "GuideCalls": ".guide_calls",
    },
)
