from typing import TYPE_CHECKING

from .._facade import lazy_facade as _lazy_facade

if TYPE_CHECKING:
    from .guide_calls import (
        MULTIPLE as MULTIPLE,
        NEGATIVE as NEGATIVE,
        GuideCalls as GuideCalls,
    )
    from .efficiency import PerturbationEfficiency as PerturbationEfficiency
    from .guide_thresholds import GuideThresholds as GuideThresholds
    from .labels import (
        CONTROL as CONTROL,
        PerturbationLabels as PerturbationLabels,
    )

__all__ = [
    "CONTROL",
    "MULTIPLE",
    "NEGATIVE",
    "GuideCalls",
    "GuideThresholds",
    "PerturbationEfficiency",
    "PerturbationLabels",
]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "CONTROL": ".labels",
        "MULTIPLE": ".guide_calls",
        "NEGATIVE": ".guide_calls",
        "GuideCalls": ".guide_calls",
        "GuideThresholds": ".guide_thresholds",
        "PerturbationEfficiency": ".efficiency",
        "PerturbationLabels": ".labels",
    },
)
