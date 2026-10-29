from typing import TYPE_CHECKING

from .._facade import lazy_facade as _lazy_facade

if TYPE_CHECKING:
    from .guide_calls import (
        MULTIPLE as MULTIPLE,
        NEGATIVE as NEGATIVE,
        GuideCalls as GuideCalls,
    )
    from .associations import PerturbationAssociations as PerturbationAssociations
    from .edistance import PerturbationEDistance as PerturbationEDistance
    from .efficiency import PerturbationEfficiency as PerturbationEfficiency
    from .guide_thresholds import GuideThresholds as GuideThresholds
    from .labels import (
        CONTROL as CONTROL,
        PerturbationLabels as PerturbationLabels,
    )
    from .mixscape import MixscapeResult as MixscapeResult
    from .signature import PerturbationSignature as PerturbationSignature

__all__ = [
    "CONTROL",
    "MULTIPLE",
    "NEGATIVE",
    "GuideCalls",
    "GuideThresholds",
    "MixscapeResult",
    "PerturbationAssociations",
    "PerturbationEDistance",
    "PerturbationEfficiency",
    "PerturbationLabels",
    "PerturbationSignature",
]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "CONTROL": ".labels",
        "MULTIPLE": ".guide_calls",
        "NEGATIVE": ".guide_calls",
        "GuideCalls": ".guide_calls",
        "GuideThresholds": ".guide_thresholds",
        "MixscapeResult": ".mixscape",
        "PerturbationAssociations": ".associations",
        "PerturbationEDistance": ".edistance",
        "PerturbationEfficiency": ".efficiency",
        "PerturbationLabels": ".labels",
        "PerturbationSignature": ".signature",
    },
)
