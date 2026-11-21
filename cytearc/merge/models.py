from dataclasses import dataclass, field
from typing import Any, Literal

from ..storage.profiles import StorageProfile

type MissingAssayPolicy = Literal["zero_fill", "error"]
type ComponentAction = Literal["write", "resume", "skip", "blocked"]


@dataclass(frozen=True, slots=True)
class AssayMergePlan:
    """Resolved plan for one assay in a DataStoreMerge.

    ``featureOverlapFraction`` is the share of the merged features that every
    source holding the assay measured. Features match by ID, or by name with
    ``feature_key="names"``. The merge writes zero counts for the other
    features in the cells of each source that lacks them, so ``1.0`` means
    that no cell reads a zero count for a feature its source never measured. A
    source whose assay holds no features measured none of the others'
    features, so the assay then reports ``0.0``, while an assay of no features
    in any source reports ``1.0``. A source without the assay is left out,
    since the merge marks its cells as not measured, so an assay that only one
    source holds reports ``1.0``.
    """

    assayName: str
    assayType: str
    sourcePresent: tuple[bool, ...]
    missingSources: tuple[str, ...]
    nFeatures: int
    featureOverlapFraction: float
    dtype: str
    chunks: tuple[int, int]
    shards: tuple[int, int] | None
    writeCountsT: bool
    estimatedWriteTasks: int
    countsAction: ComponentAction
    countsTAction: ComponentAction


@dataclass(frozen=True, slots=True)
class MergePlan:
    """Side-effect-free preview of a DataStoreMerge."""

    zarrPath: str
    outWorkspace: str | None
    sourceNames: tuple[str, ...]
    nCells: int
    assays: tuple[AssayMergePlan, ...]
    profile: StorageProfile
    seed: int | None
    missingAssayPolicy: MissingAssayPolicy
    willResume: bool
    canDump: bool
    blockedReason: str | None
    cellDataAction: ComponentAction
    manifest: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ComponentResult:
    """Outcome for one written or skipped merge component."""

    name: str
    action: ComponentAction


@dataclass(frozen=True, slots=True)
class MergeResult:
    """Result returned by DataStoreMerge.dump()."""

    zarrPath: str
    nCells: int
    assayNames: tuple[str, ...]
    components: tuple[ComponentResult, ...]
    resumed: bool
