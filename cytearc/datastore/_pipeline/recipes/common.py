"""Resolution steps that every single-modality recipe shares."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ....metadata.membership import require_measured_cells
from ....metadata.selection import valid_category_mask
from ....storage.artifacts import ArtifactRef
from ....storage.selections import read_stored_selection_mask
from ....storage.types import as_zarr_group
from ....utils.arguments import integer_argument
from ..fields import reserved_fields
from ..filtering import filter_columns, resolve_filtering, snapshot_column_values
from ..settings import (
    HVG_TOP_N_DEFAULT,
    TSNE_DIMS_DEFAULT,
    UMAP_DIMS_DEFAULT,
    column_sequence,
    require_columns,
    stage_settings,
    warn_without_sgtsnepi,
)
from ..stages import CellSource


def selection_mask(store: Any, ref: ArtifactRef) -> np.ndarray:
    """Return a datastore cell selection as its mask over the cell axis."""
    return read_stored_selection_mask(
        store.zw,
        ref,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )


def input_cell_mask(store: Any, source: CellSource) -> np.ndarray:
    """Return a run's input cells, a column or a selection, as a cell-axis mask."""
    if source.column is not None:
        return np.asarray(store.cells.fetch_all(source.column), dtype=bool)
    assert source.selection is not None
    return selection_mask(store, source.selection)


def require_labeled_cells(
    labels: np.ndarray,
    missing: np.ndarray | None,
    active: np.ndarray,
    column: str,
    *,
    purpose: str,
    blank_is_label: bool = True,
) -> None:
    """Refuse labels that leave one of the active cells unlabeled.

    ``purpose`` names what needs the labels, such as ``"Cell calling per
    capture"``. A missing value, or NaN in a float column, is no label. With
    ``blank_is_label=False``, empty or whitespace text is no label either, as
    for the samples of the perturbation methods, which ``valid_category_mask``
    checks.
    """
    if blank_is_label:
        unlabeled = np.zeros(len(active), dtype=bool) if missing is None else missing
        if labels.dtype.kind == "f":
            unlabeled = unlabeled | np.isnan(labels)
    else:
        unlabeled = ~valid_category_mask(labels, missing_mask=missing)
    count = int(np.count_nonzero(unlabeled & active))
    if count:
        verb = "has" if count == 1 else "have"
        raise ValueError(
            f"{purpose} needs a {column!r} label for every cell, "
            f"but {count} of the run's cells {verb} none"
        )


def check_labeled_cells(
    store: Any,
    source: CellSource,
    column: str,
    *,
    purpose: str,
    blank_is_label: bool = True,
) -> None:
    """Refuse a cell column that leaves one of a run's input cells unlabeled.

    A recipe checks the live column before the run starts; the stage that
    reads the labels checks the cells that it reads. ``blank_is_label`` is
    that of :func:`require_labeled_cells`.
    """
    labels, missing = snapshot_column_values(
        as_zarr_group(store.zw["cellData"], name="cellData"), column
    )
    require_labeled_cells(
        labels,
        missing,
        input_cell_mask(store, source),
        column,
        purpose=purpose,
        blank_is_label=blank_is_label,
    )


def resolve_label(label: object) -> str | None:
    if label is None:
        return None
    if not isinstance(label, str) or not label:
        raise TypeError("label must be a non-empty string or None")
    return label


def require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def modality_assay(
    store: Any,
    assay: object,
    *,
    recipe: str,
    modality: str,
    classes: tuple[type, ...],
    types: frozenset[str] | None = None,
) -> str:
    """Return the assay a recipe analyzes, given or found by modality.

    Without ``assay``, a recipe takes the default assay when it has the
    recipe's modality, else the store's only assay of that modality.
    """

    def matches(name: str) -> bool:
        candidate = store._get_assay(name)
        if not isinstance(candidate, classes):
            return False
        return types is None or getattr(candidate, "assayType", None) in types

    def preset(name: str) -> str:
        return (
            f"a writable open with assay_types={{{name!r}: {modality!r}}} "
            f"opens an assay as {modality}"
        )

    if assay is not None:
        if not isinstance(assay, str) or not assay:
            raise TypeError("assay must be a non-empty string or None")
        if not matches(assay):
            kind = getattr(store._get_assay(assay), "assayType", None)
            # Only an assay of no specific type can be opened as another one.
            hint = f", and {preset(assay)}" if kind in (None, "Assay") else ""
            raise TypeError(
                f"The {recipe} recipe analyzes {modality} assays; assay "
                f"{assay!r} is of type {kind!r}{hint}"
            )
        return assay
    default = store._defaultAssay
    if isinstance(default, str) and default and matches(default):
        return default
    found = [str(name) for name in store.assay_names if matches(name)]
    if len(found) == 1:
        return found[0]
    if not found:
        raise TypeError(
            f"The store has no {modality} assay for the {recipe} recipe; "
            f"{preset('<assay>')}"
        )
    raise ValueError(
        f"The store has several {modality} assays, {found!r}; choose one with "
        f"pipeline.{recipe}(assay=...)"
    )


def resolve_hvg(store: Any, hvg: object) -> dict[str, Any]:
    """Return the settings of highly variable gene selection.

    Raises:
        KeyError: If ``batch`` names a missing cell column.
    """
    _enabled, settings = stage_settings("hvg", hvg, optional=False)
    batch = settings.get("batch")
    if batch is not None and batch not in store.cells.columns:
        raise KeyError(f"HVG batch column {batch!r} was not found")
    return settings


def check_pca_features(dims: int, hvg: Mapping[str, Any]) -> None:
    """Refuse PCA dimensions that the highly variable genes cannot give."""
    top_n = hvg.get("top_n", HVG_TOP_N_DEFAULT)
    if dims >= top_n:
        raise ValueError(
            f"pca dims ({dims}) must be below hvg top_n ({top_n}): PCA "
            "needs at least dims + 1 features"
        )


@dataclass(frozen=True, slots=True)
class GraphSettings:
    ann_index: dict[str, Any]
    neighbors: dict[str, Any]
    connectivity: dict[str, Any]
    initialization: dict[str, Any]

    def to_config(self) -> dict[str, Any]:
        return {
            "ann_index": self.ann_index,
            "neighbors": self.neighbors,
            "connectivity": self.connectivity,
            "embedding_initialization": self.initialization,
        }


def resolve_graph(
    *,
    ann_index: object,
    neighbors: object,
    connectivity: object,
    embedding_initialization: object,
) -> GraphSettings:
    return GraphSettings(
        ann_index=stage_settings("ann_index", ann_index, optional=False)[1],
        neighbors=stage_settings("neighbors", neighbors, optional=False)[1],
        connectivity=stage_settings("connectivity", connectivity, optional=False)[1],
        initialization=stage_settings(
            "embedding_initialization", embedding_initialization, optional=False
        )[1],
    )


@dataclass(frozen=True, slots=True)
class EmbeddingSettings:
    umap: bool
    umap_settings: dict[str, Any]
    umap_dims: int
    tsne: bool
    tsne_settings: dict[str, Any]
    tsne_dims: int

    def to_config(self) -> dict[str, Any]:
        return {
            "umap": self.umap_settings if self.umap else False,
            "tsne": self.tsne_settings if self.tsne else False,
        }


def resolve_embeddings(*, umap: object, tsne: object) -> EmbeddingSettings:
    umap_enabled, umap_settings = stage_settings("umap", umap, optional=True)
    tsne_enabled, tsne_settings = stage_settings("tsne", tsne, optional=True)
    return EmbeddingSettings(
        umap=umap_enabled,
        umap_settings=umap_settings,
        umap_dims=integer_argument(
            umap_settings.get("umap_dims", UMAP_DIMS_DEFAULT), "umap_dims", minimum=1
        ),
        tsne=tsne_enabled,
        tsne_settings=tsne_settings,
        tsne_dims=integer_argument(
            tsne_settings.get("tsne_dims", TSNE_DIMS_DEFAULT), "tsne_dims", minimum=1
        ),
    )


def resolve_harmony(
    store: Any,
    harmony: object,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Return Harmony's batch columns, empty when it is off, and its settings."""
    if harmony is None or harmony is False:
        return (), {}
    if harmony is True:
        raise ValueError("harmony needs a mapping with batch_columns")
    if not isinstance(harmony, Mapping):
        raise TypeError(
            "harmony must be a mapping such as {'batch_columns': [...]}, False, or None"
        )
    _enabled, settings = stage_settings("harmony", harmony, optional=True)
    if "batch_columns" not in settings:
        raise ValueError("harmony needs batch_columns")
    columns = column_sequence(settings.pop("batch_columns"), "harmony batch_columns")
    if not columns:
        raise ValueError("harmony batch_columns must not be empty")
    require_columns(store, columns, "Harmony columns")
    return columns, settings


def harmony_config(
    columns: Sequence[str],
    settings: Mapping[str, Any],
) -> dict[str, Any] | bool:
    if not columns:
        return False
    return {"batch_columns": list(columns), **settings}


def resolve_snapshot_columns(
    store: Any,
    snapshot_columns: object,
    *,
    reserved: frozenset[str],
) -> tuple[str, ...]:
    snapshots = column_sequence(snapshot_columns, "snapshot_columns")
    collisions = set(snapshots) & reserved
    if collisions:
        raise ValueError(
            f"snapshot_columns collide with reserved run fields: {sorted(collisions)!r}"
        )
    require_columns(store, snapshots, "Snapshot columns")
    return snapshots


def resolve_run_filtering(
    store: Any,
    filtering: object,
    *,
    cells: CellSource,
    assay: str,
    default_attrs: Sequence[str],
    operation: str,
    default_on: bool = True,
    qc_stages: bool = False,
) -> dict[str, Any]:
    """Resolve ``filtering``, whose default depends on where the cells come from.

    None filters a run whose cells name a column, and not one given a
    selection or a run, whose cells are taken as already filtered. A run
    whose ``qc_stages`` call cells or keep hashtag singlets filters the cells
    they keep, whatever its input.
    """
    if filtering is None:
        filtering = default_on and (qc_stages or cells.column is not None)
    if not isinstance(filtering, bool | Mapping):
        raise TypeError("filtering must be a mapping, bool, or None")
    config = resolve_filtering(store, filtering, default_attrs=default_attrs)
    columns = tuple(config.get("attrs", ()))
    if columns:
        # A QC metric reads the assay whose preparation wrote it, as
        # qc.auto_filter reads it, so every other assay whose metric the run
        # filters on must have measured the run's cells too.
        for metric_assay in store._metric_assays(columns, ()):
            if metric_assay == assay:
                continue
            if cells.column is not None:
                require_measured_cells(
                    store.cells,
                    metric_assay,
                    cells.column,
                    operation=operation,
                    remedy="cells",
                )
            else:
                store._require_measured_cells(
                    metric_assay,
                    cells.selection,
                    operation=operation,
                    remedy="cells",
                )
    return config


def cell_snapshots(
    filtering: Mapping[str, Any],
    *,
    harmony_columns: Sequence[str] = (),
    snapshot_columns: Sequence[str] = (),
    capture_column: str | None = None,
    doublet_samples: Sequence[str] = (),
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Return the columns of a run's cell snapshot and of its other snapshots.

    The cell snapshot holds the names and the QC columns that filtering
    reads. Every other group of columns is frozen apart, under the output key
    of the stage that reads it, so that a column one stage reads never changes
    the identity of what another stage computes: a run with Harmony or with
    snapshot columns filters to the same cells as a run without them.
    """
    groups = {
        "capture_snapshot": () if capture_column is None else (capture_column,),
        "harmony_snapshot": tuple(harmony_columns),
        "doublet_sample_snapshot": tuple(doublet_samples),
        "column_snapshot": tuple(snapshot_columns),
    }
    return (
        tuple(dict.fromkeys(("names", *filter_columns(filtering)))),
        {key: columns for key, columns in groups.items() if columns},
    )


def check_tsne(embeddings: EmbeddingSettings) -> None:
    if embeddings.tsne:
        # Once the recipe is valid, so the warning precedes every stage.
        warn_without_sgtsnepi()


def run_reserved_fields(
    embeddings: EmbeddingSettings,
    partitions: Sequence[tuple[str, float]],
    extra: Sequence[str] = (),
) -> frozenset[str]:
    return reserved_fields(
        umap_dims=embeddings.umap_dims,
        tsne_dims=embeddings.tsne_dims,
        leiden_keys=[key for key, _value in partitions],
        extra=extra,
    )
