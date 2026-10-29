from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from ...metadata.artifacts import categorical_display, continuous_display
from ...metadata.rows import array_block_rows
from ...storage.arrays import linked_missing_mask
from ...storage.artifacts import ArtifactRef, artifact_group
from ...storage.pipeline_runs import PipelineFieldDescriptor
from ...storage.types import as_zarr_array


def artifact_array(root: Any, ref: ArtifactRef, name: str) -> Any:
    group = artifact_group(root, ref)
    return as_zarr_array(group[name], name=name)


def _iter_array_blocks(
    array: Any,
    *,
    value_index: int | None = None,
) -> Iterator[np.ndarray]:
    n_rows = int(array.shape[0])
    block_rows = array_block_rows(array, n_rows)
    for start in range(0, n_rows, block_rows):
        stop = min(start + block_rows, n_rows)
        if value_index is None:
            yield np.asarray(array[start:stop])
        else:
            yield np.asarray(array[start:stop, value_index])


def continuous_array_display(
    array: Any,
    *,
    value_index: int | None = None,
) -> dict[str, Any]:
    minimum: float | None = None
    maximum: float | None = None
    for block in _iter_array_blocks(array, value_index=value_index):
        numeric = np.asarray(block, dtype=np.float64)
        finite = numeric[np.isfinite(numeric)]
        if finite.size == 0:
            continue
        block_minimum = float(finite.min())
        block_maximum = float(finite.max())
        minimum = block_minimum if minimum is None else min(minimum, block_minimum)
        maximum = block_maximum if maximum is None else max(maximum, block_maximum)
    extrema = (
        np.empty(0, dtype=np.float64)
        if minimum is None or maximum is None
        else np.asarray([minimum, maximum], dtype=np.float64)
    )
    return continuous_display(extrema)


def categorical_array_display(array: Any) -> dict[str, Any]:
    distinct: list[Any] = []
    for block in _iter_array_blocks(array):
        values = np.asarray(block).reshape(-1)
        # Typed blocks hold one value type, so their distinct values carry the
        # same categories in first-seen order; object blocks can mix types.
        distinct.extend(pd.unique(values) if values.dtype != object else values)
    return categorical_display(np.asarray(distinct, dtype=object))


def _fill_for_dtype(dtype: np.dtype[Any]) -> str | int | bool:
    if dtype.kind == "f":
        return "nan"
    if dtype.kind in {"i", "u"}:
        return -1
    if dtype.kind == "b":
        return False
    return ""


def _field(
    *,
    key: str,
    axis: Literal["cells", "features"],
    ref: ArtifactRef,
    source_value: str,
    dtype: Any,
    value_index: int | None = None,
    missing_mask: str | None = None,
    display: Mapping[str, object] | None = None,
    fill: str | int | float | bool | None = None,
) -> PipelineFieldDescriptor:
    resolved_dtype = np.dtype(dtype)
    return PipelineFieldDescriptor(
        key=key,
        axis=axis,
        artifact=ref,
        source_value=source_value,
        value_index=value_index,
        dtype=resolved_dtype.str,
        fill=_fill_for_dtype(resolved_dtype) if fill is None else fill,
        missing_mask=missing_mask,
        display=None if display is None else dict(display),
    )


def _snapshot_field(
    root: Any,
    *,
    key: str,
    axis: Literal["cells", "features"],
    snapshot: ArtifactRef,
) -> PipelineFieldDescriptor:
    group = artifact_group(root, snapshot)
    array = as_zarr_array(group[key], name=key)
    missing = linked_missing_mask(group, key, values=array)
    return _field(
        key=key,
        axis=axis,
        ref=snapshot,
        source_value=key,
        dtype=array.dtype,
        missing_mask=None if missing is None else missing.basename,
    )


# The cell fields that a snapshot column may not shadow, besides the Leiden,
# UMAP, and t-SNE fields of the run and its recipe's feature fields.
RESERVED_CELL_FIELDS = frozenset(
    {
        "I",
        "ids",
        "names",
        "s_score",
        "g2m_score",
        "cell_cycle_phase",
        "umap_1",
        "umap_2",
        "paris",
        "clusters",
        "membership_strength",
        "doublet_score",
        "predicted_doublet",
    }
)


def reserved_fields(
    *,
    umap_dims: int,
    tsne_dims: int,
    leiden_keys: Sequence[str],
    extra: Sequence[str] = (),
) -> frozenset[str]:
    """Return the field names that a run's snapshot columns may not use."""
    return frozenset(
        {
            *RESERVED_CELL_FIELDS,
            *(f"umap_{index}" for index in range(1, umap_dims + 1)),
            *(f"tsne_{index}" for index in range(1, tsne_dims + 1)),
            *(f"leiden_{key}" for key in leiden_keys),
            *extra,
        }
    )


def _continuous(
    root: Any,
    *,
    key: str,
    ref: ArtifactRef,
    source: str,
    value_index: int | None = None,
) -> PipelineFieldDescriptor:
    values = artifact_array(root, ref, source)
    return _field(
        key=key,
        axis="cells",
        ref=ref,
        source_value=source,
        value_index=value_index,
        dtype=values.dtype,
        display=continuous_array_display(values, value_index=value_index),
    )


def _categorical(
    root: Any,
    *,
    key: str,
    ref: ArtifactRef,
    source: str,
) -> PipelineFieldDescriptor:
    values = artifact_array(root, ref, source)
    return _field(
        key=key,
        axis="cells",
        ref=ref,
        source_value=source,
        dtype=values.dtype,
        display=categorical_array_display(values),
    )


def cell_fields(
    store: Any,
    artifacts: Mapping[str, ArtifactRef],
    *,
    snapshot_columns: Sequence[str],
    leiden_keys: Sequence[str],
) -> list[PipelineFieldDescriptor]:
    """Return a run's cell fields: its selection, snapshots, and results.

    Results appear in one order for every recipe: cell cycle scores, UMAP and
    t-SNE coordinates, Leiden candidates, Paris, the saved clustering,
    membership strength, doublet calls, guide calls, perturbation labels,
    Mixscape classes, and WNN modality weights. A result the run did not
    produce has no field.
    """
    root = store.zw
    cell_snapshot = artifacts["cell_snapshot"]
    fields: list[PipelineFieldDescriptor] = [
        _field(
            key="I",
            axis="cells",
            ref=artifacts["analysis_cell_selection"],
            source_value="values",
            dtype=bool,
            fill=False,
        ),
        _field(
            key="ids",
            axis="cells",
            ref=cell_snapshot,
            source_value="ids",
            dtype=store.cells._get_array("ids").dtype,
            fill="",
        ),
    ]
    fields.append(
        _snapshot_field(root, key="names", axis="cells", snapshot=cell_snapshot)
    )
    fields.extend(
        _snapshot_field(
            root, key=column, axis="cells", snapshot=artifacts["column_snapshot"]
        )
        for column in snapshot_columns
    )
    if "cell_cycle" in artifacts:
        ref = artifacts["cell_cycle"]
        for key in ("s_score", "g2m_score"):
            fields.append(_continuous(root, key=key, ref=ref, source=key))
        fields.append(
            _categorical(root, key="cell_cycle_phase", ref=ref, source="phase")
        )
    for embedding in ("umap", "tsne"):
        if embedding not in artifacts:
            continue
        ref = artifacts[embedding]
        values = artifact_array(root, ref, "values")
        fields.extend(
            _continuous(
                root,
                key=f"{embedding}_{index + 1}",
                ref=ref,
                source="values",
                value_index=index,
            )
            for index in range(values.shape[1])
        )
    for key in leiden_keys:
        output_key = f"leiden_{key}"
        if output_key in artifacts:
            fields.append(
                _categorical(
                    root, key=output_key, ref=artifacts[output_key], source="values"
                )
            )
    if "paris" in artifacts:
        fields.append(
            _categorical(root, key="paris", ref=artifacts["paris"], source="labels")
        )
    if "clusters" in artifacts:
        # The saved clustering is always a Leiden candidate.
        fields.append(
            _categorical(
                root, key="clusters", ref=artifacts["clusters"], source="values"
            )
        )
    if "membership_strength" in artifacts:
        fields.append(
            _continuous(
                root,
                key="membership_strength",
                ref=artifacts["membership_strength"],
                source="values",
            )
        )
    if "doublets" in artifacts:
        ref = artifacts["doublets"]
        fields.append(_continuous(root, key="doublet_score", ref=ref, source="values"))
        fields.append(
            _categorical(
                root, key="predicted_doublet", ref=ref, source="predicted_doublets"
            )
        )
    # The guide calls and labels belong to the guide assay. Every assay shares
    # the store's cells, so the view reads them by the run's cell selection.
    if "guide_calls" in artifacts:
        fields.append(
            _categorical(
                root, key="guide_call", ref=artifacts["guide_calls"], source="values"
            )
        )
    if "perturbation_labels" in artifacts:
        ref = artifacts["perturbation_labels"]
        fields.append(
            _categorical(root, key="perturbation_label", ref=ref, source="values")
        )
        fields.append(
            _categorical(root, key="is_control", ref=ref, source="is_control")
        )
    if "mixscape" in artifacts:
        # Mixscape classifies the run's cells; its other results describe
        # each target and stay in the artifact.
        ref = artifacts["mixscape"]
        fields.append(
            _categorical(root, key="mixscape_class", ref=ref, source="values")
        )
        fields.append(
            _continuous(root, key="mixscape_posterior", ref=ref, source="posterior")
        )
    integrated = artifacts.get("integrated_graph")
    if integrated is not None:
        group = artifact_group(root, integrated)
        assays = group.attrs.get("assays")
        if "modality_weights" in group and isinstance(assays, list):
            for index, assay in enumerate(assays):
                fields.append(
                    _continuous(
                        root,
                        key=f"weight_{assay}",
                        ref=integrated,
                        source="modality_weights",
                        value_index=index,
                    )
                )
    return fields


def feature_fields(
    store: Any,
    assay: str,
    artifacts: Mapping[str, ArtifactRef],
    *,
    selection_key: str | None,
) -> list[PipelineFieldDescriptor]:
    """Return a run's feature fields: the assay's features and the selected ones.

    ``selection_key`` names the run output that selected the features the
    reduction used, such as ``"highly_variable_features"``; it becomes a
    boolean feature field of the same name.
    """
    feature_snapshot = artifacts["feature_snapshot"]
    fields = [
        _field(
            key="I",
            axis="features",
            ref=artifacts["feature_universe"],
            source_value="values",
            dtype=bool,
            fill=False,
        ),
        _field(
            key="ids",
            axis="features",
            ref=feature_snapshot,
            source_value="ids",
            dtype=store._get_assay(assay).feats._get_array("ids").dtype,
            fill="",
        ),
        _snapshot_field(
            store.zw,
            key="names",
            axis="features",
            snapshot=feature_snapshot,
        ),
    ]
    if selection_key is not None:
        selected = artifacts[selection_key]
        fields.append(
            _field(
                key=selection_key,
                axis="features",
                ref=selected,
                source_value="values",
                dtype=artifact_array(store.zw, selected, "values").dtype,
                fill=False,
            )
        )
    return fields
