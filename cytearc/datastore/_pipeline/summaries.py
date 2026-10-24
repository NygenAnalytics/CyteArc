"""Small diagnostics that a run records for each completed stage.

Each summary is computed from the artifacts that the stage produced, so a
reused artifact reports the same numbers as a computed one. Summaries read
attributes and bounded blocks of small arrays, never a matrix.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from ...metadata.rows import array_block_rows
from ...storage.artifacts import ArtifactRef, artifact_group, inspect_artifact
from ...storage.types import as_zarr_array

# A summary lists at most this many samples, clusters, or components.
SUMMARY_ITEM_LIMIT = 50


def _array(group: Any, name: str) -> Any | None:
    if name not in group:
        return None
    return as_zarr_array(group[name], name=name)


def _blocks(array: Any) -> Any:
    rows = int(array.shape[0])
    block = array_block_rows(array, rows)
    for start in range(0, rows, block):
        yield np.asarray(array[start : min(start + block, rows)])


def _true_count(array: Any) -> int:
    return sum(int(np.count_nonzero(block)) for block in _blocks(array))


def _float(value: Any, digits: int = 4) -> float | None:
    number = float(value)
    return round(number, digits) if np.isfinite(number) else None


# The inputs that name the cells a derived selection narrowed, by operation.
_SELECTION_SOURCES = {
    "filter_pipeline_cells": "input_cell_selection",
    "call_cells": "prior_cell_selection",
    "combine_capture_cell_calls": "input_cell_selection",
    "select_hashtag_singlets": "cell_selection",
}


def _selection(root: Any, ref: ArtifactRef, group: Any) -> dict[str, Any]:
    values = _array(group, "values")
    if values is None:
        return {}
    summary: dict[str, Any] = {"cells": _true_count(values)}
    status = inspect_artifact(root, ref)
    source_key = _SELECTION_SOURCES.get(str(status.operation))
    source = None if source_key is None else (status.inputs or {}).get(source_key)
    if isinstance(source, Mapping):
        before = _array(
            artifact_group(root, ArtifactRef.from_dict(dict(source))), "values"
        )
        if before is not None:
            kept = summary["cells"]
            summary["inputCells"] = _true_count(before)
            summary["removedCells"] = summary["inputCells"] - kept
    mad = (status.parameters or {}).get("mad")
    if isinstance(mad, Mapping):
        sizes = mad.get("sample_sizes")
        if isinstance(sizes, Mapping):
            summary["samples"] = len(sizes)
    if status.operation == "combine_capture_cell_calls":
        captures = (status.parameters or {}).get("captures") or []
        called: dict[str, int] = {}
        knees: dict[str, Any] = {}
        for index, capture in enumerate(captures[:SUMMARY_ITEM_LIMIT]):
            raw = (status.inputs or {}).get(f"capture_{index}")
            if not isinstance(raw, Mapping):
                continue
            capture_ref = ArtifactRef.from_dict(dict(raw))
            capture_values = _array(artifact_group(root, capture_ref), "values")
            if capture_values is not None:
                called[str(capture)] = _true_count(capture_values)
            # Each capture's calls come from its own EmptyDrops fit.
            calls = (inspect_artifact(root, capture_ref).inputs or {}).get(
                "cell_calling"
            )
            if isinstance(calls, Mapping):
                fit = artifact_group(root, ArtifactRef.from_dict(dict(calls)))
                if fit.attrs.get("knee") is not None:
                    knees[str(capture)] = fit.attrs["knee"]
        summary["calledPerCapture"] = called
        summary["kneePerCapture"] = knees
    return summary


def _hashtags(group: Any) -> dict[str, Any]:
    values = _array(group, "values")
    if values is None:
        return {}
    counts: dict[str, int] = {}
    for block in _blocks(values):
        labels, numbers = np.unique(np.asarray(block).astype(str), return_counts=True)
        for label, number in zip(labels.tolist(), numbers.tolist(), strict=True):
            counts[label] = counts.get(label, 0) + int(number)
    ordered = dict(sorted(counts.items(), key=lambda item: -item[1]))
    return {
        "hashtags": dict(list(ordered.items())[:SUMMARY_ITEM_LIMIT]),
        "negative": counts.get("Negative", 0),
        "multiplets": counts.get("Doublet", 0),
    }


def _cell_calling(group: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("retain", "knee", "inflection", "n_ambient"):
        if key in group.attrs and group.attrs[key] is not None:
            summary[key] = group.attrs[key]
    return summary


def _features(group: Any) -> dict[str, Any]:
    values = _array(group, "values")
    return {} if values is None else {"features": _true_count(values)}


def _shape(group: Any, name: str, labels: tuple[str, str]) -> dict[str, Any]:
    array = _array(group, name)
    if array is None or array.ndim != 2:
        return {}
    return {labels[0]: int(array.shape[0]), labels[1]: int(array.shape[1])}


def _reduction(group: Any) -> dict[str, Any]:
    summary = _shape(group, "data", ("cells", "dims"))
    ratio = _array(group, "variance_ratio")
    if ratio is not None:
        values = np.asarray(ratio[:SUMMARY_ITEM_LIMIT], dtype=np.float64)
        summary["varianceRatio"] = [_float(value) for value in values]
        summary["varianceExplained"] = _float(np.asarray(ratio[:]).sum())
    return summary


def _harmony(group: Any) -> dict[str, Any]:
    fit = group.attrs.get("fit_parameters")
    if not isinstance(fit, Mapping):
        return {}
    return {
        key: fit[source]
        for key, source in (("rounds", "rounds"), ("converged", "converged"))
        if source in fit
    }


def _neighbors(group: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, name in (("cells", "n_cells"), ("k", "n_neighbors")):
        if name in group.attrs:
            summary[key] = int(group.attrs[name])
    if "self_hit_rate" in group.attrs:
        summary["selfHitRate"] = _float(group.attrs["self_hit_rate"])
    return summary


def _clusters(group: Any) -> dict[str, Any]:
    values = _array(group, "values")
    if values is None:
        return {}
    counts: dict[Any, int] = {}
    for block in _blocks(values):
        labels, numbers = np.unique(np.asarray(block).ravel(), return_counts=True)
        for label, number in zip(labels.tolist(), numbers.tolist(), strict=True):
            counts[label] = counts.get(label, 0) + int(number)
    if not counts:
        return {"clusters": 0}
    sizes = sorted(counts.values(), reverse=True)
    return {
        "clusters": len(sizes),
        "largestCluster": sizes[0],
        "smallestCluster": sizes[-1],
    }


def _cluster_cut(group: Any) -> dict[str, Any]:
    if "n_clusters" not in group.attrs:
        return {}
    return {"clusters": int(group.attrs["n_clusters"])}


def _cluster_selection(group: Any) -> dict[str, Any]:
    keys = group.attrs.get("candidateKeys")
    scores = _array(group, "scores")
    summary: dict[str, Any] = {"selected": group.attrs.get("selectedKey")}
    if isinstance(keys, list | tuple) and scores is not None:
        values = np.asarray(scores[:], dtype=np.float64)
        summary["scores"] = {
            str(key): _float(value) for key, value in zip(keys, values, strict=True)
        }
    return summary


def _cell_cycle(group: Any) -> dict[str, Any]:
    phase = _array(group, "phase")
    if phase is None:
        return {}
    counts: dict[str, int] = {}
    for block in _blocks(phase):
        labels, numbers = np.unique(np.asarray(block).astype(str), return_counts=True)
        for label, number in zip(labels, numbers, strict=True):
            counts[str(label)] = counts.get(str(label), 0) + int(number)
    return {"phases": counts}


def _membership(group: Any) -> dict[str, Any]:
    values = _array(group, "values")
    if values is None or not values.shape[0]:
        return {}
    total = 0.0
    finite = 0
    below_half = 0
    for block in _blocks(values):
        numbers = np.asarray(block, dtype=np.float64)
        numbers = numbers[np.isfinite(numbers)]
        total += float(numbers.sum())
        finite += int(numbers.size)
        below_half += int(np.count_nonzero(numbers < 0.5))
    if not finite:
        return {}
    return {"mean": _float(total / finite), "belowHalf": below_half}


def _doublets(group: Any) -> dict[str, Any]:
    calls = _array(group, "predicted_doublets")
    summary: dict[str, Any] = {}
    if calls is not None:
        predicted = _true_count(calls)
        summary["predictedDoublets"] = predicted
        summary["rate"] = _float(predicted / max(1, int(calls.shape[0])))
    samples = group.attrs.get("samples")
    if isinstance(samples, list | tuple):
        summary["samples"] = [
            {
                key: sample[key]
                for key in ("labels", "cells", "threshold", "predictedDoublets")
                if isinstance(sample, Mapping) and key in sample
            }
            for sample in samples[:SUMMARY_ITEM_LIMIT]
        ]
    return summary


def _markers(group: Any) -> dict[str, Any]:
    skipped = group.attrs.get("skipped_groups")
    return {
        "groups": len(list(group.group_keys())),
        "skippedGroups": len(skipped) if isinstance(skipped, Mapping) else 0,
    }


def _integrated(group: Any) -> dict[str, Any]:
    assays = group.attrs.get("assays")
    summary: dict[str, Any] = {"assays": list(assays) if assays else []}
    weights = _array(group, "modality_weights")
    if weights is not None and assays:
        total = np.zeros(len(assays), dtype=np.float64)
        rows = 0
        for block in _blocks(weights):
            total += np.asarray(block, dtype=np.float64).sum(axis=0)
            rows += len(block)
        summary["meanWeights"] = {
            str(assay): _float(value / max(1, rows))
            for assay, value in zip(assays, total, strict=True)
        }
    return summary


_BY_KIND: Mapping[str, Callable[[Any], dict[str, Any]]] = {
    "feature_selection": _features,
    "normalized": lambda group: _shape(group, "data", ("cells", "features")),
    "reduction": _reduction,
    "batch_correction": _harmony,
    "neighbors": _neighbors,
    "connectivity_map": _neighbors,
    "embedding": lambda group: _shape(group, "values", ("cells", "dims")),
    "embedding_initialization": lambda group: _shape(
        group, "cluster_centers", ("centroids", "dims")
    ),
    "cluster_labels": _clusters,
    "cluster_cut": _cluster_cut,
    "cluster_selection": _cluster_selection,
    "cell_cycle": _cell_cycle,
    "membership_strength": _membership,
    "doublet_score": _doublets,
    "marker_table": _markers,
    "integrated_graph": _integrated,
    "hto_identity": _hashtags,
    "cell_calling": _cell_calling,
}


def summarize_stage(
    store: Any,
    stage: str,
    outputs: Sequence[tuple[str, ArtifactRef]],
) -> dict[str, Any] | None:
    """Return the diagnostics of one completed stage, keyed by its outputs.

    A stage with one summarized output records that summary itself; one with
    several records a summary per output key. Metadata snapshots and ANN
    indexes have none, so a stage that produced only those records None.
    """
    del stage
    root = store.zw
    summaries: dict[str, dict[str, Any]] = {}
    for key, ref in outputs:
        group = artifact_group(root, ref)
        if ref.kind == "cell_selection":
            summary = _selection(root, ref, group)
        else:
            summarize = _BY_KIND.get(ref.kind)
            if summarize is None:
                continue
            summary = summarize(group)
        if summary:
            summaries[key] = summary
    if not summaries:
        return None
    if len(summaries) == 1:
        return next(iter(summaries.values()))
    return summaries
