"""Small diagnostics that a run records for each completed stage.

Each summary is computed from the artifacts that the stage produced, so a
reused artifact reports the same numbers as a computed one. Summaries read
attributes and bounded blocks of small arrays, never a matrix.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np

from ...metadata.rows import array_block_rows
from ...perturbation.edistance import EDISTANCE_STATUSES
from ...perturbation.efficiency import EFFICIENCY_STATUSES
from ...perturbation.mixscape import MIXSCAPE_STATUSES
from ...perturbation.mixtures import FIT_STATUSES
from ...storage.artifacts import ArtifactRef, artifact_group, inspect_artifact
from ...storage.types import as_zarr_array
from .._operations.perturbation import _status_labels

# A summary lists at most this many samples, clusters, or components.
SUMMARY_ITEM_LIMIT = 50
# A target with fewer single-target cells than this is named in the labels'
# summary: the fewest cells of a perturbation profile by default, as in
# Replogle et al. (2022).
FEW_TARGET_CELLS = 25
# The adjusted p-value at or below which a perturbation's test counts as
# significant in a summary, as a pair of the association tests does.
SIGNIFICANT_P = 0.05


def _array(group: Any, name: str) -> Any | None:
    if name not in group:
        return None
    return as_zarr_array(group[name], name=name)


def _arrays(group: Any, *names: str) -> list[Any] | None:
    """Return the arrays of ``names``, or None when one of them is missing."""
    if any(name not in group for name in names):
        return None
    return [as_zarr_array(group[name], name=name) for name in names]


def _blocks(array: Any) -> Any:
    rows = int(array.shape[0])
    block = array_block_rows(array, rows)
    for start in range(0, rows, block):
        yield np.asarray(array[start : min(start + block, rows)])


def _true_count(array: Any) -> int:
    return sum(int(np.count_nonzero(block)) for block in _blocks(array))


def _row_ranges(array: Any) -> Iterator[tuple[int, int]]:
    """Yield bounded row ranges of ``array``, to read arrays of its rows alike."""
    rows = int(array.shape[0])
    block = array_block_rows(array, rows)
    for start in range(0, rows, block):
        yield start, min(start + block, rows)


def _status_counts(labels: np.ndarray, statuses: Sequence[str]) -> dict[str, int]:
    """Count each of ``statuses`` among the decoded ``labels``."""
    return {name: int(np.count_nonzero(labels == name)) for name in statuses}


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


def _guide_thresholds(group: Any) -> dict[str, Any]:
    status = _array(group, "status")
    if status is None or status.ndim != 2:
        return {}
    codes = np.concatenate(
        [np.asarray(block).ravel() for block in _blocks(status)]
        or [np.empty(0, dtype=np.int64)]
    )
    return {
        "samples": int(status.shape[0]),
        "guides": int(status.shape[1]),
        "statuses": _status_counts(
            _status_labels(group, FIT_STATUSES, codes), FIT_STATUSES
        ),
    }


def _guide_calls(group: Any) -> dict[str, Any]:
    arrays = _arrays(group, "n_guides", "guide_codes", "guides")
    if arrays is None:
        return {}
    n_guides, codes, guides = arrays
    # Cells with no call, one call, and several calls.
    calls = np.zeros(3, dtype=np.int64)
    for block in _blocks(n_guides):
        calls += np.bincount(
            np.minimum(np.asarray(block, dtype=np.int64), 2), minlength=3
        )
    called = np.zeros(int(guides.shape[0]), dtype=bool)
    for block in _blocks(codes):
        values = np.asarray(block, dtype=np.int64).ravel()
        called[values[values >= 0]] = True
    summary: dict[str, Any] = {
        "cells": int(calls.sum()),
        "negative": int(calls[0]),
        "single": int(calls[1]),
        "multiple": int(calls[2]),
        "calledGuides": int(np.count_nonzero(called)),
    }
    # Imported calls count the barcodes of Cell Ranger's file.
    for key, name in (
        ("fileBarcodes", "n_file_barcodes"),
        ("unmatchedBarcodes", "n_unmatched_barcodes"),
    ):
        if name in group.attrs:
            summary[key] = int(group.attrs[name])
    return summary


def _perturbation_labels(group: Any) -> dict[str, Any]:
    arrays = _arrays(
        group, "n_targets", "is_control", "target_codes", "targets", "target_is_control"
    )
    if arrays is None:
        return {}
    n_targets, is_control, target_codes, targets, target_is_control = arrays
    names = np.asarray(targets[:]).astype(str)
    rows = ~np.asarray(target_is_control[:]).astype(bool)
    # Each target's cells that carry it alone, and the cells of each label.
    single_cells = np.zeros(len(names), dtype=np.int64)
    counts = np.zeros(4, dtype=np.int64)
    for start, stop in _row_ranges(n_targets):
        n = np.asarray(n_targets[start:stop]).astype(np.int64)
        control = np.asarray(is_control[start:stop]).astype(bool)
        first = np.asarray(target_codes[start:stop, 0]).astype(np.int64)
        single_cells += np.bincount(first[n == 1], minlength=len(names))
        counts += np.asarray(
            [
                len(n),
                np.count_nonzero(control),
                np.count_nonzero((n == 0) & ~control),
                np.count_nonzero(n > 1),
            ],
            dtype=np.int64,
        )
    target_cells = single_cells[rows]
    target_names = names[rows]
    few = np.flatnonzero(target_cells < FEW_TARGET_CELLS)
    few = few[np.lexsort((target_names[few], target_cells[few]))]
    return {
        "cells": int(counts[0]),
        "controls": int(counts[1]),
        "negative": int(counts[2]),
        "multipleTargets": int(counts[3]),
        "targets": int(len(target_names)),
        "targetCells": (
            {
                "min": int(target_cells.min()),
                "median": _float(np.median(target_cells)),
                "max": int(target_cells.max()),
            }
            if len(target_cells)
            else None
        ),
        "fewCellTargets": {
            str(target_names[row]): int(target_cells[row])
            for row in few[:SUMMARY_ITEM_LIMIT]
        },
        "fewCellTargetCount": int(len(few)),
    }


def _row_table(group: Any, names: Sequence[str]) -> dict[str, np.ndarray] | None:
    """Read the arrays of a table with one row per perturbation, or None."""
    arrays = _arrays(group, "status", *names)
    if arrays is None:
        return None
    return {
        name: np.asarray(array[:])
        for name, array in zip(("status", *names), arrays, strict=True)
    }


def _efficiency(group: Any) -> dict[str, Any]:
    table = _row_table(group, ("names", "remaining_fraction", "adjusted_p_value"))
    if table is None:
        return {}
    labels = _status_labels(group, EFFICIENCY_STATUSES, table["status"])
    tested = labels == "tested"
    fraction = table["remaining_fraction"].astype(np.float64)
    adjusted = table["adjusted_p_value"].astype(np.float64)
    measured = fraction[tested & np.isfinite(fraction)]
    unmatched = table["names"].astype(str)[labels == "no_feature"]
    return {
        "by": str(group.attrs.get("by")),
        "rows": int(len(labels)),
        "statuses": _status_counts(labels, EFFICIENCY_STATUSES),
        "knockedDown": int(
            np.count_nonzero(tested & (adjusted <= SIGNIFICANT_P) & (fraction < 1))
        ),
        "medianRemainingFraction": (
            _float(np.median(measured)) if len(measured) else None
        ),
        "noFeature": unmatched[:SUMMARY_ITEM_LIMIT].tolist(),
    }


def _edistance(group: Any, parameters: Mapping[str, Any]) -> dict[str, Any]:
    table = _row_table(group, ("p_value", "adjusted_p_value"))
    if table is None:
        return {}
    labels = _status_labels(group, EDISTANCE_STATUSES, table["status"])
    tested = labels == "tested"
    permutations = int(parameters["n_permutations"])
    # No draw reached the observed E-distance: the smallest p-value a test of
    # these permutations gives. Without permutations p-values are NaN.
    floor = 1.0 / (permutations + 1)
    p_value = table["p_value"].astype(np.float64)
    adjusted = table["adjusted_p_value"].astype(np.float64)
    return {
        "by": str(group.attrs.get("by")),
        "rows": int(len(labels)),
        "statuses": _status_counts(labels, EDISTANCE_STATUSES),
        "significant": int(np.count_nonzero(tested & (adjusted <= SIGNIFICANT_P))),
        "atFloor": int(np.count_nonzero(tested & (p_value <= floor * (1 + 1e-9)))),
        "nPermutations": permutations,
    }


def _signature(group: Any) -> dict[str, Any]:
    arrays = _arrays(group, "values", "neighbors", "is_control")
    if arrays is None:
        return {}
    values, neighbors, is_control = arrays
    # The samples of the sample column, which the loader's cells name. Without
    # a sample column every cell is in one sample named None, which counts as
    # none, as pandas counts the loader's labels.
    samples = group.attrs.get("sample_names") or ()
    return {
        "cells": int(values.shape[0]),
        "features": int(values.shape[1]),
        "controls": _true_count(is_control),
        "samples": sum(name is not None for name in samples),
        "neighbors": int(neighbors.shape[1]),
    }


def _mixscape(group: Any, parameters: Mapping[str, Any]) -> dict[str, Any]:
    table = _row_table(group, ("targets", "n_cells", "n_ko"))
    if table is None:
        return {}
    labels = _status_labels(group, MIXSCAPE_STATUSES, table["status"])
    few = labels == "few_de_genes"
    return {
        "rows": int(len(labels)),
        "statuses": _status_counts(labels, MIXSCAPE_STATUSES),
        "cells": int(table["n_cells"].sum()),
        "knockedOut": int(table["n_ko"].sum()),
        "fewDeGenes": sorted(set(table["targets"].astype(str)[few].tolist()))[
            :SUMMARY_ITEM_LIMIT
        ],
        "perturbationType": str(parameters["perturbation_type"]),
    }


def _associations(group: Any, parameters: Mapping[str, Any]) -> dict[str, Any]:
    arrays = _arrays(group, "sets", "pass_qc", "significant", "groups", "group_guides")
    if arrays is None:
        return {}
    sets, pass_qc, significant, pair_groups, group_guides = arrays
    names = [str(name) for name in group.attrs["set_names"]]
    discovery, calibration, positive = (
        names.index(name) for name in ("discovery", "calibration", "positive")
    )
    # Each set's pairs, those that pass QC, and those found significant, read
    # in bounded blocks, since a trans run holds millions of pairs.
    pairs = np.zeros(len(names), dtype=np.int64)
    passing = np.zeros(len(names), dtype=np.int64)
    found = np.zeros(len(names), dtype=np.int64)
    false_groups: list[np.ndarray] = []
    for start, stop in _row_ranges(sets):
        codes = np.asarray(sets[start:stop]).astype(np.int64)
        qc = np.asarray(pass_qc[start:stop]).astype(bool)
        hits = np.asarray(significant[start:stop]).astype(bool)
        pairs += np.bincount(codes, minlength=len(names))
        passing += np.bincount(codes[qc], minlength=len(names))
        found += np.bincount(codes[hits], minlength=len(names))
        false = hits & (codes == calibration)
        if false.any():
            false_groups.append(np.asarray(pair_groups[start:stop])[false])
    # The false discoveries of each group of control guides, most first.
    guides = np.asarray(group_guides[:]).astype(str)
    labels, counts = np.unique(
        guides[np.concatenate(false_groups or [np.empty(0, dtype=np.int64)])],
        return_counts=True,
    )
    order = np.lexsort((labels, -counts))[:SUMMARY_ITEM_LIMIT]
    return {
        "discoveryPairs": int(pairs[discovery]),
        "testedDiscoveryPairs": int(passing[discovery]),
        "discoveries": int(found[discovery]),
        "calibrationPairs": int(pairs[calibration]),
        "falseDiscoveries": int(group.attrs["n_false_discoveries"]),
        "calibrationGrade": str(group.attrs["calibration_grade"]),
        "calibrationStatus": str(group.attrs["calibration_status"]),
        "falseDiscoveriesByControlGuides": {
            str(labels[index]): int(counts[index]) for index in order
        },
        "positivePairs": int(pairs[positive]),
        "positiveDetected": int(found[positive]),
        "alpha": float(parameters["alpha"]),
    }


def _profiles(group: Any) -> dict[str, Any]:
    arrays = _arrays(group, "values", "left_out", "n_zero_sd")
    if arrays is None:
        return {}
    values, left_out, n_zero_sd = arrays
    return {
        "perturbations": int(values.shape[0]),
        "features": int(values.shape[1]),
        "leftOut": int(left_out.shape[0]),
        "zeroSdFeatures": int(np.count_nonzero(np.asarray(n_zero_sd[:]))),
        "samples": len(group.attrs.get("sample_names") or ()),
    }


def _perturbation_clusters(group: Any) -> dict[str, Any]:
    cluster = _array(group, "cluster")
    if cluster is None:
        return {}
    labels = np.asarray(cluster[:], dtype=np.int64)
    sizes = np.bincount(labels[labels >= 0])
    return {
        "perturbations": int(len(labels)),
        "clusters": int(np.count_nonzero(sizes)),
        "unassigned": int(np.count_nonzero(labels < 0)),
        "largestCluster": int(sizes.max(initial=0)),
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
    "guide_thresholds": _guide_thresholds,
    "guide_calls": _guide_calls,
    "perturbation_labels": _perturbation_labels,
    "perturbation_efficiency": _efficiency,
    "perturbation_signature": _signature,
    "perturbation_profiles": _profiles,
    "perturbation_clusters": _perturbation_clusters,
}
# The summaries that also read the artifact's recorded parameters.
_BY_KIND_WITH_PARAMETERS: Mapping[
    str, Callable[[Any, Mapping[str, Any]], dict[str, Any]]
] = {
    "perturbation_edistance": _edistance,
    "mixscape_classes": _mixscape,
    "perturbation_associations": _associations,
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
        elif ref.kind in _BY_KIND_WITH_PARAMETERS:
            summary = _BY_KIND_WITH_PARAMETERS[ref.kind](
                group, inspect_artifact(root, ref).parameters or {}
            )
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
