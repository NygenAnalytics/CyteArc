"""Stored differential-abundance results: slot layouts, writers, and readers.

A ``category_tests`` slot records the propeller design in its attributes and
holds the categories x samples ``counts``, the per-category statistics in a
``stats`` array whose columns ``stat_columns`` names, and the empirical Bayes
``df_prior`` and ``s2_prior``. Proportions are recomputed from the counts.
"""

from typing import Any

import numpy as np
import pandas as pd
import zarr

from ...abundance.propeller import PropellerFit, transformed_proportions
from ...abundance.results import (
    CategoryTestResult,
    NeighborhoodCounts,
    Neighborhoods,
    NeighborhoodTestResult,
)
from ...metadata.arguments import CategoryTestArguments
from ...metadata.selection import CellField
from ...storage.artifacts import ArtifactRef
from ...storage.types import as_zarr_array

_CATEGORY_SLOT_ATTRIBUTES = frozenset(
    {
        "categories",
        "category_field",
        "cell_selection",
        "condition_by",
        "groups",
        "method",
        "n_cells",
        "robust",
        "robust_used",
        "sample_by",
        "sample_groups",
        "samples",
        "stat_columns",
        "transform",
        "trend",
        "category_labels",
    }
)


def _write_float_array(group: zarr.Group, name: str, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values, dtype=np.float64)
    target = group.create_array(name, shape=array.shape, dtype=np.float64)
    if array.size:
        target[...] = array


def write_category_slot(
    group: zarr.Group,
    result: CategoryTestResult,
    *,
    arguments: CategoryTestArguments,
) -> None:
    """Write a propeller result into its started slot."""
    fit = result.fit
    columns = [str(column) for column in fit.table.columns]
    group.attrs.update(
        {
            "method": fit.method,
            "transform": fit.transform,
            "robust": arguments.robust,
            "robust_used": fit.robust,
            "trend": fit.trend,
            "category_labels": list(fit.categories),
            "samples": list(fit.samples),
            "groups": list(fit.groups),
            "sample_groups": list(fit.sample_groups),
            "stat_columns": columns,
            "sample_by": arguments.sample_by,
            "condition_by": arguments.condition_by,
            "category_field": arguments.category_field,
            "categories": (
                arguments.categories.to_dict()
                if arguments.categories is not None
                else None
            ),
            "cell_selection": (
                arguments.cell_selection.to_dict()
                if arguments.cell_selection is not None
                else None
            ),
            "n_cells": arguments.n_cells,
        }
    )
    counts = np.ascontiguousarray(fit.counts, dtype=np.int64)
    target = group.create_array("counts", shape=counts.shape, dtype=np.int64)
    target[...] = counts
    _write_float_array(group, "stats", fit.table.to_numpy(dtype=np.float64))
    _write_float_array(group, "df_prior", fit.df_prior)
    _write_float_array(group, "s2_prior", fit.s2_prior)


def _string_list(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"Category test {name} metadata is invalid")
    return tuple(value)


def read_category_slot(
    slot_group: zarr.Group,
    *,
    artifact: ArtifactRef,
) -> CategoryTestResult:
    """Read the propeller result that a complete slot stores."""
    attrs: dict[str, Any] = dict(slot_group.attrs)
    missing = sorted(_CATEGORY_SLOT_ATTRIBUTES.difference(attrs))
    if missing:
        raise ValueError(
            "Category test artifact is missing metadata "
            f"({', '.join(missing)}); rerun abundance.test_categories"
        )
    categories = _string_list(attrs["category_labels"], "category")
    samples = _string_list(attrs["samples"], "sample")
    groups = _string_list(attrs["groups"], "group")
    sample_groups = _string_list(attrs["sample_groups"], "sample group")
    columns = _string_list(attrs["stat_columns"], "column")
    counts = np.asarray(as_zarr_array(slot_group["counts"], name="counts")[:])
    stats = np.asarray(as_zarr_array(slot_group["stats"], name="stats")[:])
    if counts.shape != (len(categories), len(samples)) or stats.shape != (
        len(categories),
        len(columns),
    ):
        raise ValueError("Category test arrays do not match their metadata")
    proportions, transformed = transformed_proportions(counts, attrs["transform"])
    proportions_fit = PropellerFit(
        method=attrs["method"],
        transform=attrs["transform"],
        robust=bool(attrs["robust_used"]),
        trend=bool(attrs["trend"]),
        categories=categories,
        samples=samples,
        groups=groups,
        sample_groups=sample_groups,
        counts=counts,
        proportions=proportions,
        transformed=transformed,
        df_prior=np.asarray(as_zarr_array(slot_group["df_prior"], name="df_prior")[:]),
        s2_prior=np.asarray(as_zarr_array(slot_group["s2_prior"], name="s2_prior")[:]),
        table=pd.DataFrame(
            stats, index=pd.Index(categories, name="category"), columns=list(columns)
        ),
    )
    categories_ref = attrs["categories"]
    cell_selection = attrs["cell_selection"]
    category_field = attrs["category_field"]
    return CategoryTestResult(
        fit=proportions_fit,
        categories=(
            ArtifactRef.from_dict(categories_ref)
            if categories_ref is not None
            else None
        ),
        category_field=(
            CellField(category_field, kind="categorical")
            if category_field is not None
            else None
        ),
        cell_selection=(
            ArtifactRef.from_dict(cell_selection)
            if cell_selection is not None
            else None
        ),
        sample_by=str(attrs["sample_by"]),
        condition_by=str(attrs["condition_by"]),
        robust_requested=bool(attrs["robust"]),
        n_cells=int(attrs["n_cells"]),
        artifact=artifact,
    )


# Milo artifacts. A ``neighborhoods`` slot holds ``index_rows`` (positions in
# the graph's cell selection), ``index_cells`` (physical rows), ``sizes``,
# ``kth_distance``, and the cell-major membership CSR ``membership_indptr``
# (graph cells + 1) and ``membership_indices`` (neighbourhood ids). A
# ``neighborhood_counts`` slot holds ``counts`` (neighbourhoods x samples) and
# the sample labels in its attributes. A ``neighborhood_tests`` slot holds the
# per-neighbourhood ``stats`` named by ``stat_columns``, the ``tested`` mask,
# the ``design`` matrix, and the per-sample ``norm_factors`` and ``lib_size``.

NEIGHBORHOOD_ARRAYS = (
    "index_rows",
    "index_cells",
    "sizes",
    "kth_distance",
    "membership_indptr",
    "membership_indices",
)
NEIGHBORHOOD_TEST_COLUMNS = (
    "log_fc",
    "log_cpm",
    "f_statistic",
    "p_value",
    "fdr",
    "spatial_fdr",
)


def write_small_array(group: zarr.Group, name: str, values: np.ndarray) -> None:
    """Write a whole array that fits comfortably in memory."""
    array = np.ascontiguousarray(values)
    target = group.create_array(name, shape=array.shape, dtype=array.dtype)
    if array.size:
        target[...] = array


def read_neighborhoods_slot(
    slot_group: zarr.Group, *, artifact: ArtifactRef
) -> Neighborhoods:
    """Read the neighbourhoods that a complete slot stores."""
    attrs: dict[str, Any] = dict(slot_group.attrs)
    required = {
        "neighbors",
        "cell_selection",
        "k",
        "n_cells",
        "prop",
        "refined",
        "seed",
    }
    missing = sorted(required.difference(attrs))
    if missing:
        raise ValueError(
            "Neighbourhood artifact is missing metadata "
            f"({', '.join(missing)}); rerun abundance.make_neighborhoods"
        )

    def read(name: str) -> np.ndarray:
        return np.asarray(as_zarr_array(slot_group[name], name=name)[:])

    return Neighborhoods(
        artifact=artifact,
        neighbors=ArtifactRef.from_dict(attrs["neighbors"]),
        cell_selection=ArtifactRef.from_dict(attrs["cell_selection"]),
        index_rows=read("index_rows"),
        index_cells=read("index_cells"),
        sizes=read("sizes"),
        kth_distance=read("kth_distance"),
        k=int(attrs["k"]),
        n_cells=int(attrs["n_cells"]),
        prop=float(attrs["prop"]),
        refined=bool(attrs["refined"]),
        seed=int(attrs["seed"]),
    )


def read_neighborhood_counts_slot(
    slot_group: zarr.Group, *, artifact: ArtifactRef
) -> NeighborhoodCounts:
    """Read the neighbourhood x sample counts that a complete slot stores."""
    attrs: dict[str, Any] = dict(slot_group.attrs)
    samples = _string_list(attrs.get("samples"), "sample")
    counts = np.asarray(as_zarr_array(slot_group["counts"], name="counts")[:])
    if counts.ndim != 2 or counts.shape[1] != len(samples):
        raise ValueError("Neighbourhood counts do not match their sample metadata")
    return NeighborhoodCounts(
        artifact=artifact,
        neighborhoods=ArtifactRef.from_dict(attrs["neighborhoods"]),
        sample_by=str(attrs["sample_by"]),
        samples=samples,
        counts=counts,
    )


def write_neighborhood_test_slot(
    group: zarr.Group,
    result: NeighborhoodTestResult,
    *,
    tested: np.ndarray,
) -> None:
    """Write a Milo test into its started slot."""
    group.attrs.update(
        {
            "stat_columns": list(NEIGHBORHOOD_TEST_COLUMNS),
            "design_columns": [str(column) for column in result.design.columns],
            "samples": [str(sample) for sample in result.design.index],
            "contrast": {key: float(value) for key, value in result.contrast.items()},
            "counts": result.counts.to_dict(),
            "neighborhoods": result.neighborhoods.to_dict(),
            "sample_by": result.sample_by,
            "condition_by": result.condition_by,
            "covariates": list(result.covariates),
            "common_dispersion": float(result.common_dispersion),
            "average_ql_dispersion": result.average_ql_dispersion,
            "norm_method": result.norm_method,
            "min_mean": float(result.min_mean),
            "robust": bool(result.robust),
            "legacy": bool(result.legacy),
            "fdr_weighting": result.fdr_weighting,
        }
    )
    table = result.table
    write_small_array(
        group,
        "stats",
        table[list(NEIGHBORHOOD_TEST_COLUMNS)].to_numpy(dtype=np.float64),
    )
    write_small_array(group, "tested", np.asarray(tested, dtype=bool))
    write_small_array(group, "index_cells", table["index_cell"].to_numpy(np.int64))
    write_small_array(group, "sizes", table["size"].to_numpy(np.int64))
    write_small_array(group, "design", result.design.to_numpy(dtype=np.float64))
    write_small_array(group, "norm_factors", result.norm_factors.to_numpy(np.float64))
    write_small_array(group, "lib_size", result.lib_size.to_numpy(np.float64))


def read_neighborhood_test_slot(
    slot_group: zarr.Group, *, artifact: ArtifactRef
) -> NeighborhoodTestResult:
    """Read the Milo test that a complete slot stores."""
    attrs: dict[str, Any] = dict(slot_group.attrs)
    columns = _string_list(attrs.get("stat_columns"), "column")
    design_columns = _string_list(attrs.get("design_columns"), "design column")
    samples = _string_list(attrs.get("samples"), "sample")

    def read(name: str) -> np.ndarray:
        return np.asarray(as_zarr_array(slot_group[name], name=name)[:])

    stats = read("stats")
    n_nhoods = stats.shape[0]
    table = pd.DataFrame(
        {
            "index_cell": read("index_cells"),
            "size": read("sizes"),
            **{column: stats[:, idx] for idx, column in enumerate(columns)},
            "tested": read("tested"),
        },
        index=pd.RangeIndex(n_nhoods, name="neighborhood"),
    )
    sample_index = pd.Index(samples, name=str(attrs["sample_by"]))
    return NeighborhoodTestResult(
        counts=ArtifactRef.from_dict(attrs["counts"]),
        neighborhoods=ArtifactRef.from_dict(attrs["neighborhoods"]),
        sample_by=str(attrs["sample_by"]),
        condition_by=str(attrs["condition_by"]),
        covariates=tuple(attrs["covariates"]),
        design=pd.DataFrame(
            read("design"), index=sample_index, columns=list(design_columns)
        ),
        contrast={str(key): float(value) for key, value in attrs["contrast"].items()},
        table=table,
        norm_factors=pd.Series(
            read("norm_factors"), index=sample_index, name="norm_factor"
        ),
        lib_size=pd.Series(read("lib_size"), index=sample_index, name="lib_size"),
        common_dispersion=float(attrs["common_dispersion"]),
        average_ql_dispersion=attrs["average_ql_dispersion"],
        norm_method=str(attrs["norm_method"]),
        min_mean=float(attrs["min_mean"]),
        robust=bool(attrs["robust"]),
        legacy=bool(attrs["legacy"]),
        fdr_weighting=str(attrs["fdr_weighting"]),
        artifact=artifact,
    )
