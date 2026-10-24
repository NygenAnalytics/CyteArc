import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from numbers import Real
from types import MappingProxyType
from typing import Any

import numpy as np

from ...clustering.leiden import canonical_resolution
from ...embeddings import sgtsne
from ...features.gene_reference import species_registry
from ...features.variability import (
    HVG_FLAVOR_OPTION_NAMES,
    HVG_OPTION_NAMES,
    hvg_flavor_options,
    hvg_options,
)
from ...graph.arguments import graph_load_flags, umap_graph_flags
from ...quality_control.cell_calling import validated_fdr, validated_settings
from ...quality_control.hto import _validated_positive_quantile
from ...utils.arguments import integer_argument
from ...utils.logging import logger
from .._operations.cell_calling import _CellCallingOperationsMixin
from .._operations.embeddings import _EmbeddingOperationsMixin
from .._operations.features import _FeatureOperationsMixin
from .._operations.graph import _GraphOperationsMixin
from .._operations.quality_control import _QualityControlOperationsMixin


def keyword_default(method: Callable[..., Any], keyword: str) -> Any:
    """Return the default of one keyword of a stage's method."""
    default = inspect.signature(method).parameters[keyword].default
    if default is inspect.Parameter.empty:
        raise TypeError(f"{method.__qualname__}() has no default for {keyword!r}")
    return default


# The features.hvgs options that hvg settings omit; a recipe checks a setting
# together with these, as features.hvgs checks its keywords.
_HVG_OPTION_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        name: keyword_default(_FeatureOperationsMixin._features_hvgs, name)
        for name in HVG_OPTION_NAMES
    }
)
_HVG_FLAVOR_OPTION_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        name: keyword_default(_FeatureOperationsMixin._features_hvgs, name)
        for name in HVG_FLAVOR_OPTION_NAMES
    }
)
# The embedding dimensions, and so the run fields, of settings that omit them.
UMAP_DIMS_DEFAULT: int = keyword_default(
    _EmbeddingOperationsMixin._embeddings_umap, "umap_dims"
)
TSNE_DIMS_DEFAULT: int = keyword_default(
    _EmbeddingOperationsMixin._embeddings_tsne, "tsne_dims"
)
PCA_DIMS_DEFAULT: int = keyword_default(_GraphOperationsMixin._reduction_pca, "dims")
HVG_TOP_N_DEFAULT: int = _HVG_OPTION_DEFAULTS["top_n"]
PEAKS_TOP_N_DEFAULT: int = keyword_default(
    _QualityControlOperationsMixin._features_prevalent_peaks, "top_n"
)
LSI_DIMS_DEFAULT: int = keyword_default(_GraphOperationsMixin._reduction_lsi, "dims")
LSI_SKIP_FIRST_DEFAULT: bool = keyword_default(
    _GraphOperationsMixin._reduction_lsi, "skip_first"
)
# The numeric embeddings.tsne settings, which sgtsne_settings checks, at the
# embeddings.tsne defaults that tsne settings omit.
_TSNE_SETTING_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        name: keyword_default(_EmbeddingOperationsMixin._embeddings_tsne, name)
        for name in inspect.signature(sgtsne.sgtsne_settings).parameters
    }
)
# The graph-loading keywords of embeddings.tsne, embeddings.umap, and
# clusters.leiden, which graph_load_flags and umap_graph_flags check.
_GRAPH_FLAGS = ("symmetric_graph", "graph_upper_only")
# The default Leiden candidates of a recipe that scores them by silhouette.
DEFAULT_LEIDEN_PARTITIONS: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25)

# The settings that each stage keyword accepts. The pipeline supplies the
# artifacts a stage consumes, so only its settings appear here. A setting
# means exactly what the same keyword means on the stage's method, and
# omitting it uses that method's default. Some keys configure the recipe
# instead: Leiden ``partitions`` and ``selected``, the ADT panel's
# ``exclude``, the HTO ``assay`` of hashing, and the capture
# ``sample_column`` of cell calling. Harmony's ``batch_columns`` is the
# required keyword of its method.
STAGE_SETTINGS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "cell_calling": frozenset(
            {
                "lower",
                "retain",
                "n_iter",
                "alpha",
                "fdr",
                "random_seed",
                "sample_column",
            }
        ),
        "hashing": frozenset({"assay", "random_seed", "positive_quantile"}),
        "cell_cycle": frozenset(
            {
                "s_genes",
                "g2m_genes",
                "ctrl_size",
                "log_transform",
                "n_bins",
                "rand_seed",
            }
        ),
        "hvg": frozenset(
            {
                "top_n",
                "min_cells",
                "max_cells",
                "min_mean",
                "max_mean",
                "min_var",
                "max_var",
                "n_bins",
                "lowess_frac",
                "blacklist",
                "keep_bounds",
                "bin_strategy",
                "flavor",
                "batch",
                "span",
                "theta",
                "clip",
            }
        ),
        "prevalent_peaks": frozenset({"top_n"}),
        "panel": frozenset({"exclude"}),
        "normalization": frozenset({"log_transform", "renormalize_subset"}),
        "pca": frozenset({"dims", "feat_scaling", "batch_size"}),
        "lsi": frozenset(
            {
                "dims",
                "skip_first",
                "rand_state",
                "solver",
                "n_iter",
                "n_oversamples",
                "batch_size",
            }
        ),
        "harmony": frozenset({"batch_columns", "harmony_params", "batch_size"}),
        "ann_index": frozenset(
            {
                "ann_metric",
                "ann_efc",
                "ann_ef",
                "ann_m",
                "ann_parallel",
                "rand_state",
                "batch_size",
            }
        ),
        "neighbors": frozenset({"k", "ann_ef", "batch_size"}),
        "connectivity": frozenset({"local_connectivity", "bandwidth"}),
        "integration": frozenset({"chunk_size", "l2_normalize"}),
        "embedding_initialization": frozenset(
            {
                "n_centroids",
                "rand_state",
                "batch_size",
                "kmeans_sampling",
                "kmeans_batch_size",
            }
        ),
        "umap": frozenset(
            {
                "symmetric_graph",
                "graph_upper_only",
                "umap_dims",
                "spread",
                "min_dist",
                "n_epochs",
                "repulsion_strength",
                "initial_alpha",
                "negative_sample_rate",
                "use_density_map",
                "dens_lambda",
                "dens_frac",
                "dens_var_shift",
                "random_seed",
                "parallel",
            }
        ),
        "tsne": frozenset(
            {
                "symmetric_graph",
                "graph_upper_only",
                "tsne_dims",
                "lambda_scale",
                "max_iter",
                "early_iter",
                "alpha",
                "box_h",
            }
        ),
        "leiden": frozenset(
            {
                "partitions",
                "selected",
                "backend",
                "symmetric_graph",
                "graph_upper_only",
                "random_seed",
            }
        ),
        "membership_strength": frozenset(),
        "paris": frozenset({"n_clusters", "min_cluster_size"}),
        "doublets": frozenset(
            {
                "cluster_sample_fraction",
                "max_cells_per_cluster",
                "sample_column",
                "simulation_ratio",
                "heterotypic_fraction",
                "save_k",
                "expected_doublet_rate",
                "threshold",
                "smoothing_t",
                "normalize_scores",
                "random_seed",
            }
        ),
        "markers": frozenset(),
    }
)


def parameter_value(value: Any, name: str) -> Any:
    """Return a JSON-compatible copy of one stage setting for the run record."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"{name} must be finite; stage settings are JSON values and "
                "never infinity or NaN"
            )
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{name} keys must be strings")
        return {
            key: parameter_value(item, f"{name}[{key!r}]")
            for key, item in value.items()
        }
    if isinstance(value, Sequence | np.ndarray) and not isinstance(value, bytes):
        return [
            parameter_value(item, f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{name} must be a number, string, boolean, None, list, or mapping")


def column_sequence(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of column names")
    columns = tuple(value)
    if any(not isinstance(column, str) or not column for column in columns):
        raise TypeError(f"{name} must contain non-empty strings")
    if len(columns) != len(set(columns)):
        raise ValueError(f"{name} must not contain duplicates")
    return columns


def finite_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be a finite number")
    return resolved


# The integer settings that a recipe checks before a run record exists.
_POSITIVE_INTEGER_SETTINGS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "neighbors": ("k",),
        "prevalent_peaks": ("top_n",),
        "lsi": ("dims",),
    }
)
# The qc.call_cells settings at the defaults that cell_calling settings omit.
_CELL_CALLING_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        name: keyword_default(_CellCallingOperationsMixin._qc_call_cells, name)
        for name in ("lower", "retain", "n_iter", "alpha", "fdr", "random_seed")
    }
)


def _check_stage_values(stage: str, value: Mapping[str, Any]) -> None:
    """Run a stage method's own checks on settings, before a run record exists.

    Late stages would otherwise fail only after the stages before them.
    """
    if stage == "hvg":
        hvg_options(
            **{
                name: value.get(name, default)
                for name, default in _HVG_OPTION_DEFAULTS.items()
            }
        )
        hvg_flavor_options(
            **{
                name: value.get(name, default)
                for name, default in _HVG_FLAVOR_OPTION_DEFAULTS.items()
            }
        )
        if not isinstance(value.get("batch"), str | None):
            raise TypeError("hvg['batch'] must be a cell metadata column")
    elif stage in {"tsne", "umap", "leiden"}:
        check_flags = umap_graph_flags if stage == "umap" else graph_load_flags
        check_flags(*(value.get(flag, False) for flag in _GRAPH_FLAGS))
        if stage == "tsne":
            sgtsne.sgtsne_settings(
                **{
                    name: value.get(name, default)
                    for name, default in _TSNE_SETTING_DEFAULTS.items()
                }
            )
    elif stage == "harmony" and "harmony_params" in value:
        from ...embeddings.harmony import validate_harmony_parameters

        validate_harmony_parameters(value["harmony_params"])
    elif stage == "cell_calling":
        options = {
            name: value.get(name, default)
            for name, default in _CELL_CALLING_DEFAULTS.items()
        }
        validated_fdr(options.pop("fdr"))
        validated_settings(**options)
    elif stage == "hashing" and "positive_quantile" in value:
        _validated_positive_quantile(value["positive_quantile"])
    # The counts and sizes that these stages take must be positive integers.
    for key in _POSITIVE_INTEGER_SETTINGS.get(stage, ()):
        if key in value:
            integer_argument(value[key], f"{stage}['{key}']", minimum=1)


def stage_settings(
    stage: str,
    value: object,
    *,
    optional: bool,
    allowed: frozenset[str] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Validate one stage keyword into whether it runs and its settings.

    An optional stage takes a bool, which switches it, or a mapping of
    settings, which enables it. A stage that always runs takes None for its
    defaults or a mapping of settings.
    """
    if allowed is None:
        allowed = STAGE_SETTINGS[stage]
    if isinstance(value, bool):
        if not optional:
            raise TypeError(f"{stage} must be a mapping because the stage always runs")
        return value, {}
    if value is None and not optional:
        return True, {}
    if not isinstance(value, Mapping):
        expected = "a mapping or bool" if optional else "a mapping or None"
        raise TypeError(f"{stage} must be {expected}")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown {stage} settings {unknown!r}; allowed: {sorted(allowed)!r}"
        )
    _check_stage_values(stage, value)
    return True, {
        key: parameter_value(item, f"{stage}[{key!r}]") for key, item in value.items()
    }


def resolve_species(species: object) -> str | None:
    if species is None:
        return None
    known = species_registry()
    if not isinstance(species, str) or species not in known:
        raise ValueError(f"species must be one of {sorted(known)!r}")
    return species


def resolve_leiden(
    value: object,
    *,
    default_partitions: tuple[float, ...] = DEFAULT_LEIDEN_PARTITIONS,
) -> tuple[tuple[tuple[str, float], ...], str | None, dict[str, Any]]:
    """Return the Leiden candidates, the selected key, and the other settings.

    The candidates are ``(key, resolution)`` pairs, empty when Leiden is off.
    """
    enabled, settings = stage_settings("leiden", value, optional=True)
    if not enabled:
        return (), None, {}
    raw_partitions = settings.pop("partitions", None)
    if raw_partitions is None:
        raw_partitions = list(default_partitions)
    if isinstance(raw_partitions, str | bytes) or not isinstance(
        raw_partitions,
        Sequence,
    ):
        raise TypeError("leiden partitions must be a non-empty sequence")
    partitions = tuple(
        (str(resolution), resolution)
        for resolution in map(canonical_resolution, raw_partitions)
    )
    if not partitions:
        raise ValueError("leiden partitions must not be empty")
    keys = [key for key, _resolution in partitions]
    if len(keys) != len(set(keys)):
        raise ValueError("leiden partitions contain duplicate resolutions")
    selected: str | None = None
    raw_selected = settings.pop("selected", None)
    if raw_selected is not None:
        selected = str(canonical_resolution(raw_selected))
        if selected not in dict(partitions):
            raise ValueError("leiden selected must be one of the partitions")
    return partitions, selected, settings


def leiden_config(
    partitions: tuple[tuple[str, float], ...],
    selected: str | None,
    settings: Mapping[str, Any],
) -> dict[str, Any] | bool:
    """Return the recorded Leiden settings, with the resolved candidates."""
    if not partitions:
        return False
    return {
        **settings,
        "partitions": [value for _key, value in partitions],
        "selected": None if selected is None else dict(partitions)[selected],
    }


def require_columns(store: Any, columns: Sequence[str], what: str) -> None:
    missing = [column for column in columns if column not in store.cells.columns]
    if missing:
        raise KeyError(f"{what} were not found: {missing!r}")


def warn_without_sgtsnepi() -> None:
    """Warn, before a run starts, when its t-SNE stage cannot compute.

    The run is not refused: the stage reuses a matching embedding that the
    store holds without importing ``sgtsnepi``, and only a new embedding
    needs it.
    """
    try:
        sgtsne.require_sgtsnepi()
    except ImportError as error:
        logger.warning(
            "This run enables t-SNE, and its t-SNE stage can only reuse an "
            "embedding that the store already holds; computing a new one fails "
            f"at that stage, after the earlier stages. {error}"
        )
