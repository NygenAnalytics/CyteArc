"""Describe each recipe's stage keywords: the settings each takes and the defaults the recipe uses.

The description is built from what resolves a recipe call: the stage keywords
of the recipe's method, the settings that ``STAGE_SETTINGS`` lets each one
take, and the signatures of the public methods in ``STAGE_METHODS`` that the
stage passes them to. A setting added to a stage therefore appears with its
method's annotation and default. The few settings whose default the recipe
sets itself, the keys that configure the recipe and the defaults that it fits
to its data or to the rest of the call, are declared here, with their rules.
"""

import inspect
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from .filtering import FILTERING_SETTINGS
from .recipes import adt, atac, rna
from .settings import (
    DEFAULT_LEIDEN_PARTITIONS,
    MULTIMODAL_LEIDEN_PARTITIONS,
    PCA_DIMS_DEFAULT,
    PEAKS_TOP_N_DEFAULT,
    STAGE_METHODS,
    STAGE_SETTINGS,
)


def _json_value(value: Any) -> Any:
    """Return ``value`` as JSON values; a float that is not finite as ``"inf"``, ``"-inf"`` or ``"nan"``."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    raise TypeError(f"A described default must be a JSON value, not {value!r}")


@dataclass(frozen=True, slots=True)
class SettingDescription:
    """One setting that a stage keyword's mapping takes.

    Attributes:
        name: The setting's name, a key of the stage keyword's mapping.
        method: The public method whose keyword the setting is, such as
            ``"features.hvgs"``, or None for a key that configures the recipe,
            such as Leiden ``partitions``.
        annotation: The setting's type: as the method's signature annotates
            it, or as the recipe takes it where the recipe takes less, such as
            a cell column for ``hvg`` ``batch``.
        default: What the recipe uses when a call omits the setting: the
            method's default unless the recipe sets its own. None when the
            setting is required or ``rule`` sets the default.
        rule: How the recipe sets the default from its data or from the rest
            of the call, such as the ATAC recipe's ``prevalent_peaks``
            ``top_n``, or when the setting is required; None otherwise.
        required: Whether every call that passes the stage's mapping must
            give the setting, such as Harmony's ``batch_columns``.
    """

    name: str
    method: str | None
    annotation: str
    default: Any = None
    rule: str | None = None
    required: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the setting as JSON values, a default that is not a finite number as text."""
        return {
            "name": self.name,
            "method": self.method,
            "annotation": self.annotation,
            "default": _json_value(self.default),
            "rule": self.rule,
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class StageDescription:
    """One stage keyword of a recipe.

    Attributes:
        keyword: The recipe keyword, such as ``"hvg"``.
        methods: The public methods that the stage runs, whose keywords its
            settings are, such as ``("features.hvgs",)``; empty for the ADT
            panel, whose only setting configures the recipe.
        runs: Whether the stage runs when a call omits the keyword, or None
            when ``rule`` decides.
        rule: When the stage runs if the keyword is omitted, for ``runs`` of
            None, such as ``filtering``; None otherwise.
        settings: The settings that the keyword's mapping takes, the keys that
            configure the recipe first and then each method's keywords in the
            order of its signature.
    """

    keyword: str
    methods: tuple[str, ...]
    runs: bool | None
    rule: str | None
    settings: tuple[SettingDescription, ...]

    def __getitem__(self, name: str) -> SettingDescription:
        """Return the description of the setting ``name``."""
        for setting in self.settings:
            if setting.name == name:
                return setting
        raise KeyError(
            f"{self.keyword} takes no setting {name!r}; it takes "
            f"{[setting.name for setting in self.settings]!r}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stage as JSON values."""
        return {
            "keyword": self.keyword,
            "methods": list(self.methods),
            "runs": self.runs,
            "rule": self.rule,
            "settings": [setting.to_dict() for setting in self.settings],
        }


@dataclass(frozen=True, slots=True)
class RecipeDescription:
    """The stage keywords of one recipe, in the order of its method's signature.

    Attributes:
        recipe: The recipe, such as ``"rna"``.
        stages: One description for each stage keyword.
    """

    recipe: str
    stages: tuple[StageDescription, ...]

    def __getitem__(self, keyword: str) -> StageDescription:
        """Return the description of the stage keyword ``keyword``."""
        for stage in self.stages:
            if stage.keyword == keyword:
                return stage
        raise KeyError(
            f"The {self.recipe} recipe has no stage keyword {keyword!r}; it has "
            f"{[stage.keyword for stage in self.stages]!r}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the description as JSON values."""
        return {
            "recipe": self.recipe,
            "stages": [stage.to_dict() for stage in self.stages],
        }


# A declared setting whose default is its method's.
_METHOD_DEFAULT = object()


@dataclass(frozen=True, slots=True)
class _Declared:
    """A setting that the recipe takes or sets otherwise than its method does.

    ``annotation`` replaces the method's annotation when given. ``default``
    replaces the method's default unless it is ``_METHOD_DEFAULT``, and a
    ``rule`` sets the default instead of any value. A ``recipe_key``
    configures the recipe and is no method's keyword.
    """

    annotation: str | None = None
    default: Any = _METHOD_DEFAULT
    rule: str | None = None
    recipe_key: bool = False


def _filter_columns(modality: str, suffixes: tuple[str, ...]) -> _Declared:
    """The default filtering columns of a recipe: its assay's QC columns of ``suffixes``."""
    names = [f"<assay>_{suffix}" for suffix in suffixes]
    columns = (
        names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"
    )
    rule = f"the {modality} assay's QC columns that preparation wrote, {columns}"
    if "nCounts" in suffixes:
        rule += ", without <assay>_nCounts for an assay of normalized values"
    return _Declared("collections.abc.Sequence[str]", rule=rule)


_MANUAL_BOUNDS = "required with method 'manual', one bound for each column of attrs"
# The settings whose type, default or rule is the recipe's own, by recipe,
# stage keyword and setting; "*" declares a setting for every recipe.
_DECLARED: Mapping[tuple[str, ...], _Declared] = MappingProxyType(
    {
        ("*", "cell_calling", "sample_column"): _Declared(
            "str | None", None, recipe_key=True
        ),
        ("*", "hashing", "assay"): _Declared(
            "str | None", rule="the store's only assay of type HTO", recipe_key=True
        ),
        ("*", "filtering", "method"): _Declared(
            "Literal['auto', 'mad', 'gaussian', 'manual']", "auto", recipe_key=True
        ),
        ("rna", "filtering", "attrs"): _filter_columns("RNA", rna.FILTER_SUFFIXES),
        ("perturbation", "filtering", "attrs"): _filter_columns(
            "RNA", rna.FILTER_SUFFIXES
        ),
        ("atac", "filtering", "attrs"): _filter_columns("ATAC", atac.FILTER_SUFFIXES),
        ("adt", "filtering", "attrs"): _filter_columns("ADT", adt.FILTER_SUFFIXES),
        ("*", "filtering", "lows"): _Declared(rule=_MANUAL_BOUNDS),
        ("*", "filtering", "highs"): _Declared(rule=_MANUAL_BOUNDS),
        ("*", "hvg", "batch"): _Declared("str | None"),
        ("atac", "prevalent_peaks", "top_n"): _Declared(
            rule=(
                f"the smaller of {PEAKS_TOP_N_DEFAULT} and one fewer than the "
                "assay's peaks"
            )
        ),
        ("*", "panel", "exclude"): _Declared(
            "str | None", adt.DEFAULT_PANEL_EXCLUDE, recipe_key=True
        ),
        ("adt", "pca", "dims"): _Declared(
            rule=(
                f"the smaller of {PCA_DIMS_DEFAULT} and one fewer than the panel's "
                "antibodies"
            )
        ),
        ("*", "leiden", "partitions"): _Declared(
            "collections.abc.Sequence[float]",
            DEFAULT_LEIDEN_PARTITIONS,
            recipe_key=True,
        ),
        ("multimodal", "leiden", "partitions"): _Declared(
            "collections.abc.Sequence[float]",
            MULTIMODAL_LEIDEN_PARTITIONS,
            recipe_key=True,
        ),
        ("*", "leiden", "selected"): _Declared("float | None", None, recipe_key=True),
        ("rna", "doublets", "sample_column"): _Declared(
            rule=(
                "the Harmony batch_columns when the run corrects batches with "
                "Harmony, else None"
            )
        ),
        ("*", "guide_calling", "path"): _Declared(
            "str | os.PathLike[str] | None", None
        ),
        ("*", "associations", "positive_controls"): _Declared(
            "Literal['targets'] | None"
        ),
        ("*", "profiles", "cells"): _Declared(
            "Literal['knocked_out', 'labelled']",
            rule="'knocked_out' when mixscape runs, else 'labelled'",
            recipe_key=True,
        ),
    }
)
# Whether a stage whose keyword defaults to None runs, or the rule that
# decides, by recipe and keyword; other such stages always run.
_RUNS_WHEN_OMITTED: Mapping[tuple[str, ...], bool | str] = MappingProxyType(
    {
        ("*", "harmony"): False,
        ("rna", "filtering"): (
            "when cells is a cell column, or when cell calling or hashing runs"
        ),
        ("atac", "filtering"): "when cells is a cell column",
        ("perturbation", "filtering"): "when cells is a cell column",
        ("adt", "filtering"): False,
        ("perturbation", "signature"): "when mixscape runs",
        ("perturbation", "profiles"): "when clusters runs",
    }
)


def _declared[T](
    table: Mapping[tuple[str, ...], T], recipe: str, *key: str
) -> T | None:
    """Return the recipe's own entry for ``key``, or the entry of every recipe."""
    return table.get((recipe, *key), table.get(("*", *key)))


def _method_name(method: Callable[..., Any]) -> str:
    """Return the public name of a stage method, such as ``features.hvgs`` for ``_features_hvgs``."""
    namespace, _separator, name = method.__name__.removeprefix("_").partition("_")
    return f"{namespace}.{name}"


def _setting(
    recipe: str,
    stage: str,
    name: str,
    parameters: Mapping[str, tuple[str, inspect.Parameter]],
) -> SettingDescription:
    """Describe one setting from its method's parameter and the recipe's declaration."""
    declared = _declared(_DECLARED, recipe, stage, name)
    found = parameters.get(name)
    if declared is None and found is None:
        raise KeyError(
            f"{stage}[{name!r}] is no keyword of {list(STAGE_METHODS[stage])!r} and "
            "declares no default or rule of the recipe"
        )
    if declared is not None and declared.recipe_key:
        found = None
    method = None if found is None else found[0]
    parameter = None if found is None else found[1]
    if declared is not None and declared.annotation is not None:
        annotation = declared.annotation
    else:
        assert parameter is not None
        annotation = inspect.formatannotation(parameter.annotation)
    if declared is not None and declared.rule is not None:
        return SettingDescription(name, method, annotation, rule=declared.rule)
    if declared is not None and declared.default is not _METHOD_DEFAULT:
        return SettingDescription(name, method, annotation, declared.default)
    assert parameter is not None
    if parameter.default is inspect.Parameter.empty:
        return SettingDescription(name, method, annotation, required=True)
    return SettingDescription(name, method, annotation, parameter.default)


def _stage(recipe: str, keyword: str, default: object) -> StageDescription:
    """Describe one stage keyword, whose default in the recipe's signature is ``default``."""
    methods = STAGE_METHODS[keyword]
    names = FILTERING_SETTINGS if keyword == "filtering" else STAGE_SETTINGS[keyword]
    # The first method that takes a setting is the one that the stage passes
    # it to.
    parameters: dict[str, tuple[str, inspect.Parameter]] = {}
    for method in methods:
        for name, parameter in inspect.signature(method).parameters.items():
            if name in names and name not in parameters:
                parameters[name] = (_method_name(method), parameter)
    settings = [_setting(recipe, keyword, name, parameters) for name in names]
    order = list(parameters)
    settings.sort(
        key=lambda setting: (
            setting.method is not None,
            order.index(setting.name) if setting.method is not None else 0,
            setting.name,
        )
    )
    runs: bool | None
    rule = None
    if isinstance(default, bool):
        runs = default
    elif default is inspect.Parameter.empty:
        runs = True
    else:
        omitted = _declared(_RUNS_WHEN_OMITTED, recipe, keyword)
        if isinstance(omitted, str):
            runs, rule = None, omitted
        else:
            runs = True if omitted is None else omitted
    return StageDescription(
        keyword=keyword,
        methods=tuple(_method_name(method) for method in methods),
        runs=runs,
        rule=rule,
        settings=tuple(settings),
    )


def describe_recipe(recipe: str, signature: inspect.Signature) -> RecipeDescription:
    """Describe the stage keywords of ``recipe``, whose method has ``signature``.

    A stage keyword is a keyword of the recipe's method that ``STAGE_SETTINGS``
    or the filtering settings name; the recipe's other keywords, such as
    ``assay`` and ``label``, are not stages.

    Raises:
        KeyError: If a stage setting is no keyword of the stage's methods and
            declares no default or rule of the recipe.
    """
    return RecipeDescription(
        recipe=recipe,
        stages=tuple(
            _stage(recipe, keyword, parameter.default)
            for keyword, parameter in signature.parameters.items()
            if keyword in STAGE_SETTINGS or keyword == "filtering"
        ),
    )
