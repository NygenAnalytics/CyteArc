"""Every public flag takes Python and NumPy booleans and refuses integers and text.

The flags are found from the signatures of the public surface: the namespace
methods, the DataStore and its artifact, pipeline and cell accessors, the
assays, the plotting functions and the names that ``cytearc`` and its domain
packages export. A parameter is a flag when its annotation is ``bool``, or
``bool | None`` for one that also takes None. Every public function checks its
flags first, so each call here passes a placeholder for every other required
argument: a refused flag raises before the placeholder is read, and an
accepted one gives the same outcome as the Python boolean it equals.
"""

import dataclasses
import importlib
import inspect
import types
import typing
from collections.abc import Callable, Iterator
from types import FunctionType
from typing import Any

import numpy as np
import pytest

import cytearc
import cytearc.plotting as splt
from cytearc.assay import ADTassay, ATACassay, Assay, RNAassay
from cytearc.datastore import namespaces
from cytearc.datastore.artifact_accessor import ArtifactAccessor
from cytearc.datastore.datastore import DataStore
from cytearc.datastore.pipeline_accessor import PipelineAccessor
from cytearc.metadata import MetaData
from cytearc.plotting import PlotOutputSettings
from cytearc.regulons import ModuleParameters
from cytearc.utils.arguments import bool_argument, optional_bool_argument
from tests.storage_helpers import open_store, write_count_store


class _Placeholder:
    """An argument that no check accepts, standing in for every required one."""

    def __repr__(self) -> str:
        return "<placeholder>"


PLACEHOLDER = _Placeholder()

# Calls that change process-wide settings when they succeed; each test of
# theirs restores what it found.
_GLOBAL_SETTINGS = {"cytearc.configure_output", "cytearc.utils.configure_output"}


def _flag_kind(annotation: Any) -> str | None:
    """``"flag"`` for ``bool``, ``"optional"`` for ``bool | None``, else None."""
    if annotation is inspect.Parameter.empty:
        return None
    if isinstance(annotation, str):
        parts = sorted(part.strip() for part in annotation.split("|"))
        return {("bool",): "flag", ("None", "bool"): "optional"}.get(tuple(parts))
    if annotation is bool:
        return "flag"
    if (
        isinstance(annotation, types.UnionType)
        or typing.get_origin(annotation) is typing.Union
    ):
        arguments = set(typing.get_args(annotation))
        if arguments == {bool, type(None)}:
            return "optional"
    return None


def _is_record(cls: type) -> bool:
    return dataclasses.is_dataclass(cls) or hasattr(cls, "model_fields")


def _methods(cls: type) -> Iterator[tuple[str, FunctionType]]:
    """The public functions of ``cls`` and of its bases in CyteArc."""
    seen: set[str] = set()
    for owner in cls.__mro__:
        if not owner.__module__.startswith("cytearc"):
            continue
        for name, member in vars(owner).items():
            if name.startswith("_") or name in seen:
                continue
            if isinstance(member, staticmethod | classmethod):
                member = member.__func__
            if isinstance(member, FunctionType):
                seen.add(name)
                yield name, member


def _surface() -> Iterator[tuple[str, Callable[[Any], Callable[..., Any]], Any]]:
    """Each public callable: its label, how to reach it from a store, its function."""
    for name in namespaces.__all__:
        attribute = name.removesuffix("Accessor").lower()
        for method, function in _methods(getattr(namespaces, name)):
            yield (
                f"ds.{attribute}.{method}",
                lambda store, a=attribute, m=method: getattr(getattr(store, a), m),
                function,
            )
    for method, function in _methods(DataStore):
        yield (
            f"DataStore.{method}",
            lambda store, m=method: getattr(store, m),
            function,
        )
    for label, cls, attribute in (
        ("ds.artifacts", ArtifactAccessor, "artifacts"),
        ("ds.pipeline", PipelineAccessor, "pipeline"),
        ("ds.cells", MetaData, "cells"),
    ):
        for method, function in _methods(cls):
            yield (
                f"{label}.{method}",
                lambda store, a=attribute, m=method: getattr(getattr(store, a), m),
                function,
            )
    for cls, assay in (
        (Assay, "RNA"),
        (RNAassay, "RNA"),
        (ATACassay, "ATAC"),
        (ADTassay, "ADT"),
    ):
        for method, function in _methods(cls):
            yield (
                f"{cls.__name__}.{method}",
                lambda store, a=assay, m=method: getattr(getattr(store, a), m),
                function,
            )
    for name in splt.__all__:
        value = getattr(splt, name)
        if inspect.isfunction(value):
            yield f"cytearc.plotting.{name}", lambda _store, v=value: v, value
    for package in (
        "cytearc",
        "cytearc.clones",
        "cytearc.repertoire",
        "cytearc.variants",
        "cytearc.perturbation",
    ):
        module = importlib.import_module(package)
        for name in getattr(module, "__all__", []):
            value = getattr(module, name)
            if inspect.isfunction(value):
                yield f"{package}.{name}", lambda _store, v=value: v, value
            elif inspect.isclass(value) and value.__module__.startswith("cytearc"):
                if not _is_record(value) and "__init__" in vars(value):
                    yield (
                        f"{package}.{name}",
                        lambda _store, v=value: v,
                        vars(value)["__init__"],
                    )


def _public_flags() -> list[tuple[str, Callable[[Any], Callable[..., Any]], str, str]]:
    found = {}
    for label, reach, function in _surface():
        for name, parameter in inspect.signature(function).parameters.items():
            kind = _flag_kind(parameter.annotation)
            if kind is not None:
                found[(label, name)] = (label, reach, name, kind)
    return sorted(found.values(), key=lambda row: (row[0], row[2]))


PUBLIC_FLAGS = _public_flags()


def _arguments(function: Any, flag: str, value: object) -> dict[str, Any]:
    """A placeholder for every required argument but ``self``, and the flag."""
    arguments: dict[str, Any] = {}
    for name, parameter in inspect.signature(function).parameters.items():
        if name in ("self", "cls"):
            continue
        if name == flag:
            arguments[name] = value
        elif parameter.default is inspect.Parameter.empty and parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            arguments[name] = PLACEHOLDER
    return arguments


def _called(call: Callable[..., Any], arguments: dict[str, Any]) -> Any:
    """Call, and start a generator, which checks its arguments when it starts."""
    result = call(**arguments)
    if inspect.isgenerator(result):
        try:
            next(result, None)
        finally:
            result.close()
    return result


def _outcome(call: Callable[..., Any], arguments: dict[str, Any]) -> tuple[str, str]:
    """What a call gives: the error it raises, or the type of what it returns."""
    try:
        result = _called(call, arguments)
    except Exception as error:  # noqa: BLE001
        return type(error).__name__, str(error)
    return "returned", type(result).__name__


@pytest.fixture(scope="module")
def flag_store(tmp_path_factory):
    path = tmp_path_factory.mktemp("flags") / "store.zarr"
    write_count_store(
        str(path),
        {
            "RNA": np.arange(24, dtype=np.uint16).reshape(6, 4),
            "ATAC": np.arange(24, dtype=np.uint16).reshape(6, 4),
            "ADT": np.arange(18, dtype=np.uint16).reshape(6, 3),
        },
        "uint16",
    )
    return open_store(path)


@pytest.fixture
def restored_output_settings():
    from cytearc.utils import logging as cytearc_logging

    saved = dataclasses.replace(cytearc_logging._config)
    try:
        yield
    finally:
        cytearc.configure_output(
            level=saved.level,
            progress=saved.progress,
            timestamps=saved.timestamps,
            stdlib=saved.stdlib,
        )


def test_the_surface_holds_the_flags_of_every_namespace_and_plot() -> None:
    # The discovery can pass by finding nothing, so it must find the flags
    # that each part of the surface is known to hold.
    labels = {(label, name) for label, _reach, name, _kind in PUBLIC_FLAGS}
    for expected in (
        ("ds.clones.barcode_clones", "prune"),
        ("ds.clones.barcode_clones", "trex_rules"),
        ("ds.repertoire.clonotypes", "same_v_gene"),
        ("ds.variants.import_counts", "swap_strands"),
        ("ds.reduction.pca", "feat_scaling"),
        ("ds.graph.load", "symmetric"),
        ("ds.perturbation.profiles", "mask_targets"),
        ("ds.mapping.compute_scores", "weighted"),
        ("DataStore.select_cells", "keep_bounds"),
        ("ds.artifacts.prune_incomplete", "dry_run"),
        ("ds.pipeline.rna", "markers"),
        ("ds.cells.insert", "overwrite"),
        ("ATACassay.normed", "log_transform"),
        ("cytearc.plotting.clone_fates", "normalize"),
        ("cytearc.plotting.embedding", "sort_values"),
        ("cytearc.CrToZarr", "overwrite"),
        ("cytearc.configure_output", "stdlib"),
    ):
        assert expected in labels
    assert len(PUBLIC_FLAGS) >= 300


@pytest.mark.parametrize(
    ("label", "reach", "flag", "kind"),
    PUBLIC_FLAGS,
    ids=[f"{label}-{flag}" for label, _reach, flag, _kind in PUBLIC_FLAGS],
)
def test_every_public_flag_refuses_integers_and_text(
    flag_store, label, reach, flag, kind
) -> None:
    call = reach(flag_store)
    message = "a boolean or None" if kind == "optional" else "a boolean"
    refused: list[object] = [1, 0, np.int64(1), "yes", "False", 1.0]
    if kind == "flag":
        refused.append(None)
    for value in refused:
        with pytest.raises(TypeError, match=rf"^{flag} must be {message}$"):
            _called(call, _arguments(call, flag, value))


@pytest.mark.parametrize(
    ("label", "reach", "flag", "kind"),
    PUBLIC_FLAGS,
    ids=[f"{label}-{flag}" for label, _reach, flag, _kind in PUBLIC_FLAGS],
)
def test_every_public_flag_takes_numpy_booleans_as_python_ones(
    flag_store, request, label, reach, flag, kind
) -> None:
    if label in _GLOBAL_SETTINGS:
        request.getfixturevalue("restored_output_settings")
    call = reach(flag_store)
    for numpy_value, value in ((np.True_, True), (np.False_, False)):
        found = _outcome(call, _arguments(call, flag, numpy_value))
        assert found == _outcome(call, _arguments(call, flag, value))
        # The outcome is the next check's, not a refusal of the flag.
        assert not found[1].startswith(f"{flag} must be")
    if kind == "optional":
        assert not _outcome(call, _arguments(call, flag, None))[1].startswith(
            f"{flag} must be"
        )


@pytest.mark.parametrize("value", [np.True_, np.False_, True, False])
def test_bool_argument_returns_a_python_boolean_for_numpy_and_python_ones(
    value,
) -> None:
    expected = value is True or value is np.True_
    found = bool_argument(value, "flag")
    assert type(found) is bool and found is expected
    found = optional_bool_argument(value, "flag")
    assert type(found) is bool and found is expected


@pytest.mark.parametrize("value", [1, 0, np.int8(0), 1.0, "yes", "", b"1", [True]])
def test_optional_bool_argument_refuses_integers_and_text(value) -> None:
    with pytest.raises(TypeError, match="^flag must be a boolean or None$"):
        optional_bool_argument(value, "flag")
    assert optional_bool_argument(None, "flag") is None


def test_settings_objects_take_numpy_booleans_and_refuse_integers() -> None:
    # Two dataclasses that callers build hold flags: the plot output
    # settings and the regulon module rules.
    settings = PlotOutputSettings("panel.png", transparent=np.True_)
    assert type(settings.transparent) is bool and settings.transparent
    assert type(settings.exact_size) is bool and settings.exact_size
    with pytest.raises(TypeError, match="^exact_size must be a boolean$"):
        PlotOutputSettings("panel.png", exact_size=1)
    rules = ModuleParameters(keepOnlyActivating=np.False_)
    assert type(rules.keepOnlyActivating) is bool and not rules.keepOnlyActivating
    with pytest.raises(TypeError, match="^keepOnlyActivating must be a boolean$"):
        ModuleParameters(keepOnlyActivating="no")
