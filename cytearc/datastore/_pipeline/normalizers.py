"""The normalizer of a recipe run: the setting that names it, and how a run applies it.

A run of the ``rna``, ``atac``, ``adt`` or ``perturbation`` recipe normalizes
its assay with the normalizer that its ``normalization`` ``method`` names, or
with the assay's ``normMethod`` when the call names none. The run records the
normalizer by the name of a built-in normalizer, or by the identity that
provenance records for any other, so a rerun in any session normalizes as the
run did. Every stage reads the normalizer from the assay's ``normMethod``, so
a run sets it there while its stages run, and so does a plan of the run,
since the identities of its artifacts depend on it. A ``multimodal`` run
ranks each assay's markers with the normalizer of the run that analyzed it.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import MappingProxyType
from typing import Any

from ...assay import (
    ATACassay,
    NormMethod,
    PearsonResidualNormalizer,
    RNAassay,
    norm_clr,
    norm_dummy,
    norm_lib_size,
    norm_lib_size_log,
    norm_pearson_residuals,
    norm_tf_idf,
)
from ...storage.artifacts import ArtifactRef, callable_identity, inspect_artifact
from ...storage.pipeline_runs import PipelineRunRecord, load_pipeline_stage_records

# The built-in normalizers that a run names, each by its function's name
# without ``norm_``; ``passthrough`` is ``norm_dummy``, which passes the
# stored values through.
BUILTIN_NORMALIZERS: Mapping[str, NormMethod] = MappingProxyType(
    {
        "lib_size": norm_lib_size,
        "lib_size_log": norm_lib_size_log,
        "pearson_residuals": norm_pearson_residuals,
        "clr": norm_clr,
        "tf_idf": norm_tf_idf,
        "passthrough": norm_dummy,
    }
)
# The keys of each identity that provenance records for a normalizer: its
# artifact_identity, or its module and qualified name, with the digest of
# its code for a function defined outside CyteArc.
_IDENTITY_KEYS = (
    frozenset({"identity"}),
    frozenset({"module", "qualname"}),
    frozenset({"module", "qualname", "code"}),
)

type _IdentityKey = tuple[tuple[str, str], ...]
type MethodSetting = str | dict[str, str]
"""A recorded ``normalization`` ``method``: a built-in's name or an identity."""


def _identity_key(identity: Mapping[str, str]) -> _IdentityKey:
    return tuple(sorted(identity.items()))


_BUILTIN_IDENTITIES: Mapping[_IdentityKey, str] = MappingProxyType(
    {
        _identity_key(callable_identity(normalizer)): name
        for name, normalizer in BUILTIN_NORMALIZERS.items()
    }
)


def _setting(identity: Mapping[str, str]) -> MethodSetting:
    """Return the method that records ``identity``: a built-in's name, or the identity."""
    return _BUILTIN_IDENTITIES.get(_identity_key(identity), dict(identity))


def _is_identity(value: Mapping[Any, Any]) -> bool:
    return frozenset(value) in _IDENTITY_KEYS and all(
        isinstance(item, str) and item for item in value.values()
    )


def check_method_setting(value: object) -> None:
    """Refuse a ``normalization`` ``method`` that names no normalizer.

    A method is None, which takes the assay's ``normMethod``, the name of a
    built-in normalizer, or the identity of a custom normalizer as a run
    records it and ``run.settings()`` returns it.

    Raises:
        TypeError: If the method is neither a name nor a mapping, such as
            the normalizer itself.
        ValueError: If the name is no built-in normalizer's, or the mapping
            is no identity.
    """
    names = sorted(BUILTIN_NORMALIZERS)
    if value is None:
        return
    if isinstance(value, str):
        if value not in BUILTIN_NORMALIZERS:
            raise ValueError(
                f"normalization['method'] must be one of {names!r}, or the "
                f"identity of a custom normalizer that a run recorded; got {value!r}"
            )
        return
    if isinstance(value, Mapping):
        if not _is_identity(value):
            raise ValueError(
                "normalization['method'] must be one of "
                f"{names!r}, or the identity of a custom normalizer that a run "
                "recorded: a mapping of 'identity', or of 'module', 'qualname' "
                f"and 'code', to text; got {dict(value)!r}"
            )
        return
    if callable(value):
        raise TypeError(
            "normalization['method'] takes the name of a built-in normalizer, "
            f"one of {names!r}, not the normalizer itself; a run applies a "
            "custom normalizer that is the assay's normMethod"
        )
    raise TypeError(
        f"normalization['method'] must be one of {names!r}, or the identity "
        f"of a custom normalizer that a run recorded; got {value!r}"
    )


def normalizer_setting(normalizer: NormMethod) -> MethodSetting:
    """Return how a run records ``normalizer``: a built-in's name, or its identity.

    A normalizer is built-in when provenance identifies it as one, so a
    ``PearsonResidualNormalizer`` with the default options records
    ``"pearson_residuals"``.

    Raises:
        ValueError: If provenance cannot identify the normalizer, as for a
            lambda without an ``artifact_identity``.
    """
    return _setting(callable_identity(normalizer))


def normalizer_label(normalizer: NormMethod) -> str:
    """Name a normalizer in a message: by its recorded method, or its qualified name."""
    try:
        setting = normalizer_setting(normalizer)
    except ValueError:
        name = getattr(normalizer, "__qualname__", type(normalizer).__qualname__)
        return f"{name}, which provenance cannot identify"
    if isinstance(setting, str):
        return setting
    return f"the custom normalizer {setting!r}"


def method_normalizer(assay: Any, method: object) -> NormMethod | None:
    """Return the normalizer that a checked ``method`` names for ``assay``, or None.

    None names the assay's ``normMethod``, and a name its built-in
    normalizer. A run applies a custom normalizer only as the assay's
    ``normMethod``, so an identity names the ``normMethod`` when provenance
    identifies it so, and None is returned when it does not.
    """
    if method is None:
        normalizer: NormMethod = assay.normMethod
        return normalizer
    assert isinstance(method, str | Mapping)
    named = method if isinstance(method, str) else _setting(method)
    if isinstance(named, str):
        return BUILTIN_NORMALIZERS[named]
    try:
        current = callable_identity(assay.normMethod)
    except ValueError:
        return None
    return assay.normMethod if current == named else None


def _required_class(normalizer: NormMethod) -> type | None:
    """Return the assay class that hands ``normalizer`` what it reads, or None for any.

    The library-size and Pearson residual normalizers read each cell's total,
    which only an RNA assay hands them, and TF-IDF the document frequencies
    that an ATAC assay fits. CLR, passthrough and custom normalizers read the
    counts alone, as far as CyteArc knows.
    """
    if isinstance(normalizer, PearsonResidualNormalizer) or any(
        normalizer is builtin for builtin in (norm_lib_size, norm_lib_size_log)
    ):
        return RNAassay
    if normalizer is norm_tf_idf:
        return ATACassay
    return None


def resolve_normalizer(
    store: Any, assay_name: str, method: object
) -> tuple[NormMethod, MethodSetting]:
    """Return the normalizer that a run of ``assay_name`` applies, and the method it records.

    ``method`` is the ``normalization`` ``method`` setting, which
    :func:`check_method_setting` checked, and names the normalizer as
    :func:`method_normalizer` takes it.

    Raises:
        ValueError: If the assay's ``normMethod`` lacks the identity that
            ``method`` names, the normalizer needs another assay class, or
            provenance cannot identify the assay's ``normMethod``.
    """
    assay = store._get_assay(assay_name)
    names = sorted(BUILTIN_NORMALIZERS)
    normalizer = method_normalizer(assay, method)
    if normalizer is None:
        raise ValueError(
            f"normalization['method'] is the custom normalizer {method!r}, which "
            "a run applies only as the assay's normMethod, but assay "
            f"{assay_name!r} normalizes with {normalizer_label(assay.normMethod)}; "
            "set the assay's normMethod to that normalizer, or pass "
            f"normalization={{'method': ...}} with one of {names!r}"
        )
    try:
        recorded = normalizer_setting(normalizer)
    except ValueError as error:
        raise ValueError(
            f"A run records the normalizer of assay {assay_name!r} by its "
            f"identity: {error}"
        ) from error
    required = _required_class(normalizer)
    if required is not None and not isinstance(assay, required):
        applicable = [
            name
            for name in names
            if isinstance(assay, _required_class(BUILTIN_NORMALIZERS[name]) or object)
        ]
        raise ValueError(
            f"Assay {assay_name!r} opens as {type(assay).__name__}, but "
            f"{normalizer_label(normalizer)} needs an {required.__name__}; pass "
            f"normalization={{'method': ...}} with one of {applicable!r}"
        )
    return normalizer, recorded


def feat_scaling_default(normalizer: NormMethod) -> bool:
    """Return the PCA ``feat_scaling`` that a run of ``normalizer`` takes by default.

    Pearson residuals are already standardized, so their PCA does not
    rescale them; every other normalizer's PCA scales its features.
    """
    return not isinstance(normalizer, PearsonResidualNormalizer)


def add_applied_feat_scaling(settings: dict[str, Any], method: MethodSetting) -> None:
    """Give the settings of a run recorded before the method the PCA scaling it applied.

    Before the ``normalization`` ``method`` existed, a PCA scaled its
    features unless the call set ``feat_scaling``, and the run recorded only
    what the call gave. A run of Pearson residuals now defaults to no
    scaling, and so may a custom normalizer, whose recorded identity does
    not say whether it is a ``PearsonResidualNormalizer``. The ``pca``
    settings of such a run that set no ``feat_scaling`` therefore get True,
    as the run applied it, while they run a PCA, with ``dims`` above 0. A
    run of another built-in normalizer, whose default is still True, keeps
    its settings.
    """
    pca = settings.get("pca")
    if not isinstance(pca, dict) or "feat_scaling" in pca:
        return
    if isinstance(method, str) and feat_scaling_default(BUILTIN_NORMALIZERS[method]):
        return
    dims = pca.get("dims")
    if isinstance(dims, int) and not isinstance(dims, bool) and dims > 0:
        pca["feat_scaling"] = True


@contextmanager
def applied_normalizers(
    store: Any, normalizers: Mapping[str, NormMethod]
) -> Iterator[None]:
    """Make each run normalizer its assay's ``normMethod`` while the block runs.

    ``normalizers`` maps assay names to the normalizers that a run applies.
    The session's normalizers come back however the block ends, completed,
    failed, or cancelled.
    """
    assays = [
        (store._get_assay(name), normalizer) for name, normalizer in normalizers.items()
    ]
    session = [assay.normMethod for assay, _normalizer in assays]
    try:
        for assay, normalizer in assays:
            assay.normMethod = normalizer
        yield
    finally:
        for (assay, _normalizer), method in zip(assays, session, strict=True):
            assay.normMethod = method


def recorded_normalizer(store: Any, record: PipelineRunRecord) -> MethodSetting | None:
    """Return the ``normalization`` ``method`` of the normalizer that a run applied.

    A run records its method. A run recorded before the setting existed, as
    every run of CyteArc 1.4.0 is, applied the normalizer that its saved
    normalized values record in their provenance. A run that saved no
    normalized values gives None, so a call with its settings takes the
    assay's ``normMethod``.

    Raises:
        ValueError: If the run's saved normalized values are missing or do
            not record their normalizer.
    """
    normalization = record.config.get("stages", {}).get("normalization")
    if isinstance(normalization, Mapping) and "method" in normalization:
        method = normalization["method"]
        return dict(method) if isinstance(method, Mapping) else method
    ref = _saved_normalized(store, record)
    if ref is None:
        return None
    status = inspect_artifact(store.zw, ref)
    identity = (status.parameters or {}).get("normalization_method")
    if not status.complete or not isinstance(identity, Mapping):
        raise ValueError(
            f"Pipeline run {record.run_id}, recorded by CyteArc "
            f"{record.cytearc_version}, records no normalization method, and its "
            f"normalized values, artifact {ref.artifact_id[:12]}, are missing or "
            "record no normalization_method; report()['run']['config'] holds the "
            "settings that it recorded"
        )
    return _setting(
        {key: value for key, value in identity.items() if key != "external_hook"}
    )


def _saved_normalized(store: Any, record: PipelineRunRecord) -> ArtifactRef | None:
    """Return the normalized values that a run's normalization stage saved, or None.

    A stage that was interrupted before it saved them records no output.
    """
    for stage in load_pipeline_stage_records(store.zw, record.run_id):
        if stage.stage == "normalization":
            saved = {output.output_key: output.artifact for output in stage.outputs}
            return saved.get("normalized")
    return None
