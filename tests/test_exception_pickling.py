"""Every CyteArc exception class survives a pickle round trip.

Remote runners such as Modal, multiprocessing, and joblib pickle an exception
to report it to the calling process, which rebuilds it from the pickle. An
exception that cannot be rebuilt reaches the caller as a deserialization
error instead of the failure. Each class therefore has a sample here, and a
guard fails when a module's source defines an exception class without one.
"""

import ast
import builtins
import pickle
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import cytearc
from cytearc.agent.models import AnalysisInputError, NeedsInput
from cytearc.agent.provider import (
    DecisionValidationError,
    PromptTooLarge,
    ProviderError,
    RequestBudgetExceeded,
)
from cytearc.agent.records import RecordError, RunLockedError
from cytearc.datastore.pipeline_run import PipelineExecutionError
from cytearc.merge.metadata import _SourceConflict
from cytearc.metadata.membership import UnmeasuredCellsError
from cytearc.readers._rds._types import (
    RdsClosedError,
    RdsError,
    RdsFormatError,
    RdsLimitError,
)
from cytearc.readers._seurat.errors import (
    MatrixSourceError,
    ResourceLimitError,
    UnsafeSidecarError,
    UnsupportedMatrixOperation,
)
from cytearc.readers.seurat import SeuratImportError
from cytearc.storage.errors import ArtifactResolutionError
from cytearc.storage.finite_values import NonFiniteArtifactError
from cytearc.storage.sharding import CountLayoutMemoryError
from cytearc.utils.shutdown import ShutdownRequest, ShutdownRequested

RUN_ID = "a" * 64

SAMPLES: dict[type[BaseException], Callable[[], BaseException]] = {
    AnalysisInputError: lambda: AnalysisInputError("No column names the donors"),
    NeedsInput: lambda: NeedsInput("Which column holds the donor?", field="donor"),
    ProviderError: lambda: ProviderError("The model returned no decision"),
    DecisionValidationError: lambda: DecisionValidationError("The repair failed"),
    RequestBudgetExceeded: lambda: RequestBudgetExceeded("No requests remain"),
    PromptTooLarge: lambda: PromptTooLarge("The request exceeds 4096 bytes"),
    RecordError: lambda: RecordError("The run record is missing"),
    RunLockedError: lambda: RunLockedError("Another process holds the lock"),
    PipelineExecutionError: lambda: PipelineExecutionError(
        RUN_ID, "pca", ValueError("bad pca")
    ),
    _SourceConflict: lambda: _SourceConflict("The sources disagree on 'batch'"),
    UnmeasuredCellsError: lambda: UnmeasuredCellsError(
        "run_pca", "ADT", "ADT_I", 2, 9, "graph"
    ),
    RdsError: lambda: RdsError("Unexpected tag", path="$.x", offset=5),
    RdsFormatError: lambda: RdsFormatError("Bad header", path="$", offset=0),
    RdsLimitError: lambda: RdsLimitError("depth", 70, 64, path="$.y", offset=7),
    RdsClosedError: lambda: RdsClosedError("The document is closed", path="$.z"),
    MatrixSourceError: lambda: MatrixSourceError("The shapes disagree"),
    ResourceLimitError: lambda: ResourceLimitError("The block exceeds its limit"),
    UnsafeSidecarError: lambda: UnsafeSidecarError("The sidecar links outside"),
    UnsupportedMatrixOperation: lambda: UnsupportedMatrixOperation(
        "@assays$RNA", "scale", "DelayedMatrix", "no local implementation"
    ),
    SeuratImportError: lambda: SeuratImportError(
        "Unsupported assay",
        object_path="@assays$RNA",
        code="unsupported_assay",
        context={"class": "Assay5"},
    ),
    ArtifactResolutionError: lambda: ArtifactResolutionError(
        "Stored selection is unavailable",
        code="artifact_missing",
        context={"kind": "cell_selection"},
    ),
    NonFiniteArtifactError: lambda: NonFiniteArtifactError("run_pca", "data", 7),
    CountLayoutMemoryError: lambda: CountLayoutMemoryError("countsT needs 8 GB"),
    ShutdownRequested: lambda: ShutdownRequested(
        ShutdownRequest(
            requested_at_ns=1,
            reason="SIGTERM received",
            signal_number=15,
            signal_name="SIGTERM",
        )
    ),
}


def _name(error_type: type[BaseException]) -> str:
    return f"{error_type.__module__}.{error_type.__qualname__}"


_BUILTIN_EXCEPTIONS = frozenset(
    name
    for name, value in vars(builtins).items()
    if isinstance(value, type) and issubclass(value, BaseException)
)


def _base_name(base: ast.expr) -> str | None:
    """Return the name a class base ends in, such as ``Error`` of ``errors.Error``."""
    if isinstance(base, ast.Subscript):
        base = base.value
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return None


def _class_bases() -> dict[str, set[str]]:
    """Return the base names of each module-level or nested class CyteArc defines.

    The sources are parsed rather than imported, so modules of optional
    integrations, such as Modal, are found where those are not installed.
    """
    root = Path(cytearc.__file__).parent
    classes: dict[str, set[str]] = {}
    for path in sorted(root.rglob("*.py")):
        parts = ("cytearc", *path.relative_to(root).with_suffix("").parts)
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        pending: list[tuple[ast.stmt, str]] = [
            (node, "") for node in ast.parse(path.read_bytes()).body
        ]
        while pending:
            node, prefix = pending.pop()
            if isinstance(node, ast.ClassDef):
                qualname = f"{prefix}{node.name}"
                classes[f"{module}.{qualname}"] = {
                    name for base in node.bases if (name := _base_name(base))
                }
                pending.extend((child, f"{qualname}.") for child in node.body)
            elif isinstance(node, ast.If | ast.Try):
                for block in ("body", "orelse", "handlers", "finalbody"):
                    for child in getattr(node, block, []):
                        if isinstance(child, ast.ExceptHandler):
                            pending.extend((item, prefix) for item in child.body)
                        else:
                            pending.append((child, prefix))
    return classes


def _defined_exception_names() -> set[str]:
    """Return the qualified names of the exception classes CyteArc defines.

    A class is an exception when a base is a built-in exception, one of these
    classes, or another class named like an error, exception, or warning.
    """
    classes = _class_bases()
    exception_bases = set(_BUILTIN_EXCEPTIONS)
    found: set[str] = set()
    while True:
        added = {
            name
            for name, bases in classes.items()
            if name not in found
            and any(
                base in exception_bases
                or base.endswith(("Error", "Exception", "Warning"))
                for base in bases
            )
        }
        if not added:
            return found
        found |= added
        exception_bases |= {name.rsplit(".", 1)[-1] for name in added}


def test_every_exception_class_has_a_pickling_sample() -> None:
    defined = _defined_exception_names()
    sampled = {_name(error_type) for error_type in SAMPLES}

    missing = sorted(defined - sampled)
    stale = sorted(sampled - defined)
    assert not missing, f"Add a sample to SAMPLES for {missing}"
    assert not stale, f"Remove the samples of classes no module defines: {stale}"


@pytest.mark.parametrize("error_type", list(SAMPLES), ids=_name)
def test_exception_survives_a_pickle_round_trip(
    error_type: type[BaseException],
) -> None:
    error = SAMPLES[error_type]()
    assert type(error) is error_type
    error.add_note("Recorded while the run cleaned up")

    restored = pickle.loads(pickle.dumps(error))

    assert type(restored) is error_type
    assert restored.args == error.args
    assert str(restored) == str(error)
    assert vars(restored) == vars(error)


def _raised_from(cause: BaseException) -> PipelineExecutionError:
    try:
        raise PipelineExecutionError(RUN_ID, "pca", cause) from cause
    except PipelineExecutionError as error:
        return error


def test_pipeline_execution_error_carries_its_cause() -> None:
    error = _raised_from(np.linalg.LinAlgError("SVD did not converge"))

    restored = pickle.loads(pickle.dumps(error))

    assert type(restored.__cause__) is np.linalg.LinAlgError
    assert str(restored.__cause__) == "SVD did not converge"
    assert restored.__suppress_context__
    unraised = pickle.loads(pickle.dumps(SAMPLES[PipelineExecutionError]()))
    assert unraised.__cause__ is None


class _RequiredArgumentError(Exception):
    """An error that its pickle cannot rebuild, like many third-party errors."""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


def _unpicklable_error() -> ValueError:
    error = ValueError("bad pca")
    error.lock = threading.Lock()  # type: ignore[attr-defined]
    return error


@pytest.mark.parametrize(
    ("cause", "expected"),
    [
        (_RequiredArgumentError("bad pca", 3), f"{__name__}._RequiredArgumentError"),
        (_unpicklable_error(), "builtins.ValueError"),
    ],
    ids=["cannot-rebuild", "cannot-pickle"],
)
def test_pipeline_execution_error_names_a_cause_that_cannot_round_trip(
    cause: BaseException, expected: str
) -> None:
    error = _raised_from(cause)

    restored = pickle.loads(pickle.dumps(error))

    assert str(restored) == str(error)
    assert (restored.run_id, restored.stage) == (RUN_ID, "pca")
    assert type(restored.__cause__) is RuntimeError
    assert str(restored.__cause__) == f"{expected}: bad pca"
