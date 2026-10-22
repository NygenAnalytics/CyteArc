"""Explicit new-run, resume, and read-only interfaces."""

import asyncio
import sys
from collections.abc import Coroutine
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import AnalysisConfig, RuntimeConfig, Study
from .records import (
    RunRecords,
    procedure_identity,
    process_state,
    relative_locator,
    run_lock,
    source_lock,
)
from .result import AnalysisRun


def _sync(coroutine: Coroutine[Any, Any, AnalysisRun]) -> AnalysisRun:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    coroutine.close()
    raise RuntimeError(
        "An event loop is running; await the corresponding async agent API"
    )


def _require_prompt_budget(runtime: RuntimeConfig) -> None:
    """Refuse a request limit below the fixed evidence budget plus overhead."""
    from .workflow import EVIDENCE_LIMIT

    if runtime.maxPromptBytes < EVIDENCE_LIMIT:
        raise ValueError(
            f"maxPromptBytes must be at least {EVIDENCE_LIMIT:,}: decision evidence "
            "is trimmed to a fixed budget that leaves room for instructions, "
            "schema and repair feedback within this many bytes"
        )


def analyze_rna(
    source: str | Path,
    *,
    run_dir: str | Path | None = None,
    model: Any,
    study: Study | dict[str, Any],
    config: AnalysisConfig | None = None,
    runtime: RuntimeConfig | None = None,
) -> AnalysisRun:
    """Start a new analysis, by default in ./agent_runs/<agentRunId>."""
    return _sync(
        analyze_rna_async(
            source,
            run_dir=run_dir,
            model=model,
            study=study,
            config=config,
            runtime=runtime,
        )
    )


async def analyze_rna_async(
    source: str | Path,
    *,
    run_dir: str | Path | None = None,
    model: Any,
    study: Study | dict[str, Any],
    config: AnalysisConfig | None = None,
    runtime: RuntimeConfig | None = None,
) -> AnalysisRun:
    """Notebook-compatible entry point; numerical pipeline calls are sequential."""
    supplied = Study.model_validate(study)
    scientific = config or AnalysisConfig()
    if scientific.maxCandidates not in {4, 5}:
        raise ValueError(
            "New analyses require maxCandidates=4 or 5 for the four native probes; "
            "older saved configurations remain readable"
        )
    operational = runtime or RuntimeConfig()
    _require_prompt_budget(operational)
    location = Path(source).expanduser().resolve()
    run_id = uuid4().hex[:16]
    destination = (
        Path.cwd() / "agent_runs" / run_id
        if run_dir is None
        else Path(run_dir).expanduser()
    ).resolve()
    if not location.is_dir():
        raise ValueError("source must be an existing prepared local CyteArc store")
    if destination == location or destination.is_relative_to(location):
        raise ValueError("run_dir must be outside the numerical store")
    if model is None:
        raise ValueError("Supply a configured model or model identifier")
    # __version__ also resolves when CyteArc runs from source without installed
    # distribution metadata, where importlib.metadata.version raises.
    from .. import __version__
    from .workflow import run_workflow

    with run_lock(destination), source_lock(location):
        records = RunRecords.create(
            destination,
            {
                "format": "cytearc-rna-analysis",
                "procedure": "bounded-rna",
                "runId": run_id,
                "procedureIdentity": procedure_identity(),
                "inputEvidence": "evidence/inspect.json",
                "frozenPolicy": "evidence/preprocess.json",
                "randomSeed": scientific.randomSeed,
                "software": {
                    "python": sys.version.split()[0],
                    "cytearc": __version__,
                    "pydantic-ai-slim": version("pydantic-ai-slim"),
                    "pydantic": version("pydantic"),
                },
                "source": relative_locator(location, destination),
                "study": supplied.model_dump(mode="json"),
                "config": scientific.model_dump(mode="json"),
            },
        )
        result = AnalysisRun(destination, source=location)
        try:
            await run_workflow(
                records, location, model, supplied, scientific, operational
            )
        finally:
            _finish(result, records)
        return result


def resume_rna(
    run_dir: str | Path,
    *,
    model: Any,
    answers: dict[str, Any] | None = None,
    runtime: RuntimeConfig | None = None,
    source: str | Path | None = None,
    abandon_unfinished_pipeline: bool = False,
) -> AnalysisRun:
    """Resume unchanged scientific work, optionally with new operational settings.

    Set ``abandon_unfinished_pipeline`` only after confirming that the process
    which started the last unfinished pipeline has stopped. Its liveness cannot
    be checked on Windows, or when only a reusable process ID was recorded.
    """
    return _sync(
        resume_rna_async(
            run_dir,
            model=model,
            answers=answers,
            runtime=runtime,
            source=source,
            abandon_unfinished_pipeline=abandon_unfinished_pipeline,
        )
    )


async def resume_rna_async(
    run_dir: str | Path,
    *,
    model: Any,
    answers: dict[str, Any] | None = None,
    runtime: RuntimeConfig | None = None,
    source: str | Path | None = None,
    abandon_unfinished_pipeline: bool = False,
) -> AnalysisRun:
    """Resume one saved run without migrating histories or changing accepted work.

    ``abandon_unfinished_pipeline`` records that the process which started the
    last unfinished pipeline has stopped, so its pipeline can run again under a
    new label. A process confirmed to be running is never abandoned.
    """
    from .evidence import open_store, verify_source
    from .workflow import run_workflow, stage_result, validate_final

    destination = Path(run_dir).expanduser().resolve()
    with run_lock(destination):
        result = AnalysisRun(destination, source=source)
        records = RunRecords(destination)
        if records.manifest["procedureIdentity"] != procedure_identity():
            raise ValueError(
                "CyteArc implementation or prompts changed; start a new run directory"
            )
        supplied = Study.model_validate(records.manifest["study"])
        scientific = AnalysisConfig.model_validate(records.manifest["config"])
        for row in records.events():
            if row["kind"] == "inputsResolved":
                supplied = Study.model_validate(row["study"])
                scientific = AnalysisConfig.model_validate(row["config"])
        previous = records.latest("invocationStarted")
        if (
            runtime is None
            and previous is not None
            and previous.get("modelSettings", {}).get("keys")
        ):
            raise ValueError(
                "The previous invocation supplied modelSettings, which are not saved "
                "because they can hold credentials. Resume with "
                "runtime=RuntimeConfig(modelSettings=...)"
            )
        operational = runtime or RuntimeConfig.model_validate(
            previous["runtime"] if previous else {}
        )
        _require_prompt_budget(operational)
        abandoned = (
            _unfinished_pipeline(records) if abandon_unfinished_pipeline else None
        )
        if answers:
            pending = {row["questionId"]: row for row in result.pending_questions}
            if not pending or set(answers) != set(pending):
                raise ValueError(
                    "Answer the exact pending questionIds; unrelated answers cannot change this run"
                )
            for key, value in answers.items():
                if value is None or isinstance(value, str) and not value.strip():
                    raise ValueError(f"Answer {key} must not be empty")
                field = pending[key].get("field")
                if field and stage_result(records, "inspect") is None:
                    if field == "assay" and scientific.assay is None:
                        scientific = AnalysisConfig.model_validate(
                            {**scientific.model_dump(), "assay": value}
                        )
                    elif (
                        field in {"organism", "tissue", "sampleColumn", "captureColumn"}
                        and getattr(supplied, field) is None
                    ):
                        supplied = Study.model_validate(
                            {**supplied.model_dump(), field: value}
                        )
                    else:
                        raise ValueError(
                            "This answer would change scientific inputs; start a new run"
                        )
        elif result.status == "needsInput":
            return result
        with source_lock(result.source):
            inspected = stage_result(records, "inspect")
            if inspected:
                verify_source(
                    result.source, inspected, supplied, scientific, operational
                )
            final = None
            if result.status == "completed":
                final = stage_result(records, "finalize")
                if not final:
                    raise ValueError("Completed analysis has no final pipeline record")
            elif model is None:
                raise ValueError("Supply a model to resume unfinished decisions")
            if source is not None and inspected:
                _validate_relocation(
                    open_store(result.source, scientific, operational), records
                )
            elif final is not None:
                validate_final(
                    open_store(result.source, scientific, operational), final
                )
            if answers:
                records.append("answers", answers=answers)
                records.append(
                    "inputsResolved",
                    study=supplied.model_dump(mode="json"),
                    config=scientific.model_dump(mode="json"),
                )
            if source is not None and inspected:
                records.append(
                    "sourceRebound",
                    source=relative_locator(result.source, destination),
                    fingerprint=inspected["fingerprint"],
                )
            if abandoned is not None:
                records.append(
                    "pipelineAbandoned",
                    operation=abandoned["operation"],
                    label=abandoned["label"],
                    plannedSequence=abandoned["sequence"],
                    process=abandoned["process"],
                    reason="The caller confirmed that the process stopped",
                )
            if result.status == "completed":
                _finish(result, records)
                return result
            try:
                await run_workflow(
                    records, result.source, model, supplied, scientific, operational
                )
            finally:
                if source is not None and inspected is None:
                    # An initial inspection question has no frozen source yet.
                    inspected = stage_result(records, "inspect")
                    if inspected is not None:
                        records.append(
                            "sourceRebound",
                            source=relative_locator(result.source, destination),
                            fingerprint=inspected["fingerprint"],
                        )
                _finish(result, records)
        return result


def _unfinished_pipeline(records: RunRecords) -> dict[str, Any]:
    """Return the last planned pipeline that never ended, if it may be abandoned."""
    events = records.events()
    planned = next(
        (event for event in reversed(events) if event["kind"] == "pipelinePlanned"),
        None,
    )
    if planned is None or any(
        event["kind"] in {"pipelineCompleted", "pipelineFailed", "pipelineAbandoned"}
        and event.get("label") == planned["label"]
        for event in events
    ):
        raise ValueError("No unfinished pipeline attempt can be abandoned")
    if process_state(planned["process"]) == "running":
        raise ValueError(
            "The process that started the unfinished pipeline is still running; "
            "it cannot be abandoned"
        )
    return planned


def _validate_relocation(store: Any, records: RunRecords) -> None:
    """Require saved numerical history before recording a replacement locator."""
    from .evidence import require_clean_analysis
    from .workflow import stage_result, validate_final

    if records.latest("pipelinePlanned") is None:
        inspected = stage_result(records, "inspect")
        if inspected is not None:
            require_clean_analysis(store, inspected["assay"])
    for event in records.events():
        if event["kind"] != "pipelineCompleted":
            continue
        try:
            run = store.pipeline.open(run_id=event["runId"])
        except (KeyError, ValueError, RuntimeError) as error:
            raise ValueError(
                "Replacement source cannot open a saved pipeline run; "
                "relocate the complete analysis store, including its pipeline history"
            ) from error
        if run.status != "completed":
            raise ValueError("Replacement source has an incomplete saved pipeline run")
        if any(not store.artifacts.inspect(ref).complete for ref in run.values()):
            raise ValueError("Replacement source has incomplete saved artifacts")
    final = stage_result(records, "finalize")
    if final is not None:
        validate_final(store, final)


def open_analysis(
    run_dir: str | Path, *, source: str | Path | None = None
) -> AnalysisRun:
    """Inspect saved outcomes without opening a provider or numerical store."""
    return AnalysisRun(run_dir, source=source)


def _finish(result: AnalysisRun, records: RunRecords) -> None:
    """Publish and report one invocation's outcome from a single source check."""
    with result._shared_binding():
        _publish_result(result, records)
        _report(result, records)


def _publish_result(result: AnalysisRun, records: RunRecords) -> None:
    if result.status != "completed":
        return
    from .compact_result import publish_result

    try:
        publish_result(result)
    except Exception as error:
        records.append(
            "resultPublicationError",
            errorType=type(error).__name__,
            message="Compact result publication failed; completed science is unchanged. Resume to retry.",
        )


def _report(result: AnalysisRun, records: RunRecords) -> None:
    if result.status == "completed":
        try:
            result.save_plots()
        except Exception as error:
            records.append(
                "reportError",
                stage="report",
                errorType=type(error).__name__,
                message="A report plot could not be saved; analysis results are unchanged. Regenerate figures with save_plots().",
            )
    try:
        result.report()
    except Exception as error:
        records.append(
            "reportError",
            errorType=type(error).__name__,
            message="Report rendering failed; saved analysis status and results are unchanged",
        )
