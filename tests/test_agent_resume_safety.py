"""Resume stays recoverable, verifies its source once, and saves no credentials."""

import json
import os
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from cytearc.agent import RuntimeConfig, open_analysis, resume_rna
from cytearc.agent import api, workflow
from cytearc.agent.records import RunRecords, digest
from tests.test_agent_recovery import _analyze, _model
from tests.test_agent_recovery import science as science


def _unsupported_at(stage: str, *, repeat: bool) -> FunctionModel:
    """Refuse the objective at ``stage``; optionally comply after one repair."""
    ordinary = _model([])
    refusals = 0

    async def respond(messages: Any, info: Any) -> ModelResponse:
        nonlocal refusals
        payload = json.loads(messages[-1].parts[-1].content)
        if payload["stage"] == stage and (repeat or refusals == 0):
            refusals += 1
            measured = payload["evidence"]
            options = measured.get("eligibleOptions") or [
                row["optionId"] for row in measured["candidates"][0]["partitions"]
            ]
            answer = {
                "action": "defer",
                "question": "Is descriptive discovery really the objective?",
                "deferralReason": "unsupportedObjective",
                "evidenceIds": options[:1],
                "rationale": "Metadata asks for a differential test instead.",
            }
            return ModelResponse(parts=[ToolCallPart("decision", answer)])
        return await ordinary.function(messages, info)

    return FunctionModel(respond)


@pytest.mark.parametrize(("stage", "measured"), [("explore", 1), ("finalists", 2)])
def test_late_unsupported_objective_is_repaired_and_never_strands_a_run(
    science: Any, tmp_path: Path, stage: str, measured: int
) -> None:
    result = _analyze(science, tmp_path, _unsupported_at(stage, repeat=True))
    records = RunRecords(result.run_dir)
    assert result.status == "failed"
    status = records.latest("status")
    assert (status["stage"], status["errorType"]) == (
        stage,
        "DecisionValidationError",
    )
    assert not any(
        row["kind"] == "decisionAccepted" and row["stage"] == stage
        for row in records.events()
    )
    rejected = records.latest("decisionRejected")
    assert any(
        "unsupportedObjective is a context deferral" in error
        for error in rejected["errors"]
    )
    assert len(science.calls) == measured
    # A resumed run makes a fresh attempt and reuses the pipelines it created.
    observed: list[Any] = []
    resumed = resume_rna(result.run_dir, model=_model(observed))
    assert resumed.status == "completed"
    assert len(science.calls) == 3
    assert [call["label"] for call in science.calls] == [
        f"agent_{records.manifest['runId']}_{operation}_0"
        for operation in ("screen_c0", "finalist_0", "final")
    ]


@pytest.mark.parametrize("stage", ["explore", "finalists"])
def test_late_unsupported_objective_receives_a_semantic_repair(
    science: Any, tmp_path: Path, stage: str
) -> None:
    result = _analyze(science, tmp_path, _unsupported_at(stage, repeat=False))
    assert result.status == "completed"
    records = RunRecords(result.run_dir)
    accepted = [
        row["output"]
        for row in records.events()
        if row["kind"] == "decisionAccepted" and row["stage"] == stage
    ]
    assert [row["deferralReason"] for row in accepted] == [None]
    assert len(science.calls) == 3


def test_source_is_verified_once_per_invocation_and_before_each_write(
    science: Any, tmp_path: Path
) -> None:
    result = _analyze(
        science, tmp_path, _model([], context_question="Which tissue was sampled?")
    )
    assert result.status == "needsInput"
    # The new run computed its fingerprint during inspection; nothing to repeat.
    assert science.verify.call_count == 0
    question = result.pending_questions[0]
    resumed = resume_rna(
        result.run_dir, model=_model([]), answers={question["questionId"]: "Blood"}
    )
    assert resumed.status == "completed"
    # Once on resume, once before each of three pipeline writes, and once for
    # the compact result and previews derived at the end.
    assert science.verify.call_count == 5
    assert len(science.calls) == 3


def test_new_completed_analysis_verifies_only_before_writes_and_outputs(
    science: Any, tmp_path: Path
) -> None:
    result = _analyze(science, tmp_path, _model([]))
    assert result.status == "completed"
    assert science.verify.call_count == 3 + 1


def test_every_file_an_event_references_is_bound_by_its_digest(
    science: Any, tmp_path: Path
) -> None:
    result = _analyze(science, tmp_path, _model([]))
    records = RunRecords(result.run_dir)
    bound = set()
    for event in records.events():
        for field in ("evidence", "requestPath", "responsePath"):
            # A request names its response before it exists; the response
            # event that follows records that file's digest.
            if field in event and (event["kind"], field) != (
                "modelRequest",
                "responsePath",
            ):
                sha256 = event[field.removesuffix("Path") + "Sha256"]
                assert sha256 == records.file_sha256(event[field])
                bound.add((event["kind"], field))
    assert bound == {
        ("stageCompleted", "evidence"),
        ("explorationPlanned", "evidence"),
        ("candidateMeasured", "evidence"),
        ("finalistMeasured", "evidence"),
        ("modelRequest", "requestPath"),
        ("modelResponse", "responsePath"),
    }
    accepted = [row for row in records.events() if row["kind"] == "decisionAccepted"]
    responses = {
        row["callId"]: row["responseSha256"]
        for row in records.events()
        if row["kind"] == "modelResponse"
    }
    assert [row["responseSha256"] for row in accepted] == [
        responses[row["callId"]] for row in accepted
    ]


def test_model_settings_are_saved_as_names_and_a_digest_only(
    science: Any, tmp_path: Path
) -> None:
    settings = {"temperature": 0.2, "extra_body": {"region": "secret-endpoint-value"}}
    result = _analyze(
        science,
        tmp_path,
        _model([], context_question="Which tissue was sampled?"),
        runtime=RuntimeConfig(maxRequests=12, modelSettings=settings),
    )
    assert result.status == "needsInput"
    records = RunRecords(result.run_dir)
    invocation = records.latest("invocationStarted")
    assert invocation["runtime"] == {"maxRequests": 12}
    assert invocation["modelSettings"] == {
        "keys": ["extra_body", "temperature"],
        "sha256": digest(settings),
    }
    for path in (records.path / "events").iterdir():
        assert "secret-endpoint-value" not in path.read_text()
    question = result.pending_questions[0]
    before = records.events()
    with pytest.raises(ValueError, match="supplied modelSettings, which are not saved"):
        resume_rna(
            result.run_dir, model=_model([]), answers={question["questionId"]: "Blood"}
        )
    assert records.events() == before
    resumed = resume_rna(
        result.run_dir,
        model=_model([]),
        answers={question["questionId"]: "Blood"},
        runtime=RuntimeConfig(modelSettings=settings),
    )
    assert resumed.status == "completed"


def test_resume_refuses_a_prompt_limit_below_the_evidence_budget(
    science: Any, tmp_path: Path
) -> None:
    result = _analyze(
        science, tmp_path, _model([], context_question="Which tissue was sampled?")
    )
    records = RunRecords(result.run_dir)
    before = records.events()
    question = result.pending_questions[0]
    with pytest.raises(ValueError, match="maxPromptBytes must be at least 65,536"):
        resume_rna(
            result.run_dir,
            model=_model([]),
            answers={question["questionId"]: "Blood"},
            runtime=RuntimeConfig(maxPromptBytes=40_000),
        )
    # No pipeline has been planned, so there is nothing to abandon.
    with pytest.raises(ValueError, match="No unfinished pipeline attempt"):
        resume_rna(
            result.run_dir,
            model=_model([]),
            answers={question["questionId"]: "Blood"},
            abandon_unfinished_pipeline=True,
        )
    assert records.events() == before


class _HardStop(BaseException):
    """A process killed before it could journal the pipeline outcome."""


def _stranded(science: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    execute = workflow.execute_pipeline
    append = RunRecords.append

    def stop(*args: Any, **kwargs: Any) -> Any:
        raise _HardStop

    def unjournaled(records: Any, kind: str, **payload: Any) -> Any:
        if kind == "pipelineFailed":
            raise OSError("process stopped before journaling")
        return append(records, kind, **payload)

    with monkeypatch.context() as patch:
        patch.setattr(workflow, "execute_pipeline", stop)
        patch.setattr(RunRecords, "append", unjournaled)
        result = _analyze(science, tmp_path, _model([]))
    assert result.status == "failed"
    records = RunRecords(result.run_dir)
    planned = records.latest("pipelinePlanned")
    assert planned["operation"] == "screen_c0"
    assert records.latest("pipelineFailed") is None
    assert science.calls == []
    monkeypatch.setattr(workflow, "execute_pipeline", execute)
    # Windows cannot probe whether the recorded process still runs.
    monkeypatch.setattr(workflow, "process_alive", lambda identity: True)
    monkeypatch.setattr(api, "process_state", lambda identity: "unknown")
    return result, records, planned


def test_unknown_stopped_process_requires_and_accepts_explicit_abandonment(
    science: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, records, planned = _stranded(science, tmp_path, monkeypatch)
    blocked = resume_rna(result.run_dir, model=_model([]))
    assert blocked.status == "failed"
    status = records.latest("status")
    assert status["errorType"] == "PipelineProcessActive"
    assert "abandon_unfinished_pipeline=True" in status["message"]
    assert science.calls == []
    resumed = resume_rna(
        result.run_dir, model=_model([]), abandon_unfinished_pipeline=True
    )
    assert resumed.status == "completed"
    abandoned = records.latest("pipelineAbandoned")
    assert {
        key: abandoned[key] for key in ("operation", "label", "plannedSequence")
    } == {
        "operation": "screen_c0",
        "label": planned["label"],
        "plannedSequence": planned["sequence"],
    }
    assert abandoned["process"] == planned["process"]
    assert science.calls[0]["label"] == (
        f"agent_{records.manifest['runId']}_screen_c0_1"
    )
    before = records.events()
    with pytest.raises(ValueError, match="No unfinished pipeline attempt"):
        resume_rna(result.run_dir, model=None, abandon_unfinished_pipeline=True)
    assert records.events() == before


def test_a_confirmed_running_process_is_never_abandoned(
    science: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, records, _ = _stranded(science, tmp_path, monkeypatch)
    monkeypatch.setattr(api, "process_state", lambda identity: "running")
    before = records.events()
    with pytest.raises(ValueError, match="still running; it cannot be abandoned"):
        resume_rna(result.run_dir, model=_model([]), abandon_unfinished_pipeline=True)
    assert records.events() == before
    assert science.calls == []


def test_locators_fall_back_to_absolute_paths_when_no_relative_path_exists(
    science: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relpath = os.path.relpath

    def across_drives(path: Any, start: Any = None) -> str:
        if str(path).startswith(str(tmp_path)) or str(start).startswith(str(tmp_path)):
            raise ValueError("path is on mount 'D:', start on mount 'C:'")
        return relpath(path, start) if start is not None else relpath(path)

    monkeypatch.setattr(os.path, "relpath", across_drives)
    result = _analyze(science, tmp_path, _model([]))
    assert result.status == "completed"
    records = RunRecords(result.run_dir)
    assert records.manifest["source"] == str(science.source)
    assert open_analysis(result.run_dir).source == science.source
    moved = tmp_path / "moved-source"
    moved.mkdir()
    resume_rna(result.run_dir, model=None, source=moved)
    assert records.latest("sourceRebound")["source"] == str(moved)
    assert open_analysis(result.run_dir).source == moved
