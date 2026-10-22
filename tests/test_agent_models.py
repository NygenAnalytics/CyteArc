"""Scientific and operational requests fail closed before a run is created."""

from typing import Any

import pytest
from pydantic import ValidationError

from cytearc.agent.models import AnalysisConfig, RuntimeConfig, Study


@pytest.mark.parametrize("field", ["context", "objective"])
def test_study_rejects_whitespace_in_required_scientific_context(field: str) -> None:
    values = {"context": "Supplied observations", "objective": "Discover populations"}
    values[field] = " \t\n "
    with pytest.raises(ValidationError, match="Study text must not be blank"):
        Study.model_validate(values)


@pytest.mark.parametrize("resolutions", [(), (0.25, 0.5, 0.75, 1.0, 1.25)])
def test_scientific_search_requires_one_to_four_resolutions(
    resolutions: tuple[float, ...],
) -> None:
    with pytest.raises(ValidationError, match="one to four Leiden resolutions"):
        AnalysisConfig(resolutions=resolutions)


@pytest.mark.parametrize(
    "resolutions",
    [(0.5, 0.5), (0.0,), (-0.5,), (float("nan"),), (float("inf"),)],
)
def test_scientific_search_rejects_duplicate_or_invalid_resolutions(
    resolutions: tuple[float, ...],
) -> None:
    with pytest.raises(ValidationError, match="distinct positive finite values"):
        AnalysisConfig(resolutions=resolutions)


@pytest.mark.parametrize(
    "values",
    [
        {"qcPolicy": "manual"},
        {"qcPolicy": "retain", "qcBounds": {"RNA_nCounts": (10, None)}},
        {"qcPolicy": "gentleMad5", "qcBounds": {"RNA_nCounts": (10, None)}},
    ],
)
def test_manual_thresholds_cannot_be_ignored_or_invented_by_other_qc_policies(
    values: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError, match="supplied only for manual QC"):
        AnalysisConfig.model_validate(values)


@pytest.mark.parametrize(
    "bounds",
    [
        {"": (1.0, 2.0)},
        {"RNA_nCounts": (float("nan"), None)},
        {"RNA_nCounts": (None, float("inf"))},
    ],
)
def test_manual_qc_requires_named_finite_bounds(
    bounds: dict[str, tuple[float | None, float | None]],
) -> None:
    with pytest.raises(ValidationError, match="names and finite values"):
        AnalysisConfig(qcPolicy="manual", qcBounds=bounds)


def test_manual_qc_accepts_asymmetric_bounds_but_rejects_inverted_ranges() -> None:
    bounds = {"RNA_nCounts": (100.0, None), "RNA_percentMito": (None, 15.0)}
    request = AnalysisConfig(qcPolicy="manual", qcBounds=bounds)
    assert request.qcBounds == bounds
    with pytest.raises(ValidationError, match="lower bound must not exceed upper"):
        AnalysisConfig(qcPolicy="manual", qcBounds={"RNA_nCounts": (100.0, 50.0)})


@pytest.mark.parametrize(
    "values",
    [
        {"maxCandidates": 6},
        {"maxCandidates": 0},
        {"maxFinalists": 3},
        {"maxFinalists": 0},
        {"hvgCount": 1},
        {"pcaDims": 1},
        {"neighborsK": 1},
        {"randomSeed": -1},
    ],
)
def test_scientific_request_enforces_supported_execution_limits(
    values: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError) as error:
        AnalysisConfig.model_validate(values)
    assert error.value.errors()[0]["loc"] == (next(iter(values)),)


@pytest.mark.parametrize(
    "settings",
    [
        {"API-Key": "not-a-real-key"},
        {"extra_body": {"authorization": "not-a-real-token"}},
        {"metadata": [{"password": "not-a-real-password"}]},
        {"metadata": ({"cookie": "not-a-real-cookie"},)},
        {"extra_headers": {"custom": "value"}},
    ],
)
def test_runtime_rejects_credentials_even_in_nested_saved_settings(
    settings: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError, match="Configure credentials on the provider"):
        RuntimeConfig(modelSettings=settings)


@pytest.mark.parametrize("setting", ["max_tokens", "max_output_tokens", "timeout"])
def test_runtime_requires_explicit_output_and_deadline_overrides(setting: str) -> None:
    with pytest.raises(ValidationError, match="explicit runtime output-token"):
        RuntimeConfig(modelSettings={setting: 256})


def test_runtime_preserves_safe_model_configuration_and_explicit_overrides() -> None:
    settings = {
        "temperature": 0.2,
        "metadata": [{"purpose": "provisional annotation"}],
        "extra_body": {"sampling": {"top_p": 0.8}, "stop": ("END",)},
    }
    runtime = RuntimeConfig(
        modelSettings=settings, maxOutputTokens=512, decisionTimeout=45.0
    )
    assert runtime.modelSettings == settings
    assert runtime.model_dump(exclude_unset=True) == {
        "modelSettings": settings,
        "maxOutputTokens": 512,
        "decisionTimeout": 45.0,
    }
    assert RuntimeConfig().model_dump(exclude_unset=True) == {}


@pytest.mark.parametrize(
    "values",
    [
        {"maxRequests": 0},
        {"maxRequestsPerDecision": 4},
        {"maxPromptBytes": 1023},
        {"maxOutputTokens": 127},
        {"decisionTimeout": 0},
        {"decisionTimeout": float("inf")},
        {"nthreads": 0},
    ],
)
def test_runtime_rejects_invalid_provider_and_resource_limits(
    values: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError) as error:
        RuntimeConfig.model_validate(values)
    assert error.value.errors()[0]["loc"] == (next(iter(values)),)


def test_requests_reject_unknown_fields_and_reassignment_of_frozen_policy() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AnalysisConfig.model_validate({"autoRepair": True})
    request = AnalysisConfig()
    with pytest.raises(ValidationError, match="Instance is frozen"):
        request.scoreDoublets = True
    assert request.scoreDoublets is False


def test_lenient_native_probe_policy_is_frozen_and_old_budgets_remain_readable() -> (
    None
):
    config = AnalysisConfig()
    assert config.interactionMode == "lenient"
    assert config.maxCandidates == 5
    assert AnalysisConfig(interactionMode="strict", maxCandidates=4).maxCandidates == 4
    for old_budget in (1, 2, 3):
        assert (
            AnalysisConfig.model_validate({"maxCandidates": old_budget}).maxCandidates
            == old_budget
        )
    with pytest.raises(ValidationError, match="Instance is frozen"):
        config.interactionMode = "strict"


@pytest.mark.parametrize("budget", [1, 2, 3])
def test_new_runs_refuse_old_search_budgets_before_creating_records(
    budget: int, tmp_path: Any
) -> None:
    from cytearc.agent import analyze_rna

    destination = tmp_path / "analysis"
    with pytest.raises(ValueError, match="four native probes"):
        analyze_rna(
            tmp_path,
            run_dir=destination,
            model="unused-model",
            study=Study(context="Observed cells", objective="Describe populations"),
            config=AnalysisConfig(maxCandidates=budget),
        )
    assert not destination.exists()


@pytest.mark.parametrize(
    "settings",
    [
        {"extra_body": {"credentials": "not-a-real-credential"}},
        {"extra_body": {"auth": "not-a-real-token"}},
        {"extra_body": {"key": "not-a-real-key"}},
        {"extra_body": {"bearer": "not-a-real-token"}},
        {"extra_body": {"aws_access_key_id": "not-a-real-id"}},
        {"extra_body": {"privateKey": "not-a-real-key"}},
        {"extra_body": {"x-api-key": "not-a-real-key"}},
        {"extra_body": [{"passwd": "not-a-real-password"}]},
        {"extra_body": {"oauth": {"client": "not-a-real-client"}}},
        {"extra_body": {"session": "not-a-real-session"}},
    ],
)
def test_runtime_rejects_credential_names_the_earlier_denylist_missed(
    settings: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError, match="Configure credentials on the provider"):
        RuntimeConfig(modelSettings=settings)


def test_runtime_keeps_known_non_secret_settings_with_similar_names() -> None:
    settings = {
        "openai_prompt_cache_key": "analysis-cache",
        "bedrock_additional_model_requests_fields": {"top_k": 4},
        "extra_body": {"keyword_boost": 2, "monkey_patch": False},
    }
    assert RuntimeConfig(modelSettings=settings).modelSettings == settings


@pytest.mark.parametrize(
    ("resolutions", "labels"),
    [
        ((1.0, 1.0000001), "['1', '1']"),
        ((0.5, 0.75, 0.7500000001), "['0.5', '0.75', '0.75']"),
    ],
)
def test_resolutions_must_keep_distinct_option_labels(
    resolutions: tuple[float, ...], labels: str
) -> None:
    from cytearc.agent.models import partition_option_id

    # Both values would otherwise share one partition option ID.
    assert len({partition_option_id("c0", value) for value in resolutions}) < len(
        resolutions
    )
    with pytest.raises(ValidationError) as raised:
        AnalysisConfig(resolutions=resolutions)
    assert f"option labels {labels} must be distinct" in str(raised.value)
    assert AnalysisConfig(resolutions=(1.0, 1.00001)).resolutions == (1.0, 1.00001)


@pytest.mark.parametrize(
    ("field", "limit"),
    [
        ("context", 8000),
        ("objective", 2000),
        ("organism", 200),
        ("tissue", 200),
        ("batchCorrectionEvidence", 4000),
        ("sampleColumn", 128),
        ("captureColumn", 128),
    ],
)
def test_study_text_that_enters_every_prompt_is_bounded(field: str, limit: int) -> None:
    supplied = {"context": "Observed cells", "objective": "Describe populations"}
    with pytest.raises(ValidationError) as raised:
        Study.model_validate({**supplied, field: "x" * (limit + 1)})
    assert raised.value.errors()[0]["loc"] == (field,)
    assert raised.value.errors()[0]["ctx"]["max_length"] == limit
    accepted = Study.model_validate({**supplied, field: "x" * limit})
    assert getattr(accepted, field) == "x" * limit


@pytest.mark.parametrize(
    ("field", "value", "location"),
    [
        ("technicalBatchColumns", [f"batch{index}" for index in range(17)], ()),
        ("protectedColumns", [f"group{index}" for index in range(17)], ()),
        ("protectedColumns", ["x" * 129], (0,)),
        ("featureExclusions", [f"GENE{index}" for index in range(201)], ()),
        ("featureExclusions", ["G" * 129], (0,)),
    ],
)
def test_study_role_and_exclusion_lists_are_bounded(
    field: str, value: list[str], location: tuple[int, ...]
) -> None:
    supplied = {"context": "Observed cells", "objective": "Describe populations"}
    with pytest.raises(ValidationError) as raised:
        Study.model_validate({**supplied, field: value})
    assert raised.value.errors()[0]["loc"] == (field, *location)


def test_new_runs_and_resumes_refuse_a_prompt_limit_below_the_evidence_budget(
    tmp_path: Any,
) -> None:
    from cytearc.agent import analyze_rna

    destination = tmp_path / "analysis"
    with pytest.raises(ValueError, match="maxPromptBytes must be at least 65,536"):
        analyze_rna(
            tmp_path,
            run_dir=destination,
            model="unused-model",
            study=Study(context="Observed cells", objective="Describe populations"),
            runtime=RuntimeConfig(maxPromptBytes=65_535),
        )
    assert not destination.exists()
    # Saved runtimes with smaller limits remain readable.
    assert RuntimeConfig(maxPromptBytes=1024).maxPromptBytes == 1024
