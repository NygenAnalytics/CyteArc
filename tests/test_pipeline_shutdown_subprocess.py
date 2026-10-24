import json
import os
import select
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from cytearc import DataStore
from cytearc.storage.pipeline_runs import (
    list_pipeline_run_records,
    load_pipeline_stage_records,
)
from tests.test_pipeline_run_record_paths import (
    _GRAPH_STAGE_STATUSES,
    _GRAPH_STAGES,
    _PCA,
    _write_graph_store,
)

# Each child runs the RNA recipe with the settings given as its last argument.
_CHILD = textwrap.dedent(
    """
    import json
    import sys
    import time

    from cytearc import DataStore
    from cytearc.utils.shutdown import shutdown_checkpoint

    store = DataStore(sys.argv[1], default_assay="RNA")
    cooperative = sys.argv[2] == "cooperative"

    def paused_pca(self, *_args, **_kwargs):
        print("READY", flush=True)
        while True:
            if cooperative:
                shutdown_checkpoint()
            time.sleep(0.01)

    type(store)._reduction_pca = paused_pca
    store.pipeline.rna(**json.loads(sys.argv[3]))
    """
)


_SIGNAL_RACE_CHILD = textwrap.dedent(
    """
    import json
    import os
    import signal
    import sys
    import time

    import cytearc.datastore._pipeline.executor as executor
    from cytearc import DataStore
    from cytearc.utils.shutdown import current_shutdown_token

    store = DataStore(sys.argv[1], default_assay="RNA")
    case = sys.argv[2]

    def requested():
        token = current_shutdown_token()
        return token is not None and token.requested

    def pca_failing_after_signal(self, *_args, **_kwargs):
        # PCA never checks for shutdown, so SIGTERM only marks the run's
        # token, and the stage then fails for a reason of its own.
        print("READY", flush=True)
        while not requested():
            time.sleep(0.01)
        raise RuntimeError("PCA failed after SIGTERM")

    open_completed_run = executor.open_pipeline_run

    def open_then_signal(*args, **kwargs):
        # SIGTERM arrives once the run is durably complete, while the run is
        # handed back to the caller.
        run = open_completed_run(*args, **kwargs)
        os.kill(os.getpid(), signal.SIGTERM)
        while not requested():
            time.sleep(0.01)
        return run

    if case == "operation_failure":
        type(store)._reduction_pca = pca_failing_after_signal
    else:
        executor.open_pipeline_run = open_then_signal
    store.pipeline.rna(**json.loads(sys.argv[3]))
    """
)


# A user stops the run, and one signal from the test then arrives before PCA
# reaches its next checkpoint.
_CANCEL_THEN_SIGNAL_CHILD = textwrap.dedent(
    """
    import json
    import signal
    import sys
    import time

    from cytearc import CancellationToken, DataStore
    from cytearc.utils.shutdown import current_shutdown_token, shutdown_checkpoint

    # The signals take their default behavior, whatever the parent ignores.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    store = DataStore(sys.argv[1], default_assay="RNA")
    cancel = CancellationToken()

    def cancelled_pca(self, *_args, **_kwargs):
        cancel.cancel("stopped by the user")
        token = current_shutdown_token()
        print("READY", flush=True)
        while token.signal_record is None:
            time.sleep(0.01)
        print("SIGNALLED", token.signal_record.signal_name, flush=True)
        shutdown_checkpoint()

    type(store)._reduction_pca = cancelled_pca
    try:
        store.pipeline.rna(cancel=cancel, **json.loads(sys.argv[3]))
    except BaseException as error:
        print("RAISED", type(error).__name__, flush=True)
    """
)


def _prepared_store(path: Path) -> Path:
    """Write a small RNA store and prepare it as every child opens it.

    The signal contract does not depend on data size, so each child reaches
    its PCA stage, or completes its run, after only a few quick stages.
    """
    _write_graph_store(path)
    DataStore(str(path), default_assay="RNA")
    return path


def _child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "2",
            "NUMEXPR_MAX_THREADS": "1",
        }
    )
    return environment


def _start_child(path: Path, child: str, case: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", child, str(path), case, json.dumps(_GRAPH_STAGES)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_child_environment(),
    )


def _wait_for_stdout(process: subprocess.Popen[str], marker: str) -> None:
    assert process.stdout is not None
    observed: list[str] = []
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        ready, _, _ = select.select(
            [process.stdout],
            [],
            [],
            max(0.0, deadline - time.monotonic()),
        )
        if not ready:
            break
        line = process.stdout.readline().strip()
        if line == marker:
            return
        if line:
            observed.append(line)
        if process.poll() is not None:
            break
    process.kill()
    _stdout, stderr = process.communicate(timeout=10)
    pytest.fail(
        f"Pipeline child did not emit {marker!r}: "
        + " | ".join(observed[-3:])
        + f"\n{stderr}"
    )


def _communicate(
    process: subprocess.Popen[str],
    *,
    timeout: float,
) -> tuple[str, str]:
    try:
        return process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate(timeout=10)
        pytest.fail(f"Pipeline child timed out:\n{stdout}\n{stderr}")


def _paused_pipeline(path: Path, *, cooperative: bool) -> subprocess.Popen[str]:
    process = _start_child(
        path,
        _CHILD,
        "cooperative" if cooperative else "blocked",
    )
    _wait_for_stdout(process, "READY")
    return process


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal contract")
def test_sigterm_commits_interrupted_pipeline_before_propagation(
    tmp_path: Path,
) -> None:
    location = _prepared_store(tmp_path / "sigterm.zarr")
    process = _paused_pipeline(location, cooperative=True)

    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=30) == -signal.SIGTERM
    root = DataStore(str(location), default_assay="RNA").zw
    runs = list_pipeline_run_records(root, limit=100)
    assert len(runs) == 1
    assert runs[0].status == "interrupted"
    assert runs[0].complete
    interruption = runs[0].interruption
    assert interruption is not None
    assert (
        interruption.kind,
        interruption.signal_number,
        interruption.signal_name,
    ) == (
        "signal",
        signal.SIGTERM,
        "SIGTERM",
    )
    stages = load_pipeline_stage_records(root, runs[0].run_id)
    assert [(stage.stage, stage.status) for stage in stages] == [
        *_GRAPH_STAGE_STATUSES[:_PCA],
        ("pca", "interrupted"),
    ]
    assert all(stage.complete for stage in stages)
    assert stages[-1].interruption is not None
    assert stages[-1].interruption.signal_name == "SIGTERM"


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal contract")
@pytest.mark.parametrize(
    ("signum", "returncode", "raised"),
    [
        # SIGTERM's default action then ends the process.
        (signal.SIGTERM, -signal.SIGTERM, []),
        # SIGINT's handler then raises KeyboardInterrupt to the caller.
        (signal.SIGINT, 0, ["RAISED KeyboardInterrupt"]),
    ],
    ids=["sigterm", "sigint"],
)
def test_a_signal_after_a_cancellation_takes_effect_once_the_run_is_recorded(
    tmp_path: Path,
    signum: signal.Signals,
    returncode: int,
    raised: list[str],
) -> None:
    location = _prepared_store(tmp_path / f"cancel-{signum.name}.zarr")
    process = _start_child(location, _CANCEL_THEN_SIGNAL_CHILD, signum.name)
    _wait_for_stdout(process, "READY")

    process.send_signal(signum)
    stdout, stderr = _communicate(process, timeout=30)

    # The signal did not escalate: PCA saw it, stopped at its checkpoint, and
    # the signal took its usual effect only once the run was recorded.
    assert process.returncode == returncode, stderr
    lines = stdout.splitlines()
    assert f"SIGNALLED {signum.name}" in lines, stdout
    assert [line for line in lines if line.startswith("RAISED")] == raised, stdout
    root = DataStore(str(location), default_assay="RNA").zw
    runs = list_pipeline_run_records(root, limit=100)
    assert len(runs) == 1
    assert (runs[0].status, runs[0].complete) == ("interrupted", True)
    interruption = runs[0].interruption
    assert interruption is not None
    # The run records the cancellation, which came before the signal.
    assert (
        interruption.kind,
        interruption.message,
        interruption.signal_name,
    ) == ("cancelled", "stopped by the user", None)
    stages = load_pipeline_stage_records(root, runs[0].run_id)
    assert [(stage.stage, stage.status) for stage in stages] == [
        *_GRAPH_STAGE_STATUSES[:_PCA],
        ("pca", "interrupted"),
    ]
    assert all(stage.complete for stage in stages)
    assert stages[-1].interruption is not None
    assert stages[-1].interruption.kind == "cancelled"


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal contract")
def test_pending_sigterm_propagates_after_operation_failure(
    tmp_path: Path,
) -> None:
    location = _prepared_store(tmp_path / "sigterm-operation-failure.zarr")
    process = _start_child(location, _SIGNAL_RACE_CHILD, "operation_failure")
    _wait_for_stdout(process, "READY")

    process.send_signal(signal.SIGTERM)
    _stdout, stderr = _communicate(process, timeout=30)
    returncode = process.returncode

    root = DataStore(str(location), default_assay="RNA").zw
    runs = list_pipeline_run_records(root, limit=100)
    assert len(runs) == 1
    assert runs[0].status == "failed"
    assert runs[0].complete
    assert runs[0].interruption is None
    assert runs[0].error is not None
    assert (runs[0].error.type, runs[0].error.message) == (
        "RuntimeError",
        "PCA failed after SIGTERM",
    )
    stages = load_pipeline_stage_records(root, runs[0].run_id)
    assert [(stage.stage, stage.status) for stage in stages] == [
        *_GRAPH_STAGE_STATUSES[:_PCA],
        ("pca", "failed"),
    ]
    assert all(stage.complete for stage in stages)
    assert stages[-1].error is not None
    assert stages[-1].error.type == "RuntimeError"
    # The operation failure is recorded first; the pending signal then ends
    # the process.
    assert returncode == -signal.SIGTERM, stderr


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal contract")
def test_pending_sigterm_propagates_after_completed_run_handoff(
    tmp_path: Path,
) -> None:
    location = _prepared_store(tmp_path / "sigterm-completed-handoff.zarr")
    process = _start_child(location, _SIGNAL_RACE_CHILD, "completed_handoff")

    _stdout, stderr = _communicate(process, timeout=60)
    returncode = process.returncode

    root = DataStore(str(location), default_assay="RNA").zw
    runs = list_pipeline_run_records(root, limit=100)
    assert len(runs) == 1
    assert runs[0].status == "completed"
    assert runs[0].complete
    assert runs[0].interruption is None
    stages = load_pipeline_stage_records(root, runs[0].run_id)
    assert [(stage.stage, stage.status) for stage in stages] == list(
        _GRAPH_STAGE_STATUSES
    )
    assert all(stage.complete for stage in stages)
    assert returncode == -signal.SIGTERM, stderr


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal contract")
def test_sigkill_leaves_incomplete_run_and_a_new_run_reuses_only_complete_outputs(
    tmp_path: Path,
) -> None:
    location = _prepared_store(tmp_path / "sigkill.zarr")
    process = _paused_pipeline(location, cooperative=False)

    process.send_signal(signal.SIGKILL)
    assert process.wait(timeout=30) == -signal.SIGKILL
    store = DataStore(str(location), default_assay="RNA")
    incomplete = list_pipeline_run_records(store.zw, limit=100)
    assert len(incomplete) == 1
    assert incomplete[0].status == "running"
    assert not incomplete[0].complete
    assert load_pipeline_stage_records(store.zw, incomplete[0].run_id)[-1].status == (
        "running"
    )

    recovered = store.pipeline.rna(**_GRAPH_STAGES)
    assert recovered.status == "completed"
    report = recovered.report()
    # Stages the killed run completed are reused; PCA onward is created.
    dispositions = {
        stage["stage"]: {plan["disposition"] for plan in stage["plans"]}
        for stage in report["stages"]
        if stage["plans"]
    }
    assert dispositions == {
        "input_snapshot": {"reused"},
        "highly_variable_features": {"reused"},
        "normalization": {"reused"},
        "pca": {"created"},
        "ann_index": {"created"},
        "neighbors": {"created"},
        "connectivity": {"created"},
    }
    assert all(
        store.artifacts.inspect(plan_ref).complete for plan_ref in recovered.values()
    )
