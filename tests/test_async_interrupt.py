"""An interrupted caller in a running event loop stops its storage operation."""

import asyncio
import os
import signal
import threading
import time

import pytest

from cytearc.storage.async_execution import AsyncStorageRunner
from cytearc.storage.budget import ResourceBudget
from cytearc.storage.execution import WorkShape, plan_operation
from tests.storage_helpers import reset_zarr_runtime


@pytest.fixture(autouse=True)
def _reset_runtime() -> None:
    reset_zarr_runtime()
    yield
    reset_zarr_runtime()


def _runner() -> AsyncStorageRunner:
    plan = plan_operation(
        ResourceBudget(memoryBytes=1 << 30, workers=1),
        WorkShape(nUnits=12, unitBytes=1024),
    )
    return AsyncStorageRunner(operation=plan)


@pytest.mark.skipif(not hasattr(signal, "SIGINT"), reason="needs SIGINT")
def test_an_interrupted_notebook_cell_stops_its_operation_before_it_returns():
    events: list[str] = []
    first_unit = threading.Event()

    def unit(index: int) -> None:
        first_unit.set()
        time.sleep(0.5)
        events.append(f"unit {index}")

    async def operation(runner: AsyncStorageRunner) -> str:
        for index in range(12):
            await runner.compute(lambda index=index: unit(index))
        events.append("finished")
        return "done"

    async def cell() -> str:
        # Synchronous CyteArc code inside the kernel's running loop.
        return _runner().run(operation)

    def interrupt() -> None:
        first_unit.wait()
        time.sleep(0.05)
        # Ctrl-C in a notebook: the default handler raises KeyboardInterrupt
        # in the main thread, which is waiting for the helper thread.
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=interrupt, daemon=True).start()
    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(KeyboardInterrupt):
            loop.run_until_complete(cell())
    finally:
        loop.close()

    stopped = list(events)
    time.sleep(1.2)
    # The unit running at the interruption finished; nothing ran after the
    # caller saw the interruption, and the operation never completed.
    assert events == stopped
    assert "finished" not in events
    assert 1 <= len(events) <= 2
    assert not any(
        thread.name == "cytearc-storage-operation" for thread in threading.enumerate()
    )


def test_an_operation_that_finishes_inside_a_running_loop_returns_its_result():
    async def operation(runner: AsyncStorageRunner) -> int:
        return await runner.compute(lambda: 7)

    async def cell() -> int:
        return _runner().run(operation)

    assert asyncio.run(cell()) == 7


def test_an_interruption_after_the_operation_finished_still_propagates(monkeypatch):
    finished = threading.Event()

    async def operation(runner: AsyncStorageRunner) -> int:
        finished.set()
        return 1

    original_join = threading.Thread.join
    calls: list[str] = []

    def join(thread: threading.Thread, timeout: float | None = None) -> None:
        if thread.name == "cytearc-storage-operation" and not calls:
            calls.append("interrupted")
            original_join(thread, timeout)
            # The operation's loop has closed when the caller is interrupted.
            raise KeyboardInterrupt
        return original_join(thread, timeout)

    monkeypatch.setattr(threading.Thread, "join", join)

    async def cell() -> int:
        return _runner().run(operation)

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(cell())
    assert finished.is_set() and calls == ["interrupted"]
