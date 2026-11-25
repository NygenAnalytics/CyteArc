"""The helpers that read the budget a memory refusal names and trace what a call or a pipe end holds."""

import pytest

from tests.memory_helpers import (
    TracedPipeEnd,
    named_budget,
    traced_charge,
    traced_reruns,
)


def test_named_budget_reads_the_budget_that_a_refusal_closes_with() -> None:
    assert (
        named_budget(
            MemoryError(
                "Counting guides holds about 5000 bytes, more than the memory budget "
                "of 100 bytes. Raise mem_budget to at least 5000 bytes."
            )
        )
        == 5000
    )
    assert (
        named_budget(
            MemoryError(
                "Labeling holds about 812 bytes. Raise mem_budget to at least 812 "
                "bytes, or pass a smaller cell_selection."
            )
        )
        == 812
    )
    # A refusal that names no budget asks for the bytes it holds.
    assert (
        named_budget(MemoryError("A batch holds about 77 bytes. Raise mem_budget."))
        == 77
    )


@pytest.mark.parametrize(
    "message",
    [
        # A budget other than the bytes the call holds.
        "Labeling holds about 812 bytes. Raise mem_budget to at least 900 bytes.",
        # No closing budget.
        "Labeling holds about 812 bytes, more than the budget.",
        # A closing request that says nothing of what the call holds.
        "Labeling does not fit. Raise mem_budget.",
    ],
)
def test_named_budget_refuses_a_refusal_that_names_no_budget_it_holds(
    message: str,
) -> None:
    with pytest.raises(AssertionError):
        named_budget(MemoryError(message))


def test_traced_reruns_call_once_untraced_before_the_traced_calls() -> None:
    import gc
    import tracemalloc

    tracing: list[bool] = []
    finalized: list[bool] = []

    class Garbage:
        """An object that only a collection frees, which notes whether tracemalloc traced as it went."""

        def __init__(self) -> None:
            self.cycle = self

        def __del__(self) -> None:
            finalized.append(tracemalloc.is_tracing())

    def call() -> int:
        if not tracing:
            Garbage()
        tracing.append(tracemalloc.is_tracing())
        return len(tracing)

    # The first call runs untraced, so it alone can be a first call of the
    # functions that every call runs; the traced calls' results come back.
    # What the untraced call leaves for a collection goes before tracing.
    enabled = gc.isenabled()
    gc.disable()
    try:
        assert traced_reruns(call, runs=2) == [2, 3]
    finally:
        if enabled:
            gc.enable()
    assert tracing == [False, True, True]
    assert finalized == [False]
    assert not tracemalloc.is_tracing()


def test_traced_charge_leaves_out_what_only_a_first_call_allocates() -> None:
    kept: list[bytearray] = []

    def call(_ledger: object) -> None:
        # The first call keeps 4 MB, as a cache or coverage's record of a
        # function would; every call holds 1 MB while it runs.
        if not kept:
            kept.append(bytearray(4_000_000))
        held = bytearray(1_000_000)
        del held

    traced = traced_charge(call)

    assert len(kept) == 1
    assert 1_000_000 <= traced.held < 2_000_000


def test_a_traced_pipe_end_makes_no_first_call_in_a_traced_send() -> None:
    import sys
    import tracemalloc

    # A monitoring tool of its own sees each function that starts to run.
    monitoring = sys.monitoring
    tool = next(tool for tool in range(6) if monitoring.get_tool(tool) is None)
    started: list[object] = []
    monitoring.use_tool_id(tool, "functions that a traced pipe end runs")
    try:
        monitoring.register_callback(
            tool,
            monitoring.events.PY_START,
            lambda code, _offset: started.append(code),
        )
        monitoring.set_events(tool, monitoring.events.PY_START)
        pipe = TracedPipeEnd()
        made = set(started)
        tracemalloc.start()
        try:
            started.clear()
            pipe.send(("band", [8]))
            pipe.send_bytes(bytes(8))
            sent = set(started)
        finally:
            tracemalloc.stop()
            monitoring.set_events(tool, 0)
    finally:
        monitoring.register_callback(tool, monitoring.events.PY_START, None)
        monitoring.free_tool_id(tool)

    # Every function that the traced sends ran had run as the pipe end was
    # made, and only the traced sends were recorded.
    assert sent and sent <= made
    assert [form for form, _value in pipe.sent] == ["pickled", "bytes"]
    assert len(pipe.held) == 2
