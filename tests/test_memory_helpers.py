"""The helpers that compare a traced peak with its charge, read the budget a memory refusal names and trace what a call or a pipe end holds."""

import pytest

from tests.memory_helpers import (
    TracedPipeEnd,
    climb_refusals,
    held_within_charge,
    named_budget,
    traced_charge,
    traced_reruns,
)


def test_a_peak_stays_within_its_charge_up_to_a_tenth_of_it_and_128_kib() -> None:
    # A charge of 1,000,000 bytes allows 1,100,000 and 131,072 more.
    assert held_within_charge(1_231_072, 1_000_000)
    assert not held_within_charge(1_231_073, 1_000_000)
    # A call charged nothing may hold the 131,072 bytes alone.
    assert held_within_charge(131_072, 0)
    assert not held_within_charge(131_073, 0)
    assert held_within_charge(0, 1_000_000)
    # Three arrays of 8 bytes for each of 30,000 cells pass a charge that
    # leaves one of them out, and a charge a tenth short is within the slack.
    assert not held_within_charge(720_000, 480_000)
    assert held_within_charge(720_000, 648_000)


def test_named_budget_reads_the_budget_that_a_refusal_closes_with() -> None:
    from cytearc.utils.memory_budget import MemoryBudgetError, memory_refusal

    # 5,000 bytes and a quarter are within one MiB.
    assert (
        named_budget(
            MemoryError(
                "Counting guides holds about 5000 bytes, more than the memory budget "
                "of 100 bytes. Raise mem_budget to at least 1 MiB."
            )
        )
        == 1024**2
    )
    # 4,000,000 bytes and a quarter are 4.77 MiB; a remedy may hold periods.
    assert (
        named_budget(
            MemoryError(
                "Labeling holds about 4000000 bytes. Raise mem_budget to at least 5 "
                "MiB, or pass policy=CountMatrixPolicy(unitBytes=8) from "
                "cytearc.storage.count_matrix."
            )
        )
        == 5 * 1024**2
    )
    # A refusal of its own words names its suggested budget.
    error = MemoryBudgetError(
        "One row does not fit", needed_bytes=3_000_000, budget_bytes=10, item="a row"
    )
    assert named_budget(error) == error.suggested_budget_bytes == 4 * 1024**2
    assert named_budget(memory_refusal("Labeling", "the labels", 812, 10)) == 1024**2


@pytest.mark.parametrize(
    "message",
    [
        # A budget other than the suggested budget of the bytes the call holds.
        "Labeling holds about 4000000 bytes. Raise mem_budget to at least 4 MiB.",
        # No closing budget.
        "Labeling holds about 812 bytes, more than the budget.",
        # A closing request without a budget.
        "Labeling holds about 812 bytes. Raise mem_budget.",
        # A budget in bytes rather than in whole MiB.
        "Labeling holds about 812 bytes. Raise mem_budget to at least 812 bytes.",
    ],
)
def test_named_budget_refuses_a_refusal_that_names_no_budget_it_holds(
    message: str,
) -> None:
    with pytest.raises(AssertionError):
        named_budget(MemoryError(message))


def test_named_budget_refuses_a_message_that_names_another_budget_than_its_field() -> (
    None
):
    from cytearc.utils.memory_budget import MemoryBudgetError

    class Misnamed(MemoryBudgetError):
        def __str__(self) -> str:
            return "One row does not fit. Raise mem_budget to at least 9 MiB."

    with pytest.raises(AssertionError):
        named_budget(
            Misnamed("One row", needed_bytes=3_000_000, budget_bytes=10, item="a row")
        )


def test_a_climb_refuses_a_step_that_refuses_the_budget_that_it_named() -> None:
    from cytearc.utils.memory_budget import MemoryBudgetError

    def refusing(item: str, needed: int, budget: int) -> MemoryBudgetError:
        return MemoryBudgetError(
            "It does not fit.", needed_bytes=needed, budget_bytes=budget, item=item
        )

    # Two steps that need 1,000 and 2,000 bytes whatever the budget refuse
    # once each: the climb goes on at a quarter more than each need, 1,250
    # and 2,500 bytes, unrounded.
    budgets: list[int] = []

    def two_steps(budget: int) -> int:
        budgets.append(budget)
        if budget < 1_000:
            raise refusing("the first step", 1_000, budget)
        if budget < 2_000:
            raise refusing("the second step", 2_000, budget)
        return budget

    refusals, ran = climb_refusals(two_steps)
    assert [error.item for error in refusals] == ["the first step", "the second step"]
    assert budgets == [1, 1_250, 2_500]
    assert ran == 2_500
    # To each need instead, the climb runs at 2,000 bytes.
    budgets.clear()
    _refusals, ran = climb_refusals(two_steps, at_needs=True)
    assert budgets == [1, 1_000, 2_000] and ran == 2_000

    # A step whose need grows faster than a quarter of the budget refuses
    # the budget that its refusal gave, which the climb refuses.
    def growing(budget: int) -> int:
        if budget < 3 * 1024**2:
            raise refusing("the step", 2 * budget + 1, budget)
        return budget

    with pytest.raises(AssertionError, match="the step is refused again"):
        climb_refusals(growing)


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
