"""The helper that reads the budget a memory refusal names."""

import pytest

from tests.memory_helpers import named_budget


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
