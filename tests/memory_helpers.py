"""Read the budget that a memory refusal names, and give a domain function a ledger of a budget."""

import re
from typing import Any


def named_budget(refusal: BaseException) -> int:
    """Return the memory budget that a refusal names in its closing ``raise mem_budget to at least N bytes``.

    A refusal that says the call holds about M bytes must name M. A refusal
    that closes with ``Raise mem_budget.`` names no budget, as Mixscape's
    refusal of a classification batch does, and asks for the M bytes the call
    holds.
    """
    message = str(refusal)
    held = re.search(r"holds? about (\d+) bytes", message)
    if message.endswith(" Raise mem_budget."):
        assert held is not None, f"The refusal names no budget: {message}"
        return int(held.group(1))
    named = re.search(
        r"[Rr]aise mem_budget to at least (\d+) bytes(?:, or [^.]*)?\.$", message
    )
    assert named is not None, f"The refusal does not end naming a budget: {message}"
    assert held is None or held.group(1) == named.group(1), (
        f"The refusal names a budget other than what it holds: {message}"
    )
    return int(named.group(1))


def budgeted(budget: int | None, doing: str = "Calling clones") -> Any:
    """A memory ledger of ``budget`` bytes that holds nothing yet, or None for no budget.

    Domain functions that a test calls alone charge it as they would an
    operation's ledger that held nothing beside them.
    """
    from cytearc.utils.memory_budget import MemoryLedger

    return None if budget is None else MemoryLedger(budget, doing)
