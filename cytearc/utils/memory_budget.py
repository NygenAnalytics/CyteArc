"""Charge an estimate of what an operation holds against its memory budget before it allocates.

An operation creates one :class:`MemoryLedger` with its budget and charges it
before each step allocates: an estimate of what the step keeps, and of the
transient peak of what it builds and lets go. The estimates count the arrays
that grow with the data, such as blocks, coordinates, graphs and per-cell IDs
and labels, and ``READ_ALLOWANCE_BYTES`` for each read of stored arrays in
flight. They leave out the interpreter, the other objects and buffers of the
libraries, and other allocations that do not grow with the data, so a step can
hold a little more than its charge. The ledger refuses a charge that would pass
the budget with a :class:`MemoryBudgetError` that names the step, the bytes
that it needs and the budget, so a call is refused before it reads or writes
what would not fit, and a budget of the bytes it names runs that step.
Planners that size work outside any ledger raise the same error with the same
fields in their own words, and :func:`memory_refusal` words the refusal of a
step that has none.

The shared costs below estimate the structures that many steps build, at a
simple cost for each item: Python strings, fixed-width text, object arrays,
sorts and hash tables.
"""

import sys
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = [
    "READ_ALLOWANCE_BYTES",
    "MemoryBudgetError",
    "MemoryLedger",
    "fixed_text_bytes",
    "hash_table_bytes",
    "held_array_bytes",
    "least_admitted_budget",
    "memory_refusal",
    "object_array_bytes",
    "python_text_bytes",
    "refusal_words",
    "sort_bytes",
]


#: What one read of stored arrays in flight holds beside the arrays that grow
#: with the data: the objects of Zarr's read and of the chunks it decodes,
#: NumPy's buffers of a read that takes values by index arrays, and the
#: buffer of a file read in part, as a read of part of a shard is, which
#: Python sizes by the block of the file system. On a local file system, a
#: read held at most 50 KB of these on Python 3.13 and 97 KB on Python 3.14,
#: whose buffer of a file read in part is larger; a file system with larger
#: blocks gives each read a larger buffer.
READ_ALLOWANCE_BYTES = 128 * 1024


class MemoryBudgetError(MemoryError):
    """A step of a call would hold more memory than the call's budget, by the estimate that the step charges.

    The estimate counts the arrays that grow with the data and leaves out the
    memory of the interpreter and its libraries, so the budget plans the work
    rather than capping the process.

    Attributes:
        needed_bytes: The estimated bytes that the call holds at the step
            that was refused: what it held before and the step's own charge.
            A budget of at least this many bytes runs that step, and a later
            step can need more. A count write whose producer hands it a batch
            beyond the producer's reservation names instead what the write
            would reserve with the batch.
        budget_bytes: The memory budget that the step would pass.
        item: What the step's charge was for, such as ``"the cells' IDs"``.
    """

    def __init__(
        self, message: str, *, needed_bytes: int, budget_bytes: int, item: str
    ) -> None:
        super().__init__(message)
        self.needed_bytes = int(needed_bytes)
        self.budget_bytes = int(budget_bytes)
        self.item = str(item)

    def __reduce__(self) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
        return (
            _restore_memory_budget_error,
            (
                type(self),
                str(self.args[0]),
                self.needed_bytes,
                self.budget_bytes,
                self.item,
            ),
            self.__dict__,
        )


def _restore_memory_budget_error(
    error_type: type[MemoryBudgetError],
    message: str,
    needed_bytes: int,
    budget_bytes: int,
    item: str,
) -> MemoryBudgetError:
    """Rebuild a pickled MemoryBudgetError, whose fields are keywords."""
    error = MemoryBudgetError.__new__(error_type)
    MemoryBudgetError.__init__(
        error,
        message,
        needed_bytes=needed_bytes,
        budget_bytes=budget_bytes,
        item=item,
    )
    return error


def memory_refusal(
    doing: str,
    item: str,
    needed_bytes: int,
    budget_bytes: int,
    *,
    remedy: str | None = None,
) -> MemoryBudgetError:
    """Return the refusal of a step that needs more bytes than the memory budget.

    The message reads "{doing} holds about {needed_bytes} bytes for {item},
    more than the memory budget of {budget_bytes} bytes. Raise mem_budget to
    at least {needed_bytes} bytes.", with ", or {remedy}" before the closing
    period when a step can also be made smaller.

    Args:
        doing: What the call does, such as ``"Importing contigs for 8 cells"``.
        item: What the refused charge is for.
        needed_bytes: The bytes that the call needs at that step.
        budget_bytes: The memory budget.
        remedy: Another way to fit the budget, such as ``"pass a smaller
            cell_selection"``, without a closing period.

    Raises:
        TypeError: If a number of bytes is not an integer.
        ValueError: If a number of bytes is negative, or ``needed_bytes``
            does not pass ``budget_bytes``.
    """
    for value, name in ((needed_bytes, "needed_bytes"), (budget_bytes, "budget_bytes")):
        if isinstance(value, bool) or not isinstance(value, int | np.integer):
            raise TypeError(f"{name} must be a number of bytes, not {value!r}")
        if value < 0:
            raise ValueError(f"{name} must be at least 0, not {value}")
    if needed_bytes <= budget_bytes:
        raise ValueError(
            f"A refusal needs more than the budget of {budget_bytes} bytes, not {needed_bytes}"
        )
    needed_bytes, budget_bytes = int(needed_bytes), int(budget_bytes)
    closing = f"Raise mem_budget to at least {needed_bytes} bytes"
    if remedy:
        closing += f", or {remedy}"
    return MemoryBudgetError(
        f"{doing} holds about {needed_bytes} bytes for {item}, more than the "
        f"memory budget of {budget_bytes} bytes. {closing}.",
        needed_bytes=needed_bytes,
        budget_bytes=budget_bytes,
        item=item,
    )


def refusal_words(error: MemoryBudgetError) -> str:
    """Return the message of ``error`` without the sentence that closes it with its bytes.

    A step that refuses inside a larger one, such as a conversion beside the
    data it converts, closes its message with "It needs at least
    {needed_bytes} bytes." in its own terms. The larger step keeps these
    words and states the bytes of its whole budget after them.
    """
    return str(error).removesuffix(f" It needs at least {error.needed_bytes} bytes.")


class MemoryLedger:
    """What one call holds, charged against its memory budget before each step.

    A step charges what it will keep with :meth:`hold`, which it lets go with
    :meth:`release`, and what it builds and lets go with :meth:`peak`. A
    budget of None never refuses.

    Attributes:
        budget: The memory budget in bytes, or None for none.
        doing: What the call does, as its refusals name it; a step can set
            its own.
        remedy: Another way to fit the budget that refusals name, or None;
            a charge can pass its own.
        codec_threads: The threads of Zarr's codec pool that the call's reads
            decode on: the store's ``nthreads``, whatever the machine, as
            ``storage.async_execution.zarr_decoding_threads`` gives them.
        charged: The bytes held by charges not released.
    """

    def __init__(
        self,
        budget: int | None,
        doing: str,
        *,
        remedy: str | None = None,
        codec_threads: int = 1,
    ) -> None:
        if budget is not None and (
            isinstance(budget, bool) or not isinstance(budget, int) or budget < 0
        ):
            raise ValueError(
                f"A memory budget must be a non-negative number of bytes or None, not {budget!r}"
            )
        if (
            isinstance(codec_threads, bool)
            or not isinstance(codec_threads, int)
            or codec_threads < 1
        ):
            raise ValueError(
                f"codec_threads must be a number of threads of at least 1, not {codec_threads!r}"
            )
        self.budget = budget
        self.doing = doing
        self.remedy = remedy
        self.codec_threads = codec_threads
        self.charged = 0

    @property
    def held_bytes(self) -> int:
        """The bytes charged and not released."""
        return self.charged

    @property
    def room(self) -> int | None:
        """The bytes that a step may still take: the budget less ``held_bytes``, None without a budget."""
        if self.budget is None:
            return None
        return max(0, self.budget - self.held_bytes)

    def refuse_beyond(self, item: str, needed: int, remedy: str | None = None) -> None:
        """Raise :class:`MemoryBudgetError` if holding ``needed`` bytes in all would pass the budget."""
        if self.budget is not None and needed > self.budget:
            raise memory_refusal(
                self.doing,
                item,
                needed,
                self.budget,
                remedy=self.remedy if remedy is None else remedy,
            )

    def hold(self, item: str, nbytes: int, *, remedy: str | None = None) -> int:
        """Charge ``nbytes`` that the call keeps for ``item``, refusing them beyond the budget.

        Returns:
            ``nbytes``, which :meth:`release` lets go.
        """
        nbytes = _bytes(nbytes)
        self.refuse_beyond(item, self.held_bytes + nbytes, remedy)
        self.charged += nbytes
        return nbytes

    def peak(self, item: str, nbytes: int, *, remedy: str | None = None) -> None:
        """Check that ``nbytes`` more, which a step builds and lets go, fit beside what is held."""
        nbytes = _bytes(nbytes)
        self.refuse_beyond(item, self.held_bytes + nbytes, remedy)

    def release(self, nbytes: int) -> None:
        """Let go of ``nbytes`` that :meth:`hold` charged, once they are freed."""
        nbytes = _bytes(nbytes)
        if nbytes > self.charged:
            raise ValueError(
                f"Releasing {nbytes} bytes, more than the {self.charged} bytes charged"
            )
        self.charged -= nbytes


def _bytes(nbytes: int) -> int:
    resolved = int(nbytes)
    if resolved < 0:
        raise ValueError(
            f"A charge must be a non-negative number of bytes, not {nbytes}"
        )
    return resolved


#: The bytes of a Python string beside its characters, between the 41 of
#: ASCII text and the 60 of text beyond U+FFFF.
_STRING_BYTES = 49


def python_text_bytes(n_values: int, characters: int, *, widest: int = 0x7F) -> int:
    """Return the bytes of ``n_values`` Python strings that hold ``characters`` characters in all.

    Each string takes about 49 bytes beside its characters, which take a
    byte each, two when ``widest``, the widest code point of any string, is
    beyond U+00FF, and four when it is beyond U+FFFF.
    """
    if widest > 0x10FFFF:
        raise ValueError(f"A code point cannot exceed U+10FFFF, not {widest:#x}")
    width = 1 if widest <= 0xFF else 2 if widest <= 0xFFFF else 4
    return int(n_values) * _STRING_BYTES + width * int(characters)


def fixed_text_bytes(n_values: int, width: int) -> int:
    """Return the bytes of ``n_values`` values of NumPy or Zarr text ``width`` characters wide.

    Fixed-width text stores every value at the width of the widest, four
    bytes per character.
    """
    return 4 * int(width) * int(n_values)


def object_array_bytes(n_values: int) -> int:
    """Return the bytes of an object array's references to ``n_values`` values, beside the values."""
    return 8 * int(n_values)


# The most times that a step's named need is raised before its refusal is
# passed on: a need that grows with its budget passes within a few.
_ADMISSION_CLIMBS = 64


def least_admitted_budget(
    admit: Callable[[int], object], refused: MemoryBudgetError
) -> int:
    """Return the least budget at which ``admit`` passes a step that it ``refused``.

    ``admit(budget)`` checks the step at ``budget``, planning it but running
    nothing, and raises :class:`MemoryBudgetError` for a budget that it
    refuses. A step whose need grows with its budget, as shares of the
    budget do, can refuse the budget that its refusal names: the budget is
    raised to each larger need that a refusal names until the step passes.
    The least budget that passes, above the refused one, is then found by
    bisection.

    A step that refuses each need that it names, 64 times over, keeps the
    need that its refusal named, as one that passes at no budget would.
    """
    low, high = int(refused.budget_bytes), int(refused.needed_bytes)
    for _ in range(_ADMISSION_CLIMBS):
        try:
            admit(high)
        except MemoryBudgetError as error:
            low, high = high, max(int(error.needed_bytes), high + 1)
        else:
            break
    else:
        return int(refused.needed_bytes)
    while high - low > 1:
        middle = (low + high) // 2
        try:
            admit(middle)
        except MemoryBudgetError:
            low = middle
        else:
            high = middle
    return high


def held_array_bytes(values: np.ndarray) -> int:
    """Return the bytes of an array, with those of the objects that an object array holds.

    Each object counts once, as ``sys.getsizeof`` measures it, however many
    elements refer to it, as the rows of a column share the strings that
    pandas reads once. A reader measures the text it keeps once, rather
    than at every step that charges it.
    """
    held = int(values.nbytes)
    if values.dtype.kind == "O":
        flat = values.ravel()
        _ids, first = np.unique(
            np.fromiter(map(id, flat), dtype=np.intp, count=flat.size),
            return_index=True,
        )
        held += sum(map(sys.getsizeof, flat[first]))
    return held


def sort_bytes(
    n_values: int, itemsize: int, *, stable: bool = False, order: bool = False
) -> int:
    """Return the bytes that sorting ``n_values`` values of ``itemsize`` bytes holds, beside them.

    ``np.sort`` returns a copy, and with ``order`` ``np.argsort`` returns the
    order as int64; a stable sort holds as much again while it merges.
    """
    n_values, itemsize = int(n_values), int(itemsize)
    held = 8 * n_values if order else n_values * itemsize
    return 2 * held if stable else held


#: The bytes of pandas' hash table for each value that it hashes: 16 bytes a
#: bucket, with 1.3 to 2.6 buckets a value, 30 bytes on average.
_HASH_TABLE_BYTES = 32


def hash_table_bytes(n_values: int) -> int:
    """Return the bytes of the hash table that pandas builds to hash ``n_values`` values.

    ``pd.factorize``, ``pd.unique`` and ``Index.get_indexer`` hash their
    values into a table that pandas sizes for every value given, whether or
    not the values repeat, about 32 bytes a value. Beside the table, pandas
    returns codes and the distinct values, and an object array hashes the
    values' strings, which fixed-width text becomes first.
    """
    return _HASH_TABLE_BYTES * max(0, int(n_values))
