"""Feature streams draw their progress bar once they are iterated.

The consumer advances a stream's bar as it uses each item, so the progress
that a pipeline stage observes comes from the stage's own thread. Closing a
stream early releases the item it last handed off before the producer stops.
"""

import threading
import weakref
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import tqdm.auto

from cytearc.composite import create as composite_create
from cytearc.composite import create_composite
from cytearc.storage.count_matrix import CountMatrixPolicy
from cytearc.storage.feature_stream import (
    advance_on_consumption,
    map_feature_cell_bands,
    map_feature_read_groups,
)
from cytearc.storage.virtual_arrays import virtual_array_reader
from cytearc.utils.progress import observe_progress
from tests.test_composite import _write_source


@pytest.fixture(scope="module")
def stores(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """A small store, and a composite whose counts a reader computes from two."""
    directory = tmp_path_factory.mktemp("stream_progress")
    rng = np.random.default_rng(5)
    sources = {}
    for name in ("a", "b"):
        values = rng.integers(0, 20, size=(24, 32)).astype(np.uint16)
        sources[name] = _write_source(
            directory / f"{name}.zarr",
            values,
            [f"g{feature:02d}" for feature in range(32)],
            policy=CountMatrixPolicy(1024, 128),
        )
    with pytest.MonkeyPatch.context() as patch:
        # Small chunks give the composite's streams several units.
        patch.setattr(
            composite_create, "DEFAULT_COUNT_MATRIX_POLICY", CountMatrixPolicy(512, 128)
        )
        joint = create_composite(
            sources,
            at=directory / "joint.zarr",
            features="intersection",
            mem_budget="64M",
            nthreads=2,
        )
    return SimpleNamespace(store=sources["a"], composite=joint)


@pytest.fixture
def drawn_bars(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    """Record the description of every progress bar that CyteArc creates.

    A bar counts whether it is displayed or not: a hidden bar still exists,
    and an observed one still reports to its observer.
    """
    drawn: list[str | None] = []
    base = tqdm.auto.tqdm

    class RecordedBar(base):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            drawn.append(kwargs.get("desc"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(tqdm.auto, "tqdm", RecordedBar)
    return drawn


_STREAMS = {
    "read_groups": map_feature_read_groups,
    "cell_bands": map_feature_cell_bands,
}


@pytest.mark.parametrize("stream", list(_STREAMS))
@pytest.mark.parametrize("source", ["store", "composite"])
def test_a_stream_draws_its_bar_only_once_it_is_iterated(
    stores: SimpleNamespace,
    drawn_bars: list[str | None],
    source: str,
    stream: str,
) -> None:
    mapping = _STREAMS[stream]
    counts_t = getattr(stores, source).RNA.matrixGroup["countsT"]
    # A composite's counts come from its reader, which streams them itself.
    assert (virtual_array_reader(counts_t) is not None) == (source == "composite")

    # A stream that is made and closed but never iterated draws no bar.
    unused = mapping(counts_t, lambda unit: unit, progress="Never iterated")
    unused.close()
    assert drawn_bars == []

    updates: list[tuple[str | None, int, int | None, int]] = []

    def observe(
        _bar: int, description: str | None, done: int, total: int | None
    ) -> None:
        updates.append((description, done, total, threading.get_ident()))

    with observe_progress(observe):
        units = sum(mapping(counts_t, lambda _unit: 1, progress="Iterated"))

    assert drawn_bars == ["Iterated"]
    # The consumer advanced the bar once for each unit it used.
    assert units > 1
    assert [update[:3] for update in updates] == [
        ("Iterated", done, units) for done in range(1, units + 1)
    ]
    assert {update[3] for update in updates} == {threading.get_ident()}


class _Unit:
    """A handed-off item that a weak reference can watch."""


class _Bar:
    def __init__(self) -> None:
        self.updates = 0
        self.closed = False

    def update(self, units: int = 1) -> None:
        self.updates += units

    def close(self) -> None:
        self.closed = True


def _handoff(released: list[bool]) -> Iterator[_Unit]:
    """Hand off three units, as a bounded handoff does.

    A unit is released when the consumer moves past it or closes the stream,
    and the producer may then reuse its buffers, so ``released`` notes
    whether anything still held each unit at that moment.
    """
    for _index in range(3):
        unit = _Unit()
        watched = weakref.ref(unit)
        try:
            yield unit
        finally:
            del unit
            released.append(watched() is not None)


def _bar_factory(bars: list[_Bar]) -> Callable[[], _Bar]:
    def make_bar() -> _Bar:
        bars.append(_Bar())
        return bars[-1]

    return make_bar


@pytest.mark.parametrize("with_bar", [True, False], ids=["bar", "no_bar"])
def test_closing_a_stream_early_releases_the_item_it_handed_off(
    with_bar: bool,
) -> None:
    released: list[bool] = []
    bars: list[_Bar] = []
    stream = advance_on_consumption(
        _handoff(released), _bar_factory(bars) if with_bar else None
    )

    first = next(stream)
    del first
    stream.close()

    # Nothing held the unit when the producer was told it was released.
    assert released == [False]
    if with_bar:
        (bar,) = bars
        assert (bar.updates, bar.closed) == (0, True)


def test_a_consumed_stream_advances_its_bar_after_each_item() -> None:
    released: list[bool] = []
    bars: list[_Bar] = []
    stream = advance_on_consumption(_handoff(released), _bar_factory(bars))
    assert bars == []

    seen: list[tuple[int, list[bool]]] = []
    for unit in stream:
        del unit
        (bar,) = bars
        seen.append((bar.updates, list(released)))

    # Each unit is released, and the bar advanced, once the consumer asks for
    # the next one.
    assert seen == [(0, []), (1, [False]), (2, [False, False])]
    (bar,) = bars
    assert (bar.updates, bar.closed, released) == (3, True, [False, False, False])
