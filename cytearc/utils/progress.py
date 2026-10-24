from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
import itertools
import sys
from typing import Any

from .logging import logger, progress_enabled

tqdm_params = {
    "bar_format": "{desc}: {percentage:3.0f}%| {bar} {n_fmt}/{total_fmt} [{elapsed}]",
    "dynamic_ncols": True,
    "colour": "#34abeb",
}

type ProgressObserver = Callable[[int, str | None, int, int | None], None]
"""Receives ``(bar, description, completed, total)`` after a bar advances.

``bar`` identifies one progress bar within the process, so bars that share a
description are told apart.
"""

_OBSERVER: ContextVar[ProgressObserver | None] = ContextVar(
    "cytearc_progress_observer",
    default=None,
)


@contextmanager
def observe_progress(observer: ProgressObserver) -> Iterator[None]:
    """Report every CyteArc progress bar of this context to ``observer``.

    The observer receives each bar's identifier, description, units
    completed, and total after every update, whether or not bars are
    displayed. An observer's exception is logged and never stops the work.
    """
    if not callable(observer):
        raise TypeError("observer must be callable")
    token = _OBSERVER.set(observer)
    try:
        yield
    finally:
        _OBSERVER.reset(token)


def progress_requested() -> bool:
    """Return whether progress is displayed or observed in this context."""
    return progress_enabled() or _OBSERVER.get() is not None


# The identifiers of observed bars, unique within the process.
_BAR_IDS = itertools.count()


class _ObservedBar:
    """A progress bar that also reports each update to an observer."""

    __slots__ = ("_bar", "_completed", "_description", "_id", "_observer", "_total")

    def __init__(
        self,
        bar: Any,
        observer: ProgressObserver,
        description: str | None,
        total: int | None,
    ) -> None:
        self._bar = bar
        self._id = next(_BAR_IDS)
        self._observer = observer
        self._description = description
        self._total = total
        self._completed: float = 0

    def _notify(self) -> None:
        try:
            self._observer(
                self._id, self._description, int(self._completed), self._total
            )
        except Exception:
            logger.exception("A progress observer failed")

    def update(self, n: float | None = 1) -> Any:
        result = self._bar.update(n)
        self._completed += 1 if n is None else n
        self._notify()
        return result

    def __iter__(self) -> Iterator[Any]:
        for item in self._bar:
            yield item
            self._completed += 1
            self._notify()

    def close(self) -> None:
        self._bar.close()

    def __enter__(self) -> "_ObservedBar":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bar, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _ObservedBar.__slots__:
            object.__setattr__(self, name, value)
            return
        # The bar's own attributes, such as its total, are set on the bar.
        setattr(self._bar, name, value)
        if name == "total":
            object.__setattr__(self, "_total", None if value is None else int(value))

    def __len__(self) -> int:
        return len(self._bar)

    def __bool__(self) -> bool:
        return bool(self._bar)


def tqdmbar(*args: Any, **kwargs: Any) -> Any:
    """Create a progress bar using CyteArc's display defaults.

    In a context that observes progress (see ``observe_progress``), the
    bar also reports its updates, even when bars are not displayed.
    """
    params = dict(tqdm_params)
    for name in kwargs:
        if name in params:
            del params[name]
    if "disable" in kwargs:
        kwargs["disable"] = not progress_enabled() or bool(kwargs["disable"])
    else:
        params["disable"] = not progress_enabled()
    from tqdm.auto import tqdm

    bar = tqdm(*args, **kwargs, **params)
    observer = _OBSERVER.get()
    if observer is None:
        return bar
    total = kwargs.get("total")
    if total is None and args:
        try:
            total = len(args[0])
        except TypeError:
            total = None
    return _ObservedBar(
        bar,
        observer,
        kwargs.get("desc"),
        None if total is None else int(total),
    )


def iter_progress[T](
    iterable: Iterable[T],
    *,
    desc: str | None = None,
    total: int | None = None,
    disable: bool | None = None,
) -> Iterator[T]:
    """Yield values while updating and reliably closing a progress bar."""
    iterator = iter(iterable)
    kwargs: dict[str, Any] = {"desc": desc, "total": total}
    if disable is not None:
        kwargs["disable"] = disable
    progress = None
    try:
        progress = tqdmbar(**kwargs)
        while True:
            try:
                item = next(iterator)
            except StopIteration:
                return
            try:
                yield item
            finally:
                del item
            progress.update()
    finally:
        original = sys.exception()
        errors: list[BaseException] = []
        for resource in (iterator, progress):
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as error:
                    errors.append(error)
        if errors:
            if original is not None and not isinstance(original, GeneratorExit):
                errors.insert(0, original)
            if len(errors) == 1:
                raise errors[0]
            raise BaseExceptionGroup("Progress stream cleanup failed", errors)
