import signal
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import FrameType
from typing import Any, Iterator


@dataclass(frozen=True, slots=True)
class ShutdownRequest:
    requested_at_ns: int
    reason: str
    signal_number: int | None = None
    signal_name: str | None = None
    # True when a CancellationToken asked for the shutdown.
    cancelled: bool = False


class ShutdownRequested(BaseException):
    """Raised at a safe checkpoint after cooperative shutdown was requested."""

    def __init__(self, request: ShutdownRequest) -> None:
        self.request = request
        super().__init__(request.reason)

    def __reduce__(self) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
        return (type(self), (self.request,), self.__dict__)


@dataclass(frozen=True, slots=True)
class _Claim:
    """The first shutdown request with the signal behavior it interrupted."""

    request: ShutdownRequest
    previous_handler: Any
    frame: FrameType | None


# The keys of a token's claims: the first request of any kind, which a
# checkpoint raises and an interrupted run records, and the first signal,
# which is propagated after cleanup even when a cancellation came first.
_FIRST_REQUEST = "first"
_FIRST_SIGNAL = "signal"


class ShutdownToken:
    """Thread-safe, runtime-neutral cooperative shutdown state."""

    __slots__ = ("_claims",)

    def __init__(self) -> None:
        # Holds the first request's claim and the first signal's, which are
        # never replaced.
        self._claims: dict[str, _Claim] = {}

    @property
    def requested(self) -> bool:
        return _FIRST_REQUEST in self._claims

    @property
    def request_record(self) -> ShutdownRequest | None:
        claim = self._claims.get(_FIRST_REQUEST)
        return None if claim is None else claim.request

    @property
    def signal_record(self) -> ShutdownRequest | None:
        """The first signal received, which may follow a cancellation."""
        claim = self._claims.get(_FIRST_SIGNAL)
        return None if claim is None else claim.request

    def request(
        self,
        *,
        reason: str = "shutdown requested",
        signal_number: int | None = None,
        previous_handler: Any = None,
        frame: FrameType | None = None,
        cancelled: bool = False,
    ) -> bool:
        """Record a request and return whether it is the first of its kind.

        A signal is the first of its kind when no signal came before it, even
        after a cancellation, so that only a second signal escalates. Any
        other request is the first when no request came before it.
        """

        if not isinstance(reason, str) or not reason:
            raise TypeError("reason must be a non-empty string")
        if cancelled and signal_number is not None:
            raise ValueError("a cancellation is not a signal")
        signal_name: str | None = None
        if signal_number is not None:
            try:
                signal_name = signal.Signals(signal_number).name
            except ValueError:
                signal_name = f"SIGNAL_{signal_number}"
        claim = _Claim(
            request=ShutdownRequest(
                requested_at_ns=time.time_ns(),
                reason=reason,
                signal_number=signal_number,
                signal_name=signal_name,
                cancelled=cancelled,
            ),
            previous_handler=previous_handler,
            frame=frame,
        )
        first = self._claims.setdefault(_FIRST_REQUEST, claim) is claim
        if signal_number is None:
            return first
        return self._claims.setdefault(_FIRST_SIGNAL, claim) is claim

    def checkpoint(self) -> None:
        claim = self._claims.get(_FIRST_REQUEST)
        if claim is not None:
            raise ShutdownRequested(claim.request)

    def propagate(self) -> None:
        """Continue with the prior signal behavior after durable cleanup.

        A received signal goes to the handler it interrupted, even when a
        cancellation stopped the work first. Without a signal, a request that
        no cancellation made raises ``ShutdownRequested``, and a cancellation
        returns: its caller reports it.
        """

        claim = self._claims.get(_FIRST_SIGNAL)
        if claim is None:
            first = self._claims.get(_FIRST_REQUEST)
            if first is not None and not first.request.cancelled:
                raise ShutdownRequested(first.request)
            return
        request = claim.request
        assert request.signal_number is not None
        previous = claim.previous_handler
        if callable(previous):
            previous(request.signal_number, claim.frame)
            raise ShutdownRequested(request)
        if previous == signal.SIG_IGN:
            raise ShutdownRequested(request)
        signal.raise_signal(request.signal_number)
        raise ShutdownRequested(request)


class CancellationToken:
    """Ask a running pipeline to stop at its next safe checkpoint.

    Pass the token as a recipe's ``cancel`` argument and call :meth:`cancel`
    from any thread, such as a UI's stop button. Long operations check for
    cancellation between bounded blocks of work, so the run stops soon after
    and raises :class:`~cytearc.PipelineCancelled`. The run is recorded as
    interrupted. A stage whose work had not finished is recorded as
    interrupted too, and one whose work had finished as completed, so the run
    stops between stages. Completed stages keep their artifacts, which a
    later run reuses. A cancellation that arrives after the run completed
    leaves it completed, and the recipe returns it.

    A token stays cancelled: a run given a cancelled token raises before it
    writes anything. One token may cancel several runs at once. A signal that
    stops a run never cancels the token, and a signal that arrives after a
    cancellation still takes its usual effect once the run is recorded.
    """

    __slots__ = ("_lock", "_reason", "_tokens")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reason: str | None = None
        # The shutdown tokens of the runs that listen to this cancellation.
        self._tokens: set[ShutdownToken] = set()

    def cancel(self, reason: str = "cancelled") -> bool:
        """Cancel, and return whether this was the token's first cancellation."""
        if not isinstance(reason, str) or not reason:
            raise TypeError("reason must be a non-empty string")
        with self._lock:
            first = self._reason is None
            if first:
                self._reason = reason
            resolved = self._reason
            tokens = tuple(self._tokens)
        assert resolved is not None
        for token in tokens:
            token.request(reason=resolved, cancelled=True)
        return first

    @property
    def cancelled(self) -> bool:
        """Whether :meth:`cancel` has been called."""
        return self._reason is not None

    @property
    def reason(self) -> str | None:
        """The reason given to the first :meth:`cancel`, or None."""
        return self._reason

    @contextmanager
    def _linked(self, token: ShutdownToken) -> Iterator[None]:
        """Request shutdown on ``token`` when this token is cancelled."""
        with self._lock:
            self._tokens.add(token)
            reason = self._reason
        if reason is not None:
            token.request(reason=reason, cancelled=True)
        try:
            yield
        finally:
            with self._lock:
                self._tokens.discard(token)

    def __repr__(self) -> str:
        state = f"cancelled, reason={self.reason!r}" if self.cancelled else "active"
        return f"CancellationToken({state})"


@contextmanager
def cancellation_scope(
    cancel: CancellationToken | None,
    token: ShutdownToken,
) -> Iterator[None]:
    """Turn a cancellation of ``cancel`` into a shutdown request on ``token``."""
    if cancel is None:
        yield
        return
    if not isinstance(cancel, CancellationToken):
        raise TypeError("cancel must be a CancellationToken or None")
    with cancel._linked(token):
        yield


_CURRENT_TOKEN: ContextVar[ShutdownToken | None] = ContextVar(
    "cytearc_shutdown_token",
    default=None,
)


@contextmanager
def shutdown_scope(token: ShutdownToken) -> Iterator[ShutdownToken]:
    if not isinstance(token, ShutdownToken):
        raise TypeError("token must be a ShutdownToken")
    context_token = _CURRENT_TOKEN.set(token)
    try:
        yield token
    finally:
        _CURRENT_TOKEN.reset(context_token)


def current_shutdown_token() -> ShutdownToken | None:
    return _CURRENT_TOKEN.get()


def shutdown_checkpoint() -> None:
    token = current_shutdown_token()
    if token is not None:
        token.checkpoint()


class TemporarySignalGuard:
    """Temporarily translate catchable termination signals into token requests."""

    __slots__ = ("_installed", "_token", "available", "unavailable_reason")

    def __init__(self, token: ShutdownToken) -> None:
        if not isinstance(token, ShutdownToken):
            raise TypeError("token must be a ShutdownToken")
        self._token = token
        self._installed: dict[int, Any] = {}
        self.available = False
        self.unavailable_reason: str | None = None

    def __enter__(self) -> "TemporarySignalGuard":
        if threading.current_thread() is not threading.main_thread():
            self.unavailable_reason = "signal handlers require the main thread"
            return self
        candidates = tuple(
            candidate
            for name in ("SIGTERM", "SIGINT", "SIGHUP")
            if (candidate := getattr(signal, name, None)) is not None
        )
        try:
            for signum in candidates:
                previous = signal.getsignal(signum)
                # A handler installed outside Python reads as None and could
                # not be restored, so that signal keeps its current handler.
                if previous is None or previous == signal.SIG_IGN:
                    continue

                def handler(
                    received: int,
                    frame: FrameType | None,
                    *,
                    prior: Any = previous,
                ) -> None:
                    first = self._token.request(
                        reason=f"received {signal.Signals(received).name}",
                        signal_number=received,
                        previous_handler=prior,
                        frame=frame,
                    )
                    if not first:
                        self._escalate(received, frame, prior)

                # Record the prior handler first, so that a handler installed
                # just before an error is always put back.
                self._installed[int(signum)] = previous
                signal.signal(signum, handler)
        except BaseException:
            # A guard that fails to enter is never exited, so it puts back the
            # handlers it had already replaced before the error propagates.
            self._restore()
            raise
        self.available = bool(self._installed)
        if not self.available:
            self.unavailable_reason = "no catchable termination signals are available"
        return self

    @staticmethod
    def _escalate(signum: int, frame: FrameType | None, prior: Any) -> None:
        signal.signal(signum, prior)
        if callable(prior):
            prior(signum, frame)
            return
        if prior == signal.SIG_IGN:
            return
        signal.raise_signal(signum)

    def _restore(self) -> None:
        for signum, previous in self._installed.items():
            signal.signal(signum, previous)
        self._installed.clear()

    def __exit__(self, *_exc: object) -> None:
        self._restore()
