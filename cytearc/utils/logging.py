import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from loguru import logger as _loguru_logger

_CYTEARC_RECORD = "cytearc"

logger = _loguru_logger.bind(**{_CYTEARC_RECORD: True})
"""Loguru's logger, marking each record that CyteArc logs through it."""


STDLIB_LOGGER_NAME = "cytearc"
"""The standard-library logger that receives CyteArc's records when bridged."""

# The record extras that name the pipeline run and stage a record came from.
RUN_ID_EXTRA = "cytearc_run_id"
STAGE_EXTRA = "cytearc_stage"
# A record with this extra set, such as a host callback's failure, is shown
# but never collected into a stage's warnings.
UNCOLLECTED_EXTRA = "cytearc_uncollected"


@dataclass(slots=True)
class _OutputConfig:
    level: str = "INFO"
    progress: bool = True
    timestamps: bool = False
    filepath: str | None = None
    stdlib: bool = False


_config = _OutputConfig()
_handler_id: int | None = None


def _flushing_stdout_sink(message: Any) -> None:
    sys.stdout.write(str(message))
    sys.stdout.flush()


def _stdlib_sink(message: Any) -> None:
    """Forward one CyteArc record to the standard-library ``cytearc`` logger.

    The forwarded record keeps the level number and name, such as
    ``SUCCESS``, and the module, file, line, and function that logged it. A
    record logged during a pipeline stage carries the run and stage as the
    ``cytearc_run_id`` and ``cytearc_stage`` attributes, so a host's handlers
    can route it.
    """
    import logging

    record = message.record
    target = logging.getLogger(STDLIB_LOGGER_NAME)
    level = record["level"].no
    if not target.isEnabledFor(level):
        return
    exception = record["exception"]
    forwarded = target.makeRecord(
        STDLIB_LOGGER_NAME,
        level,
        record["file"].path,
        record["line"],
        record["message"],
        (),
        (
            None
            if exception is None
            else (exception.type, exception.value, exception.traceback)
        ),
        func=record["function"],
        extra={
            RUN_ID_EXTRA: record["extra"].get(RUN_ID_EXTRA),
            STAGE_EXTRA: record["extra"].get(STAGE_EXTRA),
        },
    )
    forwarded.levelname = record["level"].name
    target.handle(forwarded)


def _log_format() -> str:
    if _config.timestamps:
        return "{time:YYYY-MM-DDTHH:mm:ss.SSSZ} | <level>{level}</level> | {message}"
    return "<level>{level}</level>: {message}"


def _cytearc_record(record: Any) -> bool:
    """Return whether CyteArc logged ``record``, the only records its sink shows."""
    return record["extra"].get(_CYTEARC_RECORD) is True


_LOGURU_FILTER = os.environ.get("LOGURU_FILTER")


def _other_record(record: Any) -> bool:
    """Return whether loguru's default handler shows ``record``.

    That handler shows every record that CyteArc did not log, as filtered by
    ``LOGURU_FILTER``, a module name whose records and whose submodules'
    records it keeps.
    """
    if _cytearc_record(record):
        return False
    if not _LOGURU_FILTER:
        return True
    name = record["name"]
    return name is not None and f"{name}.".startswith(f"{_LOGURU_FILTER}.")


def _remove_cytearc_handler() -> None:
    global _handler_id
    if _handler_id is None:
        return
    try:
        logger.remove(_handler_id)
    except ValueError:
        pass
    _handler_id = None


def _install_cytearc_handler() -> None:
    global _handler_id
    _remove_cytearc_handler()
    if _config.stdlib:
        _handler_id = logger.add(
            _stdlib_sink,
            colorize=False,
            format="{message}",
            level=_config.level,
            filter=_cytearc_record,
        )
        return
    if _config.filepath is None:
        _handler_id = logger.add(
            _flushing_stdout_sink,
            colorize=None,
            format=_log_format(),
            level=_config.level,
            filter=_cytearc_record,
        )
        return
    _handler_id = logger.add(
        _config.filepath,
        colorize=False,
        format=_log_format(),
        level=_config.level,
        filter=_cytearc_record,
    )


def _leave_cytearc_records_out_of_loguru_default() -> None:
    """Keep loguru's default handler for other code, without CyteArc's records.

    Loguru prints every record on stderr through its default handler 0 until
    an application removes it. CyteArc shows its own records on stdout, so
    that handler would print them a second time. When the handler is still
    installed, it is replaced by one with the same sink and settings that
    leaves out CyteArc's records; other code keeps its stderr output, level,
    and format. An application that removed it manages its own handlers, and
    they are not touched.
    """
    try:
        logger.remove(0)
    except ValueError:
        return
    logger.add(sys.stderr, filter=_other_record)


def _resolve_level(level: str | None) -> str:
    if level is None:
        raise ValueError("Please provide a value for level recognized by Loguru")
    try:
        return logger.level(level).name
    except ValueError as error:
        raise ValueError(
            f"Please provide a value for level recognized by Loguru: {level!r}"
        ) from error


_leave_cytearc_records_out_of_loguru_default()
_install_cytearc_handler()


def progress_enabled() -> bool:
    """Return whether CyteArc progress reporting is enabled."""
    return _config.progress


# The warnings that CyteArc logs or issues in the current context, while a
# caller collects them.
_WARNINGS: ContextVar[list[str] | None] = ContextVar(
    "cytearc_collected_warnings",
    default=None,
)


def note_warning(message: str) -> None:
    """Add ``message`` to the warnings that the current context collects."""
    collected = _WARNINGS.get()
    if collected is not None:
        collected.append(str(message))


@contextmanager
def collect_warnings() -> Iterator[list[str]]:
    """Collect the warnings that CyteArc logs or issues in this context.

    The list gains each WARNING or higher record that CyteArc logs, and each
    warning that it issues with :func:`cytearc.utils.warnings.warn`, from this
    thread or asyncio task, so concurrent work elsewhere is not collected.
    Display is unchanged.
    """
    collected: list[str] = []
    token = _WARNINGS.set(collected)

    def collecting(record: Any) -> bool:
        return (
            _cytearc_record(record)
            and not record["extra"].get(UNCOLLECTED_EXTRA)
            and _WARNINGS.get() is collected
        )

    handler = logger.add(
        lambda message: collected.append(message.record["message"]),
        level="WARNING",
        filter=collecting,
        format="{message}",
        catch=True,
    )
    try:
        yield collected
    finally:
        try:
            logger.remove(handler)
        except ValueError:
            pass
        _WARNINGS.reset(token)


def configure_output(
    *,
    level: str | None = None,
    progress: bool | None = None,
    timestamps: bool | None = None,
    stdlib: bool | None = None,
) -> None:
    """Configure CyteArc's logging and progress output.

    Unspecified settings retain their current values. The level and
    timestamps apply to CyteArc's own log sink, which shows only the records
    logged through ``cytearc.logger``, so other code's loguru output keeps its
    own handlers and levels.

    Args:
        level: The lowest level that CyteArc's sink shows, such as
            ``"WARNING"``.
        progress: Whether to display progress bars. Pipeline progress events
            are emitted either way.
        timestamps: Whether log lines start with a timestamp.
        stdlib: True sends CyteArc's records, at or above ``level``, to the
            standard-library logger named ``"cytearc"`` instead of stdout or
            the file that ``set_verbosity`` chose, for an application that
            configures ``logging``. A forwarded record keeps the details of
            the call that logged it: its ``module``, ``pathname``, ``lineno``,
            and ``funcName``, and loguru's level name, such as ``SUCCESS``, as
            its ``levelname``. A record logged during a pipeline stage carries
            the run and stage as its ``cytearc_run_id`` and ``cytearc_stage``
            attributes. False restores the stdout sink, or the file that
            ``set_verbosity`` chose.
    """
    update_handler = False
    if level is not None:
        _config.level = _resolve_level(level)
        update_handler = True
    if progress is not None:
        if not isinstance(progress, bool):
            raise TypeError("progress must be a bool")
        _config.progress = progress
    if timestamps is not None:
        if not isinstance(timestamps, bool):
            raise TypeError("timestamps must be a bool")
        _config.timestamps = timestamps
        update_handler = True
    if stdlib is not None:
        if not isinstance(stdlib, bool):
            raise TypeError("stdlib must be a bool")
        _config.stdlib = stdlib
        update_handler = True
    if update_handler:
        _install_cytearc_handler()


def set_verbosity(level: str | None = None, filepath: str | None = None) -> None:
    """Set CyteArc's logging level and optional output file.

    Like :func:`configure_output`, this changes only CyteArc's own sink. It
    selects stdout or the file, so it also ends a ``stdlib`` bridge.
    """
    _config.level = _resolve_level(level)
    _config.filepath = filepath
    _config.stdlib = False
    _install_cytearc_handler()
