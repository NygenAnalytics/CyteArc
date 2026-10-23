import os
import sys
from dataclasses import dataclass
from typing import Any

from loguru import logger as _loguru_logger

_CYTEARC_RECORD = "cytearc"

logger = _loguru_logger.bind(**{_CYTEARC_RECORD: True})
"""Loguru's logger, marking each record that CyteArc logs through it."""


@dataclass(slots=True)
class _OutputConfig:
    level: str = "INFO"
    progress: bool = True
    timestamps: bool = False
    filepath: str | None = None


_config = _OutputConfig()
_handler_id: int | None = None


def _flushing_stdout_sink(message: Any) -> None:
    sys.stdout.write(str(message))
    sys.stdout.flush()


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


def configure_output(
    *,
    level: str | None = None,
    progress: bool | None = None,
    timestamps: bool | None = None,
) -> None:
    """Configure CyteArc's logging and progress output.

    Unspecified settings retain their current values. The level and
    timestamps apply to CyteArc's own log sink, which shows only the records
    logged through ``cytearc.logger``, so other code's loguru output keeps its
    own handlers and levels.
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
    if update_handler:
        _install_cytearc_handler()


def set_verbosity(level: str | None = None, filepath: str | None = None) -> None:
    """Set CyteArc's logging level and optional output file.

    Like :func:`configure_output`, this changes only CyteArc's own sink.
    """
    _config.level = _resolve_level(level)
    _config.filepath = filepath
    _install_cytearc_handler()
