import json
import re
import subprocess
import sys

import pytest

from cytearc.utils import configure_output, logger, set_verbosity
from cytearc.utils.logging import _config, progress_enabled


@pytest.fixture(autouse=True)
def _restore_output_configuration():
    yield
    set_verbosity("INFO")
    configure_output(progress=False, timestamps=False)


def test_import_defaults_are_notebook_first_in_isolated_process():
    code = """
import json
import cytearc
from cytearc.utils.logging import _config, progress_enabled

cytearc.logger.info("default output")
print(json.dumps({"level": _config.level, "progress": progress_enabled()}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.splitlines()

    assert lines[0] == "INFO: default output"
    assert json.loads(lines[-1]) == {"level": "INFO", "progress": True}
    assert "\x1b" not in result.stdout


def test_configure_output_updates_settings_independently(capsys):
    configure_output(level="WARNING", progress=True)
    configure_output(timestamps=True)
    logger.warning("timestamped")
    output = capsys.readouterr().out

    assert _config.level == "WARNING"
    assert progress_enabled()
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}"
        r"[+-]\d{2}:\d{2} \| WARNING \| timestamped\n",
        output,
    )


def test_set_verbosity_file_output_has_no_ansi_codes(tmp_path):
    path = tmp_path / "cytearc.log"
    set_verbosity("INFO", str(path))
    logger.warning("file output")
    output = path.read_text(encoding="utf-8")

    assert output == "WARNING: file output\n"
    assert "\x1b" not in output


def test_reconfiguration_preserves_caller_sink_and_configured_level():
    captured: list[str] = []
    sink = logger.add(
        lambda message: captured.append(message.record["message"]),
        level="TRACE",
    )
    try:
        configure_output(level="WARNING")
        logger.debug("caller only")
        logger.warning("visible warning")
    finally:
        logger.remove(sink)

    assert captured == ["caller only", "visible warning"]
    assert _config.level == "WARNING"


def test_reconfiguration_tolerates_a_stale_cytearc_handler(capsys):
    logger.remove()
    configure_output(level="DEBUG")
    logger.debug("restored")

    assert capsys.readouterr().out == "DEBUG: restored\n"


@pytest.mark.parametrize("setting", ["progress", "timestamps"])
def test_configure_output_requires_boolean_settings(setting):
    with pytest.raises(TypeError, match=setting):
        configure_output(**{setting: 1})


def test_loading_tolerates_an_already_removed_default_handler():
    import importlib.util

    import cytearc.utils.logging as cytearc_logging

    # An application, like this test session, may remove Loguru's default
    # handler 0 before CyteArc's logging module loads.
    with pytest.raises(ValueError):
        logger.remove(0)
    spec = importlib.util.spec_from_file_location(
        "cytearc_logging_copy", cytearc_logging.__file__
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._config == module._OutputConfig()
    assert module._handler_id is not None
    # Removing the copy's handler succeeds only because loading installed it.
    logger.remove(module._handler_id)


def _run_isolated(code: str, **environment: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter whose stderr holds only its own output."""
    import os

    # A coverage run measures subprocesses through these variables, and its
    # messages would join the child's stderr.
    inherited = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("COVERAGE_", "COV_CORE_"))
    }
    return subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env={**inherited, **environment},
    )


# Loguru colours its default handler on CI services and in PyCharm even when
# stderr is a pipe, so other code's records are compared without the colours.
_ANSI_COLOUR = re.compile(r"\x1b\[[0-9;]*m")


def _stderr_lines(result: subprocess.CompletedProcess) -> list[str]:
    """Return the lines of ``result``'s stderr without terminal colours."""
    return _ANSI_COLOUR.sub("", result.stderr).splitlines()


def test_host_loguru_output_keeps_its_default_handler_and_level():
    code = """
from loguru import logger
import cytearc
import cytearc.utils.logging

logger.debug("host debug")
cytearc.logger.info("cytearc info")
cytearc.configure_output(level="ERROR")
logger.warning("host warning")
cytearc.logger.warning("cytearc warning")
cytearc.logger.error("cytearc error")
"""
    result = _run_isolated(code)
    errors = _stderr_lines(result)

    # CyteArc's records appear once, on stdout, in CyteArc's format.
    assert result.stdout.splitlines() == ["INFO: cytearc info", "ERROR: cytearc error"]
    # Other code keeps loguru's default stderr handler, its DEBUG level, and
    # its format, and CyteArc's level setting does not silence it.
    assert len(errors) == 2
    assert "| DEBUG    | __main__:<module>:6 - host debug" in errors[0]
    assert "| WARNING  | __main__:<module>:9 - host warning" in errors[1]


def test_an_application_that_configured_loguru_keeps_its_handlers():
    code = """
import sys
from loguru import logger

logger.remove()
host = logger.add(sys.stdout, format="HOST {level} {message}", level="DEBUG")
import cytearc
import cytearc.utils.logging

logger.info("host first")
logger.debug("host debug")
cytearc.configure_output(level="ERROR")
logger.info("host info")
logger.remove(host)
"""
    result = _run_isolated(code)

    # CyteArc's own sink shows none of the application's records, and the
    # application's handler prints each of them once, below CyteArc's level too.
    assert result.stdout.splitlines() == [
        "HOST INFO host first",
        "HOST DEBUG host debug",
        "HOST INFO host info",
    ]
    assert result.stderr == ""


def test_the_replaced_default_handler_honors_loguru_filter():
    code = """
from loguru import logger
import cytearc

logger.patch(lambda record: record.update(name="hostapp.sub")).info("kept")
logger.patch(lambda record: record.update(name="hostapplication")).info("other")
logger.info("main module")
cytearc.logger.info("cytearc")
"""
    result = _run_isolated(code, LOGURU_FILTER="hostapp")
    errors = _stderr_lines(result)

    assert len(errors) == 1 and errors[0].endswith(" - kept")
    assert result.stdout.splitlines() == ["INFO: cytearc"]


def test_default_handler_filter_keeps_only_other_code_records(monkeypatch):
    import cytearc.utils.logging as cytearc_logging

    def record(name: str | None, **extra: object) -> dict[str, object]:
        return {"name": name, "extra": extra}

    keeps = cytearc_logging._other_record
    monkeypatch.setattr(cytearc_logging, "_LOGURU_FILTER", None)
    assert keeps(record("host"))
    assert not keeps(record("host", cytearc=True))
    monkeypatch.setattr(cytearc_logging, "_LOGURU_FILTER", "")
    assert keeps(record(None))
    monkeypatch.setattr(cytearc_logging, "_LOGURU_FILTER", "hostapp")
    assert keeps(record("hostapp")) and keeps(record("hostapp.sub"))
    assert not keeps(record("hostapplication"))
    assert not keeps(record(None))
    assert cytearc_logging._cytearc_record(record("anything", cytearc=True))
    assert not cytearc_logging._cytearc_record(record("cytearc.storage"))
