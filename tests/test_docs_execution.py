from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def executed_notebook():
    pytest.importorskip("modal")
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("nbclient")
    from docs import modal_docs

    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# Example"),
            nbformat.v4.new_code_cell('print("RESULT")'),
            nbformat.v4.new_code_cell(""),
        ],
        metadata={
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
                "language": "python",
            },
            "executed_at": "old execution",
            "executed_code_sha256": "old fingerprint",
        },
    )
    source = nbformat.writes(notebook)
    result = modal_docs.execute_page.local(source)
    return source, result


@pytest.fixture
def execution_source(tmp_path, monkeypatch, executed_notebook):
    from docs import modal_docs

    source = tmp_path / "source"
    source.mkdir()
    (source / "tutorials").mkdir()
    (source / "example.ipynb").write_text(executed_notebook[0])
    monkeypatch.setattr(modal_docs, "SOURCE_DIR", source)
    return modal_docs


def _submit_results(monkeypatch, runner, results, *, before_result=None):
    submitted = []

    def spawn(source):
        index = len(submitted)
        submitted.append(source)

        def get(*, timeout):
            if before_result is not None:
                before_result(index, submitted)
            result = results[index]
            if isinstance(result, Exception):
                raise result
            return result

        return SimpleNamespace(object_id=f"local-{index}", get=get)

    monkeypatch.setattr(runner, "execute_page", SimpleNamespace(spawn=spawn))
    return submitted


def test_execute_page_runs_json_notebook_and_records_success(executed_notebook):
    import nbformat

    notebook = nbformat.reads(executed_notebook[1], as_version=4)

    assert notebook.cells[1].execution_count == 1
    assert notebook.cells[1].outputs[0].text == "RESULT\n"
    assert notebook.cells[2].source == ""
    assert datetime.fromisoformat(notebook.metadata.executed_at).utcoffset() is not None
    assert "executed_code_sha256" not in notebook.metadata
    timing = notebook.cells[1].metadata.execution
    assert datetime.fromisoformat(
        timing["iopub.status.idle"]
    ) >= datetime.fromisoformat(timing["iopub.status.busy"])


def test_execution_timings_preserve_results_and_are_idempotent(executed_notebook):
    import nbformat

    from docs.modal_docs import add_execution_timings

    notebook = nbformat.reads(executed_notebook[1], as_version=4)
    notebook.cells[1].metadata.execution.update(
        {
            "iopub.status.busy": "2026-10-08T12:34:56+02:00",
            "iopub.status.idle": "2026-10-08T12:34:57.250000+02:00",
        }
    )
    original = deepcopy(notebook)

    add_execution_timings(notebook)

    assert notebook.metadata == original.metadata
    assert notebook.cells[0] == original.cells[0]
    assert notebook.cells[2] == original.cells[2]
    cell = notebook.cells[1]
    assert cell.source == original.cells[1].source
    assert cell.execution_count == original.cells[1].execution_count
    assert cell.metadata == original.cells[1].metadata
    assert cell.outputs[:-1] == original.cells[1].outputs
    footer = cell.outputs[-1]
    assert footer.output_type == "display_data"
    assert footer.metadata.cytearc_execution_timing is True
    assert (
        footer.data["text/plain"]
        == "Started: 2026-10-08T10:34:56Z | Wall time: 1.250 s"
    )
    assert footer.data["text/plain"] in footer.data["text/html"]
    nbformat.validate(notebook)
    first = deepcopy(notebook)

    add_execution_timings(notebook)

    assert notebook == first


@pytest.mark.parametrize("failure", ["missing", "invalid", "naive", "negative"])
def test_execution_timings_reject_missing_or_invalid_records(
    executed_notebook, failure
):
    import nbformat

    from docs.modal_docs import add_execution_timings

    notebook = nbformat.reads(executed_notebook[1], as_version=4)
    timing = notebook.cells[1].metadata.execution
    if failure == "missing":
        timing.pop("iopub.status.idle")
    elif failure == "invalid":
        timing["iopub.status.busy"] = "invalid"
    elif failure == "naive":
        timing["iopub.status.busy"] = "2026-10-08T12:00:00"
    else:
        timing["iopub.status.busy"] = "2026-10-08T12:00:00Z"
        timing["iopub.status.idle"] = "2026-10-08T11:00:00Z"

    with pytest.raises(ValueError, match="timing"):
        add_execution_timings(notebook)

    assert notebook.cells[1].outputs[0].text == "RESULT\n"
    assert len(notebook.cells[1].outputs) == 1


def test_execute_page_does_not_return_failed_execution(executed_notebook):
    import nbformat
    from nbclient.exceptions import CellExecutionError

    from docs import modal_docs

    notebook = nbformat.reads(executed_notebook[0], as_version=4)
    notebook.cells[1].source = 'raise RuntimeError("execution failed")'

    with pytest.raises(CellExecutionError, match="execution failed"):
        modal_docs.execute_page.local(nbformat.writes(notebook))


@pytest.mark.parametrize("change", ["empty", "skip_execution"])
def test_reexecution_clears_old_results_before_running_changed_cells(
    execution_source, executed_notebook, monkeypatch, change
):
    import nbformat

    from docs.build_docs import code_fingerprint

    runner = execution_source
    path = runner.SOURCE_DIR / "example.ipynb"
    notebook = nbformat.reads(executed_notebook[1], as_version=4)
    notebook.metadata["executed_code_sha256"] = code_fingerprint(notebook)
    runner.add_execution_timings(notebook)
    cell = notebook.cells[1]
    cell.metadata["custom"] = {"preserve": True}
    cell.source = "" if change == "empty" else 'print("UPDATED")'
    if change == "skip_execution":
        cell.metadata["tags"] = ["skip-execution"]
    source = nbformat.writes(notebook)
    path.write_text(source)
    result = runner.execute_page.local(source)
    _submit_results(monkeypatch, runner, [result])

    if change == "empty":
        runner.main("example")
        saved = nbformat.read(path, as_version=4)
        assert saved.cells[1].outputs == []
        assert saved.cells[1].execution_count is None
        assert saved.metadata.executed_code_sha256 == code_fingerprint(saved)
    else:
        with pytest.raises(RuntimeError, match="unexecuted cell 2"):
            runner.main("example")
        assert path.read_text() == source

    returned = nbformat.reads(result, as_version=4)
    assert returned.cells[1].outputs == []
    assert returned.cells[1].execution_count is None
    assert "execution" not in returned.cells[1].metadata
    assert returned.cells[1].metadata.custom == {"preserve": True}
    assert returned.cells[0] == notebook.cells[0]
    if change == "skip_execution":
        assert returned.cells[1].metadata.tags == ["skip-execution"]


@pytest.mark.parametrize(
    "name", ["example", "example.ipynb", "tutorials/example", "tutorials/example.ipynb"]
)
def test_page_path_resolves_notebooks(execution_source, name):
    runner = execution_source
    tutorial = runner.SOURCE_DIR / "tutorials" / "example.ipynb"
    if name.startswith("tutorials/"):
        tutorial.write_text((runner.SOURCE_DIR / "example.ipynb").read_text())

    expected = (
        tutorial
        if name.startswith("tutorials/")
        else runner.SOURCE_DIR / "example.ipynb"
    )
    assert runner.page_path(name) == expected


def test_page_path_resolves_tutorial_stem(execution_source):
    runner = execution_source
    path = runner.SOURCE_DIR / "example.ipynb"
    destination = runner.SOURCE_DIR / "tutorials" / path.name
    path.rename(destination)

    assert runner.page_path("example") == destination


@pytest.mark.parametrize("name", ["example.md", "tutorials/example.md"])
def test_page_path_rejects_markdown_aliases(execution_source, name):
    runner = execution_source
    (runner.SOURCE_DIR / name).write_text("# Source-only page\n")

    with pytest.raises(ValueError, match="Expected a documentation notebook"):
        runner.page_path(name)


@pytest.mark.parametrize("name", ["../outside.ipynb", "linked.ipynb"])
def test_page_path_rejects_notebooks_outside_source(execution_source, name):
    runner = execution_source
    outside = runner.SOURCE_DIR.parent / "outside.ipynb"
    outside.write_text((runner.SOURCE_DIR / "example.ipynb").read_text())
    (runner.SOURCE_DIR / "linked.ipynb").symlink_to(outside)

    with pytest.raises(ValueError, match="Unknown documentation page"):
        runner.page_path(name)


def test_main_validates_and_replaces_canonical_notebook(
    execution_source, executed_notebook, monkeypatch
):
    import nbformat

    from docs.build_docs import code_fingerprint, validate_execution

    runner = execution_source
    path = runner.SOURCE_DIR / "example.ipynb"
    submitted = _submit_results(monkeypatch, runner, [executed_notebook[1]])

    runner.main("example example.ipynb")

    assert submitted == [executed_notebook[0]]
    notebook = nbformat.read(path, as_version=4)
    validate_execution(notebook, Path("example.ipynb"))
    assert notebook.metadata.executed_code_sha256 == code_fingerprint(notebook)
    assert notebook.cells[1].outputs[0].text == "RESULT\n"
    assert notebook.cells[1].outputs[-1].metadata.cytearc_execution_timing is True
    assert {entry.name for entry in runner.SOURCE_DIR.iterdir()} == {
        "example.ipynb",
        "tutorials",
    }


def test_main_submits_all_pages_before_waiting_and_preserves_failed_source(
    execution_source, executed_notebook, monkeypatch
):
    import nbformat

    runner = execution_source
    failed = runner.SOURCE_DIR / "example.ipynb"
    successful = runner.SOURCE_DIR / "tutorials" / "second.ipynb"
    successful.write_text(executed_notebook[0])

    def assert_submitted(index, submitted):
        assert len(submitted) == 2

    _submit_results(
        monkeypatch,
        runner,
        [RuntimeError("kernel failed"), executed_notebook[1]],
        before_result=assert_submitted,
    )

    with pytest.raises(RuntimeError, match="example.ipynb: kernel failed"):
        runner.main("example second")

    assert failed.read_text() == executed_notebook[0]
    assert (
        nbformat.read(successful, as_version=4).cells[1].outputs[0].text == "RESULT\n"
    )


def test_main_preserves_edits_made_during_execution(
    execution_source, executed_notebook, monkeypatch
):
    import nbformat

    runner = execution_source
    path = runner.SOURCE_DIR / "example.ipynb"
    edited = nbformat.reads(executed_notebook[0], as_version=4)
    edited.cells[0].source = "# User edit while execution is running"
    text = nbformat.writes(edited)

    def edit_source(index, submitted):
        path.write_text(text)

    _submit_results(
        monkeypatch, runner, [executed_notebook[1]], before_result=edit_source
    )

    with pytest.raises(RuntimeError, match="Notebook changed during execution"):
        runner.main("example")

    assert path.read_text() == text
    assert not list(runner.SOURCE_DIR.glob(".example-*"))


@pytest.mark.parametrize("change", ["code", "empty_cell"])
def test_main_rejects_results_from_different_code(
    execution_source, executed_notebook, monkeypatch, change
):
    import nbformat

    runner = execution_source
    result = nbformat.reads(executed_notebook[1], as_version=4)
    if change == "code":
        result.cells[1].source = 'print("OTHER")'
    else:
        result.cells.pop()
    _submit_results(monkeypatch, runner, [nbformat.writes(result)])

    with pytest.raises(RuntimeError, match="Executed code differs"):
        runner.main("example")

    assert (runner.SOURCE_DIR / "example.ipynb").read_text() == executed_notebook[0]


@pytest.mark.parametrize("failure", ["timestamp", "unexecuted", "error", "timing"])
def test_main_rejects_incomplete_results_without_overwriting_source(
    execution_source, executed_notebook, monkeypatch, failure
):
    import nbformat

    runner = execution_source
    notebook = nbformat.reads(executed_notebook[1], as_version=4)
    if failure == "timestamp":
        notebook.metadata.pop("executed_at")
    elif failure == "unexecuted":
        notebook.cells[1].execution_count = None
    elif failure == "timing":
        notebook.cells[1].metadata.pop("execution")
    else:
        notebook.cells[1].outputs = [
            nbformat.v4.new_output(
                "error", ename="RuntimeError", evalue="failed", traceback=["failed"]
            )
        ]
    _submit_results(monkeypatch, runner, [nbformat.writes(notebook)])

    with pytest.raises(RuntimeError, match="Notebook execution failed"):
        runner.main("example")

    assert (runner.SOURCE_DIR / "example.ipynb").read_text() == executed_notebook[0]


def test_main_checks_all_sources_before_submitting(execution_source, monkeypatch):
    import nbformat

    runner = execution_source
    invalid = runner.SOURCE_DIR / "tutorials" / "invalid.ipynb"
    notebook = nbformat.read(runner.SOURCE_DIR / "example.ipynb", as_version=4)
    notebook.metadata.pop("kernelspec")
    nbformat.write(notebook, invalid)
    submitted = _submit_results(monkeypatch, runner, [])

    with pytest.raises(ValueError, match="Not an executable notebook"):
        runner.main("example invalid")

    assert submitted == []
