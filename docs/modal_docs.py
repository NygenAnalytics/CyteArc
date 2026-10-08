"""Execute selected documentation notebooks in separate Modal containers."""

from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import modal

from profiling.modal_support import await_function_call

DOCS_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = DOCS_ROOT / "source"
TIMEOUT_SECONDS = 7_200

app = modal.App("cytearc-docs")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "libfftw3-dev", "libmetis-dev")
    .uv_sync(
        groups=["docs-modal"],
        extras=["agent", "cytebase", "docs", "extra"],
        extra_options="--no-default-groups",
        env={"HNSWLIB_NO_NATIVE": "1"},
    )
    .env(
        {
            "CYTEARC_WORKERS": "4",
            "MPLBACKEND": "module://matplotlib_inline.backend_inline",
            "PYTHONPATH": "/root",
        }
    )
    .add_local_python_source("cytearc", copy=True)
    .add_local_file(
        str(DOCS_ROOT / "modal_docs.py"), "/root/docs/modal_docs.py", copy=True
    )
    .add_local_file(
        str(DOCS_ROOT.parent / "profiling" / "modal_support.py"),
        "/root/profiling/modal_support.py",
        copy=True,
    )
)


def page_path(name: str) -> Path:
    relative = Path(name)
    if relative.suffix not in {"", ".ipynb"}:
        raise ValueError(f"Expected a documentation notebook, not {name}")
    relative = relative.with_suffix(".ipynb")
    candidates = [SOURCE_DIR / relative, SOURCE_DIR / "tutorials" / relative]
    for candidate in candidates:
        path = candidate.resolve()
        if path.is_relative_to(SOURCE_DIR.resolve()) and path.is_file():
            return path
    raise ValueError(f"Unknown documentation page: {name}")


def add_execution_timings(notebook) -> None:
    import nbformat

    for index, cell in enumerate(notebook.cells, start=1):
        if cell.cell_type != "code":
            continue
        cell.outputs = [
            output
            for output in cell.outputs
            if output.get("metadata", {}).get("cytearc_execution_timing") is not True
        ]
        if not cell.source.strip():
            continue
        if type(cell.execution_count) is not int or cell.execution_count < 1:
            raise ValueError(
                f"Cannot show execution timing for unexecuted cell {index}"
            )
        timing = cell.metadata.get("execution", {})
        try:
            started = datetime.fromisoformat(timing["iopub.status.busy"])
            finished = datetime.fromisoformat(timing["iopub.status.idle"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Missing or invalid execution timing in cell {index}"
            ) from error
        if started.utcoffset() is None or finished.utcoffset() is None:
            raise ValueError(
                f"Execution timing in cell {index} must include a timezone"
            )
        elapsed = (finished - started).total_seconds()
        if elapsed < 0:
            raise ValueError(f"Execution timing in cell {index} has a negative runtime")
        stamp = started.astimezone(UTC).isoformat().replace("+00:00", "Z")
        text = f"Started: {stamp} | Wall time: {elapsed:.3f} s"
        cell.outputs.append(
            nbformat.v4.new_output(
                "display_data",
                data={"text/plain": text, "text/html": f"<small>{text}</small>"},
                metadata={"cytearc_execution_timing": True},
            )
        )


@app.function(
    image=image,
    cpu=4,
    memory=8_192,
    timeout=TIMEOUT_SECONDS,
    retries=0,
    single_use_containers=True,
    include_source=False,
)
def execute_page(source: str) -> str:
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.reads(source, as_version=4)
    nbformat.validate(notebook)
    if not notebook.metadata.get("kernelspec") or not any(
        cell.cell_type == "code" for cell in notebook.cells
    ):
        raise ValueError("The selected page is not an executable notebook")
    notebook.metadata.pop("executed_at", None)
    notebook.metadata.pop("executed_code_sha256", None)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            cell.metadata.pop("execution", None)
    with TemporaryDirectory(prefix="cytearc-docs-") as working_directory:
        NotebookClient(
            notebook,
            timeout=600,
            allow_errors=False,
            record_timing=True,
            resources={"metadata": {"path": working_directory}},
        ).execute()
    notebook.metadata["executed_at"] = datetime.now(UTC).isoformat()
    return nbformat.writes(notebook)


@app.local_entrypoint()
def main(pages: str) -> None:
    import nbformat

    from docs.build_docs import code_fingerprint, validate_execution

    sources = list(dict.fromkeys(page_path(name) for name in pages.split()))
    if not sources:
        raise ValueError(
            "Specify affected pages, for example --pages 'tea_seq cell_cycle'"
        )
    snapshots = []
    for path in sources:
        source = path.read_text()
        notebook = nbformat.reads(source, as_version=4)
        nbformat.validate(notebook)
        if not notebook.metadata.get("kernelspec") or not any(
            cell.cell_type == "code" for cell in notebook.cells
        ):
            raise ValueError(f"Not an executable notebook: {path}")
        snapshots.append((path, source, code_fingerprint(notebook)))

    calls = [
        (path, source, fingerprint, execute_page.spawn(source))
        for path, source, fingerprint in snapshots
    ]
    for path, _source, _fingerprint, call in calls:
        print(f"Submitted {path.relative_to(SOURCE_DIR)}: {call.object_id}", flush=True)
    failures = []
    for path, source, fingerprint, call in calls:
        relative = path.relative_to(SOURCE_DIR)
        try:
            result = await_function_call(
                call, pollSeconds=20, deadlineSeconds=TIMEOUT_SECONDS + 300
            )
            notebook = nbformat.reads(result, as_version=4)
            nbformat.validate(notebook)
            if code_fingerprint(notebook) != fingerprint:
                raise ValueError("Executed code differs from the submitted notebook")
            notebook.metadata["executed_code_sha256"] = fingerprint
            validate_execution(notebook, relative)
            add_execution_timings(notebook)
            with TemporaryDirectory(prefix=f".{path.stem}-", dir=path.parent) as work:
                temporary = Path(work) / path.name
                nbformat.write(notebook, temporary)
                if path.read_text() != source:
                    raise ValueError(
                        "Notebook changed during execution; preserving current edits"
                    )
                temporary.replace(path)
            print(f"Saved {relative}: {notebook.metadata.executed_at}", flush=True)
        except Exception as error:
            failures.append(f"{relative}: {error}")
            print(f"FAILED {relative}: {error}", flush=True)
    if failures:
        raise RuntimeError("Notebook execution failed:\n" + "\n".join(failures))
