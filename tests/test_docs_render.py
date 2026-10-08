from copy import deepcopy
from pathlib import Path
import shutil

import pytest


@pytest.fixture
def executable_docs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("jupytext")
    nbformat = pytest.importorskip("nbformat")
    yaml = pytest.importorskip("yaml")
    from docs import build_docs

    source = tmp_path / "source"
    source.mkdir()
    (source / "myst.yml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "project": {
                    "github": "https://github.com/example/docs",
                    "toc": [{"file": "example.ipynb"}],
                },
            }
        )
    )
    saved = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# Example"),
            nbformat.v4.new_code_cell(
                'print("RESULT")',
                execution_count=1,
                outputs=[
                    nbformat.v4.new_output("stream", name="stdout", text="RESULT\n")
                ],
            ),
        ],
        metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}},
    )
    saved.metadata["executed_at"] = "2026-10-07T22:00:00+00:00"
    saved.metadata["executed_code_sha256"] = build_docs.code_fingerprint(saved)
    nbformat.write(saved, source / "example.ipynb")
    monkeypatch.setattr(build_docs, "SOURCE", source)
    monkeypatch.setattr(build_docs, "SITE", tmp_path / "site")
    return build_docs


def test_html_requires_saved_execution_for_every_executable_page(
    executable_docs,
) -> None:
    import nbformat

    build_docs = executable_docs
    notebook = nbformat.read(build_docs.SOURCE / "example.ipynb", as_version=4)
    notebook.metadata.pop("executed_at")
    notebook.metadata.pop("executed_code_sha256")
    notebook.cells[1].execution_count = None
    notebook.cells[1].outputs = []
    for page in ("example.ipynb", "second.ipynb"):
        nbformat.write(notebook, build_docs.SOURCE / page)

    with pytest.raises(ValueError, match="Saved execution is missing") as error:
        build_docs.prepare_site()

    assert "example.ipynb" in str(error.value)
    assert "second.ipynb" in str(error.value)
    assert "--preview" in str(error.value)


@pytest.mark.parametrize("preview", [False, True])
def test_code_cells_require_kernel_metadata(executable_docs, preview: bool) -> None:
    import nbformat

    build_docs = executable_docs
    source = build_docs.SOURCE / "example.ipynb"
    notebook = nbformat.read(source, as_version=4)
    notebook.metadata.pop("kernelspec")
    nbformat.write(notebook, source)

    with pytest.raises(ValueError, match="example.ipynb has no kernelspec metadata"):
        build_docs.prepare_site(preview=preview)


def test_plain_python_examples_need_no_execution(executable_docs) -> None:
    import yaml

    build_docs = executable_docs
    (build_docs.SOURCE / "example.ipynb").unlink()
    (build_docs.SOURCE / "example.md").write_text(
        '# Example\n\n```python\nprint("RESULT")\n```\n'
    )
    config_path = build_docs.SOURCE / "myst.yml"
    config = yaml.safe_load(config_path.read_text())
    config["project"]["toc"] = [{"file": "example.md"}]
    config_path.write_text(yaml.safe_dump(config))

    site = build_docs.prepare_site()

    assert (site / "example.md").is_file()
    assert not (site / "example.ipynb").exists()


def test_source_and_edit_links_target_master(executable_docs) -> None:
    import nbformat
    import yaml

    build_docs = executable_docs
    (build_docs.SOURCE / "guide.md").write_text("# Guide\n")

    site = build_docs.prepare_site()
    notebook = nbformat.read(site / "example.ipynb", as_version=4)
    guide = yaml.safe_load((site / "guide.md").read_text().split("---", 2)[1])

    for page, metadata in (("example.ipynb", notebook.metadata), ("guide.md", guide)):
        assert metadata["source_url"] == (
            f"https://github.com/example/docs/blob/master/docs/source/{page}"
        )
        assert metadata["edit_url"] == (
            f"https://github.com/example/docs/edit/master/docs/source/{page}"
        )


def test_saved_notebooks_restore_complete_outputs_in_a_clean_checkout(
    executable_docs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nbformat
    import yaml

    build_docs = executable_docs
    held = build_docs.SOURCE / "held.ipynb"
    held.write_text("Excluded outputs")
    config_path = build_docs.SOURCE / "myst.yml"
    config = yaml.safe_load(config_path.read_text())
    config["project"]["exclude"] = ["held.ipynb"]
    config_path.write_text(yaml.safe_dump(config))
    checkout = tmp_path / "checkout" / "docs"
    shutil.copytree(build_docs.SOURCE, checkout / "source")
    monkeypatch.setattr(build_docs, "SOURCE", checkout / "source")
    monkeypatch.setattr(build_docs, "SITE", checkout / "build" / "site")
    assert not (checkout / "build").exists()
    source = build_docs.SOURCE / "example.ipynb"
    current = nbformat.read(source, as_version=4)
    current.cells[0].source = "# Updated"
    nbformat.write(current, source)
    source_text = source.read_text()

    site = build_docs.prepare_site()
    notebook = nbformat.read(site / "example.ipynb", as_version=4)

    assert notebook.cells[0].source == "# Updated"
    assert notebook.cells[1].outputs[0].text == "RESULT\n"
    assert notebook.cells[1].execution_count == 1
    assert notebook.metadata["executed_at"] == "2026-10-07T22:00:00+00:00"
    assert (
        notebook.metadata["executed_code_sha256"]
        == (current.metadata["executed_code_sha256"])
    )
    assert not (site / "held.ipynb").exists()
    assert source.read_text() == source_text

    current.cells[1].source = 'print("CHANGED")'
    nbformat.write(current, source)
    with pytest.raises(ValueError, match="Code changed"):
        build_docs.prepare_site()


def test_preview_labels_missing_execution_without_fabricating_outputs(
    executable_docs,
) -> None:
    import nbformat

    build_docs = executable_docs
    source = build_docs.SOURCE / "example.ipynb"
    current = nbformat.read(source, as_version=4)
    current.metadata.pop("executed_at")
    current.metadata.pop("executed_code_sha256")
    current.cells[1].execution_count = None
    current.cells[1].outputs = []
    nbformat.write(current, source)
    source_text = source.read_text()

    site = build_docs.prepare_site(preview=True)
    notebook = nbformat.read(site / "example.ipynb", as_version=4)

    assert notebook.cells[0].cell_type == "markdown"
    assert notebook.cells[0].source.startswith(":::{warning} Source preview\n")
    assert "Figures and results are not available" in notebook.cells[0].source
    assert "executed_at" not in notebook.metadata
    assert "executed_code_sha256" not in notebook.metadata
    assert all(
        cell.execution_count is None and cell.outputs == []
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    assert source.read_text() == source_text


@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("timestamp", "no completed execution timestamp"),
        ("fingerprint", "fingerprint"),
        ("unexecuted", "unexecuted cell"),
        ("error", "has an error in cell"),
        ("changed_code", "Code changed"),
    ],
)
def test_build_rejects_invalid_saved_execution(
    executable_docs, preview: bool, failure: str, message: str
) -> None:
    import nbformat

    build_docs = executable_docs
    saved_path = build_docs.SOURCE / "example.ipynb"
    saved = nbformat.read(saved_path, as_version=4)
    if failure == "timestamp":
        saved.metadata.pop("executed_at")
    elif failure == "fingerprint":
        saved.metadata.pop("executed_code_sha256")
    elif failure == "unexecuted":
        saved.cells[1].execution_count = None
    elif failure == "error":
        saved.cells[1].outputs = [
            nbformat.v4.new_output(
                "error", ename="RuntimeError", evalue="failed", traceback=["failed"]
            )
        ]
    else:
        saved.cells[1].source = 'print("OLD RESULT")'
    nbformat.write(saved, saved_path)

    with pytest.raises(ValueError, match=message):
        build_docs.prepare_site(preview=preview)


def test_completed_cells_need_not_produce_outputs(executable_docs) -> None:
    import nbformat

    build_docs = executable_docs
    saved_path = build_docs.SOURCE / "example.ipynb"
    saved = nbformat.read(saved_path, as_version=4)
    saved.cells[1].source = "value = 1"
    saved.cells[1].outputs = []
    saved.metadata["executed_code_sha256"] = build_docs.code_fingerprint(saved)
    nbformat.write(saved, saved_path)

    site = build_docs.prepare_site()
    notebook = nbformat.read(site / "example.ipynb", as_version=4)

    assert notebook.cells[1].execution_count == 1
    assert notebook.cells[1].outputs == []
    assert "Source preview" not in notebook.cells[0].source


@pytest.mark.parametrize(
    ("exclude", "suffix"),
    [
        ("tutorials/held.ipynb", ".ipynb"),
        ("tutorials/**", ".ipynb"),
        ("**/held.ipynb", ".ipynb"),
        ("tutorials/held.md", ".md"),
    ],
)
def test_excluded_pages_need_no_execution_and_leave_no_generated_routes(
    executable_docs, exclude: str, suffix: str
) -> None:
    import nbformat
    import yaml

    build_docs = executable_docs
    held = build_docs.SOURCE / "tutorials" / f"held{suffix}"
    held.parent.mkdir()
    if suffix == ".md":
        held.write_text("# Held\n\n```{code-cell} python\nraise RuntimeError\n```\n")
    else:
        held.write_text("Invalid excluded notebook")
    config_path = build_docs.SOURCE / "myst.yml"
    config = yaml.safe_load(config_path.read_text())
    config["project"]["exclude"] = [exclude]
    config["project"]["toc"].append(
        {"title": "Tutorials", "children": [{"file": f"tutorials/held{suffix}"}]}
    )
    config_path.write_text(yaml.safe_dump(config))
    old_html = build_docs.SITE / "_build" / "html"
    old_html.mkdir(parents=True)
    (old_html / "held.html").write_text("Old held page")
    (old_html / "held.png").write_bytes(b"old figure")
    (build_docs.SITE / "held.ipynb").write_text("Old staged notebook")

    site = build_docs.prepare_site()
    notebook = nbformat.read(site / "example.ipynb", as_version=4)
    staged_config = yaml.safe_load((site / "myst.yml").read_text())

    assert notebook.cells[1].outputs[0].text == "RESULT\n"
    assert not (site / "tutorials" / "held.md").exists()
    assert not (site / "tutorials" / "held.ipynb").exists()
    assert not (site / "held.ipynb").exists()
    assert not old_html.exists()
    assert staged_config["project"]["toc"][1]["children"] == []
    assert held.is_file()
    assert yaml.safe_load(config_path.read_text()) == config


@pytest.mark.parametrize("change", ["edit", "reorder", "delete"])
@pytest.mark.parametrize("preview", [False, True])
def test_saved_outputs_follow_unchanged_code_and_current_prose(
    executable_docs, change: str, preview: bool
) -> None:
    import nbformat

    build_docs = executable_docs
    source = build_docs.SOURCE / "example.ipynb"
    saved = nbformat.read(source, as_version=4)
    saved.cells.append(
        nbformat.v4.new_code_cell(
            'print("SECOND")',
            execution_count=2,
            outputs=[nbformat.v4.new_output("stream", name="stdout", text="SECOND\n")],
        )
    )
    saved.metadata["executed_code_sha256"] = build_docs.code_fingerprint(saved)
    current = deepcopy(saved)
    current.cells[0].source = "# Edited prose"
    current.cells.insert(2, nbformat.v4.new_markdown_cell("More explanation."))
    nbformat.write(current, source)

    site = build_docs.prepare_site(preview=preview)
    rendered = nbformat.read(site / "example.ipynb", as_version=4)

    assert rendered.cells[0].source == "# Edited prose"
    assert rendered.cells[2].source == "More explanation."
    assert [cell for cell in rendered.cells if cell.cell_type == "code"] == (
        [cell for cell in saved.cells if cell.cell_type == "code"]
    )
    assert rendered.metadata["executed_at"] == saved.metadata["executed_at"]
    assert (
        rendered.metadata["executed_code_sha256"]
        == (saved.metadata["executed_code_sha256"])
    )

    changed = deepcopy(current)
    if change == "edit":
        changed.cells[1].source = 'print("CHANGED")'
    elif change == "reorder":
        changed.cells[1], changed.cells[3] = changed.cells[3], changed.cells[1]
    else:
        del changed.cells[3]
    nbformat.write(changed, source)
    with pytest.raises(ValueError, match='PAGES="example"'):
        build_docs.prepare_site(preview=preview)


@pytest.mark.parametrize("state", ["count", "output", "timestamp", "fingerprint"])
def test_preview_rejects_partial_execution(executable_docs, state: str) -> None:
    import nbformat

    build_docs = executable_docs
    source = build_docs.SOURCE / "example.ipynb"
    current = nbformat.read(source, as_version=4)
    current.metadata.pop("executed_at")
    current.metadata.pop("executed_code_sha256")
    current.cells[1].execution_count = None
    current.cells[1].outputs = []
    if state == "count":
        current.cells[1].execution_count = 1
    elif state == "output":
        current.cells[1].outputs = [
            nbformat.v4.new_output("stream", name="stdout", text="Partial result\n")
        ]
    elif state == "timestamp":
        current.metadata["executed_at"] = "2026-10-07T22:00:00+00:00"
    else:
        current.metadata["executed_code_sha256"] = build_docs.code_fingerprint(current)
    nbformat.write(current, source)
    source_text = source.read_text()

    with pytest.raises(ValueError):
        build_docs.prepare_site(preview=True)

    assert source.read_text() == source_text


@pytest.mark.parametrize("preview", [False, True])
def test_executable_markdown_requires_notebook_source(
    executable_docs, preview: bool
) -> None:
    import yaml

    build_docs = executable_docs
    (build_docs.SOURCE / "example.ipynb").unlink()
    (build_docs.SOURCE / "example.md").write_text(
        "---\nkernelspec:\n  name: python3\n  display_name: Python 3\n---\n"
        '# Example\n\n```{code-cell} python\nprint("RESULT")\n```\n'
    )
    config_path = build_docs.SOURCE / "myst.yml"
    config = yaml.safe_load(config_path.read_text())
    config["project"]["toc"] = [{"file": "example.md"}]
    config_path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match=r"\.ipynb") as error:
        build_docs.prepare_site(preview=preview)

    assert "example.md" in str(error.value)


@pytest.mark.parametrize(
    ("base_url", "prefix"),
    [
        (None, ""),
        ("", ""),
        ("/", ""),
        ("/CyteArc", "/CyteArc"),
        ("/CyteArc/", "/CyteArc"),
        ("/docs/CyteArc/", "/docs/CyteArc"),
    ],
)
def test_prepared_sidebar_urls_respect_site_base_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    base_url: str | None,
    prefix: str,
) -> None:
    pytest.importorskip("jupytext")
    pytest.importorskip("nbformat")
    yaml = pytest.importorskip("yaml")
    from docs import build_docs

    source = tmp_path / "source"
    source.mkdir()
    (source / "index.md").write_text("# Example\n")
    unchanged_urls = [
        "https://example.org/api/",
        "//example.org/api/",
        "mailto:docs@example.org",
        "relative.html",
        "#section",
    ]
    config = {
        "version": 1,
        "project": {
            "github": "https://github.com/example/docs",
            "toc": [
                {"file": "index.md"},
                {"url": "/quickstart#section"},
                {
                    "title": "Reference",
                    "children": [
                        {"url": "/api/index.html", "open_in_same_tab": True},
                        *({"url": url} for url in unchanged_urls),
                    ],
                },
            ],
        },
    }
    source_config = yaml.safe_dump(config)
    (source / "myst.yml").write_text(source_config)
    monkeypatch.setattr(build_docs, "SOURCE", source)
    monkeypatch.setattr(build_docs, "SITE", tmp_path / "site")
    if base_url is None:
        monkeypatch.delenv("BASE_URL", raising=False)
    else:
        monkeypatch.setenv("BASE_URL", base_url)

    site = build_docs.prepare_site()
    toc = yaml.safe_load((site / "myst.yml").read_text())["project"]["toc"]

    assert toc[0] == {"file": "index.md"}
    assert toc[1]["url"] == f"{prefix}/quickstart#section"
    assert toc[2]["children"][0] == {
        "url": f"{prefix}/api/index.html",
        "open_in_same_tab": True,
    }
    assert [entry["url"] for entry in toc[2]["children"][1:]] == unchanged_urls
    assert (source / "myst.yml").read_text() == source_config
