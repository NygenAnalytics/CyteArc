from copy import deepcopy
from datetime import datetime
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess
from xml.etree import ElementTree

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
    current.cells[1].metadata["figure_descriptions"] = [
        "An embedding that will be generated when this page is executed."
    ]
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
    assert notebook.cells[2].metadata.figure_descriptions == (
        current.cells[1].metadata.figure_descriptions
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


@pytest.mark.parametrize("figure_count", [1, 2])
def test_staged_figure_uses_authored_alt_without_changing_saved_output(
    executable_docs,
    figure_count: int,
) -> None:
    import nbformat

    build_docs = executable_docs
    source = build_docs.SOURCE / "example.ipynb"
    notebook = nbformat.read(source, as_version=4)
    image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aD1sAAAAASUVORK5CYII="
    descriptions = [
        "UMAP of the prepared PBMC analysis, colored by RNA cluster.",
        "The same UMAP colored by MS4A1 expression.",
    ][:figure_count]
    notebook.cells[1].metadata["figure_descriptions"] = descriptions
    notebook.cells[1].outputs = [
        nbformat.v4.new_output(
            "display_data",
            data={
                "image/png": image,
                "text/plain": "<Figure size 400x300 with 1 Axes>",
            },
        )
        for _ in descriptions
    ]
    nbformat.write(notebook, source)
    original = source.read_bytes()

    site = build_docs.prepare_site()

    staged = nbformat.read(site / "example.ipynb", as_version=4)
    assert [
        output.data["text/plain"] for output in staged.cells[1].outputs
    ] == descriptions
    assert all(output.data["image/png"] == image for output in staged.cells[1].outputs)
    assert build_docs.code_fingerprint(staged) == notebook.metadata.executed_code_sha256
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    "descriptions",
    [None, [], [""], ["   "], ["First", "Unexpected second"], "not a list"],
)
def test_staged_figure_rejects_missing_description(
    executable_docs, descriptions
) -> None:
    import nbformat

    build_docs = executable_docs
    source = build_docs.SOURCE / "example.ipynb"
    notebook = nbformat.read(source, as_version=4)
    notebook.cells[1].outputs = [
        nbformat.v4.new_output(
            "display_data",
            data={"image/png": "aW1hZ2U=", "text/plain": "<Figure with 1 Axes>"},
        )
    ]
    if descriptions is not None:
        notebook.cells[1].metadata["figure_descriptions"] = descriptions
    nbformat.write(notebook, source)

    with pytest.raises(ValueError, match="example.ipynb.*metadata.figure_descriptions"):
        build_docs.prepare_site()


@pytest.fixture(params=["", "/CyteArc"])
def rendered_discovery_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> tuple[Path, Path, str]:
    pytest.importorskip("jupytext")
    pytest.importorskip("nbformat")
    pytest.importorskip("yaml")
    base_url = request.param
    monkeypatch.setenv("BASE_URL", base_url)
    html = tmp_path / "_build" / "html"
    api_text = tmp_path / "api-text"
    for directory in (html, api_text, html / "api" / "_sources"):
        directory.mkdir(parents=True)

    references = []
    for name, route in (("index", "/"), ("quickstart", "/quickstart")):
        references.append({"kind": "page", "url": route, "data": f"/{name}.json"})
        children = (
            [
                {
                    "type": "outputs",
                    "id": "saved-result",
                    "children": [
                        {
                            "type": "output",
                            "jupyter_data": {
                                "output_type": "execute_result",
                                "data": {
                                    "text/plain": {"content": "cell_count\n    1000"}
                                },
                            },
                        }
                    ],
                }
            ]
            if name == "quickstart"
            else []
        )
        (html / f"{name}.json").write_text(
            json.dumps(
                {
                    "frontmatter": {"title": name},
                    "mdast": {"type": "root", "children": children},
                }
            )
        )
        saved_output = (
            '<div data-mdast-node-id="saved-result">Loading...'
            '<img src="plot.png"/><p>LEAKED_OUTPUT</p></div>'
            if name == "quickstart"
            else ""
        )
        page = html / route.strip("/") / "index.html"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(
            f"<html><head><title>{name}</title></head>"
            f"<body><nav>Site navigation</nav><article><h1>{name}</h1>"
            "<p>Saved result: 42 cells.</p>"
            "<pre>def count():\n    return 42\n\nprint(count())\n42</pre>"
            "<table><tr><th>Group</th><th>Cells</th></tr>"
            "<tr><td>T cells</td><td>42</td></tr></table>"
            '<p>Read <a href="details#reading">the details</a>.</p>'
            '<img src="plot.png" alt="Cell embedding">'
            f"{saved_output}"
            "<p>Analysis continues after the saved output.</p>"
            "</article><footer>Footer navigation</footer></body></html>"
        )
    references.extend(
        [
            {
                "kind": "page",
                "identifier": "quickstart",
                "url": "/quickstart",
                "data": "/quickstart.json",
            },
            {
                "kind": "heading",
                "identifier": "results",
                "url": "/quickstart",
                "data": "/quickstart.json",
            },
        ]
    )
    (html / "myst.xref.json").write_text(json.dumps({"references": references}))
    (html / "api" / "_sources" / "datastore.md").write_text(
        "# DataStore\n\n```{autoclass} cytearc.DataStore\n:members:\n```\n"
    )
    (html / "api" / "datastore.html").write_text(
        '<html><head><link rel="canonical" '
        'href="https://docs.nygen.io/CyteArc/api/datastore.html" /></head>'
        "<body>DataStore(path, *, mode='r')</body></html>"
    )
    (api_text / "datastore.txt").write_text(
        "DataStore API\n=============\n\nDataStore(path, *, mode='r')\n"
        "    Open a saved datastore.\n"
    )
    for relative in (
        "build/api/datastore.html",
        "api/search.html",
        "api/genindex.html",
    ):
        page = html / relative
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text("<html><head></head><body>Extra generated page</body></html>")
    (html / "llms.txt").write_text(
        "# CyteArc\n\n- [Home](./): overview\n"
        "\n## Getting started\n\n"
        "- [Quick start](quickstart/): first analysis\n"
        "\n## API reference\n\n"
        "- [DataStore](api/datastore.html): public API\n"
    )
    (html / "robots.txt").write_text(
        "User-agent: *\nAllow: /\nSitemap: http://localhost:3000/sitemap.xml\n"
    )
    (html / "sitemap.xml").write_text(
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        "<url><loc>http://localhost:3000/quickstart</loc></url></urlset>"
    )
    return html, api_text, base_url


def _discovery_links(page: Path) -> list[dict[str, str | None]]:
    class LinkParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.links: list[dict[str, str | None]] = []

        def handle_starttag(
            self, tag: str, attrs: list[tuple[str, str | None]]
        ) -> None:
            if tag == "link":
                self.links.append(dict(attrs))

    parser = LinkParser()
    parser.feed(page.read_text())
    return parser.links


def test_discovery_metadata_uses_public_urls_and_local_export_links(
    rendered_discovery_files: tuple[Path, Path, str],
) -> None:
    from docs import build_docs

    html, api_text, base_url = rendered_discovery_files
    build_docs.write_discovery_files(html, api_text, revision="0123456789ab")

    public = "https://docs.nygen.io/CyteArc"
    pages = [
        ("index.html", "/", "/index.txt", "text/plain"),
        (
            "quickstart/index.html",
            "/quickstart/",
            "/quickstart/index.txt",
            "text/plain",
        ),
        (
            "api/datastore.html",
            "/api/datastore.html",
            "/api/datastore.txt",
            "text/plain",
        ),
    ]
    for page, route, export, media_type in pages:
        links = _discovery_links(html / page)
        canonicals = [link for link in links if link.get("rel") == "canonical"]
        assert len(canonicals) == 1
        assert canonicals[0]["href"] == f"{public}{route}"
        descriptions = [link for link in links if link.get("rel") == "describedby"]
        assert len(descriptions) == 1
        assert descriptions[0]["href"] == f"{base_url}/llms.txt"
        alternatives = [
            link
            for link in links
            if link.get("rel") == "alternate" and link.get("type") == media_type
        ]
        assert len(alternatives) == 1
        assert alternatives[0]["href"] == f"{base_url}{export}"
    assert "<h1>quickstart</h1>" in (html / "quickstart" / "index.html").read_text()

    sitemap = ElementTree.parse(html / "sitemap.xml")
    namespace = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
    assert sitemap.getroot().tag == f"{namespace}urlset"
    locations = [
        element.text for element in sitemap.findall(f"{namespace}url/{namespace}loc")
    ]
    assert len(locations) == 3
    assert set(locations) == {f"{public}{route}" for _, route, _, _ in pages}
    robots = (html / "robots.txt").read_text()
    assert f"Sitemap: {public}/sitemap.xml" in robots
    assert "http://localhost" not in robots
    assert "User-agent: *" in robots
    assert "Allow: /" in robots


def test_discovery_exports_preserve_content_and_identify_revision(
    rendered_discovery_files: tuple[Path, Path, str],
) -> None:
    from docs import build_docs

    html, api_text, _ = rendered_discovery_files
    revision = "0123456789ab"
    build_docs.write_discovery_files(html, api_text, revision=revision)

    for relative in ("index.txt", "quickstart/index.txt", "api/datastore.txt"):
        exported = (html / relative).read_text()
        assert revision in exported
        assert "{autoclass}" not in exported
    for relative in ("index.txt", "quickstart/index.txt"):
        exported = (html / relative).read_text()
        assert "Saved result: 42 cells." in exported
        assert "def count():\n    return 42\n\nprint(count())\n42" in exported
        assert all(value in exported for value in ("Group", "Cells", "T cells", "42"))
        assert "Site navigation" not in exported
        assert "Footer navigation" not in exported
    quickstart = (html / "quickstart" / "index.txt").read_text()
    assert "https://docs.nygen.io/CyteArc/quickstart/details#reading" in quickstart
    assert "https://docs.nygen.io/CyteArc/quickstart/plot.png" in quickstart
    assert "Cell embedding" in quickstart
    assert quickstart.count("cell_count\n    1000") == 1
    assert "Loading..." not in quickstart
    assert "LEAKED_OUTPUT" not in quickstart
    assert "Analysis continues after the saved output." in quickstart
    api = (html / "api" / "datastore.txt").read_text()
    assert "DataStore(path, *, mode='r')\n    Open a saved datastore." in api
    llms = (html / "llms.txt").read_text()
    assert revision in llms
    links = dict(re.findall(r"\[([^\]]+)\]\(([^)]+)\)", llms))
    assert {label: links[label] for label in ("Home", "Quick start", "DataStore")} == {
        "Home": "https://docs.nygen.io/CyteArc/index.txt",
        "Quick start": "https://docs.nygen.io/CyteArc/quickstart/index.txt",
        "DataStore": "https://docs.nygen.io/CyteArc/api/datastore.txt",
    }


def test_discovery_rejects_missing_api_text(
    rendered_discovery_files: tuple[Path, Path, str],
) -> None:
    from docs import build_docs

    html, api_text, _ = rendered_discovery_files
    path = api_text / "datastore.txt"
    path.unlink()

    with pytest.raises(FileNotFoundError, match=re.escape(path.name)):
        build_docs.write_discovery_files(html, api_text, revision="0123456789ab")


@pytest.mark.parametrize(
    "content", ["<p>Navigation without an article</p>", "<article> </article>"]
)
def test_discovery_rejects_missing_or_empty_article(
    rendered_discovery_files: tuple[Path, Path, str], content: str
) -> None:
    from docs import build_docs

    html, api_text, _ = rendered_discovery_files
    (html / "quickstart" / "index.html").write_text(
        f"<html><head></head><body>{content}</body></html>"
    )

    with pytest.raises(ValueError, match="quickstart"):
        build_docs.write_discovery_files(html, api_text, revision="0123456789ab")


@pytest.mark.parametrize("as_lines", [False, True])
def test_saved_html_tables_become_native_nodes_without_changing_output_data(
    as_lines: bool,
) -> None:
    from docs.render_outputs import render_outputs

    content = (
        '<div><style>.dataframe { text-align: right; }</style><table class="dataframe">'
        '<thead><tr><th></th><th colspan="2">Counts</th></tr></thead><tbody>'
        '<tr><th rowspan="2" valign="top">Donor A</th><td>1000</td><td>&lt;5</td></tr>'
        "<tr><td>1200</td><td>6</td></tr></tbody></table>"
        "<p>2 rows × 2 columns</p></div>"
    )
    output = {
        "output_type": "execute_result",
        "execution_count": 1,
        "metadata": {},
        "data": {
            "text/html": [content] if as_lines else content,
            "text/plain": "Counts\nDonor A 1000 <5\n        1200 6",
        },
    }
    node = {"type": "output", "jupyter_data": deepcopy(output), "children": []}

    render_outputs(node)

    assert node["type"] == "block"
    assert node["jupyter_data"] == output
    table, summary = node["children"]
    assert table["type"] == "table"
    assert len(table["children"]) == 3
    heading = table["children"][0]["children"][1]
    assert heading == {
        "type": "tableCell",
        "header": True,
        "colspan": 2,
        "children": [{"type": "text", "value": "Counts"}],
    }
    row = table["children"][1]["children"]
    assert row[0]["header"] is True
    assert row[0]["rowspan"] == 2
    assert row[1]["header"] is False
    assert row[1]["children"][0]["value"] == "1000"
    assert row[2]["children"][0]["value"] == "<5"
    assert summary == {
        "type": "paragraph",
        "children": [{"type": "text", "value": "2 rows × 2 columns"}],
    }


def test_saved_output_transform_preserves_streams_and_renders_timing() -> None:
    from docs.render_outputs import render_outputs

    timing = {
        "type": "output",
        "jupyter_data": {
            "output_type": "display_data",
            "metadata": {"cytearc_execution_timing": True},
            "data": {
                "text/html": "<small>Wall time: 1.234 s</small>",
                "text/plain": ["Wall time: ", "1.234 s"],
            },
        },
        "children": [],
    }
    stream = {
        "type": "output",
        "jupyter_data": {"output_type": "stream", "name": "stdout", "text": "Done"},
        "children": [],
    }
    tree = {"type": "outputs", "children": [deepcopy(stream), deepcopy(timing)]}

    render_outputs(tree)

    assert tree["children"][0] == stream
    rendered = tree["children"][1]
    assert rendered["jupyter_data"] == timing["jupyter_data"]
    assert rendered["children"] == [
        {
            "type": "paragraph",
            "children": [{"type": "text", "value": "Wall time: 1.234 s"}],
        }
    ]


@pytest.mark.parametrize(
    "content",
    [
        "<table><tr><td><script>unknown()</script></td></tr></table>",
        "<table><tr><td>Unclosed",
        '<table><tr><td rowspan="0">Invalid span</td></tr></table>',
        "<div>Unrecognized output</div>",
    ],
)
def test_saved_output_transform_rejects_unsupported_or_invalid_html(
    content: str,
) -> None:
    from docs.render_outputs import render_outputs

    node = {
        "type": "output",
        "jupyter_data": {"data": {"text/html": content}},
        "children": [],
    }

    with pytest.raises(ValueError):
        render_outputs(node)


@pytest.fixture
def documentation_git_checkout(tmp_path: Path) -> Path:
    pytest.importorskip("jupytext")
    pytest.importorskip("nbformat")
    pytest.importorskip("yaml")
    repository = tmp_path / "checkout"
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--shared",
            "--no-checkout",
            str(Path(__file__).resolve().parents[1]),
            str(repository),
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "checkout",
            "HEAD",
            "--",
            "README.md",
            "LICENSE",
            "docs/source/reference/license.md",
            "docs/source/references.bib",
        ],
        cwd=repository,
        check=True,
    )
    return repository


def test_last_modified_uses_source_and_include_history(
    documentation_git_checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from docs import build_docs

    repository = documentation_git_checkout
    monkeypatch.setattr(build_docs, "SOURCE", repository / "docs/source")
    monkeypatch.setattr(build_docs, "SITE", repository / "docs/build/site")
    bibliography = repository / "docs/source/references.bib"
    source = repository / "docs/source/reference/license.md"
    dependencies = build_docs.myst_source_files(
        {
            "location": "/reference/license.md",
            "dependencies": [],
            "mdast": {
                "type": "root",
                "children": [
                    {"type": "include", "file": "../references.bib"},
                    {
                        "type": "include",
                        "file": "../../../../README.md",
                        "children": [{"type": "include", "file": "LICENSE"}],
                    },
                    {"type": "cite", "label": "paper"},
                ],
            },
        },
        {bibliography},
    )
    assert dependencies == {
        source,
        bibliography,
        repository / "README.md",
        repository / "LICENSE",
    }
    dates = [
        subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", str(path)],
            cwd=repository,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for path in dependencies
    ]
    expected = max(dates, key=datetime.fromisoformat)
    assert build_docs.latest_source_commit(repository, dependencies) == expected
    for path in dependencies:
        path.touch()
    assert build_docs.latest_source_commit(repository, dependencies) == expected


@pytest.mark.parametrize(
    "change", ["unstaged", "staged", "untracked", "deleted", "outside"]
)
def test_last_modified_omits_uncommitted_or_untracked_dependencies(
    documentation_git_checkout: Path, change: str
) -> None:
    from docs import build_docs

    repository = documentation_git_checkout
    source = repository / "LICENSE"
    dependencies = {source}
    if change in {"unstaged", "staged"}:
        source.write_text(source.read_text() + "\nUncommitted change\n")
        if change == "staged":
            subprocess.run(["git", "add", "LICENSE"], cwd=repository, check=True)
    elif change == "untracked":
        include = repository / "untracked.txt"
        include.write_text("Untracked included content\n")
        dependencies.add(include)
    elif change == "deleted":
        source.unlink()
    else:
        dependencies.add(repository.parent / "outside.txt")
    assert build_docs.latest_source_commit(repository, dependencies) is None


def test_last_modified_ignores_unrelated_working_tree_changes(
    documentation_git_checkout: Path,
) -> None:
    from docs import build_docs

    repository = documentation_git_checkout
    dependencies = {repository / "LICENSE"}
    original = build_docs.latest_source_commit(repository, dependencies)
    assert original is not None
    (repository / "README.md").write_text("Unrelated local change\n")
    assert build_docs.latest_source_commit(repository, dependencies) == original


def test_last_modified_tracks_lazy_and_wrapped_api_implementations() -> None:
    addnodes = pytest.importorskip("sphinx.addnodes")
    pytest.importorskip("jupytext")
    pytest.importorskip("nbformat")
    pytest.importorskip("yaml")
    from docs import build_docs

    doctree = addnodes.desc()
    for module, fullname, kind in (
        ("cytearc.writers", "to_mtx", "function"),
        ("cytearc.datastore.namespaces", "FeaturesAccessor.normalize", "method"),
        ("cytearc.readers", "CrReader.autoNames", "attribute"),
        ("cytearc.merge", "MergePlan", "class"),
    ):
        description = addnodes.desc(objtype=kind)
        description += addnodes.desc_signature(module=module, fullname=fullname)
        doctree += description
    paths = build_docs.api_source_files(doctree)
    package = Path(__file__).resolve().parents[1] / "cytearc"
    assert package / "writers/export.py" in paths
    assert package / "datastore/_operations/graph.py" in paths
    assert package / "readers/cellranger.py" in paths
    assert package / "merge/models.py" in paths
    assert all(path.is_relative_to(package) for path in paths)


def test_sitemap_includes_only_known_last_modified_dates(
    rendered_discovery_files: tuple[Path, Path, str],
) -> None:
    from docs import build_docs

    html, api_text, _ = rendered_discovery_files
    date = "2026-10-08T20:02:08+02:00"
    build_docs.write_discovery_files(
        html, api_text, revision="0123456789ab", last_modified={"/quickstart/": date}
    )
    namespace = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
    entries = ElementTree.parse(html / "sitemap.xml").findall(f"{namespace}url")
    dated = {
        entry.findtext(f"{namespace}loc"): entry.findtext(f"{namespace}lastmod")
        for entry in entries
        if entry.find(f"{namespace}lastmod") is not None
    }
    assert dated == {"https://docs.nygen.io/CyteArc/quickstart/": date}


def test_text_export_preserves_saved_markdown_instead_of_object_repr(
    rendered_discovery_files: tuple[Path, Path, str],
) -> None:
    from docs import build_docs

    html, api_text, _ = rendered_discovery_files
    document = json.loads((html / "quickstart.json").read_text())
    output = document["mdast"]["children"][0]["children"][0]["jupyter_data"]
    markdown = "| Dataset | Cells |\n| --- | ---: |\n| PBMC | 1000 |"
    output["data"] = {
        "text/plain": {"content": "<IPython.core.display.Markdown object>"},
        "text/markdown": {"content": markdown},
    }
    (html / "quickstart.json").write_text(json.dumps(document))

    build_docs.write_discovery_files(html, api_text, revision="0123456789ab")

    exported = (html / "quickstart" / "index.txt").read_text()
    assert markdown in exported
    assert "<IPython.core.display.Markdown object>" not in exported
