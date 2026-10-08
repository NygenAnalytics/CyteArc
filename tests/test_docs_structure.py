import json
import re
import tomllib
from fnmatch import fnmatch
from pathlib import Path

import yaml


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DOCS_SOURCE = _REPOSITORY_ROOT / "docs" / "source"
_MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
_AGENT_PAGES = {
    "analysis_with_agents.md",
    "tutorials/agent_workflow.md",
    "tutorials/garrido_trigo_agents.md",
    "developers/agent_decisions.md",
    "reference/api/agent.md",
}


def _project() -> dict:
    return yaml.safe_load((_DOCS_SOURCE / "myst.yml").read_text())["project"]


def _source_text(path: Path) -> str:
    if path.suffix != ".ipynb":
        return path.read_text()
    notebook = json.loads(path.read_text())
    return "\n".join(
        cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        for cell in notebook["cells"]
    )


def _toc_files(entries: list[dict]) -> set[str]:
    files = {entry["file"] for entry in entries if "file" in entry}
    for entry in entries:
        files.update(_toc_files(entry.get("children", [])))
    return files


def test_llms_index_links_to_local_documentation() -> None:
    contents = (_DOCS_SOURCE / "llms.txt").read_text()
    targets = _MARKDOWN_LINK.findall(contents)

    assert targets
    for target in targets:
        assert "://" not in target
        page = target.split("#", maxsplit=1)[0]
        if page.startswith("api/"):
            source = _DOCS_SOURCE / "reference" / page.replace(".html", ".md")
        else:
            stem = _DOCS_SOURCE / (
                "index" if page == "./" else page.rstrip("/").replace("-", "_")
            )
            candidates = [
                path
                for suffix in (".md", ".ipynb")
                if (path := stem.with_suffix(suffix)).is_file()
            ]
            assert len(candidates) == 1, (target, candidates)
            source = candidates[0]
        assert source.resolve().is_relative_to(_DOCS_SOURCE.resolve())
        assert source.is_file(), target


def test_navigation_prioritizes_remote_workflows_and_preserves_tutorials() -> None:
    toctree = _project()["toc"]
    sections = {
        entry["title"]: entry["children"] for entry in toctree if "children" in entry
    }
    captions = list(sections)

    assert captions[:3] == [
        "Getting started",
        "scRNA-seq analysis",
        "scATAC-seq and multimodal",
    ]
    assert captions[-2:] == ["Reference", "Developers"]
    assert {
        "concepts/memory_and_execution.md",
        "reference/remote_storage.md",
    } <= _toc_files(sections["Reference"])
    assert sections["Getting started"][0]["file"] == "quickstart.ipynb"
    assert sections["Getting started"][1]["file"] == ("tutorials/remote_stores.ipynb")
    assert all(
        "children" not in entry for entries in sections.values() for entry in entries
    )

    get_started = _toc_files(sections["Getting started"])
    exploration = _toc_files(sections["scRNA-seq analysis"])
    modalities = _toc_files(sections["scATAC-seq and multimodal"])
    assert not {"scanpy.md", "seurat.md"} & get_started
    assert {"scanpy.md", "seurat.md"} <= _toc_files(sections["Data management"])
    assert "scanpy_and_seurat.md" not in get_started
    assert {
        "tutorials/scrna_seq.ipynb",
        "tutorials/plotting.ipynb",
        "tutorials/annotation.ipynb",
    } <= exploration
    assert "AI-assisted analysis" not in sections
    assert {
        "tutorials/scatac_seq.ipynb",
        "tutorials/cite_seq.ipynb",
        "tutorials/tea_seq.ipynb",
        "tutorials/multimodal_diagnostics.ipynb",
        "tutorials/hto_demultiplexing.md",
    } <= modalities

    published = _toc_files(toctree)
    for page in published:
        assert (_DOCS_SOURCE / page).is_file(), page
    tutorials = [
        path
        for path in (_DOCS_SOURCE / "tutorials").iterdir()
        if path.suffix in {".md", ".ipynb"}
    ]
    for tutorial in tutorials:
        page = tutorial.relative_to(_DOCS_SOURCE).as_posix()
        if page not in _AGENT_PAGES:
            assert page in published


def test_focused_scanpy_and_seurat_guides_are_reachable() -> None:
    toctree = _toc_files(_project()["toc"])
    landing = (_DOCS_SOURCE / "scanpy_and_seurat.md").read_text()

    assert (_DOCS_SOURCE / "scanpy.md").is_file()
    assert (_DOCS_SOURCE / "seurat.md").is_file()
    assert "(scanpy_and_seurat)=" in landing
    assert "[](scanpy.md)" in landing
    assert "[](seurat.md)" in landing
    assert "scanpy.md" in toctree
    assert "seurat.md" in toctree
    assert "scanpy_and_seurat.md" in toctree


def test_agent_pages_are_withheld_and_their_sources_are_retained() -> None:
    project = _project()
    assert "llms.txt" in project["static_files"]
    assert not _AGENT_PAGES & _toc_files(project["toc"])
    for page in _AGENT_PAGES:
        assert (_DOCS_SOURCE / page).is_file()
        assert any(fnmatch(page, pattern) for pattern in project["exclude"]), page


def test_public_documentation_does_not_link_to_withheld_agent_pages() -> None:
    targets = {
        page.removesuffix(".md")
        for page in _AGENT_PAGES
        if not page.startswith("reference/api/")
    }
    targets.update(target.replace("_", "-") for target in list(targets))
    targets.add("api/agent.html")
    sources = [
        path
        for path in _DOCS_SOURCE.rglob("*")
        if path.suffix in {".md", ".ipynb"}
        and path.relative_to(_DOCS_SOURCE).as_posix() not in _AGENT_PAGES
    ]
    sources.extend([_REPOSITORY_ROOT / "README.md", _DOCS_SOURCE / "llms.txt"])

    for source in sources:
        contents = _source_text(source)
        for target in targets:
            assert target not in contents, (source, target)

    api_index = (_DOCS_SOURCE / "reference" / "api" / "index.md").read_text()
    assert "agent" not in api_index.splitlines()


def test_installation_uses_one_environment_for_install_and_runtime() -> None:
    contents = (_DOCS_SOURCE / "installation.md").read_text()

    assert "uv venv --python 3.12" in contents
    # Importing DataStore loads the modules that a bare import cytearc defers.
    assert (
        'python -c "from cytearc import DataStore; import cytearc; '
        'print(cytearc.__version__)"' in contents
    )
    assert 'python -c "import cytearc; print' not in contents
    assert "uv pip install jupyterlab\njupyter lab" in contents
    assert "uv run jupyter lab" not in contents
    assert "pywin32" not in contents
    assert contents.index("sudo apt install python3-dev python3-venv") < contents.index(
        "python -m venv .venv"
    )
    assert "## Next steps" in contents


def test_installation_links_match_the_published_route() -> None:
    metadata = tomllib.loads((_REPOSITORY_ROOT / "pyproject.toml").read_text())

    assert (
        metadata["project"]["urls"]["Installation"]
        .rstrip("/")
        .endswith("/installation")
    )
    assert (
        "https://nygenanalytics.github.io/CyteArc/installation"
        in (_REPOSITORY_ROOT / "README.md").read_text()
    )


def test_trajectory_tutorials_use_source_sink_sign_convention() -> None:
    tutorials = (
        "pseudotime.ipynb",
        "expression_dynamics.ipynb",
        "fate_mapping.ipynb",
    )

    for tutorial in tutorials:
        contents = _source_text(_DOCS_SOURCE / "tutorials" / tutorial)
        assert "source_sink_vector[source] = -1.0 / source.sum()" in contents
        assert "source_sink_vector[sink] = 1.0 / sink.sum()" in contents
