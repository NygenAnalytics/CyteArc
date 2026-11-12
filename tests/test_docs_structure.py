import json
import re
import tomllib
from fnmatch import fnmatch
from pathlib import Path

import pytest
import yaml


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DOCS_SOURCE = _REPOSITORY_ROOT / "docs" / "source"
_MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
_DOI_LINK = re.compile(
    r"""(?:https?://(?:dx\.)?doi\.org/|\bdoi:)(10\.\d{4,9}/[^\s)<>\]"']+)""",
    re.IGNORECASE,
)
_AGENT_PAGES = {"analysis_with_agents.md"}


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


def _pages() -> list[Path]:
    return sorted(
        path for path in _DOCS_SOURCE.rglob("*") if path.suffix in {".md", ".ipynb"}
    )


def _prose(path: Path) -> str:
    """A page's text outside code: Markdown, or a notebook's Markdown cells."""
    if path.suffix != ".ipynb":
        text = path.read_text()
    else:
        cells = json.loads(path.read_text())["cells"]
        text = "\n".join(
            "".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown"
        )
    return re.sub(r"```.*?```|`[^`\n]*`", "", text, flags=re.DOTALL)


def _anchors(path: Path) -> set[str]:
    """The anchors of a page: its labels, and the words of each heading without one."""
    text = _prose(path)
    labels = set(re.findall(r"^\(([^)\s]+)\)=\s*$", text, re.MULTILINE))
    headings = re.findall(
        r"^(\([^)\s]+\)=\s*\n(?:\s*\n)?)?#+ (.+)$", text, re.MULTILINE
    )
    return labels | {
        re.sub(r"[^\w\s-]", "", heading).strip().lower().replace(" ", "-")
        for labelled, heading in headings
        if not labelled
    }


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


def test_markdown_pages_open_with_their_own_title() -> None:
    # The discovery export reads a page from its first h1, and MyST renders an
    # h1 only for a heading in the page itself, not for one an include brings in.
    untitled = []
    for page in sorted(_toc_files(_project()["toc"])):
        if not page.endswith(".md"):
            continue
        body = re.sub(
            r"\A---\n.*?\n---\n", "", (_DOCS_SOURCE / page).read_text(), flags=re.S
        )
        # MyST targets such as "(installation)=" may precede the heading.
        lines = [line for line in body.splitlines() if line.strip()]
        first = next((line for line in lines if not re.fullmatch(r"\(.+\)=", line)), "")
        if not first.startswith("# "):
            untitled.append(page)

    assert untitled == []


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
        "tutorials/mitochondrial_lineage_tracing.ipynb",
        "tutorials/cite_seq.ipynb",
        "tutorials/tea_seq.ipynb",
        "tutorials/multimodal_diagnostics.ipynb",
        "tutorials/hto_demultiplexing.ipynb",
        "tutorials/immune_repertoires.ipynb",
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
        assert tutorial.relative_to(_DOCS_SOURCE).as_posix() in published


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
    assert "../build/api" not in project["static_files"]
    assert not _AGENT_PAGES & _toc_files(project["toc"])
    for page in _AGENT_PAGES:
        assert (_DOCS_SOURCE / page).is_file()
        assert any(fnmatch(page, pattern) for pattern in project["exclude"]), page


def test_public_documentation_does_not_link_to_withheld_agent_pages() -> None:
    targets = {page.removesuffix(".md") for page in _AGENT_PAGES}
    targets.update(target.replace("_", "-") for target in list(targets))
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


def test_installation_uses_one_environment_for_install_and_runtime() -> None:
    contents = (_DOCS_SOURCE / "installation.md").read_text()

    assert "uv venv --python 3.13" in contents
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
        "https://docs.nygen.io/CyteArc/installation"
        in (_REPOSITORY_ROOT / "README.md").read_text()
    )


def test_trajectory_tutorials_use_source_sink_sign_convention() -> None:
    tutorials = (
        "pseudotime.ipynb",
        "expression_dynamics.ipynb",
        "fate_mapping.ipynb",
        "lineage_dynamics.ipynb",
        "lineage_curves.ipynb",
        "lineage_tracing.ipynb",
    )

    for tutorial in tutorials:
        contents = _source_text(_DOCS_SOURCE / "tutorials" / tutorial)
        assert "source_sink_vector[source] = -1.0 / source.sum()" in contents
        assert "source_sink_vector[sink] = 1.0 / sink.sum()" in contents


def test_every_linked_doi_has_a_local_citation() -> None:
    # MyST looks up each DOI that references.bib lacks on doi.org during every
    # build, and a strict build fails whenever one lookup does.
    bibliography = (_DOCS_SOURCE / "references.bib").read_text()
    recorded = {
        doi.lower()
        for doi in re.findall(
            r"^\s*doi\s*=\s*\{([^}]+)\}", bibliography, re.IGNORECASE | re.MULTILINE
        )
    }
    excluded = _project()["exclude"]
    linked = {
        match.group(1).rstrip(".,;").lower(): page
        for path in sorted(_DOCS_SOURCE.rglob("*"))
        if path.suffix in {".md", ".ipynb"}
        and not any(
            fnmatch(page := path.relative_to(_DOCS_SOURCE).as_posix(), pattern)
            for pattern in excluded
        )
        for match in _DOI_LINK.finditer(_source_text(path))
    }
    missing = {doi: page for doi, page in linked.items() if doi not in recorded}

    assert linked
    assert not missing, (
        "Add the BibTeX of each DOI to docs/source/references.bib, for example "
        "from `curl -L -H 'Accept: application/x-bibtex' https://doi.org/<doi>`: "
        f"{missing}"
    )


def test_the_notices_open_and_close_each_code_fence_on_a_line_of_its_own() -> None:
    # A licence whose last line lacks a line break once swallowed the fence
    # that closes it, and the notices after it rendered as its code.
    notices = (_REPOSITORY_ROOT / "THIRD_PARTY_NOTICES.md").read_text()
    lines = notices.splitlines()
    inline = [
        number for number, line in enumerate(lines, 1) if "```" in line.strip()[3:]
    ]
    fences = [line for line in lines if line.startswith("```")]
    assert not inline, f"Fences that share a line with text, on lines {inline}"
    assert len(fences) % 2 == 0
    assert all(fence == "```" for fence in fences[1::2])


def test_pages_hold_no_em_dashes_or_html_comments() -> None:
    # The site publishes each page's source, so a comment becomes public.
    found = [
        (path.relative_to(_DOCS_SOURCE).as_posix(), mark)
        for path in _pages()
        for mark in ("\u2014", "<!--")
        if mark in _source_text(path)
    ]
    assert not found


_NUMBERED_HEADING = re.compile(
    r"^#{1,6}[ \t]+(?:\d+(?:\.\d+)*[.)]|\d+(?:\.\d+)+|0\d+)(?:[ \t].*)?$",
    re.MULTILINE,
)


def test_numbered_headings_are_told_apart_from_headings_that_open_with_a_number() -> (
    None
):
    numbered = [
        "## 1. Choose datasets",
        "### 2) Select",
        "## 1.2 Topic",
        "## 3.1. Topic",
    ]
    numbered.append("## 01 Introduction")
    named = ["## 10x Genomics libraries", "## 3' RNA counts", "## 2024 releases"]
    named.extend(["## 1.5x faster reads", "## Choose datasets", "    ## 1. In code"])
    assert [bool(_NUMBERED_HEADING.search(text)) for text in numbered] == [True] * 5
    assert [bool(_NUMBERED_HEADING.search(text)) for text in named] == [False] * 6


def test_headings_name_their_sections_without_numbers() -> None:
    # The order of a page already numbers its sections, and a number becomes
    # the start of the heading's anchor.
    found = [
        (path.relative_to(_DOCS_SOURCE).as_posix(), heading.group(0))
        for path in _pages()
        for heading in _NUMBERED_HEADING.finditer(_prose(path))
    ]
    assert not found


def test_every_citation_names_an_entry_of_the_bibliography() -> None:
    # A citation of a removed entry renders as its key, and MyST only warns.
    bibliography = (_DOCS_SOURCE / "references.bib").read_text()
    entries = set(re.findall(r"^@\w+\{([^,\s]+),", bibliography, re.MULTILINE))
    cited = {
        key: path.relative_to(_DOCS_SOURCE).as_posix()
        for path in _pages()
        for key in re.findall(r"(?<![\w@./])@([A-Za-z][\w:-]*\w)", _prose(path))
    }
    assert cited
    assert {key: page for key, page in cited.items() if key not in entries} == {}


def test_every_figure_of_a_built_notebook_has_a_description() -> None:
    # The build gives each figure its description as alt text, and refuses a
    # figure without one; the description lives in the cell's metadata.
    excluded = _project()["exclude"]
    undescribed = []
    for path in _pages():
        page = path.relative_to(_DOCS_SOURCE).as_posix()
        if path.suffix != ".ipynb" or any(fnmatch(page, rule) for rule in excluded):
            continue
        for cell in json.loads(path.read_text())["cells"]:
            outputs = cell.get("outputs", [])
            figures = [
                o
                for o in outputs
                if any(k.startswith("image/") for k in o.get("data", {}))
            ]
            texts = cell["metadata"].get("figure_descriptions", [])
            if len(figures) != len(texts) or not all(str(t).strip() for t in texts):
                undescribed.append((page, cell.get("id")))
    assert not undescribed


def test_committed_notebooks_match_the_code_they_executed() -> None:
    # The build refuses a notebook whose code changed after it ran. Checking
    # here finds such a notebook without a full build, excluded pages included.
    pytest.importorskip("jupytext")
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("yaml")
    from docs.build_docs import validate_execution

    checked = 0
    for path in _pages():
        if path.suffix != ".ipynb":
            continue
        notebook = nbformat.read(path, as_version=4)
        if "executed_code_sha256" not in notebook.metadata:
            continue
        validate_execution(notebook, path.relative_to(_DOCS_SOURCE))
        checked += 1
    assert checked


def test_api_pages_link_labels_through_the_ref_role() -> None:
    # Sphinx renders a Markdown link to a bare label, such as
    # [text](cell_aligned_kinds), as a link to a missing page.
    bare = re.compile(r"\[[^\]]+\]\(([^)\s/.#:]+)\)")
    found = [
        (path.name, target)
        for path in sorted((_DOCS_SOURCE / "reference" / "api").glob("*.md"))
        for target in bare.findall(_prose(path))
    ]
    assert not found


def test_links_to_a_section_of_another_page_find_it() -> None:
    # A label above a heading takes the place of the heading's words as its
    # anchor, so a link to the words finds no target, as the build warns.
    link = re.compile(r"\]\(([^)\s#:]+\.(?:md|ipynb))#([^)\s]+)\)")
    missing = [
        (path.relative_to(_DOCS_SOURCE).as_posix(), f"{file}#{anchor}")
        for path in _pages()
        for file, anchor in link.findall(_prose(path))
        if anchor not in _anchors((path.parent / file).resolve())
    ]
    assert not missing


def test_links_within_a_built_page_name_a_label() -> None:
    # MyST warns of a link to a heading's words on its own page, and a label
    # names the heading whatever its words become.
    excluded = _project()["exclude"]
    unlabelled = []
    for path in _pages():
        page = path.relative_to(_DOCS_SOURCE).as_posix()
        if any(fnmatch(page, rule) for rule in excluded):
            continue
        text = _prose(path)
        labels = set(re.findall(r"^\(([^)\s]+)\)=\s*$", text, re.MULTILINE))
        unlabelled += [
            (page, anchor)
            for anchor in re.findall(r"\]\(#([^)\s]+)\)", text)
            if anchor not in labels
        ]
    assert not unlabelled


def test_labels_of_built_pages_stay_apart_once_normalised() -> None:
    # MyST writes a label's underscores as hyphens, so (a_b)= beside (a-b)= is
    # one label twice, which the build warns of.
    excluded = _project()["exclude"]
    labels = [
        label.lower().replace("_", "-")
        for path in _pages()
        if not any(
            fnmatch(path.relative_to(_DOCS_SOURCE).as_posix(), rule)
            for rule in excluded
        )
        for label in re.findall(r"^\(([^)\s]+)\)=\s*$", _prose(path), re.MULTILINE)
    ]
    assert len(labels) == len(set(labels))
