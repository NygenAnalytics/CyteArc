"""Render current documentation with saved notebook outputs, without execution."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from fnmatch import fnmatchcase
from pathlib import Path

import jupytext
import nbformat
import yaml

DOCS = Path(__file__).resolve().parent
SOURCE = DOCS / "source"
SITE = DOCS / "build" / "site"


def code_fingerprint(notebook: nbformat.NotebookNode) -> str:
    sources = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    encoded = json.dumps(sources, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_execution(notebook: nbformat.NotebookNode, page: Path) -> None:
    nbformat.validate(notebook)
    if not notebook.metadata.get("executed_at"):
        raise ValueError(f"Notebook {page} has no completed execution timestamp")
    fingerprint = notebook.metadata.get("executed_code_sha256")
    if not fingerprint:
        raise ValueError(f"Notebook {page} has no executed-code fingerprint")
    if fingerprint != code_fingerprint(notebook):
        raise ValueError(
            f"Code changed in {page} since execution. Rerun:\n"
            f'make -C docs execute-docs-modal PAGES="{page.with_suffix("")}"'
        )
    for index, cell in enumerate(notebook.cells, start=1):
        if cell.cell_type != "code":
            continue
        if any(output.output_type == "error" for output in cell.get("outputs", [])):
            raise ValueError(f"Notebook {page} has an error in cell {index}")
        count = cell.get("execution_count")
        if cell.source.strip() and (type(count) is not int or count < 1):
            raise ValueError(f"Notebook {page} has an unexecuted cell {index}")


def rewrite_includes(text: str, page: Path) -> str:
    def include_path(match: re.Match[str]) -> str:
        original = (SOURCE / page.parent / match[2]).resolve()
        return f"{match[1]}{original.relative_to(SITE / page.parent, walk_up=True)}"

    return re.sub(
        r"(?m)^((?:`{3,}|:{3,})\{include\}[ \t]+)(\S+)",
        include_path,
        text,
    )


def prepare_site(*, preview: bool = False) -> Path:
    config = yaml.safe_load((SOURCE / "myst.yml").read_text())
    excludes = config["project"].get("exclude", [])

    def excluded(path: Path) -> bool:
        relative = path.as_posix()
        return any(
            fnmatchcase(relative, pattern)
            or (pattern.startswith("**/") and fnmatchcase(relative, pattern[3:]))
            for pattern in excludes
        )

    def ignore(directory: str, names: list[str]) -> set[str]:
        relative = Path(directory).relative_to(SOURCE)
        return shutil.ignore_patterns(
            "_build", "cytearc_datasets", "__pycache__", ".ipynb_checkpoints"
        )(directory, names) | {name for name in names if excluded(relative / name)}

    if SITE.exists():
        shutil.rmtree(SITE)
    shutil.copytree(
        SOURCE,
        SITE,
        ignore=ignore,
    )
    pages = {
        path.relative_to(SITE): (
            nbformat.read(path, as_version=4)
            if path.suffix == ".ipynb"
            else jupytext.read(path, fmt="md:myst")
        )
        for path in sorted(SITE.rglob("*"))
        if path.suffix in {".md", ".ipynb"}
        and "_build" not in path.relative_to(SITE).parts
    }
    executable = {
        page
        for page, notebook in pages.items()
        if any(cell.cell_type == "code" for cell in notebook.cells)
    }
    for page in executable:
        if page.suffix != ".ipynb":
            raise ValueError(
                f"Executable Markdown page {page} must be an .ipynb notebook. "
                "Keep its code and outputs in one source file."
            )
        if not pages[page].metadata.get("kernelspec"):
            raise ValueError(
                f"Executable page {page} has no kernelspec metadata. "
                "Add the kernel configuration for its execution environment "
                "before building."
            )
    missing = []
    github = config["project"]["github"].rstrip("/")
    for page, notebook in pages.items():
        links = {
            key: f"{github}/{action}/master/docs/source/{page.as_posix()}"
            for key, action in (("source_url", "blob"), ("edit_url", "edit"))
        }
        staged = SITE / page
        if page.suffix == ".ipynb":
            nbformat.validate(notebook)
            code = [cell for cell in notebook.cells if cell.cell_type == "code"]
            pristine = (
                "executed_at" not in notebook.metadata
                and "executed_code_sha256" not in notebook.metadata
                and all(
                    cell.execution_count is None and not cell.outputs for cell in code
                )
            )
            if code and pristine:
                missing.append(page.as_posix())
                if preview:
                    notebook.cells.insert(
                        0,
                        nbformat.v4.new_markdown_cell(
                            ":::{warning} Source preview\n"
                            "This page has not been executed. Figures and results "
                            "are not available.\n:::"
                        ),
                    )
            elif code or "executed_code_sha256" in notebook.metadata:
                validate_execution(notebook, page)
            for cell in notebook.cells:
                if cell.cell_type == "markdown":
                    cell.source = rewrite_includes(cell.source, page)
            notebook.metadata.pop("jupytext", None)
            notebook.metadata.update(links)
            nbformat.write(notebook, staged)
        else:
            text = rewrite_includes(staged.read_text(), page)
            frontmatter = re.match(r"\A---\n(.*?)\n---\n", text, re.DOTALL)
            metadata = yaml.safe_load(frontmatter[1]) or {} if frontmatter else {}
            metadata.update(links)
            body = text[frontmatter.end() :] if frontmatter else text
            staged.write_text("---\n" + yaml.safe_dump(metadata) + "---\n" + body)

    base_url = os.environ.get("BASE_URL", "").rstrip("/")

    def update_toc(entries: list[dict]) -> None:
        entries[:] = [
            entry
            for entry in entries
            if "file" not in entry or not excluded(Path(entry["file"]))
        ]
        for entry in entries:
            url = entry.get("url", "")
            if url.startswith("/") and not url.startswith("//"):
                entry["url"] = f"{base_url}{url}"
            update_toc(entry.get("children", []))

    update_toc(config["project"]["toc"])
    config["project"]["static_files"] = [
        str((SOURCE / path).resolve())
        for path in config["project"].get("static_files", [])
    ]
    (SITE / "myst.yml").write_text(yaml.safe_dump(config, sort_keys=False))
    if missing:
        if not preview:
            raise ValueError(
                "Saved execution is missing for:\n  "
                + "\n  ".join(missing)
                + "\nExecute these pages before building HTML, or use --preview "
                "for a labelled source preview."
            )
        print(
            "Unexecuted notebooks:\n  " + "\n  ".join(missing),
            flush=True,
        )
    return SITE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Allow missing executions and label those pages as source previews",
    )
    args = parser.parse_args()
    subprocess.run(
        ["myst", "build", "--html", "--strict"],
        cwd=prepare_site(preview=args.preview),
        check=True,
    )


if __name__ == "__main__":
    main()
