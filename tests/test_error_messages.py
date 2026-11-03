"""The clone stack's error messages quote values cut to 80 characters.

A message that repeats a value, such as a barcode, a label, a path or a
contig ID, quotes it through ``quoted_value``, which keeps the start and the
end of a value of thousands of characters, so the message stays readable. A
guard refuses every ``!r`` and ``repr()`` of the stack's modules outside a
``__repr__`` method, since a message may be built anywhere before it is
raised.
"""

import ast

import pandas as pd
import pytest

from cytearc.utils.arguments import QUOTED_LIMIT, quoted_value
from cytearc.writers.airr import write_airr_rearrangements
from tests.clone_helpers import TCR_ROWS, receptor_store, tcr_file
from tests.stack_modules import CYTEARC_ROOT, stack_module_paths

# A value cut to the limit keeps this many characters at its start.
HEAD = (QUOTED_LIMIT - 3) // 2


def _plain_quotes(source: str) -> list[int]:
    """The lines of the ``!r`` fields and ``repr()`` calls outside ``__repr__``."""
    tree = ast.parse(source)
    allowed = {
        id(node)
        for function in ast.walk(tree)
        if isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef)
        and function.name == "__repr__"
        for node in ast.walk(function)
    }
    lines = []
    for node in ast.walk(tree):
        if id(node) in allowed:
            continue
        quoted = isinstance(node, ast.FormattedValue) and node.conversion == ord("r")
        called = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "repr"
        )
        if quoted or called:
            lines.append(node.lineno)
    return sorted(set(lines))


def test_stack_modules_quote_values_through_quoted_value() -> None:
    found = [
        f"{path.relative_to(CYTEARC_ROOT).as_posix()}:{line}"
        for path in stack_module_paths()
        for line in _plain_quotes(path.read_text())
    ]
    assert not found, (
        "These lines quote a value with !r or repr(), which repeats a long "
        f"value in full; quote it with quoted_value: {found}"
    )


def test_the_guard_finds_every_quote_outside_a_repr_method() -> None:
    source = "\n".join(
        [
            'raise ValueError(f"site {site!r} repeats")',
            'warn(f"{label!r} is ambiguous")',
            'message = f"{site!r}"',
            'shown = ", ".join(repr(name) for name in names)',
            'raise ValueError(f"site {quoted_value(site)} repeats")',
            "class Record:",
            "    def __repr__(self):",
            '        return f"Record({self.name!r}, {repr(self.size)})"',
            'raise TypeError(\n    "x " f"{name!r}"\n)',
        ]
    )
    assert _plain_quotes(source) == [1, 2, 3, 4, 10]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("CASS", "'CASS'"),
        (310, "310"),
        (None, "None"),
        ("A" * (QUOTED_LIMIT - 2), "'" + "A" * (QUOTED_LIMIT - 2) + "'"),
        (
            "A" * (QUOTED_LIMIT - 1),
            "'" + "A" * (HEAD - 1) + "..." + "A" * (QUOTED_LIMIT - HEAD - 4) + "'",
        ),
    ],
)
def test_quoted_value_keeps_short_values_and_cuts_long_ones(value, expected) -> None:
    found = quoted_value(value)
    assert found == expected
    assert len(found) <= QUOTED_LIMIT == 80


def test_a_long_path_keeps_its_file_name() -> None:
    folder = "/data/" + "experiment_2026_lineage_" * 6
    first = quoted_value(f"{folder}/sample_1/clones.csv")
    second = quoted_value(f"{folder}/sample_2/clones.csv")
    assert len(first) == len(second) == QUOTED_LIMIT
    assert first.endswith("sample_1/clones.csv'") and first != second


def test_a_repeated_contig_id_of_20000_characters_is_quoted_in_80(tmp_path) -> None:
    store = receptor_store(tmp_path)
    long_id = "c" * 20_000
    rows = [row.replace("cell0_contig_1", long_id) for row in TCR_ROWS[:1]] * 2
    with pytest.raises(ValueError) as raised:
        store.repertoire.import_receptors(
            tcr_file(tmp_path, rows), cell_selection=store.snapshot_cell_selection()
        )
    message = str(raised.value)
    shown = "'" + "c" * (HEAD - 1) + "..." + "c" * (QUOTED_LIMIT - HEAD - 4) + "'"
    assert message.startswith(f"Contig ID {shown} repeats")
    assert len(message) < 300


def test_the_airr_writer_quotes_a_contig_id_of_20000_characters_in_80(
    tmp_path,
) -> None:
    # A tab in a field cannot be written, and the refusal names its contig.
    table = pd.DataFrame(
        {
            "sequence_id": ["c" * 20_000, "short"],
            "v_call": ["TRBV1\tTRBV2", "TRBV3\tTRBV4"],
            "junction_aa": ["CASS", "CASR"],
        }
    )
    with pytest.raises(ValueError) as raised:
        write_airr_rearrangements(table, tmp_path / "out.tsv")
    message = str(raised.value)
    assert "in contig 'ccc" in message and "..." in message
    assert len(message) < 400
    assert not (tmp_path / "out.tsv").exists()
