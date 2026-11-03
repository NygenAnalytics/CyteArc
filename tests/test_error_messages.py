"""The clone stack's error messages quote values cut to 80 characters.

A message that repeats a value, such as a barcode, a label or a contig ID,
quotes it through ``quoted_value``, so a value of thousands of characters
leaves the message readable. A guard refuses a plain ``!r`` in the stack's
raised errors and warnings.
"""

import ast

import pytest

from cytearc.utils.arguments import QUOTED_LIMIT, quoted_value
from tests.clone_helpers import TCR_ROWS, receptor_store, tcr_file
from tests.stack_modules import CYTEARC_ROOT, stack_module_paths


def _plain_quotes(source: str) -> list[int]:
    """The lines of ``!r`` fields in the errors raised and warnings issued."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        parts = []
        if isinstance(node, ast.Raise) and node.exc is not None:
            parts.append(node.exc)
        elif isinstance(node, ast.Call):
            called = node.func
            name = (
                called.id
                if isinstance(called, ast.Name)
                else getattr(called, "attr", "")
            )
            if name == "warn":
                parts.extend(node.args)
        for part in parts:
            lines.extend(
                field.lineno
                for field in ast.walk(part)
                if isinstance(field, ast.FormattedValue)
                and field.conversion == ord("r")
            )
    return sorted(set(lines))


def test_stack_messages_quote_values_through_quoted_value() -> None:
    found = [
        f"{path.relative_to(CYTEARC_ROOT).as_posix()}:{line}"
        for path in stack_module_paths()
        for line in _plain_quotes(path.read_text())
    ]
    assert not found, (
        "These messages quote a value with !r, which repeats a long value in "
        f"full; quote it with quoted_value: {found}"
    )


def test_the_guard_finds_quotes_in_raised_errors_and_warnings_alone() -> None:
    source = "\n".join(
        [
            'raise ValueError(f"site {site!r} repeats")',
            'warn(f"{label!r} is ambiguous")',
            'warnings.warn(f"{value!r}")',
            'raise ValueError(f"site {quoted_value(site)} repeats")',
            'message = f"{site!r}"',
            'raise TypeError(\n    "x " f"{name!r}"\n)',
        ]
    )
    assert _plain_quotes(source) == [1, 2, 3, 7]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("CASS", "'CASS'"),
        (310, "310"),
        (None, "None"),
        ("A" * (QUOTED_LIMIT - 2), "'" + "A" * (QUOTED_LIMIT - 2) + "'"),
        ("A" * (QUOTED_LIMIT - 1), "'" + "A" * (QUOTED_LIMIT - 4) + "..."),
    ],
)
def test_quoted_value_keeps_short_values_and_cuts_long_ones(value, expected) -> None:
    found = quoted_value(value)
    assert found == expected
    assert len(found) <= QUOTED_LIMIT == 80


def test_a_repeated_contig_id_of_20000_characters_is_quoted_in_80(tmp_path) -> None:
    store = receptor_store(tmp_path)
    long_id = "c" * 20_000
    rows = [row.replace("cell0_contig_1", long_id) for row in TCR_ROWS[:1]] * 2
    with pytest.raises(ValueError) as raised:
        store.repertoire.import_receptors(
            tcr_file(tmp_path, rows), cell_selection=store.snapshot_cell_selection()
        )
    message = str(raised.value)
    assert message.startswith("Contig ID 'ccc")
    assert "c" * (QUOTED_LIMIT - 4) + "..." in message
    assert len(message) < 300
