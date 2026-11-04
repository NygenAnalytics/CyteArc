"""Warnings that CyteArc issues to the code that calls it."""

import os
import warnings
from typing import Any

from .logging import note_warning

# The directory of the ``cytearc`` package. The trailing separator keeps a
# sibling directory whose name starts with ``cytearc`` outside the prefix.
_PACKAGE_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "")


class DiagnosticWarning(UserWarning):
    """A warning that an operation gives from a diagnostic it saved with its artifact.

    An operation gives the warning after it computes its artifact and again
    whenever it reuses it, from the diagnostics the artifact holds. Filter
    this category to leave CyteArc's own warnings out, and read the numbers
    from ``ds.artifacts.inspect(ref).diagnostics`` to apply other thresholds.

    Attributes:
        operation: The operation that saved the artifact, such as
            ``"import_clones"``.
        diagnostic: The diagnostic that the warning reports, such as
            ``"unmatchedBarcodes"``.
        value: The diagnostic's saved value.
    """

    def __init__(
        self, message: str, operation: str, diagnostic: str, value: Any
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.diagnostic = diagnostic
        self.value = value

    def __str__(self) -> str:
        return str(self.args[0])

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (
            type(self),
            (self.args[0], self.operation, self.diagnostic, self.value),
        )


def warn(message: str | Warning, category: type[Warning] = UserWarning) -> None:
    """Issue a warning that points at the first frame outside the package.

    ``message`` is text, issued in ``category``, or a warning instance, such as
    a :class:`DiagnosticWarning`, issued in its own class. A caller that
    collects warnings, such as a pipeline stage, also records it.
    """
    note_warning(str(message))
    if isinstance(message, Warning):
        warnings.warn(message, skip_file_prefixes=(_PACKAGE_DIRECTORY,))
    else:
        warnings.warn(message, category, skip_file_prefixes=(_PACKAGE_DIRECTORY,))
