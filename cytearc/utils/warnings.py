"""Warnings that CyteArc issues to the code that calls it."""

import os
import warnings

# The directory of the ``cytearc`` package. The trailing separator keeps a
# sibling directory whose name starts with ``cytearc`` outside the prefix.
_PACKAGE_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "")


def warn(message: str, category: type[Warning] = UserWarning) -> None:
    """Issue a warning that points at the first frame outside the package."""
    warnings.warn(message, category, skip_file_prefixes=(_PACKAGE_DIRECTORY,))
