import math
from numbers import Real

import numpy as np


def float_argument(value: object, name: str) -> float:
    """Return ``value`` as a finite Python float.

    Python and NumPy integers and floats are accepted, so ``1`` and ``1.0``
    give the same value, and booleans are rejected.

    Args:
        value: The argument to validate.
        name: Argument name used in error messages.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` is not finite.
    """
    if isinstance(value, bool | np.bool_) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved


def bool_argument(value: object, name: str) -> bool:
    """Return ``value`` as a Python boolean.

    Python and NumPy booleans are accepted, so ``np.True_`` gives ``True``,
    and integers and text are rejected.

    Args:
        value: The argument to validate.
        name: Argument name used in error messages.

    Raises:
        TypeError: If ``value`` is not a boolean.
    """
    if not isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def optional_bool_argument(value: object, name: str) -> bool | None:
    """Return ``value`` as a Python boolean, or None.

    Accepts what :func:`bool_argument` accepts, and None.

    Args:
        value: The argument to validate.
        name: Argument name used in error messages.

    Raises:
        TypeError: If ``value`` is neither None nor a boolean.
    """
    if value is None:
        return None
    if not isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be a boolean or None")
    return bool(value)


#: The most characters of a value that an error message quotes.
QUOTED_LIMIT = 80


def quoted_value(value: object) -> str:
    """Return ``repr(value)`` for an error message, cut to ``QUOTED_LIMIT``.

    A value longer than that, such as a contig ID of 20,000 characters, keeps
    its start and its end around ``...``, so the message stays readable and a
    long path still shows its file's name.
    """
    text = repr(value)
    if len(text) <= QUOTED_LIMIT:
        return text
    head = (QUOTED_LIMIT - 3) // 2
    return text[:head] + "..." + text[head + 3 - QUOTED_LIMIT :]


def optional_name_argument(
    value: object, name: str, *, noun: str = "a non-empty string"
) -> str | None:
    """Return ``value``, a name such as a column's, or None.

    Args:
        value: The argument to validate.
        name: Argument name used in error messages.
        noun: What the name must be, for the error message, such as
            ``"a feature column name"``.

    Raises:
        TypeError: If ``value`` is neither None nor a non-empty string.
    """
    if value is not None and (not isinstance(value, str) or not value):
        raise TypeError(f"{name} must be {noun} or None")
    return value


def clip_fraction_argument(value: object, name: str = "clip_fraction") -> float:
    """Return a two-sided clipping fraction as a Python float."""
    resolved = float_argument(value, name)
    if not 0.0 <= resolved < 0.5:
        raise ValueError(f"{name} must be at least 0 and less than 0.5")
    return resolved


def integer_argument(
    value: object,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return ``value`` as a Python integer within inclusive bounds.

    Python and NumPy integers are accepted and booleans are rejected.

    Args:
        value: The argument to validate.
        name: Argument name used in error messages.
        minimum: Smallest accepted value, if any.
        maximum: Largest accepted value, if any.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If ``value`` lies outside the bounds.
    """
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an integer")
    resolved = int(value)
    if minimum is not None and resolved < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and resolved > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return resolved
