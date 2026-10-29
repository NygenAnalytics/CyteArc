"""The names of the groups of cells that repertoire summaries count within.

A group is a combination of labels, one in each key column, such as a sample
and a receptor type. Tables and figures name it by its labels joined by
commas. These functions work on labels and never touch storage.
"""

from collections.abc import Sequence

__all__ = ["group_names"]


def group_names(group_keys: Sequence[dict[str, str]]) -> list[str]:
    """Name each group by its labels joined by commas, or ``"all cells"``.

    Raises:
        ValueError: If two groups get one name, as labels that hold a comma
            can give them.
    """
    names = [", ".join(key.values()) if key else "all cells" for key in group_keys]
    first: dict[str, dict[str, str]] = {}
    for key, name in zip(group_keys, names, strict=True):
        if name in first and first[name] != key:
            raise ValueError(
                f"The groups {first[name]} and {key} are both named {name!r}; "
                "remove the commas from their labels"
            )
        first.setdefault(name, key)
    return names
