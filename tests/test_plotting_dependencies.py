"""Optional plotting dependency loader tests."""

import builtins

import pytest

from cytearc.plotting._deps import (
    require_kneed,
    require_matplotlib,
    require_seaborn,
)


def test_optional_dependency_loaders_return_the_installed_objects():
    import kneed
    import matplotlib
    import matplotlib.pyplot
    import seaborn

    plt, mpl = require_matplotlib()

    assert plt is matplotlib.pyplot
    assert mpl is matplotlib
    assert require_seaborn() is seaborn
    assert require_kneed() is kneed.KneeLocator


@pytest.mark.parametrize(
    ("loader", "blocked_package", "message"),
    [
        (
            require_matplotlib,
            "matplotlib",
            "CyteArc plotting requires matplotlib. "
            "Install with: pip install 'cytearc[extra]'",
        ),
        (
            require_seaborn,
            "seaborn",
            "CyteArc plotting requires seaborn. Install with: pip install 'cytearc[extra]'",
        ),
        (
            require_kneed,
            "kneed",
            "CyteArc elbow detection requires kneed. "
            "Install with: pip install 'cytearc[extra]'",
        ),
    ],
)
def test_optional_dependency_errors_are_actionable(
    monkeypatch,
    loader,
    blocked_package,
    message,
):
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.split(".", 1)[0] == blocked_package:
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError) as raised:
        loader()

    assert str(raised.value) == message
    assert isinstance(raised.value.__cause__, ModuleNotFoundError)
    assert str(raised.value.__cause__).split(".", 1)[0] == blocked_package
