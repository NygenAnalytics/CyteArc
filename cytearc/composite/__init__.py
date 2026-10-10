from typing import TYPE_CHECKING

from .._facade import lazy_facade as _lazy_facade

if TYPE_CHECKING:
    from .create import create_composite as create_composite

__all__ = ["create_composite"]

__getattr__, __dir__ = _lazy_facade(
    __name__, {"create_composite": ".create"}, set_module=True
)
