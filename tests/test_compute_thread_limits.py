import pytest
from threadpoolctl import ThreadpoolController, threadpool_limits

from cytearc.utils.compute import (
    enter_thread_limit,
    exit_thread_limit,
    process_thread_limit,
)


def _blas_threads() -> set[int]:
    return {
        lib.num_threads
        for lib in ThreadpoolController().lib_controllers
        if lib.user_api == "blas"
    }


@pytest.fixture
def two_blas_threads():
    # The test session pins BLAS to one thread; raise it so a change shows.
    with threadpool_limits(limits=2, user_api="blas"):
        if not _blas_threads() or max(_blas_threads()) < 2:
            pytest.skip("needs a multi-threaded BLAS")
        yield


def test_the_latest_remaining_limit_applies_after_a_middle_exit(two_blas_threads):
    first = enter_thread_limit(1)
    second = enter_thread_limit(2)
    third = enter_thread_limit(1)
    exit_thread_limit(third)
    assert _blas_threads() == {2}
    exit_thread_limit(second)
    assert _blas_threads() == {1}
    exit_thread_limit(first)
    assert _blas_threads() == {2}


def test_a_scope_without_a_limit_still_restores_the_original(two_blas_threads):
    unchanged = enter_thread_limit(None)
    assert _blas_threads() == {2}
    limited = enter_thread_limit(1)
    exit_thread_limit(limited)
    # Only the unchanged scope remains, so the original limits apply.
    assert _blas_threads() == {2}
    exit_thread_limit(unchanged)
    assert _blas_threads() == {2}


def test_the_context_manager_restores_after_an_exception(two_blas_threads):
    with pytest.raises(RuntimeError, match="stage failed"):
        with process_thread_limit(1):
            assert _blas_threads() == {1}
            raise RuntimeError("stage failed")
    assert _blas_threads() == {2}


def _built_controllers(monkeypatch: pytest.MonkeyPatch) -> list[None]:
    """Note each ThreadpoolController built from now on."""
    built: list[None] = []
    build = ThreadpoolController.__init__

    def noted(self: ThreadpoolController) -> None:
        built.append(None)
        build(self)

    monkeypatch.setattr(ThreadpoolController, "__init__", noted)
    return built


def test_scopes_reuse_one_controller_while_nothing_is_imported(monkeypatch):
    built = _built_controllers(monkeypatch)
    exit_thread_limit(enter_thread_limit(None))
    before = len(built)

    for _ in range(50):
        exit_thread_limit(enter_thread_limit(1))
        with process_thread_limit(None):
            pass

    # Each scope built two controllers, one at each end, which only the cycle
    # collector frees.
    assert len(built) == before


def test_a_scope_after_an_import_builds_the_controller_again(monkeypatch):
    import sys
    import types

    built = _built_controllers(monkeypatch)
    exit_thread_limit(enter_thread_limit(None))
    before = len(built)
    # An extension module that loads a BLAS or OpenMP runtime is imported.
    stand_in = types.ModuleType("cytearc_native_library_stand_in")
    monkeypatch.setitem(sys.modules, stand_in.__name__, stand_in)

    exit_thread_limit(enter_thread_limit(None))
    exit_thread_limit(enter_thread_limit(None))

    assert len(built) == before + 1


def test_many_scopes_leave_no_controller_to_the_cycle_collector():
    import gc

    exit_thread_limit(enter_thread_limit(None))
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    gc.set_debug(gc.DEBUG_SAVEALL)
    try:
        # As many scopes as the blocks of a stream that limits each block's
        # BLAS threads.
        for _ in range(40):
            with process_thread_limit(1):
                pass
        gc.collect()
        left = [obj for obj in gc.garbage if isinstance(obj, ThreadpoolController)]
    finally:
        gc.set_debug(0)
        gc.garbage.clear()
        if enabled:
            gc.enable()

    assert left == []
