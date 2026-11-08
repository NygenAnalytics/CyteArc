"""Raise the errors that a cleanup collected, without keeping them in a reference cycle."""

__all__ = ["raise_collected_errors"]


def raise_collected_errors(
    errors: list[BaseException],
    message: str,
    *,
    ended_by: BaseException | None = None,
) -> None:
    """Raise the errors in ``errors``, if any, and empty the list.

    ``ended_by`` is the error that ended the work before its cleanup ran. It
    leads the other errors, unless it is the ``GeneratorExit`` that closing a
    generator throws, which is no failure. One error is raised as itself and
    several as a ``BaseExceptionGroup`` with ``message``. With no errors this
    returns, and a caller that caught ``ended_by`` raises it again itself.

    An error's traceback refers to each frame it passed through. A frame that
    kept an error raised in it, in a list or a local, would form a reference
    cycle with it, and that cycle keeps the frame and everything it refers to,
    such as blocks of data, until Python's cycle collector runs. So the list is
    empty before the error leaves, and a caller should catch ``ended_by`` with
    ``except ... as``, whose name Python unbinds when the clause ends.

    Args:
        errors: Errors collected in cleanup, in the order they happened.
        message: Message of the group that several errors are raised in.
        ended_by: The error that ended the work, if any.
    """
    if not errors:
        return
    if ended_by is not None and not isinstance(ended_by, GeneratorExit):
        errors.insert(0, ended_by)
    try:
        if len(errors) == 1:
            raise errors[0]
        raise BaseExceptionGroup(message, list(errors))
    finally:
        errors.clear()
