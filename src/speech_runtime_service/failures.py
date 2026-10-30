"""Failures that are logged without the payload that may have caused them."""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import logging
    from types import TracebackType
    from typing import Self


class SpeakingFailedError(Exception):
    """A speaking stream failed; the cause is logged, never carried."""

    def __init__(self) -> None:
        """Carry only the fixed message."""
        super().__init__("speaking failed")


def log_failure(logger: logging.Logger, message: str, error: BaseException) -> None:
    """Log the fixed message, each chained exception's type and its stack frames.

    An exception message, argument or local variable can hold the text a person
    spoke or heard, so none of them is ever logged.
    """
    lines = [message]
    seen: set[int] = set()
    link: BaseException | None = error
    while link is not None and id(link) not in seen:
        seen.add(id(link))
        kind = type(link)
        lines.append(f"{kind.__module__}.{kind.__qualname__}")
        lines.extend(
            f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}'
            + (f"\n    {frame.line}" if frame.line else "")
            for frame in traceback.extract_tb(link.__traceback__)
        )
        link = link.__cause__ or link.__context__
    logger.error("\n".join(lines))


class PayloadFreeFailure:
    """Log an Exception that leaves the block by type and frames, never its message.

    BaseException (cancellation, generator close) and the `expected` exceptions
    pass through untouched. Any other Exception is logged and then either
    raised as `reraise_as` without its cause, or swallowed and recorded in
    `failed` so the caller can close with a fixed reason.
    """

    def __init__(
        self,
        logger: logging.Logger,
        message: str,
        *,
        expected: tuple[type[Exception], ...] = (),
        reraise_as: type[Exception] | None = None,
    ) -> None:
        """Name where and how a failure is logged and what becomes of it."""
        self._logger = logger
        self._message = message
        self._expected = expected
        self._reraise_as = reraise_as
        self.failed = False

    def __enter__(self) -> Self:
        """Guard the block."""
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Log an unexpected Exception; swallow it unless it is re-raised."""
        del exception_type, traceback
        if not isinstance(error, Exception) or isinstance(error, self._expected):
            return False
        log_failure(self._logger, self._message, error)
        self.failed = True
        if self._reraise_as is not None:
            raise self._reraise_as from None
        return True
