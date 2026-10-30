"""Failures that are logged without the payload that may have caused them."""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import logging


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
