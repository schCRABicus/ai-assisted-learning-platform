"""Progress reporting for the LLM generation pipeline.

Long-running steps — task generation, schema validation, and per-judge
verification — emit ``ProgressEvent`` objects through a ``ProgressListener``
so a caller (the Streamlit admin page, a logger, a test) can surface live
progress. The pipeline itself is decoupled from any specific sink: it only
calls ``listener.on_progress(event)`` and never inspects the result.

Events carry a coarse ``stage`` ("generate", "validate", "verify", "retry",
"done", "error") used by the UI to pick an icon, a human-readable ``message``,
and optionally ``attempt``/``total`` for a progress bar.
"""

from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProgressEvent:
    """One discrete step in the generation pipeline."""

    stage: str
    message: str
    attempt: int | None = None
    total: int | None = None


class ProgressListener(ABC):
    """Receives generation progress events.

    The base implementation is a no-op, so a caller can pass a bare
    ``ProgressListener()`` where progress is unwanted; the concrete listeners
    below collect or log events.
    """

    async def on_progress(self, event: ProgressEvent) -> None:
        """Handle ``event``. Subclasses override; the default does nothing."""


class CollectingProgressListener(ProgressListener):
    """Append every event to ``events`` so a caller can read them back.

    Useful for tests and for the admin page, which polls a list the worker
    thread appends to.
    """

    def __init__(self, events: list[ProgressEvent]) -> None:
        self.events = events

    async def on_progress(self, event: ProgressEvent) -> None:
        self.events.append(event)


class LoggingProgressListener(ProgressListener):
    """Log every event at INFO level with its stage and attempt window."""

    async def on_progress(self, event: ProgressEvent) -> None:
        window = (
            f" ({event.attempt}/{event.total})" if event.attempt is not None else ""
        )
        logger.info("progress[%s]%s: %s", event.stage, window, event.message)


async def report_progress(
    listener: ProgressListener | None,
    stage: str,
    message: str,
    *,
    attempt: int | None = None,
    total: int | None = None,
) -> None:
    """Forward a progress event to ``listener``, skipping a ``None`` listener."""
    if listener is None:
        return
    await listener.on_progress(
        ProgressEvent(stage=stage, message=message, attempt=attempt, total=total)
    )