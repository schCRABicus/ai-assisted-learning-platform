from __future__ import annotations

import logging

import pytest

from agentic_learning_portal.api.progress import (
    CollectingProgressListener,
    LoggingProgressListener,
    ProgressEvent,
    ProgressListener,
    report_progress,
)


def test_progress_event_fields() -> None:
    event = ProgressEvent(stage="generate", message="Generating...", attempt=1, total=3)

    assert event.stage == "generate"
    assert event.message == "Generating..."
    assert event.attempt == 1
    assert event.total == 3


def test_progress_event_attempt_defaults() -> None:
    event = ProgressEvent(stage="done", message="Done.")

    assert event.attempt is None
    assert event.total is None


@pytest.mark.asyncio
async def test_base_listener_is_a_noop() -> None:
    # The base ProgressListener can be used directly as a no-op sink.
    await ProgressListener().on_progress(
        ProgressEvent(stage="done", message="Nothing to see here.")
    )


@pytest.mark.asyncio
async def test_collecting_listener_appends_events() -> None:
    log: list[ProgressEvent] = []
    listener = CollectingProgressListener(log)

    await listener.on_progress(ProgressEvent(stage="verify", message="Checking..."))
    await listener.on_progress(ProgressEvent(stage="done", message="Done."))

    assert [e.stage for e in log] == ["verify", "done"]


@pytest.mark.asyncio
async def test_report_progress_skips_none_listener() -> None:
    # A None listener must be a silent no-op, not an error.
    await report_progress(None, "generate", "Should not raise.")


@pytest.mark.asyncio
async def test_report_progress_forwards_event() -> None:
    log: list[ProgressEvent] = []
    listener = CollectingProgressListener(log)

    await report_progress(listener, "retry", "Retrying...", attempt=2, total=5)

    assert log == [ProgressEvent(stage="retry", message="Retrying...", attempt=2, total=5)]


@pytest.mark.asyncio
async def test_logging_listener_logs_events(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="agentic_learning_portal.api.progress"):
        await LoggingProgressListener().on_progress(
            ProgressEvent(stage="generate", message="Hi", attempt=1, total=3)
        )

    assert any("Hi" in record.message for record in caplog.records)