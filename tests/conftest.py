"""Shared fixtures for the portal test suite."""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

from agentic_learning_portal.storage.storage_factory import reset_storage


@pytest.fixture(autouse=True)
def _forbid_real_llm_calls(monkeypatch):
    """Fail any test that lets a real LLM call (or any HTTP call) escape.

    Every LLM call in the app funnels through ``api/llm.py``: the structured
    path builds a ``pydantic_ai.Agent`` and the raw path opens an
    ``httpx.AsyncClient``. Tests that exercise those code paths patch the seams
    themselves (``Agent``, ``httpx.AsyncClient``, ``_call_llm``, judge seams,
    …); this guard turns any unpached use into an immediate failure so a slow,
    billed, network-bound call can never slip through. An ``AssertionError``
    here means a test is reaching the network — patch the seam instead.

    ``httpx.AsyncClient`` is a module-level attribute shared by every importer,
    so this also blocks the Wolfram judge's direct httpx path and any accidental
    HTTP call, which is exactly what a test suite should never do.
    """

    def _forbidden(*args, **kwargs):
        raise AssertionError(
            "A test tried to make a real network call (LLM or HTTP). Patch the "
            "seam (e.g. api.llm.Agent, httpx.AsyncClient, Generator._call_llm, "
            "a judge's _query_*/_translate_to_query) instead of hitting the "
            "network."
        )

    monkeypatch.setattr("agentic_learning_portal.api.llm.Agent", _forbidden)
    monkeypatch.setattr("agentic_learning_portal.api.llm.httpx.AsyncClient", _forbidden)


@pytest.fixture
def portal_env(tmp_path, monkeypatch):
    """Point the portal storage at a fresh backend with an env-seeded admin.

    Used by tests that drive the portal UI (or its auth guards) through
    AppTest: ``get_storage()`` constructs a storage seeded with the
    ``ADMIN_USERNAME``/``ADMIN_PASSWORD`` admin on construction. Tests run
    against the in-memory (dict-backed) backend by default so nothing touches
    SQLite — ``tests/storage/test_sqlite_storage.py`` is the only place that
    exercises ``SqliteStorage`` directly. ``PORTAL_DB_PATH`` is still set so a
    test can flip back to the file backend by overriding ``STORAGE_BACKEND``.
    """
    monkeypatch.setenv("STORAGE_BACKEND", "memory")
    monkeypatch.setenv("PORTAL_DB_PATH", str(tmp_path / "portal.db"))
    monkeypatch.setenv("ADMIN_USERNAME", "boss")
    monkeypatch.setenv("ADMIN_PASSWORD", "hunter2")
    reset_storage()
    yield
    reset_storage()


@pytest.fixture(scope="session", autouse=True)
def load_test_environment():
    # Define paths relative to the root directory
    root_dir = Path(__file__).parent.parent
    env_base = root_dir / '.env'
    env_test = root_dir / '.env.test'

    # Step 1: Load the base .env file first
    if env_base.exists():
        load_dotenv(dotenv_path=env_base, override=False)

    # Step 2: Load the .env.test file second, overwriting duplicates
    if env_test.exists():
        load_dotenv(dotenv_path=env_test, override=True)