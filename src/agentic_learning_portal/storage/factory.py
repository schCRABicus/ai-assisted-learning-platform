import os
import threading

from agentic_learning_portal.storage import Storage, InMemoryStorage, SqliteStorage


# Where the portal's SQLite database lives (used when ``STORAGE_BACKEND`` is
# ``"sqlite"``). File-backed so users, the seeded admin, and password hashes
# survive Streamlit's per-run process, and so a script rerun logs in against
# the same data. Override with PORTAL_DB_PATH.
DEFAULT_DB_PATH = "portal.db"

# A ``SqliteStorage`` must be used on the thread that constructed it (Python
# 3.14 removed the post-connect ``check_same_thread`` switch), and Streamlit may
# run a session's script runs on different threads — so keep one storage per
# thread, all backed by the same ``PORTAL_DB_PATH`` file. ``_STORAGES`` maps
# thread id → instance so :func:`reset_storage` can drop every instance.
_STORAGES: dict[int, Storage] = {}

# In memory mode the portal hands out one process-wide ``InMemoryStorage``:
# unlike SQLite there is no per-thread connection to worry about, and the store
# must be shared so a user created on one thread (a test's main thread) is
# visible to a page script running on another (AppTest's). ``reset_storage``
# drops it for test isolation.
_MEMORY_STORAGE: InMemoryStorage | None = None

class StorageFactory:

    def __init__(self):
        self.storage_backend = os.getenv("STORAGE_BACKEND", "sqlite")

    def get_storage(self) -> Storage:
        """Return the :class:`Storage` the portal should use.

        The backend is chosen by ``STORAGE_BACKEND`` (``"sqlite"`` or ``"memory"``),
        read at call time so tests and ``.env`` can switch it after import. The
        sqlite backend keeps one instance per thread (``seed_admin_from_env`` runs
        on construction, so the ``.env``-configured admin is available from the
        start); the memory backend is a single shared instance.
        """
        # Read the env var at call time so tests (and .env) can switch backends
        # after ``auth`` is imported.
        if self.storage_backend == "memory":
            global _MEMORY_STORAGE
            if _MEMORY_STORAGE is None:
                _MEMORY_STORAGE = InMemoryStorage()
            return _MEMORY_STORAGE

        ident = threading.get_ident()
        storage = _STORAGES.get(ident)
        if storage is None:
            storage = SqliteStorage(os.getenv("PORTAL_DB_PATH", DEFAULT_DB_PATH))
            _STORAGES[ident] = storage
        return storage


def reset_storage() -> None:
    """Forget every storage instance (tests use this for isolation).

        The in-memory backend is dropped wholesale. For sqlite, only the instance
        bound to the calling thread is closed explicitly; the others are dropped and
        closed by garbage collection when their thread ends (a connection can only
        be closed from the thread that created it). The next ``get_storage``
        re-opens the DB, re-seeding the admin from env.
        """
    global _MEMORY_STORAGE
    _MEMORY_STORAGE = None
    ident = threading.get_ident()
    current = _STORAGES.pop(ident, None)
    if current is not None:
        current.close()
    _STORAGES.clear()