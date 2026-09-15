"""Thread-local write attribution shared by writers and optional observers.

No I/O, locks, fd lookup or serving dependency. Scope the actual sync operation;
the final destination and the physical (possibly temporary) file are distinct.
This is observation metadata, never a durability or execution policy.
"""

import os
import threading
from contextlib import contextmanager
from pathlib import Path

IO_CONTEXT = threading.local()


@contextmanager
def file_context(path, *, physical_path=None, write_kind="jsonl"):
    previous = getattr(IO_CONTEXT, "fields", None)
    previous_name = getattr(IO_CONTEXT, "name", None)
    IO_CONTEXT.fields = dict(
        log_name=Path(path).name,
        log_path=os.fspath(path),
        physical_path=os.fspath(path if physical_path is None else physical_path),
        write_kind=write_kind,
    )
    IO_CONTEXT.name = Path(path).name  # Compatibility with the original observer.
    try:
        yield
    finally:
        IO_CONTEXT.fields = previous
        IO_CONTEXT.name = previous_name


def sync_attribution():
    fields = getattr(IO_CONTEXT, "fields", None)
    return dict(fields) if fields is not None else dict(
        log_name=None, log_path=None, physical_path=None, write_kind=None
    )
