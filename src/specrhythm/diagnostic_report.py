"""Optional post-run report sink; no serving/GPU dependency or implicit activation."""

import os

_SINK = None


def install(sink):
    global _SINK
    _SINK = (os.getpid(), sink)


def defer(path, build):
    if _SINK is None or _SINK[0] != os.getpid():
        return False
    return _SINK[1](path, build)
