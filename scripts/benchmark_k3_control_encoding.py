#!/usr/bin/env python3
"""Offline CPU encoding probe; never a GPU throughput estimate or a pass threshold."""

import argparse
import io
import json
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    args = parser.parse_args()
    runtime = json.loads(args.runtime.read_text())
    step = next(s for s in runtime["target_steps"] if s.get("window"))
    packet = {"pp_admission": step["ping_admission"]}

    class Sink(io.StringIO):
        calls = 0

        def write(self, value):
            self.calls += 1
            return super().write(value)

    original = Sink()
    json.dump(packet, original, allow_nan=False)
    encoded = json.dumps(packet, allow_nan=False)
    assert encoded == original.getvalue()
    result = dict(source=str(args.runtime), scope="first measured actual admission subtree",
                  byte_equal=True, bytes=len(encoded.encode()), streaming_writes=original.calls,
                  one_shot_writes=1, samples_ms={})
    for name in ("streaming", "one_shot"):
        values = []
        for _ in range(20):
            sink = io.StringIO()
            start = time.perf_counter_ns()
            if name == "streaming":
                json.dump(packet, sink, allow_nan=False)
            else:
                sink.write(json.dumps(packet, allow_nan=False))
            values.append((time.perf_counter_ns()-start)/1e6)
        result["samples_ms"][name] = dict(values=values, median=statistics.median(values))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
