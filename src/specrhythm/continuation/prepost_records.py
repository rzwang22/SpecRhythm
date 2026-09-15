"""Bounded protocol evidence; follows already-read trace phase, never performs I/O."""

from specrhythm.continuation.trace import TRACE, CausalTrace


class Records:
    def __init__(self):
        self.trace = CausalTrace(True, layout="phased")

    def append(self, row):
        self.trace.follow_control({"diagnostic_phase": TRACE.phase})
        self.trace.event("prepost_record", **row)

    def report(self):
        return self.trace.report()

    def rows(self):
        return self.report()["rows"]
