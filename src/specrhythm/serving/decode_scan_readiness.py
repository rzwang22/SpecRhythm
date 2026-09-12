"""Scan-only pre-allocation wait and bounded, coalesced host evidence."""

from __future__ import annotations


class ScanBatchWait(RuntimeError):
    """Return to the coordinator without calling the stock allocator or model."""

    def __init__(self, evidence):
        super().__init__("decode scan awaiting full cohort before stock allocation")
        self.evidence = evidence


def coalesce(rows, row, *, limit=512):
    """Keep first/last times and exact observation count; bound transition history.

    At saturation the last slot explicitly counts omitted transitions. This is
    diagnostic evidence, never input to scheduling or performance qualification.
    """
    payload = {k: v for k, v in row.items() if k != "timestamp_ns"}
    if rows and rows[-1]["value"] == payload:
        rows[-1]["last_ns"] = row["timestamp_ns"]
        rows[-1]["observations"] += 1
    elif len(rows) < limit:
        rows.append({"value": payload, "first_ns": row["timestamp_ns"],
                     "last_ns": row["timestamp_ns"], "observations": 1})
    else:
        last = rows[-1]
        last.update(value=payload, last_ns=row["timestamp_ns"],
                    observations=last["observations"] + 1,
                    omitted_transitions=last.get("omitted_transitions", 0) + 1)


class ReadinessEvidence:
    def __init__(self):
        self.rows = []
        self.wait_start = None
        self.wait_count = self.poll_count = self.wait_ns = 0
        self.event_cursor = 0

    def lifecycle(self, clock):
        for event in clock.events[self.event_cursor:]:
            if event["event"] in ("finished", "resources-released", "admitted"):
                coalesce(self.rows, {k: event[k] for k in
                         ("event", "timestamp_ns", "request_id", "cohort") if k in event})
        self.event_cursor = len(clock.events)

    def waiting(self, evidence):
        self.poll_count += 1
        entering = self.wait_start is None
        if entering:
            self.wait_start = evidence["timestamp_ns"]
            self.wait_count += 1
        coalesce(self.rows, {**evidence, "event": "wait-enter" if entering else "wait-state"})
        return entering

    def close(self, now, reason):
        if self.wait_start is not None:
            self.wait_ns += now - self.wait_start
            coalesce(self.rows, {"event": reason, "timestamp_ns": now,
                                 "wait_start_ns": self.wait_start})
            self.wait_start = None

    def report(self):
        return {"schema_version": "specrhythm.decode-scan-readiness.v1",
                "scope": "scan PingPong; host readiness before stock allocation",
                "wait_count": self.wait_count, "wait_poll_count": self.poll_count,
                "closed_wait_ns": self.wait_ns, "open_wait_start_ns": self.wait_start,
                "max_records": 512, "transitions": self.rows,
                "history_complete": not any(r.get("omitted_transitions") for r in self.rows)}
