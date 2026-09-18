"""Bounded mailbox; feedback acknowledgement is independent of proposal readiness."""

from specrhythm.serving.eager_owner import EagerOwner


class PingPrePostOwner(EagerOwner):
    def __init__(self, factory, *, timeout_seconds=900):
        super().__init__(factory, timeout_seconds=timeout_seconds, queue_capacity=256)

    def _has_work(self):
        return self.machine is not None and self.machine.has_work()

    def _dispatch(self, operation, payload):
        m = self.machine
        if operation == "pp_register":
            return m.register(payload["requests"])
        if operation == "pp_admit":
            return m.admit(payload)
        if operation == "pp_feedback":
            return m.feedback(payload["synchronizations"])
        if operation == "pp_stop":
            return m.begin_drain()
        if operation == "pp_eligibility":
            return m.update_eligibility(
                payload["enabled"], payload["decision_version"], payload["eligible_request_ids"]
            )
        if operation == "status":
            inflight = set(m.normal) | {r["request_id"] for r in m._pending_rows()}
            inflight.update(
                rid
                for rid, w in m.works.items()
                if not m.core._state(rid).continuations[w.work_id].completed
            )
            return dict(
                inflight_request_ids=sorted(inflight),
                pending_work=sorted(inflight),
                ready_request_ids=list(m.ready),
                failures=[],
            )
        return super()._dispatch(operation, payload)
