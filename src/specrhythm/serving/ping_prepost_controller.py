"""Coordinator opportunities over immutable homes; only owner claim mutates readiness."""


import time


class PingPrePostController:
    def __init__(self, mode=None, configuration="k3-b16-v1"):
        from specrhythm.serving.k3 import MODES, geometry

        self.geometry = geometry(mode, configuration) if mode in MODES else None
        self.opportunity = 0
        self.admitted = 0

    def select(self, clock, client):
        from specrhythm.serving.dual_batch import enabled, unpack

        available = time.monotonic_ns()
        value = client.call(
            "pp_admit_command" if enabled() else "pp_admit",
            dict(
                active_request_ids=[
                    rid for rid, r in clock.rows.items() if r["state"] == "ACTIVE"
                ],
                opportunity=self.opportunity,
                normal_cohort="A" if (self.geometry and self.geometry["serial_idle_gate"])
                or self.admitted % 2 == 0 else "B",
                capacity=self.geometry["target_request_ceiling"] if self.geometry else 8,
            ),
        )
        if enabled():
            value = unpack(value)
        value["target_available_observed_ns"] = available
        value["admission_response_ns"] = time.monotonic_ns()
        # Empty polls do not consume A/B roles, but each snapshot has a fresh identity.
        self.opportunity += 1
        self.admitted += bool(value["claims"])
        return value
