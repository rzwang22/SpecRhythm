"""Coordinator opportunities over immutable homes; only owner claim mutates readiness."""


class PingPrePostController:
    def __init__(self, mode=None):
        from specrhythm.serving.k3 import MODES, geometry

        self.geometry = geometry(mode) if mode in MODES else None
        self.opportunity = 0
        self.admitted = 0

    def select(self, clock, client):
        value = client.call(
            "pp_admit",
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
        # Empty polls do not consume A/B roles, but each snapshot has a fresh identity.
        self.opportunity += 1
        self.admitted += bool(value["claims"])
        return value
