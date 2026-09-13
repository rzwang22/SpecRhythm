"""Coordinator opportunities over immutable homes; only owner claim mutates readiness."""


class PingPrePostController:
    def __init__(self):
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
                normal_cohort="A" if self.admitted % 2 == 0 else "B",
                capacity=8,
            ),
        )
        # Empty polls do not consume A/B roles, but each snapshot has a fresh identity.
        self.opportunity += 1
        self.admitted += bool(value["claims"])
        return value
