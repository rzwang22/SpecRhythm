"""Opt-in, actual three-candidate protocol; never aliases the old P1/P4 modes."""

MODES = ("serial-k3", "pingpong-k3", "pingpong-eager-k3")
PROTOCOL = "specrhythm.uniform-k3.v1"
RESIDENT_POLICY = "k3-normalize-once-v1"
PARAMETERS = dict(
    candidate_length=3, eager_candidate_limit=3, serial_extension_steps=2, target_request_ceiling=8
)


def accounting_complete(value):
    """Final owner-lifetime candidate conservation, separate from output commits."""
    keys = (
        "generated", "submitted", "discarded_early", "discarded_unsubmitted", "live_unsubmitted",
    )
    return (
        isinstance(value, dict)
        and all(type(value.get(k)) is int and value[k] >= 0 for k in keys)
        and value["live_unsubmitted"] == 0
        and value["generated"] == sum(value[k] for k in keys[1:4])
    )
