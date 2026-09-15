"""Opt-in, actual three-candidate protocol; never aliases the old P1/P4 modes."""

MODES = ("serial-k3", "serial-eager-k3", "pingpong-k3", "pingpong-eager-k3")
PROTOCOL = "specrhythm.uniform-k3.v1"
RESIDENT_POLICY = "k3-normalize-once-v1"
PARAMETERS = dict(
    candidate_length=3, eager_candidate_limit=3, serial_extension_steps=2
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


SERIAL_MODES = ("serial-k3", "serial-eager-k3")
EAGER_MODES = ("serial-eager-k3", "pingpong-eager-k3")
GEOMETRY_VERSION = "specrhythm.k3-four-mode.v1"


def geometry(mode):
    from specrhythm.serving.common import require

    require(mode in MODES, "unknown K3 execution geometry", mode=mode)
    serial = mode in SERIAL_MODES
    return dict(schema_version=GEOMETRY_VERSION, active_limit=16,
                home_capacities={"A": 16} if serial else {"A": 8, "B": 8},
                target_request_ceiling=16 if serial else 8,
                draft_physical_batch_ceiling=16, eager=mode in EAGER_MODES,
                serial_idle_gate=serial,
                warmup_unit="16 completed request verification opportunities")
