"""Opt-in, actual three-candidate protocol; never aliases the old P1/P4 modes."""

MODES = ("serial-k3", "serial-eager-k3", "pingpong-k3", "pingpong-eager-k3")
PROTOCOL = "specrhythm.uniform-k3.v1"
RESIDENT_POLICY = "k3-normalize-once-frame-once-v2"
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
B16 = "k3-b16-v1"
B64 = "k3-b64-v1"
CONFIGURATIONS = (B16, B64)


def configuration_of(record):
    """Absent declarations retain B16 only. Explicit invalid values never fall back."""
    from specrhythm.serving.common import require

    value = record.get("k3_configuration", B16)
    require(type(value) is str and value in CONFIGURATIONS, "invalid K3 configuration")
    return value


def configuration_fields(configuration):
    configuration_of({"k3_configuration": configuration})
    return {"k3_configuration": configuration} if configuration != B16 else {}


def validate_point(point, manifest):
    """Fail before engine creation if any declaration along the launch chain differs."""
    from specrhythm.serving.common import require
    from specrhythm.serving.k3_validation import matching

    matching(point, manifest)
    config = configuration_of(manifest)
    require(configuration_of(point) == config, "K3 point/manifest configuration mismatch")
    mode = point["mode"]
    if mode not in MODES:
        return
    g = geometry(mode, config)
    meta = manifest["fixed_diagnostic"]["capacity"][mode]
    require(point["runtime_mode"] == mode
            and type(point["batch"]) is int and point["batch"] == g["active_limit"]
            and type(manifest["active_limit"]) is int
            and manifest["active_limit"] == g["active_limit"]
            and configuration_of(meta) == config
            and matches_geometry(meta.get("execution_geometry"), mode, config)
            and type(meta["active_request_limit"]) is int
            and meta["active_request_limit"] == g["active_limit"]
            and type(meta["max_requests_per_target_forward"]) is int
            and meta["max_requests_per_target_forward"] == g["target_request_ceiling"],
            "K3 launch point/manifest/capacity geometry mismatch")


def geometry(mode, configuration=B16):
    from specrhythm.serving.common import require

    require(mode in MODES, "unknown K3 execution geometry", mode=mode)
    configuration_of({"k3_configuration": configuration})
    active = 64 if configuration == B64 else 16
    serial = mode in SERIAL_MODES
    return dict(schema_version=GEOMETRY_VERSION, active_limit=active,
                **configuration_fields(configuration),
                home_capacities={"A": active} if serial else {"A": active // 2, "B": active // 2},
                target_request_ceiling=active if serial else active // 2,
                draft_physical_batch_ceiling=active, eager=mode in EAGER_MODES,
                serial_idle_gate=serial,
                warmup_unit=f"{active} completed request verification opportunities")


def matches_geometry(value, mode, configuration=B16):
    """Typed structural equality: booleans/floats cannot stand in for capacities."""
    def same(a, b):
        return type(a) is type(b) and (
            set(a) == set(b) and all(same(a[k], b[k]) for k in b)
            if isinstance(b, dict) else a == b)

    return same(value, geometry(mode, configuration))
