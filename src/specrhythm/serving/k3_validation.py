"""Validation policy is independent of execution geometry and GPU algorithms."""

from specrhythm.serving.common import read_json, require
from specrhythm.serving.k3 import B64, B128, configuration_of

STRICT = "strict-output"
EXPLORATION = "performance-exploration"
PROFILES = (STRICT, EXPLORATION)
KNOWN_DIFFERENCE = {
    "execution_sha": "f6f67aa1e1d7aea2a81665ec628d0ae857c148ee",
    "archive_sha256": "d9e31c43f66d501451823180638b66e31c0e84a92ac932be2e01b4b47eaa1606",
    "exact_requests_out_of_64": {"serial-k3": 54, "serial-eager-k3": 54,
                                 "pingpong-k3": 64, "pingpong-eager-k3": 64},
    "shared_serial_mismatches": 10,
    "performance": "NOT_RUN in that historical experiment",
    "resolution": "UNRESOLVED; current policy does not change historical results",
}


def profile_of(record):
    require(isinstance(record, dict), "validation_profile requires an object record")
    value = record.get("validation_profile", STRICT)
    require(type(value) is str and value in PROFILES, "invalid validation_profile")
    require(value != EXPLORATION or configuration_of(record) in (B64, B128),
            "performance-exploration requires explicit k3-b64-v1 or k3-b128-v1")
    return value


def fields(value, configuration):
    if value is None:
        return {}  # Legacy/B16 records retain their original strict contract.
    profile_of(dict(validation_profile=value, k3_configuration=configuration))
    return dict(validation_profile=value)


def matching(*records):
    values = [profile_of(r) for r in records]
    require(len(set(values)) == 1, "validation_profile differs across producer records")
    return values[0]


def not_run(record):
    if "validation_profile" not in record:
        return {}
    policy = profile_of(record)
    return dict(validation_profile=policy, k3_configuration=configuration_of(record),
                full_output_comparison_run=False,
                output_equivalence_status="NOT_RUN",
                output_equivalence_scope="this performance/capacity point",
                output_equivalence_reason="explicit performance exploration; full output "
                "equivalence not verified in this run" if policy == EXPLORATION else
                "full output equivalence is checked in the separate joint diagnostic",
                known_output_difference=KNOWN_DIFFERENCE)


def plan(value, configuration):
    return dict(schema_version="specrhythm.k3-validation-plan.v1",
                k3_configuration=configuration, **fields(value, configuration),
                full_output_comparison_planned=value == STRICT,
                known_output_difference=KNOWN_DIFFERENCE)


def read_plan(directory):
    path = directory / "validation-plan.json"
    if not path.exists():
        return {}  # Missing plan can never opt into exploration.
    require(path.stat().st_size <= 128 * 1024, "validation plan exceeds bounded size")
    value = read_json(path)
    policy = profile_of(value)
    require("validation_profile" in value
            and value.get("schema_version") == "specrhythm.k3-validation-plan.v1"
            and type(value.get("full_output_comparison_planned")) is bool
            and value["full_output_comparison_planned"] == (policy == STRICT),
            "invalid validation plan comparison declaration")
    return value
