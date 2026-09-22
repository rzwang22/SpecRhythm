"""Plan-derived, bounded archive counts; not producer retention or GPU budgets."""

from specrhythm.serving.common import read_json, require


def count_budget(directory, *, logical=512, unique=512):
    legacy = dict(profile="legacy-v1", logical_files=logical, unique_files=unique)
    path = directory / "dual-batch-plan.json"
    if not path.exists():
        return legacy
    require(path.stat().st_size < 128 * 1024, "export plan exceeds declaration budget")
    plan = read_json(path)
    require(isinstance(plan, dict), "export plan must be an object")
    selected = plan.get("cpu_comparison", False)
    require(type(selected) is bool, "cpu_comparison must be a boolean")
    if not selected:
        return legacy
    from specrhythm.serving.dual_batch_run import CPU_CASES, CPU_ORDER
    from specrhythm.serving.k3 import B128

    cases = {k: dict(mode=v[0], target_dispatch=v[1], target_cpu=v[2])
             for k, v in CPU_CASES.items()}
    require(plan.get("cases") == cases and plan.get("order") == list(CPU_ORDER)
            and plan.get("smoke_cases") == [*cases, "S0"]
            and plan.get("k3_configuration") == B128
            and plan.get("validation_profile") == "performance-exploration",
            "invalid ordinary CPU export plan; no expanded budget granted")
    # Observed 545 logical / 438 unique files across 13 runs. Allocate 64/run
    # (reports, native sidecars, logs, receipts) + 64 orchestration files. This
    # does not depend on the observed file count, and cannot grow with retries.
    counts = dict(capacity_runs=len(cases), smoke_runs=len(plan["smoke_cases"]),
                  performance_runs=len(CPU_ORDER))
    runs = sum(counts.values())
    limit = 64 + 64 * runs
    return dict(profile="ordinary-cpu-13-runs-v1", **counts, planned_runs=runs,
                per_run_files=64, orchestration_files=64,
                logical_files=limit, unique_files=limit)


def export_summary(result):
    """Small terminal receipt; the inventory retains every error and source path."""
    rows = result.get("inventory", [])
    errors = result.get("export_errors", [])
    code = result["export_validation_exit_code"]
    return dict(export_status=result["export_status"], export_exit_code=code,
                failure_layer="archive_evidence" if code == 41 else
                "comparison" if code == 42 else None,
                source_first_exit_code=result["first_exit_code"],
                logical_files=len(result.get("logical_paths", {})),
                limits=result.get("limits"),
                omitted_limit=sum(r.get("status") == "OMITTED_LIMIT" for r in rows),
                missing=sum(r.get("status") == "MISSING" for r in rows),
                error_count=len(errors), first_errors=errors[:5],
                full_errors="archive inventory.json: export_errors and inventory")
