"""Best-effort K3 pre-drive shutdown, preserving the original startup exception."""

import time

from specrhythm.serving.common import require
from specrhythm.serving.fixed_artifacts import record_error
from specrhythm.serving.fixed_settle import remaining
from specrhythm.serving.s2_pool import publish


class StartupCleanup:
    def __init__(self, directory, mode, error, deadline_ns):
        self.directory = directory
        self.result = dict(
            schema_version="specrhythm.k3-startup-cleanup.v1", mode=mode,
            run_directory=str(directory.resolve()),
            phase="startup_failure_cleanup", status="RUNNING", deadline_ns=deadline_ns,
            original_error=str(error), execution_failed=True, actions={}, recording_errors=[],
            cleanup_qualification="PENDING_SUPERVISOR",
        )
        self.completed = False

    def save(self):
        for name in ("startup-cleanup.json", "drain-state.json"):
            try:
                publish(self.directory / name, self.result)
            except Exception as error:
                self.result["recording_errors"].append(dict(file=name, error=str(error)))
                record_error(self.directory, error, "startup_cleanup_recording")

    def finish(self, llm, client_factory):
        if self.completed:
            return self.result
        self.completed = True  # Never repeat an acknowledged or failed shutdown attempt.
        deadline = self.result["deadline_ns"]
        self.save()  # Supervisor observes the unchanged drain budget, even on a hung RPC.
        for name in ("draft_shutdown", "target_engine_shutdown"):
            entry = self.result["actions"][name] = dict(start_ns=time.monotonic_ns())
            try:
                timeout = remaining(deadline)
                if name == "draft_shutdown":
                    client = client_factory()
                    client.timeout_seconds = timeout
                    receipt = client.call("shutdown", {"deadline_ns": deadline})
                    require(receipt.get("shutdown") is True
                            and receipt.get("physical_live_requests") == 0
                            and receipt.get("pending_work") == [],
                            "K3 pre-drive Draft shutdown lacks empty-owner receipt")
                    entry.update(status="ACKNOWLEDGED", receipt=receipt)
                elif llm is None:
                    entry.update(status="NO_HANDLE", reason="engine constructor did not return; "
                                 "partial worker cleanup requires owned supervisor evidence")
                else:
                    llm.llm_engine.engine_core.shutdown(timeout=timeout)
                    entry["status"] = "RETURNED"
                remaining(deadline)
            except Exception as error:
                entry.update(status="FAILED", error=str(error))
                record_error(self.directory, error, "startup_" + name)
            entry["end_ns"] = time.monotonic_ns()
        # Failure remains failure, even if both release APIs return successfully.
        self.result.update(status="FAILED", end_ns=time.monotonic_ns(),
            release_attempts_complete=all(r["status"] in ("ACKNOWLEDGED", "RETURNED")
                                         for r in self.result["actions"].values()))
        self.save()
        return self.result
