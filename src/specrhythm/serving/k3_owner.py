"""Informational snapshots never wait for a Draft token; claims still use the owner."""

import copy

from specrhythm.serving.ping_prepost_owner import PingPrePostOwner


class K3Owner(PingPrePostOwner):
    def _publish_status(self):
        # Called by the single owner after mutations/fences, never from socket code.
        # Assignment publishes a new immutable-by-convention object under CPython.
        self._status_snapshot = super()._dispatch("status", {})

    def call(self, operation, payload):
        if operation == "status" and self.failure is None and not self.closed:
            # The consumer uses status for population only. Admission, release,
            # stop and Serial's idle gate always validate on the authoritative owner.
            return copy.deepcopy(self._status_snapshot)
        return super().call(operation, payload)

    def _dispatch(self, operation, payload):
        if operation == "k3_idle":
            return dict(idle=not self.machine.has_work())
        return super()._dispatch(operation, payload)
