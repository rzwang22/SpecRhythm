"""Same physical batching/checks; complete three-token reuse needs no tail forward."""

from specrhythm.continuation.ping_prepost_backend import PingPrePostBackendMixin
from specrhythm.serving.k3 import PARAMETERS, PROTOCOL


class K3BackendMixin(PingPrePostBackendMixin):
    protocol, parameters = PROTOCOL, PARAMETERS
    uniform_candidate_length = 3
