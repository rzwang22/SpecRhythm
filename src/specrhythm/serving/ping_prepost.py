"""Explicit PingPong pre/post protocol; cohorts are homes, opportunities are roles."""

from specrhythm.serving.k3 import MODES as K3_MODES

MODES = ("pingpong-prepost3", "pingpong-eager-prepost3")
PROTOCOL = "specrhythm.pingpong-prepost3.v1"
PURPOSES = (
    "prepost_lookahead",
    "prepost_post",
    "prepost_extension",
    "prepost_repair",
    "prepost_mixed",
)
OWNER_ID = "draft-gpu0-prepost-owner"

# Routing only. Old pair runners continue to use MODES.

SCHEDULED_MODES = (*MODES, *K3_MODES)
