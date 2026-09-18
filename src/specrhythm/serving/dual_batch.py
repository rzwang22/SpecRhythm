"""Compatibility facade for the measured ordinary PingPong path.

The historical dual-batch mode guard remains in shared_control.validate_mode.
Serial must explicitly select shared-command and still uses K3Machine.admit's
idle gate and the sole owner; this module does not change its protocol.
"""

from specrhythm.serving.shared_control import (  # noqa: F401
    SCHEMA,
    CommandPublisher,
    control_scope,
    control_transaction,
    current_control,
    decode_control,
    enabled,
    encode_control,
    pack,
    run_target,
    unpack,
    validate_mode,
)
