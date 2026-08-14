"""Phase-3 GPU readiness, real-trace, and calibration interfaces.

The modules in this package do not change the Phase-2 simulator.  PyTorch and
Transformers are optional and imported only by the real GPU backend.
"""

from specrhythm.phase3.config import Phase3Config, load_phase3_config
from specrhythm.phase3.trace import RealTraceRecord, TraceStore

__all__ = ["Phase3Config", "RealTraceRecord", "TraceStore", "load_phase3_config"]
