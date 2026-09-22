"""Call-local immutable admission records with byte-identical checkpoint framing.

Only scalar audit fields are accepted. No dynamic scheduling state is cached, and
no record is delayed: the existing writer owns persistence, budgets and failures.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from specrhythm.phase4.resident_setup import ADMISSION_EVENT_SCHEMA


@dataclass(frozen=True, init=False)
class PreparedAdmission(Mapping):
    _values: Mapping
    line: bytes

    def __init__(self, common, dynamic):
        if set(common) & set(dynamic):
            raise ValueError("admission shared/dynamic field collision")
        value = {**common, **dynamic}
        if (value.get("schema_version") != ADMISSION_EVENT_SCHEMA
                or "record_sha256" in value
                or type(value.get("request_id")) is not str
                or any(type(v) not in (str, int, bool, type(None)) for v in value.values())):
            raise ValueError("prepared admission requires immutable scalar audit fields")
        body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        checksum = hashlib.sha256(body).hexdigest().encode("ascii")
        # In this schema request_id is the first sorted key after record_sha256.
        # Quotes in string values are JSON-escaped and cannot match this delimiter.
        marker = b',"request_id":'
        if body.count(marker) != 1 or any("record_sha256" < k < "request_id" for k in value):
            raise ValueError("admission checksum insertion contract changed")
        line = body.replace(marker, b',"record_sha256":"' + checksum + b'"' + marker, 1)
        object.__setattr__(self, "_values", MappingProxyType(value))
        object.__setattr__(self, "line", line + b"\n")

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)
