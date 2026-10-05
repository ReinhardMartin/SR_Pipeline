import copy
import json
from pathlib import Path
from threading import RLock
from typing import Any

from core.provenance import canonical_hash, write_json_atomic


class Checkpoint:
    def __init__(self, path: Path, fingerprint: str):
        self.path = path
        self.fingerprint = fingerprint
        self.values = {}
        self._lock = RLock()
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if (saved["fingerprint"] == fingerprint and isinstance(saved["values"], dict)
                    and saved["checksum"] == canonical_hash(saved["values"])):
                self.values = saved["values"]
        except (OSError, ValueError, KeyError, TypeError):
            pass

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return copy.deepcopy(self.values.get(key, default))

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            values = {**self.values, key: copy.deepcopy(value)}
            self._save_locked(values)

    def save(self, values: dict) -> None:
        with self._lock:
            self._save_locked(values)

    def _save_locked(self, values: dict) -> None:
        snapshot = copy.deepcopy(values)
        write_json_atomic(self.path, {
            "fingerprint": self.fingerprint,
            "values": snapshot,
            "checksum": canonical_hash(snapshot),
        })
        self.values = snapshot
