import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(stage: str, inputs: dict[str, str | Path], settings: dict) -> dict:
    input_hashes = {
        label: file_hash(path)
        for label, path in sorted(inputs.items())
    }
    identity = {"stage": stage, "inputs": input_hashes, "settings": settings}
    return {
        **identity,
        "fingerprint": canonical_hash(identity),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def artifact_hash(path: str | Path) -> str:
    path = Path(path)
    if not path.is_dir():
        return file_hash(path)
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Empty artifact directory: {path}")
    return canonical_hash({item.relative_to(path).as_posix(): file_hash(item) for item in files})


def write_manifest(path: Path, manifest: dict, outputs: dict[str, Path]) -> None:
    write_json_atomic(path, {**manifest, "outputs": {key: artifact_hash(value) for key, value in outputs.items()}})


def manifest_current(path: Path, expected: dict, outputs: dict[str, Path]) -> bool:
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        return (
            saved["fingerprint"] == expected["fingerprint"]
            and saved["outputs"] == {key: artifact_hash(value) for key, value in outputs.items()}
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def write_json_atomic(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_text_atomic(path: str | Path, value: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value.rstrip() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
