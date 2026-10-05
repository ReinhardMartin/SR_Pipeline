import json
from pathlib import Path

from services.errors import ServiceError


def read_json(path: Path):
    if not path.exists():
        raise ServiceError(404, "Not found")
    return json.loads(path.read_text(encoding="utf-8"))
