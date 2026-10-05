from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from uuid import UUID, uuid4

from core.provenance import write_json_atomic
from core.workspace_lock import workspace_lock
from screening.domain.models import BatchMetadata, BatchSummary


class InvalidBatchId(ValueError):
    pass


class UnsafeBatchContents(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ScreeningBatchPaths:
    directory: Path

    @property
    def batch_id(self) -> str:
        return self.directory.name

    @property
    def papers(self) -> Path:
        return self.directory / "papers.json"

    @property
    def results(self) -> Path:
        return self.directory / "results.json"

    @property
    def criteria(self) -> Path:
        return self.directory / "criteria.json"

    @property
    def metadata(self) -> Path:
        return self.directory / "meta.json"

    @property
    def panel(self) -> Path:
        return self.directory / "panel.json"

    @property
    def lock(self) -> Path:
        return self.directory / "screening.lock"


class ScreeningBatchRepository:
    """Owns screening batch identity, layout, JSON persistence, and locking."""

    def __init__(self, configuration_or_root):
        self._source = configuration_or_root

    @property
    def root(self) -> Path:
        if isinstance(self._source, Path):
            return self._source
        return self._source.paths.screening

    def paths(self, batch_id: str) -> ScreeningBatchPaths:
        try:
            canonical = str(UUID(batch_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise InvalidBatchId("Screening batch IDs must be canonical UUIDs") from exc
        if batch_id != canonical:
            raise InvalidBatchId("Screening batch IDs must use canonical UUID form")
        root = self.root.resolve()
        directory = (root / canonical).resolve()
        if not directory.is_relative_to(root):
            raise InvalidBatchId("Screening batch path escapes its repository")
        return ScreeningBatchPaths(directory)

    def list(self) -> list[BatchSummary]:
        if not self.root.exists():
            return []
        summaries = []
        for directory in sorted(self.root.iterdir()):
            if not directory.is_dir():
                continue
            try:
                paths = self.paths(directory.name)
                if not paths.papers.is_file() or not paths.metadata.is_file():
                    continue
                metadata = BatchMetadata.model_validate(self.read(paths.metadata))
                papers = self.read(paths.papers)
                if not isinstance(papers, list):
                    continue
            except (InvalidBatchId, OSError, ValueError, TypeError, KeyError):
                continue
            summaries.append(
                BatchSummary(
                    batch_id=paths.batch_id,
                    filename=metadata.filename,
                    paper_count=len(papers),
                    has_results=paths.results.is_file(),
                )
            )
        return summaries

    def create(
        self, *, filename: str, source: bytes, papers: list[dict]
    ) -> ScreeningBatchPaths:
        paths = self.paths(str(uuid4()))
        paths.directory.mkdir(parents=True, exist_ok=False)
        suffix = Path(filename).suffix.lower()
        (paths.directory / f"source{suffix}").write_bytes(source)
        self.write(paths.papers, papers)
        self.write(
            paths.metadata,
            BatchMetadata(filename=filename).model_dump(exclude_none=True),
        )
        return paths

    @staticmethod
    def read(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def write(path: Path, value: Any) -> None:
        write_json_atomic(path, value)

    @contextmanager
    def locked(self, batch_id: str) -> Iterator[ScreeningBatchPaths]:
        paths = self.paths(batch_id)
        with workspace_lock(paths.lock):
            yield paths

    def delete(self, batch_id: str) -> None:
        paths = self.paths(batch_id)
        if not paths.directory.exists():
            raise FileNotFoundError(paths.directory)
        with workspace_lock(paths.lock):
            entries = [entry for entry in paths.directory.iterdir() if entry != paths.lock]
            if any(not entry.is_file() for entry in entries):
                raise UnsafeBatchContents(
                    "Batch contains directories and cannot be deleted safely"
                )
            for entry in entries:
                entry.unlink()
        paths.lock.unlink(missing_ok=True)
        paths.directory.rmdir()

