import hashlib
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from core.provenance import file_hash, write_json_atomic
from core.recovery_import import index_papers, match_record, pdf_source, read_library
from services.errors import ServiceError
from services.files import read_json
from screening.infrastructure.repository import (
    InvalidBatchId,
    ScreeningBatchRepository,
)


class Recovery:
    def __init__(
        self,
        configuration,
        screening,
        jobs,
        batches: ScreeningBatchRepository | None = None,
    ):
        self.configuration = configuration
        self.screening = screening
        self.jobs = jobs
        self.batches = batches or ScreeningBatchRepository(configuration)
        self.lock = threading.Lock()

    def batch_dir(self, batch_id: str) -> Path:
        try:
            paths = self.batches.paths(batch_id)
        except InvalidBatchId as exc:
            raise ServiceError(400, "Invalid batch ID") from exc
        if not paths.results.is_file():
            raise ServiceError(404, "Run screening before importing recovered papers")
        return paths.directory

    def inspect_pdf(self, relative: str) -> tuple[Path, str]:
        source = pdf_source(self.configuration.paths.input_dir, relative)
        if not source.is_file():
            raise ValueError("PDF not found in input folder")
        if source.stat().st_size > self.configuration.max_pdf_bytes:
            raise ValueError("PDF exceeds configured size limit")
        with source.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise ValueError("File does not have a PDF signature")
        return source, file_hash(source)

    def preview(self, batch_id: str, data: bytes, filename: str) -> dict:
        directory = self.batch_dir(batch_id)
        papers = self.screening.recovery(batch_id)
        limit = self.configuration.load_config()["uploads"]["max_records"]
        try:
            records = read_library(data, filename, limit)
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from exc
        output = []
        index = index_papers(papers)
        for record in records:
            match = match_record(record, papers, index)
            try:
                _, digest = self.inspect_pdf(record["pdf_path"])
                error = None
            except (OSError, ValueError) as exc:
                digest, error = None, str(exc)
            output.append({**record, **match, "sha256": digest, "file_error": error})
        result = {"import_id": str(uuid4()), "filename": Path(filename).name,
                  "records": output, "targets": [{"row": p["row"], "title": p["title"], "doi": p.get("doi", "")} for p in papers]}
        write_json_atomic(directory / f"recovery-import-{result['import_id']}.json", result)
        return result

    def confirm(self, batch_id: str, import_id: str, library_row: int, screening_row: int) -> dict:
        directory = self.batch_dir(batch_id)
        try:
            import_id = str(UUID(import_id))
        except ValueError as exc:
            raise ServiceError(400, "Invalid import ID") from exc
        with self.lock:
            preview = read_json(directory / f"recovery-import-{import_id}.json")
            record = next((r for r in preview["records"] if r["library_row"] == library_row), None)
            target = next((p for p in self.screening.recovery(batch_id) if p["row"] == screening_row), None)
            if record is None or target is None:
                raise ServiceError(400, "Select an imported record and a currently eligible recovery target")
            if self.jobs.is_active(target["record_id"]):
                raise ServiceError(409, "This paper has an active extraction job")
            try:
                source, digest = self.inspect_pdf(record["pdf_path"])
                if digest != record["sha256"]:
                    raise ServiceError(409, "PDF changed since preview; import the list again")
                with source.open("rb") as handle:
                    content = handle.read(self.configuration.max_pdf_bytes + 1)
                if len(content) > self.configuration.max_pdf_bytes or hashlib.sha256(content).hexdigest() != digest:
                    raise ServiceError(409, "PDF changed while linking; import the list again")
                destination = pdf_source(self.configuration.paths.input_dir, target["pdf_filename"])
                if destination.exists():
                    if file_hash(destination) != digest:
                        raise ServiceError(409, "A different PDF is already attached to this screening record")
                else:
                    fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".upload")
                    try:
                        with os.fdopen(fd, "wb") as handle:
                            handle.write(content)
                        os.link(temporary, destination)
                    finally:
                        Path(temporary).unlink(missing_ok=True)
            except (OSError, ValueError) as exc:
                raise ServiceError(400, str(exc)) from exc
            path = directory / "pdf_links.json"
            links = read_json(path) if path.exists() else {}
            link = {"record_id": target["record_id"], "screening_row": screening_row,
                    "pdf_filename": target["pdf_filename"], "original_path": record["pdf_path"],
                    "sha256": digest, "import_id": import_id, "library_row": library_row,
                    "citation": {key: record[key] for key in ("title", "authors", "year", "doi")},
                    "match_method": record["match_method"] if record["match_status"] == "proposed" and any(c["row"] == screening_row for c in record["candidates"]) else "manual", "confirmed_at": datetime.now(timezone.utc).isoformat()}
            links[str(screening_row)] = link
            write_json_atomic(path, links)
            return link
