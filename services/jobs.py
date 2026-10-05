import asyncio
import json
import logging
import uuid

from core.provenance import write_json_atomic

from services.configuration import Configuration
from services.errors import ServiceError


logger = logging.getLogger(__name__)


class JobQueue:
    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.records = {}
        self.tasks = set()
        self.semaphore = None
        self.locks = {}
        self.active = {}
        self.history_limit = 500

    def spawn(self, coro):
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    def persist(self) -> None:
        write_json_atomic(self.configuration.paths.jobs, self.records)

    def restore(self) -> None:
        if self.configuration.paths.jobs.exists():
            try:
                restored = json.loads(
                    self.configuration.paths.jobs.read_text(encoding="utf-8")
                )
                if not isinstance(restored, dict) or any(
                    (
                        not isinstance(job, dict)
                        or job.get("status")
                        not in {"queued", "running", "done", "error"}
                        for job in restored.values()
                    )
                ):
                    raise ValueError("Invalid job history")
                self.records.clear()
                self.records.update(restored)
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"Cannot restore job history from {self.configuration.paths.jobs}"
                ) from exc
        interrupted = False
        for job in self.records.values():
            if job.get("status") in {"queued", "running"}:
                job.update(
                    status="error", error="API restarted before this job completed"
                )
                interrupted = True
        while len(self.records) > self.history_limit:
            self.records.pop(next(iter(self.records)))
        if interrupted:
            self.persist()

    def create(self, message: str = "queued") -> str:
        while len(self.records) >= self.history_limit:
            completed = next(
                (
                    key
                    for key, value in self.records.items()
                    if value["status"] in {"done", "error"}
                ),
                None,
            )
            if completed is None:
                break
            self.records.pop(completed, None)
        job_id = str(uuid.uuid4())
        self.records[job_id] = {"status": "queued", "message": message, "error": None}
        self.persist()
        return job_id

    def is_active(self, stem: str) -> bool:
        job_id = self.active.get(stem)
        if job_id and self.records.get(job_id, {}).get("status") in {
            "queued",
            "running",
        }:
            return True
        self.active.pop(stem, None)
        return False

    def start(self, stem: str, message: str, operation) -> str:
        if self.is_active(stem):
            raise ServiceError(
                409, f"A pipeline job is already queued or running for '{stem}'"
            )
        job_id = self.create(message)
        self.active[stem] = job_id
        self.spawn(self.run(job_id, stem, operation))
        return job_id

    async def run(self, job_id: str, stem: str, operation) -> None:
        semaphore = self.semaphore
        if semaphore is None:
            self.records[job_id].update(
                status="error", error="API execution queue is not initialized"
            )
            self.active.pop(stem, None)
            self.persist()
            return
        lock = self.locks.setdefault(stem, asyncio.Lock())
        try:
            async with semaphore:
                async with lock:
                    self.records[job_id]["status"] = "running"
                    self.persist()
                    await operation(job_id)
                    self.records[job_id].update(status="done", message="done")
        except asyncio.CancelledError:
            self.records[job_id].update(
                status="error", error="Job interrupted; rerun to resume"
            )
            raise
        except Exception as exc:
            logger.exception("Pipeline job %s failed for %s", job_id, stem)
            self.records[job_id].update(status="error", error=str(exc))
        finally:
            if self.active.get(stem) == job_id:
                self.active.pop(stem, None)
            self.locks.pop(stem, None)
            self.persist()
