import asyncio
from contextlib import asynccontextmanager

from core.workspace_lock import workspace_lock

from services.artifacts import Artifacts
from services.configuration import Configuration
from services.exports import Exports
from services.extraction import Extraction
from services.jobs import JobQueue
from services.models import Models
from screening.application.review import Review
from screening.application.recovery import Recovery
from screening.application.service import Screening
from screening.infrastructure.repository import ScreeningBatchRepository


class Services:
    def __init__(
        self, configuration: Configuration | None = None, models: Models | None = None
    ):
        self.configuration = configuration or Configuration()
        self.screening_batches = ScreeningBatchRepository(self.configuration)
        self.models = models or Models(self.configuration)
        self.jobs = JobQueue(self.configuration)
        self.artifacts = Artifacts(self.configuration, self.models)
        self.extraction = Extraction(
            self.configuration, self.models, self.artifacts, self.jobs
        )
        self.exports = Exports(self.configuration, self.artifacts)
        self.screening = Screening(
            self.configuration, self.models, self.screening_batches
        )
        self.review = Review(
            self.configuration, self.screening, self.screening_batches
        )
        self.recovery = Recovery(
            self.configuration, self.screening, self.jobs, self.screening_batches
        )

    @asynccontextmanager
    async def lifespan(self, app):
        with workspace_lock(self.configuration.paths.jobs.with_suffix(".lock")):
            config = self.configuration.load_config()
            self.jobs.semaphore = asyncio.Semaphore(
                config["execution"]["max_concurrent_papers"]
            )
            self.jobs.history_limit = config["execution"]["job_history_limit"]
            self.jobs.restore()
            try:
                self.models.start(config)
                yield
            finally:
                if self.jobs.tasks:
                    await asyncio.gather(*self.jobs.tasks, return_exceptions=True)
                self.models.close()
