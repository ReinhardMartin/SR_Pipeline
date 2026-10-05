import asyncio
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypeVar

from screening.domain.calibration import calibrate_thresholds
from core.provenance import build_manifest, write_json_atomic
from screening.domain.policy import Screener
from screening.domain.models import (
    ScreeningAgreement,
    ScreeningCriterionLabelRequest,
    ScreeningCriteriaDocument,
    ScreeningLabelRequest,
    ScreeningPanelReviewRequest,
    ScreeningThresholdRequest,
)
from schemas import StatusResult

from services.configuration import Configuration
from services.errors import ServiceError
from screening.application.service import Screening
from screening.application.panel import ScreeningPanel
from screening.infrastructure.repository import ScreeningBatchRepository


T = TypeVar("T")


class Review:
    def __init__(
        self,
        configuration: Configuration,
        screening: Screening,
        batches: ScreeningBatchRepository | None = None,
    ):
        self.configuration = configuration
        self.screening = screening
        self.batches = batches or ScreeningBatchRepository(configuration)

    def _batch_directory(self, batch_id: str) -> Path:
        try:
            return self.batches.paths(batch_id).directory
        except ValueError as error:
            raise ServiceError(400, "Invalid batch ID") from error

    async def _run_locked(self, directory: Path, operation: Callable[[], T]) -> T:
        def run() -> T:
            with self.batches.locked(directory.name):
                return operation()

        try:
            return await asyncio.to_thread(run)
        except RuntimeError as error:
            if "Another pipeline process owns" not in str(error):
                raise
            raise ServiceError(409, str(error)) from error

    async def label_panel_paper(
        self,
        batch_id: str,
        row: int,
        role: Literal["second_reviewer", "judge"],
        body: ScreeningPanelReviewRequest,
    ) -> StatusResult:
        directory = self._batch_directory(batch_id)

        await self._run_locked(
            directory,
            lambda: ScreeningPanel(self.configuration).submit(
                directory, row, role, body
            ),
        )
        return StatusResult(status="saved")

    async def run_panel_judges(
        self, batch_id: str, row: int | None = None
    ) -> None:
        directory = self._batch_directory(batch_id)
        rows = {row} if row is not None else None
        await self._run_locked(
            directory,
            lambda: ScreeningPanel(self.configuration).run_judges(
                directory, rows
            ),
        )

    async def label_screening_paper(
        self, batch_id: str, row: int, body: ScreeningLabelRequest
    ) -> StatusResult:
        directory = self._batch_directory(batch_id)

        def save() -> None:
            path = directory / "results.json"
            if not path.exists():
                raise ServiceError(404, "No results yet; run screening first")
            results = self.screening.validate_results(
                json.loads(path.read_text(encoding="utf-8"))
            )
            match = next((p for p in results if p["row"] == row), None)
            if match is None:
                raise ServiceError(404, f"No paper at row {row}")
            if match.get("consensus") is not None:
                raise ServiceError(
                    409, "Use the assigned panel reviewer or judge controls"
                )
            match["human_decision"] = body.human_decision
            match["human_labelled_at"] = (
                datetime.now(timezone.utc).isoformat()
                if body.human_decision
                else None
            )
            write_json_atomic(path, results)

        await self._run_locked(directory, save)
        return StatusResult(status="saved")

    async def label_screening_criterion(
        self,
        batch_id: str,
        row: int,
        criterion_index: int,
        body: ScreeningCriterionLabelRequest,
    ) -> StatusResult:
        directory = self._batch_directory(batch_id)

        def save() -> None:
            path = directory / "results.json"
            if not path.exists():
                raise ServiceError(404, "No results yet; run screening first")
            results = self.screening.validate_results(
                json.loads(path.read_text(encoding="utf-8"))
            )
            paper = next((item for item in results if item["row"] == row), None)
            if paper is None:
                raise ServiceError(404, f"No paper at row {row}")
            if criterion_index < 0 or criterion_index >= len(
                paper.get("criteria", [])
            ):
                raise ServiceError(404, "Criterion not found")
            criterion = paper["criteria"][criterion_index]
            allowed = (
                {"passed", "failed", "unresolved"}
                if criterion["type"] == "include"
                else {"hit", "cleared", "unresolved"}
            )
            if (
                body.human_decision is not None
                and body.human_decision not in allowed
            ):
                raise ServiceError(
                    400, "Human decision is not valid for this criterion type"
                )
            if body.human_finding is not None:
                expected = {
                    "supported": (
                        "passed" if criterion["type"] == "include" else "hit"
                    ),
                    "contradicted": (
                        "failed"
                        if criterion["type"] == "include"
                        else "cleared"
                    ),
                    "not_established": "unresolved",
                    "uncertain": "unresolved",
                }[body.human_finding]
                if body.human_decision != expected:
                    raise ServiceError(
                        400, "Reviewer finding and decision do not match"
                    )
            criterion["human_finding"] = body.human_finding
            criterion["human_decision"] = body.human_decision
            criterion["human_labelled_at"] = (
                datetime.now(timezone.utc).isoformat()
                if body.human_decision
                else None
            )
            write_json_atomic(path, results)

        await self._run_locked(directory, save)
        return StatusResult(status="saved")

    async def get_screening_agreement(self, batch_id: str) -> ScreeningAgreement:
        path = self._batch_directory(batch_id) / "results.json"
        if not path.exists():
            raise ServiceError(404, "No results yet; run screening first")
        results = self.screening.validate_results(
            json.loads(path.read_text(encoding="utf-8"))
        )
        labels = ["Include", "Likely include", "Exclude", "Maybe"]
        confusion = {h: {m: 0 for m in labels} for h in labels}
        reviewed = agreeing = 0
        for p in results:
            h = p.get("human_decision")
            if not h:
                continue
            decision = p.get("decision")
            if decision not in labels:
                continue
            reviewed += 1
            confusion[h][decision] += 1
            if h == decision:
                agreeing += 1
        return ScreeningAgreement(
            reviewed=reviewed,
            agreeing=agreeing,
            agreement_rate=agreeing / reviewed if reviewed else None,
            confusion=confusion,
        )

    async def get_screening_calibration(self, batch_id: str) -> dict:
        batch_dir = self._batch_directory(batch_id)
        path = batch_dir / "results.json"
        if not path.exists():
            raise ServiceError(404, "No results yet; run screening first")
        papers = self.screening.validate_results(
            json.loads(path.read_text(encoding="utf-8"))
        )
        if any(paper.get("backend") == "llm" for paper in papers):
            raise ServiceError(
                400, "Threshold calibration is only available for NLI results"
            )
        cfg = self.configuration.load_config()["screening"]
        criteria_path = batch_dir / "criteria.json"
        if not criteria_path.is_file():
            raise ServiceError(409, "Batch criteria snapshot is missing; rerun screening")
        criteria = [
            item.model_dump()
            for item in ScreeningCriteriaDocument.model_validate_json(
                criteria_path.read_text(encoding="utf-8")
            ).criteria
        ]
        thresholds = {
            criterion["label"]: criterion.get(
                "confidence_threshold", cfg["confidence_threshold"]
            )
            for criterion in criteria
        }
        return await asyncio.to_thread(
            calibrate_thresholds,
            papers,
            current_thresholds=thresholds,
            default_threshold=cfg["confidence_threshold"],
            options=cfg.get("calibration", {}),
        )

    async def apply_screening_calibration(
        self, batch_id: str, body: ScreeningThresholdRequest
    ) -> StatusResult:
        batch_dir = self._batch_directory(batch_id)
        return await self._run_locked(
            batch_dir,
            lambda: self._apply_screening_calibration_unlocked(batch_dir, body),
        )

    def _apply_screening_calibration_unlocked(
        self, batch_dir: Path, body: ScreeningThresholdRequest
    ) -> StatusResult:
        if (batch_dir / "panel.json").exists():
            raise ServiceError(
                409,
                "Threshold calibration cannot rewrite a panel-managed run; "
                "disable the panel and rerun screening before calibrating",
            )
        results_path = batch_dir / "results.json"
        if not results_path.exists():
            raise ServiceError(404, "No results yet; run screening first")
        criteria_path = batch_dir / "criteria.json"
        if not criteria_path.is_file():
            raise ServiceError(409, "Batch criteria snapshot is missing; rerun screening")
        criteria = [
            item.model_dump()
            for item in ScreeningCriteriaDocument.model_validate_json(
                criteria_path.read_text(encoding="utf-8")
            ).criteria
        ]
        labels = {criterion["label"] for criterion in criteria}
        unknown = set(body.criterion_thresholds) - labels
        if unknown:
            raise ServiceError(
                400, f"Unknown criterion label(s): {', '.join(sorted(unknown))}"
            )
        invalid = {
            label: value
            for label, value in body.criterion_thresholds.items()
            if not 0.5 <= value <= 1.0
        }
        if invalid:
            raise ServiceError(
                400, "Every criterion threshold must be between 0.5 and 1.0"
            )
        config = self.configuration.load_config()
        default = config["screening"]["confidence_threshold"]
        thresholds = {
            criterion["label"]: body.criterion_thresholds.get(
                criterion["label"], criterion.get("confidence_threshold", default)
            )
            for criterion in criteria
        }
        results = self.screening.validate_results(
            json.loads(results_path.read_text(encoding="utf-8"))
        )
        if any(paper.get("backend") == "llm" for paper in results):
            raise ServiceError(
                400, "Threshold calibration is only available for NLI results"
            )
        screener = Screener(
            nli=None,
            confidence_threshold=default,
            criterion_thresholds=body.criterion_thresholds,
        )
        reclassified = []
        for paper in results:
            if paper.get("screening_status") == "screened" and paper.get("criteria"):
                decision = screener._decide(paper["criteria"])
                reclassified.append({**paper, **decision})
            else:
                reclassified.append(paper)
        write_json_atomic(results_path, reclassified)
        calibrated_criteria = [
            {
                **criterion,
                "confidence_threshold": thresholds[criterion["label"]],
            }
            for criterion in criteria
        ]
        write_json_atomic(
            criteria_path, {"schema_version": "2.2", "criteria": calibrated_criteria}
        )
        current_criteria = self.configuration.load_screening_criteria()
        self.configuration.save_screening_criteria(
            [
                {
                    **criterion,
                    "confidence_threshold": thresholds.get(
                        criterion["label"], criterion["confidence_threshold"]
                    ),
                }
                for criterion in current_criteria
            ]
        )
        if criteria_path.exists():
            manifest = build_manifest(
                "screen",
                {
                    "papers": batch_dir / "papers.json",
                    "criteria": criteria_path,
                    "screener_code": self.configuration.paths.base
                    / "screening/domain/policy.py",
                },
                {
                    "screening": config["screening"],
                    "criterion_thresholds": thresholds,
                    "nli": config["nli"],
                },
            )
            write_json_atomic(batch_dir / "screening.manifest.json", manifest)
        return StatusResult(status="saved_and_reclassified")
