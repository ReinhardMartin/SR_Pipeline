from fastapi import APIRouter, Depends
from typing import Literal
from screening.domain.models import (
    ScreeningAgreement,
    ScreeningCriterionLabelRequest,
    ScreeningLabelRequest,
    ScreeningPanelReviewRequest,
    ScreeningThresholdRequest,
)
from schemas import JobCreated, StatusResult
from services.runtime import Services

from routes.dependencies import get_services

router = APIRouter()


@router.put("/screening/{batch_id}/papers/{row}/panel/{role}", response_model=StatusResult)
async def review_panel_paper(batch_id: str, row: int, role: Literal["second_reviewer", "judge"],
                             body: ScreeningPanelReviewRequest, *, services: Services = Depends(get_services)):
    return await services.review.label_panel_paper(batch_id, row, role, body)


@router.post(
    "/screening/{batch_id}/papers/{row}/panel/judge/run",
    response_model=JobCreated,
)
async def run_panel_judge(
    batch_id: str,
    row: int,
    *,
    services: Services = Depends(get_services),
):
    directory = services.review._batch_directory(batch_id)
    if not (directory / "panel.json").is_file():
        raise ServiceError(409, "Run screening with the review panel enabled first")
    job_id = services.jobs.start(
        f"screening:{batch_id}",
        "adjudicating screening paper",
        lambda _: services.review.run_panel_judges(batch_id, row),
    )
    return JobCreated(job_id=job_id)


@router.post("/screening/{batch_id}/panel/judge/run", response_model=JobCreated)
async def run_pending_panel_judges(
    batch_id: str,
    *,
    services: Services = Depends(get_services),
):
    directory = services.review._batch_directory(batch_id)
    if not (directory / "panel.json").is_file():
        raise ServiceError(409, "Run screening with the review panel enabled first")
    job_id = services.jobs.start(
        f"screening:{batch_id}",
        "adjudicating pending screening papers",
        lambda _: services.review.run_panel_judges(batch_id),
    )
    return JobCreated(job_id=job_id)


@router.put("/screening/{batch_id}/papers/{row}/label", response_model=StatusResult)
async def label_screening_paper(
    batch_id: str,
    row: int,
    body: ScreeningLabelRequest,
    *,
    services: Services = Depends(get_services),
):
    return await services.review.label_screening_paper(batch_id, row, body)


@router.put(
    "/screening/{batch_id}/papers/{row}/criteria/{criterion_index}/label",
    response_model=StatusResult,
)
async def label_screening_criterion(
    batch_id: str,
    row: int,
    criterion_index: int,
    body: ScreeningCriterionLabelRequest,
    *,
    services: Services = Depends(get_services),
):
    return await services.review.label_screening_criterion(
        batch_id, row, criterion_index, body
    )


@router.get("/screening/{batch_id}/agreement", response_model=ScreeningAgreement)
async def get_screening_agreement(
    batch_id: str, *, services: Services = Depends(get_services)
):
    return await services.review.get_screening_agreement(batch_id)


@router.get("/screening/{batch_id}/calibration")
async def get_screening_calibration(
    batch_id: str, *, services: Services = Depends(get_services)
):
    return await services.review.get_screening_calibration(batch_id)


@router.put("/screening/{batch_id}/calibration/apply", response_model=StatusResult)
async def apply_screening_calibration(
    batch_id: str,
    body: ScreeningThresholdRequest,
    *,
    services: Services = Depends(get_services),
):
    return await services.review.apply_screening_calibration(batch_id, body)
