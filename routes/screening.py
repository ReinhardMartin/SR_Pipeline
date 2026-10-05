import asyncio
import json
from pathlib import Path

from screening.application.handoff import RECOVERY_COLUMNS
from screening.infrastructure.citations import TEMPLATE_COLUMNS, csv_bytes, read_citations
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response
from schemas import (
    JobCreated,
    RecoveryLinkRequest,
    StatusResult,
)
from screening.domain.models import (
    ScreeningBatchInfo,
    ScreeningCriteria,
    ScreeningCriteriaDocument,
    ScreeningResults,
    ScreeningPrompts,
    ScreeningUploadResult,
)
from services.runtime import Services
from screening.infrastructure.repository import (
    InvalidBatchId,
    UnsafeBatchContents,
)

from routes.dependencies import get_services

router = APIRouter()


def _batch_directory(services: Services, batch_id: str) -> Path:
    try:
        return services.screening_batches.paths(batch_id).directory
    except InvalidBatchId as exc:
        raise HTTPException(400, "Invalid screening batch ID") from exc


@router.get("/screening/criteria", response_model=ScreeningCriteria)
async def get_screening_criteria(*, services: Services = Depends(get_services)):
    return ScreeningCriteria(criteria=services.configuration.load_screening_criteria())


@router.put("/screening/criteria", response_model=StatusResult)
async def put_screening_criteria(
    body: ScreeningCriteria, *, services: Services = Depends(get_services)
):
    labels = [c.label.strip() for c in body.criteria]
    if len(labels) != len(set(labels)):
        raise HTTPException(400, "Screening criterion labels must be unique")
    if not any((c.type == "include" for c in body.criteria)):
        raise HTTPException(400, "At least one inclusion criterion is required")
    services.configuration.save_screening_criteria(
        [c.model_dump() for c in body.criteria]
    )
    return StatusResult(status="saved")


@router.get("/screening/prompts", response_model=ScreeningPrompts)
async def get_screening_prompts(*, services: Services = Depends(get_services)):
    return ScreeningPrompts(**services.configuration.load_screening_prompts())


@router.put("/screening/prompts", response_model=StatusResult)
async def put_screening_prompts(
    body: ScreeningPrompts, *, services: Services = Depends(get_services)
):
    if any(
        job["status"] in {"running", "queued"}
        for job in services.jobs.records.values()
    ):
        raise HTTPException(
            409, "Wait for active jobs to finish before changing screening prompts"
        )
    try:
        services.configuration.save_screening_prompts(body)
    except (OSError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return StatusResult(status="saved")


@router.get("/screening/batches", response_model=list[ScreeningBatchInfo])
async def list_screening_batches(*, services: Services = Depends(get_services)):
    return [
        ScreeningBatchInfo(
            batch_id=batch.batch_id,
            filename=batch.filename,
            paper_count=batch.paper_count,
            has_results=batch.has_results,
        )
        for batch in services.screening_batches.list()
    ]


@router.post("/screening/upload", response_model=ScreeningUploadResult)
async def upload_screening(
    file: UploadFile = File(...), *, services: Services = Depends(get_services)
):
    filename = Path((file.filename or "").replace("\\", "/")).name
    if Path(filename).suffix.lower() not in {".ris", ".csv", ".xlsx"}:
        raise HTTPException(400, "Upload a .ris, .csv, or .xlsx file")
    data = await file.read(services.configuration.max_screening_bytes + 1)
    if len(data) > services.configuration.max_screening_bytes:
        raise HTTPException(413, "Citation file exceeds the configured upload limit")
    try:
        papers, columns = await asyncio.to_thread(read_citations, data, filename)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not papers:
        raise HTTPException(
            400, "No papers found ; every row is missing both title and abstract"
        )
    batch = services.screening_batches.create(
        filename=filename, source=data, papers=papers
    )
    return ScreeningUploadResult(
        batch_id=batch.batch_id,
        filename=filename,
        paper_count=len(papers),
        columns=columns,
    )


@router.get("/screening/template.csv")
async def screening_template(*, services: Services = Depends(get_services)):
    return Response(
        content=csv_bytes(TEMPLATE_COLUMNS, []),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=screening_template.csv"},
    )


@router.get("/screening/{batch_id}/recovery")
async def get_screening_recovery(
    batch_id: str, *, services: Services = Depends(get_services)
):
    batch_id = _batch_directory(services, batch_id).name
    return {
        "folder": str(services.configuration.paths.input_dir),
        "papers": services.screening.recovery(batch_id),
    }


@router.get("/screening/{batch_id}/recovery.csv")
async def download_screening_recovery(
    batch_id: str, *, services: Services = Depends(get_services)
):
    batch_id = _batch_directory(services, batch_id).name
    papers = services.screening.recovery(batch_id)
    return Response(
        content=csv_bytes(
            RECOVERY_COLUMNS,
            ([paper.get(key, "") for key in RECOVERY_COLUMNS] for paper in papers),
        ),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=papers_to_recover.csv"},
    )


@router.post("/screening/{batch_id}/run", response_model=JobCreated)
async def run_screening(batch_id: str, *, services: Services = Depends(get_services)):
    batch_dir = _batch_directory(services, batch_id)
    batch_id = batch_dir.name
    if not (batch_dir / "papers.json").exists():
        raise HTTPException(404, "Batch not found")
    if not services.configuration.load_screening_criteria():
        raise HTTPException(400, "Define screening criteria first")
    job_id = services.jobs.start(
        f"screening:{batch_id}", "screening", lambda _: services.screening.run(batch_id)
    )
    return JobCreated(job_id=job_id)


@router.get("/screening/{batch_id}/results", response_model=ScreeningResults)
async def get_screening_results(
    batch_id: str, *, services: Services = Depends(get_services)
):
    batch_dir = _batch_directory(services, batch_id)
    batch_id = batch_dir.name
    path = batch_dir / "results.json"
    if not path.exists():
        raise HTTPException(404, "No results yet ; run screening first")
    criteria_path = batch_dir / "criteria.json"
    if not criteria_path.is_file():
        raise HTTPException(409, "Batch criteria snapshot is missing; rerun screening")
    criteria = [
        item.model_dump()
        for item in ScreeningCriteriaDocument.model_validate_json(
            criteria_path.read_text(encoding="utf-8")
        ).criteria
    ]
    return ScreeningResults(
        batch_id=batch_id,
        criteria=criteria,
        papers=services.screening.validate_results(
            json.loads(path.read_text(encoding="utf-8"))
        ),
    )


@router.get("/screening/{batch_id}/results.xlsx")
async def download_screening_results(
    batch_id: str, *, services: Services = Depends(get_services)
):
    batch_dir = _batch_directory(services, batch_id)
    batch_id = batch_dir.name
    path = batch_dir / "results.json"
    if not path.exists():
        raise HTTPException(404, "No results yet ; run screening first")
    results = services.screening.validate_results(
        json.loads(path.read_text(encoding="utf-8"))
    )
    criteria_path = batch_dir / "criteria.json"
    if not criteria_path.is_file():
        raise HTTPException(409, "Batch criteria snapshot is missing; rerun screening")
    criteria = [
        item.model_dump()
        for item in ScreeningCriteriaDocument.model_validate_json(
            criteria_path.read_text(encoding="utf-8")
        ).criteria
    ]
    content = await asyncio.to_thread(
        services.screening.export_excel, results, criteria
    )
    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f"attachment; filename=screening_results_{batch_id[:8]}.xlsx"
        },
    )


@router.delete("/screening/{batch_id}", response_model=StatusResult)
async def delete_screening_batch(
    batch_id: str, *, services: Services = Depends(get_services)
):
    batch_dir = _batch_directory(services, batch_id)
    batch_id = batch_dir.name
    if not batch_dir.exists():
        raise HTTPException(404, "Batch not found")
    if services.jobs.is_active(f"screening:{batch_id}"):
        raise HTTPException(
            409, "Wait for screening to finish before deleting this batch"
        )
    try:
        services.screening_batches.delete(batch_id)
    except (RuntimeError, UnsafeBatchContents) as exc:
        raise HTTPException(409, str(exc)) from exc
    return StatusResult(status="deleted")


@router.get("/screening/{batch_id}/recovery/template.csv")
async def recovery_template(batch_id: str):
    from core.recovery_import import LIBRARY_COLUMNS

    return Response(
        csv_bytes(LIBRARY_COLUMNS, []),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=reviewer_library.csv"},
    )


@router.post("/screening/{batch_id}/recovery/import")
async def import_recovery(
    batch_id: str,
    file: UploadFile = File(...),
    *,
    services: Services = Depends(get_services),
):
    batch_id = _batch_directory(services, batch_id).name
    data = await file.read(services.configuration.max_screening_bytes + 1)
    if len(data) > services.configuration.max_screening_bytes:
        raise HTTPException(413, "Citation list exceeds configured upload limit")
    return await asyncio.to_thread(
        services.recovery.preview, batch_id, data, file.filename or ""
    )


@router.post("/screening/{batch_id}/recovery/link")
async def link_recovery(
    batch_id: str,
    body: RecoveryLinkRequest,
    *,
    services: Services = Depends(get_services),
):
    batch_id = _batch_directory(services, batch_id).name
    return await asyncio.to_thread(
        services.recovery.confirm,
        batch_id,
        body.import_id,
        body.library_row,
        body.screening_row,
    )
