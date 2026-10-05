from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from schemas import BulkRunResult, JobCreated, PaperStatus, UploadResult
from services.files import read_json
from services.runtime import Services

from routes.dependencies import get_services

router = APIRouter()


@router.post("/upload", response_model=UploadResult)
async def upload(
    file: UploadFile = File(...), *, services: Services = Depends(get_services)
):
    filename = Path((file.filename or "").replace("\\", "/")).name
    if not filename or Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(400, "Only PDF files are accepted")
    data = await file.read(services.configuration.max_pdf_bytes + 1)
    if len(data) > services.configuration.max_pdf_bytes:
        raise HTTPException(413, "PDF exceeds the configured upload limit")
    if not data.startswith(b"%PDF-"):
        raise HTTPException(400, "Uploaded file does not have a PDF signature")
    services.configuration.paths.input_dir.mkdir(parents=True, exist_ok=True)
    dest = services.configuration.paths.input_dir / filename
    if services.jobs.is_active(dest.stem):
        raise HTTPException(
            409, "Cannot replace a PDF while its pipeline job is active"
        )
    dest.write_bytes(data)
    return UploadResult(stem=Path(filename).stem)


@router.get("/papers", response_model=list[PaperStatus])
async def list_papers(*, services: Services = Depends(get_services)):
    stems: set[str] = set()
    if services.configuration.paths.output_dir.exists():
        stems.update(
            (
                d.name
                for d in services.configuration.paths.output_dir.iterdir()
                if d.is_dir()
            )
        )
    if services.configuration.paths.input_dir.exists():
        stems.update(
            (
                f.stem
                for f in services.configuration.paths.input_dir.iterdir()
                if f.suffix.lower() == ".pdf"
            )
        )
    return [services.artifacts.status(s) for s in sorted(stems)]


@router.post("/papers/{stem}/parse", response_model=JobCreated)
async def run_parse(stem: str, *, services: Services = Depends(get_services)):
    pdf_path = services.artifacts.paper_pdf(stem)
    if pdf_path is None:
        raise HTTPException(404, f"No PDF found for '{stem}'")
    job_id = services.jobs.start(
        stem, "parsing", lambda _job_id: services.extraction.parse(stem, pdf_path)
    )
    return JobCreated(job_id=job_id)


@router.post("/papers/{stem}/index", response_model=JobCreated)
async def run_index(stem: str, *, services: Services = Depends(get_services)):
    if not services.artifacts.status(stem)["has_parsed"]:
        raise HTTPException(400, "Parse the paper first")
    job_id = services.jobs.start(
        stem, "indexing", lambda _job_id: services.extraction.index(stem)
    )
    return JobCreated(job_id=job_id)


@router.post("/papers/{stem}/retrieve", response_model=JobCreated)
async def run_retrieve(stem: str, *, services: Services = Depends(get_services)):
    if not services.artifacts.status(stem)["has_index"]:
        raise HTTPException(400, "Index the paper first")
    if not services.configuration.paths.query_plan.exists():
        raise HTTPException(400, "Generate or write a query plan first")
    job_id = services.jobs.start(
        stem, "retrieving", lambda _job_id: services.extraction.retrieve(stem)
    )
    return JobCreated(job_id=job_id)


@router.post("/papers/{stem}/extract", response_model=JobCreated)
async def run_extract(stem: str, *, services: Services = Depends(get_services)):
    if not services.artifacts.status(stem)["has_evidence"]:
        raise HTTPException(400, "Retrieve first")
    job_id = services.jobs.start(
        stem, "extracting", lambda _job_id: services.extraction.extract(stem)
    )
    return JobCreated(job_id=job_id)


@router.post("/papers/{stem}/extract-full", response_model=JobCreated)
async def run_extract_full(stem: str, *, services: Services = Depends(get_services)):
    if not services.artifacts.status(stem)["has_parsed"]:
        raise HTTPException(400, "Parse the paper first")
    job_id = services.jobs.start(
        stem, "full extraction", lambda _job_id: services.extraction.extract_full(stem)
    )
    return JobCreated(job_id=job_id)


@router.post("/papers/{stem}/run", response_model=JobCreated)
async def run_all(stem: str, *, services: Services = Depends(get_services)):
    pdf_path = services.artifacts.paper_pdf(stem)
    if pdf_path is None:
        raise HTTPException(404, f"No PDF found for '{stem}'")
    if not services.configuration.paths.query_plan.exists():
        raise HTTPException(400, "Generate or write a query plan first")
    job_id = services.jobs.start(
        stem,
        "queued",
        lambda queued_job_id: services.extraction.run_all(
            queued_job_id, stem, pdf_path
        ),
    )
    return JobCreated(job_id=job_id)


@router.post("/papers/run-all", response_model=BulkRunResult)
async def run_all_papers(
    force: bool = False, *, services: Services = Depends(get_services)
):
    return BulkRunResult(**services.extraction.enqueue_all(force))


@router.get("/papers/{stem}/extraction")
async def get_extraction(stem: str, *, services: Services = Depends(get_services)):
    return read_json(
        services.artifacts.require_directory(stem) / f"{stem}.extraction.json"
    )


@router.get("/papers/{stem}/retrieve-debug")
async def get_retrieve_debug(stem: str, *, services: Services = Depends(get_services)):
    return read_json(
        services.artifacts.require_directory(stem) / f"{stem}.retrieve_debug.json"
    )


@router.get("/papers/{stem}/full_extraction")
async def get_full_extraction(stem: str, *, services: Services = Depends(get_services)):
    return read_json(
        services.artifacts.require_directory(stem) / f"{stem}.full_extraction.json"
    )


@router.get("/papers/{stem}/evidence")
async def get_evidence(stem: str, *, services: Services = Depends(get_services)):
    return read_json(
        services.artifacts.require_directory(stem) / f"{stem}.evidence.json"
    )


@router.get("/papers/{stem}/chunks")
async def get_chunks(stem: str, *, services: Services = Depends(get_services)):
    return read_json(services.artifacts.require_directory(stem) / f"{stem}.chunks.json")
