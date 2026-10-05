from fastapi import APIRouter, Depends, HTTPException
from schemas import JobStatus
from services.runtime import Services

from routes.dependencies import get_services

router = APIRouter()


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def job_status(job_id: str, *, services: Services = Depends(get_services)):
    if job_id not in services.jobs.records:
        raise HTTPException(404, "Job not found")
    return services.jobs.records[job_id]
