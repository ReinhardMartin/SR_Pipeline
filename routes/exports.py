import csv
import io
from typing import Literal

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from schemas import ExportResult
from services.runtime import Services

from routes.dependencies import get_services

router = APIRouter()


@router.get("/export", response_model=ExportResult)
async def export_results(
    source: Literal["rag", "full"] = "rag",
    *,
    services: Services = Depends(get_services),
):
    return services.exports.build_export(source)


@router.get("/export.csv")
async def export_csv(
    source: Literal["rag", "full"] = "rag",
    *,
    services: Services = Depends(get_services),
):
    result = services.exports.build_export(source)
    buf = io.StringIO()
    writer = csv.writer(buf)
    header = ["paper"]
    for label in result.fields:
        header.extend(
            [label, f"{label} status", f"{label} confidence", f"{label} evidence"]
        )
    writer.writerow(header)
    for paper in result.papers:
        row = [paper.stem]
        for label in result.fields:
            fr = paper.fields[label]
            row.extend(
                [
                    fr.value or "",
                    fr.status or "",
                    fr.confidence or "",
                    " | ".join(fr.evidence),
                ]
            )
        writer.writerow(row)
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=extraction_results.csv"},
    )
