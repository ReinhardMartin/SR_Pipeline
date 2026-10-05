from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from services.runtime import Services

from routes.dependencies import get_services

router = APIRouter()


@router.get("/health")
async def health(*, services: Services = Depends(get_services)):
    try:
        current = services.models.current_fingerprint == services.models.fingerprint(
            services.configuration.load_config()
        )
    except (OSError, ValueError):
        current = False
    if not current:
        raise HTTPException(503, "Configuration changed; restart required")
    return {"status": "ok"}


@router.get("/", response_class=HTMLResponse)
async def index(*, services: Services = Depends(get_services)):
    return HTMLResponse(
        content=(services.configuration.paths.base / "static/index.html").read_text(
            encoding="utf-8"
        ),
        headers={"Cache-Control": "no-store"},
    )
