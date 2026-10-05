from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from routes import (
    configuration,
    exports,
    jobs,
    papers,
    review,
    screening,
    system,
    tools,
)
from services.errors import ServiceError
from services.runtime import Services


def create_app(services: Services | None = None) -> FastAPI:
    services = services or Services()
    app = FastAPI(title="Extraction Pipeline", lifespan=services.lifespan)
    app.state.services = services
    app.mount(
        "/static",
        StaticFiles(directory=str(services.configuration.paths.base / "static")),
        name="static",
    )

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    for module in (
        system,
        configuration,
        papers,
        exports,
        jobs,
        tools,
        screening,
        review,
    ):
        app.include_router(module.router)
    return app


app = create_app()
