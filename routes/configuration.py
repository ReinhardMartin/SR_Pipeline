import asyncio

from core.provenance import canonical_hash
from core.settings import PipelineConfig
from fastapi import APIRouter, Depends, HTTPException
from schemas import DataTable, PlanEntry, StatusResult
from services.files import read_json
from services.query_plans import (
    StaleQueryPlanError,
    delete_query_plan,
    save_query_plan,
)
from services.runtime import Services

from routes.dependencies import get_services

router = APIRouter()


@router.get("/config", response_model=PipelineConfig)
async def get_config(*, services: Services = Depends(get_services)):
    return PipelineConfig(**services.configuration.load_config())


@router.put("/config", response_model=StatusResult)
async def put_config(
    cfg: PipelineConfig, *, services: Services = Depends(get_services)
):
    if any(
        (
            job["status"] in {"running", "queued"}
            for job in services.jobs.records.values()
        )
    ):
        raise HTTPException(
            409, "Wait for active jobs to finish before changing configuration"
        )
    new_config = cfg.model_dump()
    restart_required = (
        services.models.current_fingerprint != services.models.fingerprint(new_config)
    )
    services.configuration.save_config(new_config)
    return StatusResult(
        status="saved_restart_required" if restart_required else "saved"
    )


@router.get("/data-table", response_model=DataTable)
async def get_data_table(*, services: Services = Depends(get_services)):
    return DataTable(**services.configuration.load_data_table())


@router.get("/plan")
async def get_plan(*, services: Services = Depends(get_services)):
    return read_json(services.configuration.paths.query_plan)


@router.delete("/plan", response_model=StatusResult)
async def delete_plan(*, services: Services = Depends(get_services)):
    delete_query_plan(services.configuration)
    return StatusResult(status="deleted")


@router.post("/plan/generate")
async def generate_plan(*, services: Services = Depends(get_services)):
    fields = services.configuration.load_fields()
    schema_identity = canonical_hash(services.configuration.load_data_table())
    plan = await asyncio.to_thread(services.models.instances["planner"].plan, fields)
    try:
        save_query_plan(
            services.configuration,
            plan,
            expected_schema_fingerprint=schema_identity,
        )
    except StaleQueryPlanError as exc:
        raise HTTPException(409, str(exc)) from exc
    return plan


@router.put("/plan", response_model=StatusResult)
async def put_plan(
    body: dict[str, PlanEntry], *, services: Services = Depends(get_services)
):
    labels = {f["label"] for f in services.configuration.load_fields()}
    unknown = set(body) - labels
    if unknown:
        raise HTTPException(
            400, f"Unknown field label(s): {', '.join(sorted(unknown))}"
        )
    missing = labels - set(body)
    if missing:
        raise HTTPException(
            400, f"Missing field label(s): {', '.join(sorted(missing))}"
        )
    save_query_plan(
        services.configuration,
        {label: entry.model_dump() for label, entry in body.items()},
    )
    return StatusResult(status="saved")


@router.put("/data-table", response_model=StatusResult)
async def put_data_table(
    body: DataTable, *, services: Services = Depends(get_services)
):
    group_labels = [group.label.strip() for group in body.groups]
    if not group_labels:
        raise HTTPException(400, "Define at least one extraction group")
    if len(group_labels) != len(set(group_labels)):
        raise HTTPException(400, "Extraction group labels must be unique")
    fields = [field for group in body.groups for field in group.fields]
    if any((not group.fields for group in body.groups)):
        raise HTTPException(
            400, "Every extraction group must contain at least one field"
        )
    if not fields:
        raise HTTPException(400, "Define at least one extraction field")
    labels = [field.label.strip() for field in fields]
    ids = [field.id.strip() for field in fields]
    if len(labels) != len(set(labels)):
        raise HTTPException(400, "Extraction field labels must be unique")
    if len(ids) != len(set(ids)):
        raise HTTPException(400, "Extraction field IDs must be unique")
    normalized = {
        "schema_version": "3.0",
        "groups": [
            {
                "label": group.label.strip(),
                "fields": [
                    field.model_dump(exclude_none=True) for field in group.fields
                ],
            }
            for group in body.groups
        ],
    }
    current = DataTable(**services.configuration.load_data_table()).model_dump(
        exclude_none=True
    )
    changed = normalized != current
    had_queries = services.configuration.paths.query_plan.exists()
    services.configuration.save_data_table(normalized)
    if changed:
        delete_query_plan(services.configuration)
    return StatusResult(
        status="saved_queries_invalidated" if changed and had_queries else "saved"
    )
