import json
from pathlib import Path

from core.provenance import canonical_hash, write_json_atomic
from core.workspace_lock import workspace_lock

from services.configuration import Configuration


PLAN_SCHEMA_VERSION = "1.0"


class StaleQueryPlanError(ValueError):
    pass


def metadata_path(plan_path: Path) -> Path:
    return plan_path.with_name(f"{plan_path.name}.manifest.json")


def schema_fingerprint(configuration: Configuration) -> str:
    return canonical_hash(configuration.load_data_table())


def save_query_plan(
    configuration: Configuration,
    plan: dict,
    *,
    expected_schema_fingerprint: str | None = None,
) -> None:
    schema_lock = configuration.paths.data_table.with_name(
        f"{configuration.paths.data_table.name}.lock"
    )
    with workspace_lock(schema_lock):
        current_schema_fingerprint = schema_fingerprint(configuration)
        if (
            expected_schema_fingerprint is not None
            and current_schema_fingerprint != expected_schema_fingerprint
        ):
            raise StaleQueryPlanError(
                "The extraction schema changed while the query plan was being generated"
            )
        write_json_atomic(configuration.paths.query_plan, plan)
        write_json_atomic(
            metadata_path(configuration.paths.query_plan),
            {
                "schema_version": PLAN_SCHEMA_VERSION,
                "schema_fingerprint": current_schema_fingerprint,
                "plan_fingerprint": canonical_hash(plan),
            },
        )


def load_current_query_plan(configuration: Configuration) -> dict:
    plan_path = configuration.paths.query_plan
    manifest_path = metadata_path(plan_path)
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StaleQueryPlanError(
            "The query plan has no valid identity; regenerate or save it again"
        ) from exc
    if manifest.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise StaleQueryPlanError("The query plan identity version is unsupported")
    if manifest.get("plan_fingerprint") != canonical_hash(plan):
        raise StaleQueryPlanError("The query plan changed without being saved")
    if manifest.get("schema_fingerprint") != schema_fingerprint(configuration):
        raise StaleQueryPlanError(
            "The extraction schema changed; regenerate the query plan"
        )
    return plan


def delete_query_plan(configuration: Configuration) -> None:
    configuration.paths.query_plan.unlink(missing_ok=True)
    metadata_path(configuration.paths.query_plan).unlink(missing_ok=True)
