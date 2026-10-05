import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.config:
        os.environ["PIPELINE_CONFIG"] = str(args.config.resolve())

    from core.settings import (
        ParseConfig,
        load_prompt,
        read_config,
        render_prompt,
        resolve_path,
    )
    from schemas import DataTable, ScreeningCriteria

    cfg = read_config()
    for name in ("schema", "criteria", "parser", "mineru"):
        key = "data_table" if name == "schema" else name
        path = resolve_path(cfg["paths"][key])
        if not path.is_file():
            parser.error(f"Missing {name} file: {cfg['paths'][key]}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if name == "schema":
            DataTable.model_validate(data)
        elif name == "criteria":
            criteria = ScreeningCriteria.model_validate(data)
            if not any(item.type == "include" for item in criteria.criteria):
                parser.error("At least one inclusion criterion is required")
        elif name == "parser":
            ParseConfig.model_validate(data)

    for prompt in (
        "extraction_single_field_system",
        "extraction_field_group_system",
        "extraction_full_document_system",
        "extraction_query_planner_system",
    ):
        load_prompt(prompt)

    screening = cfg["screening"]
    prompt_paths = []
    if screening["backend"] == "llm":
        key = (
            "criterion_instructions"
            if screening["llm"]["execution_mode"] == "per_criterion"
            else "instructions"
        )
        prompt_paths.append(("primary screening", screening["llm"][key]))
    if screening["panel"]["enabled"]:
        second = screening["panel"]["second_reviewer"]
        if second["kind"] == "llm":
            prompt_paths.append(("second reviewer", second["llm"]["instructions"]))
        if screening["panel"]["judge"]["kind"] == "llm":
            prompt_paths.append(
                ("panel judge", screening["panel"]["judge_instructions"])
            )
    for label, value in prompt_paths:
        prompt_path = resolve_path(value)
        if not prompt_path.is_file() or not prompt_path.read_text(
            encoding="utf-8"
        ).strip():
            parser.error(f"Missing or empty {label} prompt: {value}")
    for name, values in {
        "planner_field": {"label": "check", "description": "check"},
        "field": {
            "label": "check",
            "description": "check",
            "rules": "",
            "value_type": "string",
            "allowed_values": None,
        },
        "field_id": {"id": "check", "specification": "check"},
        "passage": {"id": "check", "section": "check", "text": "check"},
        "extract": {"specification": "check", "passages": "check"},
        "group": {"label": "check", "specifications": "check", "passages": "check"},
        "full": {"specifications": "check", "paper": "check"},
    }.items():
        render_prompt(name, **values)
    if args.check:
        print("Configuration and required files are valid.")
        return

    import uvicorn

    server = cfg["server"]
    uvicorn.run(
        "api:app",
        app_dir=str(Path(__file__).resolve().parent),
        host=server["host"],
        port=server["port"],
        log_level=server["log_level"],
        workers=1,
        timeout_graceful_shutdown=server["shutdown_timeout"],
    )


if __name__ == "__main__":
    main()
