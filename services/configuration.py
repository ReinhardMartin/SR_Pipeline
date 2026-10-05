import json
from pathlib import Path
from typing import Any

from core.provenance import write_json_atomic, write_text_atomic
from core.settings import read_config, resolve_path
from core.workspace_lock import workspace_lock
from screening.domain.models import ScreeningCriteriaDocument, ScreeningPrompts

from services.paths import AppPaths


class Configuration:
    def __init__(self, paths: AppPaths | None = None):
        self.paths = paths or AppPaths.from_config(read_config())
        self.max_pdf_bytes = read_config(self.paths.config)["uploads"]["max_pdf_bytes"]
        self.max_screening_bytes = read_config(self.paths.config)["uploads"][
            "max_screening_bytes"
        ]

    def load_config(self) -> dict:
        return read_config(self.paths.config)

    @staticmethod
    def _save(path: Path, value: Any) -> None:
        lock_path = path.with_name(f"{path.name}.lock")
        with workspace_lock(lock_path):
            write_json_atomic(path, value)

    def save_config(self, cfg: dict) -> None:
        self._save(self.paths.config, cfg)

    def load_data_table(self) -> dict:
        return json.loads(self.paths.data_table.read_text(encoding="utf-8"))

    def load_fields(self) -> list[dict]:
        fields = []
        for group in self.load_data_table()["groups"]:
            for field in group["fields"]:
                fields.append({**field, "group": group["label"]})
        return fields

    def save_data_table(self, data_table: dict) -> None:
        self._save(self.paths.data_table, data_table)

    def load_screening_criteria(self) -> list[dict]:
        if not self.paths.criteria.exists():
            return []
        document = ScreeningCriteriaDocument.model_validate_json(
            self.paths.criteria.read_text(encoding="utf-8")
        )
        return [criterion.model_dump() for criterion in document.criteria]

    def save_screening_criteria(self, criteria: list[dict]) -> None:
        document = ScreeningCriteriaDocument(
            schema_version="2.2", criteria=criteria
        )
        self._save(self.paths.criteria, document.model_dump())

    def screening_prompt_paths(self) -> dict[str, Path]:
        screening = self.load_config()["screening"]
        return {
            "all_criteria": resolve_path(screening["llm"]["instructions"]),
            "per_criterion": resolve_path(
                screening["llm"]["criterion_instructions"]
            ),
            "judge": resolve_path(screening["panel"]["judge_instructions"]),
        }

    def load_screening_prompts(self) -> dict[str, str]:
        return {
            name: path.read_text(encoding="utf-8")
            for name, path in self.screening_prompt_paths().items()
        }

    def save_screening_prompts(self, prompts: ScreeningPrompts) -> None:
        paths = self.screening_prompt_paths()
        if len(set(paths.values())) != len(paths):
            raise ValueError(
                "Each editable screening prompt must use a different file path"
            )
        lock_path = self.paths.config.with_name("screening-prompts.lock")
        with workspace_lock(lock_path):
            for name, path in paths.items():
                write_text_atomic(path, getattr(prompts, name))
