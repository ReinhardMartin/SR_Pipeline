from dataclasses import dataclass
from pathlib import Path

from core.settings import CONFIG_PATH, ROOT, resolve_path


@dataclass
class AppPaths:
    base: Path
    config: Path
    input_dir: Path
    output_dir: Path
    data_table: Path
    query_plan: Path
    jobs: Path
    screening: Path
    criteria: Path

    @classmethod
    def from_config(cls, config: dict):
        paths = config["paths"]
        return cls(
            base=ROOT,
            config=CONFIG_PATH,
            input_dir=resolve_path(paths["input"]),
            output_dir=resolve_path(paths["output"]),
            data_table=resolve_path(paths["data_table"]),
            query_plan=resolve_path(paths["query_plan"]),
            jobs=resolve_path(paths["jobs"]),
            screening=resolve_path(paths["screening"]),
            criteria=resolve_path(paths["criteria"]),
        )
