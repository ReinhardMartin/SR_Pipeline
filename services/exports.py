import json
from typing import Literal

from schemas import ExportResult, FieldResult, PaperExport

from services.artifacts import Artifacts
from services.configuration import Configuration


class Exports:
    def __init__(self, configuration: Configuration, artifacts: Artifacts):
        self.configuration = configuration
        self.artifacts = artifacts

    def stringify_value(self, raw) -> str | None:
        if raw is None:
            return None
        return (
            json.dumps(raw, ensure_ascii=False)
            if isinstance(raw, (dict, list))
            else str(raw)
        )

    def paper_extraction(self, stem: str, source: str) -> dict:
        status_key = "has_full_extraction" if source == "full" else "has_extraction"
        if not self.artifacts.status(stem)[status_key]:
            return {}
        doc_dir = self.artifacts.paper_dir(stem)
        filename = (
            f"{stem}.full_extraction.json"
            if source == "full"
            else f"{stem}.extraction.json"
        )
        path = doc_dir / filename
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def build_export(self, source: Literal["rag", "full"]) -> ExportResult:
        field_labels = [f["label"] for f in self.configuration.load_fields()]
        stems: set[str] = set()
        if self.configuration.paths.output_dir.exists():
            stems.update(
                (
                    d.name
                    for d in self.configuration.paths.output_dir.iterdir()
                    if d.is_dir()
                )
            )
        papers = []
        for stem in sorted(stems):
            data = self.paper_extraction(stem, source)
            fields = {
                label: FieldResult(
                    value=self.stringify_value(data.get(label, {}).get("value")),
                    status=data.get(label, {}).get("status"),
                    confidence=data.get(label, {}).get("confidence"),
                    evidence=[
                        item.get("quote", "")
                        for item in data.get(label, {}).get("evidence", [])
                        if item.get("quote")
                    ]
                    if source == "rag"
                    else [data.get(label, {}).get("source_quote")]
                    if data.get(label, {}).get("source_quote")
                    else [],
                    error=data.get(label, {}).get("error"),
                )
                for label in field_labels
            }
            papers.append(PaperExport(stem=stem, fields=fields))
        return ExportResult(fields=field_labels, papers=papers)
