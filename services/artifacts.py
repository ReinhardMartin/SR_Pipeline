import json
from pathlib import Path

from core.extraction_schema import field_errors
from core.parser import parse_current, parsed_directory
from core.provenance import (
    artifact_hash,
    build_manifest,
    manifest_current,
    write_manifest,
)
from core.settings import resolve_path

from services.configuration import Configuration
from services.errors import ServiceError
from services.models import Models


class Artifacts:
    def __init__(self, configuration: Configuration, models: Models):
        self.configuration = configuration
        self.models = models

    def paper_pdf(self, stem: str) -> Path | None:
        matches = [
            p
            for p in self.configuration.paths.input_dir.glob("*")
            if p.is_file() and p.stem == stem and (p.suffix.lower() == ".pdf")
        ]
        return matches[0] if len(matches) == 1 else None

    def paper_dir(self, stem: str) -> Path | None:
        directory = parsed_directory(stem, self.configuration.paths.output_dir)
        return directory if (directory / f"{stem}.md").is_file() else None

    def parsed_current(self, stem: str, doc_dir: Path) -> bool:
        pdf = self.paper_pdf(stem)
        return pdf is not None and parse_current(pdf, doc_dir)

    def stage_files(
        self, stage: str, stem: str, doc_dir: Path
    ) -> tuple[Path, dict[str, Path]]:
        suffixes = {
            "index": ("chunks.json", "embeddings.npy"),
            "retrieve": ("evidence.json", "retrieve_debug.json"),
            "extract": ("extraction.json",),
            "extract_full": ("full_extraction.json",),
        }
        outputs = {suffix: doc_dir / f"{stem}.{suffix}" for suffix in suffixes[stage]}
        if stage == "index":
            outputs["keywords"] = doc_dir / "keyword_index"
        name = {"extract": "extraction", "extract_full": "full_extraction"}.get(
            stage, stage
        )
        return (doc_dir / f"{stem}.{name}.manifest.json", outputs)

    def stage_manifest(self, stage: str, stem: str, doc_dir: Path) -> dict:
        cfg = self.configuration.load_config()
        md = doc_dir / f"{stem}.md"
        parse_meta = doc_dir / f"{stem}.parse.manifest.json"
        llm = {key: value for key, value in cfg["llm"].items() if key != "api_key_env"}
        match stage:
            case "index":
                inputs = {
                    "markdown": md,
                    "parse": parse_meta,
                    **{
                        name: self.configuration.paths.base / "core" / f"{name}.py"
                        for name in ("chunking", "encoders", "keyword_search")
                    },
                }
                settings = {
                    "chunking": cfg["chunking"],
                    "embedding": cfg["embedding"],
                    "keyword_searcher": cfg["retrieval"].get(
                        "keyword_searcher", "bm25"
                    ),
                    "language": cfg["retrieval"].get("language", "english"),
                }
            case "retrieve":
                inputs = {
                    "index": self.stage_files("index", stem, doc_dir)[0],
                    "plan": self.configuration.paths.query_plan,
                    "schema": self.configuration.paths.data_table,
                    "code": self.configuration.paths.base / "core/retriever.py",
                    "reranker": self.configuration.paths.base / "core/reranker.py",
                }
                settings = {"retrieval": cfg["retrieval"], "reranker": cfg["reranker"]}
            case "extract" | "extract_full":
                inputs = {
                    "schema": self.configuration.paths.data_table,
                    "validation": self.configuration.paths.base
                    / "core/extraction_schema.py",
                }
                settings = {
                    "llm": llm,
                    "prompts": artifact_hash(resolve_path(cfg["paths"]["prompts"])),
                }
                if stage == "extract":
                    inputs.update(
                        retrieval=self.stage_files("retrieve", stem, doc_dir)[0],
                        code=self.configuration.paths.base / "llm/extractor.py",
                    )
                else:
                    inputs.update(
                        markdown=md,
                        parse=parse_meta,
                        code=self.configuration.paths.base / "llm/full_extractor.py",
                    )
                    settings["full_extraction"] = cfg["full_extraction"]
            case _:
                raise ValueError(f"Unknown stage: {stage}")
        return build_manifest(stage, inputs, settings)

    def artifact_current(self, stage: str, stem: str, doc_dir: Path) -> bool:
        path, outputs = self.stage_files(stage, stem, doc_dir)
        try:
            return manifest_current(
                path, self.stage_manifest(stage, stem, doc_dir), outputs
            )
        except (OSError, ValueError):
            return False

    def status(self, stem: str) -> dict:
        doc_dir = self.paper_dir(stem)
        parsed = doc_dir is not None and self.parsed_current(stem, doc_dir)
        indexed = parsed and self.artifact_current("index", stem, doc_dir)
        retrieved = indexed and self.artifact_current("retrieve", stem, doc_dir)
        return {
            "stem": stem,
            "has_pdf": self.paper_pdf(stem) is not None,
            "has_parsed": parsed,
            "has_index": indexed,
            "has_evidence": retrieved,
            "has_extraction": retrieved
            and self.artifact_current("extract", stem, doc_dir)
            and self.extraction_complete(doc_dir / f"{stem}.extraction.json"),
            "has_full_extraction": parsed
            and self.artifact_current("extract_full", stem, doc_dir)
            and self.extraction_complete(
                doc_dir / f"{stem}.full_extraction.json", full=True
            ),
        }

    def extraction_complete(self, path: Path, *, full: bool = False) -> bool:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            fields = self.configuration.load_fields()
            return (
                bool(fields)
                and isinstance(data, dict)
                and (set(data) == {f["label"] for f in fields})
                and all(
                    (
                        not field_errors(data[field["label"]], field, full=full)
                        for field in fields
                    )
                )
            )
        except (OSError, ValueError, TypeError):
            return False

    def require_stage(self, stem: str, stage: str) -> Path:
        if not self.status(stem)[f"has_{stage}"]:
            raise RuntimeError(f"Paper has no current {stage} artifacts")
        return self.paper_dir(stem)

    def begin_stage(self, stage: str, stem: str, doc_dir: Path) -> dict:
        manifest = self.stage_manifest(stage, stem, doc_dir)
        self.stage_files(stage, stem, doc_dir)[0].unlink(missing_ok=True)
        return manifest

    def finish_stage(
        self, stage: str, stem: str, doc_dir: Path, manifest: dict
    ) -> None:
        self.models.require_current()
        upstream = {
            "index": "parsed",
            "retrieve": "index",
            "extract": "evidence",
            "extract_full": "parsed",
        }[stage]
        self.require_stage(stem, upstream)
        if (
            self.stage_manifest(stage, stem, doc_dir)["fingerprint"]
            != manifest["fingerprint"]
        ):
            raise RuntimeError("Stage inputs or settings changed during execution")
        path, outputs = self.stage_files(stage, stem, doc_dir)
        write_manifest(path, manifest, outputs)

    def require_directory(self, stem: str) -> Path:
        doc_dir = self.paper_dir(stem)
        if doc_dir is None:
            raise ServiceError(404, f"Paper '{stem}' not found")
        return doc_dir
