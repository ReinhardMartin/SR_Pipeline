import asyncio
import json
import logging
from pathlib import Path

import numpy as np
from core.checkpoint import Checkpoint
from core.parser import parse, parse_files
from core.provenance import write_json_atomic

from services.artifacts import Artifacts
from services.configuration import Configuration
from services.errors import ServiceError
from services.jobs import JobQueue
from services.models import Models
from services.query_plans import StaleQueryPlanError, load_current_query_plan


logger = logging.getLogger(__name__)


class Extraction:
    def __init__(
        self,
        configuration: Configuration,
        models: Models,
        artifacts: Artifacts,
        jobs: JobQueue,
    ):
        self.configuration = configuration
        self.models = models
        self.artifacts = artifacts
        self.jobs = jobs

    async def parse(self, stem: str, pdf_path: Path) -> None:
        await parse(pdf_path, self.configuration.paths.output_dir)

    async def index(self, stem: str) -> None:
        self.models.require_current()
        doc_dir = self.artifacts.require_stage(stem, "parsed")
        manifest = self.artifacts.begin_stage("index", stem, doc_dir)
        _, outputs = self.artifacts.stage_files("index", stem, doc_dir)
        ret_cfg = self.configuration.load_config()["retrieval"]

        def run() -> None:
            chunks = self.models.instances["chunker"].create_chunks(
                (doc_dir / f"{stem}.md").read_text(encoding="utf-8")
            )
            if not chunks:
                raise RuntimeError("Parsed document contains no indexable chunks")
            embeddings = np.asarray(
                self.models.instances["embedder"].embed_chunks(chunks), dtype=np.float32
            )
            if (
                embeddings.ndim != 2
                or embeddings.shape[0] != len(chunks)
                or (not embeddings.shape[1])
                or (not np.isfinite(embeddings).all())
            ):
                raise RuntimeError("Invalid embedding matrix")
            write_json_atomic(outputs["chunks.json"], chunks)
            np.save(outputs["embeddings.npy"], embeddings)
            self.models.make_keyword_searcher(ret_cfg).index_chunks(
                chunks, storage_path=str(outputs["keywords"])
            )
            self.artifacts.finish_stage("index", stem, doc_dir, manifest)

        await asyncio.to_thread(run)

    async def retrieve(self, stem: str) -> None:
        self.models.require_current()
        doc_dir = self.artifacts.require_stage(stem, "index")
        try:
            query_plan = load_current_query_plan(self.configuration)
        except StaleQueryPlanError as exc:
            raise ServiceError(409, str(exc)) from exc
        if set(query_plan) != {
            field["label"] for field in self.configuration.load_fields()
        }:
            raise RuntimeError(
                "Retrieval queries do not match the extraction plan; regenerate them"
            )
        manifest = self.artifacts.begin_stage("retrieve", stem, doc_dir)
        ret_cfg = self.configuration.load_config()["retrieval"]

        def run() -> None:
            retriever = self.models.make_retriever()
            retriever.load(
                doc_dir / f"{stem}.chunks.json",
                doc_dir / f"{stem}.embeddings.npy",
                doc_dir / "keyword_index",
            )
            evidence, debug = ({}, {})
            for label, entry in query_plan.items():
                debug[label] = retriever.retrieve_debug(
                    entry["queries"], entry["keywords"]
                )
                candidates = retriever.candidates(
                    entry["queries"],
                    entry["keywords"],
                    prefilter=ret_cfg.get("use_prefilter", False),
                )
                evidence[label] = retriever.select_for_extraction(
                    entry["queries"],
                    candidates,
                    self.models.get_reranker(),
                    top_k_per_query=ret_cfg.get("rerank_top_k_per_query", 2),
                    max_chunks=ret_cfg.get("max_evidence_chunks", 12),
                    fusion_reserve=ret_cfg.get("fusion_reserve", 4),
                )
            write_json_atomic(doc_dir / f"{stem}.evidence.json", evidence)
            write_json_atomic(doc_dir / f"{stem}.retrieve_debug.json", debug)
            self.artifacts.finish_stage("retrieve", stem, doc_dir, manifest)

        await asyncio.to_thread(run)

    async def extract(self, stem: str) -> None:
        self.models.require_current()
        doc_dir = self.artifacts.require_stage(stem, "evidence")
        manifest = self.artifacts.begin_stage("extract", stem, doc_dir)
        groups = self.configuration.load_data_table()["groups"]
        evidence = json.loads(
            (doc_dir / f"{stem}.evidence.json").read_text(encoding="utf-8")
        )
        result = await asyncio.to_thread(
            self.models.instances["extractor"].extract_groups,
            groups,
            evidence,
            self.configuration.load_config()["llm"].get(
                "max_group_evidence_chunks", 24
            ),
            checkpoint=Checkpoint(
                doc_dir / f"{stem}.extraction.checkpoint.json", manifest["fingerprint"]
            ),
        )
        output_path = doc_dir / f"{stem}.extraction.json"
        write_json_atomic(output_path, result)
        if not self.artifacts.extraction_complete(output_path):
            raise RuntimeError("Extraction returned incomplete or invalid fields")
        self.artifacts.finish_stage("extract", stem, doc_dir, manifest)

    async def extract_full(self, stem: str) -> None:
        self.models.require_current()
        doc_dir = self.artifacts.require_stage(stem, "parsed")
        manifest = self.artifacts.begin_stage("extract_full", stem, doc_dir)
        fields = self.configuration.load_fields()
        extractor = self.models.instances["full_extractor"]

        def run() -> None:
            paper_text, token_count = extractor.prepare(
                (doc_dir / f"{stem}.md").read_text(encoding="utf-8")
            )
            result = extractor.extract_all(
                fields,
                paper_text,
                token_count,
                checkpoint=Checkpoint(
                    doc_dir / f"{stem}.full_extraction.checkpoint.json",
                    manifest["fingerprint"],
                ),
            )
            output_path = doc_dir / f"{stem}.full_extraction.json"
            write_json_atomic(output_path, result)
            if not self.artifacts.extraction_complete(output_path, full=True):
                raise RuntimeError(
                    "Full extraction returned incomplete or invalid fields"
                )
            self.artifacts.finish_stage("extract_full", stem, doc_dir, manifest)

        await asyncio.to_thread(run)

    async def run_all(self, job_id: str, stem: str, pdf_path: Path) -> None:
        self.jobs.records[job_id]["message"] = "parsing"
        self.jobs.persist()
        await self.parse(stem, pdf_path)
        self.jobs.records[job_id]["message"] = "indexing"
        self.jobs.persist()
        if not self.artifacts.status(stem)["has_index"]:
            await self.index(stem)
        self.jobs.records[job_id]["message"] = "retrieving"
        self.jobs.persist()
        if not self.artifacts.status(stem)["has_evidence"]:
            await self.retrieve(stem)
        self.jobs.records[job_id]["message"] = "extracting"
        self.jobs.persist()
        if not self.artifacts.status(stem)["has_extraction"]:
            await self.extract(stem)

    async def run_bulk(self, entries: list[tuple[str, Path, str]]) -> None:
        """Parse bounded PDF batches, then finish each paper independently."""
        semaphore = self.jobs.semaphore
        if semaphore is None:
            for stem, _pdf_path, job_id in entries:
                self.jobs.records[job_id].update(
                    status="error", error="API execution queue is not initialized"
                )
                self.jobs.active.pop(stem, None)
            self.jobs.persist()
            return
        try:
            async with semaphore:
                batch_size = (
                    self.configuration.load_config()
                    .get("execution", {})
                    .get("mineru_batch_size", 10)
                )
                for start in range(0, len(entries), batch_size):
                    batch = entries[start : start + batch_size]
                    for _stem, _pdf_path, job_id in batch:
                        self.jobs.records[job_id].update(
                            status="running", message="parsing batch"
                        )
                    self.jobs.persist()
                    needs_parsing = [
                        entry
                        for entry in batch
                        if not self.artifacts.status(entry[0])["has_parsed"]
                    ]
                    batch_error: Exception | None = None
                    if needs_parsing:
                        try:
                            await parse_files(
                                [
                                    pdf_path
                                    for _stem, pdf_path, _job_id in needs_parsing
                                ],
                                self.configuration.paths.output_dir,
                            )
                        except Exception as exc:
                            logger.exception(
                                "Batch parsing failed; retrying affected papers individually"
                            )
                            batch_error = exc
                    if batch_error is not None:
                        for stem, pdf_path, job_id in needs_parsing:
                            if self.artifacts.status(stem)["has_parsed"]:
                                continue
                            self.jobs.records[job_id]["message"] = (
                                "retrying parse individually"
                            )
                            self.jobs.persist()
                            try:
                                await parse(
                                    pdf_path, self.configuration.paths.output_dir
                                )
                            except Exception as exc:
                                logger.exception(
                                    "Individual parsing retry failed for %s", stem
                                )
                                self.jobs.records[job_id].update(
                                    status="error",
                                    error=f"MinerU batch failed ({batch_error}); individual retry failed: {exc}",
                                )
                    for stem, _pdf_path, job_id in batch:
                        if self.jobs.records[job_id]["status"] == "error":
                            self.jobs.active.pop(stem, None)
                            self.jobs.persist()
                            continue
                        try:
                            if not self.artifacts.status(stem)["has_parsed"]:
                                raise RuntimeError(
                                    "MinerU did not produce a current parsed document"
                                )
                            self.jobs.records[job_id]["message"] = "indexing"
                            self.jobs.persist()
                            if not self.artifacts.status(stem)["has_index"]:
                                await self.index(stem)
                            self.jobs.records[job_id]["message"] = "retrieving"
                            self.jobs.persist()
                            if not self.artifacts.status(stem).get(
                                "has_evidence", False
                            ):
                                await self.retrieve(stem)
                            self.jobs.records[job_id]["message"] = "extracting"
                            self.jobs.persist()
                            if not self.artifacts.status(stem).get(
                                "has_extraction", False
                            ):
                                await self.extract(stem)
                            self.jobs.records[job_id].update(
                                status="done", message="done"
                            )
                        except Exception as exc:
                            logger.exception("Bulk pipeline failed for %s", stem)
                            self.jobs.records[job_id].update(
                                status="error", error=str(exc)
                            )
                        finally:
                            self.jobs.active.pop(stem, None)
                            self.jobs.persist()
        except Exception as exc:
            logger.exception("Bulk pipeline orchestration failed")
            for stem, _pdf_path, job_id in entries:
                if self.jobs.records[job_id]["status"] not in {"done", "error"}:
                    self.jobs.records[job_id].update(
                        status="error", error=f"Bulk pipeline failed: {exc}"
                    )
                self.jobs.active.pop(stem, None)
            self.jobs.persist()

    def enqueue_all(self, force: bool = False):
        if not self.configuration.paths.query_plan.exists():
            raise ServiceError(400, "Generate or write a query plan first")
        stems: set[str] = set()
        if self.configuration.paths.output_dir.exists():
            stems.update(
                (
                    d.name
                    for d in self.configuration.paths.output_dir.iterdir()
                    if d.is_dir()
                )
            )
        if self.configuration.paths.input_dir.exists():
            stems.update(
                (
                    f.stem
                    for f in self.configuration.paths.input_dir.iterdir()
                    if f.suffix.lower() == ".pdf"
                )
            )
        job_ids: dict[str, str] = {}
        skipped: list[str] = []
        entries: list[tuple[str, Path, str]] = []
        for stem in sorted(stems):
            pdf_path = self.artifacts.paper_pdf(stem)
            if (
                pdf_path is None
                or self.jobs.is_active(stem)
                or (not force and self.artifacts.status(stem)["has_extraction"])
            ):
                skipped.append(stem)
                continue
            job_id = self.jobs.create("queued")
            self.jobs.active[stem] = job_id
            job_ids[stem] = job_id
            entries.append((stem, pdf_path, job_id))
        if entries:
            self.jobs.spawn(self.run_bulk(entries))
        return {"job_ids": job_ids, "skipped": skipped}
