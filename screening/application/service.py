import asyncio
import io
import json
from contextlib import ExitStack

import openpyxl
from core.checkpoint import Checkpoint
from core.provenance import build_manifest, write_json_atomic
from screening.domain.policy import Screener
from screening.domain.models import ScreeningCriteriaDocument, ScreeningPaperResult
from screening.application.handoff import recovery_list
from screening.infrastructure.citations import spreadsheet_value
from core.settings import resolve_path
from screening.application.panel import ScreeningPanel
from screening.infrastructure.llm_engine import LlmScreener

from services.configuration import Configuration
from services.errors import ServiceError
from services.files import read_json
from services.models import Models
from screening.infrastructure.repository import ScreeningBatchRepository


class Screening:
    def __init__(
        self,
        configuration: Configuration,
        models: Models,
        batches: ScreeningBatchRepository | None = None,
    ):
        self.configuration = configuration
        self.models = models
        self.batches = batches or ScreeningBatchRepository(configuration)

    def export_excel(self, papers: list[dict], criteria: list[dict]) -> bytes:
        wb = openpyxl.Workbook()
        ws = wb.active
        extra_keys = list(dict.fromkeys(key for paper in papers for key in paper.get("extra", {})))
        header = (
            ["row", "title", "abstract"]
            + extra_keys
            + [
                "screening_status",
                "decision",
                "reason",
                "decision_confidence",
                "review_group", "blocking_criteria", "score_gap",
                "human_decision",
                "input_issues", "llm_reason", "llm_quote", "llm_model_confidence",
                "panel_status", "final_decision", "second_reviewer", "judge",
            ]
        )
        for c in criteria:
            header += [
                f"{c['label']} (decision)",
                f"{c['label']} (human decision)",
                f"{c['label']} (threshold)",
                f"{c['label']} (confidence)",
                f"{c['label']} (entailment)",
                f"{c['label']} (contradiction)",
                f"{c['label']} (neutral)",
                f"{c['label']} (finding)",
                f"{c['label']} (explicit evidence required)",
                f"{c['label']} (reason)", f"{c['label']} (quote)", f"{c['label']} (model confidence)",
            ]
        ws.append([spreadsheet_value(value) for value in header])
        for p in papers:
            by_label = {c["label"]: c for c in p["criteria"]}
            row = [p["row"], p["title"], p["abstract"]] + [
                p.get("extra", {}).get(k, "") for k in extra_keys
            ]
            row += [
                p.get("screening_status", "screened"),
                p.get("decision") or "",
                p.get("reason", ""),
                round(p["decision_confidence"], 4)
                if p.get("decision_confidence") is not None
                else "",
                p.get("review_order", {}).get("group", ""),
                p.get("review_order", {}).get("blocking_count", ""),
                p.get("review_order", {}).get("score_gap"),
                p.get("human_decision") or "",
                json.dumps(p.get("input_issues", []), ensure_ascii=False),
                (p.get("llm_output") or {}).get("reason", ""),
                (p.get("llm_output") or {}).get("quote", ""),
                (p.get("llm_output") or {}).get("confidence"),
                (p.get("consensus") or {}).get("status"), (p.get("consensus") or {}).get("decision"),
                json.dumps((p.get("consensus") or {}).get("second_reviewer"), ensure_ascii=False),
                json.dumps((p.get("consensus") or {}).get("judge"), ensure_ascii=False),
            ]
            for c in criteria:
                cr = by_label.get(c["label"])
                if cr is None:
                    row += [""] * 12
                else:
                    row += [
                        cr["criterion_decision"],
                        cr.get("human_decision") or "",
                        round(cr["confidence_threshold"], 4) if cr.get("confidence_threshold") is not None else "",
                        round(cr["confidence"], 4) if cr.get("confidence") is not None else "",
                        round(cr["entailment"], 4) if cr.get("entailment") is not None else "",
                        round(cr["contradiction"], 4) if cr.get("contradiction") is not None else "",
                        round(cr["neutral"], 4) if cr.get("neutral") is not None else "",
                        cr.get("finding", ""),
                        cr.get("explicit_evidence_required", False),
                        cr.get("reason", ""), cr.get("quote", ""), cr.get("model_confidence"),
                    ]
            ws.append([spreadsheet_value(value) for value in row])
        buf = io.BytesIO()
        wb.save(buf)
        return buf.getvalue()

    @staticmethod
    def validate_results(papers: list[dict]) -> list[dict]:
        """Validate persisted results against the only supported result schema."""
        return [
            ScreeningPaperResult.model_validate(paper).model_dump()
            for paper in papers
        ]

    async def run(self, batch_id: str) -> None:
        self.models.require_current()
        batch = self.batches.paths(batch_id)
        batch_dir = batch.directory
        papers = self.batches.read(batch.papers)
        criteria = self.configuration.load_screening_criteria()
        if not criteria:
            raise RuntimeError("No screening criteria defined yet")
        config = self.configuration.load_config()
        cfg = config["screening"]
        if cfg["panel"]["enabled"] and cfg.get("backend", "nli") != "llm":
            raise RuntimeError(
                "The independent review panel is available only for LLM screening"
            )
        criteria_path = batch_dir / "criteria.json"
        write_json_atomic(
            criteria_path,
            ScreeningCriteriaDocument(
                schema_version="2.2", criteria=criteria
            ).model_dump(),
        )
        results_path = batch_dir / "results.json"
        prior_labels = {}
        if results_path.exists():
            prior = json.loads(results_path.read_text(encoding="utf-8"))
            current_criteria = {(c["label"], c["statement"], c["type"], c.get("explicit_evidence_required", False)) for c in criteria}
            stale_reviews = [
                {"row": paper["row"], "criterion": item}
                for paper in prior for item in paper.get("criteria", [])
                if item.get("human_decision") and
                (item["label"], item.get("statement"), item.get("type"), item.get("explicit_evidence_required", False)) not in current_criteria
            ]
            if stale_reviews:
                history_path = batch_dir / "criterion_review_history.json"
                history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
                write_json_atomic(history_path, history + [item for item in stale_reviews if item not in history])
            prior_labels = {
                p["row"]: {
                    "human_decision": p.get("human_decision"),
                    "human_labelled_at": p.get("human_labelled_at"),
                    "criteria": {
                        (item["label"], item.get("statement"), item.get("type"), item.get("explicit_evidence_required", False)): {
                            "human_decision": item.get("human_decision"),
                            "human_finding": item.get("human_finding"),
                            "human_labelled_at": item.get("human_labelled_at"),
                        }
                        for item in p.get("criteria", [])
                    },
                }
                for p in prior
            }

        def _run():
            inputs = {
                "papers": batch_dir / "papers.json", "criteria": criteria_path,
                "code": self.configuration.paths.base / "screening/domain/policy.py",
                "service_code": self.configuration.paths.base / "screening/application/service.py",
                "retry_code": self.configuration.paths.base / "core/retry.py",
            }
            if cfg.get("backend", "nli") == "llm":
                prompt_key = "criterion_instructions" if cfg["llm"]["execution_mode"] == "per_criterion" else "instructions"
                prompt_path = resolve_path(cfg["llm"][prompt_key])
                inputs.update(contract_code=self.configuration.paths.base / "screening/infrastructure/llm_contract.py", instructions=prompt_path, llm_code=self.configuration.paths.base / "screening/infrastructure/llm_engine.py")
                screener = LlmScreener(cfg["llm"], prompt_path.read_text(encoding="utf-8"))
            else:
                inputs["nli_code"] = self.configuration.paths.base / "core/nli.py"
                screener = Screener(nli=self.models.get_nli(), confidence_threshold=cfg["confidence_threshold"])
            decisions = []
            with ExitStack() as stack:
                client = stack.enter_context(screener.client()) if cfg.get("backend") == "llm" else None
                runtime = screener.runtime_identity(client) if client is not None else {}
                identity = build_manifest(
                    "screen", inputs,
                    {"screening": {k: v for k, v in cfg.items() if k != "panel"}, "runtime": runtime,
                     **({"nli": config["nli"]} if cfg.get("backend") != "llm" else {})},
                )
                checkpoint = Checkpoint(batch_dir / "screening.checkpoint.json", identity["fingerprint"])
                if cfg.get("backend") == "llm":
                    decisions = screener.screen_batch(
                        papers,
                        criteria,
                        checkpoint=checkpoint,
                        cache_prefix="primary",
                        client=client,
                    )
                else:
                    size = cfg["checkpoint_batch_size"]
                    for start in range(0, len(papers), size):
                        paper_batch = papers[start : start + size]
                        key = str(start)
                        cached = checkpoint.get(key)
                        if cached is None or any(
                            item["screening_status"] in {
                                "screening_failed", "output_too_long"
                            }
                            for item in cached
                        ):
                            cached = screener.screen_batch(paper_batch, criteria)
                            checkpoint.put(key, cached)
                        decisions.extend(cached)
            results = []
            for paper, decision in zip(papers, decisions):
                prior_review = prior_labels.get(paper["row"], {})
                prior_criteria = prior_review.get("criteria", {})
                for item in decision["criteria"]:
                    prior_item = prior_criteria.get((item["label"], item.get("statement"), item.get("type"), item.get("explicit_evidence_required", False)), {})
                    item["human_decision"] = prior_item.get("human_decision")
                    item["human_finding"] = prior_item.get("human_finding")
                    item["human_labelled_at"] = prior_item.get("human_labelled_at")
                results.append(
                    {
                        **paper,
                        **decision,
                        "human_decision": prior_review.get("human_decision"),
                        "human_labelled_at": prior_review.get("human_labelled_at"),
                    }
                )
            write_json_atomic(results_path, results)
            write_json_atomic(batch_dir / "screening.manifest.json", identity)
            if cfg["panel"]["enabled"]:
                for paper in results:
                    paper["consensus"] = {"status": "pending", "decision": None}
                write_json_atomic(results_path, results)
                ScreeningPanel(self.configuration).run(batch_dir, results, identity["fingerprint"], criteria, cfg["panel"])
                panel_failures = [p for p in results if p["consensus"]["status"] == "second_reviewer_failed"]
                if panel_failures:
                    raise RuntimeError(f"Panel review failed for {len(panel_failures)} papers; inspect the saved review audit and rerun to retry")
            failures = [
                result for result in results
                if result["screening_status"] in {"screening_failed", "output_too_long"}
            ]
            if failures:
                first = failures[0]
                raise RuntimeError(
                    f"Screening failed for {len(failures)}/{len(results)} papers. "
                    f"First failure (row {first['row']}): {first['reason']}"
                )

        def locked_run():
            with self.batches.locked(batch_id):
                _run()
        await asyncio.to_thread(locked_run)

    def recovery(self, batch_id: str) -> list[dict]:
        try:
            batch_dir = self.batches.paths(batch_id).directory
        except ValueError as exc:
            raise ServiceError(400, "Invalid screening batch ID") from exc
        papers = read_json(batch_dir / "results.json")
        return recovery_list(batch_dir.name, papers, self.configuration.paths.input_dir)
