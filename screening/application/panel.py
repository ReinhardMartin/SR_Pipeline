import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from core.provenance import canonical_hash, file_hash, write_json_atomic
from core.settings import resolve_path
from screening.domain.models import PanelState, ReviewAssessment
from screening.infrastructure.llm_engine import LlmScreener
from services.errors import ServiceError


DECISIONS = {"Include", "Exclude"}
PANEL_SCHEMA_VERSION = "2.0"


def resolve_reviews(first, second, judge=None):
    if first.get("screening_status") == "missing_abstract":
        return "manual_full_text_required", None
    if first.get("screening_status") != "screened":
        return "first_failed", None
    if not second:
        return "awaiting_second_reviewer", None
    if second.get("decision") not in DECISIONS:
        return "second_reviewer_failed", None
    if first.get("decision") == second["decision"] and second["decision"] in DECISIONS:
        return "agreed", second["decision"]
    if not judge:
        return "awaiting_judge", None
    if judge.get("decision") not in DECISIONS:
        return "judge_failed", None
    return "adjudicated", judge["decision"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _useful_reason(result: dict) -> str | None:
    output = result.get("llm_output") or {}
    reason = output.get("reason")
    if reason:
        return reason
    fallback = result.get("reason")
    return fallback if fallback not in {"llm_paper_decision", "all_criteria_passed"} else None


def _primary_assessment(paper: dict) -> dict:
    output = paper.get("llm_output") or {}
    request = paper.get("llm_request") or {}
    return ReviewAssessment(
        role="primary",
        kind="llm",
        decision=paper.get("decision"),
        reason=_useful_reason(paper),
        quote=output.get("quote"),
        confidence=output.get("confidence"),
        criteria=paper.get("criteria", []),
        model=request.get("model"),
        screening_status=paper.get("screening_status", "screened"),
    ).model_dump()


def _llm_assessment(result: dict, role: str, model: str) -> dict:
    output = result.get("llm_output") or {}
    return ReviewAssessment(
        role=role,
        kind="llm",
        decision=result.get("decision"),
        reason=_useful_reason(result),
        quote=output.get("quote"),
        confidence=output.get("confidence"),
        criteria=result.get("criteria", []),
        model=model,
        reviewed_at=_utc_now(),
        screening_status=result.get("screening_status", "screening_failed"),
    ).model_dump()


def _human_assessment(role: str, decision: str, reason: str) -> dict:
    return ReviewAssessment(
        role=role,
        kind="human",
        decision=decision,
        reason=reason.strip() or None,
        reviewed_at=_utc_now(),
    ).model_dump()


class ScreeningPanel:
    def __init__(self, configuration):
        self.configuration = configuration

    @staticmethod
    def _read_state(path: Path) -> dict:
        return PanelState.model_validate_json(
            path.read_text(encoding="utf-8")
        ).model_dump()

    @staticmethod
    def _prompt_path(role: str, config: dict) -> Path:
        actor = config[role]
        value = (
            config["judge_instructions"]
            if role == "judge"
            else actor["llm"]["instructions"]
        )
        return resolve_path(value)

    @staticmethod
    def _engine(role: str, config: dict) -> tuple[LlmScreener, Path]:
        actor_config = copy.deepcopy(config[role]["llm"])
        if role == "judge" and config.get("require_judge_reason", True):
            actor_config["include_reason"] = True
        prompt = ScreeningPanel._prompt_path(role, config)
        return LlmScreener(actor_config, prompt.read_text(encoding="utf-8")), prompt

    def run(self, batch_dir, papers, primary_fingerprint, criteria, config):
        """Run only the independent second model; adjudication is always explicit."""
        path = batch_dir / "panel.json"
        identities = {
            "judge_prompt": file_hash(self._prompt_path("judge", config))
        }
        actor = config["second_reviewer"]
        if actor["kind"] == "llm":
            engine, prompt = self._engine("second_reviewer", config)
            with engine.client() as connection:
                identities["second_reviewer"] = {
                    "runtime": engine.runtime_identity(connection),
                    "prompt": file_hash(prompt),
                }
                fingerprint = self._fingerprint(
                    primary_fingerprint, config, identities
                )
                state = self._state(
                    path, batch_dir, fingerprint, config, criteria, identities
                )
                self._run_second_reviews(
                    path, batch_dir, state, papers, criteria, engine, connection
                )
            return

        fingerprint = self._fingerprint(primary_fingerprint, config, identities)
        state = self._state(path, batch_dir, fingerprint, config, criteria, identities)
        self._run_second_reviews(
            path, batch_dir, state, papers, criteria, engine=None, client=None
        )

    @staticmethod
    def _fingerprint(primary_fingerprint: str, config: dict, identities: dict) -> str:
        return canonical_hash(
            {
                "primary": primary_fingerprint,
                "config": config,
                "actors": identities,
                "code": file_hash(__file__),
            }
        )

    def _state(self, path, batch_dir, fingerprint, config, criteria, identities):
        state = self._read_state(path) if path.exists() else {}
        if state.get("fingerprint") != fingerprint:
            if state:
                prior = state.get("fingerprint") or canonical_hash(state)
                write_json_atomic(batch_dir / f"panel-history-{prior}.json", state)
            state = {
                "schema_version": PANEL_SCHEMA_VERSION,
                "fingerprint": fingerprint,
                "config": config,
                "criteria": criteria,
                "actor_identities": identities,
                "reviews": {},
                "history": [],
            }
        return state

    def _run_second_reviews(
        self, path, batch_dir, state, papers, criteria, engine, client
    ):
        candidates = []
        for paper in papers:
            reviews = state["reviews"].setdefault(str(paper["row"]), {})
            if (
                paper.get("screening_status") == "screened"
                and engine is not None
                and (reviews.get("second_reviewer") or {}).get("decision") not in DECISIONS
            ):
                candidates.append((paper, reviews))
                continue
            self.present(paper, reviews, state)

        if candidates:
            payloads = [
                {
                    "row": paper["row"],
                    "title": paper.get("title"),
                    "abstract": paper.get("abstract"),
                }
                for paper, _reviews in candidates
            ]
            results = engine.screen_batch(payloads, criteria, client=client)
            for (paper, reviews), result in zip(candidates, results):
                reviews["second_reviewer"] = _llm_assessment(
                    result, "second_reviewer", engine.config["model"]
                )
                self.present(paper, reviews, state)

        write_json_atomic(path, state)
        write_json_atomic(batch_dir / "results.json", papers)

    def run_judges(self, batch_dir: Path, rows: set[int] | None = None) -> int:
        """Run an LLM judge for one requested paper or all pending papers."""
        path = batch_dir / "panel.json"
        if not path.exists():
            raise ServiceError(409, "Run screening with the review panel enabled first")
        state = self._read_state(path)
        current = self.configuration.load_config()["screening"]
        config = current["panel"]
        if current["backend"] != "llm" or not config["enabled"]:
            raise ServiceError(409, "The LLM review panel is not enabled")
        if config != state["config"]:
            raise ServiceError(409, "Panel settings changed; rerun screening first")
        if config["judge"]["kind"] != "llm":
            raise ServiceError(400, "The judge role is assigned to a human")

        papers = json.loads((batch_dir / "results.json").read_text(encoding="utf-8"))
        by_row = {paper["row"]: paper for paper in papers}
        if rows is not None and not rows.issubset(by_row):
            raise ServiceError(404, "Paper not found")

        candidates = []
        for paper in papers:
            if rows is not None and paper["row"] not in rows:
                continue
            reviews = state["reviews"].setdefault(str(paper["row"]), {})
            status, _ = resolve_reviews(
                paper, reviews.get("second_reviewer"), reviews.get("judge")
            )
            if status in {"awaiting_judge", "judge_failed"}:
                candidates.append((paper, reviews))
            elif rows is not None:
                raise ServiceError(409, "This paper is not awaiting LLM adjudication")

        if not candidates:
            return 0

        engine, prompt = self._engine("judge", config)
        failures = []
        with engine.client() as client:
            identity = {
                "runtime": engine.runtime_identity(client),
                "prompt": file_hash(prompt),
            }
            prior_identity = state["actor_identities"].get("judge")
            if prior_identity is not None and prior_identity != identity:
                raise ServiceError(
                    409, "The judge model deployment changed; rerun screening first"
                )
            state["actor_identities"]["judge"] = identity
            payloads = [
                {
                    "row": paper["row"],
                    "title": paper.get("title"),
                    "abstract": paper.get("abstract"),
                    "reviewer_assessments": [
                        _primary_assessment(paper),
                        reviews["second_reviewer"],
                    ],
                }
                for paper, reviews in candidates
            ]
            results = engine.screen_batch(
                payloads, state["criteria"], client=client
            )
            for (paper, reviews), result in zip(candidates, results):
                reviews["judge"] = _llm_assessment(
                    result, "judge", engine.config["model"]
                )
                self.present(paper, reviews, state)
                if paper["consensus"]["status"] == "judge_failed":
                    failures.append(paper["row"])
            write_json_atomic(path, state)
            write_json_atomic(batch_dir / "results.json", papers)
        if failures:
            raise RuntimeError(
                f"Judge review failed for {len(failures)} paper(s); continue again to retry"
            )
        return len(candidates)

    @staticmethod
    def present(paper, reviews, state):
        status, decision = resolve_reviews(
            paper, reviews.get("second_reviewer"), reviews.get("judge")
        )
        paper["consensus"] = {
            "schema_version": PANEL_SCHEMA_VERSION,
            "status": status,
            "decision": decision,
            "fingerprint": state["fingerprint"],
            "criteria": state["criteria"],
            "roles": {
                role: state["config"][role]["kind"]
                for role in ("second_reviewer", "judge")
            },
            "require_judge_reason": state["config"].get(
                "require_judge_reason", True
            ),
            "primary": _primary_assessment(paper),
            **reviews,
        }

    def submit(self, batch_dir, row, role, body):
        path = batch_dir / "panel.json"
        if not path.exists():
            raise ServiceError(409, "Run screening with the review panel enabled first")
        state = self._read_state(path)
        if state["fingerprint"] != body.fingerprint:
            raise ServiceError(409, "The screening run changed; reload the results")
        screening = self.configuration.load_config()["screening"]
        config = screening["panel"]
        if screening.get("backend", "llm") != "llm" or config != state["config"]:
            raise ServiceError(409, "Panel settings changed; rerun screening first")
        if config[role]["kind"] != "human":
            raise ServiceError(400, "This role is assigned to an LLM")
        if (
            role == "judge"
            and config.get("require_judge_reason", True)
            and not body.reason.strip()
        ):
            raise ServiceError(400, "The judge must provide a reason")

        papers = json.loads((batch_dir / "results.json").read_text(encoding="utf-8"))
        paper = next((item for item in papers if item["row"] == row), None)
        if paper is None:
            raise ServiceError(404, "Paper not found")
        reviews = state["reviews"].setdefault(str(row), {})
        status, _ = resolve_reviews(
            paper, reviews.get("second_reviewer"), reviews.get("judge")
        )
        editing_human_judge = (reviews.get("judge") or {}).get("kind") == "human"
        if paper.get("screening_status") != "screened" or (
            role == "judge"
            and status not in {"awaiting_judge", "judge_failed"}
            and not editing_human_judge
        ):
            raise ServiceError(409, "This paper is not ready for this review")

        state["history"].append(
            {
                "row": row,
                "role": role,
                "previous": reviews.get(role),
                "changed_at": _utc_now(),
            }
        )
        if role == "second_reviewer":
            reviews.pop("judge", None)
        reviews[role] = _human_assessment(role, body.decision, body.reason)
        self.present(paper, reviews, state)
        write_json_atomic(path, state)
        write_json_atomic(batch_dir / "results.json", papers)
