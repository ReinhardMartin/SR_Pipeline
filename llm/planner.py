from core.settings import load_prompt, render_prompt
import json
import re
from typing import Callable

SYSTEM_PROMPT = load_prompt("extraction_query_planner_system")


class Planner:
    def __init__(self, llm_fn: Callable[[str, str], str], retries: int = 2, batch_size: int = 4):
        self._llm = llm_fn
        self._retries = retries
        self._batch_size = batch_size

    def plan(self, fields: list[dict]) -> dict[str, dict]:
        combined: dict[str, dict] = {}
        for start in range(0, len(fields), self._batch_size):
            batch = fields[start : start + self._batch_size]
            combined.update(self._plan_batch(batch))
        return combined

    def _plan_batch(self, fields: list[dict]) -> dict[str, dict]:
        fields_text = "\n\n".join(self._format_field(f) for f in fields)
        last_exc = None
        for _ in range(self._retries + 1):
            try:
                decomposed = self._parse(self._llm(SYSTEM_PROMPT, fields_text))
                self._validate(decomposed, fields)
                return {f["label"]: decomposed[f["label"]] for f in fields}
            except (ValueError, json.JSONDecodeError) as e:
                last_exc = e
        raise last_exc

    def _format_field(self, field: dict) -> str:
        return render_prompt("planner_field", label=field["label"], description=field["description"])

    @staticmethod
    def _validate(plan: dict[str, dict], fields: list[dict]) -> None:
        expected = {f["label"] for f in fields}
        missing = expected - set(plan)
        if missing:
            raise ValueError(f"Planner omitted fields: {', '.join(sorted(missing))}")
        for label in expected:
            if not plan[label]["queries"]:
                raise ValueError(f"Planner returned no query for {label!r}")
            if len(plan[label]["keywords"]) < 2:
                raise ValueError(f"Planner returned too few keywords for {label!r}")

    def _parse(self, response: str) -> dict[str, dict]:
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON in planner response: {response!r}")
            data = json.loads(match.group())
        if not isinstance(data, dict):
            raise ValueError("Planner response must be a JSON object")
        return {
            label: {
                "queries":  [q for q in content.get("queries",  []) if isinstance(q, str) and q.strip()],
                "keywords": [k for k in content.get("keywords", []) if isinstance(k, str) and k.strip()],
            }
            for label, content in data.items()
            if isinstance(label, str) and isinstance(content, dict)
        }
