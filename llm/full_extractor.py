from core.settings import load_prompt, read_config, render_prompt, field_prompt
import json
import re
from typing import Any, Callable

import tiktoken

from core.extraction_schema import field_errors


_SYSTEM = load_prompt("extraction_full_document_system")

_ENCODING = tiktoken.get_encoding(read_config()["full_extraction"]["tokenizer"])


def _failure(status: str, error: str | None = None) -> dict:
    result = {
        "value": None,
        "status": status,
        "confidence": "low",
        "source_quote": None,
        "source_section": None,
        "notes": None,
    }
    if error:
        result["error"] = error
    return result


def strip_references(text: str) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("#"):
            heading = line.lstrip("#").strip().casefold()
            if heading in {"references", "bibliography", "works cited"}:
                return "\n".join(lines[:i]).rstrip()
    return text


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _extract_json(response: str) -> dict:
    try:
        value = json.loads(response)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", response):
        try:
            value, _ = decoder.raw_decode(response[match.start() :])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    raise ValueError("LLM response did not contain a valid JSON object")


def _normalize(text: str) -> str:
    return " ".join(str(text).split()).casefold()


class FullExtractor:
    """Bounded full-paper comparator.

    RAG remains the production path. This comparator is intentionally unavailable
    for papers that do not fit safely in one request rather than truncating them.
    """

    def __init__(
        self,
        llm_fn: Callable[[str, str], str],
        max_context_tokens: int = 32000,
        max_output_tokens: int = 4096,
    ):
        self._llm = llm_fn
        self._max_context_tokens = max_context_tokens
        self._max_output_tokens = max_output_tokens

    def prepare(self, md_text: str) -> tuple[str, int]:
        cleaned = strip_references(md_text)
        return cleaned, count_tokens(cleaned)

    def _user_message(self, fields: list[dict], paper_text: str) -> str:
        specs = "\n\n".join(field_prompt(field) for field in fields)
        return render_prompt("full", specifications=specs, paper=paper_text)

    def extract_all(self, fields: list[dict], paper_text: str, token_count: int, *, checkpoint=None) -> dict[str, dict]:
        results = {}
        if checkpoint is not None:
            results = {field["label"]: checkpoint.values[field["label"]] for field in fields
                       if field["label"] in checkpoint.values
                       and not field_errors(checkpoint.values[field["label"]], field, full=True)}
            fields = [field for field in fields if field["label"] not in results]
        if not fields:
            return results
        user_message = self._user_message(fields, paper_text)
        request_tokens = count_tokens(_SYSTEM) + count_tokens(user_message) + self._max_output_tokens
        if request_tokens > self._max_context_tokens:
            error = (
                f"Full-paper request requires about {request_tokens:,} tokens including output reserve; "
                f"configured safe limit is {self._max_context_tokens:,}. Use validated RAG extraction."
            )
            return {**results, **{field["label"]: _failure("context_too_large", error) for field in fields}}

        try:
            data = _extract_json(self._llm(_SYSTEM, user_message))
        except Exception as exc:
            return {**results, **{field["label"]: _failure("llm_failed", str(exc)) for field in fields}}
        results.update({field["label"]: self._validate(data.get(field["label"]), field, paper_text) for field in fields})
        if checkpoint is not None:
            checkpoint.save(results)
        return results

    def _validate(self, raw: Any, field: dict, paper_text: str) -> dict:
        if not isinstance(raw, dict):
            return _failure("invalid_output", "Missing field object in LLM response")

        errors = field_errors(raw, field, full=True)
        status, value = raw.get("status"), raw.get("value")
        confidence = raw.get("confidence")
        quote = raw.get("source_quote")
        if isinstance(quote, str) and quote.strip() and _normalize(quote) not in _normalize(paper_text):
            errors.append("source quote is not present in the paper")

        if errors:
            result = _failure("invalid_output", "; ".join(errors))
            result["raw_output"] = raw
            return result
        return {
            "value": value,
            "status": status,
            "confidence": confidence,
            "source_quote": quote,
            "source_section": raw.get("source_section"),
            "notes": raw.get("notes"),
        }
