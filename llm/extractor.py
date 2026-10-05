from core.settings import load_prompt, render_prompt, field_prompt
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from typing import Callable

from core.extraction_schema import field_errors


SYSTEM_PROMPT = load_prompt("extraction_single_field_system")

GROUP_SYSTEM_PROMPT = load_prompt("extraction_field_group_system")

def _failure(status: str, error: str | None = None) -> dict:
    result = {
        "value": None,
        "status": status,
        "confidence": "low",
        "evidence": [],
        "notes": None,
    }
    if error:
        result["error"] = error
    return result


def _table_to_text(html: str) -> str:
    class _P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows, self._row, self._cell = [], [], []

        def handle_starttag(self, tag, _):
            if tag == "tr":
                self._row = []
            elif tag in ("td", "th"):
                self._cell = []

        def handle_endtag(self, tag):
            if tag in ("td", "th"):
                self._row.append(" ".join(self._cell).strip())
            elif tag == "tr" and self._row:
                self.rows.append(self._row)

        def handle_data(self, data):
            self._cell.append(data)

    parser = _P()
    parser.feed(html)
    return "\n".join("\t".join(row) for row in parser.rows if any(cell for cell in row))


def _chunk_text(chunk: dict) -> str:
    if chunk.get("kind") == "table" and chunk.get("raw"):
        return _table_to_text(chunk["raw"])
    return chunk["text"]


def _normalize(text: str) -> str:
    return " ".join(str(text).split()).casefold()


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


class Extractor:
    def __init__(self, llm_fn: Callable[[str, str], str], max_workers: int = 4):
        self._llm = llm_fn
        self._max_workers = max_workers

    def extract(self, field: dict, evidence: list[dict] | str) -> dict:
        if not evidence:
            return _failure("no_evidence")
        if isinstance(evidence, str):
            passages = evidence
            evidence_by_id: dict[str, str] = {}
        else:
            evidence_by_id = {c["id"]: _chunk_text(c) for c in evidence}
            passages = "\n\n".join(
                render_prompt("passage", id=c["id"], section=c["section"], text=evidence_by_id[c["id"]]) for c in evidence
            )

        user_msg = render_prompt("extract", specification=field_prompt(field), passages=passages)
        parsed = _extract_json(self._llm(SYSTEM_PROMPT, user_msg))
        return self._validate(parsed, field, evidence_by_id)

    @staticmethod
    def _group_evidence(
        fields: list[dict], evidence_map: dict, max_chunks: int
    ) -> list[dict]:
        """Deduplicate evidence while giving every child field a fair share."""
        if max_chunks < len(fields):
            raise ValueError(
                f"max_group_evidence_chunks ({max_chunks}) must be at least the "
                f"number of fields in the group ({len(fields)})"
            )
        per_field = [evidence_map.get(field["label"], []) for field in fields]
        selected: list[dict] = []
        seen: set[str] = set()
        depth = 0
        while len(selected) < max_chunks and any(depth < len(items) for items in per_field):
            for items in per_field:
                if depth >= len(items):
                    continue
                chunk = items[depth]
                chunk_id = chunk.get("id")
                if chunk_id and chunk_id not in seen:
                    selected.append(chunk)
                    seen.add(chunk_id)
                    if len(selected) >= max_chunks:
                        break
            depth += 1
        return selected

    def extract_group(self, group: dict, evidence_map: dict, max_chunks: int = 24) -> dict[str, dict]:
        fields = group.get("fields", [])
        if not fields:
            return {}
        evidence = self._group_evidence(fields, evidence_map, max_chunks)
        if not evidence:
            return {field["label"]: _failure("no_evidence") for field in fields}

        evidence_by_id = {chunk["id"]: _chunk_text(chunk) for chunk in evidence}
        passages = "\n\n".join(
            render_prompt("passage", id=c["id"], section=c["section"], text=evidence_by_id[c["id"]])
            for c in evidence
        )
        specifications = "\n\n".join(
            render_prompt("field_id", id=field["id"], specification=field_prompt(field)) for field in fields
        )
        user_msg = render_prompt("group", label=group.get("label", ""), specifications=specifications, passages=passages)

        try:
            parsed = _extract_json(self._llm(GROUP_SYSTEM_PROMPT, user_msg))
        except Exception:
            return {
                field["label"]: self._extract_with_failure_capture(
                    field, evidence_map.get(field["label"], [])
                )
                for field in fields
            }

        results: dict[str, dict] = {}
        for field in fields:
            raw = parsed.get(field["id"])
            validated = self._validate(raw if isinstance(raw, dict) else {}, field, evidence_by_id)
            if validated["status"] == "invalid_output":
                validated = self._extract_with_failure_capture(
                    field, evidence_map.get(field["label"], [])
                )
            results[field["label"]] = validated
        return results

    def _extract_with_failure_capture(self, field: dict, evidence: list[dict] | str) -> dict:
        try:
            return self.extract(field, evidence)
        except Exception as exc:
            return _failure("llm_failed", str(exc))

    def extract_groups(
        self, groups: list[dict], evidence_map: dict, max_chunks: int = 24, *, checkpoint=None
    ) -> dict[str, dict]:
        """Extract once per topic group, retrying only invalid child fields."""
        results = {}
        if checkpoint is not None:
            results = {
                field["label"]: checkpoint.values[field["label"]]
                for group in groups for field in group["fields"]
                if field["label"] in checkpoint.values
                and not field_errors(checkpoint.values[field["label"]], field)
            }
        work_groups = []
        for group in groups:
            fields = [field for field in group.get("fields", []) if field["label"] not in results]
            for start in range(0, len(fields), max_chunks):
                work_groups.append({**group, "fields": fields[start : start + max_chunks]})
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(self.extract_group, group, evidence_map, max_chunks): group
                for group in work_groups
            }
            for future in as_completed(futures):
                group = futures[future]
                try:
                    results.update(future.result())
                except Exception as exc:
                    for field in group.get("fields", []):
                        results[field["label"]] = _failure("llm_failed", str(exc))
                if checkpoint is not None:
                    checkpoint.save(results)
            return results

    def extract_all(self, fields: list[dict], evidence_map: dict) -> dict[str, dict]:
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(self.extract, field, evidence_map.get(field["label"], [])): field["label"]
                for field in fields
            }
            results = {}
            for future in as_completed(futures):
                label = futures[future]
                try:
                    results[label] = future.result()
                except Exception as exc:
                    results[label] = _failure("llm_failed", str(exc))
            return results

    def _validate(self, data: dict, field: dict, evidence_by_id: dict[str, str]) -> dict:
        errors = field_errors(data, field)
        if not isinstance(data, dict):
            return _failure("invalid_output", "; ".join(errors))
        status = data.get("status")
        confidence = data.get("confidence")
        value = data.get("value")
        evidence = data.get("evidence", [])

        if not isinstance(evidence, list):
            errors.append("evidence must be a list")
            evidence = []

        validated_evidence = []
        for item in evidence:
            if not isinstance(item, dict):
                errors.append("evidence entry is not an object")
                continue
            source_id, quote = item.get("source_id"), item.get("quote")
            if not isinstance(source_id, str) or source_id not in evidence_by_id:
                errors.append(f"unknown source_id: {source_id!r}")
                continue
            if not isinstance(quote, str) or not quote.strip():
                errors.append(f"missing quote for source_id {source_id!r}")
                continue
            if _normalize(quote) not in _normalize(evidence_by_id[source_id]):
                errors.append(f"quote is not present in source_id {source_id!r}")
                continue
            validated_evidence.append({"source_id": source_id, "quote": quote.strip()})

        if errors:
            return {
                "value": None,
                "status": "invalid_output",
                "confidence": "low",
                "evidence": validated_evidence,
                "notes": data.get("notes"),
                "validation_errors": errors,
                "raw_output": data,
            }

        return {
            "value": value,
            "status": status,
            "confidence": confidence,
            "evidence": validated_evidence,
            "notes": data.get("notes"),
        }
