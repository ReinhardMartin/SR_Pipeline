import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import BoundedSemaphore, Lock
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field
from screening.infrastructure.llm_contract import build_response_model, validate_response

from core.retry import retry_call
from screening.domain.policy import Screener
from core.settings import ScreeningLlmConfig


class OutputTokenLimitError(ValueError):
    """The provider stopped generation because its output allowance was exhausted."""


class CriterionInput(BaseModel):
    model_config = ConfigDict(strict=True)
    label: str = Field(min_length=1, pattern=r"\S")
    statement: str = Field(min_length=1, pattern=r"\S")
    type: Literal["include", "exclude"]
    explicit_evidence_required: bool = False


class LlmScreener:
    def __init__(self, config: dict, instructions: str):
        if not instructions.strip():
            raise ValueError("Screening instructions are empty")
        self.options = ScreeningLlmConfig.model_validate(config)
        self.config = self.options.model_dump()
        self._contracts = {}
        self._contract_lock = Lock()
        self._request_slots = BoundedSemaphore(
            self.config["max_concurrent_requests"]
        )
        self.instructions = instructions
        self.native_context_guard = False

    def runtime_identity(self, client):
        cfg = self.config
        if cfg["provider"] != "ollama":
            return {"revision": cfg["cache_revision"] or str(uuid.uuid4())}

        def metadata(endpoint):
            response = client.get(cfg["base_url"].rstrip("/") + "/api/" + endpoint)
            response.raise_for_status()
            return response.json()

        def resolve():
            version = metadata("version")["version"]
            name = cfg["model"]
            if ":" not in name.rsplit("/", 1)[-1]:
                name += ":latest"
            models = metadata("tags")["models"]
            model = next((item for item in models if item.get("name") == name), None)
            if not model or not isinstance(model.get("digest"), str) or not model["digest"]:
                raise ValueError(f"Cannot resolve installed Ollama model digest: {name}")
            parts = version.split("-", 1)[0].split(".")
            self.native_context_guard = len(parts) == 3 and all(p.isdigit() for p in parts) and tuple(map(int, parts)) >= (0, 34, 2)
            return {"model_digest": model["digest"], "ollama_version": version}

        return retry_call(resolve, retries=cfg["retries"], initial=cfg["retry_initial_seconds"], maximum=cfg["retry_max_seconds"])

    def client(self):
        return httpx.Client(base_url=self.config["base_url"], timeout=self.config["timeout"], trust_env=False)

    def screen_batch(self, papers: list[dict], criteria: list[dict], checkpoint=None, cache_prefix="", client=None) -> list[dict]:
        if client is None:
            with self.client() as connection:
                self.runtime_identity(connection)
                return self.screen_batch(papers, criteria, checkpoint, cache_prefix, connection)
        if not papers:
            return []

        def screen_one(index: int, paper: dict) -> tuple[int, dict]:
            identity = paper.get("row", index)
            prefix = f"{cache_prefix}:paper:{identity}"
            cached = checkpoint.get(prefix) if checkpoint else None
            if cached is not None and cached.get("screening_status") not in {
                "screening_failed", "output_too_long"
            }:
                return index, cached
            result = self._screen(client, paper, criteria, checkpoint, prefix)
            if checkpoint:
                checkpoint.put(prefix, result)
            return index, result

        request_limit = self.config["max_concurrent_requests"]
        if len(papers) == 1 or request_limit == 1:
            return [screen_one(index, paper)[1] for index, paper in enumerate(papers)]

        results = [None] * len(papers)
        worker_count = min(len(papers), request_limit * 2)
        with ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix="screening-llm"
        ) as executor:
            futures = {
                executor.submit(screen_one, index, paper): index
                for index, paper in enumerate(papers)
            }
            for future in as_completed(futures):
                index, result = future.result()
                results[index] = result
        return results

    def _contract(self, ids):
        key = tuple(ids) if self.options.execution_mode == "per_criterion" else ()
        with self._contract_lock:
            if key not in self._contracts:
                model = build_response_model(self.options)
                self._contracts[key] = model, model.model_json_schema()
            return self._contracts[key]

    def _screen(self, client, paper, criteria, checkpoint=None, cache_prefix=""):
        if self.config["execution_mode"] == "all_criteria":
            return self._screen_group(client, paper, criteria)
        calls = []
        findings = []
        for start in range(max(len(criteria), 1)):
            key = f"{cache_prefix}:criterion_group:{start}"
            result = checkpoint.get(key) if checkpoint else None
            reused = result is not None and result["screening_status"] == "screened"
            if not reused:
                result = self._screen_group(client, paper, criteria[start:start + 1], start)
                if checkpoint:
                    checkpoint.put(key, result)
            calls.append({**result, "cache_hit": reused})
            if result["screening_status"] != "screened":
                return {**result, "llm_calls": calls}
            findings.append(result["llm_output"])
        output = {"criteria": findings}
        audit = calls[0] if len(calls) == 1 else {}
        return {**audit, **self._assessment(output, criteria), "backend": "llm", "llm_output": output,
                "llm_calls": calls, "elapsed_seconds": sum(call.get("elapsed_seconds", 0) for call in calls if not call["cache_hit"])}

    def _screen_group(self, client, paper, criteria, offset=0):
        cfg = self.config
        try:
            if any(paper.get(key) is not None and not isinstance(paper[key], str) for key in ("title", "abstract")):
                raise ValueError("Title and abstract must be strings")
            validated = [CriterionInput.model_validate(c) for c in criteria]
        except ValueError as error:
            return {**Screener._unprocessed("screening_failed", f"Invalid screening input: {error}"), "backend": "llm"}
        ids = list(range(offset + 1, offset + len(criteria) + 1))
        if cfg["execution_mode"] == "per_criterion":
            criterion_payload = [
                {"statement": item.statement}
                for item in validated
            ]
        else:
            criterion_payload = [
                item.model_dump(exclude={"label"}) for item in validated
            ]
        payload = {
            "criteria": criterion_payload,
            "title": paper.get("title") or "",
            "abstract": paper.get("abstract") or "",
        }
        if "reviewer_assessments" in paper:
            payload["reviewer_assessments"] = paper["reviewer_assessments"]
        response_model, schema = self._contract(ids)
        text = json.dumps(payload, ensure_ascii=False)
        request = self._request(text, schema)
        audit = {"backend": "llm", "llm_request": request, "llm_response": None, "llm_attempts": []}
        if not payload["abstract"].strip():
            return {**Screener._unprocessed("missing_abstract", "missing_abstract"), **audit}
        if not criteria:
            return {**Screener._unprocessed("screening_failed", "no_criteria"), **audit}
        input_size = len(json.dumps(request, ensure_ascii=False).encode("utf-8"))
        budget = cfg["context_tokens"] - cfg["max_output_tokens"] - cfg["template_token_reserve"]
        audit["context_check"] = "server" if self.native_context_guard else "conservative_bytes"
        if not self.native_context_guard and input_size > budget:
            return {**Screener._unprocessed("input_too_long", "Conservative input budget exceeded; no text was truncated"), **audit}

        start = time.perf_counter()
        try:
            key_name = cfg["api_key_env"]
            key = os.environ.get(key_name) if key_name else None
            if key_name and not key:
                raise ValueError(f"Missing LLM credential: {key_name}")
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            endpoint = "api/generate" if cfg["provider"] == "ollama" else "chat/completions"

            def generate(current_request):
                with self._request_slots:
                    response = client.post(
                        cfg["base_url"].rstrip("/") + "/" + endpoint,
                        json=current_request,
                        headers=headers,
                    )
                    response.raise_for_status()
                    return response

            output_token_budget = cfg["max_output_tokens"]
            for attempt in range(cfg["response_retries"] + 1):
                audit["llm_response"] = None
                request = self._request(text, schema, output_token_budget)
                audit["llm_request"] = request
                response = retry_call(lambda: generate(request), retries=cfg["retries"],
                                      initial=cfg["retry_initial_seconds"], maximum=cfg["retry_max_seconds"])
                try:
                    body = response.json()
                    raw, completion = self._read_response(body)
                    audit["token_usage"] = ({"input": body.get("prompt_eval_count"), "output": body.get("eval_count")}
                                            if cfg["provider"] == "ollama" else body.get("usage"))
                    audit["llm_response"] = raw
                    if completion == "output_limit":
                        raise OutputTokenLimitError(
                            f"Model exhausted its {output_token_budget}-token output budget"
                        )
                    if completion == "refused":
                        raise ValueError("Model refused the screening request")
                    if completion != "complete":
                        raise ValueError("Model response was incomplete")
                    if not isinstance(raw, str):
                        raise ValueError("Model returned no final text")
                    output = validate_response(response_model, self._json_output(raw), paper)
                    assessment = self._assessment(output, criteria)
                    audit["llm_attempts"].append({
                        "response": raw, "error": None,
                        "output_token_budget": output_token_budget,
                    })
                    break
                except ValueError as error:
                    audit["llm_attempts"].append({
                        "response": audit["llm_response"], "error": str(error),
                        "output_token_budget": output_token_budget,
                    })
                    if attempt == cfg["response_retries"]:
                        raise
                    if isinstance(error, OutputTokenLimitError):
                        maximum = (
                            cfg["context_tokens"] - cfg["template_token_reserve"] - 1
                            if self.native_context_guard
                            else cfg["context_tokens"] - cfg["template_token_reserve"] - input_size
                        )
                        increased = min(
                            int(output_token_budget * cfg["output_token_retry_multiplier"]),
                            maximum,
                        )
                        if increased <= output_token_budget:
                            raise OutputTokenLimitError(
                                f"Model exhausted its {output_token_budget}-token output budget; "
                                "the context window leaves no room to increase it"
                            ) from error
                        output_token_budget = increased
        except (httpx.HTTPError, ValueError) as error:
            reason = str(error)
            status = "output_too_long" if isinstance(error, OutputTokenLimitError) else "screening_failed"
            if isinstance(error, httpx.HTTPStatusError):
                try:
                    detail = error.response.json().get("error", "")
                except (ValueError, AttributeError):
                    detail = ""
                if isinstance(detail, str) and detail:
                    reason = detail
                    if "context" in detail.lower() and any(word in detail.lower() for word in ("exceed", "length", "full")):
                        status = "input_too_long"
            if isinstance(error, httpx.ConnectError):
                reason = f"Cannot connect to LLM server at {cfg['base_url']}. Check that the server is running. {error}"
            elif isinstance(error, httpx.TimeoutException):
                reason = f"LLM server at {cfg['base_url']} timed out. Check server availability and the configured timeout."
            return {**Screener._unprocessed(status, reason), **audit,
                    "elapsed_seconds": time.perf_counter() - start}
        return {**assessment, **audit, "llm_output": output,
                "elapsed_seconds": time.perf_counter() - start}

    def _assessment(self, output, criteria):
        if self.config["execution_mode"] == "all_criteria":
            decision = {"yes": "Include", "no": "Exclude"}[output["decision"]]
            return {"screening_status": "screened", "decision": decision, "criteria": [],
                    "decision_confidence": None, "reason": "llm_paper_decision",
                    "review_order": Screener._review_order(decision, [])}
        findings = output.get("criteria") if "criteria" in output else [output]
        if not isinstance(findings, list) or len(findings) != len(criteria):
            raise ValueError("Expected exactly one finding per criterion in input order")
        results = []
        for criterion, item in zip(criteria, findings):
            if not isinstance(item, dict) or item.get("decision") not in {"yes", "no", "uncertain"}:
                raise ValueError("Invalid criterion decision")
            finding = {
                "yes": "supported",
                "no": "contradicted",
                "uncertain": "uncertain",
            }[item["decision"]]
            inclusion = criterion["type"] == "include"
            decision = {"supported": "passed" if inclusion else "hit",
                        "contradicted": "failed" if inclusion else "cleared",
                        "uncertain": "failed" if inclusion and criterion.get("explicit_evidence_required") else "unresolved"}[finding]
            results.append({"label": criterion["label"], "statement": criterion["statement"],
                            "type": criterion["type"], "backend": "llm", "finding": finding,
                            "explicit_evidence_required": criterion.get("explicit_evidence_required", False),
                            "criterion_decision": decision, "reason": item.get("reason"),
                            "quote": item.get("quote"), "model_confidence": item.get("confidence")})
        return Screener.aggregate(results)

    def _request(self, payload, schema=None, max_output_tokens=None):
        cfg = self.config
        output_tokens = max_output_tokens or cfg["max_output_tokens"]
        schema = schema if schema is not None else self._contract([1])[1]
        instructions = self.instructions
        if cfg["structured_output"] != "json_schema":
            instructions += "\nResponse JSON schema:\n" + json.dumps(schema)
        if cfg["provider"] == "ollama":
            request = {
                "model": cfg["model"], "system": instructions, "prompt": payload,
                "stream": False, "keep_alive": cfg["keep_alive"],
                "options": {"num_ctx": cfg["context_tokens"], "num_predict": output_tokens},
            }
            if self.native_context_guard:
                request.update(truncate=False, shift=False)
            if cfg["temperature"] is not None:
                request["options"]["temperature"] = cfg["temperature"]
            if cfg["structured_output"] != "none":
                request["format"] = schema if cfg["structured_output"] == "json_schema" else "json"
            if cfg["think"] is not None:
                request["think"] = cfg["think"]
            return request
        request = {
            "model": cfg["model"],
            "messages": [{"role": "system", "content": instructions}, {"role": "user", "content": payload}],
            cfg["token_parameter"]: output_tokens,
        }
        if cfg["temperature"] is not None:
            request["temperature"] = cfg["temperature"]
        if cfg["reasoning_effort"] is not None:
            request["reasoning_effort"] = cfg["reasoning_effort"]
        if cfg["structured_output"] == "json_schema":
            request["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "screening_decision", "strict": True, "schema": schema,
            }}
        elif cfg["structured_output"] == "json":
            request["response_format"] = {"type": "json_object"}
        return request

    def _read_response(self, response):
        if not isinstance(response, dict):
            raise ValueError("Model response must be an object")
        if self.config["provider"] == "ollama":
            raw = response.get("response")
            reason = response.get("done_reason")
            completion = (
                "complete" if response.get("done") is True and reason == "stop"
                else "output_limit" if reason in {"length", "max_tokens"}
                else "incomplete"
            )
        else:
            choices = response.get("choices")
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                raise ValueError("Expected one model response choice")
            choice = choices[0]
            message = choice.get("message")
            if not isinstance(message, dict):
                raise ValueError("Model returned no message")
            raw = message.get("content")
            reason = choice.get("finish_reason")
            completion = (
                "refused" if message.get("refusal")
                else "complete" if reason == "stop"
                else "output_limit" if reason in {"length", "max_tokens"}
                else "incomplete"
            )
        return raw if isinstance(raw, str) else None, completion

    @staticmethod
    def _json_output(raw):
        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate JSON key: {key}")
                result[key] = value
            return result
        def invalid_constant(value):
            raise ValueError(f"Invalid JSON constant: {value}")
        return json.loads(raw, object_pairs_hook=unique_keys, parse_constant=invalid_constant)
