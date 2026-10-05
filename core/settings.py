from pathlib import Path
import json
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
CONFIG_PATH = Path(os.environ.get("PIPELINE_CONFIG", ROOT / "config/pipeline_config.json")).expanduser().resolve()


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class LlmConfig(ConfigModel):
    provider: Literal["openai_compatible"] = "openai_compatible"
    model: str
    base_url: str
    api_key_env: str
    max_workers: int = Field(default=2, ge=1)
    max_concurrent_requests: int = Field(default=2, ge=1)
    retries: int = Field(default=2, ge=0, le=20)
    timeout: float = Field(default=120, gt=0)
    retry_initial_seconds: float = Field(default=1, gt=0)
    retry_max_seconds: float = Field(default=60, gt=0)
    max_tokens: int = Field(default=1024, ge=1)
    token_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    temperature: float | None = Field(default=0.0, ge=0.0, le=2.0)
    json_mode: bool = True
    planner_batch_size: int = Field(default=4, ge=1)
    max_group_evidence_chunks: int = Field(default=24, ge=1, le=100)


class ChunkingConfig(ConfigModel):
    tokenizer: str = "cl100k_base"
    max_tokens: int
    min_tokens: int
    overlap_tokens: int
    caption_max_tokens: int = 45
    chunk_level: Literal["paragraph", "sentence"] = "paragraph"


class RetrievalConfig(ConfigModel):
    candidates_per_query: int = Field(ge=1)
    use_prefilter: bool = False
    rerank_top_k_per_query: int = Field(default=2, ge=1)
    max_evidence_chunks: int = Field(default=12, ge=1)
    max_fused_candidates: int = Field(default=100, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    fusion_reserve: int = Field(default=4, ge=0)
    keyword_searcher: Literal["stem", "bm25"] = "stem"
    language: str = "english"


class EmbeddingConfig(ConfigModel):
    service_url: str | None = None
    model: str
    revision: str | None = None
    batch_size: int = Field(default=32, ge=1)
    device: str | None = None
    api_key_env: str = ""
    retries: int = Field(default=2, ge=0, le=20)
    query_prefix: str = ""
    doc_prefix: str = ""
    timeout: float = Field(default=30.0, gt=0)


class RerankerConfig(ConfigModel):
    device: str | None = None
    retries: int = Field(default=2, ge=0, le=20)
    api_key_env: str = ""
    service_url: str | None = None
    model: str
    revision: str | None = None
    timeout: float = Field(default=30.0, gt=0)


class NliConfig(ConfigModel):
    device: str | None = None
    model: str = Field(min_length=1)
    revision: str | None = None
    batch_size: int = Field(default=16, ge=1)


class SpladeConfig(ConfigModel):
    model: str
    revision: str | None = None
    batch_size: int = 16
    max_length: int = 256


class FullExtractionConfig(ConfigModel):
    tokenizer: str = "cl100k_base"
    max_context_tokens: int = Field(gt=0)
    max_tokens: int = Field(default=4096, gt=0)


class ExecutionConfig(ConfigModel):
    warm_models: bool = False
    max_concurrent_papers: int = Field(default=1, ge=1, le=16)
    mineru_batch_size: int = Field(default=10, ge=1, le=100)
    job_history_limit: int = Field(default=500, ge=10, le=10000)


class CalibrationConfig(ConfigModel):
    diagnostic_min_papers: int = Field(default=10, ge=1)
    neutral_rate_threshold: float = Field(default=0.5, gt=0, le=1)
    uncertain_rate_threshold: float = Field(default=0.3, gt=0, le=1)
    min_reviews: int = Field(default=20, ge=2)
    max_disagreement_rate: float = Field(default=0.0, ge=0, le=1)
    bootstrap_enabled: bool = False
    bootstrap_samples: int = Field(default=100, ge=20, le=1000)
    random_seed: int = 0
    max_threshold_spread: float = Field(default=0.1, ge=0, le=0.5)


class ScreeningLlmConfig(ConfigModel):
    provider: Literal["ollama", "openai_compatible"] = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    api_key_env: str = ""
    structured_output: Literal["json_schema", "json", "none"] = "json_schema"
    token_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    think: bool | str | None = None
    reasoning_effort: Literal["none", "low", "medium", "high"] | None = None
    model: str = Field(default="qwen2.5:1.5b", min_length=1)
    cache_revision: str = ""
    instructions: str = "config/prompts/screening_all_criteria_system.txt"
    criterion_instructions: str = "config/prompts/screening_per_criterion_system.txt"
    execution_mode: Literal["all_criteria", "per_criterion"] = "all_criteria"
    allow_uncertainty: bool = True
    include_reason: bool = True
    extract_evidence: bool = False
    include_confidence_score: bool = False
    temperature: float | None = Field(default=0, ge=0, le=2)
    context_tokens: int = Field(default=8192, ge=512)
    max_output_tokens: int = Field(default=512, ge=16)
    output_token_retry_multiplier: float = Field(default=2.0, gt=1, le=8)
    template_token_reserve: int = Field(default=256, ge=128)
    timeout: float = Field(default=120, gt=0)
    retries: int = Field(default=2, ge=0, le=5)
    response_retries: int = Field(default=1, ge=0, le=3)
    max_concurrent_requests: int = Field(default=1, ge=1, le=16)
    retry_initial_seconds: float = Field(default=1, gt=0)
    retry_max_seconds: float = Field(default=30, gt=0)
    keep_alive: str = "5m"

    @model_validator(mode="after")
    def check_settings(self):
        from urllib.parse import urlsplit
        url = urlsplit(self.base_url)
        if url.scheme not in {"http", "https"} or not url.netloc:
            raise ValueError("Screening LLM base_url must be an absolute HTTP(S) URL")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("Use api_key_env for credentials; base_url must not contain credentials, query, or fragment")
        if self.provider == "ollama" and self.reasoning_effort is not None:
            raise ValueError("Use think for Ollama; reasoning_effort is for OpenAI-compatible servers")
        if self.provider == "openai_compatible" and self.think is not None:
            raise ValueError("Use reasoning_effort for OpenAI-compatible servers; think is for Ollama")
        if self.retry_initial_seconds > self.retry_max_seconds:
            raise ValueError("Initial retry delay must not exceed its maximum")
        if self.max_output_tokens + self.template_token_reserve >= self.context_tokens:
            raise ValueError("Screening LLM context must leave room for input")
        return self


class ScreeningReviewerConfig(ConfigModel):
    kind: Literal["human", "llm"] = "human"
    llm: ScreeningLlmConfig | None = None

    @model_validator(mode="after")
    def check_model(self):
        if self.kind == "llm" and self.llm is None:
            raise ValueError("An LLM reviewer requires its own llm configuration")
        if self.kind == "human" and self.llm is not None:
            raise ValueError("A human reviewer must not include unused LLM configuration")
        if self.llm is not None and self.llm.execution_mode != "all_criteria":
            raise ValueError("Additional reviewers use an overall paper decision (all_criteria)")
        if self.kind == "llm" and self.llm.provider == "openai_compatible" and not self.llm.cache_revision.strip():
            raise ValueError("A remote panel model requires cache_revision so reviews can resume safely")
        return self


class ScreeningPanelConfig(ConfigModel):
    enabled: bool = False
    second_reviewer: ScreeningReviewerConfig = Field(default_factory=ScreeningReviewerConfig)
    judge: ScreeningReviewerConfig = Field(default_factory=ScreeningReviewerConfig)
    judge_instructions: str = "config/prompts/screening_judge_system.txt"
    require_judge_reason: bool = True


class ScreeningConfig(ConfigModel):
    backend: Literal["nli", "llm"] = "nli"
    llm: ScreeningLlmConfig = Field(default_factory=ScreeningLlmConfig)
    panel: ScreeningPanelConfig = Field(default_factory=ScreeningPanelConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    confidence_threshold: float = Field(default=0.8, ge=0.5, le=1.0)
    checkpoint_batch_size: int = Field(default=32, ge=1)

    @model_validator(mode="after")
    def check_panel_backend(self):
        if self.panel.enabled and self.backend != "llm":
            raise ValueError("The independent review panel is available only for LLM screening")
        return self


class PathsConfig(ConfigModel):
    input: str = "data/pdfs"
    output: str = "data/output"
    data_table: str = "data_table.json"
    query_plan: str = "data/query_plan.json"
    jobs: str = "data/jobs.json"
    screening: str = "data/screening"
    criteria: str = "screening_criteria.json"
    prompts: str = "config/prompts"
    parser: str = "config/parse_parameters.json"
    mineru: str = "config/mineru.json"


class UploadConfig(ConfigModel):
    max_records: int = Field(default=50000, gt=0)
    max_pdf_bytes: int = Field(default=52428800, gt=0)
    max_screening_bytes: int = Field(default=20971520, gt=0)


class ServerConfig(ConfigModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: Literal["critical", "error", "warning", "info", "debug", "trace"] = "info"
    shutdown_timeout: int = Field(default=180, ge=1)


class ParseConfig(ConfigModel):
    api_url: str | None = None
    backend: str = "pipeline"
    parse_method: Literal["auto", "txt", "ocr"] = "auto"
    language: str = "en"
    formula_enable: bool = True
    table_enable: bool = True
    image_analysis: bool = False
    server_url: str | None = None
    effort: str | None = None
    start_page_id: int = Field(default=0, ge=0)
    end_page_id: int | None = Field(default=None, ge=0)
    return_md: Literal[True] = True
    return_middle_json: bool = False
    return_model_output: bool = False
    return_content_list: bool = False
    return_images: bool = True
    response_format_zip: Literal[True] = True
    return_original_file: bool = False
    http_timeout: float = Field(default=600, gt=0)
    task_timeout: float = Field(default=3600, gt=0)


class PipelineConfig(ConfigModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    uploads: UploadConfig = Field(default_factory=UploadConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    llm: LlmConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    embedding: EmbeddingConfig
    reranker: RerankerConfig
    nli: NliConfig
    splade: SpladeConfig
    full_extraction: FullExtractionConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    screening: ScreeningConfig

    @model_validator(mode="after")
    def check_limits(self):
        if self.llm.retry_initial_seconds > self.llm.retry_max_seconds:
            raise ValueError("Initial retry delay must not exceed its maximum")
        if not 0 <= self.chunking.overlap_tokens < self.chunking.max_tokens:
            raise ValueError("chunking.overlap_tokens must be smaller than max_tokens")
        if not 0 < self.chunking.min_tokens <= self.chunking.max_tokens:
            raise ValueError("chunking.min_tokens must be within the chunk size")
        if self.full_extraction.max_tokens >= self.full_extraction.max_context_tokens:
            raise ValueError("Full extraction output reserve must be smaller than its context")
        from urllib.parse import urlsplit
        for url in (self.llm.base_url, self.embedding.service_url, self.reranker.service_url):
            if url is not None and (urlsplit(url).scheme not in {"http", "https"} or not urlsplit(url).netloc):
                raise ValueError("Provider URLs must be absolute HTTP(S) URLs")
        return self


def resolve_path(value: str) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_config(path: Path = CONFIG_PATH) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    for name, value in os.environ.items():
        if not name.startswith("PIPELINE__"):
            continue
        keys = name.removeprefix("PIPELINE__").lower().split("__")
        target = data
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        try:
            target[keys[-1]] = json.loads(value)
        except json.JSONDecodeError:
            target[keys[-1]] = value
    return PipelineConfig.model_validate(data).model_dump()


def load_prompt(name: str) -> str:
    directory = resolve_path(read_config()["paths"]["prompts"])
    text = (directory / f"{name}.txt").read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Empty prompt: {name}")
    return text


def render_prompt(name: str, **values) -> str:
    directory = resolve_path(read_config()["paths"]["prompts"])
    templates = json.loads(
        (directory / "llm_user_message_templates.json").read_text(encoding="utf-8")
    )
    return templates[name].format(**values)


def field_prompt(field: dict) -> str:
    return render_prompt("field", **{
        "label": field["label"], "description": field["description"],
        "rules": field.get("rules") or "", "value_type": field.get("value_type", "string"),
        "allowed_values": field.get("allowed_values"),
    })


