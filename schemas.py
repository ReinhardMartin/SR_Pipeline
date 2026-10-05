from typing import Literal

from pydantic import BaseModel, Field
from screening.domain.models import (
    ScreeningAgreement,
    ScreeningBatchInfo,
    ScreeningCriteria,
    ScreeningCriterion,
    ScreeningCriterionLabelRequest,
    ScreeningCriterionResult,
    ScreeningInputIssue,
    ScreeningLabelRequest,
    ScreeningPanelReviewRequest,
    PanelConsensus,
    ReviewAssessment,
    ScreeningPaperResult,
    ScreeningResults,
    ScreeningReviewOrder,
    ScreeningThresholdRequest,
    ScreeningUploadResult,
)


class FieldSpec(BaseModel):
    id: str = Field(min_length=1, pattern="^[a-z0-9][a-z0-9_-]*$")
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)
    rules: str | None = None
    value_type: Literal["string", "integer", "number", "boolean", "array"] = "string"
    allowed_values: list[str] | None = None


class FieldGroup(BaseModel):
    label: str = Field(min_length=1)
    fields: list[FieldSpec]


class DataTable(BaseModel):
    schema_version: str = "3.0"
    groups: list[FieldGroup]


class PlanEntry(BaseModel):
    queries: list[str] = Field(min_length=1)
    keywords: list[str] = Field(min_length=2)


class PaperStatus(BaseModel):
    stem: str
    has_pdf: bool
    has_parsed: bool
    has_index: bool
    has_evidence: bool
    has_extraction: bool
    has_full_extraction: bool


class JobStatus(BaseModel):
    status: str
    message: str
    error: str | None = None


class JobCreated(BaseModel):
    job_id: str


class UploadResult(BaseModel):
    stem: str


class StatusResult(BaseModel):
    status: str


class BulkRunResult(BaseModel):
    job_ids: dict[str, str]
    skipped: list[str]


class FieldResult(BaseModel):
    value: str | None = None
    status: str | None = None
    confidence: str | None = None
    evidence: list[str] = Field(default_factory=list)
    error: str | None = None


class PaperExport(BaseModel):
    stem: str
    fields: dict[str, FieldResult]


class ExportResult(BaseModel):
    fields: list[str]
    papers: list[PaperExport]


class EncodeRequest(BaseModel):
    text: str = Field(min_length=1)
    is_query: bool = True


class EncodeResult(BaseModel):
    dim: int
    norm: float
    vector: list[float]


class SearchRequest(BaseModel):
    stem: str
    query: str
    top_n: int = 5


class RerankRequest(BaseModel):
    query: str = Field(min_length=1)
    documents: list[str] = Field(min_length=1)


class RerankedDocument(BaseModel):
    document: str
    score: float


class NliRequest(BaseModel):
    premise: str = Field(min_length=1)
    hypothesis: str | None = None
    hypotheses: list[str] | None = None


class NliResult(BaseModel):
    hypothesis: str
    entailment: float


class RecoveryLinkRequest(BaseModel):
    import_id: str
    library_row: int = Field(ge=1)
    screening_row: int = Field(ge=1)
