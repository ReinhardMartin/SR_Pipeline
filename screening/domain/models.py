from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ScreeningRecord(BaseModel):
    """A normalized citation accepted by the screening domain."""

    row: int = Field(ge=1)
    title: str
    abstract: str
    authors: str = ""
    year: str = ""
    doi: str = ""
    journal: str = ""
    url: str = ""
    source_id: str = ""
    extra: dict[str, str] = Field(default_factory=dict)


class BatchMetadata(BaseModel):
    """Versioned metadata persisted with every imported citation batch."""

    model_config = ConfigDict(extra="forbid", strict=True)

    filename: str
    record_schema_version: Literal["1.0"] = "1.0"


@dataclass(frozen=True, slots=True)
class BatchSummary:
    batch_id: str
    filename: str
    paper_count: int
    has_results: bool


class ScreeningCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    explicit_evidence_required: bool = False
    label: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    type: Literal["include", "exclude"]
    confidence_threshold: float = Field(default=0.8, ge=0.5, le=1.0)


class ScreeningCriteria(BaseModel):
    criteria: list[ScreeningCriterion]


class ScreeningCriteriaDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.2"]
    criteria: list[ScreeningCriterion]


class ScreeningPrompts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    all_criteria: str = Field(min_length=1, max_length=50000, pattern=r"\S")
    per_criterion: str = Field(min_length=1, max_length=50000, pattern=r"\S")
    judge: str = Field(min_length=1, max_length=50000, pattern=r"\S")


class ScreeningUploadResult(BaseModel):
    batch_id: str
    filename: str
    paper_count: int
    columns: list[str]


class ScreeningBatchInfo(BaseModel):
    batch_id: str
    filename: str
    paper_count: int
    has_results: bool


class ScreeningCriterionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quote: str | None = None
    model_confidence: float | None = None
    human_finding: Literal["supported", "contradicted", "not_established", "uncertain"] | None = None
    explicit_evidence_required: bool = False
    label: str
    statement: str
    type: Literal["include", "exclude"]
    backend: Literal["nli", "llm"] = "nli"
    reason: str | None = None
    entailment: float | None = None
    contradiction: float | None = None
    neutral: float | None = None
    status: Literal["entailment", "contradiction", "neutral"] | None = None
    confidence: float | None = None
    confidence_threshold: float | None = Field(default=None, ge=0.5, le=1.0)
    finding: Literal["supported", "contradicted", "not_established", "uncertain"] | None = None
    criterion_decision: Literal["passed", "failed", "cleared", "hit", "unresolved"]
    human_decision: Literal["passed", "failed", "cleared", "hit", "unresolved"] | None = None
    human_labelled_at: str | None = None


class ScreeningInputIssue(BaseModel):
    criterion: str
    status: Literal["input_too_long"]
    token_count: int
    token_limit: int
    excess_tokens: int


class ScreeningReviewOrder(BaseModel):
    group: Literal["uncertain", "missing_evidence", "confirm_inclusion_gaps", "optional_check", "needs_processing"]
    group_rank: int = Field(ge=0, le=4)
    blocking_count: int = Field(ge=0)
    score_gap: float | None = Field(default=None, ge=0, le=1)


class ReviewAssessment(BaseModel):
    """A normalized assessment produced by a model or human reviewer."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["primary", "second_reviewer", "judge"]
    kind: Literal["llm", "human"]
    decision: Literal["Include", "Likely include", "Exclude", "Maybe"] | None = None
    reason: str | None = None
    quote: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    criteria: list[ScreeningCriterionResult] = Field(default_factory=list)
    model: str | None = None
    reviewed_at: str | None = None
    screening_status: str = "screened"


class PanelConsensus(BaseModel):
    """Public, typed view of the persisted panel state for one paper."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = "2.0"
    status: Literal[
        "pending",
        "awaiting_second_reviewer",
        "second_reviewer_failed",
        "awaiting_judge",
        "judge_failed",
        "agreed",
        "adjudicated",
        "manual_full_text_required",
        "first_failed",
    ]
    decision: Literal["Include", "Exclude"] | None = None
    fingerprint: str | None = None
    criteria: list[ScreeningCriterion] = Field(default_factory=list)
    roles: dict[Literal["second_reviewer", "judge"], Literal["human", "llm"]] = Field(default_factory=dict)
    require_judge_reason: bool = True
    primary: ReviewAssessment | None = None
    second_reviewer: ReviewAssessment | None = None
    judge: ReviewAssessment | None = None


class PanelReviews(BaseModel):
    model_config = ConfigDict(extra="forbid")

    second_reviewer: ReviewAssessment | None = None
    judge: ReviewAssessment | None = None

    @model_validator(mode="after")
    def validate_roles_and_decisions(self):
        for role in ("second_reviewer", "judge"):
            assessment = getattr(self, role)
            if assessment is None:
                continue
            if assessment.role != role:
                raise ValueError(f"{role} assessment has the wrong role")
            if assessment.decision not in {None, "Include", "Exclude"}:
                raise ValueError(f"{role} decisions must be Include or Exclude")
        return self


class PanelHistoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row: int = Field(ge=1)
    role: Literal["second_reviewer", "judge"]
    previous: ReviewAssessment | None
    changed_at: str


class PanelState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"]
    fingerprint: str = Field(min_length=64, max_length=64)
    config: dict
    criteria: list[ScreeningCriterion]
    actor_identities: dict[str, object]
    reviews: dict[str, PanelReviews]
    history: list[PanelHistoryEntry]


class ScreeningPaperResult(ScreeningRecord):
    model_config = ConfigDict(extra="forbid")

    consensus: PanelConsensus | None = None
    backend: Literal["nli", "llm"] = "nli"
    llm_request: dict | None = None
    llm_response: str | None = None
    llm_attempts: list[dict] = Field(default_factory=list)
    llm_calls: list[dict] = Field(default_factory=list)
    llm_output: dict | None = None
    context_check: str | None = None
    token_usage: dict | None = None
    elapsed_seconds: float | None = None
    screening_status: Literal[
        "screened", "missing_abstract", "screening_failed", "input_too_long",
        "output_too_long",
    ]
    input_issues: list[ScreeningInputIssue] = Field(default_factory=list)
    decision: Literal["Include", "Likely include", "Exclude", "Maybe"] | None = None
    decision_confidence: float | None = None
    review_order: ScreeningReviewOrder
    reason: str
    criteria: list[ScreeningCriterionResult]
    human_decision: Literal["Include", "Likely include", "Exclude", "Maybe"] | None = None
    human_labelled_at: str | None = None


class ScreeningResults(BaseModel):
    batch_id: str
    criteria: list[ScreeningCriterion]
    papers: list[ScreeningPaperResult]


class ScreeningPanelReviewRequest(BaseModel):
    model_config = {"extra": "forbid"}
    decision: Literal["Include", "Exclude"]
    reason: str = Field(default="", max_length=10000)
    fingerprint: str = Field(min_length=64, max_length=64)


class ScreeningLabelRequest(BaseModel):
    human_decision: Literal["Include", "Likely include", "Exclude", "Maybe"] | None = None


class ScreeningCriterionLabelRequest(BaseModel):
    human_finding: Literal["supported", "contradicted", "not_established", "uncertain"] | None = None
    human_decision: Literal["passed", "failed", "cleared", "hit", "unresolved"] | None = None


class ScreeningThresholdRequest(BaseModel):
    criterion_thresholds: dict[str, float] = Field(min_length=1)


class ScreeningAgreement(BaseModel):
    reviewed: int
    agreeing: int
    agreement_rate: float | None
    confusion: dict[str, dict[str, int]]

