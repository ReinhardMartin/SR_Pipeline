import copy
import math
from numbers import Real
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from core.settings import ScreeningLlmConfig


class Response(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


def build_response_model(options: ScreeningLlmConfig):
    choices = ["yes", "no"]
    if options.allow_uncertainty:
        choices.append("uncertain")
    description = "yes: established by concrete title/abstract evidence; no: contradicted or not established."
    if options.allow_uncertainty:
        description += " uncertain: too incomplete or conflicting to decide."
    fields = {}
    if options.execution_mode == "all_criteria":
        choices = ["yes", "no"]
        description = "yes: every criterion is established; no: at least one criterion is contradicted or not established."
        fields = {}
    if options.include_reason:
        fields["reason"] = (str, Field(
            min_length=1,
            pattern=r"\S",
            description=(
                "State only the decisive title/abstract evidence and why it establishes or fails to "
                "establish the criterion. Do not repeat the criterion and do not summarize the abstract."
            ),
        ))
    fields["decision"] = (Literal[tuple(choices)], Field(description=description))
    if options.extract_evidence:
        fields["quote"] = (str | None, Field(description="Exact contiguous quote from the title or abstract supporting the assessment; null when no relevant passage exists."))
    if options.include_confidence_score:
        fields["confidence"] = (
            float | None,
            Field(
                ge=0,
                le=1,
                description=(
                    "Optional self-reported certainty as a decimal from 0 to 1; "
                    "use null if unavailable. This is not a calibrated probability."
                ),
            ),
        )
    assessment = create_model("ScreeningAssessment", __base__=Response, **fields)
    return assessment


def validate_response(model, value, paper):
    candidate = copy.deepcopy(value)
    if isinstance(candidate, dict):
        assessments = candidate.get("criteria", [candidate])
        if isinstance(assessments, list):
            for assessment in assessments:
                if not isinstance(assessment, dict) or "confidence" not in assessment:
                    continue
                confidence = assessment["confidence"]
                if (
                    isinstance(confidence, Real)
                    and not isinstance(confidence, bool)
                    and math.isfinite(float(confidence))
                    and not 0 <= float(confidence) <= 1
                ):

                    assessment["confidence"] = None
    output = model.model_validate(candidate).model_dump()
    for assessment in output.get("criteria", [output]):
        quote = assessment.get("quote")
        if quote is not None and (not quote.strip() or not any(quote in (paper.get(key) or "") for key in ("title", "abstract"))):
            raise ValueError("Evidence quote must occur verbatim in the title or abstract")
    return output
