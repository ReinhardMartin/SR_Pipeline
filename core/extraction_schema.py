import math


STATUSES = ("extracted", "not_reported", "not_applicable", "unclear", "conflicting_evidence")
CONFIDENCES = ("high", "medium", "low")


def valid_value(value, kind: str) -> bool:
    match kind:
        case "integer":
            return type(value) is int
        case "number":
            return type(value) is int or type(value) is float and math.isfinite(value)
        case "boolean":
            return type(value) is bool
        case "array":
            return isinstance(value, list) and all(isinstance(item, str) for item in value)
        case "string":
            return isinstance(value, str)
        case _:
            return False


def field_errors(data, field: dict, *, full: bool = False) -> list[str]:
    if not isinstance(data, dict):
        return ["Expected a field object"]
    errors = []
    if not {"value", "status", "confidence"} <= data.keys():
        errors.append("Missing required field properties")
    status, value = data.get("status"), data.get("value")
    if status not in STATUSES:
        errors.append("Invalid status")
    if data.get("confidence") not in CONFIDENCES:
        errors.append("Invalid confidence")
    if data.get("error") or data.get("validation_errors"):
        errors.append("Field contains errors")
    if data.get("notes") is not None and not isinstance(data["notes"], str):
        errors.append("Notes must be text or null")
    if value is not None and not valid_value(value, field.get("value_type", "string")):
        errors.append("Value has the wrong type")
    if status == "extracted" and value is None:
        errors.append("Extracted status requires a value")
    if status in ("not_reported", "not_applicable") and value is not None:
        errors.append("Absent values must be null")
    allowed = field.get("allowed_values")
    if value is not None and allowed:
        values = value if isinstance(value, list) else [value]
        if any(item not in allowed for item in values):
            errors.append("Value is outside the allowed set")
    if full:
        quote = data.get("source_quote")
        if "source_quote" not in data or quote is not None and not isinstance(quote, str):
            errors.append("Source quote must be text or null")
        if status == "extracted" and (not isinstance(quote, str) or not quote.strip()):
            errors.append("Extracted status requires a source quote")
        if data.get("source_section") is not None and not isinstance(data["source_section"], str):
            errors.append("Source section must be text or null")
    else:
        evidence = data.get("evidence")
        if not isinstance(evidence, list):
            errors.append("Evidence must be a list")
        elif any(
            not isinstance(item, dict) or any(
                not isinstance(item.get(key), str) or not item[key].strip()
                for key in ("source_id", "quote")
            ) for item in evidence
        ):
            errors.append("Evidence requires source IDs and quotes")
        if status == "extracted" and not evidence:
            errors.append("Extracted status requires evidence")
    return errors
