from __future__ import annotations

from collections import defaultdict
import random

from core.settings import CalibrationConfig
from screening.domain.policy import NLI_LABELS, Screener, _status


HUMAN_DECISIONS = {
    "include": {"passed", "failed", "unresolved"},
    "exclude": {"hit", "cleared", "unresolved"},
}


def reviewer_class(item: dict) -> str | None:
    finding = item.get("human_finding")
    if finding:
        return {"supported": "entailment", "contradicted": "contradiction",
                "not_established": "neutral"}.get(finding)
    if item.get("explicit_evidence_required") and item.get("human_decision") == "failed":
        return None
    return {"passed": "entailment", "hit": "entailment", "failed": "contradiction",
            "cleared": "contradiction", "unresolved": "neutral"}.get(item.get("human_decision"))


def labelled_criterion_items(papers: list[dict], label: str | None = None) -> list[dict]:
    items = []
    for paper in papers:
        if paper.get("screening_status", "screened") != "screened":
            continue
        for criterion in paper.get("criteria", []):
            if label is not None and criterion.get("label") != label:
                continue
            if reviewer_class(criterion) is not None:
                items.append(criterion)
    return items


def _model_decision(criterion: dict, threshold: float) -> str:
    status = _status(criterion)
    confidence = max(float(criterion[name]) for name in NLI_LABELS)
    return Screener._criterion_decision(
        criterion["type"], status, confidence, threshold,
        criterion.get("explicit_evidence_required", False),
    )


def evaluate_criterion_threshold(
    papers: list[dict], label: str, threshold: float
) -> dict:
    items = labelled_criterion_items(papers, label)
    return _prediction_metrics(items, threshold)


def _prediction_metrics(items: list[dict], threshold: float) -> dict:
    accepted = [item for item in items if max(item[name] for name in NLI_LABELS) >= threshold]
    errors = sum(_status(item) != reviewer_class(item) for item in accepted)
    return _metrics(len(items), len(accepted), errors, threshold)


def _metrics(reviewed: int, accepted: int, errors: int, threshold: float) -> dict:
    return {"threshold": threshold, "reviewed": reviewed, "accepted": accepted,
            "agreements": accepted - errors, "disagreements": errors,
            "uncertain": reviewed - accepted,
            "coverage": accepted / reviewed if reviewed else None,
            "disagreement_rate": errors / accepted if accepted else None}


def _best_threshold(items: list[dict], current: float, max_disagreement_rate: float = 0.0) -> dict | None:
    rows = sorted((max(item[name] for name in NLI_LABELS), _status(item) != reviewer_class(item))
                  for item in items)
    boundaries = sorted({0.5, 1.0} | {score for score, _ in rows if 0.5 <= score <= 1})
    candidates = sorted({0.5, 1.0, current} |
                        {(a + b) / 2 for a, b in zip(boundaries, boundaries[1:])})
    accepted, errors = len(rows), sum(error for _, error in rows)
    best = None
    index = 0
    for threshold in candidates:
        while index < len(rows) and rows[index][0] < threshold:
            accepted -= 1
            errors -= rows[index][1]
            index += 1
        if not accepted or errors / accepted > max_disagreement_rate:
            continue
        metrics = _metrics(len(rows), accepted, errors, threshold)
        key = (-accepted, errors, abs(threshold - current), threshold)
        if best is None or key < best[0]:
            best = key, metrics
    return best[1] if best else None


def _stability(items: list[dict], current: float, label: str, cfg: dict) -> dict:
    if not cfg["bootstrap_enabled"]:
        return {"status": "disabled", "lower": None, "upper": None, "samples": 0}
    if len(items) < cfg["min_reviews"]:
        return {"status": "insufficient_reviews", "lower": None, "upper": None, "samples": 0}
    groups = defaultdict(list)
    for item in items:
        groups[reviewer_class(item)].append(item)
    rng = random.Random(f"{cfg['random_seed']}:{label}")
    estimates = []
    for _ in range(cfg["bootstrap_samples"]):
        result = _best_threshold([item for key in sorted(groups)
                                  for item in rng.choices(groups[key], k=len(groups[key]))],
                                 current, cfg["max_disagreement_rate"])
        if result is not None:
            estimates.append(result["threshold"])
    if len(estimates) != cfg["bootstrap_samples"]:
        return {"status": "no_feasible_resamples", "lower": None, "upper": None,
                "samples": len(estimates)}
    estimates.sort()
    lower = estimates[int(0.1 * (len(estimates) - 1))]
    upper = estimates[int(0.9 * (len(estimates) - 1))]
    return {"status": "stable" if upper - lower <= cfg["max_threshold_spread"] else "unstable",
            "lower": lower, "upper": upper, "samples": len(estimates)}


def _diagnostics(items: list[dict], threshold: float, cfg: dict) -> dict:
    neutral = sum(_status(item) == "neutral" and max(item[name] for name in NLI_LABELS) >= threshold
                  for item in items)
    uncertain = sum(max(item[name] for name in NLI_LABELS) < threshold for item in items)
    total = len(items)
    neutral_rate = neutral / total if total else None
    uncertain_rate = uncertain / total if total else None
    flags = []
    if total >= cfg["diagnostic_min_papers"]:
        if neutral_rate >= cfg["neutral_rate_threshold"]:
            flags.append("frequently_neutral")
        if uncertain_rate >= cfg["uncertain_rate_threshold"]:
            flags.append("frequently_uncertain")
    return {"assessed": total, "neutral_count": neutral, "uncertain_count": uncertain,
            "neutral_rate": neutral_rate, "uncertain_rate": uncertain_rate, "flags": flags,
            "enough_papers": total >= cfg["diagnostic_min_papers"]}


def _aggregate(reports: list[dict], key: str, thresholds: dict[str, float]) -> dict:
    metrics = [report[key] for report in reports]
    result = _metrics(sum(m["reviewed"] for m in metrics), sum(m["accepted"] for m in metrics),
                      sum(m["disagreements"] for m in metrics), 0)
    result.pop("threshold")
    return {"criterion_thresholds": thresholds, **result}


def calibrate_thresholds(
    papers: list[dict],
    current_thresholds: dict[str, float],
    default_threshold: float = 0.8,
    options: dict | None = None,
) -> dict:
    cfg = CalibrationConfig(**(options or {})).model_dump()
    by_label = defaultdict(list)
    for paper in papers:
        if paper.get("screening_status", "screened") == "screened":
            for item in paper.get("criteria", []):
                by_label[item["label"]].append(item)
    reports = []
    current_map = {}
    suggested_map = {}
    for label in current_thresholds or by_label:
        threshold = float(current_thresholds.get(label, default_threshold))
        all_items = by_label[label]
        items = [item for item in all_items if reviewer_class(item) is not None]
        current = _prediction_metrics(items, threshold)
        candidate = _best_threshold(items, threshold, cfg["max_disagreement_rate"])
        status = "no_reviews" if not items else "no_feasible_threshold" if candidate is None else "suggested"
        stability = _stability(items, threshold, label, cfg)
        suggested = candidate or current
        current_map[label] = threshold
        suggested_map[label] = suggested["threshold"]
        reports.append({
            "label": label, "type": all_items[0]["type"] if all_items else None,
            "reviewed": len(items), "ready": candidate is not None, "tuning_status": status,
            "current_threshold": threshold, "suggested_threshold": suggested["threshold"],
            "candidate_threshold": candidate["threshold"] if candidate else None, "stability": stability,
            "max_disagreement_rate": cfg["max_disagreement_rate"],
            "diagnostics": _diagnostics(all_items, threshold, cfg),
            "current": current, "suggested": suggested,
        })
    return {
        "ready": bool(reports) and all(report["ready"] for report in reports),
        "warning": "Choose the threshold accepting the most reviewed predictions within the disagreement limit. "
                   "Neutral agreement is counted independently of eligibility rules. "
                   "Resampling is optional and does not block suggestions; these are in-sample results.",
        "current": _aggregate(reports, "current", current_map),
        "suggested": _aggregate(reports, "suggested", suggested_map),
        "criteria": reports,
    }
