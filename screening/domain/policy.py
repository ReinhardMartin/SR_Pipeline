from __future__ import annotations

import math
from numbers import Real

NLI_LABELS = ("entailment", "contradiction", "neutral")


def _status(scores: dict) -> str:
    return max(NLI_LABELS, key=scores.get)


class Screener:
    def __init__(
        self,
        nli,
        confidence_threshold: float = 0.8,
        criterion_thresholds: dict[str, float] | None = None,
    ):
        self._nli = nli
        self._confidence_threshold = confidence_threshold
        self._criterion_thresholds = criterion_thresholds or {}

    def _threshold(self, criterion: dict) -> float:
        if criterion["label"] in self._criterion_thresholds:
            return float(self._criterion_thresholds[criterion["label"]])
        value = criterion.get("confidence_threshold")
        return float(self._confidence_threshold if value is None else value)

    def screen_batch(self, papers: list[dict], criteria: list[dict]) -> list[dict]:
        if not criteria:
            return [self._unprocessed("screening_failed", "no_criteria") for _ in papers]

        results: list[dict | None] = [None] * len(papers)
        eligible_indices: list[int] = []
        texts: list[str] = []
        for index, paper in enumerate(papers):
            title = str(paper.get("title") or "").strip()
            abstract = str(paper.get("abstract") or "").strip()
            if not abstract:
                results[index] = self._unprocessed("missing_abstract", "missing_abstract")
                continue
            eligible_indices.append(index)
            texts.append(f"Title: {title}\nAbstract: {abstract}".strip())

        pairs = [(text, criterion["statement"]) for text in texts for criterion in criteria]
        classified = self._nli.classify_batch(pairs) if pairs else []
        if not isinstance(classified, list) or len(classified) != len(pairs):
            for index in eligible_indices:
                results[index] = self._unprocessed("screening_failed", "invalid_classifier_response")
            return results

        criterion_count = len(criteria)
        for batch_position, paper_index in enumerate(eligible_indices):
            row = classified[
                batch_position * criterion_count : (batch_position + 1) * criterion_count
            ]
            issues = [
                {"criterion": criterion["label"], **scores}
                for criterion, scores in zip(criteria, row)
                if isinstance(scores, dict) and scores.get("status") == "input_too_long"
            ]
            if issues:
                results[paper_index] = {
                    **self._unprocessed("input_too_long", "input_too_long"),
                    "input_issues": issues,
                }
                continue
            if any(
                not isinstance(scores, dict) or any(
                    not isinstance(scores.get(label), Real)
                    or isinstance(scores[label], bool)
                    or not 0 <= scores[label] <= 1
                    for label in NLI_LABELS
                )
                or not math.isclose(sum(scores[label] for label in NLI_LABELS), 1.0, abs_tol=1e-4)
                for scores in row
            ):
                results[paper_index] = self._unprocessed("screening_failed", "invalid_classifier_scores")
                continue
            per_criterion = [
                self._criterion_result(criterion, scores)
                for criterion, scores in zip(criteria, row)
            ]
            results[paper_index] = self._decide(per_criterion)

        return [
            result if result is not None else self._unprocessed("screening_failed", "screening_failed")
            for result in results
        ]

    def _criterion_result(self, criterion: dict, scores: dict) -> dict:
        entailment = float(scores["entailment"])
        contradiction = float(scores["contradiction"])
        neutral = float(scores["neutral"])
        status = _status(scores)
        confidence = max(entailment, contradiction, neutral)
        threshold = self._threshold(criterion)
        return {
            "label": criterion["label"],
            "type": criterion["type"],
            "statement": criterion["statement"],
            "explicit_evidence_required": criterion.get("explicit_evidence_required", False),
            "entailment": entailment,
            "contradiction": contradiction,
            "neutral": neutral,
            "status": status,
            "confidence": confidence,
            "confidence_threshold": threshold,
            "finding": self._finding(status, confidence, threshold),
            "criterion_decision": self._criterion_decision(
                criterion["type"], status, confidence, threshold,
                criterion.get("explicit_evidence_required", False),
            ),
        }

    @staticmethod
    def _criterion_decision(
        criterion_type: str, status: str, confidence: float, threshold: float,
        explicit_evidence_required: bool = False,
    ) -> str:
        if confidence < threshold:
            return "unresolved"
        if status == "neutral":
            return "failed" if criterion_type == "include" and explicit_evidence_required else "unresolved"
        if criterion_type == "include":
            return "passed" if status == "entailment" else "failed"
        return "hit" if status == "entailment" else "cleared"

    @staticmethod
    def _finding(status: str, confidence: float, threshold: float) -> str:
        if confidence < threshold:
            return "uncertain"
        return {"entailment": "supported", "contradiction": "contradicted", "neutral": "not_established"}[status]

    @staticmethod
    def _unprocessed(screening_status: str, reason: str) -> dict:
        return {
            "screening_status": screening_status,
            "decision": None,
            "decision_confidence": None,
            "review_order": {"group": "needs_processing", "group_rank": 4,
                             "blocking_count": 0, "score_gap": None},
            "criteria": [],
            "reason": reason,
        }

    def _decide(self, per_criterion: list[dict]) -> dict:
        criteria = []
        for criterion in per_criterion:
            item = dict(criterion)
            item["status"] = _status(item)
            item["confidence"] = max(float(item[label]) for label in NLI_LABELS)
            item["confidence_threshold"] = self._threshold(item)
            item["finding"] = self._finding(item["status"], item["confidence"], item["confidence_threshold"])
            item["criterion_decision"] = self._criterion_decision(
                item["type"], item["status"], item["confidence"],
                item["confidence_threshold"], item.get("explicit_evidence_required", False),
            )
            criteria.append(item)

        return self.aggregate(criteria)

    @staticmethod
    def aggregate(criteria: list[dict]) -> dict:
        includes = [item for item in criteria if item["type"] == "include"]
        excludes = [item for item in criteria if item["type"] == "exclude"]
        failed_inclusions = [item for item in includes if item["criterion_decision"] == "failed"]
        exclusion_hits = [item for item in excludes if item["criterion_decision"] == "hit"]
        unresolved = [item for item in criteria if item["criterion_decision"] == "unresolved"]
        all_inclusions_passed = bool(includes) and all(
            item["criterion_decision"] == "passed" for item in includes
        )
        all_exclusions_cleared = all(
            item["criterion_decision"] == "cleared" for item in excludes
        )

        if failed_inclusions or exclusion_hits:
            failures = failed_inclusions + exclusion_hits
            decision = "Exclude"
            reason = "criterion_failed:" + ", ".join(item["label"] for item in failures)
            decision_confidence = max((item["confidence"] for item in failures if item.get("confidence") is not None), default=None)
        elif all_inclusions_passed and all_exclusions_cleared:
            decision = "Include"
            reason = "all_criteria_passed"
            decision_confidence = min((item["confidence"] for item in criteria if item.get("confidence") is not None), default=None)
        else:
            likely_include = (
                all_exclusions_cleared
                and any(item["criterion_decision"] == "passed" for item in includes)
                and all(item["finding"] == "not_established" for item in unresolved)
            )
            decision = "Likely include" if likely_include else "Maybe"
            labels = ", ".join(item["label"] for item in unresolved)
            reason = f"unresolved_criteria:{labels}"
            decision_confidence = None

        return {
            "screening_status": "screened",
            "decision": decision,
            "decision_confidence": decision_confidence,
            "review_order": Screener._review_order(decision, criteria),
            "criteria": criteria,
            "reason": reason,
        }

    @staticmethod
    def _review_order(decision: str, criteria: list[dict]) -> dict:
        unresolved = [c for c in criteria if c["criterion_decision"] == "unresolved"]
        uncertain = [c for c in unresolved if c["finding"] == "uncertain"]
        if decision == "Maybe":
            group, rank = ("uncertain", 0) if uncertain else ("missing_evidence", 1)
            relevant = uncertain or unresolved
        elif decision == "Likely include":
            group, rank, relevant = "confirm_inclusion_gaps", 2, unresolved
        else:
            group, rank = "optional_check", 3
            relevant = [c for c in criteria if c["criterion_decision"] in {"failed", "hit"}] if decision == "Exclude" else criteria
        gaps = []
        for criterion in relevant:
            if any(criterion.get(label) is None for label in NLI_LABELS):
                continue
            scores = sorted((criterion[label] for label in NLI_LABELS), reverse=True)
            gaps.append(scores[0] - scores[1])
        return {"group": group, "group_rank": rank,
                "blocking_count": len(unresolved) if decision == "Maybe" else 0,
                "score_gap": min(gaps) if gaps else None}
