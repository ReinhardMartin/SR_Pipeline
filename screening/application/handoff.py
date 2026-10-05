from pathlib import Path
from uuid import UUID

from screening.infrastructure.citations import TEMPLATE_COLUMNS, normalize_doi, text


RECOVERY_COLUMNS = [
    "record_id", "title", "authors", "year", "doi", "journal", "url", "source_id",
    "decision", "decision_source", "recovery_reason", "pdf_filename", "file_status",
]


def recovery_list(batch_id: str, papers: list[dict], pdf_directory: Path) -> list[dict]:
    batch_id = str(UUID(batch_id))
    output = []
    seen = set()
    for paper in papers:
        missing = paper.get("screening_status") == "missing_abstract"
        human = paper.get("human_decision")
        decision = human if human in {"Include", "Likely include", "Exclude", "Maybe"} else paper.get("decision")
        consensus = paper.get("consensus")
        if consensus is not None:
            decision = consensus.get("decision")
            if decision not in {"Include", "Exclude"} and not missing:
                continue
        if decision == "Exclude":
            continue
        if decision not in {"Include", "Likely include", "Maybe"} and not missing:
            continue
        row = paper["row"]
        if not isinstance(row, int) or isinstance(row, bool) or row < 1 or row in seen:
            raise ValueError("Screening records must have unique positive row numbers")
        seen.add(row)
        record_id = f"sr-{batch_id}-r{row}"
        filename = f"{record_id}.pdf"
        extra = {key.casefold(): value for key, value in paper.get("extra", {}).items()}
        metadata = {key: text(paper.get(key) or extra.get(key)) for key in TEMPLATE_COLUMNS if key != "abstract"}
        metadata["doi"] = normalize_doi(metadata["doi"])
        path = pdf_directory / filename
        file_status = "missing"
        if path.is_file():
            try:
                with path.open("rb") as handle:
                    file_status = "present" if handle.read(5) == b"%PDF-" else "invalid_signature"
            except OSError:
                file_status = "unreadable"
        output.append({
            **metadata, "record_id": record_id, "batch_id": batch_id, "row": row,
            "decision": decision, "decision_source": "panel" if consensus is not None else "human" if human in {"Include", "Likely include", "Exclude", "Maybe"} else "screening",
            "recovery_reason": "advance_to_full_text" if decision == "Include" else "confirm_inclusion_gaps" if decision == "Likely include" else "resolve_missing_abstract" if missing else "resolve_uncertainty",
            "pdf_filename": filename, "file_status": file_status,
        })
    return output
