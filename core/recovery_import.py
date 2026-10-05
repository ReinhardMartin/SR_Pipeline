from collections import defaultdict
import csv
import io
import re
import unicodedata
from zipfile import BadZipFile
from openpyxl.utils.exceptions import InvalidFileException
from pathlib import Path, PureWindowsPath

import openpyxl

from screening.infrastructure.citations import normalize_doi, text


LIBRARY_COLUMNS = ["title", "authors", "year", "doi", "pdf_path"]


def read_library(data: bytes, filename: str, max_records: int) -> list[dict]:
    suffix = Path(filename).suffix.lower()
    workbook = None
    try:
        if suffix == ".csv":
            rows = iter(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
        elif suffix == ".xlsx":
            workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
            rows = iter(workbook.active.values)
        else:
            raise ValueError("Upload a CSV or XLSX citation list")
        header = [text(v).casefold() for v in next(rows, [])]
        if len(set(header)) != len(header):
            raise ValueError("Column names must be unique")
        if "pdf_path" not in header or not {"title", "doi"}.intersection(header):
            raise ValueError("Provide pdf_path and at least one of title or doi")
        records = []
        for number, row in enumerate(rows, 1):
            if number > max_records:
                raise ValueError(f"Import is limited to {max_records} rows")
            if not any(text(v) for v in row):
                continue
            if any(text(v) for v in row[len(header):]):
                raise ValueError(f"Row {number} has values beyond its header")
            values = dict(zip(header, map(text, row)))
            record = {key: values.get(key, "") for key in LIBRARY_COLUMNS}
            if not record["pdf_path"] or not (record["title"] or record["doi"]):
                raise ValueError(f"Row {number} needs a PDF path and title or DOI")
            records.append({"library_row": number, **record})
        return records
    except (UnicodeError, OSError, KeyError, BadZipFile, InvalidFileException) as exc:
        raise ValueError("Could not read citation list") from exc
    finally:
        if workbook is not None:
            workbook.close()


def pdf_source(root: Path, relative: str) -> Path:
    windows = PureWindowsPath(relative)
    path = Path(relative.replace("\\", "/"))
    if windows.drive or windows.root or path.is_absolute() or ".." in path.parts or ":" in relative:
        raise ValueError("PDF paths must be relative to the input folder")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()) or resolved.suffix.lower() != ".pdf":
        raise ValueError("PDF path must point to a PDF inside the input folder")
    return resolved


def normalized_title(value: str) -> str:
    return re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def index_papers(papers: list[dict]) -> dict:
    index = {"doi": defaultdict(list), "title": defaultdict(list)}
    for paper in papers:
        index["doi"][normalize_doi(paper.get("doi", "")).casefold()].append(paper)
        index["title"][normalized_title(paper.get("title", ""))].append(paper)
    return index


def match_record(record: dict, papers: list[dict], index: dict | None = None) -> dict:
    index = index if index is not None else index_papers(papers)
    doi = normalize_doi(record["doi"]).casefold()
    title = normalized_title(record["title"])
    doi_matches = index["doi"].get(doi, []) if doi else []
    title_matches = index["title"].get(title, []) if title else []
    matches = doi_matches or title_matches
    method = "doi" if doi_matches else "title" if title_matches else None
    status = "unmatched" if not matches else "ambiguous" if len(matches) > 1 else "proposed"
    if len(matches) == 1:
        candidate = matches[0]
        other_doi = normalize_doi(candidate.get("doi", "")).casefold()
        if (doi and other_doi and doi != other_doi) or (
            doi_matches and title and candidate.get("title") and title != normalized_title(candidate["title"])
        ):
            status = "metadata_conflict"
    return {"match_status": status, "match_method": method,
            "candidates": [{"row": p["row"], "title": p["title"], "doi": p.get("doi", "")}
                           for p in matches]}
