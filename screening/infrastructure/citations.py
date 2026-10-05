import csv
import io
import re
from pathlib import Path

import openpyxl
from screening.domain.models import ScreeningRecord


TEMPLATE_COLUMNS = ["title", "abstract", "authors", "year", "doi", "journal", "url", "source_id"]
from core.settings import read_config

MAX_RECORDS = read_config()["uploads"]["max_records"]


def text(value) -> str:
    return "" if value is None else str(value).strip()


def normalize_doi(value: str) -> str:
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value.strip(), flags=re.I)


def _record(row: int, values: dict, extra: dict) -> dict:
    normalized = {key: text(values.get(key)) for key in TEMPLATE_COLUMNS}
    normalized["doi"] = normalize_doi(normalized["doi"])
    return ScreeningRecord(row=row, **normalized, extra=extra).model_dump()


def _read_rows(rows) -> tuple[list[dict], list[str]]:
    rows = iter(rows)
    header = [text(value) for value in next(rows, [])]
    lower = [value.casefold() for value in header]
    named = [value for value in lower if value]
    if len(named) != len(set(named)):
        raise ValueError("Column names must be unique (ignoring case)")
    if "title" not in lower or "abstract" not in lower:
        raise ValueError("Provide 'title' and 'abstract' columns; download the CSV template")
    papers = []
    for row_number, row in enumerate(rows, start=1):
        if row_number > MAX_RECORDS:
            raise ValueError(f"Import is limited to {MAX_RECORDS:,} data rows")
        if any(text(value) for value in row[len(header):]):
            raise ValueError(f"Data row {row_number} has more values than the header")
        if any(not key and i < len(row) and text(row[i]) for i, key in enumerate(lower)):
            raise ValueError(f"Data row {row_number} has a value in an unnamed column")
        values = {key: text(row[i]) if i < len(row) else "" for i, key in enumerate(lower)}
        if not values["title"] and not values["abstract"]:
            continue
        extra = {
            name: text(row[i]) if i < len(row) else ""
            for i, name in enumerate(header) if name and lower[i] not in {"title", "abstract"}
        }
        papers.append(_record(row_number, values, extra))
    return papers, header


def _read_ris(data: str) -> tuple[list[dict], list[str]]:
    papers = []
    tags: dict[str, list[str]] | None = None
    last_tag = None
    tag_pattern = re.compile(r"^([A-Z0-9]{2})  -(?: ?)(.*)$")

    def first(*keys):
        return next((value for key in keys for value in tags.get(key, []) if value.strip()), "")

    for line_number, line in enumerate(data.splitlines(), start=1):
        if not line.strip():
            continue
        match = tag_pattern.match(line)
        if not match:
            if tags is None or last_tag is None:
                raise ValueError(f"RIS line {line_number}: expected a 'TY  -' record")
            tags[last_tag][-1] += "\n" + line.strip()
            continue
        tag, value = match.groups()
        if tag == "TY":
            if tags is not None:
                raise ValueError(f"RIS line {line_number}: previous record has no 'ER  -' terminator")
            tags = {tag: [value.strip()]}
        elif tags is None:
            raise ValueError(f"RIS line {line_number}: record must start with 'TY  -'")
        elif tag == "ER":
            if len(papers) >= MAX_RECORDS:
                raise ValueError(f"Import is limited to {MAX_RECORDS:,} records")
            values = dict(
                title=first("TI", "T1", "CT"), abstract=first("AB", "N2"),
                authors="; ".join(tags.get("AU") or tags.get("A1") or []),
                year=first("PY", "Y1", "DA").split("/")[0], doi=first("DO"),
                journal=first("JF", "JO", "T2", "JA"), url=first("UR"), source_id=first("ID", "AN"),
            )
            if not values["title"] and not values["abstract"]:
                raise ValueError(f"RIS record {len(papers) + 1} has neither title nor abstract")
            papers.append(_record(len(papers) + 1, values, {key: "\n".join(items) for key, items in tags.items()}))
            tags = None
        else:
            tags.setdefault(tag, []).append(value.strip())
        last_tag = tag if tags is not None else None
    if tags is not None:
        raise ValueError("Final RIS record has no 'ER  -' terminator")
    return papers, TEMPLATE_COLUMNS.copy()


def read_citations(data: bytes, filename: str) -> tuple[list[dict], list[str]]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".xlsx":
        try:
            workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        except Exception as exc:
            raise ValueError(f"Could not read workbook: {exc}") from exc
        try:
            if workbook.active is None:
                raise ValueError("Workbook has no active worksheet")
            return _read_rows(workbook.active.iter_rows(values_only=True))
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Could not read worksheet: {exc}") from exc
        finally:
            workbook.close()
    if suffix not in {".csv", ".ris"}:
        raise ValueError("Upload a .ris, .csv, or .xlsx file")
    try:
        decoded = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV and RIS files must use UTF-8 encoding; re-export the file as UTF-8") from exc
    if suffix == ".ris":
        return _read_ris(decoded)
    try:
        return _read_rows(csv.reader(io.StringIO(decoded, newline=""), strict=True))
    except csv.Error as exc:
        raise ValueError(f"Malformed CSV: {exc}") from exc


def csv_bytes(columns: list[str], rows) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(columns)
    for row in rows:
        writer.writerow([spreadsheet_value(value) for value in row])
    return buffer.getvalue().encode("utf-8-sig")


def spreadsheet_value(value):
    """Preserve untrusted text instead of letting spreadsheets evaluate it."""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value
