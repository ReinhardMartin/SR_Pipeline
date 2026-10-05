import json
import asyncio
import logging
import os
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path

import httpx
from mineru.cli import api_client as _api_client
from mineru.cli.common import image_suffixes, pdf_suffixes
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_path

from core.settings import read_config, resolve_path, ParseConfig
from core.provenance import build_manifest, manifest_current, write_manifest

logger = logging.getLogger(__name__)

_CONFIG_PATH = resolve_path(read_config()["paths"]["parser"])
_MINERU_CFG  = resolve_path(read_config()["paths"]["mineru"])
_SUPPORTED_SUFFIXES = set(pdf_suffixes + image_suffixes)


def _setup_mineru_env() -> None:
    os.environ.setdefault("MINERU_TOOLS_CONFIG_JSON", str(_MINERU_CFG))


def _load_config(config_path: Path) -> dict:
    p = Path(config_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, encoding="utf-8") as f:
        return ParseConfig.model_validate(json.load(f)).model_dump()


def _collect_input_files(input_path: Path) -> list[Path]:
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if path.is_file():
        suffix = guess_suffix_by_path(path)
        if suffix not in _SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {path.name}")
        return [path]
    if not path.is_dir():
        raise ValueError(f"Input path must be a file or directory: {path}")
    files = sorted(
        (f.resolve() for f in path.iterdir()
         if f.is_file() and guess_suffix_by_path(f) in _SUPPORTED_SUFFIXES),
        key=lambda f: f.name,
    )
    if not files:
        raise ValueError(f"No supported files found in: {path}")
    return files


def _fix_wsl_tempdir() -> None:
    if os.name == "nt":
        return
    if not str(Path(tempfile.gettempdir())).startswith("/mnt/"):
        return
    os.environ["TMPDIR"] = "/tmp"
    tempfile.tempdir = None


def parsed_directory(stem: str, output_dir: Path) -> Path:
    return output_dir / stem / _load_config(_CONFIG_PATH)["parse_method"]


def parse_manifest(pdf: Path, config_path: Path = _CONFIG_PATH) -> dict:
    tools_config = Path(os.environ.get("MINERU_TOOLS_CONFIG_JSON", str(_MINERU_CFG))).expanduser()
    return build_manifest("parse", {
        "pdf": pdf, "parser_code": Path(__file__), "tools_config": tools_config,
    }, {"parameters": _load_config(config_path), "mineru_version": version("mineru")})


def parse_current(pdf: Path, doc_dir: Path, config_path: Path = _CONFIG_PATH) -> bool:
    markdown = doc_dir / f"{pdf.stem}.md"
    try:
        return bool(markdown.read_text(encoding="utf-8").strip()) and manifest_current(
            doc_dir / f"{pdf.stem}.parse.manifest.json", parse_manifest(pdf, config_path), {"markdown": markdown},
        )
    except (OSError, ValueError):
        return False


async def parse(
    input_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path = _CONFIG_PATH,
) -> None:
    await parse_files(_collect_input_files(Path(input_path)), output_dir, config_path)


async def parse_files(
    files: list[str | Path],
    output_dir: str | Path,
    config_path: str | Path = _CONFIG_PATH,
) -> None:
    """Submit an explicit set of documents as one MinerU parsing task."""
    _setup_mineru_env()
    cfg = _load_config(Path(config_path))

    backend    = cfg["backend"]
    api_url    = cfg.get("api_url") or None
    server_url = cfg.get("server_url") or None
    effort     = cfg.get("effort") or None

    if backend.endswith("http-client") and not server_url:
        raise ValueError(
            f"backend='{backend}' requires 'server_url' to be set in parse_parameters.json"
        )

    input_files = [Path(item).expanduser().resolve() for item in files]
    if not input_files:
        raise ValueError("No input files were supplied")
    if len(input_files) != len(set(input_files)):
        raise ValueError("Duplicate input files were supplied")
    if len({path.stem.casefold() for path in input_files}) != len(input_files):
        raise ValueError("Input filenames must have unique stems")
    for path in input_files:
        if not path.is_file():
            raise FileNotFoundError(f"Input file does not exist: {path}")
        if guess_suffix_by_path(path) not in _SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {path.name}")
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    parse_method = cfg["parse_method"]
    input_files = [f for f in input_files if not parse_current(f, output_path / f.stem / parse_method, config_path)]
    if not input_files:
        logger.info("All files already parsed, skipping.")
        return

    manifests = {path: parse_manifest(path, config_path) for path in input_files}
    for path in input_files:
        (output_path / path.stem / parse_method / f"{path.stem}.parse.manifest.json").unlink(missing_ok=True)

    logger.info("Parsing %d file(s) with backend=%s", len(input_files), backend)

    form_data = _api_client.build_parse_request_form_data(
        lang_list=[cfg["language"]],
        backend=backend,
        parse_method=parse_method,
        formula_enable=cfg["formula_enable"],
        table_enable=cfg["table_enable"],
        image_analysis=cfg["image_analysis"],
        server_url=server_url,
        start_page_id=cfg["start_page_id"],
        end_page_id=cfg.get("end_page_id"),
        return_md=cfg["return_md"],
        return_middle_json=cfg["return_middle_json"],
        return_model_output=cfg["return_model_output"],
        return_content_list=cfg["return_content_list"],
        return_images=cfg["return_images"],
        response_format_zip=cfg["response_format_zip"],
        return_original_file=cfg["return_original_file"],
    )
    
    if effort is not None:
        form_data["effort"] = effort

    local_server: _api_client.LocalAPIServer | None = None

    async with httpx.AsyncClient(
        timeout=cfg["http_timeout"],
        follow_redirects=True,
    ) as http:
        try:
            if api_url is None:
                _fix_wsl_tempdir()
                local_server = _api_client.LocalAPIServer()
                local_server.start()
                health = await _api_client.wait_for_local_api_ready(http, local_server)
            else:
                health = await _api_client.fetch_server_health(
                    http, _api_client.normalize_base_url(api_url)
                )

            logger.info("API ready: %s", health.base_url)

            submit = await _api_client.submit_parse_task(
                base_url=health.base_url,
                upload_assets=[
                    _api_client.UploadAsset(path=f, upload_name=f.name)
                    for f in input_files
                ],
                form_data=form_data,
            )

            last_status: str | None = None

            def _on_status(snap: _api_client.TaskStatusSnapshot) -> None:
                nonlocal last_status
                msg = snap.status
                if snap.queued_ahead is not None:
                    msg += f" (queued_ahead={snap.queued_ahead})"
                if msg != last_status:
                    last_status = msg
                    logger.info("Task status: %s", msg)

            await asyncio.wait_for(_api_client.wait_for_task_result(
                client=http,
                submit_response=submit,
                task_label=f"{len(input_files)} file(s)",
                status_snapshot_callback=_on_status,
            ), timeout=cfg["task_timeout"])

            result_zip = await _api_client.download_result_zip(
                client=http,
                submit_response=submit,
                task_label=f"{len(input_files)} file(s)",
            )

        finally:
            if local_server is not None:
                local_server.stop()

    try:
        with tempfile.TemporaryDirectory(dir=output_path, prefix=".parse-") as directory:
            staging = Path(directory)
            _api_client.safe_extract_zip(result_zip, staging)
            failed = []
            for pdf in input_files:
                source = staging / pdf.stem / parse_method
                markdown = source / f"{pdf.stem}.md"
                if not markdown.is_file() or not markdown.read_text(encoding="utf-8").strip():
                    failed.append(pdf.name)
                    continue
                if parse_manifest(pdf, config_path)["fingerprint"] != manifests[pdf]["fingerprint"]:
                    failed.append(pdf.name)
                    continue
                destination = output_path / pdf.stem / parse_method
                shutil.copytree(source, destination, dirs_exist_ok=True)
                write_manifest(destination / f"{pdf.stem}.parse.manifest.json", manifests[pdf], {
                    "markdown": destination / f"{pdf.stem}.md",
                })
            if failed:
                raise RuntimeError(f"Missing, empty, or changed parse inputs: {', '.join(failed)}")
        logger.info("Extracted %d file(s) to: %s", len(input_files), output_path)
    finally:
        result_zip.unlink(missing_ok=True)
