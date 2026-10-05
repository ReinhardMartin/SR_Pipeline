import asyncio
import json

import numpy as np
from core.encoders import DenseSearcher
from core.keyword_search import BM25Searcher, StemKeywordSearcher
from core.splade import SpladeSearcher
from fastapi import APIRouter, Depends, HTTPException
from schemas import (
    EncodeRequest,
    EncodeResult,
    NliRequest,
    NliResult,
    RerankedDocument,
    RerankRequest,
    SearchRequest,
)
from services.runtime import Services

from routes.dependencies import get_services

router = APIRouter()


@router.post("/tools/encode", response_model=EncodeResult)
async def tools_encode(
    body: EncodeRequest, *, services: Services = Depends(get_services)
):

    def _run():
        encoder = services.models.instances["encoder"]
        vec = (
            encoder.encode_query(body.text)
            if body.is_query
            else encoder.encode_docs([body.text])[0]
        )
        return EncodeResult(dim=len(vec), norm=float(np.linalg.norm(vec)), vector=vec)

    return await asyncio.to_thread(_run)


@router.post("/tools/dense-search")
async def tools_dense_search(
    body: SearchRequest, *, services: Services = Depends(get_services)
):
    doc_dir = services.artifacts.require_directory(body.stem)
    chunks_path, embeddings_path = (
        doc_dir / f"{body.stem}.chunks.json",
        doc_dir / f"{body.stem}.embeddings.npy",
    )
    if not (chunks_path.exists() and embeddings_path.exists()):
        raise HTTPException(400, "Paper not indexed yet")

    def _run():
        searcher = DenseSearcher(encoder=services.models.instances["encoder"])
        searcher.load(chunks_path, embeddings_path)
        return searcher.search(body.query, body.top_n)

    return await asyncio.to_thread(_run)


@router.post("/tools/bm25-search")
async def tools_bm25_search(
    body: SearchRequest, *, services: Services = Depends(get_services)
):
    doc_dir = services.artifacts.require_directory(body.stem)
    chunks_path = doc_dir / f"{body.stem}.chunks.json"
    if not chunks_path.exists():
        raise HTTPException(400, "Paper not indexed yet")
    language = services.configuration.load_config()["retrieval"].get(
        "language", "english"
    )

    def _run():
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        searcher = BM25Searcher(language=language)
        searcher.index_chunks(chunks)
        return searcher.search(body.query, body.top_n)

    return await asyncio.to_thread(_run)


@router.post("/tools/stem-search")
async def tools_stem_search(
    body: SearchRequest, *, services: Services = Depends(get_services)
):
    doc_dir = services.artifacts.require_directory(body.stem)
    keyword_index_path = doc_dir / "keyword_index"
    if not keyword_index_path.exists():
        raise HTTPException(400, "Paper not indexed yet")
    language = services.configuration.load_config()["retrieval"].get(
        "language", "english"
    )

    def _run():
        searcher = StemKeywordSearcher(language=language)
        searcher.load(keyword_index_path)
        return searcher.search(body.query, body.top_n)

    return await asyncio.to_thread(_run)


@router.post("/tools/splade-search")
async def tools_splade_search(
    body: SearchRequest, *, services: Services = Depends(get_services)
):
    doc_dir = services.artifacts.require_directory(body.stem)
    chunks_path = doc_dir / f"{body.stem}.chunks.json"
    if not chunks_path.exists():
        raise HTTPException(400, "Paper not indexed yet")

    def _run():
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        searcher = SpladeSearcher(encoder=services.models.get_splade_encoder())
        searcher.index_chunks(chunks)
        return searcher.search(body.query, body.top_n)

    return await asyncio.to_thread(_run)


@router.post("/tools/rerank", response_model=list[RerankedDocument])
async def tools_rerank(
    body: RerankRequest, *, services: Services = Depends(get_services)
):

    def _run():
        scores = services.models.get_reranker().predict(
            [(body.query, doc) for doc in body.documents]
        )
        ranked = sorted(zip(body.documents, scores), key=lambda x: x[1], reverse=True)
        return [RerankedDocument(document=doc, score=score) for doc, score in ranked]

    return await asyncio.to_thread(_run)


@router.post("/tools/nli", response_model=list[NliResult])
async def tools_nli(body: NliRequest, *, services: Services = Depends(get_services)):
    hypotheses = body.hypotheses or ([body.hypothesis] if body.hypothesis else [])
    if not hypotheses:
        raise HTTPException(400, "'hypothesis' or 'hypotheses' is required")

    def _run():
        scores = services.models.get_nli().entails_batch(
            [(body.premise, h) for h in hypotheses]
        )
        return [
            NliResult(hypothesis=h, entailment=score)
            for h, score in zip(hypotheses, scores)
        ]

    try:
        return await asyncio.to_thread(_run)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
