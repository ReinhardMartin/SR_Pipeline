import json
from collections import defaultdict
from pathlib import Path

import numpy as np

_RECALL_EXCLUDED_KINDS = {"formula", "code"}

_EMPTY_RESULT = {"dense": [], "sparse": [], "intersection": [], "kept": []}


class Retriever:
    def __init__(
        self,
        dense_searcher,
        keyword_searcher,
        candidates_per_query: int,
        max_fused_candidates: int = 100,
        rrf_k: int = 60,
    ):
        self._dense = dense_searcher
        self._keyword = keyword_searcher
        self._candidates_per_query = candidates_per_query
        self._max_fused_candidates = max_fused_candidates
        self._rrf_k = rrf_k
        self._chunks: list[dict] = []
        self._idx: dict[str, int] = {}
        self._by_section: dict[str, list[dict]] = {}
        self._section_order: dict[str, int] = {}

    def load(self, chunks_path: str | Path, embeddings_path: str | Path, keyword_index_path: str | Path) -> None:
        self._chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
        self._idx = {c["id"]: i for i, c in enumerate(self._chunks)}

        by_section: dict[str, list[int]] = defaultdict(list)
        for i, chunk in enumerate(self._chunks):
            by_section[chunk["section"]].append(i)
        self._by_section = {
            section: [self._chunks[i] for i in indices]
            for section, indices in by_section.items()
        }
        self._section_order = {s: i for i, s in enumerate(self._by_section)}

        self._dense.index_chunks(self._chunks, np.load(embeddings_path))
        self._keyword.load(str(keyword_index_path))

    @property
    def sections(self) -> list[str]:
        return list(self._by_section.keys())

    def retrieve(self, queries: list[str], keywords: list[str] | None = None) -> list[dict]:
        return self.retrieve_debug(queries, keywords)["kept"]

    def candidates(self, queries: list[str], keywords: list[str] | None = None, prefilter: bool = False) -> list[dict]:
        if not prefilter:
            return [c for c in self._chunks if c["kind"] not in _RECALL_EXCLUDED_KINDS]
        return self.retrieve_debug(queries, keywords)["kept"]

    def rerank(self, queries: list[str], chunks: list[dict], reranker) -> list[tuple[dict, float]]:
        if not queries or not chunks:
            return []
        pairs = [(q, f"Section: {c.get('section', 'Unknown')}\n{c['text']}") for q in queries for c in chunks]
        scores = reranker.predict(pairs)
        n = len(chunks)
        best = [-float("inf")] * n
        for qi in range(len(queries)):
            offset = qi * n
            for ci in range(n):
                s = scores[offset + ci]
                if s > best[ci]:
                    best[ci] = s
        return sorted(zip(chunks, best), key=lambda x: x[1], reverse=True)

    def select_for_extraction(
        self,
        queries: list[str],
        chunks: list[dict],
        reranker,
        top_k_per_query: int = 2,
        max_chunks: int = 12,
        fusion_reserve: int = 0,
    ) -> list[dict]:
        """Select evidence while guaranteeing coverage for every atomic query.

        The previous global top-k could devote every evidence slot to one query.
        Here each query contributes its strongest passages before a global cap is
        applied. Selection metadata is retained for audit and validation.
        """
        if not queries or not chunks:
            return []
        if max_chunks < len(queries):
            raise ValueError(
                f"max_chunks ({max_chunks}) must be at least the number of queries "
                f"({len(queries)}) to guarantee query coverage"
            )

        pairs = [
            (query, f"Section: {chunk.get('section', 'Unknown')}\n{chunk['text']}")
            for query in queries
            for chunk in chunks
        ]
        scores = reranker.predict(pairs)
        n_chunks = len(chunks)
        selected: dict[str, dict] = {}
        required_ids: list[str] = []

        for query_index, query in enumerate(queries):
            offset = query_index * n_chunks
            ranked_indices = sorted(
                range(n_chunks),
                key=lambda i: scores[offset + i],
                reverse=True,
            )[:top_k_per_query]
            if ranked_indices:
                required_id = chunks[ranked_indices[0]]["id"]
                if required_id not in required_ids:
                    required_ids.append(required_id)
            for chunk_index in ranked_indices:
                chunk = chunks[chunk_index]
                score = float(scores[offset + chunk_index])
                entry = selected.setdefault(
                    chunk["id"],
                    {"chunk": chunk, "score": score, "matched_queries": []},
                )
                entry["score"] = max(entry["score"], score)
                entry["matched_queries"].append(query)

        limited = [selected[chunk_id] for chunk_id in required_ids]
        limited_ids = set(required_ids)
        for chunk in chunks[:fusion_reserve]:
            if len(limited) >= max_chunks:
                break
            if chunk["id"] not in limited_ids:
                limited.append(
                    selected.get(
                        chunk["id"],
                        {"chunk": chunk, "score": 0.0, "matched_queries": ["fusion_reserve"]},
                    )
                )
                limited_ids.add(chunk["id"])
        for entry in sorted(selected.values(), key=lambda e: e["score"], reverse=True):
            if len(limited) >= max_chunks:
                break
            if entry["chunk"]["id"] not in limited_ids:
                limited.append(entry)
                limited_ids.add(entry["chunk"]["id"])
        annotated = []
        for entry in limited:
            chunk = dict(entry["chunk"])
            chunk["rerank_score"] = entry["score"]
            chunk["selected_for_queries"] = entry["matched_queries"]
            annotated.append(self._with_paragraph_context(chunk))
        return self._sort_by_section(annotated)

    def _with_paragraph_context(self, chunk: dict) -> dict:
        pid = chunk.get("paragraph_id")
        if not pid:
            return chunk
        siblings = sorted(
            (c for c in self._chunks if c.get("paragraph_id") == pid),
            key=lambda c: self._idx[c["id"]],
        )
        if len(siblings) <= 1:
            return chunk
        expanded = dict(chunk)
        expanded["text"] = " ".join(s["text"] for s in siblings)
        return expanded

    def retrieve_debug(self, queries: list[str], keywords: list[str] | None = None) -> dict:
        if not self._chunks or not queries:
            return dict(_EMPTY_RESULT)

        dense_hits, sparse_hits, fusion_scores = self._recall(queries, keywords or [])
        union_ids = set(dense_hits) | set(sparse_hits)
        if not union_ids:
            return dict(_EMPTY_RESULT)
        intersection_ids = set(dense_hits) & set(sparse_hits)

        def annotate(cid: str) -> dict:
            chunk = dict(self._chunks[self._idx[cid]])
            chunk["matched_by"] = {"dense": dense_hits.get(cid, []), "sparse": sparse_hits.get(cid, [])}
            chunk["fusion_score"] = fusion_scores.get(cid, 0.0)
            return chunk

        ranked_union_ids = sorted(union_ids, key=lambda cid: fusion_scores.get(cid, 0.0), reverse=True)
        kept_ids = ranked_union_ids[: self._max_fused_candidates]

        return {
            "dense":        self._sort_by_section([annotate(cid) for cid in dense_hits]),
            "sparse":       self._sort_by_section([annotate(cid) for cid in sparse_hits]),
            "intersection": self._sort_by_section([annotate(cid) for cid in intersection_ids]),
            "kept":         [annotate(cid) for cid in kept_ids],
        }

    def _recall(
        self, queries: list[str], keywords: list[str]
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, float]]:
        dense_hits: dict[str, list[str]] = {}
        fusion_scores: dict[str, float] = defaultdict(float)
        for query in queries:
            top_candidates = self._dense.search_scored(query)[: self._candidates_per_query]
            for rank, (chunk, _score) in enumerate(top_candidates, start=1):
                if chunk["kind"] not in _RECALL_EXCLUDED_KINDS:
                    dense_hits.setdefault(chunk["id"], []).append(query)
                    fusion_scores[chunk["id"]] += 1.0 / (self._rrf_k + rank)

        sparse_hits: dict[str, list[str]] = {}
        for kw in keywords:
            if hasattr(self._keyword, "search_scored"):
                sparse_results = self._keyword.search_scored(kw, self._candidates_per_query)
                sparse_chunks = [chunk for chunk, _ in sparse_results]
            else:
                sparse_chunks = self._keyword.search(kw, self._candidates_per_query)
            for rank, chunk in enumerate(sparse_chunks, start=1):
                if chunk["kind"] not in _RECALL_EXCLUDED_KINDS:
                    sparse_hits.setdefault(chunk["id"], []).append(kw)
                    fusion_scores[chunk["id"]] += 1.0 / (self._rrf_k + rank)

        return dense_hits, sparse_hits, dict(fusion_scores)

    def _sort_by_section(self, chunks: list[dict]) -> list[dict]:
        return sorted(chunks, key=lambda c: (self._section_order.get(c["section"], 9999), self._idx[c["id"]]))
