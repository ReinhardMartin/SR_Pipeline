import httpx
import math
import os
from sentence_transformers import CrossEncoder
from core.retry import retry_call


class Reranker:
    def __init__(self, model: str, url: str | None = None, timeout: float = 30.0, revision: str | None = None,
                 device: str | None = None, retries: int = 2, api_key_env: str = ""):
        self._model_name = model
        self._url = url
        self._timeout = timeout
        self._retries = retries
        key = os.environ.get(api_key_env) if api_key_env else None
        if url and api_key_env and not key:
            raise ValueError(f"Missing reranker credential: {api_key_env}")
        self._headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._local_model = CrossEncoder(model, revision=revision, device=device) if url is None else None

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self._local_model is not None:
            return self._local_model.predict(pairs).tolist()
        return self._predict_remote(pairs)

    def _predict_remote(self, pairs: list[tuple[str, str]]) -> list[float]:
        groups: dict[str, list[tuple[int, str]]] = {}
        for i, (query, doc) in enumerate(pairs):
            groups.setdefault(query, []).append((i, doc))

        scores: list[float] = [0.0] * len(pairs)
        with httpx.Client(base_url=self._url, timeout=self._timeout, headers=self._headers) as client:
            for query, items in groups.items():
                docs = [doc for _, doc in items]
                def send():
                    response = client.post("rerank", json={"model": self._model_name, "query": query, "documents": docs})
                    response.raise_for_status()
                    return response
                resp = retry_call(send, retries=self._retries)
                by_doc_position = {r["index"]: r["relevance_score"] for r in resp.json()["results"]}
                if set(by_doc_position) != set(range(len(docs))) or not all(
                    isinstance(score, (int, float)) and math.isfinite(score) for score in by_doc_position.values()
                ):
                    raise ValueError("Reranker returned incomplete or invalid scores")
                for position, (i, _) in enumerate(items):
                    scores[i] = by_doc_position.get(position, 0.0)
        return scores

    def health_check(self) -> None:
        self.predict([("ping", "pong")])
