import json
import math
import threading
import os
from pathlib import Path

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from core.retry import retry_call


class Encoder:
    def __init__(
        self,
        model: str,
        revision: str | None = None,
        url: str | None = None,
        batch_size: int = 32,
        query_prefix: str = "",
        doc_prefix: str = "",
        timeout: float = 30.0,
        device: str | None = None,
        api_key_env: str = "",
        retries: int = 2,
    ):
        self._model_name = model
        self._device = device
        self._retries = retries
        self._revision = revision
        self._batch_size = batch_size
        self._query_prefix = query_prefix
        self._doc_prefix = doc_prefix
        self._local_model: SentenceTransformer | None = None
        self._lock = threading.Lock()
        key = os.environ.get(api_key_env) if api_key_env else "unused"
        if url and not key:
            raise ValueError(f"Missing embedding credential: {api_key_env}")
        self._client = OpenAI(base_url=url, api_key=key, timeout=timeout, max_retries=0) if url else None

    def _get_local_model(self) -> SentenceTransformer:
        if self._local_model is None:
            with self._lock:
                if self._local_model is None:
                    self._local_model = SentenceTransformer(
                        self._model_name, revision=self._revision, device=self._device
                    )
        return self._local_model

    def _encode(self, texts: list[str], prefix: str) -> list[list[float]]:
        prefixed = [f"{prefix}{t}" for t in texts] if prefix else texts
        if self._client is not None:
            vecs: list[list[float]] = []
            for i in range(0, len(prefixed), self._batch_size):
                batch = prefixed[i : i + self._batch_size]
                response = retry_call(
                    lambda: self._client.embeddings.create(input=batch, model=self._model_name),
                    retries=self._retries,
                )
                if sorted(item.index for item in response.data) != list(range(len(batch))):
                    raise ValueError("Embedding service returned missing or duplicate rows")
                vecs.extend(item.embedding for item in sorted(response.data, key=lambda x: x.index))
            return [_normalize(v) for v in vecs]
        return self._get_local_model().encode(
            prefixed, batch_size=self._batch_size, normalize_embeddings=True
        ).tolist()

    def encode_docs(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, self._doc_prefix)

    def encode_query(self, text: str) -> list[float]:
        return self._encode([text], self._query_prefix)[0]

    def health_check(self) -> None:
        self.encode_query("ping") if self._client is not None else self._get_local_model()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()


def _normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm > 0 else v


class Embedder:
    def __init__(self, encoder: Encoder):
        self.encoder = encoder

    def embed_chunks(self, chunks: list[dict]) -> list[list[float]]:
        if not chunks:
            return []
        return self.encoder.encode_docs(
            [f"Section: {c.get('section', 'Unknown')}\n{c['text']}" for c in chunks]
        )


class DenseSearcher:
    def __init__(self, encoder: Encoder):
        self.encoder = encoder
        self.chunks: list[dict] = []
        self.matrix: np.ndarray | None = None

    def index_chunks(self, chunks: list[dict], embeddings: np.ndarray) -> None:
        self.chunks = chunks
        self.matrix = embeddings

    def load(self, chunks_path: Path, embeddings_path: Path) -> None:
        self.chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
        self.matrix = np.load(embeddings_path)

    def search_scored(self, query: str) -> list[tuple[dict, float]]:
        if self.matrix is None or not query.strip():
            return []
        q = np.array(self.encoder.encode_query(query), dtype=np.float32)
        scores = self.matrix @ q
        order = np.argsort(scores)[::-1]
        return [(self.chunks[i], float(scores[i])) for i in order]

    def search(self, query: str, top_n: int) -> list[dict]:
        return [c for c, _ in self.search_scored(query)[:top_n]]
