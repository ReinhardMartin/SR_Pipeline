import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


class SpladeEncoder:
    def __init__(
        self,
        model: str,
        batch_size: int = 16,
        max_length: int = 256,
        revision: str | None = None,
    ):
        self._tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self._model = AutoModelForMaskedLM.from_pretrained(model, revision=revision)
        self._model.eval()
        self._batch_size = batch_size
        self._max_length = max_length

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._model.config.vocab_size), dtype=np.float32)
        vecs = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            tokens = self._tokenizer(
                batch, padding=True, truncation=True, max_length=self._max_length, return_tensors="pt"
            )
            with torch.no_grad():
                logits = self._model(**tokens).logits
            weighted = torch.log1p(torch.relu(logits)) * tokens["attention_mask"].unsqueeze(-1)
            pooled, _ = torch.max(weighted, dim=1)
            vecs.append(pooled.numpy())
        return np.concatenate(vecs, axis=0)

    def health_check(self) -> None:
        self.encode(["ping"])


class SpladeSearcher:
    def __init__(self, encoder: SpladeEncoder):
        self.encoder = encoder
        self.chunks: list[dict] = []
        self.matrix: np.ndarray | None = None

    def index_chunks(self, chunks: list[dict], storage_path: Path | None = None) -> None:
        self.chunks = chunks
        self.matrix = self.encoder.encode(
            [f"{c.get('section', '')} {c['text']}" for c in chunks]
        )
        if storage_path is not None:
            path = Path(storage_path)
            path.mkdir(parents=True, exist_ok=True)
            np.save(path / "splade_matrix.npy", self.matrix)
            (path / "corpus.json").write_text(json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8")

    def load(self, storage_path: Path) -> None:
        path = Path(storage_path)
        matrix_file, corpus_file = path / "splade_matrix.npy", path / "corpus.json"
        if not matrix_file.exists() or not corpus_file.exists():
            raise FileNotFoundError(f"SPLADE index not found: {path}")
        self.matrix = np.load(matrix_file)
        self.chunks = json.loads(corpus_file.read_text(encoding="utf-8"))

    def search(self, query: str, top_n: int) -> list[dict]:
        if self.matrix is None or not self.chunks or not query.strip():
            return []
        query_vec = self.encoder.encode([query])[0]
        scores = self.matrix @ query_vec
        k = min(top_n, len(self.chunks))
        top_idx = np.argsort(scores)[::-1][:k]
        return [self.chunks[i] for i in top_idx]
