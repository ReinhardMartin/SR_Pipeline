import json
import re
from collections import defaultdict
from pathlib import Path

import bm25s
import bm25s.stopwords as _bm25s_stopwords
import Stemmer

_WORD_RE = re.compile(r"[a-z0-9]+")


def _stopwords_for(language: str):
    if language == "english":
        return _bm25s_stopwords.STOPWORDS_EN
    return getattr(_bm25s_stopwords, f"STOPWORDS_{language.upper()}", _bm25s_stopwords.STOPWORDS_EN)


class BM25Searcher:
    def __init__(self, language: str = "english"):
        self.retriever = bm25s.BM25()
        self.stemmer = Stemmer.Stemmer(language)
        self.language = language
        self.chunks: list[dict] = []
        self._is_indexed = False

    def index_chunks(self, chunks: list[dict], storage_path: Path | None = None) -> None:
        if not chunks:
            return
        self.chunks = chunks
        corpus_tokens = bm25s.tokenize(
            [f"{c.get('section', '')} {c['text']}" for c in chunks],
            stopwords=self.language,
            stemmer=self.stemmer,
            show_progress=False,
        )
        self.retriever.index(corpus_tokens, show_progress=False)
        self._is_indexed = True
        if storage_path is not None:
            Path(storage_path).mkdir(parents=True, exist_ok=True)
            self.retriever.save(str(storage_path), corpus=self.chunks)

    def load(self, storage_path: Path) -> None:
        path = Path(storage_path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 index not found: {path}")
        self.retriever = bm25s.BM25.load(str(path), load_corpus=True)
        self.chunks = self.retriever.corpus
        self._is_indexed = True

    def search(self, query: str, top_n: int) -> list[dict]:
        return [document for document, _ in self.search_scored(query, top_n)]

    def search_scored(self, query: str, top_n: int) -> list[tuple[dict, float]]:
        if not self._is_indexed:
            raise RuntimeError("BM25Searcher not indexed — call index_chunks() or load() first")
        if not query.strip():
            return []
        k = min(top_n, len(self.chunks))
        if k <= 0:
            return []
        query_tokens = bm25s.tokenize(
            [query], stopwords=self.language, stemmer=self.stemmer, show_progress=False
        )
        results = self.retriever.retrieve(
            query_tokens, corpus=self.chunks, k=k, show_progress=False
        )
        return [
            (doc, float(score))
            for doc, score in zip(results.documents[0], results.scores[0])
            if score > 0
        ]


class StemKeywordSearcher:
    def __init__(self, language: str = "english"):
        self.stemmer = Stemmer.Stemmer(language)
        self.language = language
        self._stopwords = _stopwords_for(language)
        self.chunks: list[dict] = []
        self._by_id: dict[str, dict] = {}
        self._order: dict[str, int] = {}
        self._index: dict[str, set[str]] = {}
        self._is_indexed = False

    def _tokenize(self, text: str) -> set[str]:
        words = [w for w in _WORD_RE.findall(text.lower()) if w not in self._stopwords]
        return set(self.stemmer.stemWords(words)) if words else set()

    def index_chunks(self, chunks: list[dict], storage_path: Path | None = None) -> None:
        if not chunks:
            return
        self.chunks = chunks
        self._by_id = {c["id"]: c for c in chunks}
        self._order = {c["id"]: i for i, c in enumerate(chunks)}

        index: dict[str, set[str]] = defaultdict(set)
        for c in chunks:
            for stem in self._tokenize(f"{c.get('section', '')} {c['text']}"):
                index[stem].add(c["id"])
        self._index = dict(index)
        self._is_indexed = True

        if storage_path is not None:
            self._save(Path(storage_path))

    def _save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "inverted_index.json").write_text(
            json.dumps({stem: sorted(ids) for stem, ids in self._index.items()}, ensure_ascii=False),
            encoding="utf-8",
        )
        (path / "corpus.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8"
        )

    def load(self, storage_path: Path) -> None:
        path = Path(storage_path)
        index_file, corpus_file = path / "inverted_index.json", path / "corpus.json"
        if not index_file.exists() or not corpus_file.exists():
            raise FileNotFoundError(f"Stem keyword index not found: {path}")
        raw_index = json.loads(index_file.read_text(encoding="utf-8"))
        self._index = {stem: set(ids) for stem, ids in raw_index.items()}
        self.chunks = json.loads(corpus_file.read_text(encoding="utf-8"))
        self._by_id = {c["id"]: c for c in self.chunks}
        self._order = {c["id"]: i for i, c in enumerate(self.chunks)}
        self._is_indexed = True

    def search(self, query: str, top_n: int) -> list[dict]:
        if not self._is_indexed:
            raise RuntimeError("StemKeywordSearcher not indexed — call index_chunks() or load() first")
        if not query.strip():
            return []
        matched_ids: set[str] = set()
        for stem in self._tokenize(query):
            matched_ids.update(self._index.get(stem, ()))
        ordered = sorted(matched_ids, key=lambda i: self._order[i])
        return [self._by_id[i] for i in ordered[:top_n]]
