import threading

from core.chunking import Chunker
from core.encoders import DenseSearcher, Embedder, Encoder
from core.keyword_search import BM25Searcher, StemKeywordSearcher
from core.nli import NLIClassifier
from core.provenance import artifact_hash, canonical_hash, file_hash
from core.reranker import Reranker
from core.retriever import Retriever
from core.settings import resolve_path
from core.splade import SpladeEncoder
from llm.client import ChatClient
from llm.extractor import Extractor
from llm.full_extractor import FullExtractor
from llm.planner import Planner

from services.configuration import Configuration


class Models:
    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.instances = {}
        self.current_fingerprint = None
        self.reranker_lock = threading.Lock()
        self.nli_lock = threading.Lock()
        self.splade_lock = threading.Lock()

    @staticmethod
    def _startup_prompt_fingerprint(cfg: dict) -> str:
        prompt_root = resolve_path(cfg["paths"]["prompts"])
        if not prompt_root.is_dir():
            return artifact_hash(prompt_root)
        screening = cfg["screening"]
        runtime_loaded = {
            resolve_path(screening["llm"]["instructions"]),
            resolve_path(screening["llm"]["criterion_instructions"]),
            resolve_path(screening["panel"]["judge_instructions"]),
        }
        for role in ("second_reviewer", "judge"):
            llm = screening["panel"][role].get("llm")
            if llm:
                runtime_loaded.add(resolve_path(llm["instructions"]))
                runtime_loaded.add(resolve_path(llm["criterion_instructions"]))
        files = sorted(path for path in prompt_root.rglob("*") if path.is_file())
        return canonical_hash({
            path.relative_to(prompt_root).as_posix(): file_hash(path)
            for path in files
            if path.resolve() not in runtime_loaded
        })

    def fingerprint(self, cfg: dict) -> str:
        return canonical_hash(
            {
                "prompts": self._startup_prompt_fingerprint(cfg),
                "deployment": {key: cfg[key] for key in ("paths", "uploads", "server")},
                "screening_engine": {key: cfg["screening"].get(key) for key in ("backend", "llm")},
                "components": {
                    key: cfg.get(key)
                    for key in (
                        "llm",
                        "chunking",
                        "embedding",
                        "reranker",
                        "nli",
                        "splade",
                        "full_extraction",
                        "execution",
                    )
                },
            }
        )

    def require_current(self) -> None:
        if self.current_fingerprint != self.fingerprint(
            self.configuration.load_config()
        ):
            raise RuntimeError(
                "Runtime model/chunking configuration changed; restart the API before running this stage"
            )

    def call_llm(self, system: str, user: str, max_tokens: int) -> str:
        return self.instances["llm"].complete(system, user, max_tokens)

    def llm_fn(self, system: str, user: str) -> str:
        return self.call_llm(
            system, user, self.configuration.load_config()["llm"]["max_tokens"]
        )

    def llm_fn_full(self, system: str, user: str) -> str:
        return self.call_llm(
            system,
            user,
            self.configuration.load_config()["full_extraction"]["max_tokens"],
        )

    def make_keyword_searcher(self, cfg: dict):
        language = cfg.get("language", "english")
        if cfg.get("keyword_searcher", "stem") == "bm25":
            return BM25Searcher(language=language)
        return StemKeywordSearcher(language=language)

    def make_retriever(self) -> Retriever:
        cfg = self.configuration.load_config()["retrieval"]
        return Retriever(
            dense_searcher=DenseSearcher(encoder=self.instances["encoder"]),
            keyword_searcher=self.make_keyword_searcher(cfg),
            candidates_per_query=cfg["candidates_per_query"],
            max_fused_candidates=cfg.get("max_fused_candidates", 100),
            rrf_k=cfg.get("rrf_k", 60),
        )

    def get_reranker(self) -> Reranker:
        if "reranker" not in self.instances:
            with self.reranker_lock:
                if "reranker" not in self.instances:
                    cfg = self.configuration.load_config()["reranker"]
                    self.instances["reranker"] = Reranker(
                        model=cfg["model"],
                        url=cfg.get("service_url") or None,
                        timeout=cfg["timeout"],
                        revision=cfg["revision"],
                        device=cfg["device"],
                        retries=cfg["retries"],
                        api_key_env=cfg["api_key_env"],
                    )
        return self.instances["reranker"]

    def get_nli(self) -> NLIClassifier:
        if "nli" not in self.instances:
            with self.nli_lock:
                if "nli" not in self.instances:
                    cfg = self.configuration.load_config()["nli"]
                    self.instances["nli"] = NLIClassifier(
                        model=cfg["model"],
                        batch_size=cfg.get("batch_size", 16),
                        revision=cfg.get("revision"),
                        device=cfg["device"],
                    )
        return self.instances["nli"]

    def get_splade_encoder(self) -> SpladeEncoder:
        if "splade_encoder" not in self.instances:
            with self.splade_lock:
                if "splade_encoder" not in self.instances:
                    cfg = self.configuration.load_config()["splade"]
                    self.instances["splade_encoder"] = SpladeEncoder(
                        model=cfg["model"],
                        batch_size=cfg.get("batch_size", 16),
                        max_length=cfg.get("max_length", 256),
                        revision=cfg.get("revision"),
                    )
        return self.instances["splade_encoder"]

    def start(self, cfg: dict):
        emb = cfg["embedding"]
        encoder = Encoder(
            model=emb["model"],
            revision=emb["revision"],
            url=emb["service_url"] or None,
            batch_size=emb["batch_size"],
            query_prefix=emb["query_prefix"],
            doc_prefix=emb["doc_prefix"],
            timeout=emb["timeout"],
            device=emb["device"],
            retries=emb["retries"],
            api_key_env=emb["api_key_env"],
        )
        self.instances["encoder"] = encoder
        if cfg["execution"]["warm_models"]:
            encoder.health_check()
        self.instances["llm"] = ChatClient(cfg["llm"])
        self.instances.update(
            chunker=Chunker(**cfg["chunking"]),
            embedder=Embedder(encoder),
            planner=Planner(
                self.llm_fn,
                retries=cfg["llm"]["retries"],
                batch_size=cfg["llm"]["planner_batch_size"],
            ),
            extractor=Extractor(self.llm_fn, max_workers=cfg["llm"]["max_workers"]),
            full_extractor=FullExtractor(
                self.llm_fn_full,
                max_context_tokens=cfg["full_extraction"]["max_context_tokens"],
                max_output_tokens=cfg["full_extraction"]["max_tokens"],
            ),
        )
        self.current_fingerprint = self.fingerprint(cfg)

    def close(self):
        for name in ("llm", "encoder"):
            if name in self.instances:
                self.instances[name].close()
        self.instances.clear()
        self.current_fingerprint = None
