import numpy as np
from sentence_transformers import CrossEncoder


class NLIClassifier:
    def __init__(self, model: str, batch_size: int = 16, revision: str | None = None, device: str | None = None):
        self._model = CrossEncoder(model, revision=revision, device=device)
        self._batch_size = batch_size
        self._entail_idx = self._find_label_index("entail")
        self._contra_idx = self._find_label_index("contra")
        self._neutral_idx = self._find_label_index("neutral")
        if min(self._entail_idx, self._contra_idx, self._neutral_idx) < 0:
            raise ValueError(
                f"NLI model {model!r} does not expose entailment, contradiction, and neutral labels"
            )

    def _find_label_index(self, needle: str) -> int:
        id2label = getattr(self._model.config, "id2label", {})
        for idx, label in id2label.items():
            if needle in str(label).lower():
                return int(idx)
        return -1

    def entails(self, premise: str, hypothesis: str) -> float:
        return self.entails_batch([(premise, hypothesis)])[0]

    def entails_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        results = self.classify_batch(pairs)
        if any(item.get("status") == "input_too_long" for item in results):
            raise ValueError("NLI input exceeds the model context limit")
        return [c["entailment"] for c in results]

    def classify_batch(self, pairs: list[tuple[str, str]]) -> list[dict]:
        if not pairs:
            return []
        limit = self._model.max_seq_length
        if not isinstance(limit, int) or not 0 < limit < 1_000_000:
            raise ValueError("NLI model does not expose a usable context limit")
        encoded = self._model.tokenizer(
            [pair[0] for pair in pairs], [pair[1] for pair in pairs],
            truncation=False, padding=False, add_special_tokens=True,
        )
        results = [None] * len(pairs)
        valid = []
        for index, tokens in enumerate(encoded["input_ids"]):
            count = len(tokens)
            if count > limit:
                results[index] = {"status": "input_too_long", "token_count": count,
                                  "token_limit": limit, "excess_tokens": count - limit}
            else:
                valid.append(index)
        if not valid:
            return results
        logits = np.array(
            self._model.predict([pairs[i] for i in valid], batch_size=self._batch_size, show_progress_bar=False),
            dtype=np.float32,
        )
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        predictions = [
            {
                "entailment": float(row[self._entail_idx]) if self._entail_idx >= 0 else 0.0,
                "contradiction": float(row[self._contra_idx]) if self._contra_idx >= 0 else 0.0,
                "neutral": float(row[self._neutral_idx]) if self._neutral_idx >= 0 else 0.0,
            }
            for row in probs
        ]
        for index, prediction in zip(valid, predictions):
            results[index] = prediction
        return results

    def health_check(self) -> None:
        self.entails("ping", "pong")
