from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence
import numpy as np


@dataclass
class PredictionBatch:
    labels: list[str]
    scores: list[float]
    per_example_seconds: float


class PrototypeClassifier:
    """Zero-shot cosine prototype classifier using SentenceTransformer embeddings."""

    def __init__(self, model_id: str, labels: Sequence[str], device: str = 'cpu', batch_size: int = 128, revision: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model_id = model_id
        self.labels = list(labels)
        self.device = device
        self.batch_size = int(batch_size)
        kwargs = {'device': device}
        if revision:
            kwargs['revision'] = revision
        self.model = SentenceTransformer(model_id, **kwargs)
        self.label_embeddings = self.model.encode(
            self.labels,
            batch_size=min(self.batch_size, max(1, len(self.labels))),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def predict(self, texts: Sequence[str]) -> PredictionBatch:
        if not texts:
            return PredictionBatch([], [], 0.0)
        start = perf_counter()
        emb = self.model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        sims = np.matmul(emb, self.label_embeddings.T)
        idx = sims.argmax(axis=1)
        elapsed = perf_counter() - start
        return PredictionBatch(
            labels=[self.labels[int(i)] for i in idx],
            scores=[float(sims[j, int(i)]) for j, i in enumerate(idx)],
            per_example_seconds=elapsed / len(texts),
        )
