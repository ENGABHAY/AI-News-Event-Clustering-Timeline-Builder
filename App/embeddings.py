"""
embeddings.py
=============
Stage 5 — Text Embedding Generation.
Generates Sentence-Transformer embeddings and caches them to .npy so
re-runs on the same dataset don't recompute them.
"""

import hashlib
import os

import numpy as np
import pandas as pd

from config import PipelineConfig


class EmbeddingGenerator:
    """Generates and caches sentence embeddings for the news text."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        os.makedirs(config.cache_dir, exist_ok=True)

    @staticmethod
    def _dataset_fingerprint(texts: list) -> str:
        """Short hash derived from the actual text content, so a different
        dataset (or a dataset that changed) never reuses another one's
        cached embeddings, even if cache_name collides."""
        h = hashlib.sha1()
        h.update(str(len(texts)).encode())
        for t in (texts[0], texts[len(texts) // 2], texts[-1]) if texts else ():
            h.update(t.encode(errors="ignore"))
        return h.hexdigest()[:10]

    def generate(
        self,
        df: pd.DataFrame,
        cache_name: str = "news_embeddings.npy",
        force_recompute: bool = False,
    ) -> np.ndarray:
        texts = df["text"].fillna("").astype(str).tolist()

        fingerprint = self._dataset_fingerprint(texts)
        stem, ext = os.path.splitext(cache_name)
        cache_path = os.path.join(self.config.cache_dir, f"{stem}_{fingerprint}{ext}")

        if os.path.exists(cache_path) and not force_recompute:
            cached = np.load(cache_path)
            # Defensive check even with fingerprinting: never hand back a
            # cached array whose row count doesn't match the current data.
            if cached.shape[0] == len(texts):
                return cached

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.config.embedding_model, device=self.config.embedding_device)
        embeddings = model.encode(
            texts,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        np.save(cache_path, embeddings)
        return embeddings