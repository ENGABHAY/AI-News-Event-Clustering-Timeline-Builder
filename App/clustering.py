"""
clustering.py
=============
Stage 6 — Event Clustering: UMAP dimensionality reduction + HDBSCAN.
"""

import numpy as np
import pandas as pd

from config import PipelineConfig


class EventClusterer:
    """Reduces embedding dimensionality then clusters into events."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def cluster(self, df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
        import hdbscan
        import umap

        if embeddings is None or len(embeddings) == 0:
            raise ValueError(
                "No embeddings to cluster (0 rows). This usually means the "
                "dataset ended up empty after cleaning — check the row count "
                "after the cleaning step, and that dates parsed correctly."
            )

        cfg = self.config

        reducer = umap.UMAP(
            n_neighbors=cfg.umap_n_neighbors,
            n_components=cfg.umap_n_components,
            metric=cfg.umap_metric,
            random_state=cfg.random_state,
        )
        reduced = reducer.fit_transform(embeddings)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=cfg.hdbscan_min_cluster_size,
            metric=cfg.hdbscan_metric,
        )
        labels = clusterer.fit_predict(reduced)

        df = df.copy()
        df["cluster"] = labels
        return df