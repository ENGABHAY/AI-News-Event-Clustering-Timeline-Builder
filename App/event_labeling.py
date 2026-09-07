"""
event_labeling.py
==================
Stage 7b — Cluster Labeling (representative headline): the article
closest to each cluster's centroid becomes that event's label.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class EventLabeler:
    """Assigns a representative-headline event_label to each cluster."""

    def label(self, df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
        cluster_labels = {}
        clusters = sorted(df[df["cluster"] != -1]["cluster"].unique())

        for cluster in clusters:
            cluster_df = df[df["cluster"] == cluster]
            cluster_embeddings = embeddings[cluster_df.index]
            centroid = cluster_embeddings.mean(axis=0)
            similarity = cosine_similarity(cluster_embeddings, centroid.reshape(1, -1)).flatten()
            best_idx = similarity.argmax()
            cluster_labels[cluster] = cluster_df.iloc[best_idx]["headline"]

        df = df.copy()
        df["event_label"] = df["cluster"].map(cluster_labels).fillna("general_news")
        return df

    @staticmethod
    def top_articles(df: pd.DataFrame, embeddings: np.ndarray, cluster: int, n: int = 5) -> pd.DataFrame:
        """Helper used by summarization: n articles closest to the cluster centroid."""
        cluster_df = df[df["cluster"] == cluster]
        cluster_embeddings = embeddings[cluster_df.index]
        centroid = cluster_embeddings.mean(axis=0)
        similarity = cosine_similarity(cluster_embeddings, centroid.reshape(1, -1)).flatten()
        top_idx = similarity.argsort()[-n:][::-1]
        return cluster_df.iloc[top_idx].sort_values("date")
