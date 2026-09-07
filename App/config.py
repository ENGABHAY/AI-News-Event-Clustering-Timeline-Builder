"""
config.py
=========
Single shared configuration object passed into every stage class.
Keeping it in one place means main.py only has to build one object
from the sidebar controls.
"""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"          # "cuda" if a GPU is available
    umap_n_neighbors: int = 15
    umap_n_components: int = 50
    umap_metric: str = "cosine"
    hdbscan_min_cluster_size: int = 4
    hdbscan_metric: str = "euclidean"
    tfidf_ngram_range: tuple = (1, 2)
    tfidf_min_df: int = 2
    tfidf_max_df: float = 0.95
    top_keywords_per_cluster: int = 10
    random_state: int = 42
    cache_dir: str = "cache"
