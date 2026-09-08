"""
pipeline.py
===========
NewsEventPipeline — orchestrates the stage classes below. This file
holds NO clustering/embedding/summarization logic itself; it only
wires the classes from the other modules together in order, matching
the notebook's 11 stages:

  1  DataLoader              -> load_data()
  2  DataCleaner             -> clean_data()
  4  FeatureEngineer         -> feature_engineering()
  5  EmbeddingGenerator      -> generate_embeddings()      (cached .npy)
  6  EventClusterer          -> cluster_events()
  7a KeywordExtractor        -> extract_keywords()
  7b EventLabeler            -> label_events()
  8  EventTableBuilder       -> build_event_table()
  10 EventSummarizer         -> summarize_event_bart() / summarize_event_gemini()
  11 EventExporter           -> export_top_events()
"""

from typing import Optional

import numpy as np
import pandas as pd

from config import PipelineConfig
from data_loader import DataLoader
from data_cleaning import DataCleaner
from feature_engineering import FeatureEngineer
from embeddings import EmbeddingGenerator
from clustering import EventClusterer
from keyword_extraction import KeywordExtractor
from event_labeling import EventLabeler
from event_table import EventTableBuilder
from summarization import EventSummarizer
from export import EventExporter


class NewsEventPipeline:
    """Thin orchestrator over the stage classes. Holds state on `self`
    so the Streamlit UI can run stages independently and inspect
    intermediate results."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        self._loader = DataLoader()
        self._cleaner = DataCleaner()
        self._feature_engineer = FeatureEngineer()
        self._embedder = EmbeddingGenerator(self.config)
        self._clusterer = EventClusterer(self.config)
        self._keyword_extractor = KeywordExtractor(self.config)
        self._labeler = EventLabeler()
        self._table_builder = EventTableBuilder()
        self._summarizer = EventSummarizer()
        self._exporter = EventExporter(self._summarizer)

        self.df: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None
        self.event_table: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # Stage 1
    # ------------------------------------------------------------------ #
    def load_data(self, path: str) -> pd.DataFrame:
        self.df = self._loader.load(path)
        return self.df

    # ------------------------------------------------------------------ #
    # Stage 2
    # ------------------------------------------------------------------ #
    def clean_data(self) -> pd.DataFrame:
        self.df = self._cleaner.clean(self.df)
        return self.df

    # ------------------------------------------------------------------ #
    # Stage 4
    # ------------------------------------------------------------------ #
    def feature_engineering(self) -> pd.DataFrame:
        self.df = self._feature_engineer.transform(self.df)
        return self.df

    # ------------------------------------------------------------------ #
    # Stage 5
    # ------------------------------------------------------------------ #
    def generate_embeddings(self, cache_name: str = "news_embeddings.npy", force_recompute: bool = False) -> np.ndarray:
        self.embeddings = self._embedder.generate(self.df, cache_name, force_recompute)
        return self.embeddings

    # ------------------------------------------------------------------ #
    # Stage 6
    # ------------------------------------------------------------------ #
    def cluster_events(self) -> pd.DataFrame:
        self.df = self._clusterer.cluster(self.df, self.embeddings)
        return self.df

    # ------------------------------------------------------------------ #
    # Stage 7a
    # ------------------------------------------------------------------ #
    def extract_keywords(self) -> pd.DataFrame:
        self.df = self._keyword_extractor.extract(self.df)
        return self.df

    # ------------------------------------------------------------------ #
    # Stage 7b
    # ------------------------------------------------------------------ #
    def label_events(self) -> pd.DataFrame:
        self.df = self._labeler.label(self.df, self.embeddings)
        return self.df

    # ------------------------------------------------------------------ #
    # Stage 8
    # ------------------------------------------------------------------ #
    def build_event_table(self) -> pd.DataFrame:
        self.event_table = self._table_builder.build(self.df)
        return self.event_table

    # ------------------------------------------------------------------ #
    # Stage 10
    # ------------------------------------------------------------------ #
    def summarize_event_bart(self, cluster: int) -> str:
        return self._summarizer.summarize_event_bart(self.df, cluster)

    def summarize_event_gemini(self, cluster: int, api_key: str) -> dict:
        return self._summarizer.summarize_event_gemini(self.df, self.embeddings, self.event_table, cluster, api_key)

    # ------------------------------------------------------------------ #
    # Stage 11
    # ------------------------------------------------------------------ #
    def export_top_events(
        self, n: int = 100, method: str = "bart", gemini_api_key: Optional[str] = None, out_path: str = "top_events.csv"
    ) -> pd.DataFrame:
        return self._exporter.export_top_events(
            self.df, self.embeddings, self.event_table, n, method, gemini_api_key, out_path
        )

    # ------------------------------------------------------------------ #
    # Convenience: run stages 1-8 in one call
    # ------------------------------------------------------------------ #
    def run_core_pipeline(self, data_path: str) -> pd.DataFrame:
        self.load_data(data_path)
        self.clean_data()
        self.feature_engineering()
        self.generate_embeddings()
        self.cluster_events()
        self.extract_keywords()
        self.label_events()
        return self.build_event_table()
