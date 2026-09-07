"""
keyword_extraction.py
======================
Stage 7a — Cluster Labeling (keywords): TF-IDF top keywords per cluster.
"""

from string import punctuation

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from config import PipelineConfig


class KeywordExtractor:
    """Extracts TF-IDF keywords for each cluster."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    @staticmethod
    def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
        import nltk
        from nltk.stem import WordNetLemmatizer

        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        lemmatizer = WordNetLemmatizer()

        def remove_pun(text: str) -> str:
            return "".join(c for c in text if c not in punctuation)

        def lemmatize_text(text: str) -> str:
            return " ".join(lemmatizer.lemmatize(w) for w in text.split())

        df = df.copy()
        df["text"] = df["text"].fillna("").str.lower().apply(remove_pun).apply(lemmatize_text)
        return df

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns df with a new 'keywords' column filled in."""
        cfg = self.config
        processed = self._preprocess(df)

        df_cluster = processed[processed["cluster"] != -1].copy()
        cluster_docs = df_cluster.groupby("cluster")["text"].apply(" ".join)

        vectorizer = CountVectorizer(
            stop_words="english",
            ngram_range=cfg.tfidf_ngram_range,
            min_df=cfg.tfidf_min_df,
            max_df=cfg.tfidf_max_df,
        )
        count_matrix = vectorizer.fit_transform(cluster_docs)
        tfidf = TfidfTransformer().fit_transform(count_matrix)
        words = np.array(vectorizer.get_feature_names_out())

        k = cfg.top_keywords_per_cluster
        keyword_text = {}
        for i, cluster in enumerate(cluster_docs.index):
            scores = tfidf[i].toarray().flatten()
            top_idx = scores.argsort()[-k:][::-1]
            keyword_text[cluster] = ", ".join(words[top_idx])

        df = df.copy()
        df["keywords"] = df["cluster"].map(keyword_text).fillna("general_news")
        return df
