"""
feature_engineering.py
=======================
Stage 4 — Feature Engineering: build the combined text field used
for embedding.
"""

import pandas as pd


class FeatureEngineer:
    """Builds the text field that gets embedded."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["text"] = df["headline"].astype(str) + " " + df["short_description"].astype(str)
        return df
