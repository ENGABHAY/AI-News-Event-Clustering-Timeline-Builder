"""
data_cleaning.py
================
Stage 2 — Data Cleaning: nulls, duplicates, dtypes.
"""

import numpy as np
import pandas as pd


class DataCleaner:
    """Cleans the raw news dataframe."""

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df = df.replace("", np.nan)
        df["authors"] = df["authors"].fillna("Unknown")

        if "short_description" in df.columns and "headline" in df.columns:
            df["short_description"] = df["short_description"].fillna(df["headline"])

        df = df.dropna(subset=["headline", "short_description"])

        # drop_duplicates() must hash every cell — columns holding dicts/lists
        # (e.g. an 'authors' field stored as a list of objects instead of a
        # plain string) crash it with "unhashable type: 'dict'". Convert any
        # such column to its string representation first.
        for col in df.columns:
            if df[col].apply(lambda v: isinstance(v, (dict, list))).any():
                df[col] = df[col].astype(str)

        df = df.drop_duplicates()

        if "date" in df.columns:
            # errors="coerce" turns any value pandas can't parse as a date
            # (e.g. a corrupted/malformed field in the source file) into NaT
            # instead of raising and killing the whole run.
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            n_bad = df["date"].isna().sum()
            if n_bad:
                df = df.dropna(subset=["date"])

        return df.reset_index(drop=True)