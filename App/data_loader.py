"""
data_loader.py
==============
Stage 1 — Data Loading & Setup.
"""

import pandas as pd


class DataLoader:
    """Loads the raw news dataset from disk."""

    @staticmethod
    def _looks_malformed(df: pd.DataFrame) -> bool:
        """True if the parse produced a suspicious near-empty dataframe where
        columns hold dicts/lists instead of scalar values — the tell-tale sign
        that a 'columns'-oriented JSON file (one big object, not one record
        per line) got misread as JSON-lines."""
        if len(df) > 1:
            return False
        return any(
            df[col].apply(lambda v: isinstance(v, (dict, list))).any()
            for col in df.columns
        )

    def load(self, path: str) -> pd.DataFrame:
        if path.endswith(".csv"):
            df = pd.read_csv(path)

        elif path.endswith(".json"):
            try:
                df = pd.read_json(path, lines=True)
            except ValueError:
                df = None

            if df is None or self._looks_malformed(df):
                # Not one-record-per-line after all — let pandas auto-detect
                # the actual layout (e.g. a single "columns"-oriented object).
                df = pd.read_json(path, lines=False)

        else:
            raise ValueError("Expected a .json (lines=True) or .csv file")

        if len(df) == 0:
            raise ValueError(
                "Loaded dataset has 0 rows. Check that the file is valid "
                "JSON-lines (one record per line) or a proper CSV."
            )

        return df