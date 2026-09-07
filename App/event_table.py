"""
event_table.py
==============
Stage 8 — Event Table Construction: one row per event (cluster).
"""

import pandas as pd


class EventTableBuilder:
    """Aggregates the article-level dataframe into an event-level table."""

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        event_table = (
            df.groupby("cluster")
            .agg(
                representative_headline=("event_label", "first"),
                keywords=("keywords", "first"),
                article_count=("headline", "count"),
                start_date=("date", "min"),
                end_date=("date", "max"),
            )
            .reset_index()
        )
        event_table["duration_days"] = (event_table["end_date"] - event_table["start_date"]).dt.days + 1
        event_table = event_table[event_table["cluster"] != -1]
        return event_table.sort_values("start_date").reset_index(drop=True)
