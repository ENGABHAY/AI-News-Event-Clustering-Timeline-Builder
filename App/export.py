"""
export.py
=========
Stage 11 — Final Output Generation & Export.
"""

from typing import Optional

import pandas as pd

from summarization import EventSummarizer


class EventExporter:
    """Summarizes and exports the top-N events by article count."""

    def __init__(self, summarizer: EventSummarizer):
        self.summarizer = summarizer

    def export_top_events(
        self,
        df: pd.DataFrame,
        embeddings,
        event_table: pd.DataFrame,
        n: int = 100,
        method: str = "bart",
        gemini_api_key: Optional[str] = None,
        out_path: str = "top_events.csv",
    ) -> pd.DataFrame:
        top = event_table.nlargest(n, "article_count").copy()
        summaries, authors_col, refs_col = [], [], []

        for _, row in top.iterrows():
            cluster = row["cluster"]
            try:
                if method == "bart":
                    summary = self.summarizer.summarize_event_bart(df, cluster)
                elif method == "gemini":
                    result = self.summarizer.summarize_event_gemini(
                        df, embeddings, event_table, cluster, gemini_api_key
                    )
                    summary = result["summary"]
                else:
                    summary = ""
            except Exception as e:  # keep exporting even if one cluster fails
                summary = f"[error: {e}]"
            summaries.append(summary)

            cluster_df = df[df["cluster"] == cluster]
            authors_col.append(cluster_df["authors"].dropna().astype(str).str.strip().unique().tolist())
            refs_col.append(cluster_df["link"].dropna().astype(str).str.strip().unique().tolist())

        top["summary"] = summaries
        top["authors"] = authors_col
        top["references"] = refs_col
        top.to_csv(out_path, index=False)
        return top
