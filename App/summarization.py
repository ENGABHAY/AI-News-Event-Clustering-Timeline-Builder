"""
summarization.py
=================
Stage 10 — Automated Event Summarization.
Two independent paths: a local BART transformer, and Google Gemini
(structured JSON report). Heavy libraries are imported lazily.
"""

import json

import pandas as pd

from event_labeling import EventLabeler


class EventSummarizer:
    """Summarizes a single event (cluster) using BART or Gemini."""

    def __init__(self):
        self._bart_model = None
        self._bart_tokenizer = None
        self._bart_device = None

    # ------------------------------------------------------------------ #
    # BART (local, no API key)
    # ------------------------------------------------------------------ #
    def _load_bart(self):
        import torch
        from transformers import BartForConditionalGeneration, BartTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._bart_device = device
        self._bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
        self._bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        self._bart_model.eval()

    def summarize_event_bart(self, df: pd.DataFrame, cluster: int, max_tokens: int = 900) -> str:
        import torch

        if cluster == -1:
            return "General news articles not assigned to a specific event."

        if self._bart_model is None:
            self._load_bart()

        model, tokenizer, device = self._bart_model, self._bart_tokenizer, self._bart_device

        def summarize_text(text: str, max_length=150, min_length=50) -> str:
            inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                ids = model.generate(
                    inputs["input_ids"], max_length=max_length, min_length=min_length,
                    num_beams=4, early_stopping=True,
                )
            return tokenizer.decode(ids[0], skip_special_tokens=True)

        articles = df[df["cluster"] == cluster].sort_values("date")

        chunks, current, current_tokens = [], [], 0
        for _, row in articles.iterrows():
            text = f"On {row['date'].strftime('%Y-%m-%d')}, {row['headline']}. {row['short_description']}"
            n_tok = len(tokenizer.encode(text, add_special_tokens=False))
            if current_tokens + n_tok > max_tokens and current:
                chunks.append(" ".join(current))
                current, current_tokens = [], 0
            current.append(text)
            current_tokens += n_tok
        if current:
            chunks.append(" ".join(current))

        chunk_summaries = [summarize_text(c) for c in chunks]
        if len(chunk_summaries) == 1:
            return chunk_summaries[0]
        return summarize_text(" ".join(chunk_summaries), max_length=180, min_length=60)

    # ------------------------------------------------------------------ #
    # Gemini (LLM, structured JSON report)
    # ------------------------------------------------------------------ #
    def summarize_event_gemini(
        self, df: pd.DataFrame, embeddings, event_table: pd.DataFrame, cluster: int, api_key: str
    ) -> dict:
        from google import genai

        client = genai.Client(api_key=api_key)
        event_row = event_table[event_table["cluster"] == cluster].iloc[0]
        articles = EventLabeler.top_articles(df, embeddings, cluster)

        prompt = f"""You are a professional news analyst.

The following articles belong to the same news cluster.
Create a factual event report based ONLY on the information given.

EVENT INFORMATION
Representative headline: {event_row['representative_headline']}
Keywords: {event_row['keywords']}
Start date: {event_row['start_date']}
End date: {event_row['end_date']}
Article count: {event_row['article_count']}

ARTICLES
"""
        for i, (_, row) in enumerate(articles.iterrows(), start=1):
            prompt += (
                f"\nARTICLE {i}\nHeadline: {row['headline']}\n"
                f"Date: {row['date']}\nAuthor: {row['authors']}\n"
                f"Category: {row['category']}\nDescription: {row['short_description']}\n"
                f"URL: {row['link']}\n"
            )
        prompt += (
            "\nTASK\nReturn a JSON object with: event_title, summary (3-5 sentences), "
            "timeline (list of {date, event}), key_people, organizations, locations."
        )

        event_schema = {
            "type": "object",
            "properties": {
                "event_title": {"type": "string"},
                "summary": {"type": "string"},
                "timeline": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"date": {"type": "string"}, "event": {"type": "string"}},
                        "required": ["date", "event"],
                    },
                },
                "key_people": {"type": "array", "items": {"type": "string"}},
                "organizations": {"type": "array", "items": {"type": "string"}},
                "locations": {"type": "array", "items": {"type": "string"}},
            },
        }

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config={"response_mime_type": "application/json", "response_json_schema": event_schema},
        )
        return json.loads(response.text)
