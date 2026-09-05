"""
News Event Clustering & Summarization — Core Pipeline
=======================================================

Reusable stage functions for the news event clustering pipeline, extracted
from the original "News_clustering" notebook. This module has no CLI of its
own — it's imported by the FastAPI backend (main.py), which calls each stage
in order and reports progress to the Streamlit frontend.

Stages:
  1. load_data            — read a JSON-Lines or CSV file into a DataFrame
  2. clean_data            — handle missing values, duplicates, date parsing
  3. run_eda                — save EDA plots to disk (optional)
  4. build_text_field       — combine headline + short_description
  5. generate_embeddings    — Sentence-Transformers embeddings
  6. cluster_events         — UMAP dimensionality reduction + HDBSCAN
  7. label_clusters         — TF-IDF keywords + representative headline
  8. build_event_table      — one row per event (cluster)
  9. visualize_events       — save event-analysis plots to disk (optional)
 10. summarize_with_llm     — Gemini-based event summaries (optional)
 11. summarize_with_bart    — BART-based event summaries (optional)

Required input columns: headline, short_description, authors, date
(category is optional but improves the EDA charts).
"""

import json
import os
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ==========================================================================
# 1. DATA LOADING
# ==========================================================================
def load_data(input_path: str) -> pd.DataFrame:
    """Load the input file as either JSON-Lines or CSV, based on extension."""
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".json":
        df = pd.read_json(input_path, lines=True)
    elif ext in (".jsonl",):
        df = pd.read_json(input_path, lines=True)
    elif ext == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Please provide a .json, "
            f".jsonl, or .csv file."
        )

    print(f"[1/9] Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ==========================================================================
# 2. DATA CLEANING
# ==========================================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, duplicates, and datetime conversion."""
    df = df.copy()

    required_cols = ["headline", "short_description", "authors", "date"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Input file is missing required column(s): {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    # Treat empty strings as missing
    df = df.replace("", np.nan)

    # Fill missing authors with 'Unknown'
    df["authors"] = df["authors"].fillna("Unknown")

    # Fill missing short_description with the headline
    df["short_description"] = df["short_description"].fillna(df["headline"])

    # Drop any remaining rows with missing headline/short_description
    df = df.dropna(subset=["headline", "short_description"])

    # Remove duplicate rows
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]

    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df = df.reset_index(drop=True)

    print(
        f"[2/9] Cleaned data: removed {before - after} duplicate rows, "
        f"{df.shape[0]} rows remaining"
    )
    return df


# ==========================================================================
# 3. EXPLORATORY DATA ANALYSIS (optional, saves plots to disk)
# ==========================================================================
def run_eda(df: pd.DataFrame, plots_dir: str) -> None:
    """Generate and save the same EDA plots produced in the notebook."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(plots_dir, exist_ok=True)

    # Category-wise news counts
    if "category" in df.columns:
        plt.figure(figsize=(19, 6))
        sns.countplot(data=df, x="category")
        plt.xticks(rotation=90)
        plt.title("Category wise news counts")
        plt.grid(axis="both", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "category_counts.png"))
        plt.close()

        # Least / most common categories
        plt.figure(figsize=(20, 6))
        plt.subplot(1, 2, 1)
        (df["category"].value_counts(normalize=True, ascending=True) * 100).head(5).plot(
            kind="barh"
        )
        plt.title("The categories with least news (%)")
        plt.grid(axis="both", alpha=0.2)

        plt.subplot(1, 2, 2)
        (df["category"].value_counts(normalize=True, ascending=False) * 100).head(5).plot(
            kind="barh"
        )
        plt.title("The categories with most news (%)")
        plt.grid(axis="both", alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "category_share.png"))
        plt.close()

    # Articles published over time
    articles_per_month = df.groupby(df["date"].dt.to_period("M")).size()
    plt.figure(figsize=(19, 5))
    articles_per_month.plot()
    plt.title("Articles Published Over Time")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "articles_over_time.png"))
    plt.close()

    print(f"[3/9] EDA plots saved to '{plots_dir}'")


# ==========================================================================
# 4. FEATURE ENGINEERING — combined text field
# ==========================================================================
def build_text_field(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["headline"] + " " + df["short_description"]
    print("[4/9] Built combined 'text' field (headline + short_description)")
    return df


# ==========================================================================
# 5. TEXT EMBEDDING GENERATION
# ==========================================================================
def generate_embeddings(
    df: pd.DataFrame,
    output_dir: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 128,
    use_cache: bool = True,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    import torch

    emb_path = os.path.join(output_dir, "news_embeddings.npy")
    if use_cache and os.path.exists(emb_path):
        print(f"[5/9] Loading cached embeddings from '{emb_path}'")
        return np.load(emb_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)

    embeddings = model.encode(
        df["text"].fillna("").astype(str).tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    np.save(emb_path, embeddings)
    print(f"[5/9] Generated embeddings for {len(df)} articles (device={device})")
    return embeddings


# ==========================================================================
# 6. EVENT CLUSTERING — UMAP + HDBSCAN
# ==========================================================================
def cluster_events(
    embeddings: np.ndarray,
    output_dir: str,
    n_neighbors: int = 15,
    n_components: int = 50,
    min_cluster_size: int = 4,
    use_cache: bool = True,
) -> np.ndarray:
    import umap
    import hdbscan

    labels_path = os.path.join(output_dir, "news_labels.npy")
    if use_cache and os.path.exists(labels_path):
        print(f"[6/9] Loading cached cluster labels from '{labels_path}'")
        return np.load(labels_path)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric="cosine",
        random_state=42,
    )
    reduced_emb = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(reduced_emb)

    np.save(labels_path, labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[6/9] Clustering complete: {n_clusters} events found "
          f"({(labels == -1).sum()} articles unclustered/noise)")
    return labels


# ==========================================================================
# 7. CLUSTER LABELING — keywords (TF-IDF) + representative headline
# ==========================================================================
def label_clusters(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    import nltk
    from string import punctuation
    from nltk.stem import WordNetLemmatizer
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    df = df.copy()

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4", quiet=True)

    lemmatizer = WordNetLemmatizer()

    def remove_pun(text):
        return "".join(char for char in text if char not in punctuation)

    def normalize_text(text):
        text = remove_pun(str(text).lower())
        return " ".join(lemmatizer.lemmatize(w) for w in text.split())

    df["text_norm"] = df["text"].apply(normalize_text)

    # ---- Keyword extraction (TF-IDF) ----
    df_cluster = df[df["cluster"] != -1].copy()
    cluster_docs = df_cluster.groupby("cluster")["text_norm"].apply(" ".join)

    vectorizer = CountVectorizer(stop_words="english", max_features=5000)
    count_matrix = vectorizer.fit_transform(cluster_docs)

    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(count_matrix)

    feature_names = np.array(vectorizer.get_feature_names_out())

    cluster_keywords = {}
    for i, cluster_id in enumerate(cluster_docs.index):
        row = tfidf_matrix[i].toarray().flatten()
        top_idx = row.argsort()[::-1][:10]
        cluster_keywords[cluster_id] = [
            (feature_names[j], row[j]) for j in top_idx if row[j] > 0
        ]

    cluster_keyword_text = {
        cluster: ", ".join(word for word, score in keywords)
        for cluster, keywords in cluster_keywords.items()
    }
    df["keywords"] = df["cluster"].map(cluster_keyword_text)

    # ---- Representative headline extraction (similarity-based) ----
    cluster_labels = {}
    clusters = sorted(df[df["cluster"] != -1]["cluster"].unique())

    for cluster in clusters:
        cluster_df = df[df["cluster"] == cluster]
        cluster_embeddings = embeddings[cluster_df.index]
        centroid = cluster_embeddings.mean(axis=0)
        similarity = cosine_similarity([centroid], cluster_embeddings)[0]
        best_idx = similarity.argmax()
        representative_headline = cluster_df.iloc[best_idx]["headline"]
        cluster_labels[cluster] = representative_headline

    df["event_label"] = df["cluster"].map(cluster_labels)
    df["event_label"] = df["event_label"].fillna("general_news")

    df = df.drop(columns=["text_norm"])
    df = df.reset_index(drop=True)

    print(f"[7/9] Labeled {len(clusters)} clusters with keywords and representative headlines")
    return df


# ==========================================================================
# 8. EVENT TABLE CONSTRUCTION
# ==========================================================================
def build_event_table(df: pd.DataFrame) -> pd.DataFrame:
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

    event_table["duration_days"] = (
        event_table["end_date"] - event_table["start_date"]
    ).dt.days + 1

    event_table = event_table[event_table["cluster"] != -1].copy()
    event_table = event_table.sort_values("start_date").reset_index(drop=True)

    print(f"[8/9] Built event table with {len(event_table)} events")
    return event_table


def visualize_events(event_table: pd.DataFrame, plots_dir: str) -> None:
    """Generate and save the event-level analysis plots from the notebook."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plots_dir, exist_ok=True)

    top10 = event_table.nlargest(10, "article_count").sort_values("article_count")
    plt.figure(figsize=(12, 6))
    plt.barh(top10["representative_headline"], top10["article_count"])
    plt.xlabel("Number of Articles")
    plt.ylabel("Event")
    plt.title("Top 10 Events by Article Count")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "top10_events_by_article_count.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(event_table["duration_days"], bins=30)
    plt.xlabel("Duration (Days)")
    plt.ylabel("Number of Events")
    plt.title("Distribution of Event Duration")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "event_duration_distribution.png"))
    plt.close()

    print(f"[9/9] Event analysis plots saved to '{plots_dir}'")


# ==========================================================================
# 9. (OPTIONAL) AUTOMATED EVENT SUMMARIZATION
# ==========================================================================
def summarize_with_llm(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    event_table: pd.DataFrame,
    api_key: str,
    top_n: int = 20,
) -> pd.DataFrame:
    """Summarize the top-N events using Google Gemini, as in the notebook."""
    from google import genai
    from sklearn.metrics.pairwise import cosine_similarity

    client = genai.Client(api_key=api_key)

    event_schema = {
        "type": "object",
        "properties": {
            "event_title": {"type": "string"},
            "summary": {"type": "string"},
            "timeline": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string"},
                        "development": {"type": "string"},
                    },
                },
            },
        },
    }

    def build_prompt(event_row, representative_articles):
        articles_text = "\n".join(
            f"- ({a['date'].date()}) {a['headline']}: {a['short_description']}"
            for a in representative_articles
        )
        return f"""
You are a professional news analyst.

The following articles belong to the same news cluster.

Your task is to create a factual event report based ONLY
on the information provided in these articles.

EVENT INFORMATION
------------------
Representative headline: {event_row['representative_headline']}
Keywords: {event_row['keywords']}
Date range: {event_row['start_date']} to {event_row['end_date']}

ARTICLES
--------
{articles_text}
"""

    def generate_summary(prompt):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": event_schema,
            },
        )
        return response.text

    top_events = event_table.nlargest(top_n, "article_count").copy()
    top_events["summary"] = ""

    for idx, row in top_events.iterrows():
        cluster = row["cluster"]
        cluster_df = df[df["cluster"] == cluster]
        cluster_embeddings = embeddings[cluster_df.index]
        centroid = cluster_embeddings.mean(axis=0)
        similarity = cosine_similarity([centroid], cluster_embeddings)[0]
        top_idx = similarity.argsort()[::-1][:5]
        representative_articles = cluster_df.iloc[top_idx].to_dict("records")

        try:
            prompt = build_prompt(row, representative_articles)
            top_events.at[idx, "summary"] = generate_summary(prompt)
        except Exception as e:
            top_events.at[idx, "summary"] = f"ERROR: {e}"

    return top_events


def summarize_with_bart(
    df: pd.DataFrame,
    event_table: pd.DataFrame,
    top_n: int = 20,
    max_tokens: int = 900,
) -> pd.DataFrame:
    """Summarize the top-N events using a pretrained BART model, as in the notebook."""
    import torch
    from transformers import BartForConditionalGeneration, BartTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    def bart_summarize(text, max_length=150, min_length=40):
        inputs = tokenizer(
            [text], max_length=1024, truncation=True, return_tensors="pt"
        ).to(device)
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=max_length,
            min_length=min_length,
            early_stopping=True,
        )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def create_cluster_chunks(cluster):
        articles = df[df["cluster"] == cluster].copy().sort_values("date")
        chunks, current_articles, current_len = [], [], 0

        for _, article in articles.iterrows():
            piece = f"{article['headline']}. {article['short_description']}"
            piece_len = len(piece.split())
            if current_len + piece_len > max_tokens and current_articles:
                chunks.append(" ".join(current_articles))
                current_articles, current_len = [], 0
            current_articles.append(piece)
            current_len += piece_len

        if current_articles:
            chunks.append(" ".join(current_articles))
        return chunks

    def generate_event_summary(cluster):
        if cluster == -1:
            return "General news articles not assigned to a specific event."
        chunks = create_cluster_chunks(cluster)
        chunk_summaries = [bart_summarize(c) for c in chunks]
        combined = " ".join(chunk_summaries)
        if len(chunk_summaries) > 1:
            return bart_summarize(combined)
        return combined

    top_events = event_table.nlargest(top_n, "article_count").copy()
    top_events["summary"] = ""

    for idx, row in top_events.iterrows():
        try:
            top_events.at[idx, "summary"] = generate_event_summary(row["cluster"])
        except Exception as e:
            top_events.at[idx, "summary"] = f"ERROR: {e}"

    return top_events


