"""
main.py
=======
Streamlit UI for the News Event Clustering & Summarization pipeline.

Run with:
    streamlit run main.py
"""

import pandas as pd
import streamlit as st

from pipeline import NewsEventPipeline, PipelineConfig

st.set_page_config(page_title="News Event Clustering", layout="wide")
st.title("📰 News Event Clustering & Summarization")

# ---------------------------------------------------------------------- #
# Session state
# ---------------------------------------------------------------------- #
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "stage" not in st.session_state:
    st.session_state.stage = 0  # how many stages have completed

# ---------------------------------------------------------------------- #
# Sidebar — data source + hyperparameters
# ---------------------------------------------------------------------- #
with st.sidebar:
    st.header("1. Data")
    uploaded_file = st.file_uploader("News dataset (.json lines or .csv)", type=["json", "csv"])

    st.header("2. Clustering parameters")
    n_neighbors = st.slider("UMAP n_neighbors", 5, 50, 15)
    n_components = st.slider("UMAP n_components", 5, 100, 50)
    min_cluster_size = st.slider("HDBSCAN min_cluster_size", 2, 30, 4)
    embedding_model = st.text_input("Sentence-Transformer model", "sentence-transformers/all-MiniLM-L6-v2")
    device = st.selectbox("Embedding device", ["cpu", "cuda"], index=0)

    st.header("3. Summarization")
    summary_method = st.radio("Method", ["BART (local)", "Gemini (API)"])
    gemini_key = ""
    if summary_method == "Gemini (API)":
        gemini_key = st.text_input("Gemini API key", type="password")

    run_button = st.button("🚀 Run pipeline (stages 1-8)", use_container_width=True)

# ---------------------------------------------------------------------- #
# Run core pipeline
# ---------------------------------------------------------------------- #
if run_button:
    if uploaded_file is None:
        st.error("Upload a dataset first.")
        st.stop()

    suffix = ".json" if uploaded_file.name.endswith(".json") else ".csv"
    tmp_path = f"uploaded_dataset{suffix}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    config = PipelineConfig(
        embedding_model=embedding_model,
        embedding_device=device,
        umap_n_neighbors=n_neighbors,
        umap_n_components=n_components,
        hdbscan_min_cluster_size=min_cluster_size,
    )
    pipeline = NewsEventPipeline(config)

    with st.status("Running pipeline...", expanded=True) as status:
        st.write("Loading data...")
        pipeline.load_data(tmp_path)
        st.write(f"Loaded {len(pipeline.df)} rows.")

        st.write("Cleaning data (nulls, duplicates, dtypes)...")
        pipeline.clean_data()
        st.write(f"{len(pipeline.df)} rows remain after cleaning.")
        if len(pipeline.df) == 0:
            status.update(label="Pipeline failed: 0 rows after cleaning", state="error")
            st.error(
                "No rows survived cleaning. This usually means the dataset "
                "wasn't parsed correctly (wrong JSON format) or every row "
                "had an unparseable date. Check the raw file's structure."
            )
            st.stop()

        st.write("Building combined text field...")
        pipeline.feature_engineering()

        st.write("Generating sentence embeddings (cached after first run)...")
        pipeline.generate_embeddings()

        st.write("Reducing dimensions (UMAP) + clustering (HDBSCAN)...")
        pipeline.cluster_events()

        st.write("Extracting per-cluster TF-IDF keywords...")
        pipeline.extract_keywords()

        st.write("Labeling events with representative headlines...")
        pipeline.label_events()

        st.write("Building the event table...")
        pipeline.build_event_table()

        status.update(label="Pipeline complete ✅", state="complete")

    st.session_state.pipeline = pipeline
    st.session_state.stage = 8

pipeline: NewsEventPipeline = st.session_state.pipeline

# ---------------------------------------------------------------------- #
# Results
# ---------------------------------------------------------------------- #
if pipeline is None:
    st.info("Upload a dataset and click **Run pipeline** in the sidebar to get started.")
    st.stop()

df = pipeline.df
event_table = pipeline.event_table

tab_overview, tab_events, tab_summary, tab_export = st.tabs(
    ["📊 Overview", "🗂️ Events", "📝 Summarize an event", "⬇️ Export"]
)

# ---- Overview / EDA -----------------------------------------------------
with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Articles", len(df))
    col2.metric("Events found", event_table["cluster"].nunique())
    col3.metric("Avg articles / event", round(event_table["article_count"].mean(), 1))
    col4.metric("Avg event duration (days)", round(event_table["duration_days"].mean(), 1))

    st.subheader("Articles per category")
    if "category" in df.columns:
        st.bar_chart(df["category"].value_counts())

    st.subheader("Articles published over time")
    if "date" in df.columns:
        monthly = df.groupby(df["date"].dt.to_period("M")).size()
        monthly.index = monthly.index.astype(str)
        st.line_chart(monthly)

    st.subheader("Top 10 largest events")
    top10 = event_table.nlargest(10, "article_count")[["representative_headline", "article_count", "duration_days"]]
    st.bar_chart(top10.set_index("representative_headline")["article_count"])

# ---- Event table ---------------------------------------------------------
with tab_events:
    st.subheader("Event table")
    search = st.text_input("Filter by keyword or headline")
    view = event_table
    if search:
        mask = (
            view["representative_headline"].str.contains(search, case=False, na=False)
            | view["keywords"].str.contains(search, case=False, na=False)
        )
        view = view[mask]
    st.dataframe(
        view[["cluster", "representative_headline", "keywords", "article_count", "start_date", "end_date", "duration_days"]],
        use_container_width=True,
    )

# ---- Summarize a single event --------------------------------------------
with tab_summary:
    st.subheader("Generate a summary for one event")
    cluster_options = event_table.sort_values("article_count", ascending=False)["cluster"].tolist()
    chosen_cluster = st.selectbox(
        "Choose an event (cluster id)",
        cluster_options,
        format_func=lambda c: f"{c} — {event_table[event_table['cluster'] == c].iloc[0]['representative_headline']}",
    )

    if st.button("Generate summary"):
        with st.spinner("Summarizing..."):
            if summary_method == "BART (local)":
                summary_text = pipeline.summarize_event_bart(chosen_cluster)
                st.write(summary_text)
            else:
                if not gemini_key:
                    st.error("Enter a Gemini API key in the sidebar first.")
                else:
                    result = pipeline.summarize_event_gemini(chosen_cluster, gemini_key)
                    st.markdown(f"### {result.get('event_title', '')}")
                    st.write(result.get("summary", ""))
                    if result.get("timeline"):
                        st.subheader("Timeline")
                        st.table(pd.DataFrame(result["timeline"]))
                    st.write("**Key people:**", ", ".join(result.get("key_people", [])) or "—")
                    st.write("**Organizations:**", ", ".join(result.get("organizations", [])) or "—")
                    st.write("**Locations:**", ", ".join(result.get("locations", [])) or "—")

# ---- Export ---------------------------------------------------------------
with tab_export:
    st.subheader("Summarize + export the top-N events")
    n = st.number_input("Number of top events", min_value=5, max_value=500, value=20, step=5)

    if st.button("Run export"):
        with st.spinner(f"Summarizing top {n} events — this can take a while..."):
            method = "bart" if summary_method == "BART (local)" else "gemini"
            result_df = pipeline.export_top_events(
                n=int(n), method=method, gemini_api_key=gemini_key, out_path="top_events.csv"
            )
        st.success(f"Exported {len(result_df)} events.")
        st.dataframe(result_df, use_container_width=True)
        st.download_button(
            "Download CSV",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="top_events.csv",
            mime="text/csv",
        )