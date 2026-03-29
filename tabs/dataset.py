import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

# Loading dataset from HuggingFace (cached after first load)
@st.cache_data(show_spinner=False)
def load_df():
    csv_path = hf_hub_download(
        repo_id   = "harshuajmani/nlp_geoCanada_embeddings",
        filename  = "index_data.csv",
        repo_type = "dataset"
    )
    return pd.read_csv(csv_path).fillna("")

def render():
    st.subheader("Dataset Information")

    with st.spinner("Loading dataset..."):
        df = load_df()

    # ── Section 1: Key Stats ──────────────────────────────────────────────────
    st.markdown("### 📈 Key Stats")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records",       f"{len(df):,}")
    col2.metric("Unique Organizations", f"{df['org'].nunique():,}")
    col3.metric("Unique Subjects",      f"{df['subject'].nunique():,}")
    col4.metric("Languages",            "EN + FR")
    st.divider()

    # ── Section 2: Before vs After Cleaning ──────────────────────────────────
    st.markdown("### 🧹 Before vs After Cleaning")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Before Cleaning**")
        st.table(pd.DataFrame({
            "Property":  ["Total records", "Fields per record", "Bilingual fields", "Empty descriptions", "Markdown links in text", "Identical EN/FR titles"],
            "Value":     ["46,468",         "65+",               "title, notes only", "~3,000",             "Yes",                    "~2,000"]
        }))

    with col2:
        st.markdown("**After Cleaning**")
        st.table(pd.DataFrame({
            "Property":  ["Total records", "Fields kept", "Bilingual fields",                     "Empty descriptions", "Markdown links in text", "Identical EN/FR titles"],
            "Value":     ["46,468",         "8",           "title, desc, keywords, org, subject",  "0",                  "Removed",                "Filtered for evaluation"]
        }))
    st.divider()

    # ── Section 3: Column Coverage ────────────────────────────────────────────
    st.markdown("### 📋 Column Info & Coverage")
    coverage = pd.DataFrame({
        "Column":      ["title_en", "title_fr", "desc_en",  "desc_fr",  "keywords_en", "keywords_fr", "org",   "subject"],
        "Type":        ["text",     "text",     "text",     "text",     "text",        "text",        "text",  "text"],
        "Non-empty":   [
            (df['title_en']    != '').sum(),
            (df['title_fr']    != '').sum(),
            (df['desc_en']     != '').sum(),
            (df['desc_fr']     != '').sum(),
            (df['keywords_en'] != '').sum(),
            (df['keywords_fr'] != '').sum(),
            (df['org']         != '').sum(),
            (df['subject']     != '').sum(),
        ],
        "Coverage %": [
            f"{(df['title_en']    != '').mean()*100:.1f}%",
            f"{(df['title_fr']    != '').mean()*100:.1f}%",
            f"{(df['desc_en']     != '').mean()*100:.1f}%",
            f"{(df['desc_fr']     != '').mean()*100:.1f}%",
            f"{(df['keywords_en'] != '').mean()*100:.1f}%",
            f"{(df['keywords_fr'] != '').mean()*100:.1f}%",
            f"{(df['org']         != '').mean()*100:.1f}%",
            f"{(df['subject']     != '').mean()*100:.1f}%",
        ]
    })
    st.dataframe(coverage, use_container_width=True)
    st.divider()

    # ── Section 4: Top Organizations ──────────────────────────────────────────
    st.markdown("### 🏛️ Top 10 Publishing Organizations")
    top_orgs = df['org'].value_counts().head(10)
    fig, ax  = plt.subplots(figsize=(10, 4))
    ax.barh(top_orgs.index, top_orgs.values, color='#2196F3')
    ax.set_xlabel("Number of Datasets")
    ax.tick_params(axis='y', labelsize=8)
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1e2130')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    plt.tight_layout()
    st.pyplot(fig)
    st.divider()

    # ── Section 5: Top Subjects ───────────────────────────────────────────────
    st.markdown("### 📂 Top 10 Subjects")
    top_subjects = df['subject'].value_counts().head(10)
    fig2, ax2    = plt.subplots(figsize=(10, 4))
    ax2.barh(top_subjects.index, top_subjects.values, color='#4CAF50')
    ax2.set_xlabel("Number of Datasets")
    ax2.tick_params(axis='y', labelsize=8)
    fig2.patch.set_facecolor('#0e1117')
    ax2.set_facecolor('#1e2130')
    ax2.tick_params(colors='white')
    ax2.xaxis.label.set_color('white')
    plt.tight_layout()
    st.pyplot(fig2)
    st.divider()

    # ── Section 6: Sample Records ─────────────────────────────────────────────
    st.markdown("### 📄 Sample Records")
    st.caption("Showing first 100 records from the cleaned dataset")
    st.dataframe(
        df[['title_en', 'title_fr', 'org', 'subject', 'keywords_en']].head(20),
        use_container_width=True,
        height=400
    )