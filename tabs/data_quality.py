# tabs/data_quality.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
    st.subheader("✅ Data Quality Report")

    with st.spinner("Loading dataset..."):
        df = load_df()

    # ── Section 1: Overall Quality Score ─────────────────────────────────────
    st.markdown("### 🏆 Overall Quality Score")

    en_coverage      = (df['desc_en']     != '').mean() * 100
    fr_coverage      = (df['desc_fr']     != '').mean() * 100
    keyword_coverage = (df['keywords_en'] != '').mean() * 100
    bilingual        = ((df['title_en'] != '') & (df['title_fr'] != '')).mean() * 100
    overall_score    = round((en_coverage + fr_coverage + keyword_coverage + bilingual) / 4, 1)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Overall Score",       f"{overall_score}%")
    col2.metric("EN Description",      f"{en_coverage:.1f}%")
    col3.metric("FR Description",      f"{fr_coverage:.1f}%")
    col4.metric("Keyword Coverage",    f"{keyword_coverage:.1f}%")
    col5.metric("Bilingual Records",   f"{bilingual:.1f}%")
    st.divider()

    # ── Section 2: Missing Values ─────────────────────────────────────────────
    st.markdown("### 🔍 Missing Values per Field")

    fields  = ['title_en', 'title_fr', 'desc_en', 'desc_fr', 'keywords_en', 'keywords_fr', 'org', 'subject']
    missing = [(df[f] == '').sum() for f in fields]
    pct     = [round((df[f] == '').mean() * 100, 2) for f in fields]

    missing_df = pd.DataFrame({
        "Field":        fields,
        "Missing":      missing,
        "Missing %":    pct,
        "Present":      [len(df) - m for m in missing],
        "Present %":    [round(100 - p, 2) for p in pct],
        "Status":       ["✅" if p < 5 else "⚠️" if p < 20 else "❌" for p in pct]
    })
    st.dataframe(missing_df, use_container_width=True)
    st.divider()

    # ── Section 3: Bilingual Completeness ────────────────────────────────────
    st.markdown("### 🌐 Bilingual Completeness")

    both_titles = ((df['title_en'] != '') & (df['title_fr'] != '')).sum()
    both_desc   = ((df['desc_en']  != '') & (df['desc_fr']  != '')).sum()
    both_kw     = ((df['keywords_en'] != '') & (df['keywords_fr'] != '')).sum()
    en_only     = ((df['title_en'] != '') & (df['title_fr'] == '')).sum()
    fr_only     = ((df['title_en'] == '') & (df['title_fr'] != '')).sum()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Bilingual field pairs**")
        st.table(pd.DataFrame({
            "Field Pair":      ["Title EN + FR", "Desc EN + FR", "Keywords EN + FR"],
            "Both Present":    [f"{both_titles:,}", f"{both_desc:,}", f"{both_kw:,}"],
            "Coverage":        [
                f"{both_titles/len(df)*100:.1f}%",
                f"{both_desc/len(df)*100:.1f}%",
                f"{both_kw/len(df)*100:.1f}%"
            ]
        }))

    with col2:
        st.markdown("**Title language distribution**")
        fig, ax = plt.subplots(figsize=(5, 4))
        labels  = ['Both EN+FR', 'EN only', 'FR only']
        values  = [both_titles, en_only, fr_only]
        colors  = ['#4CAF50', '#2196F3', '#FF9800']
        ax.pie(values, labels=labels, colors=colors,
               autopct='%1.1f%%', textprops={'color': 'white'})
        fig.patch.set_facecolor('#0e1117')
        st.pyplot(fig)
    st.divider()

    # ── Section 4: Text Length Distribution ──────────────────────────────────
    st.markdown("### 📏 Description Length Distribution")
    st.caption("How long are the descriptions? Longer = more searchable content")

    df['len_en'] = df['desc_en'].str.len()
    df['len_fr'] = df['desc_fr'].str.len()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**English Description Length**")
        st.table(pd.DataFrame({
            "Stat":  ["Min", "Max", "Mean", "Median"],
            "Value": [
                f"{df['len_en'].min():,} chars",
                f"{df['len_en'].max():,} chars",
                f"{df['len_en'].mean():,.0f} chars",
                f"{df['len_en'].median():,.0f} chars",
            ]
        }))

    with col2:
        st.markdown("**French Description Length**")
        st.table(pd.DataFrame({
            "Stat":  ["Min", "Max", "Mean", "Median"],
            "Value": [
                f"{df['len_fr'].min():,} chars",
                f"{df['len_fr'].max():,} chars",
                f"{df['len_fr'].mean():,.0f} chars",
                f"{df['len_fr'].median():,.0f} chars",
            ]
        }))

    # Length histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    axes[0].hist(df['len_en'].clip(0, 2000), bins=30, color='#2196F3', edgecolor='none')
    axes[0].set_title("EN Description Length", color='white')
    axes[0].set_xlabel("Characters", color='white')
    axes[0].tick_params(colors='white')
    axes[0].set_facecolor('#1e2130')

    axes[1].hist(df['len_fr'].clip(0, 2000), bins=30, color='#4CAF50', edgecolor='none')
    axes[1].set_title("FR Description Length", color='white')
    axes[1].set_xlabel("Characters", color='white')
    axes[1].tick_params(colors='white')
    axes[1].set_facecolor('#1e2130')

    fig.patch.set_facecolor('#0e1117')
    plt.tight_layout()
    st.pyplot(fig)
    st.divider()

    # ── Section 5: Duplicate Check ────────────────────────────────────────────
    st.markdown("### 🔁 Duplicate Check")

    dup_titles = df['title_en'].duplicated().sum()
    dup_ids    = df['id'].duplicated().sum() if 'id' in df.columns else 0
    identical  = (df['title_en'] == df['title_fr']).sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Duplicate EN Titles", f"{dup_titles:,}",
                help="Records sharing identical English titles")
    col2.metric("Duplicate IDs",       f"{dup_ids:,}",
                help="Records sharing identical IDs")
    col3.metric("Identical EN=FR Title", f"{identical:,}",
                help="Records where EN and FR title are the same — may indicate missing translation")

    if dup_titles > 0:
        st.warning(f"⚠️ {dup_titles:,} duplicate English titles found — may affect search ranking")
    else:
        st.success("✅ No duplicate titles found")
    st.divider()

    # ── Section 6: Cleaning Steps Applied ────────────────────────────────────
    st.markdown("### 🧹 Cleaning Steps Applied")
    st.markdown("""
    | Step | Action | Impact |
    |------|--------|--------|
    | 1 | Extracted `title_translated` and `notes_translated` bilingual fields | Correct EN/FR separation |
    | 2 | Removed markdown links from descriptions | Cleaner searchable text |
    | 3 | Dropped records with empty EN + FR descriptions | Removed ~3,000 records |
    | 4 | Filled missing descriptions with title text | Zero empty descriptions |
    | 5 | Filtered titles < 5 words for evaluation | Cleaner test queries |
    | 6 | Removed stop words for query generation | More meaningful queries |
    | 7 | Filtered identical EN/FR titles for evaluation | Avoids ambiguous matches |
    """)