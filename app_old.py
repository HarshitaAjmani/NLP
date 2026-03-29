import streamlit as st
import pandas as pd
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import os

#Setting up the Streamlit app configuration.

st.set_page_config(
    page_title= "Semantic Search - Canada Open Data",
    layout="wide"
)


#Adding custom CSS styles for the search results display.

st.markdown("""
    <style>
        .result-card {
            background-color: #1e2130;
            border-left: 4px solid #4CAF50;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        }
        .score-badge {
            color: white;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 13px;
            font-weight: bold;
        }
        .rank-badge {
            background-color: #2196F3;
            color: white;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 13px;
        }
        .title-text {
            font-size: 16px;
            font-weight: bold;
            color: #ffffff;
        }
        .desc-text {
            font-size: 13px;
            color: #aaaaaa;
            margin-top: 8px;
        }
        .org-text {
            font-size: 12px;
            color: #888888;
            margin-top: 8px;
        }
    </style>
""", unsafe_allow_html=True)


#Initializing the Hugging Face repository and model.

hf_repo = "harshuajmani/nlp_geoCanada_embeddings"
model = "intfloat/multilingual-e5-base"


#Loading the SentenceTransformer model and FAISS indexes with caching to optimize performance.

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model, device=device)



@st.cache_resource
def load_data():
    with st.spinner("⬇️ Downloading data from HuggingFace (first time only)..."):

        # Download files
        index_en_path  = hf_hub_download(repo_id=hf_repo, filename="index_en.faiss",   repo_type="dataset")
        index_fr_path  = hf_hub_download(repo_id=hf_repo, filename="index_fr.faiss",   repo_type="dataset")
        csv_path       = hf_hub_download(repo_id=hf_repo, filename="index_data.csv",   repo_type="dataset")

        # Load FAISS indexes directly
        index_en = faiss.read_index(index_en_path)
        index_fr = faiss.read_index(index_fr_path)

        # Load metadata
        df = pd.read_csv(csv_path).fillna("")

    return df, index_en, index_fr


#Defining the search function that takes a user query, encodes it, and retrieves the top-k relevant results from the appropriate FAISS index.'''

def search(query, search_idx, top_k=5):
    vec   = model.encode(f"query: {query}")
    vec   = vec / np.linalg.norm(vec)
    vec   = vec.astype('float32').reshape(1, -1)
    index = index_en if search_idx == 'en' else index_fr
    scores, indices = index.search(vec, top_k)
    return indices[0].tolist(), scores[0].tolist()


model = load_model()
df, index_en, index_fr = load_data()


#Creating the user interface with Streamlit, 
#including input fields for the search query, 
#language selection, and number of results to display.

col1, col2, col3 = st.columns([5, 1, 1])

with col1:
    query = st.text_input(
        "Search",
        placeholder="e.g. water quality monitoring Ontario...",
        label_visibility="collapsed"
    )
with col2:
    lang = st.selectbox(
        "Language",
        options=["en", "fr"],
        format_func=lambda x: "EN English" if x == "en" else "🇫🇷 French",
        label_visibility="collapsed"
    )
with col3:
    top_k = st.selectbox(
        "Results",
        options=[5, 10, 15],
        label_visibility="collapsed"
    )

#When the user submits a query, 
#perform the search and display the results in a styled format, 
#including the title, description, organization, subject, and a relevance score badge.

if query:
    with st.spinner("🔍 Searching..."):
        indices, scores = search(query, lang, top_k)

    st.markdown(f"**{len(indices)} results** for *'{query}'* in **{'English' if lang == 'en' else 'French'}** index")
    st.divider()

    for rank, (idx, score) in enumerate(zip(indices, scores), 1):
        row = df.iloc[idx]

        score_color = (
            "#4CAF50" if score >= 0.85 else
            "#FF9800" if score >= 0.75 else
            "#F44336"
        )

        st.markdown(f"""
            <div class="result-card">
                <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                    <span class="rank-badge">#{rank}</span>
                    <span class="score-badge" style="background-color:{score_color}">
                        Score: {score:.4f}
                    </span>
                </div>
                <div class="title-text">🇬🇧 {row['title_en']}</div>
                <div class="title-text" style="color:#aaaaaa; font-size:14px;">
                    🇫🇷 {row['title_fr']}
                </div>
                <div class="desc-text">{str(row['desc_en'])[:250]}...</div>
                <div class="org-text">
                    🏛️ {row['org']} &nbsp;|&nbsp; 📂 {row['subject']}
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.progress(float(score))

else:
    st.markdown("""
        <div style="text-align:center; padding:60px; color:#555555;">
            <h3>🔍 Enter a query to search</h3>
            <p>Try: "water quality", "forest fire", "climate change Canada"</p>
            <p>Or in French: "qualité de l'eau", "incendies de forêt"</p>
        </div>
    """, unsafe_allow_html=True)


#Adding a metrics summary section at the bottom of the app to display key information about the dataset, model, and languages used.

st.divider()
st.caption("multilingual-e5-base · FAISS vector search · 46,468 datasets · open.canada.ca")
col1, col2, col3 = st.columns(3)
col1.metric("Total Datasets", "46,468")
col2.metric("Model", "multilingual-e5-base")
col3.metric("Languages", "EN + FR")