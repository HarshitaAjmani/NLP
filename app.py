import streamlit as st
import pandas as pd
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import os

from tabs import search, dataset, evaluation, workflow, training, data_quality

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Search Demo",
    "Project Workflow",
    "Dataset Info",
    "Data Quality",
    "Training Summary",
    "Evaluation Metrics"
])

#Setting up the Streamlit app configuration.

st.set_page_config(
    page_title= "Semantic Search - Canada Open Data",
    layout="wide",
    initial_sidebar_state="collapsed"
)

#Adding custom CSS styles for the search results display.

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
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
        .footer {
            text-align: center;
            color: #555555;
            font-size: 12px;
            padding: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

#Initializing the Hugging Face repository and model.=
hf_repo = "harshuajmani/nlp_geoCanada_embeddings"
model = "intfloat/multilingual-e5-base"

#Loading the SentenceTransformer model and FAISS indexes with caching to optimize performance.
@st.cache_resource(show_spinner=False)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model, device=device)


# Loading the SentenceTransformer model and FAISS indexes with caching.
@st.cache_resource(show_spinner=False)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model=model, device=device)

@st.cache_resource(show_spinner=False)
def load_data():
    index_en_path = hf_hub_download(repo_id=hf_repo, filename="index_en.faiss", repo_type="dataset")
    index_fr_path = hf_hub_download(repo_id=hf_repo, filename="index_fr.faiss", repo_type="dataset")
    csv_path      = hf_hub_download(repo_id=hf_repo, filename="index_data.csv", repo_type="dataset")
    index_en      = faiss.read_index(index_en_path)
    index_fr      = faiss.read_index(index_fr_path)
    df            = pd.read_csv(csv_path).fillna("")
    return df, index_en, index_fr

with st.spinner("Loading model and search index..."):
    model = load_model()
    df, index_en, index_fr = load_data()

# Header
st.markdown("""
    <div style='margin-bottom: 8px;'>
        <span style='font-size: 24px; font-weight: bold;'>🗺️ Semantic Search - Canada Open Data</span>
        <span style='font-size: 13px; color: #888888; margin-left: 12px;'>
            Semantic search over 46,468 Canadian government geospatial datasets · EN + FR
        </span>
    </div>
""", unsafe_allow_html=True)

st.divider()

# Navigation tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Search Demo",
    "Project Workflow",
    "Dataset Info",
    "Data Quality",
    "Training Summary",
    "Evaluation Metrics"
])

# Rendering each tab
with tab1: search.render(model, df, index_en, index_fr)
with tab2: workflow.render()
with tab3: dataset.render()
with tab4: data_quality.render()
with tab5: training.render()
with tab6: evaluation.render()

# Footer
st.divider()
st.caption("multilingual-e5-base · FAISS vector search · 46,468 datasets · open.canada.ca")