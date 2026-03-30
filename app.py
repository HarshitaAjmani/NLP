import streamlit as st
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

from tabs import search, dataset, evaluation, workflow, training, data_quality

#Setting up the Streamlit app configuration.
st.set_page_config(
    page_title="Semantic Search - Canada Open Data",
    layout="wide",
    initial_sidebar_state="collapsed"
)

#Adding custom CSS styles for the search results display.──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700&family=DM+Sans:wght@300;400;600&display=swap');

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }
.stApp { background: #0e0e10; font-family: 'DM Sans', sans-serif; color: #fff; }

/* ── st.tabs: restyle the tab bar ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0e0e10;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    gap: 1rem;
    padding: 0 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: rgba(255,255,255,0.38);
    font-family: 'DM Sans', sans-serif;
    font-size: 0.875rem;
    font-weight: 400;
    border: none;
    padding: 0.75rem 0.75rem;
}
.stTabs [aria-selected="true"] {
    background: transparent;
    color: #ffffff;
    font-weight: 600;
}
/* Replace default blue underline with purple */
.stTabs [data-baseweb="tab-highlight"] {
    background: #7c6af7;
    height: 2px;
}
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ── Result cards ── */
.result-card {
    background-color: #1a1a1f;
    border-left: 3px solid #7c6af7;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 10px;
}
.score-badge { color: white; padding: 2px 10px; border-radius: 12px; font-size: 13px; font-weight: bold; }
.rank-badge  { background-color: #2196F3; color: white; padding: 2px 10px; border-radius: 12px; font-size: 13px; }
.title-text  { font-size: 15px; font-weight: 600; color: #ffffff; }
.desc-text   { font-size: 13px; color: #aaaaaa; margin-top: 6px; }
.org-text    { font-size: 12px; color: #666666; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

# Page heading using native st.columns
col_title, col_badge = st.columns([6, 1])

with col_title:
    st.markdown(
        "<span style='font-family:Syne,sans-serif; font-size:2.0rem; font-weight:700; letter-spacing:-0.02em;'>"
        "<span style='color:#7c6af7'>Semantic</span>"
        "<span style='color:#ffffff'>Search</span>"
        "</span>",
        unsafe_allow_html=True
    )
    st.caption(" Dataset Discovery Tool for Canada Open Data Portal")

with col_badge:
    st.markdown(
        "<div style='display:flex; justify-content:flex-end; padding-top:0.4rem;'>"
        "<a href='https://huggingface.co/intfloat/multilingual-e5-base' target='_blank' style='text-decoration:none;'>"
        "<span style='border:1px solid rgba(124,106,247,0.4); background:rgba(124,106,247,0.1); "
        "color:#a89ef9; padding:0.35rem 0.9rem; border-radius:999px; font-size:1.1rem;'>"
        "● HuggingFace model</span></a></div>",
        unsafe_allow_html=True
    )


#Initializing the Hugging Face repository and model.=
hf_repo = "harshuajmani/nlp_geoCanada_embeddings"
model_name = "intfloat/multilingual-e5-base"

#Loading the SentenceTransformer model and FAISS indexes with caching to optimize performance.
@st.cache_resource(show_spinner=False)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)

@st.cache_resource(show_spinner=False)
def load_data():
    index_en_path = hf_hub_download(repo_id=hf_repo, filename="index_en.faiss", repo_type="dataset")
    index_fr_path = hf_hub_download(repo_id=hf_repo, filename="index_fr.faiss", repo_type="dataset")
    csv_path      = hf_hub_download(repo_id=hf_repo, filename="index_data.csv",  repo_type="dataset")
    index_en      = faiss.read_index(index_en_path)
    index_fr      = faiss.read_index(index_fr_path)
    df            = pd.read_csv(csv_path).fillna("")
    return df, index_en, index_fr

with st.spinner("Loading model and search index…"):
    _model = load_model()
    df, index_en, index_fr = load_data()


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Search Demo",
    "Project Workflow",
    "Dataset Info",
    "Data Quality",
    "Training Summary",
    "Evaluation Metrics"
])

# Rendering each tab

with tab1: search.render(_model, df, index_en, index_fr)
with tab2: workflow.render()
with tab3: dataset.render()
with tab4: data_quality.render()
with tab5: training.render()
with tab6: evaluation.render()



# Footer
st.divider()
 
col_name, col_github = st.columns([6, 1])
 
with col_name:
    st.markdown(
        "<span style='color:rgba(255,255,255,255); font-size:1.1rem;'>Harshita Ajmani</span>",
        unsafe_allow_html=True
    )
 
with col_github:
    st.markdown(
        """
        <div style='display:flex; justify-content:flex-end;'>
          <a href='https://github.com/HarshitaAjmani/Semantic-search-Canada-open-data'
             target='_blank'
             style='display:inline-flex; align-items:center; gap:0.4rem;
                    border:1px solid rgba(255,255,255,255);
                    background:rgba(255,255,255,0.05);
                    color:rgba(255,255,255,0.45);
                    padding:0.32rem 0.8rem; border-radius:999px;
                    font-size:1.1rem; text-decoration:none;
                    transition: all 0.15s;'>
            <svg width='13' height='13' viewBox='0 0 24 24' fill='currentColor'
                 xmlns='http://www.w3.org/2000/svg'>
              <path d='M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57
                       0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41
                       -1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815
                       2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925
                       0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23
                       .96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65
                       .24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925
                       .435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57
                       A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z'/>
            </svg>
            GitHub
          </a>
        </div>
        """,
        unsafe_allow_html=True
    )