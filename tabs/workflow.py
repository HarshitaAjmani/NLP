# tabs/workflow.py
import streamlit as st
import pandas as pd

def render():

    # ── Fade-in CSS animation ─────────────────────────────────────────────────
    st.markdown("""
    <style>
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(30px); }
            to   { opacity: 1; transform: translateY(0);    }
        }
        .fade-1  { animation: fadeInUp 0.5s ease forwards; animation-delay: 0.1s; opacity: 0; }
        .fade-2  { animation: fadeInUp 0.5s ease forwards; animation-delay: 0.3s; opacity: 0; }
        .fade-3  { animation: fadeInUp 0.5s ease forwards; animation-delay: 0.5s; opacity: 0; }
        .fade-4  { animation: fadeInUp 0.5s ease forwards; animation-delay: 0.7s; opacity: 0; }
        .fade-5  { animation: fadeInUp 0.5s ease forwards; animation-delay: 0.9s; opacity: 0; }
        .fade-6  { animation: fadeInUp 0.5s ease forwards; animation-delay: 1.1s; opacity: 0; }
        .fade-7  { animation: fadeInUp 0.5s ease forwards; animation-delay: 1.3s; opacity: 0; }
        .fade-8  { animation: fadeInUp 0.5s ease forwards; animation-delay: 1.5s; opacity: 0; }

        .timeline-step {
            background-color: #1e2130;
            border-left: 4px solid #4CAF50;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 8px;
        }
        .step-number {
            background-color: #4CAF50;
            color: white;
            border-radius: 50%;
            width: 28px;
            height: 28px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 14px;
            margin-right: 10px;
        }
        .step-title {
            font-size: 16px;
            font-weight: bold;
            color: #ffffff;
            display: inline;
        }
        .step-subtitle {
            font-size: 12px;
            color: #888888;
            margin-left: 38px;
        }
        .tech-card {
            background-color: #1e2130;
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 6px;
            border-left: 3px solid #2196F3;
        }
        .flow-box {
            background-color: #1e2130;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 12px;
            text-align: center;
            color: white;
            font-size: 13px;
        }
        .flow-arrow {
            text-align: center;
            font-size: 20px;
            color: #4CAF50;
            margin: 2px 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── Section 1: Project Timeline ───────────────────────────────────────────
    st.markdown("""
    <div class="fade-1">
        <h3>🗓️ Project Timeline</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="fade-2">
        <div class="timeline-step">
            <span class="step-number">1</span>
            <span class="step-title">Data Collection</span>
            <div class="step-subtitle">Download Canada Open Data Catalog · 46,468 records · JSONL format</div>
        </div>
    </div>
    <div class="fade-3">
        <div class="timeline-step">
            <span class="step-number">2</span>
            <span class="step-title">Data Cleaning</span>
            <div class="step-subtitle">Extract bilingual fields · Remove noise · Save cleaned CSV</div>
        </div>
    </div>
    <div class="fade-4">
        <div class="timeline-step">
            <span class="step-number">3</span>
            <span class="step-title">Embedding Generation</span>
            <div class="step-subtitle">multilingual-e5-base · 46,468 × 768 vectors · GPU accelerated</div>
        </div>
    </div>
    <div class="fade-5">
        <div class="timeline-step">
            <span class="step-number">4</span>
            <span class="step-title">Search Index</span>
            <div class="step-subtitle">FAISS IndexFlatIP · Cosine similarity · EN + FR indexes</div>
        </div>
    </div>
    <div class="fade-6">
        <div class="timeline-step">
            <span class="step-number">5</span>
            <span class="step-title">Evaluation</span>
            <div class="step-subtitle">100 queries · 4 metrics · Bilingual + cross-lingual testing</div>
        </div>
    </div>
    <div class="fade-7">
        <div class="timeline-step">
            <span class="step-number">6</span>
            <span class="step-title">Deployment</span>
            <div class="step-subtitle">HuggingFace Hub · Streamlit Cloud · Live demo</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Section 2: Step by Step Breakdown ────────────────────────────────────
    st.markdown("""
    <div class="fade-8"><h3>⚙️ Step by Step Breakdown</h3></div>
    """, unsafe_allow_html=True)

    with st.expander("📥 Step 1 — Data Collection", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Source**
            - Official Canada Open Data Portal
            - URL: `open.canada.ca/static/od-do-canada.jsonl.gz`
            - Format: JSON Lines (one record per line)
            - Size: ~500MB compressed

            **What each record contains**
            - 65+ fields per record
            - Bilingual metadata (EN + FR)
            - Government dataset descriptions
            - Keywords, organizations, subjects
            """)
        with col2:
            st.markdown("""
            **Download code**
```python
            import json

            records = []
            with open("od-do-canada.jsonl", "r",
                      encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))

            # Result: 46,468 records loaded
```
            """)

    with st.expander("🧹 Step 2 — Data Cleaning", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Fields extracted**
            - `title_translated` → EN + FR titles
            - `notes_translated` → EN + FR descriptions
            - `keywords`         → EN + FR keywords
            - `organization`     → Publishing department
            - `subject`          → Topic category

            **Cleaning applied**
            - Removed markdown links from descriptions
            - Dropped records with empty EN + FR descriptions
            - Filled missing descriptions with title text
            - Saved as `cleaned_data.csv`
            """)
        with col2:
            st.markdown("""
            **Why `notes_translated` not `notes`?**
```
            notes field:
              Plain string — English only
              "This dataset contains..."

            notes_translated field:
              Bilingual dict ✅
              {
                "en": "This dataset...",
                "fr": "Ce jeu de données..."
              }
```
            The `_translated` suffix signals
            bilingual content in this dataset.
            """)

    with st.expander("🤖 Step 3 — Embedding Generation", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Model:** `intfloat/multilingual-e5-base`

            **The passage prefix trick**
```python
            # Documents need "passage: " prefix
            text = f"passage: {title}. {desc} {keywords}"

            # Queries need "query: " prefix
            query = f"query: {user_input}"
```

            **Why prefixes matter**
            Without them the model treats queries
            and documents identically — accuracy drops.
            With them it knows the intent:
            - `passage:` = encode this document
            - `query:`   = find similar documents
            """)
        with col2:
            st.markdown("""
            **Generation stats**
```
            Device    : NVIDIA RTX 5070 Laptop
            Batch size: 64
            EN time   : 242.6 seconds
            FR time   : 233.3 seconds

            Output shape: (46468, 768)
            → 46,468 records
            → 768 dimensions each

            Saved as float16 for efficiency:
            142MB → 71MB per file
```
            """)

    with st.expander("🔍 Step 4 — Search Index (FAISS)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **What is FAISS?**
            Facebook AI Similarity Search —
            finds nearest vectors in milliseconds
            across 46,468 records.

            **Why IndexFlatIP?**
```
            IndexFlatIP = Inner Product index

            On normalized vectors:
            Inner Product = Cosine Similarity

            We normalize embeddings → use IP
            → fast cosine similarity search
```

            **Two separate indexes**
            - `index_en.faiss` → English descriptions
            - `index_fr.faiss` → French descriptions
            """)
        with col2:
            st.markdown("""
            **Search pipeline**
```python
            # 1. Encode user query
            vec = model.encode("query: water quality")

            # 2. Normalize
            vec = vec / np.linalg.norm(vec)

            # 3. Search FAISS
            scores, indices = index.search(vec, k=5)

            # 4. Return top 5 results
            # Takes: ~5ms for 46,468 records ⚡
```

            **Pre-built index advantage**
```
            Raw embeddings → build index at runtime
            = 10-15 seconds startup

            Pre-built index → load directly
            = instant startup ✅
```
            """)

    with st.expander("📊 Step 5 — Evaluation", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **100 queries across 4 types**
```
            EN → EN  : 25 queries (monolingual)
            FR → FR  : 25 queries (monolingual)
            EN → FR  : 25 queries (cross-lingual)
            FR → EN  : 25 queries (cross-lingual)
```

            **Query generation**
            - Sampled records with titles ≥ 5 words
            - Extracted 4 meaningful words (no stop words)
            - Used as search queries
            - True answer = the source record
            """)
        with col2:
            st.markdown("""
            **4 metrics measured**
```
            Metric 1: Avg similarity score
                      → Are results relevant?

            Metric 2: EN vs FR gap
                      → Are both languages equal?

            Metric 3: Cross-lingual consistency
                      → Do EN/FR searches agree?

            Metric 4: Score distribution (std)
                      → Are scores consistent?
```
            """)

    with st.expander("🚀 Step 6 — Deployment", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **The deployment challenge**
```
            Problem:
            Embeddings = 142MB each
            GitHub limit = 100MB
            → Cannot push to GitHub ❌

            Solution:
            Upload to HuggingFace Hub ✅
            App downloads on first run
            Streamlit caches forever after
```

            **Optimizations applied**
            - float32 → float16 (142MB → 71MB)
            - Pre-built FAISS indexes saved
            - `@st.cache_resource` for zero reload
            """)
        with col2:
            st.markdown("""
            **Deployment stack**
```
            Code hosting:
              GitHub repository

            Large files hosting:
              HuggingFace Hub (free)
              harshuajmani/nlp_geoCanada_embeddings

            App hosting:
              Streamlit Cloud (free)
              Auto-deploys on GitHub push

            First load  : ~2-3 mins (download)
            After that  : instant (cached) ✅
```
            """)

    st.divider()

    # ── Section 3: Data Flow Diagram ──────────────────────────────────────────
    st.markdown("### 🔄 Data Flow Diagram")

    col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)

    with col1:
        st.markdown("""
        <div class="flow-box">
            📥<br>Raw JSONL<br>
            <small>46,468 records<br>500MB</small>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="flow-arrow">→</div>',
                    unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="flow-box">
            🧹<br>Cleaned CSV<br>
            <small>8 fields<br>bilingual</small>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="flow-arrow">→</div>',
                    unsafe_allow_html=True)
    with col5:
        st.markdown("""
        <div class="flow-box">
            🤖<br>Embeddings<br>
            <small>46468×768<br>float16</small>
        </div>
        """, unsafe_allow_html=True)
    with col6:
        st.markdown('<div class="flow-arrow">→</div>',
                    unsafe_allow_html=True)
    with col7:
        st.markdown("""
        <div class="flow-box">
            🔍<br>FAISS Index<br>
            <small>EN + FR<br>142MB each</small>
        </div>
        """, unsafe_allow_html=True)
    with col8:
        st.markdown('<div class="flow-arrow">→</div>',
                    unsafe_allow_html=True)
    with col9:
        st.markdown("""
        <div class="flow-box">
            🎯<br>Search Results<br>
            <small>Top-K<br>~5ms</small>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Section 4: Tech Stack ─────────────────────────────────────────────────
    st.markdown("### 🛠️ Tech Stack")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Core Libraries**")
        st.table(pd.DataFrame({
            "Library":          [
                "sentence-transformers",
                "faiss-cpu",
                "numpy",
                "pandas",
                "torch",
                "huggingface-hub"
            ],
            "Purpose": [
                "Load and run multilingual-e5-base",
                "Vector similarity search",
                "Vector operations and normalization",
                "Data loading and cleaning",
                "GPU acceleration for embeddings",
                "Upload/download large files"
            ]
        }))

    with col2:
        st.markdown("**Infrastructure**")
        st.table(pd.DataFrame({
            "Tool":          [
                "Streamlit",
                "HuggingFace Hub",
                "Streamlit Cloud",
                "GitHub",
                "NVIDIA RTX 5070"
            ],
            "Purpose": [
                "Web UI framework",
                "Large file storage (free)",
                "App deployment (free)",
                "Code hosting + CI/CD",
                "GPU for embedding generation"
            ]
        }))

    st.divider()

    # ── Section 5: File Structure ─────────────────────────────────────────────
    st.markdown("### 📁 Project File Structure")

    st.markdown("""
```
    NLP/
    ├── app.py                    ← Main Streamlit app
    │
    ├── tabs/                     ← One file per UI tab
    │   ├── __init__.py
    │   ├── search.py             ← Search Demo tab
    │   ├── dataset.py            ← Dataset Info tab
    │   ├── evaluation.py         ← Evaluation Metrics tab
    │   ├── workflow.py           ← Project Workflow tab
    │   ├── model_tab.py          ← Model Summary tab
    │   └── data_quality.py       ← Data Quality tab
    │
    ├── src/                      ← Data pipeline scripts
    │   ├── load_data.py          ← Download + clean dataset
    │   ├── embeddings.py         ← Generate embeddings
    │   ├── search.py             ← Search function + CLI
    │   ├── evaluate.py           ← Evaluation metrics
    │   └── upload_to_hf.py       ← Upload to HuggingFace
    │
    ├── data/                     ← Local data (not in GitHub)
    │   ├── od-do-canada.jsonl    ← Raw dataset
    │   ├── cleaned_data.csv      ← Cleaned records
    │   ├── embeddings_en.npy     ← English vectors
    │   ├── embeddings_fr.npy     ← French vectors
    │   ├── index_data.csv        ← Lookup table
    │   └── evaluation_results.*  ← Eval outputs
    │
    ├── requirements.txt          ← Python dependencies
    └── README.md                 ← Project documentation
```
    """)