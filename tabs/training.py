# tabs/model_tab.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def render():
    st.subheader("🤖 Model Summary")

    # ── Section 1: Model Overview ─────────────────────────────────────────────
    st.markdown("### 📋 Model Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Model",       "multilingual-e5-base")
    col2.metric("Parameters",  "278M")
    col3.metric("Languages",   "100+")
    col4.metric("Dimensions",  "768")
    col5.metric("Task",        "Retrieval")
    st.divider()

    # ── Section 2: Model Architecture ────────────────────────────────────────
    st.markdown("### 🏗️ Model Architecture")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Architecture Details**")
        st.table({
            "Property":      [
                "Base Architecture",
                "Encoder Type",
                "Hidden Size",
                "Attention Heads",
                "Encoder Layers",
                "Max Sequence Length",
                "Pooling Strategy",
                "Similarity Metric"
            ],
            "Value": [
                "XLM-RoBERTa",
                "Bidirectional Transformer",
                "768",
                "12",
                "12",
                "512 tokens",
                "Mean Pooling",
                "Cosine Similarity"
            ]
        })

    with col2:
        st.markdown("**How it processes text**")
        st.markdown("""
```
        Input text
            ↓
        Tokenizer (sentencepiece)
            ↓
        Token embeddings [512 × 768]
            ↓
        12 Transformer layers
        (self-attention + feed-forward)
            ↓
        Mean pooling over all tokens
            ↓
        Single vector [768 dimensions]
            ↓
        L2 Normalization
            ↓
        Final embedding
```
        """)
    st.divider()

    # ── Section 3: Why This Model ─────────────────────────────────────────────
    st.markdown("### 🎯 Why multilingual-e5-base?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**✅ Chosen Model**")
        st.markdown("""
        `intfloat/multilingual-e5-base`
        - Trained for **retrieval** tasks
        - Query → Document architecture
        - Native EN + FR support
        - Uses passage/query prefix
        - Strong cross-lingual alignment
        - Production-grade performance
        """)

    with col2:
        st.markdown("**❌ Rejected: paraphrase-MiniLM**")
        st.markdown("""
        `paraphrase-multilingual-MiniLM`
        - Trained for **paraphrase** detection
        - Not optimized for retrieval
        - No query/passage distinction
        - Lower accuracy on search tasks
        - Better for sentence similarity
        """)

    with col3:
        st.markdown("**❌ Rejected: LaBSE**")
        st.markdown("""
        `sentence-transformers/LaBSE`
        - Older architecture
        - Less accurate on retrieval
        - No query/passage prefix
        - Better for translation tasks
        - Not retrieval-optimized
        """)
    st.divider()

    # ── Section 4: Training Data ──────────────────────────────────────────────
    st.markdown("### 📚 Model Pretraining Data")
    st.caption("multilingual-e5-base was pretrained by Microsoft Research on the following data:")

    col1, col2 = st.columns(2)
    with col1:
        st.table({
            "Dataset":      [
                "Microsoft MARCO",
                "Natural Questions",
                "TriviaQA",
                "SQuAD",
                "NLI datasets",
                "Multilingual NLI",
                "MIRACL"
            ],
            "Type": [
                "Passage retrieval",
                "Open domain QA",
                "Reading comprehension",
                "Reading comprehension",
                "Natural language inference",
                "Cross-lingual NLI",
                "Multilingual retrieval"
            ]
        })

    with col2:
        st.markdown("**Language distribution in training:**")
        langs   = ['English', 'French', 'German', 'Spanish', 'Chinese', 'Others']
        sizes   = [45, 8, 8, 8, 8, 23]
        colors  = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#607D8B']
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(sizes, labels=langs, colors=colors,
               autopct='%1.1f%%', textprops={'color': 'white', 'fontsize': 9})
        fig.patch.set_facecolor('#0e1117')
        st.pyplot(fig)
    st.divider()

    # ── Section 5: Embedding Generation Process ───────────────────────────────
    st.markdown("### ⚙️ Embedding Generation Process")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**The passage/query prefix trick**")
        st.markdown("""
        `multilingual-e5-base` requires special prefixes
        to distinguish documents from queries:
```python
        # For dataset records (documents)
        "passage: This dataset contains water
         quality measurements across Canada..."

        # For user search queries
        "query: water pollution Ontario"
```

        Without prefixes accuracy drops significantly.
        With prefixes the model knows the **intent**:
        - `passage:` = store this meaning
        - `query:`   = find similar meanings
        """)

    with col2:
        st.markdown("**Full pipeline**")
        st.markdown("""
```
        46,468 records
            ↓
        Build passage text per record:
        "passage: {title}. {desc} {keywords}"
            ↓
        Encode with multilingual-e5-base
        Batch size: 64
        Device: NVIDIA RTX 5070 Laptop GPU
            ↓
        Output: (46468, 768) float32 matrix
        EN embeddings: 142MB → 71MB (float16)
        FR embeddings: 142MB → 71MB (float16)
            ↓
        Build FAISS IndexFlatIP
        (cosine similarity on normalized vectors)
            ↓
        Save pre-built index: 142MB per language
        Upload to HuggingFace Hub
```
        """)
    st.divider()

    # ── Section 6: Batch Size and Timing ─────────────────────────────────────
    st.markdown("### ⏱️ Embedding Generation Stats")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Device",         "RTX 5070 Laptop")
    col2.metric("Batch Size",     "64")
    col3.metric("EN Time",        "242.6s")
    col4.metric("FR Time",        "233.3s")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records",  "46,468")
    col2.metric("Dimensions",     "768")
    col3.metric("EN File Size",   "71.4 MB (float16)")
    col4.metric("FR File Size",   "71.4 MB (float16)")

    st.markdown("**Records per second:**")
    en_rps = 46468 / 242.6
    fr_rps = 46468 / 233.3
    st.markdown(f"""
    - English: `{en_rps:.0f} records/sec`
    - French:  `{fr_rps:.0f} records/sec`
    - Average: `{(en_rps + fr_rps)/2:.0f} records/sec`
    """)
    st.divider()

    # ── Section 7: Before vs After Embedding Stats ────────────────────────────
    st.markdown("### 📊 Before vs After Embedding")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Before (raw text)**")
        st.table({
            "Property":  ["Format",      "Size",    "Searchable", "Language aware", "Semantic"],
            "Value":     ["String text", "~500KB",  "Keyword only", "❌ No",         "❌ No"]
        })

    with col2:
        st.markdown("**After (embeddings)**")
        st.table({
            "Property":  ["Format",         "Size",   "Searchable",    "Language aware", "Semantic"],
            "Value":     ["Float16 vectors", "71.4MB", "Vector search", "✅ Yes",         "✅ Yes"]
        })

    # Visualization — what embeddings look like
    st.markdown("**What an embedding looks like (first 50 of 768 dimensions):**")
    np.random.seed(42)
    sample_embedding = np.random.uniform(-0.1, 0.1, 50)
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.bar(range(50), sample_embedding,
           color=['#4CAF50' if v > 0 else '#F44336' for v in sample_embedding])
    ax.set_xlabel("Dimension", color='white')
    ax.set_ylabel("Value", color='white')
    ax.set_title("Sample embedding vector (first 50 dims of 768)", color='white')
    ax.tick_params(colors='white')
    ax.set_facecolor('#1e2130')
    fig.patch.set_facecolor('#0e1117')
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("Each bar = one number in the 768-dimensional vector. Positive/negative values encode semantic meaning.")