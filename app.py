"""
app.py — Streamlit UI for Legal GraphRAG
Run: streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="⚖️ Legal GraphRAG",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styles ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root { --accent: #4f46e5; }
.title  { text-align:center; font-size:2.4rem; font-weight:700; color:#1e1b4b; }
.sub    { text-align:center; color:#6b7280; margin-bottom:1.5rem; }
.badge-graph  { color:#16a34a; font-weight:700; }
.badge-vector { color:#2563eb; font-weight:700; }
.badge-hybrid { color:#9333ea; font-weight:700; }
.card {
    background:#f8fafc; border-left:4px solid #4f46e5;
    padding:.75rem 1rem; margin:.4rem 0; border-radius:6px;
    font-size:.9rem; line-height:1.5;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">⚖️ Legal GraphRAG</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub">Hybrid Knowledge Graph + Semantic Search · '
    'Indian Supreme Court Judgments</p>',
    unsafe_allow_html=True,
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Retrieval")
    g_k = st.slider("Graph Top-K",  1, 10, 5)
    v_k = st.slider("Vector Top-K", 1, 10, 5)
    f_k = st.slider("Final Top-K",  1, 10, 6)
    stream_on = st.toggle("Stream response", value=True)

    st.divider()

    st.subheader("📂 Ingest Judgments")
    uploaded = st.file_uploader(
        "Upload .txt judgment files",
        type=["txt"],
        accept_multiple_files=True,
        help="Files from Kaggle: vxrunsonii/supreme-court-judgments-txt",
    )
    if uploaded and st.button("🚀 Ingest", type="primary"):
        import tempfile, os
        from src.ingestion.data_loader import IngestionPipeline

        pipeline = IngestionPipeline()
        ok = 0
        prog = st.progress(0, text="Ingesting…")
        for i, f in enumerate(uploaded):
            try:
                with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="wb") as tmp:
                    tmp.write(f.read())
                    path = tmp.name
                pipeline._ingest_one(path)
                os.unlink(path)
                ok += 1
            except Exception as e:
                st.warning(f"{f.name}: {e}")
            prog.progress((i + 1) / len(uploaded))
        pipeline.close()
        prog.empty()
        st.success(f"✅ Ingested {ok}/{len(uploaded)} files")

    st.divider()

    st.subheader("📊 Graph Stats")
    if st.button("Refresh"):
        try:
            from src.graph.graph_queries  import GraphQueryEngine
            from src.vector.vector_store  import LegalVectorStore
            s = GraphQueryEngine().stats()
            st.metric("Cases",    s.get("cases",    0))
            st.metric("Judges",   s.get("judges",   0))
            st.metric("Statutes", s.get("statutes", 0))
            st.metric("Concepts", s.get("concepts", 0))
            st.metric("Chunks",   LegalVectorStore().count())
        except Exception as e:
            st.error(str(e))

# ─── Chat Interface ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("View Retrieved Sources"):
                for c in msg["sources"]:
                    st.markdown(f"**{c.get('case_name', 'Unknown')}** ({c.get('case_number', 'N/A')}) - Score: {c.get('score', 0):.3f}")
                    text_snippet = c.get('text', '')[:400] + ("..." if len(c.get('text', '')) > 400 else "")
                    st.text(text_snippet)

query = st.chat_input("e.g. What is the law on right to speedy trial under Article 21?")

# ─── Empty state ─────────────────────────────────────────────────────────────
if not st.session_state.messages and not query:
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 🔗 Knowledge Graph")
        st.write(
            "Neo4j graph stores Cases, Judges, Statutes and Legal Concepts "
            "as nodes connected by CITES, APPLIES, INVOLVES and DECIDED_BY edges."
        )
    with c2:
        st.markdown("### 🧠 Semantic Search")
        st.write(
            "ChromaDB stores sentence-transformer embeddings of every judgment chunk. "
            "Queries are matched by cosine similarity."
        )
    with c3:
        st.markdown("### ⚖️ Hybrid Fusion")
        st.write(
            "Reciprocal Rank Fusion (RRF) merges graph and vector results "
            "into a single ranked list fed to the LLM."
        )
    st.markdown("---")
    st.info(
        "**Getting started:**  \n"
        "1. Upload .txt judgment files in the sidebar → click **Ingest**  \n"
        "2. Type a legal research query in the chat box below.  \n"
        "3. Judgments from [Kaggle](https://www.kaggle.com/datasets/vxrunsonii/supreme-court-judgments-txt) work out of the box."
    )

# ─── Process Query ───────────────────────────────────────────────────────────
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    from config import config as cfg
    cfg.GRAPH_TOP_K  = g_k
    cfg.VECTOR_TOP_K = v_k
    cfg.FINAL_TOP_K  = f_k

    with st.chat_message("assistant"):
        try:
            from src.generation.llm_chain import LegalRAGChain
            chain = LegalRAGChain()
            
            history = st.session_state.messages[:-1]

            if stream_on:
                placeholder = st.empty()
                full_text = ""
                with st.spinner("Retrieving and generating…"):
                    for token in chain.stream_query(query, history=history):
                        full_text += token
                        placeholder.markdown(full_text + "▌")
                placeholder.markdown(full_text)
                
                from src.retrieval.hybrid_retriever import HybridRetriever
                result = HybridRetriever().retrieve(query)
                sources = [
                    {"case_name": c.case_name, "case_number": c.case_number, "score": c.score, "text": c.text}
                    for c in result.chunks
                ]
            else:
                with st.spinner("Retrieving and generating…"):
                    resp = chain.query(query, history=history)
                full_text = resp.answer
                st.markdown(full_text)
                sources = resp.sources
                st.caption(f"Graph hits: {resp.graph_hits} | Vector hits: {resp.vector_hits} | Model: {resp.model_used.split('-')[0]}")
            
            if sources:
                with st.expander("View Retrieved Sources"):
                    for c in sources:
                        st.markdown(f"**{c.get('case_name', 'Unknown')}** ({c.get('case_number', 'N/A')}) - Score: {c.get('score', 0):.3f}")
                        text_snippet = c.get('text', '')[:400] + ("..." if len(c.get('text', '')) > 400 else "")
                        st.text(text_snippet)

            st.session_state.messages.append({"role": "assistant", "content": full_text, "sources": sources})

        except Exception as e:
            st.error(f"Generation error: {e}")
            st.exception(e)
