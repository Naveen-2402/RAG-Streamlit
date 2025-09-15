# src/app_streamlit.py
# Streamlit UI wrapper around rag_logic.py (same core logic), with numbered stepper,
# insights panel, and a stable chat UI. Background colors use Streamlit defaults.

import os
import io
import json
import time
import streamlit as st

from config import (
    get_paths,
    PROJECT_ROOT,
    check_credentials,
    CONTEXT_K,
    # Optional: if available, these will be displayed in Insights
    DENSE_EMBEDDING_MODEL,
    RERANKER_MODEL,
)
from rag_logic import (
    build_index_stepwise,
    delete_document_artifacts,
    answer_with_citations,
)

# ------------ UI helpers ------------
def draw_stepper(current_step: int, steps=None):
    steps = steps or ["Upload & Start", "Analyzing PDF", "Planning Chunks", "Segmenting Content", "Creating Index"]
    cols = st.columns(len(steps))
    for i, c in enumerate(cols):
        with c:
            label = f"Step {i+1}. {steps[i]}"
            if i < current_step:
                st.markdown(f"[done] **{label}**")
            elif i == current_step:
                st.markdown(f"[running] **{label}**")
            else:
                st.markdown(f"[ ] {label}")

def file_bytes(path: str):
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None

def ensure_dirs():
    os.makedirs(os.path.join(PROJECT_ROOT, "input"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "output"), exist_ok=True)

def count_pages_from_raw_md(raw_md_path: str):
    try:
        with open(raw_md_path, "r", encoding="utf-8") as f:
            s = f.read()
        pb = s.count("<!-- PageBreak -->")
        return pb + 1 if pb > 0 else None
    except Exception:
        return None

def simplify_hits(hits):
    out = []
    for h in hits or []:
        out.append({
            "chunk_id": h.get("chunk_id"),
            "score": round(float(h.get("score", 0)), 3),
            "breadcrumbs": h.get("breadcrumbs", ""),
            "preview": h.get("preview", "")[:400] + ("..." if len(h.get("preview", "")) > 400 else "")
        })
    return out

def render_history():
    for i, turn in enumerate(st.session_state.chat):
        with st.chat_message(turn["role"]):
            # If we stored top hits for this assistant turn, show them before the answer (collapsed)
            if turn["role"] == "assistant" and turn.get("debug", {}).get("hits"):
                with st.expander("Top hits used"):
                    for h in turn["debug"]["hits"]:
                        st.markdown(
                            f"- #{h['chunk_id']} · score={h['score']:.3f} · {h['breadcrumbs']}\n\n"
                            f"```text\n{h['preview']}\n```"
                        )
            # Then show the message content (answer or user text)
            st.markdown(turn["content"])
            if turn["role"] == "assistant" and turn.get("citations"):
                st.caption(f"Citations: {', '.join(str(c) for c in turn['citations'])}")

def compute_basic_stats_from_paths(paths: dict):
    stats = {}
    # Headings count
    try:
        if os.path.exists(paths.get("refined_md", "")):
            with open(paths["refined_md"], "r", encoding="utf-8") as f:
                stats["total_headings"] = sum(1 for line in f if line.lstrip().startswith("#"))
    except Exception:
        pass

    # Chunk count
    try:
        chunks_dir = paths.get("chunks_dir", "")
        if chunks_dir and os.path.isdir(chunks_dir):
            stats["chunk_count"] = len([f for f in os.listdir(chunks_dir) if f.startswith("chunk_") and f.endswith(".md")])
    except Exception:
        pass

    # Validation from chunk plan JSON, if present
    try:
        if os.path.exists(paths.get("chunk_plan", "")):
            with open(paths["chunk_plan"], "r", encoding="utf-8") as f:
                plan = json.load(f)
            stats["validation"] = (plan.get("validation", {}) or {}).get("status", "")
            if "document_title" in plan:
                stats["document_title"] = plan["document_title"]
    except Exception:
        pass

    # Approx pages (from raw MD pagebreaks)
    pages = count_pages_from_raw_md(paths.get("raw_md", ""))
    if pages:
        stats["pages"] = pages

    # Optional model names from config (if available)
    try:
        stats["embedding_model"] = DENSE_EMBEDDING_MODEL
    except Exception:
        pass
    try:
        stats["reranker_model"] = RERANKER_MODEL
    except Exception:
        pass

    return stats

# ------------ App State ------------
if "pdf_base_name" not in st.session_state:
    st.session_state.pdf_base_name = None
if "index" not in st.session_state:
    st.session_state.index = None
if "paths" not in st.session_state:
    st.session_state.paths = {}
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of dicts: {"role": "user"/"assistant", "content": str, "citations": [...], "debug": {...}}
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = None

# ------------ Sidebar ------------
st.sidebar.title("Controls")
with st.sidebar.expander("Credentials check", expanded=False):
    try:
        ok = check_credentials()
        st.write("Loaded" if ok else "Missing values – check .env")
    except Exception:
        st.write("Error checking credentials")

reset_clicked = st.sidebar.button("Delete & Reset", type="primary")
if reset_clicked:
    base = st.session_state.get("pdf_base_name")
    if base:
        delete_document_artifacts(base)
    st.session_state.clear()
    st.rerun()

# ------------ Main ------------
st.title("Document Q&A (Streamlit)")

ensure_dirs()

STEPS = ["Upload & Start", "Analyzing PDF", "Planning Chunks", "Segmenting Content", "Creating Index"]

# Stage 0: Upload screen
if st.session_state.index is None:

    st.markdown("Step 1: Upload your PDF")
    draw_stepper(0, STEPS)

    uploaded = st.file_uploader("Drag & drop a PDF here", type=["pdf"])

    base_name_input = None
    if uploaded is not None:
        default_base = os.path.splitext(uploaded.name)[0]
        base_name_input = st.text_input("Base name (used for outputs)", value=default_base)

        if st.button("Start Processing"):
            if not base_name_input.strip():
                st.error("Please provide a base name.")
            else:
                base = base_name_input.strip()
                st.session_state.pdf_base_name = base
                paths = get_paths(base)
                st.session_state.paths = paths

                # save uploaded file into input/
                in_path = paths["pdf_input"]
                with open(in_path, "wb") as f:
                    f.write(uploaded.getbuffer())

                # Stepper + pipeline
                step_placeholder = st.empty()
                note_placeholder = st.empty()

                # Show Step 2 explicitly with a default-themed info note
                step_placeholder.empty()
                draw_stepper(1, STEPS)
                note_placeholder.info(
                    "⚡ Step 2: Analyzing your PDF and finding the best chunking strategy. ⏳ This may take a few seconds..."
                )


                with st.spinner("Processing..."):
                    result = build_index_stepwise(base)

                # Accept both (idx, paths) and (idx, paths, stats)
                idx, paths_out, stats = None, paths, {}
                try:
                    if isinstance(result, (list, tuple)) and len(result) == 3:
                        idx, paths_out, stats = result
                    elif isinstance(result, (list, tuple)) and len(result) == 2:
                        idx, paths_out = result
                        stats = compute_basic_stats_from_paths(paths_out)
                    else:
                        idx = None
                except Exception:
                    idx = None

                if idx is None:
                    st.error("Pipeline failed. Check console logs.")
                    st.stop()

                st.session_state.index = idx
                st.session_state.paths = paths_out

                # Add or augment insights
                if not stats:
                    stats = compute_basic_stats_from_paths(paths_out)
                else:
                    # ensure pages present if possible
                    pages = count_pages_from_raw_md(paths_out.get("raw_md"))
                    if pages:
                        stats["pages"] = pages
                st.session_state.doc_stats = stats

                # Completed
                step_placeholder.empty()
                note_placeholder.empty()
                draw_stepper(5, STEPS)
                st.success("Document is ready for Q&A.")

                # Insights panel
                st.markdown("### Document Insights")
                cols = st.columns(5)
                with cols[0]: st.metric("Headings", st.session_state.doc_stats.get("total_headings", "-"))
                with cols[1]: st.metric("Chunks", st.session_state.doc_stats.get("chunk_count", "-"))
                with cols[2]: st.metric("Validation", (st.session_state.doc_stats.get("validation", "") or "").capitalize())
                with cols[3]: st.metric("Embedding", st.session_state.doc_stats.get("embedding_model", "-"))
                with cols[4]: st.metric("Reranker", st.session_state.doc_stats.get("reranker_model", "-"))
                if "pages" in st.session_state.doc_stats:
                    st.caption(f"Approx. pages: {st.session_state.doc_stats['pages']}")

                st.rerun()

    st.info("Place a PDF into input/ via this uploader. The app will run OCR, refine headings, chunk, index, then open chat.")

# Stage 1: Chat UI
else:
    paths = st.session_state.paths
    base = st.session_state.pdf_base_name
    idx = st.session_state.index

    st.markdown("### Downloads")
    colA, colB, colC, colD = st.columns(4)
    with colA:
        if os.path.exists(paths.get("raw_md", "")):
            st.download_button("Raw MD", data=file_bytes(paths["raw_md"]), file_name=os.path.basename(paths["raw_md"]))
    with colB:
        if os.path.exists(paths.get("refined_md", "")):
            st.download_button("Refined MD", data=file_bytes(paths["refined_md"]), file_name=os.path.basename(paths["refined_md"]))
    with colC:
        if os.path.exists(paths.get("chunk_plan", "")):
            st.download_button("Chunk Plan (JSON)", data=file_bytes(paths["chunk_plan"]), file_name=os.path.basename(paths["chunk_plan"]))
    with colD:
        if os.path.exists(paths.get("metadata", "")):
            st.download_button("Metadata (JSON)", data=file_bytes(paths["metadata"]))

    if st.session_state.doc_stats:
        st.divider()
        st.markdown("### Document Insights")
        stats = st.session_state.doc_stats
        cols = st.columns(5)
        with cols[0]: st.metric("Headings", stats.get("total_headings", "-"))
        with cols[1]: st.metric("Chunks", stats.get("chunk_count", "-"))
        with cols[2]: st.metric("Validation", (stats.get("validation", "") or "").capitalize())
        with cols[3]: st.metric("Embedding", stats.get("embedding_model", "-"))
        with cols[4]: st.metric("Pages", stats.get("pages", "-"))

    st.divider()
    st.markdown("### Chat")

    # Render existing history (stable; uses default Streamlit backgrounds)
    render_history()

    if prompt := st.chat_input("Ask a question about the document..."):
        # 1) show user message immediately & persist
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2) Do retrieval + answer and display both Top hits and Answer immediately
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and composing..."):
                # retrieval
                hits = []
                try:
                    hits = idx.retrieve(prompt, k=5)
                except Exception as e:
                    st.error(f"Retrieval failed: {e}")

                # show top hits first (live view)
                if hits:
                    st.markdown("**Top hits**")
                    for h in hits:
                        st.markdown(
                            f"- #{h['chunk_id']} · score={float(h.get('score',0)):.3f} · {h.get('breadcrumbs','')}\n\n"
                            f"```text\n{h.get('preview','')}\n```"
                        )
                else:
                    st.info("No strong hits found.")

                # answer
                try:
                    ans = answer_with_citations(idx, prompt, k_ctx=CONTEXT_K)
                    cits = ans.get("citations", [])
                    # show answer immediately (live view)
                    st.markdown(ans.get("answer", ""))

                    # persist assistant turn with hits for stable re-render
                    st.session_state.chat.append({
                        "role": "assistant",
                        "content": ans.get("answer", ""),
                        "citations": cits,
                        "debug": {"hits": simplify_hits(hits)}
                    })
                except Exception as e:
                    msg = f"Answer generation failed: {e}"
                    st.markdown(msg)
                    st.session_state.chat.append({
                        "role": "assistant",
                        "content": msg,
                        "citations": [],
                        "debug": {"hits": simplify_hits(hits)}
                    })

        # 3) Re-render in stable mode so the conversation persists cleanly
        st.rerun()
