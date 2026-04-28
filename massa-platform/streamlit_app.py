"""
MASSA Platform — Streamlit Dashboard

Three tabs:
    Chat      — multi-turn conversation with the financial agent
    Upload    — ingest PDF, DOCX, or XLSX documents
    Dashboard — live system health and recent agent activity

Run with:
    streamlit run streamlit_app.py

The backend must be running separately:
    uvicorn src.api.app:app --port 8000 --reload
"""

import json
import time

import requests
import streamlit as st

# ── Config ───────────────────────────────────────────────────────────────────

BACKEND = "http://localhost:8000"
PAGE_TITLE = "MASSA — Financial Intelligence"

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def backend_is_up() -> bool:
    """Quick liveness check against the health endpoint."""
    try:
        r = requests.get(f"{BACKEND}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def post_chat(message: str, history: list[dict]) -> dict | None:
    try:
        r = requests.post(
            f"{BACKEND}/chat",
            json={"message": message, "history": history},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        st.error("Request timed out. The agent may be processing a complex query.")
    except Exception as exc:
        st.error(f"Chat error: {exc}")
    return None


def post_ingest(file_bytes: bytes, filename: str) -> dict | None:
    try:
        r = requests.post(
            f"{BACKEND}/ingest",
            files={"file": (filename, file_bytes)},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        st.error("Ingestion timed out. Large files may take longer — try again.")
    except Exception as exc:
        st.error(f"Ingestion error: {exc}")
    return None


def get_health() -> dict | None:
    try:
        r = requests.get(f"{BACKEND}/health", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"Could not fetch health report: {exc}")
    return None


def get_logs(limit: int = 20) -> list[dict]:
    try:
        r = requests.get(f"{BACKEND}/logs", params={"limit": limit}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


# ── Backend status banner ─────────────────────────────────────────────────────

if not backend_is_up():
    st.error(
        "Backend is not reachable. "
        "Start it with: `uvicorn src.api.app:app --port 8000 --reload`"
    )
    st.stop()

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_chat, tab_upload, tab_dashboard = st.tabs(["Chat", "Upload Documents", "Dashboard"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ═══════════════════════════════════════════════════════════════════════════

with tab_chat:
    st.header("Financial Agent")
    st.caption("Ask questions about the documents and financial data you have ingested.")

    # Session state for conversation
    if "history" not in st.session_state:
        st.session_state.history = []       # [{role, content}, ...]
    if "display_turns" not in st.session_state:
        st.session_state.display_turns = [] # [{question, answer, tool_calls}, ...]

    # Render conversation history
    for turn in st.session_state.display_turns:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn.get("tool_calls"):
                with st.expander(f"Tools used ({len(turn['tool_calls'])})"):
                    for tc in turn["tool_calls"]:
                        st.markdown(f"**{tc['name']}**")
                        if tc.get("input"):
                            st.json(tc["input"])
                        if tc.get("result"):
                            st.text(tc["result"][:500] + ("..." if len(tc["result"]) > 500 else ""))

    # Input
    user_input = st.chat_input("Ask about revenue, EBITDA, margins, or anything in your documents...")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = post_chat(user_input, st.session_state.history)

            if result:
                st.write(result["response"])

                if result.get("tool_calls"):
                    with st.expander(f"Tools used ({len(result['tool_calls'])})"):
                        for tc in result["tool_calls"]:
                            st.markdown(f"**{tc['name']}**")
                            if tc.get("input"):
                                st.json(tc["input"])
                            if tc.get("result"):
                                st.text(tc["result"][:500] + ("..." if len(tc["result"]) > 500 else ""))

                # Update state
                st.session_state.history = result["history"]
                st.session_state.display_turns.append({
                    "question": user_input,
                    "answer": result["response"],
                    "tool_calls": result.get("tool_calls", []),
                })

    # Clear conversation button
    if st.session_state.display_turns:
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state.history = []
            st.session_state.display_turns = []
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — UPLOAD DOCUMENTS
# ═══════════════════════════════════════════════════════════════════════════

with tab_upload:
    st.header("Upload Documents")
    st.caption("Supported formats: PDF, DOCX, XLSX. Duplicate files are automatically skipped.")

    uploaded_files = st.file_uploader(
        "Choose one or more documents",
        type=["pdf", "docx", "xlsx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Ingest selected files", type="primary"):
            results = []
            progress = st.progress(0, text="Starting ingestion...")

            for i, f in enumerate(uploaded_files):
                progress.progress(
                    (i) / len(uploaded_files),
                    text=f"Processing {f.name}...",
                )
                file_bytes = f.read()
                result = post_ingest(file_bytes, f.name)
                if result:
                    results.append(result)

            progress.progress(1.0, text="Done.")
            time.sleep(0.5)
            progress.empty()

            if results:
                st.subheader("Ingestion Results")
                for r in results:
                    if r.get("skipped"):
                        st.warning(f"**{r['filename']}** — already ingested, skipped.")
                    else:
                        st.success(
                            f"**{r['filename']}** — "
                            f"{r['chunk_count']} chunks from {r['page_count']} pages."
                        )

    # Show what's already ingested
    st.divider()
    st.subheader("Ingested Documents")

    health = get_health()
    if health:
        col1, col2, col3 = st.columns(3)
        col1.metric("Documents", health.get("total_documents", 0))
        col2.metric("Chunks", health.get("total_chunks", 0))
        coverage = health.get("embedding_coverage", 0)
        col3.metric("Embedding coverage", f"{coverage * 100:.1f}%")

        if health.get("data_quality_issues"):
            for issue in health["data_quality_issues"]:
                st.warning(f"Data quality: {issue}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

with tab_dashboard:
    st.header("System Health Dashboard")

    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        refresh = st.button("Refresh", key="refresh_dashboard")

    health = get_health()

    if health:
        # ── Overall status ────────────────────────────────────────────────
        status = health.get("data_quality_pass", True)
        if status:
            st.success("System status: OK — all checks passing")
        else:
            st.error("System status: DEGRADED — data quality issues detected")
            for issue in health.get("data_quality_issues", []):
                st.warning(issue)

        st.divider()

        # ── Data layer ────────────────────────────────────────────────────
        st.subheader("Data Layer")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Documents", health.get("total_documents", 0))
        c2.metric("Chunks", health.get("total_chunks", 0))
        coverage = health.get("embedding_coverage", 0)
        c3.metric("Embedding coverage", f"{coverage * 100:.1f}%")
        c4.metric("Missing embeddings", health.get("missing_embeddings", 0))
        c5.metric("Empty chunks", health.get("empty_chunks", 0))

        dup = health.get("duplicate_chunks", 0)
        if dup > 0:
            st.warning(f"{dup} duplicate content hashes detected.")

        st.divider()

        # ── Agent activity (last 24 h) ────────────────────────────────────
        st.subheader("Agent Activity — Last 24 h")
        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("Total calls", health.get("total_calls_24h", 0))

        avg_lat = health.get("avg_latency_ms")
        a2.metric("Avg latency", f"{avg_lat:.0f} ms" if avg_lat else "—")

        avg_faith = health.get("avg_faithfulness_score")
        a3.metric("Avg faithfulness", f"{avg_faith * 100:.1f}%" if avg_faith is not None else "—")

        halluc = health.get("hallucination_rate")
        a4.metric("Hallucination rate", f"{halluc * 100:.1f}%" if halluc is not None else "—")

        avg_out = health.get("avg_output_tokens")
        a5.metric("Avg output tokens", f"{avg_out:.0f}" if avg_out else "—")

        st.divider()

    # ── Recent interactions ───────────────────────────────────────────────
    st.subheader("Recent Interactions")
    log_limit = st.slider("Show last N interactions", min_value=5, max_value=100, value=20, step=5)
    logs = get_logs(limit=log_limit)

    if logs:
        # Build a display-friendly table
        rows = []
        for log in logs:
            rows.append({
                "Time": log.get("logged_at", "")[:19].replace("T", " "),
                "Question": log.get("question", "")[:80] + ("..." if len(log.get("question", "")) > 80 else ""),
                "Tools": ", ".join(log.get("tools_called") or []) or "—",
                "Latency (ms)": log.get("latency_ms", ""),
                "In tokens": log.get("input_tokens", ""),
                "Out tokens": log.get("output_tokens", ""),
                "Faithfulness": (
                    f"{log['faithfulness_score']:.2f}"
                    if log.get("faithfulness_score") is not None else "—"
                ),
                "Hallucination": (
                    "Yes" if log.get("hallucination_detected") is True
                    else "No" if log.get("hallucination_detected") is False
                    else "—"
                ),
            })

        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No interactions logged yet. Ask the agent a question in the Chat tab.")
