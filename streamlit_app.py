"""
Run with:  streamlit run streamlit_app.py
"""

import uuid
import requests
import streamlit as st


API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Documentation Assistant",
    page_icon="📚",
    layout="wide",
)

# -- Compact sidebar styling --
st.markdown(
    """
    <style>
        /* Tighten sidebar padding */
        section[data-testid="stSidebar"] > div { padding-top: 1rem; }
        section[data-testid="stSidebar"] .block-container { padding: 0; }

        /* Reduce gaps between sidebar elements */
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {
            gap: 0.35rem !important;
        }

        /* Smaller dividers */
        section[data-testid="stSidebar"] hr {
            margin: 0.4rem 0 !important;
        }

        /* Compact metrics */
        section[data-testid="stSidebar"] [data-testid="stMetric"] {
            padding: 0.3rem 0 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stMetric"] label {
            font-size: 0.75rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 RAG Documentation Assistant")
st.caption("Ask questions about technical documentation — powered by LangGraph")


# ---- Session State ----

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---- Sidebar ----

with st.sidebar:
    # -- Branding --
    st.markdown(
        """
        <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
            <span style="font-size: 2rem;">📚</span>
            <h3 style="margin: 0.25rem 0 0 0; font-weight: 700;">RAG Assistant</h3>
            <p style="margin: 0; font-size: 0.8rem; opacity: 0.6;">Documentation Q&A System</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # -- System Status (compact) --
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        status = health["status"]
        provider = health["llm_provider"].upper()

        status_color = "#00c853" if status == "healthy" else "#ff5252"
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; justify-content:space-between;
                        background: rgba(255,255,255,0.05); border-radius: 8px;
                        padding: 0.6rem 0.8rem; margin-bottom: 0.5rem;">
                <div style="display:flex; align-items:center; gap:0.5rem;">
                    <span style="display:inline-block; width:8px; height:8px;
                                 border-radius:50%; background:{status_color};"></span>
                    <span style="font-size:0.85rem; font-weight:600;">{status.capitalize()}</span>
                </div>
                <span style="font-size:0.7rem; opacity:0.5; background:rgba(255,255,255,0.08);
                             padding: 2px 8px; border-radius: 4px;">{provider}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        st.error("⚠ Server not reachable")

    st.divider()

    # -- Indexed Documents --
    st.markdown("##### 📄 Indexed Documents")

    try:
        response = requests.get(f"{API_URL}/documents", timeout=5)
        if response.status_code == 200:
            data = response.json()

            # Summary metrics row
            col1, col2 = st.columns(2)
            col1.metric("Documents", data["total_documents"])
            col2.metric("Chunks", data["total_chunks"])

            # Document cards
            for doc in data["documents"]:
                name = doc["source"].replace(".md", "").replace("_", " ").title()
                chunks = doc["chunk_count"]
                pct = int((chunks / max(data["total_chunks"], 1)) * 100)

                st.markdown(
                    f"""
                    <div style="background:rgba(255,255,255,0.04); border-radius:6px;
                                padding:0.5rem 0.7rem; margin-bottom:0.4rem;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:0.82rem; font-weight:500;">📝 {name}</span>
                            <span style="font-size:0.72rem; opacity:0.5;">{chunks} chunks</span>
                        </div>
                        <div style="background:rgba(255,255,255,0.08); border-radius:3px;
                                    height:4px; margin-top:0.35rem; overflow:hidden;">
                            <div style="background:linear-gradient(90deg, #667eea, #764ba2);
                                        height:100%; width:{pct}%; border-radius:3px;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.warning("Could not fetch documents")
    except requests.ConnectionError:
        st.error("⚠ Cannot connect to API. Start the server first.")

    st.divider()

    # -- Add Documents --
    st.markdown("##### ➕ Add Documents")
    new_url = st.text_input("Paste a documentation URL:", placeholder="https://docs.example.com/page")
    if st.button("🔗 Ingest URL", use_container_width=True) and new_url:
        try:
            with st.spinner("Fetching & indexing..."):
                resp = requests.post(
                    f"{API_URL}/ingest",
                    json={"urls": [new_url]},
                    timeout=60,
                )
            if resp.status_code == 201:
                st.success(resp.json()["message"])
                st.rerun()
            else:
                st.error(f"Error: {resp.text}")
        except Exception as e:
            st.error(f"Failed: {e}")

    st.divider()

    # -- Session info (subtle) --
    st.markdown(
        f"<p style='font-size:0.7rem; opacity:0.35; text-align:center;'>"
        f"Session: {st.session_state.session_id[:8]}…</p>",
        unsafe_allow_html=True,
    )


# ---- Chat Interface ----

# Show chat history
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show extra info for assistant messages
        if msg["role"] == "assistant":
            meta = msg.get("metadata", {})
            if meta:
                with st.expander("📋 Details"):
                    col1, col2, col3 = st.columns(3)
                    col1.write(f"**Query Type:** {meta.get('query_type', 'N/A')}")
                    col2.write(f"**Retries:** {meta.get('retry_count', 0)}")
                    col3.write(f"**Web Search:** {'Yes' if meta.get('web_search_used') else 'No'}")

                    if meta.get("rewritten_query"):
                        st.write(f"**Rewritten Query:** {meta['rewritten_query']}")

                    if meta.get("citations"):
                        st.write("**Sources:**")
                        for cite in meta["citations"]:
                            st.write(f"• {cite['source']}")

            # Feedback buttons
            feedback = msg.get("feedback")
            question = ""
            if idx > 0 and st.session_state.messages[idx - 1]["role"] == "user":
                question = st.session_state.messages[idx - 1]["content"]

            if feedback:
                val = "👍" if feedback == "thumbs_up" else "👎"
                st.markdown(
                    f"""
                    <div style="display:inline-flex; align-items:center; gap:0.4rem;
                                background:rgba(255,255,255,0.05); padding:0.2rem 0.6rem;
                                border-radius:12px; margin-top:0.4rem;">
                        <span style="font-size:0.8rem; opacity:0.6;">Recorded feedback:</span>
                        <span style="font-size:0.9rem;">{val}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                col_up, col_down, _ = st.columns([0.05, 0.05, 0.9])
                with col_up:
                    if st.button("👍", key=f"up_{idx}"):
                        try:
                            resp = requests.post(
                                f"{API_URL}/feedback",
                                json={
                                    "session_id": st.session_state.session_id,
                                    "question": question,
                                    "rating": "thumbs_up",
                                },
                                timeout=5,
                            )
                            if resp.status_code == 200:
                                msg["feedback"] = "thumbs_up"
                                st.toast("✅ Feedback recorded! Thank you.")
                                st.rerun()
                            else:
                                st.error("Failed to record feedback")
                        except Exception as e:
                            st.error(f"Could not send feedback: {e}")
                with col_down:
                    if st.button("👎", key=f"down_{idx}"):
                        try:
                            resp = requests.post(
                                f"{API_URL}/feedback",
                                json={
                                    "session_id": st.session_state.session_id,
                                    "question": question,
                                    "rating": "thumbs_down",
                                },
                                timeout=5,
                            )
                            if resp.status_code == 200:
                                msg["feedback"] = "thumbs_down"
                                st.toast("✅ Feedback recorded! Thank you.")
                                st.rerun()
                            else:
                                st.error("Failed to record feedback")
                        except Exception as e:
                            st.error(f"Could not send feedback: {e}")

# Chat input
user_input = st.chat_input("Ask a question about the docs...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get answer from API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={
                        "question": user_input,
                        "session_id": st.session_state.session_id,
                    },
                    timeout=120,
                )

                if response.status_code == 200:
                    data = response.json()

                    # Save to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "feedback": None,
                        "metadata": {
                            "query_type": data.get("query_type"),
                            "rewritten_query": data.get("rewritten_query"),
                            "web_search_used": data.get("web_search_used"),
                            "retry_count": data.get("retry_count"),
                            "citations": data.get("citations", []),
                        },
                    })
                    st.rerun()

                else:
                    error_msg = response.json().get("detail", "Unknown error")
                    st.error(f"Error: {error_msg}")

            except requests.ConnectionError:
                st.error("Cannot connect to the API. Is the server running?")
            except Exception as e:
                st.error(f"Something went wrong: {e}")
