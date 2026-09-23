"""DocuChat Streamlit Web Application - Presentation Layer with FastAPI Backend Integration."""

import os
from typing import Any, Dict, List, Set, Tuple
import requests
import streamlit as st

# 1. Configuration
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

# 2. Page Configuration
st.set_page_config(
    page_title="DocuChat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 3. Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []

if "has_documents" not in st.session_state:
    st.session_state.has_documents = False


import ast


def format_answer_text(raw_answer: Any) -> str:
    """Format answer text cleanly handling string or structured content blocks."""
    if isinstance(raw_answer, str):
        stripped = raw_answer.strip()
        if stripped.startswith("[{") and stripped.endswith("}]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    return format_answer_text(parsed)
            except Exception:
                pass
        return raw_answer
    if isinstance(raw_answer, list):
        text_parts = []
        for item in raw_answer:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(str(item["text"]))
            else:
                text_parts.append(str(item))
        return "\n".join(text_parts).strip()
    return str(raw_answer)


def deduplicate_sources(raw_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate source citations by (file, page) tuple client-side."""
    unique: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, int]] = set()

    for src in raw_sources:
        file_name = src.get("file")
        page_num = src.get("page")
        if file_name is not None and page_num is not None:
            key = (str(file_name), int(page_num))
            if key not in seen:
                seen.add(key)
                unique.append({"file": file_name, "page": page_num})
    return unique


# 4. Sidebar: Document Upload & Management
with st.sidebar:
    st.header("Upload a document")
    st.caption("Upload text-searchable PDF documents to ask questions.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF file (max 20 MB).",
    )

    process_button = st.button(
        "Process document",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

    if process_button and uploaded_file is not None:
        file_name = uploaded_file.name

        # Client-side duplicate check
        if file_name in st.session_state.documents:
            st.warning(f"'{file_name}' is already uploaded and indexed.")
        else:
            with st.spinner(f"Uploading and processing '{file_name}'..."):
                try:
                    files = {
                        "file": (
                            file_name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    }
                    response = requests.post(
                        f"{API_URL}/upload",
                        files=files,
                        timeout=60,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.documents.append(file_name)
                        st.session_state.has_documents = True
                        chunks_count = data.get("chunks_created", 0)
                        st.success(
                            f"Successfully processed **{file_name}** ({chunks_count} chunks indexed)."
                        )
                    else:
                        # Extract error detail from response JSON if present
                        try:
                            error_detail = response.json().get("detail", response.text)
                        except Exception:
                            error_detail = response.text or f"HTTP {response.status_code}"
                        st.error(f"Upload failed: {error_detail}")

                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                    st.error("Could not reach the backend — is it running?")
                except Exception as exc:
                    st.error(f"An unexpected error occurred: {str(exc)}")

    # Uploaded Documents Section
    st.subheader("Indexed Documents")
    if not st.session_state.documents:
        st.info("No documents uploaded yet.")
    else:
        for doc in st.session_state.documents:
            st.markdown(f"- 📄 **{doc}**")

# 5. Main Chat Area
st.title("📄 DocuChat")
st.caption("Grounded Question Answering powered by RAG and Google Gemini")

# Render existing conversation history
chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        if not st.session_state.has_documents:
            st.info("👈 Please upload and process a PDF document in the sidebar to start chatting.")
        else:
            st.write("Ask any question about your uploaded document below.")
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                sources = msg.get("sources", [])
                if sources:
                    with st.expander("Sources"):
                        for src in sources:
                            st.markdown(f"📄 {src['file']}, Page {src['page']}")

# Chat Input at bottom
input_placeholder = (
    "Ask a question about your documents..."
    if st.session_state.has_documents
    else "Upload a document first..."
)

user_input = st.chat_input(
    placeholder=input_placeholder,
    disabled=not st.session_state.has_documents,
)

if user_input:
    # 1. Append and render user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)

    # 2. Call backend /ask endpoint and render response
    with chat_container:
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                try:
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={"question": user_input},
                        timeout=60,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        answer_text = format_answer_text(data.get("answer", ""))
                        raw_sources = data.get("sources", [])
                        unique_sources = deduplicate_sources(raw_sources)

                        # If refusal response, do not display misleading sources
                        if answer_text.strip() == "I don't know based on the provided documents.":
                            unique_sources = []

                        st.markdown(answer_text)
                        if unique_sources:
                            with st.expander("Sources"):
                                for src in unique_sources:
                                    st.markdown(f"📄 {src['file']}, Page {src['page']}")

                        # Save turn to session state with citations
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer_text,
                            "sources": unique_sources,
                        })

                    else:
                        try:
                            error_detail = response.json().get("detail", response.text)
                        except Exception:
                            error_detail = response.text or f"HTTP {response.status_code}"
                        error_msg = f"⚠️ Backend error: {error_detail}"
                        st.markdown(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "sources": [],
                        })

                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                    conn_error_msg = "⚠️ Could not reach the backend — is it running?"
                    st.markdown(conn_error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": conn_error_msg,
                        "sources": [],
                    })
                except Exception as exc:
                    unexp_error_msg = f"⚠️ An unexpected error occurred: {str(exc)}"
                    st.markdown(unexp_error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": unexp_error_msg,
                        "sources": [],
                    })
