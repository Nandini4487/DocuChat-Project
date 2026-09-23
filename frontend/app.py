"""DocuChat Streamlit Web Application - Presentation Layer."""

import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="DocuChat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []

if "has_documents" not in st.session_state:
    st.session_state.has_documents = False

# 3. Sidebar: Document Upload & Management
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
        # In Task 8 (placeholder mode): simulate document processing state
        if uploaded_file.name not in st.session_state.documents:
            st.session_state.documents.append(uploaded_file.name)
        st.session_state.has_documents = True
        st.success(f"Processed '{uploaded_file.name}' (Layout placeholder mode).")

    # Static progress indicator placeholder for Task 9
    st.subheader("Upload Progress")
    st.progress(0, text="Ready (Task 9 placeholder)")

    # Uploaded Documents Section
    st.subheader("Indexed Documents")
    if not st.session_state.documents:
        st.info("No documents uploaded yet.")
    else:
        for doc in st.session_state.documents:
            st.markdown(f"- 📄 **{doc}**")

# 4. Main Chat Area
st.title("📄 DocuChat")
st.caption("Grounded Question Answering powered by RAG and Google Gemini")

# Render message history
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

# Chat Input at the bottom
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

    # 2. Append and render placeholder assistant response (Task 8 scope)
    assistant_placeholder = (
        "This is a placeholder response — backend not connected yet."
    )
    st.session_state.messages.append({"role": "assistant", "content": assistant_placeholder})
    with chat_container:
        with st.chat_message("assistant"):
            st.markdown(assistant_placeholder)
