# app.py
import os
import tempfile
import streamlit as st
from streamlit_chat import message
from rag import ChatPDF

# Set page configuration
st.set_page_config(page_title="PDF Assistant")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "assistant" not in st.session_state:
    st.session_state["assistant"] = ChatPDF()

if "k" not in st.session_state:
    st.session_state["k"] = 5

if "score_threshold" not in st.session_state:
    st.session_state["score_threshold"] = 0.1

def display_messages():
    """Display the chat history."""
    st.subheader("Chat History")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    """Process the user input."""
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_input = st.session_state["user_input"].strip()
        st.session_state["messages"].append((user_input, True))
        
        with st.session_state["thinking_spinner"], st.spinner("Thinking..."):
            response = st.session_state["assistant"].ask(
                user_input,
                k=st.session_state["k"],
                score_threshold=st.session_state["score_threshold"]
            )
            
        st.session_state["messages"].append((response, False))
        st.session_state["user_input"] = ""

# Sidebar with information and controls
with st.sidebar:
    st.title("About")
    st.markdown("""
    ### How it works
    1. Upload a PDF document
    2. Ask questions about the document
    3. Get answers using Llama2
    
    ### Technical Details
    This app uses:
    - Llama2 (7B) for text generation
    - FAISS for vector storage
    - Ollama for model serving
    
    ### Requirements
    - Ollama must be running locally
    - Llama2 model must be pulled (`ollama pull llama2:7b`)
    """)
    
    st.divider()
    st.subheader("Advanced Settings")
    
    # Number of retrieved documents
    st.session_state["k"] = st.slider(
        "Number of retrieved documents (k)",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of document chunks to retrieve for each question"
    )
    
    # Score threshold
    st.session_state["score_threshold"] = st.slider(
        "Similarity score threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="Minimum similarity score for retrieved documents (lower = more results but potentially less relevant)"
    )

# Main interface
st.title("💬 Chat with your PDF")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Create a temporary file to store the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    
    try:
        with st.spinner("Processing PDF..."):
            st.session_state["assistant"].ingest(tmp_file_path)
        st.success("PDF processed successfully!")
        
        # Display chat interface
        display_messages()
        
        # Chat input
        st.text_input("Ask a question about your PDF:", key="user_input", on_change=process_input)
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
    finally:
        # Clean up the temporary file
        os.unlink(tmp_file_path)
else:
    st.info("👆 Upload a PDF file to get started!")

# Add a clear chat button
if st.button("Clear Chat"):
    st.session_state["messages"] = []
    st.session_state["assistant"].clear()
    st.experimental_rerun()