import streamlit as st
import requests
import uuid
from typing import List

# API configuration
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="SmartDoc Bot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Sidebar
# Sidebar
with st.sidebar:
    st.title("📁 Document Management")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, DOCX)",
        type=["pdf", "docx", "doc"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if st.button("Upload Documents", type="primary"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                try:
                    files = [
                        ("files", (file.name, file.getvalue(), file.type))
                        for file in uploaded_files
                    ]
                    
                    response = requests.post(
                        f"{API_URL}/api/upload",
                        files=files
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {result['message']}")
                        st.session_state.uploaded_files.extend(result['file_ids'])
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Upload failed')}")
                        
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
        else:
            st.warning("Please select files to upload")
    
    # Display stats
    st.divider()
    try:
        health = requests.get(f"{API_URL}/api/health").json()
        doc_count = health.get("documents_count", 0)
        st.metric("Documents in DB", doc_count)
        st.metric("Status", health.get("status", "unknown"))
    except:
        st.error("⚠️ Cannot connect to API")
        doc_count = 0
    
    # Clear documents button with confirmation
    st.divider()
    
    # Initialize confirmation state
    if "confirm_clear_docs" not in st.session_state:
        st.session_state.confirm_clear_docs = False
    
    if doc_count > 0:
        if not st.session_state.confirm_clear_docs:
            if st.button("🗑️ Clear All Documents", type="secondary"):
                st.session_state.confirm_clear_docs = True
                st.rerun()
        else:
            st.warning("⚠️ This will delete ALL documents!")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("✅ Confirm", type="primary"):
                    with st.spinner("Deleting documents..."):
                        try:
                            response = requests.delete(f"{API_URL}/api/documents/clear-all")
                            
                            if response.status_code == 200:
                                result = response.json()
                                st.success(f"✅ {result['message']}")
                                st.session_state.uploaded_files = []
                                st.session_state.confirm_clear_docs = False
                                st.rerun()
                            else:
                                st.error(f"Error: {response.json().get('detail', 'Delete failed')}")
                                st.session_state.confirm_clear_docs = False
                                
                        except Exception as e:
                            st.error(f"Connection error: {str(e)}")
                            st.session_state.confirm_clear_docs = False
            
            with col2:
                if st.button("❌ Cancel"):
                    st.session_state.confirm_clear_docs = False
                    st.rerun()
    
    # Clear chat button
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()


# Main chat interface
st.title("🤖 SmartDoc Bot")
st.caption("Ask questions about your uploaded documents")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if available
        if message["role"] == "assistant" and "sources" in message:
            if message["sources"]:
                with st.expander("📚 Sources"):
                    for source in set(message["sources"]):
                        st.text(f"• {source}")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/api/chat",
                    json={
                        "query": prompt,
                        "session_id": st.session_state.session_id
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    
                    st.markdown(answer)
                    
                    # Display sources
                    if sources:
                        with st.expander("📚 Sources"):
                            for source in set(sources):
                                st.text(f"• {source}")
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    error_msg = response.json().get("detail", "Unknown error")
                    st.error(f"Error: {error_msg}")
                    
            except Exception as e:
                st.error(f"Connection error: {str(e)}")
