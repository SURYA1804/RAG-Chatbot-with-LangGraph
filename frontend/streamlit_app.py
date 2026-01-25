import streamlit as st
import uuid
from typing import List
import io
import os
import sys
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.graph import create_rag_agent
from backend.utils import process_document
from backend.vectore_store import VectorStoreManager

# ✅ CACHE EXPENSIVE INITIALIZATION
@st.cache_resource
def load_components():
    """Cache vector store and agent - only loads ONCE"""
    vector_store = VectorStoreManager()
    agent = create_rag_agent(vector_store)
    return vector_store, agent

# Load once and reuse
vector_store, agent = load_components()

# Page config (runs every time - unavoidable but fast)
st.set_page_config(page_title="SmartDoc Bot", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    /* Dark global background like ChatGPT */
    .stApp {
        background: #111827;
        color: #f9fafb;
        font-family: "Inter", "Segoe UI", -apple-system, sans-serif;
    }

    /* Main container */
    .main-container {
        max-width: 1100px;
        margin: 0 auto;
        padding: 1rem;
    }

    /* Dark header */
    .header-section {
        text-align: center;
        padding: 2.5rem 2rem;
        background: rgba(31, 41, 55, 0.8);
        border-radius: 24px;
        box-shadow: 0 25px 50px rgba(0,0,0,0.5);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(75, 85, 99, 0.3);
        margin-bottom: 2rem;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 25px;
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.3);
    }

    .main-title {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #f9fafb, #e5e7eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 1rem 0;
        font-weight: 800;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 1.15rem;
        margin: 0;
    }

    /* Dark chat container */
    .chat-container {
        background: rgba(17, 24, 39, 0.95);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 25px 50px rgba(0,0,0,0.6);
        backdrop-filter: blur(30px);
        border: 1px solid rgba(55, 65, 81, 0.5);
        max-height: 75vh;
        overflow-y: auto;
        margin-bottom: 2rem;
    }

    /* Chat bubbles - ChatGPT style */
    .message-bubble {
        display: flex;
        gap: 12px;
        margin-bottom: 20px;
        align-items: flex-start;
        animation: fadeIn 0.4s ease-out;
        padding: 0 4px;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        font-weight: 600;
        flex-shrink: 0;
        margin-top: 2px;
    }

    /* Bot messages on RIGHT */
    .bot-message {
        justify-content: flex-end;
    }

    .avatar-bot {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }

    .bot-bubble {
        background: #343541 !important;
        color: #ececf1 !important;
        border-radius: 18px 18px 4px 18px;
        border-bottom-right-radius: 4px;
        max-width: 80%;
        margin-left: 52px;
        border: 1px solid rgba(75, 85, 99, 0.3);
    }

    /* User messages on LEFT */
    .avatar-user {
        background: linear-gradient(135deg, #f97316, #ea580c);
        color: white;
        box-shadow: 0 4px 12px rgba(249, 115, 22, 0.4);
    }

    .user-bubble {
        background: linear-gradient(135deg, #202123, #2a2b32) !important;
        color: #ececf1 !important;
        border-radius: 18px 18px 18px 4px;
        border-bottom-left-radius: 4px;
        max-width: 80%;
        margin-right: 52px;
        border: 1px solid rgba(75, 85, 99, 0.3);
    }

    .bubble-content {
        padding: 16px 20px;
        border-radius: 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        line-height: 1.65;
        font-size: 15px;
    }

    .timestamp {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 8px;
        font-weight: 500;
        font-family: 'SF Mono', monospace;
    }

    /* Sidebar dark theme */
    .sidebar-title {
        font-size: 1.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #10b981, #059669);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }

    .metric-container {
        background: rgba(31, 41, 55, 0.8);
        padding: 1.2rem;
        border-radius: 16px;
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        border: 1px solid rgba(55, 65, 81, 0.6);
    }

    /* Enhanced button styling */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: none !important;
        letter-spacing: 0.5px;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 30px rgba(16, 185, 129, 0.4) !important;
    }

    /* File uploader dark theme */
    .uploadedFileUploader {
        background: rgba(55, 65, 81, 0.5);
        border-radius: 12px;
        border: 2px dashed #4b5563;
        padding: 1rem;
    }

    /* Dark expander */
    .streamlit-expanderHeader {
        background: rgba(55, 65, 81, 0.8) !important;
        border-radius: 12px;
        padding: 12px 16px;
        font-weight: 600;
        color: #f9fafb;
        border: 1px solid rgba(75, 85, 99, 0.5);
    }

    /* Chat input enhancement */
    .stChatInput input {
        background: rgba(55, 65, 81, 0.8) !important;
        border: 1px solid rgba(75, 85, 99, 0.5) !important;
        border-radius: 20px !important;
        color: #f9fafb !important;
        padding: 16px 20px !important;
        font-size: 15px !important;
    }

    /* Scrollbar */
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }
    .chat-container::-webkit-scrollbar-track {
        background: rgba(55, 65, 81, 0.3);
        border-radius: 10px;
    }
    .chat-container::-webkit-scrollbar-thumb {
        background: rgba(156, 163, 175, 0.6);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ✅ CACHE METRICS
@st.cache_data(ttl=30)  # Refresh every 30s max
def get_cached_metrics():
    """Cache document count - avoids DB calls"""
    try:
        return vector_store.get_collection_count()
    except:
        return 0

# FIXED timestamp functions (keep these)
def format_time_stored(dt_obj=None):
    if dt_obj is None:
        return datetime.now().strftime("%H:%M")
    return dt_obj.strftime("%H:%M")

def get_stored_datetime():
    return datetime.now()

# Session state initialization (runs every time - fast)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "confirm_clear_docs" not in st.session_state:
    st.session_state.confirm_clear_docs = False
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False  # Track processing state



# Header (fast)
st.markdown("""
<div class='header-section'>
    <div class='badge'>🧠 RAG-Powered Document Assistant</div>
    <h1 class='main-title'>SmartDoc Bot</h1>
    <p class='subtitle'>Upload PDFs & DOCX files. Ask questions. Get precise answers.</p>
</div>
""", unsafe_allow_html=True)

# OPTIMIZED SIDEBAR - Only run when needed
with st.sidebar:
    st.markdown("<div class='sidebar-title'>📁 Document Control Panel</div>", unsafe_allow_html=True)
    
    # File upload (widget interaction triggers rerun - normal)
    uploaded_files = st.file_uploader(
        "Choose PDF, DOCX, or DOC files",
        type=["pdf", "docx", "doc"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    # ✅ PROCESS DOCUMENTS ONLY WHEN BUTTON CLICKED
    if st.button("🚀 Process & Index Documents", type="primary", use_container_width=True):
        if uploaded_files:
            st.session_state.doc_processed = True
            with st.spinner("🔄 Processing your documents..."):
                try:
                    file_ids = []
                    for file in uploaded_files:
                        file_id = str(uuid.uuid4())
                        file_ids.append(file_id)
                        content = file.getvalue()
                        chunks = process_document(content, file.name, file_id)
                        vector_store.add_documents(chunks, file_id)
                    
                    st.session_state.uploaded_files.extend(file_ids)
                    st.success(f"✅ Indexed {len(uploaded_files)} document(s)!")
                    st.cache_data.clear()  # Clear metrics cache after update
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("👆 Please select files first")

    # ✅ CACHED METRICS - Only calls DB once every 30s
    st.markdown("### 📊 Knowledge Base")
    doc_count = get_cached_metrics()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📚 Documents", doc_count)
    with col2:
        st.metric("🟢 Status", "Ready")

    # Clear logic (conditional)
    st.markdown("### 🗑️ Clear Database")
    if doc_count > 0 and not st.session_state.confirm_clear_docs:
        if st.button("🗑️ Clear All Documents", type="secondary", use_container_width=True):
            st.session_state.confirm_clear_docs = True
            st.rerun()
    elif st.session_state.confirm_clear_docs:
        st.warning("⚠️ This will **permanently delete** all documents!")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Confirm Delete", type="primary"):
                with st.spinner("🗑️ Deleting..."):
                    count = vector_store.clear_document()
                    st.session_state.uploaded_files = []
                    st.session_state.confirm_clear_docs = False
                    st.session_state.doc_processed = False
                    st.cache_data.clear()
                    st.success(f"✅ Cleared {count} documents")
                    st.rerun()
        with col2:
            if st.button("❌ Cancel", type="secondary"):
                st.session_state.confirm_clear_docs = False
                st.rerun()
    
    # New chat (lightweight)
    if st.button("🧹 New Chat Session", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()


# Display messages (fast iteration)
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    stored_time = message.get("timestamp")
    time_str = format_time_stored(stored_time)
    
    if role == "user":
        st.markdown(f"""
        <div class='message-bubble'>
            <div class='avatar avatar-user'>👤</div>
            <div class='bubble-content user-bubble'>
                <div>{content}</div>
                <div class='timestamp'>{time_str}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='message-bubble bot-message'>
            <div class='bubble-content bot-bubble'>
                <div>{content}</div>
                <div class='timestamp'>{time_str}</div>
            </div>
            <div class='avatar avatar-bot'>🤖</div>
        </div>
        """, unsafe_allow_html=True)
        
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Sources"):
                for source in set(message["sources"]):
                    st.markdown(f"• {source}")

st.markdown("</div>", unsafe_allow_html=True)

# Chat input - ONLY runs when user types
if prompt := st.chat_input("💭 Ask a question about your documents..."):
    now_time = get_stored_datetime()
    
    user_msg = {
        "role": "user",
        "content": prompt,
        "timestamp": now_time
    }
    st.session_state.messages.append(user_msg)
    
    time_str = format_time_stored(now_time)
    st.markdown(f"""
    <div class='message-bubble'>
        <div class='avatar avatar-user'>👤</div>
        <div class='bubble-content user-bubble'>
            <div>{prompt}</div>
            <div class='timestamp'>{time_str}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("🤖 Analyzing documents..."):
        try:
            config = {"configurable": {"thread_id": st.session_state.session_id}}
            result = agent.invoke({"messages": [{"role": "user", "content": prompt}]}, config=config)
            
            answer = result["messages"][-1]["content"]
            sources = result.get("sources", [])
            
            assistant_msg = {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "timestamp": get_stored_datetime()
            }
            st.session_state.messages.append(assistant_msg)
            st.rerun()
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Error: {str(e)}",
                "timestamp": get_stored_datetime()
            })
            st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
