# 🤖 SmartDoc Bot - RAG-Powered Document Q&A System


![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)


## ✨ Why SmartDoc Bot?

SmartDoc Bot is **more than just a chatbot** — it’s like having a **smart research assistant** at your fingertips.  
No more scrolling through endless pages or struggling to find the right section.  
Simply **upload your files** and **ask questions in plain language** — the bot will:

- 🧠 **Read & Understand** your documents  
- 💬 **Respond clearly** in natural conversation  
- 📌 **Point you back to exact sources** so you can trust the information  

---

### 🌟 Think of it as:
- 📚 **Your personal study buddy** for textbooks and reports  
- 🧾 **A quick insight extractor** for contracts, policies, or meeting notes  
- 🕵️ **A reliable assistant** that saves time by highlighting the most relevant parts of your files  

---

### 💡 Designed for You
SmartDoc Bot feels **conversational and natural** — no special commands needed.  
Just ask questions the way you normally would, and it takes care of the rest.  

Whether you’re a **student**, **researcher**, or **professional**, SmartDoc Bot helps you:  
- Cut through the noise  
- Focus on what matters most  
- Work smarter, not harder 🚀  

---


## ✨ Features

### 🎯 Core Capabilities
- **Multi-format Support**: PDF and DOCX document processing
- **Intelligent Table Extraction**: Advanced parsing of tables for structured data
- **Conversational Context**: Follow-up questions with memory
- **Intent Classification**: Understands user intent (location queries, metrics, pricing, etc.)
- **Query Reformulation**: Generates multiple query variations for better retrieval
- **Source Citations**: Tracks and displays document sources for answers
- **Dark ChatGPT-style UI**: Modern, responsive interface

### 🧠 Advanced RAG Pipeline
1. **Contextual Understanding** - Resolves follow-up questions
2. **Intent Classification** - Routes queries intelligently
3. **Query Expansion** - Multiple query variations with synonyms
4. **Hybrid Retrieval** - Vector search with relevance scoring
5. **Relevance Checking** - Validates document relevance before answering
6. **Smart Generation** - Context-aware answer synthesis

### 🔍 Document Processing
- **LLM-powered Extraction**: Structured information extraction from tables
- **Smart Chunking**: Semantic-aware text splitting (2000 chars, 400 overlap)
- **Metadata Enrichment**: Automatic entity detection and tagging
- **Table Intelligence**: Converts tables to natural language for retrieval

---


### Key Features in Action
- ✅ Upload multiple PDFs/DOCX
- ✅ Ask questions in natural language
- ✅ Get timestamped responses
- ✅ View source documents
- ✅ Clear chat or database anytime

---

## 📦 Installation

### Prerequisites
- Python 3.12
- pip package manager
- Groq API key ([Get one here](https://console.groq.com))


### Project Structure

smartdoc-bot/<br>
├── frontend/<br>
│   └── streamlit_app.py       # 🎨 Streamlit UI<br>
├── backend/<br>
│   ├── graph.py               # 🧠 LangGraph RAG agent<br>
│   ├── utils.py               # ⚙️ Document processing<br>
│   └── vectore_store.py       # 🗄️ ChromaDB manager<br>
├── chroma_db/                 # 💾 Vector DB storage (auto-created)<br>
├── .env                       # 🔑 API keys<br>
├── requirements.txt           # 📦 Dependencies<br>
└── README.md                  # 📖 Project documentation<br>



### 🏗️ Architecture

┌─────────────┐<br>
│   Streamlit │ 🎨 User Interface<br>
└──────┬──────┘<br>
       │<br>
       ▼<br>
┌─────────────────────────────────────────────┐<br>
│         LangGraph RAG Agent                
│  ┌────────────────────────────────────── <br>
│  │ 1. Contextualize (follow-ups)        <br>
│  │ 2. Classify Intent                  <br>
│  │ 3. Reformulate Query                <br>
│  │ 4. Retrieve (ChromaDB)             <br>
│  │ 5. Check Relevance                  <br>
│  │ 6. Generate / Out-of-Scope Handler   <br>
│  └──────────────────────────────────────<br>
└───────────┬─────────────────────────────────┘<br>
            │<br>
            ▼<br>
     ┌──────────────┐<br>
     │   ChromaDB   │ 🗄️ Vector Store<br>
     └──────────────┘<br>



## ❤️ Made with Love

Made with ❤️ by [Surya](https://github.com/Surya1804)
