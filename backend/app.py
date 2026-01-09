from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import logging
from pathlib import Path
import uuid
import os

from vectore_store import VectorStoreManager
from graph import create_rag_agent
from utils import process_document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize vector store manager
vector_store = VectorStoreManager()

# Initialize LangGraph agent
agent = create_rag_agent(vector_store)

class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str

class UploadResponse(BaseModel):
    status: str
    message: str
    documents_processed: int
    file_ids: List[str]

@app.post("/api/upload", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and process documents (PDF, DOCX).
    Documents are chunked and stored in ChromaDB.
    """
    try:
        file_ids = []
        total_chunks = 0
        
        for file in files:
            # Validate file type
            if not file.filename.endswith(('.pdf', '.docx', '.doc')):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file.filename}. Only PDF and DOCX allowed."
                )
            
            # Generate unique file ID
            file_id = str(uuid.uuid4())
            file_ids.append(file_id)
            
            # Read file content
            content = await file.read()
            
            # Process document (extract text and chunk)
            chunks = process_document(content, file.filename, file_id)
            
            # Store in ChromaDB
            vector_store.add_documents(chunks, file_id)
            total_chunks += len(chunks)
            
            logger.info(f"Processed {file.filename}: {len(chunks)} chunks")
        
        return UploadResponse(
            status="success",
            message=f"Successfully processed {len(files)} document(s)",
            documents_processed=len(files),
            file_ids=file_ids
        )
        
    except Exception as e:
        logger.error(f"Error uploading documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint that uses LangGraph agent to process queries
    with RAG capabilities using ChromaDB.
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Configuration for thread management
        config = {
            "configurable": {
                "thread_id": request.session_id
            }
        }
        
        # Invoke LangGraph agent
        result = agent.invoke(
            {"messages": [{"role": "user", "content": request.query}]},
            config=config
        )
        
        # Extract answer and sources (accessing as dictionary)
        answer = result["messages"][-1]["content"]  # ✅ Fixed
        sources = result.get("sources", [])
        
        logger.info(f"Chat session {request.session_id}: Query processed")
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vector_store": "connected",
        "documents_count": vector_store.get_collection_count()
    }

@app.delete("/api/documents/{file_id}")
async def delete_document(file_id: str):
    """Delete a document from ChromaDB"""
    try:
        vector_store.delete_document(file_id)
        return {"status": "success", "message": f"Deleted document {file_id}"}
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.delete("/api/documents/clear-all")
async def clear_all_documents():
    """Delete all documents from ChromaDB"""
    try:
        count = vector_store.clear_document()
        logger.info(f"Cleared all documents from ChromaDB")
        return {
            "status": "success",
            "message": f"Deleted {count} document(s)",
            "documents_deleted": count
        }
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
