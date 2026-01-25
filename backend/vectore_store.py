from datetime import time
import gc
import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Dict
import logging
import shutil
import os
logger = logging.getLogger(__name__)
from pathlib import Path

class VectorStoreManager:
    """Manages ChromaDB operations for document storage and retrieval"""
    
    def __init__(self, collection_name: str = "rag_documents"):
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"ChromaDB initialized with collection: {collection_name}")
        logger.info(f"Current document count: {self.collection.count()}")
    
    def add_documents(self, chunks: List[Dict], file_id: str):
        """Add document chunks to ChromaDB"""
        try:
            logger.info(f"Adding {len(chunks)} chunks for file_id: {file_id}")
            
            texts = [chunk["text"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            ids = [f"{file_id}_{i}" for i in range(len(chunks))]
            
            # Log structured summary if present
            for i, metadata in enumerate(metadatas):
                if metadata.get('type') == 'structured_summary':
                    logger.info(f"Structured summary detected in chunk {i}")
                    logger.info(f"Preview: {texts[i][:300]}...")
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Add to collection
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully added {len(chunks)} chunks to ChromaDB")
            logger.info(f"New total document count: {self.collection.count()}")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}", exc_info=True)
            raise
    
    def search(self, query: str, k: int = 15) -> List[Dict]:
        """Search for relevant documents with logging"""
        try:
            logger.debug(f"Searching for query: '{query}' with k={k}")
            
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self.collection.count())
            )
            
            # Format results
            documents = []
            if results['documents'] and len(results['documents']) > 0:
                for i in range(len(results['documents'][0])):
                    documents.append({
                        "text": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i]
                    })
            
            logger.debug(f"Found {len(documents)} documents")
            logger.info("Retrieved Documents:")
            for doc in documents:
                logger.info(f"- Distance: {doc['distance']:.4f}, Metadata: {doc['metadata']},Text : {doc['text']}...")
            return documents
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}", exc_info=True)
            return []
    
    def delete_document(self, file_id: str):
        """Delete all chunks associated with a file_id"""
        try:
            # Get all IDs starting with file_id
            all_ids = self.collection.get()['ids']
            ids_to_delete = [id for id in all_ids if id.startswith(file_id)]
            
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunks for file {file_id}")
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise
    
    def get_collection_count(self) -> int:
        """Get total number of documents in collection"""
        count = self.collection.count()
        logger.debug(f"Current collection count: {count}")
        return count
    
    def clear_document(self) -> dict:
        """Clear all documents from collection"""
        try:
            initial_count = self.collection.count()
            collection_name = self.collection.name
            
            logger.info("=" * 80)
            logger.info("🗑️  STARTING CHROMA WIPE")
            logger.info(f"Docs: {initial_count}")
            
            if initial_count == 0:
                logger.info("Already empty")
                return {"deleted_count": 0, "success": True}
            
            # === FIX: Delete and recreate collection ===
            self.client.delete_collection(name=collection_name)
            logger.info(f"✓ Collection deleted: {collection_name}")
            
            # ⚠️ CRITICAL: Recreate collection reference immediately
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            final_count = self.collection.count()
            
            logger.info("✅ DONE!")
            logger.info(f"Before: {initial_count} → After: {final_count}")
            
            return {
                "success": True,
                "deleted_count": initial_count,
                "final_count": final_count
            }
            
        except Exception as e:
            logger.error(f"💥 ERROR: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
