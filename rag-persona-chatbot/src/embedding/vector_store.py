from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np

from ..core.config import settings
from ..core.logger import get_logger
from ..preprocessing.data_loader import ChatThread

logger = get_logger(__name__)


class VectorStore:
    """Manages vector storage and retrieval using ChromaDB."""
    
    def __init__(self):
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=settings.vector_db.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(settings.vector_db.collection_name)
            logger.info("Loaded existing collection", name=settings.vector_db.collection_name)
        except:
            self.collection = self.client.create_collection(
                name=settings.vector_db.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Created new collection", name=settings.vector_db.collection_name)
    
    def add_embeddings(self, embedded_threads: List[Tuple[ChatThread, List[float]]]):
        """Add embeddings to the vector store."""
        logger.info("Adding embeddings to vector store", count=len(embedded_threads))
        
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for i, (thread, embedding) in enumerate(embedded_threads):
            # Create unique ID
            thread_id = f"{thread.label}_{i}_{thread.create_time}"
            
            # Prepare metadata
            metadata = {
                "label": thread.label,
                "title": thread.title,
                "create_time": thread.create_time,
                "update_time": thread.update_time,
                "message_count": len(thread.messages)
            }
            
            # Prepare document text
            doc_text = self._prepare_document(thread)
            
            ids.append(thread_id)
            embeddings.append(embedding)
            metadatas.append(metadata)
            documents.append(doc_text)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        
        logger.info("Embeddings added successfully")
    
    def _prepare_document(self, thread: ChatThread) -> str:
        """Prepare document text for storage."""
        messages = []
        for msg in thread.messages[:20]:  # Limit to first 20 messages
            messages.append(f"{msg.author}: {msg.text}")
        
        return "\n".join([
            f"Title: {thread.title}",
            f"Persona: {thread.label}",
            "Messages:",
            *messages
        ])
    
    def query(
        self,
        query_embedding: List[float],
        persona: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> Dict:
        """Query the vector store."""
        if top_k is None:
            top_k = settings.retrieval.top_k
        
        # Build where clause for filtering
        where_filter = {"label": {"$eq": persona}} if persona else None
        
        logger.debug("Querying vector store", persona=persona, top_k=top_k)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            where=where_filter,
            n_results=top_k,
            include=["metadatas", "documents", "distances"]
        )
        
        return dict(results)
    
    def clear(self):
        """Clear all data from the collection."""
        logger.warning("Clearing all data from collection")
        self.client.delete_collection(settings.vector_db.collection_name)
        self.collection = self.client.create_collection(
            name=settings.vector_db.collection_name,
            metadata={"hnsw:space": "cosine"}
        )