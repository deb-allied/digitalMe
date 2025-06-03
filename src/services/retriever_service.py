from typing import List, Dict, Any, Optional

from config.settings import config
from src.services.vectorstore_service import VectorStoreService
from src.utils.logger import LoggerSetup


class RetrieverService:
    """Service for retrieving relevant messages based on queries."""
    
    def __init__(self, vectorstore_service: VectorStoreService):
        self.logger = LoggerSetup.get_logger(self.__class__.__name__)
        self.vectorstore_service = vectorstore_service
    
    def retrieve(self, query: str, personality_type: str, 
                top_k: Optional[int] = None,
                similarity_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant messages for a query and personality."""
        try:
            if top_k is None:
                top_k = config.retriever.top_k
            if similarity_threshold is None:
                similarity_threshold = config.retriever.similarity_threshold
            
            self.logger.info(
                "Retrieving messages for personality '%s' with query: %s",
                personality_type,
                query[:50] + "..." if len(query) > 50 else query
            )
            
            # Query vector store
            results = self.vectorstore_service.query_by_personality(
                query=query,
                personality_type=personality_type,
                top_k=top_k
            )
            
            # Process results
            retrieved_messages = []
            
            if results and results.get("documents") and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0] if results.get("metadatas") else []
                distances = results["distances"][0] if results.get("distances") else []
                
                for idx, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
                    # Convert distance to similarity score (assuming cosine distance)
                    similarity_score = 1 - dist
                    
                    # Filter by similarity threshold
                    if similarity_score >= similarity_threshold:
                        retrieved_messages.append({
                            "content": doc,
                            "metadata": meta,
                            "similarity_score": similarity_score,
                            "rank": idx + 1
                        })
            
            self.logger.info(
                "Retrieved %d messages above threshold %.2f",
                len(retrieved_messages),
                similarity_threshold
            )
            
            return retrieved_messages
        except Exception as e:
            self.logger.error("Failed to retrieve messages: %s", str(e))
            raise
    
    def get_context_for_personality(self, personality_type: str, 
                                  limit: int = 10) -> List[Dict[str, Any]]:
        """Get general context messages for a personality without specific query."""
        try:
            # Use a generic query to get some representative messages
            generic_query = "Tell me about your experience and background"
            
            return self.retrieve(
                query=generic_query,
                personality_type=personality_type,
                top_k=limit,
                similarity_threshold=0.0  # Get all results
            )
        except Exception as e:
            self.logger.error("Failed to get context: %s", str(e))
            raise