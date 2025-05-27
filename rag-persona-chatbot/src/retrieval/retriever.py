from typing import List, Dict, Optional, Tuple

from ..core.config import settings
from ..core.logger import get_logger
from ..embedding.embedder import TextEmbedder
from ..embedding.vector_store import VectorStore

logger = get_logger(__name__)


class Document:
    """Represents a retrieved document."""
    
    def __init__(self, id: str, content: str, metadata: Dict, score: float):
        self.id = id
        self.content = content
        self.metadata = metadata
        self.score = score


class PersonaRetriever:
    """Handles persona-based document retrieval."""
    
    def __init__(self, embedder: TextEmbedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store
        logger.info("PersonaRetriever initialized")
    
    def retrieve(
        self,
        query: str,
        persona: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Document]:
        """Retrieve relevant documents for a query."""
        logger.info("Retrieving documents", query=query[:50], persona=persona)
        
        # Embed the query
        query_embedding = self.embedder.embed_single(query)
        
        # Query vector store
        results = self.vector_store.query(
            query_embedding=query_embedding,
            persona=persona,
            top_k=top_k if top_k is not None else settings.retrieval.top_k
        )
        
        # Convert to Document objects
        documents = []
        
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                doc = Document(
                    id=results['ids'][0][i],
                    content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i],
                    score=1 - results['distances'][0][i]  # Convert distance to similarity
                )
                
                # Filter by similarity threshold
                if doc.score >= settings.retrieval.similarity_threshold:
                    documents.append(doc)
        
        logger.info("Documents retrieved", count=len(documents))
        return documents