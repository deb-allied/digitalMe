from typing import List, Optional

from openai import OpenAI

from ..core.config import settings
from ..core.logger import get_logger
from .retriever import Document

logger = get_logger(__name__)


class Reranker:
    """Reranks retrieved documents for better relevance."""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai.api_key)
        logger.info("Reranker initialized")
    
    def rerank(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> List[Document]:
        """Rerank documents based on relevance to query."""
        if top_k is None:
            top_k = settings.retrieval.rerank_top_k
        
        if len(documents) <= top_k:
            return documents
        
        logger.debug("Reranking documents", input_count=len(documents), top_k=top_k)
        
        # Build reranking prompt
        prompt = self._build_rerank_prompt(query, documents)
        
        try:
            response = self.client.chat.completions.create(
                model=settings.openai.classification_model,
                messages=[
                    {"role": "system", "content": "You are an expert at ranking document relevance."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            # Parse response
            message_content = response.choices[0].message.content
            if message_content is None:
                logger.error("OpenAI response message content is None")
                return documents[:top_k]
            ranking_text = message_content.strip()
            rankings = self._parse_rankings(ranking_text, documents)
            
            return rankings[:top_k]
            
        except Exception as e:
            logger.error("Error during reranking", error=str(e))
            return documents[:top_k]
    
    def _build_rerank_prompt(self, query: str, documents: List[Document]) -> str:
        """Build prompt for reranking."""
        doc_summaries = []
        for i, doc in enumerate(documents[:10]):  # Limit to 10 for API constraints
            summary = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
            doc_summaries.append(f"{i+1}. {summary}")
        
        prompt = f"""Given the query: "{query}"

Rank these documents by relevance (most relevant first):
{chr(10).join(doc_summaries)}

Return only the numbers in order of relevance, separated by commas."""
        
        return prompt
    
    def _parse_rankings(self, ranking_text: str, documents: List[Document]) -> List[Document]:
        """Parse ranking response."""
        try:
            indices = [int(x.strip()) - 1 for x in ranking_text.split(',')]
            reranked = [documents[i] for i in indices if 0 <= i < len(documents)]
            
            # Add any missing documents
            for doc in documents:
                if doc not in reranked:
                    reranked.append(doc)
            
            return reranked
            
        except:
            logger.warning("Failed to parse rankings, returning original order")
            return documents