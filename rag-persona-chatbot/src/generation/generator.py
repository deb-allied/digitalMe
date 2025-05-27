from typing import List, Optional

from openai import OpenAI

from ..core.config import settings
from ..core.logger import get_logger
from ..retrieval.retriever import Document
from .persona_manager import PersonaManager

logger = get_logger(__name__)


class ResponseGenerator:
    """Generates responses using OpenAI with retrieved context."""
    
    def __init__(self, persona_manager: PersonaManager):
        self.client = OpenAI(api_key=settings.openai.api_key)
        self.persona_manager = persona_manager
        logger.info("ResponseGenerator initialized")
    
    def generate(
        self,
        query: str,
        documents: List[Document],
        persona: str
    ) -> str:
        """Generate a response based on query and context."""
        logger.info("Generating response", query=query[:50], persona=persona, doc_count=len(documents))
        
        # Get persona prompt
        system_prompt = self.persona_manager.get_persona_prompt(persona)
        
        # Build context from documents
        context = self._build_context(documents)
        
        # Build user prompt
        user_prompt = self._build_user_prompt(query, context)
        
        try:
            response = self.client.chat.completions.create(
                model=settings.openai.generation_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=settings.openai.temperature,
                max_tokens=settings.openai.max_tokens
            )
            
            generated_text = response.choices[0].message.content or ""
            logger.info("Response generated successfully")
            
            return generated_text
            
        except Exception as e:
            logger.error("Error generating response", error=str(e))
            return self._fallback_response(persona)
    
    def _build_context(self, documents: List[Document]) -> str:
        """Build context string from documents."""
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(documents):
            context_parts.append(f"[Document {i+1}]")
            context_parts.append(f"Title: {doc.metadata.get('title', 'N/A')}")
            context_parts.append(f"Content: {doc.content}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _build_user_prompt(self, query: str, context: str) -> str:
        """Build the user prompt with query and context."""
        prompt = f"""Based on the following context, please answer the question. If the context doesn't contain enough information to answer accurately, please say so.

Context:
{context}

Question: {query}

Remember to answer based only on the provided context. Do not make up information."""
        
        return prompt
    
    def _fallback_response(self, persona: str) -> str:
        """Generate a fallback response when generation fails."""
        return f"I apologize, but I'm having trouble generating a response right now. As a {persona}, I want to ensure I provide accurate information. Please try rephrasing your question or try again later."