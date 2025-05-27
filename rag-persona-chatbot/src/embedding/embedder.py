from typing import List, Dict, Tuple
import numpy as np

from openai import OpenAI
import tiktoken

from ..core.config import settings
from ..core.logger import get_logger
from ..preprocessing.data_loader import ChatThread

logger = get_logger(__name__)


class TextEmbedder:
    """Handles text embedding using OpenAI."""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai.api_key)
        self.model = settings.openai.embedding_model
        self.encoder = tiktoken.encoding_for_model("gpt-4")
        logger.info("TextEmbedder initialized", model=self.model)
    
    def _prepare_text(self, chat_thread: ChatThread) -> str:
        """Prepare chat thread text for embedding."""
        # Combine title and messages
        text_parts = [f"Title: {chat_thread.title}", f"Persona: {chat_thread.label}"]
        
        for msg in chat_thread.messages:
            text_parts.append(f"{msg.author}: {msg.text}")
        
        full_text = "\n".join(text_parts)
        
        # Truncate if too long
        tokens = self.encoder.encode(full_text)
        if len(tokens) > 8000:  # Leave some buffer
            tokens = tokens[:8000]
            full_text = self.encoder.decode(tokens)
        
        return full_text
    
    def embed_single(self, text: str) -> List[float]:
        """Embed a single text."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            embedding = response.data[0].embedding
            return embedding
            
        except Exception as e:
            logger.error("Error creating embedding", error=str(e))
            raise
    
    def embed_chat_threads(self, chat_threads: List[ChatThread]) -> List[Tuple[ChatThread, List[float]]]:
        """Embed multiple chat threads."""
        logger.info("Starting embedding process", total_threads=len(chat_threads))
        
        embedded_threads = []
        
        for thread in chat_threads:
            try:
                text = self._prepare_text(thread)
                embedding = self.embed_single(text)
                embedded_threads.append((thread, embedding))
                
            except Exception as e:
                logger.error(
                    "Failed to embed thread",
                    title=thread.title,
                    error=str(e)
                )
        
        logger.info("Embedding complete", embedded_count=len(embedded_threads))
        return embedded_threads