from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config.settings import config
from src.utils.logger import LoggerSetup


class EmbeddingService:
    """Service for generating embeddings using HuggingFace models."""
    
    def __init__(self):
        self.logger = LoggerSetup.get_logger(self.__class__.__name__)
        self.model_name = config.embedding.model_name
        self.device = config.embedding.device
        self.model: Optional[SentenceTransformer] = None
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the embedding model."""
        try:
            self.logger.info("Initializing embedding model: %s", self.model_name)
            
            # Show progress bar for model loading
            with tqdm(total=1, desc="Loading embedding model", unit="model") as pbar:
                self.model = SentenceTransformer(
                    self.model_name, 
                    device=self.device
                )
                self.model.max_seq_length = config.embedding.max_seq_length
                pbar.update(1)
            
            self.logger.info("Embedding model initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize embedding model: %s", str(e))
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not self.model:
            raise RuntimeError("Embedding model not initialized")
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
            return embedding.tolist()
        except Exception as e:
            self.logger.error("Failed to generate embedding: %s", str(e))
            raise
    
    def generate_embeddings(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """Generate embeddings for multiple texts with progress bar."""
        if not self.model:
            raise RuntimeError("Embedding model not initialized")
        
        if batch_size is None:
            batch_size = config.embedding.batch_size
        
        try:
            # For small batches, don't show progress
            if len(texts) < 10:
                embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            else:
                # Show progress for larger batches
                embeddings = self.model.encode(
                    texts, 
                    convert_to_numpy=True, 
                    show_progress_bar=True,
                    batch_size=batch_size
                )
            
            return embeddings.tolist()
        except Exception as e:
            self.logger.error("Failed to generate embeddings: %s", str(e))
            raise