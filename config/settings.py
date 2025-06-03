import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ChromaDBConfig:
    """ChromaDB configuration settings."""
    persist_directory: str = "./chroma_db"
    collection_name: str = "personality_conversations"


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "intfloat/multilingual-e5-small"
    device: str = "cuda"  # Change to "cuda" if GPU available
    max_seq_length: int = 512
    batch_size: int = 32  # Batch size for embedding generation


@dataclass
class OpenAIConfig:
    """OpenAI API configuration."""
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 500


@dataclass
class RetrieverConfig:
    """Retriever configuration."""
    top_k: int = 5
    similarity_threshold: float = 0.1


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_level: str = "INFO"
    log_file: Optional[str] = "app.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class Config:
    """Central configuration for the application."""
    chromadb: ChromaDBConfig = field(default_factory=ChromaDBConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data_file: str = "./data/conversations.json"


# Global config instance
config = Config()