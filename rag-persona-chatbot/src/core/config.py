import os
from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class OpenAIConfig(BaseModel):
    api_key: str
    embedding_model: str = "text-embedding-3-small"
    classification_model: str = "gpt-4-turbo"
    generation_model: str = "gpt-4-turbo"
    temperature: float = 0.3
    max_tokens: int = 2000


class VectorDBConfig(BaseModel):
    persist_directory: str = "./chroma_db"
    collection_name: str = "persona_chats"
    chunk_size: int = 1000
    chunk_overlap: int = 200


class RetrievalConfig(BaseModel):
    top_k: int = 5
    rerank_top_k: int = 3
    similarity_threshold: float = 0.7


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"
    log_file: str = "logs/rag_chatbot.log"


class Settings(BaseSettings):
    openai: OpenAIConfig
    vector_db: VectorDBConfig
    personas: List[str]
    retrieval: RetrievalConfig
    logging: LoggingConfig

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @classmethod
    def from_yaml(cls, config_path: Path) -> "Settings":
        """Load settings from YAML file with environment variable substitution."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Replace environment variables
        config_dict = cls._replace_env_vars(config_dict)
        
        return cls(**config_dict)
    
    @staticmethod
    def _replace_env_vars(config: Any) -> Any:
        """Recursively replace environment variables in config."""
        if isinstance(config, dict):
            return {k: Settings._replace_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [Settings._replace_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        return config


# Singleton instance
settings = Settings.from_yaml(Path("config/settings.yaml"))