from typing import Optional, Dict, Any, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, ValidationError
import os
from pathlib import Path

# Ensure environment variables are loaded
from dotenv import load_dotenv
load_dotenv()

class Settings:
    def __init__(self):
        self.chroma_persist_directory: Path = Path("chroma_db")
    
    @property
    def chroma_persist_directory_str(self) -> str:
        """Return chroma persist directory as string."""
        return str(self.chroma_persist_directory)
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API Key",
        alias="OPENAI_API_KEY"
    )
    openai_model: str = Field(
        default="gpt-3.5-turbo", 
        alias="OPENAI_MODEL"
    )
    openai_temperature: float = Field(
        default=0.7, 
        ge=0.0, 
        le=1.0,
        alias="OPENAI_TEMPERATURE"
    )
    openai_max_tokens: int = Field(
        default=500, 
        gt=0,
        alias="OPENAI_MAX_TOKENS"
    )
    
    # HuggingFace Embedding Configuration
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL"
    )
    embedding_device: str = Field(
        default="gpu",
        alias="EMBEDDING_DEVICE"
    )
    
    # ChromaDB Configuration
    chroma_collection_prefix: str = Field(
        default="personality_",
        alias="CHROMA_COLLECTION_PREFIX"
    )
    
    # Data Configuration
    personality_data_path: Path = Field(
        default=Path("backend/src/data/pruned_personality.json"),
        alias="PERSONALITY_DATA_PATH"
    )
    
    # Retrieval Configuration
    retrieval_top_k: int = Field(
        default=5, 
        gt=0,
        alias="RETRIEVAL_TOP_K"
    )
    similarity_threshold: float = Field(
        default=0.7, 
        ge=0.0, 
        le=1.0,
        alias="SIMILARITY_THRESHOLD"
    )
    
    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL"
    )
    log_file: Path = Field(
        default=Path("logs/personality_qa.log"),
        alias="LOG_FILE"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return upper_v

    @field_validator("log_file", "chroma_persist_directory", mode="before")
    @classmethod
    def ensure_path_type(cls, v) -> Path:
        """Ensure value is converted to Path and create parent directories"""
        path_obj = Path(v) if not isinstance(v, Path) else v
        return path_obj
    
    @field_validator("personality_data_path", mode="before")
    @classmethod
    def ensure_data_path_type(cls, v) -> Path:
        """Ensure personality data path is converted to Path"""
        return Path(v) if not isinstance(v, Path) else v

    def model_post_init(self, __context) -> None:
        """Post-initialization hook to create directories and validate settings"""
        # Create necessary directories
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_directory.mkdir(parents=True, exist_ok=True)
    
    def validate_openai_config(self) -> None:
        """Validate OpenAI configuration when needed"""
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key is required for chatbot functionality. "
                "Please set OPENAI_API_KEY in your .env file or environment variables."
            )
    
    def get_openai_api_key(self) -> str:
        """Get OpenAI API key with validation"""
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file."
            )
        return self.openai_api_key
    
    def is_openai_configured(self) -> bool:
        """Check if OpenAI is properly configured"""
        return self.openai_api_key is not None and len(self.openai_api_key.strip()) > 0

# Safe initialization with error handling
def create_settings() -> Settings:
    """Create settings instance with proper error handling"""
    try:
        return Settings()
    except ValidationError as e:
        print(f"❌ Configuration error: {e}")
        print("\n📝 Please check your .env file and ensure all required values are set.")
        print("\nRequired .env file format:")
        print("OPENAI_API_KEY=your_openai_api_key_here")
        raise
    except Exception as e:
        print(f"❌ Unexpected error loading configuration: {e}")
        raise

# Initialize settings as a singleton
settings = create_settings()

# Convenience function to check if system is ready
def check_system_requirements() -> Dict[str, bool]:
    """Check if all system requirements are met"""
    checks = {
        "openai_configured": settings.is_openai_configured(),
        "data_path_exists": settings.personality_data_path.exists(),
        "log_directory_writable": os.access(settings.log_file.parent, os.W_OK),
        "chroma_directory_writable": os.access(settings.chroma_persist_directory.parent, os.W_OK)
    }
    return checks

def print_system_status() -> None:
    """Print current system configuration status"""
    print("🔧 System Configuration Status:")
    print("=" * 40)
    
    checks = check_system_requirements()
    
    print(f"✅ OpenAI Configured: {'Yes' if checks['openai_configured'] else '❌ No'}")
    print(f"✅ Data File Exists: {'Yes' if checks['data_path_exists'] else '❌ No'}")
    print(f"✅ Log Directory Writable: {'Yes' if checks['log_directory_writable'] else '❌ No'}")
    print(f"✅ ChromaDB Directory Writable: {'Yes' if checks['chroma_directory_writable'] else '❌ No'}")
    
    print(f"\n📁 Data Path: {settings.personality_data_path}")
    print(f"📁 ChromaDB Path: {settings.chroma_persist_directory}")
    print(f"📁 Log Path: {settings.log_file}")
    print(f"🤖 Embedding Model: {settings.embedding_model}")
    
    if not all(checks.values()):
        print("\n⚠️  Some requirements are not met. Please check the configuration.")
    else:
        print("\n✅ All requirements met! System ready to run.")
