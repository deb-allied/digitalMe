# 🧠 DigitalMe - AI-Powered Personality Chat System

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/poetry-1.0+-purple.svg)](https://python-poetry.org/)
[![Gradio](https://img.shields.io/badge/gradio-4.0+-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

DigitalMe is an advanced AI chatbot system that can embody multiple personalities based on conversation data. It uses state-of-the-art embeddings and vector search to provide contextually relevant responses while maintaining distinct personality traits.

## 🌟 Features

- **Multiple Personalities**: Create and manage distinct AI personalities with unique conversational styles
- **Context-Aware Responses**: Uses vector embeddings to find relevant context from conversation history
- **Auto-Initialization**: Services initialize automatically on startup with optional data loading
- **Modern Web Interface**: Clean, responsive Gradio interface with dark theme
- **Real-time Chat**: Interactive chat interface with personality selection
- **Single Query Mode**: Ask one-off questions to specific personalities
- **Personality Management**: Add new messages and expand personality knowledge bases
- **Comprehensive Logging**: Detailed logging system for debugging and monitoring
- **Batch Processing**: Efficient batch operations for loading large conversation datasets

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Data Format](#-data-format)
- [API Reference](#-api-reference)
- [File Structure](#-file-structure)
- [Development](#-development)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## 🏗 Architecture

### System Overview

```
┌─────────────────┐
│  Gradio Web UI  │
└────────┬────────┘
         │
┌────────┴────────┐
│ Chatbot Service │
└────────┬────────┘
         │
┌────────┴────────┐
│Retriever Service│
└────────┬────────┘
         │
┌────────┴────────┐
│VectorStore      │
│Service          │
└────────┬────────┘
         │
┌────────┴────────┐
│Embedding Service│
└─────────────────┘
```

### Components

1. **Frontend (Gradio UI)**
   - Tab-based interface for different functionalities
   - Real-time chat interface
   - System status monitoring
   - Data management tools

2. **Backend Services**
   - **Embedding Service**: Converts text to vector embeddings using sentence transformers
   - **VectorStore Service**: Manages storage and retrieval of conversation vectors
   - **Retriever Service**: Finds relevant context based on queries
   - **Chatbot Service**: Generates responses using retrieved context

3. **Data Models**
   - Structured conversation data with personality metadata
   - Message threading and context preservation

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- Poetry (for dependency management)
- 4GB+ RAM recommended
- GPU optional but recommended for faster embeddings

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/digitalme.git
cd digitalme

# Install dependencies with Poetry
poetry install

# Activate the virtual environment
poetry shell

# Run the application
poetry run python run.py
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/digitalme.git
cd digitalme

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python run.py
```

### Docker Installation

```bash
# Build the Docker image
docker build -t digitalme .

# Run the container
docker run -p 7860:7860 digitalme
```

## ⚙️ Configuration

### Configuration File (`config/settings.py`)

```python
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Data settings
    data_file: str = "data/conversations.json"
    vector_db_path: str = "data/vectorstore"
    
    # Model settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Retriever settings
    top_k_results: int = 5
    similarity_threshold: float = 0.7
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 7860
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/digitalme.log"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

config = Settings()
```

### Environment Variables (.env)

Create a `.env` file in the root directory:

```env
# Data paths
DATA_FILE=data/conversations.json
VECTOR_DB_PATH=data/vectorstore

# Model configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7

# Server configuration
HOST=0.0.0.0
PORT=7860

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/digitalme.log
```

## 📖 Usage

### Starting the Application

```bash
# Using Poetry
poetry run python run.py

# Using Python directly
python run.py
```

The application will:
1. Auto-initialize all services
2. Load the default data file if it exists
3. Launch the web interface at `http://localhost:7860`

### Web Interface Guide

#### 1. System Tab
- View system status and initialization state
- Load custom data files
- Monitor service health
- Reinitialize services if needed

#### 2. Chat Tab
- Select a personality from the dropdown
- Have interactive conversations
- Messages are context-aware based on personality history
- Clear chat to start fresh

#### 3. Query Tab
- Ask single questions without maintaining conversation history
- Get detailed responses with context information
- See how many context messages were used

#### 4. Manage Tab
- Add new messages to existing personalities
- Create new personalities
- View all available personalities
- Expand knowledge bases

### Command Line Usage

For programmatic access:

```python
from src.services.chatbot_service import ChatbotService
from src.services.retriever_service import RetrieverService
from src.services.vectorstore_service import VectorStoreService
from src.services.embedding_service import EmbeddingService

# Initialize services
embedding_service = EmbeddingService()
vectorstore_service = VectorStoreService(embedding_service)
retriever_service = RetrieverService(vectorstore_service)
chatbot_service = ChatbotService(retriever_service)

# Load data
vectorstore_service.load_conversations("data/conversations.json")

# Ask a question
result = chatbot_service.answer_question(
    query="Tell me about yourself",
    personality_type="Assistant"
)
print(result['answer'])
```

## 📊 Data Format

### Conversation Data Structure

The system expects JSON data in the following format:

```json
{
  "conversations": [
    {
      "personality_type": "Assistant",
      "conversation_title": "Introduction",
      "messages": [
        {
          "role": "user",
          "content": "Hello, who are you?"
        },
        {
          "role": "assistant",
          "content": "I'm a helpful AI assistant designed to answer questions and assist with various tasks."
        }
      ]
    },
    {
      "personality_type": "Teacher",
      "conversation_title": "Math Lesson",
      "messages": [
        {
          "role": "user",
          "content": "Can you explain algebra?"
        },
        {
          "role": "assistant",
          "content": "Of course! Algebra is a branch of mathematics that uses letters and symbols to represent numbers and quantities in formulas and equations."
        }
      ]
    }
  ]
}
```

### Data Schema

```python
from pydantic import BaseModel
from typing import List

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class Conversation(BaseModel):
    personality_type: str
    conversation_title: str
    messages: List[Message]

class ConversationData(BaseModel):
    conversations: List[Conversation]
```

## 🔧 API Reference

### EmbeddingService

```python
class EmbeddingService:
    def __init__(self, model_name: str = None):
        """Initialize embedding model"""
        
    def embed_text(self, text: str) -> np.ndarray:
        """Convert text to embedding vector"""
        
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Convert multiple texts to embeddings"""
```

### VectorStoreService

```python
class VectorStoreService:
    def __init__(self, embedding_service: EmbeddingService):
        """Initialize vector store with embedding service"""
        
    def add_message(self, message: str, personality_type: str, 
                   conversation_title: str) -> str:
        """Add single message to vector store"""
        
    def add_conversations_batch(self, conversations: List[Conversation]):
        """Add multiple conversations in batch"""
        
    def search(self, query: str, personality_type: str, 
              top_k: int = 5) -> List[Dict]:
        """Search for similar messages"""
        
    def get_all_personalities(self) -> List[str]:
        """Get list of all personality types"""
        
    def clear_collection(self):
        """Clear all data from vector store"""
```

### RetrieverService

```python
class RetrieverService:
    def __init__(self, vectorstore_service: VectorStoreService):
        """Initialize retriever with vector store"""
        
    def retrieve_context(self, query: str, personality_type: str,
                        max_results: int = 5) -> List[str]:
        """Retrieve relevant context for query"""
```

### ChatbotService

```python
class ChatbotService:
    def __init__(self, retriever_service: RetrieverService):
        """Initialize chatbot with retriever"""
        
    def answer_question(self, query: str, 
                       personality_type: str) -> Dict[str, Any]:
        """Generate answer based on personality and context"""
        # Returns: {
        #     "query": str,
        #     "personality_type": str,
        #     "answer": str,
        #     "context_used": int
        # }
```

## 📁 File Structure

```
DigitalMe:.
│   .env
│   .gitattributes
│   .gitignore
│   app.log
│   LICENSE
│   poetry.lock
│   pyproject.toml
│   README.md
│
├───.vscode
│       settings.json
│
├───backend
│   │   main.py
│   │
│   ├───data
│   │       pruned.json
│   │       pruned123.json
│   │
│   ├───data_preprocessors
│   │       data_preprocess_augment.py
│   │       gpt_data_simplifier.py
│   │       __init__.py
│   │
│   ├───logs
│   │       app.log
│   │       personality_qa.log
│   │
│   ├───utils
│   │   │   chroma_access.py
│   │   │   logger.py
│   │   │   __init__.py
│   │
│   └───vector_data_handler
│       │   db.py
│       │   processor.py
│       │   __init__.py
│
├───chroma_db
│   │   chroma.sqlite3
│   │
│   ├───1202d742-e02e-4b77-bc85-faa50d56d758
│   │       data_level0.bin
│   │       header.bin
│   │       index_metadata.pickle
│   │       length.bin
│   │       link_lists.bin
│
├───config
│   │   settings.py
│
├───data
│       conversations.json
│       pruned123.json
│
├───logs
│       app.log
│       personality_qa.log
│
└───src
    │   gradio_main.py
    │   main.py
    │
    ├───models
    │   │   data_models.py
    │   │   __init__.py
    │
    ├───services
    │   │   chatbot_service.py
    │   │   embedding_service.py
    │   │   retriever_service.py
    │   │   vectorstore_service.py
    │   │   __init__.py
    │
    ├───utils
    │   │   chroma_access.py
    │   │   logger.py
    │   │   __init__.py
```

## 🛠 Development

### Setting Up Development Environment

```bash
# Install development dependencies
poetry install --with dev

# Run tests
poetry run pytest

# Run linting
poetry run flake8 src/
poetry run black src/

# Run type checking
poetry run mypy src/
```

### Adding New Features

1. **Adding a New Personality**
   - Prepare conversation data in the required JSON format
   - Load through the UI or programmatically
   - The system will automatically index and make it available

2. **Extending the Backend**
   - Follow the service pattern in `src/services/`
   - Add corresponding tests in `tests/`
   - Update configuration if needed

3. **Customizing the UI**
   - Modify `src/ui/app.py` or the single-file version
   - Update CSS in the `custom_css` variable
   - Add new tabs or components as needed

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src

# Run specific test file
poetry run pytest tests/test_embedding.py

# Run with verbose output
poetry run pytest -v
```

## 🐛 Troubleshooting

### Common Issues

1. **Services fail to initialize**
   - Check Python version (3.9+ required)
   - Verify all dependencies are installed
   - Check available memory (4GB+ recommended)
   - Review logs in `logs/digitalme.log`

2. **Data loading fails**
   - Verify JSON format matches the schema
   - Check file permissions
   - Ensure file path is correct
   - Validate JSON syntax

3. **Slow performance**
   - Consider using GPU for embeddings
   - Reduce `top_k_results` in configuration
   - Use smaller embedding model
   - Enable batch processing for large datasets

4. **Port already in use**
   - Change port in configuration
   - Kill existing process: `lsof -ti:7860 | xargs kill -9`
   - Use different port: `PORT=7861 python run.py`

### Debug Mode

Enable debug logging:

```bash
LOG_LEVEL=DEBUG python run.py
```

### Performance Optimization

1. **Use GPU acceleration**:
   ```python
   # In config/settings.py
   device: str = "cuda" if torch.cuda.is_available() else "cpu"
   ```

2. **Adjust batch sizes**:
   ```python
   batch_size: int = 32  # Increase for better throughput
   ```

3. **Cache embeddings**:
   ```python
   enable_cache: bool = True
   cache_size: int = 10000
   ```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `poetry run pytest`
5. Commit with conventional commits: `git commit -m 'feat: add amazing feature'`
6. Push to your fork: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Code Style

- Use Black for formatting: `poetry run black src/`
- Follow PEP 8 guidelines
- Add type hints to all functions
- Write docstrings for all public methods
- Keep functions focused and small

### Testing Requirements

- Write tests for new features
- Maintain 80%+ code coverage
- Test edge cases
- Use meaningful test names

### Commit Message Format

Follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test additions/changes
- `chore:` Maintenance tasks

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Gradio](https://gradio.app/) for the web interface framework
- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [Anthropic](https://www.anthropic.com/) for AI inspiration
- All contributors and users of DigitalMe

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/digitalme/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/digitalme/discussions)
- **Email**: your.email@example.com

---

Made with ❤️ by the DigitalMe Team