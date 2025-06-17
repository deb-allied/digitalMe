# 🧠 DigitalMe - Multi-Personality AI Communication Platform

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/poetry-1.0+-purple.svg)](https://python-poetry.org/)
[![Gradio](https://img.shields.io/badge/gradio-4.0+-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Executive Summary
DigitalMe represents a paradigm shift in AI-powered communication, introducing a multi-personality chatbot system designed to enable seamless professional discourse across diverse domains. By leveraging advanced embeddings and vector search technologies, DigitalMe creates distinct AI personalities that can authentically communicate within specific professional contexts, effectively allowing a single user to engage with domain experts using their specialized knowledge and communication patterns. It aims to address commnication gaps between varying professions and helps address them in an effective manner, using user data and an internal chat model which can effectively mimic the specified professions talking style and expertise, while not having the knowledge access of other personalities. Every personality is catered to only that specific one, without generalising all the data resources and information lingo thus providing effective reasoning with a few prompts without frustration.

The different personalities will be created using our model, based on the professional context assesment.

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

## Core Technology Architecture
### Personality-Driven AI Framework
DigitalMe employs a sophisticated personality modeling system that goes beyond traditional chatbots by creating independent AI agents, each trained on profession-specific conversational data. The system currently processes ChatGPT conversation dumps in JSON format, with plans to integrate multiple data sources for enhanced training capabilities.
### Vector-Based Knowledge Retrieval
The platform utilizes state-of-the-art embedding technologies combined with vector search algorithms to ensure contextually relevant responses. This approach enables:

- Semantic Understanding: Deep comprehension of domain-specific terminology and concepts
- Contextual Relevance: Responses that align with professional standards and expectations
- Personality Consistency: Maintaining distinct communication styles across interactions

### Independent Agent Architecture
Each personality operates as an autonomous agent with:

- Unique Knowledge Base: Profession-specific data repositories with minimal overlap
- Distinct Communication Patterns: Industry-appropriate language, tone, and expertise levels
- Specialized Context Awareness: Understanding of domain-specific challenges and solutions

## Product Differentiation & Value Proposition
### Revolutionary Multi-Personality Approach
Unlike conventional AI assistants that attempt to be generalists, DigitalMe creates specialist personalities that excel in their respective domains. This approach delivers:
- Professional Authenticity: Each personality communicates with the depth and nuance expected from industry professionals, using appropriate jargon, methodologies, and problem-solving approaches.
- Domain Expertise: Personalities are trained on extensive professional conversations, ensuring they understand not just the technical aspects but also the cultural and practical elements of each profession.
- Contextual Adaptability: The system recognizes when to switch between personalities or blend expertise for cross-functional challenges.
### Scalable Knowledge Architecture
The platform's design allows for:

- Rapid Personality Development: New professional personalities can be created by training on relevant conversation data
- Knowledge Base Expansion: Continuous integration of new data sources and conversation types
- Cross-Domain Intelligence: Strategic knowledge overlap enables collaboration between personalities when complex, multi-disciplinary problems arise

## Use Cases & Applications
### Professional Communication Enhancement

- Consultants: Switch between client-specific communication styles and industry expertise
- Business Development: Engage prospects using their industry's language and concerns
- Technical Sales: Communicate complex solutions in terms relevant to different professional audiences
- Project Management: Coordinate across diverse teams using appropriate professional contexts

### Knowledge Amplification

- Research & Development: Access specialized knowledge across multiple domains without hiring multiple experts
- Strategic Planning: Leverage diverse professional perspectives for comprehensive decision-making
- Training & Education: Provide realistic professional interaction scenarios for skill development

## Technical Innovation
### Advanced Embedding Systems
The platform employs cutting-edge natural language processing to create rich, multidimensional representations of professional communication patterns, ensuring that personality responses are both technically accurate and stylistically appropriate.
### Intelligent Context Management
DigitalMe's architecture maintains conversation context while switching between personalities, enabling seamless transitions and collaborative problem-solving across professional domains.
### Scalable Vector Search
The system's vector search capabilities ensure rapid retrieval of relevant information while maintaining personality-specific response patterns, delivering both speed and authenticity.
## Future Roadmap
### Continuous Learning Integration
- Reinforcement Learning Pipeline: Implementation of continuous training mechanisms that allow personalities to evolve based on successful interactions and user feedback.
- Real-Time Adaptation: Development of systems that can adjust personality responses based on conversation outcomes and user satisfaction metrics.
### Enhanced Context Awareness
- Multi-Modal Integration: Expansion beyond text to include voice patterns, communication preferences, and behavioral modeling.
- Situational Intelligence: Advanced context recognition that adapts communication style based on meeting types, urgency levels, and relationship dynamics.
### Data Source Diversification
- Professional Network Integration: Incorporation of LinkedIn conversations, industry forums, and professional documentation.
- Academic & Research Data: Integration of scientific papers, industry reports, and technical documentation for enhanced expertise depth.
- Real-World Interaction Data: Collection and processing of actual professional interactions (with appropriate privacy protections) to improve authenticity.
### Competitive Advantage
DigitalMe's unique approach addresses critical limitations in current AI communication tools:
- Depth Over Breadth: Instead of creating a single AI that knows a little about everything, DigitalMe creates multiple AIs that know a lot about specific domains.
- Professional Credibility: Users can engage with confidence, knowing their AI counterpart truly understands their professional context and challenges.
- Scalable Expertise: Organizations can access multiple domains of professional expertise without the overhead of hiring specialists in every field.
- Authentic Communication: Each personality maintains the communication patterns and problem-solving approaches specific to their profession, ensuring natural and credible interactions.
## Market Impact
DigitalMe represents a fundamental shift toward specialized AI that can serve as trusted professional counterparts rather than generic assistants. This approach has the potential to revolutionize how professionals collaborate, learn, and solve complex problems by providing access to authentic, specialized expertise across multiple domains within a single platform.
The system's modular, personality-driven architecture positions it as a scalable solution for organizations seeking to enhance their professional capabilities without the traditional constraints of hiring and training specialized personnel across multiple domains.

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
poetry run python -m src.gradio_main

#For shell only run (testing purposes)
poetry run python -m src.main
```
- In the shell only run, its wired to install the entire json file first, which takes 5 mins on a GPU. To create the json file in its exact format, use the data_preprocessing_augmented file, here one will need gpt 3.5 api key to access and create it, will take approx 20 to 30mins (for large chatgpt data input). One can use better models for the format creation, the prompt for the format is given in the code.

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
```

### Environment Variables (.env)

Create a `.env` file in the root directory:

```env
#OpenAI Resources
OPENAI_API_KEY="<Your key here>"
OPENAI_MODEL=gpt-3.5-turbo

# Data Configuration
PERSONALITY_DATA_PATH=r"<Path to json file>"
```

## 📖 Usage

### Starting the Application

```bash
# Using Poetry
poetry run python -m src.gradio_main
```

The application will:
1. Auto-initialize all services
2. Load the default chroma data file if it exists
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
            "title": "AI Workforce Transition Discussion",
            "summary": "The conversation highlights the importance of transitioning individuals to outcome-based models in the face of AI-driven workforce shifts. It emphasizes the need for humans to focus on critical thinking, problem-solving, and creativity to complement AI. The potential consequences of not proactively leading this transition are discussed, with a call for businesses to take the lead to avoid government intervention. The invitation to a virtual chat on shaping the future of work is extended, focusing on the challenges and opportunities presented by AI in the workplace.",
            "professional_personality": {
                "primary_profession_type": "Consultant",
                "secondary_profession_type": "Researcher",
                "professional_traits": {
                    "analytical_thinking": 8,
                    "communication_skills": 6,
                    "creativity": 5,
                    "problem_solving": 9,
                    "leadership": 7,
                    "attention_to_detail": 6,
                    "empathy": 4,
                    "strategic_thinking": 9,
                    "technical_aptitude": 5,
                    "business_acumen": 7
                },
                "professional_characteristics": [
                    "Strategic thinker",
                    "Problem solver",
                    "Analytical",
                    "Adaptable",
                    "Innovative"
                ],
                "likely_industries": [
                    "Management Consulting",
                    "Research & Development",
                    "Technology",
                    "Business Strategy"
                ],
                "confidence": 0.85
            },
            "messages": [
                "The **real economic shock isn’t job loss, but the failure to shift individuals to outcome-based models.** We need an approach where humans complement AI by focusing on critical thinking, problem-solving, and creativity—skills machines can’t easily replicate.  \n\n**Are we ready for this transition?** If businesses don’t lead it proactively, governments will be forced to step in, and that rarely leads to optimal solutions.  \n\nLet’s take this conversation further. **I’d love to invite you to a virtual chat on this topic.** Let me know if you’re open to a discussion on how we shape the Now of Work.  \n\n---\n\nWould you like me to personalize it further?",
                "Here's a 200-word reply offering a virtual chat on the subject:  \n\n---\n\n**A thought-provoking post, Dr. Apostolos.** The AI-driven workforce shift you highlight is accelerating, and DBS Bank’s move is just the tip of the iceberg. While AI’s efficiency gains are undeniable, the deeper issue is how we reimagine work and value creation in an AI-augmented world.  \n\nOne key distinction is that **AI does not replace jobs—it replaces tasks**. The challenge is transitioning human workers to focus on outcomes rather than outputs. This is where **Flexible Resource Architecture (FRA)** and **Digital Human Resource Architecture (DHRA)** come in, ensuring a balance between automation and human ingenuity.  \n\nThe **real economic shock isn’t job loss, but the failure to shift individuals to outcome-based models.** We need an approach where humans complement AI by focusing on critical thinking, problem-solving, and creativity—skills machines can’t easily replicate.",
                "I want you to pull not push and And give some facts that support, you know, the 2.2 billion workers, 50% of what they do will get done, not by them. We already have 47% attrition in the US, 33% in the UK, 20% fewer jobs. So yes, AI won't take someone's job, it's the percentage of the job in the effect. And we're already seeing all the knock-on effects of it over the last 12 months now, and it's a significant problem. There's a societal problem, obviously. We've then got the country PLC problems that exist in governments around the world, major economic, political, societal changes. There's massive tax implications, and essentially... Yeah, so just use some tools around that, and obviously you can share. We've got lots of research on this that we're happy to go off and share if you'd like to do a virtual meeting.",
                "The knock-on effects of AI-driven change over the past 12 months alone have been significant. As you highlight, DBS Bank is leading this transition, but they won’t be the last. The real challenge is **not whether AI will replace jobs, but how businesses and governments handle the shifting balance of work and taxation**.  \n\nWe’ve been researching these trends extensively. If you’re interested, we’d be happy to share insights and discuss solutions in a **virtual chat**. Let me know if that would be of value.  \n\n---\n\nThis keeps the tone open-ended, fact-driven, and invites engagement. Let me know if you’d like any refinements!",
                "Here’s your refined response incorporating the key facts while **pulling, not pushing**, and inviting discussion:  \n\n---\n\n**A great discussion, Dr. Apostolos.** The impact of AI on the workforce isn’t theoretical—it’s already happening, and the numbers are staggering.  \n\nGlobally, there are **2.2 billion workers**, and AI is expected to take over **50% of their tasks**, not full jobs, but enough to create profound economic shifts. In the **US, attrition is at 47%**, in the **UK, it’s 33%**, and we’ve already seen **20% fewer job vacancies**. The effects aren’t just corporate; they extend to **Country PLCs**, where tax bases are under strain, and governments face mounting economic, political, and societal pressures."
            ],
            "message_count": 5,
            "total_characters": 3793,
            "created_date": "2025-02-27T15:13:33.789408"
        },
    ],
    "statistics": {
        "total_conversations": 755,
        "profession_distribution": {
            "HR_Professional": 17,
            "Legal_Professional": 63,
            "Consultant": 328,
            "Engineer": 29,
            "Financial_Analyst": 84,
            "Marketing_Professional": 19,
            "Entrepreneur": 40,
            "Researcher": 85,
            "Sales_Professional": 20,
            "Operations_Specialist": 7,
            "Creative_Designer": 16,
            "Technical_Support": 20,
            "Manager": 15,
            "Teacher": 4,
            "Healthcare_Professional": 4,
            "Insurance_Professional": 1,
            "Politician": 1,
            "Motivational Speaker": 1,
            "Sports Analyst": 1
        },
        "analysis_timestamp": "2025-05-29T13:08:22.687277"
    }
}
```

### Data Schema

```python
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ProfessionalTraits(BaseModel):
    """Professional traits model."""
    analytical_thinking: int = Field(ge=0, le=10)
    communication_skills: int = Field(ge=0, le=10)
    creativity: int = Field(ge=0, le=10)
    problem_solving: int = Field(ge=0, le=10)
    leadership: int = Field(ge=0, le=10)
    attention_to_detail: int = Field(ge=0, le=10)
    empathy: int = Field(ge=0, le=10)
    strategic_thinking: int = Field(ge=0, le=10)
    technical_aptitude: int = Field(ge=0, le=10)
    business_acumen: int = Field(ge=0, le=10)


class ProfessionalPersonality(BaseModel):
    """Professional personality model."""
    primary_profession_type: str
    secondary_profession_type: Optional[str] = None
    professional_traits: ProfessionalTraits
    professional_characteristics: List[str]
    likely_industries: List[str]


class Conversation(BaseModel):
    """Conversation model."""
    title: str
    summary: str
    professional_personality: ProfessionalPersonality
    messages: List[str]
    message_count: int
    total_characters: int
    created_date: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConversationData(BaseModel):
    """Root data model for conversations."""
    conversations: List[Conversation]
    statistics: Dict[str, Any]


class MessageDocument(BaseModel):
    """Document model for vector storage."""
    id: str
    content: str
    personality_type: str
    conversation_title: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
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

Made with ❤️ by Debshishu Ghosh