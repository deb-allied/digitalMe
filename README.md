# digitalMe

How to Run

Install Poetry (if not already installed):
bashcurl -sSL https://install.python-poetry.org | python3 -

Clone and setup the project:
bashcd personality-qa-chatbot
poetry install

Set up environment variables:

Create a .env file in the project root
Add your OpenAI API key: OPENAI_API_KEY=your_key_here


Copy your conversation data:

Create a data directory in the project root
Copy your paste.txt content to data/conversations.json


Run the application:
Interactive Mode:
bashpoetry run python -m src.main
Single Query Mode:
bashpoetry run python -m src.main --query "What's your approach to data analysis?" "Engineer"


Features

Modular Architecture: Clean separation of concerns with models, services, and utilities
OOPS Compliance: All components use proper object-oriented design
Centralized Configuration: Single config/settings.py file for all settings
Comprehensive Logging: Lazy formatting and configurable log levels
Poetry Package Management: Modern dependency management
Personality-based Storage: Vector store organized by professional personalities
Individual Message Retrieval: Each message stored and retrieved separately
Similarity-based Context: Retrieved messages ranked by similarity scores
Interactive Chat Sessions: Real-time Q&A with chosen personalities
Add New Messages: Dynamically add messages to existing personalities
GPT-3.5 Integration: Uses OpenAI's GPT-3.5 for generating responses

Architecture Overview

EmbeddingService: Generates embeddings using HuggingFace models
VectorStoreService: Manages ChromaDB for vector storage
RetrieverService: Retrieves relevant messages based on queries
ChatbotService: Integrates with GPT-3.5 for Q&A functionality
Main Application: Orchestrates all services and provides UI

The system efficiently stores conversation messages individually, retrieves them based on semantic similarity, and generates personality-consistent responses using GPT-3.5.
Progress Tracking
The application now includes comprehensive progress bars using tqdm for:

Model Initialization: Shows progress when loading embedding models
Service Initialization: Tracks initialization of all services
Data Loading: Shows progress when loading conversations and messages
Batch Embeddings: Progress bar for generating multiple embeddings
ChromaDB Operations: Visual feedback during database operations
API Calls: Progress indicator for OpenAI GPT-3.5 calls
Personality Loading: Progress when retrieving all personalities

This provides better user experience by showing real-time progress during time-consuming operations.