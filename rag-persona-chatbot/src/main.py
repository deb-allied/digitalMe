import argparse
from pathlib import Path
from typing import Optional

from .core.logger import get_logger
from .preprocessing.data_loader import DataLoader
from .preprocessing.classifier import PersonaClassifier
from .embedding.embedder import TextEmbedder
from .embedding.vector_store import VectorStore
from .retrieval.retriever import PersonaRetriever
from .retrieval.reranker import Reranker
from .generation.persona_manager import PersonaManager
from .generation.generator import ResponseGenerator
from .utils.helpers import PersonaInferencer

logger = get_logger(__name__)


class RAGPersonaChatbot:
    """Main application class for the RAG Persona Chatbot."""
    
    def __init__(self):
        logger.info("Initializing RAG Persona Chatbot")
        
        # Initialize components
        self.embedder = TextEmbedder()
        self.vector_store = VectorStore()
        self.retriever = PersonaRetriever(self.embedder, self.vector_store)
        self.reranker = Reranker()
        self.persona_manager = PersonaManager()
        self.generator = ResponseGenerator(self.persona_manager)
        self.persona_inferencer = PersonaInferencer()
        
        logger.info("All components initialized successfully")
    
    def preprocess_and_index(self, data_path: Path):
        """Preprocess data and build vector index."""
        logger.info("Starting preprocessing and indexing")
        
        # Load data
        loader = DataLoader(data_path)
        raw_data = loader.load_raw_data()
        chat_threads = loader.parse_chat_threads(raw_data)
        
        # Classify threads
        classifier = PersonaClassifier()
        classified_threads = classifier.classify_batch(chat_threads)
        
        # Embed threads
        embedded_threads = self.embedder.embed_chat_threads(classified_threads)
        
        # Store in vector database
        self.vector_store.add_embeddings(embedded_threads)
        
        logger.info("Preprocessing and indexing complete")
    
    def query(self, user_query: str, persona: Optional[str] = None) -> str:
        """Process a user query and generate a response."""
        logger.info("Processing query", query=user_query[:50], persona=persona)
        
        # Infer persona if not provided
        if not persona:
            persona = self.persona_inferencer.infer_persona(user_query)
            logger.info("Persona inferred", persona=persona)
        
        # Retrieve relevant documents
        documents = self.retriever.retrieve(user_query, persona)
        
        # Rerank documents
        if documents:
            documents = self.reranker.rerank(user_query, documents)
        
        # Generate response
        response = self.generator.generate(user_query, documents, persona)
        
        return response
    
    def interactive_mode(self):
        """Run the chatbot in interactive mode."""
        print("\n=== RAG Persona Chatbot ===")
        print("Available personas:", ", ".join(self.persona_manager.personas.keys()))
        print("Type 'quit' to exit\n")
        
        while True:
            # Get user input
            user_query = input("\nYour question: ").strip()
            
            if user_query.lower() == 'quit':
                print("Goodbye!")
                break
            
            if not user_query:
                continue
            
            # Optional: Get persona
            persona_input = input("Persona (press Enter to auto-detect): ").strip()
            persona = persona_input if persona_input else None
            
            # Get response
            try:
                response = self.query(user_query, persona)
                print(f"\n{persona or 'Auto-detected'} Response:")
                print(response)
            except Exception as e:
                print(f"Error: {str(e)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RAG Persona Chatbot")
    parser.add_argument(
        "--mode",
        choices=["preprocess", "query", "interactive"],
        default="interactive",
        help="Mode to run the application in"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/chats.json"),
        help="Path to the chat data JSON file"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query to process (for query mode)"
    )
    parser.add_argument(
        "--persona",
        type=str,
        help="Persona to use (optional)"
    )
    
    args = parser.parse_args()
    
    # Initialize chatbot
    chatbot = RAGPersonaChatbot()
    
    if args.mode == "preprocess":
        chatbot.preprocess_and_index(args.data_path)
        print("Preprocessing complete!")
    
    elif args.mode == "query":
        if not args.query:
            print("Error: --query is required for query mode")
            return
        
        response = chatbot.query(args.query, args.persona)
        print(f"\nResponse ({args.persona or 'auto-detected'}):")
        print(response)
    
    else:  # interactive mode
        chatbot.interactive_mode()


if __name__ == "__main__":
    main()