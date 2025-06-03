import json
import sys
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from src.models.data_models import ConversationData, Conversation
from src.services.embedding_service import EmbeddingService
from src.services.vectorstore_service import VectorStoreService
from src.services.retriever_service import RetrieverService
from src.services.chatbot_service import ChatbotService
from src.utils.logger import LoggerSetup
from config.settings import config


class PersonalityQAChatbot:
    """Main application class for the personality-based Q&A chatbot."""
    
    def __init__(self):
        self.logger = LoggerSetup.get_logger(self.__class__.__name__)
        self.embedding_service = None
        self.vectorstore_service = None
        self.retriever_service = None
        self.chatbot_service = None
        self._initialize_services()
    
    def _initialize_services(self) -> None:
        """Initialize all services with progress tracking."""
        try:
            self.logger.info("Initializing services...")
            
            services = [
                ("Embedding Service", lambda: EmbeddingService()),
                ("Vector Store", lambda: VectorStoreService(self.embedding_service)),
                ("Retriever", lambda: RetrieverService(self.vectorstore_service)),
                ("Chatbot", lambda: ChatbotService(self.retriever_service))
            ]
            
            with tqdm(total=len(services), desc="Initializing services", unit="service") as pbar:
                # Initialize embedding service
                self.embedding_service = services[0][1]()
                pbar.update(1)
                
                # Initialize vectorstore service
                self.vectorstore_service = services[1][1]()
                pbar.update(1)
                
                # Initialize retriever service
                self.retriever_service = services[2][1]()
                pbar.update(1)
                
                # Initialize chatbot service
                self.chatbot_service = services[3][1]()
                pbar.update(1)
            
            self.logger.info("All services initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize services: %s", str(e))
            raise
    
    def load_data(self, data_file: Optional[str] = None) -> None:
        """Load conversation data from JSON file with progress tracking."""
        try:
            if data_file is None:
                data_file = config.data_file
            
            self.logger.info("Loading data from: %s", data_file)
            
            # Read JSON file
            print(f"\nLoading data from {data_file}...")
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Parse data
            conversation_data = ConversationData(**data)
            
            # Clear existing data
            print("Clearing existing data...")
            self.vectorstore_service.clear_collection()
            
            # Add conversations to vector store with batch processing
            print(f"\nLoading {len(conversation_data.conversations)} conversations...")
            self.vectorstore_service.add_conversations_batch(conversation_data.conversations)
            
            self.logger.info(
                "Successfully loaded %d conversations",
                len(conversation_data.conversations)
            )
            
        except Exception as e:
            self.logger.error("Failed to load data: %s", str(e))
            raise
    
    def list_personalities(self) -> None:
        """List all available personalities."""
        try:
            personalities = self.vectorstore_service.get_all_personalities()
            
            print("\nAvailable Personalities:")
            print("-" * 30)
            for idx, personality in enumerate(personalities, 1):
                print(f"{idx}. {personality}")
            print()
            
        except Exception as e:
            self.logger.error("Failed to list personalities: %s", str(e))
            raise
    
    def add_message_to_personality(self, message: str, personality_type: str,
                                 conversation_title: str) -> None:
        """Add a new message to an existing personality."""
        try:
            print("\nAdding message...")
            doc_id = self.vectorstore_service.add_message(
                message=message,
                personality_type=personality_type,
                conversation_title=conversation_title
            )
            
            print(f"✓ Successfully added message with ID: {doc_id}")
            
        except Exception as e:
            self.logger.error("Failed to add message: %s", str(e))
            raise
    
    def run_interactive_mode(self) -> None:
        """Run the chatbot in interactive mode."""
        print("\n=== Personality-based Q&A Chatbot ===\n")
        
        while True:
            try:
                # Show menu
                print("\nOptions:")
                print("1. Start chat session")
                print("2. List personalities")
                print("3. Add message to personality")
                print("4. Reload data")
                print("5. Exit")
                
                choice = input("\nSelect option (1-5): ").strip()
                
                if choice == "1":
                    # Start chat session
                    self.list_personalities()
                    personality = input("Enter personality type: ").strip()
                    
                    if personality:
                        self.chatbot_service.chat_session(personality)
                
                elif choice == "2":
                    # List personalities
                    self.list_personalities()
                
                elif choice == "3":
                    # Add message
                    self.list_personalities()
                    personality = input("Enter personality type: ").strip()
                    title = input("Enter conversation title: ").strip()
                    message = input("Enter message: ").strip()
                    
                    if personality and title and message:
                        self.add_message_to_personality(message, personality, title)
                
                elif choice == "4":
                    # Reload data
                    self.load_data()
                    print("✓ Data reloaded successfully!")
                
                elif choice == "5":
                    # Exit
                    print("Goodbye!")
                    break
                
                else:
                    print("Invalid option. Please try again.")
            
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                self.logger.error("Interactive mode error: %s", str(e))
    
    def run_single_query(self, query: str, personality_type: str) -> None:
        """Run a single query."""
        try:
            result = self.chatbot_service.answer_question(query, personality_type)
            
            print(f"\nQuery: {result['query']}")
            print(f"Personality: {result['personality_type']}")
            print(f"\nAnswer: {result['answer']}")
            print(f"\nContext messages used: {result['context_used']}")
            
        except Exception as e:
            self.logger.error("Failed to run query: %s", str(e))
            raise


def main():
    """Main entry point."""
    print("\n🤖 Personality-based Q&A Chatbot")
    print("=" * 40)
    
    # Create application instance
    app = PersonalityQAChatbot()
    
    # Load initial data
    app.load_data()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--query" and len(sys.argv) >= 4:
            # Run single query mode
            query = sys.argv[2]
            personality = sys.argv[3]
            app.run_single_query(query, personality)
        else:
            print("\nUsage:")
            print("  Interactive mode: python -m src.main")
            print("  Query mode: python -m src.main --query \"<question>\" \"<personality>\"")
    else:
        # Run interactive mode
        app.run_interactive_mode()


if __name__ == "__main__":
    main()