from typing import List, Dict, Any, Optional
import uuid
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

from config.settings import config
from src.models.data_models import MessageDocument, Conversation
from src.services.embedding_service import EmbeddingService
from src.utils.logger import LoggerSetup


class VectorStoreService:
    """Service for managing vector storage using ChromaDB."""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.logger = LoggerSetup.get_logger(self.__class__.__name__)
        self.embedding_service = embedding_service
        self.client = None
        self.collection = None
        self._initialize_chromadb()
    
    def _initialize_chromadb(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            self.logger.info("Initializing ChromaDB")
            
            with tqdm(total=2, desc="Initializing ChromaDB", unit="step") as pbar:
                # Initialize client
                self.client = chromadb.PersistentClient(
                    path=config.chromadb.persist_directory,
                    settings=Settings(anonymized_telemetry=False)
                )
                pbar.update(1)
                
                # Get or create collection
                self.collection = self.client.get_or_create_collection(
                    name=config.chromadb.collection_name
                )
                pbar.update(1)
            
            self.logger.info("ChromaDB initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize ChromaDB: %s", str(e))
            raise
    
    def add_conversation(self, conversation: Conversation) -> None:
        """Add a conversation to the vector store with progress tracking."""
        try:
            personality_type = conversation.professional_personality.primary_profession_type
            
            # Process each message individually with progress bar
            messages = conversation.messages
            desc = f"Adding messages for {personality_type}"
            
            for idx, message in enumerate(tqdm(messages, desc=desc, unit="msg")):
                doc_id = str(uuid.uuid4())
                
                # Create document
                document = MessageDocument(
                    id=doc_id,
                    content=message,
                    personality_type=personality_type,
                    conversation_title=conversation.title,
                    metadata={
                        "message_index": idx,
                        "summary": conversation.summary,
                        "created_date": conversation.created_date.isoformat(),
                        "secondary_type": conversation.professional_personality.secondary_profession_type
                    }
                )
                
                # Generate embedding
                embedding = self.embedding_service.generate_embedding(message)
                
                # Add to collection
                self.collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[message],
                    metadatas=[{
                        "personality_type": personality_type,
                        "conversation_title": conversation.title,
                        **document.metadata
                    }]
                )
            
            self.logger.info(
                "Added %d messages from conversation '%s' for personality '%s'",
                len(conversation.messages),
                conversation.title,
                personality_type
            )
        except Exception as e:
            self.logger.error("Failed to add conversation: %s", str(e))
            raise
    
    def add_conversations_batch(self, conversations: List[Conversation]) -> None:
        """Add multiple conversations with batch processing and progress tracking."""
        try:
            total_messages = sum(len(conv.messages) for conv in conversations)
            
            with tqdm(total=total_messages, desc="Processing all messages", unit="msg") as pbar:
                for conversation in conversations:
                    personality_type = conversation.professional_personality.primary_profession_type
                    
                    # Prepare batch data
                    ids = []
                    embeddings = []
                    documents = []
                    metadatas = []
                    
                    # Generate embeddings for all messages at once
                    messages = conversation.messages
                    if messages:
                        # Generate embeddings in batch
                        message_embeddings = self.embedding_service.generate_embeddings(messages)
                        
                        for idx, (message, embedding) in enumerate(zip(messages, message_embeddings)):
                            doc_id = str(uuid.uuid4())
                            ids.append(doc_id)
                            embeddings.append(embedding)
                            documents.append(message)
                            metadatas.append({
                                "personality_type": personality_type,
                                "conversation_title": conversation.title,
                                "message_index": idx,
                                "summary": conversation.summary,
                                "created_date": conversation.created_date.isoformat(),
                                "secondary_type": conversation.professional_personality.secondary_profession_type
                            })
                        
                        # Add batch to collection
                        self.collection.add(
                            ids=ids,
                            embeddings=embeddings,
                            documents=documents,
                            metadatas=metadatas
                        )
                        
                        pbar.update(len(messages))
                    
                    self.logger.info(
                        "Added %d messages from conversation '%s' for personality '%s'",
                        len(messages),
                        conversation.title,
                        personality_type
                    )
        except Exception as e:
            self.logger.error("Failed to add conversations batch: %s", str(e))
            raise
    
    def add_message(self, message: str, personality_type: str, 
                   conversation_title: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a single message to an existing personality."""
        try:
            doc_id = str(uuid.uuid4())
            
            # Generate embedding
            embedding = self.embedding_service.generate_embedding(message)
            
            # Prepare metadata
            meta = {
                "personality_type": personality_type,
                "conversation_title": conversation_title
            }
            if metadata:
                meta.update(metadata)
            
            # Add to collection
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[message],
                metadatas=[meta]
            )
            
            self.logger.info(
                "Added message to personality '%s' in conversation '%s'",
                personality_type,
                conversation_title
            )
            
            return doc_id
        except Exception as e:
            self.logger.error("Failed to add message: %s", str(e))
            raise
    
    def query_by_personality(self, query: str, personality_type: str, 
                           top_k: Optional[int] = None) -> Dict[str, Any]:
        """Query messages for a specific personality."""
        try:
            if top_k is None:
                top_k = config.retriever.top_k
            
            # Generate query embedding
            query_embedding = self.embedding_service.generate_embedding(query)
            
            # Query with personality filter
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where={"personality_type": personality_type}
            )
            
            self.logger.info(
                "Queried %d results for personality '%s'",
                len(results["documents"][0]) if results["documents"] else 0,
                personality_type
            )
            
            return results
        except Exception as e:
            self.logger.error("Failed to query by personality: %s", str(e))
            raise
    
    def get_all_personalities(self) -> List[str]:
        """Get all unique personality types in the store."""
        try:
            # Get all documents
            with tqdm(desc="Loading personalities", unit="step") as pbar:
                all_docs = self.collection.get()
                pbar.update(1)
            
            # Extract unique personality types
            personalities = set()
            for metadata in all_docs.get("metadatas", []):
                if metadata and "personality_type" in metadata:
                    personalities.add(metadata["personality_type"])
            
            return sorted(list(personalities))
        except Exception as e:
            self.logger.error("Failed to get personalities: %s", str(e))
            raise
    
    def clear_collection(self) -> None:
        """Clear all data from the collection."""
        try:
            with tqdm(total=2, desc="Clearing collection", unit="step") as pbar:
                # Delete collection
                self.client.delete_collection(config.chromadb.collection_name)
                pbar.update(1)
                
                # Recreate collection
                self.collection = self.client.create_collection(
                    name=config.chromadb.collection_name
                )
                pbar.update(1)
            
            self.logger.info("Collection cleared successfully")
        except Exception as e:
            self.logger.error("Failed to clear collection: %s", str(e))
            raise