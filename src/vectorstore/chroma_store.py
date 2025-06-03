import json
import uuid
from typing import List, Dict, Any, Optional, Union
import numpy as np
import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Metadata, OneOrMany, GetResult
from chromadb.api import ClientAPI

from chromadb.config import Settings as ChromaSettings
from tqdm import tqdm
import time

from config.settings import settings
from src.embeddings.embedder import HuggingFaceEmbedder
from src.utils.logger import logger


class PersonalityVectorStore:
    """ChromaDB-based vector store for personality data with comprehensive progress tracking"""
    
    def __init__(self, embedder: Optional[HuggingFaceEmbedder] = None) -> None:
        """
        Initialize the vector store
        
        Args:
            embedder: HuggingFace embedder instance
        """
        self.embedder: HuggingFaceEmbedder = embedder or HuggingFaceEmbedder()
        self._client: Optional[chromadb.PersistentClient] = None
        self._collections: Dict[str, Collection] = {}
        
        logger.info("Initializing ChromaDB vector store")
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize ChromaDB client with proper error handling."""
        try:
            from chromadb.config import Settings as ChromaSettings
            import chromadb
            
            # Convert Path to string if necessary
            persist_dir = str(settings.chroma_persist_directory)
            
            chroma_settings = ChromaSettings(
                persist_directory=persist_dir,  # Use string conversion
                anonymized_telemetry=False
            )
            
            self.client = chromadb.PersistentClient(
                path=persist_dir,  # Also use string here
                settings=chroma_settings
            )
            
            logger.info(f"ChromaDB client initialized successfully at: {persist_dir}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
    
    def _get_collection_name(self, personality: str) -> str:
        """Generate collection name for a personality"""
        clean_personality = personality.replace(" ", "_").replace("-", "_").lower()
        return f"{settings.chroma_collection_prefix}{clean_personality}"
    
    def _get_or_create_collection(self, personality: str) -> Collection:
        """Get or create a collection for a specific personality with progress tracking"""
        collection_name = self._get_collection_name(personality)

        # Ensure the client is initialized
        if self._client is None:
            self._initialize_client()
        
        if collection_name not in self._collections:
            try:
                if self._client is None:
                    raise RuntimeError("ChromaDB client is not initialized")
                
                print(f"📁 Creating collection for personality: {personality}")
                
                with tqdm(
                    total=100,
                    desc=f"Collection: {personality[:20]}...",
                    unit="%",
                    leave=False
                ) as pbar:
                    pbar.set_description("Creating collection...")
                    self._collections[collection_name] = self._client.get_or_create_collection(
                        name=collection_name,
                        metadata={"personality": personality}
                    )
                    pbar.update(100)
                
                logger.info("Collection created/retrieved for personality: %s", personality)
                
            except Exception as e:
                logger.error("Failed to create collection for %s: %s", personality, str(e))
                raise
        
        return self._collections[collection_name]
    
    def add_personality_data(
        self, 
        personality: str, 
        messages: List[str], 
        metadata: Optional[Dict[str, Any]] = None,
        batch_size: int = 100,
        show_progress: bool = True
    ) -> None:
        """Add messages for a specific personality to the vector store with progress tracking"""
        if not messages:
            logger.warning("No messages provided for personality: %s", personality)
            return
        
        collection = self._get_or_create_collection(personality)
        total_messages = len(messages)
        
        if show_progress:
            print(f"💾 Adding {total_messages} messages for {personality}")
        
        try:
            # Process in batches for large datasets
            if total_messages > batch_size:
                self._add_data_in_batches(
                    collection, personality, messages, metadata, batch_size, show_progress
                )
            else:
                self._add_single_batch(
                    collection, personality, messages, metadata, show_progress
                )
            
            if show_progress:
                print(f"✅ Successfully added {total_messages} messages for {personality}")
            
            logger.info("Added %d messages for personality: %s", total_messages, personality)
            
        except Exception as e:
            logger.error("Failed to add data for personality %s: %s", personality, str(e))
            raise
    
    def convert_to_metadatas(
        self,
        metadata: Optional[Dict[str, Any]],
        count: int
    ) -> List[Metadata]:
        """Convert a single metadata dict to a list of valid Metadata dicts (ChromaDB compliant)."""
        if metadata is None:
            return [{} for _ in range(count)]

        # Only allow valid types (str, int, float, bool)
        safe_metadata = {
            k: v for k, v in metadata.items()
            if isinstance(k, str) and isinstance(v, (str, int, float, bool))
        }

        return [safe_metadata.copy() for _ in range(count)]

    def convert_to_embeddings(
        self,
        raw_embeddings: List[List[float]],
        expected_count: int
    ) -> List[List[float]]:
        """Ensure embeddings are valid and match the expected count"""
        if not isinstance(raw_embeddings, list):
            raise ValueError("Embeddings must be a list of vectors.")
        if len(raw_embeddings) != expected_count:
            raise ValueError(f"Expected {expected_count} embeddings, got {len(raw_embeddings)}.")
        
        # Optionally: Ensure all elements are lists of floats
        for vec in raw_embeddings:
            if not all(isinstance(val, (float, int)) for val in vec):
                raise TypeError("Each embedding must be a list of floats or ints.")
        
        # Cast all to float
        return [list(map(float, vec)) for vec in raw_embeddings]

    def _add_single_batch(
        self,
        collection: Collection,
        personality: str,
        messages: List[str],
        metadata: Optional[Dict[str, Any]],
        show_progress: bool
    ) -> None:
        """Add a single batch of messages with progress tracking"""
        
        progress_desc = f"Processing {personality[:15]}..."
        
        if show_progress:
            with tqdm(
                total=3, 
                desc=progress_desc,
                bar_format="{l_bar}{bar}| {desc}",
                leave=False
            ) as pbar:
                # Generate embeddings
                pbar.set_description("Generating embeddings...")
                embeddings = self.embedder.embed_text(messages, show_progress=False)
                pbar.update(1)
                
                # Prepare data
                pbar.set_description("Preparing data...")
                ids, metadatas = self._prepare_data(messages, personality, metadata)
                pbar.update(1)
                metadatas = self.convert_to_metadatas(metadata, count=len(messages))
                # Store in ChromaDB
                pbar.set_description("Storing in database...")
                embeddings_array = np.array(embeddings) if isinstance(embeddings, list) else embeddings
                self._store_in_chromadb(collection, embeddings_array, messages, metadatas, ids)
                pbar.update(1)
        else:
            # Process without progress bars
            embeddings = self.embedder.embed_text(messages, show_progress=False)
            ids, metadatas = self._prepare_data(messages, personality, metadata)
            metadatas = self.convert_to_metadatas(metadata, count=len(messages))
            embeddings_array = np.array(embeddings) if isinstance(embeddings, list) else embeddings
            self._store_in_chromadb(collection, embeddings_array, messages, metadatas, ids)
    
    def _add_data_in_batches(
        self,
        collection: Collection,
        personality: str,
        messages: List[str],
        metadata: Optional[Dict[str, Any]],
        batch_size: int,
        show_progress: bool
    ) -> None:
        """Add data in batches with comprehensive progress tracking"""
        
        total_messages = len(messages)
        num_batches = (total_messages + batch_size - 1) // batch_size
        
        if show_progress:
            print(f"📦 Processing in {num_batches} batches of {batch_size} messages each")
        
        # Main progress bar for all batches
        batch_pbar = tqdm(
            total=num_batches,
            desc=f"Batches for {personality[:15]}...",
            unit="batch",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} batches [{elapsed}<{remaining}, {rate_fmt}]"
        ) if show_progress else None
        
        try:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_messages)
                batch_messages = messages[start_idx:end_idx]
                
                # Process this batch
                if show_progress and batch_pbar:
                    batch_pbar.set_postfix({
                        'messages': f'{start_idx+1}-{end_idx}',
                        'size': len(batch_messages)
                    })
                
                # Generate embeddings for batch
                embeddings = self.embedder.embed_text(batch_messages, show_progress=False)
                
                # Prepare data for batch
                ids, metadatas = self._prepare_data(batch_messages, personality, metadata, start_idx)
                
                # Store batch in ChromaDB
                metadatas = self.convert_to_metadatas(metadata, count=len(messages))
                embeddings_array = np.array(embeddings) if isinstance(embeddings, list) else embeddings
                self._store_in_chromadb(collection, embeddings_array, messages, metadatas, ids)
                
                # Update batch progress
                if batch_pbar:
                    batch_pbar.update(1)
        
        finally:
            if batch_pbar:
                batch_pbar.close()
    
    def _prepare_data(
        self, 
        messages: List[str], 
        personality: str, 
        metadata: Optional[Dict[str, Any]] = None,
        start_index: int = 0
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """Prepare IDs and metadata for messages"""
        ids = [str(uuid.uuid4()) for _ in messages]
        metadatas: List[Dict[str, Any]] = []
        
        for i, message in enumerate(messages):
            msg_metadata: Dict[str, Any] = {
                "personality": personality,
                "message_index": start_index + i,
                "character_count": len(message),
                "word_count": len(message.split()),
                "batch_id": f"{personality}_{start_index//100}"
            }
            if metadata:
                msg_metadata.update(metadata)
            metadatas.append(msg_metadata)
        
        return ids, metadatas
    
    def _store_in_chromadb(
        self,
        collection: Collection,
        embeddings: np.ndarray,
        messages: List[str],
        metadatas: OneOrMany[Metadata],
        ids: List[str]
    ) -> None:
        """Store data in ChromaDB"""
        # Ensure embeddings is a numpy array
        if not isinstance(embeddings, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(embeddings)}")
        
        # Convert embeddings to list format for ChromaDB
        embeddings_list = embeddings.tolist()
        
        # Add to collection
        collection.add(
            embeddings=embeddings_list,
            documents=messages,
            metadatas=metadatas,
            ids=ids
        )
    
    def load_personality_data_from_json(self, json_path: Optional[str] = None) -> None:
        """Load personality data from JSON file with comprehensive progress tracking"""
        json_path = str(json_path or settings.personality_data_path)
        
        print(f"📂 Loading personality data from: {json_path}")
        
        try:
            # Load and parse JSON
            with tqdm(desc="Loading JSON", unit="B", unit_scale=True, leave=False) as pbar:
                with open(json_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                pbar.update(1)
            
            conversations = data.get('conversations', [])
            total_conversations = len(conversations)
            
            if total_conversations == 0:
                print("⚠️  No conversations found in the data file")
                return
            
            print(f"📊 Found {total_conversations} conversations to process")
            
            # Group conversations by personality for efficient processing
            personality_groups = self._group_conversations_by_personality(conversations)
            
            print(f"👥 Processing {len(personality_groups)} different personalities")
            
            # Process each personality group
            overall_pbar = tqdm(
                total=len(personality_groups),
                desc="Processing Personalities",
                unit="personality",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} personalities [{elapsed}<{remaining}]"
            )
            
            try:
                for personality, convs in personality_groups.items():
                    overall_pbar.set_postfix({'current': personality[:20]})
                    
                    # Extract all messages and metadata for this personality
                    all_messages = []
                    base_metadata = {}
                    
                    for conv in convs:
                        messages = conv.get('messages', [])
                        if messages:
                            all_messages.extend(messages)
                            
                            # Use metadata from first conversation as base
                            if not base_metadata:
                                base_metadata = {
                                    "created_date": conv.get('created_date', ''),
                                    "total_conversations": len(convs)
                                }
                                
                                # Add professional traits as metadata
                                traits = conv.get('professional_personality', {}).get('professional_traits', {})
                                base_metadata.update({f"trait_{k}": v for k, v in traits.items()})
                    
                    if all_messages:
                        # Add this personality's data
                        self.add_personality_data(
                            personality, 
                            all_messages, 
                            base_metadata,
                            show_progress=False  # We're showing overall progress
                        )
                    
                    overall_pbar.update(1)
            
            finally:
                overall_pbar.close()
            
            print(f"✅ Successfully loaded personality data")
            print(f"📈 Statistics:")
            print(f"   - Total conversations: {total_conversations}")
            print(f"   - Personalities: {len(personality_groups)}")
            print(f"   - Total messages: {sum(len(group) for group in personality_groups.values())}")
            
            logger.info("Successfully loaded personality data from JSON")
            
        except FileNotFoundError:
            logger.error("Personality data file not found: %s", json_path)
            raise
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON format in file %s: %s", json_path, str(e))
            raise
        except Exception as e:
            logger.error("Failed to load personality data from %s: %s", json_path, str(e))
            raise
    
    def _group_conversations_by_personality(self, conversations: List[Dict]) -> Dict[str, List[Dict]]:
        """Group conversations by personality type"""
        personality_groups: Dict[str, List[Dict]] = {}
        
        with tqdm(
            conversations, 
            desc="Grouping conversations",
            unit="conv",
            leave=False
        ) as pbar:
            for conv in pbar:
                personality_data = conv.get('professional_personality', {})
                personality = personality_data.get('primary_profession_type')
                
                if personality and conv.get('messages'):
                    if personality not in personality_groups:
                        personality_groups[personality] = []
                    personality_groups[personality].append(conv)
                    
                    pbar.set_postfix({'personalities': len(personality_groups)})
        
        return personality_groups
    
    def query_personality(
        self, 
        personality: str, 
        query: str, 
        top_k: Optional[int] = None,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """Query messages for a specific personality with progress tracking"""
        top_k = top_k or settings.retrieval_top_k
        collection = self._get_or_create_collection(personality)
        
        if show_progress:
            print(f"🔍 Searching {personality} for: '{query[:50]}...'")
        
        try:
            if show_progress:
                with tqdm(
                    total=3,
                    desc="Searching",
                    bar_format="{l_bar}{bar}| {desc}",
                    leave=False
                ) as pbar:
                    # Generate query embedding
                    pbar.set_description("Generating query embedding...")
                    query_embedding = self.embedder.embed_text(query, show_progress=False)
                    pbar.update(1)
                    
                    # Ensure query_embedding is a numpy array and convert to list
                    if not isinstance(query_embedding, np.ndarray):
                        raise TypeError(f"Expected numpy array, got {type(query_embedding)}")
                    
                    query_embedding_list = query_embedding.tolist()
                    pbar.update(1)
                    
                    # Query the collection
                    pbar.set_description("Searching database...")
                    results = collection.query(
                        query_embeddings=[query_embedding_list],
                        n_results=top_k
                    )
                    pbar.update(1)
            else:
                # Query without progress bar
                query_embedding = self.embedder.embed_text(query, show_progress=False)
                if not isinstance(query_embedding, np.ndarray):
                    raise TypeError(f"Expected numpy array, got {type(query_embedding)}")
                query_embedding_list = query_embedding.tolist()
                results = collection.query(
                    query_embeddings=[query_embedding_list],
                    n_results=top_k
                )
            
            # Format results
            formatted_results: List[Dict[str, Any]] = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    result: Dict[str, Any] = {
                        'document': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                        'distance': results['distances'][0][i] if results['distances'] and results['distances'][0] else None,
                        'similarity': 1 - results['distances'][0][i] if results['distances'] and results['distances'][0] else None,
                        'id': results['ids'][0][i] if results['ids'] and results['ids'][0] else None
                    }
                    formatted_results.append(result)
            
            if show_progress:
                print(f"✅ Found {len(formatted_results)} relevant results")
            
            logger.info("Retrieved %d results for personality %s", len(formatted_results), personality)
            return formatted_results
            
        except Exception as e:
            logger.error("Failed to query personality %s: %s", personality, str(e))
            raise
    
    def get_available_personalities(self, show_progress: bool = False) -> List[str]:
        """Get list of available personalities in the vector store"""
        try:
            if self._client is None:
                logger.warning("ChromaDB client not initialized")
                return []
            
            if show_progress:
                with tqdm(desc="Scanning collections", leave=False) as pbar:
                    collections = self._client.list_collections()
                    pbar.update(1)
            else:
                collections = self._client.list_collections()
            
            personalities: List[str] = []
            
            collection_pbar = tqdm(
                collections, 
                desc="Loading personalities",
                unit="collection",
                leave=False
            ) if show_progress else collections
            
            for collection in collection_pbar:
                if collection.name.startswith(settings.chroma_collection_prefix):
                    personality = collection.metadata.get('personality') if collection.metadata else None
                    if personality:
                        personalities.append(personality)
            
            if show_progress:
                print(f"📊 Found {len(personalities)} personalities in vector store")
            
            logger.info("Found %d personalities in vector store", len(personalities))
            return personalities
            
        except Exception as e:
            logger.error("Failed to get available personalities: %s", str(e))
            return []
    
    def get_personality_stats(self, personality: str, show_progress: bool = True) -> Dict[str, Any]:
        """Get comprehensive statistics for a specific personality"""
        try:
            collection = self._get_or_create_collection(personality)
            
            if show_progress:
                with tqdm(
                    total=3,
                    desc=f"Analyzing {personality[:15]}...",
                    leave=False
                ) as pbar:
                    # Get basic count
                    pbar.set_description("Getting document count...")
                    count = collection.count()
                    pbar.update(1)
                    
                    # Get sample documents
                    pbar.set_description("Sampling documents...")
                    sample_size = min(100, count)
                    sample_results = collection.get(limit=sample_size)
                    pbar.update(1)
                    
                    # Analyze metadata
                    pbar.set_description("Analyzing metadata...")
                    stats = self._analyze_personality_metadata(
                        personality, count, sample_results
                    )
                    pbar.update(1)
            else:
                count = collection.count()
                sample_size = min(100, count)
                sample_results = collection.get(limit=sample_size)
                stats = self._analyze_personality_metadata(
                    personality, count, sample_results
                )
            
            return stats
            
        except Exception as e:
            logger.error("Failed to get stats for personality %s: %s", personality, str(e))
            return {
                "personality": personality,
                "error": str(e),
                "total_documents": 0
            }

    def _analyze_personality_metadata(
        self, 
        personality: str, 
        count: int, 
        sample_results: 'GetResult'
    ) -> Dict[str, Any]:
        """Analyze personality metadata and return statistics"""
        stats: Dict[str, Any] = {
            "personality": personality,
            "total_documents": count,
            "collection_name": self._get_collection_name(personality),
            "sample_size": len(sample_results['documents']) if sample_results['documents'] else 0
        }
        
        # Analyze metadata if available
        if sample_results['metadatas']:
            char_counts = []
            word_counts = []
            traits = {}
            
            for meta in sample_results['metadatas']:
                if isinstance(meta.get('character_count'), int):
                    char_counts.append(meta['character_count'])
                if isinstance(meta.get('word_count'), int):
                    word_counts.append(meta['word_count'])
                
                # Extract trait scores
                for key, value in meta.items():
                    if key.startswith('trait_') and isinstance(value, (int, float)):
                        trait_name = key.replace('trait_', '')
                        if trait_name not in traits:
                            traits[trait_name] = []
                        traits[trait_name].append(value)
            
            # Calculate averages
            if char_counts:
                stats.update({
                    "avg_character_count": np.mean(char_counts),
                    "min_character_count": min(char_counts),
                    "max_character_count": max(char_counts)
                })
            
            if word_counts:
                stats.update({
                    "avg_word_count": np.mean(word_counts),
                    "min_word_count": min(word_counts),
                    "max_word_count": max(word_counts)
                })
            
            # Average trait scores
            if traits:
                avg_traits = {trait: np.mean(scores) for trait, scores in traits.items()}
                stats["average_traits"] = avg_traits
                stats["top_traits"] = sorted(avg_traits.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return stats