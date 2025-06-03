import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional, Union, Tuple
from contextlib import contextmanager
import time
from pathlib import Path

# Handle ChromaDB error imports - they might not be available in all versions
try:
    from chromadb.errors import DuplicateIDError, InvalidCollectionException
except ImportError:
    # Fallback for older ChromaDB versions
    class DuplicateIDError(Exception):
        pass
    class InvalidCollectionException(Exception):
        pass

from config.settings import CHROMA_DB_DIR, COLLECTION_NAME
from backend.utils.logger import LoggerService

logger = LoggerService(__name__).get_logger()

class ChromaDB:
    def __init__(self, collection_name: Optional[str] = None) -> None:
        self.collection_name = collection_name or COLLECTION_NAME
        self.client = None
        self.collection = None
        self._initialize_client()
        self._initialize_collection()
        logger.info("ChromaDB initialized and collection '%s' ready", self.collection_name)
        
    def _initialize_client(self) -> None:
        """Initialize ChromaDB client with optimized settings"""
        try:
            # Ensure directory exists
            chroma_path = Path(CHROMA_DB_DIR)
            chroma_path.mkdir(parents=True, exist_ok=True)
            
            # Basic settings that work across ChromaDB versions
            settings = ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
            
            self.client = chromadb.PersistentClient(
                path=str(CHROMA_DB_DIR),
                settings=settings
            )
            logger.info("ChromaDB client initialized with persist directory: %s", str(CHROMA_DB_DIR))
            
        except Exception as e:
            logger.error("Failed to initialize ChromaDB client: %s", str(e))
            raise

    def _initialize_collection(self) -> None:
        """Initialize collection with metadata configuration"""
        try:
            # Check if collection exists
            existing_collections = [col.name for col in self.client.list_collections()]
            
            if self.collection_name in existing_collections:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info("Retrieved existing collection '%s' with %d documents", 
                          self.collection_name, self.collection.count())
            else:
                # Create collection with basic metadata
                collection_metadata = {
                    "description": "Document embeddings collection",
                    "created_at": str(time.time())
                }
                
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata=collection_metadata
                )
                logger.info("Created new collection '%s'", self.collection_name)
                
        except Exception as e:
            logger.error("Failed to initialize collection '%s': %s", self.collection_name, str(e))
            raise

    @contextmanager
    def _error_handler(self, operation: str):
        """Context manager for consistent error handling"""
        try:
            start_time = time.time()
            yield
            elapsed = time.time() - start_time
            logger.debug("Operation '%s' completed in %.2f seconds", operation, elapsed)
        except DuplicateIDError as e:
            logger.warning("Duplicate ID error in %s: %s", operation, str(e))
            raise
        except InvalidCollectionException as e:
            logger.error("Invalid collection error in %s: %s", operation, str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error in %s: %s", operation, str(e))
            raise

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Union[str, int, float, bool]]],
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        batch_size: int = 1000,
        skip_duplicates: bool = True
    ) -> Dict[str, Any]:
        """
        Add documents to the collection with improved batch processing and error handling
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: List of document IDs
            embeddings: Optional pre-computed embeddings
            batch_size: Batch size for processing
            skip_duplicates: Whether to skip documents with duplicate IDs
            
        Returns:
            Dictionary with operation statistics
        """
        if not documents or len(documents) != len(metadatas) or len(documents) != len(ids):
            raise ValueError("Documents, metadatas, and ids must have the same length and be non-empty")
        
        if embeddings and len(embeddings) != len(documents):
            raise ValueError("Embeddings length must match documents length")

        stats = {
            "total_documents": len(documents),
            "added_documents": 0,
            "skipped_duplicates": 0,
            "failed_documents": 0,
            "batches_processed": 0
        }

        with self._error_handler("add_documents"):
            # Handle duplicates if requested
            if skip_duplicates:
                documents, metadatas, ids, embeddings = self._filter_duplicates(
                    documents, metadatas, ids, embeddings, stats
                )
                
            if not documents:  # All were duplicates
                logger.info("No new documents to add after duplicate filtering")
                return stats

            # Process in batches for better performance
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            for i in range(0, len(documents), batch_size):
                batch_end = min(i + batch_size, len(documents))
                batch_docs = documents[i:batch_end]
                batch_metas = metadatas[i:batch_end]
                batch_ids = ids[i:batch_end]
                batch_embeddings = embeddings[i:batch_end] if embeddings else None
                
                try:
                    add_kwargs = {
                        "documents": batch_docs,
                        "metadatas": batch_metas,
                        "ids": batch_ids
                    }
                    
                    if batch_embeddings:
                        add_kwargs["embeddings"] = batch_embeddings
                    
                    self.collection.add(**add_kwargs)
                    
                    stats["added_documents"] += len(batch_docs)
                    stats["batches_processed"] += 1
                    
                    logger.debug("Processed batch %d/%d: added %d documents", 
                               stats["batches_processed"], total_batches, len(batch_docs))
                    
                except Exception as e:
                    stats["failed_documents"] += len(batch_docs)
                    logger.error("Failed to add batch %d: %s", stats["batches_processed"] + 1, str(e))
                    if not skip_duplicates:  # Re-raise if not handling duplicates gracefully
                        raise

        logger.info("Added %d documents to collection '%s' (skipped %d duplicates, %d failed)", 
                   stats["added_documents"], self.collection_name, 
                   stats["skipped_duplicates"], stats["failed_documents"])
        
        return stats

    def _filter_duplicates(
        self, 
        documents: List[str], 
        metadatas: List[Dict], 
        ids: List[str],
        embeddings: Optional[List[List[float]]],
        stats: Dict[str, Any]
    ) -> Tuple[List[str], List[Dict], List[str], Optional[List[List[float]]]]:
        """Filter out documents with existing IDs"""
        try:
            # Get existing IDs in batches to avoid memory issues
            existing_ids = set()
            batch_size = 1000
            
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i + batch_size]
                try:
                    result = self.collection.get(ids=batch_ids, include=[])
                    existing_ids.update(result.get('ids', []))
                except Exception:
                    # If get fails, assume none exist in this batch
                    continue
            
            # Filter out duplicates
            filtered_docs, filtered_metas, filtered_ids, filtered_embeddings = [], [], [], []
            
            for idx, doc_id in enumerate(ids):
                if doc_id in existing_ids:
                    stats["skipped_duplicates"] += 1
                    logger.debug("Skipping duplicate ID: %s", doc_id)
                else:
                    filtered_docs.append(documents[idx])
                    filtered_metas.append(metadatas[idx])
                    filtered_ids.append(doc_id)
                    if embeddings:
                        filtered_embeddings.append(embeddings[idx])
            
            return (filtered_docs, filtered_metas, filtered_ids, 
                   filtered_embeddings if embeddings else None)
                   
        except Exception as e:
            logger.warning("Failed to filter duplicates, proceeding without filtering: %s", str(e))
            return documents, metadatas, ids, embeddings

    def query_documents(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Query documents from the collection with comprehensive options
        
        Args:
            query_texts: List of query texts
            query_embeddings: List of query embeddings
            n_results: Number of results to return
            where: Metadata filter conditions
            where_document: Document content filter conditions
            include: What to include in results
            
        Returns:
            Query results
        """
        if not query_texts and not query_embeddings:
            raise ValueError("Either query_texts or query_embeddings must be provided")
        
        if include is None:
            include = ["documents", "metadatas", "distances"]
        
        with self._error_handler("query_documents"):
            query_kwargs = {
                "n_results": n_results,
                "include": include
            }
            
            if query_texts:
                query_kwargs["query_texts"] = query_texts
            if query_embeddings:
                query_kwargs["query_embeddings"] = query_embeddings
            if where:
                query_kwargs["where"] = where
            if where_document:
                query_kwargs["where_document"] = where_document
            
            results = self.collection.query(**query_kwargs)
            
            logger.debug("Query returned %d result sets", len(results.get('ids', [])))
            return results

    def update_documents(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Union[str, int, float, bool]]]] = None,
        embeddings: Optional[List[List[float]]] = None
    ) -> None:
        """Update existing documents in the collection"""
        with self._error_handler("update_documents"):
            update_kwargs = {"ids": ids}
            
            if documents:
                update_kwargs["documents"] = documents
            if metadatas:
                update_kwargs["metadatas"] = metadatas
            if embeddings:
                update_kwargs["embeddings"] = embeddings
            
            self.collection.update(**update_kwargs)
            logger.info("Updated %d documents in collection '%s'", len(ids), self.collection_name)

    def delete_documents(
        self, 
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> None:
        """Delete documents from the collection"""
        if not ids and not where:
            raise ValueError("Either ids or where condition must be provided")
        
        with self._error_handler("delete_documents"):
            delete_kwargs = {}
            if ids:
                delete_kwargs["ids"] = ids
            if where:
                delete_kwargs["where"] = where
            
            self.collection.delete(**delete_kwargs)
            logger.info("Deleted documents from collection '%s'", self.collection_name)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive collection statistics"""
        try:
            count = self.collection.count()
            collection_metadata = self.collection.metadata or {}
            
            # Sample some documents to get metadata keys
            sample_size = min(100, count) if count > 0 else 0
            metadata_keys: set = set()
            
            if sample_size > 0:
                try:
                    sample = self.collection.peek(limit=sample_size)
                    for meta in sample.get('metadatas', []):
                        if meta:
                            metadata_keys.update(meta.keys())
                except AttributeError:
                    # peek might not be available in older versions
                    logger.debug("peek method not available, skipping metadata keys sampling")
            
            stats = {
                "collection_name": self.collection_name,
                "document_count": count,
                "collection_metadata": collection_metadata,
                "sample_metadata_keys": list(metadata_keys),
                "database_path": str(CHROMA_DB_DIR)
            }
            
            logger.info("Collection stats: %d documents in '%s'", count, self.collection_name)
            return stats
            
        except Exception as e:
            logger.error("Failed to get collection stats: %s", str(e))
            return {"error": str(e)}

    def reset_collection(self) -> None:
        """Reset (clear) the collection - use with caution!"""
        with self._error_handler("reset_collection"):
            self.client.delete_collection(name=self.collection_name)
            self._initialize_collection()
            logger.warning("Collection '%s' has been reset", self.collection_name)

    def close(self) -> None:
        """Close the database connection"""
        if self.client:
            # ChromaDB doesn't have explicit close, but we can clear references
            self.collection = None
            self.client = None
            logger.info("ChromaDB connection closed")

    def __enter__(self) -> 'ChromaDB':
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()