import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any

from settings.config import CHROMA_DB_DIR, COLLECTION_NAME
from logging_service.logger import LoggerService

logger = LoggerService(__name__).get_logger()

class ChromaDB:
    def __init__(self) -> None:
        self._initialize_client()
        self._initialize_collection()
        logger.info("ChromaDB initialized and collection '%s' ready", COLLECTION_NAME)
        
    def _initialize_client(self) -> None:
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_DIR),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        logger.debug("ChromaDB client initialized with persist directory: %s", str(CHROMA_DB_DIR))

    def _initialize_collection(self) -> None:
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        logger.debug("Collection '%s' fetched or created", COLLECTION_NAME)

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, str | int | float | bool]],
        ids: List[str]
    ) -> None:
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        logger.info("Added %d documents to the collection '%s'", len(documents), COLLECTION_NAME)
