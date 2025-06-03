import json
import logging
from pathlib import Path

import numpy as np
from chromadb import PersistentClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def export_chromadb_to_json_file(
    chroma_path: str = "./chroma_db",
    output_json_path: str = "./chroma_data_export.json"
):
    client = PersistentClient(path=chroma_path)
    collections = client.list_collections()

    if not collections:
        logger.info("No collections found in ChromaDB at %s", chroma_path)
        return

    all_data = {}
    for col in collections:
        logger.info("Collecting data from collection: %s", col.name)
        collection = client.get_collection(name=col.name)
        data = collection.get(include=["documents", "metadatas"])

        all_data[col.name] = data

    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4)

    logger.info("ChromaDB data successfully exported to %s", output_json_path)


if __name__ == "__main__":
    export_chromadb_to_json_file(
        chroma_path="./chroma_db",
        output_json_path="./src/utils/chroma_data_export.json"
    )
