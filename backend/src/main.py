from vector_data_handler.processor import MessageProcessor
from vector_data_handler.db import ChromaDB

def main():
    processor = MessageProcessor()
    db = ChromaDB()

    data = processor.load_data(r"C:\Projects\digitalMe\backend\src\data\pruned.json")
    documents, metadatas, ids = processor.prepare_documents(data)
    embeddings = processor.embed_documents(documents, batch_size=8)

    db.collection.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)

if __name__ == "__main__":
    main()
