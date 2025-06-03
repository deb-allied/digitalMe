import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

from vector_data_handler.processor import MessageProcessor
from vector_data_handler.db import ChromaDB
from config.settings import OPENAI_API_KEY, COLLECTION_NAME
from backend.utils.logger import LoggerService

# Import OpenAI for LLM integration
try:
    from openai import OpenAI
except ImportError:
    print("OpenAI package not installed. Install with: pip install openai")
    OpenAI = None

logger = LoggerService(__name__).get_logger()

class RAGSystem:
    """
    Complete RAG (Retrieval-Augmented Generation) System
    Handles document ingestion, retrieval, and context-aware generation
    """
    
    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.processor = MessageProcessor()
        self.db = ChromaDB(collection_name)
        self.openai_client = None
        self.similarity_threshold = 0.3  # Lowered default threshold
        
        # Initialize OpenAI client if available
        if OpenAI and OPENAI_API_KEY:
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info("OpenAI client initialized successfully")
        else:
            logger.warning("OpenAI client not available - LLM generation will be disabled")
    
    def debug_embeddings(self, sample_query: str = "test") -> Dict[str, Any]:
        """
        Debug embedding generation and storage
        """
        try:
            print("🔧 Running embedding diagnostics...")
            
            # Test query embedding
            query_embedding = self.processor.embed_documents([sample_query])[0]
            print(f"✅ Query embedding generated: shape={len(query_embedding)}")
            print(f"   Sample values: {query_embedding[:5]}")
            
            # Check database content
            stats = self.db.get_collection_stats()
            print(f"✅ Database stats: {stats.get('document_count', 0)} documents")
            
            # Try a raw query without threshold
            raw_results = self.db.query_documents(
                query_embeddings=[query_embedding],
                n_results=3,
                include=["documents", "metadatas", "distances"]
            )
            
            print(f"✅ Raw query results: {len(raw_results.get('distances', [[]])[0])} documents")
            
            if raw_results.get('distances') and raw_results['distances'][0]:
                print("   Distance samples:")
                for i, dist in enumerate(raw_results['distances'][0][:3]):
                    doc_preview = raw_results['documents'][0][i][:50] if raw_results.get('documents') else "N/A"
                    print(f"   {i+1}: distance={dist:.3f}, doc='{doc_preview}...'")
            
            return {
                "query_embedding_shape": len(query_embedding),
                "database_document_count": stats.get('document_count', 0),
                "raw_results_count": len(raw_results.get('distances', [[]])[0])
            }
            
        except Exception as e:
            print(f"❌ Debug failed: {str(e)}")
            return {"error": str(e)}
    
    def ingest_documents(self, data_path: str, batch_size: int = 32, max_workers: int = 4) -> Dict[str, Any]:
        """
        Complete document ingestion pipeline
        
        Args:
            data_path: Path to the JSON data file
            batch_size: Batch size for embedding generation
            max_workers: Number of parallel workers
            
        Returns:
            Dictionary with ingestion statistics
        """
        logger.info("Starting document ingestion pipeline")
        start_time = time.time()
        
        try:
            # Load and prepare documents
            logger.info("Loading data from %s", data_path)
            data = self.processor.load_data(data_path)
            
            logger.info("Preparing documents for embedding")
            documents, metadatas, ids = self.processor.prepare_documents(data)
            
            if not documents:
                logger.error("No documents found in data file")
                return {"error": "No documents found", "success": False}
            
            # Generate embeddings
            logger.info("Generating embeddings for %d documents", len(documents))
            embeddings = self.processor.embed_documents(documents, batch_size=batch_size)
            
            if not embeddings:
                logger.error("Failed to generate embeddings")
                return {"error": "Embedding generation failed", "success": False}
            
            # Store in vector database
            logger.info("Storing documents and embeddings in vector database")
            stats = self.db.add_documents(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings,
                batch_size=1000,  # Larger batch for DB insertion
                skip_duplicates=True
            )
            
            elapsed_time = time.time() - start_time
            
            result = {
                "success": True,
                "total_documents": len(documents),
                "embeddings_generated": len(embeddings),
                "ingestion_stats": stats,
                "elapsed_time": elapsed_time,
                "documents_per_second": len(documents) / elapsed_time
            }
            
            logger.info("Document ingestion completed successfully in %.2f seconds", elapsed_time)
            logger.info("Ingested %d documents (%.2f docs/sec)", 
                       len(documents), result["documents_per_second"])
            
            return result
            
        except Exception as e:
            logger.error("Document ingestion failed: %s", str(e))
            return {"error": str(e), "success": False}
    
    def retrieve_similar_documents(
        self, 
        query: str, 
        n_results: int = 5,
        similarity_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        debug: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar documents based on query
        
        Args:
            query: Query text
            n_results: Number of results to retrieve
            similarity_threshold: Minimum similarity score (0-1)
            metadata_filter: Optional metadata filtering
            debug: Enable debug logging
            
        Returns:
            List of similar documents with metadata and scores
        """
        try:
            # Generate query embedding
            logger.debug("Generating embedding for query: %s", query[:50] + "...")
            query_embedding = self.processor.embed_documents([query])[0]
            
            if debug:
                print(f"🔍 Query embedding shape: {len(query_embedding)}")
                print(f"🔍 Query embedding sample: {query_embedding[:5]}")
            
            # Search vector database
            search_results = self.db.query_documents(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=metadata_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            if debug:
                print(f"🔍 Raw search results keys: {search_results.keys()}")
                if 'distances' in search_results and search_results['distances']:
                    print(f"🔍 Found {len(search_results['distances'][0])} results")
                    print(f"🔍 Distance range: {min(search_results['distances'][0]):.3f} - {max(search_results['distances'][0]):.3f}")
            
            # Process and filter results
            similar_docs = []
            threshold = similarity_threshold or self.similarity_threshold
            
            if search_results.get('distances') and search_results['distances'][0]:
                for i, distance in enumerate(search_results['distances'][0]):
                    # ChromaDB uses different distance metrics - let's handle multiple cases
                    if distance <= 2.0:  # Likely cosine distance (0-2 range)
                        similarity = 1 - (distance / 2.0)  # Normalize to 0-1
                    else:  # Likely euclidean or other metric
                        # For very large distances, use exponential decay
                        similarity = 1 / (1 + distance)
                    
                    # Lower the threshold for debugging
                    debug_threshold = 0.3 if debug else threshold
                    
                    if similarity >= debug_threshold:
                        doc_info = {
                            "document": search_results['documents'][0][i],
                            "metadata": search_results['metadatas'][0][i],
                            "similarity_score": similarity,
                            "distance": distance,
                            "rank": i + 1
                        }
                        similar_docs.append(doc_info)
                        
                        if debug:
                            print(f"🔍 Doc {i+1}: distance={distance:.3f}, similarity={similarity:.3f}")
                            print(f"   Text preview: {search_results['documents'][0][i][:100]}...")
            
            if debug and not similar_docs:
                print("🔍 No documents found above threshold. Showing all results:")
                if search_results.get('distances') and search_results['distances'][0]:
                    for i, distance in enumerate(search_results['distances'][0]):
                        similarity = 1 - (distance / 2.0) if distance <= 2.0 else 1 / (1 + distance)
                        print(f"   Doc {i+1}: distance={distance:.3f}, similarity={similarity:.3f}")
                        print(f"   Text: {search_results['documents'][0][i][:100]}...")
            
            logger.info("Retrieved %d similar documents (threshold: %.2f)", 
                       len(similar_docs), threshold)
            
            return similar_docs
            
        except Exception as e:
            logger.error("Document retrieval failed: %s", str(e))
            if debug:
                print(f"🔍 Error details: {str(e)}")
            return []
    
    def generate_context_aware_response(
        self, 
        query: str, 
        context_docs: List[Dict[str, Any]],
        model: str = "gpt-3.5-turbo",
        max_context_length: int = 3000
    ) -> Dict[str, Any]:
        """
        Generate response using retrieved context
        
        Args:
            query: User query
            context_docs: Retrieved similar documents
            model: OpenAI model to use
            max_context_length: Maximum context length
            
        Returns:
            Dictionary with response and metadata
        """
        if not self.openai_client:
            return {
                "error": "OpenAI client not available",
                "response": "I can retrieve relevant documents but cannot generate responses without OpenAI API access."
            }
        
        try:
            # Prepare context from retrieved documents
            context_parts = []
            total_length = 0
            
            for doc in context_docs:
                doc_text = f"[Score: {doc['similarity_score']:.3f}] {doc['document']}"
                if total_length + len(doc_text) < max_context_length:
                    context_parts.append(doc_text)
                    total_length += len(doc_text)
                else:
                    break
            
            context = "\n\n".join(context_parts)
            
            # Create system prompt
            system_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 
            Use the context documents to provide accurate, relevant answers. If the context doesn't contain 
            enough information to answer the question, say so clearly.
            
            Context documents are prefixed with similarity scores in brackets [Score: X.XXX].
            Higher scores indicate more relevant documents."""
            
            # Create user prompt
            user_prompt = f"""Context Documents:
                            {context}

                            Question: {query}

                            Please provide a comprehensive answer based on the context above. Also add an inquisitive small question at the end based on the above context."""
            
            logger.debug("Generating response with %d context documents", len(context_parts))
            
            # Generate response
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            result = {
                "response": response.choices[0].message.content,
                "model_used": model,
                "context_documents_used": len(context_parts),
                "total_context_length": total_length,
                "token_usage": {
                    "prompt_tokens": getattr(getattr(response, 'usage', {}), 'prompt_tokens', None),
                    "completion_tokens": getattr(getattr(response, 'usage', {}), 'completion_tokens', None),
                    "total_tokens": getattr(getattr(response, 'usage', {}), 'total_tokens', None)
                }
            }
            
            logger.info("Generated response using %d context documents", len(context_parts))
            return result
            
        except Exception as e:
            logger.error("Response generation failed: %s", str(e))
            return {"error": str(e), "response": "Failed to generate response."}
    
    def query_rag_system(
        self, 
        query: str, 
        n_results: int = 5,
        similarity_threshold: float = 0.3,  # Lowered default threshold
        generate_response: bool = True,
        metadata_filter: Optional[Dict[str, Any]] = None,
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        Complete RAG query pipeline
        
        Args:
            query: User query
            n_results: Number of documents to retrieve
            similarity_threshold: Minimum similarity threshold
            generate_response: Whether to generate LLM response
            metadata_filter: Optional metadata filtering
            debug: Enable debug mode
            
        Returns:
            Complete RAG response with context and generation
        """
        logger.info("Processing RAG query: %s", query[:100] + "..." if len(query) > 100 else query)
        start_time = time.time()
        
        try:
            # Step 1: Retrieve similar documents
            similar_docs = self.retrieve_similar_documents(
                query=query,
                n_results=n_results,
                similarity_threshold=similarity_threshold,
                metadata_filter=metadata_filter,
                debug=debug
            )
            
            if not similar_docs:
                # Try with lower threshold
                if debug:
                    print("🔍 No results found, trying with very low threshold...")
                
                similar_docs = self.retrieve_similar_documents(
                    query=query,
                    n_results=n_results,
                    similarity_threshold=0.1,  # Very low threshold
                    metadata_filter=metadata_filter,
                    debug=debug
                )
                
                if not similar_docs:
                    return {
                        "query": query,
                        "retrieved_documents": [],
                        "response": "No relevant documents found for your query. Try rephrasing or check if documents are properly indexed.",
                        "debug_info": "Tried with threshold 0.1, still no results",
                        "elapsed_time": time.time() - start_time
                    }
            
            result = {
                "query": query,
                "retrieved_documents": similar_docs,
                "retrieval_stats": {
                    "documents_found": len(similar_docs),
                    "similarity_threshold": similarity_threshold,
                    "average_similarity": sum(doc["similarity_score"] for doc in similar_docs) / len(similar_docs),
                    "max_similarity": max(doc["similarity_score"] for doc in similar_docs),
                    "min_similarity": min(doc["similarity_score"] for doc in similar_docs)
                }
            }
            
            # Step 2: Generate response if requested
            if generate_response:
                generation_result = self.generate_context_aware_response(query, similar_docs)
                result.update(generation_result)
            
            result["elapsed_time"] = time.time() - start_time
            
            logger.info("RAG query completed in %.2f seconds", result["elapsed_time"])
            return result
            
        except Exception as e:
            logger.error("RAG query failed: %s", str(e))
            return {
                "query": query,
                "error": str(e),
                "elapsed_time": time.time() - start_time
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            db_stats = self.db.get_collection_stats()
            
            return {
                "database_stats": db_stats,
                "processor_info": {
                    "model_name": self.processor.model_name,
                    "device": str(self.processor.device),
                    "num_threads": getattr(self.processor, 'num_threads', 'unknown')
                },
                "system_config": {
                    "similarity_threshold": self.similarity_threshold,
                    "openai_available": self.openai_client is not None,
                    "collection_name": self.db.collection_name
                }
            }
        except Exception as e:
            logger.error("Failed to get system stats: %s", str(e))
            return {"error": str(e)}


def main():
    """Main function demonstrating complete RAG workflow"""
    
    # Initialize RAG system
    print("🚀 Initializing RAG System...")
    rag_system = RAGSystem()
    
    # Configuration
    data_path = r"C:\Projects\digitalMe\backend\src\data\pruned.json"
    
    try:
        # Step 1: Document Ingestion (one-time setup)
        print("\n📄 Starting Document Ingestion...")
        
        # Check if we need to ingest documents
        stats = rag_system.get_system_stats()
        current_doc_count = stats.get("database_stats", {}).get("document_count", 0)
        
        if current_doc_count == 0:
            print("No documents found in database. Starting ingestion...")
            ingestion_result = rag_system.ingest_documents(
                data_path=data_path,
                batch_size=32,  # Adjust based on your system
                max_workers=4   # Conservative for CPU
            )
            
            if ingestion_result.get("success"):
                print(f"✅ Successfully ingested {ingestion_result['total_documents']} documents")
                print(f"   Processing rate: {ingestion_result['documents_per_second']:.2f} docs/sec")
            else:
                print(f"❌ Ingestion failed: {ingestion_result.get('error')}")
                return
        else:
            print(f"📊 Found {current_doc_count} existing documents in database")
        
        # Step 2: Interactive RAG Query Loop
        print("\n🤖 RAG System Ready! Enter queries (commands: 'quit', 'stats', 'debug', 'test [query]')")
        print("=" * 70)
        
        while True:
            try:
                query = input("\n💬 Query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                elif query.lower() == 'stats':
                    stats = rag_system.get_system_stats()
                    print(f"\n📊 System Statistics:")
                    print(f"   Documents: {stats.get('database_stats', {}).get('document_count', 'unknown')}")
                    print(f"   Model: {stats.get('processor_info', {}).get('model_name', 'unknown')}")
                    print(f"   Device: {stats.get('processor_info', {}).get('device', 'unknown')}")
                    print(f"   OpenAI Available: {stats.get('system_config', {}).get('openai_available', False)}")
                    continue
                elif query.lower() == 'debug':
                    debug_info = rag_system.debug_embeddings()
                    print(f"\n🔧 Debug Information:")
                    for key, value in debug_info.items():
                        print(f"   {key}: {value}")
                    continue
                elif query.lower().startswith('test'):
                    # Test with debug mode
                    test_query = query[5:].strip() or "machine learning"
                    print(f"\n🧪 Testing with debug mode: '{test_query}'")
                    
                    result = rag_system.query_rag_system(
                        query=test_query,
                        n_results=5,
                        similarity_threshold=0.1,  # Very low threshold for testing
                        generate_response=False,   # Skip LLM for faster testing
                        debug=True
                    )
                    
                    if result.get("retrieved_documents"):
                        print(f"✅ Found {len(result['retrieved_documents'])} documents")
                        for doc in result['retrieved_documents'][:2]:
                            print(f"   Score: {doc['similarity_score']:.3f} - {doc['document'][:100]}...")
                    else:
                        print("❌ No documents found even with debug mode")
                        if result.get("debug_info"):
                            print(f"   Debug: {result['debug_info']}")
                    continue
                elif not query:
                    continue
                
                print(f"\n🔍 Processing query...")
                
                # Execute RAG query with lower threshold
                result = rag_system.query_rag_system(
                    query=query,
                    n_results=5,
                    similarity_threshold=0.3,  # Lower threshold for better results
                    generate_response=True
                )
                
                # Display results
                if result.get("error"):
                    print(f"❌ Error: {result['error']}")
                    continue
                
                print(f"\n📚 Retrieved {len(result['retrieved_documents'])} relevant documents:")
                for i, doc in enumerate(result['retrieved_documents'][:3], 1):  # Show top 3
                    print(f"   {i}. [Score: {doc['similarity_score']:.3f}] {doc['document'][:100]}...")
                
                if result.get("response"):
                    print(f"\n🤖 AI Response:")
                    print(f"   {result['response']}")
                    
                    if result.get("token_usage"):
                        tokens = result["token_usage"]
                        print(f"\n📊 Usage: {tokens['total_tokens']} tokens "
                              f"({tokens['prompt_tokens']} prompt + {tokens['completion_tokens']} completion)")
                
                print(f"\n⏱️  Query completed in {result['elapsed_time']:.2f} seconds")
                
            except KeyboardInterrupt:
                print("\n\n👋 Exiting...")
                break
            except Exception as e:
                print(f"❌ Error processing query: {str(e)}")
                logger.error("Query processing error: %s", str(e))
    
    except Exception as e:
        print(f"❌ System error: {str(e)}")
        logger.error("Main system error: %s", str(e))
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        rag_system.db.close()
        print("✅ RAG System shutdown complete")


if __name__ == "__main__":
    main()