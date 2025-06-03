from typing import List, Dict, Any, Optional
import openai
import time
from tqdm import tqdm

from config.settings import settings
from src.retriever.personality_retriever import PersonalityRetriever
from src.utils.logger import logger

class PersonalityQABot:
    """Q&A Chatbot that answers questions based on specific personalities with progress tracking"""
    
    def __init__(self, retriever: Optional[PersonalityRetriever] = None):
        """
        Initialize the Q&A bot
        
        Args:
            retriever: PersonalityRetriever instance
        """
        self.retriever = retriever or PersonalityRetriever()
        self._client: Optional[openai.OpenAI] = None
        self.current_personality: Optional[str] = None
        
        logger.info("PersonalityQABot initialized")
        
        # Initialize OpenAI client if API key is available
        self._initialize_openai_client()
    
    def _initialize_openai_client(self) -> None:
        """Initialize OpenAI client if API key is available"""
        try:
            if settings.is_openai_configured():
                api_key = settings.get_openai_api_key()
                self._client = openai.OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized with model: %s", settings.openai_model)
            else:
                logger.warning("OpenAI API key not configured. Chatbot functionality will be limited.")
                self._client = None
        except Exception as e:
            logger.error("Failed to initialize OpenAI client: %s", str(e))
            self._client = None
    
    def is_openai_available(self) -> bool:
        """Check if OpenAI functionality is available"""
        return self._client is not None
    
    def set_personality(self, personality: str, show_progress: bool = False) -> bool:
        """Set the current personality for the bot"""
        if show_progress:
            with tqdm(
                total=2,
                desc="Setting personality",
                bar_format="{l_bar}{bar}| {desc}",
                leave=False
            ) as pbar:
                pbar.set_description("Getting available personalities...")
                available_personalities = self.retriever.vector_store.get_available_personalities()
                pbar.update(1)
                
                if personality in available_personalities:
                    self.current_personality = personality
                    pbar.set_description("Personality set successfully")
                    pbar.update(1)
                    logger.info("Set current personality to: %s", personality)
                    return True
                else:
                    pbar.set_description("Personality not found")
                    pbar.update(1)
                    logger.warning("Personality '%s' not available. Available: %s", 
                                 personality, available_personalities)
                    return False
        else:
            available_personalities = self.retriever.vector_store.get_available_personalities()
            
            if personality in available_personalities:
                self.current_personality = personality
                logger.info("Set current personality to: %s", personality)
                return True
            else:
                logger.warning("Personality '%s' not available. Available: %s", 
                             personality, available_personalities)
                return False
    
    def _build_context_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Build the context prompt for the chatbot"""
        context_text = "\n\n".join([
            f"Document {i+1}: {doc['document']}" 
            for i, doc in enumerate(context_docs)
        ])
        
        personality_info = ""
        if context_docs:
            # Extract personality traits from first document's metadata
            metadata = context_docs[0].get('metadata', {})
            traits = {k.replace('trait_', ''): v for k, v in metadata.items() 
                     if k.startswith('trait_') and isinstance(v, (int, float))}
            
            if traits:
                top_traits = sorted(traits.items(), key=lambda x: x[1], reverse=True)[:3]
                personality_info = f"\nKey personality traits: {', '.join([f'{trait} ({score}/10)' for trait, score in top_traits])}"
        
        prompt = f"""You are an AI assistant specialized in answering questions based on the communication style and expertise of a {self.current_personality} personality type.{personality_info}

Context Information:
{context_text}

User Question: {query}

Please answer the question in the style and perspective of a {self.current_personality}, using the provided context. Be helpful, accurate, and maintain the professional characteristics typical of this personality type. If the context doesn't contain enough information to answer the question, say so clearly.

Answer:"""
        
        return prompt
    
    def ask_question(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """Ask a question to the bot with comprehensive progress tracking"""
        if not self.current_personality:
            return {
                "answer": "Please set a personality first using set_personality() method.",
                "success": False,
                "error": "No personality set"
            }
        
        # Check if OpenAI is available
        if not self.is_openai_available():
            return {
                "answer": "ChatBot functionality requires OpenAI API key. Please configure OPENAI_API_KEY in your .env file.",
                "success": False,
                "error": "OpenAI not configured",
                "context_available": True
            }
        
        logger.info("Processing question for personality '%s': %s", 
                   self.current_personality, query[:50])
        
        start_time = time.time()
        
        try:
            if show_progress:
                with tqdm(
                    total=4,
                    desc="Generating Response",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} steps [{desc}]",
                    unit="step"
                ) as pbar:
                    # Step 1: Retrieve relevant context
                    pbar.set_description("Retrieving context...")
                    context_docs = self.retriever.retrieve_context(
                        self.current_personality, query, top_k, show_progress=False
                    )
                    pbar.update(1)
                    
                    if not context_docs:
                        pbar.close()
                        return {
                            "answer": f"I don't have enough information about {self.current_personality} personality to answer that question.",
                            "success": False,
                            "error": "No relevant context found",
                            "personality": self.current_personality,
                            "processing_time": time.time() - start_time
                        }
                    
                    # Step 2: Build prompt
                    pbar.set_description("Building prompt...")
                    prompt = self._build_context_prompt(query, context_docs)
                    pbar.update(1)
                    
                    # Step 3: Query OpenAI
                    pbar.set_description("Querying OpenAI...")
                    response = self._client.chat.completions.create(
                        model=settings.openai_model,
                        messages=[
                            {
                                "role": "system",
                                "content": f"You are an AI assistant that responds in the style of a {self.current_personality} professional."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        max_tokens=settings.openai_max_tokens,
                        temperature=settings.openai_temperature
                    )
                    pbar.update(1)
                    
                    # Step 4: Process response
                    pbar.set_description("Processing response...")
                    answer = response.choices[0].message.content
                    processing_time = time.time() - start_time
                    pbar.update(1)
                    
                    # Brief pause to show completion
                    time.sleep(0.1)
            else:
                # Process without progress bars
                context_docs = self.retriever.retrieve_context(
                    self.current_personality, query, top_k, show_progress=False
                )
                
                if not context_docs:
                    return {
                        "answer": f"I don't have enough information about {self.current_personality} personality to answer that question.",
                        "success": False,
                        "error": "No relevant context found",
                        "personality": self.current_personality,
                        "processing_time": time.time() - start_time
                    }
                
                prompt = self._build_context_prompt(query, context_docs)
                
                response = self._client.chat.completions.create(
                    model=settings.openai_model,
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are an AI assistant that responds in the style of a {self.current_personality} professional."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=settings.openai_max_tokens,
                    temperature=settings.openai_temperature
                )
                
                answer = response.choices[0].message.content
                processing_time = time.time() - start_time
            
            # Prepare result
            result = {
                "answer": answer,
                "success": True,
                "personality": self.current_personality,
                "context_docs_count": len(context_docs),
                "query": query,
                "processing_time": processing_time,
                "model_used": settings.openai_model,
                "context_quality": {
                    "avg_similarity": sum(doc.get('similarity', 0) for doc in context_docs) / len(context_docs),
                    "top_similarity": max(doc.get('similarity', 0) for doc in context_docs),
                    "total_context_chars": sum(len(doc.get('document', '')) for doc in context_docs)
                }
            }
            
            logger.info("Successfully generated answer for query in %.2fs", processing_time)
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error("Failed to process question: %s", str(e))
            return {
                "answer": "Sorry, I encountered an error while processing your question.",
                "success": False,
                "error": str(e),
                "personality": self.current_personality,
                "processing_time": processing_time
            }
    
    def batch_ask_questions(
        self,
        queries: List[str],
        show_progress: bool = True,
        delay_between_queries: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Process multiple questions in batch with progress tracking"""
        if not self.current_personality:
            return [{
                "answer": "Please set a personality first using set_personality() method.",
                "success": False,
                "error": "No personality set"
            }] * len(queries)
        
        if show_progress:
            print(f"🤖 Processing {len(queries)} questions for {self.current_personality}")
        
        results = []
        
        if show_progress:
            query_pbar = tqdm(
                queries,
                desc="Processing questions",
                unit="question",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} questions [{elapsed}<{remaining}]"
            )
        else:
            query_pbar = queries
        
        try:
            for i, query in enumerate(query_pbar):
                if show_progress and hasattr(query_pbar, 'set_postfix'):
                    query_pbar.set_postfix({
                        'current': query[:30] + '...' if len(query) > 30 else query
                    })
                
                # Process individual question
                result = self.ask_question(query, show_progress=False)
                results.append(result)
                
                # Add delay between requests to respect rate limits
                if delay_between_queries > 0 and i < len(queries) - 1:
                    time.sleep(delay_between_queries)
        
        finally:
            if hasattr(query_pbar, 'close'):
                query_pbar.close()
        
        if show_progress:
            successful = sum(1 for r in results if r.get('success', False))
            total_time = sum(r.get('processing_time', 0) for r in results)
            print(f"✅ Batch processing complete: {successful}/{len(queries)} successful in {total_time:.2f}s")
        
        return results
    
    def get_context_only(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Get context documents without generating an answer (useful when OpenAI is not available)"""
        if not self.current_personality:
            return {
                "success": False,
                "error": "No personality set"
            }
        
        try:
            # Retrieve relevant context
            context_docs = self.retriever.retrieve_context(
                self.current_personality, query, top_k, show_progress=False
            )
            
            return {
                "success": True,
                "personality": self.current_personality,
                "context_docs": context_docs,
                "context_docs_count": len(context_docs),
                "query": query,
                "context_summary": {
                    "total_characters": sum(len(doc.get('document', '')) for doc in context_docs),
                    "avg_similarity": (
                        sum(doc.get('similarity', 0) for doc in context_docs) / len(context_docs)
                        if context_docs else 0
                    ),
                    "personality_coverage": len(set(
                        doc.get('metadata', {}).get('personality', 'unknown') 
                        for doc in context_docs
                    ))
                }
            }
            
        except Exception as e:
            logger.error("Failed to retrieve context: %s", str(e))
            return {
                "success": False,
                "error": str(e),
                "personality": self.current_personality
            }
    
    def get_available_personalities(self, show_progress: bool = False) -> List[str]:
        """Get list of available personalities with optional progress tracking"""
        return self.retriever.vector_store.get_available_personalities(show_progress=show_progress)
    
    def add_messages_to_personality(
        self, 
        personality: str, 
        messages: List[str], 
        metadata: Optional[Dict[str, Any]] = None,
        show_progress: bool = True
    ) -> bool:
        """Add new messages to an existing personality or create a new one with progress tracking"""
        try:
            if show_progress:
                print(f"➕ Adding {len(messages)} messages to personality: {personality}")
            
            self.retriever.vector_store.add_personality_data(
                personality, messages, metadata, show_progress=show_progress
            )
            
            if show_progress:
                print(f"✅ Successfully added messages to {personality}")
            
            logger.info("Added %d messages to personality: %s", len(messages), personality)
            return True
            
        except Exception as e:
            logger.error("Failed to add messages to personality %s: %s", personality, str(e))
            if show_progress:
                print(f"❌ Failed to add messages: {str(e)}")
            return False
    
    def configure_openai(self, api_key: str, show_progress: bool = True) -> bool:
        """Configure OpenAI client with new API key and progress tracking"""
        try:
            if show_progress:
                with tqdm(
                    total=3,
                    desc="Configuring OpenAI",
                    bar_format="{l_bar}{bar}| {desc}",
                    leave=False
                ) as pbar:
                    # Update settings
                    pbar.set_description("Updating settings...")
                    settings.openai_api_key = api_key
                    pbar.update(1)
                    
                    # Initialize client
                    pbar.set_description("Initializing client...")
                    self._client = openai.OpenAI(api_key=api_key)
                    pbar.update(1)
                    
                    # Test the connection
                    pbar.set_description("Testing connection...")
                    test_response = self._client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=5
                    )
                    pbar.update(1)
            else:
                # Configure without progress bars
                settings.openai_api_key = api_key
                self._client = openai.OpenAI(api_key=api_key)
                
                # Test connection
                test_response = self._client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )
            
            logger.info("OpenAI client configured successfully")
            return True
            
        except Exception as e:
            logger.error("Failed to configure OpenAI client: %s", str(e))
            self._client = None
            return False
    
    def get_conversation_stats(self, show_progress: bool = True) -> Dict[str, Any]:
        """Get comprehensive statistics for the current personality"""
        if not self.current_personality:
            return {"error": "No personality set"}
        
        try:
            if show_progress:
                print(f"📊 Analyzing conversation statistics for {self.current_personality}")
            
            # Get detailed personality summary
            summary = self.retriever.get_personality_summary(
                self.current_personality, show_progress=show_progress
            )
            
            # Add current session info
            summary.update({
                "current_session": {
                    "personality": self.current_personality,
                    "openai_available": self.is_openai_available(),
                    "model_configured": settings.openai_model,
                    "embedding_model": settings.embedding_model
                }
            })
            
            return summary
            
        except Exception as e:
            logger.error("Failed to get conversation stats: %s", str(e))
            return {"error": str(e)}