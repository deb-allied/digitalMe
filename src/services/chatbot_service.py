from typing import List, Dict, Any, Optional
import openai
from tqdm import tqdm
import time

from config.settings import config
from src.services.retriever_service import RetrieverService
from src.utils.logger import LoggerSetup


class ChatbotService:
    """Service for Q&A chatbot using GPT-3.5."""
    
    def __init__(self, retriever_service: RetrieverService):
        self.logger = LoggerSetup.get_logger(self.__class__.__name__)
        self.retriever_service = retriever_service
        self._validate_api_key()
        openai.api_key = config.openai.api_key
    
    def _validate_api_key(self) -> None:
        """Validate OpenAI API key."""
        if not config.openai.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
    
    def _build_prompt(self, query: str, personality_type: str, 
                     context_messages: List[Dict[str, Any]]) -> str:
        """Build prompt for GPT-3.5."""
        # Build context from retrieved messages
        context_parts = []
        for msg in context_messages:
            content = msg["content"]
            score = msg["similarity_score"]
            context_parts.append(f"[Relevance: {score:.2f}] {content}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are answering questions based on the personality type: {personality_type}

                    Based on the following context from previous conversations, please answer the user's question in a way that reflects this personality type.

                    Context from previous messages:
                    {context}

                    User Question: {query}

                    Please provide a helpful and personality-consistent response. 
                    Also consider the user's intent and provide additional context if necessary. 
                    Add a follow-up question that the user can ask regarding the content, ask it as if you are asking to help further:"""

        return prompt
    
    def answer_question(self, query: str, personality_type: str,
                       top_k: Optional[int] = None) -> Dict[str, Any]:
        """Answer a question based on personality context."""
        try:
            self.logger.info(
                "Answering question for personality '%s': %s",
                personality_type,
                query[:50] + "..." if len(query) > 50 else query
            )
            
            # Retrieve relevant context with progress
            print("\nRetrieving relevant context...")
            context_messages = self.retriever_service.retrieve(
                query=query,
                personality_type=personality_type,
                top_k=top_k
            )
            
            if not context_messages:
                self.logger.warning("No relevant context found for query")
                # Still try to answer based on personality type alone
                context_messages = []
            
            # Build prompt
            prompt = self._build_prompt(query, personality_type, context_messages)
            
            # Call GPT-3.5 with progress indicator
            print("Generating response...")
            
            # Create a simple progress bar for API call
            with tqdm(total=1, desc="Calling GPT-3.5", unit="call", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
                response = openai.chat.completions.create(
                    model=config.openai.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided personality context."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=config.openai.temperature,
                    max_tokens=config.openai.max_tokens
                )
                pbar.update(1)
            
            answer = response.choices[0].message.content.strip()
            
            result = {
                "query": query,
                "personality_type": personality_type,
                "answer": answer,
                "context_used": len(context_messages),
                "context_messages": context_messages
            }
            
            self.logger.info("Successfully generated answer")
            
            return result
            
        except Exception as e:
            self.logger.error("Failed to answer question: %s", str(e))
            raise
    
    def chat_session(self, personality_type: str) -> None:
        """Start an interactive chat session."""
        print(f"\nStarting chat session with personality: {personality_type}")
        print("Type 'exit' to end the session.\n")
        
        while True:
            try:
                query = input("You: ").strip()
                
                if query.lower() in ['exit', 'quit', 'bye']:
                    print("Ending chat session. Goodbye!")
                    break
                
                if not query:
                    continue
                
                # Get answer
                result = self.answer_question(query, personality_type)
                
                print(f"\n{personality_type}: {result['answer']}\n")
                print(f"(Used {result['context_used']} context messages)\n")
                
            except KeyboardInterrupt:
                print("\nChat session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                self.logger.error("Chat session error: %s", str(e))