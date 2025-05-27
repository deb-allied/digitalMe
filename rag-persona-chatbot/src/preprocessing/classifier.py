from typing import List, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from tqdm import tqdm

from ..core.config import settings
from ..core.logger import get_logger
from .data_loader import ChatThread

logger = get_logger(__name__)


class PersonaClassifier:
    """Classifies chat threads into personas using OpenAI."""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai.api_key)
        self.personas = settings.personas
        logger.info("PersonaClassifier initialized", personas=self.personas)
    
    def _build_classification_prompt(self, chat_thread: ChatThread) -> str:
        """Build prompt for classification."""
        messages_text = "\n".join([
            f"{msg.author}: {msg.text[:500]}"  # Limit message length
            for msg in chat_thread.messages[:10]  # Limit to first 10 messages
        ])
        
        prompt = f"""Analyze the following conversation and classify it into ONE of these personas:
{', '.join(self.personas)}

Conversation Title: {chat_thread.title}
Messages:
{messages_text}

Based on the semantics, vocabulary, and context, which persona best fits this conversation?
Respond with ONLY the persona name, nothing else."""
        
        return prompt
    
    def classify_single(self, chat_thread: ChatThread) -> str:
        """Classify a single chat thread."""
        try:
            prompt = self._build_classification_prompt(chat_thread)
            
            response = self.client.chat.completions.create(
                model=settings.openai.classification_model,
                messages=[
                    {"role": "system", "content": "You are an expert at classifying conversations into appropriate personas based on their content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            content = response.choices[0].message.content
            if content is not None:
                label = content.strip()
            else:
                logger.warning(
                    "No content returned in classification response, using default",
                    title=chat_thread.title
                )
                label = "General"
            
            # Validate label
            if label not in self.personas:
                logger.warning(
                    "Invalid label returned, using default",
                    returned_label=label,
                    title=chat_thread.title
                )
                label = "General"
            
            return label
            
        except Exception as e:
            logger.error(
                "Error classifying thread",
                title=chat_thread.title,
                error=str(e)
            )
            return "General"
    
    def classify_batch(self, chat_threads: List[ChatThread], max_workers: int = 5) -> List[ChatThread]:
        """Classify multiple chat threads in parallel."""
        logger.info("Starting batch classification", total_threads=len(chat_threads))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for thread in chat_threads:
                future = executor.submit(self.classify_single, thread)
                futures.append((thread, future))
            
            # Process results with progress bar
            classified_threads = []
            for thread, future in tqdm(futures, desc="Classifying threads"):
                label = future.result()
                thread.label = label
                classified_threads.append(thread)
                
                logger.debug(
                    "Thread classified",
                    title=thread.title,
                    label=label
                )
        
        logger.info("Batch classification complete")
        return classified_threads