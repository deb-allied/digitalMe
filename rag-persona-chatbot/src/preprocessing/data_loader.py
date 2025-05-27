import json
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field

from ..core.logger import get_logger

logger = get_logger(__name__)


class Message(BaseModel):
    author: str
    text: str


class ChatThread(BaseModel):
    title: str
    create_time: str
    update_time: str
    messages: List[Message]
    label: str = Field(default="")


class DataLoader:
    """Handles loading and parsing of chat data."""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        logger.info("DataLoader initialized", data_path=str(data_path))
    
    def load_raw_data(self) -> Dict:
        """Load raw JSON data from file."""
        logger.info("Loading raw data from %s", self.data_path)
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info("Raw data loaded successfully", record_count=len(data))
        return data
    
    def parse_chat_threads(self, raw_data: Dict) -> List[ChatThread]:
        """Parse raw data into ChatThread objects."""
        chat_threads = []
        
        for month, threads in raw_data.items():
            logger.debug("Processing month: %s", month)
            
            for thread_data in threads:
                try:
                    messages = [
                        Message(author=msg["author"], text=msg["text"])
                        for msg in thread_data["messages"]
                    ]
                    
                    thread = ChatThread(
                        title=thread_data["title"],
                        create_time=thread_data["create_time"],
                        update_time=thread_data["update_time"],
                        messages=messages
                    )
                    
                    chat_threads.append(thread)
                    
                except Exception as e:
                    logger.error(
                        "Error parsing thread",
                        title=thread_data.get("title", "Unknown"),
                        error=str(e)
                    )
        
        logger.info("Chat threads parsed", total_threads=len(chat_threads))
        return chat_threads