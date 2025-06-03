from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ProfessionalTraits(BaseModel):
    """Professional traits model."""
    analytical_thinking: int = Field(ge=0, le=10)
    communication_skills: int = Field(ge=0, le=10)
    creativity: int = Field(ge=0, le=10)
    problem_solving: int = Field(ge=0, le=10)
    leadership: int = Field(ge=0, le=10)
    attention_to_detail: int = Field(ge=0, le=10)
    empathy: int = Field(ge=0, le=10)
    strategic_thinking: int = Field(ge=0, le=10)
    technical_aptitude: int = Field(ge=0, le=10)
    business_acumen: int = Field(ge=0, le=10)


class ProfessionalPersonality(BaseModel):
    """Professional personality model."""
    primary_profession_type: str
    secondary_profession_type: Optional[str] = None
    professional_traits: ProfessionalTraits
    professional_characteristics: List[str]
    likely_industries: List[str]


class Conversation(BaseModel):
    """Conversation model."""
    title: str
    summary: str
    professional_personality: ProfessionalPersonality
    messages: List[str]
    message_count: int
    total_characters: int
    created_date: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConversationData(BaseModel):
    """Root data model for conversations."""
    conversations: List[Conversation]
    statistics: Dict[str, Any]


class MessageDocument(BaseModel):
    """Document model for vector storage."""
    id: str
    content: str
    personality_type: str
    conversation_title: str
    metadata: Dict[str, Any] = Field(default_factory=dict)