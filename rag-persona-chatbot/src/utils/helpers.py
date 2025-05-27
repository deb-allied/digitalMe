from typing import Optional

from openai import OpenAI

from ..core.config import settings
from ..core.logger import get_logger

logger = get_logger(__name__)


class PersonaInferencer:
    """Infers the best persona for a given query."""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai.api_key)
        self.personas = settings.personas
        logger.info("PersonaInferencer initialized")
    
    def infer_persona(self, query: str) -> str:
        """Infer the most appropriate persona for a query."""
        logger.debug("Inferring persona for query", query=query[:50])
        
        prompt = f"""Given the following query, determine which persona would be best suited to answer it:

Query: "{query}"

Available personas: {', '.join(self.personas)}

Respond with ONLY the persona name that best matches this query."""
        
        try:
            response = self.client.chat.completions.create(
                model=settings.openai.classification_model,
                messages=[
                    {"role": "system", "content": "You are an expert at matching queries to appropriate subject matter experts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            content = response.choices[0].message.content
            persona = content.strip() if content is not None else "General"
            
            if persona not in self.personas:
                logger.warning("Invalid persona inferred", inferred=persona)
                persona = "General"
            
            logger.info("Persona inferred", persona=persona)
            return persona
            
        except Exception as e:
            logger.error("Error inferring persona", error=str(e))
            return "General"