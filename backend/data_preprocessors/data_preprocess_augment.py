import os
import json
import re
from datetime import datetime
from typing import List, Dict, Any
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

class PersonalityClassifier:
    """Classifier for personality traits based on conversation content"""
    
    def __init__(self):
        """Initialize with OpenAI API key from .env file"""
        # Load environment variables from .env file
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file. Please add OPENAI_API_KEY=your_key_here to your .env file")
        
        self.client = OpenAI(api_key=api_key)
    
    def generate_summary(self, messages: List[str]) -> str:
        """Generate a summary of the conversation messages"""
        # Combine all messages into one text for summarization
        combined_text = " ".join(messages)
        
        # Limit text length for API efficiency
        if len(combined_text) > 8000:
            combined_text = combined_text[:8000] + "..."
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that creates concise summaries of conversations. Focus on the main topics, themes, and key points discussed."
                    },
                    {
                        "role": "user",
                        "content": f"Please provide a concise summary of this conversation:\n\n{combined_text}"
                    }
                ],
                max_tokens=200,
                temperature=0.3
            )
            content = response.choices[0].message.content
            if content is not None:
                return content.strip()
            else:
                print("Warning: OpenAI API returned no content for summary.")
                return "Summary generation failed"
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Summary generation failed"
    
    def classify_personality(self, summary: str) -> Dict[str, Any]:
        """Classify personality traits based on conversation summary in terms of professions"""
        
        personality_prompt = f"""
        Based on the following conversation summary, analyze the communication style, thinking patterns, and interests to determine which professional archetype the person most closely resembles.
        
        Summary: {summary}
        
        Classify the person's professional personality type and provide scores for different professional traits.
        
        Respond in the following JSON format:
        {{
            "primary_profession_type": "one of: Engineer, Teacher, Sales_Professional, Creative_Designer, Researcher, Manager, Consultant, Healthcare_Professional, Legal_Professional, Entrepreneur, Technical_Support, Financial_Analyst, Marketing_Professional, Operations_Specialist, HR_Professional",
            "secondary_profession_type": "second most likely profession type from the same list",
            "professional_traits": {{
                "analytical_thinking": "score from 1-10 (Engineer, Researcher, Financial_Analyst)",
                "communication_skills": "score from 1-10 (Teacher, Sales_Professional, HR_Professional)",
                "creativity": "score from 1-10 (Creative_Designer, Marketing_Professional)",
                "problem_solving": "score from 1-10 (Engineer, Consultant, Technical_Support)",
                "leadership": "score from 1-10 (Manager, Entrepreneur)",
                "attention_to_detail": "score from 1-10 (Legal_Professional, Financial_Analyst)",
                "empathy": "score from 1-10 (Healthcare_Professional, HR_Professional, Teacher)",
                "strategic_thinking": "score from 1-10 (Consultant, Manager, Entrepreneur)",
                "technical_aptitude": "score from 1-10 (Engineer, Technical_Support)",
                "business_acumen": "score from 1-10 (Sales_Professional, Financial_Analyst, Entrepreneur)"
            }},
            "professional_characteristics": ["list of 3-5 key professional characteristics"],
            "likely_industries": ["list of 2-4 industries this person would thrive in"]
        }}
        
        Provide only the JSON response, no additional text.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional assessment expert. Analyze conversations to determine professional personality types and career aptitudes. Provide only valid JSON responses."
                    },
                    {
                        "role": "user",
                        "content": personality_prompt
                    }
                ],
                max_tokens=400,
                temperature=0.2
            )
            
            # Parse the JSON response
            message_content = response.choices[0].message.content
            if message_content is None:
                print("Error: No content returned from OpenAI API.")
                return self._get_default_personality()
            response_text = message_content.strip()
            
            # Clean the response to ensure it's valid JSON
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            result = json.loads(response_text)
            return result
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            return self._get_default_personality()
        except Exception as e:
            print(f"Error classifying personality: {e}")
            return self._get_default_personality()
    
    def _get_default_personality(self) -> Dict[str, Any]:
        """Return default personality classification when API fails"""
        return {
            "primary_profession_type": "Unknown",
            "secondary_profession_type": "Unknown",
            "professional_traits": {
                "analytical_thinking": 5,
                "communication_skills": 5,
                "creativity": 5,
                "problem_solving": 5,
                "leadership": 5,
                "attention_to_detail": 5,
                "empathy": 5,
                "strategic_thinking": 5,
                "technical_aptitude": 5,
                "business_acumen": 5
            },
            "professional_characteristics": ["Unable to classify"],
            "likely_industries": ["Unknown"]
        }

def estimate_tokens(text: str) -> int:
    """Rough estimation: 1 token ≈ 4 characters for English text"""
    return len(text) // 4

def get_conversation_messages(conversation: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract and process conversation messages"""
    messages = []
    current_node = conversation.get("current_node")
    mapping = conversation.get("mapping", {})
    
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    while current_node:
        node = mapping.get(current_node, {})
        message = node.get("message") if node else None
        content = message.get("content") if message else None
        author = message.get("author", {}).get("role", "") if message else ""
        
        if content and content.get("content_type") == "text":
            parts = content.get("parts", [])
            if parts and len(parts) > 0 and len(parts[0]) > 0:
                if author != "system" or (message.get("metadata", {}) if message else {}).get("is_user_system_message"):
                    text = parts[0]
                    
                    # Filter out messages smaller than 75 characters
                    if len(text) >= 75:
                        # Split large messages into chunks using LangChain
                        chunks = text_splitter.split_text(text)
                        for chunk in chunks:
                            messages.append({"text": chunk})
        
        current_node = node.get("parent") if node else None
    
    return messages[::-1]

def write_conversations_and_json(conversations_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Process conversations and write to files with personality classification"""
    created_directories_info = []
    pruned_data = []
    
    # Professional statistics tracking
    profession_stats = {}
    
    try:
        # Initialize personality classifier
        classifier = PersonalityClassifier()
        print("✓ OpenAI client initialized successfully from .env file")
    except ValueError as e:
        print(f"❌ {e}")
        print("Please create a .env file in your project root with:")
        print("OPENAI_API_KEY=your_openai_api_key_here")
        print("Proceeding without AI analysis...")
        classifier = None
    except Exception as e:
        print(f"❌ Error initializing OpenAI client: {e}")
        print("Proceeding without AI analysis...")
        classifier = None
    
    for i, conversation in enumerate(conversations_data, 1):
        updated = conversation.get('update_time')
        if not updated:
            continue
        
        updated_date = datetime.fromtimestamp(updated)
        directory_name = updated_date.strftime('%B_%Y')
        directory_path = os.path.join('/mnt/data', directory_name)
        
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        
        title = conversation.get('title', 'Untitled')
        sanitized_title = re.sub(r"[^a-zA-Z0-9_]", "_", title)[:120]
        file_name = f"{directory_path}/{sanitized_title}_{updated_date.strftime('%d_%m_%Y_%H_%M_%S')}.txt"
        
        messages = get_conversation_messages(conversation)
        
        if not messages:  # Skip conversations with no valid messages
            continue
        
        # Extract text content for analysis
        message_texts = [msg["text"] for msg in messages]
        
        # Initialize default values
        summary = "No summary available"
        personality_data = {
            "primary_profession_type": "Unknown",
            "secondary_profession_type": "Unknown",
            "professional_traits": {
                "analytical_thinking": 5,
                "communication_skills": 5,
                "creativity": 5,
                "problem_solving": 5,
                "leadership": 5,
                "attention_to_detail": 5,
                "empathy": 5,
                "strategic_thinking": 5,
                "technical_aptitude": 5,
                "business_acumen": 5
            },
            "professional_characteristics": ["No analysis available"],
            "likely_industries": ["Unknown"]
        }
        
        # Generate summary and classify personality if API is available
        if classifier:
            print(f"Processing ({i}/{len(conversations_data)}): {title}")
            summary = classifier.generate_summary(message_texts)
            personality_data = classifier.classify_personality(summary)
            
            # Track profession statistics
            primary_prof = personality_data.get("primary_profession_type", "Unknown")
            if primary_prof in profession_stats:
                profession_stats[primary_prof] += 1
            else:
                profession_stats[primary_prof] = 1
        
        # Write plain text version
        with open(file_name, 'w', encoding="utf-8") as file:
            for message in messages:
                file.write(f"{message['text']}\n")
        
        # Add to pruned data with summary and personality
        pruned_data.append({
            "title": title,
            "summary": summary,
            "professional_personality": personality_data,
            "messages": message_texts,
            "message_count": len(messages),
            "total_characters": sum(len(msg) for msg in message_texts),
            "created_date": updated_date.isoformat()
        })
        
        created_directories_info.append({
            "directory": directory_path,
            "file": file_name
        })
    
    # Add profession statistics to the final JSON
    final_data = {
        "conversations": pruned_data,
        "statistics": {
            "total_conversations": len(pruned_data),
            "profession_distribution": profession_stats,
            "analysis_timestamp": datetime.now().isoformat()
        }
    }
    
    # Write enhanced JSON with summaries and professional personality data
    with open('backend/src/data/pruned_personality.json', 'w', encoding='utf-8') as json_file:
        json.dump(final_data, json_file, ensure_ascii=False, indent=4)
    
    return created_directories_info

def generate_profession_report(data_file: str = 'backend/src/data/pruned_personality.json'):
    """Generate a detailed report of professional personality distribution"""
    try:
        with open(data_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        conversations = data.get('conversations', [])
        if not conversations:
            print("No conversation data found")
            return
        
        # Analyze profession distribution
        profession_counts = {}
        trait_averages = {}
        industry_preferences = {}
        
        for conv in conversations:
            prof_data = conv.get('professional_personality', {})
            primary_prof = prof_data.get('primary_profession_type', 'Unknown')
            
            # Count professions
            profession_counts[primary_prof] = profession_counts.get(primary_prof, 0) + 1
            
            # Track trait scores
            traits = prof_data.get('professional_traits', {})
            for trait, score in traits.items():
                if isinstance(score, (int, float)):
                    if trait not in trait_averages:
                        trait_averages[trait] = []
                    trait_averages[trait].append(score)
            
            # Track industry preferences
            industries = prof_data.get('likely_industries', [])
            for industry in industries:
                industry_preferences[industry] = industry_preferences.get(industry, 0) + 1
        
        # Generate report
        print("\n" + "="*50)
        print("PROFESSIONAL PERSONALITY ANALYSIS REPORT")
        print("="*50)
        
        print(f"\n📊 Total Conversations Analyzed: {len(conversations)}")
        
        print(f"\n👔 Professional Type Distribution:")
        sorted_profs = sorted(profession_counts.items(), key=lambda x: x[1], reverse=True)
        for prof, count in sorted_profs:
            percentage = (count / len(conversations)) * 100
            print(f"   {prof.replace('_', ' ')}: {count} ({percentage:.1f}%)")
        
        print(f"\n📈 Average Professional Trait Scores:")
        for trait, scores in trait_averages.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                print(f"   {trait.replace('_', ' ').title()}: {avg_score:.1f}/10")
        
        print(f"\n🏭 Top Industry Preferences:")
        sorted_industries = sorted(industry_preferences.items(), key=lambda x: x[1], reverse=True)[:10]
        for industry, count in sorted_industries:
            print(f"   {industry}: {count} mentions")
        
        print("\n" + "="*50)
        
    except Exception as e:
        print(f"Error generating report: {e}")

def check_env_file():
    """Check if .env file exists and has required variables"""
    if not os.path.exists('.env'):
        print("❌ .env file not found in project root")
        print("Please create a .env file with the following content:")
        print("OPENAI_API_KEY=your_openai_api_key_here")
        return False
    
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in .env file")
        print("Please add the following line to your .env file:")
        print("OPENAI_API_KEY=your_openai_api_key_here")
        return False
    
    print("✅ .env file found with OPENAI_API_KEY")
    return True

def main():
    """Main execution function"""
    try:
        # Check .env file
        print("🔍 Checking .env configuration...")
        env_ok = check_env_file()
        
        # Load conversation data
        with open('backend/src/data/conversations.json', 'r', encoding='utf-8') as file:
            conversations_data = json.load(file)
        
        print(f"📚 Processing {len(conversations_data)} conversations...")
        
        # Process conversations
        created_directories_info = write_conversations_and_json(conversations_data)
        
        print(f"✅ Successfully processed {len(created_directories_info)} conversations")
        print("📄 Enhanced JSON with professional personality data saved to backend/src/data/pruned_personality.json")
        
        # Generate profession report if API was available
        if env_ok:
            generate_profession_report()
        
    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
    except Exception as e:
        print(f"❌ Error processing conversations: {e}")

if __name__ == "__main__":
    main()