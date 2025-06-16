import json
import sys
import threading
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from tqdm import tqdm
import gradio as gr

from src.models.data_models import ConversationData, Conversation
from src.services.embedding_service import EmbeddingService
from src.services.vectorstore_service import VectorStoreService
from src.services.retriever_service import RetrieverService
from src.services.chatbot_service import ChatbotService
from src.utils.logger import LoggerSetup
from config.settings import config


class PersonalityQAChatbotGradio:
    """Improved Gradio frontend for the personality-based Q&A chatbot."""
    
    def __init__(self):
        self.logger = LoggerSetup.get_logger(self.__class__.__name__)
        self.embedding_service = None
        self.vectorstore_service = None
        self.retriever_service = None
        self.chatbot_service = None
        self._services_initialized = False
        self.initialization_status = "Initializing..."
        
        # Auto-initialize services
        self._auto_initialize()
        
    def _auto_initialize(self):
        """Automatically initialize services on startup."""
        try:
            self.logger.info("Auto-initializing services...")
            self.initialization_status = self._initialize_services()
            
            # If initialization successful, try to load default data
            # if self._services_initialized and Path(config.data_file).exists():
            #     self.logger.info("Loading default data file...")
            #     load_result = self.load_data()
            #     if "✅" in load_result:
            #         self.initialization_status += f"\n{load_result}"
                    
        except Exception as e:
            self.logger.error(f"Auto-initialization failed: {str(e)}")
            self.initialization_status = f"❌ Auto-initialization failed: {str(e)}"
        
    def _initialize_services(self) -> str:
        """Initialize all services."""
        try:
            self.logger.info("Initializing services...")
            
            self.embedding_service = EmbeddingService()
            self.vectorstore_service = VectorStoreService(self.embedding_service)
            self.retriever_service = RetrieverService(self.vectorstore_service)
            self.chatbot_service = ChatbotService(self.retriever_service)
            
            self._services_initialized = True
            self.initialization_status = "✅ All services initialized successfully"
            self.logger.info("All services initialized successfully")
            return self.initialization_status
            
        except Exception as e:
            error_msg = f"❌ Failed to initialize services: {str(e)}"
            self.initialization_status = error_msg
            self.logger.error(f"Failed to initialize services: {str(e)}")
            return error_msg
    
    def load_data(self, data_file: Optional[str] = None) -> Tuple[str, gr.Dropdown, gr.Dropdown, gr.Dropdown]:
        """Load conversation data from JSON file and update dropdowns."""
        try:
            if not self._services_initialized:
                return "❌ Services not initialized. Please initialize services first.", gr.update(), gr.update(), gr.update()
                
            data_file = data_file or config.data_file
            self.logger.info(f"Loading data from: {data_file}")
            
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            conversation_data = ConversationData(**data)
            self.vectorstore_service.clear_collection()
            self.vectorstore_service.add_conversations_batch(conversation_data.conversations)
            
            success_msg = f"✅ Successfully loaded {len(conversation_data.conversations)} conversations"
            self.logger.info(f"Successfully loaded {len(conversation_data.conversations)} conversations")
            
            # Update all personality dropdowns
            personalities = self.get_personalities()
            dropdown_update = gr.update(choices=personalities, value=personalities[0] if personalities else "Generalist")
            
            return success_msg, dropdown_update, dropdown_update, dropdown_update
            
        except Exception as e:
            error_msg = f"❌ Failed to load data: {str(e)}"
            self.logger.error(f"Failed to load data: {str(e)}")
            return error_msg, gr.update(), gr.update(), gr.update()
    
    def get_personalities(self) -> List[str]:
        """Get all available personalities."""
        try:
            if not self._services_initialized or not self.vectorstore_service:
                return ["Generalist"]
            
            personalities = self.vectorstore_service.get_all_personalities()
            if not personalities:
                return ["Generalist"]
            
            if "Generalist" not in personalities:
                personalities.insert(0, "Generalist")
            elif personalities[0] != "Generalist":
                personalities.remove("Generalist")
                personalities.insert(0, "Generalist")
                
            return personalities
            
        except Exception as e:
            self.logger.error(f"Failed to get personalities: {str(e)}")
            return ["Generalist"]
    
    def add_message_to_personality(self, message: str, personality_type: str,
                                 conversation_title: str) -> str:
        """Add a new message to an existing personality."""
        try:
            if not self._services_initialized:
                return "❌ Services not initialized"
                
            if not message or not personality_type or not conversation_title:
                return "❌ All fields are required"
            
            doc_id = self.vectorstore_service.add_message(
                message=message,
                personality_type=personality_type,
                conversation_title=conversation_title
            )
            
            success_msg = f"✅ Successfully added message with ID: {doc_id}"
            self.logger.info(f"Added message with ID: {doc_id}")
            return success_msg
            
        except Exception as e:
            error_msg = f"❌ Failed to add message: {str(e)}"
            self.logger.error(f"Failed to add message: {str(e)}")
            return error_msg
    
    def chat_with_personality(self, message: str, personality: str, history: List[List[str]]) -> Tuple[List[List[str]], str]:
        """Process chat message and return updated history."""
        try:
            if not self._services_initialized:
                return history, "❌ Services not initialized"
                
            if not message:
                return history, ""
            
            result = self.chatbot_service.answer_question(message, personality)
            history.append([message, result['answer']])
            
            self.logger.info(f"Chat interaction - Personality: {personality}, Query: {message[:50]}")
            return history, ""
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            self.logger.error(f"Chat error: {str(e)}")
            return history, error_msg
    
    def single_query(self, query: str, personality_type: str) -> str:
        """Run a single query and return formatted result."""
        try:
            if not self._services_initialized:
                return "❌ Services not initialized"
                
            if not query or not personality_type:
                return "❌ Please enter both query and personality type"
            
            result = self.chatbot_service.answer_question(query, personality_type)
            
            formatted_result = f"""
            <div class="result-card">
                <div class="result-item">
                    <span class="result-label">Query:</span>
                    <span class="result-value">{result['query']}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Personality:</span>
                    <span class="result-value">{result['personality_type']}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Answer:</span>
                    <span class="result-value">{result['answer']}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Context Used:</span>
                    <span class="result-value">{result['context_used']} messages</span>
                </div>
            </div>
            """
            
            self.logger.info(f"Single query processed - Personality: {personality_type}")
            return formatted_result
            
        except Exception as e:
            error_msg = f"❌ Failed to run query: {str(e)}"
            self.logger.error(f"Failed to run query: {str(e)}")
            return error_msg
    
    def get_system_status(self) -> str:
        """Get current system status."""
        # Format initialization status for display
        init_display = self.initialization_status.replace('\n', '<br>')
        
        status = f"""
        <div class="status-card">
            <h3>System Status</h3>
            <div class="status-item">
                <span>Services:</span>
                <strong>{'✅ Initialized' if self._services_initialized else '❌ Not initialized'}</strong>
            </div>
            <div class="status-item">
                <span>Status:</span>
                <strong>{init_display}</strong>
            </div>
            <div class="status-item">
                <span>Data File:</span>
                <strong>{config.data_file}</strong>
            </div>
            <div class="status-item">
                <span>Personalities:</span>
                <strong>{len(self.get_personalities()) if self._services_initialized else 'N/A'}</strong>
            </div>
        </div>
        """
        return status
    
    def create_gradio_interface(self) -> gr.Blocks:
        """Create and return the Gradio interface with improved design."""
        
        # Simplified modern CSS
        custom_css = """
        /* Modern color scheme */
        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --secondary: #06b6d4;
            --success: #10b981;
            --danger: #ef4444;
            --bg-dark: #1e293b;
            --bg-darker: #0f172a;
            --text: #e2e8f0;
            --text-muted: #94a3b8;
            --border: #334155;
        }
        
        /* Base styles */
        .gradio-container {
            background: var(--bg-darker);
            color: var(--text);
        }
        
        /* Card styling */
        .gr-box {
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-radius: 12px;
        }
        
        /* Header */
        .app-header {
            background: linear-gradient(135deg, var(--primary) 0%, #8b5cf6 100%);
            padding: 2rem;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .app-header h1 {
            color: white;
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }
        
        .app-header p {
            color: rgba(255, 255, 255, 0.9);
            margin: 0.5rem 0 0 0;
        }
        
        /* Buttons */
        .gr-button {
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.2s;
        }
        
        .gr-button:hover {
            background: var(--primary-dark);
            transform: translateY(-1px);
        }
        
        .gr-button-secondary {
            background: var(--bg-dark);
            border: 1px solid var(--border);
        }
        
        /* Inputs */
        input, textarea, select {
            background: var(--bg-darker) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            color: var(--text) !important;
        }
        
        input:focus, textarea:focus, select:focus {
            border-color: var(--primary) !important;
            outline: none !important;
        }
        
        /* Tabs */
        .gr-tabs {
            background: transparent;
            border: none;
        }
        
        button.selected {
            background: var(--primary) !important;
            color: white !important;
        }
        
        /* Chat styling */
        .gr-chatbot {
            background: var(--bg-dark);
            border-radius: 12px;
        }
        
        .message {
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin: 0.5rem;
        }
        
        .user {
            background: var(--primary);
            color: white;
            margin-left: 20%;
        }
        
        .bot {
            background: var(--bg-darker);
            border: 1px solid var(--border);
            margin-right: 20%;
        }
        
        /* Status and result cards */
        .status-card, .result-card {
            background: var(--bg-darker);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
        }
        
        .status-card h3 {
            margin: 0 0 1rem 0;
            color: var(--text);
        }
        
        .status-item, .result-item {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--border);
            gap: 1rem;
        }
        
        .status-item:last-child, .result-item:last-child {
            border-bottom: none;
        }
        
        .status-item span:first-child {
            flex-shrink: 0;
            min-width: 120px;
        }
        
        .result-label {
            color: var(--text-muted);
            font-weight: 500;
        }
        
        .result-value {
            color: var(--text);
        }
        
        /* Personality list */
        .personality-item {
            padding: 0.5rem;
            margin: 0.25rem 0;
            background: var(--bg-darker);
            border-radius: 6px;
            border: 1px solid var(--border);
        }
        
        /* Info boxes */
        .info-box {
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        /* Auto-init status */
        .init-success {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: var(--success);
            padding: 0.75rem 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            text-align: center;
        }
        
        .init-warning {
            background: rgba(245, 158, 11, 0.1);
            border: 1px solid rgba(245, 158, 11, 0.3);
            color: #f59e0b;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            text-align: center;
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .gr-box {
            animation: fadeIn 0.3s ease-out;
        }
        """
        
        with gr.Blocks(
            title="DigitalMe",
            theme=gr.themes.Soft(
                primary_hue="indigo",
                secondary_hue="cyan",
                neutral_hue="slate"
            ),
            css=custom_css
        ) as interface:
            
            # Header
            gr.HTML("""
                <div class="app-header">
                    <h1>🧠 DigitalMe</h1>
                    <p>AI-Powered Personality Chat System</p>
                </div>
            """)
            
            # Auto-initialization status
            init_status_html = f"""
                <div class="{'init-success' if self._services_initialized else 'init-warning'}">
                    {'✅ System automatically initialized and ready to use!' if self._services_initialized else '⚠️ System initialization failed. Please check the System tab.'}
                </div>
            """
            gr.HTML(init_status_html)
            
            # Main content with tabs
            with gr.Tabs() as tabs:
                
                # System Tab
                with gr.Tab("🔧 System", id=1):
                    with gr.Column():
                        status_display = gr.HTML(value=self.get_system_status())
                        
                        with gr.Row():
                            reinit_btn = gr.Button("🔄 Reinitialize Services", variant="secondary")
                            refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary")
                        
                        init_status = gr.Textbox(
                            label="Initialization Log", 
                            value=self.initialization_status,
                            lines=3, 
                            interactive=False
                        )
                        
                        gr.Markdown("### 📂 Data Management")
                        with gr.Row():
                            data_file_input = gr.Textbox(
                                label="Data File Path",
                                placeholder=f"Default: {config.data_file}",
                                scale=3
                            )
                            load_btn = gr.Button("📥 Load Data", variant="primary", scale=1)
                        
                        load_status = gr.Textbox(label="Loading Status", lines=3, interactive=False)
                
                # Chat Tab
                with gr.Tab("💬 Chat", id=2):
                    with gr.Column():
                        chat_personality = gr.Dropdown(
                            label="Select Personality",
                            choices=self.get_personalities(),
                            value="Generalist"
                        )
                        
                        chatbot = gr.Chatbot(
                            height=400,
                            bubble_full_width=False,
                            show_label=False
                        )
                        
                        with gr.Row():
                            msg = gr.Textbox(
                                label="Message",
                                placeholder="Type your message and press Enter...",
                                scale=4,
                                container=False
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)
                        
                        chat_error = gr.Textbox(label="", visible=False)
                        clear_btn = gr.Button("🗑️ Clear Chat", size="sm", variant="secondary")
                
                # Query Tab
                with gr.Tab("🔍 Query", id=3):
                    with gr.Column():
                        query_personality = gr.Dropdown(
                            label="Select Personality",
                            choices=self.get_personalities(),
                            value="Generalist"
                        )
                        
                        query_input = gr.Textbox(
                            label="Your Question",
                            lines=3,
                            placeholder="Enter your question here..."
                        )
                        
                        query_btn = gr.Button("🔍 Ask Question", variant="primary", size="lg")
                        query_result = gr.HTML()
                
                # Manage Tab
                with gr.Tab("🎭 Manage", id=4):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Add New Message")
                            add_personality = gr.Dropdown(
                                label="Personality",
                                choices=self.get_personalities(),
                                allow_custom_value=True
                            )
                            add_title = gr.Textbox(label="Conversation Title")
                            add_message = gr.Textbox(label="Message", lines=3)
                            add_btn = gr.Button("➕ Add Message", variant="primary")
                            add_status = gr.Textbox(label="Status", lines=2, interactive=False)
                        
                        with gr.Column():
                            gr.Markdown("### Available Personalities")
                            personality_list = gr.Textbox(
                                label="",
                                value="\n".join([f"• {p}" for p in self.get_personalities()]),
                                lines=10,
                                interactive=False
                            )
                            refresh_personalities_btn = gr.Button("🔄 Refresh List", variant="secondary")
            
            # Event Handlers
            
            # System events
            reinit_btn.click(
                fn=self._initialize_services,
                outputs=[init_status]
            ).then(
                fn=self.get_system_status,
                outputs=[status_display]
            ).then(
                fn=lambda: gr.update(choices=self.get_personalities()),
                outputs=[chat_personality]
            ).then(
                fn=lambda: gr.update(choices=self.get_personalities()),
                outputs=[query_personality]
            ).then(
                fn=lambda: gr.update(choices=self.get_personalities()),
                outputs=[add_personality]
            )
            
            refresh_btn.click(
                fn=self.get_system_status,
                outputs=[status_display]
            )
            
            load_btn.click(
                fn=self.load_data,
                inputs=[data_file_input],
                outputs=[load_status, chat_personality, query_personality, add_personality]
            )
            
            # Chat events
            send_btn.click(
                fn=self.chat_with_personality,
                inputs=[msg, chat_personality, chatbot],
                outputs=[chatbot, chat_error]
            ).then(
                fn=lambda: "",
                outputs=[msg]
            )
            
            msg.submit(
                fn=self.chat_with_personality,
                inputs=[msg, chat_personality, chatbot],
                outputs=[chatbot, chat_error]
            ).then(
                fn=lambda: "",
                outputs=[msg]
            )
            
            clear_btn.click(
                fn=lambda: [],
                outputs=[chatbot]
            )
            
            # Query events
            query_btn.click(
                fn=self.single_query,
                inputs=[query_input, query_personality],
                outputs=[query_result]
            )
            
            # Manage events
            add_btn.click(
                fn=self.add_message_to_personality,
                inputs=[add_message, add_personality, add_title],
                outputs=[add_status]
            ).then(
                fn=lambda: ("", "", ""),
                outputs=[add_message, add_title, add_personality]
            )
            
            refresh_personalities_btn.click(
                fn=lambda: "\n".join([f"• {p}" for p in self.get_personalities()]),
                outputs=[personality_list]
            )
            
            # Auto-refresh personalities on load
            interface.load(
                fn=lambda: (
                    gr.update(choices=self.get_personalities(), value=self.get_personalities()[0]),
                    gr.update(choices=self.get_personalities(), value=self.get_personalities()[0]),
                    gr.update(choices=self.get_personalities(), value=self.get_personalities()[0])
                ),
                outputs=[chat_personality, query_personality, add_personality]
            )
        
        return interface


def main():
    """Main entry point for Gradio application."""
    print("\n🤖 Starting DigitalMe Chatbot")
    print("=" * 50)
    print("📦 Auto-initializing services...")
    
    # Create application instance (auto-initializes)
    app = PersonalityQAChatbotGradio()
    
    # Show initialization result
    if app._services_initialized:
        print("✅ Services initialized successfully!")
        personalities = app.get_personalities()
        print(f"🎭 Loaded {len(personalities)} personalities: {', '.join(personalities)}")
    else:
        print("❌ Service initialization failed. Check the application logs.")
    
    print("=" * 50)
    print("🌐 Launching web interface...")
    
    # Create Gradio interface
    interface = app.create_gradio_interface()
    
    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )


if __name__ == "__main__":
    main()