import os
import sys
import time
from typing import List, Optional
from tqdm import tqdm

from config.settings import settings, check_system_requirements, print_system_status
from src.embeddings.embedder import HuggingFaceEmbedder
from src.vectorstore.chroma_store import PersonalityVectorStore
from src.retriever.personality_retriever import PersonalityRetriever
from src.chatbot.qa_bot import PersonalityQABot
from src.utils.logger import logger

class PersonalityQASystem:
    """Main system class for personality-based Q&A with comprehensive progress tracking"""
    
    def __init__(self, show_progress: bool = True):
        """Initialize the Q&A system with progress tracking"""
        logger.info("Initializing Personality Q&A System")
        
        if show_progress:
            print("🚀 Initializing Personality Q&A System")
            print("=" * 50)
        
        # Initialize components with progress tracking
        if show_progress:
            with tqdm(
                total=4,
                desc="System Initialization",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} components [{desc}]",
                unit="component"
            ) as pbar:
                pbar.set_description("Loading embedder...")
                self.embedder = HuggingFaceEmbedder()
                pbar.update(1)
                
                pbar.set_description("Initializing vector store...")
                self.vector_store = PersonalityVectorStore(self.embedder)
                pbar.update(1)
                
                pbar.set_description("Setting up retriever...")
                self.retriever = PersonalityRetriever(self.vector_store)
                pbar.update(1)
                
                pbar.set_description("Configuring chatbot...")
                self.qa_bot = PersonalityQABot(self.retriever)
                pbar.update(1)
        else:
            self.embedder = HuggingFaceEmbedder()
            self.vector_store = PersonalityVectorStore(self.embedder)
            self.retriever = PersonalityRetriever(self.vector_store)
            self.qa_bot = PersonalityQABot(self.retriever)
        
        if show_progress:
            print("✅ System initialization complete")
        
        logger.info("System initialization complete")
    
    def setup_data(self, json_path: Optional[str] = None, show_progress: bool = True) -> bool:
        """Setup data from JSON file with comprehensive progress tracking"""
        try:
            if show_progress:
                print("\n📊 Setting up personality data...")
                print("=" * 40)
            
            logger.info("Setting up data from JSON file")
            
            # Check if data file exists first
            data_path = json_path or settings.personality_data_path
            if not os.path.exists(data_path):
                print(f"❌ Data file not found: {data_path}")
                return False
            
            if show_progress:
                print(f"📂 Loading from: {data_path}")
            
            # Load data with progress tracking
            start_time = time.time()
            self.vector_store.load_personality_data_from_json(json_path)
            load_time = time.time() - start_time
            
            if show_progress:
                print(f"\n✅ Data setup completed in {load_time:.2f} seconds")
                
                # Show summary statistics
                personalities = self.qa_bot.get_available_personalities()
                print(f"📈 Summary:")
                print(f"   - Personalities loaded: {len(personalities)}")
                print(f"   - Processing time: {load_time:.2f}s")
                
                # Show personality breakdown
                if personalities:
                    print(f"\n👥 Available personalities:")
                    for i, personality in enumerate(personalities, 1):
                        print(f"   {i}. {personality}")
            
            return True
            
        except Exception as e:
            logger.error("Failed to setup data: %s", str(e))
            if show_progress:
                print(f"❌ Data setup failed: {str(e)}")
            return False
    
    def interactive_chat(self) -> None:
        """Start interactive chat session with enhanced progress feedback"""
        print("\n🤖 Personality Q&A System")
        print("=" * 50)
        
        # Check system status with progress
        print("🔍 Checking system status...")
        with tqdm(
            total=3,
            desc="System checks",
            bar_format="{l_bar}{bar}| {desc}",
            leave=False
        ) as pbar:
            pbar.set_description("Checking configuration...")
            system_checks = check_system_requirements()
            pbar.update(1)
            
            pbar.set_description("Verifying OpenAI...")
            openai_available = system_checks.get('openai_configured', False)
            pbar.update(1)
            
            pbar.set_description("Loading personalities...")
            personalities = self.qa_bot.get_available_personalities()
            pbar.update(1)
        
        if not openai_available:
            print("⚠️  OpenAI not configured - running in context-only mode")
            print("   You can still browse personality data and get relevant context")
            print("   To enable full chatbot features, set OPENAI_API_KEY in .env file")
            print()
        
        if not personalities:
            print("❌ No personalities available. Please setup data first.")
            return
        
        print(f"\n👥 Available personalities ({len(personalities)}):")
        for i, personality in enumerate(personalities, 1):
            print(f"   {i}. {personality}")
        
        # Let user choose personality
        selected_personality = self._select_personality_interactive(personalities)
        if not selected_personality:
            return
        
        # Start chat session
        self._run_chat_session(selected_personality, openai_available)
    
    def _select_personality_interactive(self, personalities: List[str]) -> Optional[str]:
        """Interactive personality selection with enhanced UX"""
        while True:
            try:
                print(f"\n🎯 Personality Selection")
                choice = input(f"Choose personality (1-{len(personalities)}) or 'stats' for details: ").strip().lower()
                
                if choice == 'stats':
                    self._show_personality_stats(personalities)
                    continue
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(personalities):
                    selected_personality = personalities[choice_idx]
                    
                    # Show loading animation while setting personality
                    print(f"🔧 Setting up {selected_personality}...")
                    with tqdm(
                        total=100,
                        desc="Initializing personality",
                        bar_format="{l_bar}{bar}| {desc}",
                        leave=False
                    ) as pbar:
                        if self.qa_bot.set_personality(selected_personality):
                            pbar.update(100)
                            print(f"✅ Selected personality: {selected_personality}")
                            return selected_personality
                        else:
                            pbar.close()
                            print(f"❌ Failed to set personality: {selected_personality}")
                            return None
                else:
                    print(f"❌ Invalid choice. Please enter a number between 1 and {len(personalities)}")
                    
            except (ValueError, KeyboardInterrupt):
                print("\n👋 Exiting...")
                return None
    
    def _show_personality_stats(self, personalities: List[str]) -> None:
        """Show detailed statistics for all personalities"""
        print("\n📊 Personality Statistics")
        print("=" * 40)
        
        stats_pbar = tqdm(
            personalities,
            desc="Loading stats",
            unit="personality",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{desc}]"
        )
        
        try:
            for personality in stats_pbar:
                stats_pbar.set_postfix({'current': personality[:15]})
                stats = self.vector_store.get_personality_stats(personality, show_progress=False)
                
                print(f"\n{personality}:")
                print(f"   Documents: {stats.get('total_documents', 0)}")
                if 'avg_character_count' in stats:
                    print(f"   Avg length: {stats['avg_character_count']:.0f} chars")
                if 'top_traits' in stats:
                    top_traits = stats['top_traits'][:2]
                    print(f"   Top traits: {', '.join([f'{t[0]} ({t[1]:.1f})' for t in top_traits])}")
        
        finally:
            stats_pbar.close()
    
    def _run_chat_session(self, personality: str, openai_available: bool) -> None:
        """Run the main chat session with progress feedback"""
        mode_text = "Full ChatBot" if openai_available else "Context Browser"
        print(f"\n💬 {mode_text} Mode - Chatting with {personality}")
        print("Commands: 'quit' to exit, 'switch' to change personality, 'config' for OpenAI, 'stats' for personality stats")
        print("-" * 80)
        
        while True:
            try:
                query = input(f"\n[{personality}] You: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if query.lower() == 'switch':
                    self.interactive_chat()
                    break
                
                if query.lower() == 'config':
                    self._configure_openai_interactive()
                    openai_available = self.qa_bot.is_openai_available()
                    continue
                
                if query.lower() == 'stats':
                    self._show_current_personality_stats(personality)
                    continue
                
                if not query:
                    continue
                
                # Process query with progress feedback
                self._process_chat_query(personality, query, openai_available)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    def _process_chat_query(self, personality: str, query: str, openai_available: bool) -> None:
        """Process a chat query with progress tracking"""
        print(f"[{personality}] Bot: 🤔 Processing your question...")
        
        start_time = time.time()
        
        if openai_available and self.qa_bot.is_openai_available():
            # Full chatbot response with progress
            with tqdm(
                total=3,
                desc="Generating response",
                bar_format="{l_bar}{bar}| {desc}",
                leave=False
            ) as pbar:
                pbar.set_description("Retrieving context...")
                result = self.qa_bot.ask_question(query)
                pbar.update(3)  # qa_bot handles internal progress
            
            processing_time = time.time() - start_time
            
            if result['success']:
                print(f"[{personality}] Bot: {result['answer']}")
                print(f"📊 Response generated in {processing_time:.2f}s using {result['context_docs_count']} documents")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
        else:
            # Context-only mode with progress
            with tqdm(
                total=2,
                desc="Searching context",
                bar_format="{l_bar}{bar}| {desc}",
                leave=False
            ) as pbar:
                pbar.set_description("Finding relevant content...")
                result = self.qa_bot.get_context_only(query)
                pbar.update(2)
            
            processing_time = time.time() - start_time
            
            if result['success']:
                print(f"[{personality}] Context Found ({processing_time:.2f}s):")
                print("=" * 60)
                
                for i, doc in enumerate(result['context_docs'][:3], 1):
                    similarity = doc.get('similarity', 0)
                    print(f"\n📄 Document {i} (similarity: {similarity:.3f}):")
                    print(f"Content: {doc['document'][:300]}...")
                    
                    if doc['metadata']:
                        traits = {k.replace('trait_', ''): v for k, v in doc['metadata'].items() 
                                if k.startswith('trait_') and isinstance(v, (int, float))}
                        if traits:
                            top_trait = max(traits.items(), key=lambda x: x[1])
                            print(f"Top trait: {top_trait[0]} ({top_trait[1]}/10)")
                
                print(f"\n📊 Found {result['context_docs_count']} relevant documents")
                print("💡 To get AI-generated answers, configure OpenAI API key using 'config' command")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    def _show_current_personality_stats(self, personality: str) -> None:
        """Show detailed stats for current personality"""
        print(f"\n📊 Statistics for {personality}")
        print("=" * 40)
        
        with tqdm(desc="Loading detailed stats", leave=False) as pbar:
            summary = self.retriever.get_personality_summary(personality, show_progress=False)
            pbar.update(1)
        
        if summary.get('available'):
            print(f"📈 Documents: {summary.get('document_count', 0)}")
            
            if 'content_stats' in summary:
                stats = summary['content_stats']
                print(f"📝 Average message: {stats.get('avg_words', 0):.0f} words, {stats.get('avg_characters', 0):.0f} characters")
            
            if 'top_traits' in summary:
                print(f"🏆 Top traits:")
                for trait, score in summary['top_traits']:
                    print(f"   - {trait.replace('_', ' ').title()}: {score:.1f}/10")
            
            if 'communication_patterns' in summary:
                patterns = summary['communication_patterns']
                if patterns.get('patterns_available'):
                    print(f"💬 Communication style: {patterns.get('communication_style', 'Unknown')}")
                    print(f"📏 Average message length: {patterns.get('avg_message_length', 0):.0f} words")
        else:
            print(f"❌ Could not load stats: {summary.get('error', 'Unknown error')}")
    
    def _configure_openai_interactive(self) -> None:
        """Interactive OpenAI configuration with progress feedback"""
        print("\n🔧 OpenAI Configuration")
        print("=" * 30)
        
        api_key = input("Enter your OpenAI API key: ").strip()
        if api_key:
            print("🔌 Testing OpenAI connection...")
            
            with tqdm(
                total=2,
                desc="Configuring OpenAI",
                bar_format="{l_bar}{bar}| {desc}",
                leave=False
            ) as pbar:
                pbar.set_description("Validating API key...")
                success = self.qa_bot.configure_openai(api_key)
                pbar.update(2)
            
            if success:
                print("✅ OpenAI configured successfully!")
            else:
                print("❌ Failed to configure OpenAI. Please check your API key.")
        else:
            print("❌ No API key provided.")
    
    def benchmark_system(self, show_progress: bool = True) -> dict:
        """Run comprehensive system benchmarks with progress tracking"""
        if show_progress:
            print("🚀 Running System Benchmarks")
            print("=" * 40)
        
        benchmark_results = {
            "timestamp": time.time(),
            "embedding_benchmark": {},
            "retrieval_benchmark": {},
            "system_info": {
                "embedding_model": settings.embedding_model,
                "embedding_device": settings.embedding_device,
                "chroma_persist_dir": str(settings.chroma_persist_directory)
            }
        }
        
        try:
            if show_progress:
                with tqdm(
                    total=3,
                    desc="Benchmarking system",
                    unit="test"
                ) as pbar:
                    # Benchmark embedding speed
                    pbar.set_description("Benchmarking embeddings...")
                    embedding_results = self.embedder.benchmark_speed()
                    benchmark_results["embedding_benchmark"] = embedding_results
                    pbar.update(1)
                    
                    # Benchmark retrieval speed
                    pbar.set_description("Benchmarking retrieval...")
                    retrieval_results = self._benchmark_retrieval()
                    benchmark_results["retrieval_benchmark"] = retrieval_results
                    pbar.update(1)
                    
                    # System health check
                    pbar.set_description("System health check...")
                    health_results = check_system_requirements()
                    benchmark_results["health_check"] = health_results
                    pbar.update(1)
            else:
                embedding_results = self.embedder.benchmark_speed()
                benchmark_results["embedding_benchmark"] = embedding_results
                
                retrieval_results = self._benchmark_retrieval()
                benchmark_results["retrieval_benchmark"] = retrieval_results
                
                health_results = check_system_requirements()
                benchmark_results["health_check"] = health_results
            
            if show_progress:
                self._print_benchmark_results(benchmark_results)
            
            return benchmark_results
            
        except Exception as e:
            logger.error("Benchmark failed: %s", str(e))
            if show_progress:
                print(f"❌ Benchmark failed: {str(e)}")
            return {"error": str(e)}
    
    def _benchmark_retrieval(self) -> dict:
        """Benchmark retrieval performance"""
        personalities = self.qa_bot.get_available_personalities()
        if not personalities:
            return {"error": "No personalities available for benchmarking"}
        
        test_queries = [
            "How do you approach problem solving?",
            "What is your communication style?",
            "Describe your work methodology."
        ]
        
        personality = personalities[0]  # Use first available personality
        
        start_time = time.time()
        for query in test_queries:
            self.retriever.retrieve_context(
                personality, query, show_progress=False
            )
        total_time = time.time() - start_time
        
        return {
            "personality_tested": personality,
            "queries_tested": len(test_queries),
            "total_time": total_time,
            "avg_time_per_query": total_time / len(test_queries),
            "queries_per_second": len(test_queries) / total_time
        }
    
    def _print_benchmark_results(self, results: dict) -> None:
        """Print formatted benchmark results"""
        print("\n📊 Benchmark Results")
        print("=" * 30)
        
        # Embedding benchmark
        if "embedding_benchmark" in results:
            emb = results["embedding_benchmark"]
            print(f"🔧 Embedding Performance:")
            print(f"   Speed: {emb.get('avg_speed', 0):.1f} ± {emb.get('std_speed', 0):.1f} texts/second")
            print(f"   Device: {emb.get('device', 'unknown')}")
        
        # Retrieval benchmark
        if "retrieval_benchmark" in results:
            ret = results["retrieval_benchmark"]
            print(f"🔍 Retrieval Performance:")
            print(f"   Speed: {ret.get('queries_per_second', 0):.1f} queries/second")
            print(f"   Avg time: {ret.get('avg_time_per_query', 0):.3f}s per query")
        
        # Health check
        if "health_check" in results:
            health = results["health_check"]
            print(f"🏥 System Health:")
            for check, status in health.items():
                status_icon = "✅" if status else "❌"
                print(f"   {status_icon} {check.replace('_', ' ').title()}")
    
    def add_custom_personality(
        self, 
        personality: str, 
        messages: List[str], 
        traits: Optional[dict] = None,
        show_progress: bool = True
    ) -> bool:
        """Add a custom personality with messages and progress tracking"""
        if show_progress:
            print(f"➕ Adding custom personality: {personality}")
        
        metadata = {}
        if traits:
            metadata.update({f"trait_{k}": v for k, v in traits.items()})
        
        return self.qa_bot.add_messages_to_personality(personality, messages, metadata)
    
    def run_system_check(self, show_progress: bool = True) -> bool:
        """Run comprehensive system check with progress tracking"""
        if show_progress:
            print("🔍 Running Comprehensive System Check")
            print("=" * 50)
        
        all_good = True
        
        if show_progress:
            with tqdm(
                total=5,
                desc="System checks",
                unit="check"
            ) as pbar:
                # Check configuration
                pbar.set_description("Checking configuration...")
                checks = check_system_requirements()
                pbar.update(1)
                
                # Check data file
                pbar.set_description("Verifying data file...")
                if not checks['data_path_exists']:
                    print(f"❌ Data file not found: {settings.personality_data_path}")
                    all_good = False
                else:
                    print(f"✅ Data file found: {settings.personality_data_path}")
                pbar.update(1)
                
                # Check OpenAI
                pbar.set_description("Testing OpenAI...")
                if not checks['openai_configured']:
                    print("⚠️  OpenAI not configured - ChatBot features disabled")
                else:
                    print("✅ OpenAI configured")
                pbar.update(1)
                
                # Check directories
                pbar.set_description("Checking directories...")
                if not checks['log_directory_writable']:
                    print(f"❌ Cannot write to log directory: {settings.log_file.parent}")
                    all_good = False
                else:
                    print(f"✅ Log directory writable")
                
                if not checks['chroma_directory_writable']:
                    print(f"❌ Cannot write to ChromaDB directory: {settings.chroma_persist_directory}")
                    all_good = False
                else:
                    print(f"✅ ChromaDB directory writable")
                pbar.update(1)
                
                # Final validation
                pbar.set_description("Final validation...")
                time.sleep(0.5)  # Brief pause for completeness
                pbar.update(1)
        else:
            # Run checks without progress bars
            checks = check_system_requirements()
            
            if not checks['data_path_exists']:
                all_good = False
            if not checks['log_directory_writable']:
                all_good = False
            if not checks['chroma_directory_writable']:
                all_good = False
        
        return all_good

def main():
    """Main entry point with comprehensive progress tracking"""
    try:
        # Show startup banner
        print("🚀 Starting Personality Q&A System")
        print("=" * 50)
        
        # Print system status with progress
        print("🔧 Checking system configuration...")
        print_system_status()
        print()
        
        # Initialize system with progress
        system = PersonalityQASystem(show_progress=True)
        
        # Run system check with progress
        print("\n🔍 Running system verification...")
        if not system.run_system_check(show_progress=True):
            print("\n❌ System check failed. Please fix the issues above.")
            print("💡 Run with --check flag to see detailed diagnostics")
            return
        
        # Setup data with comprehensive progress tracking
        print("\n📊 Setting up personality data...")
        if not system.setup_data(show_progress=True):
            print("\n❌ Failed to setup data. Please check your configuration.")
            print(f"   Ensure your data file exists at: {settings.personality_data_path}")
            return
        
        print("\n🎉 System ready! Starting interactive chat...")
        time.sleep(1)  # Brief pause before starting chat
        
        # Start interactive chat
        system.interactive_chat()
        
    except KeyboardInterrupt:
        print("\n👋 Exiting gracefully...")
    except Exception as e:
        logger.error("System error: %s", str(e))
        print(f"\n❌ System error: {str(e)}")
        print("\n🔧 Diagnostic commands:")
        print("   python -c \"from config.settings import print_system_status; print_system_status()\"")
        print("   python main.py --check  # (if implemented)")

if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--check":
            from config.settings import print_system_status
            print_system_status()
            sys.exit(0)
        elif sys.argv[1] == "--benchmark":
            system = PersonalityQASystem(show_progress=True)
            system.benchmark_system(show_progress=True)
            sys.exit(0)
    
    main()
