from typing import List, Dict, Any, Optional
from tqdm import tqdm
import time

from src.vectorstore.chroma_store import PersonalityVectorStore
from src.embeddings.embedder import HuggingFaceEmbedder
from config.settings import settings
from src.utils.logger import logger

class PersonalityRetriever:
    """Retriever for personality-specific information with progress tracking"""
    
    def __init__(self, vector_store: Optional[PersonalityVectorStore] = None):
        """
        Initialize the retriever
        
        Args:
            vector_store: PersonalityVectorStore instance
        """
        self.vector_store = vector_store or PersonalityVectorStore()
        logger.info("PersonalityRetriever initialized")
    
    def retrieve_context(
        self, 
        personality: str, 
        query: str, 
        top_k: Optional[int] = None, 
        similarity_threshold: Optional[float] = None,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context for a query from a specific personality with progress tracking"""
        top_k = top_k or settings.retrieval_top_k
        similarity_threshold = similarity_threshold or settings.similarity_threshold
        
        if show_progress:
            print(f"🔍 Retrieving context for '{personality}' with query: '{query[:50]}...'")
        
        logger.info("Retrieving context for personality '%s' with query: %s", personality, query[:50])
        
        try:
            if show_progress:
                with tqdm(
                    total=3,
                    desc="Context Retrieval",
                    bar_format="{l_bar}{bar}| {desc}",
                    leave=False
                ) as pbar:
                    # Query the vector store
                    pbar.set_description("Searching vector store...")
                    results = self.vector_store.query_personality(
                        personality, query, top_k, show_progress=False
                    )
                    pbar.update(1)
                    
                    # Filter by similarity threshold
                    pbar.set_description("Filtering by similarity...")
                    filtered_results = self._filter_by_similarity(
                        results, similarity_threshold, show_progress=False
                    )
                    pbar.update(1)
                    
                    # Enhance results with additional metadata
                    pbar.set_description("Enhancing results...")
                    enhanced_results = self._enhance_results(filtered_results)
                    pbar.update(1)
            else:
                # Process without progress bars
                results = self.vector_store.query_personality(
                    personality, query, top_k, show_progress=False
                )
                filtered_results = self._filter_by_similarity(
                    results, similarity_threshold, show_progress=False
                )
                enhanced_results = self._enhance_results(filtered_results)
            
            if show_progress:
                print(f"✅ Retrieved {len(enhanced_results)} relevant documents")
                if enhanced_results:
                    avg_similarity = sum(r.get('similarity', 0) for r in enhanced_results) / len(enhanced_results)
                    print(f"📊 Average similarity: {avg_similarity:.3f}")
            
            logger.info("Retrieved %d relevant documents after filtering", len(enhanced_results))
            return enhanced_results
            
        except Exception as e:
            logger.error("Failed to retrieve context for personality %s: %s", personality, str(e))
            return []
    
    def _filter_by_similarity(
        self, 
        results: List[Dict[str, Any]], 
        similarity_threshold: float,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """Filter results by similarity threshold with optional progress tracking"""
        if not results:
            return []
        
        filtered_results = []
        
        if show_progress and len(results) > 10:
            iterator = tqdm(
                results, 
                desc="Filtering results",
                unit="result",
                leave=False
            )
        else:
            iterator = results
        
        for result in iterator:
            distance = result.get('distance')
            if distance is None:
                # If no distance, include the result
                filtered_results.append(result)
            else:
                similarity = 1 - distance
                result['similarity'] = similarity
                if similarity >= similarity_threshold:
                    filtered_results.append(result)
                elif show_progress and isinstance(iterator, tqdm):
                    iterator.set_postfix({'filtered_out': f'sim={similarity:.3f}'})
        
        return filtered_results
    
    def _enhance_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance results with additional computed metadata"""
        for result in results:
            metadata = result.get('metadata', {})
            document = result.get('document', '')
            
            # Add computed fields
            result['enhanced_metadata'] = {
                'relevance_score': result.get('similarity', 0),
                'document_length': len(document),
                'word_count': len(document.split()),
                'has_traits': any(k.startswith('trait_') for k in metadata.keys()),
                'personality': metadata.get('personality', 'Unknown')
            }
        
        return results
    
    def batch_retrieve_context(
        self,
        personality: str,
        queries: List[str],
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        show_progress: bool = True
    ) -> List[List[Dict[str, Any]]]:
        """Retrieve context for multiple queries in batch with progress tracking"""
        if show_progress:
            print(f"🔍 Batch retrieving context for {len(queries)} queries from '{personality}'")
        
        results = []
        
        if show_progress:
            query_pbar = tqdm(
                queries,
                desc="Processing queries",
                unit="query",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} queries [{elapsed}<{remaining}]"
            )
        else:
            query_pbar = queries
        
        try:
            for i, query in enumerate(query_pbar):
                if isinstance(query_pbar, tqdm):
                    query_pbar.set_postfix({'current': query[:30] + '...' if len(query) > 30 else query})

                query_results = self.retrieve_context(
                    personality=personality,
                    query=query,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                    show_progress=False  # Don't show nested progress
                )
                results.append(query_results)
        
        finally:
            if isinstance(query_pbar, tqdm):
                query_pbar.close()
        
        if show_progress:
            total_results = sum(len(r) for r in results)
            print(f"✅ Completed batch retrieval: {total_results} total results across {len(queries)} queries")
        
        return results
    
    def get_personality_summary(self, personality: str, show_progress: bool = True) -> Dict[str, Any]:
        """Get summary information about a personality with progress tracking"""
        if show_progress:
            print(f"📊 Generating summary for personality: {personality}")
        
        try:
            if show_progress:
                with tqdm(
                    total=4,
                    desc=f"Analyzing {personality[:15]}...",
                    bar_format="{l_bar}{bar}| {desc}",
                    leave=False
                ) as pbar:
                    # Get basic statistics
                    pbar.set_description("Getting basic stats...")
                    stats = self.vector_store.get_personality_stats(personality, show_progress=False)
                    pbar.update(1)
                    
                    # Get sample documents for characteristics analysis
                    pbar.set_description("Sampling documents...")
                    sample_results = self.vector_store.query_personality(
                        personality, "summary characteristics", top_k=5, show_progress=False
                    )
                    pbar.update(1)
                    
                    # Analyze communication patterns
                    pbar.set_description("Analyzing patterns...")
                    patterns = self._analyze_communication_patterns(sample_results)
                    pbar.update(1)
                    
                    # Compile summary
                    pbar.set_description("Compiling summary...")
                    summary = self._compile_personality_summary(personality, stats, patterns)
                    pbar.update(1)
            else:
                # Process without progress bars
                stats = self.vector_store.get_personality_stats(personality, show_progress=False)
                sample_results = self.vector_store.query_personality(
                    personality, "summary characteristics", top_k=5, show_progress=False
                )
                patterns = self._analyze_communication_patterns(sample_results)
                summary = self._compile_personality_summary(personality, stats, patterns)
            
            if show_progress:
                print(f"✅ Generated comprehensive summary for {personality}")
            
            logger.info("Generated summary for personality: %s", personality)
            return summary
            
        except Exception as e:
            logger.error("Failed to get personality summary for %s: %s", personality, str(e))
            return {
                "personality": personality, 
                "available": False, 
                "error": str(e)
            }
    
    def _analyze_communication_patterns(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze communication patterns from sample results"""
        if not results:
            return {"patterns_available": False}
        
        patterns = {
            "patterns_available": True,
            "avg_message_length": 0,
            "common_themes": [],
            "communication_style": "analytical"  # default
        }
        
        # Analyze message lengths
        lengths = []
        for result in results:
            doc = result.get('document', '')
            lengths.append(len(doc.split()))
        
        if lengths:
            patterns["avg_message_length"] = sum(lengths) / len(lengths)
            patterns["message_length_range"] = [min(lengths), max(lengths)]
        
        # Basic communication style analysis based on content
        all_text = " ".join([r.get('document', '') for r in results]).lower()
        
        if any(word in all_text for word in ['data', 'analysis', 'metrics', 'statistics']):
            patterns["communication_style"] = "analytical"
        elif any(word in all_text for word in ['team', 'manage', 'leadership', 'strategy']):
            patterns["communication_style"] = "leadership-focused"
        elif any(word in all_text for word in ['creative', 'design', 'innovative', 'artistic']):
            patterns["communication_style"] = "creative"
        elif any(word in all_text for word in ['teach', 'learn', 'explain', 'understand']):
            patterns["communication_style"] = "educational"
        
        return patterns
    
    def _compile_personality_summary(
        self, 
        personality: str, 
        stats: Dict[str, Any], 
        patterns: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile comprehensive personality summary"""
        summary = {
            "personality": personality,
            "available": True,
            "document_count": stats.get('total_documents', 0),
            "collection_name": stats.get('collection_name', ''),
            "communication_patterns": patterns
        }
        
        # Add trait information if available
        if 'average_traits' in stats:
            summary["average_traits"] = stats['average_traits']
            summary["top_traits"] = stats.get('top_traits', [])
        
        # Add content statistics
        if 'avg_character_count' in stats:
            summary["content_stats"] = {
                "avg_characters": stats['avg_character_count'],
                "avg_words": stats.get('avg_word_count', 0),
                "content_range": {
                    "min_chars": stats.get('min_character_count', 0),
                    "max_chars": stats.get('max_character_count', 0)
                }
            }
        
        return summary
    
    def compare_personalities(
        self, 
        personalities: List[str], 
        query: str,
        show_progress: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """Compare how different personalities respond to the same query"""
        if show_progress:
            print(f"⚖️  Comparing {len(personalities)} personalities for query: '{query[:50]}...'")
        
        results = {}
        
        if show_progress:
            personality_pbar = tqdm(
                personalities,
                desc="Comparing personalities",
                unit="personality"
            )
        else:
            personality_pbar = personalities
        
        try:
            for personality in personality_pbar:
                if isinstance(personality_pbar, tqdm):
                    personality_pbar.set_postfix({'current': personality[:20]})
                
                personality_results = self.retrieve_context(
                    personality=personality,
                    query=query,
                    show_progress=False
                )
                
                # Compile comparison metrics
                results[personality] = {
                    "result_count": len(personality_results),
                    "avg_similarity": (
                        sum(r.get('similarity', 0) for r in personality_results) / len(personality_results)
                        if personality_results else 0
                    ),
                    "top_result_similarity": (
                        personality_results[0].get('similarity', 0)
                        if personality_results else 0
                    ),
                    "sample_response": (
                        personality_results[0].get('document', '')[:200] + '...'
                        if personality_results else "No relevant content found"
                    )
                }
        
        finally:
            if isinstance(personality_pbar, tqdm):
                personality_pbar.close()
        
        if show_progress:
            print(f"✅ Comparison complete across {len(personalities)} personalities")
        
        return results