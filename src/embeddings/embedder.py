from typing import List, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time

from config.settings import settings
from src.utils.logger import logger

class HuggingFaceEmbedder:
    """HuggingFace-based text embedder using sentence-transformers with progress tracking"""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the embedder
        
        Args:
            model_name: HuggingFace model name for embeddings
            device: Device to run the model on ('cpu', 'cuda', etc.)
        """
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self._model = None
        
        logger.info("Initializing HuggingFace embedder with model: %s", self.model_name)
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the sentence transformer model with progress tracking"""
        try:
            print(f"🔧 Loading embedding model: {self.model_name}")
            
            # Create a progress bar for model loading
            with tqdm(
                total=100, 
                desc="Loading Model", 
                unit="%",
                bar_format="{l_bar}{bar}| {n:.0f}%|{desc}",
                leave=False
            ) as pbar:
                # Simulate model loading progress (actual loading doesn't provide progress)
                pbar.set_description("Downloading model files...")
                pbar.update(20)
                
                start_time = time.time()
                self._model = SentenceTransformer(self.model_name, device=self.device)
                load_time = time.time() - start_time
                
                pbar.set_description("Model loaded successfully")
                pbar.update(80)
                time.sleep(0.1)  # Brief pause for visual feedback
            
            print(f"✅ Successfully loaded embedding model in {load_time:.2f}s")
            print(f"📊 Model dimension: {self.get_embedding_dimension()}")
            print(f"🖥️  Device: {self.device}")
            
            logger.info("Successfully loaded embedding model: %s", self.model_name)
            
        except Exception as e:
            logger.error("Failed to load embedding model %s: %s", self.model_name, str(e))
            raise
    
    def embed_text(
        self, 
        text: Union[str, List[str]], 
        batch_size: int = 32,
        show_progress: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for text(s) with progress tracking
        
        Args:
            text: Single text string or list of text strings
            batch_size: Batch size for processing large lists
            show_progress: Whether to show progress bar
            
        Returns:
            Embedding vector(s) as numpy array(s)
        """
        if not self._model:
            raise RuntimeError("Model not loaded")
        
        try:
            # Handle single text
            if isinstance(text, str):
                if show_progress:
                    with tqdm(
                        total=1, 
                        desc="Generating Embedding", 
                        unit="text",
                        leave=False
                    ) as pbar:
                        embeddings = self._model.encode([text], convert_to_numpy=True)
                        pbar.update(1)
                        return embeddings[0]
                else:
                    embeddings = self._model.encode([text], convert_to_numpy=True)
                    return embeddings[0]
            
            # Handle list of texts
            total_texts = len(text)
            if total_texts == 0:
                return np.array([])
            
            logger.info("Generating embeddings for %d texts", total_texts)
            
            if show_progress and total_texts > 1:
                # Process in batches with progress bar
                all_embeddings = []
                
                with tqdm(
                    total=total_texts,
                    desc="Generating Embeddings",
                    unit="texts",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                ) as pbar:
                    
                    for i in range(0, total_texts, batch_size):
                        batch = text[i:i + batch_size]
                        batch_embeddings = self._model.encode(
                            batch, 
                            convert_to_numpy=True,
                            show_progress_bar=False  # Disable internal progress bar
                        )
                        all_embeddings.append(batch_embeddings)
                        
                        # Update progress
                        pbar.update(len(batch))
                        pbar.set_postfix({
                            'batch': f'{i//batch_size + 1}/{(total_texts-1)//batch_size + 1}',
                            'batch_size': len(batch)
                        })
                
                # Concatenate all batches
                embeddings = np.vstack(all_embeddings)
                
            else:
                # Process without progress bar for small datasets
                embeddings = self._model.encode(
                    text, 
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
            
            logger.debug("Generated embeddings for %d text(s)", total_texts)
            return embeddings
            
        except Exception as e:
            logger.error("Failed to generate embeddings: %s", str(e))
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model"""
        if not self._model:
            raise RuntimeError("Model not loaded")
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError("Embedding dimension could not be determined")
        return dim
    
    def benchmark_speed(self, sample_texts: Optional[List[str]] = None, iterations: int = 3) -> dict:
        """Benchmark embedding speed with progress tracking"""
        if not self._model:
            raise RuntimeError("Model not loaded")
        
        # Use default sample texts if none provided
        if sample_texts is None:
            sample_texts = [
                "This is a short sample text for benchmarking.",
                "Here's another sample text with different content and length for testing purposes.",
                "A third sample text that provides variety in the benchmark dataset for measuring performance."
            ]
        
        print("🚀 Running embedding speed benchmark...")
        
        results = {
            'model': self.model_name,
            'device': self.device,
            'sample_count': len(sample_texts),
            'iterations': iterations,
            'times': [],
            'speeds': []
        }
        
        with tqdm(total=iterations, desc="Benchmarking", unit="iter") as pbar:
            for i in range(iterations):
                start_time = time.time()
                self.embed_text(sample_texts, show_progress=False)
                end_time = time.time()
                
                iteration_time = end_time - start_time
                speed = len(sample_texts) / iteration_time
                
                results['times'].append(iteration_time)
                results['speeds'].append(speed)
                
                pbar.update(1)
                pbar.set_postfix({
                    'time': f'{iteration_time:.3f}s',
                    'speed': f'{speed:.1f} texts/s'
                })
        
        # Calculate statistics
        avg_time = np.mean(results['times'])
        avg_speed = np.mean(results['speeds'])
        std_speed = np.std(results['speeds'])
        
        results.update({
            'avg_time': avg_time,
            'avg_speed': avg_speed,
            'std_speed': std_speed
        })
        
        print(f"📊 Benchmark Results:")
        print(f"   Average Time: {avg_time:.3f}s")
        print(f"   Average Speed: {avg_speed:.1f} ± {std_speed:.1f} texts/second")
        print(f"   Device: {self.device}")
        
        return results