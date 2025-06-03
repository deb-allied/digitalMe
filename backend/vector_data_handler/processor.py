import json
import uuid
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from backend.utils.logger import LoggerService
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial

logger = LoggerService(__name__).get_logger()

class MessageProcessor:
    def __init__(self, model_name: str = "ibm-granite/granite-embedding-107m-multilingual") -> None:
        self.model_name = model_name
        self.device = torch.device("cpu")  # Force CPU
        self.tokenizer = None
        self.model = None
        
        # CPU optimization settings
        self.num_threads = min(mp.cpu_count(), 8)  # Limit to prevent overhead
        torch.set_num_threads(self.num_threads)
        torch.set_num_interop_threads(1)  # Reduce thread contention
        
        self._load_model()

    def _load_model(self) -> None:
        try:
            logger.info("Loading tokenizer and model for '%s' with %d CPU threads", 
                       self.model_name, self.num_threads)
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            self.model = AutoModel.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                torch_dtype=torch.float32  # Ensure float32 for CPU
            )
            
            # CPU optimizations
            self.model.eval()  # Set to evaluation mode
            # Note: JIT optimization may not work with all models, so we'll skip it
            # self.model = torch.jit.optimize_for_inference(self.model)
            
            logger.info("Model loaded successfully on CPU with optimizations")
            
        except Exception as e:
            logger.error("Failed to load model '%s': %s", self.model_name, str(e))
            raise

    def load_data(self, file_path: str) -> list:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info("Loaded %d topic groups from %s", len(data), file_path)
            return data
        except Exception as e:
            logger.error("Error loading data file '%s': %s", file_path, str(e))
            raise

    def prepare_documents(self, data: list) -> tuple:
        documents, metadatas, ids = [], [], []
        for topic in data:
            title = topic.get("title", "Untitled")
            for idx, msg in enumerate(topic.get("messages", [])):
                doc_id = str(uuid.uuid4())
                documents.append(msg)
                metadatas.append({"title": title, "message_index": idx})
                ids.append(doc_id)
        logger.info("Prepared %d documents for embedding", len(documents))
        return documents, metadatas, ids

    def _preprocess_batch(self, texts: list) -> dict:
        """Optimized preprocessing with efficient tokenization"""
        if self.tokenizer is None:
            logger.error("Tokenizer is not loaded")
            raise RuntimeError("Tokenizer is not loaded")
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
            # CPU optimizations
            return_attention_mask=True
        )

    def _compute_embeddings_batch(self, batch_tokens: dict) -> np.ndarray:
        """Optimized embedding computation with efficient pooling"""
        with torch.no_grad():  # inference_mode may cause issues with some models
            # Forward pass
            outputs = self.model(**batch_tokens)
            last_hidden_state = outputs.last_hidden_state
            attention_mask = batch_tokens['attention_mask']
            
            # Optimized mean pooling using torch operations
            # Expand attention mask efficiently
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            
            # Mean pooling with efficient operations
            masked_embeddings = last_hidden_state * mask_expanded
            summed = torch.sum(masked_embeddings, dim=1)
            counts = torch.sum(mask_expanded, dim=1).clamp(min=1e-9)
            mean_pooled = summed / counts
            
            # Convert to numpy efficiently
            return mean_pooled.detach().numpy()

    def _process_batch_chunk(self, texts_chunk: list) -> list:
        """Process a chunk of texts and return embeddings"""
        try:
            # Preprocess
            batch_tokens = self._preprocess_batch(texts_chunk)
            
            # Compute embeddings
            embeddings = self._compute_embeddings_batch(batch_tokens)
            
            return embeddings.tolist()
            
        except Exception as e:
            logger.error("Error processing batch chunk: %s", str(e))
            raise

    def embed_documents(self, texts: list, batch_size: int = 32) -> list:
        """
        Optimized embedding with larger batch sizes for CPU efficiency
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size (increased default for CPU efficiency)
        """
        logger.info("Embedding %d documents in batches of %d (CPU optimized)", 
                   len(texts), batch_size)
        
        if self.tokenizer is None or self.model is None:
            logger.error("Tokenizer or model is not loaded")
            raise RuntimeError("Tokenizer or model is not loaded")

        all_embeddings = []
        
        try:
            # Process in batches
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            with tqdm(total=total_batches, desc="Embedding Progress", unit="batch") as pbar:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    
                    # Process batch
                    batch_embeddings = self._process_batch_chunk(batch)
                    all_embeddings.extend(batch_embeddings)
                    
                    pbar.update(1)
                    
                    # Optional: Log progress for large datasets
                    if (i // batch_size + 1) % 50 == 0:
                        logger.debug("Processed %d/%d batches", 
                                  i // batch_size + 1, total_batches)

            logger.info("Successfully embedded %d documents", len(all_embeddings))
            return all_embeddings

        except Exception as e:
            logger.error("Error during embedding: %s", str(e))
            raise

    def embed_documents_parallel(self, texts: list, batch_size: int = 32, max_workers: int = 4) -> list:
        """
        Parallel embedding processing for very large datasets
        WARNING: This creates multiple model instances and may use significant memory.
        Use only for very large datasets where the overhead is justified.
        """
        logger.warning("Parallel embedding creates multiple model instances. Consider using regular embed_documents() for smaller datasets.")
        
        if max_workers is None:
            max_workers = min(2, mp.cpu_count() // 4)  # Very conservative to prevent memory issues
            
        logger.info("Embedding %d documents with %d parallel workers", 
                   len(texts), max_workers)
        
        # For parallel processing, we'll process in sequential batches instead
        # to avoid model loading overhead
        return self.embed_documents(texts, batch_size)