import json
import uuid
import torch
from transformers import AutoTokenizer, AutoModel
from logging_service.logger import LoggerService
from tqdm import tqdm

logger = LoggerService(__name__).get_logger()

class MessageProcessor:
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v3") -> None:
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            logger.info("Loading tokenizer and model for '%s'", self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
            self.model.to(self.device)
            logger.info("Model loaded successfully on device: %s", self.device)
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

    # Inside the class
    def embed_documents(self, texts: list, batch_size: int = 16) -> list:
        logger.info("Embedding %d documents in batches of %d", len(texts), batch_size)
        all_embeddings = []

        try:
            if self.tokenizer is None or self.model is None:
                logger.error("Tokenizer or model is not loaded. Call _load_model() before embedding.")
                raise RuntimeError("Tokenizer or model is not loaded.")

            total_batches = (len(texts) + batch_size - 1) // batch_size
            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Progress", unit="batch"):
                batch = texts[i:i + batch_size]
                batch_tokens = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**batch_tokens)
                    last_hidden_state = outputs.last_hidden_state
                    attention_mask = batch_tokens.attention_mask

                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                summed = torch.sum(last_hidden_state * mask_expanded, 1)
                counts = torch.clamp(mask_expanded.sum(1), min=1e-9)
                mean_pooled = summed / counts

                all_embeddings.extend(mean_pooled.cpu().numpy().tolist())

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            logger.info("All batches embedded successfully")
            return all_embeddings

        except Exception as e:
            logger.error("Error during batched embedding: %s", str(e))
            raise


