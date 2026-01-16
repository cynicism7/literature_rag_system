#query->embedding
from sentence_transformers import SentenceTransformer
import torch

class QueryEncoder:
    def __init__(self, model_name, device=None):
        # 如果没有指定设备，自动检测
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            print("警告: CUDA不可用，使用CPU")
            device = "cpu"
        
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, query: str):
        return self.model.encode(
            [query],
            normalize_embeddings=True
        )
