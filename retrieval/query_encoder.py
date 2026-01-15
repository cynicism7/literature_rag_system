#query->embedding
from sentence_transformers import SentenceTransformer

class QueryEncoder:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def encode(self, query: str):
        return self.model.encode(
            [query],
            normalize_embeddings=True
        )
