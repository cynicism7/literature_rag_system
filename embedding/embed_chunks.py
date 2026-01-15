#chunk->embedding
import json
import numpy as np
from sentence_transformers import SentenceTransformer

class ChunkEmbedder:
    def __init__(self, model_name, device):
        self.model = SentenceTransformer(model_name, device=device)

    def embed_file(self, chunk_file, out_file):
        texts = []
        meta = []

        with open(chunk_file) as f:
            for line in f:
                obj = json.loads(line)
                texts.append(obj["text"])
                meta.append((obj["paper_id"], obj["chunk_id"]))

        vectors = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True
        )

        np.save(out_file, vectors)
        return meta
