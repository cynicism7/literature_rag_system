#chunk->embedding
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from database.db import DB
import yaml
import torch

class ChunkEmbedder:
    def __init__(self, model_name, device):
        # 自动检测CUDA是否可用
        if device == "cuda" and not torch.cuda.is_available():
            print("警告: 配置使用CUDA，但PyTorch未编译CUDA支持，自动切换到CPU")
            device = "cpu"
        
        self.model = SentenceTransformer(model_name, device=device)

        with open("config/system.yaml") as f:
            cfg = yaml.safe_load(f)
        self.db = DB(cfg["db"])

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

        for (paper_id, chunk_id) in meta:
            self.db.execute("""
            INSERT OR REPLACE INTO embedding
            (paper_id, chunk_id, vector_path)
            VALUES (:paper_id, :chunk_id, :path)
            """, {
                "paper_id": paper_id,
                "chunk_id": chunk_id,
                "path": out_file
            })

        return meta
