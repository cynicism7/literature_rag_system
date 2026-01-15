#大模型batch控制
import os
from embed_chunks import ChunkEmbedder

def run_embedding(chunk_dir, emb_dir, model_cfg):
    os.makedirs(emb_dir, exist_ok=True)
    embedder = ChunkEmbedder(
        model_cfg["model"],
        model_cfg["device"]
    )

    for fname in os.listdir(chunk_dir):
        if not fname.endswith(".jsonl"):
            continue
        out = fname.replace(".jsonl", ".npy")
        embedder.embed_file(
            f"{chunk_dir}/{fname}",
            f"{emb_dir}/{out}"
        )
