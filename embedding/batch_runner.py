#大模型batch控制
import os
from embedding.embed_chunks import ChunkEmbedder

def run_embedding(chunk_dir, emb_dir, model_cfg):
    os.makedirs(emb_dir, exist_ok=True)
    print(f"初始化embedding模型: {model_cfg['model']}")
    print(f"配置设备: {model_cfg['device']}")
    embedder = ChunkEmbedder(
        model_cfg["model"],
        model_cfg["device"]
    )
    print(f"实际使用设备: {embedder.model.device}")

    for fname in os.listdir(chunk_dir):
        if not fname.endswith(".jsonl"):
            continue
        out = fname.replace(".jsonl", ".npy")
        embedder.embed_file(
            f"{chunk_dir}/{fname}",
            f"{emb_dir}/{out}"
        )
