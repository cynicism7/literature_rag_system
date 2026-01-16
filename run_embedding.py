"""
第二步：向量化（Embedding）
将chunks转换为向量embeddings
"""
import yaml
from embedding.batch_runner import run_embedding

if __name__ == "__main__":
    # 加载配置
    with open("config/system.yaml", "r", encoding="utf-8") as f:
        system_cfg = yaml.safe_load(f)
    
    with open("config/embedding.yaml", "r", encoding="utf-8") as f:
        embedding_cfg = yaml.safe_load(f)
    
    chunk_dir = system_cfg["paths"]["chunks"]
    emb_dir = system_cfg["paths"]["embeddings"]
    
    print("[EMBEDDING] 开始向量化")
    print(f"Chunk目录: {chunk_dir}")
    print(f"Embedding目录: {emb_dir}")
    print(f"模型: {embedding_cfg['model']}")
    print(f"设备: {embedding_cfg['device']}")
    
    run_embedding(chunk_dir, emb_dir, embedding_cfg)
    
    print("[EMBEDDING] 向量化完成")

