"""
第三步：构建FAISS索引
将所有embeddings合并并构建向量索引
"""
import os
import yaml
import numpy as np
import faiss
from index.build_faiss import build_index
from database.db import DB
import json

def load_all_embeddings(emb_dir, db):
    """加载所有embedding文件并构建元数据映射"""
    all_vectors = []
    metadata = []  # 存储 (paper_id, chunk_id, vector_index) 的映射
    
    # 从数据库获取所有embedding记录
    rows = db.query("""
        SELECT paper_id, chunk_id, vector_path
        FROM embedding
        ORDER BY paper_id, chunk_id
    """)
    
    # 按文件分组加载
    file_to_meta = {}
    for paper_id, chunk_id, vector_path in rows:
        if vector_path not in file_to_meta:
            file_to_meta[vector_path] = []
        file_to_meta[vector_path].append((paper_id, chunk_id))
    
    # 加载每个文件
    for vector_path, meta_list in file_to_meta.items():
        if not os.path.exists(vector_path):
            print(f"警告: 文件不存在 {vector_path}")
            continue
        
        vectors = np.load(vector_path)
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)
        
        # 记录每个向量的元数据
        for i, (paper_id, chunk_id) in enumerate(meta_list):
            vector_idx = len(all_vectors) + i
            metadata.append({
                "vector_index": vector_idx,
                "paper_id": paper_id,
                "chunk_id": chunk_id
            })
        
        all_vectors.append(vectors)
        print(f"已加载: {vector_path}, 向量数: {len(vectors)}")
    
    if not all_vectors:
        raise ValueError("没有找到任何embedding文件")
    
    # 合并所有向量
    combined_vectors = np.vstack(all_vectors)
    print(f"总向量数: {len(combined_vectors)}, 维度: {combined_vectors.shape[1]}")
    
    return combined_vectors, metadata

if __name__ == "__main__":
    # 加载配置
    with open("config/system.yaml", "r", encoding="utf-8") as f:
        system_cfg = yaml.safe_load(f)
    
    with open("config/embedding.yaml", "r", encoding="utf-8") as f:
        embedding_cfg = yaml.safe_load(f)
    
    with open("config/faiss.yaml", "r", encoding="utf-8") as f:
        faiss_cfg = yaml.safe_load(f)
    
    emb_dir = system_cfg["paths"]["embeddings"]
    index_dir = system_cfg["paths"]["faiss_index"]
    os.makedirs(index_dir, exist_ok=True)
    
    db = DB(system_cfg["db"])
    
    print("[BUILD INDEX] 开始构建FAISS索引")
    print(f"Embedding目录: {emb_dir}")
    print(f"索引目录: {index_dir}")
    
    # 1. 加载所有embeddings
    vectors, metadata = load_all_embeddings(emb_dir, db)
    dim = vectors.shape[1]
    
    # 2. 构建索引
    print(f"开始训练索引 (维度: {dim}, 向量数: {len(vectors)})...")
    
    # 调整nlist：不能大于向量数，建议为向量数的1/4到1/2
    nlist = faiss_cfg.get("nlist", 65536)
    if nlist > len(vectors):
        nlist = max(1, len(vectors) // 2)
        print(f"警告: nlist({faiss_cfg.get('nlist', 65536)})大于向量数，自动调整为: {nlist}")
    
    index = build_index(
        vectors, 
        dim,
        nlist=nlist,
        m=faiss_cfg.get("pq_m", 64),
        nbits=8
    )
    
    # 3. 保存索引
    index_path = os.path.join(index_dir, "faiss.index")
    faiss.write_index(index, index_path)
    print(f"索引已保存: {index_path}")
    
    # 4. 保存元数据映射
    metadata_path = os.path.join(index_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"元数据已保存: {metadata_path}")
    
    print("[BUILD INDEX] 索引构建完成")
    print(f"索引统计: {index.ntotal} 个向量")

