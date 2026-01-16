#建立FAISS shard
import faiss
import numpy as np

def build_index(vectors: np.ndarray, dim: int, nlist: int = 65536, m: int = 64, nbits: int = 8):
    """
    构建FAISS索引
    
    根据向量数量自动选择最适合的索引类型：
    - 小数据集(<1000): 使用IndexFlatIP（精确但较慢）
    - 中等数据集(1000-10000): 使用IndexIVFFlat（需要训练）
    - 大数据集(>10000): 使用IndexIVFPQ（压缩索引，节省内存）
    
    Args:
        vectors: 向量数组（应该已经L2归一化）
        dim: 向量维度
        nlist: IVF聚类中心数
        m: PQ压缩的段数
        nbits: PQ每段的比特数
    """
    num_vectors = len(vectors)
    vectors = vectors.astype('float32')
    faiss.normalize_L2(vectors)
    
    # 根据向量数量选择索引类型
    if num_vectors < 1000:
        # 小数据集：使用简单的精确索引
        print(f"使用 IndexFlatIP（精确索引，适合小数据集）")
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        return index
    
    elif num_vectors < 10000:
        # 中等数据集：使用IVF-Flat（需要训练，但比IVF-PQ简单）
        print(f"使用 IndexIVFFlat（IVF索引，适合中等数据集）")
        # 调整nlist：建议为向量数的1/4到1/2，但至少39（FAISS要求）
        adjusted_nlist = min(nlist, max(39, num_vectors // 4))
        if adjusted_nlist != nlist:
            print(f"  调整nlist: {nlist} -> {adjusted_nlist}")
        
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, adjusted_nlist)
        index.train(vectors)
        index.add(vectors)
        return index
    
    else:
        # 大数据集：使用IVF-PQ（压缩索引）
        print(f"使用 IndexIVFPQ（压缩索引，适合大数据集）")
        # 调整nlist：建议为向量数的1/4到1/2
        adjusted_nlist = min(nlist, num_vectors // 4)
        if adjusted_nlist != nlist:
            print(f"  调整nlist: {nlist} -> {adjusted_nlist}")
        
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, adjusted_nlist, m, nbits)
        index.train(vectors)
        index.add(vectors)
        return index
