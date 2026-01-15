#建立FAISS shard
import faiss
import numpy as np

def build_index(vectors: np.ndarray, dim: int):
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(
        quantizer,
        dim,
        nlist=65536,
        m=64,
        nbits=8
    )
    index.train(vectors)
    index.add(vectors)
    return index
