#召回逻辑
from index.search import search

def recall(index, query_vec, meta, top_k=200):
    ids, scores = search(index, query_vec, top_k)
    results = []
    for i, s in zip(ids, scores):
        pid, cid = meta[i]
        results.append({
            "paper_id": pid,
            "chunk_id": cid,
            "score": float(s)
        })
    return results
