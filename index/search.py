#Top-K 向量召回
def search(index, query_vec, top_k=200):
    D, I = index.search(query_vec, top_k)
    return I[0], D[0]
