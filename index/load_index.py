#索引加载与路由
import faiss

def load_index(path):
    return faiss.read_index(path)
