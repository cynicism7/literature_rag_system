from retrieval.query_encoder import QueryEncoder
from index.search import search

def run_query(query):
    encoder = QueryEncoder("BAAI/bge-large-en-v1.5")
    qvec = encoder.encode(query)
    # index.search → aggregate → rerank → output
