#paper/chunk引用
def build_citation(paper, chunk_ids):
    return {
        "paper_id": paper["paper_id"],
        "title": paper["title"],
        "year": paper["year"],
        "chunks": chunk_ids
    }
