#输出格式
def format_paper_results(papers, mode="brief"):
    """
    papers: list of dict
    mode: brief | detailed
    """
    results = []

    for p in papers:
        if mode == "brief":
            results.append({
                "paper_id": p["paper_id"],
                "score": round(p["score"], 4),
                "hits": p.get("hits", 0),
                "title": p.get("title", "")
            })
        else:
            results.append({
                "paper_id": p["paper_id"],
                "title": p.get("title", ""),
                "year": p.get("year"),
                "score": round(p["score"], 4),
                "matched_chunks": p.get("chunk_ids", []),
                "reason": p.get("reason", "")
            })

    return results
