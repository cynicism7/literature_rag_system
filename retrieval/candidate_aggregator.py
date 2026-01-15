#chunk->paper聚合
from collections import defaultdict

def aggregate(chunks):
    papers = defaultdict(list)
    for c in chunks:
        papers[c["paper_id"]].append(c["score"])

    results = []
    for pid, scores in papers.items():
        results.append({
            "paper_id": pid,
            "score": sum(scores) / len(scores),
            "hits": len(scores)
        })
    return sorted(results, key=lambda x: x["score"], reverse=True)
