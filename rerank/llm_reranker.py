#7B模型精排
def rerank(llm, query, paper):
    prompt = f"""
Query: {query}
Title: {paper['title']}
Abstract: {paper['abstract']}

Score relevance from 0-5 and say keep or discard.
"""
    out = llm.generate(prompt)
    return parse(out)
