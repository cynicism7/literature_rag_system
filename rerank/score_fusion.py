#embedding+LLM分数
def fuse(embedding_score, llm_score, alpha=0.7):
    return alpha * embedding_score + (1 - alpha) * llm_score
