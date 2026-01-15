#sliding window 分块
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    cid = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append({
            "chunk_id": cid,
            "text": chunk
        })
        cid += 1
        start += chunk_size - overlap

    return chunks
