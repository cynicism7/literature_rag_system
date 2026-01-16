"""
智能问答系统
支持问题理解、关键词提取、向量检索和答案生成
"""
import os
import yaml
import numpy as np
import faiss
import json
import re
from index.load_index import load_index
from retrieval.query_encoder import QueryEncoder
from retrieval.recall import recall
from database.db import DB

def extract_keywords(query, method="simple"):
    """
    提取查询中的关键词
    
    Args:
        query: 查询文本
        method: 提取方法
            - "simple": 简单方法，移除停用词
            - "full": 使用完整查询（推荐用于向量检索）
    """
    if method == "full":
        # 对于向量检索，使用完整查询效果更好
        return query
    
    # 简单的关键词提取：移除常见疑问词和标点
    stop_words = {
        '什么', '是什么', '定义', '如何', '怎样', '为什么', '哪个', '哪些',
        '的', '了', '吗', '呢', '？', '?', 'what', 'is', 'are', 'how', 
        'why', 'which', 'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for'
    }
    
    # 移除标点
    query_clean = re.sub(r'[^\w\s]', ' ', query)
    words = query_clean.split()
    
    # 过滤停用词
    keywords = [w for w in words if w.lower() not in stop_words and len(w) > 1]
    
    if keywords:
        return ' '.join(keywords)
    else:
        # 如果没有提取到关键词，使用原查询
        return query

def load_metadata(metadata_path):
    """加载元数据映射"""
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    meta_list = [None] * len(metadata)
    for item in metadata:
        idx = item["vector_index"]
        meta_list[idx] = (item["paper_id"], item["chunk_id"])
    return meta_list

def get_chunk_text(db, paper_id, chunk_id):
    """从数据库获取chunk文本"""
    rows = db.query("""
        SELECT text, section
        FROM chunk
        WHERE paper_id = :paper_id AND chunk_id = :chunk_id
    """, {"paper_id": paper_id, "chunk_id": chunk_id})
    
    if rows:
        return rows[0][0], rows[0][1] or ""
    return None, ""

def get_paper_info(db, paper_id):
    """从数据库获取paper信息"""
    rows = db.query("""
        SELECT title, authors, year, abstract, path
        FROM paper
        WHERE paper_id = :paper_id
    """, {"paper_id": paper_id})
    
    if rows:
        return {
            "title": rows[0][0] or "未知标题",
            "authors": rows[0][1] or "未知作者",
            "year": rows[0][2] or "未知年份",
            "abstract": rows[0][3] or "无摘要",
            "path": rows[0][4] or ""
        }
    return None

def get_paper_chunks(db, paper_id):
    """获取论文的所有chunks，用于显示目录/章节信息"""
    rows = db.query("""
        SELECT chunk_id, section, text
        FROM chunk
        WHERE paper_id = :paper_id
        ORDER BY chunk_id
    """, {"paper_id": paper_id})
    
    chunks = []
    sections = set()
    for chunk_id, section, text in rows:
        chunks.append({
            "chunk_id": chunk_id,
            "section": section or "",
            "text_preview": text[:100] + "..." if len(text) > 100 else text
        })
        if section:
            sections.add(section)
    
    return chunks, sorted(list(sections))

def aggregate_to_papers(retrieved_chunks, db):
    """
    将chunk级别的结果聚合到paper级别
    
    Args:
        retrieved_chunks: chunk级别的检索结果列表
        db: 数据库连接
    
    Returns:
        按论文聚合的结果列表
    """
    from collections import defaultdict
    
    # 按paper_id聚合
    paper_chunks = defaultdict(list)
    for chunk in retrieved_chunks:
        paper_id = chunk["paper_id"]
        paper_chunks[paper_id].append(chunk)
    
    # 构建论文级别的结果
    paper_results = []
    for paper_id, chunks in paper_chunks.items():
        # 计算论文的总体相似度（使用最高分或平均分）
        scores = [c["score"] for c in chunks]
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        
        # 获取论文信息
        paper_info = get_paper_info(db, paper_id)
        if not paper_info:
            continue
        
        # 获取论文的chunks和章节信息
        all_chunks, sections = get_paper_chunks(db, paper_id)
        
        # 获取匹配的chunk信息
        matched_chunks = []
        for chunk in chunks:
            matched_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "section": chunk.get("section", ""),
                "score": chunk["score"],
                "text_preview": chunk.get("text", "")[:150] + "..." if len(chunk.get("text", "")) > 150 else chunk.get("text", "")
            })
        
        paper_results.append({
            "paper_id": paper_id,
            "title": paper_info["title"],
            "authors": paper_info["authors"],
            "year": paper_info["year"],
            "abstract": paper_info["abstract"],
            "path": paper_info["path"],
            "max_score": max_score,
            "avg_score": avg_score,
            "matched_chunks_count": len(chunks),
            "matched_chunks": matched_chunks,
            "all_sections": sections,  # 论文的所有章节（目录）
            "total_chunks": len(all_chunks)
        })
    
    # 按最高分排序
    paper_results.sort(key=lambda x: x["max_score"], reverse=True)
    
    return paper_results

def generate_answer_simple(query, retrieved_chunks, top_n=3):
    """
    基于检索结果生成简单答案（不使用LLM）
    
    Args:
        query: 原始问题
        retrieved_chunks: 检索到的chunks列表，每个包含paper_id, chunk_id, score, text等
        top_n: 使用前N个最相关的chunks
    """
    if not retrieved_chunks:
        return "抱歉，没有找到相关信息。", []
    
    # 选择最相关的前N个chunks
    top_chunks = retrieved_chunks[:top_n]
    
    # 提取关键信息
    answer_parts = []
    sources = []
    
    for i, chunk in enumerate(top_chunks, 1):
        text = chunk.get('text', '')
        paper_info = chunk.get('paper_info', {})
        score = chunk.get('score', 0)
        
        # 提取最相关的句子（包含关键词的句子）
        sentences = re.split(r'[.!?。！？]\s+', text)
        relevant_sentences = []
        
        # 提取查询中的关键词（简单方法）
        query_words = set(re.findall(r'\w+', query.lower()))
        
        for sent in sentences:
            sent_words = set(re.findall(r'\w+', sent.lower()))
            # 计算句子与查询的重叠度
            overlap = len(query_words & sent_words)
            if overlap > 0:
                relevant_sentences.append((overlap, sent.strip()))
        
        # 选择重叠度最高的句子
        if relevant_sentences:
            relevant_sentences.sort(reverse=True)
            best_sentence = relevant_sentences[0][1]
            if best_sentence:
                answer_parts.append(best_sentence)
        
        # 记录来源
        sources.append({
            "paper_id": chunk.get('paper_id'),
            "title": paper_info.get('title', '未知'),
            "chunk_id": chunk.get('chunk_id'),
            "score": score,
            "text_preview": text[:150] + "..." if len(text) > 150 else text
        })
    
    # 组合答案
    if answer_parts:
        answer = " ".join(answer_parts[:2])  # 使用前2个最相关的句子
        if len(answer) < 50:
            # 如果答案太短，使用第一个chunk的完整文本
            answer = top_chunks[0].get('text', '')[:300]
    else:
        # 如果没有找到相关句子，使用第一个chunk的开头部分
        answer = top_chunks[0].get('text', '')[:300]
    
    return answer, sources

def qa_search(query_text, top_k=10, answer_top_n=3, use_keywords=True, group_by_paper=True):
    """
    智能问答搜索
    
    Args:
        query_text: 问题文本
        top_k: 检索的chunk数量
        answer_top_n: 用于生成答案的chunk数量
        use_keywords: 是否提取关键词（对于向量检索，通常使用完整查询更好）
        group_by_paper: 是否按论文聚合结果（True=按论文，False=按chunk）
    """
    # 加载配置
    with open("config/system.yaml", "r", encoding="utf-8") as f:
        system_cfg = yaml.safe_load(f)
    
    with open("config/embedding.yaml", "r", encoding="utf-8") as f:
        embedding_cfg = yaml.safe_load(f)
    
    with open("config/faiss.yaml", "r", encoding="utf-8") as f:
        faiss_cfg = yaml.safe_load(f)
    
    index_dir = system_cfg["paths"]["faiss_index"]
    index_path = os.path.join(index_dir, "faiss.index")
    metadata_path = os.path.join(index_dir, "metadata.json")
    
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"索引文件不存在: {index_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")
    
    # 1. 提取关键词（用于显示，实际检索使用完整查询）
    keywords = extract_keywords(query_text, method="simple")
    print(f"问题: {query_text}")
    if keywords != query_text:
        print(f"提取的关键词: {keywords}")
    print(f"检索查询: {query_text} (使用完整问题以获得更好的语义匹配)")
    
    # 2. 加载索引和元数据
    print(f"\n加载索引...")
    index = load_index(index_path)
    
    if hasattr(index, 'nprobe'):
        nprobe = faiss_cfg.get("nprobe", 32)
        index.nprobe = nprobe
    
    metadata = load_metadata(metadata_path)
    
    # 3. 初始化查询编码器
    device = embedding_cfg.get("device", "cpu")
    encoder = QueryEncoder(embedding_cfg["model"], device=device)
    
    # 4. 编码查询（使用完整查询，不是关键词）
    print(f"编码查询...")
    query_vec = encoder.encode(query_text)
    query_vec = query_vec.astype('float32')
    if len(query_vec.shape) == 1:
        query_vec = query_vec.reshape(1, -1)
    faiss.normalize_L2(query_vec)
    
    # 5. 搜索
    print(f"搜索 top-{top_k} 结果...")
    results = recall(index, query_vec, metadata, top_k=top_k)
    
    # 6. 加载数据库
    db = DB(system_cfg["db"])
    
    # 7. 获取详细信息
    retrieved_chunks = []
    for result in results:
        paper_id = result["paper_id"]
        chunk_id = result["chunk_id"]
        score = result["score"]
        
        chunk_text, section = get_chunk_text(db, paper_id, chunk_id)
        if not chunk_text:
            continue
        
        paper_info = get_paper_info(db, paper_id)
        
        retrieved_chunks.append({
            "paper_id": paper_id,
            "chunk_id": chunk_id,
            "text": chunk_text,
            "section": section,
            "score": score,
            "paper_info": paper_info or {}
        })
    
    # 8. 生成答案
    print(f"\n基于检索结果生成答案...")
    answer, sources = generate_answer_simple(query_text, retrieved_chunks, top_n=answer_top_n)
    
    # 9. 显示结果
    print("\n" + "="*80)
    print("【问题】")
    print(query_text)
    print("\n" + "="*80)
    print("【答案】")
    print(answer)
    print("\n" + "="*80)
    
    # 10. 按论文聚合或按chunk显示
    if group_by_paper:
        # 按论文聚合显示
        paper_results = aggregate_to_papers(retrieved_chunks, db)
        print(f"【相关文献】（共{len(paper_results)}篇）")
        
        for i, paper in enumerate(paper_results, 1):
            print(f"\n{'='*80}")
            print(f"【文献 {i}】最高相似度: {paper['max_score']:.4f} | 平均相似度: {paper['avg_score']:.4f}")
            print(f"{'='*80}")
            print(f"标题: {paper['title']}")
            print(f"作者: {paper['authors']}")
            print(f"年份: {paper['year']}")
            print(f"论文ID: {paper['paper_id']}")
            
            # 显示目录/章节
            if paper['all_sections']:
                print(f"目录/章节: {', '.join(paper['all_sections'][:10])}")  # 最多显示10个章节
                if len(paper['all_sections']) > 10:
                    print(f"  ... 共{len(paper['all_sections'])}个章节")
            
            print(f"匹配的chunks: {paper['matched_chunks_count']}个 (共{paper['total_chunks']}个chunks)")
            
            # 显示匹配的chunk预览
            print(f"\n匹配内容预览:")
            for j, chunk in enumerate(paper['matched_chunks'][:3], 1):  # 最多显示3个匹配chunk
                print(f"  [{j}] Chunk {chunk['chunk_id']} (相似度: {chunk['score']:.4f})")
                if chunk['section']:
                    print(f"      章节: {chunk['section']}")
                print(f"      内容: {chunk['text_preview']}")
            
            if len(paper['matched_chunks']) > 3:
                print(f"  ... 还有{len(paper['matched_chunks']) - 3}个匹配chunk")
            
            # 显示摘要
            if paper['abstract'] and paper['abstract'] != "无摘要":
                print(f"\n摘要: {paper['abstract'][:200]}...")
    else:
        # 按chunk显示（原有方式）
        print(f"【参考来源】（共{len(sources)}个）")
        for i, source in enumerate(sources, 1):
            print(f"\n来源 {i}:")
            print(f"  论文: {source['title']} (ID: {source['paper_id']})")
            print(f"  相似度: {source['score']:.4f}")
            print(f"  内容预览: {source['text_preview']}")
    
    return {
        "query": query_text,
        "keywords": keywords,
        "answer": answer,
        "sources": sources,
        "all_chunks": retrieved_chunks,
        "papers": aggregate_to_papers(retrieved_chunks, db) if group_by_paper else None
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python run_qa.py <问题> [top_k] [answer_top_n]")
        print("示例: python run_qa.py '机器学习的定义是什么' 10 3")
        sys.exit(1)
    
    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    answer_top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    group_by_paper = True  # 默认按论文聚合
    
    try:
        qa_search(query, top_k=top_k, answer_top_n=answer_top_n, group_by_paper=group_by_paper)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

