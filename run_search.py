"""
第四步：关键字搜索
使用向量检索查找相关文献chunks
"""
import os
import yaml
import numpy as np
import faiss
import json
from index.load_index import load_index
from retrieval.query_encoder import QueryEncoder
from retrieval.recall import recall
from database.db import DB


def load_metadata(metadata_path):
    """加载元数据映射"""
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    # 转换为列表，索引为vector_index
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

def aggregate_to_papers(results, db):
    """将chunk级别的结果聚合到paper级别"""
    from collections import defaultdict
    
    # 按paper_id聚合
    paper_chunks = defaultdict(list)
    for result in results:
        paper_id = result["paper_id"]
        paper_chunks[paper_id].append(result)
    
    # 构建论文级别的结果
    paper_results = []
    for paper_id, chunks in paper_chunks.items():
        scores = [c["score"] for c in chunks]
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        
        paper_info = get_paper_info(db, paper_id)
        if not paper_info:
            continue
        
        all_chunks, sections = get_paper_chunks(db, paper_id)
        
        matched_chunks = []
        for chunk in chunks:
            chunk_text, section = get_chunk_text(db, chunk["paper_id"], chunk["chunk_id"])
            matched_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "section": section,
                "score": chunk["score"],
                "text_preview": chunk_text[:150] + "..." if chunk_text and len(chunk_text) > 150 else (chunk_text or "")
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
            "all_sections": sections,
            "total_chunks": len(all_chunks)
        })
    
    paper_results.sort(key=lambda x: x["max_score"], reverse=True)
    return paper_results

def search(query_text, top_k=10, nprobe=32, group_by_paper=True):
    """执行搜索
    
    Args:
        query_text: 查询文本
        top_k: 返回结果数量
        nprobe: FAISS搜索参数
        group_by_paper: 是否按论文聚合（True=按论文，False=按chunk）
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
    
    # 检查文件是否存在
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"索引文件不存在: {index_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")
    
    # 1. 加载索引和元数据
    print(f"加载索引: {index_path}")
    index = load_index(index_path)
    print(f"索引已加载，包含 {index.ntotal} 个向量")
    
    # 设置nprobe（对于IVF索引）
    if hasattr(index, 'nprobe'):
        nprobe = faiss_cfg.get("nprobe", nprobe)
        index.nprobe = nprobe
        print(f"设置 nprobe = {nprobe}")
    
    metadata = load_metadata(metadata_path)
    print(f"元数据已加载，包含 {len(metadata)} 个映射")
    
    # 2. 初始化查询编码器
    print(f"初始化查询编码器: {embedding_cfg['model']}")
    device = embedding_cfg.get("device", "cpu")
    encoder = QueryEncoder(embedding_cfg["model"], device=device)
    
    # 3. 编码查询
    print(f"编码查询: {query_text}")
    query_vec = encoder.encode(query_text)
    query_vec = query_vec.astype('float32')
    # 确保是2D数组 (1, dim)
    if len(query_vec.shape) == 1:
        query_vec = query_vec.reshape(1, -1)
    faiss.normalize_L2(query_vec)
    
    # 4. 搜索
    print(f"搜索 top-{top_k} 结果...")
    results = recall(index, query_vec, metadata, top_k=top_k)
    
    # 5. 加载数据库
    db = DB(system_cfg["db"])
    
    # 6. 获取详细信息并显示
    print("\n" + "="*80)
    print(f"搜索结果 (查询: {query_text})")
    print("="*80)
    
    if group_by_paper:
        # 按论文聚合显示
        paper_results = aggregate_to_papers(results, db)
        print(f"\n找到 {len(paper_results)} 篇相关文献\n")
        
        for i, paper in enumerate(paper_results, 1):
            print(f"{'='*80}")
            print(f"【文献 {i}】最高相似度: {paper['max_score']:.4f} | 平均相似度: {paper['avg_score']:.4f}")
            print(f"{'='*80}")
            print(f"标题: {paper['title']}")
            print(f"作者: {paper['authors']}")
            print(f"年份: {paper['year']}")
            print(f"论文ID: {paper['paper_id']}")
            
            # 显示目录/章节
            if paper['all_sections']:
                print(f"目录/章节: {', '.join(paper['all_sections'][:10])}")
                if len(paper['all_sections']) > 10:
                    print(f"  ... 共{len(paper['all_sections'])}个章节")
            
            print(f"匹配的chunks: {paper['matched_chunks_count']}个 (共{paper['total_chunks']}个chunks)")
            
            # 显示匹配的chunk预览
            print(f"\n匹配内容预览:")
            for j, chunk in enumerate(paper['matched_chunks'][:3], 1):
                print(f"  [{j}] Chunk {chunk['chunk_id']} (相似度: {chunk['score']:.4f})")
                if chunk['section']:
                    print(f"      章节: {chunk['section']}")
                print(f"      内容: {chunk['text_preview']}")
            
            if len(paper['matched_chunks']) > 3:
                print(f"  ... 还有{len(paper['matched_chunks']) - 3}个匹配chunk")
            
            # 显示摘要
            if paper['abstract'] and paper['abstract'] != "无摘要":
                print(f"\n摘要: {paper['abstract'][:200]}...")
            print()
    else:
        # 按chunk显示（原有方式）
        for i, result in enumerate(results, 1):
            paper_id = result["paper_id"]
            chunk_id = result["chunk_id"]
            score = result["score"]
            
            chunk_text, section = get_chunk_text(db, paper_id, chunk_id)
            if not chunk_text:
                continue
            
            paper_info = get_paper_info(db, paper_id)
            
            print(f"\n【结果 {i}】相似度: {score:.4f}")
            print(f"论文ID: {paper_id}")
            if paper_info:
                print(f"标题: {paper_info['title']}")
                print(f"作者: {paper_info['authors']}")
                print(f"年份: {paper_info['year']}")
            if section:
                print(f"章节: {section}")
            print(f"Chunk ID: {chunk_id}")
            print(f"内容预览: {chunk_text[:200]}...")
            print("-" * 80)
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python run_search.py <查询文本> [top_k]")
        print("示例: python run_search.py 'machine learning' 10")
        sys.exit(1)
    
    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    try:
        search(query, top_k=top_k)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()