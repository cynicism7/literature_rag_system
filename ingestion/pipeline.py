#ingestion总调度
import os, json
from ingestion.pdf_parser import parse_pdf
from ingestion.chunker import chunk_text
from ingestion.metadata_extractor import extract_metadata
from database.db import DB
import yaml

def load_db():
    with open("config/system.yaml") as f:
        cfg = yaml.safe_load(f)
    return DB(cfg["db"])


def run_ingestion(pdf_dir, chunk_out_dir):
    os.makedirs(chunk_out_dir, exist_ok=True)
    db = load_db()

    for fname in os.listdir(pdf_dir):
        if not fname.endswith(".pdf"):
            continue

        paper_id = fname.replace(".pdf", "")
        pdf_path = os.path.join(pdf_dir, fname)

        # 1. PDF → text
        text = parse_pdf(pdf_path)

        # 2. 提取元数据
        meta = extract_metadata(text)
        meta.update({
            "paper_id": paper_id,
            "path": pdf_path
        })

        # 3. 写 paper 表
        db.execute("""
        INSERT OR REPLACE INTO paper
        (paper_id, title, year, abstract, path)
        VALUES (:paper_id, :title, :year, :abstract, :path)
        """, meta)

        # 4. 切 chunk
        chunks = chunk_text(text)

        # 5. 写 chunk 表 + jsonl（两者都保留）
        with open(f"{chunk_out_dir}/{paper_id}.jsonl", "w", encoding="utf-8") as f:
            for c in chunks:
                c["paper_id"] = paper_id
                c["section"] = ""
                f.write(json.dumps(c) + "\n")

                db.execute("""
                INSERT OR REPLACE INTO chunk
                (paper_id, chunk_id, text, section)
                VALUES (:paper_id, :chunk_id, :text, :section)
                """, c)


