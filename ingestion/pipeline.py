#ingestion总调度
import os, json
from pdf_parser import parse_pdf
from chunker import chunk_text

def run_ingestion(pdf_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(pdf_dir):
        if not fname.endswith(".pdf"):
            continue
        pid = fname.replace(".pdf", "")
        text = parse_pdf(os.path.join(pdf_dir, fname))
        chunks = chunk_text(text)

        with open(f"{out_dir}/{pid}.jsonl", "w", encoding="utf-8") as f:
            for c in chunks:
                c["paper_id"] = pid
                f.write(json.dumps(c) + "\n")
