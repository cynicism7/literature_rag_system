#请求/返回结构
import json
import csv
from pathlib import Path

def export_json(data, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def export_csv(rows, path: str):
    if not rows:
        return

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def export_markdown(papers, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, p in enumerate(papers, 1):
            f.write(f"### {i}. {p.get('title','')}\n")
            f.write(f"- Paper ID: {p['paper_id']}\n")
            f.write(f"- Score: {p['score']}\n")
            if "reason" in p:
                f.write(f"- Reason: {p['reason']}\n")
            f.write("\n")
