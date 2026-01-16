from ingestion.pipeline import run_ingestion

if __name__ == "__main__":
    import yaml

    with open("config/system.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pdf_dir = cfg["paths"]["raw_pdfs"]
    chunk_dir = cfg["paths"]["chunks"]

    print("[INGESTION] start")
    print("PDF dir:", pdf_dir)
    print("Chunk dir:", chunk_dir)

    run_ingestion(pdf_dir, chunk_dir)

    print("[INGESTION] done")