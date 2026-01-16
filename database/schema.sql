CREATE TABLE IF NOT EXISTS paper (
    paper_id TEXT PRIMARY KEY,
    title TEXT,
    authors TEXT,
    year INTEGER,
    abstract TEXT,
    path TEXT
);

CREATE TABLE IF NOT EXISTS chunk (
    paper_id TEXT,
    chunk_id INTEGER,
    text TEXT,
    section TEXT,
    PRIMARY KEY (paper_id, chunk_id)
);

CREATE TABLE IF NOT EXISTS embedding (
    paper_id TEXT,
    chunk_id INTEGER,
    vector_path TEXT,
    PRIMARY KEY (paper_id, chunk_id)
);
