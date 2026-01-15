CREATE TABLE paper (
    paper_id TEXT PRIMARY KEY,
    title TEXT,
    authors TEXT,
    year INTEGER,
    abstract TEXT,
    path TEXT
);

CREATE TABLE chunk (
    chunk_id INTEGER,
    paper_id TEXT,
    text TEXT,
    section TEXT,
    PRIMARY KEY (paper_id, chunk_id)
);

CREATE TABLE embedding (
    paper_id TEXT,
    chunk_id INTEGER,
    vector_path TEXT,
    PRIMARY KEY (paper_id, chunk_id)
);
