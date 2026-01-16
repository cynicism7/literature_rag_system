from sqlalchemy import create_engine, text

class DB:
    def __init__(self, cfg: dict):
        if cfg["type"] != "sqlite":
            raise ValueError("This project is locked to SQLite")

        uri = f"sqlite:///{cfg['path']}"
        self.engine = create_engine(
            uri,
            echo=False,
            future=True
        )

    def execute(self, sql: str, params: dict | None = None):
        with self.engine.begin() as conn:
            conn.execute(text(sql), params or {})

    def query(self, sql: str, params: dict | None = None):
        with self.engine.begin() as conn:
            result = conn.execute(text(sql), params or {})
            return result.fetchall()
