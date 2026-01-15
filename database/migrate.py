#表升级工具
from db import DB

def migrate(db_path, schema_path):
    db = DB(db_path)
    with open(schema_path) as f:
        db.conn.executescript(f.read())
