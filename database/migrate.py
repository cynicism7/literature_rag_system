from database.db import DB

def migrate(db_cfg, schema_path):
    db = DB(db_cfg)
    with open(schema_path, encoding="utf-8") as f:
        sql = f.read()
    db.execute(sql)
