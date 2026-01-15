#DB连接与基础封装
import sqlite3

class DB:
    def __init__(self, path):
        self.conn = sqlite3.connect(path)

    def execute(self, sql, params=()):
        cur = self.conn.cursor()
        cur.execute(sql, params)
        self.conn.commit()
        return cur

    def query(self, sql, params=()):
        cur = self.conn.cursor()
        cur.execute(sql, params)
        return cur.fetchall()
