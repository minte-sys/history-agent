"""Keyword search over the passage index (SQLite FTS5, BM25 ranking)."""
import re
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "corpus" / "index.sqlite"

STOP = set("""a an and are as at be but by for from had has have he her his how i in is it its of on or
she that the their them they this to was were what when where which who whom why will with did does do
about into than then there these those you your""".split())

COLUMNS = ("id", "source_id", "figure", "author", "work", "translator", "editor", "date",
           "section", "url", "section_order", "n", "text")


class Corpus:
    def __init__(self, path: Path = DB_PATH):
        self.db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)

    def figures(self):
        q = "SELECT figure, work, translator, date, COUNT(*) FROM passages GROUP BY source_id ORDER BY figure"
        return [dict(zip(("figure", "work", "translator", "date", "passages"), r)) for r in self.db.execute(q)]

    def search(self, query: str, figure: str | None = None, limit: int = 8):
        words = [w for w in re.findall(r"[\w']+", query.lower()) if w not in STOP and len(w) > 1]
        if not words:
            return []
        # OR the terms so partial matches still rank; BM25 rewards passages with more of them.
        match = " OR ".join('"' + w.replace('"', "") + '"' for w in words)
        sql = ("SELECT p." + ", p.".join(COLUMNS) + ", bm25(passages_fts, 1.0, 2.0, 3.0, 1.0) AS score "
               "FROM passages_fts JOIN passages p ON p.rowid = passages_fts.rowid "
               "WHERE passages_fts MATCH ?")
        args: list = [match]
        if figure:
            sql += " AND p.figure LIKE ?"
            args.append(f"%{figure}%")
        sql += " ORDER BY score LIMIT ?"
        args.append(limit)
        return [dict(zip(COLUMNS + ("score",), r)) for r in self.db.execute(sql, args)]

    def get(self, pid: str):
        r = self.db.execute("SELECT " + ", ".join(COLUMNS) + " FROM passages WHERE id = ?", (pid,)).fetchone()
        return dict(zip(COLUMNS, r)) if r else None

    def neighbours(self, pid: str, before: int = 1, after: int = 1):
        p = self.get(pid)
        if not p:
            return []
        q = ("SELECT " + ", ".join(COLUMNS) + " FROM passages WHERE source_id = ? AND section_order = ? "
             "AND n BETWEEN ? AND ? ORDER BY n")
        return [dict(zip(COLUMNS, r)) for r in self.db.execute(q, (p["source_id"], p["section_order"], p["n"] - before, p["n"] + after))]
