"""Split corpus/sections.jsonl into passages and build a SQLite full-text index.

A passage is a run of whole paragraphs of roughly TARGET characters, so a
citation always points at a real place in a real section. Each passage gets a
stable id: <source_id>:<section order>.<passage number>, e.g. lincoln-papers:812.1
"""
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = 1200   # aim for passages of about this many characters
MAX = 2400      # split a single very long paragraph at sentence ends past this


def split_long(par: str):
    if len(par) <= MAX:
        return [par]
    out, cur = [], ""
    for sent in par.replace("? ", "?\u0000").replace("! ", "!\u0000").replace(". ", ".\u0000").split("\u0000"):
        if cur and len(cur) + len(sent) > TARGET:
            out.append(cur.strip())
            cur = ""
        cur += sent + " "
    if cur.strip():
        out.append(cur.strip())
    return out


def passages(text: str):
    paras = [p for raw in text.split("\n\n") for p in split_long(raw.strip()) if p]
    cur = []
    size = 0
    for p in paras:
        if cur and size + len(p) > TARGET:
            yield "\n\n".join(cur)
            cur, size = [], 0
        cur.append(p)
        size += len(p)
    if cur:
        yield "\n\n".join(cur)


def main():
    db_path = ROOT / "corpus" / "index.sqlite"
    db_path.unlink(missing_ok=True)
    db = sqlite3.connect(db_path)
    db.executescript(
        """
        CREATE TABLE passages (
            id TEXT PRIMARY KEY, source_id TEXT, figure TEXT, author TEXT, work TEXT,
            translator TEXT, editor TEXT, date TEXT, section TEXT, url TEXT,
            section_order INTEGER, n INTEGER, text TEXT
        );
        CREATE VIRTUAL TABLE passages_fts USING fts5(
            text, section, figure, work,
            content='passages', content_rowid='rowid',
            tokenize='porter unicode61 remove_diacritics 2'
        );
        """
    )
    rows = 0
    with (ROOT / "corpus" / "sections.jsonl").open(encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            for n, text in enumerate(passages(s["text"]), 1):
                pid = f"{s['source_id']}:{s['order']}.{n}"
                db.execute(
                    "INSERT INTO passages VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (pid, s["source_id"], s["figure"], s["author"], s["work"], s.get("translator"),
                     s.get("editor"), s.get("date"), s["section"], s["url"], s["order"], n, text),
                )
                rows += 1
    db.execute("INSERT INTO passages_fts(rowid, text, section, figure, work) "
               "SELECT rowid, text, section, figure, work FROM passages")
    db.commit()
    print(f"indexed {rows:,} passages into {db_path}")


if __name__ == "__main__":
    main()
