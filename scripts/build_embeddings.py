"""Embed every passage with Gemini's free embedding model, for meaning-based search.

Writes corpus/embeddings.npy (float16, one row per passage, in passage rowid order) and
corpus/embeddings_ids.json. Safe to re-run: finished batches are cached in corpus/cache/emb/.
"""
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
from google import genai
from google.genai import errors, types

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
from semantic import DIM, EMBED_MODEL, doc_text  # noqa: E402

BATCH = 100
CACHE = ROOT / "corpus" / "cache" / "emb"


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(ROOT / "corpus" / "index.sqlite")
    rows = db.execute("SELECT id, work, section, text FROM passages ORDER BY rowid").fetchall()
    client = genai.Client()
    vecs = []
    for b in range(0, len(rows), BATCH):
        chunk = rows[b:b + BATCH]
        f = CACHE / f"{b:06d}_{len(chunk)}.npy"
        if f.exists():
            vecs.append(np.load(f))
            continue
        texts = [doc_text(w, s, t) for _, w, s, t in chunk]
        for attempt in range(8):
            try:
                r = client.models.embed_content(
                    model=EMBED_MODEL, contents=texts,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=DIM))
                break
            except (errors.ClientError, errors.ServerError) as e:
                wait = 20 * (attempt + 1)
                print(f"  batch {b}: {getattr(e, 'code', '?')}, waiting {wait}s", flush=True)
                time.sleep(wait)
        else:
            raise SystemExit(f"gave up at batch {b}")
        arr = np.array([e.values for e in r.embeddings], dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr.astype(np.float16)
        np.save(f, arr)
        vecs.append(arr)
        print(f"{b + len(chunk):,}/{len(rows):,}", flush=True)
        time.sleep(62)   # free tier: about 100 embedded texts per minute
    np.save(ROOT / "corpus" / "embeddings.npy", np.vstack(vecs))
    (ROOT / "corpus" / "embeddings_ids.json").write_text(json.dumps([r[0] for r in rows]))
    print("done")


if __name__ == "__main__":
    main()
