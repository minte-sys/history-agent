"""Build the word-vector files used for meaning-based search (see app/semantic.py).

Needs spaCy's en_core_web_md locally (only to build; the web app reads the .npy files):
    pip install spacy https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl
"""
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
from semantic import SIF_A, tokens  # noqa: E402


def main():
    import spacy

    vocab_vecs = spacy.load("en_core_web_md", exclude=["parser", "ner", "tagger", "lemmatizer",
                                                       "attribute_ruler", "senter", "tok2vec"]).vocab
    db = sqlite3.connect(ROOT / "corpus" / "index.sqlite")
    rows = db.execute("SELECT id, section, text FROM passages ORDER BY rowid").fetchall()
    toks = [tokens(f"{s} {t}") for _, s, t in rows]
    freq = Counter(w for ts in toks for w in ts)
    total = sum(freq.values())

    words, vecs = [], []
    for w, _ in freq.most_common():
        if vocab_vecs.has_vector(w):
            words.append(w)
            vecs.append(vocab_vecs.get_vector(w))
    wv = np.array(vecs, dtype=np.float32)
    weight = np.array([SIF_A / (SIF_A + freq[w] / total) for w in words], dtype=np.float32)
    index = {w: i for i, w in enumerate(words)}

    mat = np.zeros((len(rows), wv.shape[1]), dtype=np.float32)
    for i, ts in enumerate(toks):
        r = [index[t] for t in ts if t in index]
        if r:
            mat[i] = (wv[r] * weight[r, None]).sum(0) / len(r)
    # Remove the first principal component (shared "topic-free" direction), as in SIF.
    _, _, vt = np.linalg.svd(mat - mat.mean(0), full_matrices=False)
    pc = vt[0]
    mat -= np.outer(mat @ pc, pc)
    mat /= np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9)

    out = ROOT / "corpus"
    np.save(out / "wordvecs.npy", wv.astype(np.float16))
    (out / "wordvecs_vocab.json").write_text(json.dumps(words))
    np.savez(out / "wordvecs_meta.npz", weight=weight, pc=pc)
    np.save(out / "passage_vecs.npy", mat.astype(np.float16))
    (out / "passage_vecs_ids.json").write_text(json.dumps([r[0] for r in rows]))
    print(f"{len(words):,} words with vectors; {len(rows):,} passages")


if __name__ == "__main__":
    main()
