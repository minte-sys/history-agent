"""Meaning-based search with word vectors, run locally (no API, no quota).

Each passage and each question becomes the SIF-weighted average of its words' vectors
(GloVe-style vectors from spaCy's en_core_web_md, shipped as corpus/wordvecs.npy), with the
common direction removed. Cosine similarity then finds passages that use related words even
when they don't share the question's exact words ("ruler" ~ "king", "fight" ~ "battle").
Built by scripts/build_embeddings.py. If the files are missing, callers use keyword search only.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "corpus"
SIF_A = 1e-3
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower().replace("’", "'"))


class Semantic:
    def __init__(self):
        self.ok = False
        try:
            self.words = json.loads((ROOT / "wordvecs_vocab.json").read_text())
            self.wv = np.load(ROOT / "wordvecs.npy").astype(np.float32)
            meta = np.load(ROOT / "wordvecs_meta.npz")
            self.weight = meta["weight"].astype(np.float32)
            self.pc = meta["pc"].astype(np.float32)
            self.mat = np.load(ROOT / "passage_vecs.npy").astype(np.float32)
            self.ids = json.loads((ROOT / "passage_vecs_ids.json").read_text())
            self.index = {w: i for i, w in enumerate(self.words)}
            self.ok = len(self.ids) == len(self.mat)
        except (OSError, ValueError, KeyError):
            pass

    def embed(self, text: str):
        rows = [self.index[t] for t in tokens(text) if t in self.index]
        if not rows:
            return None
        v = (self.wv[rows] * self.weight[rows, None]).sum(0) / len(rows)
        v -= self.pc * (self.pc @ v)
        n = np.linalg.norm(v)
        return v / n if n else None

    def search(self, query: str, allowed: set[str] | None = None, limit: int = 8) -> list[str]:
        """Return passage ids, best first. `allowed` restricts to those ids (e.g. one figure)."""
        if not self.ok:
            return []
        q = self.embed(query)
        if q is None:
            return []
        order = np.argsort(-(self.mat @ q))
        out = []
        for i in order:
            pid = self.ids[i]
            if allowed is None or pid in allowed:
                out.append(pid)
                if len(out) >= limit:
                    break
        return out
