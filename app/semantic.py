"""Meaning-based search: cosine similarity between the question and pre-computed passage embeddings.

Uses Gemini's free embedding model. If the embeddings file or the API is unavailable, callers fall
back to keyword search only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

EMBED_MODEL = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
DIM = 768
ROOT = Path(__file__).resolve().parent.parent / "corpus"


def doc_text(work: str, section: str, text: str) -> str:
    return f"{work}, {section}\n{text}"


class Semantic:
    def __init__(self):
        self.ok = False
        try:
            self.mat = np.load(ROOT / "embeddings.npy").astype(np.float32)
            self.ids = json.loads((ROOT / "embeddings_ids.json").read_text())
            self.ok = len(self.ids) == len(self.mat)
        except (OSError, ValueError):
            pass
        self._client = None

    def _embed(self, query: str):
        from google import genai
        from google.genai import types

        if self._client is None:
            self._client = genai.Client()
        r = self._client.models.embed_content(
            model=EMBED_MODEL, contents=[query],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=DIM))
        v = np.array(r.embeddings[0].values, dtype=np.float32)
        return v / np.linalg.norm(v)

    def search(self, query: str, allowed: set[str] | None = None, limit: int = 8) -> list[str]:
        """Return passage ids, best first. `allowed` restricts to those ids (e.g. one figure)."""
        if not self.ok:
            return []
        try:
            q = self._embed(query)
        except Exception:
            return []
        scores = self.mat @ q
        order = np.argsort(-scores)
        out = []
        for i in order:
            pid = self.ids[i]
            if allowed is None or pid in allowed:
                out.append(pid)
                if len(out) >= limit:
                    break
        return out
