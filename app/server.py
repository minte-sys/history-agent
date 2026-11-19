"""Web server: a question page plus a JSON API.

Run locally:  cd app && uvicorn server:app --port 8000
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agent import ModelBusy, ModelError, answer
from search import Corpus

app = FastAPI(title="Primary Source History Agent")
corpus = Corpus()
STATIC = Path(__file__).parent / "static"

# Simple per-IP rate limit so a public page can't run up the API bill.
RATE = 20          # questions
WINDOW = 3600      # per hour
_hits: dict[str, deque] = defaultdict(deque)


def _allow(ip: str) -> bool:
    now, q = time.time(), _hits[ip]
    while q and now - q[0] > WINDOW:
        q.popleft()
    if len(q) >= RATE:
        return False
    q.append(now)
    return True


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/library")
def library():
    return corpus.figures()


@app.get("/api/passage/{pid}")
def passage(pid: str):
    p = corpus.get(pid)
    if not p:
        raise HTTPException(404, "No such passage")
    return p


@app.post("/api/ask")
def ask(body: Ask, request: Request):
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0]
    if not _allow(ip):
        raise HTTPException(429, "Too many questions from you this hour. Please try again later.")
    try:
        return answer(body.question.strip(), corpus)
    except ModelBusy:
        raise HTTPException(503, "The service is busy. Please try again in a minute.")
    except ModelError:
        raise HTTPException(502, "Something went wrong while answering. Please try again.")


@app.get("/healthz")
def healthz():
    return {"ok": True}
