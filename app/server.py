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
from figure import Turn, personas, reply
from mythbuster import check
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


class FigureMsg(BaseModel):
    figure: str = Field(max_length=80)
    message: str = Field(min_length=2, max_length=500)
    history: list[Turn] = Field(default_factory=list, max_length=20)


class CheckBody(BaseModel):
    text: str = Field(min_length=8, max_length=600)


def _client_ip(request: Request) -> str:
    return request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0]


def _guard(request: Request):
    if not _allow(_client_ip(request)):
        raise HTTPException(429, "Too many questions from you this hour. Please try again later.")


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


@app.get("/api/context/{pid}")
def context(pid: str, before: int = 1, after: int = 1):
    """The passage plus its neighbours in the same section, for the source viewer."""
    p = corpus.get(pid)
    if not p:
        raise HTTPException(404, "No such passage")
    before, after = max(0, min(before, 5)), max(0, min(after, 5))
    return {"passage": p, "neighbours": corpus.neighbours(pid, before, after),
            "section_passages": corpus.section_size(pid)}


@app.post("/api/ask")
def ask(body: Ask, request: Request):
    _guard(request)
    try:
        return answer(body.question.strip(), corpus)
    except ModelBusy:
        raise HTTPException(503, "The service is busy. Please try again in a minute.")
    except ModelError:
        raise HTTPException(502, "Something went wrong while answering. Please try again.")


@app.get("/api/figures")
def figures():
    """Figures you can chat with: those whose texts are their own words."""
    return personas(corpus)


@app.post("/api/figure")
def figure_chat(body: FigureMsg, request: Request):
    """Ask the figure: a reply built only from that figure's own words."""
    if body.figure not in {p["id"] for p in personas(corpus)}:
        raise HTTPException(404, "That figure isn't in the library.")
    _guard(request)
    try:
        return reply(body.figure, body.message.strip(), body.history, corpus)
    except ModelBusy:
        raise HTTPException(503, "The service is busy. Please try again in a minute.")
    except ModelError:
        raise HTTPException(502, "Something went wrong while answering. Please try again.")


@app.post("/api/check")
def myth_check(body: CheckBody, request: Request):
    """Myth buster: is this quote or claim in the sources? Works without the model for exact quotes."""
    _guard(request)
    return check(body.text.strip(), corpus)


@app.get("/healthz")
def healthz():
    return {"ok": True}
