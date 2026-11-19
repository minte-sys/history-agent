"""Answer a history question from the corpus only, with verified quotes.

Pipeline (code-controlled, two model calls):
  1. plan:   Claude turns the question into a few keyword searches (and a figure filter).
  2. search: run them against the SQLite index, keep the best distinct passages.
  3. answer: Claude writes an answer whose every claim rests on a quote from those passages,
             or says the sources don't cover it.
  4. verify: code checks each quote appears word for word in the cited passage. Claims whose
             quote fails are dropped; if nothing survives, the answer becomes "not supported".
"""
from __future__ import annotations

import os
import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, Field

from search import Corpus

# Which model answers. "gemini" (free tier, default) or "claude" (paid, stronger).
PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()
# Tried in order; the next is used when one is overloaded or times out.
GEMINI_MODELS = os.environ.get("GEMINI_MODELS", "gemini-3.5-flash-lite,gemini-3.5-flash,gemini-3.1-flash-lite").split(",")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_PASSAGES = 14

_clients: dict = {}


class ModelError(RuntimeError):
    """The model call failed or returned nothing usable."""


class ModelBusy(ModelError):
    """Rate limit or quota hit (common on Gemini's free tier)."""


# ---------- structured outputs ----------

class SearchPlan(BaseModel):
    figure: str | None = Field(description="The one figure from the list the question is about, exactly as listed, or null if several or none.")
    queries: list[str] = Field(description="2 to 4 short keyword searches, using words the source itself would likely use.")


class Claim(BaseModel):
    statement: str = Field(description="One sentence of the answer, in plain words.")
    quote: str = Field(description="The exact words from the passage that support the statement, copied character for character, 5 to 60 words.")
    passage_id: str = Field(description="The id of the passage the quote comes from.")


class Answer(BaseModel):
    verdict: Literal["answered", "partial", "not_supported"]
    summary: str = Field(description="A short direct answer in 1 to 3 sentences, or an explanation of what the sources do and don't cover.")
    claims: list[Claim]
    caveats: str | None = Field(description="Translation, date or attribution issues the reader should know, or null.")


# ---------- prompts ----------

PLAN_SYSTEM = """You plan keyword searches over a library of primary historical sources.
The library holds only these works:
{catalog}

Return the figure the question is about (exactly as named above) if it is about one of them, and
2 to 4 short keyword queries. Use concrete words the original text (or its old English translation)
would contain: names, places, objects, period vocabulary. Avoid modern paraphrase."""

ANSWER_SYSTEM = """You answer history questions using ONLY the primary-source passages provided.

Rules:
- Every claim must be backed by a quote copied exactly, character for character, from one passage,
  with that passage's id. Do not fix spelling, punctuation or archaic wording inside quotes. Do not
  join text from different places with "..."; use separate claims instead.
- Use nothing from memory or general knowledge, even if you are confident it is true. If the passages
  don't contain the answer, set verdict to "not_supported", leave claims empty, and say plainly what
  the sources do cover that is closest to the question.
- Use "partial" when the passages answer only part of the question, and say which part is missing.
- These are the figure's own words (or a translation of them). Reflect that: "Caesar writes that...",
  not "Caesar did...". Note when a translation, the author's bias, or the date limits what the text shows.
- Keep it short: at most 6 claims."""


def catalog_text(corpus: Corpus) -> str:
    lines = []
    for w in corpus.figures():
        tr = f", tr. {w['translator']}" if w["translator"] else ""
        lines.append(f"- {w['figure']}: {w['work']} ({w['date']}{tr})")
    return "\n".join(lines)


def _call(system: str, user: str, schema: type[BaseModel], effort: str) -> BaseModel:
    return _call_claude(system, user, schema, effort) if PROVIDER == "claude" else _call_gemini(system, user, schema, effort)


def _call_gemini(system, user, schema, effort):
    from google import genai
    from google.genai import errors, types

    client = _clients.get("gemini") or _clients.setdefault(
        "gemini", genai.Client(http_options=types.HttpOptions(timeout=45_000)))  # reads GEMINI_API_KEY
    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_schema=schema,
        temperature=0.2,
    )
    last: Exception | None = None
    for model in (m.strip() for m in GEMINI_MODELS if m.strip()):
        try:
            resp = client.models.generate_content(model=model, contents=user, config=config)
        except errors.ClientError as e:
            if e.code in (404, 429):   # model unavailable or quota used up: try the next one
                last = e
                continue
            raise ModelError(str(e)) from e
        except Exception as e:         # 5xx, overload, timeout, dropped connection
            last = e
            continue
        parsed = resp.parsed
        if parsed is None:
            try:
                parsed = schema.model_validate_json(resp.text or "")
            except Exception as e:
                last = e
                continue
        return parsed
    raise ModelBusy(f"all Gemini models failed: {last}")


def _call_claude(system, user, schema, effort):
    import anthropic

    client = _clients.get("claude") or _clients.setdefault("claude", anthropic.Anthropic())
    try:
        resp = client.beta.messages.parse(
            model=CLAUDE_MODEL,
            max_tokens=8000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
            output_config={"effort": effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
    except anthropic.RateLimitError as e:
        raise ModelBusy(str(e)) from e
    except anthropic.APIStatusError as e:
        if e.status_code >= 500:
            raise ModelBusy(str(e)) from e
        raise ModelError(str(e)) from e
    except anthropic.APIConnectionError as e:
        raise ModelBusy(str(e)) from e
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        raise ModelError(f"model returned no answer (stop_reason={resp.stop_reason})")
    return resp.parsed_output


# ---------- quote verification ----------

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "—": "-", "–": "-",
                         "⁠": "", " ": "", " ": " "})


def normalise(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).translate(_QUOTES)
    s = s.replace("--", "-")
    s = re.sub(r"[_*]", "", s)          # italics markers in Gutenberg text
    return re.sub(r"\s+", " ", s).strip().lower()


def quote_in(quote: str, text: str) -> bool:
    q = normalise(quote).strip(" .,;:'\"")
    return len(q) >= 12 and q in normalise(text)


# ---------- pipeline ----------

def gather(corpus: Corpus, question: str, plan: SearchPlan):
    seen, out = set(), []
    queries = [question] + plan.queries
    per_query = max(4, MAX_PASSAGES // len(queries) + 1)
    for q in queries:
        for figure in ([plan.figure, None] if plan.figure else [None]):
            for p in corpus.search(q, figure=figure, limit=per_query):
                if p["id"] not in seen:
                    seen.add(p["id"])
                    out.append(p)
            if plan.figure and len(out) >= 4:
                break
    return out[:MAX_PASSAGES]


def format_passages(passages) -> str:
    blocks = []
    for p in passages:
        meta = f"{p['author']}, {p['work']}, {p['section']}"
        if p["translator"]:
            meta += f" (tr. {p['translator']})"
        blocks.append(f'<passage id="{p["id"]}" source="{meta}">\n{p["text"]}\n</passage>')
    return "\n\n".join(blocks)


def answer(question: str, corpus: Corpus | None = None) -> dict:
    corpus = corpus or Corpus()
    catalog = catalog_text(corpus)
    plan = _call(PLAN_SYSTEM.format(catalog=catalog), question, SearchPlan, effort="low")
    passages = gather(corpus, question, plan)
    by_id = {p["id"]: p for p in passages}
    if not passages:
        return {"question": question, "verdict": "not_supported",
                "summary": "No passage in the library matches this question.", "claims": [],
                "caveats": None, "searched": plan.queries, "dropped": []}

    user = (f"Library catalogue:\n{catalog}\n\nPassages:\n{format_passages(passages)}\n\n"
            f"Question: {question}")
    ans = _call(ANSWER_SYSTEM, user, Answer, effort="medium")

    claims, dropped = [], []
    for c in ans.claims:
        p = by_id.get(c.passage_id) or corpus.get(c.passage_id)
        if p and quote_in(c.quote, p["text"]):
            claims.append({
                "statement": c.statement, "quote": c.quote, "passage_id": p["id"],
                "figure": p["figure"], "author": p["author"], "work": p["work"], "section": p["section"],
                "translator": p["translator"], "editor": p["editor"], "url": p["url"],
            })
        else:
            dropped.append({"statement": c.statement, "quote": c.quote, "passage_id": c.passage_id})

    verdict, summary = ans.verdict, ans.summary
    if verdict != "not_supported" and not claims:
        verdict = "not_supported"
        summary = ("I couldn't back an answer with exact quotes from the sources, so I won't give one. "
                   "Try rephrasing, or ask about something the listed works cover.")
    elif dropped and verdict == "answered":
        verdict = "partial"
    return {"question": question, "verdict": verdict, "summary": summary, "claims": claims,
            "caveats": ans.caveats, "searched": plan.queries, "dropped": dropped}


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(answer(" ".join(sys.argv[1:])), indent=2, ensure_ascii=False))
