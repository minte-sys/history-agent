# Primary Source History

Ask a question about a famous historical figure. The answer is built only from that
figure's own writings (in public-domain translations), every claim carries a quote that
code checks word for word against the source, and if the sources don't cover the
question the agent says so instead of guessing.

## How it works

1. **Library**: `corpus/sources.json` lists the texts. `scripts/fetch_sources.py` downloads them
   (Standard Ebooks and Project Gutenberg mirrors), keeps only the figure's own words (editors'
   introductions, translators' prefaces and letters written *to* the figure are removed), and
   writes `corpus/sections.jsonl`.
2. **Search**: `scripts/build_index.py` splits sections into roughly 1,200-character passages at
   paragraph breaks, giving each one a stable id (`lincoln-papers:944.1`). It then builds a SQLite FTS5
   index at `corpus/index.sqlite`.
3. **Agent** (`app/agent.py`): Gemini (free tier, the default) or Claude Sonnet 5.5 (`LLM_PROVIDER=claude`) plans 2–4 keyword searches, reads the best
   ~14 passages, and answers in a fixed structure in which each claim has a quote and a passage id.
   Code then drops any claim whose quote isn't verbatim in its passage. If nothing survives,
   the answer becomes "Not in the sources".
4. **Web** (`app/server.py`, `app/static/index.html`): FastAPI with one page, `/api/ask`,
   `/api/passage/{id}` and `/api/library`. A per-IP limit allows 20 questions an hour.

## Run locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...               # free key from aistudio.google.com; never commit it
# or: export LLM_PROVIDER=claude ANTHROPIC_API_KEY=...
python scripts/fetch_sources.py         # only needed to rebuild the corpus
python scripts/build_index.py
cd app && uvicorn server:app --port 8000
```

## Test

```bash
python tests/run_eval.py                # 30 questions, including 10 it should decline
```

## Add a source

Add an entry to `corpus/sources.json`. Use `standardebooks` with the repo name, or `gitenberg`
with a splitter in `scripts/split_gutenberg.py`. Then run
`python scripts/fetch_sources.py <id>` followed by `python scripts/build_index.py`. Check the new text for material
that isn't by the figure, such as introductions, notes, or letters from other people.

## Deploy

`render.yaml` defines a Render web service. Set `GEMINI_API_KEY` in the Render dashboard (or `LLM_PROVIDER=claude` plus `ANTHROPIC_API_KEY`).
