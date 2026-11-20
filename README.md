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
3. **Meaning-based search** (`app/semantic.py`): word vectors (spaCy `en_core_web_md`, shipped as
   `corpus/wordvecs.npy`) averaged per passage with SIF weighting, so related words match without an
   API or quota. Results are fused with keyword results (reciprocal rank fusion). Rebuild with
   `python scripts/build_embeddings.py` after changing the corpus. Gemini's free embedding tier allows
   only 1,000 texts a day, which is too few for 12,000 passages.
4. **Agent** (`app/agent.py`): Gemini (free tier, the default) or Claude Sonnet 5.5 (`LLM_PROVIDER=claude`) plans 2–4 keyword searches, reads the best
   ~14 passages, and answers in a fixed structure in which each claim has a quote and a passage id.
   Code then drops any claim whose quote isn't verbatim in its passage. If nothing survives,
   the answer becomes "Not in the sources".
5. **Web** (`app/server.py`, `app/static/index.html`): FastAPI with one page, `/api/ask`,
   `/api/passage/{id}`, `/api/context/{id}` (source viewer) and `/api/library`. Compare questions
   ("How do X and Y describe…") are searched per figure and shown side by side. Voice input and
   read-aloud use the browser's built-in speech features. A per-IP limit allows 20 questions an hour.

6. **Ask the figure** (`app/figure.py`, `/api/figures`, `/api/figure`): chat with one of 14 figures
   whose texts are their own words. Each reply is 1 to 4 lines; every line is a first-person
   paraphrase plus the exact quote behind it, checked word for word and required to come from that
   figure's own text. Questions their writings don't cover (modern things, events after their death)
   get an out-of-character "not covered" instead of an invented answer. Anonymous chronicles, the
   three-author Federalist and Thucydides (built around other people's speeches) are left out.
7. **Myth buster** (`app/mythbuster.py`, `/api/check`): paste a quote or claim and get **Verified**,
   **Misquoted** (wording changed, meaning twisted, or someone else wrote it) or **Not found**, with the
   real passage. Candidates come from 3-word phrase search plus keyword and meaning search; code
   fuzzy-matches the wording, and a "verified" verdict needs at least 90% word-level similarity to a real
   passage by the person named. If the model is busy, exact quotes are still checked by code alone.
   Results have shareable links (`/?check=...#myth`).
8. **Debate** (`app/debate.py`, `/api/debate`): put one question to two to four of the figures and get
   their answers side by side. Each side is built only from that figure's own text, every line carries
   a quote checked word for word, and a quote from anyone else's text is rejected. Figures whose
   writings don't touch the question sit it out. A short "where they agree / differ" note is written
   afterwards from the verified lines only. Three model calls per debate, whatever the number of
   figures. Results have shareable links (`/?debate=...&with=a,b#debate`).
9. **Map and timeline** (`corpus/places.json`, `scripts/build_map.py`, `/api/map`): about 260 moments
   pinned to a place and year, shown on an OpenStreetMap map (Leaflet, served from `app/static/vendor`)
   with a timeline, era filters and a "play" mode. `places.json` holds the curated pins (place, year,
   section and search words); the build script picks each quote from the text itself and checks it is
   verbatim. Napoleon's letters are pinned automatically from the place and date at the head of each
   letter. Citations elsewhere on the site get an "On the map" link when their section is pinned.
   Places and years are approximate editorial placements, and the page says so.
10. **Voices** (`app/static/index.html`): in Ask the figure and Debate the figures speak their replies
   aloud, using the browser's free built-in text to speech (Web Speech API), so there is no server cost.
   Each figure has a voice profile (preferred accent, such as British for Caesar, Irish then British for
   Shackleton and Indian English for Gandhi, plus pitch and pace), matched to the voices on the
   visitor's device; figures in one debate get different voices where the device has enough. Debates
   play as turns (each side's first line, then each side's second), then a narrator voice reads where
   they agree and differ. Visitors can switch auto-play off, or have the figures read their original
   quotes instead of the plain version. Voices differ by device and browser.

## Run locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...               # free key from aistudio.google.com; never commit it
# or: export LLM_PROVIDER=claude ANTHROPIC_API_KEY=...
python scripts/fetch_sources.py         # only needed to rebuild the corpus
python scripts/build_index.py
python scripts/build_embeddings.py     # needs spaCy + en_core_web_md, see the script
python scripts/build_map.py            # map pins, after build_index.py
cd app && uvicorn server:app --port 8000
```

## Test

```bash
python tests/run_eval.py                # 38 questions: answers, refusals and comparisons
python tests/run_features.py            # Ask the figure, Myth buster, Debate and map: 16 checks
```

## Add a source

Add an entry to `corpus/sources.json`. Use `standardebooks` with the repo name, or `gitenberg`
with a splitter in `scripts/split_gutenberg.py`. Then run
`python scripts/fetch_sources.py <id>` followed by `python scripts/build_index.py`. Check the new text for material
that isn't by the figure, such as introductions, notes, or letters from other people.

## Deploy

`render.yaml` defines a Render web service. Set `GEMINI_API_KEY` in the Render dashboard (or `LLM_PROVIDER=claude` plus `ANTHROPIC_API_KEY`).
