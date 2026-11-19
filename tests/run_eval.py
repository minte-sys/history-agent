"""Run the question set through the agent and score it.

Checks, per question:
  - verdict matches what we expect (answer vs. "not in the sources")
  - for answerable questions, at least one surviving claim cites the expected source
  - every surviving quote is verbatim in its passage (the agent enforces this; we re-check)
Writes tests/results.json and prints a summary.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from agent import ModelBusy, answer, quote_in  # noqa: E402
from search import Corpus  # noqa: E402

HERE = Path(__file__).parent
PAUSE = float(os.environ.get("EVAL_PAUSE", "6"))  # seconds between questions, for free-tier limits


def main():
    corpus = Corpus()
    cases = json.loads((HERE / "questions.json").read_text())
    if len(sys.argv) > 1:
        cases = [c for i, c in enumerate(cases) if str(i) in sys.argv[1:]]
    results, passed = [], 0
    for i, c in enumerate(cases):
        t = time.time()
        try:
            for attempt in range(5):  # free-tier rate limits: back off and retry
                try:
                    r = answer(c["q"], corpus)
                    break
                except ModelBusy:
                    if attempt == 4:
                        raise
                    time.sleep(15 * (attempt + 1))
            time.sleep(PAUSE)
        except Exception as e:  # keep going; record the failure
            r = {"verdict": "error", "summary": repr(e), "claims": [], "dropped": []}
        ok_verdict = (
            r["verdict"] == c["expect"]
            or (c["expect"] == "answered_or_partial" and r["verdict"] in ("answered", "partial"))
            or (c["expect"] == "answered" and r["verdict"] == "partial")
        )
        ok_cite = "must_cite" not in c or any(cl["passage_id"].startswith(c["must_cite"] + ":") for cl in r["claims"])
        ok_quotes = all(quote_in(cl["quote"], corpus.get(cl["passage_id"])["text"]) for cl in r["claims"])
        ok_compare = not c.get("compare") or r.get("compare")
        ok = ok_verdict and ok_cite and ok_quotes and ok_compare
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'} [{time.time() - t:4.1f}s] {r['verdict']:<14} {c['q']}"
              + ("" if ok else f"  (expected {c['expect']}; cite ok={ok_cite}; quotes ok={ok_quotes}; compare ok={ok_compare})"))
        results.append({**c, "result": r, "pass": ok})
    (HERE / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    dropped = sum(len(x["result"].get("dropped", [])) for x in results)
    print(f"\n{passed}/{len(cases)} passed; {dropped} claims dropped for unverifiable quotes")


if __name__ == "__main__":
    main()
