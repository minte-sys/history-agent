"""Check Ask the figure and Myth buster against known cases.

Myth buster: each case has the verdict we expect and, where there is one, the source it must cite.
Ask the figure: answerable questions must come back with verified quotes from that figure only;
out-of-period questions must come back "not covered" with no lines.
Run: python tests/run_features.py   (uses the model; pauses between calls for free-tier limits)
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from agent import ModelBusy, quote_in  # noqa: E402
from figure import reply  # noqa: E402
from mythbuster import check  # noqa: E402
from search import Corpus  # noqa: E402

PAUSE = float(os.environ.get("EVAL_PAUSE", "4"))

MYTHS = [
    ('"Government of the people, by the people, for the people" - Abraham Lincoln', "verified", "lincoln-papers"),
    ("Caesar wrote that all Gaul is divided into three parts", "verified", "caesar-gallic-war"),
    ("Sun Tzu said: all warfare is based on deception", "verified", "suntzu-art-of-war"),
    ('"It is better to be feared than loved." - Machiavelli', "misquoted", "machiavelli-prince"),
    ('"The supreme art of war is to subdue the enemy without fighting." - Abraham Lincoln', "misquoted", None),
    ('"Be the change you wish to see in the world." - Gandhi', "not_found", None),
    ('"Veni, vidi, vici." - Julius Caesar', "not_found", None),
]

FIGURES = [
    ("douglass-narrative", "How did you learn to read?", True),
    ("machiavelli-prince", "Is it better to be loved or feared?", True),
    ("shackleton-south", "What happened to the Endurance?", True),
    ("lincoln-papers", "What do you think of the iPhone?", False),
    ("caesar-gallic-war", "What did you think of the Battle of Waterloo?", False),
]


def retry(fn, *a):
    for attempt in range(4):
        try:
            return fn(*a)
        except ModelBusy:
            if attempt == 3:
                raise
            time.sleep(15 * (attempt + 1))


def main():
    corpus = Corpus()
    passed = total = 0
    for text, want, src in MYTHS:
        r = retry(check, text, corpus)
        s = r.get("source")
        ok = r["verdict"] == want and (src is None or (s and s["passage_id"].startswith(src + ":")))
        ok = ok and (not s or quote_in(s["quote"], corpus.get(s["passage_id"])["text"]))
        passed += bool(ok); total += 1
        print(f"{'PASS' if ok else 'FAIL'} myth   {r['verdict']:<10} {text}")
        time.sleep(PAUSE)
    for pid, q, covered in FIGURES:
        r = retry(reply, pid, q, [], corpus)
        own = all(l["passage_id"].startswith(pid + ":") and quote_in(l["quote"], corpus.get(l["passage_id"])["text"])
                  for l in r["lines"])
        ok = own and r["covered"] == covered
        passed += ok; total += 1
        print(f"{'PASS' if ok else 'FAIL'} figure {'covered' if r['covered'] else 'not covered':<11} {pid}: {q}")
        time.sleep(PAUSE)
    print(f"\n{passed}/{total} passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
