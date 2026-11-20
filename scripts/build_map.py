"""Build corpus/map.json: pins for the map and timeline, each with a verbatim quote.

Reads the curated pins in corpus/places.json. For each pin it searches only the named section
for the pin's search words, picks the best-matching sentence, and checks it appears word for word
in the passage (so no quote on the map is typed by hand). Napoleon's letters are pinned
automatically from the place and date written at the top of each letter, using the gazetteer below.

Run after build_index.py:  python scripts/build_map.py
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
from agent import quote_in  # noqa: E402

DB = ROOT / "corpus" / "index.sqlite"
PLACES = ROOT / "corpus" / "places.json"
OUT = ROOT / "corpus" / "map.json"
MAX_WORDS = 60

# Places named at the head of Napoleon's letters -> (display name, lat, lon).
GAZETTEER = {
    "Chanceaux Post House": ("Chanceaux, Burgundy", 47.51, 4.72), "Marmirolo": ("Marmirolo, Italy", 45.22, 10.76),
    "Milan": ("Milan", 45.46, 9.19), "Verona": ("Verona", 45.44, 10.99), "Castiglione": ("Castiglione delle Stiviere", 45.39, 10.49),
    "Brescia": ("Brescia", 45.54, 10.22), "Roverbella": ("Roverbella, Italy", 45.27, 10.77), "Ala": ("Ala, Trentino", 45.76, 11.0),
    "Ronco": ("Ronco all'Adige", 45.34, 11.24), "Modena": ("Modena", 44.65, 10.93), "Forli": ("Forlì", 44.22, 12.04),
    "Ancona": ("Ancona", 43.62, 13.52), "Tolentino": ("Tolentino", 43.21, 13.28), "Lausanne": ("Lausanne", 46.52, 6.63),
    "Torre di Garofolo": ("Torre di Garofoli, near Marengo", 44.9, 8.7), "Malmaison": ("Malmaison, near Paris", 48.87, 2.17),
    "Paris": ("Paris", 48.86, 2.35), "St. Cloud": ("Saint-Cloud", 48.84, 2.22), "Trianon": ("Trianon, Versailles", 48.81, 2.11),
    "Fontainebleau": ("Fontainebleau", 48.4, 2.7), "Rambouillet": ("Rambouillet", 48.64, 1.83),
    "Pont-de-Bricques": ("Pont-de-Briques, near Boulogne", 50.69, 1.62), "Boulogne": ("Boulogne", 50.73, 1.61),
    "Calais": ("Calais", 50.95, 1.86), "Ostend": ("Ostend", 51.22, 2.93), "Treves": ("Trier", 49.75, 6.64),
    "Ettlingen": ("Ettlingen", 48.94, 8.41), "Louisbourg": ("Ludwigsburg", 48.9, 9.19),
    "Abbaye d'Elchingen": ("Elchingen Abbey, near Ulm", 48.45, 10.1), "Elchingen": ("Elchingen Abbey, near Ulm", 48.45, 10.1),
    "Augsburg": ("Augsburg", 48.37, 10.9), "Haag": ("Haag, Bavaria", 48.16, 12.18), "Vienna": ("Vienna", 48.21, 16.37),
    "Schoenbrunn": ("Schönbrunn, Vienna", 48.18, 16.31), "Brunn": ("Brno", 49.19, 16.61), "Austerlitz": ("Austerlitz", 49.15, 16.88),
    "Munich": ("Munich", 48.14, 11.58), "Bamberg": ("Bamberg", 49.89, 10.89), "Gera": ("Gera", 50.88, 12.08),
    "Jena": ("Jena", 50.93, 11.59), "Weimar": ("Weimar", 50.98, 11.33), "Wittenberg": ("Wittenberg", 51.87, 12.65),
    "Potsdam": ("Potsdam", 52.39, 13.06), "Berlin": ("Berlin", 52.52, 13.4), "Kustrin": ("Küstrin", 52.57, 14.64),
    "Meseritz": ("Międzyrzecz", 52.44, 15.58), "Posen": ("Poznań", 52.41, 16.93), "Warsaw": ("Warsaw", 52.23, 21.01),
    "Golymin": ("Gołymin", 52.81, 20.87), "Pultusk": ("Pułtusk", 52.7, 21.08), "Willemberg": ("Wielbark", 53.4, 20.95),
    "Preussich-Eylau": ("Eylau (Bagrationovsk)", 54.39, 20.64), "Eylau": ("Eylau (Bagrationovsk)", 54.39, 20.64),
    "Landsberg": ("Górowo Iławeckie", 54.28, 20.49), "Liebstadt": ("Miłakowo", 54.02, 20.07),
    "Osterode": ("Ostróda", 53.7, 19.97), "Finckenstein": ("Finckenstein Palace (Kamieniec)", 53.68, 19.49),
    "Dantzig": ("Gdańsk", 54.35, 18.65), "Marienburg": ("Malbork", 54.04, 19.03), "Friedland": ("Friedland (Pravdinsk)", 54.44, 21.01),
    "Tilsit": ("Tilsit (Sovetsk)", 55.08, 21.88), "Dresden": ("Dresden", 51.05, 13.74), "Venice": ("Venice", 45.44, 12.33),
    "Udine": ("Udine", 46.06, 13.24), "Bayonne": ("Bayonne", 43.49, -1.47), "Erfurt": ("Erfurt", 50.98, 11.03),
    "Tolosa": ("Tolosa, Spain", 43.13, -2.08), "Burgos": ("Burgos", 42.34, -3.7), "Chamartin": ("Chamartín, Madrid", 40.46, -3.68),
    "Madrid": ("Madrid", 40.42, -3.7), "Benavento": ("Benavente, Spain", 42.0, -5.68), "Donauwoerth": ("Donauwörth", 48.72, 10.78),
    "Enns": ("Enns", 48.21, 14.48), "Saint-Polten": ("Sankt Pölten", 48.2, 15.62), "Ebersdorf": ("Kaiserebersdorf, Vienna", 48.16, 16.47),
    "Wolkersdorf": ("Wolkersdorf", 48.38, 16.52), "Krems": ("Krems", 48.41, 15.6), "Gumbinnen": ("Gumbinnen (Gusev)", 54.59, 22.2),
}
NAP_HEAD = re.compile(r"_(?:Imperial Headquarters, )?([A-Z][^_,\d]{2,40})_?,\s*_?(?:the )?\"?(?:\w+ \d+(?:st|nd|rd|th)?,?\s*)?(\d{4})")


def sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+(?=[\"“‘_A-Z])", text) if s.strip()]


def strip_heading(text: str, section: str) -> str:
    """Drop the chapter heading that opens a section's first passage ("XII  Seventh year of...")."""
    title = [w for w in re.sub(r"[:\u2060]", " ", section).split()]
    words = re.sub("\u2060", " ", text).split()
    k = len(title)
    if k and [w.strip(".") for w in words[:k]] == [w.strip(".") for w in title]:
        return " ".join(words[k:])
    return text


def best_sentence(text: str, terms: str) -> str | None:
    words = [w.lower() for w in re.findall(r"[\w']+", terms) if len(w) > 2]
    best, best_score = None, 0.0
    for s in sentences(text):
        n = len(s.split())
        if n < 6:
            continue
        low = s.lower()
        score = sum(1 for w in words if w in low) - (0.02 * max(0, n - 40))
        if score > best_score:
            best, best_score = s, score
    if best is None:
        return None
    ws = best.split()
    if len(ws) > MAX_WORDS:   # keep the window with the most search words, still contiguous and verbatim
        hit = [any(t in w.lower() for t in words) for w in ws]
        start = max(range(len(ws) - MAX_WORDS + 1), key=lambda i: (sum(hit[i:i + MAX_WORDS]), i))
        best = " ".join(ws[start:start + MAX_WORDS])
    return best.strip("_ ").strip()


def resolve_section(db, source: str, pin: dict) -> int:
    if "order" in pin:
        return pin["order"]
    rows = db.execute("SELECT DISTINCT section_order, section FROM passages WHERE source_id = ? ORDER BY section_order",
                      (source,)).fetchall()
    exact = [o for o, s in rows if s == pin["section"]]
    if exact:
        return exact[0]
    pre = [o for o, s in rows if s.startswith(pin["section"])]
    if not pre:
        raise SystemExit(f"no section {pin['section']!r} in {source}")
    if len(pre) > 1:
        print(f"  note: {pin['section']!r} matches {len(pre)} sections in {source}; using the first")
    return pre[0]


def pick(db, source: str, order: int, terms: str):
    rows = db.execute("SELECT id, text, section FROM passages WHERE source_id = ? AND section_order = ? ORDER BY n",
                      (source, order)).fetchall()
    words = [w.lower() for w in re.findall(r"[\w']+", terms) if len(w) > 2]
    scored = sorted(rows, key=lambda r: -sum(r[1].lower().count(w) > 0 for w in words))
    for pid, text, section in scored:
        q = best_sentence(strip_heading(text, section), terms)
        if q and quote_in(q, text):
            return pid, q, section
    return None


def _cut(quote: str, db, pid: str) -> bool:
    """True when the quote starts or ends mid-sentence, so the page can show an ellipsis."""
    starts_mid = quote[:1].islower()
    ends_mid = not quote.rstrip("\"”’'_)").endswith((".", "!", "?", ";", ":"))
    return starts_mid or ends_mid


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60]


def year_label(y: int, approx: bool) -> str:
    s = f"{-y} BCE" if y < 0 else (f"{y} CE" if y < 600 else str(y))
    return ("c. " + s) if approx else s


def main():
    db = sqlite3.connect(DB)
    meta = {r[0]: {"figure": r[1], "work": r[2], "date": r[3], "translator": r[4], "author": r[5]} for r in db.execute(
        "SELECT source_id, figure, work, date, translator, author FROM passages GROUP BY source_id")}
    pins, misses = [], []
    for pin in json.loads(PLACES.read_text())["pins"]:
        src = pin["source"]
        if pin.get("auto") == "letters":
            pins += napoleon(db, src, meta[src])
            continue
        order = resolve_section(db, src, pin)
        got = pick(db, src, order, pin["terms"])
        if not got:
            misses.append(pin["title"])
            continue
        pid, quote, section = got
        pins.append({"id": slug(f"{src.split('-')[0]}-{pin['title']}"), "source_id": src, "title": pin["title"],
                     "place": pin["place"], "lat": pin["lat"], "lon": pin["lon"], "year": pin["year"],
                     "year_label": year_label(pin["year"], pin.get("approx", False)),
                     "passage_id": pid, "quote": quote, "section": section,
                     "cut": _cut(quote, db, pid), **meta[src]})
    pins.sort(key=lambda p: (p["year"], p["source_id"]))
    ids = [p["id"] for p in pins]
    assert len(ids) == len(set(ids)), "pin ids must be unique"
    sections = {}
    for p in pins:
        sid, so = p["passage_id"].split(":")[0], p["passage_id"].split(":")[1].split(".")[0]
        sections.setdefault(f"{sid}:{so}", p["id"])
    OUT.write_text(json.dumps({"pins": pins, "sections": sections}, ensure_ascii=False, indent=0))
    print(f"{len(pins)} pins written to {OUT.relative_to(ROOT)}")
    if misses:
        print("no quote found for:", ", ".join(misses))


def napoleon(db, src: str, meta: dict) -> list[dict]:
    out = []
    rows = db.execute("SELECT id, text, section, section_order FROM passages WHERE source_id = ? AND n = 1 ORDER BY section_order",
                      (src,)).fetchall()
    for pid, text, section, order in rows:
        m = NAP_HEAD.search(text)
        if not m or m.group(1).strip() not in GAZETTEER:
            continue
        place, lat, lon = GAZETTEER[m.group(1).strip()]
        year = int(m.group(2))
        nl = text.find("\n", m.end())
        body = text[nl:] if nl != -1 else text[m.end():]
        quote = next((s.strip("_ ") for s in sentences(body)
                      if len(s.split()) >= 8 and not s.startswith("_") and "TO JOSEPHINE" not in s and "TO THE EMPRESS" not in s), None)
        if not quote:
            continue
        ws = quote.split()
        quote = " ".join(ws[:MAX_WORDS])
        if not quote_in(quote, text):
            continue
        out.append({"id": slug(f"napoleon-{section}"), "source_id": src, "title": f"Letter to Josephine ({section})",
                    "place": place, "lat": lat, "lon": lon, "year": year, "year_label": str(year),
                    "passage_id": pid, "quote": quote, "section": section, **meta})
    return out


if __name__ == "__main__":
    main()
