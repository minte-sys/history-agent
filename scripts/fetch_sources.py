"""Download the public-domain texts listed in corpus/sources.json.

Writes corpus/sections.jsonl: one record per section (chapter, book, letter...)
with the metadata needed to cite it. Only text written by the figure is kept;
editors' introductions, translators' prefaces and similar front matter are
skipped.
"""
import html
import json
import re
import sys
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
CACHE = CORPUS / "cache"
UA = {"User-Agent": "history-agent-corpus-builder/0.1"}

SE_RAW = "https://raw.githubusercontent.com/standardebooks/{repo}/master/src/epub/{path}"
SE_PAGE = "https://standardebooks.org/ebooks/{slug}/text/{name}"
GIT_RAW = "https://raw.githubusercontent.com/GITenberg/{repo}/master/{gid}.txt"
PG_PAGE = "https://www.gutenberg.org/ebooks/{gid}"

MAIN_PREFIXES = ("chapter-", "book-", "part-")


def get(url: str) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = CACHE / re.sub(r"[^A-Za-z0-9._-]", "_", url)[-180:]
    if key.exists():
        return key.read_text(encoding="utf-8")
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                text = r.read().decode("utf-8")
            key.write_text(text, encoding="utf-8")
            return text
        except Exception as e:  # network hiccup: back off and retry
            if attempt == 3:
                raise
            print(f"  retry {url}: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)
    raise RuntimeError(url)


class XhtmlText(HTMLParser):
    """Turn a Standard Ebooks XHTML file into plain paragraphs.

    Drops footnote reference markers (<a epub:type="noteref">) so the stored
    text is what the author wrote, not the edition's apparatus.
    """

    BLOCKS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "tr", "header", "br"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.title, self._skip, self._in_title = [], "", 0, False
        self._heading = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if self._skip:
            if tag in ("a", "sup", "span", "title"):
                self._skip += 1
            return
        if tag == "a" and "noteref" in (a.get("epub:type") or ""):
            self._skip = 1
            return
        if tag == "title":
            self._in_title = True
        if tag in self.BLOCKS:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if self._skip:
            if tag in ("a", "sup", "span", "title"):
                self._skip -= 1
            return
        if tag == "title":
            self._in_title = False
        if tag in self.BLOCKS:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_title:
            self.title += data
            return
        self.parts.append(data)

    def text(self):
        t = "".join(self.parts)
        t = re.sub(r"[ \t\r\f\v]+", " ", t)
        t = re.sub(r" *\n *", "\n", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip()


def se_sections(src):
    opf = get(SE_RAW.format(repo=src["repo"], path="content.opf"))
    manifest = dict(re.findall(r'<item[^>]*id="([^"]+)"[^>]*href="(text/[^"]+\.xhtml)"', opf))
    manifest.update({k: v for v, k in re.findall(r'<item[^>]*href="(text/[^"]+\.xhtml)"[^>]*id="([^"]+)"', opf)})
    spine = re.findall(r'<itemref[^>]*idref="([^"]+)"', opf)
    slug = src["repo"].replace("_", "/")
    order = 0
    for idref in spine:
        href = manifest.get(idref)
        if not href:
            continue
        name = href[len("text/"):-len(".xhtml")]
        if not (name.startswith(MAIN_PREFIXES) or name in src.get("extra", [])):
            continue
        p = XhtmlText()
        p.feed(get(SE_RAW.format(repo=src["repo"], path=href)))
        body = p.text()
        if len(body) < 40:
            continue
        order += 1
        yield {
            "section": html.unescape(p.title).strip() or name,
            "order": order,
            "url": SE_PAGE.format(slug=slug, name=name),
            "text": body,
        }


def strip_gutenberg(raw: str) -> str:
    raw = raw.replace("\r\n", "\n")
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG[^\n]*\n", raw)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", raw)
    return raw[start.end() if start else 0 : end.start() if end else len(raw)].strip()


def git_sections(src):
    text = strip_gutenberg(get(GIT_RAW.format(repo=src["repo"], gid=src["gutenberg_id"])))
    out = CORPUS / "raw" / f"{src['id']}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    # Split into sections in a per-source splitter (see split_gutenberg.py).
    from split_gutenberg import SPLITTERS

    url = PG_PAGE.format(gid=src["gutenberg_id"])
    for i, (title, body) in enumerate(SPLITTERS[src["id"]](text), 1):
        yield {"section": title, "order": i, "url": url, "text": body}


def main():
    sys.path.insert(0, str(Path(__file__).parent))
    sources = json.loads((CORPUS / "sources.json").read_text(encoding="utf-8"))
    only = set(sys.argv[1:])
    out_path = CORPUS / "sections.jsonl"
    existing = []
    if only and out_path.exists():
        existing = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines()]
        existing = [r for r in existing if r["source_id"] not in only]
    records = list(existing)
    for src in sources:
        if only and src["id"] not in only:
            continue
        gen = se_sections if src["kind"] == "standardebooks" else git_sections
        n = chars = 0
        for sec in gen(src):
            meta = {k: src.get(k) for k in ("figure", "author", "work", "translator", "editor", "date")}
            records.append({"source_id": src["id"], **meta, **sec})
            n += 1
            chars += len(sec["text"])
        print(f"{src['id']}: {n} sections, {chars:,} chars")
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} sections to {out_path}")


if __name__ == "__main__":
    main()
