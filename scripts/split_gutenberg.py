"""Per-source splitters that turn a Gutenberg plain text into (title, text) sections.

Each splitter keeps only the figure's own words and drops the editor's
introductions, chronology notes and footnote markers.
"""
import re

FOOTNOTE_MARK = re.compile(r"\[\d+\]")


def _clean(text: str) -> str:
    text = FOOTNOTE_MARK.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    # Re-flow hard-wrapped lines inside paragraphs; keep paragraph breaks.
    paras = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    return "\n\n".join(p for p in paras if p)


def napoleon(text: str):
    """Napoleon's Letters to Josephine: letters are 'No. N.' within 'SERIES X'."""
    body = text[: text.index("\nNOTES\n")]
    series = None
    # Tokenise by SERIES headings and letter numbers.
    tokens = re.split(r"\n(SERIES [A-Z]+|No\. \d+\.)\n", body)
    for i in range(1, len(tokens), 2):
        head, chunk = tokens[i], tokens[i + 1]
        if head.startswith("SERIES"):
            series = head.split()[1]
            continue
        # The editor's chronology follows the '* * *' separator; drop it.
        chunk = re.split(r"\n\s*\*\s+\*\s+\*", chunk)[0]
        # Drop bracketed editorial notes such as "[This letter is translated from ...]".
        chunk = re.sub(r"\n\[[^\]]*\]\s*(?=\n|$)", "\n", "\n" + chunk)
        chunk = _clean(chunk)
        if series and len(chunk) > 40:
            yield f"Series {series}, Letter {head[4:-1]}", chunk


LETTERHEADS = ("EXECUTIVE MANSION", "WAR DEPARTMENT", "DEAR ", "MY DEAR", "A. LINCOLN", "SIR", "GENTLEMEN",
               "FELLOW", "WASHINGTON", "SPRINGFIELD", "HEADQUARTERS", "NAVY DEPARTMENT", "DEPARTMENT OF")


def _is_title(line: str) -> bool:
    letters = re.sub(r"[^A-Za-z]", "", line)
    if len(letters) < 6 or not line.isupper():
        return False
    return not line.startswith(LETTERHEADS)


def lincoln(text: str):
    """Papers and Writings of Abraham Lincoln: documents separated by 3+ blank lines.

    Starts at Lincoln's own writings, skipping the editors' introductions.
    """
    start = text.index("THE WRITINGS OF ABRAHAM LINCOLN, 1832-1843")
    body = text[start:]
    blocks = re.split(r"\n\s*\n\s*\n\s*\n", body)
    title, parts, year = None, [], None

    def flush():
        if title and parts:
            doc = _clean("\n\n".join(parts))
            # The editor sometimes prints other people's documents (e.g. "GENERAL GRANT'S REPLY").
            if (re.search(r"'S REPLY", title) and "LINCOLN" not in title) or re.match(r"(TELEGRAM |LETTER )?FROM ", title) or "TO A. LINCOLN" in title:
                return None
            if len(doc) > 40:
                # Date the document from its own dateline, not the volume's year headings.
                m = re.search(r"\b(18[0-6]\d)\b", doc[:300])
                yr = m.group(1) if m else None
                return (f"{title} ({yr})" if yr and yr not in title else title), doc
        return None

    for block in blocks:
        b = block.strip("\n")
        if not b.strip():
            continue
        first = b.strip().splitlines()[0].strip()
        if re.fullmatch(r"1[89]\d\d", first) and len(b.strip().splitlines()) == 1:
            year = first
            continue
        if first.startswith(("VOLUME", "THE WRITINGS OF ABRAHAM LINCOLN", "End of ", "Produced by")):
            continue
        if _is_title(first):
            r = flush()
            if r:
                yield r
            lines = b.strip().splitlines()
            # A title can wrap onto a second all-caps line.
            title_lines = [first]
            rest = lines[1:]
            while rest and rest[0].strip() and rest[0].strip().isupper() and not rest[0].strip().startswith(LETTERHEADS):
                title_lines.append(rest.pop(0).strip())
            title = " ".join(title_lines).rstrip(".,")
            parts = ["\n".join(rest)]
        else:
            parts.append(b)
    r = flush()
    if r:
        yield r


SPLITTERS = {"napoleon-letters-josephine": napoleon, "lincoln-papers": lincoln}


# ---------- added sources ----------

PAREN_NOTE = re.compile(r"\s?\((\d{1,3})\)")


def anglo_saxon_chronicle(text: str):
    """One section per year entry ("A.D. 871."). Translator's endnotes and (n) markers removed."""
    # Drop every ENDNOTES block (they run until the next entry or the next blank-line-separated heading).
    text = re.sub(r"\nENDNOTES:\n.*?(?=\n(?:A\.D\. ?\d+\.|B\.C\. ?\d+\.|THE ANGLO-SAXON CHRONICLE)|\Z)", "\n", text, flags=re.S)
    start = text.find("\nA.D. 1.")
    body = text[start:]
    parts = re.split(r"\n(A\.D\. ?\d+\.)", body)
    for i in range(1, len(parts), 2):
        year = parts[i].replace("A.D.", "").strip(" .")
        chunk = PAREN_NOTE.sub("", parts[i + 1])
        chunk = re.sub(r"\(\(|\)\)", "", chunk)      # Ingram's (( )) marks text from other manuscripts
        chunk = _clean(chunk)
        if len(chunk) > 30:
            yield f"Year {year}", chunk


def federalist(text: str):
    """One section per paper, with the paper's own author."""
    parts = re.split(r"\n(FEDERALIST No\. \d+)\n", "\n" + text)
    for i in range(1, len(parts), 2):
        num = parts[i].split()[-1]
        chunk = parts[i + 1]
        lines = [l.strip() for l in chunk.strip().splitlines()]
        title = lines[0] if lines else ""
        m = re.search(r"\n(HAMILTON|MADISON|JAY|MADISON, with HAMILTON|HAMILTON OR MADISON)\n", chunk)
        author = m.group(1).title().replace("With", "with") if m else "Publius"
        body = chunk[m.end():] if m else chunk
        body = re.split(r"\n\s*\*\*\* ?END OF|\nPUBLIUS\.?\n", body)[0]
        body = _clean(PAREN_NOTE.sub("", body))
        if len(body) > 200:
            yield f"No. {num}: {title}", body, {"author": f"{author} (as Publius)"}


def anabasis(text: str):
    """One section per chapter (BOOK n, chapter in roman numerals); footnotes and line numbers removed."""
    body = text[text.index("\nBOOK I\n"):]
    book = None
    for block in re.split(r"\n(BOOK [IVX]+)\n", body):
        if re.fullmatch(r"BOOK [IVX]+", block.strip()):
            book = block.strip().title()
            continue
        if not book:
            continue
        block = re.sub(r"\n {3,}\((?:In the|The )[^)]*\)\s*\n", "\n", block)  # translator's book summaries
        chapters = re.split(r"\n\s*([IVX]+)\.?[ \t]*\n", "\n" + block)
        for j in range(1, len(chapters), 2):
            chunk = chapters[j + 1]
            chunk = re.sub(r"(?m)^\s*\(\d+\) .*(?:\n(?!\s*\n).*)*", "", chunk)  # footnote paragraphs
            chunk = re.sub(r"(?m)\s+\d{1,3}$", "", chunk)                   # trailing line numbers
            chunk = _clean(PAREN_NOTE.sub("", chunk))
            if len(chunk) > 200:
                yield f"{book}, Chapter {chapters[j]}", chunk


SPLITTERS.update({
    "anglo-saxon-chronicle": anglo_saxon_chronicle,
    "federalist-papers": federalist,
    "xenophon-anabasis": anabasis,
})
