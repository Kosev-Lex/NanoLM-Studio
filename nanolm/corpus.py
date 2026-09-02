"""Corpus: text extraction, ONE canonical cleaning pipeline, and the
SQLite-backed document library.

Design principles
-----------------
* Documents keep their identity forever.  No merged blob.
* Raw extracted text is stored alongside cleaned text, so the corpus can
  be re-cleaned later with an improved pipeline without re-importing.
* Every transformation is a small named step; the pipeline is an ordered
  list you can read top to bottom.
* The split-word repair is line-preserving and uses a word-frequency
  (zipf) heuristic instead of bare dictionary membership, which avoids
  the classic "a bout" -> "about" false-positive family.
"""
from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import tempfile
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from .config import DB_PATH, DOC_DIR, RAW_DIR

# ---------------------------------------------------------------
# Optional dependencies (each guarded; failures are loud, not silent)
# ---------------------------------------------------------------
try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import ebooklib
    from ebooklib import epub
except ImportError:
    ebooklib = None
    epub = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import docx as python_docx
except ImportError:
    python_docx = None

try:
    from wordfreq import zipf_frequency
except ImportError:
    zipf_frequency = None

SUPPORTED_EXTS = {".txt", ".md", ".markdown", ".pdf", ".epub", ".docx", ".html", ".htm"}


class ExtractionError(RuntimeError):
    """Raised when a file cannot be converted to text (with a reason)."""


# ===============================================================
# EXTRACTION
# ===============================================================
def extract_text(path: Path) -> str:
    """Extract raw text from a supported file.  Raises ExtractionError
    with a human-readable reason instead of silently returning ''."""
    ext = path.suffix.lower()

    if ext in {".txt", ".md", ".markdown"}:
        return path.read_text(encoding="utf-8", errors="replace")

    if ext == ".pdf":
        if pypdf is None:
            raise ExtractionError("pypdf not installed (pip install pypdf)")
        try:
            reader = pypdf.PdfReader(str(path))
            pages = [(page.extract_text() or "") for page in reader.pages]
        except Exception as e:
            raise ExtractionError(f"PDF parse failed: {e}") from e
        text = "\n\n".join(pages)
        if not text.strip():
            raise ExtractionError(
                "PDF produced no text (likely a scanned/image PDF needing OCR)"
            )
        return text

    if ext == ".epub":
        if epub is None or BeautifulSoup is None:
            raise ExtractionError(
                "ebooklib + beautifulsoup4 required (pip install ebooklib beautifulsoup4)"
            )
        try:
            book = epub.read_epub(str(path))
        except Exception as e:
            raise ExtractionError(f"EPUB parse failed: {e}") from e
        chunks = []
        # NOTE: get_type() returns the ITEM_DOCUMENT int constant.
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                try:
                    soup = BeautifulSoup(item.get_content(), "html.parser")
                    chunks.append(soup.get_text(" "))
                except Exception:
                    continue
        text = "\n".join(chunks)
        if not text.strip():
            raise ExtractionError("EPUB contained no readable document items")
        return text

    if ext in {".html", ".htm"}:
        if BeautifulSoup is None:
            raise ExtractionError("beautifulsoup4 required (pip install beautifulsoup4)")
        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
        for node in soup(["script", "style", "noscript"]):
            node.decompose()
        return soup.get_text("\n")

    if ext == ".docx":
        if python_docx is None:
            raise ExtractionError("python-docx required (pip install python-docx)")
        d = python_docx.Document(str(path))
        return "\n".join(p.text for p in d.paragraphs)

    raise ExtractionError(f"Unsupported extension: {ext}")


# ===============================================================
# CLEANING PIPELINE
# ===============================================================
@dataclass
class CleanOptions:
    strip_gutenberg: bool = True
    strip_headers_footers: bool = True
    merge_split_words: bool = True


def _step_normalize_unicode(t: str) -> str:
    """NFC normalize; unify newlines; drop control chars except \\n and \\t."""
    t = unicodedata.normalize("NFC", t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(
        ch for ch in t
        if ch in ("\n", "\t") or unicodedata.category(ch)[0] != "C"
    )


_GUT_START = re.compile(r"\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*{3}", re.IGNORECASE | re.DOTALL)
_GUT_END = re.compile(r"\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*", re.IGNORECASE | re.DOTALL)


def _step_strip_gutenberg(t: str) -> str:
    """Remove Project Gutenberg boilerplate when markers are present.
    Only acts if a marker is actually found -- never guesses."""
    m = _GUT_START.search(t)
    if m:
        t = t[m.end():]
    m = _GUT_END.search(t)
    if m:
        t = t[: m.start()]
    return t


def _step_fix_hyphenation(t: str) -> str:
    """Re-join words hyphen-broken across lines: 'exam-\\nple of' ->
    'example\\nof' (whole continuation word joins; line count preserved)."""
    return re.sub(r"(\w+)-[ \t]*\n[ \t]*(\w+)", r"\1\2\n", t)


def _step_strip_headers_footers(t: str) -> str:
    """Drop page-number lines, rule lines, and short lines repeated many
    times (typical running headers).  Frequency-based, so a chapter title
    that appears once survives while 'CHAPTER TITLE   page' repeated on
    every page does not."""
    lines = t.split("\n")
    stripped = [ln.strip() for ln in lines]
    counts = Counter(s for s in stripped if s and len(s) <= 60)
    out = []
    for ln, s in zip(lines, stripped):
        if not s:
            out.append(ln)
            continue
        if re.fullmatch(r"\d{1,4}", s):                # bare page number
            continue
        if re.fullmatch(r"[-=~_.\u2022*]{3,}", s):     # rule / separator line
            continue
        if len(s) <= 60 and counts[s] >= 5:            # repeated running header
            continue
        out.append(ln)
    return "\n".join(out)


# ---- split-word repair (line-preserving, zipf heuristic) -------------
_PUNCT_EDGE = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)


def _split_edges(token: str):
    m = _PUNCT_EDGE.match(token)
    return m.group(1), m.group(2), m.group(3)


def _zipf(word: str) -> float:
    return zipf_frequency(word, "en") if word else 0.0


def _merge_line(tokens: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        # Case 1: a run of single letters: "m a r k e t" -> "market".
        # Require >= 4 letters so initialisms like "U S A" survive.
        j = i
        while j < n:
            _, core, post = _split_edges(tokens[j])
            if len(core) == 1 and core.isalpha() and (post == "" or j == n - 1 or post in {".", ",", ";", ":", "!", "?"}):
                j += 1
                if post:        # punctuation ends the run
                    break
            else:
                break
        run_len = j - i
        if run_len >= 4:
            pres, cores, posts = zip(*(_split_edges(tk) for tk in tokens[i:j]))
            merged = "".join(cores)
            if _zipf(merged.lower()) >= 3.0:
                out.append(pres[0] + merged + posts[-1])
                i = j
                continue

        # Case 2: two-token split: "m illion" -> "million".
        if i + 1 < n:
            lpre, lcore, lpost = _split_edges(tokens[i])
            rpre, rcore, rpost = _split_edges(tokens[i + 1])
            if (lcore.isalpha() and rcore.isalpha() and not lpost and not rpre
                    and 1 <= len(lcore) <= 8 and 1 <= len(rcore) <= 12):
                merged = lcore + rcore
                f_m = _zipf(merged.lower())
                f_l = _zipf(lcore.lower())
                f_r = _zipf(rcore.lower())
                # Merge when (a) one fragment is gibberish and the merge is a
                # real word, or (b) the merge is much more frequent than both
                # parts.  Rule (a) catches "m illion" (illion: 0); the high
                # frequency of real words like "a"/"in" blocks "a bout".
                gibberish_side = min(f_l, f_r) < 1.5
                strong_gain = f_m - max(f_l, f_r) >= 1.5
                if f_m >= 3.0 and (gibberish_side or strong_gain):
                    out.append(lpre + merged + rpost)
                    i += 2
                    continue
        out.append(tokens[i])
        i += 1
    return out


def _step_merge_split_words(t: str) -> str:
    """Repair OCR/PDF intra-word spaces.  Operates per line, so document
    structure (newlines, paragraphs) is fully preserved."""
    if zipf_frequency is None:
        return t  # optional dependency missing -> step is skipped, not faked
    fixed_lines = []
    for line in t.split("\n"):
        tokens = line.split()
        if len(tokens) < 2:
            fixed_lines.append(line)
            continue
        fixed_lines.append(" ".join(_merge_line(tokens)))
    return "\n".join(fixed_lines)


def _step_normalize_whitespace(t: str) -> str:
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" ?\n ?", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def clean_text(raw: str, opts: Optional[CleanOptions] = None) -> str:
    """THE cleaning pipeline.  Order matters and is intentional."""
    opts = opts or CleanOptions()
    steps: list[Callable[[str], str]] = [_step_normalize_unicode]
    if opts.strip_gutenberg:
        steps.append(_step_strip_gutenberg)
    steps.append(_step_fix_hyphenation)
    if opts.strip_headers_footers:
        steps.append(_step_strip_headers_footers)
    if opts.merge_split_words:
        steps.append(_step_merge_split_words)
    steps.append(_step_normalize_whitespace)

    t = raw
    for step in steps:
        t = step(t)
    return t


# ===============================================================
# DOCUMENT LIBRARY (SQLite)
# ===============================================================
_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    source_path TEXT,
    sha256 TEXT UNIQUE NOT NULL,
    added_at REAL NOT NULL,
    chars INTEGER NOT NULL,
    tokens INTEGER,
    active INTEGER NOT NULL DEFAULT 1,
    tags TEXT NOT NULL DEFAULT ''
);
"""


@dataclass
class DocumentRow:
    id: int
    title: str
    source_path: str
    sha256: str
    added_at: float
    chars: int
    tokens: Optional[int]
    active: bool
    tags: str


class CorpusLibrary:
    """Document-centric corpus store.  Each method opens its own SQLite
    connection, which makes the class safe to call from worker threads."""

    def __init__(self, db_path: Path = DB_PATH, *, raw_dir: Optional[Path] = None,
                 doc_dir: Optional[Path] = None):
        self.db_path = Path(db_path)
        is_default = self.db_path.resolve() == DB_PATH.resolve()
        base = self.db_path.parent
        self.raw_dir = Path(raw_dir) if raw_dir is not None else (RAW_DIR if is_default else base / "raw")
        self.doc_dir = Path(doc_dir) if doc_dir is not None else (DOC_DIR if is_default else base / "documents")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.doc_dir.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=15.0)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA busy_timeout=15000")
        return c

    # ---------- paths ----------
    def _doc_path(self, doc_id: int) -> Path:
        return self.doc_dir / f"{doc_id:05d}.txt"

    def _raw_path(self, doc_id: int) -> Path:
        return self.raw_dir / f"{doc_id:05d}.txt"

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        """Durably replace a UTF-8 text file without exposing partial data."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise

    # ---------- CRUD ----------
    def add_document(self, src: Path, opts: Optional[CleanOptions] = None) -> tuple[Optional[DocumentRow], str]:
        """Extract -> clean -> dedupe -> store.  Returns (row|None, message)."""
        raw = extract_text(src)                       # may raise ExtractionError
        cleaned = clean_text(raw, opts)
        if not cleaned:
            return None, f"skipped (empty after cleaning): {src.name}"
        digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()

        raw_path: Optional[Path] = None
        doc_path: Optional[Path] = None
        try:
            with self._conn() as c:
                dup = c.execute("SELECT id, title FROM documents WHERE sha256=?", (digest,)).fetchone()
                if dup:
                    return None, f"skipped (duplicate of #{dup['id']} '{dup['title']}'): {src.name}"
                cur = c.execute(
                    "INSERT INTO documents (title, source_path, sha256, added_at, chars) VALUES (?,?,?,?,?)",
                    (src.stem, str(src), digest, time.time(), len(cleaned)),
                )
                doc_id = int(cur.lastrowid)
                raw_path, doc_path = self._raw_path(doc_id), self._doc_path(doc_id)
                self._atomic_write(raw_path, raw)
                self._atomic_write(doc_path, cleaned)
        except Exception:
            for path in (raw_path, doc_path):
                if path is not None:
                    path.unlink(missing_ok=True)
            raise
        return self.get(doc_id), f"added #{doc_id}: {src.name} ({len(cleaned):,} chars)"

    def reclean(self, doc_id: int, opts: Optional[CleanOptions] = None) -> str:
        """Re-run the cleaning pipeline on the stored RAW text."""
        raw_p = self._raw_path(doc_id)
        if not raw_p.exists():
            return f"#{doc_id}: no raw text stored; cannot re-clean"
        cleaned = clean_text(raw_p.read_text(encoding="utf-8"), opts)
        if not cleaned:
            return f"#{doc_id}: cleaning produced no text; existing document kept"
        digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        doc_path = self._doc_path(doc_id)
        old_cleaned = doc_path.read_text(encoding="utf-8") if doc_path.exists() else None
        try:
            with self._conn() as c:
                dup = c.execute(
                    "SELECT id, title FROM documents WHERE sha256=? AND id<>?", (digest, doc_id)
                ).fetchone()
                if dup:
                    return f"#{doc_id}: re-clean would duplicate #{dup['id']} '{dup['title']}'; unchanged"
                c.execute(
                    "UPDATE documents SET chars=?, sha256=?, tokens=NULL WHERE id=?",
                    (len(cleaned), digest, doc_id),
                )
                self._atomic_write(doc_path, cleaned)
        except Exception:
            if old_cleaned is not None:
                self._atomic_write(doc_path, old_cleaned)
            else:
                doc_path.unlink(missing_ok=True)
            raise
        return f"#{doc_id}: re-cleaned ({len(cleaned):,} chars)"

    def remove(self, doc_id: int):
        with self._conn() as c:
            c.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        for p in (self._doc_path(doc_id), self._raw_path(doc_id)):
            if p.exists():
                p.unlink()

    def get(self, doc_id: int) -> Optional[DocumentRow]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        return self._row(r) if r else None

    def list_documents(self) -> list[DocumentRow]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM documents ORDER BY id").fetchall()
        return [self._row(r) for r in rows]

    def set_active(self, doc_id: int, active: bool):
        with self._conn() as c:
            c.execute("UPDATE documents SET active=? WHERE id=?", (1 if active else 0, doc_id))

    def set_tags(self, doc_id: int, tags: str):
        with self._conn() as c:
            c.execute("UPDATE documents SET tags=? WHERE id=?", (tags, doc_id))

    def set_token_count(self, doc_id: int, n: int):
        with self._conn() as c:
            c.execute("UPDATE documents SET tokens=? WHERE id=?", (n, doc_id))

    def clear_token_counts(self) -> None:
        """Invalidate displayed counts after the tokenizer changes."""
        with self._conn() as c:
            c.execute("UPDATE documents SET tokens=NULL")

    def get_text(self, doc_id: int) -> str:
        return self._doc_path(doc_id).read_text(encoding="utf-8")

    def get_raw_text(self, doc_id: int) -> str:
        p = self._raw_path(doc_id)
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def active_documents(self) -> list[DocumentRow]:
        return [d for d in self.list_documents() if d.active]

    def stats(self) -> dict:
        docs = self.list_documents()
        active = [d for d in docs if d.active]
        token_counts_complete = bool(active) and all(d.tokens is not None for d in active)
        return {
            "documents": len(docs),
            "active": len(active),
            "chars": sum(d.chars for d in active),
            "tokens": sum(int(d.tokens) for d in active) if token_counts_complete else None,
        }

    def corpus_fingerprint(self) -> str:
        """Identify the exact ordered active corpus used by token caches."""
        h = hashlib.sha256()
        for doc in self.active_documents():
            h.update(f"{doc.id}:{doc.sha256}\n".encode("utf-8"))
        return h.hexdigest()[:16]

    @staticmethod
    def _row(r: sqlite3.Row) -> DocumentRow:
        return DocumentRow(
            id=r["id"], title=r["title"], source_path=r["source_path"] or "",
            sha256=r["sha256"], added_at=r["added_at"], chars=r["chars"],
            tokens=r["tokens"], active=bool(r["active"]), tags=r["tags"],
        )


def iter_supported_files(path: Path) -> Iterable[Path]:
    if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
        yield path
    elif path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                yield p
