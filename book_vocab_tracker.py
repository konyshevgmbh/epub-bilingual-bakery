"""
Adaptive Vocabulary Tracker for EPUB books.

Annotates words from a vocabulary list (e.g. Goethe B1) inline in an EPUB:
  - Each matching word gets a superscript number: Bahnhof¹
  - Annotated at most MAX_ANNOTATIONS_PER_WORD times per word
  - Two consecutive annotations of the same word are at least INTERVAL words apart
  - No sentence translation — vocabulary glosses only
  - Final chapter: numbered glossary + coverage summary (black = seen, red = missed)

Usage:
    python book_vocab_tracker.py input.epub output.epub --vocab vocab_b1.tsv
    python book_vocab_tracker.py input.epub output.epub --vocab vocab_b1.tsv --interval 150 --max 3
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ebooklib
from bs4 import BeautifulSoup, NavigableString, Tag
from ebooklib import epub

# ── Optional spaCy for German lemmatization ───────────────────────────────────
try:
    import spacy
    _nlp = spacy.load("de_core_news_sm")
    _SPACY_OK = True
except Exception:
    _SPACY_OK = False


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class VocabEntry:
    lemma: str          # canonical German form (as in the vocab file)
    translation: str    # Russian translation


@dataclass
class AnnotationRecord:
    number: int         # sequential annotation number (1, 2, 3 …)
    surface: str        # word form as it appeared in text ("Männern")
    lemma: str          # canonical lemma ("Mann")
    translation: str    # Russian translation


@dataclass
class TrackerState:
    vocab: Dict[str, VocabEntry]            # lemma_lower -> VocabEntry
    interval: int                           # min words between two same-word annotations
    max_per_word: int                       # max total annotations per word

    # mutable state
    global_word_pos: int = 0               # words seen so far in the whole book
    annotation_counter: int = 0            # next annotation number
    last_shown_pos: Dict[str, int] = field(default_factory=dict)   # lemma -> word pos
    shown_count: Dict[str, int] = field(default_factory=dict)      # lemma -> times shown
    annotation_log: List[AnnotationRecord] = field(default_factory=list)
    noun_articles: Dict[str, str] = field(default_factory=dict)    # lemma_lower -> "der/die/das"


# ── Vocabulary loading ────────────────────────────────────────────────────────

def load_vocab(path: str) -> Dict[str, VocabEntry]:
    """Load tab-separated vocab file: lemma TAB translation (# comments ignored)."""
    vocab: Dict[str, VocabEntry] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        lemma, translation = parts[0].strip(), parts[1].strip()
        vocab[lemma.lower()] = VocabEntry(lemma=lemma, translation=translation)
    return vocab


# ── Lemmatization ─────────────────────────────────────────────────────────────

_GENDER_TO_ARTICLE = {"Masc": "der", "Fem": "die", "Neut": "das"}


def get_lemmas_and_articles(tokens: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Return (lemmas_list, {lemma_lower: article}) using spaCy when available."""
    if _SPACY_OK:
        doc = _nlp(" ".join(tokens))
        lemmas = [t.lemma_.lower() for t in doc]
        articles: Dict[str, str] = {}
        for token, lemma_l in zip(doc, lemmas):
            if token.pos_ == "NOUN":
                gender_list = token.morph.get("Gender")
                if gender_list:
                    art = _GENDER_TO_ARTICLE.get(gender_list[0], "")
                    if art:
                        articles[lemma_l] = art
        return lemmas, articles

    # Fallback: lowercase + rough suffix stripping, no article info
    result = []
    for tok in tokens:
        l = tok.lower()
        if len(l) > 5:
            for suffix in ("innen", "ungen", "ieren", "ende", "sten",
                           "ern", "ung", "est", "end", "en", "er", "em", "es", "e"):
                if l.endswith(suffix) and len(l) - len(suffix) > 3:
                    l = l[: -len(suffix)]
                    break
        result.append(l)
    return result, {}


# ── Text tokenization ─────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüßА-Яа-яЁё]+(?:[-'][A-Za-zÄÖÜäöüßА-Яа-яЁё]+)*")


def tokenize(text: str) -> List[Tuple[str, int, int]]:
    """Return list of (word, start, end) for alphabetic tokens."""
    return [(m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


# ── Core annotation logic ─────────────────────────────────────────────────────

def should_annotate(state: TrackerState, lemma_lower: str) -> bool:
    if lemma_lower not in state.vocab:
        return False
    count = state.shown_count.get(lemma_lower, 0)
    if count >= state.max_per_word:
        return False
    last = state.last_shown_pos.get(lemma_lower, -state.interval - 1)
    return (state.global_word_pos - last) >= state.interval


def record_annotation(state: TrackerState, surface: str, lemma_lower: str) -> int:
    entry = state.vocab[lemma_lower]
    state.annotation_counter += 1
    num = state.annotation_counter
    state.annotation_log.append(AnnotationRecord(
        number=num,
        surface=surface,
        lemma=entry.lemma,
        translation=entry.translation,
    ))
    state.last_shown_pos[lemma_lower] = state.global_word_pos
    state.shown_count[lemma_lower] = state.shown_count.get(lemma_lower, 0) + 1
    return num


# ── HTML processing ───────────────────────────────────────────────────────────

def annotate_text_node(text: str, state: TrackerState, soup: BeautifulSoup) -> List:
    """
    Split a plain text node into a list of strings and <sup> Tags.
    Returns a list of BS4 nodes to replace the original NavigableString.
    """
    tokens = tokenize(text)
    if not tokens:
        return [NavigableString(text)]

    # Batch lemmatization + article detection
    words = [w for w, _, _ in tokens]
    lemmas, articles = get_lemmas_and_articles(words)

    # Persist newly detected articles into state
    for lemma_l, art in articles.items():
        if lemma_l in state.vocab:
            state.noun_articles[lemma_l] = art

    # Build output: walk through text, emit inline glosses
    result = []
    cursor = 0

    for (surface, start, end), lemma_lower in zip(tokens, lemmas):
        state.global_word_pos += 1

        if cursor < start:
            result.append(NavigableString(text[cursor:start]))

        if should_annotate(state, lemma_lower):
            # occurrence = 1 on first annotation, 2 on second, etc.
            occurrence = state.shown_count.get(lemma_lower, 0) + 1
            record_annotation(state, surface, lemma_lower)
            entry = state.vocab[lemma_lower]

            # **surface** ( *[lemma — ]translation*ⁿ )
            wrapper = soup.new_tag("span", attrs={"class": "vocab-entry"})

            bold = soup.new_tag("b")
            bold.string = surface
            wrapper.append(bold)
            wrapper.append(" ( ")

            italic = soup.new_tag("i")
            if surface.lower() != lemma_lower and surface != entry.lemma:
                italic.string = f"{entry.lemma} — {entry.translation}"
            else:
                italic.string = entry.translation
            wrapper.append(italic)

            sup = soup.new_tag("sup", attrs={"class": "vocab-ref"})
            sup.string = str(occurrence)
            wrapper.append(sup)
            wrapper.append(" )")

            result.append(wrapper)
        else:
            result.append(NavigableString(surface))

        cursor = end

    if cursor < len(text):
        result.append(NavigableString(text[cursor:]))

    return result


def process_element(element, state: TrackerState, soup: BeautifulSoup):
    """Recursively walk HTML element tree and annotate text nodes."""
    # Don't touch existing annotations or <sup>/<script>/<style>
    if isinstance(element, Tag) and element.name in ("sup", "script", "style",
                                                      "code", "pre"):
        return

    if isinstance(element, NavigableString):
        text = str(element)
        if not text.strip():
            return
        replacements = annotate_text_node(text, state, soup)
        # Only replace if we actually added annotations
        if any(isinstance(r, Tag) for r in replacements):
            parent = element.parent
            idx = list(parent.children).index(element)
            element.extract()
            for i, node in enumerate(replacements):
                parent.insert(idx + i, node)
        return

    for child in list(element.children):
        process_element(child, state, soup)


def process_document(item, state: TrackerState):
    soup = BeautifulSoup(item.get_content(), "html.parser")
    for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "li"]):
        process_element(tag, state, soup)
    item.set_content(str(soup).encode("utf-8"))


# ── CSS ───────────────────────────────────────────────────────────────────────

VOCAB_CSS = """\
/* inline gloss: **Urteil** ( *приговор*¹ ) */
span.vocab-entry b  { font-weight: bold; }
span.vocab-entry i  { font-style: italic; color: #555; font-size: 0.92em; }
sup.vocab-ref {
    font-size: 0.6em;
    color: #1a5276;
    vertical-align: super;
}
/* glossary / coverage page */
.vocab-seen   { color: #000000; }
.vocab-missed { color: #c0392b; }
.vocab-coverage { margin: 1em 0; }
.vocab-coverage-title { font-size: 1.2em; font-weight: bold; margin-bottom: 0.5em; }
table.vocab-glossary {
    border-collapse: collapse; width: 100%; font-size: 0.88em;
}
table.vocab-glossary td {
    padding: 0.18em 0.45em; border-bottom: 1px solid #eee; vertical-align: top;
}
td.vocab-art   { color: #888; width: 2.4em; font-style: italic; }
td.vocab-lemma { font-weight: bold; }
td.vocab-count { color: #1a5276; width: 3em; text-align: center; }
.alpha-letter  { font-size: 1.3em; font-weight: bold; color: #1a5276;
                 margin-top: 0.8em; border-bottom: 1px solid #1a5276; }
"""


# ── Glossary & Coverage chapter ───────────────────────────────────────────────

def _article_display(lemma_lower: str, state: TrackerState) -> str:
    """Return article string ('der '/'die '/'das ') or '' if unknown/not a noun."""
    entry = state.vocab.get(lemma_lower)
    # Only show article for nouns (canonical lemma starts with uppercase in German)
    if not entry or not entry.lemma[0].isupper():
        return ""
    art = state.noun_articles.get(lemma_lower, "")
    return art + " " if art else ""


def build_glossary_chapter(state: TrackerState) -> epub.EpubHtml:
    seen_lemmas = set(state.shown_count.keys())
    all_lemmas = sorted(state.vocab.keys())

    lines = ["<html><body>"]

    # ── Section 1: alphabetical index (lemma → count) ─────────────────────
    lines.append('<h1>Словарь встреченных слов</h1>')
    lines.append('<p style="font-size:0.85em;color:#555">Слово — перевод — сколько раз встречено (¹²³⁴).</p>')

    # Build {lemma_lower: [numbers]}
    from collections import defaultdict
    index: Dict[str, List[int]] = defaultdict(list)
    for rec in state.annotation_log:
        index[rec.lemma.lower()].append(rec.number)

    SUPERSCRIPTS = {1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹"}

    lines.append('<table class="vocab-glossary">')
    current_letter = ""
    for lemma_lower in sorted(index.keys(), key=lambda x: x.lstrip("äöü").replace("ä","a").replace("ö","o").replace("ü","u").replace("ß","ss")):
        entry = state.vocab[lemma_lower]
        art = _article_display(lemma_lower, state)
        count = len(index[lemma_lower])
        shown_sup = "".join(SUPERSCRIPTS.get(i, str(i)) for i in range(1, count + 1))
        letter = entry.lemma[0].upper()
        if letter != current_letter:
            current_letter = letter
            lines.append(f'<tr><td colspan="4" class="alpha-letter">{letter}</td></tr>')
        lines.append(
            f'<tr>'
            f'<td class="vocab-art">{art.strip()}</td>'
            f'<td class="vocab-lemma">{entry.lemma}</td>'
            f'<td>{entry.translation}</td>'
            f'<td class="vocab-count">{shown_sup}</td>'
            f'</tr>'
        )
    lines.append("</table>")

    # ── Section 3: coverage summary ───────────────────────────────────────
    lines.append('<div class="vocab-coverage">')
    lines.append('<p class="vocab-coverage-title">Покрытие словаря B1</p>')

    total = len(all_lemmas)
    seen_count = sum(1 for l in all_lemmas if l in seen_lemmas)
    pct = round(100 * seen_count / total) if total else 0
    lines.append(f'<p>Встречено в тексте: <b>{seen_count}</b> из {total} слов ({pct}%)</p>')
    lines.append('<p style="line-height:1.8em">')
    for lemma_lower in all_lemmas:
        entry = state.vocab[lemma_lower]
        art = _article_display(lemma_lower, state)
        css = "vocab-seen" if lemma_lower in seen_lemmas else "vocab-missed"
        display = f'{art}{entry.lemma}'
        lines.append(f'<span class="{css}" title="{entry.translation}">{display}</span> ')
    lines.append('</p>')
    lines.append("</div>")

    lines.append("</body></html>")

    chapter = epub.EpubHtml(
        title="Словарь",
        file_name="vocab_glossary.xhtml",
        lang="ru",
    )
    chapter.content = "\n".join(lines).encode("utf-8")
    return chapter


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_epub(args):
    print(f"Loading vocabulary from {args.vocab} …")
    vocab = load_vocab(args.vocab)
    print(f"  {len(vocab)} entries loaded")
    if not _SPACY_OK:
        print("  Warning: spaCy not available, using fallback lemmatizer")

    state = TrackerState(
        vocab=vocab,
        interval=args.interval,
        max_per_word=args.max,
    )

    print(f"Reading {args.input} …")
    book = epub.read_epub(args.input)

    # Inject CSS
    css_item = epub.EpubItem(
        uid="vocab_css",
        file_name="styles/vocab.css",
        media_type="text/css",
        content=VOCAB_CSS.encode("utf-8"),
    )
    book.add_item(css_item)

    # Add CSS reference to each document
    docs = [item for item in book.items
            if item.get_type() == ebooklib.ITEM_DOCUMENT]
    for doc in docs:
        doc.add_item(css_item)

    # Process documents
    print(f"Annotating {len(docs)} document(s) …")
    for doc in docs:
        process_document(doc, state)

    print(f"  Words processed: {state.global_word_pos}")
    print(f"  Annotations added: {state.annotation_counter}")

    # Append glossary chapter
    glossary = build_glossary_chapter(state)
    book.add_item(glossary)

    # Add to spine and toc
    spine = book.spine or []
    if spine:
        book.spine = list(spine) + [("vocab_glossary", True)]
    else:
        book.spine = [("vocab_glossary", True)]

    toc = list(book.toc) if book.toc else []
    toc.append(epub.Link("vocab_glossary.xhtml", "Словарь", "vocab_glossary"))
    book.toc = toc

    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    print(f"Writing {args.output} …")
    epub.write_epub(args.output, book)

    # ── Final report ───────────────────────────────────────────────────────
    seen = set(state.shown_count.keys())
    total = len(vocab)
    seen_count = sum(1 for l in vocab if l in seen)
    missed = [vocab[l].lemma for l in sorted(vocab) if l not in seen]

    print(f"\n{'='*60}")
    print(f"  Vocabulary coverage: {seen_count}/{total} words ({round(100*seen_count/total)}%)")
    print(f"  Total annotations:   {state.annotation_counter}")
    print(f"  Avg per word:        {state.annotation_counter/seen_count:.1f}" if seen_count else "")
    if missed:
        print(f"\n  Words NOT found in text ({len(missed)}):")
        print("  " + ", ".join(missed[:40]) + ("…" if len(missed) > 40 else ""))
    print(f"{'='*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Adaptive vocabulary tracker for EPUB language learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input",  help="Input EPUB file")
    parser.add_argument("output", help="Output EPUB file")
    parser.add_argument("--vocab", required=True,
                        help="Vocabulary TSV file: lemma TAB translation")
    parser.add_argument("--interval", type=int, default=100,
                        help="Min words between two annotations of the same word")
    parser.add_argument("--max", type=int, default=9,
                        help="Max annotations per word across whole book")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.vocab).exists():
        print(f"Error: vocab file not found: {args.vocab}", file=sys.stderr)
        sys.exit(1)

    process_epub(args)


if __name__ == "__main__":
    main()
