"""
book_parser.py - Robust PDF and EPUB parser extracting chapters, paragraphs, and sentences.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
try:
    import pymupdf as fitz
except ImportError:
    import fitz
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import nltk


# Ensure punkt tokenizer is loaded
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
        nltk.download('punkt', quiet=True)
    except Exception:
        pass


@dataclass
class SentenceItem:
    sentence_index: int
    text: str
    paragraph_index: int


@dataclass
class Chapter:
    index: int
    title: str
    paragraphs: List[List[SentenceItem]] = field(default_factory=list)
    sentences: List[SentenceItem] = field(default_factory=list)

    @property
    def plain_text(self) -> str:
        return " ".join(s.text for s in self.sentences)


@dataclass
class ParsedBook:
    filepath: str
    title: str
    format: str  # 'pdf' or 'epub'
    chapters: List[Chapter] = field(default_factory=list)

    @property
    def total_sentences(self) -> int:
        return sum(len(c.sentences) for c in self.chapters)


def clean_text_for_reading(text: str) -> str:
    """Cleans up raw text, fixes hyphenations across lines, normalizes whitespace."""
    if not text:
        return ""
    # Fix hyphenated words at line breaks (e.g. "con- \n tinue" -> "continue")
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    # Replace line breaks inside sentences with single space
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Remove control characters except newlines/tabs
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    """Splits text into clean sentences using NLTK or regex fallback."""
    text = clean_text_for_reading(text)
    if not text:
        return []
    try:
        raw_sents = nltk.sent_tokenize(text)
    except Exception:
        # Fallback regex if NLTK fails
        raw_sents = re.split(r'(?<=[.!?])\s+', text)

    clean_sents = []
    for s in raw_sents:
        s = s.strip()
        if len(s) > 1 and any(c.isalnum() for c in s):
            clean_sents.append(s)
    return clean_sents


def parse_pdf(filepath: str) -> ParsedBook:
    """Parses PDF document using PyMuPDF (fitz)."""
    doc = fitz.open(filepath)
    title = os.path.basename(filepath)
    if doc.metadata and doc.metadata.get("title"):
        t = doc.metadata.get("title").strip()
        if t:
            title = t

    toc = doc.get_toc()  # [[lvl, title, page_num], ...]
    chapters: List[Chapter] = []

    # If TOC exists with multiple entries, group by TOC chapters
    if toc and len(toc) > 1:
        # Filter level 1 or 2 items
        bookmarks = [item for item in toc if item[0] <= 2]
        for i, bm in enumerate(bookmarks):
            start_page = max(0, bm[2] - 1)
            end_page = doc.page_count - 1
            if i + 1 < len(bookmarks):
                end_page = max(start_page, bookmarks[i + 1][2] - 2)

            ch_title = bm[1].strip() or f"Chapter {i + 1}"
            chapter = Chapter(index=i, title=ch_title)
            
            sent_counter = 0
            para_counter = 0

            for p_no in range(start_page, min(end_page + 1, doc.page_count)):
                page = doc.load_page(p_no)
                page_text = page.get_text("text")
                paras = [p.strip() for p in page_text.split("\n\n") if p.strip()]

                for raw_p in paras:
                    sents = split_into_sentences(raw_p)
                    if not sents:
                        continue
                    para_items = []
                    for s_text in sents:
                        item = SentenceItem(
                            sentence_index=sent_counter,
                            text=s_text,
                            paragraph_index=para_counter
                        )
                        para_items.append(item)
                        chapter.sentences.append(item)
                        sent_counter += 1
                    chapter.paragraphs.append(para_items)
                    para_counter += 1

            if chapter.sentences:
                chapters.append(chapter)

    # If no TOC or failed to parse chapters, divide by pages or groups of pages
    if not chapters:
        # Group pages: 1 page = 1 chapter for easy navigation and reading
        for p_no in range(doc.page_count):
            page = doc.load_page(p_no)
            page_text = page.get_text("text")
            paras = [p.strip() for p in page_text.split("\n\n") if p.strip()]
            
            chapter = Chapter(index=p_no, title=f"Page {p_no + 1}")
            sent_counter = 0
            para_counter = 0

            for raw_p in paras:
                sents = split_into_sentences(raw_p)
                if not sents:
                    continue
                para_items = []
                for s_text in sents:
                    item = SentenceItem(
                        sentence_index=sent_counter,
                        text=s_text,
                        paragraph_index=para_counter
                    )
                    para_items.append(item)
                    chapter.sentences.append(item)
                    sent_counter += 1
                chapter.paragraphs.append(para_items)
                para_counter += 1

            if chapter.sentences:
                chapters.append(chapter)

    # Reindex chapters sequentially
    for idx, ch in enumerate(chapters):
        ch.index = idx

    return ParsedBook(
        filepath=os.path.abspath(filepath),
        title=title,
        format="pdf",
        chapters=chapters
    )


def parse_epub(filepath: str) -> ParsedBook:
    """Parses EPUB book using ebooklib and BeautifulSoup."""
    book = epub.read_epub(filepath)
    title = os.path.basename(filepath)
    meta_title = book.get_metadata('DC', 'title')
    if meta_title:
        title = meta_title[0][0]

    chapters: List[Chapter] = []
    ch_idx = 0

    # Read items in spine order
    for item_id, linear in book.spine:
        item = book.get_item_with_id(item_id)
        if not item or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue

        soup = BeautifulSoup(item.get_content(), 'html.parser')

        # Try to find chapter heading
        heading = soup.find(['h1', 'h2', 'h3'])
        ch_title = heading.get_text().strip() if heading else f"Chapter {ch_idx + 1}"
        if not ch_title:
            ch_title = f"Chapter {ch_idx + 1}"

        chapter = Chapter(index=ch_idx, title=ch_title)
        sent_counter = 0
        para_counter = 0

        # Extract paragraphs / blocks
        blocks = soup.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'blockquote', 'li'])
        if not blocks:
            # Fallback to whole text
            text = soup.get_text()
            paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        else:
            paras = [b.get_text().strip() for b in blocks if b.get_text().strip()]

        for raw_p in paras:
            sents = split_into_sentences(raw_p)
            if not sents:
                continue
            para_items = []
            for s_text in sents:
                item = SentenceItem(
                    sentence_index=sent_counter,
                    text=s_text,
                    paragraph_index=para_counter
                )
                para_items.append(item)
                chapter.sentences.append(item)
                sent_counter += 1
            chapter.paragraphs.append(para_items)
            para_counter += 1

        if chapter.sentences:
            chapters.append(chapter)
            ch_idx += 1

    return ParsedBook(
        filepath=os.path.abspath(filepath),
        title=title,
        format="epub",
        chapters=chapters
    )


def load_book(filepath: str) -> ParsedBook:
    """Loads either PDF or EPUB based on extension."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return parse_pdf(filepath)
    elif ext in [".epub", ".mobi"]:
        return parse_epub(filepath)
    else:
        # Fallback: try PDF then EPUB
        try:
            return parse_pdf(filepath)
        except Exception:
            return parse_epub(filepath)
