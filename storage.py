"""
storage.py - SQLite storage for reading progress, highlights, and bookmarks.
"""

import os
import sqlite3
from typing import List, Dict, Optional, Any
from datetime import datetime

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reader_data.db")


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Books metadata & state
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS books (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath TEXT UNIQUE NOT NULL,
                title TEXT,
                total_chapters INTEGER DEFAULT 0,
                last_opened TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Reading progress (last position)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reading_progress (
                book_path TEXT PRIMARY KEY,
                chapter_index INTEGER DEFAULT 0,
                sentence_index INTEGER DEFAULT 0,
                scroll_ratio REAL DEFAULT 0.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (book_path) REFERENCES books (filepath) ON DELETE CASCADE
            )
        """)

        # Highlights and annotations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS highlights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_path TEXT NOT NULL,
                chapter_index INTEGER NOT NULL,
                sentence_index INTEGER DEFAULT 0,
                selected_text TEXT NOT NULL,
                color TEXT DEFAULT '#ffeb3b',
                note TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Bookmarks
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bookmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_path TEXT NOT NULL,
                chapter_index INTEGER NOT NULL,
                sentence_index INTEGER DEFAULT 0,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()


# --- Book Operations ---

def record_book_opened(filepath: str, title: str, total_chapters: int = 0):
    norm_path = os.path.abspath(filepath)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO books (filepath, title, total_chapters, last_opened)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(filepath) DO UPDATE SET
                title = excluded.title,
                total_chapters = excluded.total_chapters,
                last_opened = CURRENT_TIMESTAMP
        """, (norm_path, title, total_chapters))
        conn.commit()


def get_recent_books(limit: int = 10) -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT b.*, p.chapter_index, p.sentence_index, p.scroll_ratio
            FROM books b
            LEFT JOIN reading_progress p ON b.filepath = p.book_path
            ORDER BY b.last_opened DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]


# --- Reading Progress ---

def save_reading_progress(filepath: str, chapter_index: int, sentence_index: int = 0, scroll_ratio: float = 0.0):
    norm_path = os.path.abspath(filepath)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO reading_progress (book_path, chapter_index, sentence_index, scroll_ratio, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(book_path) DO UPDATE SET
                chapter_index = excluded.chapter_index,
                sentence_index = excluded.sentence_index,
                scroll_ratio = excluded.scroll_ratio,
                updated_at = CURRENT_TIMESTAMP
        """, (norm_path, chapter_index, sentence_index, scroll_ratio))
        conn.commit()


def get_reading_progress(filepath: str) -> Optional[Dict[str, Any]]:
    norm_path = os.path.abspath(filepath)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT chapter_index, sentence_index, scroll_ratio, updated_at
            FROM reading_progress
            WHERE book_path = ?
        """, (norm_path,))
        row = cursor.fetchone()
        return dict(row) if row else None


# --- Highlights ---

def add_highlight(filepath: str, chapter_index: int, sentence_index: int, selected_text: str, color: str = "#ffeb3b", note: str = "") -> int:
    norm_path = os.path.abspath(filepath)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO highlights (book_path, chapter_index, sentence_index, selected_text, color, note)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (norm_path, chapter_index, sentence_index, selected_text.strip(), color, note.strip()))
        conn.commit()
        return cursor.lastrowid


def update_highlight_note(highlight_id: int, note: str):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE highlights SET note = ? WHERE id = ?
        """, (note.strip(), highlight_id))
        conn.commit()


def delete_highlight(highlight_id: int):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM highlights WHERE id = ?", (highlight_id,))
        conn.commit()


def get_highlights_for_book(filepath: str, chapter_index: Optional[int] = None) -> List[Dict[str, Any]]:
    norm_path = os.path.abspath(filepath)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if chapter_index is not None:
            cursor.execute("""
                SELECT * FROM highlights
                WHERE book_path = ? AND chapter_index = ?
                ORDER BY sentence_index ASC, created_at ASC
            """, (norm_path, chapter_index))
        else:
            cursor.execute("""
                SELECT * FROM highlights
                WHERE book_path = ?
                ORDER BY chapter_index ASC, sentence_index ASC, created_at ASC
            """, (norm_path,))
        return [dict(row) for row in cursor.fetchall()]


# --- Bookmarks ---

def add_bookmark(filepath: str, chapter_index: int, sentence_index: int, title: str) -> int:
    norm_path = os.path.abspath(filepath)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO bookmarks (book_path, chapter_index, sentence_index, title)
            VALUES (?, ?, ?, ?)
        """, (norm_path, chapter_index, sentence_index, title))
        conn.commit()
        return cursor.lastrowid


def delete_bookmark(bookmark_id: int):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM bookmarks WHERE id = ?", (bookmark_id,))
        conn.commit()


def get_bookmarks_for_book(filepath: str) -> List[Dict[str, Any]]:
    norm_path = os.path.abspath(filepath)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM bookmarks
            WHERE book_path = ?
            ORDER BY chapter_index ASC, sentence_index ASC
        """, (norm_path,))
        return [dict(row) for row in cursor.fetchall()]


# Initialize DB on load
init_db()

