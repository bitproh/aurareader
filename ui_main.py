"""
ui_main.py - Elegant PyQt6 Desktop Reader with Kokoro-ONNX Neural TTS,
Bookmarks, Highlights, Auto-Resume, and Offline Summarization.
"""

import os
import sys
from typing import Optional, List, Dict, Any, Tuple

from PyQt6.QtCore import Qt, QUrl, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import (
    QAction, QFont, QTextCursor, QTextCharFormat, QColor,
    QKeySequence, QIcon, QPalette
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTextBrowser, QListWidget, QListWidgetItem, QTabWidget,
    QPushButton, QComboBox, QSlider, QLabel, QFileDialog, QMessageBox,
    QInputDialog, QMenu, QToolBar, QStatusBar, QFrame, QScrollArea,
    QSizePolicy
)

import storage
import book_parser
import summarizer
from tts_engine import TTSEngine, AudioPlayer


# --- Themes Definition ---
THEMES = {
    "dark": {
        "name": "Dark Charcoal",
        "bg_main": "#18181b",
        "bg_sidebar": "#27272a",
        "bg_reader": "#1f1f23",
        "text_main": "#f4f4f5",
        "text_secondary": "#a1a1aa",
        "active_sentence_bg": "#451a03",
        "active_sentence_border": "#f59e0b",
        "active_sentence_text": "#fef3c7",
        "accent": "#3b82f6",
        "border": "#3f3f46",
    },
    "sepia": {
        "name": "Warm Sepia",
        "bg_main": "#fbf0d9",
        "bg_sidebar": "#f4e4c1",
        "bg_reader": "#fbf0d9",
        "text_main": "#433422",
        "text_secondary": "#7f6850",
        "active_sentence_bg": "#e6c794",
        "active_sentence_border": "#b45309",
        "active_sentence_text": "#292015",
        "accent": "#9a3412",
        "border": "#e6d5b8",
    },
    "light": {
        "name": "Modern Light",
        "bg_main": "#f8fafc",
        "bg_sidebar": "#f1f5f9",
        "bg_reader": "#ffffff",
        "text_main": "#0f172a",
        "text_secondary": "#64748b",
        "active_sentence_bg": "#fef3c7",
        "active_sentence_border": "#d97706",
        "active_sentence_text": "#78350f",
        "accent": "#2563eb",
        "border": "#e2e8f0",
    }
}


class TTSBridge(QObject):
    """Thread-safe signal dispatcher for TTS audio callbacks."""
    sentence_started = pyqtSignal(int, str)
    state_changed = pyqtSignal(bool, bool)
    playback_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    buffering_status = pyqtSignal(str)


class ReaderWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AuraReader — Natural Neural Audiobook & E-Reader")
        self.resize(1280, 840)

        # State
        self.current_theme = "dark"
        self.font_size = 17
        self.book: Optional[book_parser.ParsedBook] = None
        self.current_chapter_idx: int = 0
        self.current_sentence_idx: int = 0
        self.sentence_char_ranges: List[Tuple[int, int]] = []

        # TTS Engine & Player
        self.tts_engine = TTSEngine()
        self.audio_player = AudioPlayer(self.tts_engine)
        self.tts_bridge = TTSBridge()

        # Connect audio player callbacks to Qt bridge signals
        self.audio_player.on_sentence_started = lambda idx, txt: self.tts_bridge.sentence_started.emit(idx, txt)
        self.audio_player.on_state_changed = lambda p, ps: self.tts_bridge.state_changed.emit(p, ps)
        self.audio_player.on_finished = lambda: self.tts_bridge.playback_finished.emit()
        self.audio_player.on_error = lambda err: self.tts_bridge.error_occurred.emit(err)
        self.audio_player.on_buffering_status = lambda msg: self.tts_bridge.buffering_status.emit(msg)

        self.tts_bridge.sentence_started.connect(self._on_tts_sentence_started)
        self.tts_bridge.state_changed.connect(self._on_tts_state_changed)
        self.tts_bridge.playback_finished.connect(self._on_tts_finished)
        self.tts_bridge.error_occurred.connect(self._on_tts_error)
        self.tts_bridge.buffering_status.connect(self._on_buffering_status)

        # Autosave timer
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(3000)
        self.autosave_timer.timeout.connect(self._autosave_progress)
        self.autosave_timer.start()

        self._init_ui()
        self._apply_theme()

        # Check for last opened book
        QTimer.singleShot(400, self._check_last_opened_book)

    def _init_ui(self):
        # Top Toolbar
        self._create_toolbar()

        # Main Central Widget with horizontal splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.main_splitter)

        # 1. Left Sidebar: TOC, Highlights, Bookmarks
        self._create_sidebar()

        # 2. Center: Reader View
        self._create_reader_view()

        # 3. Right Sidebar: Summary Drawer
        self._create_summary_panel()

        self.main_splitter.addWidget(self.sidebar_tabs)
        self.main_splitter.addWidget(self.reader_container)
        self.main_splitter.addWidget(self.summary_panel)

        # Initial splitter sizes: 260px left, 680px center, 320px right
        self.main_splitter.setSizes([260, 680, 320])

        # Bottom Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.status_reading_info = QLabel("No book loaded. Click 'Open Book' to begin.")
        self.status_sentence_info = QLabel("")
        self.status_bar.addWidget(self.status_reading_info, 1)
        self.status_bar.addPermanentWidget(self.status_sentence_info)

    def _create_toolbar(self):
        self.toolbar = QToolBar("Controls")
        self.toolbar.setMovable(False)
        self.toolbar.setFixedHeight(54)
        self.addToolBar(self.toolbar)

        # Open File Button
        self.btn_open = QPushButton("📂 Open Book")
        self.btn_open.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_open.clicked.connect(self.open_file_dialog)
        self.toolbar.addWidget(self.btn_open)

        self.toolbar.addSeparator()

        # Audio Playback Controls
        self.btn_prev = QPushButton("⏮ Prev")
        self.btn_prev.setToolTip("Previous sentence")
        self.btn_prev.clicked.connect(self.previous_sentence)
        self.toolbar.addWidget(self.btn_prev)

        self.btn_play_pause = QPushButton("▶ Play")
        self.btn_play_pause.setToolTip("Play / Pause narration (Space)")
        self.btn_play_pause.setStyleSheet("font-weight: bold; min-width: 80px;")
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        self.toolbar.addWidget(self.btn_play_pause)

        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.setToolTip("Stop narration (Esc)")
        self.btn_stop.clicked.connect(self.stop_playback)
        self.toolbar.addWidget(self.btn_stop)

        self.btn_next = QPushButton("⏭ Next")
        self.btn_next.setToolTip("Next sentence")
        self.btn_next.clicked.connect(self.next_sentence)
        self.toolbar.addWidget(self.btn_next)

        self.toolbar.addSeparator()

        # Engine Selector
        lbl_engine = QLabel(" Engine: ")
        self.toolbar.addWidget(lbl_engine)
        self.combo_engine = QComboBox()
        self.combo_engine.addItem("🎙️ Kokoro (Offline)", "kokoro")
        self.combo_engine.addItem("⚡ Edge-TTS (Fast Neural)", "edge")
        self.combo_engine.currentIndexChanged.connect(self._on_engine_changed)
        self.toolbar.addWidget(self.combo_engine)

        # Voice Selector
        lbl_voice = QLabel("  Voice: ")
        self.toolbar.addWidget(lbl_voice)
        self.combo_voices = QComboBox()
        self.combo_voices.setMinimumWidth(180)
        self._populate_voices()
        self.combo_voices.currentIndexChanged.connect(self._on_voice_changed)
        self.toolbar.addWidget(self.combo_voices)

        # Sentence Gap (Pause between sentences)
        lbl_gap = QLabel("  Gap: ")
        self.toolbar.addWidget(lbl_gap)
        self.combo_gap = QComboBox()
        self.combo_gap.addItem("0.04s (Tight)", 0.04)
        self.combo_gap.addItem("0.08s (Natural)", 0.08)
        self.combo_gap.addItem("0.15s (Standard)", 0.15)
        self.combo_gap.addItem("0.25s (Relaxed)", 0.25)
        self.combo_gap.setCurrentIndex(1)  # 0.08s default
        self.combo_gap.currentIndexChanged.connect(self._on_gap_changed)
        self.toolbar.addWidget(self.combo_gap)

        # Speed Selector
        lbl_speed = QLabel("  Speed: ")
        self.toolbar.addWidget(lbl_speed)
        self.combo_speed = QComboBox()
        for sp in [0.75, 1.0, 1.15, 1.25, 1.5, 1.75, 2.0]:
            self.combo_speed.addItem(f"{sp:.2f}x", sp)
        self.combo_speed.setCurrentIndex(1)  # 1.0x
        self.combo_speed.currentIndexChanged.connect(self._on_speed_changed)
        self.toolbar.addWidget(self.combo_speed)

        self.toolbar.addSeparator()

        # Summarize Chapter Button
        self.btn_summarize = QPushButton("✨ Summarize Chapter")
        self.btn_summarize.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_summarize.clicked.connect(self.summarize_current_chapter)
        self.toolbar.addWidget(self.btn_summarize)

        self.toolbar.addSeparator()

        # Font sizing
        btn_font_minus = QPushButton("A-")
        btn_font_minus.setFixedWidth(36)
        btn_font_minus.clicked.connect(self.decrease_font_size)
        self.toolbar.addWidget(btn_font_minus)

        btn_font_plus = QPushButton("A+")
        btn_font_plus.setFixedWidth(36)
        btn_font_plus.clicked.connect(self.increase_font_size)
        self.toolbar.addWidget(btn_font_plus)

        # Theme Selector
        lbl_theme = QLabel("  Theme: ")
        self.toolbar.addWidget(lbl_theme)
        self.combo_theme = QComboBox()
        self.combo_theme.addItem("Dark", "dark")
        self.combo_theme.addItem("Sepia", "sepia")
        self.combo_theme.addItem("Light", "light")
        self.combo_theme.currentIndexChanged.connect(self._on_theme_changed)
        self.toolbar.addWidget(self.combo_theme)

    def _populate_voices(self):
        eng = self.combo_engine.currentData() if hasattr(self, 'combo_engine') else "kokoro"
        voices = self.tts_engine.get_available_voices(eng)
        self.combo_voices.blockSignals(True)
        self.combo_voices.clear()
        preferred_default = "af_sarah" if eng == "kokoro" else "en-US-ChristopherNeural"
        default_idx = 0
        for i, (v_id, label) in enumerate(voices):
            self.combo_voices.addItem(label, v_id)
            if v_id == preferred_default:
                default_idx = i
        self.combo_voices.setCurrentIndex(default_idx)
        self.combo_voices.blockSignals(False)
        self._on_voice_changed()

    def _on_engine_changed(self):
        eng = self.combo_engine.currentData()
        self.audio_player.set_engine(eng)
        self._populate_voices()

    def _on_gap_changed(self):
        gap = self.combo_gap.currentData()
        if gap is not None:
            self.audio_player.set_sentence_gap(gap)

    def _on_buffering_status(self, msg: str):
        self.status_reading_info.setText(msg)

    def _create_sidebar(self):
        self.sidebar_tabs = QTabWidget()
        self.sidebar_tabs.setMinimumWidth(200)

        # Tab 1: TOC Chapters
        self.list_toc = QListWidget()
        self.list_toc.itemClicked.connect(self._on_toc_item_clicked)
        self.sidebar_tabs.addTab(self.list_toc, "📑 Chapters")

        # Tab 2: Highlights
        self.list_highlights = QListWidget()
        self.list_highlights.itemClicked.connect(self._on_highlight_item_clicked)
        self.sidebar_tabs.addTab(self.list_highlights, "⭐ Highlights")

        # Tab 3: Bookmarks
        self.list_bookmarks = QListWidget()
        self.list_bookmarks.itemClicked.connect(self._on_bookmark_item_clicked)
        self.sidebar_tabs.addTab(self.list_bookmarks, "🔖 Bookmarks")

    def _create_reader_view(self):
        self.reader_container = QWidget()
        reader_layout = QVBoxLayout(self.reader_container)
        reader_layout.setContentsMargins(12, 12, 12, 12)

        # Header with chapter title and bookmark button
        header_bar = QHBoxLayout()
        self.lbl_chapter_title = QLabel("Welcome to AuraReader")
        self.lbl_chapter_title.setStyleSheet("font-size: 18px; font-weight: bold;")
        header_bar.addWidget(self.lbl_chapter_title, 1)

        self.btn_add_bookmark = QPushButton("🔖 Bookmark Spot")
        self.btn_add_bookmark.clicked.connect(self.add_bookmark_at_current_spot)
        header_bar.addWidget(self.btn_add_bookmark)

        reader_layout.addLayout(header_bar)

        # Interactive Text Reader
        self.reader_view = QTextBrowser()
        self.reader_view.setReadOnly(True)
        self.reader_view.setOpenLinks(False)
        self.reader_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.reader_view.customContextMenuRequested.connect(self._show_reader_context_menu)
        reader_layout.addWidget(self.reader_view)

    def _create_summary_panel(self):
        self.summary_panel = QFrame()
        self.summary_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self.summary_panel.setMinimumWidth(260)
        
        summary_layout = QVBoxLayout(self.summary_panel)
        summary_layout.setContentsMargins(12, 12, 12, 12)

        header = QHBoxLayout()
        lbl_sum_title = QLabel("💡 Chapter Summary")
        lbl_sum_title.setStyleSheet("font-size: 15px; font-weight: bold;")
        header.addWidget(lbl_sum_title, 1)

        btn_copy_sum = QPushButton("📋 Copy")
        btn_copy_sum.setFixedWidth(64)
        btn_copy_sum.clicked.connect(self._copy_summary_to_clipboard)
        header.addWidget(btn_copy_sum)
        summary_layout.addLayout(header)

        self.txt_summary = QTextBrowser()
        self.txt_summary.setReadOnly(True)
        self.txt_summary.setPlaceholderText(
            "Click '✨ Summarize Chapter' or right-click selected text to generate an instant offline summary of key takeaways."
        )
        summary_layout.addWidget(self.txt_summary)

    # --- File Loading & Book State ---

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open E-Book or PDF",
            "",
            "Books (*.pdf *.epub);;PDF Files (*.pdf);;EPUB Files (*.epub);;All Files (*.*)"
        )
        if file_path:
            self.load_book_file(file_path)

    def load_book_file(self, filepath: str):
        try:
            self.stop_playback()
            self.status_reading_info.setText(f"Loading '{os.path.basename(filepath)}'...")
            QApplication.processEvents()

            parsed = book_parser.load_book(filepath)
            if not parsed.chapters:
                QMessageBox.warning(self, "Empty Book", "Could not extract any readable chapters or text from this file.")
                return

            self.book = parsed
            storage.record_book_opened(filepath, parsed.title, len(parsed.chapters))

            # Populate TOC
            self.list_toc.clear()
            for ch in self.book.chapters:
                item = QListWidgetItem(f"{ch.title} ({len(ch.sentences)} sents)")
                self.list_toc.addItem(item)

            # Check saved reading progress
            progress = storage.get_reading_progress(filepath)
            target_ch = 0
            target_sent = 0

            if progress:
                target_ch = min(progress.get("chapter_index", 0), len(self.book.chapters) - 1)
                target_sent = progress.get("sentence_index", 0)
                self.status_reading_info.setText(
                    f"Resumed '{self.book.title}' from {self.book.chapters[target_ch].title}, sentence {target_sent + 1}."
                )
            else:
                self.status_reading_info.setText(f"Opened '{self.book.title}'")

            # Load chapter into reader
            self.load_chapter(target_ch, target_sent)
            self._refresh_highlights_list()
            self._refresh_bookmarks_list()

        except Exception as e:
            QMessageBox.critical(self, "Error Loading Book", f"Failed to open book:\n{str(e)}")

    def load_chapter(self, chapter_index: int, initial_sentence_index: int = 0):
        if not self.book or chapter_index >= len(self.book.chapters):
            return

        self.current_chapter_idx = chapter_index
        chapter = self.book.chapters[chapter_index]
        self.lbl_chapter_title.setText(f"{self.book.title} — {chapter.title}")

        # Update TOC selection
        self.list_toc.setCurrentRow(chapter_index)

        # Prepare sentence list for audio player
        sentence_texts = [s.text for s in chapter.sentences]
        self.audio_player.set_content(sentence_texts, initial_sentence_index)

        # Build document text and record exact character ranges for sentence highlighting
        self.sentence_char_ranges.clear()
        full_text = ""
        curr_offset = 0

        for para in chapter.paragraphs:
            for s_item in para:
                st = s_item.text
                start = curr_offset
                full_text += st + " "
                end = curr_offset + len(st)
                self.sentence_char_ranges.append((start, end))
                curr_offset = len(full_text)
            full_text += "\n\n"
            curr_offset = len(full_text)

        self.reader_view.setPlainText(full_text)
        self._apply_reader_formatting()

        # Apply saved user highlights for this chapter
        self._apply_chapter_highlights(chapter_index)

        # Scroll to initial sentence
        self.highlight_sentence(initial_sentence_index, scroll=True)
        self.status_sentence_info.setText(f"Sentence {initial_sentence_index + 1} of {len(chapter.sentences)}")

        # Auto clear previous summary for fresh chapter view
        self.txt_summary.clear()

    # --- Highlighting & Reader Formatting ---

    def _apply_reader_formatting(self):
        font = QFont("Georgia", self.font_size)
        self.reader_view.setFont(font)

    def highlight_sentence(self, sentence_index: int, scroll: bool = True):
        """Highlights the active spoken sentence with a warm distinct background."""
        if not self.sentence_char_ranges or sentence_index >= len(self.sentence_char_ranges):
            return

        self.current_sentence_idx = sentence_index
        start, end = self.sentence_char_ranges[sentence_index]

        # Reset selection and format
        doc = self.reader_view.document()
        cursor = QTextCursor(doc)

        # Clear previous active highlight formatting across document by reapplying normal char format
        cursor.select(QTextCursor.SelectionType.Document)
        default_format = QTextCharFormat()
        theme = THEMES[self.current_theme]
        default_format.setForeground(QColor(theme["text_main"]))
        default_format.setBackground(QColor("transparent"))
        cursor.setCharFormat(default_format)

        # Re-apply permanent user highlights
        self._apply_chapter_highlights(self.current_chapter_idx)

        # Apply active voice sentence format
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)

        active_format = QTextCharFormat()
        active_format.setBackground(QColor(theme["active_sentence_bg"]))
        active_format.setForeground(QColor(theme["active_sentence_text"]))
        active_format.setFontWeight(QFont.Weight.DemiBold)
        cursor.mergeCharFormat(active_format)

        if scroll:
            self.reader_view.setTextCursor(cursor)
            self.reader_view.ensureCursorVisible()

        total = len(self.sentence_char_ranges)
        self.status_sentence_info.setText(f"Sentence {sentence_index + 1} of {total}")

    def _apply_chapter_highlights(self, chapter_index: int):
        if not self.book:
            return
        highlights = storage.get_highlights_for_book(self.book.filepath, chapter_index)
        doc = self.reader_view.document()
        
        for h in highlights:
            text = h["selected_text"]
            color_hex = h.get("color", "#ffeb3b")
            # Search for highlight text
            cursor = doc.find(text)
            while not cursor.isNull():
                h_format = QTextCharFormat()
                h_format.setBackground(QColor(color_hex))
                h_format.setForeground(QColor("#000000"))
                cursor.mergeCharFormat(h_format)
                cursor = doc.find(text, cursor)

    # --- Audio Player Events & Actions ---

    def _on_tts_sentence_started(self, sentence_index: int, text: str):
        self.highlight_sentence(sentence_index, scroll=True)

    def _on_tts_state_changed(self, is_playing: bool, is_paused: bool):
        if is_playing and not is_paused:
            self.btn_play_pause.setText("⏸ Pause")
            self.status_reading_info.setText("Narration in progress...")
        elif is_playing and is_paused:
            self.btn_play_pause.setText("▶ Resume")
            self.status_reading_info.setText("Narration paused.")
        else:
            self.btn_play_pause.setText("▶ Play")
            self.status_reading_info.setText("Playback stopped.")

    def _on_tts_finished(self):
        # Auto-advance to next chapter if available!
        if self.book and self.current_chapter_idx + 1 < len(self.book.chapters):
            self.status_reading_info.setText(f"Completed chapter. Moving to next chapter...")
            self.load_chapter(self.current_chapter_idx + 1, 0)
            self.audio_player.play(0)
        else:
            self.status_reading_info.setText("Reached the end of the book!")

    def _on_tts_error(self, err_msg: str):
        self.status_reading_info.setText(f"TTS Notice: {err_msg}")

    def toggle_play_pause(self):
        if not self.book:
            self.open_file_dialog()
            return

        if self.audio_player.is_playing:
            if self.audio_player.is_paused:
                self.audio_player.resume()
            else:
                self.audio_player.pause()
        else:
            self.audio_player.play(self.current_sentence_idx)

    def stop_playback(self):
        self.audio_player.stop()

    def previous_sentence(self):
        target = max(0, self.current_sentence_idx - 1)
        self.audio_player.skip_to(target)
        self.highlight_sentence(target, scroll=True)

    def next_sentence(self):
        if self.book:
            chapter = self.book.chapters[self.current_chapter_idx]
            target = min(len(chapter.sentences) - 1, self.current_sentence_idx + 1)
            self.audio_player.skip_to(target)
            self.highlight_sentence(target, scroll=True)

    def _on_voice_changed(self):
        voice_id = self.combo_voices.currentData()
        if voice_id:
            self.audio_player.set_voice(voice_id)

    def _on_speed_changed(self):
        speed = self.combo_speed.currentData()
        if speed:
            self.audio_player.set_speed(speed)

    # --- Context Menu: Highlights, Notes, Summarize, Read from Here ---

    def _show_reader_context_menu(self, pos):
        menu = QMenu(self)
        cursor = self.reader_view.textCursor()
        selected_text = cursor.selectedText().strip()

        # Determine sentence clicked
        click_cursor = self.reader_view.cursorForPosition(pos)
        click_char_pos = click_cursor.position()
        target_sent_idx = 0
        for i, (s_start, s_end) in enumerate(self.sentence_char_ranges):
            if s_start <= click_char_pos <= s_end:
                target_sent_idx = i
                break

        # Action: Read from here
        act_read_here = menu.addAction(f"▶ Read from sentence {target_sent_idx + 1}")
        act_read_here.triggered.connect(lambda: self._read_from_sentence(target_sent_idx))
        menu.addSeparator()

        if selected_text:
            # Highlighting actions
            menu_hl = menu.addMenu("🖍 Highlight Selection")
            colors = [
                ("🟡 Yellow", "#ffeb3b"),
                ("🟢 Green", "#a7f3d0"),
                ("🔵 Blue", "#bfdbfe"),
                ("🟣 Pink", "#fbcfe8"),
            ]
            for label, hex_color in colors:
                act = menu_hl.addAction(label)
                act.triggered.connect(lambda checked, col=hex_color, txt=selected_text, s_idx=target_sent_idx: 
                                      self._add_highlight(txt, s_idx, col))

            # Add Note
            act_note = menu.addAction("📝 Add Note to Highlight...")
            act_note.triggered.connect(lambda: self._add_note_to_selection(selected_text, target_sent_idx))

            menu.addSeparator()

            # Summarize selection
            act_sum_sel = menu.addAction("💡 Summarize Selected Text")
            act_sum_sel.triggered.connect(lambda: self._summarize_text_selection(selected_text))
            menu.addSeparator()

        # Copy action
        if selected_text:
            act_copy = menu.addAction("📋 Copy Text")
            act_copy.triggered.connect(cursor.selectedText)

        menu.exec(self.reader_view.mapToGlobal(pos))

    def _read_from_sentence(self, sent_idx: int):
        self.audio_player.skip_to(sent_idx)
        self.highlight_sentence(sent_idx, scroll=True)
        if not self.audio_player.is_playing:
            self.audio_player.play(sent_idx)

    def _add_highlight(self, text: str, sent_idx: int, color_hex: str, note: str = ""):
        if not self.book:
            return
        storage.add_highlight(self.book.filepath, self.current_chapter_idx, sent_idx, text, color_hex, note)
        self._apply_chapter_highlights(self.current_chapter_idx)
        self._refresh_highlights_list()
        self.status_reading_info.setText(f"Highlight saved.")

    def _add_note_to_selection(self, text: str, sent_idx: int):
        note, ok = QInputDialog.getMultiLineText(self, "Add Annotation Note", f"Enter note for: '{text[:50]}...'")
        if ok and note.strip():
            self._add_highlight(text, sent_idx, "#ffeb3b", note.strip())

    def _refresh_highlights_list(self):
        self.list_highlights.clear()
        if not self.book:
            return
        highlights = storage.get_highlights_for_book(self.book.filepath)
        for h in highlights:
            snippet = h["selected_text"]
            if len(snippet) > 60:
                snippet = snippet[:57] + "..."
            note_indicator = f" [Note: {h['note'][:30]}...]" if h.get("note") else ""
            item_text = f"Ch {h['chapter_index'] + 1}: \"{snippet}\"{note_indicator}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, h)
            self.list_highlights.addItem(item)

    def _refresh_bookmarks_list(self):
        self.list_bookmarks.clear()
        if not self.book:
            return
        bookmarks = storage.get_bookmarks_for_book(self.book.filepath)
        for bm in bookmarks:
            item_text = f"Ch {bm['chapter_index'] + 1}, Sent {bm['sentence_index'] + 1} — {bm['title']}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, bm)
            self.list_bookmarks.addItem(item)

    def add_bookmark_at_current_spot(self):
        if not self.book:
            return
        title, ok = QInputDialog.getText(
            self, "Add Bookmark", "Bookmark Name:",
            text=f"Spot in {self.book.chapters[self.current_chapter_idx].title}"
        )
        if ok and title.strip():
            storage.add_bookmark(
                self.book.filepath,
                self.current_chapter_idx,
                self.current_sentence_idx,
                title.strip()
            )
            self._refresh_bookmarks_list()
            self.status_reading_info.setText(f"Bookmark added: {title.strip()}")

    # --- Sidebar Navigation Handlers ---

    def _on_toc_item_clicked(self, item: QListWidgetItem):
        row = self.list_toc.row(item)
        if row != self.current_chapter_idx:
            self.stop_playback()
            self.load_chapter(row, 0)

    def _on_highlight_item_clicked(self, item: QListWidgetItem):
        h = item.data(Qt.ItemDataRole.UserRole)
        if not h:
            return
        ch_idx = h["chapter_index"]
        s_idx = h.get("sentence_index", 0)
        if ch_idx != self.current_chapter_idx:
            self.stop_playback()
            self.load_chapter(ch_idx, s_idx)
        else:
            self.highlight_sentence(s_idx, scroll=True)

    def _on_bookmark_item_clicked(self, item: QListWidgetItem):
        bm = item.data(Qt.ItemDataRole.UserRole)
        if not bm:
            return
        ch_idx = bm["chapter_index"]
        s_idx = bm["sentence_index"]
        if ch_idx != self.current_chapter_idx:
            self.stop_playback()
            self.load_chapter(ch_idx, s_idx)
        else:
            self.highlight_sentence(s_idx, scroll=True)

    # --- Summarization ---

    def summarize_current_chapter(self):
        if not self.book or self.current_chapter_idx >= len(self.book.chapters):
            return
        chapter = self.book.chapters[self.current_chapter_idx]
        self.txt_summary.setHtml("<i>Generating chapter summary...</i>")
        QApplication.processEvents()

        text = chapter.plain_text
        summary_sents = summarizer.summarize_text(text, sentence_count=6)
        formatted = summarizer.format_summary_bullet_points(summary_sents)

        html = f"""
        <div style="font-family: sans-serif; line-height: 1.6;">
            <h3 style="margin-top: 0;">Chapter Takeaways: {chapter.title}</h3>
            {formatted.replace('• ', '<p>• ').replace('\n\n', '</p>')}
        </div>
        """
        self.txt_summary.setHtml(html)

    def _summarize_text_selection(self, selected_text: str):
        self.txt_summary.setHtml("<i>Summarizing selected text...</i>")
        QApplication.processEvents()
        summary_sents = summarizer.summarize_text(selected_text, sentence_count=4)
        formatted = summarizer.format_summary_bullet_points(summary_sents)
        html = f"""
        <div style="font-family: sans-serif; line-height: 1.6;">
            <h3 style="margin-top: 0;">Selection Summary</h3>
            {formatted.replace('• ', '<p>• ').replace('\n\n', '</p>')}
        </div>
        """
        self.txt_summary.setHtml(html)

    def _copy_summary_to_clipboard(self):
        text = self.txt_summary.toPlainText().strip()
        if text:
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            self.status_reading_info.setText("Summary copied to clipboard!")

    # --- Autosave Progress ---

    def _autosave_progress(self):
        if self.book:
            storage.save_reading_progress(
                self.book.filepath,
                self.current_chapter_idx,
                self.current_sentence_idx
            )

    def _check_last_opened_book(self):
        recent = storage.get_recent_books(limit=1)
        if recent:
            last = recent[0]
            fpath = last["filepath"]
            if os.path.exists(fpath):
                self.load_book_file(fpath)

    # --- Font & Theme Controls ---

    def increase_font_size(self):
        self.font_size = min(36, self.font_size + 2)
        self._apply_reader_formatting()

    def decrease_font_size(self):
        self.font_size = max(11, self.font_size - 2)
        self._apply_reader_formatting()

    def _on_theme_changed(self):
        theme_key = self.combo_theme.currentData()
        if theme_key in THEMES:
            self.current_theme = theme_key
            self._apply_theme()
            self._apply_reader_formatting()
            self.highlight_sentence(self.current_sentence_idx, scroll=False)

    def _apply_theme(self):
        t = THEMES[self.current_theme]
        qss = f"""
        QMainWindow {{
            background-color: {t["bg_main"]};
        }}
        QWidget {{
            color: {t["text_main"]};
        }}
        QToolBar {{
            background-color: {t["bg_sidebar"]};
            border-bottom: 1px solid {t["border"]};
            spacing: 6px;
            padding: 4px;
        }}
        QPushButton {{
            background-color: {t["bg_reader"]};
            color: {t["text_main"]};
            border: 1px solid {t["border"]};
            border-radius: 6px;
            padding: 6px 12px;
        }}
        QPushButton:hover {{
            background-color: {t["border"]};
        }}
        QComboBox {{
            background-color: {t["bg_reader"]};
            color: {t["text_main"]};
            border: 1px solid {t["border"]};
            border-radius: 6px;
            padding: 4px 8px;
        }}
        QTabWidget::pane {{
            border: 1px solid {t["border"]};
            background-color: {t["bg_sidebar"]};
        }}
        QTabBar::tab {{
            background-color: {t["bg_main"]};
            color: {t["text_secondary"]};
            padding: 8px 14px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
        }}
        QTabBar::tab:selected {{
            background-color: {t["bg_sidebar"]};
            color: {t["text_main"]};
            font-weight: bold;
        }}
        QListWidget {{
            background-color: {t["bg_sidebar"]};
            color: {t["text_main"]};
            border: none;
            padding: 4px;
        }}
        QListWidget::item {{
            padding: 8px;
            border-radius: 4px;
        }}
        QListWidget::item:selected {{
            background-color: {t["accent"]};
            color: #ffffff;
        }}
        QTextBrowser {{
            background-color: {t["bg_reader"]};
            color: {t["text_main"]};
            border: 1px solid {t["border"]};
            border-radius: 8px;
            padding: 16px;
            line-height: 1.7;
        }}
        QStatusBar {{
            background-color: {t["bg_sidebar"]};
            color: {t["text_secondary"]};
            border-top: 1px solid {t["border"]};
        }}
        QSplitter::handle {{
            background-color: {t["border"]};
        }}
        """
        self.setStyleSheet(qss)

    def closeEvent(self, event):
        self.stop_playback()
        self._autosave_progress()
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.toggle_play_pause()
            event.accept()
        elif event.key() == Qt.Key.Key_Escape:
            self.stop_playback()
            event.accept()
        else:
            super().keyPressEvent(event)
