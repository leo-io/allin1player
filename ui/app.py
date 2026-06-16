from __future__ import annotations
import copy
import dataclasses
import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
import soundfile as sf

from domain.models import Arrangement
from file_io.arrangement_store import ArrangementStore
from playback.engine import PlaybackEngine
from playback.state import TransportState, VirtualQueue
from ui.renderer import CanvasRenderer

logger = logging.getLogger(__name__)


class ArrangementEditor:
    def __init__(self, arrangement_path=None):
        self.arrangement_path = None
        self.audio_data = None
        self.sr = None
        self.total_ms = 0

        self.lock = threading.RLock()
        self.arrangement: Arrangement | None = None
        self.renderer = None
        self.state = TransportState()
        self.engine = None
        self.audio_path = None

        # Audio cache: maps audiosource path -> (audio_data, sr)
        self.audio_cache: dict[str, tuple[np.ndarray, int]] = {}
        self.cache_lock = threading.RLock()

        # Clipboard for bar editing (list of bars, supports multi-bar copy)
        self._clipboard_bars = []

        # Clipboard for section editing (list of sections, supports multi-section copy)
        self._clipboard_sections = []

        # Undo/redo history (snapshots of the arrangement + selection state)
        self._undo_stack = []
        self._redo_stack = []
        self._history_limit = 100

        # WAV files only
        self.audio_files = sorted(list(Path(".").glob("*.wav")))

        self._build_ui()

        if arrangement_path:
            self.load_file(arrangement_path)
        elif self.audio_files:
            # Try to load arrangement for first WAV file
            first_wav = self.audio_files[0]
            arrangement_p = ArrangementStore.arrangement_path_for(first_wav)
            if arrangement_p.exists():
                self.load_file(str(arrangement_p))
            else:
                # Try to create from analysis JSON
                analysis_p = first_wav.with_suffix(".json")
                if analysis_p.exists():
                    self.create_and_load_arrangement(str(first_wav), str(analysis_p))
                else:
                    logger.warning(f"No arrangement or analysis file found for {first_wav}")

    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("Arrangement Editor")
        self.root.configure(bg="#1a1a2e")
        self.root.minsize(1000, 500)

        main_frame = tk.Frame(self.root, bg="#1a1a2e")
        main_frame.pack(fill="both", expand=True)

        content = tk.Frame(main_frame, bg="#1a1a2e")
        content.pack(fill="both", expand=True)

        top = tk.Frame(content, bg="#1a1a2e")
        top.pack(fill="x", padx=16, pady=(14, 4))
        self.filename_label = tk.Label(top, font=("Segoe UI", 11, "bold"),
                                       fg="#e0e0e0", bg="#1a1a2e")
        self.filename_label.pack(side="left")
        self.time_label = tk.Label(top, text=self._fmt(0), font=("Segoe UI", 11),
                                   fg="#aaaaaa", bg="#1a1a2e")
        self.time_label.pack(side="right")

        self.menu_bar = tk.Menu(self.root, bg="#1a1a2e", fg="#e0e0e0")
        self.root.config(menu=self.menu_bar)

        file_menu = tk.Menu(self.menu_bar, tearoff=0, bg="#1a1a2e", fg="#e0e0e0")
        file_menu.add_command(label="Save", command=self._cmd_save)
        self.menu_bar.add_cascade(label="File", menu=file_menu)

        self.edit_menu = tk.Menu(self.menu_bar, tearoff=0, bg="#1a1a2e", fg="#e0e0e0")
        self.edit_menu.add_command(label="Undo", command=self._cmd_undo,
                                   accelerator="Ctrl+Z")
        self.edit_menu.add_command(label="Redo", command=self._cmd_redo,
                                   accelerator="Ctrl+Y")
        self.edit_menu.add_separator()
        self.edit_menu.add_command(label="Copy", command=self._cmd_copy_bar,
                                   accelerator="Ctrl+C")
        self.edit_menu.add_command(label="Paste", command=self._cmd_paste_bar,
                                   accelerator="Ctrl+V")
        self.edit_menu.add_command(label="Delete", command=self._cmd_delete_bar,
                                   accelerator="Del")
        self.menu_bar.add_cascade(label="Edit", menu=self.edit_menu)

        cframe = tk.Frame(content, bg="#1a1a2e")
        cframe.pack(fill="both", expand=True, padx=10, pady=(4, 14))
        self.canvas = tk.Canvas(cframe, bg="#16213e", highlightthickness=0)
        vbar = ttk.Scrollbar(cframe, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Button-3>", self._on_canvas_right_click)

        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.bind("<space>", lambda e: self.pause() if self.state.playing else self.play())
        self.root.bind("<Key-l>", lambda e: self._on_key("l"))
        self.root.bind("<Key-L>", lambda e: self._on_key("l"))
        self.root.bind("<Key-Escape>", lambda e: self._on_key("escape"))
        self.root.bind("<Control-s>", lambda e: self._cmd_save())
        self.root.bind("<Control-S>", lambda e: self._cmd_save())
        self.root.bind("<Delete>", lambda e: self._cmd_delete())
        self.root.bind("<Control-c>", lambda e: self._cmd_copy())
        self.root.bind("<Control-C>", lambda e: self._cmd_copy())
        self.root.bind("<Control-v>", lambda e: self._cmd_paste())
        self.root.bind("<Control-V>", lambda e: self._cmd_paste())
        self.root.bind("<Control-z>", lambda e: self._cmd_undo())
        self.root.bind("<Control-y>", lambda e: self._cmd_redo())
        self.root.bind("<Control-Y>", lambda e: self._cmd_redo())
        self.root.bind("<Control-Shift-Z>", lambda e: self._cmd_redo())
        self.root.bind("<Key-F2>", lambda e: self._cmd_rename_section())

    # ------------------------------------------------------------------ file loading

    def create_and_load_arrangement(self, audio_path: str, analysis_path: str):
        """Create arrangement from analysis JSON and load it."""
        logger.debug(f"[CHECKPOINT] Entering create_and_load_arrangement()")
        logger.info(f"Creating arrangement from analysis: {analysis_path}")

        try:
            audio_p = Path(audio_path)
            analysis_p = Path(analysis_path)

            # Load or create arrangement
            logger.debug("Loading arrangement via ArrangementStore.load_or_create()")
            arrangement = ArrangementStore.load_or_create(str(audio_p), str(analysis_p))

            # Save as master
            master_p = ArrangementStore.master_path_for(audio_p)
            logger.info(f"Saving master arrangement to: {master_p}")
            ArrangementStore._save_to_path(arrangement, str(master_p))
            logger.debug(f"[CHECKPOINT] Master arrangement saved")

            # Now load the created master file
            logger.debug(f"Loading created master arrangement")
            self.load_file(str(master_p))

        except Exception as e:
            error_msg = f"Error creating arrangement from analysis: {type(e).__name__}: {e}"
            logger.error(f"[EXCEPTION] {error_msg}")

    # ------------------------------------------------------------------ audio loading

    def _load_audio_by_source(self, audiosource: str) -> tuple[np.ndarray, int] | None:
        """Load audio from the specified source file. Caches results for performance."""
        logger.debug(f"[CHECKPOINT] Entering _load_audio_by_source() with audiosource={audiosource}")

        if not audiosource:
            logger.warning("Empty audiosource provided")
            return None

        # Check cache first
        with self.cache_lock:
            if audiosource in self.audio_cache:
                logger.debug(f"Audio cache hit for {audiosource}")
                return self.audio_cache[audiosource]

        # Load from file
        try:
            source_path = Path(audiosource)
            if not source_path.exists():
                logger.error(f"[EXCEPTION] Audio source file not found: {source_path}")
                return None

            logger.debug(f"Loading audio from: {source_path}")
            raw, sr = sf.read(str(source_path), always_2d=True)
            logger.debug(f"Audio loaded: sr={sr}, shape={raw.shape}")

            # Convert to mono if needed
            if raw.shape[1] > 1:
                logger.debug(f"Converting {raw.shape[1]} channels to mono")
                raw = raw.mean(axis=1, keepdims=True)

            audio_data = raw.astype(np.float32)
            logger.debug(f"Audio preprocessed: {len(audio_data)} samples")

            # Cache it
            with self.cache_lock:
                self.audio_cache[audiosource] = (audio_data, sr)
                logger.debug(f"Audio cached for {audiosource}")

            logger.debug(f"[CHECKPOINT] Exiting _load_audio_by_source() successfully")
            return (audio_data, sr)

        except Exception as e:
            error_msg = f"Error loading audio from {audiosource}: {type(e).__name__}: {e}"
            logger.error(f"[EXCEPTION] {error_msg}")
            return None

    def _get_audio_for_bar(self, bar_idx: int) -> tuple[np.ndarray, int] | None:
        """Get audio data for a specific bar by loading from its audiosource."""
        logger.debug(f"[CHECKPOINT] Entering _get_audio_for_bar() with bar_idx={bar_idx}")

        if self.arrangement is None:
            logger.warning("No arrangement loaded")
            return None

        bars_flat = self.arrangement.bars
        if bar_idx < 0 or bar_idx >= len(bars_flat):
            logger.error(f"[EXCEPTION] Bar index out of range: {bar_idx}, total bars: {len(bars_flat)}")
            return None

        bar = bars_flat[bar_idx]
        logger.debug(f"Bar {bar_idx} audiosource: {bar.audiosource}")

        result = self._load_audio_by_source(bar.audiosource)
        if result:
            audio_data, sr = result
            total_ms = len(audio_data) * 1000 / sr
            logger.debug(f"[CHECKPOINT] Exiting _get_audio_for_bar() with audio: sr={sr}, duration={total_ms:.0f}ms")
        return result

    # ------------------------------------------------------------------ commands

    def _cmd_save(self):
        logger.debug(f"[CHECKPOINT] Entering _cmd_save()")
        if self.arrangement is None or self.audio_path is None:
            logger.warning("Cannot save: arrangement or audio_path is None")
            return
        try:
            arrangement_p = ArrangementStore.arrangement_path_for(self.audio_path)
            logger.info(f"Saving arrangement to: {arrangement_p}")
            ArrangementStore.save(self.arrangement, self.audio_path)
            self.state.dirty = False
            logger.info(f"[CHECKPOINT] Save completed successfully")
        except Exception as e:
            error_msg = f"Error saving arrangement: {e}"
            logger.error(f"[EXCEPTION] {error_msg}")
            import tkinter.messagebox as messagebox
            messagebox.showerror("Save Error", error_msg)

    # ------------------------------------------------------------------ file loading

    def load_file(self, arrangement_path):
        logger.debug(f"[CHECKPOINT] Entering load_file() with arrangement_path={arrangement_path}")
        self.stop()
        with self.cache_lock:
            self.audio_cache.clear()
        logger.debug("Audio cache cleared")

        p = Path(arrangement_path)
        if not p.exists():
            logger.error(f"[EXCEPTION] Arrangement file not found: {p}")
            return

        # Load arrangement
        try:
            logger.debug(f"[CHECKPOINT] Loading arrangement from {p}")
            self.arrangement = ArrangementStore._load(str(p))
            logger.debug(f"[CHECKPOINT] Arrangement loaded: {self.arrangement.name}")

            # Validate audio sources
            self._validate_audio_sources()
        except Exception as e:
            error_msg = f"Error loading arrangement: {e}"
            logger.error(f"[EXCEPTION] {error_msg}")
            return

        # Load audio from first bar's audiosource
        if not self.arrangement.bars:
            logger.error("[EXCEPTION] Arrangement has no bars")
            return

        first_bar = self.arrangement.bars[0]
        logger.debug(f"Loading audio from first bar's audiosource: {first_bar.audiosource}")

        audio_result = self._load_audio_by_source(first_bar.audiosource)
        if not audio_result:
            error_msg = "Failed to load audio from bar audiosource"
            logger.error(f"[EXCEPTION] {error_msg}")
            return

        self.audio_data, self.sr = audio_result
        self.total_ms = len(self.audio_data) * 1000 / self.sr
        logger.info(f"Audio loaded: sr={self.sr}, duration={self.total_ms:.0f}ms, samples={len(self.audio_data)}")

        # Process arrangement with loaded audio parameters
        try:
            logger.debug(f"[CHECKPOINT] Reindexing arrangement with sr={self.sr}")
            self.arrangement.reindex(self.sr)
            logger.debug(f"[CHECKPOINT] Arrangement reindexed")

            logger.debug(f"[CHECKPOINT] Setting bar frames: total_frames={len(self.audio_data)}")
            self.arrangement.set_bar_frames(self.sr, len(self.audio_data))
            logger.debug(f"[CHECKPOINT] Bar frames set")
        except Exception as e:
            error_msg = f"Error processing arrangement: {e}"
            logger.error(f"[EXCEPTION] {error_msg}")
            return

        # Reset transport state for new file
        self.state.frame_idx = 0
        self.state.playing = False
        self.state.paused = False
        self.state.vq = None
        self.state.loop_range = None
        self.state.dirty = False
        self.state.cursor = None
        self.state.play_bar = None
        self.state.selection = None
        self.state.section_selection = None
        self.state.section_cursor = None
        self.state.has_section_clipboard = False
        self._clipboard_bars = []
        self._clipboard_sections = []
        self._undo_stack = []
        self._redo_stack = []

        # Create playback engine
        self.audio_path = first_bar.audiosource
        self.engine = PlaybackEngine(
            audio_data=self.audio_data,
            arrangement=self.arrangement,
            state=self.state,
            lock=self.lock,
            sr=self.sr,
        )
        logger.debug("[CHECKPOINT] PlaybackEngine created")

        # Initialize renderer
        try:
            self.renderer = CanvasRenderer(self.canvas, self.arrangement)
            logger.debug(f"[CHECKPOINT] Renderer initialized")

        except Exception as e:
            error_msg = f"Error initializing renderer: {e}"
            logger.error(f"[EXCEPTION] {error_msg}")
            return

        # Update UI labels
        try:
            label_text = p.name
            logger.info(f"Loaded arrangement: {p}")

            self.arrangement_path = p
            self.root.title(f"Arrangement Editor — {label_text}")
            self.filename_label.config(text=label_text)
            self.time_label.config(text=self._fmt(0))
            logger.debug(f"[CHECKPOINT] UI updated: {label_text}")

            self.renderer.draw_all(self.state)
            logger.info(f"[CHECKPOINT] Exiting load_file() successfully")

        except Exception as e:
            error_msg = f"Error updating UI labels: {e}"
            logger.error(f"[EXCEPTION] {error_msg}")
            # Still continue - the main loading succeeded
            self.arrangement_path = p
            self.root.title(f"Arrangement Editor")
            self.filename_label.config(text=self.arrangement.name)
            self.time_label.config(text=self._fmt(0))
            self.renderer.draw_all(self.state)

    # ------------------------------------------------------------------ event handlers

    def _validate_audio_sources(self) -> bool:
        """Validate that all bars reference consistent audio sources."""
        logger.debug("[CHECKPOINT] Validating audio sources")
        if not self.arrangement or not self.arrangement.bars:
            return False

        audio_sources = set(bar.audiosource for bar in self.arrangement.bars)
        if len(audio_sources) > 1:
            logger.warning(f"Multiple audio sources detected in arrangement: {audio_sources}")
            logger.warning("Current playback engine supports single audio source per arrangement")

        logger.debug(f"Audio sources validated: {len(audio_sources)} unique source(s)")
        return True

    def _on_canvas_click(self, event):
        logger.debug(f"[CHECKPOINT] Canvas click at x={event.x}, y={event.y}")
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        is_ctrl = bool(event.state & 0x4)
        is_shift = bool(event.state & 0x1)

        if self.arrangement is None:
            logger.warning("Canvas clicked but no arrangement loaded")
            return

        bar_idx = self.renderer.bar_at_xy(cx, cy)
        if bar_idx is not None:
            if is_ctrl:
                self._queue_bars([bar_idx])
                self.renderer.draw_all(self.state)
            elif is_shift:
                # Extend a contiguous selection from the anchor (cursor) to the
                # shift-clicked bar. The anchor stays put so repeated
                # shift-clicks grow/shrink the same range.
                anchor = self.state.cursor if self.state.cursor is not None else bar_idx
                self.state.cursor = anchor
                self.state.selection = (min(anchor, bar_idx), max(anchor, bar_idx))
                self.state.selected = ("bar", bar_idx)
                self.renderer.draw_all(self.state)
            else:
                # Plain click selects a single bar / moves the insertion cursor
                # and clears any range. To seek-and-play, Ctrl+click a bar.
                self.state.selected = ("bar", bar_idx)
                self.state.cursor = bar_idx
                self.state.selection = None
                # Clear section state when clicking a bar
                self.state.section_selection = None
                self.state.section_cursor = None
                self.renderer.draw_all(self.state)
            return

        sec_idx = self.renderer.section_at_xy(cx, cy)
        if sec_idx is not None:
            is_ctrl_shift = is_ctrl and is_shift
            if is_ctrl_shift:
                # Multi-select sections with Ctrl+Shift+click
                anchor_idx = self.state.selected[1] if (self.state.selected and
                                                        self.state.selected[0] == "section") else sec_idx
                self.state.section_selection = (min(anchor_idx, sec_idx), max(anchor_idx, sec_idx))
                self.state.selected = ("section", sec_idx)
                self.state.section_cursor = sec_idx
            elif is_ctrl:
                # Ctrl+click on section: queue the section's bars
                if sec_idx < len(self.arrangement.sections):
                    sec = self.arrangement.sections[sec_idx]
                    if sec.bars:
                        sec_first = self._section_first_bar_idx(sec)
                        self._queue_bars(range(sec_first, sec_first + len(sec.bars)))
            else:
                # Plain click selects a single section
                self.state.selected = ("section", sec_idx)
                self.state.section_cursor = sec_idx
                self.state.section_selection = None
            # Clear bar state when clicking a section
            self.state.selection = None
            self.state.cursor = None
            self.renderer.draw_all(self.state)
            return

        self.state.selected = None
        self.state.selection = None
        self.state.section_selection = None
        self.state.section_cursor = None
        self.renderer.draw_all(self.state)

    def _on_canvas_right_click(self, event):
        logger.debug(f"[CHECKPOINT] Canvas right-click at x={event.x}, y={event.y}")
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        if self.arrangement is None:
            return

        # Check for section click first
        sec_idx = self.renderer.section_at_xy(cx, cy)
        if sec_idx is not None:
            # If right-clicking outside current selection, reselect to that section
            sec_rng = self._selected_section_range()
            if not (sec_rng is not None and sec_rng[0] <= sec_idx <= sec_rng[1]):
                self.state.selected = ("section", sec_idx)
                self.state.section_cursor = sec_idx
                self.state.section_selection = None
            self.renderer.draw_all(self.state)

            sec_rng = self._selected_section_range()
            n_sel = (sec_rng[1] - sec_rng[0] + 1) if sec_rng else 0
            n_clip = len(self._clipboard_sections)
            copy_label = self._count_label("Copy", n_sel, "sections")
            delete_label = self._count_label("Delete", n_sel, "sections")
            paste_label = self._count_label("Paste", n_clip, "sections")

            menu = tk.Menu(self.root, tearoff=0, bg="#1a1a2e", fg="#e0e0e0")

            menu.add_command(
                label="Rename",
                command=self._cmd_rename_section,
                accelerator="F2"
            )
            menu.add_command(
                label=copy_label,
                command=self._cmd_copy_section,
                accelerator="Ctrl+C"
            )
            menu.add_command(
                label=paste_label,
                command=self._cmd_paste_section,
                accelerator="Ctrl+V",
                state="normal" if n_clip else "disabled"
            )
            menu.add_command(
                label=delete_label,
                command=self._cmd_delete_section,
                accelerator="Delete",
                state="normal" if n_sel < len(self.arrangement.sections) else "disabled"
            )
            menu.add_separator()

            menu.add_command(
                label="Queue Section",
                command=self._cmd_queue_section,
                accelerator="Ctrl+click"
            )
            has_queue = self.state.vq is not None and not self.state.vq.is_empty()
            menu.add_command(
                label="Clear Queue",
                command=self._cmd_clear_queue,
                state="normal" if has_queue else "disabled"
            )
            menu.add_command(
                label="Toggle Loop Queue",
                command=self._cmd_toggle_queue_loop,
                state="normal" if has_queue else "disabled"
            )

            try:
                menu.post(event.x_root, event.y_root)
            except tk.TclError:
                pass
            return

        # Bar right-click
        bar_idx = self.renderer.bar_at_xy(cx, cy)
        if bar_idx is None:
            return

        # Keep an existing multi-bar selection if the right-click lands inside
        # it; otherwise move the cursor here and clear the selection.
        sel = self.state.selection
        if not (sel is not None and sel[0] <= bar_idx <= sel[1]):
            self.state.cursor = bar_idx
            self.state.selection = None
        self.renderer.draw_all(self.state)

        rng = self._selected_range()
        n_sel = (rng[1] - rng[0] + 1) if rng else 0
        n_clip = len(self._clipboard_bars)
        copy_label = self._count_label("Copy", n_sel, "bars")
        delete_label = self._count_label("Delete", n_sel, "bars")
        paste_label = self._count_label("Paste", n_clip, "bars")

        menu = tk.Menu(self.root, tearoff=0, bg="#1a1a2e", fg="#e0e0e0")

        menu.add_command(
            label="Undo",
            command=self._cmd_undo,
            accelerator="Ctrl+Z",
            state="normal" if self._undo_stack else "disabled"
        )
        menu.add_command(
            label="Redo",
            command=self._cmd_redo,
            accelerator="Ctrl+Y",
            state="normal" if self._redo_stack else "disabled"
        )
        menu.add_separator()

        menu.add_command(
            label=copy_label,
            command=self._cmd_copy_bar,
            accelerator="Ctrl+C"
        )
        menu.add_command(
            label=paste_label,
            command=self._cmd_paste_bar,
            accelerator="Ctrl+V",
            state="normal" if n_clip else "disabled"
        )
        menu.add_command(
            label=delete_label,
            command=self._cmd_delete_bar,
            accelerator="Delete"
        )
        menu.add_separator()

        has_queue = self.state.vq is not None and not self.state.vq.is_empty()
        menu.add_command(
            label="Clear Queue",
            command=self._cmd_clear_queue,
            state="normal" if has_queue else "disabled"
        )
        menu.add_command(
            label="Toggle Loop Queue",
            command=self._cmd_toggle_queue_loop,
            state="normal" if has_queue else "disabled"
        )

        try:
            menu.post(event.x_root, event.y_root)
        except tk.TclError:
            pass

    def _cmd_clear_queue(self):
        with self.lock:
            self.state.vq = None
            self.state.dirty = True
        if self.renderer:
            self.renderer.draw_all(self.state)

    def _cmd_toggle_queue_loop(self):
        vq = self.state.vq
        if vq is not None and not vq.is_empty():
            with self.lock:
                vq.looping = not vq.looping
                self.state.dirty = True
            if self.renderer:
                self.renderer.draw_all(self.state)

    def _on_key(self, key):
        if key == "l":
            vq = self.state.vq
            if vq is not None and not vq.is_empty():
                with self.lock:
                    vq.looping = not vq.looping
                    self.state.dirty = True
                self.renderer.draw_all(self.state)
                return
            if self.state.selected and self.arrangement is not None:
                bars_flat = self.arrangement.bars
                with self.lock:
                    if self.state.selected[0] == "bar":
                        bi = self.state.selected[1]
                        target_range = (bi, bi + 1)
                    elif self.state.selected[0] == "section":
                        si = self.state.selected[1]
                        sec = self.arrangement.sections[si]
                        sec_first_bar_idx = bars_flat.index(sec.bars[0]) if sec.bars else 0
                        sec_end_bar_idx = sec_first_bar_idx + len(sec.bars)
                        target_range = (sec_first_bar_idx, sec_end_bar_idx)
                    else:
                        return
                    self.state.loop_range = None if self.state.loop_range == target_range else target_range
                self.renderer.draw_all(self.state)
        elif key == "escape":
            with self.lock:
                self.state.vq = None
                self.state.dirty = True
            if self.arrangement is not None:
                self.renderer.draw_all(self.state)

    def _on_resize(self, event):
        if self.arrangement is None or self.renderer is None:
            return
        self.renderer.draw_all(self.state)

    # ------------------------------------------------------------------ utilities

    @staticmethod
    def _fmt(ms):
        m, s = divmod(ms // 1000, 60)
        return f"{m}:{s:02d}"

    def _time_str(self, ms):
        total = self.total_ms
        m, s = divmod(ms // 1000, 60)
        tm, ts = divmod(int(total) // 1000, 60)
        return f"{m}:{s:02d} / {tm}:{ts:02d}"

    # ------------------------------------------------------------------ bar editing

    def _section_for_bar(self, bar_idx):
        """Return (section, local_index) for the given flat bar_idx."""
        if self.arrangement is None:
            return None
        cumulative = 0
        for sec in self.arrangement.sections:
            local = bar_idx - cumulative
            if 0 <= local < len(sec.bars):
                return (sec, local)
            cumulative += len(sec.bars)
        return None

    def _reindex_arrangement(self):
        """Reassign sequential bar.idx values and rebuild playback arrays.

        Caller must hold self.lock so the audio thread never reads a
        half-updated bar list.
        """
        flat_idx = 0
        for sec_idx, sec in enumerate(self.arrangement.sections):
            sec.idx = sec_idx
            new_bars = []
            for bar in sec.bars:
                new_bars.append(dataclasses.replace(bar, idx=flat_idx))
                flat_idx += 1
            sec.bars = new_bars
        self.arrangement.reindex(self.sr)
        self.arrangement.set_bar_frames(self.sr, len(self.audio_data))

    def _clamp_play_position(self):
        """Keep cursor/play_bar within the current bar count. Holds lock."""
        total_bars = sum(len(s.bars) for s in self.arrangement.sections)
        if total_bars <= 0:
            return
        if self.state.play_bar is not None:
            self.state.play_bar = max(0, min(self.state.play_bar, total_bars - 1))
        if self.state.cursor is not None:
            self.state.cursor = max(0, min(self.state.cursor, total_bars - 1))

    def _selected_range(self):
        """Inclusive (lo, hi) flat-bar range to operate on: the active
        multi-bar selection, else the single bar at the cursor, else None."""
        if self.state.selection is not None:
            return self.state.selection
        if self.state.cursor is not None:
            return (self.state.cursor, self.state.cursor)
        return None

    def _selected_section_range(self):
        """Inclusive (lo, hi) section-index range: the active multi-section
        selection, else the single section from state.selected, else None."""
        if self.state.section_selection is not None:
            return self.state.section_selection
        if self.state.selected and self.state.selected[0] == "section":
            sec_idx = self.state.selected[1]
            return (sec_idx, sec_idx)
        return None

    def _section_flat_bar_range(self, sec_lo, sec_hi):
        """Inclusive flat-bar (lo, hi) spanning sections sec_lo..sec_hi."""
        lo = None
        hi = None
        for i in range(sec_lo, sec_hi + 1):
            if i < len(self.arrangement.sections):
                sec = self.arrangement.sections[i]
                if sec.bars:
                    sec_first = self._section_first_bar_idx(sec)
                    sec_last = sec_first + len(sec.bars) - 1
                    if lo is None:
                        lo = sec_first
                    hi = sec_last
        return (lo, hi) if lo is not None and hi is not None else None

    def _section_first_bar_idx(self, sec):
        """Flat-bar index of a section's first bar."""
        return self.arrangement.bars.index(sec.bars[0])

    def _queue_bars(self, bar_indices):
        """Append bars to the virtual queue, creating it if needed."""
        with self.lock:
            if self.state.vq is None:
                self.state.vq = VirtualQueue(
                    bars=list(bar_indices),
                    entry_bar=bar_indices[0],
                    pos=0,
                    active=True,
                )
            else:
                self.state.vq.bars.extend(bar_indices)
            self.state.dirty = True

    @staticmethod
    def _count_label(verb, n, noun):
        """Menu label that pluralizes only when more than one item is affected."""
        return f"{verb} {n} {noun}" if n > 1 else verb

    # ------------------------------------------------------------------ undo/redo

    def _snapshot(self):
        """Capture the editable state (bars + selection) for undo/redo."""
        return (
            copy.deepcopy(self.arrangement.sections),
            self.state.cursor,
            self.state.selection,
            self.state.section_cursor,
            self.state.section_selection,
        )

    def _push_undo(self):
        """Record the current state before a mutating edit. Clears redo."""
        if self.arrangement is None:
            return
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > self._history_limit:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _restore(self, snapshot):
        """Install a snapshot, rebuild playback arrays, and redraw."""
        sections, cursor, selection, section_cursor, section_selection = snapshot
        with self.lock:
            self.arrangement.sections = copy.deepcopy(sections)
            self._reindex_arrangement()
            self.state.cursor = cursor
            self.state.selection = selection
            self.state.section_cursor = section_cursor
            self.state.section_selection = section_selection
            self._clamp_play_position()
            self.state.dirty = True
        self.renderer.draw_all(self.state)

    def _cmd_undo(self):
        if self.arrangement is None or not self._undo_stack:
            logger.debug("Undo: nothing to undo")
            return
        self._redo_stack.append(self._snapshot())
        self._restore(self._undo_stack.pop())
        logger.info(f"Undo (history: {len(self._undo_stack)} left)")

    def _cmd_redo(self):
        if self.arrangement is None or not self._redo_stack:
            logger.debug("Redo: nothing to redo")
            return
        self._undo_stack.append(self._snapshot())
        self._restore(self._redo_stack.pop())
        logger.info(f"Redo (history: {len(self._redo_stack)} left)")

    def _cmd_delete_bar(self):
        """Delete the selected bar(s). Works across sections; empties are dropped."""
        if self.arrangement is None:
            return
        rng = self._selected_range()
        if rng is None:
            return
        lo, hi = rng
        total_bars = sum(len(s.bars) for s in self.arrangement.sections)
        if hi - lo + 1 >= total_bars:
            logger.warning("Cannot delete all bars in the arrangement")
            return
        self._push_undo()
        with self.lock:
            flat = 0
            for sec in self.arrangement.sections:
                kept = []
                for bar in sec.bars:
                    if not (lo <= flat <= hi):
                        kept.append(bar)
                    flat += 1
                sec.bars = kept
            # Drop any section fully emptied by the deletion.
            self.arrangement.sections = [s for s in self.arrangement.sections if s.bars]
            self._reindex_arrangement()
            new_total = sum(len(s.bars) for s in self.arrangement.sections)
            self.state.cursor = max(0, min(lo, new_total - 1))
            self.state.selection = None
            self._clamp_play_position()
            self.state.dirty = True
        logger.info(f"Deleted bars {lo}..{hi}")
        self.renderer.draw_all(self.state)

    def _cmd_copy_bar(self):
        """Copy the selected bar(s) to the clipboard."""
        if self.arrangement is None:
            return
        rng = self._selected_range()
        if rng is None:
            return
        lo, hi = rng
        bars_flat = self.arrangement.bars
        self._clipboard_bars = [bars_flat[i] for i in range(lo, hi + 1)
                                if 0 <= i < len(bars_flat)]
        logger.info(f"Copied {len(self._clipboard_bars)} bar(s) to clipboard")

    def _cmd_paste_bar(self):
        """Paste clipboard bars after the cursor position."""
        if self.arrangement is None or not self._clipboard_bars or self.state.cursor is None:
            logger.warning("Paste: no arrangement, clipboard, or cursor")
            return
        result = self._section_for_bar(self.state.cursor)
        if result is None:
            return
        sec, local_idx = result
        self._push_undo()
        with self.lock:
            new_bars = [dataclasses.replace(b, idx=0) for b in self._clipboard_bars]
            sec.bars[local_idx + 1:local_idx + 1] = new_bars
            self._reindex_arrangement()
            self.state.cursor = self.state.cursor + len(new_bars)
            self.state.selection = None
            self._clamp_play_position()
            self.state.dirty = True
        logger.info(f"Pasted {len(new_bars)} bar(s) after bar {self.state.cursor - len(new_bars)}")
        self.renderer.draw_all(self.state)

    def _cmd_rename_section(self):
        """Rename the selected section via dialog."""
        if self.arrangement is None or not self.state.selected or self.state.selected[0] != "section":
            return
        sec_idx = self.state.selected[1]
        if sec_idx >= len(self.arrangement.sections):
            return
        sec = self.arrangement.sections[sec_idx]
        from tkinter import simpledialog
        new_name = simpledialog.askstring("Rename Section", f"New name for '{sec.name}':",
                                          initialvalue=sec.name)
        if new_name and new_name.strip():
            self._push_undo()
            with self.lock:
                sec.name = new_name.strip()
                self.state.dirty = True
            logger.info(f"Renamed section {sec_idx} to '{new_name}'")
            self.renderer.draw_all(self.state)

    def _cmd_delete_section(self):
        """Delete the selected section(s). Refuse if it would remove every section."""
        if self.arrangement is None:
            return
        rng = self._selected_section_range()
        if rng is None:
            return
        lo, hi = rng
        if hi - lo + 1 >= len(self.arrangement.sections):
            logger.warning("Cannot delete all sections in the arrangement")
            return
        self._push_undo()
        with self.lock:
            self.arrangement.sections = [s for i, s in enumerate(self.arrangement.sections)
                                         if not (lo <= i <= hi)]
            self._reindex_arrangement()
            # Clamp section_cursor to valid range
            new_cursor = min(lo, len(self.arrangement.sections) - 1) if self.arrangement.sections else None
            self.state.section_cursor = new_cursor
            self.state.section_selection = None
            self._clamp_play_position()
            self.state.dirty = True
        logger.info(f"Deleted sections {lo}..{hi}")
        self.renderer.draw_all(self.state)

    def _cmd_copy_section(self):
        """Copy the selected section(s) to the clipboard."""
        if self.arrangement is None:
            return
        rng = self._selected_section_range()
        if rng is None:
            return
        lo, hi = rng
        self._clipboard_sections = [copy.deepcopy(s) for s in self.arrangement.sections[lo:hi + 1]]
        self.state.has_section_clipboard = True
        logger.info(f"Copied {len(self._clipboard_sections)} section(s) to clipboard")

    def _cmd_paste_section(self):
        """Paste clipboard sections at the cursor position (before that section)."""
        if self.arrangement is None or not self._clipboard_sections:
            logger.warning("Paste: no arrangement or section clipboard")
            return
        insert_idx = self.state.section_cursor if self.state.section_cursor is not None else len(self.arrangement.sections)
        self._push_undo()
        with self.lock:
            new_secs = [copy.deepcopy(s) for s in self._clipboard_sections]
            self.arrangement.sections[insert_idx:insert_idx] = new_secs
            self._reindex_arrangement()
            self.state.section_cursor = insert_idx + len(new_secs) - 1
            self.state.section_selection = None
            self._clamp_play_position()
            self.state.dirty = True
        logger.info(f"Pasted {len(new_secs)} section(s) at position {insert_idx}")
        self.renderer.draw_all(self.state)

    def _cmd_queue_section(self):
        """Queue the selected section's bars."""
        if self.arrangement is None or not self.state.selected or self.state.selected[0] != "section":
            return
        sec_idx = self.state.selected[1]
        if sec_idx >= len(self.arrangement.sections):
            return
        sec = self.arrangement.sections[sec_idx]
        if sec.bars:
            sec_first = self._section_first_bar_idx(sec)
            bar_indices = list(range(sec_first, sec_first + len(sec.bars)))
            self._queue_bars(bar_indices)
            logger.info(f"Queued section {sec_idx} ({len(bar_indices)} bars)")
            self.renderer.draw_all(self.state)

    # Contextual dispatchers for shared shortcuts
    def _cmd_delete(self):
        """Delete contextually: section if a section is selected, else bar."""
        if self.state.selected and self.state.selected[0] == "section":
            self._cmd_delete_section()
        else:
            self._cmd_delete_bar()

    def _cmd_copy(self):
        """Copy contextually: section if a section is selected, else bar."""
        if self.state.selected and self.state.selected[0] == "section":
            self._cmd_copy_section()
        else:
            self._cmd_copy_bar()

    def _cmd_paste(self):
        """Paste contextually: section if section clipboard non-empty, else bar."""
        if self.state.has_section_clipboard:
            self._cmd_paste_section()
        else:
            self._cmd_paste_bar()

    # ------------------------------------------------------------------ poll loop

    def _beat_offset_in_bar(self, play_bar, frame_idx):
        """Which beat (0-based) within play_bar the read head is on."""
        if play_bar is None or self.arrangement is None:
            return None
        bars = self.arrangement.bars
        if play_bar < 0 or play_bar >= len(bars):
            return None
        bar = bars[play_bar]
        if not bar.beats:
            return None
        beat_frames = [int(round(b.start_ms * self.sr / 1000)) for b in bar.beats]
        off = int(np.searchsorted(beat_frames, frame_idx, side="right") - 1)
        return max(0, min(off, len(bar.beats) - 1))

    def _poll(self):
        if not self.state.playing:
            # Engine reached the end of the arrangement (or was stopped).
            self.root.after_idle(self._playback_finished)
            return
        with self.lock:
            frame_idx = self.state.frame_idx
            play_bar = self.state.play_bar
            dirty = self.state.dirty
            self.state.dirty = False
        if dirty:
            self.renderer.draw_all(self.state)
        beat_offset = self._beat_offset_in_bar(play_bar, frame_idx)
        self.renderer.update_playhead(play_bar, beat_offset, self.state)
        pos = int(frame_idx * 1000 / self.sr)
        self.time_label.config(text=self._time_str(min(pos, int(self.total_ms))))
        self.root.after(50, self._poll)

    def _playback_finished(self):
        if self.engine is not None:
            self.engine.stop()
        self.time_label.config(text=self._fmt(0))
        if self.renderer:
            self.renderer.draw_all(self.state)

    # ------------------------------------------------------------------ playback controls

    def _start_bar(self):
        """Bar to begin playback from: the cursor, else selected section/bar, else 0."""
        if self.state.cursor is not None:
            return self.state.cursor
        if self.state.selected:
            if self.state.selected[0] == "bar":
                return self.state.selected[1]
            elif self.state.selected[0] == "section":
                # Start from first bar of selected section
                sec_idx = self.state.selected[1]
                if sec_idx < len(self.arrangement.sections):
                    sec = self.arrangement.sections[sec_idx]
                    if sec.bars:
                        return self._section_first_bar_idx(sec)
        return 0

    def play(self):
        if self.audio_data is None:
            return
        if self.state.paused:
            self.engine.pause()
            self.root.after(50, self._poll)
            return
        self.engine.stop()
        self.engine.play(from_bar=self._start_bar())
        self.root.after(50, self._poll)

    def pause(self):
        if not self.state.playing:
            self.engine.stop()
            self.engine.play(from_bar=self._start_bar())
            self.root.after(50, self._poll)
            return
        self.engine.pause()

    def stop(self):
        if self.engine is not None:
            self.engine.stop()
        self.root.after_idle(self._playback_finished)

    def quit(self):
        if self.state.dirty:
            result = tk.messagebox.askyesnocancel(
                "Unsaved Changes",
                "Save project before quitting?"
            )
            if result is None:
                return
            if result:
                self._cmd_save()
        self.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()
