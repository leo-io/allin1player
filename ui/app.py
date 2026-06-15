from __future__ import annotations
import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
import soundfile as sf

from domain.models import Arrangement
from file_io.arrangement_store import ArrangementStore
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

        # Audio cache: maps audiosource path -> (audio_data, sr)
        self.audio_cache: dict[str, tuple[np.ndarray, int]] = {}
        self.cache_lock = threading.RLock()

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

        cframe = tk.Frame(content, bg="#1a1a2e")
        cframe.pack(fill="both", expand=True, padx=10, pady=(4, 14))
        self.canvas = tk.Canvas(cframe, bg="#16213e", highlightthickness=0)
        vbar = ttk.Scrollbar(cframe, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.bind("<space>", lambda e: self.pause() if self.state.playing else self.play())
        self.root.bind("<Key-l>", lambda e: self._on_key("l"))
        self.root.bind("<Key-L>", lambda e: self._on_key("l"))
        self.root.bind("<Key-Escape>", lambda e: self._on_key("escape"))
        self.root.bind("<Control-s>", lambda e: self._cmd_save())
        self.root.bind("<Control-S>", lambda e: self._cmd_save())

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

        if self.arrangement is None:
            logger.warning("Canvas clicked but no arrangement loaded")
            return

        bar_idx = self.renderer.bar_at_xy(cx, cy)
        if bar_idx is not None:
            if is_ctrl:
                with self.lock:
                    if self.state.vq is None:
                        self.state.vq = VirtualQueue(
                            bars=[bar_idx],
                            entry_bar=bar_idx,
                            pos=0,
                            active=self.state.playing,
                        )
                    else:
                        self.state.vq.bars.append(bar_idx)
                    self.state.dirty = True
                self.renderer.draw_all(self.state)
            else:
                self.state.selected = ("bar", bar_idx)
                if self.state.playing:
                    with self.lock:
                        self.state.vq = VirtualQueue(
                            bars=[bar_idx],
                            entry_bar=bar_idx,
                            pos=0,
                            active=True,
                        )
                        self.state.dirty = True
                self.renderer.draw_all(self.state)
            return

        sec_idx = self.renderer.section_at_xy(cx, cy)
        if sec_idx is not None:
            self.state.selected = ("section", sec_idx)
            self.renderer.draw_all(self.state)
            return

        self.state.selected = None
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

    # ------------------------------------------------------------------ poll loop

    def _poll(self):
        if not self.state.playing:
            return
        with self.lock:
            frame_idx = self.state.frame_idx
            pos = int(frame_idx * 1000 / self.sr)
            dirty = self.state.dirty
            self.state.dirty = False
        if pos >= self.total_ms:
            self.root.after_idle(self._playback_finished)
            return
        if dirty:
            self.renderer.draw_all(self.state)
        self.renderer.update_playhead(frame_idx, pos, self.state)
        self.time_label.config(text=self._time_str(min(pos, int(self.total_ms))))
        self.root.after(50, self._poll)

    def _playback_finished(self):
        if self.engine is not None:
            self.engine.stop()
        self.time_label.config(text=self._fmt(0))
        if self.renderer:
            self.renderer.draw_all(self.state)

    # ------------------------------------------------------------------ playback controls

    def play(self):
        if self.audio_data is None:
            return
        if self.state.paused:
            self.engine.pause()
            self.root.after(50, self._poll)
            return
        self.engine.stop()
        with self.lock:
            self.state.frame_idx = 0
        self.engine.play()
        self.root.after(50, self._poll)

    def pause(self):
        if not self.state.playing:
            self.engine.stop()
            with self.lock:
                self.state.frame_idx = 0
            self.engine.play()
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
                self._cmd_save_project()
        self.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()
