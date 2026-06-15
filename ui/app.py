from __future__ import annotations
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


class ArrangementEditor:
    def __init__(self, audio_path=None, beats_path=None):
        self.audio_path = None
        self.audio_data = None
        self.sr = None
        self.total_ms = 0

        self.lock = threading.RLock()
        self.state = TransportState()
        self.arrangement: Arrangement | None = None
        self.engine = None
        self.renderer = None

        self.audio_files = sorted(
            list(Path(".").glob("*.mp3")) + list(Path(".").glob("*.wav"))
        )

        self._build_ui()

        if audio_path:
            self.load_file(audio_path, beats_path)
        elif self.audio_files:
            self.load_file(str(self.audio_files[0]))

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

    # ------------------------------------------------------------------ commands

    def _cmd_save(self):
        if self.arrangement is None or self.audio_path is None:
            return
        ArrangementStore.save(self.arrangement, self.audio_path)
        self.state.dirty = False

    # ------------------------------------------------------------------ file loading

    def load_file(self, audio_path, beats_path=None):
        self.stop()

        p = Path(audio_path)
        if not p.exists():
            return
        bp = Path(beats_path) if beats_path else p.with_suffix(".json")
        if not bp.exists():
            return

        try:
            raw, self.sr = sf.read(str(p), always_2d=True)
        except Exception as e:
            import sys
            sys.stderr.write(
                f"Could not decode {p.name}: {e}\n"
                "MP3 support requires soundfile >= 0.12 (libsndfile >= 1.1): "
                "pip install -U soundfile\n"
            )
            return
        if raw.shape[1] > 1:
            raw = raw.mean(axis=1, keepdims=True)
        self.audio_data = raw.astype(np.float32)
        self.total_ms = len(self.audio_data) * 1000 / self.sr

        self.arrangement = ArrangementStore.load_or_create(str(p), str(bp))
        self.arrangement.reindex(self.sr)
        self.arrangement.set_bar_frames(self.sr, len(self.audio_data))

        with self.lock:
            self.state = TransportState()

        self.engine = PlaybackEngine(self.audio_data, self.arrangement, self.state, self.lock, self.sr)
        self.renderer = CanvasRenderer(self.canvas, self.arrangement)

        self.audio_path = p
        self.root.title(f"Arrangement Editor — {p.name}")
        self.filename_label.config(text=p.name)
        self.time_label.config(text=self._fmt(0))

        self.renderer.draw_all(self.state)

    # ------------------------------------------------------------------ event handlers

    def _on_canvas_click(self, event):
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        is_ctrl = bool(event.state & 0x4)

        if self.arrangement is None:
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
