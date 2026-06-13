from __future__ import annotations
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
import soundfile as sf

from domain.models import ArrangementDocument, SongStructure, Command
from file_io.project_store import ProjectStore
from file_io.chord_loader import ChordLoader
from playback.engine import PlaybackEngine
from playback.state import TransportState, VirtualQueue
from ui.renderer import CanvasRenderer


class BeatPlayer:
    def __init__(self, audio_path=None, beats_path=None):
        self.audio_path = None
        self.audio_data = None
        self.sr = None
        self.total_ms = 0

        self.lock = threading.RLock()
        self.state = TransportState()
        self.song: SongStructure | None = None
        self.doc: ArrangementDocument | None = None
        self.engine = None
        self.renderer = None
        self.show_chords = True
        self.undo_stack: list[Command] = []
        self.redo_stack: list[Command] = []

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
        self.root.title("Allin1 Player")
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

        version_menu = tk.Menu(self.menu_bar, tearoff=0, bg="#1a1a2e", fg="#e0e0e0")
        version_menu.add_command(label="Save Project", command=self._cmd_save_project)
        version_menu.add_command(label="New Version", command=self._cmd_new_version)
        version_menu.add_command(label="Rename Version", command=self._cmd_rename_version)
        version_menu.add_separator()
        version_menu.add_command(label="Undo", command=self._cmd_undo)
        version_menu.add_command(label="Redo", command=self._cmd_redo)
        version_menu.add_separator()
        self.show_chords_var = tk.BooleanVar(value=True)
        version_menu.add_checkbutton(label="Show Chords", variable=self.show_chords_var,
                                     command=self._toggle_chords)
        self.menu_bar.add_cascade(label="Version", menu=version_menu)

        cframe = tk.Frame(content, bg="#1a1a2e")
        cframe.pack(fill="both", expand=True, padx=10, pady=(4, 0))
        self.canvas = tk.Canvas(cframe, bg="#16213e", highlightthickness=0)
        vbar = ttk.Scrollbar(cframe, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        ctrl = tk.Frame(content, bg="#1a1a2e")
        ctrl.pack(pady=(10, 14))
        self.play_btn = tk.Button(ctrl, text="Play", width=8, command=self.play)
        self.play_btn.pack(side="left", padx=4)
        self.pause_btn = tk.Button(ctrl, text="Pause", width=8, command=self.pause,
                                   state="disabled")
        self.pause_btn.pack(side="left", padx=4)
        self.stop_btn = tk.Button(ctrl, text="Stop", width=8, command=self.stop,
                                  state="disabled")
        self.stop_btn.pack(side="left", padx=4)
        tk.Button(ctrl, text="Quit", width=8, command=self.quit).pack(side="left", padx=4)

        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.bind("<space>", lambda e: self.pause() if self.state.playing else self.play())
        self.root.bind("<Key-l>", lambda e: self._on_key("l"))
        self.root.bind("<Key-L>", lambda e: self._on_key("l"))
        self.root.bind("<Key-Escape>", lambda e: self._on_key("escape"))
        self.root.bind("<Key-c>", lambda e: self._on_key("c"))
        self.root.bind("<Key-C>", lambda e: self._on_key("c"))
        self.root.bind("<Control-z>", lambda e: self._cmd_undo())
        self.root.bind("<Control-Z>", lambda e: self._cmd_undo())
        self.root.bind("<Control-y>", lambda e: self._cmd_redo())
        self.root.bind("<Control-Y>", lambda e: self._cmd_redo())
        self.root.bind("<Control-s>", lambda e: self._cmd_save_project())
        self.root.bind("<Control-S>", lambda e: self._cmd_save_project())

    # ------------------------------------------------------------------ command layer

    def _execute(self, cmd: Command):
        self.undo_stack.append(cmd)
        self.redo_stack.clear()
        self._apply(cmd)
        self.state.dirty = True

    def _apply(self, cmd: Command):
        doc = self.doc
        if doc is None:
            return
        v = doc.active_version
        if v is None:
            return

        if cmd.type == "rename_section":
            sec_idx = cmd.params["section_idx"]
            new_name = cmd.params["name"]
            if sec_idx < len(doc.source.sections):
                old = doc.source.sections[sec_idx]
                doc.source.sections[sec_idx] = type(old)(
                    idx=old.idx, name=new_name,
                    first_bar=old.first_bar, end_bar=old.end_bar
                )
        elif cmd.type == "set_active_version":
            doc.active_version_idx = cmd.params["version_idx"]
        elif cmd.type == "rename_version":
            if cmd.params["version_idx"] < len(doc.versions):
                old = doc.versions[cmd.params["version_idx"]]
                doc.versions[cmd.params["version_idx"]] = type(old)(
                    name=cmd.params["name"],
                    section_ordering=old.section_ordering,
                )

    def _cmd_undo(self):
        if not self.undo_stack:
            return
        cmd = self.undo_stack.pop()
        self.redo_stack.append(cmd)
        self._rebuild_from_doc()
        self.state.dirty = True

    def _cmd_redo(self):
        if not self.redo_stack:
            return
        cmd = self.redo_stack.pop()
        self.undo_stack.append(cmd)
        self._apply(cmd)
        self._rebuild_from_doc()
        self.state.dirty = True

    def _rebuild_from_doc(self):
        if self.doc is None:
            return
        self.song = self.doc.source
        self.renderer = CanvasRenderer(self.canvas, self.song)
        self.renderer.draw_all(self.state, self.show_chords)

    def _cmd_save_project(self):
        if self.doc is None or self.audio_path is None:
            return
        ProjectStore.save(self.doc, self.audio_path)
        self.state.dirty = False

    def _cmd_new_version(self):
        if self.doc is None:
            return
        count = len(self.doc.versions)
        self.doc.add_version(f"Version {count + 1}")
        self._execute(Command("set_active_version", {"version_idx": self.doc.active_version_idx}))
        self._rebuild_from_doc()

    def _cmd_rename_version(self):
        if self.doc is None:
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("Rename Version")
        dialog.configure(bg="#1a1a2e")
        tk.Label(dialog, text="New name:", bg="#1a1a2e", fg="#e0e0e0").pack(padx=10, pady=5)
        entry = tk.Entry(dialog, width=30)
        entry.pack(padx=10, pady=5)
        entry.focus()

        def do_rename():
            name = entry.get().strip()
            if name and self.doc:
                self._execute(Command("rename_version", {
                    "version_idx": self.doc.active_version_idx,
                    "name": name,
                }))
                self._rebuild_from_doc()
            dialog.destroy()

        tk.Button(dialog, text="OK", command=do_rename).pack(pady=5)

    def _toggle_chords(self):
        self.show_chords = self.show_chords_var.get()
        if self.renderer is not None:
            self.renderer.draw_all(self.state, self.show_chords)

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

        self.doc, self.song = ProjectStore.load_or_create(str(p), str(bp))

        cp = p.with_suffix(".madmom.chords.txt")
        self.song.chords = ChordLoader.load(str(cp), self.song)
        self.song.beat_chords = ChordLoader.compute_beat_chords(
            self.song.chords, self.song.beat_times_ms,
        )

        self.song.bar_frames = np.array(
            [int(round(self.song.beat_times_ms[b.start_beat_idx] * self.sr / 1000))
             for b in self.song.bars]
            + [len(self.audio_data)],
            dtype=np.int64,
        )
        self.song.total_frames = len(self.audio_data)

        with self.lock:
            self.state = TransportState()

        self.engine = PlaybackEngine(self.audio_data, self.song, self.state, self.lock, self.sr)
        self.renderer = CanvasRenderer(self.canvas, self.song)

        self.audio_path = p
        self.root.title(f"Allin1 Player — {p.name}")
        self.filename_label.config(text=p.name)
        self.time_label.config(text=self._fmt(0))
        self.play_btn.config(state="normal", text="Play")
        self.pause_btn.config(state="disabled", text="Pause")
        self.stop_btn.config(state="disabled")

        self.renderer.draw_all(self.state, self.show_chords)

    # ------------------------------------------------------------------ event handlers

    def _on_canvas_click(self, event):
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        is_ctrl = bool(event.state & 0x4)

        if self.song is None:
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
                self.renderer.draw_all(self.state, self.show_chords)
            else:
                self.state.selected = ("bar", bar_idx)
                self.renderer.draw_all(self.state, self.show_chords)
            return

        sec_idx = self.renderer.section_at_xy(cx, cy)
        if sec_idx is not None:
            self.state.selected = ("section", sec_idx)
            self.renderer.draw_all(self.state, self.show_chords)
            return

        self.state.selected = None
        self.renderer.draw_all(self.state, self.show_chords)

    def _on_key(self, key):
        if key == "l":
            vq = self.state.vq
            if vq is not None and not vq.is_empty():
                with self.lock:
                    vq.looping = not vq.looping
                    self.state.dirty = True
                self.renderer.draw_all(self.state, self.show_chords)
                return
            if self.state.selected and self.song is not None:
                with self.lock:
                    if self.state.selected[0] == "bar":
                        bi = self.state.selected[1]
                        target_range = (bi, bi + 1)
                    elif self.state.selected[0] == "section":
                        si = self.state.selected[1]
                        sec = self.song.sections[si]
                        target_range = (sec.first_bar, sec.end_bar)
                    else:
                        return
                    self.state.loop_range = None if self.state.loop_range == target_range else target_range
                self.renderer.draw_all(self.state, self.show_chords)
        elif key == "c":
            self.show_chords_var.set(not self.show_chords_var.get())
            self._toggle_chords()
        elif key == "escape":
            with self.lock:
                self.state.vq = None
                self.state.dirty = True
            if self.song is not None:
                self.renderer.draw_all(self.state, self.show_chords)

    def _on_resize(self, event):
        if self.song is None or self.renderer is None:
            return
        self.renderer.draw_all(self.state, self.show_chords)

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
            self.renderer.draw_all(self.state, self.show_chords)
        self.renderer.update_playhead(frame_idx, pos, self.state)
        self.time_label.config(text=self._time_str(min(pos, int(self.total_ms))))
        self.root.after(50, self._poll)

    def _playback_finished(self):
        if self.engine is not None:
            self.engine.stop()
        self.play_btn.config(state="normal", text="Play")
        self.pause_btn.config(state="disabled", text="Pause")
        self.stop_btn.config(state="disabled")
        self.time_label.config(text=self._fmt(0))
        if self.renderer:
            self.renderer.draw_all(self.state, self.show_chords)

    # ------------------------------------------------------------------ playback controls

    def play(self):
        if self.audio_data is None:
            return
        if self.state.paused:
            self.engine.pause()
            self.pause_btn.config(text="Pause")
            self.root.after(50, self._poll)
            return
        self.engine.stop()
        with self.lock:
            self.state.frame_idx = 0
        self.engine.play()
        self.play_btn.config(state="disabled")
        self.pause_btn.config(state="normal", text="Pause")
        self.stop_btn.config(state="normal")
        self.root.after(50, self._poll)

    def pause(self):
        if not self.state.playing:
            self.engine.stop()
            with self.lock:
                self.state.frame_idx = 0
            self.engine.play()
            self.play_btn.config(state="disabled")
            self.pause_btn.config(state="normal", text="Pause")
            self.stop_btn.config(state="normal")
            self.root.after(50, self._poll)
            return
        self.engine.pause()
        if self.state.paused:
            self.pause_btn.config(text="Resume")
        else:
            self.pause_btn.config(text="Pause")

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
