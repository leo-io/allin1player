#!/usr/bin/env python3
"""
Allin1 Player: plays audio with synchronized beat/bar visualization
from .json files produced by the allin1 music structure analyzer
(beats/downbeats/beat_positions in seconds + labeled segments).
"""

import argparse
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
import sounddevice as sd
import soundfile as sf


def load_structure(path):
    """Build the song hierarchy from an allin1 .json file.

    Returns a list of sections, each:
        {"name": str, "bars": [bar, ...]}
    where each bar is a list of beats:
        [{"time_ms": int, "beat": int}, ...]
    """
    import json
    with open(path) as f:
        data = json.load(f)

    # group the flat beat list into bars (new bar at each beat_position == 1;
    # a leading partial bar is kept if the track doesn't start on a downbeat)
    all_bars = []
    current = []
    for t, num in zip(data["beats"], data["beat_positions"]):
        if num == 1 and current:
            all_bars.append(current)
            current = []
        current.append({"time_ms": int(round(t * 1000)), "beat": int(num)})
    if current:
        all_bars.append(current)

    # distribute bars into labeled segments; drop segments that contain no
    # bars (e.g. the "start"/"end" silence markers)
    sections = []
    label_counts = {}
    for seg in data.get("segments", []):
        seg_start_ms = seg["start"] * 1000
        seg_end_ms = seg["end"] * 1000
        seg_bars = [b for b in all_bars if seg_start_ms <= b[0]["time_ms"] < seg_end_ms]
        if not seg_bars:
            continue
        label = seg["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
        sections.append({"name": f"{label} {label_counts[label]}", "bars": seg_bars})

    if not sections and all_bars:
        sections = [{"name": "track", "bars": all_bars}]

    # keep only the first occurrence number when a label appears once
    for sec in sections:
        label = sec["name"].rsplit(" ", 1)[0]
        if label_counts.get(label, 0) == 1:
            sec["name"] = label

    return sections


def flatten_structure(sections):
    """Derive the flat arrays the player/drawing code consumes.

    Returns (times_ms, numbers, bars, section_ranges, section_names) where
    bars is [(first_beat_idx, beat_numbers_array), ...] and section_ranges
    is [(first_bar_idx, end_bar_idx), ...].
    """
    times_ms = []
    numbers = []
    bars = []
    section_ranges = []
    section_names = []
    for sec in sections:
        sec_bar_start = len(bars)
        for bar in sec["bars"]:
            start_idx = len(times_ms)
            for beat in bar:
                times_ms.append(beat["time_ms"])
                numbers.append(beat["beat"])
            bars.append((start_idx, np.array([b["beat"] for b in bar], dtype=int)))
        section_ranges.append((sec_bar_start, len(bars)))
        section_names.append(sec["name"])
    return (np.array(times_ms, dtype=np.int64), np.array(numbers, dtype=int),
            bars, section_ranges, section_names)


class BeatPlayer:
    BAR_W = 80
    MIN_BEAT_W = 18
    BAR_H = 44
    BAR_GAP = 6
    ROW_GAP = 8
    PAD = 10
    ROW_LABEL_W = 48

    DOWNBEAT = "#ffd700"
    BEAT_OFF = "#3a3a3a"
    BEAT_ON = "#ff6b35"
    BAR_BG = "#1e2a45"
    BAR_ACTIVE = "#2a3a5a"
    QUEUE_OUTLINE = "#44ff44"
    LOOP_OUTLINE = "#ff4444"
    SELECT_OUTLINE = "#66ccff"
    SECTION_HEADER_H = 24

    SIDEBAR_W = 200

    def __init__(self, audio_path=None, beats_path=None):
        self.audio_path = None
        self.audio_data = None
        self.sr = None
        self.total_ms = 0
        self.beat_times_ms = np.array([], dtype=np.int64)
        self.beat_numbers = np.array([], dtype=int)
        self.structure = []
        self.bars = []
        self.sections = []
        self.section_names = []
        self.n_bars = 0

        self.stream = None
        self.frame_idx = 0
        self.playing = False
        self.paused = False
        self.lock = threading.Lock()
        self.queue = []
        self.loop_range = None
        self.selected = None
        self.bar_frames = np.array([], dtype=np.int64)
        self._dirty = False

        self.audio_files = sorted(
            list(Path(".").glob("*.mp3")) + list(Path(".").glob("*.wav"))
        )

        self._build_ui()

        if audio_path:
            self.load_file(audio_path, beats_path)
        elif self.audio_files:
            self.load_file(str(self.audio_files[0]))

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

        self.structure = load_structure(str(bp))
        (self.beat_times_ms, self.beat_numbers, self.bars,
         self.sections, self.section_names) = flatten_structure(self.structure)
        self.n_bars = len(self.bars)

        # frame position of each bar's first beat, plus end-of-audio sentinel,
        # so the audio callback can jump at exact sample boundaries
        self.bar_frames = np.array(
            [int(round(self.beat_times_ms[s] * self.sr / 1000)) for s, _ in self.bars]
            + [len(self.audio_data)],
            dtype=np.int64,
        )

        self.frame_idx = 0
        self.playing = False
        self.paused = False
        self.queue.clear()
        self.loop_range = None
        self.selected = None
        self._dirty = False

        self.audio_path = p
        self.root.title(f"Allin1 Player — {p.name}")
        self.filename_label.config(text=p.name)
        self.time_label.config(text=self._fmt(0))
        self.bar_var.set("--")
        self.beat_var.set("--")
        self.play_btn.config(state="normal", text="Play")
        self.pause_btn.config(state="disabled", text="Pause")
        self.stop_btn.config(state="disabled")

        for i, f in enumerate(self.audio_files):
            if f.name == p.name:
                self.sidebar_list.selection_clear(0, tk.END)
                self.sidebar_list.selection_set(i)
                self.sidebar_list.see(i)
                break

        self._draw_all_bars()
        self._update_ui(0)

    def _on_sidebar_select(self, event):
        sel = self.sidebar_list.curselection()
        if sel:
            idx = sel[0]
            self.load_file(str(self.audio_files[idx]))

    def _on_canvas_click(self, event):
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        cid = self.canvas.find_closest(cx, cy)
        if not cid:
            self.selected = None
            self._draw_all_bars()
            if self.audio_data is not None:
                self._update_ui(self._current_ms())
            return
        cid = cid[0]

        for sec_item in self.section_items:
            if sec_item["bg"] == cid or sec_item["text"] == cid:
                self.selected = ("section", sec_item["sec_idx"])
                self._draw_all_bars()
                if self.audio_data is not None:
                    self._update_ui(self._current_ms())
                return

        for item in self.bar_items:
            if item["bg"] == cid or any(r == cid for r, t in item["beats"]):
                bi = item["bar_idx"]
                self.selected = ("bar", bi)
                if self.playing:
                    with self.lock:
                        self.queue = [bi]
                self._draw_all_bars()
                if self.audio_data is not None:
                    self._update_ui(self._current_ms())
                return

    def _bar_outline(self, bar_idx):
        if self.loop_range and self.loop_range[0] <= bar_idx < self.loop_range[1]:
            return (self.LOOP_OUTLINE, 2)
        if self.selected == ("bar", bar_idx):
            return (self.SELECT_OUTLINE, 2)
        if bar_idx in self.queue:
            return (self.QUEUE_OUTLINE, 2)
        return ("#334", 1)

    def _toggle_loop_selected(self):
        if not self.selected:
            return
        if self.selected[0] == "bar":
            bi = self.selected[1]
            target_range = (bi, bi + 1)
        elif self.selected[0] == "section":
            si = self.selected[1]
            target_range = self.sections[si]
        else:
            return
        with self.lock:
            self.loop_range = None if self.loop_range == target_range else target_range
        self._draw_all_bars()
        if self.audio_data is not None:
            self._update_ui(self._current_ms())

    def _toggle_queue_selected(self):
        if not self.selected:
            return
        if self.selected[0] == "bar":
            bi = self.selected[1]
            with self.lock:
                if bi in self.queue:
                    self.queue.remove(bi)
                else:
                    self.queue.append(bi)
        elif self.selected[0] == "section":
            si = self.selected[1]
            start, end = self.sections[si]
            bars_in_section = list(range(start, end))
            with self.lock:
                if all(b in self.queue for b in bars_in_section):
                    for b in bars_in_section:
                        self.queue.remove(b)
                else:
                    for b in bars_in_section:
                        if b not in self.queue:
                            self.queue.append(b)
        self._draw_all_bars()
        if self.audio_data is not None:
            self._update_ui(self._current_ms())

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("Allin1 Player")
        self.root.configure(bg="#1a1a2e")
        self.root.minsize(1000, 500)

        main_frame = tk.Frame(self.root, bg="#1a1a2e")
        main_frame.pack(fill="both", expand=True)

        # --- left sidebar ---
        sidebar = tk.Frame(main_frame, bg="#0f0f23", width=self.SIDEBAR_W)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="TRACKS", font=("Segoe UI", 10, "bold"),
                 fg="#888", bg="#0f0f23").pack(pady=(12, 4))

        self.sidebar_list = tk.Listbox(
            sidebar, bg="#0f0f23", fg="#c0c0c0",
            selectbackground="#2a3a5a", selectforeground="#ffd700",
            font=("Segoe UI", 10), borderwidth=0, highlightthickness=0,
            activestyle="none"
        )
        self.sidebar_list.pack(fill="both", expand=True, padx=6, pady=(0, 8))
        self.sidebar_list.bind("<<ListboxSelect>>", self._on_sidebar_select)

        for f in self.audio_files:
            self.sidebar_list.insert(tk.END, f.name)

        # --- main content ---
        content = tk.Frame(main_frame, bg="#1a1a2e")
        content.pack(side="left", fill="both", expand=True)

        top = tk.Frame(content, bg="#1a1a2e")
        top.pack(fill="x", padx=16, pady=(14, 4))
        self.filename_label = tk.Label(top, font=("Segoe UI", 11, "bold"),
                                       fg="#e0e0e0", bg="#1a1a2e")
        self.filename_label.pack(side="left")
        self.time_label = tk.Label(top, text=self._fmt(0), font=("Segoe UI", 11),
                                   fg="#aaaaaa", bg="#1a1a2e")
        self.time_label.pack(side="right")

        digits = tk.Frame(content, bg="#1a1a2e")
        digits.pack(pady=(2, 2))
        tk.Label(digits, text="BAR", font=("Segoe UI", 9), fg="#888",
                 bg="#1a1a2e").grid(row=0, column=0, padx=(0, 16))
        tk.Label(digits, text="BEAT", font=("Segoe UI", 9), fg="#888",
                 bg="#1a1a2e").grid(row=0, column=1, padx=(16, 0))
        self.bar_var = tk.StringVar(value="--")
        self.beat_var = tk.StringVar(value="--")
        tk.Label(digits, textvariable=self.bar_var, font=("Segoe UI", 36, "bold"),
                 fg="#ffd700", bg="#1a1a2e").grid(row=1, column=0, padx=(0, 16))
        tk.Label(digits, textvariable=self.beat_var, font=("Segoe UI", 36, "bold"),
                 fg="#ff6b35", bg="#1a1a2e").grid(row=1, column=1, padx=(16, 0))

        cframe = tk.Frame(content, bg="#1a1a2e")
        cframe.pack(fill="both", expand=True, padx=10, pady=(4, 0))
        self.canvas = tk.Canvas(cframe, bg="#16213e", highlightthickness=0)
        vbar = ttk.Scrollbar(cframe, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.bar_items = []

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
        self.root.bind("<space>", lambda e: self.pause() if self.playing else self.play())
        self.root.bind("<Key-l>", lambda e: self._toggle_loop_selected())
        self.root.bind("<Key-L>", lambda e: self._toggle_loop_selected())
        self.root.bind("<Key-q>", lambda e: self._toggle_queue_selected())
        self.root.bind("<Key-Q>", lambda e: self._toggle_queue_selected())

    # ------------------------------------------------------------------ bar drawing

    def _bars_per_row(self, cw):
        if cw < self.ROW_LABEL_W + 60 + self.PAD * 2:
            return 1
        available = cw - self.ROW_LABEL_W - self.PAD * 2
        return max(1, int(available // (self.BAR_W + self.BAR_GAP)))

    def _draw_all_bars(self):
        self.canvas.delete("all")
        self.bar_items = []
        self.section_items = []
        cw = max(self.canvas.winfo_width(), 600)
        bpr = self._bars_per_row(cw)
        actual_avail = cw - self.PAD * 2 - self.ROW_LABEL_W - self.BAR_GAP * (bpr - 1)
        actual_bar_w = max(50, actual_avail // bpr)

        y = self.PAD
        for sec_idx, (sec_start, sec_end) in enumerate(self.sections):
            sec_is_selected = self.selected == ("section", sec_idx)
            sec_is_looping = self.loop_range == (sec_start, sec_end)
            header_text = f"{self.section_names[sec_idx].upper()} — BARS {sec_start + 1}–{sec_end}"
            header_bg_color = "#2a3a5a" if (sec_is_selected or sec_is_looping) else "#1a2a40"
            header_text_color = "#ffd700" if sec_is_looping else "#aaa" if sec_is_selected else "#666"

            header_bg = self.canvas.create_rectangle(
                self.PAD, y, cw - self.PAD, y + self.SECTION_HEADER_H,
                fill=header_bg_color, outline="#445", width=1
            )
            header_text_id = self.canvas.create_text(
                self.PAD + 10, y + self.SECTION_HEADER_H // 2,
                text=header_text, fill=header_text_color, font=("Segoe UI", 9),
                anchor="w"
            )
            self.section_items.append({
                "bg": header_bg,
                "text": header_text_id,
                "sec_idx": sec_idx,
                "range": (sec_start, sec_end),
            })
            y += self.SECTION_HEADER_H + 4

            for row_offset in range(0, sec_end - sec_start, bpr):
                row_bars_indices = list(range(sec_start + row_offset, min(sec_start + row_offset + bpr, sec_end)))
                row_bars = [self.bars[i] for i in row_bars_indices]
                x = self.PAD + self.ROW_LABEL_W

                self.canvas.create_text(
                    self.PAD + self.ROW_LABEL_W // 2, y + self.BAR_H // 2,
                    text=str(row_bars_indices[0] + 1), fill="#666", font=("Segoe UI", 9),
                    anchor="center"
                )

                for bi_in_row, bar_idx in enumerate(row_bars_indices):
                    start_idx, bar_beats = self.bars[bar_idx]
                    nb = len(bar_beats)
                    beat_w = max(self.MIN_BEAT_W, int((actual_bar_w - 2) / nb))
                    bw = beat_w * nb + 2
                    rx, ry = x, y

                    outline, width = self._bar_outline(bar_idx)
                    bg = self.canvas.create_rectangle(
                        rx, ry, rx + bw, ry + self.BAR_H,
                        fill=self.BAR_BG, outline=outline, width=width
                    )

                    beat_rects = []
                    for bii, bn in enumerate(bar_beats):
                        bx = rx + 1 + bii * beat_w
                        bw2 = max(1, beat_w - 1)
                        bh = self.BAR_H - 4
                        is_db = bn == 1
                        fill = "#665533" if is_db else self.BEAT_OFF
                        r = self.canvas.create_rectangle(
                            bx, ry + 2, bx + bw2, ry + 2 + bh,
                            fill=fill, outline="#445" if is_db else "#2a2a2a", width=1
                        )
                        t = None
                        if is_db and beat_w >= 14:
                            t = self.canvas.create_text(
                                bx + bw2 // 2, ry + 2 + bh // 2,
                                text=str(bn), fill="#ddd", font=("Segoe UI", 7)
                            )
                        beat_rects.append((r, t))

                    self.bar_items.append({
                        "bg": bg,
                        "beats": beat_rects,
                        "start_idx": start_idx,
                        "bar_idx": bar_idx,
                        "bar_num": bar_idx + 1,
                        "n_beats": nb,
                    })
                    x += bw + self.BAR_GAP
                y += self._row_total_h

        self.canvas.configure(scrollregion=(0, 0, cw, y + self.PAD))

    @property
    def _row_total_h(self):
        return self.BAR_H + self.ROW_GAP

    def _on_canvas_resize(self, event):
        if not hasattr(self, 'bar_items'):
            return
        self._draw_all_bars()
        if self.audio_data is not None:
            self._update_ui(self._current_ms())

    # ------------------------------------------------------------------ display updates

    @staticmethod
    def _fmt(ms):
        m, s = divmod(ms // 1000, 60)
        return f"{m}:{s:02d}"

    def _time_str(self, ms):
        total = self.total_ms
        m, s = divmod(ms // 1000, 60)
        tm, ts = divmod(int(total) // 1000, 60)
        return f"{m}:{s:02d} / {tm}:{ts:02d}"

    def _current_ms(self):
        return int(self.frame_idx * 1000 / self.sr) if self.sr else 0

    def _update_ui(self, ms):
        self.time_label.config(text=self._time_str(min(ms, int(self.total_ms))))

        idx = np.searchsorted(self.beat_times_ms, ms, side="right") - 1
        if idx < 0 or idx >= len(self.beat_times_ms):
            self.bar_var.set("--")
            self.beat_var.set("--")
            for item in self.bar_items:
                bi = item["bar_idx"]
                outline, width = self._bar_outline(bi)
                self.canvas.itemconfig(item["bg"], fill=self.BAR_BG, outline=outline, width=width)
                for bii, (r, t) in enumerate(item["beats"]):
                    is_db = self.beat_numbers[item["start_idx"] + bii] == 1
                    self.canvas.itemconfig(r, fill="#665533" if is_db else self.BEAT_OFF)
            return

        bnum = self.beat_numbers[idx]
        current_item = None
        for item in self.bar_items:
            si = item["start_idx"]
            if si <= idx < si + item["n_beats"]:
                current_item = item
                break

        self.bar_var.set(str(current_item["bar_num"]) if current_item else "--")
        self.beat_var.set(str(bnum))

        for item in self.bar_items:
            si = item["start_idx"]
            bi = item["bar_idx"]
            if item is current_item:
                self.canvas.itemconfig(item["bg"], fill=self.BAR_ACTIVE, outline=self.DOWNBEAT, width=2)
                beat_offset = idx - si
                for bii, (r, t) in enumerate(item["beats"]):
                    is_db = self.beat_numbers[si + bii] == 1
                    if bii == beat_offset:
                        fill = self.DOWNBEAT if is_db else self.BEAT_ON
                    else:
                        fill = "#665533" if is_db else self.BEAT_OFF
                    self.canvas.itemconfig(r, fill=fill)
            else:
                outline, width = self._bar_outline(bi)
                self.canvas.itemconfig(item["bg"], fill=self.BAR_BG, outline=outline, width=width)
                for bii, (r, t) in enumerate(item["beats"]):
                    is_db = self.beat_numbers[si + bii] == 1
                    self.canvas.itemconfig(r, fill="#665533" if is_db else self.BEAT_OFF)

        if current_item:
            bbox = self.canvas.bbox(current_item["bg"])
            if bbox:
                _, y0, _, y1 = bbox
                vh = self.canvas.winfo_height()
                yview = self.canvas.yview()
                total_h = self.canvas.bbox("all")[3] if self.canvas.bbox("all") else 1
                vis_y0 = yview[0] * total_h
                vis_y1 = yview[1] * total_h
                if y1 > vis_y1 - 10 or y0 < vis_y0 + 10:
                    target = max(0, (y0 - vh // 3) / total_h)
                    self.canvas.yview_moveto(target)

    # ------------------------------------------------------------------ playback

    def _next_jump(self):
        """Next frame boundary where playback must jump, and where it lands.

        Returns (boundary_frame, target_frame, is_queue) or (None, None, False).
        Caller must hold self.lock.
        """
        pos = self.frame_idx
        if self.loop_range is not None:
            start = int(self.bar_frames[self.loop_range[0]])
            end = int(self.bar_frames[self.loop_range[1]])
            if pos < end and start < end:
                return end, start, False
        if self.queue:
            b = int(np.searchsorted(self.bar_frames, pos, side="right"))
            if b < len(self.bar_frames):
                return int(self.bar_frames[b]), None, True
        return None, None, False

    def _audio_callback(self, outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        with self.lock:
            if not self.playing:
                outdata[:] = 0
                return
            filled = 0
            while filled < frames:
                boundary, target, is_queue = self._next_jump()
                limit = boundary if boundary is not None else len(self.audio_data)
                take = min(frames - filled, limit - self.frame_idx)
                if take > 0:
                    outdata[filled:filled + take] = \
                        self.audio_data[self.frame_idx:self.frame_idx + take]
                    self.frame_idx += take
                    filled += take
                if self.frame_idx >= limit:
                    if boundary is None:
                        outdata[filled:] = 0
                        raise sd.CallbackStop()
                    if is_queue:
                        target = int(self.bar_frames[self.queue.pop(0)])
                        self._dirty = True
                    self.frame_idx = target

    def _poll(self):
        if not self.playing:
            return
        with self.lock:
            pos = int(self.frame_idx * 1000 / self.sr)
            dirty = self._dirty
            self._dirty = False
        if pos >= self.total_ms:
            self.root.after_idle(self._playback_finished)
            return
        if dirty:
            self._draw_all_bars()
        self._update_ui(pos)
        self.root.after(50, self._poll)

    def _playback_finished(self):
        if self.stream is not None:
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.playing = False
        self.paused = False
        with self.lock:
            self.frame_idx = 0
        self.play_btn.config(state="normal", text="Play")
        self.pause_btn.config(state="disabled", text="Pause")
        self.stop_btn.config(state="disabled")
        self._update_ui(0)

    def play(self):
        if self.audio_data is None:
            return
        if self.paused:
            return
        self._playback_finished()
        self.playing = True
        self.stream = sd.OutputStream(
            samplerate=self.sr,
            channels=1,
            callback=self._audio_callback,
            blocksize=1024,
        )
        self.stream.start()
        self.play_btn.config(state="disabled")
        self.pause_btn.config(state="normal")
        self.stop_btn.config(state="normal")
        self.root.after(50, self._poll)

    def pause(self):
        if not self.playing:
            if self.stream is not None:
                try:
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None
            self.play()
            return
        if self.paused:
            self.paused = False
            try:
                self.stream.start()
            except sd.PortAudioError:
                pass
            self.pause_btn.config(text="Pause")
            self.root.after(50, self._poll)
        else:
            try:
                self.stream.stop()
            except sd.PortAudioError:
                pass
            self.paused = True
            self.pause_btn.config(text="Resume")

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.abort()
            except Exception:
                pass
            self.stream = None
        self.playing = False
        self.root.after_idle(self._playback_finished)

    def quit(self):
        self.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="Play audio with allin1 .json bar/beat visualization")
    ap.add_argument("audio", type=str, nargs="?", default=None,
                    help="Audio file (optional — shows sidebar with all .mp3/.wav files)")
    ap.add_argument("--json", "-j", type=str, default=None,
                    help="allin1 .json file (default: same name with .json extension)")
    args = ap.parse_args()

    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            sys.exit(f"Audio file not found: {audio_path}")
        json_path = Path(args.json) if args.json else audio_path.with_suffix(".json")
        if not json_path.exists():
            sys.exit(f"JSON file not found: {json_path}")
        BeatPlayer(audio_path=audio_path, beats_path=json_path).run()
    else:
        BeatPlayer().run()


if __name__ == "__main__":
    main()
