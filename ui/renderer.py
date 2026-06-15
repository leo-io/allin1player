from __future__ import annotations
import tkinter as tk
import numpy as np

from domain.models import Arrangement
from playback.state import TransportState


DOWNBEAT = "#ffd700"
BEAT_OFF = "#3a3a3a"
BEAT_ON = "#ff6b35"
BAR_BG = "#1e2a45"
BAR_ACTIVE = "#2a3a5a"
QUEUE_OUTLINE = "#44ff44"
LOOP_OUTLINE = "#ff4444"
SELECT_OUTLINE = "#66ccff"
VQUEUE_BG = "#2d1b4e"
VQUEUE_OUTLINE = "#a78bfa"
SECTION_HEADER_H = 24
BAR_W = 80
MIN_BEAT_W = 18
BAR_H = 44
BAR_GAP = 6
ROW_GAP = 8
PAD = 10
ROW_LABEL_W = 48


class CanvasRenderer:
    def __init__(self, canvas: tk.Canvas, arrangement: Arrangement):
        self.canvas = canvas
        self.arrangement = arrangement
        self.bar_items = []
        self.section_items = []

    @property
    def _row_total_h(self):
        return BAR_H + ROW_GAP

    def _bars_per_row(self, cw):
        if cw < ROW_LABEL_W + 60 + PAD * 2:
            return 1
        available = cw - ROW_LABEL_W - PAD * 2
        return max(1, int(available // (BAR_W + BAR_GAP)))

    def _bar_bg_color(self, bar_idx: int, state: TransportState) -> str:
        vq = state.vq
        if vq is not None and bar_idx in vq.bars:
            return VQUEUE_BG
        return BAR_BG

    def _bar_outline(self, bar_idx: int, state: TransportState) -> tuple:
        if state.loop_range and state.loop_range[0] <= bar_idx < state.loop_range[1]:
            return (LOOP_OUTLINE, 2)
        if state.selected == ("bar", bar_idx):
            return (SELECT_OUTLINE, 2)
        vq = state.vq
        if vq is not None and bar_idx in vq.bars:
            return (VQUEUE_OUTLINE, 2)
        return ("#334", 1)

    def _queue_label(self, bar_idx: int, vq) -> str:
        if vq is None:
            return ""
        positions = vq.positions_of(bar_idx)
        if not positions:
            return ""
        return "".join(f"[{p}]" for p in positions)

    def draw_all(self, state: TransportState):
        self.canvas.delete("all")
        self.bar_items = []
        self.section_items = []
        cw = max(self.canvas.winfo_width(), 600)
        bpr = self._bars_per_row(cw)
        actual_avail = cw - PAD * 2 - ROW_LABEL_W - BAR_GAP * (bpr - 1)
        actual_bar_w = max(50, actual_avail // bpr)

        bars_flat = self.arrangement.bars
        y = PAD
        for sec in self.arrangement.sections:
            sec_first_bar_idx = bars_flat.index(sec.bars[0]) if sec.bars else 0
            sec_end_bar_idx = sec_first_bar_idx + len(sec.bars)

            sec_is_selected = state.selected == ("section", sec.idx)
            sec_is_looping = state.loop_range == (sec_first_bar_idx, sec_end_bar_idx)
            header_text = f"{sec.name.upper()} [{len(sec.bars)}]"
            header_bg_color = "#2a3a5a" if (sec_is_selected or sec_is_looping) else "#1a2a40"
            header_text_color = "#ffd700" if sec_is_looping else "#aaa" if sec_is_selected else "#666"

            header_bg = self.canvas.create_rectangle(
                PAD, y, cw - PAD, y + SECTION_HEADER_H,
                fill=header_bg_color, outline="#445", width=1
            )
            header_text_id = self.canvas.create_text(
                PAD + 10, y + SECTION_HEADER_H // 2,
                text=header_text, fill=header_text_color, font=("Segoe UI", 9),
                anchor="w"
            )
            self.section_items.append({
                "bg": header_bg,
                "text": header_text_id,
                "sec_idx": sec.idx,
                "range": (sec_first_bar_idx, sec_end_bar_idx),
            })
            y += SECTION_HEADER_H + 4

            for row_offset in range(0, len(sec.bars), bpr):
                row_bars = sec.bars[row_offset:row_offset + bpr]
                x = PAD

                for bar in row_bars:
                    bar_idx = bars_flat.index(bar)
                    nb = bar.n_beats
                    beat_w = max(MIN_BEAT_W, int((actual_bar_w - 2) / nb))
                    bw = beat_w * nb + 2
                    rx, ry = x, y

                    outline, width = self._bar_outline(bar_idx, state)
                    bg = self.canvas.create_rectangle(
                        rx, ry, rx + bw, ry + BAR_H,
                        fill=self._bar_bg_color(bar_idx, state),
                        outline=outline, width=width,
                    )

                    beat_rects = []
                    vq = state.vq
                    for bii, beat in enumerate(bar.beats):
                        bx = rx + 1 + bii * beat_w
                        bw2 = max(1, beat_w - 1)
                        bh = BAR_H - 4
                        is_db = beat.position == 1
                        fill = "#665533" if is_db else BEAT_OFF
                        r = self.canvas.create_rectangle(
                            bx, ry + 2, bx + bw2, ry + 2 + bh,
                            fill=fill, outline="#445" if is_db else "#2a2a2a", width=1,
                        )

                        t = None
                        if beat_w >= 14:
                            if vq is not None and bii == nb - 1 and bar_idx in vq.bars:
                                label = self._queue_label(bar_idx, vq)
                                if label:
                                    t = self.canvas.create_text(
                                        bx + bw2 // 2, ry + 2 + bh // 2,
                                        text=label, fill="#ddd", font=("Segoe UI", 7),
                                    )
                            elif is_db:
                                bar_num = len(sec.bars[:row_offset]) + row_bars.index(bar) + 1
                                t = self.canvas.create_text(
                                    bx + bw2 // 2, ry + 2 + bh // 2,
                                    text=str(bar_num), fill="#ddd", font=("Segoe UI", 7),
                                )
                        beat_rects.append((r, t))

                    self.bar_items.append({
                        "bg": bg,
                        "beats": beat_rects,
                        "bar_idx": bar_idx,
                        "bar_num": bar_idx + 1,
                        "n_beats": nb,
                        "beats_data": bar.beats,
                    })
                    x += bw + BAR_GAP
                y += self._row_total_h

        self.canvas.configure(scrollregion=(0, 0, cw, y + PAD))

    def update_playhead(self, frame_idx: int, ms: float, state: TransportState):
        if len(self.arrangement.beat_times_ms) == 0:
            for item in self.bar_items:
                bi = item["bar_idx"]
                outline, width = self._bar_outline(bi, state)
                self.canvas.itemconfig(item["bg"],
                                       fill=self._bar_bg_color(bi, state),
                                       outline=outline, width=width)
                for bii, (r, t) in enumerate(item["beats"]):
                    beat = item["beats_data"][bii]
                    is_db = beat.position == 1
                    self.canvas.itemconfig(r, fill="#665533" if is_db else BEAT_OFF)
            return

        beat_idx = int(np.searchsorted(self.arrangement.beat_times_ms, ms, side="right") - 1)
        if beat_idx < 0 or beat_idx >= len(self.arrangement.beat_times_ms):
            for item in self.bar_items:
                bi = item["bar_idx"]
                outline, width = self._bar_outline(bi, state)
                self.canvas.itemconfig(item["bg"],
                                       fill=self._bar_bg_color(bi, state),
                                       outline=outline, width=width)
                for bii, (r, t) in enumerate(item["beats"]):
                    beat = item["beats_data"][bii]
                    is_db = beat.position == 1
                    self.canvas.itemconfig(r, fill="#665533" if is_db else BEAT_OFF)
            return

        current_item = None
        beat_offset_in_bar = None
        cumulative_idx = 0
        for item in self.bar_items:
            bar_beat_count = len(item["beats_data"])
            if cumulative_idx <= beat_idx < cumulative_idx + bar_beat_count:
                current_item = item
                beat_offset_in_bar = beat_idx - cumulative_idx
                break
            cumulative_idx += bar_beat_count

        for item in self.bar_items:
            bi = item["bar_idx"]
            if item is current_item and beat_offset_in_bar is not None:
                self.canvas.itemconfig(item["bg"],
                                       fill=BAR_ACTIVE, outline=DOWNBEAT, width=2)
                for bii, (r, t) in enumerate(item["beats"]):
                    beat = item["beats_data"][bii]
                    is_db = beat.position == 1
                    if bii == beat_offset_in_bar:
                        fill = DOWNBEAT if is_db else BEAT_ON
                    else:
                        fill = "#665533" if is_db else BEAT_OFF
                    self.canvas.itemconfig(r, fill=fill)
            else:
                outline, width = self._bar_outline(bi, state)
                self.canvas.itemconfig(item["bg"],
                                       fill=self._bar_bg_color(bi, state),
                                       outline=outline, width=width)
                for bii, (r, t) in enumerate(item["beats"]):
                    beat = item["beats_data"][bii]
                    is_db = beat.position == 1
                    self.canvas.itemconfig(r, fill="#665533" if is_db else BEAT_OFF)

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

    def bar_at_xy(self, x: int, y: int) -> int | None:
        cid = self.canvas.find_closest(x, y)
        if not cid:
            return None
        cid = cid[0]
        for item in self.bar_items:
            if item["bg"] == cid or any(r == cid for r, t in item["beats"]):
                return item["bar_idx"]
        return None

    def section_at_xy(self, x: int, y: int) -> int | None:
        cid = self.canvas.find_closest(x, y)
        if not cid:
            return None
        cid = cid[0]
        for sec_item in self.section_items:
            if sec_item["bg"] == cid or sec_item["text"] == cid:
                return sec_item["sec_idx"]
        return None
