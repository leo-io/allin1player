from __future__ import annotations
import tkinter as tk

from domain.models import Mix

# Reuse the arrangement palette/spacing so the two editors look consistent.
from ui.renderer import (
    BAR_BG,
    SELECT_BG,
    SELECT_OUTLINE,
    PAD,
    ROW_GAP,
)

BLOCK_H = 56
BLOCK_GAP = 10
BLOCK_MIN_W = 90
# Each bar in a section adds this many px of width, so longer sections look wider.
PX_PER_BAR = 6
BLOCK_MAX_W = 320

PLAYING_OUTLINE = "#44ff88"
PLAYING_BG = "#1a3a2a"

TRANSITION_COLOR = "#cc2222"
TRANSITION_BG = "#3a0a0a"

MULTI_SELECT_BG = "#1e2a4a"
MULTI_SELECT_OUTLINE = "#6688cc"


class MixRenderer:
    """Draws mix sections as plain blocks (no beat/bar cells) on a canvas."""

    def __init__(self, canvas: tk.Canvas, mix: Mix):
        self.canvas = canvas
        self.mix = mix
        self.section_items: list[dict] = []

    def _block_width(self, sec) -> int:
        return max(BLOCK_MIN_W, min(BLOCK_MAX_W, BLOCK_MIN_W + sec.bar_count() * PX_PER_BAR))

    @staticmethod
    def _section_color(sec) -> str:
        """Return the source color for this section (from its first bar), or empty string."""
        if sec.bars and sec.bars[0].color:
            return sec.bars[0].color
        return ""

    @staticmethod
    def _darken(hex_color: str, factor: float = 0.45) -> str:
        """Blend hex_color toward black by factor (0 = original, 1 = black)."""
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        r = int(r * (1 - factor))
        g = int(g * (1 - factor))
        b = int(b * (1 - factor))
        return f"#{r:02x}{g:02x}{b:02x}"

    def draw_all(
        self,
        selected: int | None = None,
        playing_sec: int | None = None,
        selection: set[int] | None = None,
    ):
        self.canvas.delete("all")
        self.section_items = []
        cw = max(self.canvas.winfo_width(), 600)
        sel_set = selection or set()

        x = PAD
        y = PAD
        row_h = BLOCK_H + ROW_GAP

        for sec_idx, sec in enumerate(self.mix.sections):
            bw = self._block_width(sec)

            # Wrap to a new row when the block would overflow the canvas width.
            if x > PAD and x + bw > cw - PAD:
                x = PAD
                y += row_h

            src_color = self._section_color(sec)
            is_sel = (selected == sec_idx)
            is_in_multi = (sec_idx in sel_set and not is_sel)
            is_playing = (playing_sec == sec_idx)
            is_transition = sec.is_transition

            if is_playing:
                fill = PLAYING_BG
                outline = PLAYING_OUTLINE
                outline_w = 2
            elif is_transition and is_sel:
                fill = TRANSITION_COLOR
                outline = "#ff6666"
                outline_w = 2
            elif is_transition and is_in_multi:
                fill = TRANSITION_BG
                outline = MULTI_SELECT_OUTLINE
                outline_w = 2
            elif is_transition:
                fill = TRANSITION_BG
                outline = TRANSITION_COLOR
                outline_w = 1
            elif is_sel:
                fill = src_color if src_color else SELECT_BG
                outline = SELECT_OUTLINE
                outline_w = 2
            elif is_in_multi:
                fill = MULTI_SELECT_BG
                outline = MULTI_SELECT_OUTLINE
                outline_w = 2
            else:
                fill = self._darken(src_color) if src_color else BAR_BG
                outline = src_color if src_color else "#334"
                outline_w = 1

            bg = self.canvas.create_rectangle(
                x, y, x + bw, y + BLOCK_H,
                fill=fill, outline=outline, width=outline_w,
            )
            fade_info = (f"fade-out: {len(sec.fade_out_bars)}  fade-in: {len(sec.fade_in_bars)}"
                         if is_transition else f"[{sec.bar_count()} bars]")
            name_text = ("▶ " if is_playing else "") + sec.name.upper()
            name_id = self.canvas.create_text(
                x + bw // 2, y + BLOCK_H // 2 - 8,
                text=name_text,
                fill=PLAYING_OUTLINE if is_playing else (TRANSITION_COLOR if is_transition else "#e0e0e0"),
                font=("Segoe UI", 10, "bold"),
            )
            meta_id = self.canvas.create_text(
                x + bw // 2, y + BLOCK_H // 2 + 12,
                text=fade_info,
                fill="#ff8888" if is_transition else ("#ccc" if src_color else "#888"),
                font=("Segoe UI", 8),
            )
            # Position index badge (mix order), top-left.
            idx_id = self.canvas.create_text(
                x + 8, y + 8,
                text=str(sec_idx + 1),
                fill="#66ccff", font=("Segoe UI", 8, "bold"), anchor="nw",
            )
            self.section_items.append({
                "bg": bg,
                "ids": (name_id, meta_id, idx_id),
                "sec_idx": sec_idx,
            })
            x += bw + BLOCK_GAP

        if not self.mix.sections:
            self.canvas.create_text(
                cw // 2, 60,
                text="Empty mix — use Edit ▸ Add Section… to import sections.",
                fill="#666", font=("Segoe UI", 11),
            )

        self.canvas.configure(scrollregion=(0, 0, cw, y + BLOCK_H + PAD))

    def section_at_xy(self, x: int, y: int) -> int | None:
        cid = self.canvas.find_closest(x, y)
        if not cid:
            return None
        cid = cid[0]
        for item in self.section_items:
            if item["bg"] == cid or cid in item["ids"]:
                return item["sec_idx"]
        return None
