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


class MixRenderer:
    """Draws mix sections as plain blocks (no beat/bar cells) on a canvas."""

    def __init__(self, canvas: tk.Canvas, mix: Mix):
        self.canvas = canvas
        self.mix = mix
        self.section_items: list[dict] = []

    def _block_width(self, sec) -> int:
        return max(BLOCK_MIN_W, min(BLOCK_MAX_W, BLOCK_MIN_W + sec.bar_count() * PX_PER_BAR))

    def draw_all(self, selected: int | None = None):
        self.canvas.delete("all")
        self.section_items = []
        cw = max(self.canvas.winfo_width(), 600)

        x = PAD
        y = PAD
        row_h = BLOCK_H + ROW_GAP

        for sec_idx, sec in enumerate(self.mix.sections):
            bw = self._block_width(sec)

            # Wrap to a new row when the block would overflow the canvas width.
            if x > PAD and x + bw > cw - PAD:
                x = PAD
                y += row_h

            is_sel = (selected == sec_idx)
            bg = self.canvas.create_rectangle(
                x, y, x + bw, y + BLOCK_H,
                fill=SELECT_BG if is_sel else BAR_BG,
                outline=SELECT_OUTLINE if is_sel else "#334",
                width=2 if is_sel else 1,
            )
            name_id = self.canvas.create_text(
                x + bw // 2, y + BLOCK_H // 2 - 8,
                text=sec.name.upper(),
                fill="#e0e0e0", font=("Segoe UI", 10, "bold"),
            )
            meta_id = self.canvas.create_text(
                x + bw // 2, y + BLOCK_H // 2 + 12,
                text=f"[{sec.bar_count()} bars]",
                fill="#888", font=("Segoe UI", 8),
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
