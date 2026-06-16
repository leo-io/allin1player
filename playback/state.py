from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VirtualQueue:
    bars: list[int]
    entry_bar: int
    pos: int = 0
    looping: bool = False
    active: bool = False

    def is_empty(self) -> bool:
        return len(self.bars) == 0

    def positions_of(self, bar_idx: int) -> list[int]:
        return [i + 1 for i, b in enumerate(self.bars) if b == bar_idx]

    def advance(self) -> int | None:
        if self.pos >= len(self.bars):
            return None
        result = self.bars[self.pos]
        self.pos += 1
        return result


@dataclass
class TransportState:
    frame_idx: int = 0
    playing: bool = False
    paused: bool = False
    loop_range: tuple | None = None
    selected: tuple | None = None
    # Inclusive flat-bar range (lo, hi) for multi-bar edits, or None.
    selection: tuple | None = None
    dirty: bool = False
    vq: VirtualQueue | None = None
    cursor: int | None = None
    play_bar: int | None = None
    # Section editing state
    section_selection: tuple | None = None  # Inclusive (lo, hi) section-index range
    section_cursor: int | None = None  # Section index marking paste insertion line
    has_section_clipboard: bool = False  # Gates rendering of red insertion line
