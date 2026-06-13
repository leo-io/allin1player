from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class Bar:
    idx: int
    start_beat_idx: int
    n_beats: int
    beat_numbers: tuple


@dataclass(frozen=True)
class Section:
    idx: int
    name: str
    first_bar: int
    end_bar: int

    def bar_count(self) -> int:
        return self.end_bar - self.first_bar


@dataclass(frozen=True)
class Chord:
    start_ms: float
    end_ms: float
    name: str


@dataclass
class SongStructure:
    beat_times_ms: np.ndarray = field(repr=False)
    beat_numbers: np.ndarray = field(repr=False)
    bars: list[Bar] = field(repr=False)
    sections: list[Section] = field(repr=False)
    bar_frames: np.ndarray | None = None
    total_frames: int = 0
    chords: list[Chord] = field(default_factory=list)
    beat_chords: list[str] = field(default_factory=list)

    def bar_at_frame(self, frame_idx: int) -> int:
        idx = int(np.searchsorted(self.bar_frames, frame_idx, side="right") - 1)
        return max(0, min(idx, len(self.bars) - 1))

    def bar_at_ms(self, ms: float) -> int:
        idx = int(np.searchsorted(self.beat_times_ms, ms, side="right") - 1)
        if idx < 0:
            return 0
        for bar in self.bars:
            if bar.start_beat_idx <= idx < bar.start_beat_idx + bar.n_beats:
                return bar.idx
        return len(self.bars) - 1

    def next_bar_frame(self, bar_idx: int) -> int:
        return int(self.bar_frames[bar_idx + 1])

    def chord_at_beat(self, beat_idx: int) -> str:
        if 0 <= beat_idx < len(self.beat_chords):
            return self.beat_chords[beat_idx]
        return ""


@dataclass(frozen=True)
class ArrangementVersion:
    name: str
    section_ordering: list[tuple[int, int, int]]  # (source_sec_idx, source_first_bar, source_end_bar)


@dataclass
class ArrangementDocument:
    source: SongStructure
    versions: list[ArrangementVersion] = field(default_factory=list)
    active_version_idx: int = 0
    schema_version: int = 1

    @property
    def active_version(self) -> ArrangementVersion | None:
        if not self.versions:
            return None
        return self.versions[self.active_version_idx]

    def add_version(self, name: str) -> ArrangementVersion:
        v = ArrangementVersion(
            name=name,
            section_ordering=[
                (s.idx, s.first_bar, s.end_bar) for s in self.source.sections
            ],
        )
        self.versions.append(v)
        self.active_version_idx = len(self.versions) - 1
        return v

    def get_ordered_bars(self, version_idx: int | None = None) -> list[int]:
        if version_idx is None:
            version_idx = self.active_version_idx
        if version_idx < 0 or version_idx >= len(self.versions):
            return list(range(len(self.source.bars)))
        version = self.versions[version_idx]
        bars = []
        for _, first_bar, end_bar in version.section_ordering:
            bars.extend(range(first_bar, end_bar))
        return bars


@dataclass(frozen=True)
class Command:
    type: str
    params: dict = field(default_factory=dict)
