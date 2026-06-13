from __future__ import annotations
from dataclasses import dataclass, field
import copy
import numpy as np


@dataclass(frozen=True)
class Beat:
    time_ms: int
    position: int
    chord: str = ""


@dataclass(frozen=True)
class Bar:
    idx: int
    beats: tuple[Beat, ...]

    @property
    def start_beat_idx(self) -> int:
        if not self.beats:
            return 0
        beat_times = [b.time_ms for b in self.beats]
        return int(np.searchsorted(beat_times, min(beat_times)))

    @property
    def n_beats(self) -> int:
        return len(self.beats)

    @property
    def beat_numbers(self) -> tuple:
        return tuple(b.position for b in self.beats)


@dataclass
class Section:
    idx: int
    name: str
    bars: list[Bar]

    def bar_count(self) -> int:
        return len(self.bars)


@dataclass
class Arrangement:
    name: str
    master: bool
    sections: list[Section]
    beat_times_ms: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64), repr=False)
    beat_numbers: np.ndarray = field(default_factory=lambda: np.array([], dtype=int), repr=False)
    bar_frames: np.ndarray | None = field(default=None, repr=False)
    total_frames: int = 0

    def reindex(self, sr: int):
        beat_times = []
        beat_numbers = []
        for sec in self.sections:
            for bar in sec.bars:
                for beat in bar.beats:
                    beat_times.append(beat.time_ms)
                    beat_numbers.append(beat.position)

        self.beat_times_ms = np.array(beat_times, dtype=np.int64)
        self.beat_numbers = np.array(beat_numbers, dtype=int)

    def set_bar_frames(self, sr: int, total_frames: int):
        bars_flat = self.bars
        frames = []
        for bar in bars_flat:
            if bar.beats:
                frame = int(round(bar.beats[0].time_ms * sr / 1000))
                frames.append(frame)
        frames.append(total_frames)
        self.bar_frames = np.array(frames, dtype=np.int64)
        self.total_frames = total_frames

    @property
    def bars(self) -> list[Bar]:
        result = []
        for sec in self.sections:
            result.extend(sec.bars)
        return result

    def bar_at_frame(self, frame_idx: int) -> int:
        if self.bar_frames is None:
            return 0
        idx = int(np.searchsorted(self.bar_frames, frame_idx, side="right") - 1)
        return max(0, min(idx, len(self.bars) - 1))

    def bar_at_ms(self, ms: float) -> int:
        if len(self.beat_times_ms) == 0:
            return 0
        idx = int(np.searchsorted(self.beat_times_ms, ms, side="right") - 1)
        if idx < 0:
            return 0
        bars_flat = self.bars
        beat_idx = 0
        for bar_idx, bar in enumerate(bars_flat):
            if beat_idx <= idx < beat_idx + len(bar.beats):
                return bar_idx
            beat_idx += len(bar.beats)
        return len(bars_flat) - 1 if bars_flat else 0

    def next_bar_frame(self, bar_idx: int) -> int:
        if self.bar_frames is None:
            return 0
        return int(self.bar_frames[bar_idx + 1])

    def chord_at_beat(self, beat_idx: int) -> str:
        beat_count = 0
        for bar in self.bars:
            for beat in bar.beats:
                if beat_count == beat_idx:
                    return beat.chord
                beat_count += 1
        return ""

    def clone(self) -> Arrangement:
        return copy.deepcopy(self)
