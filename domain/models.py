from __future__ import annotations
from dataclasses import dataclass, field
import copy
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Beat:
    time_ms: int
    position: int


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
    creationdate: int = field(default_factory=lambda: int(time.time() * 1000))
    beat_times_ms: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64), repr=False)
    beat_numbers: np.ndarray = field(default_factory=lambda: np.array([], dtype=int), repr=False)
    bar_frames: np.ndarray | None = field(default=None, repr=False)
    total_frames: int = 0

    def reindex(self, sr: int):
        logger.debug(f"Reindexing arrangement '{self.name}' (sr={sr})")
        beat_times = []
        beat_numbers = []
        for sec in self.sections:
            for bar in sec.bars:
                for beat in bar.beats:
                    beat_times.append(beat.time_ms)
                    beat_numbers.append(beat.position)

        self.beat_times_ms = np.array(beat_times, dtype=np.int64)
        self.beat_numbers = np.array(beat_numbers, dtype=int)
        logger.debug(f"Reindex complete: {len(beat_times)} beats indexed")

    def set_bar_frames(self, sr: int, total_frames: int):
        logger.debug(f"Setting bar frames for '{self.name}': sr={sr}, total_frames={total_frames}")
        bars_flat = self.bars
        frames = []
        for bar in bars_flat:
            if bar.beats:
                frame = int(round(bar.beats[0].time_ms * sr / 1000))
                frames.append(frame)
        frames.append(total_frames)
        self.bar_frames = np.array(frames, dtype=np.int64)
        self.total_frames = total_frames
        logger.debug(f"Bar frames set: {len(frames)} frame boundaries, duration={total_frames / sr:.2f}s")

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

    def clone(self) -> Arrangement:
        logger.debug(f"Cloning arrangement '{self.name}'")
        cloned = copy.deepcopy(self)
        logger.debug(f"Clone created: {len(cloned.sections)} sections")
        return cloned
