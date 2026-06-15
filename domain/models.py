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
        logger.debug(f"[CHECKPOINT] Entering reindex() with name='{self.name}', sr={sr}")
        try:
            beat_times = []
            beat_numbers = []
            section_count = len(self.sections)
            bar_count = sum(len(sec.bars) for sec in self.sections)

            for sec_idx, sec in enumerate(self.sections):
                try:
                    for bar in sec.bars:
                        for beat in bar.beats:
                            try:
                                beat_times.append(beat.time_ms)
                                beat_numbers.append(beat.position)
                            except AttributeError as e:
                                error_msg = f"Beat missing required attributes in section {sec_idx}: {e}"
                                logger.error(f"[EXCEPTION] {error_msg} | beat={beat}")
                                raise
                except Exception as e:
                    error_msg = f"Error processing section {sec_idx}: {type(e).__name__}: {e}"
                    logger.error(f"[EXCEPTION] {error_msg}")
                    raise

            try:
                self.beat_times_ms = np.array(beat_times, dtype=np.int64)
                self.beat_numbers = np.array(beat_numbers, dtype=int)
                logger.debug(f"[CHECKPOINT] Arrays created: {len(beat_times)} beats, {section_count} sections, {bar_count} bars")
                logger.debug(f"[CHECKPOINT] Exiting reindex() successfully")
            except Exception as e:
                error_msg = f"Error creating numpy arrays: {type(e).__name__}: {e}"
                logger.error(f"[EXCEPTION] {error_msg}")
                raise

        except Exception as e:
            error_msg = f"Unexpected error in reindex(): {type(e).__name__}: {e}"
            logger.error(f"[EXCEPTION] {error_msg}", exc_info=True)
            raise

    def set_bar_frames(self, sr: int, total_frames: int):
        logger.debug(f"[CHECKPOINT] Entering set_bar_frames() with sr={sr}, total_frames={total_frames}")
        try:
            if sr <= 0:
                error_msg = f"Invalid sample rate: sr={sr} (must be > 0)"
                logger.error(f"[EXCEPTION] {error_msg}")
                raise ValueError(error_msg)

            if total_frames < 0:
                error_msg = f"Invalid total_frames: {total_frames} (must be >= 0)"
                logger.error(f"[EXCEPTION] {error_msg}")
                raise ValueError(error_msg)

            bars_flat = self.bars
            logger.debug(f"[CHECKPOINT] Processing {len(bars_flat)} bars")

            frames = []
            for bar_idx, bar in enumerate(bars_flat):
                try:
                    if bar.beats:
                        try:
                            time_ms = bar.beats[0].time_ms
                            frame = int(round(float(time_ms) * sr / 1000))
                            if frame < 0:
                                logger.warning(f"Negative frame calculated for bar {bar_idx}: time_ms={time_ms}, frame={frame}")
                            frames.append(frame)
                        except (ValueError, TypeError) as e:
                            error_msg = f"Error calculating frame for bar {bar_idx}: {e}"
                            logger.error(f"[EXCEPTION] {error_msg} | bar.beats[0].time_ms={bar.beats[0].time_ms}")
                            raise
                except Exception as e:
                    error_msg = f"Error processing bar {bar_idx}: {type(e).__name__}: {e}"
                    logger.error(f"[EXCEPTION] {error_msg}")
                    raise

            frames.append(total_frames)
            logger.debug(f"[CHECKPOINT] Frame list built: {len(frames)} boundaries")

            try:
                self.bar_frames = np.array(frames, dtype=np.int64)
                self.total_frames = total_frames
                duration_sec = total_frames / sr if sr > 0 else 0
                logger.debug(f"[CHECKPOINT] Bar frames set: {len(frames)} boundaries, duration={duration_sec:.2f}s")
                logger.debug(f"[CHECKPOINT] Exiting set_bar_frames() successfully")
            except Exception as e:
                error_msg = f"Error creating numpy array: {type(e).__name__}: {e}"
                logger.error(f"[EXCEPTION] {error_msg}")
                raise

        except Exception as e:
            error_msg = f"Unexpected error in set_bar_frames(): {type(e).__name__}: {e}"
            logger.error(f"[EXCEPTION] {error_msg}", exc_info=True)
            raise

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
        logger.debug(f"[CHECKPOINT] Entering clone() with name='{self.name}'")
        try:
            source_sections = len(self.sections)
            source_bars = len(self.bars)
            logger.debug(f"[CHECKPOINT] Source: {source_sections} sections, {source_bars} bars")

            try:
                cloned = copy.deepcopy(self)
                logger.debug(f"[CHECKPOINT] Deepcopy completed successfully")
            except Exception as e:
                error_msg = f"Error during deepcopy: {type(e).__name__}: {e}"
                logger.error(f"[EXCEPTION] {error_msg}")
                raise

            cloned_sections = len(cloned.sections)
            cloned_bars = len(cloned.bars)
            logger.debug(f"[CHECKPOINT] Clone created: {cloned_sections} sections, {cloned_bars} bars")

            if cloned_sections != source_sections or cloned_bars != source_bars:
                logger.warning(f"Clone size mismatch: source({source_sections}s, {source_bars}b) != clone({cloned_sections}s, {cloned_bars}b)")

            logger.debug(f"[CHECKPOINT] Exiting clone() successfully")
            return cloned
        except Exception as e:
            error_msg = f"Unexpected error in clone(): {type(e).__name__}: {e}"
            logger.error(f"[EXCEPTION] {error_msg}", exc_info=True)
            raise
