from __future__ import annotations
from dataclasses import dataclass, field
import copy
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Beat:
    start_ms: int
    finish_ms: int
    position: int
    volume: int = 100


@dataclass(frozen=True)
class Bar:
    idx: int
    beats: tuple[Beat, ...]
    audiosource: str = ""
    color: str = ""

    @property
    def start_beat_idx(self) -> int:
        if not self.beats:
            return 0
        beat_times = [b.start_ms for b in self.beats]
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
    local_key: str = ""
    local_bpm: float = 0.0
    is_transition: bool = False
    transition_start_bpm: float = 0.0
    transition_finish_bpm: float = 0.0
    fade_out_bars: list[Bar] = field(default_factory=list)
    fade_in_bars: list[Bar] = field(default_factory=list)

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
                                beat_times.append(beat.start_ms)
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
                            start_ms = bar.beats[0].start_ms
                            frame = int(round(float(start_ms) * sr / 1000))
                            if frame < 0:
                                logger.warning(f"Negative frame calculated for bar {bar_idx}: start_ms={start_ms}, frame={frame}")
                            frames.append(frame)
                        except (ValueError, TypeError) as e:
                            error_msg = f"Error calculating frame for bar {bar_idx}: {e}"
                            logger.error(f"[EXCEPTION] {error_msg} | bar.beats[0].start_ms={bar.beats[0].start_ms}")
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

    def source_bounds(self, sr: int) -> np.ndarray:
        """Sorted unique source-frame start positions of all bars, plus
        total_frames as the final boundary.

        Each bar's audio slice is [start, next_boundary). Because edits only
        reorder/duplicate/delete bars (never invent new start_ms), this set is
        invariant under editing and stays sorted, so searchsorted against it is
        always valid even when bars are duplicated.
        """
        starts = set()
        for bar in self.bars:
            if bar.beats:
                starts.add(int(round(float(bar.beats[0].start_ms) * sr / 1000)))
        starts.add(int(self.total_frames))
        return np.array(sorted(starts), dtype=np.int64)

    def bar_source_slice(self, bar: Bar, sr: int, bounds: np.ndarray) -> tuple[int, int]:
        """Return the [start, end) source-frame slice for a single bar."""
        if not bar.beats:
            return (0, 0)
        start = int(round(float(bar.beats[0].start_ms) * sr / 1000))
        pos = int(np.searchsorted(bounds, start, side="right"))
        end = int(bounds[pos]) if pos < len(bounds) else int(self.total_frames)
        return (start, end)

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


@dataclass
class Mix:
    """A higher-level arrangement of whole sections (the "mix-sections").

    Each entry is a full Section (with its bars/beats/audiosource embedded), so a
    Mix is self-contained. Unlike Arrangement, a Mix carries no beat/frame arrays
    because it is edited and rendered at the section-block level only.
    """
    name: str
    sections: list[Section] = field(default_factory=list)
    creationdate: int = field(default_factory=lambda: int(time.time() * 1000))

    def clone(self) -> Mix:
        return copy.deepcopy(self)
