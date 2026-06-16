from __future__ import annotations
import sys
import threading
import numpy as np
import sounddevice as sd

from domain.models import Arrangement
from playback.state import TransportState


class PlaybackEngine:
    """Sequence-driven playback.

    Playback walks the arrangement's bars in order; each bar plays its own
    slice of the single source WAV. `state.play_bar` (a flat index into
    `arrangement.bars`) is the source of truth for position, since once bars are
    duplicated a raw frame index no longer maps to a unique bar.
    """

    def __init__(self, audio_data: np.ndarray, arrangement: Arrangement,
                 state: TransportState, lock: threading.RLock, sr: int):
        self.audio_data = audio_data
        self.arrangement = arrangement
        self.state = state
        self.lock = lock
        self.sr = sr
        self.stream = None
        self._bounds = arrangement.source_bounds(sr)

    def _slice(self, bar_idx: int) -> tuple[int, int]:
        bars = self.arrangement.bars
        if not bars or bar_idx < 0 or bar_idx >= len(bars):
            return (0, 0)
        return self.arrangement.bar_source_slice(bars[bar_idx], self.sr, self._bounds)

    def play(self, from_bar: int | None = None):
        with self.lock:
            bars = self.arrangement.bars
            self._bounds = self.arrangement.source_bounds(self.sr)
            if from_bar is None:
                from_bar = self.state.play_bar if self.state.play_bar is not None else 0
            from_bar = max(0, min(from_bar, len(bars) - 1)) if bars else 0
            self.state.play_bar = from_bar
            start, _ = self._slice(from_bar)
            self.state.frame_idx = start
            self.state.playing = True
            self.state.paused = False
        self.stream = sd.OutputStream(
            samplerate=self.sr,
            channels=1,
            callback=self._audio_callback,
            blocksize=1024,
        )
        self.stream.start()

    def pause(self):
        with self.lock:
            if not self.state.playing:
                self.state.playing = True
                self.state.paused = False
                if self.stream is not None:
                    self.stream.start()
                return
            if self.state.paused:
                self.state.paused = False
                if self.stream is not None:
                    self.stream.start()
            else:
                self.state.paused = True
                if self.stream is not None:
                    self.stream.stop()

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.abort()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        with self.lock:
            self.state.playing = False
            self.state.paused = False
            self.state.frame_idx = 0

    def _advance_bar(self) -> int | None:
        """Pick the next bar to play at a bar boundary, or None to stop."""
        bars = self.arrangement.bars
        n = len(bars)
        cur = self.state.play_bar if self.state.play_bar is not None else 0

        # 1. loop range
        lr = self.state.loop_range
        if lr is not None and lr[0] <= cur < lr[1]:
            return lr[0] if cur + 1 >= lr[1] else cur + 1

        # 2. live virtual queue (seek-to-bar is done by queuing a single bar)
        vq = self.state.vq
        if vq is not None and not vq.is_empty():
            nxt = vq.advance()
            if nxt is not None:
                self.state.dirty = True
                return nxt if 0 <= nxt < n else None
            if vq.looping:
                vq.pos = 0
                self.state.dirty = True
                first = vq.bars[0]
                return first if 0 <= first < n else None
            resume = vq.entry_bar + 1
            self.state.vq = None
            self.state.dirty = True
            return resume if 0 <= resume < n else None

        # 3. arrangement order
        return cur + 1 if cur + 1 < n else None

    def _audio_callback(self, outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        with self.lock:
            if not self.state.playing:
                outdata[:] = 0
                return

            bars = self.arrangement.bars
            n = len(bars)
            total = len(self.audio_data)
            filled = 0

            while filled < frames:
                if self.state.play_bar is None or self.state.play_bar >= n:
                    outdata[filled:] = 0
                    self.state.playing = False
                    raise sd.CallbackStop()

                start, end = self._slice(self.state.play_bar)
                end = min(end, total)
                # If the read head fell outside the current slice (just jumped
                # here, or the bar list changed under us), reseat to its start.
                if self.state.frame_idx < start or self.state.frame_idx >= end:
                    self.state.frame_idx = start

                take = min(frames - filled, end - self.state.frame_idx)
                if take > 0:
                    outdata[filled:filled + take] = \
                        self.audio_data[self.state.frame_idx:self.state.frame_idx + take]
                    self.state.frame_idx += take
                    filled += take

                # Reached the end of this bar's slice -> advance.
                if self.state.frame_idx >= end:
                    nxt = self._advance_bar()
                    if nxt is None:
                        outdata[filled:] = 0
                        self.state.playing = False
                        raise sd.CallbackStop()
                    self.state.play_bar = nxt
                    nstart, _ = self._slice(nxt)
                    self.state.frame_idx = nstart
                    self.state.dirty = True
