from __future__ import annotations
import sys
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf

from domain.models import SongStructure
from playback.state import TransportState


class PlaybackEngine:
    def __init__(self, audio_data: np.ndarray, song: SongStructure,
                 state: TransportState, lock: threading.RLock, sr: int):
        self.audio_data = audio_data
        self.song = song
        self.state = state
        self.lock = lock
        self.sr = sr
        self.stream = None

    def play(self, from_frame: int | None = None):
        with self.lock:
            if from_frame is not None:
                self.state.frame_idx = from_frame
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

    def _next_jump(self):
        pos = self.state.frame_idx

        if self.state.loop_range is not None:
            start = int(self.song.bar_frames[self.state.loop_range[0]])
            end = int(self.song.bar_frames[self.state.loop_range[1]])
            if pos < end and start < end:
                return end, start, False

        vq = self.state.vq
        if vq is not None and vq.active:
            b = int(np.searchsorted(self.song.bar_frames, pos, side="right"))
            if b < len(self.song.bar_frames):
                return int(self.song.bar_frames[b]), None, True

        return None, None, False

    def _audio_callback(self, outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        with self.lock:
            if not self.state.playing:
                outdata[:] = 0
                return
            filled = 0
            while filled < frames:
                boundary, target, is_vq = self._next_jump()
                limit = boundary if boundary is not None else len(self.audio_data)
                take = min(frames - filled, limit - self.state.frame_idx)
                if take > 0:
                    outdata[filled:filled + take] = \
                        self.audio_data[self.state.frame_idx:self.state.frame_idx + take]
                    self.state.frame_idx += take
                    filled += take
                if self.state.frame_idx >= limit:
                    if boundary is None:
                        outdata[filled:] = 0
                        raise sd.CallbackStop()
                    if is_vq:
                        vq = self.state.vq
                        next_bar = vq.advance() if vq else None
                        if next_bar is None:
                            if vq and vq.looping:
                                vq.pos = 0
                                target = int(self.song.bar_frames[vq.bars[0]])
                            else:
                                resume = (vq.entry_bar + 1) if vq else 0
                                self.state.vq = None
                                self.state.dirty = True
                                if resume < len(self.song.bars):
                                    target = int(self.song.bar_frames[resume])
                                else:
                                    outdata[filled:] = 0
                                    raise sd.CallbackStop()
                        else:
                            target = int(self.song.bar_frames[next_bar])
                        self.state.dirty = True
                    self.state.frame_idx = target
