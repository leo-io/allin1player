from __future__ import annotations
import sys
import logging
import threading
from pathlib import Path
from typing import Callable
from queue import Queue

import numpy as np
import sounddevice as sd
import soundfile as sf

from domain.models import Section

logger = logging.getLogger(__name__)


class MixPlaybackEngine:
    """On-demand section playback for the Mix Editor.

    Maintains a 3-section rolling cache: the section before, the current, and
    the section after. Pre-loading happens in a background daemon thread so the
    next section is already decoded before it needs to play.

    Usage::

        engine = MixPlaybackEngine(
            on_section_change=lambda idx: ...,
            on_stop=lambda: ...,
        )
        engine.play(sections, start_sec=0)   # non-blocking
        engine.stop()                         # abort + free
    """

    def __init__(
        self,
        on_section_change: Callable[[int], None] | None = None,
        on_stop: Callable[[], None] | None = None,
    ):
        self._lock = threading.Lock()
        self._stream: sd.OutputStream | None = None
        self._stream_sr: int = 44100  # the stream's fixed sample rate
        self._playing = False
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None

        self._sections: list[Section] = []

        # Rolling cache: section_index -> (audio, sr)
        self._cache: dict[int, tuple[np.ndarray, int]] = {}
        self._loading: set[int] = set()   # indices currently being pre-loaded
        self._cache_lock = threading.Lock()

        # Callback state: tracks position in playback (never read from worker, only callback)
        self._callback_sec_idx: int = 0
        self._callback_frame: int = 0
        self._callback_audio: np.ndarray | None = None
        self._callback_end_frame: int = 0

        # Section change notifications from callback to worker
        self._section_change_queue: Queue[int] = Queue()

        self.on_section_change = on_section_change
        self.on_stop = on_stop

    # ------------------------------------------------------------------ public

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def current_section_idx(self) -> int:
        with self._lock:
            return self._callback_sec_idx

    def play(self, sections: list[Section], start_sec: int = 0) -> None:
        """Start playing *sections* from *start_sec*. Non-blocking."""
        self.stop()
        if not sections:
            return
        self._sections = list(sections)
        self._callback_sec_idx = max(0, min(start_sec, len(sections) - 1))
        self._stop_event.clear()
        self._playing = True
        self._worker = threading.Thread(target=self._run, daemon=True, name="MixPlayback")
        self._worker.start()

    def stop(self) -> None:
        """Stop playback and free all buffers including the pre-load cache."""
        self._playing = False
        self._stop_event.set()
        if self._stream is not None:
            try:
                self._stream.abort()
            except Exception:
                pass
        if self._worker is not None:
            self._worker.join(timeout=3.0)
            self._worker = None
        with self._cache_lock:
            self._cache.clear()
            self._loading.clear()
        self._clear_audio()

    # ------------------------------------------------------------------ worker

    def _run(self) -> None:
        """Prime cache, open single stream, process section advances from callback."""
        start_idx = self._callback_sec_idx

        # Prime the 3-section window synchronously before playing the first note.
        for i in [start_idx - 1, start_idx, start_idx + 1]:
            if 0 <= i < len(self._sections) and not self._stop_event.is_set():
                self._load_into_cache(i)

        # Determine stream sample rate from the first section.
        audio, sr = self._get_from_cache_or_load(start_idx)
        if audio is None or self._stop_event.is_set():
            self._playing = False
            if self.on_stop:
                try:
                    self.on_stop()
                except Exception:
                    pass
            return

        self._stream_sr = sr
        with self._lock:
            self._callback_audio = audio
            self._callback_end_frame = len(audio)
            self._callback_frame = 0

        # Fire the initial section change notification.
        if self.on_section_change:
            try:
                self.on_section_change(start_idx)
            except Exception:
                pass

        done = threading.Event()

        def _finished():
            done.set()

        try:
            self._stream = sd.OutputStream(
                samplerate=self._stream_sr,
                channels=1,
                callback=self._callback,
                finished_callback=_finished,
                blocksize=1024,
            )
            self._stream.start()

            # Main loop: process section changes from the callback and manage
            # pre-loading/eviction. The callback handles playback sequencing.
            last_seen_idx = start_idx
            while not done.is_set() and not self._stop_event.is_set():
                # Drain any pending section changes from the callback.
                try:
                    while True:
                        idx = self._section_change_queue.get_nowait()
                        if idx != last_seen_idx:
                            last_seen_idx = idx
                            if self.on_section_change:
                                try:
                                    self.on_section_change(idx)
                                except Exception:
                                    pass
                        # Schedule pre-load of idx+2.
                        self._schedule_preload(idx + 2)
                        # Evict sections that have already played.
                        self._evict_through(max(0, idx - 1))
                except:
                    pass

                # Wait for stream to finish or stop to be signaled.
                if done.wait(timeout=0.1):
                    break

        except Exception as e:
            logger.error(f"Stream error: {e}")
        finally:
            try:
                if self._stream is not None:
                    self._stream.close()
            except Exception:
                pass
            self._stream = None

        self._playing = False
        if self.on_stop:
            try:
                self.on_stop()
            except Exception:
                pass


    # ------------------------------------------------------------------ cache management

    def _load_into_cache(self, idx: int) -> None:
        """Synchronously load section *idx* into the cache (no-op if already present)."""
        with self._cache_lock:
            if idx in self._cache or idx in self._loading:
                return
        sec = self._sections[idx]
        audio, sr = self._load_section_audio(sec)
        if audio is not None:
            with self._cache_lock:
                self._cache[idx] = (audio, sr)
            logger.info(f"Cached section {idx} ('{sec.name}')")

    def _get_from_cache_or_load(self, idx: int) -> tuple[np.ndarray | None, int]:
        """Return audio for *idx* from cache; load synchronously as fallback."""
        with self._cache_lock:
            if idx in self._cache:
                audio, sr = self._cache[idx]
                logger.info(f"Cache hit: section {idx} ('{self._sections[idx].name}')")
                return audio, sr
        # Pre-load worker may be in flight — wait by loading synchronously.
        self._load_into_cache(idx)
        with self._cache_lock:
            return self._cache.get(idx, (None, 0))

    def _schedule_preload(self, idx: int) -> None:
        """Start a background daemon thread to pre-load section *idx* into cache."""
        if idx < 0 or idx >= len(self._sections):
            return
        with self._cache_lock:
            if idx in self._cache or idx in self._loading:
                return
            self._loading.add(idx)
        t = threading.Thread(
            target=self._preload_worker,
            args=(idx,),
            daemon=True,
            name=f"MixPreload-{idx}",
        )
        t.start()

    def _preload_worker(self, idx: int) -> None:
        try:
            if self._stop_event.is_set():
                return
            sec = self._sections[idx]
            audio, sr = self._load_section_audio(sec)
            if audio is not None and not self._stop_event.is_set():
                with self._cache_lock:
                    self._cache[idx] = (audio, sr)
                logger.info(f"Pre-loaded section {idx} ('{sec.name}')")
        finally:
            with self._cache_lock:
                self._loading.discard(idx)

    def _evict_through(self, idx: int) -> None:
        """Remove cache entries for section indices ≤ *idx* (already played)."""
        with self._cache_lock:
            stale = [k for k in list(self._cache) if k <= idx]
            for k in stale:
                del self._cache[k]
        if stale:
            logger.debug(f"Evicted sections {stale} from cache")

    # ------------------------------------------------------------------ audio loading

    def _load_section_audio(self, sec: Section) -> tuple[np.ndarray | None, int]:
        """Load and slice the WAV for *sec*. Returns (audio_slice, sr) or (None, 0)."""
        if not sec.bars or not sec.bars[0].beats:
            logger.warning(f"Section '{sec.name}' has no bars/beats — skipping")
            return None, 0

        audiosource = sec.bars[0].audiosource
        if not audiosource:
            logger.error(f"Section '{sec.name}' has no audiosource")
            return None, 0

        src_path = Path(audiosource)
        if not src_path.exists():
            logger.error(f"Audio file not found for section '{sec.name}': {src_path}")
            return None, 0

        try:
            raw, sr = sf.read(str(src_path), always_2d=True)
            if raw.shape[1] > 1:
                raw = raw.mean(axis=1, keepdims=True)
            audio = raw.astype(np.float32)

            start_ms = sec.bars[0].beats[0].start_ms
            last_bar = sec.bars[-1]
            end_ms = last_bar.beats[-1].finish_ms if last_bar.beats else 0
            if end_ms <= start_ms:
                end_ms = int(len(audio) * 1000 / sr)

            start_f = max(0, int(round(start_ms * sr / 1000)))
            end_f = min(len(audio), int(round(end_ms * sr / 1000)))

            sliced = audio[start_f:end_f]
            dur_s = len(sliced) / sr
            logger.info(f"Loaded '{sec.name}': {start_ms}–{end_ms} ms ({dur_s:.2f}s, {len(sliced)} frames)")
            return sliced, sr

        except Exception as e:
            logger.error(f"Failed to load audio for '{sec.name}': {e}")
            return None, 0

    # ------------------------------------------------------------------ helpers

    def _resample(self, audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
        """Resample audio from sr_from to sr_to using linear interpolation."""
        if sr_from == sr_to:
            return audio
        ratio = sr_to / sr_from
        n_samples = int(len(audio) * ratio)
        if n_samples <= 0:
            return np.zeros((1, audio.shape[1]), dtype=np.float32)
        old_indices = np.linspace(0, len(audio) - 1, len(audio))
        new_indices = np.linspace(0, len(audio) - 1, n_samples)
        resampled = np.interp(new_indices, old_indices, audio[:, 0], left=0, right=0)
        return resampled.reshape(-1, 1).astype(np.float32)

    def _callback(self, outdata, frames, time_info, status) -> None:
        """Fill audio buffer, sequencing across sections without gaps."""
        if status:
            print(status, file=sys.stderr)

        if not self._playing:
            outdata[:] = 0
            raise sd.CallbackStop()

        filled = 0
        while filled < frames:
            if self._callback_audio is None:
                # Load the next section.
                if self._callback_sec_idx >= len(self._sections):
                    # No more sections: zero-fill the rest and stop.
                    outdata[filled:] = 0
                    raise sd.CallbackStop()

                next_audio, next_sr = self._get_from_cache_or_load(self._callback_sec_idx)
                if next_audio is None:
                    outdata[filled:] = 0
                    raise sd.CallbackStop()

                # Resample if needed to match stream rate.
                if next_sr != self._stream_sr:
                    next_audio = self._resample(next_audio, next_sr, self._stream_sr)

                with self._lock:
                    self._callback_audio = next_audio
                    self._callback_end_frame = len(next_audio)
                    self._callback_frame = 0

                # Notify worker of section change.
                self._section_change_queue.put(self._callback_sec_idx)
                self._callback_sec_idx += 1

            # Fill from current section.
            remaining = self._callback_end_frame - self._callback_frame
            take = min(frames - filled, remaining)
            if take > 0:
                outdata[filled : filled + take] = self._callback_audio[
                    self._callback_frame : self._callback_frame + take
                ]
                self._callback_frame += take
                filled += take

            # If we exhausted this section, prepare to load the next one (loop continues).
            if self._callback_frame >= self._callback_end_frame:
                with self._lock:
                    self._callback_audio = None
                    self._callback_end_frame = 0
                    self._callback_frame = 0

    def _clear_audio(self) -> None:
        with self._lock:
            self._callback_audio = None
            self._callback_frame = 0
            self._callback_end_frame = 0
