"""Render beat-aligned mixed audio for transition sections.

A transition Section carries ``fade_out_bars`` (the outgoing song's tail) and
``fade_in_bars`` (the incoming song's head) but an empty ``bars`` list, so it
produces no audio. :func:`render_transition` overlays the two — beat by beat —
into a single mono WAV and returns the ``bars`` that point at it, so the existing
:class:`~playback.mix_engine.MixPlaybackEngine` can play it like any other
section.

Design (see the project plan):

* **Per-beat re-anchor** — for each beat index, the fade-out beat's audio is
  placed at the start of the matching fade-in beat slot and summed, so every beat
  stays downbeat-aligned across the whole transition.
* **Fade-in is the master clock** — each mixed beat keeps the incoming song's
  natural duration, so the fade-in component is a bit-exact copy of the incoming
  audio and hands off seamlessly to the following section.
* No gain automation and no time-stretching (out of scope for now). The summed
  signal is only clipped to ``[-1, 1]`` to guard against overflow.
"""
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from domain.models import Bar, Beat, Section

logger = logging.getLogger(__name__)


def _ms_to_frames(ms: float, sr: int) -> int:
    """Convert milliseconds to a frame index using the project-wide convention."""
    return int(round(float(ms) * sr / 1000))


def _load_mono(audiosource: str) -> tuple[np.ndarray | None, int]:
    """Load *audiosource* as mono float32. Returns (audio, sr) or (None, 0)."""
    src_path = Path(audiosource)
    if not src_path.exists():
        logger.error(f"Transition audio source not found: {src_path}")
        return None, 0
    raw, sr = sf.read(str(src_path), always_2d=True)
    if raw.shape[1] > 1:
        raw = raw.mean(axis=1, keepdims=True)
    return raw.astype(np.float32), sr


def _flatten_beats(bars: list[Bar]) -> list[Beat]:
    return [beat for bar in bars for beat in bar.beats]


def _segment(audio: np.ndarray, start_ms: float, n_frames: int, sr: int) -> np.ndarray:
    """Slice *n_frames* from *audio* starting at *start_ms*, zero-padded if short."""
    start_f = max(0, _ms_to_frames(start_ms, sr))
    seg = audio[start_f : start_f + n_frames]
    if len(seg) < n_frames:
        pad = np.zeros((n_frames - len(seg), audio.shape[1]), dtype=np.float32)
        seg = np.concatenate([seg, pad], axis=0)
    return seg


def render_transition(section: Section, out_dir: Path, sr: int = 44100) -> list[Bar]:
    """Render *section*'s overlaid transition audio and return its ``bars``.

    Writes ``transitions/transition_<idx>.wav`` under *out_dir* and builds bars
    that mirror the fade-in grouping, referencing the rendered file. Returns an
    empty list (and renders nothing) when the section lacks usable fade bars.
    """
    if not section.fade_out_bars or not section.fade_in_bars:
        logger.warning(f"Transition '{section.name}' missing fade bars — nothing to render")
        return []

    out_src = section.fade_out_bars[0].audiosource
    in_src = section.fade_in_bars[0].audiosource
    out_audio, out_sr = _load_mono(out_src)
    in_audio, in_sr = _load_mono(in_src)
    if out_audio is None or in_audio is None:
        return []

    # Master clock is the incoming song; align everything to its sample rate.
    sr = in_sr
    if out_sr != in_sr:
        logger.warning(
            f"Transition '{section.name}': sample-rate mismatch "
            f"(fade-out {out_sr} vs fade-in {in_sr}); using fade-in rate, no resample"
        )

    out_beats = _flatten_beats(section.fade_out_bars)
    in_beats = _flatten_beats(section.fade_in_bars)
    n = min(len(out_beats), len(in_beats))
    if n == 0:
        logger.warning(f"Transition '{section.name}' has no paired beats")
        return []

    # Mixed audio: per-beat overlay, fade-in slot durations as the master grid.
    mixed_segments: list[np.ndarray] = []
    slot_frames: list[int] = []
    for i in range(n):
        in_beat = in_beats[i]
        out_beat = out_beats[i]
        n_frames = _ms_to_frames(in_beat.finish_ms - in_beat.start_ms, sr)
        if n_frames <= 0:
            continue
        in_seg = _segment(in_audio, in_beat.start_ms, n_frames, sr)
        out_seg = _segment(out_audio, out_beat.start_ms, n_frames, sr)
        mixed = np.clip(in_seg + out_seg, -1.0, 1.0)
        mixed_segments.append(mixed)
        slot_frames.append(n_frames)

    if not mixed_segments:
        logger.warning(f"Transition '{section.name}' produced no audio")
        return []

    buffer = np.concatenate(mixed_segments, axis=0)

    out_dir = Path(out_dir) / "transitions"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"transition_{section.idx}.wav"
    sf.write(str(wav_path), buffer, sr)
    logger.info(
        f"Rendered transition '{section.name}': {len(buffer)} frames "
        f"({len(buffer) / sr:.2f}s) -> {wav_path}"
    )

    # Build bars mirroring the fade-in grouping; beats are contiguous from ms 0.
    audiosource = str(wav_path)
    bars: list[Bar] = []
    beat_i = 0
    cursor_ms = 0
    for bar in section.fade_in_bars:
        new_beats: list[Beat] = []
        for beat in bar.beats:
            if beat_i >= len(slot_frames):
                break
            dur_ms = round(slot_frames[beat_i] * 1000 / sr)
            new_beats.append(
                Beat(start_ms=cursor_ms, finish_ms=cursor_ms + dur_ms, position=beat.position)
            )
            cursor_ms += dur_ms
            beat_i += 1
        if new_beats:
            bars.append(
                Bar(idx=bar.idx, beats=tuple(new_beats), audiosource=audiosource, color=bar.color)
            )
    return bars
