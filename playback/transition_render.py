"""Render beat-aligned mixed audio for transition sections.

A transition Section carries ``fade_out_bars`` (the outgoing song's tail) and
``fade_in_bars`` (the incoming song's head) but an empty ``bars`` list, so it
produces no audio. :func:`render_transition` overlays the two — beat by beat —
into a single mono WAV and returns the ``bars`` that point at it, so the existing
:class:`~playback.mix_engine.MixPlaybackEngine` can play it like any other
section.

Design:

* **Per-beat gain envelope** — out_gain fades 1→0, in_gain fades 0→1 across the
  beat count, summing to 1.0 so the mix never clips.
* **Per-beat time-stretching** — each beat's duration is interpolated between the
  outgoing and incoming natural durations (``target_dur_ms``), and both segments
  are time-stretched to match before mixing.
* **Sin² tapering** — 64-frame fade at beat boundaries eliminates waveform
  discontinuity clicks at every cut point.
* **Gapless boundary invariant** — beat 0 has out_gain=1 and target=out_dur, so
  the first beat is a bit-exact copy of the outgoing segment; beat N-1 has
  in_gain=1 and target=in_dur, handing off seamlessly to the next section.
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


def _stretch(audio: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    """Time-stretch *audio* so its length becomes ratio * original length.

    ratio = target_frames / source_frames. Skipped when ratio ≈ 1.
    Tries pyrubberband first, falls back to librosa.
    """
    if abs(ratio - 1.0) < 0.001 or len(audio) == 0:
        return audio
    # time_stretch(rate) uses speed convention: >1 is faster (shorter output).
    speed = 1.0 / ratio
    mono = audio[:, 0]
    try:
        import pyrubberband as pyrb
        stretched = pyrb.time_stretch(mono, sr, speed)
    except Exception:
        import librosa
        stretched = librosa.effects.time_stretch(mono.astype(np.float32), rate=speed)
    return stretched.reshape(-1, 1).astype(np.float32)


def _taper(seg: np.ndarray, taper_frames: int = 64) -> np.ndarray:
    """Apply sin²-ramp fade to the first and last *taper_frames* samples."""
    n = len(seg)
    if n == 0 or taper_frames == 0:
        return seg
    tf = min(taper_frames, n // 2)
    result = seg.copy()
    ramp = np.sin(np.linspace(0.0, np.pi / 2, tf, dtype=np.float32)) ** 2
    result[:tf] *= ramp.reshape(-1, 1)
    result[-tf:] *= ramp[::-1].reshape(-1, 1)
    return result


def _ensure_frames(seg: np.ndarray, n_frames: int) -> np.ndarray:
    """Crop or zero-pad *seg* to exactly *n_frames*."""
    if len(seg) >= n_frames:
        return seg[:n_frames]
    pad = np.zeros((n_frames - len(seg), seg.shape[1]), dtype=np.float32)
    return np.concatenate([seg, pad], axis=0)


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

    sr = in_sr
    if out_sr != in_sr:
        try:
            import librosa
            out_audio = librosa.resample(
                out_audio[:, 0], orig_sr=out_sr, target_sr=in_sr
            ).reshape(-1, 1).astype(np.float32)
            logger.info(f"Resampled fade-out audio from {out_sr} to {in_sr} Hz")
        except Exception as exc:
            logger.warning(
                f"Transition '{section.name}': resample failed ({exc}); "
                f"proceeding without resample — timing will be off"
            )

    out_beats = _flatten_beats(section.fade_out_bars)
    in_beats = _flatten_beats(section.fade_in_bars)
    n = min(len(out_beats), len(in_beats))
    if n == 0:
        logger.warning(f"Transition '{section.name}' has no paired beats")
        return []

    mixed_segments: list[np.ndarray] = []
    slot_frames: list[int] = []

    for i in range(n):
        in_gain = (i / (n - 1)) if n > 1 else 0.0
        out_gain = 1.0 - in_gain

        out_dur_ms = out_beats[i].finish_ms - out_beats[i].start_ms
        in_dur_ms = in_beats[i].finish_ms - in_beats[i].start_ms
        target_dur_ms = out_dur_ms + (in_dur_ms - out_dur_ms) * in_gain
        target_frames = _ms_to_frames(target_dur_ms, sr)
        if target_frames <= 0:
            continue

        out_seg_raw = _segment(out_audio, out_beats[i].start_ms, _ms_to_frames(out_dur_ms, sr), sr)
        in_seg_raw = _segment(in_audio, in_beats[i].start_ms, _ms_to_frames(in_dur_ms, sr), sr)

        if out_gain > 0.01:
            out_seg = _ensure_frames(
                _stretch(out_seg_raw, sr, target_frames / max(len(out_seg_raw), 1)),
                target_frames,
            )
        else:
            out_seg = np.zeros((target_frames, out_audio.shape[1]), dtype=np.float32)

        if in_gain > 0.01:
            in_seg = _ensure_frames(
                _stretch(in_seg_raw, sr, target_frames / max(len(in_seg_raw), 1)),
                target_frames,
            )
        else:
            in_seg = np.zeros((target_frames, in_audio.shape[1]), dtype=np.float32)

        out_seg = _taper(out_seg)
        in_seg = _taper(in_seg)
        mixed = out_seg * out_gain + in_seg * in_gain
        mixed_segments.append(mixed)
        slot_frames.append(target_frames)

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

    audiosource = str(wav_path)
    bars: list[Bar] = []
    beat_i = 0
    cursor_ms = 0.0
    for bar in section.fade_in_bars:
        new_beats: list[Beat] = []
        for beat in bar.beats:
            if beat_i >= len(slot_frames):
                break
            dur_ms = slot_frames[beat_i] * 1000 / sr
            new_beats.append(
                Beat(
                    start_ms=cursor_ms,
                    finish_ms=cursor_ms + dur_ms,
                    position=beat.position,
                )
            )
            cursor_ms += dur_ms
            beat_i += 1
        if new_beats:
            bars.append(
                Bar(idx=bar.idx, beats=tuple(new_beats), audiosource=audiosource, color=bar.color)
            )
    return bars
