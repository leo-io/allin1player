"""Tests for playback.transition_render.render_transition."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domain.models import Bar, Beat, Section  # noqa: E402
from playback.transition_render import render_transition, _ms_to_frames  # noqa: E402

SR = 44100


def _bar(idx: int, start_ms: int, beat_ms: int, source: str) -> Bar:
    beats = tuple(
        Beat(start_ms=start_ms + i * beat_ms, finish_ms=start_ms + (i + 1) * beat_ms, position=i + 1)
        for i in range(4)
    )
    return Bar(idx=idx, beats=beats, audiosource=source, color="#000000")


def _write_const_wav(path: Path, value: float, seconds: float) -> None:
    sf.write(str(path), np.full((int(SR * seconds), 1), value, dtype=np.float32), SR)


def test_render_transition(tmp_path: Path):
    out_src = str(tmp_path / "out.wav")
    in_src = str(tmp_path / "in.wav")
    _write_const_wav(Path(out_src), 0.2, 30.0)
    _write_const_wav(Path(in_src), 0.5, 30.0)

    # fade-out beats are 480ms, fade-in beats are 500ms — different grids.
    fade_out = [_bar(160 + i, 1000 + i * 4 * 480, 480, out_src) for i in range(2)]
    fade_in = [_bar(i, 2000 + i * 4 * 500, 500, in_src) for i in range(2)]
    sec = Section(idx=33, name="Transition - 1", bars=[], is_transition=True,
                  fade_out_bars=fade_out, fade_in_bars=fade_in)

    bars = render_transition(sec, tmp_path, sr=SR)

    # bars mirror the fade-in grouping
    assert len(bars) == 2
    total_beats = sum(len(b.beats) for b in bars)
    assert total_beats == 8

    wav_path = tmp_path / "transitions" / "transition_33.wav"
    assert wav_path.exists()
    assert bars[0].audiosource == str(wav_path)

    rendered, sr = sf.read(str(wav_path), always_2d=True)
    assert sr == SR

    # No NaN or inf in the output
    assert not np.any(np.isnan(rendered))
    assert not np.any(np.isinf(rendered))

    # Total frame count = sum of per-beat target durations
    # target_ms[i] = 480 + (500 - 480) * (i / 7) for i in 0..7
    n_beats = 8
    out_beat_ms, in_beat_ms = 480, 500
    expected_frames = sum(
        _ms_to_frames(out_beat_ms + (in_beat_ms - out_beat_ms) * (i / (n_beats - 1)), SR)
        for i in range(n_beats)
    )
    assert len(rendered) == expected_frames

    # Beat 0: out_gain=1.0, in_gain=0.0 → audio dominated by fade-out value (0.2)
    # Sample the middle of beat 0 (far from taper edges)
    beat0_frames = _ms_to_frames(out_beat_ms, SR)  # target_dur for beat 0
    mid0 = beat0_frames // 2
    assert np.allclose(rendered[mid0 - 50 : mid0 + 50], 0.2, atol=0.05)

    # Beat 7: out_gain=0.0, in_gain=1.0 → audio dominated by fade-in value (0.5)
    beat7_frames = _ms_to_frames(in_beat_ms, SR)  # target_dur for beat 7
    mid7 = len(rendered) - beat7_frames // 2
    assert np.allclose(rendered[mid7 - 50 : mid7 + 50], 0.5, atol=0.05)
