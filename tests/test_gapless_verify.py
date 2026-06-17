"""Verification test for gapless playback with real audio files."""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domain.models import Bar, Beat, Section  # noqa: E402
from playback.mix_engine import MixPlaybackEngine  # noqa: E402


def _bar(idx: int, start_ms: int, beat_ms: int, source: str) -> Bar:
    beats = tuple(
        Beat(start_ms=start_ms + i * beat_ms, finish_ms=start_ms + (i + 1) * beat_ms, position=i + 1)
        for i in range(4)
    )
    return Bar(idx=idx, beats=beats, audiosource=source, color="#000000")


def test_gapless_playback_real_audio():
    """Test gapless playback with real audio from test mix."""
    audio_path = Path("music/20260612_163540-sleeping/20260612_163540-sleeping.wav")

    if not audio_path.exists():
        print(f"[SKIP] Audio file not found at {audio_path}")
        return

    # Load the audio to understand its properties
    raw, sr = sf.read(str(audio_path), always_2d=True)
    if raw.shape[1] > 1:
        raw = raw.mean(axis=1, keepdims=True)
    audio = raw.astype(np.float32)

    print(f"[OK] Loaded audio: {len(audio)} samples at {sr} Hz ({len(audio) / sr:.1f}s)")

    # Create 3 sections from different parts of the same audio
    sections = []
    section_starts_ms = [0, 2080, 4160]  # milliseconds into the audio

    for i, start_ms in enumerate(section_starts_ms):
        bars = [_bar(i, start_ms, 520, str(audio_path))]
        sec = Section(idx=i, name=f"Section {i}", bars=bars)
        sections.append(sec)

    print(f"[OK] Created {len(sections)} test sections")

    # Create a custom output sink to capture the audio
    captured_audio = []

    import sounddevice as sd
    original_OutputStream = sd.OutputStream

    class CaptureOutputStream:
        def __init__(self, samplerate, channels, callback, finished_callback, blocksize):
            self.samplerate = samplerate
            self.channels = channels
            self.callback = callback
            self.finished_callback = finished_callback
            self.blocksize = blocksize
            self.running = False

        def start(self):
            self.running = True
            try:
                while self.running:
                    outdata = np.zeros((self.blocksize, self.channels), dtype=np.float32)
                    try:
                        self.callback(outdata, self.blocksize, None, None)
                        captured_audio.append(outdata.copy())
                    except Exception:
                        break
            finally:
                if self.finished_callback:
                    self.finished_callback()

        def close(self):
            self.running = False

        def abort(self):
            self.running = False

    # Monkey-patch to use our capture stream
    sd.OutputStream = CaptureOutputStream

    try:
        engine = MixPlaybackEngine()
        engine.play(sections)

        # Wait for playback to complete
        timeout = 30
        start = time.time()
        while engine.playing and time.time() - start < timeout:
            time.sleep(0.1)

        assert not engine.playing, "Playback should complete"
        print(f"[OK] Playback completed successfully")

        # Concatenate captured audio
        if captured_audio:
            full_audio = np.concatenate(captured_audio, axis=0)
            print(f"[OK] Captured {len(full_audio)} samples ({len(full_audio) / sr:.2f}s)")

            # Check for gaps: look for sequences of samples near zero at section boundaries
            # A gap would show as a block of near-zero samples between sections
            threshold = 0.01  # Amplitude threshold for "silence"
            silence_threshold = int(sr * 0.05)  # 50ms of near-silence

            is_silent = np.abs(full_audio[:, 0]) < threshold
            silent_runs = np.where(np.diff(np.concatenate(([False], is_silent, [False])).astype(int)) != 0)[0]

            gap_found = False
            for i in range(1, len(silent_runs), 2):
                silence_duration = silent_runs[i] - silent_runs[i - 1]
                if silence_duration > silence_threshold:
                    print(f"[WARN] Found silence gap of {silence_duration} samples ({silence_duration / sr:.3f}s) at sample {silent_runs[i - 1]}")
                    gap_found = True

            if not gap_found:
                print("[OK] No significant silence gaps detected between sections")
            else:
                print("[FAIL] Audio gaps detected")
                return False

    finally:
        sd.OutputStream = original_OutputStream

    return True


if __name__ == "__main__":
    success = test_gapless_playback_real_audio()
    sys.exit(0 if success is not False else 1)
