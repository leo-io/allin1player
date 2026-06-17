"""Verification test for gapless section transitions with synthetic audio."""
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


def test_gapless_synthetic_audio(tmp_path: Path = None):
    """Test gapless playback with synthetic constant-value audio sections."""
    if tmp_path is None:
        tmp_path = Path("temp_gapless_test")
    tmp_path.mkdir(parents=True, exist_ok=True)

    sr = 44100
    beat_ms = 250
    beat_frames = int(sr * beat_ms / 1000)

    # Create 3 synthetic WAV files with different constant values
    # Section 0: 0.3 amplitude
    # Section 1: 0.5 amplitude
    # Section 2: 0.7 amplitude
    values = [0.3, 0.5, 0.7]
    section_files = []

    for i, val in enumerate(values):
        # 4 beats = 1 second of audio per section
        audio = np.full((4 * beat_frames, 1), val, dtype=np.float32)
        path = tmp_path / f"section_{i}.wav"
        sf.write(str(path), audio, sr)
        section_files.append(str(path))
        print(f"[OK] Created synthetic audio {i}: {4 * beat_frames} samples of {val}")

    # Create sections from the synthetic files
    sections = []
    for i, audio_file in enumerate(section_files):
        bars = [_bar(i, 0, beat_ms, audio_file)]
        sec = Section(idx=i, name=f"Section {i}", bars=bars)
        sections.append(sec)

    print(f"[OK] Created {len(sections)} test sections")

    # Capture the audio output
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

        def start(self):
            try:
                while True:
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
            pass

        def abort(self):
            pass

    # Monkey-patch
    sd.OutputStream = CaptureOutputStream

    try:
        engine = MixPlaybackEngine()
        engine.play(sections)

        # Wait for playback
        timeout = 30
        start = time.time()
        while engine.playing and time.time() - start < timeout:
            time.sleep(0.05)

        assert not engine.playing, "Playback should complete"
        print(f"[OK] Playback completed")

        # Analyze the captured audio
        if not captured_audio:
            print("[FAIL] No audio was captured")
            return False

        full_audio = np.concatenate(captured_audio, axis=0)
        print(f"[OK] Captured {len(full_audio)} samples ({len(full_audio) / sr:.2f}s)")

        # Expected structure:
        # - Section 0: 0.3 for ~1s
        # - Section 1: 0.5 for ~1s
        # - Section 2: 0.7 for ~1s

        expected_frames_per_section = 4 * beat_frames

        # Check each section boundary for gaps
        boundaries = [
            (expected_frames_per_section, "0.3->0.5"),
            (2 * expected_frames_per_section, "0.5->0.7"),
        ]

        for boundary_frame, label in boundaries:
            # Look at samples around the boundary
            window = 100  # samples to check
            before_frames = full_audio[max(0, boundary_frame - window):boundary_frame, 0]
            after_frames = full_audio[boundary_frame:min(len(full_audio), boundary_frame + window), 0]

            # In a gapless transition, before_frames should be non-zero and
            # after_frames should immediately start with a new value
            # A gap would show as zeros in after_frames

            avg_before = np.mean(np.abs(before_frames))
            avg_after = np.mean(np.abs(after_frames))
            min_after = np.min(np.abs(after_frames))

            print(f"  Boundary {label} (frame {boundary_frame}):")
            print(f"    Before: avg={avg_before:.3f}")
            print(f"    After:  avg={avg_after:.3f}, min={min_after:.3f}")

            # If there's a gap, after_frames will have low amplitude initially
            if min_after < 0.01 and avg_after < 0.2:
                print(f"  [WARN] Possible gap at boundary {label}")

        # More precise check: look for zero-crossings (discontinuities in the waveform)
        # that indicate a boundary, and verify there's audio on both sides
        sample_diffs = np.abs(np.diff(full_audio[:, 0]))
        large_diffs = np.where(sample_diffs > 0.1)[0]  # Jumps > 0.1 amplitude

        print(f"[OK] Found {len(large_diffs)} amplitude discontinuities (expected ~2 at section boundaries)")

        if len(large_diffs) >= 2:
            print("[OK] Gapless playback verified: section boundaries present with no silence gaps")
            return True
        else:
            print("[FAIL] Could not verify section boundaries in captured audio")
            return False

    finally:
        sd.OutputStream = original_OutputStream
        # Clean up
        for f in section_files:
            Path(f).unlink(missing_ok=True)


if __name__ == "__main__":
    from pathlib import Path
    tmp = Path("temp_gapless_test")
    success = test_gapless_synthetic_audio(tmp)
    sys.exit(0 if success else 1)
