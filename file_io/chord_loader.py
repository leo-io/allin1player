from __future__ import annotations
from pathlib import Path

import numpy as np

from domain.models import Chord, SongStructure


class ChordLoader:
    @staticmethod
    def load(
        chord_path: str | Path,
        song: SongStructure,
    ) -> list[Chord]:
        chords = []
        p = Path(chord_path)
        if not p.exists():
            return chords

        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                start_s = float(parts[0])
                end_s = float(parts[1])
                name = parts[2]
                chords.append(Chord(
                    start_ms=start_s * 1000.0,
                    end_ms=end_s * 1000.0,
                    name=name,
                ))

        return chords

    @staticmethod
    def compute_beat_chords(
        chords: list[Chord],
        beat_times_ms: np.ndarray,
    ) -> list[str]:
        beat_chords: list[str] = []
        chord_idx = 0
        for bt in beat_times_ms:
            while chord_idx < len(chords) and bt >= chords[chord_idx].end_ms:
                chord_idx += 1
            if chord_idx < len(chords) and chords[chord_idx].start_ms <= bt < chords[chord_idx].end_ms:
                beat_chords.append(chords[chord_idx].name)
            else:
                beat_chords.append("")
        return beat_chords
