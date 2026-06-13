from __future__ import annotations
from pathlib import Path

from domain.models import Arrangement, Beat, Bar, Section


class ChordLoader:
    @staticmethod
    def assign_chords(arrangement: Arrangement, chord_path: str | Path) -> None:
        chords = ChordLoader._load_chords(chord_path)
        if not chords:
            return

        new_sections = []
        chord_idx = 0
        for sec in arrangement.sections:
            new_bars = []
            for bar in sec.bars:
                new_beats = []
                for beat in bar.beats:
                    while chord_idx < len(chords) and beat.time_ms >= chords[chord_idx][1]:
                        chord_idx += 1
                    chord_name = ""
                    if chord_idx < len(chords) and chords[chord_idx][0] <= beat.time_ms < chords[chord_idx][1]:
                        chord_name = chords[chord_idx][2]
                    new_beats.append(Beat(
                        time_ms=beat.time_ms,
                        position=beat.position,
                        chord=chord_name,
                    ))
                new_bars.append(Bar(idx=bar.idx, beats=tuple(new_beats)))
            new_sections.append(Section(idx=sec.idx, name=sec.name, bars=new_bars))

        arrangement.sections[:] = new_sections

    @staticmethod
    def _load_chords(chord_path: str | Path) -> list[tuple]:
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
                start_ms = float(parts[0]) * 1000.0
                end_ms = float(parts[1]) * 1000.0
                name = parts[2]
                chords.append((start_ms, end_ms, name))

        return chords
