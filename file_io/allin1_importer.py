from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from domain.models import Bar, Section, SongStructure


class Allin1Importer:
    @staticmethod
    def load(path: str | Path) -> SongStructure:
        raw = Allin1Importer._load_raw(path)
        return Allin1Importer._flatten(raw)

    @staticmethod
    def _load_raw(path: str | Path) -> dict:
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _build_bar_list(data: dict) -> list[list[dict]]:
        all_bars = []
        current = []
        for t, num in zip(data["beats"], data["beat_positions"]):
            if num == 1 and current:
                all_bars.append(current)
                current = []
            current.append({"time_ms": int(round(t * 1000)), "beat": int(num)})
        if current:
            all_bars.append(current)
        return all_bars

    @staticmethod
    def _build_sections(data: dict, all_bars: list[list[dict]]) -> list[dict]:
        sections = []
        label_counts = {}
        for seg in data.get("segments", []):
            seg_start_ms = seg["start"] * 1000
            seg_end_ms = seg["end"] * 1000
            seg_bars = [b for b in all_bars if seg_start_ms <= b[0]["time_ms"] < seg_end_ms]
            if not seg_bars:
                continue
            label = seg["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
            sections.append({"name": f"{label} {label_counts[label]}", "bars": seg_bars})

        if not sections and all_bars:
            sections = [{"name": "track", "bars": all_bars}]

        for sec in sections:
            label = sec["name"].rsplit(" ", 1)[0]
            if label_counts.get(label, 0) == 1:
                sec["name"] = label

        return sections

    @staticmethod
    def _flatten(data: dict) -> SongStructure:
        raw_bars = Allin1Importer._build_bar_list(data)
        sections_data = Allin1Importer._build_sections(data, raw_bars)

        times_ms = []
        numbers = []
        bars = []
        section_list = []
        for sec_idx, sec in enumerate(sections_data):
            sec_bar_start = len(bars)
            for bar_data in sec["bars"]:
                start_idx = len(times_ms)
                for beat in bar_data:
                    times_ms.append(beat["time_ms"])
                    numbers.append(beat["beat"])
                bars.append(Bar(
                    idx=len(bars),
                    start_beat_idx=start_idx,
                    n_beats=len(bar_data),
                    beat_numbers=tuple(b["beat"] for b in bar_data),
                ))
            section_list.append(Section(
                idx=sec_idx,
                name=sec["name"],
                first_bar=sec_bar_start,
                end_bar=len(bars),
            ))

        return SongStructure(
            beat_times_ms=np.array(times_ms, dtype=np.int64),
            beat_numbers=np.array(numbers, dtype=int),
            bars=bars,
            sections=section_list,
        )
