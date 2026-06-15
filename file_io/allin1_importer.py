from __future__ import annotations
import json
import logging
import time
from pathlib import Path
import numpy as np

from domain.models import Arrangement, Bar, Section, Beat

logger = logging.getLogger(__name__)


class Allin1Importer:
    @staticmethod
    def load(path: str | Path) -> Arrangement:
        path = Path(path)
        logger.info(f"Loading allin1 analysis JSON: {path}")
        try:
            raw = Allin1Importer._load_raw(path)
            logger.debug(f"Loaded raw data with {len(raw.get('beats', []))} beats, {len(raw.get('segments', []))} segments")
            arrangement = Allin1Importer._flatten(raw, path)
            logger.info(f"Successfully created master arrangement '{arrangement.name}' with {len(arrangement.sections)} sections, {len(arrangement.bars)} bars")
            return arrangement
        except Exception as e:
            logger.error(f"Failed to load allin1 analysis from {path}: {e}", exc_info=True)
            raise

    @staticmethod
    def _load_raw(path: str | Path) -> dict:
        path = Path(path)
        logger.debug(f"Reading JSON file: {path}")
        try:
            with open(path) as f:
                data = json.load(f)
            logger.debug(f"Successfully parsed JSON, file size info: beats={len(data.get('beats', []))}, beat_positions={len(data.get('beat_positions', []))}, segments={len(data.get('segments', []))}")
            return data
        except FileNotFoundError as e:
            logger.error(f"Analysis file not found: {path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {path}: {e}")
            raise

    @staticmethod
    def _build_bar_list(data: dict) -> list[list[dict]]:
        logger.debug("Building bar list from beats and beat_positions")
        all_bars = []
        current = []
        beats_count = len(data.get("beats", []))
        logger.debug(f"Processing {beats_count} beats")

        for t, num in zip(data.get("beats", []), data.get("beat_positions", [])):
            if num == 1 and current:
                all_bars.append(current)
                current = []
            current.append({"time_ms": int(round(t * 1000)), "beat": int(num)})
        if current:
            all_bars.append(current)

        logger.debug(f"Built {len(all_bars)} bars from {beats_count} beats")
        return all_bars

    @staticmethod
    def _build_sections(data: dict, all_bars: list[list[dict]]) -> list[dict]:
        logger.debug(f"Building sections from {len(data.get('segments', []))} segments and {len(all_bars)} bars")
        sections = []
        label_counts = {}
        segment_count = 0

        for seg in data.get("segments", []):
            seg_start_ms = seg["start"] * 1000
            seg_end_ms = seg["end"] * 1000
            seg_bars = [b for b in all_bars if seg_start_ms <= b[0]["time_ms"] < seg_end_ms]
            if not seg_bars:
                logger.debug(f"Segment '{seg.get('label', 'unknown')}' ({seg['start']:.2f}s-{seg['end']:.2f}s) has no bars, skipping")
                continue
            label = seg["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
            section_name = f"{label} {label_counts[label]}"
            sections.append({"name": section_name, "bars": seg_bars})
            logger.debug(f"Created section '{section_name}' with {len(seg_bars)} bars")
            segment_count += 1

        if not sections and all_bars:
            logger.warning(f"No sections found from segments, creating default 'track' section with {len(all_bars)} bars")
            sections = [{"name": "track", "bars": all_bars}]

        for sec in sections:
            label = sec["name"].rsplit(" ", 1)[0]
            if label_counts.get(label, 0) == 1:
                old_name = sec["name"]
                sec["name"] = label
                logger.debug(f"Renamed section '{old_name}' → '{label}' (only occurrence)")

        logger.debug(f"Section building complete: {len(sections)} sections")
        return sections

    @staticmethod
    def _flatten(data: dict, path: str | Path) -> Arrangement:
        logger.debug("Flattening raw data into domain objects")
        raw_bars = Allin1Importer._build_bar_list(data)
        sections_data = Allin1Importer._build_sections(data, raw_bars)

        section_list = []
        bar_idx = 0
        total_beats = 0

        for sec_idx, sec in enumerate(sections_data):
            bars = []
            for bar_data in sec["bars"]:
                beats = tuple(
                    Beat(time_ms=beat["time_ms"], position=beat["beat"])
                    for beat in bar_data
                )
                bars.append(Bar(idx=bar_idx, beats=beats))
                total_beats += len(beats)
                bar_idx += 1
            section_list.append(Section(idx=sec_idx, name=sec["name"], bars=bars))
            logger.debug(f"Section {sec_idx} '{sec['name']}': {len(bars)} bars, {sum(len(b.beats) for b in bars)} beats")

        file_id = Path(path).stem
        logger.debug(f"Creating master arrangement with file_id='{file_id}'")
        arrangement = Arrangement(
            name=f"{file_id}-master",
            master=True,
            sections=section_list,
            creationdate=int(time.time() * 1000),
        )
        arrangement.reindex(sr=1)
        logger.debug(f"Arrangement flattened: {len(section_list)} sections, {bar_idx} bars, {total_beats} beats")
        return arrangement
