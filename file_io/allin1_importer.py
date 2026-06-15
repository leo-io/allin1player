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
        logger.info(f"[CHECKPOINT] Entering load() with path={path}")
        try:
            logger.info(f"Loading allin1 analysis JSON: {path}")
            raw = Allin1Importer._load_raw(path)
            logger.debug(f"[CHECKPOINT] Raw JSON loaded: beats={len(raw.get('beats', []))}, segments={len(raw.get('segments', []))}")

            arrangement = Allin1Importer._flatten(raw, path)
            logger.info(f"Successfully created master arrangement '{arrangement.name}' with {len(arrangement.sections)} sections, {len(arrangement.bars)} bars")
            logger.info(f"[CHECKPOINT] Exiting load() successfully with arrangement name='{arrangement.name}'")
            return arrangement
        except FileNotFoundError as e:
            error_msg = f"Allin1 analysis file not found: {path}"
            logger.error(f"[EXCEPTION] {error_msg} | Details: {e}")
            raise FileNotFoundError(error_msg) from e
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON format in allin1 analysis {path} at line {e.lineno}, col {e.colno}: {e.msg}"
            logger.error(f"[EXCEPTION] {error_msg} | Details: {e}")
            raise ValueError(error_msg) from e
        except KeyError as e:
            error_msg = f"Missing required field in allin1 JSON: {e}"
            logger.error(f"[EXCEPTION] {error_msg} | Details: {e}")
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error loading allin1 analysis from {path}: {type(e).__name__}: {e}"
            logger.error(f"[EXCEPTION] {error_msg}", exc_info=True)
            raise

    @staticmethod
    def _load_raw(path: str | Path) -> dict:
        path = Path(path)
        logger.debug(f"[CHECKPOINT] Entering _load_raw() with path={path}")
        try:
            logger.debug(f"Reading JSON file: {path}")
            if not path.exists():
                raise FileNotFoundError(f"File does not exist: {path}")

            file_size = path.stat().st_size
            logger.debug(f"File size: {file_size} bytes")

            with open(path, encoding='utf-8') as f:
                data = json.load(f)

            beats_count = len(data.get('beats', []))
            beat_positions_count = len(data.get('beat_positions', []))
            segments_count = len(data.get('segments', []))
            logger.debug(f"[CHECKPOINT] JSON parsed: beats={beats_count}, beat_positions={beat_positions_count}, segments={segments_count}")

            if beats_count != beat_positions_count:
                logger.warning(f"Data mismatch: beats ({beats_count}) != beat_positions ({beat_positions_count})")

            logger.debug(f"[CHECKPOINT] Exiting _load_raw() successfully")
            return data
        except FileNotFoundError as e:
            error_msg = f"Analysis file not found: {path}"
            logger.error(f"[EXCEPTION] {error_msg} | Details: {e}")
            raise FileNotFoundError(error_msg) from e
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in {path} (line {e.lineno}, col {e.colno}): {e.msg}"
            logger.error(f"[EXCEPTION] {error_msg}")
            raise ValueError(error_msg) from e
        except IOError as e:
            error_msg = f"IO error reading {path}: {e}"
            logger.error(f"[EXCEPTION] {error_msg}")
            raise IOError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error in _load_raw(): {type(e).__name__}: {e}"
            logger.error(f"[EXCEPTION] {error_msg}", exc_info=True)
            raise

    @staticmethod
    def _build_bar_list(data: dict) -> list[list[dict]]:
        logger.debug("[CHECKPOINT] Entering _build_bar_list()")
        try:
            all_bars = []
            current = []
            beats_count = len(data.get("beats", []))
            beat_positions_count = len(data.get("beat_positions", []))
            logger.debug(f"Processing {beats_count} beats, {beat_positions_count} beat_positions")

            if beats_count == 0 or beat_positions_count == 0:
                logger.warning("No beats or beat_positions data found")
                logger.debug("[CHECKPOINT] Exiting _build_bar_list() with empty result")
                return all_bars

            for idx, (t, num) in enumerate(zip(data.get("beats", []), data.get("beat_positions", []))):
                try:
                    time_ms = int(round(float(t) * 1000))
                    beat_num = int(num)
                    if beat_num == 1 and current:
                        all_bars.append(current)
                        current = []
                    current.append({"time_ms": time_ms, "beat": beat_num})
                except (ValueError, TypeError) as e:
                    error_msg = f"Invalid beat data at index {idx}: time={t}, position={num}"
                    logger.error(f"[EXCEPTION] {error_msg} | Error: {e}")
                    raise ValueError(error_msg) from e

            if current:
                all_bars.append(current)

            logger.debug(f"[CHECKPOINT] Built {len(all_bars)} bars from {beats_count} beats")
            logger.debug(f"[CHECKPOINT] Exiting _build_bar_list() with {len(all_bars)} bars")
            return all_bars
        except Exception as e:
            error_msg = f"Error building bar list: {type(e).__name__}: {e}"
            logger.error(f"[EXCEPTION] {error_msg}")
            raise

    @staticmethod
    def _build_sections(data: dict, all_bars: list[list[dict]]) -> list[dict]:
        logger.debug(f"[CHECKPOINT] Entering _build_sections() with {len(data.get('segments', []))} segments, {len(all_bars)} bars")
        try:
            sections = []
            label_counts = {}
            segments = data.get("segments", [])

            if not segments:
                logger.warning("No segments found in data")

            for seg_idx, seg in enumerate(segments):
                try:
                    seg_start = seg.get("start")
                    seg_end = seg.get("end")
                    label = seg.get("label", "unknown")

                    if seg_start is None or seg_end is None:
                        error_msg = f"Segment {seg_idx} missing start/end: start={seg_start}, end={seg_end}"
                        logger.warning(f"[EXCEPTION] {error_msg}")
                        continue

                    seg_start_ms = float(seg_start) * 1000
                    seg_end_ms = float(seg_end) * 1000
                    seg_bars = [b for b in all_bars if seg_start_ms <= b[0]["time_ms"] < seg_end_ms]

                    if not seg_bars:
                        logger.debug(f"Segment '{label}' ({seg_start:.2f}s-{seg_end:.2f}s) has no bars, skipping")
                        continue

                    label_counts[label] = label_counts.get(label, 0) + 1
                    section_name = f"{label} {label_counts[label]}"
                    sections.append({"name": section_name, "bars": seg_bars})
                    logger.debug(f"Created section '{section_name}' with {len(seg_bars)} bars (seg {seg_idx})")

                except (KeyError, TypeError, ValueError) as e:
                    error_msg = f"Error processing segment {seg_idx}: {e}"
                    logger.error(f"[EXCEPTION] {error_msg}")
                    continue

            if not sections and all_bars:
                logger.warning(f"No sections found from segments, creating default 'track' section with {len(all_bars)} bars")
                sections = [{"name": "track", "bars": all_bars}]

            for sec in sections:
                try:
                    label = sec["name"].rsplit(" ", 1)[0]
                    if label_counts.get(label, 0) == 1:
                        old_name = sec["name"]
                        sec["name"] = label
                        logger.debug(f"Renamed section '{old_name}' → '{label}' (only occurrence)")
                except Exception as e:
                    logger.error(f"[EXCEPTION] Error deduplicating section name: {e}")

            logger.debug(f"[CHECKPOINT] Section building complete: {len(sections)} sections created")
            logger.debug(f"[CHECKPOINT] Exiting _build_sections()")
            return sections
        except Exception as e:
            error_msg = f"Unexpected error in _build_sections(): {type(e).__name__}: {e}"
            logger.error(f"[EXCEPTION] {error_msg}", exc_info=True)
            raise

    @staticmethod
    def _flatten(data: dict, path: str | Path) -> Arrangement:
        logger.debug("[CHECKPOINT] Entering _flatten()")
        try:
            raw_bars = Allin1Importer._build_bar_list(data)
            logger.debug(f"[CHECKPOINT] Built bar list: {len(raw_bars)} bars")

            sections_data = Allin1Importer._build_sections(data, raw_bars)
            logger.debug(f"[CHECKPOINT] Built sections: {len(sections_data)} sections")

            section_list = []
            bar_idx = 0
            total_beats = 0

            for sec_idx, sec in enumerate(sections_data):
                try:
                    bars = []
                    for bar_data in sec.get("bars", []):
                        try:
                            beats = tuple(
                                Beat(time_ms=beat["time_ms"], position=beat["beat"])
                                for beat in bar_data
                            )
                            bars.append(Bar(idx=bar_idx, beats=beats))
                            total_beats += len(beats)
                            bar_idx += 1
                        except (KeyError, TypeError) as e:
                            error_msg = f"Error creating Beat in section {sec_idx}: {e}"
                            logger.error(f"[EXCEPTION] {error_msg} | bar_data={bar_data}")
                            raise ValueError(error_msg) from e

                    section = Section(idx=sec_idx, name=sec.get("name", f"section_{sec_idx}"), bars=bars)
                    section_list.append(section)
                    section_beats = sum(len(b.beats) for b in bars)
                    logger.debug(f"[CHECKPOINT] Section {sec_idx} '{section.name}': {len(bars)} bars, {section_beats} beats")

                except Exception as e:
                    error_msg = f"Error flattening section {sec_idx}: {type(e).__name__}: {e}"
                    logger.error(f"[EXCEPTION] {error_msg}")
                    raise

            try:
                file_id = Path(path).stem
                logger.debug(f"[CHECKPOINT] Creating master arrangement: file_id='{file_id}'")
                arrangement = Arrangement(
                    name=f"{file_id}-master",
                    master=True,
                    sections=section_list,
                    creationdate=int(time.time() * 1000),
                )
                arrangement.reindex(sr=1)
                logger.debug(f"[CHECKPOINT] Arrangement created and reindexed: {len(section_list)} sections, {bar_idx} bars, {total_beats} beats")
                logger.debug(f"[CHECKPOINT] Exiting _flatten() successfully")
                return arrangement
            except Exception as e:
                error_msg = f"Error creating Arrangement object: {type(e).__name__}: {e}"
                logger.error(f"[EXCEPTION] {error_msg}")
                raise

        except Exception as e:
            error_msg = f"Unexpected error in _flatten(): {type(e).__name__}: {e}"
            logger.error(f"[EXCEPTION] {error_msg}", exc_info=True)
            raise
