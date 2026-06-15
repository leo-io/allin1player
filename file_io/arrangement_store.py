from __future__ import annotations
import json
import logging
from pathlib import Path

from domain.models import Arrangement, Section, Bar, Beat

logger = logging.getLogger(__name__)


class ArrangementStore:
    @staticmethod
    def arrangement_path_for(audio_path: str | Path) -> Path:
        p = Path(audio_path)
        return p.with_stem(p.stem + ".arrangement").with_suffix(".json")

    @staticmethod
    def master_path_for(audio_path: str | Path) -> Path:
        p = Path(audio_path)
        return p.with_stem(p.stem + ".arrangement.master").with_suffix(".json")

    @staticmethod
    def load_or_create(
        audio_path: str | Path,
        analysis_path: str | Path | None = None,
    ) -> Arrangement:
        from file_io.allin1_importer import Allin1Importer

        logger.info(f"[CHECKPOINT] Entering load_or_create() with audio_path={audio_path}")
        try:
            audio_p = Path(audio_path)
            master_p = ArrangementStore.master_path_for(audio_p)
            logger.info(f"Loading or creating arrangement for audio: {audio_p}")
            logger.debug(f"[CHECKPOINT] Master path: {master_p}")

            if master_p.exists():
                try:
                    logger.info(f"Master arrangement file found: {master_p}")
                    master = ArrangementStore._load(str(master_p))
                    master.master = True
                    logger.info(f"Loaded master arrangement '{master.name}' (created: {master.creationdate})")
                    logger.debug(f"[CHECKPOINT] Master loaded successfully")
                except Exception as e:
                    error_msg = f"Failed to load master arrangement from {master_p}: {type(e).__name__}: {e}"
                    logger.error(f"[EXCEPTION] {error_msg}")
                    raise
            else:
                try:
                    logger.info(f"Master arrangement not found at {master_p}, will parse from allin1 analysis")
                    if analysis_path is None:
                        analysis_path = audio_p.with_suffix(".json")
                    analysis_p = Path(analysis_path)
                    logger.debug(f"Using analysis path: {analysis_p}")

                    master = Allin1Importer.load(str(analysis_p))
                    logger.debug(f"[CHECKPOINT] Allin1 import successful")

                    logger.info(f"Saving master arrangement to: {master_p}")
                    ArrangementStore._save_to_path(master, str(master_p))
                    logger.debug(f"[CHECKPOINT] Master saved to {master_p}")
                except FileNotFoundError as e:
                    error_msg = f"Analysis file not found: {analysis_p if 'analysis_p' in locals() else analysis_path}"
                    logger.error(f"[EXCEPTION] {error_msg}")
                    raise FileNotFoundError(error_msg) from e
                except Exception as e:
                    error_msg = f"Failed to create master from allin1 analysis: {type(e).__name__}: {e}"
                    logger.error(f"[EXCEPTION] {error_msg}")
                    raise

            arrangement_p = ArrangementStore.arrangement_path_for(audio_p)
            logger.debug(f"Checking for user edits at: {arrangement_p}")

            if arrangement_p.exists():
                try:
                    logger.info(f"User edits found: {arrangement_p}")
                    user_arr = ArrangementStore._load(str(arrangement_p))
                    logger.info(f"Loaded user arrangement '{user_arr.name}' (will use instead of master)")
                    logger.info(f"[CHECKPOINT] Exiting load_or_create() with user arrangement")
                    return user_arr
                except Exception as e:
                    error_msg = f"Failed to load user arrangement from {arrangement_p}: {type(e).__name__}: {e}"
                    logger.error(f"[EXCEPTION] {error_msg}")
                    raise
            else:
                logger.info(f"No user edits found, using master arrangement")
                logger.info(f"[CHECKPOINT] Exiting load_or_create() with master arrangement")
                return master

        except Exception as e:
            error_msg = f"Unexpected error in load_or_create(): {type(e).__name__}: {e}"
            logger.error(f"[EXCEPTION] {error_msg}", exc_info=True)
            raise

    @staticmethod
    def _load(path: str) -> Arrangement:
        logger.debug(f"[CHECKPOINT] Entering _load() with path={path}")
        try:
            # Load JSON file
            try:
                with open(path, encoding='utf-8') as f:
                    data = json.load(f)
                logger.debug(f"[CHECKPOINT] JSON file loaded successfully")
            except FileNotFoundError:
                error_msg = f"Arrangement file not found: {path}"
                logger.error(f"[EXCEPTION] {error_msg}")
                raise FileNotFoundError(error_msg)
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON in arrangement file {path} (line {e.lineno}, col {e.colno}): {e.msg}"
                logger.error(f"[EXCEPTION] {error_msg}")
                raise ValueError(error_msg) from e
            except IOError as e:
                error_msg = f"IO error reading arrangement file {path}: {e}"
                logger.error(f"[EXCEPTION] {error_msg}")
                raise IOError(error_msg) from e

            section_count = len(data.get("sections", []))
            arrangement_name = data.get("name", "")
            logger.debug(f"[CHECKPOINT] Parsed JSON: {section_count} sections, name='{arrangement_name}'")

            # Process sections
            sections = []
            total_bars = 0
            total_beats = 0

            for sec_idx, sec_data in enumerate(data.get("sections", [])):
                try:
                    bars = []
                    for bar_idx, bar_data in enumerate(sec_data.get("bars", [])):
                        try:
                            beats = tuple(
                                Beat(
                                    time_ms=int(beat.get("time_ms", 0)),
                                    position=int(beat.get("position", 0)),
                                )
                                for beat in bar_data.get("beats", [])
                            )
                            bars.append(Bar(
                                idx=bar_data.get("idx", bar_idx),
                                beats=beats,
                                audiosource=bar_data.get("audiosource", "")
                            ))
                            total_beats += len(beats)
                        except (KeyError, TypeError, ValueError) as e:
                            error_msg = f"Error creating Beat in section {sec_idx}, bar {bar_idx}: {e}"
                            logger.error(f"[EXCEPTION] {error_msg} | beat_data={bar_data}")
                            raise ValueError(error_msg) from e

                    section = Section(
                        idx=sec_data.get("idx", sec_idx),
                        name=sec_data.get("name", f"section_{sec_idx}"),
                        bars=bars,
                    )
                    sections.append(section)
                    total_bars += len(bars)
                    logger.debug(f"  [CHECKPOINT] Section '{section.name}': {len(bars)} bars, {sum(len(b.beats) for b in bars)} beats")

                except Exception as e:
                    error_msg = f"Error processing section {sec_idx}: {type(e).__name__}: {e}"
                    logger.error(f"[EXCEPTION] {error_msg}")
                    raise

            # Create Arrangement object
            try:
                creationdate = data.get("creationdate", int(__import__("time").time() * 1000))
                arrangement = Arrangement(
                    name=arrangement_name if arrangement_name else "unnamed",
                    master=data.get("master", False),
                    sections=sections,
                    creationdate=creationdate,
                )
                logger.info(f"Loaded arrangement: name='{arrangement.name}', {total_bars} bars, {total_beats} beats, created={creationdate}")
                logger.debug(f"[CHECKPOINT] Exiting _load() successfully")
                return arrangement
            except Exception as e:
                error_msg = f"Error creating Arrangement object: {type(e).__name__}: {e}"
                logger.error(f"[EXCEPTION] {error_msg}")
                raise

        except Exception as e:
            error_msg = f"Unexpected error in _load(): {type(e).__name__}: {e}"
            logger.error(f"[EXCEPTION] {error_msg}", exc_info=True)
            raise

    @staticmethod
    def _save_to_path(arrangement: Arrangement, path: str) -> None:
        logger.debug(f"[CHECKPOINT] Entering _save_to_path() with path={path}")
        try:
            logger.debug(f"Serializing arrangement '{arrangement.name}' to {path}")
            total_bars = len(arrangement.bars)
            total_beats = sum(len(bar.beats) for bar in arrangement.bars)
            logger.debug(f"[CHECKPOINT] Data to serialize: {len(arrangement.sections)} sections, {total_bars} bars, {total_beats} beats")

            # Build JSON structure
            try:
                data = {
                    "name": arrangement.name,
                    "master": arrangement.master,
                    "creationdate": arrangement.creationdate,
                    "sections": [
                        {
                            "idx": sec.idx,
                            "name": sec.name,
                            "bars": [
                                {
                                    "idx": bar.idx,
                                    "audiosource": bar.audiosource,
                                    "beats": [
                                        {
                                            "time_ms": beat.time_ms,
                                            "position": beat.position,
                                        }
                                        for beat in bar.beats
                                    ],
                                }
                                for bar in sec.bars
                            ],
                        }
                        for sec in arrangement.sections
                    ],
                }
                logger.debug(f"[CHECKPOINT] JSON structure built successfully")
            except Exception as e:
                error_msg = f"Error building JSON structure: {type(e).__name__}: {e}"
                logger.error(f"[EXCEPTION] {error_msg}")
                raise

            # Write to file
            try:
                path_obj = Path(path)
                path_obj.parent.mkdir(parents=True, exist_ok=True)
                logger.debug(f"[CHECKPOINT] Directory ensured: {path_obj.parent}")

                with open(path, "w", encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                logger.debug(f"[CHECKPOINT] File written successfully")

                logger.info(f"Saved arrangement: {path} ({len(arrangement.sections)} sections, {total_bars} bars, {total_beats} beats)")
                logger.debug(f"[CHECKPOINT] Exiting _save_to_path() successfully")
            except IOError as e:
                error_msg = f"IO error writing to {path}: {e}"
                logger.error(f"[EXCEPTION] {error_msg}")
                raise IOError(error_msg) from e
            except Exception as e:
                error_msg = f"Unexpected error writing file: {type(e).__name__}: {e}"
                logger.error(f"[EXCEPTION] {error_msg}")
                raise

        except Exception as e:
            error_msg = f"Failed to save arrangement to {path}: {type(e).__name__}: {e}"
            logger.error(f"[EXCEPTION] {error_msg}", exc_info=True)
            raise

    @staticmethod
    def save(arrangement: Arrangement, audio_path: str | Path) -> None:
        arrangement_p = ArrangementStore.arrangement_path_for(audio_path)
        logger.info(f"Saving user edits for audio: {audio_path}")
        ArrangementStore._save_to_path(arrangement, str(arrangement_p))
