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

        audio_p = Path(audio_path)
        master_p = ArrangementStore.master_path_for(audio_p)
        logger.info(f"Loading or creating arrangement for audio: {audio_p}")

        if master_p.exists():
            logger.info(f"Master arrangement file found: {master_p}")
            master = ArrangementStore._load(str(master_p))
            master.master = True
            logger.info(f"Loaded master arrangement '{master.name}' (created: {master.creationdate})")
        else:
            logger.info(f"Master arrangement not found at {master_p}, will parse from allin1 analysis")
            if analysis_path is None:
                analysis_path = audio_p.with_suffix(".json")
            analysis_p = Path(analysis_path)
            logger.debug(f"Using analysis path: {analysis_p}")
            master = Allin1Importer.load(str(analysis_p))
            logger.info(f"Saving master arrangement to: {master_p}")
            ArrangementStore._save_to_path(master, str(master_p))

        arrangement_p = ArrangementStore.arrangement_path_for(audio_p)
        logger.debug(f"Checking for user edits at: {arrangement_p}")

        if arrangement_p.exists():
            logger.info(f"User edits found: {arrangement_p}")
            user_arr = ArrangementStore._load(str(arrangement_p))
            logger.info(f"Loaded user arrangement '{user_arr.name}' (will use instead of master)")
            return user_arr
        else:
            logger.info(f"No user edits found, using master arrangement")
            return master

    @staticmethod
    def _load(path: str) -> Arrangement:
        logger.debug(f"Loading arrangement from: {path}")
        try:
            with open(path) as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.error(f"Arrangement file not found: {path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in arrangement file {path}: {e}")
            raise

        section_count = len(data.get("sections", []))
        logger.debug(f"Parsed arrangement: {section_count} sections, name='{data.get('name', '')}'")

        sections = []
        total_bars = 0
        total_beats = 0

        for sec_data in data.get("sections", []):
            bars = []
            for bar_data in sec_data.get("bars", []):
                beats = tuple(
                    Beat(
                        time_ms=beat["time_ms"],
                        position=beat["position"],
                    )
                    for beat in bar_data.get("beats", [])
                )
                bars.append(Bar(idx=bar_data["idx"], beats=beats))
                total_beats += len(beats)
            sections.append(Section(
                idx=sec_data["idx"],
                name=sec_data["name"],
                bars=bars,
            ))
            total_bars += len(bars)
            logger.debug(f"  Section '{sec_data['name']}': {len(bars)} bars, {sum(len(b.beats) for b in bars)} beats")

        creationdate = data.get("creationdate", int(__import__("time").time() * 1000))
        arrangement = Arrangement(
            name=data.get("name", ""),
            master=data.get("master", False),
            sections=sections,
            creationdate=creationdate,
        )
        logger.info(f"Loaded arrangement: {total_bars} bars, {total_beats} beats, created={creationdate}")
        return arrangement

    @staticmethod
    def _save_to_path(arrangement: Arrangement, path: str) -> None:
        logger.debug(f"Serializing arrangement to {path}")
        total_bars = len(arrangement.bars)
        total_beats = sum(len(bar.beats) for bar in arrangement.bars)

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
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved arrangement: {path} ({len(arrangement.sections)} sections, {total_bars} bars, {total_beats} beats)")
        except Exception as e:
            logger.error(f"Failed to save arrangement to {path}: {e}", exc_info=True)
            raise

    @staticmethod
    def save(arrangement: Arrangement, audio_path: str | Path) -> None:
        arrangement_p = ArrangementStore.arrangement_path_for(audio_path)
        logger.info(f"Saving user edits for audio: {audio_path}")
        ArrangementStore._save_to_path(arrangement, str(arrangement_p))
