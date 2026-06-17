from __future__ import annotations
import json
import logging
import time
from pathlib import Path

from domain.models import Mix, Section, Bar, Beat

logger = logging.getLogger(__name__)

DEFAULT_MIX_FILENAME = "mix-editor.json"


class MixStore:
    """Load/save Mix objects to a `mix-editor.json` file.

    Mirrors ArrangementStore but persists at the section level under a top-level
    `mix-sections` key. Section/Bar/Beat (de)serialization matches the
    arrangement format so sections imported from arrangements round-trip cleanly.
    """

    @staticmethod
    def default_path(directory: str | Path = ".") -> Path:
        return Path(directory) / DEFAULT_MIX_FILENAME

    # ------------------------------------------------------------------ section (de)serialization

    @staticmethod
    def _bars_from_data(bars_data: list, offset: int = 0) -> list[Bar]:
        bars = []
        for bar_idx, bar_data in enumerate(bars_data):
            beats = tuple(
                Beat(
                    start_ms=int(beat.get("start_ms", beat.get("time_ms", 0))),
                    finish_ms=int(beat.get("finish_ms", 0)),
                    position=int(beat.get("position", 0)),
                )
                for beat in bar_data.get("beats", [])
            )
            bars.append(Bar(
                idx=bar_data.get("idx", offset + bar_idx),
                beats=beats,
                audiosource=bar_data.get("audiosource", ""),
                color=bar_data.get("color", ""),
            ))
        return bars

    @staticmethod
    def _bars_to_data(bars: list[Bar]) -> list[dict]:
        return [
            {
                "idx": bar.idx,
                "audiosource": bar.audiosource,
                "color": bar.color,
                "beats": [
                    {
                        "start_ms": beat.start_ms,
                        "finish_ms": beat.finish_ms,
                        "position": beat.position,
                    }
                    for beat in bar.beats
                ],
            }
            for bar in bars
        ]

    @staticmethod
    def _section_from_data(sec_data: dict, sec_idx: int) -> Section:
        bars = MixStore._bars_from_data(sec_data.get("bars", []))
        fade_out_bars = MixStore._bars_from_data(sec_data.get("fade_out_bars", []))
        fade_in_bars = MixStore._bars_from_data(sec_data.get("fade_in_bars", []))
        return Section(
            idx=sec_data.get("idx", sec_idx),
            name=sec_data.get("name", f"section_{sec_idx}"),
            bars=bars,
            is_transition=sec_data.get("is_transition", False),
            fade_out_bars=fade_out_bars,
            fade_in_bars=fade_in_bars,
        )

    @staticmethod
    def _section_to_data(sec: Section) -> dict:
        return {
            "idx": sec.idx,
            "name": sec.name,
            "is_transition": sec.is_transition,
            "bars": MixStore._bars_to_data(sec.bars),
            "fade_out_bars": MixStore._bars_to_data(sec.fade_out_bars),
            "fade_in_bars": MixStore._bars_to_data(sec.fade_in_bars),
        }

    # ------------------------------------------------------------------ load / save

    @staticmethod
    def load(path: str | Path) -> Mix:
        logger.debug(f"[CHECKPOINT] Entering MixStore.load() with path={path}")
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            error_msg = f"Mix file not found: {path}"
            logger.error(f"[EXCEPTION] {error_msg}")
            raise FileNotFoundError(error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in mix file {path} (line {e.lineno}, col {e.colno}): {e.msg}"
            logger.error(f"[EXCEPTION] {error_msg}")
            raise ValueError(error_msg) from e

        sections = [
            MixStore._section_from_data(sec_data, i)
            for i, sec_data in enumerate(data.get("mix-sections", []))
        ]
        mix = Mix(
            name=data.get("name", "unnamed mix"),
            sections=sections,
            creationdate=data.get("creationdate", int(time.time() * 1000)),
        )
        logger.info(f"Loaded mix '{mix.name}' with {len(sections)} section(s) from {path}")
        return mix

    @staticmethod
    def save(mix: Mix, path: str | Path) -> None:
        logger.debug(f"[CHECKPOINT] Entering MixStore.save() with path={path}")
        data = {
            "name": mix.name,
            "creationdate": mix.creationdate,
            "mix-sections": [MixStore._section_to_data(sec) for sec in mix.sections],
        }
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved mix '{mix.name}' ({len(mix.sections)} sections) to {path_obj}")

    # ------------------------------------------------------------------ import helper

    @staticmethod
    def list_arrangement_sections(arrangement_path: str | Path) -> list[Section]:
        """Return the full Section objects from an arrangement JSON file.

        Used to import sections into a mix. Reuses ArrangementStore's loader so
        the same parsing rules apply.
        """
        from file_io.arrangement_store import ArrangementStore

        arrangement = ArrangementStore._load(str(arrangement_path))
        return arrangement.sections
