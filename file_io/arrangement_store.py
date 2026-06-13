from __future__ import annotations
import json
from pathlib import Path

from domain.models import Arrangement, Section, Bar, Beat


class ArrangementStore:
    @staticmethod
    def arrangement_path_for(audio_path: str | Path) -> Path:
        p = Path(audio_path)
        return p.with_suffix(".allin1player.json")

    @staticmethod
    def load_or_create(
        audio_path: str | Path,
        analysis_path: str | Path | None = None,
    ) -> Arrangement:
        from file_io.allin1_importer import Allin1Importer

        audio_p = Path(audio_path)

        if analysis_path is None:
            analysis_path = audio_p.with_suffix(".json")
        analysis_p = Path(analysis_path)

        master = Allin1Importer.load(str(analysis_p))
        arrangement_p = ArrangementStore.arrangement_path_for(audio_p)

        if arrangement_p.exists():
            return ArrangementStore._load(str(arrangement_p))
        else:
            return master

    @staticmethod
    def _load(path: str) -> Arrangement:
        with open(path) as f:
            data = json.load(f)

        sections = []
        for sec_data in data.get("sections", []):
            bars = []
            for bar_data in sec_data.get("bars", []):
                beats = tuple(
                    Beat(
                        time_ms=beat["time_ms"],
                        position=beat["position"],
                        chord=beat.get("chord", ""),
                    )
                    for beat in bar_data.get("beats", [])
                )
                bars.append(Bar(idx=bar_data["idx"], beats=beats))
            sections.append(Section(
                idx=sec_data["idx"],
                name=sec_data["name"],
                bars=bars,
            ))

        return Arrangement(
            name=data.get("name", ""),
            master=data.get("master", False),
            sections=sections,
        )

    @staticmethod
    def save(arrangement: Arrangement, audio_path: str | Path) -> None:
        arrangement_p = ArrangementStore.arrangement_path_for(audio_path)
        data = {
            "name": arrangement.name,
            "master": arrangement.master,
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
                                    "chord": beat.chord,
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
        with open(arrangement_p, "w") as f:
            json.dump(data, f, indent=2)
