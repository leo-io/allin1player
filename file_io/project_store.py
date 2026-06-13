from __future__ import annotations
import json
from pathlib import Path

from domain.models import ArrangementDocument, ArrangementVersion, SongStructure


class ProjectStore:
    @staticmethod
    def project_path_for(audio_path: str | Path) -> Path:
        p = Path(audio_path)
        return p.with_suffix(".allin1player.json")

    @staticmethod
    def load_or_create(
        audio_path: str | Path,
        analysis_path: str | Path | None = None,
    ) -> tuple[ArrangementDocument, SongStructure]:
        from file_io.allin1_importer import Allin1Importer

        audio_p = Path(audio_path)

        if analysis_path is None:
            analysis_path = audio_p.with_suffix(".json")
        analysis_p = Path(analysis_path)

        song = Allin1Importer.load(str(analysis_p))
        project_p = ProjectStore.project_path_for(audio_p)

        if project_p.exists():
            doc = ProjectStore._load(str(project_p), song)
        else:
            doc = ArrangementDocument(source=song)
            doc.add_version("Original")

        return doc, song

    @staticmethod
    def _load(path: str, song: SongStructure) -> ArrangementDocument:
        with open(path) as f:
            data = json.load(f)

        versions = []
        for v in data.get("versions", []):
            versions.append(ArrangementVersion(
                name=v["name"],
                section_ordering=[tuple(t) for t in v["section_ordering"]],
            ))

        doc = ArrangementDocument(
            source=song,
            versions=versions,
            active_version_idx=data.get("active_version_idx", 0),
            schema_version=data.get("schema_version", 1),
        )
        return doc

    @staticmethod
    def save(doc: ArrangementDocument, audio_path: str | Path) -> None:
        project_p = ProjectStore.project_path_for(audio_path)
        data = {
            "schema_version": doc.schema_version,
            "active_version_idx": doc.active_version_idx,
            "versions": [
                {
                    "name": v.name,
                    "section_ordering": v.section_ordering,
                }
                for v in doc.versions
            ],
        }
        with open(project_p, "w") as f:
            json.dump(data, f, indent=2)
