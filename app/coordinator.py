from __future__ import annotations
from pathlib import Path

from ui.app import ArrangementEditor


def create_editor(audio_path: str | None = None, json_path: str | None = None) -> ArrangementEditor:
    return ArrangementEditor(audio_path=audio_path, beats_path=json_path)
