from __future__ import annotations
from pathlib import Path

from ui.app import BeatPlayer


def create_player(audio_path: str | None = None, json_path: str | None = None) -> BeatPlayer:
    return BeatPlayer(audio_path=audio_path, beats_path=json_path)
