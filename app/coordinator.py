from __future__ import annotations
from pathlib import Path

from ui.app import ArrangementEditor


def create_editor(arrangement_path: str | None = None) -> ArrangementEditor:
    return ArrangementEditor(arrangement_path=arrangement_path)
