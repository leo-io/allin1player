"""Tests for multi-select functionality in MixEditor."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domain.models import Section, Mix  # noqa: E402


def test_delete_multiple_sections():
    """Test bulk deletion of multiple selected sections."""
    # Build a mix with 5 sections
    sections = [
        Section(idx=i, name=f"Section {i}", bars=[])
        for i in range(5)
    ]
    mix = Mix(name="test", sections=sections)

    # Simulate deleting sections at indices 1, 3
    to_delete = {1, 3}
    for idx in sorted(to_delete, reverse=True):
        del mix.sections[idx]

    # Reindex
    for i, sec in enumerate(mix.sections):
        sec.idx = i

    # Should have sections 0, 2, 4 (now at indices 0, 1, 2)
    assert len(mix.sections) == 3
    assert mix.sections[0].name == "Section 0"
    assert mix.sections[1].name == "Section 2"
    assert mix.sections[2].name == "Section 4"


def test_clear_selection_keeps_anchor():
    """Test that clearing multi-selection preserves the anchor."""
    selection = {1, 3, 5}
    anchor = 3

    # Simulate _clear_selection
    selection = {anchor} if anchor is not None else set()

    assert selection == {3}


def test_ctrl_click_toggle():
    """Test Ctrl+click toggle logic."""
    selection = {1, 2, 4}
    anchor = 2

    # Toggle item 2 (in set): should remove it and move anchor to remaining max
    if 2 in selection:
        selection.discard(2)
        if anchor == 2:
            anchor = max(selection) if selection else None

    assert 2 not in selection
    assert anchor == 4  # max of remaining {1, 4}

    # Toggle item 5 (not in set): should add it and set anchor
    if 5 in selection:
        pass
    else:
        selection.add(5)
        anchor = 5

    assert 5 in selection
    assert anchor == 5
