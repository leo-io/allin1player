#!/usr/bin/env python3
"""
Allin1 Arrangement Editor: edits beat/bar/section arrangements
with audio playback preview, based on .json files from the allin1 analyzer.

Refactored into a modular monolith:
  domain/   — immutable musical models
  file_io/  — allin1 JSON importer + arrangement persistence
  playback/ — audio engine, transport state, scheduling
  ui/       — Tkinter shell, canvas renderer, interaction
  app/      — coordinator wiring everything together
"""

import argparse
import logging
import sys
from pathlib import Path

from app.coordinator import create_editor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('allin1player.log'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== Allin1 Arrangement Editor Started ===")
    ap = argparse.ArgumentParser(description="Edit beat/bar arrangements with audio preview")
    ap.add_argument("audio", type=str, nargs="?", default=None,
                    help="Audio file (optional — shows sidebar with all .mp3/.wav files)")
    ap.add_argument("--json", "-j", type=str, default=None,
                    help="allin1 .json file (default: same name with .json extension)")
    args = ap.parse_args()

    if args.audio:
        logger.info(f"Audio file specified: {args.audio}")
        audio_path = Path(args.audio)
        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            sys.exit(f"Audio file not found: {audio_path}")
        json_path = Path(args.json) if args.json else audio_path.with_suffix(".json")
        logger.debug(f"Using JSON file: {json_path}")
        if not json_path.exists():
            logger.error(f"JSON file not found: {json_path}")
            sys.exit(f"JSON file not found: {json_path}")
        logger.info(f"Initializing editor with audio={audio_path}, json={json_path}")
        create_editor(audio_path=str(audio_path), json_path=str(json_path)).run()
    else:
        logger.info("No audio file specified, launching with file browser")
        create_editor().run()


if __name__ == "__main__":
    main()
