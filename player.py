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
    ap = argparse.ArgumentParser(description="Edit beat/bar arrangements")
    ap.add_argument("file", type=str, nargs="?", default=None,
                    help="Arrangement JSON (*.arrangement.*.json) or Audio WAV file (*.wav)")
    ap.add_argument("--analysis", "-a", type=str, default=None,
                    help="Analysis JSON (allin1 format, optional if creating from WAV)")
    args = ap.parse_args()

    if args.file:
        logger.info(f"File specified: {args.file}")
        file_path = Path(args.file)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            sys.exit(f"File not found: {file_path}")

        # Check if it's an arrangement file or audio file
        if file_path.suffix == ".json" and "arrangement" in file_path.name:
            # It's an arrangement JSON file
            logger.info(f"Loading arrangement: {file_path}")
            create_editor(arrangement_path=str(file_path)).run()
        elif file_path.suffix == ".wav":
            # It's a WAV file, need to create/load arrangement
            logger.info(f"WAV file specified, creating/loading arrangement: {file_path}")
            analysis_path = Path(args.analysis) if args.analysis else file_path.with_suffix(".json")
            if not analysis_path.exists():
                logger.error(f"Analysis file not found: {analysis_path}")
                sys.exit(f"Analysis file not found: {analysis_path} (use --analysis to specify)")
            editor = create_editor()
            editor.create_and_load_arrangement(str(file_path), str(analysis_path))
            editor.run()
        else:
            logger.error(f"Unsupported file type: {file_path.suffix}")
            sys.exit(f"Unsupported file type. Use *.arrangement.*.json or *.wav")
    else:
        logger.info("No file specified, launching browser")
        create_editor().run()


if __name__ == "__main__":
    main()
