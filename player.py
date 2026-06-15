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
    ap.add_argument("arrangement", type=str, nargs="?", default=None,
                    help="Arrangement JSON file (*.arrangement.*.json)")
    args = ap.parse_args()

    if args.arrangement:
        logger.info(f"Arrangement file specified: {args.arrangement}")
        arrangement_path = Path(args.arrangement)
        if not arrangement_path.exists():
            logger.error(f"Arrangement file not found: {arrangement_path}")
            sys.exit(f"Arrangement file not found: {arrangement_path}")
        logger.info(f"Initializing editor with arrangement={arrangement_path}")
        create_editor(arrangement_path=str(arrangement_path)).run()
    else:
        logger.info("No arrangement file specified, launching browser")
        create_editor().run()


if __name__ == "__main__":
    main()
