#!/usr/bin/env python3
"""
Mix Editor: arranges whole sections (imported from existing arrangements) into a
mix and persists them to a `mix-editor.json` file.

Shows each section as a single block (no beat/bar visualization), mirroring the
arrangement editor's modular structure:
  domain/   — Mix + Section models
  file_io/  — MixStore (mix-editor.json persistence + section import)
  ui/       — Tkinter shell + block renderer
  app/      — coordinator wiring
"""

import argparse
import logging
import sys
from pathlib import Path

from app.coordinator import create_mix_editor

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
    logger.info("=== Mix Editor Started ===")
    ap = argparse.ArgumentParser(description="Edit a mix of sections")
    ap.add_argument("file", type=str, nargs="?", default=None,
                    help="Mix JSON to open (e.g. mix-editor.json)")
    args = ap.parse_args()

    mix_path = None
    if args.file:
        p = Path(args.file)
        if not p.exists():
            logger.error(f"File not found: {p}")
            sys.exit(f"File not found: {p}")
        mix_path = str(p)

    create_mix_editor(mix_path=mix_path).run()


if __name__ == "__main__":
    main()
