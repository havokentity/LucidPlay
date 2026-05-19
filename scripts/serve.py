#!/usr/bin/env python
"""Thin CLI wrapper around src.game_server.main()."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from src.game_server import main  # noqa: E402

if __name__ == "__main__":
    main()
