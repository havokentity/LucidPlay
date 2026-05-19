#!/usr/bin/env python
"""Thin CLI wrapper around src.capture.main()."""
import os
import sys

# Allow `python scripts/capture.py` to find the `src` package.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from src.capture import main  # noqa: E402

if __name__ == "__main__":
    main()
