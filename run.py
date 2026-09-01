#!/usr/bin/env python3
"""Convenience runner: ``python run.py``.

Adds the ``src`` directory to the path and launches the AIR Lab demo.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from airlab.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
