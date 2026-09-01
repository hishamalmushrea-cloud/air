#!/usr/bin/env python3
"""Run the velocity-aiding corruption / safety-layer study.

Usage::

    python run_corruption.py --durations 0,15,30 --n 3 --out out/corruption.csv
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from airlab.experiments import corruption_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(corruption_main())
