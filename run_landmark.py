#!/usr/bin/env python3
"""Ablate the independent landmark detector.

Usage::

    python run_landmark.py --durations 0,15,30 --n 3 --out out/landmark.csv
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from airlab.experiments import landmark_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(landmark_main())
