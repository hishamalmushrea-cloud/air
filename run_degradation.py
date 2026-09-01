#!/usr/bin/env python3
"""Run the GNSS-outage degradation study.

Usage::

    python run_degradation.py --durations 0,5,10,15,20,25 --n 10 --out out/degradation.csv
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from airlab.experiments import degradation_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(degradation_main())
