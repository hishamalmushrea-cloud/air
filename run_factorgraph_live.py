#!/usr/bin/env python3
"""Ablate the calibrated factor-graph *live safety signal*.

Usage::

    python run_factorgraph_live.py --n 3 --out out/factorgraph_live.csv
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from airlab.experiments import factorgraph_live_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(factorgraph_live_main())
