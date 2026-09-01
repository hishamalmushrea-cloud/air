#!/usr/bin/env python3
"""Characterise the factor-graph consistency detector.

Usage::

    python run_factorgraph.py --n 3 --out out/factorgraph.csv
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from airlab.experiments import factorgraph_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(factorgraph_main())
