#!/usr/bin/env python3
"""Run a random batch of AIR Lab scenarios.

Usage::

    python run_batch.py --num 20 --duration 40 --seed 42 --out out/batch.csv
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from airlab.experiments import batch_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(batch_main())
