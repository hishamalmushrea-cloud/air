#!/usr/bin/env python3
"""Compare consensus policies between the landmark detector and factor graph.

Usage::

    python run_consensus.py --n 6 --out out/consensus.csv
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from airlab.experiments import consensus_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(consensus_main())
