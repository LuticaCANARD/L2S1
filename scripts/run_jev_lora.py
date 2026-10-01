#!/usr/bin/env python3
"""Compatibility entry point; supported CLI: l2s1-train."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training' / 'src'))
from l2s1_training.run_jev_lora import *  # noqa: F401,F403

if __name__ == '__main__':
    main()
