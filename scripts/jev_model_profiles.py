"""Compatibility imports for the official training package."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training' / 'src'))
from l2s1_training.jev_model_profiles import *  # noqa: F401,F403
