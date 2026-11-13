"""Standalone evaluation for the 3d-bpp solver task."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solver_utils import run


if __name__ == "__main__":
    run(Path(__file__).resolve().parent)
