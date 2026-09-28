"""Ensure the repository root is importable when running scripts or tests."""

import sys
from pathlib import Path
 
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT)