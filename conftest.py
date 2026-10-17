"""pytest configuration: puts the repository root on ``sys.path`` so every test imports
``common``, ``method1``, ``method2``, ``baseline_cnn`` and ``demo`` like the entry points do."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
