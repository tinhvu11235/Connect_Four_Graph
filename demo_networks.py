"""Launcher for the network demo that works from any working directory.

    python path/to/Connect4GNN/demo_networks.py [--baseline-cnn PATH] [--method1 PATH] [--method2 PATH]

Equivalent to ``python -m demo.app_networks`` run from the repository root.
"""

import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)

from demo.app_networks import main  # noqa: E402

if __name__ == "__main__":
    main()
