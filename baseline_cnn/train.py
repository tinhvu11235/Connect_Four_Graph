"""Train the baseline (the repo's residual CNN) through the same bridge, so it is timed the same way.

Same flags as ``python -m train_alphazero``; ``--channels`` / ``--residual-blocks`` size the CNN.

    python -m baseline_cnn.train --run-dir runs/baseline_cnn --channels 256 --iterations 3
"""

from __future__ import annotations

import sys

from common import repo_bridge


def main(argv: list[str] | None = None) -> None:
    """Parse the repo's flags from ``argv`` (default: the command line) and train the CNN.

    Side effects: see ``common.repo_bridge.train`` (writes the run directory, uses the GPU).
    """

    repo_bridge.train(argv if argv is not None else sys.argv[1:], None, description="Train the residual CNN baseline.")


if __name__ == "__main__":
    main()
