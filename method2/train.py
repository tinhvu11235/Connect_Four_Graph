"""Train method 2 (fixed grid graph) with the AlphaZero repo's pipeline.

Same flags as ``python -m train_alphazero``; ``--channels`` is the GNN width and
``--residual-blocks`` the number of residual graph blocks.

    python -m method2.train --run-dir runs/method2 --channels 256 --iterations 3
"""

from __future__ import annotations

import sys

from common import repo_bridge
from method2.model import model_config


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` (default: the command line) and train a method 2 network.

    Side effects: see ``common.repo_bridge.train`` (writes the run directory, uses the GPU).
    """

    repo_bridge.train(
        argv if argv is not None else sys.argv[1:],
        lambda config: model_config(config.channels, config.residual_blocks),
        description="Train method 2 (grid graph GNN) with AlphaZero self-play.",
    )


if __name__ == "__main__":
    main()
