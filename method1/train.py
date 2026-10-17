"""Train method 1 (dynamic graph) with the AlphaZero repo's pipeline.

Same flags as ``python -m train_alphazero`` plus ``--line-of-sight`` /
``--no-line-of-sight`` (default on).  ``--channels`` is the GNN width and
``--residual-blocks`` the number of residual graph blocks.

    python -m method1.train --run-dir runs/method1 --channels 160 --iterations 3
"""

from __future__ import annotations

import argparse
import sys

from common import repo_bridge
from method1.model import model_config


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` (default: the command line) and train a method 1 network.

    Side effects: see ``common.repo_bridge.train`` (writes the run directory, uses the GPU).
    """

    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--line-of-sight", action=argparse.BooleanOptionalAction, default=True)
    options, rest = extra.parse_known_args(argv if argv is not None else sys.argv[1:])
    repo_bridge.train(
        rest,
        lambda config: model_config(config.channels, config.residual_blocks, options.line_of_sight),
        description="Train method 1 (dynamic graph GNN) with AlphaZero self-play.",
    )


if __name__ == "__main__":
    main()
