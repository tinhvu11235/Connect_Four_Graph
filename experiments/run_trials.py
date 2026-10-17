"""Short real training runs of the baseline CNN, method 1 and method 2, saved as evidence.

Two suites:

* ``trials``   - the settings of the repo's 90k-game run ``results/train_only_fujita``
                 (256 games/iteration, 200 simulations, 50k replay, lr 1e-4, wd 0.01,
                 100-game evaluation vs Fujita's AlphaZero every iteration); default 3 iterations.
                 ``experiments.estimate_full_run`` extrapolates them to 90k games.
* ``peak_ram`` - 1 iteration with the repo's default flags and a 20-game evaluation,
                 to measure peak process RAM.

Each network trains in ``runs/<suite>_<method>/`` (checkpoints, not committed); the
evidence goes to ``results/<suite>/<method>/``: ``command.txt``, ``console.log``,
``config.json``, ``train_metrics.csv``, ``resources.json``.

    python -m experiments.run_trials --suite trials --iterations 3
    python -m experiments.run_trials --suite peak_ram --only method1
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NETWORKS = {
    "baseline_cnn": ("baseline_cnn.train", ["--channels", "256"]),
    "method1": ("method1.train", ["--channels", "160", "--line-of-sight"]),
    "method2": ("method2.train", ["--channels", "256"]),
}
SUITES = {
    "trials": [
        "--replay-capacity", "50000", "--learning-rate", "1e-4", "--lr-warmup-steps", "0",
        "--weight-decay", "0.01", "--residual-blocks", "6", "--no-resume",
    ],
    "peak_ram": ["--eval-games", "20", "--residual-blocks", "6", "--no-resume"],
}
DEFAULT_ITERATIONS = {"trials": 3, "peak_ram": 1}
EVIDENCE_FILES = ("config.json", "train_metrics.csv", "resources.json")


def command_for(suite: str, network: str, iterations: int) -> list[str]:
    """Return the training command of ``network`` in ``suite``.  Side effects: none."""

    module, flags = NETWORKS[network]
    run_dir = ROOT / "runs" / f"{suite}_{network}"
    return [sys.executable, "-m", module, "--run-dir", str(run_dir),
            "--iterations", str(iterations), *SUITES[suite], *flags]


def run(suite: str, network: str, iterations: int) -> int:
    """Train one network, then copy its evidence into ``results/<suite>/<network>/``; return the exit code.

    Side effects: runs a training subprocess (GPU, minutes to hours); writes
    ``runs/<suite>_<network>/`` and overwrites the evidence folder.
    """

    command = command_for(suite, network, iterations)
    evidence = ROOT / "results" / suite / network
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "command.txt").write_text(" ".join(command[1:]) + "\n", encoding="utf-8")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
    print(f"=== {suite}/{network}: {' '.join(command)}", flush=True)
    with (evidence / "console.log").open("w", encoding="utf-8") as log:
        code = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    run_dir = Path(command[command.index("--run-dir") + 1])
    for name in EVIDENCE_FILES:
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, evidence / name)
    print(f"=== {suite}/{network}: exit {code}, evidence in {evidence}", flush=True)
    return code


def main() -> None:
    """Parse ``--suite``, ``--iterations`` and ``--only`` and run the selected networks one after another.

    Side effects: see ``run``; exits non-zero if any run failed.
    """

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--suite", choices=list(SUITES), default="trials")
    parser.add_argument("--iterations", type=int, help="default: 3 for trials, 1 for peak_ram")
    parser.add_argument("--only", nargs="*", choices=list(NETWORKS), default=list(NETWORKS))
    args = parser.parse_args()
    iterations = args.iterations or DEFAULT_ITERATIONS[args.suite]
    codes = [run(args.suite, network, iterations) for network in args.only]
    sys.exit(max(codes, default=0))


if __name__ == "__main__":
    main()
