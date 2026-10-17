"""Run the AlphaZero repo's own training pipeline with our networks, without editing the repo.

The repo (AlphaZeroConnect4Train) is imported read-only: no ``__pycache__`` is written
into it.  Its ``build_policy_value_model`` is wrapped so that method 1 / method 2 are
built by our code and everything else (the baseline residual CNN) by the original;
self-play, tree search, replay, mirror augmentation, loss, checkpoints and the
evaluation against Fujita's AlphaZero are the repo's, unchanged.

Where the repo is looked for: ``$ALPHAZERO_REPO`` if set, else a folder
``AlphaZeroConnect4Train`` next to this repository.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Callable

sys.dont_write_bytecode = True  # keep the repo checkout untouched
REPO_ROOT = Path(os.environ.get("ALPHAZERO_REPO", Path(__file__).resolve().parents[2] / "AlphaZeroConnect4Train"))
if not (REPO_ROOT / "train_alphazero").is_dir():
    raise ImportError(
        f"AlphaZeroConnect4Train not found at {REPO_ROOT}; clone it next to this repo or set ALPHAZERO_REPO"
    )
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

import train_alphazero.build_network as _build_network  # noqa: E402
import train_alphazero.load_checkpoint as _load_checkpoint  # noqa: E402
import train_alphazero.run_training as _run_training  # noqa: E402
from train_alphazero.configure_training import TrainingConfig, parse_args  # noqa: E402

_ORIGINAL_BUILD = _build_network.build_policy_value_model
_BUILDERS: dict[str, Callable[[dict], torch.nn.Module]] = {}


def build_any(config: dict) -> torch.nn.Module:
    """Build the network described by a ``model_config``: ours if registered, else the repo's.

    Side effects: none (returns a new, randomly initialised module).
    """

    builder = _BUILDERS.get(config.get("architecture"))
    return builder(config) if builder else _ORIGINAL_BUILD(config)


def register(architecture: str, builder: Callable[[dict], torch.nn.Module]) -> None:
    """Make the repo build ``architecture`` with ``builder``.

    Side effects: adds to the module-level registry and replaces ``build_policy_value_model``
    in the repo's already-imported modules (in memory only; no file is changed).
    """

    _BUILDERS[architecture] = builder
    for module in (_build_network, _load_checkpoint, _run_training):
        module.build_policy_value_model = build_any


def register_all() -> None:
    """Register method 1 and method 2, including their legacy architecture tags (old checkpoints).

    Idempotent.  Side effects: see ``register``.
    """

    from method1 import model as method1
    from method2 import model as method2

    for method in (method1, method2):
        for tag in (method.ARCHITECTURE, *method.LEGACY_ARCHITECTURES):
            register(tag, method.build)


def train(argv: list[str] | None, model_config: Callable[[TrainingConfig], dict] | None, description: str) -> None:
    """Run ``python -m train_alphazero`` with an optional model override, then write ``resources.json``.

    ``argv`` takes the repo's flags (``--run-dir``, ``--iterations``, ...).  ``model_config``
    maps the parsed config to our ``model_config`` (``None`` keeps the repo's CNN).
    Side effects: everything the repo's training writes into ``--run-dir`` (``state.pt``,
    ``models/*.pt``, ``train_metrics.csv``, ``config.json``), plus ``<run-dir>/resources.json``
    (parameters, wall time, peak RAM/GPU memory); uses the GPU if configured; prints progress.
    """

    register_all()
    config, resume = parse_args(argv, description=description)
    if model_config is not None:
        config.model_config = lambda: model_config(config)  # instance attribute; asdict() ignores it
    session = _run_training.TrainingSession(config, resume=resume)
    parameters = sum(p.numel() for p in session.model.parameters())
    print(f"model {session.model_config['architecture']}: {parameters:,} parameters", flush=True)
    if session.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(session.device)
    started = perf_counter()
    session.run()
    resources = {
        "model_config": session.model_config,
        "parameters": parameters,
        "wall_seconds": perf_counter() - started,
        "iterations": session.iteration,
        "games_total": session.games_total,
        "device": str(session.device),
    }
    try:
        import psutil

        memory = psutil.Process().memory_info()
        resources["peak_ram_mb"] = getattr(memory, "peak_wset", memory.rss) / 2**20  # peak on Windows
    except ImportError:
        pass
    if session.device.type == "cuda":
        resources["gpu_name"] = torch.cuda.get_device_name(session.device)
        resources["peak_gpu_allocated_mb"] = torch.cuda.max_memory_allocated(session.device) / 2**20
        resources["peak_gpu_reserved_mb"] = torch.cuda.max_memory_reserved(session.device) / 2**20
    (Path(config.run_dir) / "resources.json").write_text(json.dumps(resources, indent=2), encoding="utf-8")
    print(json.dumps(resources, indent=2), flush=True)
