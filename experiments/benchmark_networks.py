"""Micro-benchmark of every network: parameters, inference latency, train-step time, peak GPU memory.

Inference uses a batch of 256 positions (the self-play leaf batch); the train step a
batch of 1024 with AdamW, bf16 autocast on CUDA, like the repo's training.

    python -m experiments.benchmark_networks

Evidence written to ``results/benchmark/``: ``benchmark.json`` (numbers + environment)
and ``benchmark.log`` (console output).
"""

from __future__ import annotations

import json
import platform
import random
from datetime import datetime
from pathlib import Path
from time import perf_counter

from common import repo_bridge

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common.connect4_rules import ConnectFourEnv  # noqa: E402
from method1.model import model_config as method1_config  # noqa: E402
from method2.model import model_config as method2_config  # noqa: E402
from train_alphazero.build_network import residual_model_config  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "benchmark"
MODELS = {
    "Baseline CNN 128x6 (repo default)": residual_model_config(128, 6),
    "Baseline CNN 256x6": residual_model_config(256, 6),
    "Method 1 160x6, line-of-sight": method1_config(160, 6, True),
    "Method 1 256x6, no line-of-sight (ablation)": method1_config(256, 6, False),
    "Method 2 256x6": method2_config(256, 6),
}


def random_states(count: int, seed: int = 0) -> torch.Tensor:
    """Return ``count`` canonical positions ``(count, 1, 7, 6)`` from random games of 0..35 moves.

    Side effects: none (own RNG seeded with ``seed``).
    """

    rng = random.Random(seed)
    boards = []
    for _ in range(count):
        env = ConnectFourEnv()
        for _ in range(rng.randint(0, 35)):
            if env.done:
                break
            env.step(rng.choice(env.validinputs))
        boards.append(env.state * (1 if env.turn == "red" else -1))
    return torch.tensor(np.stack(boards), dtype=torch.float32)[:, None]


def timed(fn, repeats: int, device: torch.device) -> float:
    """Return the mean seconds per call of ``fn`` over ``repeats`` calls, after 5 warm-up calls.

    Side effects: whatever ``fn`` does; synchronises the GPU around the timing.
    """

    for _ in range(5):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = perf_counter()
    for _ in range(repeats):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (perf_counter() - started) / repeats


def bench(config: dict, device: torch.device, states: torch.Tensor) -> dict:
    """Benchmark one ``model_config`` on ``device``; return parameters, latencies and peak GPU memory.

    Side effects: allocates the model on ``device``; enables ``cudnn.benchmark``; resets the
    CUDA peak-memory counters.
    """

    repo_bridge.register_all()
    model = repo_bridge.build_any(config).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
    params = sum(p.numel() for p in model.parameters())
    amp = device.type == "cuda"

    leaf = states[:256].to(device).contiguous(memory_format=torch.channels_last)
    model.eval()

    def infer():
        """One inference call on the 256-position batch.  Side effects: none (inference mode)."""

        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            model(leaf)

    infer_seconds = timed(infer, 100, device)

    batch = states[:1024].to(device).contiguous(memory_format=torch.channels_last)
    target_policy = torch.softmax(torch.randn(len(batch), 7, device=device), dim=1)
    target_value = torch.rand(len(batch), device=device) * 2 - 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=device.type == "cuda")
    model.train()

    def step():
        """One optimizer step on random targets.  Side effects: updates the benchmark model's weights."""

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits, value = model(batch)
        loss = torch.nn.functional.cross_entropy(logits.float(), target_policy) + torch.nn.functional.mse_loss(
            value.float(), target_value
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    step_seconds = timed(step, 30, device)
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else float("nan")
    return {
        "architecture": config["architecture"],
        "parameters": params,
        "infer_ms_batch256": infer_seconds * 1e3,
        "positions_per_s": 256 / infer_seconds,
        "train_step_ms_batch1024": step_seconds * 1e3,
        "train_peak_gpu_mb": peak,
    }


def main() -> None:
    """Benchmark every entry of ``MODELS`` and write ``results/benchmark/benchmark.{json,log}``.

    Side effects: uses the GPU if available; overwrites both evidence files; prints the table.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    states = random_states(1024)
    environment = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    lines = [f"{k}: {v}" for k, v in environment.items()]
    for line in lines:
        print(line, flush=True)
    results = {}
    for name, config in MODELS.items():
        results[name] = r = bench(config, device, states)
        line = (f"{name:44s} params {r['parameters'] / 1e6:5.2f}M | infer@256 {r['infer_ms_batch256']:6.2f} ms"
                f" | train@1024 {r['train_step_ms_batch1024']:7.1f} ms | peak {r['train_peak_gpu_mb']:7.0f} MB")
        lines.append(line)
        print(line, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "benchmark.json").write_text(json.dumps({**environment, "results": results}, indent=2), encoding="utf-8")
    (OUT_DIR / "benchmark.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
