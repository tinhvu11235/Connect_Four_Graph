"""Extrapolate the short trials to a full 90k-game run (352 iterations x 256 games).

Two estimates per network:

* direct - the trial's steady-state iteration (self-play + 391 optimizer steps + eval) x 352;
* scaled - the repo's real 352-iteration CNN log (``results/train_only_fujita`` of
           AlphaZeroConnect4Train, 90,112 games), each component (self-play / training /
           eval) multiplied by trial(network) / trial(baseline CNN).  The log captures how
           iterations slow down as play gets stronger, which 3 iterations cannot.

Inputs (all evidence files): ``results/trials/<network>/``, ``results/peak_ram/<network>/``,
``results/benchmark/benchmark.json`` and the repo's ``results/train_only_fujita/train_metrics.csv``.

    python -m experiments.estimate_full_run

Output: ``results/estimate/full_run_estimate.md`` (also printed).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from common.repo_bridge import REPO_ROOT

RESULTS = Path(__file__).resolve().parents[1] / "results"
NETWORKS = {
    "baseline_cnn": ("Baseline CNN 256x6", "Baseline CNN 256x6"),
    "method1": ("Method 1 160x6 + line-of-sight", "Method 1 160x6, line-of-sight"),
    "method2": ("Method 2 256x6", "Method 2 256x6"),
}  # folder -> (column title, key in benchmark.json)
ITERATIONS = 352
GAMES = 90_112
STEADY_STEPS = 391  # min(3*256, 8*50000/1024, 2000) once the 50k replay is full


def rows(path: Path) -> list[dict[str, float]]:
    """Read a ``train_metrics.csv`` as a list of float dicts (empty/``nan`` -> NaN).  Side effects: reads ``path``."""

    with path.open(newline="", encoding="utf-8") as handle:
        return [{k: float(v) if v not in ("", "nan") else math.nan for k, v in r.items()} for r in csv.DictReader(handle)]


def trial_stats(network: str) -> dict[str, float]:
    """Summarise ``results/trials/<network>``: mean seconds per phase, losses, scores, memory.

    Side effects: reads the evidence files.
    """

    folder = RESULTS / "trials" / network
    data = rows(folder / "train_metrics.csv")
    steps = sum(r["optimizer_steps"] for r in data)
    resources = json.loads((folder / "resources.json").read_text(encoding="utf-8"))
    ram_file = RESULTS / "peak_ram" / network / "resources.json"
    ram = json.loads(ram_file.read_text(encoding="utf-8")) if ram_file.exists() else {}
    last = data[-1]
    return {
        "iterations": len(data),
        "selfplay_s": sum(r["selfplay_seconds"] for r in data) / len(data),
        "step_s": sum(r["training_seconds"] for r in data) / steps if steps else math.nan,
        "eval_s": sum(r["eval_seconds"] for r in data) / len(data),
        "other_s": sum(r["iteration_seconds"] - r["selfplay_seconds"] - r["training_seconds"] - r["eval_seconds"]
                       for r in data) / len(data),
        "eval_scores": [r["eval_score"] for r in data],
        "policy_loss": last["policy_loss"],
        "value_loss": last["value_loss"],
        "parameters": resources["parameters"],
        "peak_gpu_mb": resources.get("peak_gpu_reserved_mb", math.nan),
        "peak_ram_mb": ram.get("peak_ram_mb", math.nan),
        "wall_s": resources["wall_seconds"],
    }


def baseline_log() -> dict[str, float]:
    """Total seconds per phase of the repo's 352-iteration CNN run, plus its first 3 iterations and final score.

    Side effects: reads the repo's ``results/train_only_fujita/train_metrics.csv``.
    """

    data = rows(REPO_ROOT / "results" / "train_only_fujita" / "train_metrics.csv")
    total = {k: sum(r[k] for r in data) for k in ("selfplay_seconds", "training_seconds", "eval_seconds", "iteration_seconds")}
    total["other_seconds"] = total["iteration_seconds"] - total["selfplay_seconds"] - total["training_seconds"] - total["eval_seconds"]
    total["first3_selfplay"] = sum(r["selfplay_seconds"] for r in data[:3]) / 3
    total["first3_eval"] = sum(r["eval_seconds"] for r in data[:3]) / 3
    total["steps"] = sum(r["optimizer_steps"] for r in data)
    total["final_score"] = data[-1]["eval_score"]
    return total


def hours(seconds: float) -> str:
    """Format seconds as ``"12.3 h"``.  Side effects: none."""

    return f"{seconds / 3600:.1f} h"


def estimate(stats: dict, cnn: dict, log: dict) -> dict:
    """Return the direct and scaled 90k-game estimates of one network (seconds), with and without eval every iteration.

    Side effects: none.
    """

    ratios = {
        "selfplay": stats["selfplay_s"] / cnn["selfplay_s"],
        "training": stats["step_s"] / cnn["step_s"],
        "eval": stats["eval_s"] / cnn["eval_s"],
    }
    direct = ITERATIONS * (stats["selfplay_s"] + STEADY_STEPS * stats["step_s"] + stats["eval_s"] + stats["other_s"])
    parts = {
        "selfplay": log["selfplay_seconds"] * ratios["selfplay"],
        "training": log["training_seconds"] * ratios["training"],
        "eval": log["eval_seconds"] * ratios["eval"],
        "other": log["other_seconds"],
    }
    scaled = sum(parts.values())
    return {
        "ratios": ratios,
        "direct": direct,
        "direct_eval10": direct - ITERATIONS * 0.9 * stats["eval_s"],
        "scaled": scaled,
        "scaled_eval10": scaled - 0.9 * parts["eval"],
        "parts": parts,
    }


def main() -> None:
    """Build the comparison table and write ``results/estimate/full_run_estimate.md``.

    Side effects: overwrites that file; prints it.
    """

    bench = json.loads((RESULTS / "benchmark" / "benchmark.json").read_text(encoding="utf-8"))
    log = baseline_log()
    stats = {network: trial_stats(network) for network in NETWORKS}
    cnn = stats["baseline_cnn"]
    table = []
    for network, (title, bench_key) in NETWORKS.items():
        table.append((title, stats[network], bench["results"][bench_key], estimate(stats[network], cnn, log)))

    lines = [
        "# Connect 4: baseline CNN vs method 1 vs method 2 - trial runs and 90k-game estimate",
        "",
        "Generated by `python -m experiments.estimate_full_run` from the evidence in `results/`.",
        "",
        f"GPU: {bench.get('gpu', '')}.  Trials: {cnn['iterations']} iterations x 256 games each, 200 simulations,"
        " 100-game evaluation vs Fujita AlphaZero every iteration, settings of `results/train_only_fujita`.",
        f"Reference log (AlphaZero repo, CNN 256x6): {ITERATIONS} iterations, {GAMES:,} games, {hours(log['iteration_seconds'])};"
        f" its first 3 iterations averaged {log['first3_selfplay']:.0f} s self-play and {log['first3_eval']:.0f} s eval"
        f" (this machine's CNN trial: {cnn['selfplay_s']:.0f} s and {cnn['eval_s']:.0f} s).",
        "",
        "| | " + " | ".join(t[0] for t in table) + " |",
        "|---|" + "---:|" * len(table),
    ]

    def row(title, fn):
        """Append one table row; ``fn`` formats a column.  Side effects: appends to ``lines``."""

        lines.append(f"| {title} | " + " | ".join(fn(t) for t in table) + " |")

    row("Parameters", lambda t: f"{t[1]['parameters'] / 1e6:.2f} M")
    row("Inference, batch 256 (ms, benchmark)", lambda t: f"{t[2]['infer_ms_batch256']:.1f}")
    row("Train step, batch 1024 (ms, benchmark)", lambda t: f"{t[2]['train_step_ms_batch1024']:.0f}")
    row("Peak GPU memory, training run", lambda t: f"{t[1]['peak_gpu_mb']:.0f} MB")
    row("Peak process RAM (1 iteration)", lambda t: f"{t[1]['peak_ram_mb']:.0f} MB")
    row("Self-play per iteration (256 games)", lambda t: f"{t[1]['selfplay_s']:.0f} s (x{t[3]['ratios']['selfplay']:.2f})")
    row("Optimizer step in trial", lambda t: f"{t[1]['step_s'] * 1e3:.0f} ms (x{t[3]['ratios']['training']:.2f})")
    row("Eval vs Fujita per iteration (100 games)", lambda t: f"{t[1]['eval_s']:.0f} s (x{t[3]['ratios']['eval']:.2f})")
    row("Steady iteration (self-play + 391 steps + eval)", lambda t: f"{t[3]['direct'] / ITERATIONS / 60:.1f} min")
    row("**90k games, direct x352**", lambda t: f"**{hours(t[3]['direct'])}**")
    row("**90k games, scaled from repo log**", lambda t: f"**{hours(t[3]['scaled'])}**")
    row("... of which self-play / training / eval",
        lambda t: " / ".join(hours(t[3]["parts"][k]) for k in ("selfplay", "training", "eval")))
    row("90k games, eval every 10 it (direct / scaled)",
        lambda t: f"{hours(t[3]['direct_eval10'])} / {hours(t[3]['scaled_eval10'])}")
    row("Score vs Fujita per trial iteration", lambda t: ", ".join(f"{x:.2f}" for x in t[1]["eval_scores"]))
    row(f"Policy / value loss at iteration {cnn['iterations']}",
        lambda t: f"{t[1]['policy_loss']:.3f} / {t[1]['value_loss']:.3f}")
    lines += [
        "",
        f"The repo's CNN reached {log['final_score']:.2f} vs Fujita after 90k games; "
        "3 iterations say nothing about final strength, only that the pipeline learns.",
    ]
    text = "\n".join(lines)
    out = RESULTS / "estimate" / "full_run_estimate.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
