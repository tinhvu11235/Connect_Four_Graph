"""Two tiny CPU iterations of the AlphaZero repo's real training loop for each network."""

import json

import pytest

pytest.importorskip("torch")

from common import repo_bridge  # noqa: E402,F401  (fails early if the AlphaZero repo is missing)

TINY = [
    "--device", "cpu", "--iterations", "2", "--games-per-iteration", "2", "--slots", "2",
    "--mcts-simulations", "4", "--replay-capacity", "1000", "--min-replay-size", "16",
    "--batch-size", "16", "--eval-games", "0", "--residual-blocks", "1", "--channels", "16",
    "--no-resume",
]


@pytest.mark.parametrize("network", ["baseline_cnn", "method1", "method2"])
def test_two_iterations(tmp_path, network):
    """Each entry point trains, writes ``resources.json`` and a checkpoint the repo's loader reopens."""

    run_dir = tmp_path / network
    argv = TINY + ["--run-dir", str(run_dir)]
    if network == "baseline_cnn":
        from baseline_cnn.train import main
    elif network == "method1":
        from method1.train import main
    else:
        from method2.train import main
    main(argv)
    resources = json.loads((run_dir / "resources.json").read_text())
    expected = {"baseline_cnn": "residual_gpu_v1", "method1": "method1_v1", "method2": "method2_v1"}[network]
    assert resources["model_config"]["architecture"] == expected
    assert resources["iterations"] == 2
    assert (run_dir / "models" / "latest.pt").exists()
    from train_alphazero.load_checkpoint import load_policy_value_checkpoint

    model, _ = load_policy_value_checkpoint(run_dir / "models" / "latest.pt")
    assert model.model_config["architecture"] == expected
