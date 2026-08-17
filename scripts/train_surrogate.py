"""Train, evaluate and export the paper-inspired PyTorch surrogate."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frc_lab.surrogate import (  # noqa: E402
    DynamicsConfig,
    TrajectoryParameters,
    benchmark_latency,
    evaluate_one_step,
    export_torchscript,
    generate_dataset,
    normalized_rmse,
    rollout_surrogate,
    save_checkpoint,
    simulate_trajectory,
    split_by_trajectory,
    train_surrogate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--trajectories", type=int, default=90)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    config = DynamicsConfig()
    dataset = generate_dataset(args.trajectories, seed=args.seed, config=config)
    split = split_by_trajectory(dataset, seed=args.seed)
    ood = generate_dataset(16, seed=args.seed + 1, domain="ood", config=config)

    model, history = train_surrogate(
        split["train"],
        split["validation"],
        config=config,
        epochs=args.epochs,
        seed=args.seed,
    )
    in_domain_metrics = evaluate_one_step(model, split["test"], config)
    ood_metrics = evaluate_one_step(model, ood, config)

    example_parameters = TrajectoryParameters(
        initial_error=(1.18, -0.96),
        noise_scale=0.045,
        deadline=0.90,
        impulse_gain=0.62,
        impulse_interval=19,
        impulse_phase=8,
    )
    example_features, example_targets = simulate_trajectory(
        example_parameters,
        seed=args.seed + 2,
        config=config,
    )
    true_states, predicted_states = rollout_surrogate(
        model,
        example_features,
        example_targets,
        config=config,
    )
    rollout_nrmse = normalized_rmse(true_states[1:], predicted_states[1:])
    latency = benchmark_latency(model, example_features)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    train_ids = set(np.unique(split["train"].trajectory_ids).tolist())
    validation_ids = set(np.unique(split["validation"].trajectory_ids).tolist())
    test_ids = set(np.unique(split["test"].trajectory_ids).tolist())
    split_disjoint = not (
        train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids
    )
    metrics = {
        "experiment": {
            "seed": args.seed,
            "epochs": args.epochs,
            "trajectory_count": args.trajectories,
            "split_strategy": "trajectory_group_split",
            "split_disjoint": split_disjoint,
            "train_trajectories": len(train_ids),
            "validation_trajectories": len(validation_ids),
            "test_trajectories": len(test_ids),
            "ood_trajectories": len(np.unique(ood.trajectory_ids)),
        },
        "model": {
            "framework": f"PyTorch {torch.__version__}",
            "architecture": "11-48-48-2 residual MLP (SiLU)",
            "parameter_count": parameter_count,
            "physics_loss_weight": 0.18,
        },
        "in_domain": in_domain_metrics,
        "ood": ood_metrics,
        "rollout": {"nrmse": rollout_nrmse, "steps": config.steps},
        "latency_cpu_eager": latency,
        "history": history,
        "claims": {
            "paper_reproduction": False,
            "pinn_equivalence": False,
            "frc_validated": False,
            "speedup_claimed": False,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "sdsnn_surrogate.pt"
    torchscript_path = args.output_dir / "sdsnn_surrogate.torchscript.pt"
    save_checkpoint(model, checkpoint_path, config=config, metrics=metrics)
    export_torchscript(model, torchscript_path)

    scripted = torch.jit.load(io.BytesIO(torchscript_path.read_bytes()))
    check_input = torch.from_numpy(example_features[:8])
    with torch.no_grad():
        export_error = float(torch.max(torch.abs(model(check_input) - scripted(check_input))))
    metrics["model"]["torchscript_max_abs_error"] = export_error
    save_checkpoint(model, checkpoint_path, config=config, metrics=metrics)

    (args.output_dir / "surrogate_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    time = np.arange(config.steps + 1) * config.dt
    rollout_frame = pd.DataFrame(
        {
            "time_s": time,
            "true_error_1": true_states[:, 0],
            "true_error_2": true_states[:, 1],
            "predicted_error_1": predicted_states[:, 0],
            "predicted_error_2": predicted_states[:, 1],
            "mode_1": np.r_[example_features[:, 8], example_features[-1, 8]],
            "mode_2": np.r_[example_features[:, 9], example_features[-1, 9]],
            "impulse": np.r_[0.0, example_features[:, 10]],
        }
    )
    rollout_frame.to_csv(args.output_dir / "example_rollout.csv", index=False)
    print(json.dumps({key: value for key, value in metrics.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
