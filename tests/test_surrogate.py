import json
from pathlib import Path

import numpy as np
import torch

from frc_lab.surrogate import (
    DynamicsConfig,
    TrajectoryParameters,
    evaluate_one_step,
    generate_dataset,
    rollout_surrogate,
    simulate_trajectory,
    split_by_trajectory,
    train_surrogate,
)


ROOT = Path(__file__).resolve().parents[1]


def test_simulator_is_reproducible_and_applies_impulses() -> None:
    config = DynamicsConfig(steps=30)
    parameters = TrajectoryParameters(
        initial_error=(1.0, -0.8),
        noise_scale=0.03,
        deadline=0.9,
        impulse_gain=0.5,
        impulse_interval=10,
        impulse_phase=4,
    )
    features_a, targets_a = simulate_trajectory(parameters, seed=11, config=config)
    features_b, targets_b = simulate_trajectory(parameters, seed=11, config=config)

    np.testing.assert_allclose(features_a, features_b)
    np.testing.assert_allclose(targets_a, targets_b)
    assert np.flatnonzero(features_a[:, 10]).tolist() == [4, 14, 24]
    assert set(np.unique(features_a[:, 8:10])).issubset({0.0, 1.0})


def test_group_split_has_no_trajectory_leakage() -> None:
    data = generate_dataset(20, seed=5, config=DynamicsConfig(steps=12))
    split = split_by_trajectory(data, seed=5)
    id_sets = [set(np.unique(split[name].trajectory_ids)) for name in ("train", "validation", "test")]

    assert id_sets[0].isdisjoint(id_sets[1])
    assert id_sets[0].isdisjoint(id_sets[2])
    assert id_sets[1].isdisjoint(id_sets[2])
    assert set.union(*id_sets) == set(range(20))


def test_small_surrogate_trains_and_rolls_out() -> None:
    torch.set_num_threads(1)
    config = DynamicsConfig(steps=28)
    data = generate_dataset(24, seed=17, config=config)
    split = split_by_trajectory(data, seed=17)
    model, history = train_surrogate(
        split["train"],
        split["validation"],
        config=config,
        epochs=12,
        batch_size=128,
        seed=17,
    )
    metrics = evaluate_one_step(model, split["test"])

    first_id = np.unique(split["test"].trajectory_ids)[0]
    mask = split["test"].trajectory_ids == first_id
    truth, prediction = rollout_surrogate(
        model,
        split["test"].features[mask],
        split["test"].targets[mask],
        config=config,
    )

    assert history[-1]["validation_loss"] < history[0]["validation_loss"]
    assert metrics["nrmse"] < metrics["persistence_nrmse"]
    assert truth.shape == prediction.shape == (config.steps + 1, 2)
    assert np.isfinite(prediction).all()


def test_committed_artifacts_preserve_claim_boundaries() -> None:
    paper = json.loads(
        (ROOT / "data" / "candidate_paper_evidence.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (ROOT / "models" / "surrogate_metrics.json").read_text(encoding="utf-8")
    )

    assert paper["doi"] == "10.1016/j.neunet.2025.108100"
    assert paper["candidate_author_position"] == "2/5"
    assert paper["project_boundary"] == {
        "paper_reproduction": False,
        "pinn_paper": False,
        "frc_experiment": False,
        "description": "项目仅迁移切换、时滞、随机扰动、脉冲和时间约束控制的结构，使用缩放后的二维合成基准训练 PyTorch 代理模型。",
    }
    assert metrics["experiment"]["split_disjoint"] is True
    assert not any(metrics["claims"].values())
    assert metrics["model"]["torchscript_max_abs_error"] == 0.0
