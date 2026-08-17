"""Paper-inspired switched dynamics and a small PyTorch surrogate.

The benchmark retains four structural ingredients from Chen et al. (2026):
state-dependent switching, delayed state, stochastic disturbance and impulses.
It is a normalized synthetic benchmark, not a reproduction of the paper's
theorems, LMI solution, PI controller or numerical figures.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import torch
from torch import nn


FEATURE_NAMES = (
    "error_1",
    "error_2",
    "delayed_error_1",
    "delayed_error_2",
    "time_to_deadline",
    "deadline",
    "impulse_gain",
    "noise_scale",
    "mode_1",
    "mode_2",
    "impulse_flag",
)


@dataclass(frozen=True)
class DynamicsConfig:
    """Configuration for the normalized two-state synchronization benchmark."""

    dt: float = 0.02
    steps: int = 120
    delay_steps: int = 5
    switch_threshold: float = 0.75
    state_limit: float = 4.0
    theta: float = 0.60
    nu: float = 1.40
    fixed_gain: float = 0.32
    high_order_gain: float = 0.06
    prescribed_ramp: float = 0.45


@dataclass(frozen=True)
class TrajectoryParameters:
    initial_error: tuple[float, float]
    noise_scale: float
    deadline: float
    impulse_gain: float
    impulse_interval: int
    impulse_phase: int


@dataclass(frozen=True)
class TrajectoryData:
    features: np.ndarray
    targets: np.ndarray
    trajectory_ids: np.ndarray

    def subset(self, ids: np.ndarray) -> "TrajectoryData":
        mask = np.isin(self.trajectory_ids, ids)
        return TrajectoryData(
            features=self.features[mask],
            targets=self.targets[mask],
            trajectory_ids=self.trajectory_ids[mask],
        )


# The switching matrices follow the row-wise pattern of Example 1 in the source
# paper, but are deliberately scaled for this normalized Euler benchmark.
_A_LOW = np.array([[3.10, -0.13], [-3.90, 1.00]], dtype=np.float32) * 0.12
_A_HIGH = np.array([[3.15, -0.125], [-3.85, 1.10]], dtype=np.float32) * 0.12
_B_LOW = np.array([[-1.30, -0.80], [2.40, -2.40]], dtype=np.float32) * 0.08
_B_HIGH = np.array([[-1.20, -0.70], [2.50, -2.30]], dtype=np.float32) * 0.08
_DECAY = np.array([1.15, 1.20], dtype=np.float32)
_PROPORTIONAL = np.array([0.85, 0.95], dtype=np.float32)


def _row_switched_matrix(low: np.ndarray, high: np.ndarray, modes: np.ndarray) -> np.ndarray:
    return np.where(modes[:, None], high, low)


def deterministic_drift(
    error: np.ndarray,
    delayed_error: np.ndarray,
    time_to_deadline: float,
    deadline: float,
    config: DynamicsConfig = DynamicsConfig(),
) -> np.ndarray:
    """Evaluate the deterministic part of the normalized error dynamics."""

    error = np.asarray(error, dtype=np.float32)
    delayed_error = np.asarray(delayed_error, dtype=np.float32)
    modes = np.abs(error) >= config.switch_threshold
    current_matrix = _row_switched_matrix(_A_LOW, _A_HIGH, modes)
    delay_matrix = _row_switched_matrix(_B_LOW, _B_HIGH, modes)
    neural_term = current_matrix @ np.tanh(error) + delay_matrix @ np.tanh(delayed_error)

    progress = 1.0 - np.clip(time_to_deadline / max(deadline, config.dt), 0.0, 1.0)
    schedule = 1.0 + config.prescribed_ramp * progress
    nonlinear_control = (
        config.fixed_gain * np.sign(error) * np.abs(error) ** config.theta
        + config.high_order_gain * np.sign(error) * np.abs(error) ** config.nu
    )
    return -_DECAY * error + neural_term - _PROPORTIONAL * error - schedule * nonlinear_control


def sample_parameters(
    rng: np.random.Generator,
    domain: Literal["in_domain", "ood"] = "in_domain",
) -> TrajectoryParameters:
    """Sample one trajectory configuration from the training or stress domain."""

    if domain == "in_domain":
        amplitude = rng.uniform(0.45, 1.45, size=2)
        noise_scale = float(rng.uniform(0.01, 0.075))
        deadline = float(rng.uniform(0.70, 1.30))
        impulse_gain = float(rng.uniform(0.35, 0.85))
        impulse_interval = int(rng.integers(14, 25))
    else:
        amplitude = rng.uniform(1.50, 2.20, size=2)
        noise_scale = float(rng.uniform(0.09, 0.16))
        deadline = float(rng.choice([rng.uniform(0.50, 0.67), rng.uniform(1.40, 1.65)]))
        impulse_gain = float(rng.uniform(0.88, 1.08))
        impulse_interval = int(rng.integers(10, 28))

    signs = rng.choice(np.array([-1.0, 1.0]), size=2)
    phase = int(rng.integers(3, max(4, impulse_interval)))
    return TrajectoryParameters(
        initial_error=tuple((amplitude * signs).tolist()),
        noise_scale=noise_scale,
        deadline=deadline,
        impulse_gain=impulse_gain,
        impulse_interval=impulse_interval,
        impulse_phase=phase,
    )


def simulate_trajectory(
    parameters: TrajectoryParameters,
    *,
    seed: int,
    config: DynamicsConfig = DynamicsConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one Euler-Maruyama trajectory and one-step training pairs."""

    rng = np.random.default_rng(seed)
    error = np.asarray(parameters.initial_error, dtype=np.float32)
    history = [error.copy()]
    features = np.empty((config.steps, len(FEATURE_NAMES)), dtype=np.float32)
    targets = np.empty((config.steps, 2), dtype=np.float32)

    for step in range(config.steps):
        time = step * config.dt
        delayed_error = history[max(0, len(history) - 1 - config.delay_steps)]
        modes = np.abs(error) >= config.switch_threshold
        impulse_flag = (
            step >= parameters.impulse_phase
            and (step - parameters.impulse_phase) % parameters.impulse_interval == 0
        )
        time_to_deadline = max(parameters.deadline - time, 0.0)
        features[step] = np.array(
            [
                error[0],
                error[1],
                delayed_error[0],
                delayed_error[1],
                time_to_deadline,
                parameters.deadline,
                parameters.impulse_gain,
                parameters.noise_scale,
                float(modes[0]),
                float(modes[1]),
                float(impulse_flag),
            ],
            dtype=np.float32,
        )

        drift = deterministic_drift(
            error,
            delayed_error,
            time_to_deadline,
            parameters.deadline,
            config,
        )
        diffusion = parameters.noise_scale * (0.08 + 0.06 * np.abs(error))
        next_error = error + config.dt * drift
        next_error += diffusion * np.sqrt(config.dt) * rng.normal(size=2)
        if impulse_flag:
            next_error *= parameters.impulse_gain
        next_error = np.clip(next_error, -config.state_limit, config.state_limit).astype(np.float32)

        targets[step] = next_error
        error = next_error
        history.append(error.copy())

    return features, targets


def generate_dataset(
    trajectory_count: int,
    *,
    seed: int,
    domain: Literal["in_domain", "ood"] = "in_domain",
    config: DynamicsConfig = DynamicsConfig(),
) -> TrajectoryData:
    """Generate a deterministic collection of independent trajectories."""

    rng = np.random.default_rng(seed)
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    id_parts: list[np.ndarray] = []
    for trajectory_id in range(trajectory_count):
        parameters = sample_parameters(rng, domain)
        features, targets = simulate_trajectory(
            parameters,
            seed=seed * 10_000 + trajectory_id,
            config=config,
        )
        feature_parts.append(features)
        target_parts.append(targets)
        id_parts.append(np.full(config.steps, trajectory_id, dtype=np.int32))
    return TrajectoryData(
        features=np.concatenate(feature_parts),
        targets=np.concatenate(target_parts),
        trajectory_ids=np.concatenate(id_parts),
    )


def split_by_trajectory(
    data: TrajectoryData,
    *,
    seed: int = 2026,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, TrajectoryData]:
    """Split entire trajectories so adjacent samples cannot leak across sets."""

    unique_ids = np.unique(data.trajectory_ids)
    if len(unique_ids) < 3:
        raise ValueError("At least three trajectories are required for a three-way split.")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_ids)
    train_count = max(1, min(len(shuffled) - 2, round(len(shuffled) * train_fraction)))
    validation_count = max(
        1,
        min(len(shuffled) - train_count - 1, round(len(shuffled) * validation_fraction)),
    )
    train_end = train_count
    validation_end = train_count + validation_count
    return {
        "train": data.subset(shuffled[:train_end]),
        "validation": data.subset(shuffled[train_end:validation_end]),
        "test": data.subset(shuffled[validation_end:]),
    }


class SurrogateModel(nn.Module):
    """A compact residual MLP with persisted feature and target scaling."""

    def __init__(self, hidden_size: int = 48) -> None:
        super().__init__()
        input_size = len(FEATURE_NAMES)
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 2),
        )
        self.register_buffer("feature_mean", torch.zeros(input_size))
        self.register_buffer("feature_scale", torch.ones(input_size))
        self.register_buffer("delta_mean", torch.zeros(2))
        self.register_buffer("delta_scale", torch.ones(2))

    def set_scalers(self, features: np.ndarray, targets: np.ndarray) -> None:
        deltas = targets - features[:, :2]
        self.feature_mean.copy_(torch.from_numpy(features.mean(axis=0).astype(np.float32)))
        self.feature_scale.copy_(
            torch.from_numpy(np.maximum(features.std(axis=0), 1e-4).astype(np.float32))
        )
        self.delta_mean.copy_(torch.from_numpy(deltas.mean(axis=0).astype(np.float32)))
        self.delta_scale.copy_(
            torch.from_numpy(np.maximum(deltas.std(axis=0), 1e-4).astype(np.float32))
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_scale
        return self.network(normalized) * self.delta_scale + self.delta_mean

    def predict_next(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, :2] + self(features)


def _torch_drift(features: torch.Tensor, config: DynamicsConfig) -> torch.Tensor:
    error = features[:, :2]
    delayed_error = features[:, 2:4]
    modes = features[:, 8:10].unsqueeze(-1)
    dtype = features.dtype
    device = features.device

    a_low = torch.as_tensor(_A_LOW, dtype=dtype, device=device).unsqueeze(0)
    a_high = torch.as_tensor(_A_HIGH, dtype=dtype, device=device).unsqueeze(0)
    b_low = torch.as_tensor(_B_LOW, dtype=dtype, device=device).unsqueeze(0)
    b_high = torch.as_tensor(_B_HIGH, dtype=dtype, device=device).unsqueeze(0)
    a_rows = a_low * (1.0 - modes) + a_high * modes
    b_rows = b_low * (1.0 - modes) + b_high * modes
    neural_term = torch.sum(a_rows * torch.tanh(error).unsqueeze(1), dim=2)
    neural_term += torch.sum(b_rows * torch.tanh(delayed_error).unsqueeze(1), dim=2)

    progress = 1.0 - torch.clamp(features[:, 4] / torch.clamp(features[:, 5], min=config.dt), 0.0, 1.0)
    schedule = 1.0 + config.prescribed_ramp * progress
    nonlinear = (
        config.fixed_gain * torch.sign(error) * torch.abs(error).pow(config.theta)
        + config.high_order_gain * torch.sign(error) * torch.abs(error).pow(config.nu)
    )
    decay = torch.as_tensor(_DECAY, dtype=dtype, device=device)
    proportional = torch.as_tensor(_PROPORTIONAL, dtype=dtype, device=device)
    return -decay * error + neural_term - proportional * error - schedule[:, None] * nonlinear


def physics_expected_delta(features: torch.Tensor, config: DynamicsConfig) -> torch.Tensor:
    """Deterministic one-step map used as a soft dynamics residual target."""

    continuous_next = features[:, :2] + config.dt * _torch_drift(features, config)
    impulse_flag = features[:, 10:11]
    impulse_gain = features[:, 6:7]
    expected_next = continuous_next * (1.0 - impulse_flag + impulse_flag * impulse_gain)
    return expected_next - features[:, :2]


def train_surrogate(
    train: TrajectoryData,
    validation: TrajectoryData,
    *,
    config: DynamicsConfig = DynamicsConfig(),
    epochs: int = 70,
    batch_size: int = 512,
    learning_rate: float = 2e-3,
    physics_weight: float = 0.18,
    seed: int = 2026,
) -> tuple[SurrogateModel, list[dict[str, float]]]:
    """Train with supervised one-step loss plus a deterministic dynamics residual."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SurrogateModel()
    model.set_scalers(train.features, train.targets)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    train_features = torch.from_numpy(train.features)
    train_targets = torch.from_numpy(train.targets)
    validation_features = torch.from_numpy(validation.features)
    validation_targets = torch.from_numpy(validation.targets)
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_features), generator=generator)
        epoch_data_loss = 0.0
        epoch_physics_loss = 0.0
        seen = 0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            batch_features = train_features[indices]
            target_delta = train_targets[indices] - batch_features[:, :2]
            predicted_delta = model(batch_features)
            data_loss = torch.mean(((predicted_delta - target_delta) / model.delta_scale) ** 2)
            expected_delta = physics_expected_delta(batch_features, config)
            dynamics_loss = torch.mean(((predicted_delta - expected_delta) / model.delta_scale) ** 2)
            loss = data_loss + physics_weight * dynamics_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count = len(indices)
            epoch_data_loss += float(data_loss.detach()) * count
            epoch_physics_loss += float(dynamics_loss.detach()) * count
            seen += count

        model.eval()
        with torch.no_grad():
            validation_delta = validation_targets - validation_features[:, :2]
            validation_prediction = model(validation_features)
            validation_loss = torch.mean(
                ((validation_prediction - validation_delta) / model.delta_scale) ** 2
            )
        history.append(
            {
                "epoch": float(epoch),
                "data_loss": epoch_data_loss / seen,
                "physics_loss": epoch_physics_loss / seen,
                "validation_loss": float(validation_loss),
            }
        )
    return model, history


def normalized_rmse(reference: np.ndarray, prediction: np.ndarray) -> float:
    denominator = max(float(np.sqrt(np.mean(reference**2))), 1e-8)
    return float(np.sqrt(np.mean((reference - prediction) ** 2)) / denominator)


def evaluate_one_step(
    model: SurrogateModel,
    data: TrajectoryData,
    config: DynamicsConfig = DynamicsConfig(),
) -> dict[str, float]:
    features = torch.from_numpy(data.features)
    with torch.no_grad():
        prediction = model.predict_next(features).numpy()
        expected_delta = physics_expected_delta(features, config).numpy()
        predicted_delta = prediction - data.features[:, :2]
    persistence = data.features[:, :2]
    return {
        "nrmse": normalized_rmse(data.targets, prediction),
        "persistence_nrmse": normalized_rmse(data.targets, persistence),
        "dynamics_residual_rmse": float(np.sqrt(np.mean((predicted_delta - expected_delta) ** 2))),
    }


def rollout_surrogate(
    model: SurrogateModel,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    config: DynamicsConfig = DynamicsConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Roll the surrogate recursively while replaying only exogenous schedules."""

    predicted_states = [features[0, :2].copy()]
    true_states = np.vstack([features[0, :2], targets])
    model.eval()
    with torch.no_grad():
        for step, source_feature in enumerate(features):
            model_feature = source_feature.copy()
            model_feature[:2] = predicted_states[-1]
            delay_index = max(0, len(predicted_states) - 1 - config.delay_steps)
            model_feature[2:4] = predicted_states[delay_index]
            model_feature[8:10] = (
                np.abs(model_feature[:2]) >= config.switch_threshold
            ).astype(np.float32)
            tensor = torch.from_numpy(model_feature[None, :])
            next_state = model.predict_next(tensor).numpy()[0]
            predicted_states.append(np.clip(next_state, -config.state_limit, config.state_limit))
    return true_states.astype(np.float32), np.asarray(predicted_states, dtype=np.float32)


def benchmark_latency(model: SurrogateModel, sample: np.ndarray, repeats: int = 500) -> dict[str, float]:
    """Measure CPU eager latency without claiming system-level acceleration."""

    tensor = torch.from_numpy(sample[:1])
    model.eval()
    with torch.no_grad():
        for _ in range(25):
            model.predict_next(tensor)
        timings = []
        for _ in range(repeats):
            start = perf_counter()
            model.predict_next(tensor)
            timings.append((perf_counter() - start) * 1_000)
    return {
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
    }


def save_checkpoint(
    model: SurrogateModel,
    path: Path,
    *,
    config: DynamicsConfig,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "metrics": metrics,
            "feature_names": FEATURE_NAMES,
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[SurrogateModel, DynamicsConfig, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = SurrogateModel()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, DynamicsConfig(**payload["config"]), payload["metrics"]


def export_torchscript(model: SurrogateModel, path: Path) -> None:
    """Export a portable TorchScript one-step predictor."""

    path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, len(FEATURE_NAMES), dtype=torch.float32)
    traced = torch.jit.trace(model.eval(), example)
    # The LibTorch Windows path layer can fail on non-ASCII directories.  A
    # Python-owned buffer keeps the export portable without renaming the repo.
    buffer = io.BytesIO()
    torch.jit.save(traced, buffer)
    path.write_bytes(buffer.getvalue())
