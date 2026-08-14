"""Physics-inspired synthetic diagnostics for an FRC-like configuration.

This module is intentionally educational. It demonstrates how an axisymmetric
poloidal-flux representation can constrain sparse magnetic reconstruction. It
is not a Grad-Shafranov solver and is not validated against a fusion device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Grid:
    r: np.ndarray
    z: np.ndarray
    R: np.ndarray
    Z: np.ndarray


@dataclass(frozen=True)
class Equilibrium:
    grid: Grid
    psi: np.ndarray
    br: np.ndarray
    bz: np.ndarray
    coefficients: np.ndarray
    basis_flux: tuple[np.ndarray, ...]
    basis_fields: tuple[tuple[np.ndarray, np.ndarray], ...]


@dataclass(frozen=True)
class ProbeSet:
    r: np.ndarray
    z: np.ndarray
    br: np.ndarray
    bz: np.ndarray
    br_clean: np.ndarray
    bz_clean: np.ndarray
    fault_index: int | None
    fault_mode: str


@dataclass(frozen=True)
class Reconstruction:
    psi: np.ndarray
    br: np.ndarray
    bz: np.ndarray
    coefficients: np.ndarray
    predicted_br: np.ndarray
    predicted_bz: np.ndarray
    residual: np.ndarray
    anomaly_score: np.ndarray


def make_grid(
    nr: int = 76,
    nz: int = 112,
    r_max: float = 1.15,
    z_max: float = 1.50,
) -> Grid:
    """Create a regular half-plane grid for an axisymmetric configuration."""
    if nr < 8 or nz < 8:
        raise ValueError("nr and nz must both be at least 8")
    r = np.linspace(0.0, r_max, nr)
    z = np.linspace(-z_max, z_max, nz)
    R, Z = np.meshgrid(r, z)
    return Grid(r=r, z=z, R=R, Z=Z)


def _flux_basis(grid: Grid, elongation: float) -> tuple[np.ndarray, ...]:
    """Return smooth, axis-regular flux basis functions."""
    a = 0.70
    ell = a * elongation
    q = (grid.R / a) ** 2 + (grid.Z / ell) ** 2
    return (
        0.5 * grid.R**2,
        grid.R**2 * np.exp(-q),
        grid.R**2 * np.exp(-2.0 * q),
        grid.R**4 * np.exp(-q),
        grid.R**2 * (grid.Z / ell) ** 2 * np.exp(-1.35 * q),
    )


def field_from_flux(psi: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Derive Br and Bz from poloidal flux on an axisymmetric cylindrical grid."""
    dpsi_dz, dpsi_dr = np.gradient(psi, grid.z, grid.r, edge_order=2)
    br = np.zeros_like(psi)
    bz = np.zeros_like(psi)
    np.divide(-dpsi_dz, grid.R, out=br, where=grid.R > 1e-10)
    np.divide(dpsi_dr, grid.R, out=bz, where=grid.R > 1e-10)

    # At r=0, psi is quadratic in r. The limiting axial field is 2*a(z),
    # estimated from the first non-axis radial point.
    br[:, 0] = 0.0
    bz[:, 0] = 2.0 * psi[:, 1] / (grid.r[1] ** 2)
    return br, bz


def generate_equilibrium(
    grid: Grid,
    reversal_strength: float = 1.0,
    elongation: float = 1.55,
    external_field: float = 0.80,
) -> Equilibrium:
    """Generate a synthetic FRC-like reversed axial-field equilibrium."""
    basis = _flux_basis(grid, elongation)
    basis_fields = tuple(field_from_flux(phi, grid) for phi in basis)

    # The first term is the external axial field. Localized negative terms
    # reverse Bz near the axis; the final two terms add unmodeled shaping.
    coefficients = np.array(
        [
            external_field,
            -0.92 * external_field * reversal_strength,
            -0.10 * external_field * reversal_strength,
            0.16 * external_field,
            -0.08 * external_field * reversal_strength,
        ]
    )
    psi = sum(c * phi for c, phi in zip(coefficients, basis, strict=True))
    br, bz = field_from_flux(psi, grid)
    return Equilibrium(
        grid=grid,
        psi=psi,
        br=br,
        bz=bz,
        coefficients=coefficients,
        basis_flux=basis,
        basis_fields=basis_fields,
    )


def _bilinear_sample(field: np.ndarray, grid: Grid, r: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Sample a regular grid at arbitrary probe positions."""
    ir = np.clip(np.searchsorted(grid.r, r) - 1, 0, len(grid.r) - 2)
    iz = np.clip(np.searchsorted(grid.z, z) - 1, 0, len(grid.z) - 2)
    r0, r1 = grid.r[ir], grid.r[ir + 1]
    z0, z1 = grid.z[iz], grid.z[iz + 1]
    wr = (r - r0) / np.maximum(r1 - r0, 1e-12)
    wz = (z - z0) / np.maximum(z1 - z0, 1e-12)
    return (
        field[iz, ir] * (1 - wr) * (1 - wz)
        + field[iz, ir + 1] * wr * (1 - wz)
        + field[iz + 1, ir] * (1 - wr) * wz
        + field[iz + 1, ir + 1] * wr * wz
    )


def _probe_locations(count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Place probes on two diagnostic chords plus sparse interior points."""
    if count < 8:
        raise ValueError("At least 8 probes are required")
    wall_count = max(6, int(count * 0.65))
    theta = np.linspace(-0.92 * np.pi / 2, 0.92 * np.pi / 2, wall_count)
    wall_r = 0.90 + 0.08 * np.cos(theta) ** 2
    wall_z = 1.28 * np.sin(theta)

    interior_count = count - wall_count
    interior_r = rng.uniform(0.12, 0.78, interior_count)
    interior_z = rng.uniform(-1.05, 1.05, interior_count)
    r = np.concatenate([wall_r, interior_r])
    z = np.concatenate([wall_z, interior_z])
    order = rng.permutation(count)
    return r[order], z[order]


def sample_probes(
    equilibrium: Equilibrium,
    count: int = 24,
    noise_percent: float = 1.0,
    fault_mode: Literal["None", "Drift", "Spike", "Saturation"] = "None",
    seed: int = 7,
) -> ProbeSet:
    """Sample synthetic magnetic probes and optionally inject one sensor fault."""
    rng = np.random.default_rng(seed)
    r, z = _probe_locations(count, rng)
    br_clean = _bilinear_sample(equilibrium.br, equilibrium.grid, r, z)
    bz_clean = _bilinear_sample(equilibrium.bz, equilibrium.grid, r, z)
    field_scale = float(np.sqrt(np.mean(br_clean**2 + bz_clean**2)))
    sigma = max(field_scale * noise_percent / 100.0, 1e-8)
    br = br_clean + rng.normal(0.0, sigma, count)
    bz = bz_clean + rng.normal(0.0, sigma, count)

    fault_index: int | None = None
    if fault_mode != "None":
        fault_index = int(rng.integers(0, count))
        if fault_mode == "Drift":
            bz[fault_index] += 0.30 * field_scale
        elif fault_mode == "Spike":
            br[fault_index] -= 0.65 * field_scale
            bz[fault_index] += 0.50 * field_scale
        elif fault_mode == "Saturation":
            limit = 0.42 * np.max(np.abs(np.concatenate([br, bz])))
            br[fault_index] = np.sign(br[fault_index] or 1.0) * limit
            bz[fault_index] = np.sign(bz[fault_index] or 1.0) * limit
        else:
            raise ValueError(f"Unsupported fault mode: {fault_mode}")

    return ProbeSet(
        r=r,
        z=z,
        br=br,
        bz=bz,
        br_clean=br_clean,
        bz_clean=bz_clean,
        fault_index=fault_index,
        fault_mode=fault_mode,
    )


def reconstruct_from_probes(
    grid: Grid,
    probes: ProbeSet,
    elongation: float = 1.55,
    regularization: float = 1e-3,
    basis_count: int = 3,
) -> Reconstruction:
    """Reconstruct a flux-consistent field from sparse magnetic probes."""
    basis_flux = _flux_basis(grid, elongation)
    basis_fields = tuple(field_from_flux(phi, grid) for phi in basis_flux)
    if not 2 <= basis_count <= len(basis_fields):
        raise ValueError("basis_count is outside the available basis")

    y = np.concatenate([probes.br, probes.bz])
    columns: list[np.ndarray] = []
    for br_basis, bz_basis in basis_fields[:basis_count]:
        columns.append(
            np.concatenate(
                [
                    _bilinear_sample(br_basis, grid, probes.r, probes.z),
                    _bilinear_sample(bz_basis, grid, probes.r, probes.z),
                ]
            )
        )
    design = np.column_stack(columns)
    scale = np.linalg.norm(design, axis=0)
    scaled = design / np.maximum(scale, 1e-12)
    ridge = regularization * np.eye(basis_count)
    scaled_coefficients = np.linalg.solve(scaled.T @ scaled + ridge, scaled.T @ y)
    coefficients = scaled_coefficients / np.maximum(scale, 1e-12)

    br = sum(c * pair[0] for c, pair in zip(coefficients, basis_fields[:basis_count]))
    bz = sum(c * pair[1] for c, pair in zip(coefficients, basis_fields[:basis_count]))

    psi = sum(c * phi for c, phi in zip(coefficients, basis_flux[:basis_count]))

    predicted_br = _bilinear_sample(br, grid, probes.r, probes.z)
    predicted_bz = _bilinear_sample(bz, grid, probes.r, probes.z)
    residual = np.sqrt((predicted_br - probes.br) ** 2 + (predicted_bz - probes.bz) ** 2)
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    anomaly_score = 0.6745 * np.abs(residual - median) / max(mad, 1e-10)
    return Reconstruction(
        psi=psi,
        br=br,
        bz=bz,
        coefficients=coefficients,
        predicted_br=predicted_br,
        predicted_bz=predicted_bz,
        residual=residual,
        anomaly_score=anomaly_score,
    )


def idw_reconstruct(
    probes: ProbeSet,
    grid: Grid,
    power: float = 2.0,
    smoothing: float = 0.035,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent IDW interpolation baseline for Br and Bz."""
    dr = grid.R[..., None] - probes.r
    dz = grid.Z[..., None] - probes.z
    distance = np.sqrt(dr**2 + dz**2 + smoothing**2)
    weights = 1.0 / distance**power
    weights /= weights.sum(axis=-1, keepdims=True)
    br = np.sum(weights * probes.br, axis=-1)
    bz = np.sum(weights * probes.bz, axis=-1)
    br[:, 0] = 0.0
    return br, bz


def normalized_rmse(
    truth_br: np.ndarray,
    truth_bz: np.ndarray,
    predicted_br: np.ndarray,
    predicted_bz: np.ndarray,
) -> float:
    error = np.sqrt(np.mean((truth_br - predicted_br) ** 2 + (truth_bz - predicted_bz) ** 2))
    scale = np.sqrt(np.mean(truth_br**2 + truth_bz**2))
    return float(error / max(scale, 1e-12))


def cylindrical_divergence(br: np.ndarray, bz: np.ndarray, grid: Grid) -> np.ndarray:
    """Evaluate div(B) = (1/r)d(rBr)/dr + dBz/dz."""
    radial_flux = grid.R * br
    d_rbr_dr = np.gradient(radial_flux, grid.r, axis=1, edge_order=2)
    dbz_dz = np.gradient(bz, grid.z, axis=0, edge_order=2)
    divergence = np.zeros_like(br)
    np.divide(d_rbr_dr, grid.R, out=divergence, where=grid.R > 1e-8)
    divergence += dbz_dz
    divergence[:, 0] = divergence[:, 1]
    return divergence


def reversal_radius(bz: np.ndarray, grid: Grid) -> float | None:
    """Return the first midplane Bz=0 crossing away from the axis."""
    mid = int(np.argmin(np.abs(grid.z)))
    profile = bz[mid]
    crossings = np.where(np.signbit(profile[:-1]) != np.signbit(profile[1:]))[0]
    if not len(crossings):
        return None
    idx = int(crossings[0])
    x0, x1 = grid.r[idx], grid.r[idx + 1]
    y0, y1 = profile[idx], profile[idx + 1]
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def generate_shot(
    duration_ms: float = 3.2,
    points: int = 240,
    reversal_strength: float = 1.0,
    seed: int = 17,
) -> pd.DataFrame:
    """Generate a reproducible synthetic pulse for diagnostic review."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, duration_ms, points)
    formation = 1.0 / (1.0 + np.exp(-(t - 0.48) / 0.075))
    decay = np.exp(-np.clip(t - 2.18, 0.0, None) / 0.72)
    plasma = formation * decay
    compression = 1.0 + 0.17 * np.exp(-((t - 1.35) / 0.25) ** 2)
    coil_current = 42.0 * np.exp(-((t - 0.42) / 0.22) ** 2) + 13.0 * plasma
    center_bz = 0.78 - (1.72 * reversal_strength * plasma * compression)
    edge_bz = 0.78 + 0.12 * plasma
    density = 0.10 + 1.65 * plasma * compression
    energy = 0.04 + 5.2 * plasma**1.7 * compression
    center_bz += rng.normal(0.0, 0.012, points)
    density += rng.normal(0.0, 0.018, points)
    return pd.DataFrame(
        {
            "time_ms": t,
            "coil_current_kA": coil_current,
            "center_bz_T": center_bz,
            "edge_bz_T": edge_bz,
            "density_1e19_m3": density,
            "energy_kJ": energy,
        }
    )
