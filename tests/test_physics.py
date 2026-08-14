import numpy as np

from frc_lab.physics import (
    cylindrical_divergence,
    generate_equilibrium,
    idw_reconstruct,
    make_grid,
    normalized_rmse,
    reconstruct_from_probes,
    reversal_radius,
    sample_probes,
)


def test_generated_configuration_has_field_reversal() -> None:
    grid = make_grid(nr=64, nz=80)
    equilibrium = generate_equilibrium(grid, reversal_strength=1.0, elongation=1.55)

    mid = int(np.argmin(np.abs(grid.z)))
    assert equilibrium.bz[mid, 0] < 0.0
    assert equilibrium.bz[mid, -1] > 0.0
    radius = reversal_radius(equilibrium.bz, grid)
    assert radius is not None
    assert 0.2 < radius < 1.0


def test_flux_reconstruction_beats_unconstrained_idw() -> None:
    grid = make_grid(nr=58, nz=72)
    equilibrium = generate_equilibrium(grid, reversal_strength=1.05, elongation=1.55)
    probes = sample_probes(equilibrium, count=30, noise_percent=0.5, seed=12)
    reconstruction = reconstruct_from_probes(equilibrium, probes, regularization=1e-4)
    idw_br, idw_bz = idw_reconstruct(probes, grid)

    physics_error = normalized_rmse(
        equilibrium.br, equilibrium.bz, reconstruction.br, reconstruction.bz
    )
    idw_error = normalized_rmse(equilibrium.br, equilibrium.bz, idw_br, idw_bz)
    assert physics_error < idw_error
    assert physics_error < 0.25


def test_flux_representation_reduces_divergence_residual() -> None:
    grid = make_grid(nr=58, nz=72)
    equilibrium = generate_equilibrium(grid)
    probes = sample_probes(equilibrium, count=24, noise_percent=1.0, seed=8)
    reconstruction = reconstruct_from_probes(equilibrium, probes)
    idw_br, idw_bz = idw_reconstruct(probes, grid)

    physics_div = np.sqrt(np.mean(cylindrical_divergence(reconstruction.br, reconstruction.bz, grid) ** 2))
    idw_div = np.sqrt(np.mean(cylindrical_divergence(idw_br, idw_bz, grid) ** 2))
    assert physics_div < idw_div


def test_fault_is_ranked_as_anomalous() -> None:
    grid = make_grid(nr=58, nz=72)
    equilibrium = generate_equilibrium(grid)
    probes = sample_probes(equilibrium, count=26, noise_percent=0.3, fault_mode="Spike", seed=21)
    reconstruction = reconstruct_from_probes(equilibrium, probes, regularization=1e-3)

    assert probes.fault_index is not None
    top_three = np.argsort(reconstruction.anomaly_score)[-3:]
    assert probes.fault_index in top_three
