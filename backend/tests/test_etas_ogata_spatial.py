"""Tests for ETAS isotropic power-law spatial kernel."""
from __future__ import annotations

import numpy as np


def test_spatial_kernel_integrates_to_unity_on_dense_grid() -> None:
    """Magnitude-scaled isotropic power-law kernel must integrate to ~1.0
    over R^2 by construction."""
    from backend.app.ml.etas_ogata import spatial_kernel_powerlaw

    xs = np.linspace(-300, 300, 601)
    ys = np.linspace(-300, 300, 601)
    XX, YY = np.meshgrid(xs, ys)
    R = np.sqrt(XX**2 + YY**2)
    vals = spatial_kernel_powerlaw(R, mag=5.5, mc=4.5, d0=2.0, gamma=0.5, q=1.5)
    dx = xs[1] - xs[0]
    integral = float(vals.sum() * dx * dx)
    assert 0.85 < integral < 1.05, f"kernel integral {integral} not unit-ish"


def test_spatial_kernel_decays_with_distance() -> None:
    from backend.app.ml.etas_ogata import spatial_kernel_powerlaw

    near = spatial_kernel_powerlaw(
        np.array([1.0]), mag=5.0, mc=4.5, d0=2.0, gamma=0.5, q=1.5
    )
    far = spatial_kernel_powerlaw(
        np.array([100.0]), mag=5.0, mc=4.5, d0=2.0, gamma=0.5, q=1.5
    )
    assert near[0] > far[0]


def test_spatial_kernel_larger_event_has_wider_footprint() -> None:
    """Bigger magnitude -> larger characteristic radius d -> at fixed r=50 km,
    a M=6 event should give a higher density than a M=4.5 event because the
    radius scales as exp(gamma * (M - Mc))."""
    from backend.app.ml.etas_ogata import spatial_kernel_powerlaw

    r = np.array([50.0])
    big = spatial_kernel_powerlaw(r, mag=6.0, mc=4.5, d0=2.0, gamma=0.5, q=1.5)
    small = spatial_kernel_powerlaw(r, mag=4.5, mc=4.5, d0=2.0, gamma=0.5, q=1.5)
    assert big[0] > small[0]


def test_spatial_kernel_zero_distance_is_finite() -> None:
    from backend.app.ml.etas_ogata import spatial_kernel_powerlaw

    val = spatial_kernel_powerlaw(
        np.array([0.0]), mag=5.0, mc=4.5, d0=2.0, gamma=0.5, q=1.5
    )
    assert np.isfinite(val[0]) and val[0] > 0
