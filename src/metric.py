"""
Schwarzschild metric and geodesic equations for null paths (photons).

The Schwarzschild metric in spherical coordinates (t, r, θ, φ):
ds² = -(1 - 2M/r)dt² + (1 - 2M/r)⁻¹dr² + r²dθ² + r²sin²(θ)dφ²

For photon geodesics, we use the Hamiltonian formulation with conserved quantities:
- Energy: E (related to time-like Killing vector)
- Angular momentum: L (related to azimuthal Killing vector)
"""

import numpy as np
from numba import njit

@njit
def schwarzschild_metric(r, M):
    """
    Schwarzschild metric components.

    Args:
        r: Radial coordinate (in units of M)
        M: Black hole mass (geometric units, G=c=1)

    Returns:
        g_tt, g_rr, g_thth, g_phph: Metric components
    """
    rs = 2.0 * M  # Schwarzschild radius

    if r <= rs:
        # Inside event horizon - metric signature changes
        # We handle this carefully in integration
        return 0.0, 0.0, 0.0, 0.0

    f = 1.0 - rs / r
    g_tt = -f
    g_rr = 1.0 / f
    g_thth = r * r
    g_phph = r * r  # Will multiply by sin²θ when needed

    return g_tt, g_rr, g_thth, g_phph


@njit
def photon_geodesic_equations(state, M, b, q):
    """
    Geodesic equations for photon in Schwarzschild spacetime.

    We use impact parameters instead of 4-momentum:
    - b: impact parameter (related to angular momentum L)
    - q: Carter constant (for θ motion)

    State vector: [r, θ, φ, p_r, p_θ]
    where p_r = dr/dλ, p_θ = dθ/dλ (λ is affine parameter)

    Args:
        state: [r, θ, φ, p_r, p_θ]
        M: Black hole mass
        b: impact parameter (perpendicular distance if no deflection)
        q: Carter constant (θ motion parameter)

    Returns:
        derivatives: [dr/dλ, dθ/dλ, dφ/dλ, dp_r/dλ, dp_θ/dλ]
    """
    r, theta, phi, p_r, p_theta = state

    rs = 2.0 * M

    # Avoid singularities
    if r < rs * 1.001:  # Very close to horizon
        return np.zeros(5)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Avoid theta singularities
    if abs(sin_theta) < 1e-10:
        sin_theta = 1e-10

    # Effective potential terms
    f = 1.0 - rs / r
    r2 = r * r
    r3 = r2 * r

    # Radial equation: dr/dλ = p_r (by definition)
    dr_dlambda = p_r

    # Theta equation: dθ/dλ = p_θ (by definition)
    dtheta_dlambda = p_theta

    # Phi equation from conserved angular momentum
    # dφ/dλ = L / (r² sin²θ) where L is related to impact parameter b
    dphi_dlambda = b / (r2 * sin_theta * sin_theta + 1e-10)

    # Radial momentum equation
    # dp_r/dλ = -∂H/∂r where H is Hamiltonian
    # For photons: H = (1/2)g^μν p_μ p_ν = 0
    # This gives: (1-2M/r)⁻¹ p_r² + (1/r²)p_θ² + (b²/r²sin²θ) = 1

    # Effective radial potential
    V_eff = (1.0 / r2) * (q + b * b / (sin_theta * sin_theta + 1e-10))
    dV_eff_dr = -(2.0 / r3) * (q + b * b / (sin_theta * sin_theta + 1e-10))

    # Contribution from metric
    df_dr = rs / r2

    dp_r_dlambda = -0.5 * df_dr * p_r * p_r / (f * f + 1e-10) + 0.5 * dV_eff_dr / (f + 1e-10)

    # Theta momentum equation
    # dp_θ/dλ = -∂H/∂θ
    dp_theta_dlambda = (b * b * cos_theta) / (r2 * sin_theta**3 + 1e-10)

    return np.array([dr_dlambda, dtheta_dlambda, dphi_dlambda, dp_r_dlambda, dp_theta_dlambda])


@njit
def calculate_impact_parameters(r0, theta0, v_r, v_theta, v_phi, M):
    """
    Calculate conserved impact parameters from initial conditions.

    Args:
        r0, theta0: Initial position
        v_r, v_theta, v_phi: Initial velocity direction (will be normalized)
        M: Black hole mass

    Returns:
        b, q: Impact parameters
    """
    rs = 2.0 * M
    f = 1.0 - rs / r0

    # Normalize velocity (null condition)
    norm = np.sqrt(v_r*v_r/(f*f) + v_theta*v_theta/r0**2 + v_phi*v_phi/(r0**2 * np.sin(theta0)**2))
    v_r /= norm
    v_theta /= norm
    v_phi /= norm

    # Angular momentum (azimuthal)
    L = r0 * r0 * np.sin(theta0)**2 * v_phi

    # Impact parameter
    b = L

    # Carter constant for θ motion
    q = r0**4 * v_theta**2

    return b, q


@njit
def photon_sphere_radius(M):
    """Photon sphere at r = 3M (1.5 × Schwarzschild radius)."""
    return 3.0 * M


@njit
def schwarzschild_radius(M):
    """Event horizon at r = 2M."""
    return 2.0 * M


@njit
def isco_radius(M):
    """Innermost stable circular orbit at r = 6M (for non-spinning BH)."""
    return 6.0 * M
