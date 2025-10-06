"""
Kerr metric for rotating black holes.

The Kerr metric in Boyer-Lindquist coordinates includes:
- Spin parameter: a = J/M where J is angular momentum
- Frame dragging effects
- Ergosphere outside event horizon

Key features:
- Event horizon: r+ = M + sqrt(M² - a²)
- Ergosphere: r_ergo = M + sqrt(M² - a²cos²θ)
- ISCO depends on spin (can be as close as M for a=M)
"""

import numpy as np
from numba import njit

@njit
def kerr_metric(r, theta, M, a):
    """
    Kerr metric components in Boyer-Lindquist coordinates.

    Args:
        r: Radial coordinate
        theta: Polar angle
        M: Black hole mass
        a: Spin parameter (J/M), range [0, M]

    Returns:
        Metric components needed for geodesic integration
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Kerr metric functions
    Sigma = r*r + a*a * cos_theta*cos_theta
    Delta = r*r - 2.0*M*r + a*a
    A = (r*r + a*a)**2 - a*a * Delta * sin_theta*sin_theta

    # Metric components (simplified for photon geodesics)
    g_tt = -(1.0 - 2.0*M*r/Sigma)
    g_tphi = -2.0*M*r*a*sin_theta*sin_theta/Sigma
    g_rr = Sigma/Delta
    g_thth = Sigma
    g_phph = A * sin_theta*sin_theta / Sigma

    return g_tt, g_tphi, g_rr, g_thth, g_phph, Sigma, Delta


@njit
def kerr_photon_geodesic(state, M, a, E, L, Q):
    """
    Geodesic equations for photons in Kerr spacetime.

    Uses conserved quantities:
    - E: Energy
    - L: Angular momentum (z-component)
    - Q: Carter constant

    State vector: [r, θ, φ, p_r, p_θ]

    Args:
        state: Current position and momentum
        M: Black hole mass
        a: Spin parameter
        E, L, Q: Conserved quantities

    Returns:
        Derivatives for integration
    """
    r, theta, phi, p_r, p_theta = state

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    if abs(sin_theta) < 1e-10:
        sin_theta = 1e-10

    # Kerr metric functions
    Sigma = r*r + a*a * cos_theta*cos_theta
    Delta = r*r - 2.0*M*r + a*a

    if Delta < 0.01 or Sigma < 0.01:
        return np.zeros(5)

    # Radial potential
    R = (r*r + a*a)*E - a*L
    R = R*R - Delta*(Q + (L - a*E)**2)

    # Theta potential
    Theta = Q - cos_theta*cos_theta * (a*a*E*E - L*L/(sin_theta*sin_theta + 1e-10))

    # Derivatives
    dr_dlambda = p_r
    dtheta_dlambda = p_theta

    # Frame dragging in phi
    dphi_dlambda = (a*((r*r + a*a)*E - a*L) - L/(sin_theta*sin_theta + 1e-10)) / Delta

    # Momentum derivatives (Hamiltonian equations)
    dSigma_dr = 2.0*r
    dSigma_dtheta = -2.0*a*a*cos_theta*sin_theta
    dDelta_dr = 2.0*r - 2.0*M

    dp_r_dlambda = -0.5 * (dDelta_dr*(Q + (L - a*E)**2)/Delta**2 - dSigma_dr*R/(Sigma**2))
    dp_theta_dlambda = 0.5 * dSigma_dtheta * Theta / (Sigma**2)

    return np.array([dr_dlambda, dtheta_dlambda, dphi_dlambda, dp_r_dlambda, dp_theta_dlambda])


@njit
def kerr_event_horizon(M, a):
    """Event horizon radius for Kerr black hole."""
    return M + np.sqrt(M*M - a*a)


@njit
def kerr_ergosphere(M, a, theta):
    """Ergosphere radius (depends on theta)."""
    return M + np.sqrt(M*M - a*a*np.cos(theta)**2)


@njit
def kerr_isco(M, a):
    """
    ISCO radius for Kerr black hole (prograde orbit).
    For a = M (extremal), ISCO = M.
    For a = 0 (Schwarzschild), ISCO = 6M.
    """
    z1 = 1.0 + (1.0 - a*a/(M*M))**(1./3.) * ((1.0 + a/M)**(1./3.) + (1.0 - a/M)**(1./3.))
    z2 = np.sqrt(3.0*a*a/(M*M) + z1*z1)
    r_isco = M * (3.0 + z2 - np.sqrt((3.0 - z1)*(3.0 + z1 + 2.0*z2)))
    return r_isco
