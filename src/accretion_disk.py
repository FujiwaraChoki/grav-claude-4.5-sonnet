"""
Accretion disk model with realistic physics:
- Temperature profile (hotter closer to black hole)
- Doppler shifting from rotation
- Relativistic beaming
- Gravitational redshift
"""

import numpy as np
from numba import njit

@njit
def disk_temperature(r, M, T_inner=1e7):
    """
    Temperature profile of accretion disk.
    Simplified Shakura-Sunyaev model: T ∝ r^(-3/4)

    Args:
        r: Radial distance
        M: Black hole mass
        T_inner: Temperature at inner edge (ISCO)

    Returns:
        Temperature in Kelvin
    """
    r_isco = 6.0 * M
    if r < r_isco:
        return 0.0

    return T_inner * (r_isco / r)**0.75


@njit
def blackbody_color(T):
    """
    Convert temperature to RGB color (Wien approximation).

    Args:
        T: Temperature in Kelvin

    Returns:
        RGB color [0-1, 0-1, 0-1]
    """
    # Simplified blackbody color
    # Blue stars: T > 10,000 K
    # White: T ~ 6000-10,000 K
    # Orange/Red: T < 6000 K

    if T > 1e8:  # Ultra hot - X-ray region (show as bright blue-white)
        return np.array([0.8, 0.9, 1.0])
    elif T > 1e7:  # Very hot - UV (blue-white)
        return np.array([0.6, 0.7, 1.0])
    elif T > 1e6:  # Hot (blue)
        return np.array([0.3, 0.5, 1.0])
    elif T > 5e5:  # Warm (white)
        return np.array([0.9, 0.9, 0.9])
    elif T > 1e5:  # Cool (orange)
        return np.array([1.0, 0.6, 0.3])
    else:  # Cold (red)
        return np.array([1.0, 0.3, 0.1])


@njit
def disk_velocity(r, M, spin=0.0):
    """
    Keplerian velocity of disk at radius r.
    For Schwarzschild: v = sqrt(M/r)
    For Kerr: modified by frame dragging

    Args:
        r: Radial distance
        M: Black hole mass
        spin: Spin parameter (0 to 1)

    Returns:
        Azimuthal velocity (geometric units)
    """
    # Keplerian velocity (geometric units where c=1)
    v_kepler = np.sqrt(M / r)

    # Frame dragging correction for Kerr
    a = spin * M
    frame_drag = a / (r**1.5 + a)

    return v_kepler + frame_drag


@njit
def doppler_shift(velocity, cos_angle):
    """
    Relativistic Doppler shift factor.

    Args:
        velocity: Velocity of emitting material (in units of c)
        cos_angle: Cosine of angle between velocity and line of sight

    Returns:
        Doppler factor: observed_freq / emitted_freq
    """
    beta = velocity  # Already in units of c
    gamma = 1.0 / np.sqrt(1.0 - beta*beta + 1e-10)

    # Relativistic Doppler formula
    doppler_factor = 1.0 / (gamma * (1.0 - beta * cos_angle))

    return doppler_factor


@njit
def gravitational_redshift(r, M):
    """
    Gravitational redshift factor (1 + z).

    Args:
        r: Radial distance from black hole
        M: Black hole mass

    Returns:
        Redshift factor: observed_freq / emitted_freq
    """
    rs = 2.0 * M
    if r <= rs:
        return 0.0

    # For photon emitted at r and observed at infinity
    redshift_factor = 1.0 / np.sqrt(1.0 - rs / r)

    return redshift_factor


@njit
def relativistic_beaming(velocity, cos_angle):
    """
    Relativistic beaming intensity boost.
    Moving source appears brighter when moving toward observer.

    Args:
        velocity: Velocity of source
        cos_angle: Cosine of angle to line of sight

    Returns:
        Intensity multiplier
    """
    doppler = doppler_shift(velocity, cos_angle)

    # Beaming formula: I ∝ δ³ for continuum emission
    # where δ is Doppler factor
    return doppler**3


@njit
def disk_emission(r, theta, phi, M, spin=0.0, observer_angle=np.pi/4):
    """
    Calculate emission from accretion disk at given position.

    Args:
        r, theta, phi: Position in disk
        M: Black hole mass
        spin: Black hole spin parameter
        observer_angle: Inclination angle of observer

    Returns:
        RGB color including all relativistic effects
    """
    # Disk only exists in equatorial plane
    if abs(theta - np.pi/2) > 0.1:
        return np.array([0.0, 0.0, 0.0])

    r_isco = 6.0 * M if spin < 0.1 else 6.0 * M * (1.0 - 0.54*spin)  # Approximate
    r_outer = 20.0 * M

    if r < r_isco or r > r_outer:
        return np.array([0.0, 0.0, 0.0])

    # Base temperature and color
    T = disk_temperature(r, M)
    color = blackbody_color(T)

    # Velocity at this radius
    v = disk_velocity(r, M, spin)

    # Angle between velocity and line of sight
    # Simplified: depends on azimuthal position and observer inclination
    cos_angle = np.sin(observer_angle) * np.cos(phi)

    # Doppler shift
    doppler = doppler_shift(v, cos_angle)

    # Gravitational redshift
    grav_redshift = gravitational_redshift(r, M)

    # Combined frequency shift
    total_shift = doppler * grav_redshift

    # Shift color (blue shift increases, red shift decreases)
    # Approximate by scaling intensity in blue/red channels
    if total_shift > 1.0:  # Blue shift
        color[2] *= total_shift**0.5  # Increase blue
        color[0] *= 1.0 / total_shift**0.3  # Decrease red
    else:  # Red shift
        color[0] *= (1.0/total_shift)**0.5  # Increase red
        color[2] *= total_shift**0.3  # Decrease blue

    # Relativistic beaming
    beaming = relativistic_beaming(v, cos_angle)
    color *= beaming**0.3  # Scale intensity (reduced exponent for visibility)

    # Normalize to [0, 1]
    color = np.clip(color, 0.0, 1.0)

    return color


@njit
def disk_opacity(r, M):
    """
    Disk opacity/thickness (for multiple scattering effects).

    Args:
        r: Radial distance
        M: Black hole mass

    Returns:
        Opacity factor [0-1]
    """
    r_isco = 6.0 * M
    r_peak = 10.0 * M

    if r < r_isco:
        return 0.0

    # Disk is most opaque near peak temperature region
    if r < r_peak:
        return 0.8 * (r - r_isco) / (r_peak - r_isco)
    else:
        return 0.8 * np.exp(-(r - r_peak) / (5.0 * M))
