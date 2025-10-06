"""
Advanced visualization features:
- Effective potential for photon orbits
- Ray path visualization
- Tidal forces (spaghettification)
- Orbital velocity calculations
- Penrose diagrams (conceptual)
"""

import numpy as np
from numba import njit
import matplotlib.pyplot as plt

@njit
def effective_potential_photon(r, M, L):
    """
    Effective potential for photon orbits in Schwarzschild spacetime.

    V_eff = (L²/r²)(1 - 2M/r)

    Args:
        r: Radial coordinate
        M: Black hole mass
        L: Angular momentum

    Returns:
        Effective potential value
    """
    if r <= 2.0 * M:
        return 0.0

    return (L * L / (r * r)) * (1.0 - 2.0 * M / r)


@njit
def find_photon_orbits(M, L_values):
    """
    Find circular photon orbits for given angular momenta.

    Circular orbit condition: dV_eff/dr = 0

    Args:
        M: Black hole mass
        L_values: Array of angular momentum values

    Returns:
        Arrays of r and V_eff for plotting
    """
    r_values = np.linspace(2.01 * M, 10.0 * M, 500)
    potentials = []

    for L in L_values:
        V_eff = np.array([effective_potential_photon(r, M, L) for r in r_values])
        potentials.append(V_eff)

    return r_values, potentials


def plot_effective_potential(M, save_path=None):
    """
    Plot effective potential for various angular momenta.

    Args:
        M: Black hole mass
        save_path: Optional path to save figure
    """
    # Critical angular momentum for photon sphere: L_crit = 3√3 M
    L_crit = 3.0 * np.sqrt(3.0) * M

    L_values = [0.8 * L_crit, L_crit, 1.2 * L_crit, 1.5 * L_crit]
    r_values, potentials = find_photon_orbits(M, L_values)

    plt.figure(figsize=(10, 6))
    for i, L in enumerate(L_values):
        label = f"L = {L/L_crit:.2f} L_crit"
        plt.plot(r_values / M, potentials[i], label=label)

    # Mark special radii
    plt.axvline(x=2.0, color='k', linestyle='--', label='Event Horizon (2M)')
    plt.axvline(x=3.0, color='r', linestyle='--', label='Photon Sphere (3M)')
    plt.axvline(x=6.0, color='g', linestyle='--', label='ISCO (6M)')

    plt.xlabel('r / M')
    plt.ylabel('Effective Potential V_eff')
    plt.title('Effective Potential for Photon Orbits')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(2, 10)

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


@njit
def tidal_force(r, M, delta_r=1.0):
    """
    Calculate tidal force (spaghettification) at distance r.

    Tidal acceleration: a_tidal ≈ 2GM Δr / r³

    Args:
        r: Distance from black hole center
        M: Black hole mass
        delta_r: Object size (for tidal stretching)

    Returns:
        Tidal acceleration (geometric units)
    """
    if r <= 2.0 * M:
        return 0.0

    # Tidal acceleration
    a_tidal = 2.0 * M * delta_r / (r * r * r)

    return a_tidal


@njit
def orbital_velocity(r, M):
    """
    Orbital velocity for circular orbit at radius r.

    v = sqrt(M/r)

    Args:
        r: Orbital radius
        M: Black hole mass

    Returns:
        Orbital velocity (in units of c)
    """
    if r <= 2.0 * M:
        return 0.0

    return np.sqrt(M / r)


@njit
def orbital_period(r, M):
    """
    Orbital period for circular orbit at radius r.

    T = 2π r^(3/2) / sqrt(M)

    Args:
        r: Orbital radius
        M: Black hole mass

    Returns:
        Orbital period (geometric units)
    """
    if r <= 2.0 * M:
        return 0.0

    return 2.0 * np.pi * r**1.5 / np.sqrt(M)


@njit
def time_dilation_factor(r, M, v=0.0):
    """
    Time dilation factor at radius r.

    Combines gravitational and kinematic time dilation:
    - Gravitational: sqrt(1 - 2M/r)
    - Kinematic: sqrt(1 - v²)

    Args:
        r: Distance from black hole
        M: Black hole mass
        v: Velocity (in units of c)

    Returns:
        Time dilation factor (proper_time / coordinate_time)
    """
    if r <= 2.0 * M:
        return 0.0

    # Gravitational time dilation
    grav_factor = np.sqrt(1.0 - 2.0 * M / r)

    # Kinematic time dilation
    kin_factor = np.sqrt(1.0 - v * v)

    return grav_factor * kin_factor


def visualize_geodesic_path(ray_history, M, save_path=None):
    """
    Visualize a photon's geodesic path in 3D.

    Args:
        ray_history: List of (r, theta, phi) positions along path
        M: Black hole mass
        save_path: Optional path to save figure
    """
    from mpl_toolkits.mplot3d import Axes3D

    # Convert to Cartesian
    positions = np.array(ray_history)
    r = positions[:, 0]
    theta = positions[:, 1]
    phi = positions[:, 2]

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    # 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot geodesic
    ax.plot(x, y, z, 'b-', linewidth=2, label='Photon Path')

    # Draw event horizon (sphere at r = 2M)
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = 2 * M * np.outer(np.cos(u), np.sin(v))
    y_sphere = 2 * M * np.outer(np.sin(u), np.sin(v))
    z_sphere = 2 * M * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='black', alpha=0.8)

    # Draw photon sphere (transparent)
    x_photon = 3 * M * np.outer(np.cos(u), np.sin(v))
    y_photon = 3 * M * np.outer(np.sin(u), np.sin(v))
    z_photon = 3 * M * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_photon, y_photon, z_photon, color='red', alpha=0.2)

    ax.set_xlabel('X / M')
    ax.set_ylabel('Y / M')
    ax.set_zlabel('Z / M')
    ax.set_title('Photon Geodesic in Schwarzschild Spacetime')
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


class RayPathRecorder:
    """Records ray paths for visualization."""

    def __init__(self):
        self.paths = []
        self.current_path = []

    def add_point(self, r, theta, phi):
        """Add a point to current path."""
        self.current_path.append((r, theta, phi))

    def save_path(self):
        """Save current path and start new one."""
        if len(self.current_path) > 0:
            self.paths.append(self.current_path.copy())
            self.current_path = []

    def clear(self):
        """Clear all paths."""
        self.paths = []
        self.current_path = []

    def visualize_all(self, M):
        """Visualize all recorded paths."""
        if len(self.paths) == 0:
            print("No paths recorded")
            return

        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot all paths
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.paths)))
        for i, path in enumerate(self.paths):
            positions = np.array(path)
            r = positions[:, 0]
            theta = positions[:, 1]
            phi = positions[:, 2]

            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            ax.plot(x, y, z, color=colors[i], linewidth=1, alpha=0.7)

        # Draw event horizon
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x_sphere = 2 * M * np.outer(np.cos(u), np.sin(v))
        y_sphere = 2 * M * np.outer(np.sin(u), np.sin(v))
        z_sphere = 2 * M * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_sphere, y_sphere, z_sphere, color='black', alpha=0.9)

        ax.set_xlabel('X / M')
        ax.set_ylabel('Y / M')
        ax.set_zlabel('Z / M')
        ax.set_title(f'Multiple Photon Geodesics ({len(self.paths)} rays)')

        plt.show()
