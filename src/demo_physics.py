#!/usr/bin/env python3
"""
Physics demonstration and testing script.
Shows calculations and creates visualizations of black hole physics.
"""

import numpy as np
import matplotlib.pyplot as plt
from schwarzschild import (
    schwarzschild_radius,
    photon_sphere_radius,
    isco_radius
)
from kerr import (
    kerr_event_horizon,
    kerr_ergosphere,
    kerr_isco
)
from advanced_features import (
    orbital_velocity,
    orbital_period,
    time_dilation_factor,
    tidal_force,
    plot_effective_potential
)
from accretion_disk import (
    disk_temperature,
    disk_velocity,
    doppler_shift,
    gravitational_redshift,
    relativistic_beaming
)

def print_schwarzschild_radii():
    """Print characteristic radii for Schwarzschild black holes."""
    print("=" * 60)
    print("SCHWARZSCHILD BLACK HOLE - CHARACTERISTIC RADII")
    print("=" * 60)

    for M in [1.0, 2.0, 5.0, 10.0]:
        print(f"\nMass M = {M}")
        print(f"  Event Horizon (R_s):  {schwarzschild_radius(M):.2f} M")
        print(f"  Photon Sphere:        {photon_sphere_radius(M):.2f} M")
        print(f"  ISCO:                 {isco_radius(M):.2f} M")

        # Real units for solar mass black hole
        if M == 1.0:
            print(f"\n  For a solar mass BH (M☉ = 2×10³⁰ kg):")
            print(f"    Event Horizon:  ~3 km")
            print(f"    Photon Sphere:  ~4.5 km")
            print(f"    ISCO:           ~9 km")


def print_kerr_radii():
    """Print characteristic radii for Kerr black holes."""
    print("\n" + "=" * 60)
    print("KERR (ROTATING) BLACK HOLE - CHARACTERISTIC RADII")
    print("=" * 60)

    M = 1.0
    spins = [0.0, 0.3, 0.6, 0.9, 0.99]

    print(f"\nMass M = {M}, varying spin parameter a/M:")
    print(f"{'a/M':<8} {'R+ (horizon)':<15} {'ISCO':<15} {'Ergosphere (eq)':<20}")
    print("-" * 60)

    for spin in spins:
        a = spin * M
        r_horizon = kerr_event_horizon(M, a)
        r_isco = kerr_isco(M, a)
        r_ergo = kerr_ergosphere(M, a, np.pi/2)  # At equator

        print(f"{spin:<8.2f} {r_horizon:<15.3f} {r_isco:<15.3f} {r_ergo:<20.3f}")

    print(f"\nNote: For extremal Kerr (a=M), ISCO reaches the horizon!")


def demonstrate_orbital_mechanics():
    """Demonstrate orbital mechanics near black hole."""
    print("\n" + "=" * 60)
    print("ORBITAL MECHANICS")
    print("=" * 60)

    M = 1.0
    radii = [6.0, 10.0, 20.0, 50.0, 100.0]  # In units of M

    print(f"\nOrbital properties at various radii (M = {M}):")
    print(f"{'r/M':<8} {'v/c':<10} {'Period (M)':<15} {'Time Dilation':<15}")
    print("-" * 60)

    for r in radii:
        v = orbital_velocity(r, M)
        T = orbital_period(r, M)
        dilation = time_dilation_factor(r, M, v)

        print(f"{r:<8.1f} {v:<10.4f} {T:<15.2f} {dilation:<15.4f}")

    print(f"\nInterpretation:")
    print(f"  - Closer orbits → higher velocity (approaching c)")
    print(f"  - Stronger time dilation near horizon")
    print(f"  - At ISCO (6M): v = {orbital_velocity(6.0, M):.4f}c")


def demonstrate_tidal_forces():
    """Demonstrate tidal forces (spaghettification)."""
    print("\n" + "=" * 60)
    print("TIDAL FORCES (SPAGHETTIFICATION)")
    print("=" * 60)

    M = 1.0
    object_size = 1.0  # 1 meter in geometric units
    radii = [100.0, 50.0, 20.0, 10.0, 6.0, 4.0, 2.5]

    print(f"\nTidal acceleration for {object_size}M object (M = {M}):")
    print(f"{'r/M':<8} {'a_tidal (M⁻¹)':<20} {'Survivable?':<15}")
    print("-" * 60)

    for r in radii:
        a_tidal = tidal_force(r, M, object_size)
        survivable = "Yes" if a_tidal < 10.0 else "DEADLY" if a_tidal < 100 else "INSTANT DEATH"

        print(f"{r:<8.1f} {a_tidal:<20.6f} {survivable:<15}")

    print(f"\nNote: Tidal forces ∝ 1/r³ - grow rapidly near horizon!")


def demonstrate_disk_physics():
    """Demonstrate accretion disk physics."""
    print("\n" + "=" * 60)
    print("ACCRETION DISK PHYSICS")
    print("=" * 60)

    M = 1.0
    radii = [6.0, 8.0, 10.0, 15.0, 20.0]

    print(f"\nDisk properties at various radii (M = {M}):")
    print(f"{'r/M':<8} {'T (10⁶ K)':<12} {'v/c':<10} {'Grav. Redshift':<15}")
    print("-" * 60)

    for r in radii:
        T = disk_temperature(r, M) / 1e6  # In millions of Kelvin
        v = disk_velocity(r, M)
        z = gravitational_redshift(r, M)

        print(f"{r:<8.1f} {T:<12.2f} {v:<10.4f} {z:<15.4f}")

    print(f"\nDoppler shift example (disk at 10M, v={disk_velocity(10.0, M):.3f}c):")
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    v_disk = disk_velocity(10.0, M)

    for angle in angles:
        cos_angle = np.cos(angle)
        doppler = doppler_shift(v_disk, cos_angle)
        beaming = relativistic_beaming(v_disk, cos_angle)

        print(f"  Angle {np.degrees(angle):>5.0f}°: Doppler={doppler:.3f}, Beaming={beaming:.3f}")


def plot_orbits():
    """Plot orbital velocity and period vs radius."""
    M = 1.0
    r_values = np.linspace(6.0, 50.0, 200)

    v_values = [orbital_velocity(r, M) for r in r_values]
    T_values = [orbital_period(r, M) for r in r_values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Velocity plot
    ax1.plot(r_values, v_values, 'b-', linewidth=2)
    ax1.axvline(x=6.0, color='r', linestyle='--', label='ISCO (6M)')
    ax1.axvline(x=3.0, color='orange', linestyle='--', label='Photon Sphere (3M)')
    ax1.set_xlabel('r / M', fontsize=12)
    ax1.set_ylabel('v / c', fontsize=12)
    ax1.set_title('Orbital Velocity vs Radius', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Period plot
    ax2.plot(r_values, T_values, 'g-', linewidth=2)
    ax2.axvline(x=6.0, color='r', linestyle='--', label='ISCO (6M)')
    ax2.set_xlabel('r / M', fontsize=12)
    ax2.set_ylabel('Period (M)', fontsize=12)
    ax2.set_title('Orbital Period vs Radius', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('orbital_properties.png', dpi=150)
    print(f"\nOrbital properties plot saved to 'orbital_properties.png'")


def plot_time_dilation():
    """Plot time dilation vs radius."""
    M = 1.0
    r_values = np.linspace(2.01, 50.0, 300)

    # Time dilation for stationary observer
    dilation_static = [time_dilation_factor(r, M, v=0.0) for r in r_values]

    # Time dilation for orbiting observer
    dilation_orbit = [time_dilation_factor(r, M, v=orbital_velocity(r, M))
                     for r in r_values]

    plt.figure(figsize=(10, 6))
    plt.plot(r_values, dilation_static, 'b-', linewidth=2, label='Stationary observer')
    plt.plot(r_values, dilation_orbit, 'r-', linewidth=2, label='Orbiting observer')

    plt.axvline(x=2.0, color='k', linestyle='--', alpha=0.5, label='Event Horizon (2M)')
    plt.axvline(x=6.0, color='orange', linestyle='--', alpha=0.5, label='ISCO (6M)')

    plt.xlabel('r / M', fontsize=12)
    plt.ylabel('Time Dilation Factor (τ/t)', fontsize=12)
    plt.title('Time Dilation Near Black Hole', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(2, 50)
    plt.ylim(0, 1.1)

    plt.savefig('time_dilation.png', dpi=150)
    print(f"Time dilation plot saved to 'time_dilation.png'")


def main():
    """Run all physics demonstrations."""
    print("\n" + "=" * 60)
    print(" BLACK HOLE PHYSICS DEMONSTRATION")
    print("=" * 60)

    # Text demonstrations
    print_schwarzschild_radii()
    print_kerr_radii()
    demonstrate_orbital_mechanics()
    demonstrate_tidal_forces()
    demonstrate_disk_physics()

    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS...")
    print("=" * 60)

    plot_orbits()
    plot_time_dilation()
    plot_effective_potential(M=1.0, save_path='effective_potential.png')

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nPlots saved:")
    print("  - orbital_properties.png")
    print("  - time_dilation.png")
    print("  - effective_potential.png")
    print("\nRun 'python blackhole.py' to start interactive visualization")


if __name__ == "__main__":
    main()
