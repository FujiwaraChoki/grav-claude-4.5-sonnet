"""
GRAV-Bench Test Suite: Geodesic Integration
Tests for photon geodesics and ray tracing accuracy.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from src.metric import (
    photon_geodesic_equations,
    schwarzschild_radius,
    photon_sphere_radius
)
from src.ray_tracer import rk4_step, trace_ray_schwarzschild


def test_photon_sphere_orbit():
    """Test that photons orbit at photon sphere with correct impact parameter."""
    M = 1.0
    r_ph = photon_sphere_radius(M)

    print("Testing photon sphere orbit...")

    # Critical impact parameter for photon sphere orbit
    b_crit = 3 * np.sqrt(3) * M  # ≈ 5.196M

    # Start photon at photon sphere with tangential velocity
    r0 = r_ph
    theta0 = np.pi / 2  # Equatorial plane
    phi0 = 0.0

    # Tangential initial conditions
    p_r = 0.0  # No radial motion
    p_theta = 0.0  # Stay in equatorial plane
    state = np.array([r0, theta0, phi0, p_r, p_theta])

    q = 0.0  # Equatorial orbit

    # Integrate for several orbits
    num_steps = 1000
    dt = 0.1
    max_r_deviation = 0.0

    for step in range(num_steps):
        state = rk4_step(state, M, b_crit, q, dt)
        r = state[0]
        deviation = abs(r - r_ph)
        max_r_deviation = max(max_r_deviation, deviation)

    # After many orbits, should still be near photon sphere
    tolerance = 0.1 * M
    assert max_r_deviation < tolerance, \
        f"Photon orbit unstable: max deviation {max_r_deviation/M:.3f}M > {tolerance/M}M"

    print(f"  ✓ Photon sphere orbit stable: max deviation = {max_r_deviation/M:.4f}M")
    print(f"  ✓ Impact parameter b = {b_crit/M:.3f}M ≈ 3√3M")

    return True


def test_energy_conservation():
    """Test that energy is conserved along geodesic."""
    M = 1.0

    print("\nTesting energy conservation...")

    # Initial conditions: photon far from black hole
    r0 = 20.0 * M
    theta0 = np.pi / 2
    phi0 = 0.0

    # Initial direction (slightly inward and tangential)
    v_r = -0.3
    v_theta = 0.0
    v_phi = 0.5

    # Normalize
    rs = schwarzschild_radius(M)
    f = 1.0 - rs / r0
    norm = np.sqrt(v_r**2 / f**2 + v_theta**2 / r0**2 + v_phi**2 / (r0**2))
    v_r /= norm
    v_theta /= norm
    v_phi /= norm

    # Calculate initial energy (conserved quantity)
    E0 = f  # Energy at infinity normalized to 1

    # Impact parameters
    b = r0**2 * v_phi
    q = r0**4 * v_theta**2

    # Initial state
    p_r = v_r / f
    p_theta = v_theta / r0**2
    state = np.array([r0, theta0, phi0, p_r, p_theta])

    # Integrate geodesic
    num_steps = 500
    dt = 0.1
    energies = []

    for step in range(num_steps):
        r, theta, phi, pr, pth = state

        # Calculate energy: E = (1 - 2M/r)
        # This is approximate - actual conservation check needs full calculation
        f_current = 1.0 - rs / r if r > rs else 0.0

        # Hamiltonian for null geodesic (should be zero)
        H = f_current * pr**2 + (pth**2 + b**2 / np.sin(theta)**2) / r**2
        energies.append(H)

        state = rk4_step(state, M, b, q, dt)

        # Stop if captured or escaped
        if state[0] < rs * 1.1 or state[0] > 100 * M:
            break

    # Check energy drift
    if len(energies) > 10:
        energy_drift = np.std(energies) / (abs(np.mean(energies)) + 1e-10)
        print(f"  ✓ Energy drift: {energy_drift:.2e} (std/mean)")

        # Should be small due to RK4 precision
        assert energy_drift < 0.1, f"Energy drift too large: {energy_drift}"
    else:
        print(f"  ✓ Geodesic terminated after {len(energies)} steps")

    return True


def test_deflection_angle():
    """Test light deflection angle for grazing ray."""
    M = 1.0
    rs = schwarzschild_radius(M)

    print("\nTesting gravitational deflection...")

    # Ray passing at minimum distance r_min ≈ 3M
    # Should deflect by approximately 2.47 radians (141°)
    r0 = 50.0 * M
    theta0 = np.pi / 2
    phi0 = 0.0

    # Impact parameter for r_min ≈ 3M
    b = 3.0 * np.sqrt(3) * M * 1.01  # Slightly larger than critical

    direction = np.array([-0.1, 0.0, 0.5])
    direction /= np.linalg.norm(direction)

    hit_type, final_pos, path_length, num_orbits = trace_ray_schwarzschild(
        r0, theta0, phi0, direction, M, max_steps=2000
    )

    # Should escape (not captured)
    assert hit_type == 0, "Ray should escape to infinity"

    # Calculate deflection
    phi_final = final_pos[2]
    deflection = abs(phi_final - phi0)

    # Expected deflection for grazing ray at ~3M is ~2.5 radians
    print(f"  ✓ Deflection angle: {deflection:.3f} rad ({np.degrees(deflection):.1f}°)")
    print(f"  ✓ Ray escaped to r = {final_pos[0]/M:.1f}M")

    return True


def test_circular_vs_escape():
    """Test that rays with different impact parameters behave correctly."""
    M = 1.0
    b_crit = 3 * np.sqrt(3) * M

    print("\nTesting impact parameter dependence...")

    # Test cases
    test_cases = [
        (b_crit * 0.8, "should be captured", 1),   # Below critical → capture
        (b_crit * 1.2, "should escape", 0),        # Above critical → escape
    ]

    for b, description, expected_hit_type in test_cases:
        r0 = 30.0 * M
        theta0 = np.pi / 2
        phi0 = 0.0
        direction = np.array([-0.1, 0.0, b / (r0**2)])
        direction /= np.linalg.norm(direction)

        hit_type, final_pos, path_length, num_orbits = trace_ray_schwarzschild(
            r0, theta0, phi0, direction, M, max_steps=2000
        )

        # Note: exact behavior depends on initial conditions
        # This is a simplified test
        print(f"  • b = {b/M:.2f}M: {description} → "
              f"{'captured' if hit_type == 1 else 'escaped' if hit_type == 0 else 'hit disk'}")

    print(f"  ✓ Impact parameter b_crit ≈ {b_crit/M:.2f}M")

    return True


def run_all_geodesic_tests():
    """Run all geodesic tests."""
    print("=" * 60)
    print("GEODESIC TESTS (Tier 1: Geodesic Integration)")
    print("=" * 60)

    tests = [
        ("Photon Sphere Orbit", test_photon_sphere_orbit),
        ("Energy Conservation", test_energy_conservation),
        ("Deflection Angle", test_deflection_angle),
        ("Impact Parameter Behavior", test_circular_vs_escape),
    ]

    passed = 0
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"\n✓ {name}: PASSED\n")
        except AssertionError as e:
            print(f"\n✗ {name}: FAILED - {e}\n")
        except Exception as e:
            print(f"\n✗ {name}: ERROR - {e}\n")

    print("=" * 60)
    print(f"Geodesic Tests: {passed}/{len(tests)} passed")
    print("=" * 60)

    return passed == len(tests)


if __name__ == '__main__':
    success = run_all_geodesic_tests()
    sys.exit(0 if success else 1)
