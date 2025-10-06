"""
GRAV-Bench Test Suite: Schwarzschild Metric
Tests for correct implementation of the Schwarzschild metric.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from src.metric import (
    schwarzschild_metric,
    schwarzschild_radius,
    photon_sphere_radius,
    isco_radius
)


def test_metric_components():
    """Test metric components at various radii."""
    M = 1.0
    test_cases = [
        # (r/M, expected_g_tt, expected_g_rr)
        (3.0, -1/3, 3.0),       # r = 3M
        (5.0, -3/5, 5/3),       # r = 5M
        (10.0, -4/5, 5/4),      # r = 10M
    ]

    print("Testing metric components...")
    for r_over_M, expected_gtt, expected_grr in test_cases:
        r = r_over_M * M
        g_tt, g_rr, g_thth, g_phph = schwarzschild_metric(r, M)

        # Check g_tt
        error_gtt = abs(g_tt - expected_gtt)
        assert error_gtt < 1e-10, f"g_tt error at r={r_over_M}M: {error_gtt}"

        # Check g_rr
        error_grr = abs(g_rr - expected_grr)
        assert error_grr < 1e-10, f"g_rr error at r={r_over_M}M: {error_grr}"

        # Check g_theta_theta = r²
        expected_gthth = r * r
        error_gthth = abs(g_thth - expected_gthth)
        assert error_gthth < 1e-10, f"g_θθ error at r={r_over_M}M: {error_gthth}"

        print(f"  ✓ r = {r_over_M}M: g_tt = {g_tt:.6f}, g_rr = {g_rr:.6f}")

    return True


def test_critical_radii():
    """Test that critical radii are correctly defined."""
    M = 1.0

    print("\nTesting critical radii...")

    # Event horizon at r = 2M
    rs = schwarzschild_radius(M)
    assert abs(rs - 2.0 * M) < 1e-10, f"Schwarzschild radius error: {rs} != {2*M}"
    print(f"  ✓ Event horizon at r_s = {rs/M}M")

    # Photon sphere at r = 3M
    r_photon = photon_sphere_radius(M)
    assert abs(r_photon - 3.0 * M) < 1e-10, f"Photon sphere error: {r_photon} != {3*M}"
    print(f"  ✓ Photon sphere at r_ph = {r_photon/M}M")

    # ISCO at r = 6M
    r_isco = isco_radius(M)
    assert abs(r_isco - 6.0 * M) < 1e-10, f"ISCO error: {r_isco} != {6*M}"
    print(f"  ✓ ISCO at r_isco = {r_isco/M}M")

    return True


def test_asymptotic_flatness():
    """Test that metric becomes flat as r → ∞."""
    M = 1.0

    print("\nTesting asymptotic flatness...")

    for r_over_M in [100, 1000, 10000]:
        r = r_over_M * M
        g_tt, g_rr, g_thth, g_phph = schwarzschild_metric(r, M)

        # At large r, g_tt → -1, g_rr → 1
        error_gtt = abs(g_tt - (-1.0))
        error_grr = abs(g_rr - 1.0)

        assert error_gtt < 0.01, f"g_tt not flat at r={r_over_M}M: error={error_gtt}"
        assert error_grr < 0.01, f"g_rr not flat at r={r_over_M}M: error={error_grr}"

        print(f"  ✓ r = {r_over_M}M: g_tt ≈ -1 (error: {error_gtt:.2e}), "
              f"g_rr ≈ 1 (error: {error_grr:.2e})")

    return True


def test_horizon_behavior():
    """Test metric behavior near event horizon."""
    M = 1.0
    rs = schwarzschild_radius(M)

    print("\nTesting behavior near event horizon...")

    # Test points approaching horizon from outside
    for delta in [1.0, 0.1, 0.01]:
        r = rs + delta
        g_tt, g_rr, g_thth, g_phph = schwarzschild_metric(r, M)

        # g_tt should be negative and small
        assert g_tt < 0, f"g_tt should be negative at r={r/M}M"
        assert abs(g_tt) < delta / M, f"g_tt too large at r={r/M}M"

        # g_rr should be large and positive
        assert g_rr > 0, f"g_rr should be positive at r={r/M}M"
        assert g_rr > M / delta, f"g_rr too small at r={r/M}M"

        print(f"  ✓ r = {r/M:.3f}M: metric well-behaved near horizon")

    return True


def run_all_metric_tests():
    """Run all metric tests."""
    print("=" * 60)
    print("METRIC TESTS (Tier 1: Schwarzschild Metric Implementation)")
    print("=" * 60)

    tests = [
        ("Metric Components", test_metric_components),
        ("Critical Radii", test_critical_radii),
        ("Asymptotic Flatness", test_asymptotic_flatness),
        ("Horizon Behavior", test_horizon_behavior),
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
    print(f"Metric Tests: {passed}/{len(tests)} passed")
    print("=" * 60)

    return passed == len(tests)


if __name__ == '__main__':
    success = run_all_metric_tests()
    sys.exit(0 if success else 1)
