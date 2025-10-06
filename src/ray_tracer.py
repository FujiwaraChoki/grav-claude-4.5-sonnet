"""
Ray tracer for photon geodesics in curved spacetime.

Implements backward ray tracing:
1. Shoot rays from camera through each pixel
2. Integrate geodesic equations backward in time
3. Determine what each ray hits (accretion disk, stars, or escapes)
4. Calculate color with relativistic effects
"""

import numpy as np
from numba import njit
from src.metric import photon_geodesic_equations, schwarzschild_radius

@njit
def rk4_step(state, M, b, q, dt):
    """
    4th order Runge-Kutta integration step.

    Args:
        state: Current state vector [r, θ, φ, p_r, p_θ]
        M: Black hole mass
        b, q: Impact parameters
        dt: Step size (affine parameter)

    Returns:
        New state after one RK4 step
    """
    k1 = photon_geodesic_equations(state, M, b, q)
    k2 = photon_geodesic_equations(state + 0.5*dt*k1, M, b, q)
    k3 = photon_geodesic_equations(state + 0.5*dt*k2, M, b, q)
    k4 = photon_geodesic_equations(state + dt*k3, M, b, q)

    return state + (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


@njit
def adaptive_step_size(r, M, base_step=0.1):
    """
    Adaptive step size: smaller near event horizon.

    Args:
        r: Current radial position
        M: Black hole mass
        base_step: Base step size for integration

    Returns:
        Appropriate step size
    """
    rs = schwarzschild_radius(M)

    # Reduce step size near horizon
    if r < 3.0 * M:
        factor = (r - rs) / M
        if factor < 0.1:
            factor = 0.1
        return base_step * factor

    return base_step


@njit
def trace_ray_schwarzschild(r0, theta0, phi0, direction, M, max_steps=5000):
    """
    Trace a single photon ray through Schwarzschild spacetime.

    Args:
        r0, theta0, phi0: Initial camera position (spherical coords)
        direction: Initial direction vector [v_r, v_theta, v_phi]
        M: Black hole mass
        max_steps: Maximum integration steps

    Returns:
        hit_type: 0=escaped, 1=captured, 2=hit_disk
        final_position: [r, θ, φ]
        path_length: affine parameter traveled
        num_orbits: number of times ray orbited black hole
    """
    # Initial conditions
    v_r, v_theta, v_phi = direction

    # Calculate impact parameters (conserved quantities)
    rs = schwarzschild_radius(M)
    f = 1.0 - rs / r0

    # Normalize to null condition
    sin_theta = np.sin(theta0)
    norm = np.sqrt(v_r*v_r/(f*f) + v_theta*v_theta/(r0*r0) + v_phi*v_phi/(r0*r0*sin_theta*sin_theta))
    if norm < 1e-10:
        norm = 1e-10
    v_r /= norm
    v_theta /= norm
    v_phi /= norm

    # Impact parameters
    b = r0 * r0 * sin_theta * sin_theta * v_phi
    q = r0**4 * v_theta * v_theta

    # Initialize state
    p_r = v_r / f
    p_theta = v_theta / (r0 * r0)
    state = np.array([r0, theta0, phi0, p_r, p_theta])

    # Integration
    lambda_param = 0.0
    phi_initial = phi0
    num_orbits = 0
    prev_phi = phi0

    for step in range(max_steps):
        r, theta, phi = state[0], state[1], state[2]

        # Check termination conditions
        if r < rs * 1.01:  # Captured by black hole
            return 1, state[:3], lambda_param, num_orbits

        if r > 1000.0 * M:  # Escaped to infinity
            return 0, state[:3], lambda_param, num_orbits

        # Check for accretion disk intersection (equatorial plane, certain radius range)
        if abs(theta - np.pi/2) < 0.05:  # Near equatorial plane
            if rs * 3.0 < r < 20.0 * M:  # Disk extends from ISCO to ~20M
                return 2, state[:3], lambda_param, num_orbits

        # Count orbits (phi wrapping)
        if phi - prev_phi > np.pi:
            num_orbits += 1
        elif phi - prev_phi < -np.pi:
            num_orbits -= 1
        prev_phi = phi

        # Adaptive step size
        dt = adaptive_step_size(r, M, base_step=0.5)

        # RK4 integration step
        state = rk4_step(state, M, b, q, dt)
        lambda_param += dt

        # Keep theta in [0, π]
        if state[1] < 0:
            state[1] = -state[1]
            state[2] += np.pi
        if state[1] > np.pi:
            state[1] = 2*np.pi - state[1]
            state[2] += np.pi

    # Max steps reached - assume escaped
    return 0, state[:3], lambda_param, num_orbits


@njit
def camera_ray_direction(pixel_x, pixel_y, width, height, camera_pos, camera_dir, fov=60.0):
    """
    Calculate initial ray direction for a pixel.

    Args:
        pixel_x, pixel_y: Pixel coordinates
        width, height: Image dimensions
        camera_pos: Camera position [r, θ, φ]
        camera_dir: Camera orientation [theta_rot, phi_rot]
        fov: Field of view in degrees

    Returns:
        Initial direction vector in spherical coordinates [v_r, v_θ, v_φ]
    """
    # Normalized device coordinates [-1, 1]
    ndc_x = (2.0 * pixel_x / width - 1.0)
    ndc_y = (1.0 - 2.0 * pixel_y / height)

    # Apply FOV
    aspect = width / height
    fov_rad = fov * np.pi / 180.0
    tan_fov = np.tan(fov_rad / 2.0)

    x = ndc_x * tan_fov * aspect
    y = ndc_y * tan_fov
    z = -1.0  # Looking along -z

    # Normalize
    norm = np.sqrt(x*x + y*y + z*z)
    x /= norm
    y /= norm
    z /= norm

    # Convert to spherical coordinate velocities
    # This is simplified - proper transformation would use rotation matrices
    r, theta, phi = camera_pos

    # Local frame: tangent vectors in spherical coords
    # v_r points radially outward
    # v_theta points in increasing theta
    # v_phi points in increasing phi

    v_r = -z  # Camera looks inward (-r direction)
    v_theta = y / r
    v_phi = x / (r * np.sin(theta))

    return np.array([v_r, v_theta, v_phi])
