"""
Camera system for black hole visualization.
Supports orbital camera with interactive controls.
"""

import numpy as np

class Camera:
    def __init__(self, distance=30.0, theta=np.pi/3, phi=0.0, M=1.0):
        """
        Initialize camera in spherical coordinates.

        Args:
            distance: Distance from black hole (in units of M)
            theta: Polar angle (0 = north pole, π/2 = equator, π = south pole)
            phi: Azimuthal angle
            M: Black hole mass (for reference)
        """
        self.distance = distance
        self.theta = theta
        self.phi = phi
        self.M = M

        # Camera orientation
        self.pitch = 0.0  # Rotation around local x-axis
        self.yaw = 0.0    # Rotation around local y-axis

        # Field of view
        self.fov = 60.0

        # Movement parameters
        self.move_speed = 1.0
        self.rotate_speed = 0.02

    def get_position(self):
        """Get camera position in spherical coordinates."""
        return np.array([self.distance, self.theta, self.phi])

    def get_cartesian_position(self):
        """Get camera position in Cartesian coordinates."""
        x = self.distance * np.sin(self.theta) * np.cos(self.phi)
        y = self.distance * np.sin(self.theta) * np.sin(self.phi)
        z = self.distance * np.cos(self.theta)
        return np.array([x, y, z])

    def get_look_direction(self):
        """
        Get camera look direction (toward black hole center).

        Returns:
            Direction vector in spherical coordinates [v_r, v_θ, v_φ]
        """
        # Base direction: looking toward origin (inward radial)
        v_r = -1.0
        v_theta = 0.0
        v_phi = 0.0

        # Apply pitch and yaw rotations
        # Simplified rotation model
        v_theta += self.pitch
        v_phi += self.yaw

        return np.array([v_r, v_theta, v_phi])

    def orbit_horizontal(self, delta_phi):
        """Orbit around black hole horizontally (change azimuthal angle)."""
        self.phi += delta_phi * self.rotate_speed
        # Keep phi in [0, 2π]
        self.phi = self.phi % (2 * np.pi)

    def orbit_vertical(self, delta_theta):
        """Orbit around black hole vertically (change polar angle)."""
        self.theta += delta_theta * self.rotate_speed
        # Keep theta in [0.1, π - 0.1] to avoid poles
        self.theta = np.clip(self.theta, 0.1, np.pi - 0.1)

    def zoom(self, delta_distance):
        """Move camera closer or farther from black hole."""
        self.distance += delta_distance * self.move_speed
        # Don't get too close or too far
        rs = 2.0 * self.M
        self.distance = np.clip(self.distance, rs * 5, 200.0 * self.M)

    def rotate_view(self, delta_pitch, delta_yaw):
        """Rotate camera view (look around)."""
        self.pitch += delta_pitch * self.rotate_speed
        self.yaw += delta_yaw * self.rotate_speed
        # Limit pitch to avoid gimbal lock
        self.pitch = np.clip(self.pitch, -np.pi/3, np.pi/3)

    def reset(self):
        """Reset camera to default position."""
        self.distance = 30.0
        self.theta = np.pi / 3
        self.phi = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

    def set_fov(self, fov):
        """Set field of view in degrees."""
        self.fov = np.clip(fov, 30.0, 120.0)

    def get_info(self):
        """Get camera information for display."""
        cart = self.get_cartesian_position()
        rs = 2.0 * self.M
        info = {
            'position_sph': (self.distance, np.degrees(self.theta), np.degrees(self.phi)),
            'position_cart': tuple(cart),
            'distance_in_rs': self.distance / rs,
            'fov': self.fov,
            'pitch': np.degrees(self.pitch),
            'yaw': np.degrees(self.yaw)
        }
        return info
