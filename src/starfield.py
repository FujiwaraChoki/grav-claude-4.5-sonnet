"""
Background star field for demonstrating gravitational lensing.
Creates realistic star distribution and handles ray intersections.
"""

import numpy as np
from numba import njit

class StarField:
    def __init__(self, num_stars=5000, seed=42):
        """
        Initialize background star field.

        Args:
            num_stars: Number of background stars
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.num_stars = num_stars

        # Generate stars on celestial sphere (θ, φ)
        # Use uniform distribution on sphere
        self.star_theta = np.arccos(2.0 * np.random.random(num_stars) - 1.0)
        self.star_phi = 2.0 * np.pi * np.random.random(num_stars)

        # Star brightness (magnitude)
        self.star_brightness = np.random.random(num_stars)**2

        # Star colors (temperature variation)
        self.star_temp = 3000 + np.random.random(num_stars) * 20000  # 3000-23000 K

    def get_star_color(self, temperature):
        """
        Get RGB color based on stellar temperature.

        Args:
            temperature: Temperature in Kelvin

        Returns:
            RGB color array
        """
        # Simplified stellar color based on temperature
        if temperature > 10000:  # Blue stars
            return np.array([0.6, 0.7, 1.0])
        elif temperature > 7500:  # White stars
            return np.array([0.9, 0.95, 1.0])
        elif temperature > 6000:  # Yellow-white
            return np.array([1.0, 1.0, 0.9])
        elif temperature > 5000:  # Yellow
            return np.array([1.0, 0.9, 0.7])
        elif temperature > 3500:  # Orange
            return np.array([1.0, 0.7, 0.5])
        else:  # Red stars
            return np.array([1.0, 0.5, 0.3])

    def sample_direction(self, direction):
        """
        Get star color for a given viewing direction.
        Finds nearest star to the direction vector.

        Args:
            direction: Direction vector (θ, φ) on celestial sphere

        Returns:
            RGB color of nearest star, or black if none nearby
        """
        theta, phi = direction

        # Find nearest star using angular distance
        delta_theta = self.star_theta - theta
        delta_phi = np.abs(self.star_phi - phi)
        # Handle phi wrapping
        delta_phi = np.minimum(delta_phi, 2*np.pi - delta_phi)

        # Angular distance on sphere
        angular_dist = np.sqrt(delta_theta**2 + (np.sin(theta) * delta_phi)**2)

        # Find closest star
        min_idx = np.argmin(angular_dist)
        min_dist = angular_dist[min_idx]

        # If close enough, return star color
        threshold = 0.01  # About 0.5 degrees
        if min_dist < threshold:
            temp = self.star_temp[min_idx]
            brightness = self.star_brightness[min_idx]
            color = self.get_star_color(temp)
            return color * brightness
        else:
            # Empty space - dark
            return np.array([0.0, 0.0, 0.0])


@njit
def ray_to_celestial_coords(r, theta, phi):
    """
    Convert ray final position to celestial sphere coordinates.
    Assumes ray escaped to infinity.

    Args:
        r, theta, phi: Final ray position (spherical)

    Returns:
        (theta, phi): Direction on celestial sphere
    """
    # At large r, (theta, phi) approximate celestial coordinates
    return theta, phi


@njit
def add_milky_way_band(theta, phi):
    """
    Add Milky Way galactic plane brightness.

    Args:
        theta, phi: Celestial coordinates

    Returns:
        Brightness boost [0-1]
    """
    # Milky Way is roughly in equatorial plane
    # Add brightness band with sinusoidal variation
    galactic_latitude = abs(theta - np.pi/2)

    if galactic_latitude < 0.3:  # Near galactic plane
        brightness = (0.3 - galactic_latitude) / 0.3
        # Add variation along galactic plane
        variation = 0.5 + 0.5 * np.cos(phi * 3.0)
        return brightness * variation * 0.3
    else:
        return 0.0
