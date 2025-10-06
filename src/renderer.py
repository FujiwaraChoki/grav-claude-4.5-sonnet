"""
Main rendering engine that combines all components.
Handles ray tracing, shading, and image generation.
"""

import numpy as np
from numba import njit
import multiprocessing as mp
from src.ray_tracer import trace_ray_schwarzschild, camera_ray_direction
from src.accretion_disk import disk_emission
from src.starfield import StarField, ray_to_celestial_coords, add_milky_way_band

class Renderer:
    def __init__(self, width=800, height=600, M=1.0):
        """
        Initialize renderer.

        Args:
            width, height: Image resolution
            M: Black hole mass
        """
        self.width = width
        self.height = height
        self.M = M
        self.starfield = StarField(num_stars=5000)

        # Rendering options
        self.show_disk = True
        self.show_stars = True
        self.show_lensing = True
        self.render_quality = 1  # 1 = normal, 2 = high, 4 = ultra

    def render_pixel(self, x, y, camera):
        """
        Render a single pixel by tracing a ray.

        Args:
            x, y: Pixel coordinates
            camera: Camera object

        Returns:
            RGB color for pixel
        """
        # Get camera parameters
        cam_pos = camera.get_position()
        fov = camera.fov

        # Calculate ray direction from camera through pixel
        direction = camera_ray_direction(x, y, self.width, self.height, cam_pos,
                                         np.array([camera.pitch, camera.yaw]), fov)

        # Trace ray through spacetime
        hit_type, final_pos, path_length, num_orbits = trace_ray_schwarzschild(
            cam_pos[0], cam_pos[1], cam_pos[2], direction, self.M, max_steps=5000
        )

        # Determine pixel color based on what ray hit
        color = np.array([0.0, 0.0, 0.0])

        if hit_type == 1:  # Captured by black hole
            # Pure black (event horizon shadow)
            color = np.array([0.0, 0.0, 0.0])

        elif hit_type == 2 and self.show_disk:  # Hit accretion disk
            r, theta, phi = final_pos
            # Calculate disk emission with relativistic effects
            color = disk_emission(r, theta, phi, self.M, spin=0.0,
                                 observer_angle=cam_pos[1])

            # Add brightness based on number of orbits (multiple images)
            if num_orbits > 0:
                color *= (1.0 - 0.2 * num_orbits)  # Dim for higher order images

        elif hit_type == 0 and self.show_stars:  # Escaped to infinity
            r, theta, phi = final_pos
            # Get celestial coordinates
            cel_theta, cel_phi = ray_to_celestial_coords(r, theta, phi)

            # Sample star field
            color = self.starfield.sample_direction((cel_theta, cel_phi))

            # Add Milky Way band
            if self.show_lensing:
                mw_brightness = add_milky_way_band(cel_theta, cel_phi)
                color += np.array([mw_brightness, mw_brightness, mw_brightness * 1.2])

        # Clamp to valid range
        color = np.clip(color, 0.0, 1.0)

        return color

    def render_frame(self, camera, parallel=True):
        """
        Render a complete frame.

        Args:
            camera: Camera object
            parallel: Use multiprocessing for speed

        Returns:
            RGB image array [height, width, 3]
        """
        image = np.zeros((self.height, self.width, 3))

        if parallel:
            # Parallel rendering (not fully implemented - would need process pool)
            # For now, use simple loop
            pass

        # Simple sequential rendering
        total_pixels = self.width * self.height
        for y in range(self.height):
            for x in range(self.width):
                image[y, x] = self.render_pixel(x, y, camera)

            # Progress indicator
            progress = (y + 1) * self.width / total_pixels * 100
            if y % 10 == 0:
                print(f"\rRendering: {progress:.1f}%", end='', flush=True)

        print("\rRendering: 100.0%")

        return image

    def render_low_res_preview(self, camera, scale=4):
        """
        Render low resolution preview for interactive performance.

        Args:
            camera: Camera object
            scale: Downscale factor (4 = 1/4 resolution)

        Returns:
            Downscaled RGB image
        """
        preview_width = self.width // scale
        preview_height = self.height // scale
        preview = np.zeros((preview_height, preview_width, 3))

        for y in range(preview_height):
            for x in range(preview_width):
                # Sample at downscaled position
                px = x * scale
                py = y * scale
                preview[y, x] = self.render_pixel(px, py, camera)

        return preview

    def set_quality(self, quality):
        """Set render quality (1=low, 2=medium, 4=high)."""
        self.render_quality = quality

    def toggle_disk(self):
        """Toggle accretion disk rendering."""
        self.show_disk = not self.show_disk

    def toggle_stars(self):
        """Toggle star field rendering."""
        self.show_stars = not self.show_stars

    def toggle_lensing(self):
        """Toggle lensing effects."""
        self.show_lensing = not self.show_lensing
