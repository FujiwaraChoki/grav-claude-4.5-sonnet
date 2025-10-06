"""
Performance optimization utilities.
Includes parallel processing, caching, and GPU acceleration hints.
"""

import numpy as np
from numba import njit, prange
import multiprocessing as mp
from functools import lru_cache

@njit(parallel=True)
def parallel_ray_trace(width, height, cam_r, cam_theta, cam_phi, fov, M):
    """
    Parallel ray tracing using Numba's prange.

    Args:
        width, height: Image dimensions
        cam_r, cam_theta, cam_phi: Camera position
        fov: Field of view
        M: Black hole mass

    Returns:
        Hit types array [height, width]
    """
    hit_types = np.zeros((height, width), dtype=np.int32)

    # Parallel loop over pixels
    for y in prange(height):
        for x in range(width):
            # Ray tracing logic here (simplified)
            # In practice, call trace_ray_schwarzschild
            hit_types[y, x] = 0  # Placeholder

    return hit_types


class RenderCache:
    """
    Cache for rendered frames to avoid recomputation.
    Uses LRU (Least Recently Used) eviction.
    """

    def __init__(self, max_size=10):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []

    def _make_key(self, camera_state, render_options):
        """Create hashable key from camera and render state."""
        # Round values for cache hits with small movements
        r = round(camera_state[0], 2)
        theta = round(camera_state[1], 3)
        phi = round(camera_state[2], 3)

        options = tuple(sorted(render_options.items()))
        return (r, theta, phi, options)

    def get(self, camera_state, render_options):
        """Get cached frame if available."""
        key = self._make_key(camera_state, render_options)
        if key in self.cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, camera_state, render_options, frame):
        """Store frame in cache."""
        key = self._make_key(camera_state, render_options)

        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]

        self.cache[key] = frame.copy()
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

    def clear(self):
        """Clear all cached frames."""
        self.cache.clear()
        self.access_order.clear()


@njit
def adaptive_sampling(x, y, width, height, quality=1):
    """
    Adaptive sampling: more rays near interesting features.

    Args:
        x, y: Pixel coordinates
        width, height: Image dimensions
        quality: Quality level (1=normal, 2=high, 4=ultra)

    Returns:
        Number of rays to shoot for this pixel
    """
    # Center of image (where black hole likely is) gets more samples
    center_x = width / 2
    center_y = height / 2
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)

    # More samples near center
    if dist < max_dist / 3:
        return quality * 2
    else:
        return quality


@njit
def bilinear_interpolate(image, x, y):
    """
    Bilinear interpolation for smooth upscaling.

    Args:
        image: Input image array
        x, y: Continuous coordinates

    Returns:
        Interpolated color value
    """
    height, width = image.shape[:2]

    x1 = int(np.floor(x))
    x2 = min(x1 + 1, width - 1)
    y1 = int(np.floor(y))
    y2 = min(y1 + 1, height - 1)

    # Interpolation weights
    wx = x - x1
    wy = y - y1

    # Bilinear interpolation
    result = (1 - wx) * (1 - wy) * image[y1, x1] + \
             wx * (1 - wy) * image[y1, x2] + \
             (1 - wx) * wy * image[y2, x1] + \
             wx * wy * image[y2, x2]

    return result


def multiprocess_render(renderer, camera, num_processes=None):
    """
    Render using multiple processes for parallelization.

    Args:
        renderer: Renderer object
        camera: Camera object
        num_processes: Number of processes (defaults to CPU count)

    Returns:
        Rendered image
    """
    if num_processes is None:
        num_processes = mp.cpu_count()

    # Split image into tiles
    height, width = renderer.height, renderer.width
    tile_height = height // num_processes

    def render_tile(args):
        tile_idx, renderer, camera = args
        y_start = tile_idx * tile_height
        y_end = min(y_start + tile_height, height)

        tile = np.zeros((y_end - y_start, width, 3))
        for y in range(y_start, y_end):
            for x in range(width):
                tile[y - y_start, x] = renderer.render_pixel(x, y, camera)

        return tile_idx, tile

    # Create process pool and render tiles
    with mp.Pool(num_processes) as pool:
        args = [(i, renderer, camera) for i in range(num_processes)]
        results = pool.map(render_tile, args)

    # Combine tiles
    image = np.zeros((height, width, 3))
    for tile_idx, tile in sorted(results):
        y_start = tile_idx * tile_height
        y_end = min(y_start + tile_height, height)
        image[y_start:y_end] = tile

    return image


# GPU acceleration hints (requires CuPy or similar)
def gpu_available():
    """Check if GPU acceleration is available."""
    try:
        import cupy
        return True
    except ImportError:
        return False


def gpu_ray_trace(image_shape, camera_params, M):
    """
    GPU-accelerated ray tracing (requires CuPy).

    Args:
        image_shape: (height, width)
        camera_params: Camera parameters
        M: Black hole mass

    Returns:
        Rendered image on GPU
    """
    if not gpu_available():
        raise RuntimeError("GPU not available. Install CuPy for GPU acceleration.")

    import cupy as cp

    # Convert to GPU arrays
    # Implement GPU kernels here
    # This is a placeholder - full implementation would require
    # translating the ray tracing code to CUDA kernels

    print("GPU ray tracing not fully implemented. Use CPU fallback.")
    return None
