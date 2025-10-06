#!/usr/bin/env python3
"""
Batch rendering script for creating animations or multi-view renders.
Useful for creating movies or comparison images.
"""

import numpy as np
import argparse
from PIL import Image
from camera import Camera
from renderer import Renderer
import os

def render_orbit_animation(output_dir, num_frames=360, M=1.0, distance=30.0):
    """
    Render a complete orbital animation around the black hole.

    Args:
        output_dir: Directory to save frames
        num_frames: Number of frames (360 = 1 degree per frame)
        M: Black hole mass
        distance: Camera distance
    """
    os.makedirs(output_dir, exist_ok=True)

    camera = Camera(distance=distance, theta=np.pi/3, phi=0.0, M=M)
    renderer = Renderer(width=1920, height=1080, M=M)

    print(f"Rendering {num_frames} frames for orbit animation...")

    for frame in range(num_frames):
        # Update camera position
        phi = (frame / num_frames) * 2 * np.pi
        camera.phi = phi

        # Render frame
        print(f"Frame {frame+1}/{num_frames}")
        image = renderer.render_frame(camera, parallel=False)

        # Save frame
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(image_uint8)
        img.save(os.path.join(output_dir, f"frame_{frame:04d}.png"))

    print(f"Animation frames saved to {output_dir}")
    print(f"To create video: ffmpeg -i {output_dir}/frame_%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p output.mp4")


def render_zoom_sequence(output_dir, num_frames=100, M=1.0, start_distance=100.0, end_distance=10.0):
    """
    Render a zoom sequence approaching the black hole.

    Args:
        output_dir: Directory to save frames
        num_frames: Number of frames
        M: Black hole mass
        start_distance: Starting distance
        end_distance: Ending distance
    """
    os.makedirs(output_dir, exist_ok=True)

    camera = Camera(distance=start_distance, theta=np.pi/3, phi=0.0, M=M)
    renderer = Renderer(width=1920, height=1080, M=M)

    print(f"Rendering zoom sequence from {start_distance}M to {end_distance}M...")

    for frame in range(num_frames):
        # Exponential zoom (feels more natural)
        t = frame / (num_frames - 1)
        distance = start_distance * (end_distance / start_distance) ** t
        camera.distance = distance

        # Render frame
        print(f"Frame {frame+1}/{num_frames} (distance: {distance:.1f}M)")
        image = renderer.render_frame(camera, parallel=False)

        # Save frame
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(image_uint8)
        img.save(os.path.join(output_dir, f"zoom_{frame:04d}.png"))

    print(f"Zoom sequence saved to {output_dir}")


def render_mass_comparison(output_dir, masses=[1.0, 2.0, 4.0, 8.0]):
    """
    Render comparison images for different black hole masses.

    Args:
        output_dir: Directory to save images
        masses: List of black hole masses to compare
    """
    os.makedirs(output_dir, exist_ok=True)

    camera = Camera(distance=30.0, theta=np.pi/3, phi=0.0, M=1.0)

    print(f"Rendering mass comparison for masses: {masses}")

    for M in masses:
        camera.M = M
        # Adjust distance to maintain similar view
        camera.distance = 30.0 * M

        renderer = Renderer(width=1920, height=1080, M=M)

        print(f"Rendering M = {M}...")
        image = renderer.render_frame(camera, parallel=False)

        # Save image
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(image_uint8)
        img.save(os.path.join(output_dir, f"mass_{M:.1f}M.png"))

    print(f"Mass comparison saved to {output_dir}")


def render_kerr_spin_comparison(output_dir, M=1.0, spins=[0.0, 0.3, 0.6, 0.9]):
    """
    Render comparison for different Kerr spin parameters.

    Args:
        output_dir: Directory to save images
        M: Black hole mass
        spins: List of spin parameters (a/M)
    """
    os.makedirs(output_dir, exist_ok=True)

    camera = Camera(distance=30.0, theta=np.pi/3, phi=0.0, M=M)
    renderer = Renderer(width=1920, height=1080, M=M)

    print(f"Rendering Kerr spin comparison for spins: {spins}")

    for spin in spins:
        print(f"Rendering a = {spin}M...")
        # Note: Would need to integrate Kerr geodesics in renderer
        # This is a placeholder showing the structure

        image = renderer.render_frame(camera, parallel=False)

        # Save image
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(image_uint8)
        img.save(os.path.join(output_dir, f"kerr_spin_{spin:.2f}.png"))

    print(f"Kerr comparison saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Batch render black hole visualizations')
    parser.add_argument('--mode', choices=['orbit', 'zoom', 'mass', 'kerr'],
                       required=True, help='Rendering mode')
    parser.add_argument('--output', default='./output', help='Output directory')
    parser.add_argument('--frames', type=int, default=360, help='Number of frames')
    parser.add_argument('--mass', type=float, default=1.0, help='Black hole mass')

    args = parser.parse_args()

    if args.mode == 'orbit':
        render_orbit_animation(args.output, num_frames=args.frames, M=args.mass)
    elif args.mode == 'zoom':
        render_zoom_sequence(args.output, num_frames=args.frames, M=args.mass)
    elif args.mode == 'mass':
        render_mass_comparison(args.output)
    elif args.mode == 'kerr':
        render_kerr_spin_comparison(args.output, M=args.mass)


if __name__ == "__main__":
    main()
