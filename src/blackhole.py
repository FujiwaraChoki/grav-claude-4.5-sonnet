#!/usr/bin/env python3
"""
Physically Accurate Black Hole Visualization
Using Schwarzschild and Kerr metrics with full general relativistic ray tracing.

Controls:
- Arrow Keys: Orbit camera around black hole
- W/S: Zoom in/out
- Q/E: Rotate view left/right
- R: Reset camera
- Space: Toggle high quality render
- D: Toggle accretion disk
- T: Toggle star field
- L: Toggle lensing effects
- M: Cycle black hole mass
- A: Toggle Kerr (rotating) mode
- +/-: Adjust spin parameter
- ESC: Quit

Features:
- Schwarzschild and Kerr metrics
- Geodesic ray tracing with RK4 integration
- Gravitational lensing of background stars
- Accretion disk with Doppler shifting and relativistic beaming
- Photon sphere shadow at r = 3M
- Event horizon at r = 2M
- Einstein rings and multiple images
"""

import pygame
import numpy as np
import sys
from src.camera import Camera
from src.renderer import Renderer
from src.metric import schwarzschild_radius, photon_sphere_radius, isco_radius
from src.kerr import kerr_event_horizon, kerr_isco

# Initialize Pygame
pygame.init()

# Configuration
WIDTH, HEIGHT = 800, 600
FPS = 30

# Black hole parameters
M = 1.0  # Black hole mass (geometric units)
SPIN = 0.0  # Spin parameter (0 to M for Kerr)

def draw_ui(screen, camera, renderer, M, spin, kerr_mode, fps):
    """Draw UI overlay with information and controls."""
    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 18)

    # Camera info
    cam_info = camera.get_info()
    r, theta, phi = cam_info['position_sph']
    rs_ratio = cam_info['distance_in_rs']

    ui_texts = [
        f"FPS: {fps:.1f}",
        f"Camera: r={r:.1f}M ({rs_ratio:.1f} R_s)",
        f"θ={theta:.1f}°, φ={phi:.1f}°",
        f"Black Hole Mass: M={M:.1f}",
        f"Mode: {'Kerr (Rotating)' if kerr_mode else 'Schwarzschild'}",
    ]

    if kerr_mode:
        ui_texts.append(f"Spin: a={spin:.2f}M")
        ui_texts.append(f"ISCO: r={kerr_isco(M, spin):.2f}M")
    else:
        ui_texts.append(f"Schwarzschild Radius: {schwarzschild_radius(M):.2f}M")
        ui_texts.append(f"Photon Sphere: {photon_sphere_radius(M):.2f}M")
        ui_texts.append(f"ISCO: {isco_radius(M):.2f}M")

    y_offset = 10
    for text in ui_texts:
        surf = font.render(text, True, (255, 255, 255))
        screen.blit(surf, (10, y_offset))
        y_offset += 25

    # Controls
    controls = [
        "Controls:",
        "Arrows: Orbit | W/S: Zoom | Q/E: Rotate",
        "Space: HQ Render | D: Disk | T: Stars | L: Lensing",
        "M: Cycle Mass | A: Toggle Kerr | +/-: Spin | R: Reset"
    ]

    y_offset = HEIGHT - len(controls) * 20 - 10
    for text in controls:
        surf = small_font.render(text, True, (200, 200, 200))
        screen.blit(surf, (10, y_offset))
        y_offset += 20

    # Render settings
    settings = [
        f"Disk: {'ON' if renderer.show_disk else 'OFF'}",
        f"Stars: {'ON' if renderer.show_stars else 'OFF'}",
        f"Lensing: {'ON' if renderer.show_lensing else 'OFF'}",
    ]

    y_offset = 10
    for text in settings:
        surf = small_font.render(text, True, (255, 255, 100))
        screen.blit(surf, (WIDTH - 120, y_offset))
        y_offset += 20


def main():
    """Main application loop."""
    # Set up display
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Black Hole Visualization - General Relativity")
    clock = pygame.time.Clock()

    # Initialize components
    global M, SPIN
    camera = Camera(distance=30.0, theta=np.pi/3, phi=0.0, M=M)
    renderer = Renderer(width=WIDTH, height=HEIGHT, M=M)

    # State
    running = True
    high_quality = False
    kerr_mode = False
    needs_render = True
    current_frame = None

    print("Black Hole Visualization Starting...")
    print("Compiling JIT functions (first render may be slow)...")

    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                needs_render = True

                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    camera.reset()
                    print("Camera reset")

                elif event.key == pygame.K_SPACE:
                    high_quality = not high_quality
                    print(f"High quality: {high_quality}")

                elif event.key == pygame.K_d:
                    renderer.toggle_disk()
                    print(f"Accretion disk: {renderer.show_disk}")

                elif event.key == pygame.K_t:
                    renderer.toggle_stars()
                    print(f"Stars: {renderer.show_stars}")

                elif event.key == pygame.K_l:
                    renderer.toggle_lensing()
                    print(f"Lensing: {renderer.show_lensing}")

                elif event.key == pygame.K_m:
                    # Cycle mass
                    M = M * 2 if M < 8 else 1.0
                    camera.M = M
                    renderer.M = M
                    print(f"Black hole mass: {M}M")

                elif event.key == pygame.K_a:
                    kerr_mode = not kerr_mode
                    SPIN = 0.5 * M if kerr_mode else 0.0
                    print(f"Kerr mode: {kerr_mode}, Spin: {SPIN}")

                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    if kerr_mode:
                        SPIN = min(SPIN + 0.1 * M, 0.99 * M)
                        print(f"Spin: a={SPIN/M:.2f}M")

                elif event.key == pygame.K_MINUS:
                    if kerr_mode:
                        SPIN = max(SPIN - 0.1 * M, 0.0)
                        print(f"Spin: a={SPIN/M:.2f}M")

        # Continuous key handling for camera movement
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            camera.orbit_horizontal(-1)
            needs_render = True
        if keys[pygame.K_RIGHT]:
            camera.orbit_horizontal(1)
            needs_render = True
        if keys[pygame.K_UP]:
            camera.orbit_vertical(-1)
            needs_render = True
        if keys[pygame.K_DOWN]:
            camera.orbit_vertical(1)
            needs_render = True
        if keys[pygame.K_w]:
            camera.zoom(-1)
            needs_render = True
        if keys[pygame.K_s]:
            camera.zoom(1)
            needs_render = True
        if keys[pygame.K_q]:
            camera.rotate_view(0, -1)
            needs_render = True
        if keys[pygame.K_e]:
            camera.rotate_view(0, 1)
            needs_render = True

        # Render frame if needed
        if needs_render:
            print("\nRendering frame...")
            if high_quality:
                # High quality render
                frame = renderer.render_frame(camera, parallel=False)
            else:
                # Low res preview for interactivity
                frame = renderer.render_low_res_preview(camera, scale=4)
                # Upscale for display
                from scipy.ndimage import zoom
                frame = zoom(frame, (4, 4, 1), order=1)

            # Convert to pygame surface
            frame_uint8 = (frame * 255).astype(np.uint8)
            current_frame = pygame.surfarray.make_surface(frame_uint8.swapaxes(0, 1))
            needs_render = False

        # Display
        screen.fill((0, 0, 0))
        if current_frame is not None:
            screen.blit(current_frame, (0, 0))

        # Draw UI
        fps = clock.get_fps()
        draw_ui(screen, camera, renderer, M, SPIN, kerr_mode, fps)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    print("\nVisualization ended.")
    sys.exit(0)


if __name__ == "__main__":
    main()
