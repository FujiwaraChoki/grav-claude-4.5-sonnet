#!/usr/bin/env python3
"""
GRAV-Bench Submission: General Relativistic Visualization
Main entry point for black hole visualization and testing.

Usage:
    python -m src.main --interactive    # Launch interactive viewer
    python -m src.main --render         # Render single frame
    python -m src.main --validate       # Run validation suite
    python -m src.main --demo           # Generate demo images
"""

import argparse
import sys
import numpy as np


def run_interactive():
    """Launch interactive visualization using pygame."""
    print("Starting interactive black hole visualization...")
    from src.blackhole import main as blackhole_main
    blackhole_main()


def run_render(width=1920, height=1080, output='output.png'):
    """Render a single high-quality frame."""
    from src.camera import Camera
    from src.renderer import Renderer
    from PIL import Image

    print(f"Rendering {width}x{height} frame...")

    M = 1.0
    camera = Camera(distance=30.0, theta=np.pi/3, phi=0.0, M=M)
    renderer = Renderer(width=width, height=height, M=M)

    import time
    start_time = time.time()
    frame = renderer.render_frame(camera, parallel=False)
    elapsed = time.time() - start_time

    print(f"\nRender completed in {elapsed:.2f} seconds")
    print(f"Performance: {width*height/elapsed:.0f} pixels/second")

    # Save image
    frame_uint8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(frame_uint8)
    img.save(output)
    print(f"Saved to {output}")


def run_validation():
    """Run GRAV-Bench validation test suite."""
    print("Running GRAV-Bench validation suite...")
    print("=" * 60)

    from tests.validation_suite import run_all_tests
    results = run_all_tests()

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    total_points = 0
    max_points = 0

    for tier, tier_results in results.items():
        print(f"\n{tier}:")
        tier_points = tier_results.get('points', 0)
        tier_max = tier_results.get('max_points', 0)
        total_points += tier_points
        max_points += tier_max

        for test_name, test_result in tier_results.items():
            if test_name not in ['points', 'max_points']:
                status = "✓ PASS" if test_result['passed'] else "✗ FAIL"
                print(f"  {status}: {test_name} ({test_result.get('points', 0)} pts)")

        print(f"  Subtotal: {tier_points}/{tier_max} points")

    print("\n" + "=" * 60)
    print(f"TOTAL SCORE: {total_points}/{max_points} points")

    if total_points >= 75:
        grade = "PASS"
        if total_points >= 100:
            grade = "A+ (Outstanding)"
        elif total_points >= 90:
            grade = "A (Excellent)"
        elif total_points >= 80:
            grade = "B (Good)"
        else:
            grade = "C (Pass)"
        print(f"GRADE: {grade}")
    else:
        print("GRADE: F (Fail)")
        print("Need minimum 75 points to pass GRAV-Bench")

    print("=" * 60)

    return total_points >= 75


def run_demo():
    """Generate demo images for GRAV-Bench submission."""
    from src.camera import Camera
    from src.renderer import Renderer
    from PIL import Image
    import os

    print("Generating GRAV-Bench demo images...")

    M = 1.0
    demo_configs = [
        {
            'name': 'example1_shadow.png',
            'description': 'Black hole shadow with accretion disk',
            'camera': Camera(distance=30.0, theta=np.pi/3, phi=0.0, M=M),
            'disk': True,
            'stars': True
        },
        {
            'name': 'example2_lensing.png',
            'description': 'Gravitational lensing of background stars',
            'camera': Camera(distance=50.0, theta=np.pi/2.5, phi=np.pi/4, M=M),
            'disk': False,
            'stars': True
        },
        {
            'name': 'example3_disk.png',
            'description': 'Accretion disk with Doppler shifting',
            'camera': Camera(distance=25.0, theta=np.pi/2.2, phi=0.0, M=M),
            'disk': True,
            'stars': False
        },
        {
            'name': 'example4_equatorial.png',
            'description': 'Equatorial view showing Einstein ring',
            'camera': Camera(distance=40.0, theta=np.pi/2, phi=0.0, M=M),
            'disk': True,
            'stars': True
        },
        {
            'name': 'example5_polar.png',
            'description': 'Polar view of accretion disk',
            'camera': Camera(distance=35.0, theta=np.pi/6, phi=0.0, M=M),
            'disk': True,
            'stars': True
        }
    ]

    os.makedirs('images', exist_ok=True)

    for i, config in enumerate(demo_configs, 1):
        print(f"\n[{i}/{len(demo_configs)}] Rendering {config['description']}...")

        renderer = Renderer(width=800, height=600, M=M)
        renderer.show_disk = config['disk']
        renderer.show_stars = config['stars']

        frame = renderer.render_frame(config['camera'], parallel=False)

        frame_uint8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(frame_uint8)
        output_path = os.path.join('images', config['name'])
        img.save(output_path)
        print(f"Saved to {output_path}")

    print(f"\n✓ Generated {len(demo_configs)} demo images in images/")


def run_physics_demo():
    """Run physics demonstration (effective potential, time dilation, etc.)."""
    print("Running physics demonstrations...")
    from src.demo_physics import main as demo_main
    demo_main()


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='GRAV-Bench: General Relativistic Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --interactive           # Interactive viewer
  python -m src.main --render --width 1920   # Render 1080p frame
  python -m src.main --validate              # Run tests
  python -m src.main --demo                  # Generate demo images
        """
    )

    parser.add_argument('--interactive', action='store_true',
                        help='Launch interactive visualization')
    parser.add_argument('--render', action='store_true',
                        help='Render single frame')
    parser.add_argument('--validate', action='store_true',
                        help='Run GRAV-Bench validation suite')
    parser.add_argument('--demo', action='store_true',
                        help='Generate demo images')
    parser.add_argument('--physics', action='store_true',
                        help='Run physics demonstrations')

    # Render options
    parser.add_argument('--width', type=int, default=1920,
                        help='Output image width (default: 1920)')
    parser.add_argument('--height', type=int, default=1080,
                        help='Output image height (default: 1080)')
    parser.add_argument('--output', type=str, default='output.png',
                        help='Output filename (default: output.png)')

    args = parser.parse_args()

    # If no arguments, show help
    if not any([args.interactive, args.render, args.validate, args.demo, args.physics]):
        parser.print_help()
        print("\nNo action specified. Use --interactive for the viewer.")
        return 1

    try:
        if args.interactive:
            run_interactive()

        if args.render:
            run_render(width=args.width, height=args.height, output=args.output)

        if args.validate:
            success = run_validation()
            return 0 if success else 1

        if args.demo:
            run_demo()

        if args.physics:
            run_physics_demo()

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return 130
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
