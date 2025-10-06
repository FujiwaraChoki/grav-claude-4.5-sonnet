# Quick Start Guide

## Installation (1 minute)

```bash
# Install dependencies
pip install -r requirements.txt
```

## Run Interactive Visualization (Immediate)

```bash
# Start the interactive black hole viewer
python blackhole.py
```

**First run will be slower** (Numba JIT compilation) - subsequent runs are faster.

## Controls Summary

| Action | Keys |
|--------|------|
| **Move Camera** | Arrow Keys (orbit), W/S (zoom) |
| **View** | Q/E (rotate), R (reset) |
| **Quality** | Space (toggle HQ render) |
| **Features** | D (disk), T (stars), L (lensing) |
| **Black Hole** | M (cycle mass), A (Kerr mode), +/- (spin) |

## What You'll See

1. **Black Circle** = Event horizon shadow (photon sphere projection)
2. **Glowing Disk** = Accretion disk with:
   - Blue side = Moving toward you (Doppler blue shift)
   - Red side = Moving away (Doppler red shift)
   - One side brighter = Relativistic beaming
3. **Warped Stars** = Gravitational lensing of background
4. **Multiple Images** = Einstein rings (light orbiting black hole)

## Quick Demos

### 1. See Gravitational Lensing (30 seconds)
```bash
python blackhole.py
# Wait for render
# Press Arrow Keys to orbit - watch stars bend around black hole
```

### 2. Study Physics (2 minutes)
```bash
python demo_physics.py
# Shows calculations and creates plots
```

### 3. Create Animation (5-10 minutes)
```bash
python batch_render.py --mode orbit --frames 360 --output ./orbit_animation
# Creates 360 frames orbiting the black hole
# Then: ffmpeg -i ./orbit_animation/frame_%04d.png -c:v libx264 -r 30 output.mp4
```

## Performance Tips

- **First render**: 10-30 seconds (JIT compilation)
- **Interactive mode**: Uses 4x downsampling for ~1-2 FPS
- **High quality** (Space): Full resolution, 10-30 sec per frame
- **For faster preview**: Reduce resolution in `blackhole.py` (line 17-18)

## Troubleshooting

**Slow rendering?**
- Normal for first frame (compilation)
- Reduce `WIDTH, HEIGHT` in `blackhole.py`
- Use preview mode (don't press Space)

**Black screen?**
- Press T (toggle stars)
- Press D (toggle disk)
- Press R (reset camera - might be too close)

**Want better quality?**
- Press Space for high-quality render
- Increase `max_steps` in `raytracer.py` line 112
- Decrease `base_step` in `raytracer.py` line 93

## Physics Accuracy

This implements **exact** general relativistic ray tracing:
- ✓ Full Schwarzschild & Kerr metrics
- ✓ RK4 geodesic integration
- ✓ All relativistic effects (lensing, Doppler, beaming, redshift)
- ✓ Photon sphere at r=3M, event horizon at r=2M, ISCO at r=6M

**Not approximations** - this is the real physics!

## Next Steps

- Read `README.md` for full documentation
- Explore `advanced_features.py` for analysis tools
- Modify `schwarzschild.py` to experiment with geodesics
- Try Kerr mode (press A) for rotating black hole

## Example Session

```bash
python blackhole.py

# You'll see:
# - Black hole with accretion disk
# - Background stars being lensed
# - UI showing camera position and physics parameters

# Try:
# 1. Press arrows to orbit
# 2. Press W to zoom in (careful near horizon!)
# 3. Press Space for HQ render
# 4. Press M to see different masses
# 5. Press A for rotating (Kerr) black hole
# 6. Press +/- to adjust spin

# Watch for:
# - Einstein rings (multiple star images)
# - Doppler shifting (disk color asymmetry)
# - Photon sphere shadow (perfectly black circle)
# - Relativistic beaming (one side of disk brighter)
```

## Learning Resources

The code itself is heavily commented with physics explanations:
- `schwarzschild.py` - Metric and geodesics (lines 1-30)
- `accretion_disk.py` - Doppler & redshift formulas (lines 60-90)
- `raytracer.py` - Integration method (lines 15-50)

Enjoy exploring general relativity! 🕳️✨
