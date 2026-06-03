# stellar-jax

Differentiable 1D stellar evolution in JAX. Analytic gradients through the structure ODEs via the implicit function theorem.

## What it does

Solves the four stellar structure equations (hydrostatic equilibrium, mass continuity, energy generation, energy transport) with a Newton-Raphson shooting method. Gradients computed via `@jax.custom_jvp` using the IFT at convergence — ~4 ms per gradient evaluation on a single CPU core.

## Physics

- OPAL opacity tables (4D interpolation)
- OPAL EOS (pre-inverted onto logT–logP grid)
- MLT convective energy transport (Böhm-Vitense cubic, α=1.9)
- Krishna Swamy (1966) T(τ) atmosphere integration (τ=0 to τ=2/3)
- PP + CNO nuclear burning
- Schwarzschild criterion at every shell

## Quick start

```bash
pip install jax jaxlib matplotlib numpy
```

```python
from stellar import evolve_star
import jax

# Evolve a 1 solar mass star through the main sequence
r = evolve_star(1.0, Z=0.014, max_steps=200)

# Analytic gradient: ∂(log L at TAMS) / ∂M
g = jax.grad(lambda m: evolve_star(m, Z=0.02, max_steps=5)['log_L'][-1])(1.0)
```

## Accuracy

Validated against MIST v1.2 (Z=0.014):

| Mass | Δlog L | Δlog Teff | Δlog R |
|------|--------|-----------|--------|
| 1.0 M☉ | 0.10 | 0.03 | 0.08 |
| 1.2 M☉ | 0.12 | 0.03 | 0.07 |
| 1.5 M☉ | 0.12 | 0.06 | 0.08 |
| 2.0 M☉ | 0.06 | 0.06 | 0.10 |

Residuals primarily from missing element diffusion (Thoul et al. 1994).

## Performance (c5.xlarge)

| Operation | Time |
|-----------|------|
| Forward pass (200 steps, post-JIT) | <1ms |
| **Gradient (post-JIT)** | **4 ms** |
| JIT warmup (one-time) | ~25s |

## Structure

```
stellar.py          # The code (single file, ~500 lines)
data/               # OPAL opacity + EOS tables
paper/              # ApJL draft
```

## How it was built

Built by AI agents (Claude on Amazon Bedrock) orchestrated by R. Freschi (AWS). Physics feedback from A. Dotter (Dartmouth). MIST validation data was filesystem-protected during development.

## Acknowledgments

Physics feedback: A. Dotter (Dartmouth), E. Bellinger (Yale).
