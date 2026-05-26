# stellar-jax-opal

Differentiable stellar evolution in JAX with OPAL opacity + EOS tables. Solves the 4 stellar structure ODEs via Newton-Raphson shooting, fully differentiable via `@jax.custom_jvp` with implicit function theorem.

## What it does

Given a star's mass and metallicity, computes its main-sequence evolutionary track: luminosity L(t), effective temperature T_eff(t), and radius R(t) from ZAMS to TAMS. Gradients ∂L/∂mass, ∂Teff/∂Z, etc. are computed via `jax.grad` in milliseconds.

## Usage

```python
import jax; jax.config.update('jax_enable_x64', True)
from stellar import evolve_star

# Evolve a 1 solar mass star
r = evolve_star(1.0, Z=0.0142857, max_steps=200)
# Returns: star_age, log_L, log_Teff, log_R, center_h1

# Gradient of final luminosity w.r.t. mass (4ms post-JIT)
g = jax.grad(lambda m: evolve_star(m, Z=0.02, max_steps=5)['log_L'][-1])(1.0)
```

## Performance (verified on c5.2xlarge)

| Operation | Time |
|-----------|------|
| JIT warmup (one-time) | ~60s |
| Forward pass (5 steps, post-JIT) | <1ms |
| **Gradient (post-JIT)** | **4ms** |
| Forward pass (50 steps) | ~21s |

## Requirements

- JAX with float64 support
- Two data files (not included — physics data, not code):
  - `opal_4d.npz` — OPAL Rosseland mean opacity tables, available from [OPAL project](https://opalopacity.llnl.gov/)
  - `eos_compact.npz` — OPAL EOS tables (pre-parsed), derived from [MESA EOS data](https://github.com/MESAHub/mesa)

Place them in a `data/` subdirectory next to `stellar.py`, or set environment variables:
```bash
export STELLAR_OPAL_OPACITY=/path/to/opal_4d.npz
export STELLAR_OPAL_EOS=/path/to/eos_compact.npz
```

## Physics

- **Structure equations**: dP/dr, dM/dr, dL/dr, dT/dr (hydrostatic equilibrium, mass continuity, energy generation, energy transport)
- **Opacity**: OPAL Type-1 Rosseland mean (4D interpolation in X, Z, logT, logR)
- **EOS**: Pre-computed inverse OPAL EOS table (bilinear lookup for ρ, μ, ∇_ad from P, T, X, Z — no Newton iteration in hot path)
- **Nuclear energy**: PP-chain + CNO cycle with 3He non-equilibrium suppression
- **Convection**: Schwarzschild criterion with adiabatic gradient from OPAL EOS tables
- **Atmosphere**: Eddington gray from τ=2/3
- **Composition**: Stratified X(M_r) burning core profile, evolved via `jax.lax.scan`
- **Differentiability**: `@jax.custom_jvp` via implicit function theorem on the Newton-Raphson shooting residual — gradient bypasses Newton iterations entirely

## Accuracy

Validated against MIST evolutionary tracks (Choi et al. 2016):

| Mass range | log_L | log_Teff | log_R |
|-----------|-------|----------|-------|
| M ≥ 1.4 M☉ | 0.05-0.08 dex | 0.03-0.06 dex | 0.05-0.09 dex |
| 1.0-1.4 M☉ | 0.05-0.10 dex | 0.03-0.05 dex | 0.05-0.10 dex |
| M < 1.0 M☉ | 0.10-0.13 dex | 0.05-0.10 dex | 0.10-0.13 dex |

The accuracy floor (~0.05 dex) is set by OPAL EOS table X-resolution (5 points). Higher-resolution Type-2 tables would close this gap. Low-mass stars additionally need molecular opacities and element diffusion.

## How it was built

Built autonomously by AI agents (Claude Opus on Amazon Bedrock) using the [kiro-duo](https://github.com/aws-samples/sample-eks-talks-freschi/tree/stellar-jax-kiro) teacher-worker architecture. Key steps:
1. Structure solver with OPAL opacity (shooting method, RK4)
2. OPAL EOS integration (Newton inversion on tables)
3. Physics modules via subagents (MLT, atmosphere, overshooting, nuclear)
4. Speed optimization: pre-computed inverse EOS table eliminates Newton from hot path (307s → 4ms gradient)

No MIST data was accessed during computation (enforced by file permissions).

## Comparison

| | [stellar-jax-kiro](../stellar-jax-kiro/) (v1) | stellar-jax-opal (this) |
|---|---|---|
| Opacity | Kramers (analytic) | OPAL tables |
| EOS | Ideal gas | OPAL (pre-computed inverse) |
| ZAMS accuracy | 0.03 dex | 0.05-0.08 dex |
| Full track accuracy | ~1.2 dex (fails) | 0.05-0.13 dex |
| Gradient speed | 0.3s | 0.004s |
| Lines | 319 | 481 |
| Convection | min(∇_rad, ∇_ad) | Schwarzschild + OPAL ∇_ad |
