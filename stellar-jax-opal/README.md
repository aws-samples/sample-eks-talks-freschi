# stellar-jax-opal

Differentiable stellar evolution in JAX with OPAL opacity tables. Solves the 4 stellar structure ODEs via Newton-Raphson shooting, fully differentiable via `@jax.custom_jvp` with implicit function theorem.

## What it does

Given a star's mass and metallicity, computes its main-sequence evolutionary track: luminosity L(t), effective temperature T_eff(t), and radius R(t) from ZAMS to TAMS. Gradients ∂L/∂mass, ∂Teff/∂Z, etc. are computed via `jax.grad`.

## Usage

```python
import jax; jax.config.update('jax_enable_x64', True)
from stellar import evolve_star

# Evolve a 1 solar mass star
r = evolve_star(1.0, Z=0.0142857, max_steps=200)
# Returns: star_age, log_L, log_Teff, log_R, center_h1

# Gradient of final luminosity w.r.t. mass
g = jax.grad(lambda m: evolve_star(m, Z=0.02, max_steps=5)['log_L'][-1])(1.0)
```

## Requirements

- JAX with float64 support
- OPAL opacity tables (`opal_4d.npz`) — not included, see below

## OPAL Tables

The code requires OPAL opacity tables at the path specified in `stellar.py`. These are physics data (pre-computed atomic opacities) available from the [OPAL project](https://opalopacity.llnl.gov/).

## Physics

- **Structure equations**: dP/dr, dM/dr, dL/dr, dT/dr (hydrostatic equilibrium, mass continuity, energy generation, energy transport)
- **Opacity**: OPAL Type-1 Rosseland mean (4D interpolation in X, Z, logT, logR) + H⁻ bound-free/free-free for T < 7000K
- **Nuclear energy**: PP-chain + CNO cycle with 3He non-equilibrium suppression
- **Convection**: Böhm-Vitense MLT cubic (Cox & Giuli 1968) with Schwarzschild criterion
- **Atmosphere**: Eddington gray from τ=2/3 to τ=20
- **Overshooting**: Calibrated f_ov for convective cores (M > 1.15 M☉)
- **EOS**: Ideal gas + radiation + partial ionization μ(T)
- **Differentiability**: `@jax.custom_jvp` via implicit function theorem on the Newton-Raphson shooting residual

## Accuracy

Validated against MIST evolutionary tracks (Choi et al. 2016):

| Mass range | log_L | log_Teff | log_R |
|-----------|-------|----------|-------|
| M ≥ 1.4 M☉ | < 0.05 dex | < 0.05 dex | < 0.05 dex |
| 1.0-1.4 M☉ | 0.05-0.12 dex | < 0.05 dex | 0.05-0.10 dex |
| M < 1.0 M☉ | 0.10-0.30 dex | 0.05-0.15 dex | 0.10-0.30 dex |

The accuracy gap at low mass is due to simplified envelope physics (ideal gas EOS vs OPAL EOS with partial ionization, missing low-T molecular opacities). The architecture supports drop-in replacement with tabulated EOS.

## How it was built

Built autonomously by AI agents (Claude on Amazon Bedrock) using the [kiro-duo](https://github.com/aws-samples/sample-eks-talks-freschi/tree/stellar-jax-kiro) teacher-worker architecture. The agent system:
- Used OPAL opacity tables as physics input (allowed)
- Did NOT read MIST tracks during computation (validated by file permission enforcement)
- Implemented `custom_jvp` for efficient gradient computation
- Iterated through multiple physics modules via subagents (MLT, atmosphere, overshooting, nuclear)

## Comparison to previous version

| | [stellar-jax-kiro](../stellar-jax-kiro/) | stellar-jax-opal (this) |
|---|---|---|
| Opacity | Kramers (analytic) | OPAL tables (4D interpolation) |
| ZAMS accuracy | 0.03 dex | 0.02-0.05 dex |
| Full track accuracy | ~1.2 dex (fails) | 0.05 dex (M≥1.4 passes) |
| Convection | min(∇_rad, ∇_ad) | MLT cubic |
| Lines | 319 | 542 |
