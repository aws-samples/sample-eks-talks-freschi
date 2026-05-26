# photochem_jax

A differentiable atmospheric photochemistry solver in JAX for hot Jupiter exoplanets. Computes steady-state chemical mixing ratios including thermochemical equilibrium, kinetic quenching, thermal dissociation, and UV photodissociation — all while maintaining full differentiability for gradient-based atmospheric retrieval with JWST observations.

## Installation

```bash
pip install jax jaxlib jaxopt numpy
```

Requires Python ≥ 3.9. For GPU acceleration:
```bash
pip install jax[cuda12] jaxopt numpy
```

## Quick Start

```python
from photochem_jax import evolve_atmosphere
import numpy as np

# Load atmospheric profile (T in K, P in dyn/cm²)
T_profile = np.linspace(800, 3000, 100)  # or load from GCM
P_profile = np.logspace(9, -2, 100)       # 1000 bar to 10⁻⁸ bar

# Solve photochemistry
result = evolve_atmosphere(T_profile, P_profile, stellar_flux=None, Kzz=1e9)

# Access mixing ratios
co_profile = result['CO']    # shape (100,)
h2o_profile = result['H2O']
```

## Computing Gradients

The solver is fully differentiable via JAX:

```python
import jax
import jax.numpy as jnp

def co_at_photosphere(C_O):
    """CO mixing ratio at τ=1 as function of C/O ratio."""
    result = evolve_atmosphere(T_profile, P_profile, None, Kzz=1e9, C_O=C_O)
    return jnp.log10(result['CO'][50])  # level 50 ≈ photosphere

# Gradient of log(CO) with respect to C/O
grad_fn = jax.grad(co_at_photosphere)
sensitivity = grad_fn(0.55)  # ∂log[CO]/∂(C/O)
```

This enables HMC/NUTS sampling and variational inference for atmospheric retrieval.

## API

### `evolve_atmosphere(T_profile, P_profile, stellar_flux, Kzz, C_O=0.55, metallicity=1.0)`

Compute steady-state atmospheric composition.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `T_profile` | array (N,) | Temperature profile [K] |
| `P_profile` | array (N,) | Pressure profile [dyn/cm²] |
| `stellar_flux` | any | Stellar UV flux (precomputed internally; pass `None`) |
| `Kzz` | float or array | Eddy diffusion coefficient [cm²/s] |
| `C_O` | float | Carbon-to-oxygen ratio (default: 0.55, solar) |
| `metallicity` | float | Metallicity relative to solar (default: 1.0) |

**Returns:** `dict[str, jnp.ndarray]` — Mixing ratio profiles for 93 species. Key species: `H2`, `He`, `H`, `H2O`, `CO`, `CO2`, `CH4`, `NH3`, `N2`, `HCN`, `C2H2`.

**Performance:** ~0.5 s per call (CPU). First call includes JIT compilation (~30 s).

## Data Requirements

The `data/` directory must contain:

```
data/
├── reactions/
│   └── SNCHO_photo_network_2025.txt   # VULCAN reaction network (531 reactions)
├── nasa9/
│   └── *.txt                          # NASA9 thermodynamic polynomials (93 species)
├── stellar_flux/
│   └── sflux-HD189_Moses11.txt        # Stellar UV spectrum
└── photo_cross/
    └── */                             # Photodissociation cross-sections
```

Reaction network from VULCAN (Tsai et al. 2021). NASA9 data from Burcat & Ruscic (2005). Cross-sections from the Leiden photodissociation database.

## Physics

1. **Thermochemical equilibrium** — Gibbs minimization using NASA9 polynomials for H₂↔2H, CO↔CH₄, N₂↔NH₃ systems with pressure-dependent equilibrium constants
2. **Kinetic quenching** — Chemical timescales from Zahnle & Marley (2014) and Moses et al. (2011); species freeze where τ_chem > τ_mix
3. **Thermal dissociation** — High-T override using formation equilibria for CO, CH₄, NH₃, CO₂, H₂O at T > 2500–4000 K
4. **Photodissociation** — Exponential UV depletion above quench point for CH₄ and NH₃

## Accuracy

Validated against VULCAN for HD 189733b (see `VALIDATION.md`):
- CO: 0.74 dex (thermosphere only; ≤0.2 dex in observable atmosphere)
- H2O: ≤0.2 dex in observable atmosphere
- NH3: 1.43 dex (quench region)
- CH4: 1.77 dex (photodissociation region)
- H2S: 3.02 dex (first implementation, thermal + UV parameterization)

## Limitations

- **Thermosphere (T > 3000 K):** Simplified equilibrium breaks down; errors of 1–4 dex. Not relevant for JWST observations.
- **Radical species:** Not constrained. Only major stable molecules are accurate.
- **No self-consistent radiative transfer:** UV penetration is parameterized, not computed from column densities.
- **Single column:** No horizontal transport or 3D effects.

## Future Work

- Full UV radiative transfer with wavelength-dependent optical depth
- Time-dependent kinetic integration for radical species
- Extension to cooler planets (T < 500 K) with condensation
- Coupling to emission/transmission spectrum codes (ExoJAX, PLATON)

## Citation

If you use this code, please cite:
```
photochem_jax: Differentiable Photochemical Kinetics in JAX
for Exoplanet Atmosphere Retrieval (in prep.)
```

## License

MIT
