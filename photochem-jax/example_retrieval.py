"""
Example: Gradient-based retrieval with photochem_jax

Demonstrates computing ∂log[CO]/∂(C/O) — the sensitivity of CO abundance
to the carbon-to-oxygen ratio. This gradient enables HMC/NUTS sampling
for atmospheric retrieval with JWST observations.
"""

import jax
import jax.numpy as jnp
import numpy as np
from photochem_jax import evolve_atmosphere

# --- Setup atmospheric profile (HD 189733b-like) ---
n_levels = 119
T_profile = np.linspace(800, 5800, n_levels)  # K
P_profile = np.logspace(9, -2, n_levels)       # dyn/cm² (1000 bar to 10⁻⁸ bar)
Kzz = 1e9  # cm²/s eddy diffusion

# --- 1. Forward model: compute composition ---
print("Running forward model (first call includes JIT compilation ~30s)...")
result = evolve_atmosphere(T_profile, P_profile, None, Kzz, C_O=0.55)
print(f"  CO at photosphere (level 50): {result['CO'][50]:.4e}")
print(f"  H2O at photosphere:           {result['H2O'][50]:.4e}")
print(f"  CH4 at photosphere:           {result['CH4'][50]:.4e}")

# --- 2. Compute gradient: ∂log[CO]/∂(C/O) ---
print("\nComputing gradient ∂log[CO]/∂(C/O)...")

PHOTOSPHERE = 50  # level index ≈ 1 mbar

def log_co_at_photosphere(C_O):
    """Scalar function: log10(CO mixing ratio) at the photosphere."""
    result = evolve_atmosphere(T_profile, P_profile, None, Kzz, C_O=C_O)
    return jnp.log10(result['CO'][PHOTOSPHERE])

grad_fn = jax.grad(log_co_at_photosphere)
grad_value = grad_fn(0.55)
print(f"  ∂log₁₀[CO]/∂(C/O) = {grad_value:.4f}")
print(f"  Interpretation: increasing C/O by 0.1 changes log[CO] by {grad_value * 0.1:.3f} dex")

# --- 3. Sensitivity across C/O values ---
print("\nSensitivity scan across C/O ratios:")
print(f"  {'C/O':<6} {'log[CO]':<10} {'∂log[CO]/∂(C/O)':<18}")
print(f"  {'-'*6} {'-'*10} {'-'*18}")

for c_o in [0.3, 0.45, 0.55, 0.7, 0.9]:
    log_co = log_co_at_photosphere(c_o)
    grad = grad_fn(c_o)
    print(f"  {c_o:<6.2f} {log_co:<10.4f} {grad:<18.4f}")

# --- 4. Why this matters for retrieval ---
print("""
=== Why This Matters ===

Traditional retrieval (nested sampling): ~10⁶ forward model calls
Gradient-based retrieval (HMC/NUTS):     ~10³ forward model calls

At 0.5s per call:
  Nested sampling: ~6 days
  HMC/NUTS:        ~8 minutes

The gradient computed above is exactly what HMC needs to efficiently
explore the posterior distribution P(C/O | observed spectrum).

To use with NumPyro or BlackJAX:

    import numpyro
    from numpyro.infer import MCMC, NUTS

    def model(observed_co):
        C_O = numpyro.sample('C_O', numpyro.distributions.Uniform(0.1, 1.5))
        predicted = evolve_atmosphere(T_profile, P_profile, None, Kzz, C_O=C_O)
        numpyro.sample('obs', numpyro.distributions.Normal(
            jnp.log10(predicted['CO'][50]), 0.1), obs=observed_co)

    mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=1000)
    mcmc.run(jax.random.PRNGKey(0), observed_co=-3.3)
""")
