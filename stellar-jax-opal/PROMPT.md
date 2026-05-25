# Stellar JAX — Build from scratch

## What to build

A differentiable stellar evolution code in JAX. Single file `stellar.py`.

## Acceptance test

SSH to ec2-user@3.8.6.50 (key: ~/.ssh/kiro-duo-key.pem), put your code at ~/stellar-jax-final/stellar.py, then run this EXACT script. It must print PASS at the end. There is no partial credit — ALL checks must pass in a single run.

```bash
cd ~/stellar-jax-final && python3.11 test_acceptance.py
```

Deploy this as `test_acceptance.py`:

```python
"""Acceptance test. Must print PASS at the end. No partial credit."""
import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import sys, random, os
sys.path.insert(0, '.')
from stellar import evolve_star

failures = []

# --- TEST 1: Evolution produces full MS track ---
r = evolve_star(1.0, Z=0.0142857, max_steps=200)
if len(r['log_L']) < 200: failures.append(f"Track too short: {len(r['log_L'])} pts")
if r['star_age'][-1] < 5e9: failures.append(f"Age too short: {r['star_age'][-1]/1e9:.1f} Gyr")
if r['center_h1'][-1] > 0.1: failures.append(f"H not depleted: {r['center_h1'][-1]:.2f}")
if r['log_L'][-1] <= r['log_L'][0]: failures.append("Star doesn't brighten")
print(f"Test 1 (evolution): {'FAIL - ' + '; '.join(failures) if failures else 'OK'}")

# --- TEST 2: Gradient through entire solver ---
try:
    g = jax.grad(lambda m: evolve_star(m, Z=0.02, max_steps=5)['log_L'][-1])(1.0)
    if not (jnp.isfinite(g) and g != 0):
        failures.append(f"Gradient not finite/nonzero: {g}")
    print(f"Test 2 (gradient): OK, grad={float(g):.4f}")
except Exception as e:
    failures.append(f"Gradient exception: {e}")
    print(f"Test 2 (gradient): FAIL - {e}")

# --- TEST 3: Full track accuracy <0.05 dex vs MIST ---
mist_dir = '/home/ec2-user/stellar-jax/data/mist/MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_EEPS'
for i in range(5):
    mass = round(random.uniform(0.8, 2.0), 2)
    mass_int = int(round(mass * 100))
    available = sorted(int(f[:5]) for f in os.listdir(mist_dir) if f.endswith('.eep'))
    nearest = min(available, key=lambda m: abs(m - mass_int))
    fname = f'{nearest:05d}M.track.eep'
    lines = open(f'{mist_dir}/{fname}').readlines()
    for j, l in enumerate(lines):
        if l.startswith('#') and 'star_age' in l: hi = j; break
    cols = lines[hi].strip('#').split()
    data = np.loadtxt(lines[hi+1:])
    cm = {n: j for j, n in enumerate(cols)}
    ms = data[201:]  # EEP 202 (index 201) = ZAMS onward
    ms = ms[ms[:, cm['center_h1']] > 0.01]  # until TAMS
    mist_age = ms[:, cm['star_age']]

    r = evolve_star(nearest/100, Z=0.0142857, max_steps=200)
    jax_age = np.array(r['star_age'])

    star_ok = True
    for q in ['log_L', 'log_Teff', 'log_R']:
        jax_vals = np.interp(mist_age, jax_age, np.array(r[q]))
        err = float(np.max(np.abs(jax_vals - ms[:, cm[q]])))
        if err >= 0.05:
            failures.append(f"{nearest/100} Msun {q}: {err:.3f} dex")
            star_ok = False
    status = "OK" if star_ok else "FAIL"
    print(f"  {nearest/100} Msun: {status}")

# --- FINAL VERDICT ---
print("\n" + "="*40)
if failures:
    print("FAIL")
    for f in failures:
        print(f"  - {f}")
else:
    print("PASS")
```

## What the code must do

1. Solve the 4 stellar structure ODEs (dP/dr, dM/dr, dL/dr, dT/dr) via shooting method — NOT scaling relations or polytropic approximations
2. Use OPAL opacity tables (at ~/stellar-jax/data/opal/opal_4d.npz) — Kramers CANNOT achieve <0.05 dex on full tracks
3. Evolve composition over time, RE-SOLVING structure at each step (not homology L∝μ^N)
4. Be fully differentiable — use @jax.custom_jvp with implicit function theorem for the shooting solver
5. Match MIST to <0.05 dex on FULL evolutionary tracks (not just ZAMS)

## Why this is hard

Kramers opacity gives L∝μ⁴ response to composition changes. Real stars (OPAL) show L∝μ^1.3. This causes >1 dex errors during evolution. You MUST use OPAL tables in the structure integration for the full-track test to pass.

## Architecture

```python
@jax.custom_jvp
def solve_structure(M_solar, X, Z):
    # Newton-Raphson: find (log_L, log_Teff) satisfying center BCs
    # Integrate from surface inward using RK4
    # Opacity from OPAL interpolation (not Kramers)
    return log_L, log_Teff, log_R

@solve_structure.defjvp
def solve_structure_jvp(primals, tangents):
    # Implicit function theorem: dx/dp = -(dR/dx)^-1 @ (dR/dp)
    ...

def evolve_star(mass, Z, max_steps=200):
    # jax.lax.scan over timesteps
    # Each step: update composition, call solve_structure
    return {'star_age': ..., 'log_L': ..., 'log_Teff': ..., 'log_R': ..., 'center_h1': ...}
```

## Available on EC2

- ~/stellar-jax/reference/statstar.py — reference structure solver
- ~/stellar-jax/data/mist/ — MIST tracks (validation only, do NOT read in computation)
- ~/stellar-jax/data/opal/opal_4d.npz — OPAL opacity tables (USE THIS in computation)
- Python 3.11 with JAX

## Workspace

~/stellar-jax-final/
