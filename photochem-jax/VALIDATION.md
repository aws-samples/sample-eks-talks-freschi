# Validation Report: photochem_jax

Validated against VULCAN (Tsai et al. 2017, 2021) for HD 189733b with solar composition (C/O = 0.55, [M/H] = 0).

## Summary

| Test | Result |
|------|--------|
| Gradient (∂composition/∂C/O) | **PASS** — 1.05×10⁻² |
| Speed | **PASS** — 0.5 s per atmosphere |
| Accuracy (vs VULCAN) | See below |

## Major Species Accuracy

| Species | Max \|Δlog₁₀\| | Worst Level | T (K) | P (dyn/cm²) | Region |
|---------|----------------|-------------|--------|--------------|--------|
| CO | 0.74 dex | 117 | 4734 | 0.015 | Thermosphere |
| CH4 | 1.77 dex | 116 | 3484 | 0.019 | Thermosphere |
| CO2 | 1.96 dex | 115 | 2340 | 0.023 | Upper atmosphere |
| NH3 | 2.15 dex | 41 | 1272 | 1.6×10⁵ | Quench region |
| H2O | 3.58 dex | 118 | 5745 | 0.012 | Thermosphere |

## Per-Region Analysis

The 119-level atmosphere is divided into four regions based on the dominant physics:

### Deep Atmosphere (levels 0–30, T = 1500–2800 K, P > 10⁶ dyn/cm²)

Chemistry is fast; local thermochemical equilibrium holds. The Gibbs minimization solver is accurate here.

| Species | Typical Error | Notes |
|---------|--------------|-------|
| CO | ≤ 0.2 dex | Dominant C carrier, well-constrained |
| H2O | ≤ 0.2 dex | Dominant O carrier after CO |
| CH4 | < 0.1 dex | Trace at these temperatures |
| CO2 | < 0.3 dex | Trace, sensitive to K₂ accuracy |
| NH3 | < 0.5 dex | Trace at T > 1500 K |

### Quench Region (levels 30–80, T = 800–1500 K, P = 10²–10⁶ dyn/cm²)

Transport timescale becomes comparable to chemical timescale. Species "freeze" at their quench-point values.

| Species | Typical Error | Notes |
|---------|--------------|-------|
| CO | ≤ 0.2 dex | Quenches early, well-captured |
| H2O | ≤ 0.2 dex | Quenches with CO |
| CH4 | 0.5–1.2 dex | Quench value sensitive to τ_chem parameterization |
| CO2 | 0.5–0.6 dex | Quench value slightly low |
| NH3 | **2.2 dex** | Quench point at wrong pressure; N₂↔NH₃ timescale uncertain |

The NH3 quench error (2.2 dex at levels 39–43) is the dominant mid-atmosphere limitation. The N₂↔NH₃ interconversion timescale from Moses et al. (2011) may not be accurate for this T/P regime.

### Upper Atmosphere (levels 80–115, T = 800–2340 K, P = 0.02–50 dyn/cm²)

Photodissociation dominates. UV destroys CH4 and NH3 above their quench points.

| Species | Typical Error | Notes |
|---------|--------------|-------|
| CO | ≤ 0.2 dex | Stable against photolysis |
| H2O | ≤ 0.2 dex | Stable in this region |
| CH4 | 1.2–1.3 dex | Photodissociation parameterized, not full RT |
| CO2 | 0.6 dex | Quenched value propagated |
| NH3 | < 1 dex | Photodissociation applied |

### Thermosphere (levels 115–118, T = 2340–5745 K, P < 0.02 dyn/cm²)

Extreme conditions: thermal dissociation of all molecules, H₂ fully dissociated. This region is NOT observable by JWST (optical depth ≪ 1) but dominates the max-error metric.

| Species | Error | Cause |
|---------|-------|-------|
| CO | 0.4–0.7 dex | UV photodissociation not modeled |
| H2O | 1.0–3.6 dex | Equilibrium solver over-depletes at T > 5000 K |
| CH4 | 1.2–1.8 dex | Override formula pressure dependence |
| CO2 | 0.8–2.0 dex | Quench value error at boundary |
| NH3 | < 1 dex | Photodissociation handles this region |

## Accuracy in the Observable Atmosphere

JWST transmission spectroscopy probes pressures of ~10⁻¹ to 10⁴ dyn/cm² (roughly levels 50–110). In this region:

- **CO**: ≤ 0.2 dex — **retrieval-ready**
- **H2O**: ≤ 0.2 dex — **retrieval-ready**
- **CO2**: 0.6 dex — usable with caveats
- **CH4**: 1.2 dex — photodissociation region needs improvement
- **NH3**: 2.2 dex — quench parameterization needs work

## Context: How Accurate Are Other Codes?

Different photochemistry codes routinely disagree by 0.3–1 dex for the same planet:
- VULCAN vs Photochem (Hu et al.): ~0.5 dex for major species (Tsai et al. 2021)
- Equilibrium vs disequilibrium: 1–4 dex for CH4, NH3 in hot Jupiters
- Different reaction networks: 0.3–1 dex depending on rate coefficient choices

Our CO accuracy (0.74 dex, driven by one thermosphere level) is comparable to inter-code disagreements. For the observable atmosphere, CO and H2O are within the "good agreement" threshold.

## Known Limitations

1. **No radiative transfer for photodissociation** — UV optical depth is parameterized, not computed from cross-sections × column density
2. **Thermosphere accuracy** — at T > 3000 K and P < 0.1 dyn/cm², the simplified equilibrium + quenching approach breaks down
3. **NH3 quench timescale** — the N₂↔NH₃ interconversion rate is uncertain by ~2 orders of magnitude in the literature
4. **Minor/radical species** — not constrained (10–25 dex errors); these require full time-dependent kinetics

## Gradient Verification

The solver is fully differentiable. `jax.grad` of the CO mixing ratio with respect to C/O ratio gives:

```
∂[CO]/∂(C/O) = 1.05 × 10⁻²
```

This enables gradient-based inference (HMC, NUTS, variational methods) for atmospheric retrieval — the primary motivation for this code.

## Reproducing Validation

The validation oracle compares against VULCAN output for HD 189733b (Tsai et al. 2021). To reproduce: run VULCAN with the default HD 189733b configuration, then compare mixing ratio profiles in log space. The reference data is not distributed with this code due to size — contact the authors or run VULCAN yourself (github.com/exoclime/VULCAN).
