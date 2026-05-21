# Differentiable Stellar Evolution Solver in JAX

A 301-line differentiable stellar structure and evolution solver built entirely by AI agents (Claude on Amazon Bedrock via [Kiro CLI](https://kiro.dev)). This accompanies the blog post: *Using AI Agents to Build Scientific Computing Tools in Unfamiliar Domains*.

## What it does

Given a star's mass (0.8–2.0 solar masses) and metallicity, the solver:

1. Solves the four stellar structure differential equations (hydrostatic equilibrium, energy transport, energy generation, mass continuity)
2. Finds self-consistent stellar models via a shooting method with Newton-Raphson iteration
3. Evolves composition over time (hydrogen burning → helium) across the main sequence
4. Is fully differentiable with `jax.grad` — enabling gradient-based inference of stellar parameters

## Key features

- **Pure JAX** — no NumPy, no SciPy, no Python control flow in the computation path
- **JIT-compilable** — single solve < 2 seconds after compilation
- **Differentiable** — uses implicit differentiation via `jax.custom_jvp` through the iterative solver
- **Accurate** — matches MIST/MESA reference tracks to ~0.03 dex (7%) at ZAMS

## Quick start

```bash
pip install jax jaxlib numpy
```

```python
import jax
jax.config.update('jax_enable_x64', True)
from stellar import evolve_star

# Evolve a 1 solar mass star
result = evolve_star(1.0, Z=0.0142857, max_steps=200)
print(f"Track: {len(result['log_L'])} points, {result['star_age'][-1]/1e9:.1f} Gyr")

# Compute gradient of final luminosity w.r.t. mass
grad_fn = jax.grad(lambda m: evolve_star(m, Z=0.02, max_steps=5)['log_L'][-1])
print(f"d(log_L)/dM = {grad_fn(1.0):.4f}")
```

## Files

| File | Description |
|------|-------------|
| `stellar.py` | The complete solver — structure equations, shooting method, time evolution |
| `validate.py` | Validation script comparing output against MIST evolutionary tracks |

## How it was built

An autonomous agent system ran for 7 days using a teacher-student architecture:
- **Worker** (Claude Sonnet): writes code, runs tests, iterates
- **Reviewer** (Claude Opus): catches shortcuts, validates physics

The agents completed 20 of 26 tasks autonomously. The remaining tasks required human judgment — distinguishing legitimate physics from shortcuts, setting accuracy targets, and knowing when simplified physics is sufficient.

A parallel interactive session with Kiro CLI produced the final 301-line solution in one hour — demonstrating that the bottleneck in autonomous systems is feedback loop latency, not model capability.

## Related repositories

- [kiro-duo](https://github.com/freschri/kiro-duo) — The two-agent relay system
- [autonomous-agent-framework](https://github.com/freschri/autonomous-agent-framework) — Task orchestration framework

## License

This library is licensed under the MIT-0 License. See the [LICENSE](../LICENSE) file.
