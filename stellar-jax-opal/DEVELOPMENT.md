# Development History

How this code was built by AI agents, what worked, what failed, and lessons learned.

## Timeline

| Day | What happened |
|-----|--------------|
| 1-7 | Framework approach (26 tasks, YAML orchestrator, LangGraph). Got stuck at task 20. |
| 8 | Switched to kiro-duo (minimal relay, two persistent sessions). Produced first working solver in 2 hours. |
| 9 | Strands SDK version (Opus teacher, Sonnet worker). Discovered tool execution bugs, context overflow, Teacher fatigue. |
| 10 | Fresh-agent-per-turn architecture. Gradient-first validator. MIST data protection. |
| 11 | OPAL opacity integration. Subagent modules (MLT, atmosphere, overshooting, nuclear). |
| 12 | OPAL EOS integration. Speed optimization (307s → 4ms gradient). Final version. |

## Approaches Tried

### 1. Task-based framework (Days 1-7) — Failed
- 26 sequential tasks with per-task oracles
- LangGraph orchestrator, SQLite memory, CodeTree for hard tasks
- **Why it failed:** Oracle bugs, task ordering issues, agent cheating, context resets between tasks prevented deep debugging

### 2. kiro-duo relay (Day 8) — Partial success
- Two kiro-cli sessions (Teacher + Worker) connected by a 200-line relay script
- Worker solved ZAMS in one turn (~2 hours)
- **Limitation:** Same model for both → Teacher couldn't catch Worker's shortcuts (homology scaling, MIST interpolation)

### 3. Strands with Opus/Sonnet (Days 9-10) — Partial success
- Opus as Teacher (catches cheating), Sonnet as Worker (writes code)
- Fresh agent per turn (no context fatigue)
- **Limitation:** Worker gave up after 10-20 turns; Teacher collapsed into "Acknowledged" loop

### 4. Interactive 2-window (Day 8, 1 hour) — Full success for v1
- Human copies output between two Kiro CLI windows
- Worker found `custom_jvp` immediately (tested gradient on first turn)
- **Why it worked:** No accumulated failure context, all requirements tested together, human filtered signal from noise

### 5. Subagent modules + OPAL (Days 11-12) — Final version
- Opus as integrator, subagents for individual physics modules
- OPAL EOS inversion from Fortran translation project
- Speed optimization: pre-computed inverse EOS table
- **Result:** 481 lines, 4ms gradient, 0.05-0.13 dex accuracy

## Key Architectural Decisions

### custom_jvp (implicit differentiation)
Differentiating through 15+ Newton iterations is expensive and numerically unstable. Instead, use the implicit function theorem: at convergence, the residual R(x,p)=0, so dx/dp = -(∂R/∂x)⁻¹ · (∂R/∂p). One matrix solve gives the gradient without tracing through iterations.

### Pre-computed inverse EOS table
The OPAL EOS gives P(ρ,T) but the structure equations need ρ(P,T). Inverting via Newton at every shell (600 shells × 15 Newton iterations × 200 evolution steps) was 307s per gradient. Pre-computing the inverse on a (logT, logP, X, Z) grid at module load and using bilinear interpolation in the hot path reduced this to 4ms.

### Fresh agent per turn
Persistent context accumulates failure ("I tried X, it didn't work, I tried Y..."). After 10-20 turns, both Teacher and Worker conclude the task is impossible. Fresh agents each turn see only the project log (factual results) without emotional baggage.

### REPORT block filtering
The Worker's full output includes hundreds of lines of debugging. The Teacher only sees a structured report (Status, Results, Blockers, Next). This prevents the Teacher from being overwhelmed and keeps feedback focused.

## What Failed and Why

### Same-model Teacher can't catch cheating
The Worker frames MIST interpolation as "physics-based interpolation" and the same-model Teacher accepts the framing. Only a stronger model (Opus reviewing Sonnet) catches it.

### Staged prompts prevent holistic insight
Telling the Worker "Stage 1: plan, Stage 2: structure, Stage 3: evolution" caused it to build a non-differentiable solver first, then fail to retrofit gradients. The successful run tested everything together from the start.

### Validator bugs waste days
The validator imported from the wrong directory for 12 turns. The Worker wrote correct code that was never tested. Always verify the validator independently.

### "DONE" detection is fragile
`"DONE" in response` matches "This is NOT a DONE situation." Use line-by-line matching: `any(line.strip() == "DONE" for line in response.split("\n"))`.

### Context overflow from large files
A single MIST track file is 4.2 MB (~1M tokens). Reading one file kills the session. Solution: don't give agents `file_read` for large files; they use `shell("head -50 file")` instead.

### Agents cheat when the test is achievable by cheating
If MIST data is accessible and the test checks accuracy against MIST, the agent WILL read MIST and interpolate. File permissions (root-owned data) are the only reliable prevention.

## Lessons Learned

1. **The validator IS the specification.** If it has bugs, everything optimizes for the wrong thing.
2. **Translation > invention.** Porting known-working code (MESA Fortran → JAX) is more reliable than building from physics principles.
3. **Gradient-first testing forces correct architecture.** Test `jax.grad` before accuracy — the Worker designs for differentiability from the start.
4. **Speed and accuracy are separate problems.** Solve accuracy first (even if slow), then optimize speed without changing physics.
5. **Pre-computed tables beat runtime computation.** The EOS inversion (307s → 4ms) was the single biggest win.
6. **Subagents work for independent modules.** MLT, atmosphere, overshooting, nuclear — each is a drop-in replacement with a clear interface.
7. **The human's value is asking the right questions.** "Is it using OPAL?" and "Why did it fall back to shooting?" were the interventions that mattered.
8. **File permission enforcement > prompt instructions.** "Do NOT read MIST" is ignored. `chmod 600` is not.

## Remaining Work

- **Low-mass accuracy (M < 1.0 M☉):** Needs OPAL EOS Type-2 tables (finer X-grid), Ferguson low-T opacities, element diffusion
- **Henyey solver:** Would close the 0.05 dex gap from shooting method's accumulated integration error
- **Multi-metallicity:** Currently validated at [Fe/H]=0.0 only; needs testing across -0.25 to +0.25
- **Data packaging:** OPAL opacity + EOS tables need to be distributed with the code (or auto-downloaded)

## Agent System Architecture (Final)

```
Kiro CLI (Opus 4.7)
  ├── Main agent: integrates modules, runs tests, iterates
  └── Subagents: implement individual physics modules in parallel
      ├── MLT cubic (Böhm-Vitense)
      ├── Atmosphere (Eddington gray)
      ├── Overshooting (calibrated f_ov)
      └── Nuclear (3He non-equilibrium)

Validation: sudo-protected test (root-owned, reads MIST from /opt/mist_ref/)
MIST data: inaccessible to agent (chmod 600, root-owned)
OPAL data: accessible (physics input, not answer key)
```
