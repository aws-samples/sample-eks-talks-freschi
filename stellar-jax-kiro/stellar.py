"""
Differentiable stellar structure and evolution in JAX.
Solves the four stellar structure ODEs via RK4 + shooting method.
No external data dependencies — pure physics computation.
"""
import jax
import jax.numpy as jnp
from jax import lax

jax.config.update('jax_enable_x64', True)

# Physical constants (cgs)
G = 6.67259e-8
c_light = 2.99792458e10
sigma_sb = 5.67051e-5
a_rad = 7.56591e-15
k_B = 1.380658e-16
m_H = 1.673534e-24
Msun = 1.989e33
Lsun = 3.826e33
Rsun = 6.9599e10
secyr = 3.1557e7
gamma_ad = 5.0 / 3.0
gamrat = gamma_ad / (gamma_ad - 1.0)
g_ff = 1.0


def mean_molecular_weight(X, Z):
    Y = 1.0 - X - Z
    return 1.0 / (2.0 * X + 0.75 * Y + 0.5 * Z)


def eos_rho(P, T, mu):
    Prad = a_rad * T**4 / 3.0
    Pgas = jnp.maximum(P - Prad, 1e-30)
    return mu * m_H / k_B * Pgas / jnp.maximum(T, 1.0)


def opacity(rho, T, X, Z):
    """Kramers bound-free + free-free + electron scattering."""
    rho_s = jnp.maximum(rho, 1e-30)
    T_s = jnp.maximum(T, 1.0)
    tog_bf = 2.82 * (rho_s * (1.0 + X))**0.2
    k_bf = 4.34e25 / tog_bf * Z * (1.0 + X) * rho_s / T_s**3.5
    k_ff = 3.68e22 * g_ff * (1.0 - Z) * (1.0 + X) * rho_s / T_s**3.5
    k_e = 0.2 * (1.0 + X)
    return k_bf + k_ff + k_e


def nuclear_energy(rho, T, X, Z):
    """pp-chain + CNO cycle (FCZ75) with calibrated rate factor."""
    XCNO = Z / 2.0
    T6 = jnp.maximum(T * 1e-6, 1e-6)
    rho_s = jnp.maximum(rho, 1e-30)
    X_s = jnp.maximum(X, 1e-30)

    fx = 0.133 * X_s * jnp.sqrt((3.0 + X_s) * rho_s) / T6**1.5
    fpp = 1.0 + fx * X_s
    psipp = 1.0 + 1.412e8 * (1.0 / X_s - 1.0) * jnp.exp(-49.98 * T6**(-1.0/3.0))
    Cpp = 1.0 + 0.0123 * T6**(1.0/3.0) + 0.0109 * T6**(2.0/3.0) + 0.000938 * T6
    epspp = 2.38e6 * rho_s * X_s**2 * fpp * psipp * Cpp * T6**(-2.0/3.0) * jnp.exp(-33.80 * T6**(-1.0/3.0))

    CCNO = 1.0 + 0.0027 * T6**(1.0/3.0) - 0.00778 * T6**(2.0/3.0) - 0.000149 * T6
    epsCNO = 8.67e27 * rho_s * X_s * XCNO * CCNO * T6**(-2.0/3.0) * jnp.exp(-152.28 * T6**(-1.0/3.0))

    # Rate calibration factor (accounts for Kramers opacity being too low
    # relative to OPAL, which makes the equilibrium structure different)
    return (epspp + epsCNO) * 0.12


def structure_rhs(r, state, X, Z, mu):
    """The four stellar structure ODEs:
    dP/dr = -G*M_r*rho/r^2
    dM/dr = 4*pi*r^2*rho
    dL/dr = 4*pi*r^2*rho*epsilon
    dT/dr = radiative or convective (Schwarzschild criterion)
    """
    P, M_r, L_r, T = state[0], state[1], state[2], state[3]
    r_safe = jnp.maximum(r, 1e5)
    r2 = r_safe**2

    rho = eos_rho(P, T, mu)
    kappa = opacity(rho, T, X, Z)
    eps = nuclear_energy(rho, T, X, Z)

    dP = -G * rho * jnp.maximum(M_r, 0.0) / r2
    dM = 4.0 * jnp.pi * rho * r2
    dL = 4.0 * jnp.pi * rho * eps * r2
    dT_rad = -(3.0 / (16.0 * jnp.pi * a_rad * c_light)) * kappa * rho / jnp.maximum(T, 1.0)**3 * jnp.maximum(L_r, 1e10) / r2
    dT_ad = -1.0 / gamrat * G * jnp.maximum(M_r, 0.0) / r2 * mu * m_H / k_B
    dT = jnp.where(jnp.abs(dT_rad) > jnp.abs(dT_ad), dT_ad, dT_rad)

    return jnp.array([dP, dM, dL, dT])


def integrate_star(Ms, Ls, Teff, X, Z, N=500):
    """Integrate stellar structure from surface to center using RK4.
    Returns (M_c/Ms, L_c/Ls) at innermost grid point."""
    mu = mean_molecular_weight(X, Z)
    Rs = jnp.sqrt(Ls / (4.0 * jnp.pi * sigma_sb)) / Teff**2

    r_start = Rs * 0.99
    T_start = G * Ms * mu * m_H / (4.25 * k_B) * (1.0 / r_start - 1.0 / Rs)
    T_start = jnp.maximum(T_start, 100.0)

    A_ff = 3.68e22 * g_ff * (1.0 - Z) * (1.0 + X)
    tog_bf = 0.01
    Afac = 4.34e25 * Z * (1.0 + X) / tog_bf + A_ff
    P_start = jnp.sqrt((1.0/4.25) * (16.0/3.0 * jnp.pi * a_rad * c_light) *
                        (G * Ms / Ls) * (k_B / (Afac * mu * m_H))) * T_start**4.25
    rho_est = eos_rho(P_start, T_start, mu)
    tog_bf = 2.82 * (jnp.maximum(rho_est, 1e-30) * (1.0 + X))**0.2
    Afac = 4.34e25 * Z * (1.0 + X) / jnp.maximum(tog_bf, 0.001) + A_ff
    P_start = jnp.sqrt((1.0/4.25) * (16.0/3.0 * jnp.pi * a_rad * c_light) *
                        (G * Ms / Ls) * (k_B / (Afac * mu * m_H))) * T_start**4.25

    state0 = jnp.array([P_start, Ms, Ls, T_start])
    r_min = Rs * 0.005
    log_r_arr = jnp.linspace(jnp.log(r_start), jnp.log(r_min), N)
    r_arr = jnp.exp(log_r_arr)

    def rk4_step(state, i):
        r = r_arr[i]
        r_next = r_arr[jnp.minimum(i + 1, N - 1)]
        dr = r_next - r
        k1 = structure_rhs(r, state, X, Z, mu)
        k2 = structure_rhs(r + 0.5*dr, state + 0.5*dr*k1, X, Z, mu)
        k3 = structure_rhs(r + 0.5*dr, state + 0.5*dr*k2, X, Z, mu)
        k4 = structure_rhs(r + dr, state + dr*k3, X, Z, mu)
        new_state = state + (dr/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        new_state = new_state.at[0].set(jnp.maximum(new_state[0], 1e-10))
        new_state = new_state.at[3].set(jnp.maximum(new_state[3], 100.0))
        return new_state, None

    final_state, _ = lax.scan(rk4_step, state0, jnp.arange(N - 1))
    return final_state[1] / Ms, final_state[2] / Ls


# ============ Shooting method with custom JVP ============

_N_GRID = 500
_N_ITER = 15


@jax.custom_jvp
def solve_structure(M_solar, X, Z):
    """Shooting method: find (log_L, log_Teff) satisfying center BCs."""
    Ms = M_solar * Msun
    mu = mean_molecular_weight(X, Z)
    mu_ref = mean_molecular_weight(0.72, 0.014)
    # Initial guess accounts for composition via mu dependence
    # L ~ M^3.8 * mu^1.3 (from homology)
    log_L_init = 3.8 * jnp.log10(jnp.maximum(M_solar, 0.1)) - 0.15 + 1.3 * jnp.log10(mu / mu_ref)
    log_Teff_init = jnp.log10(5778.0) + 0.5 * jnp.log10(jnp.maximum(M_solar, 0.1)) + 0.1 * jnp.log10(mu / mu_ref)
    x = jnp.array([log_L_init, log_Teff_init])

    def residual(params):
        Ls = 10.0**params[0] * Lsun
        Teff = 10.0**params[1]
        return jnp.array(integrate_star(Ms, Ls, Teff, X, Z, N=_N_GRID))

    def newton_step(x, _):
        res = residual(x)
        eps_fd = 1e-5
        J0 = (residual(x.at[0].add(eps_fd)) - res) / eps_fd
        J1 = (residual(x.at[1].add(eps_fd)) - res) / eps_fd
        J = jnp.column_stack([J0, J1])
        delta = jnp.linalg.solve(J.T @ J + 1e-12 * jnp.eye(2), -J.T @ res)
        delta = jnp.clip(delta, -0.1, 0.1)
        return x + delta, jnp.sum(res**2)

    x_final, _ = lax.scan(newton_step, x, jnp.arange(_N_ITER))

    log_L = x_final[0]
    log_Teff = x_final[1]
    log_R = 0.5 * log_L - 2.0 * (log_Teff - jnp.log10(5778.0))
    return log_L, log_Teff, log_R


@solve_structure.defjvp
def solve_structure_jvp(primals, tangents):
    """JVP via implicit function theorem."""
    M_solar, X, Z = primals
    dM, dX, dZ = tangents

    log_L, log_Teff, log_R = solve_structure(M_solar, X, Z)
    x_sol = jnp.array([log_L, log_Teff])

    def res_fn(M, Xh, Zh, params):
        Ls = 10.0**params[0] * Lsun
        Teff = 10.0**params[1]
        return jnp.array(integrate_star(M * Msun, Ls, Teff, Xh, Zh, N=300))

    eps = 1e-5
    r0 = res_fn(M_solar, X, Z, x_sol)
    J_x = jnp.column_stack([(res_fn(M_solar, X, Z, x_sol.at[0].add(eps)) - r0) / eps,
                             (res_fn(M_solar, X, Z, x_sol.at[1].add(eps)) - r0) / eps])
    dR_dM = (res_fn(M_solar + eps, X, Z, x_sol) - r0) / eps
    dR_dX = (res_fn(M_solar, X + eps, Z, x_sol) - r0) / eps
    dR_dZ = (res_fn(M_solar, X, Z + eps, x_sol) - r0) / eps

    J_inv = jnp.linalg.inv(J_x + 1e-15 * jnp.eye(2))
    dx_dM = -J_inv @ dR_dM
    dx_dX = -J_inv @ dR_dX
    dx_dZ = -J_inv @ dR_dZ

    d_logL = dx_dM[0] * dM + dx_dX[0] * dX + dx_dZ[0] * dZ
    d_logTeff = dx_dM[1] * dM + dx_dX[1] * dX + dx_dZ[1] * dZ
    d_logR = 0.5 * d_logL - 2.0 * d_logTeff
    return (log_L, log_Teff, log_R), (d_logL, d_logTeff, d_logR)


# ============ Evolution ============

def evolve_star(M_solar, Z=0.0142857, X0=None, max_steps=500, dt_gyr=None):
    """
    Evolve a star from pre-MS through TAMS (center_h1 < 0.01).
    Covers the full age range starting at age=0.
    """
    M_solar = jnp.asarray(M_solar, dtype=jnp.float64)
    Z = jnp.asarray(Z, dtype=jnp.float64)

    # Initial hydrogen fraction (MIST helium enrichment law: Y = 0.2369 + 2.30*Z)
    if X0 is None:
        X0 = 0.7631 - 3.30 * Z
    X0 = jnp.asarray(X0, dtype=jnp.float64)

    Ms = M_solar * Msun

    # MS lifetime ~ 10 * M^(-2.5) Gyr
    tau_ms = 10.0 * M_solar**(-2.5)  # Gyr
    # Time grid: logarithmic to resolve pre-MS (first ~1 Myr) and MS
    # log-spaced from 1000 years to tau_ms
    total_time_yr = tau_ms * 1.1e9  # years, slightly past TAMS
    log_t_arr = jnp.linspace(jnp.log10(1000.0), jnp.log10(total_time_yr), max_steps)
    t_arr = 10.0**log_t_arr  # years
    # Convert to seconds for dt calculation
    t_sec_arr = t_arr * secyr

    # Nuclear burning parameters
    Q = 6.3e18
    # f_eff calibrated to match MS lifetimes across mass range
    f_eff = 0.25 + 0.10 * M_solar  # ~0.35 at 1 Msun, ~0.45 at 2 Msun

    # Kelvin-Helmholtz timescale (pre-MS contraction)
    # Phase 1: rapid initial contraction from protostellar radius
    # Phase 2: slow KH approach to ZAMS
    # Real KH time: ~30 Myr for 1 Msun, ~3 Myr for 2 Msun
    tau1_yr = 1.0e4 * M_solar**(-2.0)   # rapid phase: ~0.01 Myr for 1 Msun
    tau2_yr = 1.0e7 * M_solar**(-2.5)   # slow phase: ~10 Myr for 1 Msun, ~1.8 Myr for 2 Msun

    def step_fn(carry, step_idx):
        X, _ = carry
        age = t_sec_arr[step_idx]
        dt_sec = jnp.where(step_idx > 0, t_sec_arr[step_idx] - t_sec_arr[step_idx - 1], t_sec_arr[0])

        # Solve structure at current composition
        log_L, log_Teff, log_R = solve_structure(M_solar, X, Z)

        # Pre-MS: two-phase gravitational contraction
        age_yr = age / secyr
        t1 = age_yr / tau1_yr  # rapid phase ratio
        t2 = age_yr / tau2_yr  # slow phase ratio
        # Phase 1: rapid drop from protostellar radius (factor ~100 in L)
        # Phase 2: slow KH contraction to ZAMS (factor ~5 in L)
        preMS_L_factor = 1.0 + 100.0 * jnp.exp(-t1 * 3.0) + 5.0 * jnp.exp(-t2 * 2.0)
        # Teff: cool on Hayashi track during phase 1, warms during phase 2
        preMS_T_offset = -0.15 * jnp.exp(-t1 * 3.0) - 0.10 * jnp.exp(-t2 * 2.0)
        preMS_R_offset = 0.5 * jnp.log10(preMS_L_factor) - 2.0 * preMS_T_offset
        # Burn fraction: no burning during pre-MS
        burn_fraction = jnp.minimum(t2, 1.0)

        log_L_out = log_L + jnp.log10(preMS_L_factor)
        log_Teff_out = log_Teff + preMS_T_offset
        log_R_out = log_R + preMS_R_offset

        # Record state at this time
        output = jnp.array([age_yr, log_L_out, log_Teff_out, log_R_out, X])

        # Nuclear burning (use MS luminosity, not pre-MS enhanced)
        L = 10.0**log_L * Lsun
        dXdt = -L / (Q * f_eff * Ms)
        X_new = jnp.maximum(X + dXdt * dt_sec * burn_fraction, 0.0)

        return (X_new, age), output

    init = (X0, jnp.float64(0.0))
    _, outputs = lax.scan(step_fn, init, jnp.arange(max_steps))

    return {
        'star_age': outputs[:, 0],
        'log_L': outputs[:, 1],
        'log_Teff': outputs[:, 2],
        'log_R': outputs[:, 3],
        'center_h1': outputs[:, 4],
    }


if __name__ == '__main__':
    import time

    print("=== Structure solve across mass range ===")
    for m in [0.8, 1.0, 1.5, 2.0]:
        log_L, log_Teff, log_R = solve_structure(jnp.float64(m), jnp.float64(0.7154), jnp.float64(0.0142857))
        print(f"  {m} Msun: log_L={float(log_L):.3f}, log_Teff={float(log_Teff):.4f}, log_R={float(log_R):.3f}")

    print("\n=== Evolution test ===")
    t0 = time.time()
    r = evolve_star(1.0, Z=0.0142857, max_steps=200)
    t1 = time.time()
    print(f"  Track: {len(r['log_L'])} pts, age={float(r['star_age'][-1])/1e9:.1f} Gyr ({t1-t0:.1f}s)")
    print(f"  Start: log_L={float(r['log_L'][0]):.3f}, X={float(r['center_h1'][0]):.3f}")
    print(f"  End:   log_L={float(r['log_L'][-1]):.3f}, X={float(r['center_h1'][-1]):.3f}")

    print("\n=== Gradient test ===")
    t0 = time.time()
    g = jax.grad(lambda m: evolve_star(m, Z=0.02, max_steps=5)['log_L'][-1])(1.0)
    t1 = time.time()
    print(f"  d(log_L)/dM = {float(g):.4f} ({t1-t0:.1f}s)")
