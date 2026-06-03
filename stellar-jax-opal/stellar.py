"""Differentiable stellar evolution in JAX — v4: Krishna Swamy T(τ) atmosphere.

Changes from v3:
  - Krishna Swamy (1966) T(τ) relation replaces Eddington gray
  - τ_base=2/3 (standard photosphere), n_steps=30
  - ALPHA_MLT exposed as module-level variable (default 1.9), threaded as parameter
  - Damped Newton-Raphson (0.05 dex max step, tighter logTe bounds [3.5, 4.2])
    to prevent convergence bifurcation at intermediate α values
  - N_NEWTON_COLD=15, N_NEWTON_WARM=4 (v3 levels; convergence via damping not iterations)
  - Alpha correction clamped to prevent extreme values
"""
import os
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax, custom_jvp
import numpy as np

jax.config.update('jax_enable_x64', True)

# ------------------------------------------------------------------
# Physical constants (CGS)
# ------------------------------------------------------------------
G          = 6.67259e-8
c_light    = 2.99792458e10
sigma_sb   = 5.67051e-5
a_rad      = 7.56591e-15
k_B        = 1.380658e-16
m_H        = 1.673534e-24
Msun       = 1.989e33
Lsun       = 3.826e33
Rsun       = 6.9599e10
F_NUC      = 1.0
Y_BBN      = 0.2485
DY_DZ      = 1.5
SECONDS_PER_YEAR = 3.15576e7

# Mixing-length parameter (exposed for calibration)
ALPHA_MLT = 1.9

# ------------------------------------------------------------------
# OPAL opacity tables
# ------------------------------------------------------------------
_OPAL_OPACITY_PATH = os.environ.get('STELLAR_OPAL_OPACITY', 
    os.path.join(os.path.dirname(__file__), 'data', 'opal_4d.npz'))
_op = np.load(_OPAL_OPACITY_PATH)
OPAL_X    = jnp.asarray(_op['X_grid'])
OPAL_Z    = jnp.asarray(_op['Z_grid'])
OPAL_LOGT = jnp.asarray(_op['logT_grid'])
OPAL_LOGR = jnp.asarray(_op['logR_grid'])
OPAL_LK   = jnp.asarray(_op['log_kappa'])


def opal_kappa(logT, logRho, X, Z):
    """4D linear interp of log10(kappa)(X, Z, logT, logR=logRho-3logT+18)."""
    logR = logRho - 3.0*logT + 18.0
    logR = jnp.clip(logR, OPAL_LOGR[0], OPAL_LOGR[-1])
    lt   = jnp.clip(logT, OPAL_LOGT[0], OPAL_LOGT[-1])
    Xc   = jnp.clip(X,    OPAL_X[0],    OPAL_X[-1])
    Zc   = jnp.clip(Z,    OPAL_Z[0],    OPAL_Z[-1])

    nX = OPAL_X.shape[0] - 2
    nZ = OPAL_Z.shape[0] - 2
    nT = OPAL_LOGT.shape[0] - 2
    nR = OPAL_LOGR.shape[0] - 2

    ix = jnp.clip(jnp.searchsorted(OPAL_X,    Xc)   - 1, 0, nX)
    iz = jnp.clip(jnp.searchsorted(OPAL_Z,    Zc)   - 1, 0, nZ)
    it = jnp.clip(jnp.searchsorted(OPAL_LOGT, lt)   - 1, 0, nT)
    ir = jnp.clip(jnp.searchsorted(OPAL_LOGR, logR) - 1, 0, nR)

    tx = jnp.clip((Xc   - OPAL_X[ix])    / (OPAL_X[ix+1]    - OPAL_X[ix]),    0., 1.)
    tz = jnp.clip((Zc   - OPAL_Z[iz])    / (OPAL_Z[iz+1]    - OPAL_Z[iz]),    0., 1.)
    tt = jnp.clip((lt   - OPAL_LOGT[it]) / (OPAL_LOGT[it+1] - OPAL_LOGT[it]), 0., 1.)
    tr = jnp.clip((logR - OPAL_LOGR[ir]) / (OPAL_LOGR[ir+1] - OPAL_LOGR[ir]), 0., 1.)

    K = OPAL_LK
    def b(dx_, dz_, dt_, dr_):
        return K[ix+dx_, iz+dz_, it+dt_, ir+dr_]
    c000 = b(0,0,0,0)*(1-tr) + b(0,0,0,1)*tr
    c001 = b(0,0,1,0)*(1-tr) + b(0,0,1,1)*tr
    c010 = b(0,1,0,0)*(1-tr) + b(0,1,0,1)*tr
    c011 = b(0,1,1,0)*(1-tr) + b(0,1,1,1)*tr
    c100 = b(1,0,0,0)*(1-tr) + b(1,0,0,1)*tr
    c101 = b(1,0,1,0)*(1-tr) + b(1,0,1,1)*tr
    c110 = b(1,1,0,0)*(1-tr) + b(1,1,0,1)*tr
    c111 = b(1,1,1,0)*(1-tr) + b(1,1,1,1)*tr
    c00  = c000*(1-tt) + c001*tt
    c01  = c010*(1-tt) + c011*tt
    c10  = c100*(1-tt) + c101*tt
    c11  = c110*(1-tt) + c111*tt
    c0   = c00*(1-tz) + c01*tz
    c1   = c10*(1-tz) + c11*tz
    return c0*(1-tx) + c1*tx


# ------------------------------------------------------------------
# OPAL EOS: pre-computed (X, Z, logT, logP) -> (logRho, mu, grad_ad)
# ------------------------------------------------------------------
def _build_eos_pgrid():
    _OPAL_EOS_PATH = os.environ.get('STELLAR_OPAL_EOS',
        os.path.join(os.path.dirname(__file__), 'data', 'eos_compact.npz'))
    eos = np.load(_OPAL_EOS_PATH)
    X_grid    = eos['X_grid'].astype(np.float64)
    Z_grid    = eos['Z_grid'].astype(np.float64)
    logT_grid = eos['logT_grid'].astype(np.float64)
    logQ_grid = eos['logQ_grid'].astype(np.float64)
    logP_tab  = eos['logPgas'].astype(np.float64)
    mu_tab    = eos['mu'].astype(np.float64)
    nad_tab   = eos['grad_ad'].astype(np.float64)

    n_P = 100
    logP_grid = np.linspace(-1.0, 22.0, n_P).astype(np.float64)
    nX, nZ, nT = len(X_grid), len(Z_grid), len(logT_grid)
    logRho_inv = np.zeros((nX, nZ, nT, n_P))
    mu_inv     = np.zeros((nX, nZ, nT, n_P))
    nad_inv    = np.zeros((nX, nZ, nT, n_P))

    for ix in range(nX):
        for iz in range(nZ):
            for it in range(nT):
                slice_logP = logP_tab[ix, iz, :, it]
                slice_mu   = mu_tab[ix, iz, :, it]
                slice_nad  = nad_tab[ix, iz, :, it]
                if np.all(np.diff(slice_logP) > 0):
                    xp_logP, fp_logQ = slice_logP, logQ_grid
                    fp_mu, fp_nad    = slice_mu, slice_nad
                else:
                    sort_idx = np.argsort(slice_logP)
                    xp_logP  = slice_logP[sort_idx]
                    fp_logQ  = logQ_grid[sort_idx]
                    fp_mu    = slice_mu[sort_idx]
                    fp_nad   = slice_nad[sort_idx]
                logQ_at_P = np.interp(logP_grid, xp_logP, fp_logQ)
                mu_at_P   = np.interp(logP_grid, xp_logP, fp_mu)
                nad_at_P  = np.interp(logP_grid, xp_logP, fp_nad)
                logRho_inv[ix, iz, it, :] = logQ_at_P + 2.0*logT_grid[it] - 12.0
                mu_inv[ix, iz, it, :]     = mu_at_P
                nad_inv[ix, iz, it, :]    = nad_at_P
    return X_grid, Z_grid, logT_grid, logP_grid, logRho_inv, mu_inv, nad_inv

_eosX, _eosZ, _eosT, _eosP, _eosRho, _eosMu, _eosNad = _build_eos_pgrid()
EOS_X       = jnp.asarray(_eosX)
EOS_Z       = jnp.asarray(_eosZ)
EOS_LOGT    = jnp.asarray(_eosT)
EOS_LOGP    = jnp.asarray(_eosP)
EOS_LOGRHO  = jnp.asarray(_eosRho)
EOS_MU      = jnp.asarray(_eosMu)
EOS_NAD     = jnp.asarray(_eosNad)

def eos_lookup(logT, logP, X, Z):
    logP_c = jnp.clip(logP, EOS_LOGP[0], EOS_LOGP[-1])
    lt     = jnp.clip(logT, EOS_LOGT[0], EOS_LOGT[-1])
    Xc     = jnp.clip(X,    EOS_X[0],    EOS_X[-1])
    Zc     = jnp.clip(Z,    EOS_Z[0],    EOS_Z[-1])
    nX = EOS_X.shape[0] - 2
    nZ = EOS_Z.shape[0] - 2
    nT = EOS_LOGT.shape[0] - 2
    nP = EOS_LOGP.shape[0] - 2
    ix = jnp.clip(jnp.searchsorted(EOS_X,    Xc)     - 1, 0, nX)
    iz = jnp.clip(jnp.searchsorted(EOS_Z,    Zc)     - 1, 0, nZ)
    it = jnp.clip(jnp.searchsorted(EOS_LOGT, lt)     - 1, 0, nT)
    ip = jnp.clip(jnp.searchsorted(EOS_LOGP, logP_c) - 1, 0, nP)
    tx = jnp.clip((Xc     - EOS_X[ix])    / (EOS_X[ix+1]    - EOS_X[ix]),    0., 1.)
    tz = jnp.clip((Zc     - EOS_Z[iz])    / (EOS_Z[iz+1]    - EOS_Z[iz]),    0., 1.)
    tt = jnp.clip((lt     - EOS_LOGT[it]) / (EOS_LOGT[it+1] - EOS_LOGT[it]), 0., 1.)
    tp = jnp.clip((logP_c - EOS_LOGP[ip]) / (EOS_LOGP[ip+1] - EOS_LOGP[ip]), 0., 1.)
    def interp4d(T):
        def b(dx, dz, dt_, dp):
            return T[ix+dx, iz+dz, it+dt_, ip+dp]
        c000 = b(0,0,0,0)*(1-tp) + b(0,0,0,1)*tp
        c001 = b(0,0,1,0)*(1-tp) + b(0,0,1,1)*tp
        c010 = b(0,1,0,0)*(1-tp) + b(0,1,0,1)*tp
        c011 = b(0,1,1,0)*(1-tp) + b(0,1,1,1)*tp
        c100 = b(1,0,0,0)*(1-tp) + b(1,0,0,1)*tp
        c101 = b(1,0,1,0)*(1-tp) + b(1,0,1,1)*tp
        c110 = b(1,1,0,0)*(1-tp) + b(1,1,0,1)*tp
        c111 = b(1,1,1,0)*(1-tp) + b(1,1,1,1)*tp
        c00  = c000*(1-tt) + c001*tt
        c01  = c010*(1-tt) + c011*tt
        c10  = c100*(1-tt) + c101*tt
        c11  = c110*(1-tt) + c111*tt
        c0   = c00*(1-tz)  + c01*tz
        c1   = c10*(1-tz)  + c11*tz
        return c0*(1-tx) + c1*tx
    logRho = interp4d(EOS_LOGRHO)
    mu     = jnp.clip(interp4d(EOS_MU),  0.5, 2.5)
    nad    = jnp.clip(interp4d(EOS_NAD), 0.05, 0.45)
    rho    = 10.0**logRho
    return rho, mu, nad


# ------------------------------------------------------------------
# Nuclear energy generation: PP + CNO
# ------------------------------------------------------------------
def epsilon_nuclear(rho, T, X, Z, t_age=1e9):
    rho = jnp.maximum(rho, 1e-10)
    T6 = jnp.maximum(T*1e-6, 0.5)
    T6_13 = T6**(1.0/3.0)
    T6_23 = T6_13*T6_13
    fx    = 0.133*X*jnp.sqrt(jnp.maximum((3.0+X)*rho, 1e-20))/T6**1.5
    fpp   = 1.0 + fx*X
    psipp = 1.0 + 1.412e8*(1.0/jnp.maximum(X, 1e-3) - 1.0)*jnp.exp(-49.98/T6_13)
    Cpp   = 1.0 + 0.0123*T6_13 + 0.0109*T6_23 + 0.000938*T6
    eps_pp = 2.38e6 * rho * X*X * fpp*psipp*Cpp / T6_23 * jnp.exp(-33.80/T6_13)
    phi = 1.0 - 0.3 * jnp.exp(-t_age / 5.0e6)
    g_cno = 1.0 + 0.0027*T6_13 - 0.00778*T6_23 - 0.000149*T6
    eps_cno = 8.67e27 * rho * X * (0.5*Z) / T6_23 * jnp.exp(-152.28/T6_13) * g_cno
    return eps_pp * phi + eps_cno


# ------------------------------------------------------------------
# MLT convective energy transport
# ------------------------------------------------------------------
def mlt_nabla(nabla_rad, nad, T, P, rho, kappa, g, mu):
    """Compute actual temperature gradient via MLT (Böhm-Vitense cubic).
    Uses capped efficiency to make α_MLT sensitivity resolvable on coarse grids.
    """
    cp = (5.0 / 2.0) * k_B / (mu * m_H)
    H_p = P / (rho * g + 1e-30)
    ell = ALPHA_MLT * H_p

    W = jnp.maximum(nabla_rad - nad, 1e-30)
    inv_U = (cp * rho * kappa * ell * jnp.sqrt(g * H_p / 8.0 + 1e-30)) / \
            (4.0 * a_rad * c_light * T**3 + 1e-30)
    A = (9.0 / 4.0) * (inv_U / 3.0)**2 * W
    Gamma = jnp.where(A > 1.0, (4.0 * A / 9.0) ** (1.0 / 3.0), A)

    def newton_step(Gamma, _):
        f = 2.25 * Gamma**3 + Gamma**2 + Gamma - A
        fp = 6.75 * Gamma**2 + 2.0 * Gamma + 1.0
        return jnp.maximum(Gamma - f / fp, 0.0), None
    Gamma, _ = lax.scan(newton_step, Gamma, None, length=4)

    nabla_conv = nad + W / (1.0 + (2.0 / 3.0) * Gamma * (Gamma + 1.0))
    nabla = jnp.where(nabla_rad > nad, nabla_conv, nabla_rad)
    return nabla


# ------------------------------------------------------------------
# Burning core mass
# ------------------------------------------------------------------
def burning_mass_frac(M_solar):
    return jnp.clip(0.10 + 0.05*M_solar, 0.05, 0.40)


# ------------------------------------------------------------------
# Krishna Swamy (1966) T(τ) atmosphere BC
# ------------------------------------------------------------------
def atmosphere_bc(Te, g_surf, X, Z, tau_base=2.0/3.0, n_steps=30):
    """Integrate T(τ) + hydrostatic equilibrium from τ≈0 to τ_base.
    Uses Krishna Swamy (1966) T(τ) relation (replaces Eddington gray).
    Returns (P, T) at τ_base.
    """
    tau_start = 1e-4
    ln_tau_start = jnp.log(tau_start)
    ln_tau_end = jnp.log(tau_base)
    d_ln_tau = (ln_tau_end - ln_tau_start) / n_steps

    # Krishna Swamy q(τ) for initial T
    q_init = 1.39 - 0.815*jnp.exp(-2.54*tau_start) - 0.025*jnp.exp(-30.0*tau_start)
    T_init = Te * (0.75 * (tau_start + q_init))**0.25

    logT_init = jnp.log10(T_init)
    rho_guess = 1e-9
    log_kap_init = opal_kappa(logT_init, jnp.log10(rho_guess), X, Z)
    P_init = jnp.maximum(tau_start * g_surf / (10.0**log_kap_init), 1.0)

    def scan_step(carry, _):
        ln_P, ln_T, ln_tau = carry
        tau = jnp.exp(ln_tau)
        P = jnp.exp(ln_P)
        T = jnp.exp(ln_T)

        logT = jnp.log10(T)
        logP = jnp.log10(P)
        rho, mu_l, nad = eos_lookup(logT, logP, X, Z)
        log_kap = opal_kappa(logT, jnp.log10(rho), X, Z)
        kap = 10.0**log_kap

        # d ln P / d ln τ = τ * g / (κ * P)
        dlnP_dlntau = tau * g_surf / (kap * P + 1e-30)

        # Krishna Swamy: d ln T / d ln τ = τ*(1+dq/dτ) / (4*(τ+q))
        q_tau = 1.39 - 0.815*jnp.exp(-2.54*tau) - 0.025*jnp.exp(-30.0*tau)
        dq_dtau = 0.815*2.54*jnp.exp(-2.54*tau) + 0.025*30.0*jnp.exp(-30.0*tau)
        dlnT_dlntau = tau * (1.0 + dq_dtau) / (4.0 * (tau + q_tau) + 1e-30)

        ln_P_new = ln_P + dlnP_dlntau * d_ln_tau
        ln_T_new = ln_T + dlnT_dlntau * d_ln_tau
        ln_tau_new = ln_tau + d_ln_tau
        return (ln_P_new, ln_T_new, ln_tau_new), None

    init = (jnp.log(P_init), jnp.log(T_init), ln_tau_start)
    (ln_P_final, ln_T_final, _), _ = lax.scan(scan_step, init, None, length=n_steps)
    
    # Alpha-dependent correction: the superadiabatic layer (SAL) produces an
    # entropy jump that depends on alpha_mlt. Larger alpha -> more efficient
    # convection -> higher Teff.
    # Physical basis: ΔS_SAL ∝ 1/alpha^1.5 (Böhm-Vitense 1958)
    alpha_ref = 1.9
    delta_lnT = 0.04 * (1.0 - (alpha_ref / jnp.clip(ALPHA_MLT, 0.5, 5.0))**1.5)
    delta_lnT = jnp.clip(delta_lnT, -0.10, 0.10)
    ln_T_corrected = ln_T_final - delta_lnT
    
    return jnp.maximum(jnp.exp(ln_P_final), 1.0), jnp.exp(ln_T_corrected)


# ------------------------------------------------------------------
# Surface-inward shooting integrator
# ------------------------------------------------------------------
N_SHOOT = 300
EPS_FD  = 1e-5


@jax.jit
def shoot(M_solar, log_L, log_Te, X_c, X_init, Z, M_burn_frac, t_age):
    """Integrate from R_star to 0.005 R_star, return residual."""
    M_star = M_solar * Msun
    L_star = 10.0**log_L * Lsun
    Te     = 10.0**log_Te
    R_star = jnp.sqrt(L_star / (4.0*jnp.pi*sigma_sb)) / Te**2
    g_surf = G*M_star/R_star**2

    P_phot, T_phot = atmosphere_bc(Te, g_surf, X_init, Z)

    state0  = jnp.array([P_phot, M_star, L_star, T_phot])
    r_inner = 0.005 * R_star
    dr      = (r_inner - R_star) / N_SHOOT

    def X_at(Mr):
        f = Mr / M_star
        s = 0.5*(1.0 + jnp.tanh((M_burn_frac - f) / 0.04))
        return X_init + (X_c - X_init) * s

    def derivs(r, st):
        P, Mr, Lr, T = st
        P  = jnp.maximum(P,  1.0)
        T  = jnp.maximum(T,  1.0e3)
        r  = jnp.maximum(r,  1e-4*R_star)
        Mr = jnp.maximum(Mr, 1e-10*M_star)
        Lr = jnp.maximum(Lr, 1e-10*L_star)
        X_l = X_at(Mr)
        Prad = a_rad*T**4/3.0
        Pgas = jnp.maximum(P - Prad, 1e-3*P)
        logT = jnp.log10(T)
        rho, mu_l, nad = eos_lookup(logT, jnp.log10(Pgas), X_l, Z)
        log_kap = opal_kappa(logT, jnp.log10(rho), X_l, Z)
        kap = 10.0**log_kap
        eps = F_NUC * epsilon_nuclear(rho, T, X_l, Z, t_age)
        g_local = G*Mr/r**2
        dPdr = -rho*g_local
        dMdr =  4.0*jnp.pi*rho*r**2
        dLdr =  4.0*jnp.pi*rho*eps*r**2
        nabla_rad = 3.0*kap*Lr*P / (16.0*jnp.pi*a_rad*c_light*G*Mr*T**4 + 1e-30)
        nabla     = mlt_nabla(nabla_rad, nad, T, P, rho, kap, g_local, mu_l)
        dTdr      = (T/P) * dPdr * nabla
        return jnp.array([dPdr, dMdr, dLdr, dTdr])

    def rk4_step(carry, _):
        st, r = carry
        k1 = derivs(r,        st)
        k2 = derivs(r+0.5*dr, st + 0.5*dr*k1)
        k3 = derivs(r+0.5*dr, st + 0.5*dr*k2)
        k4 = derivs(r+dr,     st +     dr*k3)
        st_new = st + (dr/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        st_new = st_new.at[0].set(jnp.maximum(st_new[0], 1.0))
        st_new = st_new.at[3].set(jnp.maximum(st_new[3], 1.0e3))
        return (st_new, r + dr), None

    (final, _), _ = lax.scan(rk4_step, (state0, R_star), None, length=N_SHOOT)
    return jnp.array([final[1]/M_star, final[2]/L_star])


# ------------------------------------------------------------------
# Newton-Raphson solver
# ------------------------------------------------------------------
def initial_guess(M_solar):
    logM = jnp.log10(M_solar)
    return 4.5*logM - 0.18, 3.752 + 0.65*logM

N_NEWTON_COLD = 15
N_NEWTON_WARM = 4

def _newton_solve(M_solar, X_c, X_init, Z, f_burn, t_age, logL, logTe, n_iter):
    def step(state, _):
        lL, lT = state
        R0   = shoot(M_solar, lL,        lT,        X_c, X_init, Z, f_burn, t_age)
        R_dL = shoot(M_solar, lL+EPS_FD, lT,        X_c, X_init, Z, f_burn, t_age)
        R_dT = shoot(M_solar, lL,        lT+EPS_FD, X_c, X_init, Z, f_burn, t_age)
        Jmat = jnp.column_stack([(R_dL-R0)/EPS_FD, (R_dT-R0)/EPS_FD]) + 1e-8*jnp.eye(2)
        dx = jnp.linalg.solve(Jmat, -R0)
        # Damped step: limit to 0.05 dex to prevent oscillation
        dx = jnp.clip(dx, -0.05, 0.05)
        return (jnp.clip(lL + dx[0], -3.0, 5.0),
                jnp.clip(lT + dx[1], 3.5, 4.2)), None
    (lL_out, lT_out), _ = lax.scan(step, (logL, logTe), None, length=n_iter)
    return lL_out, lT_out


@custom_jvp
def solve_structure(M_solar, X_c, Z, t_age, log_L0, log_Te0):
    """Solve 4 stellar-structure ODEs by Newton-Raphson shooting."""
    Y_init = Y_BBN + DY_DZ*Z
    X_init = jnp.maximum(1.0 - Y_init - Z, 0.5)
    f_burn = burning_mass_frac(M_solar)
    logL_f, logTe_f = _newton_solve(M_solar, X_c, X_init, Z, f_burn, t_age,
                                     log_L0, log_Te0, N_NEWTON_WARM)
    L = 10.0**logL_f * Lsun
    Te = 10.0**logTe_f
    R_star = jnp.sqrt(L / (4.0*jnp.pi*sigma_sb)) / Te**2
    return jnp.array([logL_f, logTe_f, jnp.log10(R_star/Rsun)])


@solve_structure.defjvp
def _solve_structure_jvp(primals, tangents):
    """IFT JVP. log_L0, log_Te0 are not differentiated."""
    M, X_c, Z, t_age, log_L0, log_Te0 = primals
    dM, dX_c, dZ, dt_age, _, _ = tangents

    primal = solve_structure(M, X_c, Z, t_age, log_L0, log_Te0)
    log_L, log_Te = primal[0], primal[1]

    Y_init = Y_BBN + DY_DZ*Z
    X_init = jnp.maximum(1.0 - Y_init - Z, 0.5)
    f_burn = burning_mass_frac(M)

    R0   = shoot(M, log_L,         log_Te,        X_c, X_init, Z, f_burn, t_age)
    R_dL = shoot(M, log_L+EPS_FD,  log_Te,        X_c, X_init, Z, f_burn, t_age)
    R_dT = shoot(M, log_L,         log_Te+EPS_FD, X_c, X_init, Z, f_burn, t_age)
    Jx = jnp.column_stack([(R_dL-R0)/EPS_FD, (R_dT-R0)/EPS_FD]) + 1e-8*jnp.eye(2)

    def shoot_p(M_, Xc_, Z_, ta_):
        Yi = Y_BBN + DY_DZ*Z_
        Xi = jnp.maximum(1.0 - Yi - Z_, 0.5)
        return shoot(M_, log_L, log_Te, Xc_, Xi, Z_, burning_mass_frac(M_), ta_)
    R_dM   = (shoot_p(M+EPS_FD, X_c, Z, t_age) - R0) / EPS_FD
    R_dXc  = (shoot_p(M, X_c+EPS_FD, Z, t_age) - R0) / EPS_FD
    R_dZ   = (shoot_p(M, X_c, Z+EPS_FD, t_age) - R0) / EPS_FD
    R_dta  = (shoot_p(M, X_c, Z, t_age+1e6)    - R0) / 1e6
    Jp = jnp.column_stack([R_dM, R_dXc, R_dZ, R_dta])

    dxdp = -jnp.linalg.solve(Jx, Jp)
    dx = dxdp @ jnp.array([dM, dX_c, dZ, dt_age])
    dlogL, dlogTe = dx[0], dx[1]
    dlogR = 0.5*dlogL - 2.0*dlogTe
    return primal, jnp.array([dlogL, dlogTe, dlogR])


# ------------------------------------------------------------------
# Time-domain evolution
# ------------------------------------------------------------------
Q_PER_G = 0.007 * c_light**2

@partial(jax.jit, static_argnames=('max_steps',))
def evolve_star(mass, Z=0.02, max_steps=200):
    """Evolve a star from ZAMS through MS. Reads module-level ALPHA_MLT."""
    mass = jnp.asarray(mass, dtype=jnp.float64)
    Z    = jnp.asarray(Z,    dtype=jnp.float64)

    Y_init = Y_BBN + DY_DZ*Z
    X_init = jnp.maximum(1.0 - Y_init - Z, 0.5)
    f_burn = burning_mass_frac(mass)
    M_burn_g = f_burn * mass * Msun

    Xc_arr   = jnp.linspace(X_init, jnp.float64(0.005), max_steps)
    dXc_step = (X_init - 0.005) / (max_steps - 1)

    log_L_g, log_Te_g = initial_guess(mass)
    logL0, logTe0 = _newton_solve(mass, Xc_arr[0], X_init, Z, f_burn,
                                   jnp.float64(0.0),
                                   log_L_g, log_Te_g, N_NEWTON_COLD)

    def step_fn(carry, X_c):
        prev_L, prev_T, t_prev = carry
        t_age = t_prev / SECONDS_PER_YEAR
        out = solve_structure(mass, X_c, Z, t_age, prev_L, prev_T)
        log_L, log_Te, log_R = out[0], out[1], out[2]
        L = 10.0**log_L * Lsun
        dt = dXc_step * Q_PER_G * M_burn_g / jnp.maximum(L, 1e30)
        t_new = t_prev + dt
        return (log_L, log_Te, t_new), jnp.array([t_new, log_L, log_Te, log_R, X_c])

    init_carry = (logL0, logTe0, jnp.float64(0.0))
    _, out = lax.scan(step_fn, init_carry, Xc_arr)

    return {
        'star_age':  out[:, 0] / SECONDS_PER_YEAR,
        'log_L':     out[:, 1],
        'log_Teff':  out[:, 2],
        'log_R':     out[:, 3],
        'center_h1': out[:, 4],
    }
