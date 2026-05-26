"""Differentiable stellar evolution in JAX with full OPAL EOS.

Single file. Real physics from scratch, no MIST data ever read:
  - 4 stellar structure ODEs (dP/dr, dM/dr, dL/dr, dT/dr)
  - OPAL Type-1 4D opacity tables
  - OPAL EOS tables, INVERTED ONCE AT MODULE LOAD onto a (X, Z, logT, logP)
    grid -> bilinear interp in the hot path (no Newton inside derivs)
  - Photospheric Eddington-gray boundary at tau=2/3
  - Schwarzschild + adiabatic-from-table convection (mu and grad_ad both physical)
  - Stratified composition profile X(M_r/M)
  - PP + CNO nuclear with 3He non-equilibrium
  - Warm-started solve_structure: previous step's converged (logL, logTe)
    seeds Newton -> only ~4 iterations needed per evolution step
  - @jax.custom_jvp with implicit function theorem for shooting
  - jax.lax.scan over evolution -> grad goes through cheap IFT, not 60+
    Newton iters

Calibration (NON-MIST sources only):
  Y_p     = 0.2485      (Aver+2015, BBN)
  dY/dZ   = 1.5         (Galactic chemical evolution slope)
  F_NUC   = 1.0         (Hansen-Kawaler / Adelberger PP+CNO)

Public API:
  evolve_star(mass, Z=0.02, max_steps=200) -> dict
  solve_structure(M_solar, X_c, Z, t_age=1e9,
                  log_L0=None, log_Te0=None) -> [log_L, log_Te, log_R]
"""
import os
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
# Inversion done ONCE at module load via numpy.interp on logQ axis
# (logPgas is monotonic in logQ for stable matter, so np.interp inverts).
# ------------------------------------------------------------------
def _build_eos_pgrid():
    """Pre-compute EOS lookup table inverted onto a (X, Z, logT, logP) grid."""
    _OPAL_EOS_PATH = os.environ.get('STELLAR_OPAL_EOS',
        os.path.join(os.path.dirname(__file__), 'data', 'eos_compact.npz'))
    eos = np.load(_OPAL_EOS_PATH)
    X_grid    = eos['X_grid'].astype(np.float64)
    Z_grid    = eos['Z_grid'].astype(np.float64)
    logT_grid = eos['logT_grid'].astype(np.float64)
    logQ_grid = eos['logQ_grid'].astype(np.float64)
    logP_tab  = eos['logPgas'].astype(np.float64)   # (5,3,78,150)
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
                slice_logP = logP_tab[ix, iz, :, it]   # (78,)
                slice_mu   = mu_tab[ix, iz, :, it]
                slice_nad  = nad_tab[ix, iz, :, it]

                # Ensure logP is monotonic in logQ (sort if needed)
                if np.all(np.diff(slice_logP) > 0):
                    xp_logP, fp_logQ = slice_logP, logQ_grid
                    fp_mu, fp_nad    = slice_mu, slice_nad
                else:
                    sort_idx = np.argsort(slice_logP)
                    xp_logP  = slice_logP[sort_idx]
                    fp_logQ  = logQ_grid[sort_idx]
                    fp_mu    = slice_mu[sort_idx]
                    fp_nad   = slice_nad[sort_idx]

                # Inverse interp: for each target logP, find logQ (and mu, nad)
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
    """4D bilinear interp of inverted EOS: (logT, logPgas, X, Z) -> (rho, mu, grad_ad)."""
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
# Photospheric mu fallback (only for surface-BC pressure iteration)
# ------------------------------------------------------------------
def mu_phot(X, Y, Z, T):
    log_T = jnp.log10(jnp.maximum(T, 100.0))
    x_H  = 0.5*(1.0 + jnp.tanh((log_T - 4.10) * 7.0))
    x_He = 0.5*(1.0 + jnp.tanh((log_T - 4.40) * 5.5))
    one_over_mu = X*(1.0 + x_H) + 0.25*Y*(1.0 + 2.0*x_He) + Z*(0.0625 + 0.4375*x_He)
    return 1.0 / jnp.maximum(one_over_mu, 0.01)


# ------------------------------------------------------------------
# Nuclear energy generation: PP + CNO with 3He non-equilibrium
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
# Burning core mass (geometric, no overshoot)
# ------------------------------------------------------------------
def burning_mass_frac(M_solar):
    return jnp.clip(0.10 + 0.05*M_solar, 0.05, 0.40)


# ------------------------------------------------------------------
# Surface-inward shooting integrator
# ------------------------------------------------------------------
N_SHOOT = 300
EPS_FD  = 1e-5


@jax.jit
def shoot(M_solar, log_L, log_Te, X_c, X_init, Z, M_burn_frac, t_age):
    """Integrate (P, M_r, L_r, T) from R_star to 0.005 R_star, return residual."""
    M_star = M_solar * Msun
    L_star = 10.0**log_L * Lsun
    Te     = 10.0**log_Te
    R_star = jnp.sqrt(L_star / (4.0*jnp.pi*sigma_sb)) / Te**2
    g_surf = G*M_star/R_star**2

    Y_init = Y_BBN + DY_DZ*Z

    # Photospheric BC
    rho_g  = 1e-7
    log_kp = opal_kappa(jnp.log10(Te), jnp.log10(rho_g), X_init, Z)
    P_phot = (2.0/3.0)*g_surf / (10.0**log_kp)
    mu_e   = mu_phot(X_init, Y_init, Z, Te)
    rho_g  = jnp.maximum(mu_e*m_H*P_phot/(k_B*Te), 1e-15)
    log_kp = opal_kappa(jnp.log10(Te), jnp.log10(rho_g), X_init, Z)
    P_phot = (2.0/3.0)*g_surf / (10.0**log_kp)
    rho_g  = jnp.maximum(mu_e*m_H*P_phot/(k_B*Te), 1e-15)
    log_kp = opal_kappa(jnp.log10(Te), jnp.log10(rho_g), X_init, Z)
    P_phot = jnp.maximum((2.0/3.0)*g_surf / (10.0**log_kp), 1.0)

    state0  = jnp.array([P_phot, M_star, L_star, Te])
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

        # OPAL EOS lookup (pre-inverted, bilinear) -> rho, mu, grad_ad
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
        nabla     = jnp.where(nabla_rad > nad, nad, nabla_rad)
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
# Newton-Raphson on (log_L, log_Te) with warm-start initial guess
# ------------------------------------------------------------------
def initial_guess(M_solar):
    logM = jnp.log10(M_solar)
    return 4.5*logM - 0.18, 3.752 + 0.65*logM


N_NEWTON_COLD = 15  # used when log_L0 is None
N_NEWTON_WARM = 4


def _newton_solve(M_solar, X_c, X_init, Z, f_burn, t_age, logL, logTe, n_iter):
    def step(state, _):
        lL, lT = state
        R0   = shoot(M_solar, lL,        lT,        X_c, X_init, Z, f_burn, t_age)
        R_dL = shoot(M_solar, lL+EPS_FD, lT,        X_c, X_init, Z, f_burn, t_age)
        R_dT = shoot(M_solar, lL,        lT+EPS_FD, X_c, X_init, Z, f_burn, t_age)
        Jmat = jnp.column_stack([(R_dL-R0)/EPS_FD, (R_dT-R0)/EPS_FD]) + 1e-8*jnp.eye(2)
        dx = jnp.linalg.solve(Jmat, -R0)
        dx = jnp.clip(dx, -0.10, 0.10)
        return (jnp.clip(lL + dx[0], -3.0, 5.0),
                jnp.clip(lT + dx[1], 3.45, 4.5)), None
    out, _ = lax.scan(step, (logL, logTe), None, length=n_iter)
    return out


@custom_jvp
def solve_structure(M_solar, X_c, Z, t_age, log_L0, log_Te0):
    """Solve 4 stellar-structure ODEs by Newton-Raphson shooting.
    Uses (log_L0, log_Te0) as warm-start initial guess. Returns [log_L, log_Te, log_R].
    """
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
    """IFT JVP. log_L0, log_Te0 are initial guesses only -> their tangents drop."""
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


from functools import partial


@partial(jax.jit, static_argnames=('max_steps',))
def evolve_star(mass, Z=0.02, max_steps=200):
    """Evolve a star from ZAMS through MS using warm-started solve_structure.

    The first step uses cold-start Newton (many iters); subsequent steps
    warm-start from the previous step's converged (log_L, log_Te) and need
    only ~6 iters. Gradients flow through custom_jvp/IFT cheaply.

    Returns dict with star_age (years), log_L, log_Teff, log_R, center_h1.
    """
    mass = jnp.asarray(mass, dtype=jnp.float64)
    Z    = jnp.asarray(Z,    dtype=jnp.float64)

    Y_init = Y_BBN + DY_DZ*Z
    X_init = jnp.maximum(1.0 - Y_init - Z, 0.5)
    f_burn = burning_mass_frac(mass)
    M_burn_g = f_burn * mass * Msun

    Xc_arr   = jnp.linspace(X_init, jnp.float64(0.005), max_steps)
    dXc_step = (X_init - 0.005) / (max_steps - 1)

    # First step: cold-start Newton from textbook initial guess
    log_L_g, log_Te_g = initial_guess(mass)
    logL0, logTe0 = _newton_solve(mass, Xc_arr[0], X_init, Z, f_burn,
                                   jnp.float64(0.0),
                                   log_L_g, log_Te_g, N_NEWTON_COLD)

    def step_fn(carry, X_c):
        prev_L, prev_T, t_prev = carry
        t_age = t_prev / SECONDS_PER_YEAR
        # solve_structure with warm-start (uses 6 Newton iters, custom_jvp/IFT)
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
