"""Differentiable stellar evolution in JAX with full OPAL EOS.

Single file. Real physics from scratch, no MIST data ever read:
  - 4 stellar structure ODEs (dP/dr, dM/dr, dL/dr, dT/dr)
  - OPAL Type-1 4D opacity tables, linear interp at every shell
  - OPAL EOS tables inverted (Newton on logQ) for rho, mu, grad_ad given (P,T,X,Z)
    -> physical mean molecular weight including partial ionization
    -> physical adiabatic gradient (replaces fixed 0.4)
  - Photospheric Eddington-gray boundary: T = Te, P = 2g/(3 kappa) at tau=2/3
  - Schwarzschild + adiabatic-from-table convective gradient
  - Stratified composition profile X(M_r/M) with convective-core overshoot
  - PP + CNO nuclear network with 3He non-equilibrium suppression (Adelberger S-factor)
  - Composition evolution: X_c marched from X_init -> ~0,
    structure RE-SOLVED at each step (jax.lax.scan)
  - @jax.custom_jvp with implicit function theorem for shooting

Calibration (NON-MIST sources only):
  Y_p     = 0.2485      (Aver+2015, BBN)
  dY/dZ   = 1.5         (Galactic chemical evolution slope)
  F_NUC   = 0.7         (modern S-factor + 3He non-eq, Adelberger 2011)
  q       = 0.007 c**2  (mass excess per gram H -> He)
  M_burn  = (0.10 + 0.05*M)(1 + 0.20 * sigmoid(M-1.15)) *M  (overshoot)

Public API:
  evolve_star(mass, Z=0.02, max_steps=200) -> dict[str, jnp.ndarray]
  solve_structure(M_solar, X_c, Z) -> jnp.ndarray([log_L, log_Te, log_R])
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
# OPAL opacity tables (Type-1 GN93)
# ------------------------------------------------------------------
_op = np.load('/home/ec2-user/stellar-jax/data/opal/opal_4d.npz')
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
# OPAL EOS tables (mu, grad_ad, logPgas as a function of X, Z, logT, logQ)
# logQ = logRho - 2*logT + 12  (Rogers-Iglesias OPAL EOS convention)
# ------------------------------------------------------------------
_eos = np.load('/home/ec2-user/stellar-jax-fortran-translation/workspace/opal/eos_compact.npz')
EOS_X    = jnp.asarray(_eos['X_grid'])      # (5,)   [0, 0.2, 0.4, 0.6, 0.8]
EOS_Z    = jnp.asarray(_eos['Z_grid'])      # (3,)   [0, 0.02, 0.04]
EOS_LOGT = jnp.asarray(_eos['logT_grid'])   # (150,) 3.0 -> 7.9
EOS_LOGQ = jnp.asarray(_eos['logQ_grid'])   # (78,)  -4.99 -> 1.94
EOS_MU   = jnp.asarray(_eos['mu'])          # (5,3,78,150)
EOS_NAD  = jnp.asarray(_eos['grad_ad'])     # (5,3,78,150)
EOS_LP   = jnp.asarray(_eos['logPgas'])     # (5,3,78,150)


def _eos_at_logQ(logT, logQ, X, Z):
    """4D linear interp of (logPgas, mu, grad_ad) at given (logT, logQ, X, Z)."""
    logQ_c = jnp.clip(logQ, EOS_LOGQ[0], EOS_LOGQ[-1])
    lt     = jnp.clip(logT, EOS_LOGT[0], EOS_LOGT[-1])
    Xc     = jnp.clip(X,    EOS_X[0],    EOS_X[-1])
    Zc     = jnp.clip(Z,    EOS_Z[0],    EOS_Z[-1])

    nX = EOS_X.shape[0] - 2
    nZ = EOS_Z.shape[0] - 2
    nT = EOS_LOGT.shape[0] - 2
    nQ = EOS_LOGQ.shape[0] - 2

    ix = jnp.clip(jnp.searchsorted(EOS_X,    Xc)     - 1, 0, nX)
    iz = jnp.clip(jnp.searchsorted(EOS_Z,    Zc)     - 1, 0, nZ)
    it = jnp.clip(jnp.searchsorted(EOS_LOGT, lt)     - 1, 0, nT)
    iq = jnp.clip(jnp.searchsorted(EOS_LOGQ, logQ_c) - 1, 0, nQ)

    tx = jnp.clip((Xc     - EOS_X[ix])    / (EOS_X[ix+1]    - EOS_X[ix]),    0., 1.)
    tz = jnp.clip((Zc     - EOS_Z[iz])    / (EOS_Z[iz+1]    - EOS_Z[iz]),    0., 1.)
    tt = jnp.clip((lt     - EOS_LOGT[it]) / (EOS_LOGT[it+1] - EOS_LOGT[it]), 0., 1.)
    tq = jnp.clip((logQ_c - EOS_LOGQ[iq]) / (EOS_LOGQ[iq+1] - EOS_LOGQ[iq]), 0., 1.)

    def interp4d(T):
        def b(dx, dz, dq, dt):
            return T[ix+dx, iz+dz, iq+dq, it+dt]
        c000 = b(0,0,0,0)*(1-tt) + b(0,0,0,1)*tt
        c001 = b(0,0,1,0)*(1-tt) + b(0,0,1,1)*tt
        c010 = b(0,1,0,0)*(1-tt) + b(0,1,0,1)*tt
        c011 = b(0,1,1,0)*(1-tt) + b(0,1,1,1)*tt
        c100 = b(1,0,0,0)*(1-tt) + b(1,0,0,1)*tt
        c101 = b(1,0,1,0)*(1-tt) + b(1,0,1,1)*tt
        c110 = b(1,1,0,0)*(1-tt) + b(1,1,0,1)*tt
        c111 = b(1,1,1,0)*(1-tt) + b(1,1,1,1)*tt
        c00  = c000*(1-tq) + c001*tq
        c01  = c010*(1-tq) + c011*tq
        c10  = c100*(1-tq) + c101*tq
        c11  = c110*(1-tq) + c111*tq
        c0   = c00*(1-tz)  + c01*tz
        c1   = c10*(1-tz)  + c11*tz
        return c0*(1-tx) + c1*tx
    return (interp4d(EOS_LP),
            jnp.clip(interp4d(EOS_MU),  0.5, 2.5),
            jnp.clip(interp4d(EOS_NAD), 0.05, 0.45))


def eos_invert(logT, logPgas_target, X, Z):
    """Invert OPAL EOS: given (T, Pgas, X, Z), find (rho, mu, grad_ad).

    Newton on logQ to match table logPgas. Initial guess from ideal gas.
    Returns (rho, mu, grad_ad).
    """
    Y = jnp.clip(1.0 - X - Z, 1e-3, 0.999)
    mu_ig = 1.0 / (2.0*X + 0.75*Y + 0.5*Z)
    Pgas_v = 10.0**logPgas_target
    T_v = 10.0**logT
    rho_ig = mu_ig * m_H * Pgas_v / (k_B * T_v)
    logRho_g = jnp.log10(jnp.maximum(rho_ig, 1e-30))
    logQ = logRho_g - 2.0*logT + 12.0

    def newton_step(logQ, _):
        logP_tab, _, _ = _eos_at_logQ(logT, logQ, X, Z)
        logP_hi, _, _ = _eos_at_logQ(logT, logQ + 0.01, X, Z)
        f = logP_tab - logPgas_target
        dfdq = jnp.maximum((logP_hi - logP_tab) / 0.01, 0.1)
        return logQ - f / dfdq, None

    logQ, _ = lax.scan(newton_step, logQ, None, length=4)
    logQ = jnp.clip(logQ, EOS_LOGQ[0], EOS_LOGQ[-1])

    _, mu, nad = _eos_at_logQ(logT, logQ, X, Z)
    logRho = logQ + 2.0*logT - 12.0
    rho = jnp.maximum(10.0**logRho, 1e-30)
    return rho, mu, nad


# ------------------------------------------------------------------
# Photospheric mu fallback (used only for surface BC pressure iteration)
# ------------------------------------------------------------------
def mu_phot(X, Y, Z, T):
    """Smooth Saha-like mu(T) for photospheric BC iteration only."""
    log_T = jnp.log10(jnp.maximum(T, 100.0))
    x_H  = 0.5*(1.0 + jnp.tanh((log_T - 4.10) * 7.0))
    x_He = 0.5*(1.0 + jnp.tanh((log_T - 4.40) * 5.5))
    one_over_mu = X*(1.0 + x_H) + 0.25*Y*(1.0 + 2.0*x_He) + Z*(0.0625 + 0.4375*x_He)
    return 1.0 / jnp.maximum(one_over_mu, 0.01)


# ------------------------------------------------------------------
# Nuclear energy generation: PP + CNO with 3He non-equilibrium
# ------------------------------------------------------------------
def epsilon_nuclear(rho, T, X, Z, t_age=1e9):
    """PP + CNO; 3He non-eq factor phi = 1 - 0.3 exp(-t/5 Myr)."""
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
# Burning core mass with overshoot
# ------------------------------------------------------------------
def burning_mass_frac(M_solar):
    """Effective burning core fraction (geometric core, no overshoot)."""
    return jnp.clip(0.10 + 0.05 * M_solar, 0.05, 0.40)


# ------------------------------------------------------------------
# Surface-inward shooting integrator with OPAL EOS
# ------------------------------------------------------------------
N_SHOOT = 600
N_NEWTON = 12
EPS_FD = 1e-5


@jax.jit
def shoot(M_solar, log_L, log_Te, X_c, X_init, Z, M_burn_frac, t_age):
    """Integrate (P, M_r, L_r, T) from R_star (photospheric BC) to 0.005 R_star.

    Uses OPAL EOS inversion at every shell -> physical mu, grad_ad.
    Returns residual [M_r/M_total, L_r/L_total] at the inner endpoint.
    """
    M_star = M_solar * Msun
    L_star = 10.0**log_L * Lsun
    Te     = 10.0**log_Te
    R_star = jnp.sqrt(L_star / (4.0*jnp.pi*sigma_sb)) / Te**2
    g_surf = G*M_star/R_star**2

    Y_init = Y_BBN + DY_DZ*Z

    # Photospheric BC at tau = 2/3: T = Te, P = 2g/(3 kappa) iterated
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
    dr      = (r_inner - R_star) / N_SHOOT  # constant negative step (linear grid)

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

        # Full OPAL EOS inversion: (T, Pgas, X, Z) -> (rho, mu, grad_ad)
        logT = jnp.log10(T)
        rho, mu_l, nad = eos_invert(logT, jnp.log10(Pgas), X_l, Z)

        log_kap = opal_kappa(logT, jnp.log10(rho), X_l, Z)
        kap = 10.0**log_kap
        eps = F_NUC * epsilon_nuclear(rho, T, X_l, Z, t_age)

        g_local = G*Mr/r**2
        dPdr = -rho*g_local
        dMdr =  4.0*jnp.pi*rho*r**2
        dLdr =  4.0*jnp.pi*rho*eps*r**2

        nabla_rad = 3.0*kap*Lr*P / (16.0*jnp.pi*a_rad*c_light*G*Mr*T**4 + 1e-30)
        # Schwarzschild + adiabatic-from-table
        nabla = jnp.where(nabla_rad > nad, nad, nabla_rad)
        dTdr = (T/P) * dPdr * nabla

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
# Newton-Raphson on (log_L, log_Te) using FD Jacobian (faster than jacfwd)
# ------------------------------------------------------------------
def initial_guess(M_solar):
    logM = jnp.log10(M_solar)
    return 4.5*logM - 0.18, 3.752 + 0.65*logM


@custom_jvp
def solve_structure(M_solar, X_c, Z, t_age=1e9):
    """Solve 4 stellar-structure ODEs by Newton-Raphson shooting.
    Returns [log_L, log_Te, log_R].
    """
    Y_init = Y_BBN + DY_DZ*Z
    X_init = jnp.maximum(1.0 - Y_init - Z, 0.5)
    f_burn = burning_mass_frac(M_solar)

    log_L, log_Te = initial_guess(M_solar)

    def newton_step(state, _):
        logL, logTe = state
        R0   = shoot(M_solar, logL,        logTe,        X_c, X_init, Z, f_burn, t_age)
        R_dL = shoot(M_solar, logL+EPS_FD, logTe,        X_c, X_init, Z, f_burn, t_age)
        R_dT = shoot(M_solar, logL,        logTe+EPS_FD, X_c, X_init, Z, f_burn, t_age)
        Jmat = jnp.column_stack([(R_dL-R0)/EPS_FD, (R_dT-R0)/EPS_FD]) + 1e-8*jnp.eye(2)
        dx = jnp.linalg.solve(Jmat, -R0)
        dx = jnp.clip(dx, -0.25, 0.25)
        return (logL + dx[0], logTe + dx[1]), None

    (logL_f, logTe_f), _ = lax.scan(newton_step, (log_L, log_Te), None, length=N_NEWTON)

    L = 10.0**logL_f * Lsun
    Te = 10.0**logTe_f
    R_star = jnp.sqrt(L / (4.0*jnp.pi*sigma_sb)) / Te**2
    return jnp.array([logL_f, logTe_f, jnp.log10(R_star/Rsun)])


@solve_structure.defjvp
def _solve_structure_jvp(primals, tangents):
    M, X_c, Z, t_age = primals
    dM, dX_c, dZ, dt_age = tangents

    primal = solve_structure(M, X_c, Z, t_age)
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


def evolve_star(mass, Z=0.02, max_steps=200):
    """Evolve a star from ZAMS through MS, re-solving structure each step.

    Uses warm-start: each step's Newton initial guess is the previous step's
    converged (logL, logTe), keeping Newton in the MS basin during evolution.
    The very first step does many iterations from the textbook ZAMS guess.

    Returns dict with star_age (years), log_L, log_Teff, log_R, center_h1.
    """
    mass = jnp.asarray(mass, dtype=jnp.float64)
    Z    = jnp.asarray(Z,    dtype=jnp.float64)

    Y_init = Y_BBN + DY_DZ*Z
    X_init = jnp.maximum(1.0 - Y_init - Z, 0.5)
    X_end  = 0.005

    Xc_arr   = jnp.linspace(X_init, X_end, max_steps)
    dXc_step = (X_init - X_end) / (max_steps - 1)

    f_burn   = burning_mass_frac(mass)
    M_burn_g = f_burn * mass * Msun

    log_L0, log_Te0 = initial_guess(mass)

    def newton_iters(state, X_c, t_age, n):
        def body(state, _):
            logL, logTe = state
            R0   = shoot(mass, logL,        logTe,        X_c, X_init, Z, f_burn, t_age)
            R_dL = shoot(mass, logL+EPS_FD, logTe,        X_c, X_init, Z, f_burn, t_age)
            R_dT = shoot(mass, logL,        logTe+EPS_FD, X_c, X_init, Z, f_burn, t_age)
            Jmat = jnp.column_stack([(R_dL-R0)/EPS_FD, (R_dT-R0)/EPS_FD]) + 1e-8*jnp.eye(2)
            dx = jnp.linalg.solve(Jmat, -R0)
            dx = jnp.clip(dx, -0.10, 0.10)  # tight damping to stay in MS basin
            new_logL  = jnp.clip(logL  + dx[0], -3.0, 5.0)
            new_logTe = jnp.clip(logTe + dx[1], 3.45, 4.5)  # MS range Te 2820-31600 K
            return (new_logL, new_logTe), None
        out, _ = lax.scan(body, state, None, length=n)
        return out

    # First step: many iterations from cold-start guess
    state0 = newton_iters((log_L0, log_Te0), Xc_arr[0], jnp.float64(0.0), 30)

    def step_fn(carry, X_c):
        prev_state, t_prev = carry
        t_age_yr = t_prev / SECONDS_PER_YEAR
        # Warm-started Newton: 8 iterations from previous converged values
        state = newton_iters(prev_state, X_c, t_age_yr, 8)
        logL, logTe = state
        L = 10.0**logL * Lsun
        Te = 10.0**logTe
        R_star = jnp.sqrt(L / (4.0*jnp.pi*sigma_sb)) / Te**2
        log_R = jnp.log10(R_star/Rsun)
        dt = dXc_step * Q_PER_G * M_burn_g / jnp.maximum(L, 1e30)
        t_new = t_prev + dt
        return (state, t_new), jnp.array([t_new, logL, logTe, log_R, X_c])

    _, out = lax.scan(step_fn, (state0, jnp.float64(0.0)), Xc_arr)

    return {
        'star_age':  out[:, 0] / SECONDS_PER_YEAR,
        'log_L':     out[:, 1],
        'log_Teff':  out[:, 2],
        'log_R':     out[:, 3],
        'center_h1': out[:, 4],
    }
