"""The GIOP inverse problem. THEORY.md sect. 6.

``giop()`` is the entry point and is a faithful port of ``giop.m`` / ``giop_kb.m``,
with deviations listed in PORTING_NOTES.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla
from scipy.optimize import minimize

from . import model as _m
from . import water as _water
from .model import ConfigurationError, GiopConfig

__all__ = ["giop", "GiopResult", "FILL"]

#: Upstream's fill value for a failed retrieval (``giop.m:22-26``).
FILL = -999.0


@dataclass
class GiopResult:
    """Outputs of one GIOP inversion.

    ``x`` is the eigenvalue triple in upstream order (M_dg, M_bp, M_phi), i.e. the
    printed vector that ``run_giop.m`` compares against 0.0441 / 0.0033 / 0.3693.
    """

    wl: np.ndarray
    x: np.ndarray                    # (M_dg, M_bp, M_phi)
    adg: np.ndarray                  # m^-1
    bbp: np.ndarray                  # m^-1
    aph: np.ndarray                  # m^-1
    apg: np.ndarray                  # m^-1, adg + aph
    rrs_model_subsurface: np.ndarray  # sr^-1, the quantity giop.m returns as mrrs
    rrs_model_above: np.ndarray      # sr^-1, back-transformed through G4c
    rrs_obs: np.ndarray              # sr^-1, above surface, as supplied
    rrs_obs_subsurface: np.ndarray   # sr^-1
    aw: np.ndarray
    bbw: np.ndarray
    adgs: np.ndarray                 # eigenvectors
    bbps: np.ndarray
    aphs: np.ndarray
    sdg: float
    eta: float
    g0: object
    g1: float
    chl_seed: float
    converged: bool
    cost: float
    qc_passed: bool | None
    anchor_idx: dict
    n_starts_spread: np.ndarray | None = None
    message: str = ""

    @property
    def chl(self):
        """Retrieved M_phi. Equals chlorophyll only when aph* is Bricaud-normalised."""
        return float(self.x[2])

    @property
    def adg443(self):
        return float(self.x[0])

    @property
    def bbp443(self):
        return float(self.x[1])

    @property
    def failed(self):
        return bool(np.any(self.x == FILL)) or not np.all(np.isfinite(self.x))


def giop(wl, rrs, chl=0.2, cfg=None, sigma=None, **kwargs):
    """Invert one R_rs spectrum for (a_dg, b_bp, a_phi). THEORY.md sect. 6.

    Parameters
    ----------
    wl : array, nm
    rrs : array, sr^-1, remote-sensing reflectance **above** the sea surface
    chl : float, mg m^-3, seed chlorophyll (typically from ``empirical.get_oc``)
    cfg : GiopConfig or None
    **kwargs : shorthand for GiopConfig fields, e.g. ``inv='lmi'``

    Returns
    -------
    GiopResult
    """
    if cfg is None:
        cfg = GiopConfig(**kwargs)
    elif kwargs:
        cfg = GiopConfig(**{**cfg.__dict__, **kwargs})
    cfg.validate()

    wl = np.asarray(wl, dtype=float)
    rrs = np.asarray(rrs, dtype=float)
    if wl.shape != rrs.shape:
        raise ValueError(f"wl {wl.shape} and rrs {rrs.shape} must have the same shape")
    if wl.ndim != 1:
        raise ValueError("giop() inverts one spectrum at a time; wl must be 1-D")
    _sanity_check_units(wl, rrs)

    idx = _m.find_anchor_bands(wl)

    aw = np.asarray(cfg.aw, float) if cfg.aw is not None else _water.a_water(wl, cfg.interp)
    bbw = np.asarray(cfg.bbw, float) if cfg.bbw is not None else _water.bb_water(wl)
    if np.any(~np.isfinite(aw)):
        bad = wl[~np.isfinite(aw)]
        raise ConfigurationError(
            f"pure-water absorption is undefined at {bad.min():.1f}-{bad.max():.1f} nm "
            "(the shipped table spans 380-1150 nm). Trim the spectrum or supply aw."
        )

    g0, g1 = _resolve_g(cfg, wl, chl)

    rin = _m.rrs_above_to_below(rrs, cfg.trans)

    adgs, bbps, aphs, sdg, eta = _m.eigenvectors(wl, cfg, chl, rrs, rin, idx)
    if np.any(~np.isfinite(aphs)):
        bad = wl[~np.isfinite(aphs)]
        raise ConfigurationError(
            f"aph* is undefined at {bad.min():.1f}-{bad.max():.1f} nm. The GIOP aph* "
            "eigenvectors are tabulated over 400-700 nm only (THEORY.md sect. 4). "
            "Restrict the inversion window, e.g. wl in [400, 700]."
        )

    spread = None
    if cfg.inv == "fmin":
        if cfg.fit_shapes:
            x, converged, cst, sdg, eta, adgs, bbps, aphs = _invert_fmin_shapes(
                wl, cfg, chl, rrs, rin, idx, aw, bbw, g0, g1
            )
        else:
            x, converged, cst, spread = _invert_fmin(
                rin, aw, bbw, adgs, bbps, aphs, g0, g1, chl, cfg.n_starts
            )
    elif cfg.inv == "bounded":
        # sigma is on the ABOVE-water scale as supplied; the fit is done on subsurface
        # r_rs, so propagate it through the same transform (G4a) to keep the weighting
        # consistent with the residuals it divides.
        sig_sub = None
        if sigma is not None:
            sig_sub = np.asarray(sigma, float) * (rin / np.where(rrs == 0, np.nan, rrs))
            sig_sub = np.where(np.isfinite(sig_sub), sig_sub, np.nanmedian(sig_sub))
        x, converged, cst, cov_b, rails = _invert_bounded(
            rin, aw, bbw, adgs, bbps, aphs, g0, g1, chl, sigma=sig_sub
        )
    else:
        x, converged, cst = _invert_lmi(rin, aw, bbw, adgs, bbps, aphs, g0, g1)

    adg = x[0] * adgs
    bbp = x[1] * bbps
    aph = x[2] * aphs
    apg = adg + aph

    if converged:
        a_tot = aw + adg + aph
        bb_tot = bbw + bbp
        mrrs = _m.rrs_from_iops(a_tot, bb_tot, g0, g1)
    else:
        mrrs = np.full_like(wl, FILL)

    qc_passed = None
    if cfg.qc is not None and converged:
        qc_passed, adg, bbp, aph, apg, mrrs = _apply_qc(
            cfg.qc, wl, rin, mrrs, adg, bbp, aph, apg
        )

    above = (
        _m.rrs_below_to_above(mrrs, cfg.trans)
        if converged and qc_passed is not False
        else np.full_like(wl, FILL)
    )

    return GiopResult(
        wl=wl, x=x, adg=adg, bbp=bbp, aph=aph, apg=apg,
        rrs_model_subsurface=mrrs, rrs_model_above=above,
        rrs_obs=rrs, rrs_obs_subsurface=rin,
        aw=aw, bbw=bbw, adgs=adgs, bbps=bbps, aphs=aphs,
        sdg=sdg, eta=eta, g0=g0, g1=g1, chl_seed=float(chl),
        converged=bool(converged), cost=float(cst), qc_passed=qc_passed,
        anchor_idx=idx, n_starts_spread=spread,
    )


# --------------------------------------------------------------------------------
# solvers


def _invert_fmin(rin, aw, bbw, adgs, bbps, aphs, g0, g1, chl, n_starts=1):
    """Nelder-Mead, matching ``fminsearch`` settings at ``giop.m:177-181``.

    MATLAB's fminsearch and SciPy's Nelder-Mead share the initial-simplex rule (5 %
    relative step, 0.00025 absolute for a zero coordinate) and the paired
    TolX/TolFun termination test, so this is a like-for-like substitution.
    THEORY.md sect. 9.2.
    """
    x0 = np.array([0.01, 0.001, float(chl)])
    args = (rin, aw, bbw, adgs, bbps, aphs, g0, g1)

    starts = [x0]
    if n_starts > 1:
        rng = np.random.default_rng(0)
        for _ in range(n_starts - 1):
            starts.append(x0 * rng.uniform(0.2, 5.0, size=3))

    results = []
    for s in starts:
        res = minimize(
            _m.cost, s, args=args, method="Nelder-Mead",
            options=dict(xatol=1e-6, fatol=1e-6, maxfev=int(1e5), maxiter=int(1e3)),
        )
        results.append(res)

    ok = [r for r in results if r.status == 0]
    if not ok:
        # MATLAB: exitflag < 1 -> x = [-999, -999, -999] (giop.m:181)
        return np.full(3, FILL), False, float(results[0].fun), None

    best = min(ok, key=lambda r: r.fun)
    spread = np.ptp(np.array([r.x for r in ok]), axis=0) if len(ok) > 1 else None
    return best.x, True, float(best.fun), spread


def _invert_bounded(rin, aw, bbw, adgs, bbps, aphs, g0, g1, chl, sigma=None,
                    upper=(20.0, 5.0, 500.0)):
    """Bounded, noise-weighted least squares. NOT GIOP-DC; an opt-in improvement.

    Two things this fixes about the published inversion (THEORY.md sect. 6.1):

    * **Positivity.** GIOP-DC minimises with an unconstrained Nelder-Mead simplex, so the
      amplitudes can and do go negative: across the shape ensemble on the demo spectrum,
      25 % of members return a negative a_phi. A negative absorption coefficient is not a
      retrieval, it is a signal that the prescribed shapes are wrong. Here the amplitudes
      are bounded to [0, upper].
    * **Weighting.** GIOP-DC sums squared residuals in absolute sr^-1, so the bright
      blue-green bands dominate and the red bands, where a_phi has its second peak and
      b_bp is most separable, contribute almost nothing. Passing ``sigma`` weights each
      band by its own uncertainty, which is what makes chi2_nu interpretable.

    Uses ``scipy.optimize.least_squares`` with the trust-region-reflective method, and
    returns the Gauss-Newton covariance from the Jacobian at the solution.
    """
    from scipy.optimize import least_squares

    n = len(rin)
    if sigma is None:
        s = np.ones(n)
    else:
        s = np.asarray(sigma, dtype=float)
        if s.shape != rin.shape:
            raise ValueError("sigma must have the same shape as the spectrum")
        if np.any(s <= 0):
            raise ValueError("sigma must be positive")

    def resid(x):
        a_tot = aw + aphs * x[2] + adgs * x[0]
        bb_tot = bbw + bbps * x[1]
        return (rin - _m.rrs_from_iops(a_tot, bb_tot, g0, g1)) / s

    x0 = np.array([0.01, 0.001, max(float(chl), 1e-4)])
    lo = np.zeros(3)
    hi = np.asarray(upper, dtype=float)
    x0 = np.clip(x0, lo + 1e-12, hi - 1e-12)

    sol = least_squares(resid, x0, bounds=(lo, hi), method="trf",
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=20000)

    cost = float(np.sum(sol.fun ** 2))
    cov = None
    try:
        JtJ = sol.jac.T @ sol.jac
        cov = np.linalg.pinv(JtJ)
        if sigma is None and n > 3:
            # Without a stated sigma the covariance is only defined up to the residual
            # scale, so rescale by the reduced chi-square. With a real sigma the
            # covariance is absolute and must NOT be rescaled.
            cov = cov * cost / (n - 3)
    except np.linalg.LinAlgError:
        cov = None

    rails = [bool(abs(sol.x[i] - lo[i]) < 1e-9 or abs(sol.x[i] - hi[i]) < 1e-9)
             for i in range(3)]
    return sol.x, bool(sol.status > 0), cost, cov, rails


def _invert_lmi(rin, aw, bbw, adgs, bbps, aphs, g0, g1):
    """Linear matrix inversion. THEORY.md G12-G13, ported from ``giop.m:186-208``.

    Upstream's four-line QR expression is the semi-normal equations plus one step of
    iterative refinement; the derivation that it collapses to
    ``x = (A'A)^-1 A'b`` computed through the QR factor is in THEORY.md sect. 6.2, and
    ``tests/test_lmi_solver.py`` pins the equivalence numerically.
    """
    u = _m.u_from_rrs(rin, g0, g1)
    b = bbw * (1.0 - u) - aw * u
    A = np.column_stack([adgs * u, bbps * (u - 1.0), aphs * u])

    _, R = sla.qr(A, mode="economic")
    Atb = A.T @ b
    try:
        x = sla.solve_triangular(R, sla.solve_triangular(R.T, Atb, lower=True))
        r = b - A @ x
        err = sla.solve_triangular(R, sla.solve_triangular(R.T, A.T @ r, lower=True))
        x = x + err
    except sla.LinAlgError:
        return np.full(3, FILL), False, np.inf

    resid = float(np.sum((b - A @ x) ** 2))
    return x, True, resid


def _invert_fmin_shapes(wl, cfg, chl, rrs, rin, idx, aw, bbw, g0, g1):
    """Extension, off by default: also fit S_dg and eta. THEORY.md sect. 6.3.

    Five parameters (M_dg, M_bp, M_phi, S_dg, eta). This is **not** GIOP-DC. It is
    provided because hyperspectral field data is the one case where the extra shape
    freedom might be supportable, and because prescribing shapes is the largest
    structural assumption in the model (A2, A3).
    """
    _, _, aphs0, sdg0, eta0 = _m.eigenvectors(wl, cfg, chl, rrs, rin, idx)

    def unpack(p):
        return p[:3], float(p[3]), float(p[4])

    def obj(p):
        amps, sdg, eta = unpack(p)
        if not (0.005 <= sdg <= 0.03) or not (-1.0 <= eta <= 4.0):
            return 1e6
        adgs = np.exp(-sdg * (wl - 443.0))
        bbps = (443.0 / wl) ** eta
        return _m.cost(amps, rin, aw, bbw, adgs, bbps, aphs0, g0, g1)

    p0 = np.array([0.01, 0.001, float(chl), sdg0, eta0])
    res = minimize(
        obj, p0, method="Nelder-Mead",
        options=dict(xatol=1e-8, fatol=1e-12, maxfev=int(2e5), maxiter=int(1e4)),
    )
    amps, sdg, eta = unpack(res.x)
    adgs = np.exp(-sdg * (wl - 443.0))
    bbps = (443.0 / wl) ** eta
    if res.status != 0:
        return np.full(3, FILL), False, float(res.fun), sdg, eta, adgs, bbps, aphs0
    return amps, True, float(res.fun), sdg, eta, adgs, bbps, aphs0


# --------------------------------------------------------------------------------
# quality control, ported from giop.m:243-280


def _apply_qc(qc, wl, rin, mrrs, adg, bbp, aph, apg):
    v = (wl >= 400.0) & (wl <= 600.0)
    if not np.any(v):
        return None, adg, bbp, aph, apg, mrrs

    with np.errstate(divide="ignore", invalid="ignore"):
        rtest = np.abs(mrrs[v] - rin[v]) / rin[v]

    if not np.all(np.isfinite(rtest)) or np.any(rtest > qc):
        fill = np.full_like(wl, FILL)
        return False, fill, fill.copy(), fill.copy(), fill.copy(), fill.copy()

    # Per-band range screen; upstream flags individual bands, not the whole spectrum.
    out = []
    for arr in (aph, adg, bbp, apg):
        arr = arr.copy()
        bad = (arr < -0.005) | (arr > 5.0)
        arr[bad] = FILL
        out.append(arr)
    aph, adg, bbp, apg = out
    return True, adg, bbp, aph, apg, mrrs


# --------------------------------------------------------------------------------


def _resolve_g(cfg, wl, chl):
    """Set (g0, g1). THEORY.md sect. 5."""
    if isinstance(cfg.fq, str):
        if cfg.fq == "gordon":
            return _m.GORDON_G0, _m.GORDON_G1
        if cfg.fq == "morel":
            from .aopiop import morel_g0

            return morel_g0(wl, chl, cfg.solz, cfg.senz, cfg.relaz, cfg.interp), 0.0
        raise ConfigurationError(f"unknown fq option {cfg.fq!r}")
    g = np.asarray(cfg.fq, dtype=float).ravel()
    if g.size != 2:
        raise ConfigurationError("fq must be 'gordon', 'morel', or a (g0, g1) pair")
    return float(g[0]), float(g[1])


def _sanity_check_units(wl, rrs):
    """Catch the two unit errors that would otherwise fail silently. THEORY.md sect. 1."""
    if np.nanmax(wl) < 100:
        raise ValueError(
            f"wavelengths look like micrometres (max {np.nanmax(wl):g}); GIOP wants nm"
        )
    finite = rrs[np.isfinite(rrs)]
    if finite.size and np.nanmax(np.abs(finite)) > 1.0:
        raise ValueError(
            f"|R_rs| reaches {np.nanmax(np.abs(finite)):g} sr^-1. Ocean R_rs is order "
            "1e-4 to 1e-1 sr^-1; this looks like percent reflectance or a "
            "reflectance factor. Convert with io.rrs first (R_rs = R / pi for a "
            "Lambertian reflectance factor R)."
        )
