"""GIOP forward model: eigenvectors, IOP assembly, and the AOP-IOP closure.

THEORY.md G1-G4, G10, G11.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import aphstar as _aph
from . import water as _water
from .matlab_compat import nearest_band

__all__ = [
    "GiopConfig",
    "ConfigurationError",
    "rrs_above_to_below",
    "rrs_below_to_above",
    "rrs_from_iops",
    "eta_qaa",
    "sdg_from_option",
    "eigenvectors",
    "cost",
    "GORDON_G0",
    "GORDON_G1",
]

GORDON_G0 = 0.0949
GORDON_G1 = 0.0794

#: Nominal anchor wavelengths and the tolerance used to find them in a band set.
#: Upstream hard-codes the windows [410,415], [441,445], [546,557] (giop.m:30-32).
ANCHOR_412 = (412.0, 3.0)
ANCHOR_443 = (443.0, 2.5)
ANCHOR_555 = (551.5, 6.0)


class ConfigurationError(ValueError):
    """A gopt combination that is physically or numerically invalid."""


@dataclass
class GiopConfig:
    """Run-time parameterisation, the Python form of upstream's ``gopt`` struct.

    Defaults reproduce GIOP-DC as configured in ``run_giop.m``.
    """

    # AOP-IOP relationship: 'gordon', 'morel', or an explicit (g0, g1) pair.
    fq: object = "gordon"
    solz: float = 30.0
    senz: float | None = None
    relaz: float | None = None

    # Air-sea transmission: 'lee' (G4a) or 'flat' (G4b).
    trans: str = "lee"

    # bbp slope: 'qaa', 'gsm', or a float.
    eta: object = "qaa"
    # adg slope: 0.018 (GIOP-DC), 'qaa', 'obpg', 'gsm', or a float (nm^-1).
    sdg: object = 0.018
    # aph*: 'bricaud', 'ciotti', 'gsm', 'chase', or an explicit array over wl.
    aph: object = "bricaud"
    sf: float = 0.5

    # Pure water. None means compute from the shipped tables.
    aw: np.ndarray | None = None
    bbw: np.ndarray | None = None

    # Inversion: 'fmin' (Nelder-Mead, GIOP-DC) or 'lmi' (linear matrix inversion).
    inv: str = "fmin"
    n_starts: int = 1

    # QC: max relative |r_rs_model - r_rs_obs| over 400-600 nm. None disables.
    qc: float | None = None

    # MATLAB interp1 semantics: 'pchip' (modern 'cubic') or 'v5cubic' (historical).
    interp: str = "pchip"

    # Extension, off by default and NOT part of GIOP-DC: also fit S_dg and eta.
    fit_shapes: bool = False

    def validate(self):
        if self.inv not in ("fmin", "lmi", "bounded"):
            raise ConfigurationError(
                f"inv must be 'fmin' (GIOP-DC), 'lmi', or 'bounded'; got {self.inv!r}")
        if self.trans not in ("lee", "flat"):
            raise ConfigurationError(f"trans must be 'lee' or 'flat'; got {self.trans!r}")
        if self.inv == "lmi" and isinstance(self.fq, str) and self.fq == "morel":
            # THEORY.md sect. 5 / PORTING_NOTES.md D2: the Morel option sets g1 = 0 and
            # the LMI branch divides by g1 (giop.m:190). MATLAB returns Inf/NaN silently.
            raise ConfigurationError(
                "fq='morel' sets g1 = 0, and the LMI solver divides by g1 "
                "(giop.m:190), so this combination is a division by zero. Upstream "
                "MATLAB returns Inf/NaN without complaint. Use inv='fmin' with "
                "fq='morel', or fq='gordon' with inv='lmi'."
            )
        if self.fit_shapes and self.inv == "lmi":
            raise ConfigurationError(
                "fit_shapes=True is nonlinear in S_dg and eta, so it cannot be solved "
                "by the linear matrix inversion. Use inv='fmin'."
            )


def rrs_above_to_below(rrs, trans="lee"):
    """R_rs (above surface) -> r_rs (below surface). THEORY.md G4a/G4b."""
    rrs = np.asarray(rrs, dtype=float)
    if trans == "lee":
        return rrs / (0.52 + 1.7 * rrs)
    if trans == "flat":
        return rrs / 0.529
    raise ConfigurationError(f"unknown trans {trans!r}")


def rrs_below_to_above(rin, trans="lee"):
    """r_rs (below surface) -> R_rs (above surface). THEORY.md G4c.

    Upstream never does this, which is why ``run_giop.m:96`` plots a subsurface model
    spectrum against an above-surface measurement. See PORTING_NOTES.md D1.
    """
    rin = np.asarray(rin, dtype=float)
    if trans == "lee":
        denom = 1.0 - 1.7 * rin
        return np.where(denom > 0, 0.52 * rin / denom, np.nan)
    if trans == "flat":
        return rin * 0.529
    raise ConfigurationError(f"unknown trans {trans!r}")


def rrs_from_iops(a_tot, bb_tot, g0, g1):
    """Gordon quadratic closure, THEORY.md G1. Returns subsurface r_rs (sr^-1)."""
    u = bb_tot / (a_tot + bb_tot)
    return g0 * u + g1 * u * u


def u_from_rrs(rin, g0, g1):
    """Positive root of the Gordon quadratic for u. THEORY.md G12."""
    if g1 == 0:
        raise ConfigurationError("u_from_rrs requires g1 != 0 (see GiopConfig.validate)")
    return (-g0 + np.sqrt(g0 * g0 + 4.0 * g1 * rin)) / (2.0 * g1)


def eta_qaa(rin, i443, i555):
    """QAA v5 bbp slope from the subsurface blue/green ratio. THEORY.md G10."""
    return 2.0 * (1.0 - 1.2 * np.exp(-0.9 * rin[i443] / rin[i555]))


def sdg_from_option(option, rrs, rin, i412, i443, i555):
    """adg slope in nm^-1. THEORY.md sect. 4.6, ported from ``giop.m:112-131``.

    Note the deliberate asymmetry preserved from upstream: 'qaa' uses subsurface
    r_rs while 'obpg' uses above-surface R_rs. PORTING_NOTES.md D3.
    """
    if isinstance(option, str):
        if option == "qaa":
            return 0.015 + 0.002 / (0.6 + rin[i443] / rin[i555])
        if option == "obpg":
            s = 0.015 + 0.0038 * np.log10(rrs[i412] / rrs[i555])
            return float(np.clip(s, 0.01, 0.02))
        if option == "gsm":
            return 0.02061
        raise ConfigurationError(f"unknown sdg option {option!r}")
    return float(option)


def eigenvectors(wl, cfg, chl, rrs, rin, idx, sdg=None, eta=None):
    """Build the three spectral shapes. THEORY.md G3.

    Returns ``(adgs, bbps, aphs, sdg, eta)``. ``idx`` is the anchor-band index dict.
    ``sdg``/``eta`` override the configured values (used by ``fit_shapes``).
    """
    wl = np.asarray(wl, dtype=float)

    if eta is None:
        if isinstance(cfg.eta, str):
            if cfg.eta == "qaa":
                eta = eta_qaa(rin, idx["443"], idx["555"])
            elif cfg.eta == "gsm":
                eta = 1.03373
            else:
                raise ConfigurationError(f"unknown eta option {cfg.eta!r}")
        else:
            eta = float(cfg.eta)

    if sdg is None:
        sdg = sdg_from_option(cfg.sdg, rrs, rin, idx["412"], idx["443"], idx["555"])

    bbps = (443.0 / wl) ** eta
    adgs = np.exp(-sdg * (wl - 443.0))

    if isinstance(cfg.aph, str):
        if cfg.aph == "bricaud":
            aphs = _aph.bricaud1998(chl, wl, normalize=True, method=cfg.interp)
        elif cfg.aph == "ciotti":
            aphs = _aph.ciotti2006(wl, sf=cfg.sf, method=cfg.interp)
        elif cfg.aph == "gsm":
            aphs = _aph.gsm(wl, method=cfg.interp)
        elif cfg.aph == "chase":
            aphs = _aph.chase2017(chl, wl, method=cfg.interp)
        else:
            raise ConfigurationError(f"unknown aph option {cfg.aph!r}")
    else:
        aphs = np.asarray(cfg.aph, dtype=float)
        if aphs.shape != wl.shape:
            raise ConfigurationError(
                f"user aph* has shape {aphs.shape}, expected {wl.shape} to match wl"
            )

    return adgs, bbps, aphs, float(sdg), float(eta)


def cost(x, rin, aw, bbw, adgs, bbps, aphs, g0, g1):
    """Unweighted sum of squared reflectance residuals. THEORY.md G11.

    Ported from ``giop_cost.m``. Parameter order is upstream's:
    ``x = (M_dg, M_bp, M_phi)``.
    """
    a_tot = aw + aphs * x[2] + adgs * x[0]
    bb_tot = bbw + bbps * x[1]
    rmod = rrs_from_iops(a_tot, bb_tot, g0, g1)
    return float(np.sum((rin - rmod) ** 2))


def find_anchor_bands(wl, strict=True):
    """Locate the 412 / 443 / 555 anchor bands GIOP requires.

    Upstream takes the *first* index inside a fixed window (``giop.m:30-32``); this
    takes the *nearest* band to the nominal wavelength, which agrees on a 6-band
    satellite grid and differs on a dense field grid. PORTING_NOTES.md D5.
    """
    wl = np.asarray(wl, dtype=float)
    idx = {}
    for name, (target, tol) in (
        ("412", ANCHOR_412),
        ("443", ANCHOR_443),
        ("555", ANCHOR_555),
    ):
        i = nearest_band(wl, target, tol)
        if i is None and strict:
            raise ConfigurationError(
                f"GIOP requires a band within {tol} nm of {target} nm; the supplied "
                f"grid spans {wl.min():.1f}-{wl.max():.1f} nm and has none. "
                "GIOP-DC is anchored on Rrs(412), Rrs(443) and Rrs(547/555)."
            )
        idx[name] = i
    return idx
