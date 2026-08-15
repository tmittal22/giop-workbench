"""Morel et al. (2002) f/Q AOP-IOP relationship. THEORY.md sect. 5.

Ports ``morel_fq.m``, ``morel_fq_appb.m``, ``morel_read.m``, ``read_fq.m``, ``get_fq.m``.
Under this option g0 = f/Q(lambda) and g1 = 0, so (G1) collapses to r_rs = (f/Q) u.
"""

from __future__ import annotations

import functools
from importlib.resources import files

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .matlab_compat import fastsmooth, interp1

__all__ = ["morel_g0", "morel_fq_geometry", "morel_fq_appb", "morel_read", "read_fq"]

#: Native grid the Morel LUTs are interpolated onto before resampling to the band set.
_WL_OUT = np.arange(380.0, 701.0, 1.0)
_CHL_LUT = np.array([0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
_WL_FQ = np.array([412.5, 442.5, 490.0, 510.0, 560.0, 620.0, 660.0])
_SOLZ_LUT = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0])
_NADIR_LUT = np.array([1.078, 3.411, 6.289, 9.278, 12.30, 15.33, 18.37, 21.41, 24.45,
                       27.50, 30.54, 33.59, 36.64, 39.69, 42.73, 45.78, 48.83])
_AZM_LUT = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0,
                     150.0, 165.0, 180.0])
_WATER_N = 1.34


def morel_g0(wl, chl, solz=30.0, senz=None, relaz=None, method="pchip"):
    """f/Q at the requested wavelengths, i.e. g0 under the Morel option.

    Mirrors the branch at ``giop.m:57-67``: with no viewing geometry it uses the
    Appendix-B Q together with the tabulated f', forming f'/Q; with geometry it
    interpolates the full 5-D f/Q table.
    """
    if senz is None and relaz is None:
        _, q = morel_fq_appb(chl, solz)
        f = morel_read(chl, solz, "fp")
        fq = f / q
    else:
        fq, _ = morel_fq_geometry(chl, solz, senz or 0.0, relaz or 0.0)
    return interp1(_WL_OUT, fq, wl, method=method)


def morel_fq_appb(chl, solz):
    """Appendix B of Morel et al. (2002). Ported from ``morel_fq_appb.m``.

    Returns (f, Q) on the 380-700 nm 1 nm grid, boxcar-smoothed with width 25 exactly
    as upstream does.
    """
    dat = np.loadtxt(files("giop.data").joinpath("morel_fq_appb.txt"))
    f0, sf, q0, sq = dat[0:6], dat[6:12], dat[12:18], dat[18:24]

    wlut = np.array([412.5, 442.5, 490.0, 510.0, 560.0, 620.0, 660.0])
    z = 1.0 - np.cos(np.deg2rad(solz))
    f1 = f0 + sf * z
    q1 = q0 + sq * z

    # clamp the query wavelengths into the table span, as morel_fq_appb.m:21-24 does
    wq = np.clip(_WL_OUT, wlut[0], wlut[-1])
    cq = np.full_like(wq, float(chl))

    f = _interp2_clamped(wlut, _CHL_LUT, f1, wq, cq)
    q = _interp2_clamped(wlut, _CHL_LUT, q1, wq, cq)
    return fastsmooth(f, 25, 1, 1), fastsmooth(q, 25, 1, 1)


def morel_read(chl, solz, kind="fp"):
    """Read Morel's f, f' or mu_d LUT. Ported from ``morel_read.m``.

    File columns are solz, wl, then six chlorophyll columns [0.03 ... 10].
    The tables span 352.5-697.5 nm at 5 nm; upstream interpolates onto 380-700 nm
    and then back-fills the trailing NaNs above 697.5 nm with the last good value
    (``morel_read.m:82-84``), which is reproduced here.
    """
    fname = {"f": "morel_f.txt", "fp": "morel_fp.txt", "mud": "morel_mud.txt"}[kind]
    table = np.loadtxt(files("giop.data").joinpath(fname))

    per_solz = {}
    for s in _SOLZ_LUT:
        rows = table[table[:, 0] == s]
        rows = rows[np.argsort(rows[:, 1])]
        w = rows[:, 1]
        vals = rows[:, 2:8]
        per_solz[s] = _interp2_clamped(w, _CHL_LUT, vals.T, _WL_OUT,
                                       np.full_like(_WL_OUT, float(chl)),
                                       clamp_x=False)

    lo, hi = _bracket(solz, _SOLZ_LUT)
    if lo == hi:
        val = per_solz[lo]
    else:
        w = (solz - lo) / (hi - lo)
        val = (1.0 - w) * per_solz[lo] + w * per_solz[hi]

    return _fill_trailing_nans(val)


@functools.lru_cache(maxsize=1)
def read_fq():
    """The 5-D Morel f/Q table, shape (7 wl, 6 solz, 6 chl, 17 nadir, 13 azimuth).

    Ported from ``read_fq.m``; the flat file stores azimuth across columns and the
    other four axes down rows in C order.
    """
    data = np.loadtxt(files("giop.data").joinpath("morel_fq.dat"))
    return data.reshape(7, 6, 6, 17, 13)


def morel_fq_geometry(chl, solz, senz, relaz):
    """Full-geometry f/Q. Ports ``morel_fq.m`` + ``get_fq.m``.

    Returns ``(fq, fc)`` on the 380-700 nm grid, where ``fc`` is the ratio of the
    nadir-view, overhead-sun f/Q to the actual-geometry f/Q, i.e. the BRDF correction
    factor that normalises to nadir viewing.
    """
    foq = read_fq()
    # in-water nadir angle from Snell's law, morel_fq.m:12
    thetap = np.rad2deg(np.arcsin(np.sin(np.deg2rad(senz)) / _WATER_N))

    fq = np.array([_get_fq(i, solz, chl, thetap, relaz, foq) for i in range(7)])
    f0 = np.array([_get_fq(i, 0.0, chl, 0.0, 0.0, foq) for i in range(7)])
    fc = f0 / fq

    # PORTING_NOTES D9: morel_fq.m:26-27 interpolates the 412.5-660 nm LUT straight onto
    # 380:700 nm, so upstream returns NaN below 412.5 and above 660 nm and never fills
    # them, unlike morel_read.m which does. Those NaNs then propagate into g0 and make
    # every band NaN once they reach the cost function. Clamped here to the LUT edge,
    # which is what upstream's own Appendix-B branch does explicitly at
    # morel_fq_appb.m:21-24, so the two Morel branches now behave the same way.
    wq = np.clip(_WL_OUT, _WL_FQ[0], _WL_FQ[-1])
    return (interp1(_WL_FQ, fq, wq, "pchip"),
            interp1(_WL_FQ, fc, wq, "pchip"))


def brdf_factor(wl, chl, solz, senz, relaz, method="pchip"):
    """Morel f/Q BRDF normalisation factor. Multiply R_rs by this.

    R_rs measured at one sun-sensor geometry is not the R_rs a satellite product
    reports. Ocean-colour products are **exact normalised water-leaving reflectance**:
    what you would see looking straight down with the sun overhead. Field data taken at
    40 deg from nadir with the sun at 30-60 deg zenith is a different quantity, and
    comparing the two without correcting is comparing two different things.

    Because r_rs = (f/Q) u and u is a property of the water alone, the ratio of the
    two geometries is just the ratio of their f/Q:

        R_rs(nadir, sun overhead) = R_rs(theta_s, theta_v, dphi) x
                                    f/Q(0, 0) / f/Q(theta_s, theta_v', dphi)

    and that ratio is exactly the ``fc`` that :func:`morel_fq_geometry` already returns.
    Morel, Antoine & Gentili (2002), Applied Optics 41(30), 6289-6306.

    **What this does NOT include.** The full normalisation also carries a factor for the
    air-water transmittance and refraction, which depends on viewing angle. That term is
    a few percent at 40 deg, against an f/Q ratio that commonly reaches 10 % or more, so
    the dominant part is here and the remainder is not. Do not present output of this
    function as a complete NASA-style exact normalisation.
    """
    _, fc = morel_fq_geometry(chl, solz, senz, relaz)
    return interp1(_WL_OUT, fc, wl, method=method)


def normalize_brdf(wl, rrs, chl, solz, senz=40.0, relaz=135.0, method="pchip"):
    """Apply :func:`brdf_factor`. Returns ``(rrs_normalised, factor)``."""
    f = brdf_factor(wl, chl, solz, senz, relaz, method=method)
    return np.asarray(rrs, dtype=float) * f, f


def _get_fq(iw, s, chl_in, n, a, foq):
    """Quadrilinear interpolation in (solz, log chl, nadir, azimuth). ``get_fq.m``."""
    lchl = np.log(_CHL_LUT)
    c = np.log(float(chl_in))
    n = max(float(n), _NADIR_LUT[0])  # get_fq.m:61 forces the minimum nadir angle

    js, ds = _weights(s, _SOLZ_LUT)
    kc, dc = _weights(c, lchl)
    ln, dn = _weights(n, _NADIR_LUT)
    ma, da = _weights(a, _AZM_LUT)

    out = 0.0
    for j in range(2):
        for k in range(2):
            for l in range(2):
                for m in range(2):
                    out += (ds[j] * dc[k] * dn[l] * da[m]
                            * foq[iw, js + j, kc + k, ln + l, ma + m])
    return out


def _weights(v, grid):
    """Lower index and the (w_lo, w_hi) pair, clamping at both ends like get_fq.m."""
    nlast = len(grid) - 1
    if v <= grid[0]:
        return 0, (1.0, 0.0)
    if v >= grid[nlast]:
        return nlast - 1, (0.0, 1.0)
    i = int(np.searchsorted(grid, v, side="right") - 1)
    i = min(i, nlast - 1)
    span = grid[i + 1] - grid[i]
    return i, ((grid[i + 1] - v) / span, (v - grid[i]) / span)


def _bracket(v, grid):
    if v <= grid[0]:
        return grid[0], grid[0]
    if v >= grid[-1]:
        return grid[-1], grid[-1]
    i = int(np.searchsorted(grid, v, side="right") - 1)
    return grid[i], grid[i + 1]


def _interp2_clamped(x, y, z, xq, yq, clamp_x=True):
    """Bilinear interpolation on a (len(y), len(x)) grid, MATLAB ``interp2`` order.

    ``z`` is indexed [y, x]. Queries outside the grid are clamped rather than set to
    NaN in the chlorophyll direction, because upstream passes scalar chl values that
    routinely sit outside [0.03, 10].
    """
    z = np.asarray(z, dtype=float)
    if z.shape != (len(y), len(x)):
        z = z.T
    xq = np.atleast_1d(np.asarray(xq, dtype=float))
    yq = np.atleast_1d(np.asarray(yq, dtype=float))
    if clamp_x:
        xq = np.clip(xq, x[0], x[-1])
    yq = np.clip(yq, y[0], y[-1])
    itp = RegularGridInterpolator((y, x), z, bounds_error=False, fill_value=None)
    return itp(np.column_stack([yq, xq]))


def _fill_trailing_nans(v):
    v = np.asarray(v, dtype=float).copy()
    bad = ~np.isfinite(v)
    if bad.any():
        first_bad = int(np.flatnonzero(bad)[0])
        if first_bad > 0:
            v[bad] = v[first_bad - 1]
    return v
