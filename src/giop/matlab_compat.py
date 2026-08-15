"""MATLAB semantics that NumPy/SciPy do not reproduce by default.

Everything here exists because a literal translation of the upstream MATLAB would
silently change numbers. See THEORY.md sect. 9 and PORTING_NOTES.md.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

__all__ = ["interp1", "v5cubic", "fastsmooth", "matlab_find_first", "nearest_band"]


def v5cubic(x, y, xq):
    """MATLAB ``interp1(...,'v5cubic')``: Keys cubic convolution on a uniform grid.

    This is the interpolant MATLAB's ``'cubic'`` selected in older releases. It is a
    4-point convolution with the Keys kernel at a = -0.5, it requires uniformly spaced
    ``x``, and unlike PCHIP it can overshoot. MATLAB extrapolates linearly-ish at the
    ends by mirroring one point; here we replicate MATLAB's documented behaviour of
    returning NaN outside ``[x[0], x[-1]]`` and using endpoint extension inside.

    Raises
    ------
    ValueError
        If ``x`` is not uniformly spaced, which is exactly the condition under which
        MATLAB refuses ``'v5cubic'`` (and, we infer, why upstream's ``giop_kb.m:125``
        switched the 6-band GSM aph* call to ``'pchip'``).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xq = np.atleast_1d(np.asarray(xq, dtype=float))

    h = np.diff(x)
    if not np.allclose(h, h[0], rtol=1e-9, atol=0.0):
        raise ValueError(
            "v5cubic requires a uniformly spaced grid (MATLAB refuses non-uniform x "
            "for 'v5cubic'); got spacings in "
            f"[{h.min()}, {h.max()}]. Use method='pchip'."
        )
    dx = h[0]

    # Extend by one node at each end so the 4-point stencil is defined everywhere,
    # using MATLAB's linear end extension y[-1] = 3y[0]-3y[1]+y[2].
    ye = np.concatenate(
        ([3 * y[0] - 3 * y[1] + y[2]], y, [3 * y[-1] - 3 * y[-2] + y[-3]])
    )

    s = (xq - x[0]) / dx           # fractional index into y
    i = np.floor(s).astype(int)    # left node
    t = s - i                      # offset in [0, 1)

    # clamp so the stencil stays in range; out-of-range handled by NaN mask below
    i = np.clip(i, 0, len(y) - 2)
    t = np.where((s < 0) | (s > len(y) - 1), 0.0, s - i)

    # Keys kernel, a = -0.5
    t2, t3 = t * t, t * t * t
    w0 = -0.5 * t3 + t2 - 0.5 * t
    w1 = 1.5 * t3 - 2.5 * t2 + 1.0
    w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    w3 = 0.5 * t3 - 0.5 * t2

    j = i + 1  # index into the extended array
    out = w0 * ye[j - 1] + w1 * ye[j] + w2 * ye[j + 1] + w3 * ye[j + 2]
    out = np.where((xq < x[0]) | (xq > x[-1]), np.nan, out)
    return out


def interp1(x, y, xq, method="pchip"):
    """MATLAB-compatible 1-D interpolation with explicit method selection.

    ``method='pchip'`` matches MATLAB's current ``'cubic'``/``'pchip'``;
    ``method='v5cubic'`` matches MATLAB's historical ``'cubic'``.
    Both return NaN outside the data range, as MATLAB does without ``'extrap'``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xq = np.asarray(xq, dtype=float)
    scalar = xq.ndim == 0
    xq = np.atleast_1d(xq)

    order = np.argsort(x)
    x, y = x[order], y[order]

    if method == "pchip":
        out = PchipInterpolator(x, y, extrapolate=False)(xq)
    elif method == "v5cubic":
        out = v5cubic(x, y, xq)
    elif method == "linear":
        out = np.interp(xq, x, y, left=np.nan, right=np.nan)
    else:
        raise ValueError(f"unknown interp1 method {method!r}")

    return float(out[0]) if scalar else out


def fastsmooth(y, w, smooth_type=1, ends=0):
    """Port of ``fastsmooth.m`` (T. C. O'Haver, 2008), used by ``morel_fq_appb.m``.

    The upstream file has CR-only line endings, so it renders as a single line and the
    sequence ``end`` + ``s(k+halfw)=...`` appears as the nonsense token
    ``ends(k+halfw)=...``. This is the reconstructed intent, which is the standard
    published fastsmooth: a sliding-average of width ``w`` applied ``smooth_type``
    times (1 = boxcar, 2 = triangular, 3 = pseudo-Gaussian).
    """
    y = np.asarray(y, dtype=float)
    out = y
    for _ in range(int(smooth_type)):
        out = _sliding_average(out, w, ends)
    return out


def _sliding_average(y, smoothwidth, ends):
    w = int(round(smoothwidth))
    L = len(y)
    s = np.zeros(L)
    halfw = int(round(w / 2))
    sum_points = float(np.sum(y[:w]))
    k = 0
    for k in range(L - w):
        s[k + halfw - 1] = sum_points
        sum_points = sum_points - y[k] + y[k + w]
    if L - w > 0:
        s[k + halfw] = float(np.sum(y[L - w:]))
    out = s / w

    if ends == 1:
        startpoint = (w + 1) // 2
        out[0] = (y[0] + y[1]) / 2.0
        for k in range(1, startpoint):
            out[k] = np.mean(y[: 2 * k + 1])
            out[L - k - 1] = np.mean(y[L - 2 * k - 1:])
        out[L - 1] = (y[L - 1] + y[L - 2]) / 2.0
    return out


def matlab_find_first(mask):
    """MATLAB ``find(mask)`` then ``(1)``: index of the first True, or None."""
    idx = np.flatnonzero(np.asarray(mask))
    return int(idx[0]) if idx.size else None


def nearest_band(wl, target, tol):
    """Index of the band nearest ``target``, requiring it within ``tol`` nm.

    Deliberately different from upstream, which takes the *first* index inside a
    window (``giop.m:30-32``). On a 6-band satellite grid the two agree; on a 1 nm
    hyperspectral grid "first in window" picks the window's blue edge, which is
    arbitrary. See THEORY.md sect. 9.3 / PORTING_NOTES.md D5.
    """
    wl = np.asarray(wl, dtype=float)
    d = np.abs(wl - target)
    i = int(np.argmin(d))
    return i if d[i] <= tol else None
