"""Spectral resampling onto an inversion grid. THEORY.md sect. 10.4.

Resampling must be applied to R_rs, never to retrieved IOPs: the forward model (G1)
is nonlinear in the IOPs, so band averaging does not commute with the inversion.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

from ..matlab_compat import interp1

__all__ = ["align", "bin_spectrum", "gaussian_resample", "SATELLITE_BANDS"]

#: Nominal band centres for the sensors GIOP is routinely run on.
SATELLITE_BANDS = {
    "seawifs": [412.0, 443, 490, 510, 555, 670],
    "modisa":  [412.0, 443, 488, 531, 547, 667],
    "viirs":   [410.0, 443, 486, 551, 671],
    "meris":   [413.0, 443, 490, 510, 560, 620, 665],
    "oci":     [412.0, 443, 490, 510, 555, 670],
    "giop_demo": [412.0, 443, 490, 510, 555, 670],
}


def align(series, grid=None, method="pchip"):
    """Put several (wavelength, value) pairs on one grid.

    If every input already shares a grid the data are returned untouched, which keeps
    the common case (three scans from the same instrument) exact.
    """
    wls = [np.asarray(w, dtype=float) for w, _ in series]
    vals = [np.asarray(v, dtype=float) for _, v in series]

    if grid is None:
        same = all(w.shape == wls[0].shape and np.allclose(w, wls[0]) for w in wls)
        if same:
            return wls[0], vals
        lo = max(w.min() for w in wls)
        hi = min(w.max() for w in wls)
        if lo >= hi:
            raise ValueError(
                f"the supplied spectra do not overlap: intersection is [{lo}, {hi}] nm"
            )
        base = wls[int(np.argmax([len(w) for w in wls]))]
        grid = base[(base >= lo) & (base <= hi)]

    grid = np.asarray(grid, dtype=float)
    out = [interp1(w, v, grid, method=method) for w, v in zip(wls, vals)]
    return grid, out


def bin_spectrum(wl, values, centres, width=10.0, min_bands=1):
    """Boxcar-average onto ``centres`` using a full width of ``width`` nm.

    Returns ``(centres, binned, n_per_bin)``. Bins with fewer than ``min_bands``
    contributing channels come back NaN rather than being quietly extrapolated.
    """
    wl = np.asarray(wl, dtype=float)
    values = np.asarray(values, dtype=float)
    centres = np.atleast_1d(np.asarray(centres, dtype=float))

    out = np.full(centres.shape, np.nan)
    n = np.zeros(centres.shape, dtype=int)
    half = width / 2.0
    for i, c in enumerate(centres):
        m = (wl >= c - half) & (wl <= c + half) & np.isfinite(values)
        n[i] = int(m.sum())
        if n[i] >= min_bands:
            out[i] = float(np.mean(values[m]))
    return centres, out, n


def gaussian_resample(wl, values, centres, fwhm=10.0):
    """Convolve with a Gaussian response of the given FWHM at each centre.

    A reasonable stand-in for a real sensor response function when none is available.
    Weights are normalised over the bands actually present, so a centre near the edge
    of the measured range is biased; those come back NaN if more than half the weight
    falls outside the data.
    """
    wl = np.asarray(wl, dtype=float)
    values = np.asarray(values, dtype=float)
    centres = np.atleast_1d(np.asarray(centres, dtype=float))
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    good = np.isfinite(values)
    out = np.full(centres.shape, np.nan)
    lo, hi = wl.min(), wl.max()
    for i, c in enumerate(centres):
        # Fraction of the response that the measured range actually covers. This has
        # to be measured against the full Gaussian, not against the sum over the
        # available grid, or a centre outside the data trivially "covers" 100 %.
        coverage = 0.5 * (erf((hi - c) / (sigma * np.sqrt(2.0)))
                          - erf((lo - c) / (sigma * np.sqrt(2.0))))
        w = np.where(good, np.exp(-0.5 * ((wl - c) / sigma) ** 2), 0.0)
        if coverage > 0.5 and w.sum() > 0:
            out[i] = float(np.sum(w * np.nan_to_num(values)) / w.sum())
    return centres, out
