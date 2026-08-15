"""Above-water field radiometry: field scans -> R_rs. THEORY.md sect. 10.

This layer has no counterpart upstream, which starts from R_rs. It is where the
NaturaSpec data becomes an ocean-colour measurement, and it carries the single largest
error term in the whole chain (the sky-glint factor rho, THEORY.md A10).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

__all__ = [
    "rrs_above_water",
    "reflectance_factor_to_rrs",
    "RHO_MOBLEY1999",
    "residual_correction",
    "RrsResult",
]

#: Mobley (1999) effective air-water radiance reflectance for the recommended
#: geometry: theta_v = 40 deg from nadir, delta-phi = 135 deg from the sun, wind
#: below ~5 m/s, clear sky. doi:10.1364/AO.38.007442.
RHO_MOBLEY1999 = 0.028

#: Ruddick et al. (2006) NIR similarity ratio Rrs(780)/Rrs(870).
SIMILARITY_RATIO_780_870 = 1.912


@dataclass
class RrsResult:
    wavelength: np.ndarray
    rrs: np.ndarray
    ed: np.ndarray | None
    lw: np.ndarray | None          # water-leaving radiance, Lt - rho*Lsky
    rho: float
    offset: float                  # the residual correction actually subtracted
    method: str
    notes: list


def rrs_above_water(
    wavelength,
    l_target,
    l_sky,
    l_panel,
    panel_reflectance=0.99,
    rho=RHO_MOBLEY1999,
    residual="none",
    residual_window=(750.0, 800.0),
):
    """Three-measurement above-water R_rs. THEORY.md G16, G17.

    Parameters
    ----------
    wavelength : array, nm, common to all three scans
    l_target : array, total upwelling radiance from the water, L_t
    l_sky : array, sky radiance at the mirror-image angle, L_sky
    l_panel : array, radiance from the reference panel, L_p
    panel_reflectance : float or array
        Spectralon panel reflectance R_p. A scalar is fine for a well-characterised
        panel; supply the calibration spectrum if you have it. This multiplies every
        output value directly (THEORY.md A11).
    rho : float
        Effective air-water interface reflectance for radiance. The default 0.028 is
        Mobley (1999) for the recommended viewing geometry in low wind. It is **not**
        the Fresnel coefficient and it is sea-state dependent; at 10 m/s it can be
        0.04-0.06 and the error propagates straight into blue R_rs.
    residual : {'none', 'nir_zero', 'nir_similarity'}
        Residual glint/offset correction, see :func:`residual_correction`.

    Returns
    -------
    RrsResult
    """
    wl = np.asarray(wavelength, dtype=float)
    lt = np.asarray(l_target, dtype=float)
    ls = np.asarray(l_sky, dtype=float)
    lp = np.asarray(l_panel, dtype=float)
    for name, arr in (("l_sky", ls), ("l_panel", lp)):
        if arr.shape != lt.shape:
            raise ValueError(
                f"{name} has shape {arr.shape} but l_target has {lt.shape}; the three "
                "scans must be on the same wavelength grid (use io.resample.align)"
            )

    notes = []
    rp = np.asarray(panel_reflectance, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        ed = np.pi * lp / rp                  # G16
        lw = lt - rho * ls                    # numerator of G17
        rrs = lw / ed

    bad = ~np.isfinite(rrs)
    if bad.any():
        notes.append(f"{bad.sum()} bands non-finite (panel radiance at or below zero)")

    neg = np.sum(lw < 0)
    if neg:
        notes.append(
            f"{neg} bands have L_t < rho*L_sky, i.e. the sky-glint subtraction "
            "exceeds the measured upwelling radiance. Either rho is too large for "
            "the sea state or the sky scan does not match the target geometry."
        )

    offset, rmethod, more = residual_correction(wl, rrs, residual, residual_window)
    notes.extend(more)

    return RrsResult(wavelength=wl, rrs=rrs - offset, ed=ed, lw=lw, rho=float(rho),
                     offset=float(offset), method=rmethod, notes=notes)


def residual_correction(wl, rrs, method="none", window=(750.0, 800.0)):
    """Residual glint / offset correction. THEORY.md sect. 10.3.

    Returns ``(offset, method_used, notes)``. The offset is a scalar to subtract.

    ``'nir_zero'`` assumes R_rs is zero in the NIR window, which is a **property of
    the water body, not of the instrument**: it is reasonable in clear oceanic water
    and wrong in turbid or sediment-laden water, where it removes real signal. Default
    is ``'none'`` so that this choice is always made deliberately.
    """
    notes = []
    if method in (None, "none"):
        return 0.0, "none", notes

    wl = np.asarray(wl, dtype=float)
    rrs = np.asarray(rrs, dtype=float)

    if method == "nir_zero":
        m = (wl >= window[0]) & (wl <= window[1]) & np.isfinite(rrs)
        if not m.any():
            raise ValueError(
                f"nir_zero needs bands in {window} nm; the spectrum spans "
                f"{wl.min():.0f}-{wl.max():.0f} nm"
            )
        off = float(np.nanmean(rrs[m]))
        notes.append(
            f"nir_zero subtracted {off:.3e} sr^-1, the mean of "
            f"{window[0]:.0f}-{window[1]:.0f} nm. Invalid in turbid water, where NIR "
            "R_rs is genuinely non-zero."
        )
        return off, "nir_zero", notes

    if method == "nir_similarity":
        i780 = int(np.argmin(np.abs(wl - 780.0)))
        i870 = int(np.argmin(np.abs(wl - 870.0)))
        if abs(wl[i780] - 780) > 10 or abs(wl[i870] - 870) > 10:
            raise ValueError(
                "nir_similarity needs bands near 780 and 870 nm; nearest available "
                f"are {wl[i780]:.1f} and {wl[i870]:.1f} nm"
            )
        a = SIMILARITY_RATIO_780_870
        off = float((a * rrs[i870] - rrs[i780]) / (a - 1.0))
        notes.append(
            f"nir_similarity subtracted {off:.3e} sr^-1 using the Ruddick et al. "
            f"(2006) ratio {a} at {wl[i780]:.0f}/{wl[i870]:.0f} nm."
        )
        return off, "nir_similarity", notes

    raise ValueError(f"unknown residual method {method!r}")


def reflectance_factor_to_rrs(reflectance_factor, panel_reflectance=1.0):
    """Convert a target/panel reflectance factor to R_rs, assuming Lambertian water.

    R_rs = R_factor * R_panel / pi

    **This is not an above-water ocean-colour measurement.** It contains no sky-glint
    subtraction, so on a water surface it is L_t/E_d, which includes the reflected sky
    and is larger than R_rs by a factor that can exceed 2 in the blue. It is the right
    conversion for a diffuse Lambertian target (soil, vegetation, a panel), and it is
    provided for that case and for quick looks. For water use
    :func:`rrs_above_water`.
    """
    r = np.asarray(reflectance_factor, dtype=float)
    # A diffuse reflectance factor is <= ~1, and pi is the value that maps to R_rs = 1;
    # percent data lands in the 1-100 range, so 10 separates the two without firing on
    # legitimate bright or specular targets.
    if np.nanmax(r) > 10.0:
        warnings.warn(
            f"reflectance factor reaches {np.nanmax(r):.1f}; if this is percent, "
            "divide by 100 first (SedSpectrum.reflectance already does).",
            UserWarning, stacklevel=2,
        )
    return r * np.asarray(panel_reflectance, dtype=float) / np.pi


def rrs_from_sed_triplet(
    target, sky, panel, panel_reflectance=0.99, rho=RHO_MOBLEY1999,
    residual="none", use="radiance",
):
    """Build R_rs from three :class:`~giop.io.sed.SedSpectrum` scans.

    ``use='radiance'`` reads the ``Rad. (Target)`` column of each file, which is the
    cleanest path and requires the scans to be radiometrically calibrated.

    ``use='ratio'`` instead reads each file's reflectance column, interpreting them as
    target-over-panel ratios. This works when only reflectance was exported, and it
    requires that **all three scans used the same reference panel**, because the panel
    radiance then cancels:  R_rs = (Rf_target - rho * Rf_sky) * R_panel / pi.
    """
    from .resample import align

    if use == "radiance":
        wl, (lt, ls, lp) = align(
            [(target.wavelength, target.radiance_target),
             (sky.wavelength, sky.radiance_target),
             (panel.wavelength, panel.radiance_target)]
        )
        return rrs_above_water(wl, lt, ls, lp, panel_reflectance, rho, residual)

    if use == "ratio":
        wl, (rt, rs) = align(
            [(target.wavelength, target.reflectance),
             (sky.wavelength, sky.reflectance)]
        )
        rp = np.asarray(panel_reflectance, dtype=float)
        rrs = (rt - rho * rs) * rp / np.pi
        return RrsResult(
            wavelength=wl, rrs=rrs, ed=None, lw=None, rho=float(rho), offset=0.0,
            method=f"ratio+{residual}",
            notes=["ratio path assumes all scans share one reference panel"],
        )

    raise ValueError(f"use must be 'radiance' or 'ratio'; got {use!r}")
