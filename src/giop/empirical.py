"""Empirical band-ratio algorithms carried alongside GIOP. THEORY.md sect. 8."""

from __future__ import annotations

import warnings

import numpy as np

from .matlab_compat import interp1

__all__ = ["get_oc", "qaa_bbp", "raman_correction", "OC_ALGORITHMS"]

# Coefficient table from get_oc.m:35-55 (OC v6, operational 2009).
_OC_COEF = {
    "oc4":     (0.3272, -2.9940, 2.7218, -1.2259, -0.5683),
    "oc4e":    (0.3255, -2.7677, 2.4409, -1.1288, -0.4990),
    "oc4o":    (0.3325, -2.8278, 3.0939, -2.0917, -0.0257),
    "oc3s":    (0.2515, -2.3798, 1.5823, -0.6372, -0.5692),
    "oc3m":    (0.2424, -2.7423, 1.8017,  0.0015, -1.2280),
    "oc3e":    (0.2521, -2.2146, 1.5193, -0.7702, -0.4291),
    "oc3o":    (0.2399, -2.0825, 1.6126, -1.0848, -0.2083),
    "oc3c":    (0.3330, -4.3770, 7.6267, -7.1457,  1.6673),
    "oc2s":    (0.2511, -2.0853, 1.5035, -3.1747,  0.3383),
    "oc2e":    (0.2389, -1.9369, 1.7627, -3.0777, -0.1054),
    "oc2o":    (0.2236, -1.8296, 1.9094, -2.9481, -0.1718),
    "oc2m":    (0.2500, -2.4752, 1.4061, -2.8237,  0.5405),
    "oc2m-hi": (0.1464, -1.7953, 0.9718, -0.8319, -0.8073),
    "kd2s":   (-0.8515, -1.8263, 1.8714, -2.4414, -1.0690),
    "kd2m":   (-0.8813, -2.0584, 2.5878, -3.4885, -1.5061),
    "kd2e":   (-0.8641, -1.6549, 2.0112, -2.5174, -1.1035),
    "kd2o":   (-0.8878, -1.5135, 2.1459, -2.4943, -1.1043),
    "kd2c":   (-1.1358, -2.1146, 1.6474, -1.1428, -0.6190),
    "oc3v":    (0.2228, -2.4683, 1.5867, -0.4275, -0.7768),
    "oc2v":    (0.2230, -2.1807, 1.4434, -3.1709,  0.5863),
    "kd2v":   (-0.8730, -1.8912, 1.8021, -2.3865, -1.0453),
}

#: How each algorithm forms its band ratio: 'max3' = max(r1,r2,r3)/r4, etc.
_OC_RATIO = {
    "oc4": "max3", "oc4e": "max3", "oc4o": "max3",
    "oc3s": "max2", "oc3m": "max2", "oc3e": "max2", "oc3o": "max2",
    "oc3c": "max2", "oc3v": "max2",
    "oc2s": "r2r4", "oc2e": "r2r4", "oc2o": "r2r4", "oc2m": "r2r4",
    "oc2m-hi": "r2r4", "oc2v": "r2r4",
    "kd2s": "r2r4", "kd2m": "r2r4", "kd2e": "r2r4", "kd2o": "r2r4", "kd2v": "r2r4",
    "kd2c": "r2r3",
}

OC_ALGORITHMS = tuple(sorted(_OC_COEF))


def get_oc(r1, r2, r3, r4, algorithm="oc4"):
    """Empirical OC chlorophyll / KD algorithms. THEORY.md G14, from ``get_oc.m``.

    Arguments are R_rs at, in order: blue (~443), blue (~488/490), green (~510/531),
    green (~547/555/560). Pass -1 for a band the chosen algorithm does not use.
    ``algorithm`` defaults to OC4 (SeaWiFS), which is what ``run_giop.m`` uses.
    """
    key = str(algorithm).lower()
    if key not in _OC_COEF:
        # get_oc.m's switch falls through to OC4 for anything unrecognised; being
        # explicit is better than silently returning a different product.
        raise ValueError(f"unknown algorithm {algorithm!r}; one of {OC_ALGORITHMS}")
    a = _OC_COEF[key]
    r1, r2, r3, r4 = (np.asarray(v, dtype=float) for v in (r1, r2, r3, r4))

    mode = _OC_RATIO[key]
    if mode == "max3":
        num = np.maximum(np.maximum(r1, r2), r3)
        den = r4
    elif mode == "max2":
        num, den = np.maximum(r1, r2), r4
    elif mode == "r2r4":
        num, den = r2, r4
    else:
        num, den = r2, r3

    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log10(num / den)
        out = 10.0 ** (a[0] + a[1] * x + a[2] * x**2 + a[3] * x**3 + a[4] * x**4)
    return out


def qaa_bbp(rrs443, rrs490, rrs55x, rrs670, lambda_out=700.0):
    """QAA v6 particulate backscattering. THEORY.md sect. 8.2.

    Port of ``estimate_bbp_from_Rrs.m`` (N. Haentjens). Inputs are above-surface R_rs
    in sr^-1; output is b_bp at ``lambda_out`` in m^-1. The reference band switches
    from 55x to 670 nm when R_rs(670) >= 0.0015 sr^-1 (the QAA v5/v6 case test).
    """
    R443, R490, R55x, R670 = (np.asarray(v, dtype=float)
                              for v in (rrs443, rrs490, rrs55x, rrs670))

    def below(R):
        return R / (0.52 + 1.7 * R)

    r443, r490, r55x, r670 = below(R443), below(R490), below(R55x), below(R670)

    g0, g1 = 0.089, 0.1245
    u55x = (-g0 + np.sqrt(g0**2 + 4 * g1 * r55x)) / (2 * g1)
    u670 = (-g0 + np.sqrt(g0**2 + 4 * g1 * r670)) / (2 * g1)

    h0, h1, h2 = -1.14590292783408, -1.36582826429176, -0.469266027944581
    aw55x, bbw55x = 0.0596, 0.0009
    with np.errstate(divide="ignore", invalid="ignore"):
        X = np.log10((r443 + r490) / (r55x + 5 * (r670 / r490) * r670))
    a55x = aw55x + 10.0 ** (h0 + h1 * X + h2 * X**2)
    bbp55x = (u55x * a55x) / (1 - u55x) - bbw55x

    aw670, bbw670 = 0.439, 0.00034
    a670 = aw670 + 0.39 * (R670 / (R443 + R490)) ** 1.14
    bbp670 = (u670 * a670) / (1 - u670) - bbw670

    use_v5 = R670 < 0.0015
    lambda0 = np.where(use_v5, 555.0, 670.0)
    bbp0 = np.where(use_v5, bbp55x, bbp670)

    n = 2.0 * (1 - 1.2 * np.exp(-0.9 * r443 / r55x))
    return bbp0 * (lambda0 / lambda_out) ** n


# Lee et al. (2013) Raman coefficients, defined at these six bands only.
_RAMAN_WL = np.array([412.0, 443.0, 488.0, 531.0, 555.0, 667.0])
_RAMAN_A = np.array([0.003, 0.004, 0.011, 0.015, 0.017, 0.018])
_RAMAN_B1 = np.array([0.014, 0.015, 0.010, 0.010, 0.010, 0.010])
_RAMAN_B2 = np.array([-0.022, -0.023, -0.051, -0.070, -0.080, -0.081])


def raman_correction(wl, rrs, rrs443, rrs555, interpolate=False):
    """Remove the Raman contribution from R_rs. THEORY.md G15.

    Extracted from ``single_wv_comps.m:105-117``. Returns corrected R_rs.

    The published coefficients exist at six discrete bands. On any other grid they
    have to be interpolated, which the six-point band-specific fits do not license;
    ``interpolate=True`` will do it and warn. GIOP does not apply this by default and
    neither does this package.
    """
    wl = np.asarray(wl, dtype=float)
    rrs = np.asarray(rrs, dtype=float)

    if wl.shape == _RAMAN_WL.shape and np.allclose(wl, _RAMAN_WL, atol=6.0):
        a, b1, b2 = _RAMAN_A, _RAMAN_B1, _RAMAN_B2
    elif interpolate:
        warnings.warn(
            "Lee et al. (2013) Raman coefficients are band-specific fits at "
            "412/443/488/531/555/667 nm. Interpolating them onto another grid is an "
            "extrapolation of the fit, not of a physical function. THEORY.md sect. 8.3.",
            UserWarning, stacklevel=2,
        )
        a = interp1(_RAMAN_WL, _RAMAN_A, wl, "pchip")
        b1 = interp1(_RAMAN_WL, _RAMAN_B1, wl, "pchip")
        b2 = interp1(_RAMAN_WL, _RAMAN_B2, wl, "pchip")
    else:
        raise ValueError(
            "Raman coefficients are defined at 412/443/488/531/555/667 nm. Supply "
            "that band set, or pass interpolate=True and accept the caveat."
        )

    rf = a * (rrs443 / rrs555) + b1 * np.asarray(rrs555, float) ** b2
    return rrs / (1.0 + rf)
