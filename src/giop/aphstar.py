"""Chlorophyll-specific phytoplankton absorption eigenvectors. THEORY.md sect. 4.1-4.4."""

from __future__ import annotations

import functools
import warnings
from importlib.resources import files

import numpy as np
import scipy.io as sio

from .matlab_compat import interp1

__all__ = ["bricaud1998", "ciotti2006", "gsm", "chase2017", "APH_MODELS"]

APH_MODELS = ("bricaud", "ciotti", "gsm", "chase")

#: Normalisation target for Bricaud aph*, GIOP-DC (m^2 mg^-1) at 442 nm.
BRICAUD_NORM_VALUE = 0.055
#: The Bricaud table is on a 2 nm grid that contains 442 and NOT 443. THEORY.md G8.
BRICAUD_NORM_WL = 442.0


@functools.lru_cache(maxsize=1)
def _bricaud_table():
    """Columns: wl (nm), A_p, E_p, A_ph, E_ph. 2 nm grid, 400-700 nm."""
    path = files("giop.data").joinpath("bricaud_1998_aph.txt")
    return np.loadtxt(path, delimiter=",")


def bricaud1998(chl, wl, normalize=True, method="pchip"):
    """Bricaud et al. (1998) aph* eigenvector. THEORY.md G7, G8.

    Ported from ``get_bricaud_aph.m``. Returns aph* in m^2 mg^-1.

    ``normalize=True`` rescales so aph*(442) = 0.055 m^2 mg^-1, which is what
    GIOP-DC uses and what ``giop.m:81`` requests via the ``'norm'`` flag.

    The eigenvector depends on ``chl`` because the Bricaud power-law exponent is
    wavelength dependent; GIOP freezes it at the seed chlorophyll and does not
    iterate (THEORY.md sect. 4.1).
    """
    dat = _bricaud_table()
    chl = float(chl)
    if chl <= 0:
        raise ValueError(f"Bricaud aph* needs chl > 0 (power law in chl); got {chl}")

    aphs0 = dat[:, 3] * chl ** (dat[:, 4] - 1.0)
    if normalize:
        idx = np.flatnonzero(dat[:, 0] == BRICAUD_NORM_WL)
        if idx.size != 1:
            raise RuntimeError("Bricaud table no longer contains exactly one 442 nm row")
        aphs0 = aphs0 * BRICAUD_NORM_VALUE / aphs0[idx[0]]
    return interp1(dat[:, 0], aphs0, wl, method=method)


def bricaud1998_all(chl, wl, normalize=True, method="pchip"):
    """Full ``get_bricaud_aph.m`` return: (ap, ap*, aph, aph*)."""
    dat = _bricaud_table()
    chl = float(chl)
    ap0 = dat[:, 1] * chl ** dat[:, 2]
    aps0 = dat[:, 1] * chl ** (dat[:, 2] - 1.0)
    aph0 = dat[:, 3] * chl ** dat[:, 4]
    aphs0 = dat[:, 3] * chl ** (dat[:, 4] - 1.0)
    if normalize:
        idx = int(np.flatnonzero(dat[:, 0] == BRICAUD_NORM_WL)[0])
        aphs0 = aphs0 * BRICAUD_NORM_VALUE / aphs0[idx]
    return tuple(interp1(dat[:, 0], v, wl, method=method) for v in (ap0, aps0, aph0, aphs0))


# --- Ciotti & Bricaud 2006 -------------------------------------------------------
# Tabulated end-member size-class spectra on a 2 nm grid, 400-700 nm, verbatim from
# get_ciotti_aph.m (the 2006 values; the file also carries a commented-out 2002 set).
_CIOTTI_W0 = np.arange(400, 701, 2, dtype=float)

_CIOTTI_PICO_RAW = np.array([
    1.7439, 1.8264, 1.9128, 1.9992, 2.0895, 2.1799, 2.2702, 2.3684, 2.4666, 2.5687,
    2.6669, 2.7612, 2.8437, 2.9183, 2.9890, 3.0479, 3.1029, 3.1500, 3.1854, 3.2089,
    3.2247, 3.2325, 3.2286, 3.2168, 3.1932, 3.1540, 3.1029, 3.0361, 2.9576, 2.8712,
    2.7848, 2.6944, 2.5137, 2.4273, 2.3488, 2.2781, 2.2486, 2.2192, 2.1720, 2.1328,
    2.1013, 2.0660, 2.0267, 1.9835, 1.9285, 1.8657, 1.7989, 1.7203, 1.6339, 1.5357,
    1.4336, 1.3276, 1.2176, 1.1076, 1.0016, 0.8994, 0.8013, 0.7109, 0.6284, 0.5538,
    0.4870, 0.4320, 0.3782, 0.3307, 0.2875, 0.2486, 0.2137, 0.1842, 0.1599, 0.1402,
    0.1233, 0.1080, 0.0935, 0.0789, 0.0656, 0.0530, 0.0424, 0.0344, 0.0290, 0.0260,
    0.0258, 0.0268, 0.0304, 0.0320, 0.0331, 0.0347, 0.0355, 0.0363, 0.0382, 0.0401,
    0.0416, 0.0428, 0.0432, 0.0432, 0.0432, 0.0424, 0.0416, 0.0408, 0.0408, 0.0424,
    0.0452, 0.0503, 0.0562, 0.0628, 0.0695, 0.0758, 0.0821, 0.0880, 0.0939, 0.1002,
    0.1060, 0.1123, 0.1178, 0.1229, 0.1261, 0.1280, 0.1288, 0.1296, 0.1308, 0.1331,
    0.1371, 0.1422, 0.1493, 0.1591, 0.1728, 0.1909, 0.2137, 0.2416, 0.2757, 0.3178,
    0.3692, 0.4281, 0.5499, 0.6009, 0.6324, 0.6402, 0.6324, 0.6245, 0.5892, 0.5342,
    0.4674, 0.3967, 0.3276, 0.2635, 0.2078, 0.1618, 0.1249, 0.0958, 0.0746, 0.0601,
    0.0503,
])

_CIOTTI_MICRO_RAW = np.array([
    1.574, 1.584, 1.600, 1.617, 1.633, 1.654, 1.669, 1.674, 1.684, 1.697,
    1.708, 1.710, 1.716, 1.737, 1.763, 1.793, 1.812, 1.827, 1.830, 1.834,
    1.824, 1.800, 1.771, 1.741, 1.712, 1.685, 1.667, 1.650, 1.641, 1.631,
    1.631, 1.623, 1.616, 1.606, 1.592, 1.568, 1.542, 1.509, 1.481, 1.459,
    1.437, 1.415, 1.399, 1.387, 1.377, 1.367, 1.349, 1.338, 1.319, 1.301,
    1.271, 1.242, 1.222, 1.196, 1.169, 1.141, 1.118, 1.096, 1.075, 1.057,
    1.035, 1.013, 0.992, 0.977, 0.959, 0.944, 0.927, 0.909, 0.888, 0.868,
    0.847, 0.826, 0.806, 0.785, 0.764, 0.737, 0.711, 0.682, 0.653, 0.626,
    0.604, 0.580, 0.555, 0.535, 0.514, 0.501, 0.487, 0.478, 0.475, 0.468,
    0.464, 0.459, 0.452, 0.452, 0.449, 0.443, 0.433, 0.424, 0.416, 0.406,
    0.401, 0.400, 0.403, 0.408, 0.416, 0.429, 0.443, 0.458, 0.473, 0.487,
    0.495, 0.499, 0.504, 0.514, 0.521, 0.525, 0.532, 0.535, 0.534, 0.535,
    0.532, 0.528, 0.526, 0.528, 0.538, 0.549, 0.574, 0.605, 0.655, 0.720,
    0.798, 0.889, 0.979, 1.068, 1.147, 1.207, 1.243, 1.249, 1.227, 1.174,
    1.096, 1.004, 0.893, 0.767, 0.635, 0.516, 0.409, 0.323, 0.253, 0.200,
    0.158,
])

#: Rescaling applied by get_ciotti_aph.m to put each end member at its aph*(443).
_CIOTTI_PICO = _CIOTTI_PICO_RAW * 0.023 / 0.891
_CIOTTI_MICRO = _CIOTTI_MICRO_RAW * 0.0086 / 1.249


def ciotti2006(wl, sf=0.5, method="pchip"):
    """Ciotti & Bricaud (2006) size-fraction aph*. THEORY.md G9.

    ``sf`` is the picoplankton size fraction in [0, 1]. Independent of chlorophyll.
    Ported from ``get_ciotti_aph.m``.
    """
    sf = float(sf)
    if not 0.0 <= sf <= 1.0:
        raise ValueError(f"Ciotti size fraction Sf must be in [0, 1]; got {sf}")
    a0 = sf * _CIOTTI_PICO + (1.0 - sf) * _CIOTTI_MICRO
    return interp1(_CIOTTI_W0, a0, wl, method=method)


# --- GSM -------------------------------------------------------------------------
_GSM_WL = np.array([412.0, 443.0, 490.0, 510.0, 555.0, 670.0])
_GSM_APHS = np.array([0.006650, 0.05582, 0.02055, 0.0191, 0.010150, 0.01424])


def gsm(wl, method="pchip"):
    """GSM fixed aph* (Maritorena et al. 2002). THEORY.md sect. 4.3.

    Upstream is inconsistent here: ``giop.m:143`` uses ``'cubic'`` and
    ``giop_kb.m:125`` uses ``'pchip'`` on the same six non-uniformly spaced points.
    ``'v5cubic'`` cannot act on a non-uniform grid at all, which is the most likely
    reason for the change. Default here is ``'pchip'``, matching ``giop_kb.m``.
    """
    return interp1(_GSM_WL, _GSM_APHS, wl, method=method)


# --- Chase 2017 ------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _chase_table():
    path = files("giop.data").joinpath("chase_ap17.mat")
    return sio.loadmat(path)["chase_ap17"]


def chase2017(chl, wl, method="pchip", warn=True):
    """Chase et al. (2017) hyperspectral particulate absorption. THEORY.md sect. 4.4.

    Ported from ``get_chase_ap.m`` (K. Bisson). Returns a_p, **total particulate
    absorption**, not a_phi. Feeding it to GIOP as the phytoplankton eigenvector
    double-counts the detrital fraction that a_dg already carries; upstream's
    ``giop_kb.m:190-192`` contains a commented-out attempt to deal with this. Exposed
    because upstream exposes it, with the warning made loud.
    """
    if warn:
        warnings.warn(
            "aph='chase' uses Chase et al. (2017) a_p (TOTAL particulate absorption) "
            "as the phytoplankton eigenvector. The detrital fraction is then counted "
            "twice, once here and once in a_dg. See THEORY.md sect. 4.4.",
            UserWarning,
            stacklevel=2,
        )
    dat = _chase_table()
    chl = float(chl)
    ap0 = dat[:, 1] * chl ** dat[:, 2]
    return interp1(dat[:, 0], ap0, wl, method=method)
