"""GIOP: Generalized Ocean Colour Inversion Model, Python port.

Port of the reference MATLAB GIOP (J. Werdell, NASA GSFC, 2013; fork by K. Bisson,
OSU, 2019) at https://github.com/kelseybisson/GIOP.

Reference: Werdell et al. (2013), Applied Optics 52(10), 2019-2037,
doi:10.1364/AO.52.002019.

The governing equations, the assumption ledger, and the equation-to-code map are in
THEORY.md. Deviations from upstream are enumerated in PORTING_NOTES.md.

Quick start::

    from giop import giop, get_oc
    wl  = [412, 443, 490, 510, 555, 670]
    rrs = [0.003478, 0.004074, 0.004465, 0.003588, 0.002494, 0.000051]
    chl = get_oc(rrs[1], rrs[2], rrs[3], rrs[4], 'oc4')
    res = giop(wl, rrs, chl, qc=0.33)
    res.x            # (a_dg(443), b_bp(443), M_phi)
"""

from .empirical import get_oc, qaa_bbp, raman_correction
from .inversion import FILL, GiopResult, giop
from .model import ConfigurationError, GiopConfig
from .water import a_water, bb_water

__version__ = "0.1.0"

__all__ = [
    "giop", "GiopConfig", "GiopResult", "ConfigurationError", "FILL",
    "get_oc", "qaa_bbp", "raman_correction",
    "a_water", "bb_water",
    "__version__",
]
