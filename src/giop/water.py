"""Pure-seawater inherent optical properties. THEORY.md G5, G6."""

from __future__ import annotations

import functools
from importlib.resources import files

import numpy as np

from .matlab_compat import interp1

__all__ = ["a_water", "bb_water", "optics_coef_table", "f0_solar"]


@functools.lru_cache(maxsize=1)
def optics_coef_table():
    """``optics_coef.txt``: 1 nm grid, 380-1150 nm.

    Column 1 is wavelength (nm) and column 2 is pure-water absorption (m^-1); those
    are the only two the upstream code reads (``get_aw.m``). Column 3 is total pure
    water scattering b_w (m^-1) -- verified: b_w/2 at 443 nm is 0.0024362 against the
    0.0024447 that the (G6) power law gives, 0.35 % apart.

    **Column 7 is extraterrestrial solar irradiance F0**, in mW cm^-2 um^-1. Identified
    from the data rather than assumed: the column peaks at 451 nm, and it carries the
    solar Fraunhofer lines at their correct depths -- a 41 % dip across the Ca II H&K
    doublet at 390-400 nm, 9 % at H-alpha, 6 % at the Mg b triplet -- with a
    band-to-band maximum-to-median step ratio of 101. A smooth parameterisation cannot
    produce those. See :func:`f0_solar`.

    Columns 4-6 are not used by GIOP and are not identified here rather than guessed at.
    """
    path = files("giop.data").joinpath("optics_coef.txt")
    return np.loadtxt(path)


def a_water(wl, method="pchip"):
    """Pure-seawater absorption a_w (m^-1). THEORY.md G5, from ``get_aw.m``.

    Source of the table, per the companion ``pureH2O_iop.mat`` NOTES field:
    Pope & Fry (1997) over 380-700 nm, Smith & Baker (1981) below 380 nm,
    Kou et al. (1993) above 700 nm.

    Returns NaN outside 380-1150 nm, as MATLAB's ``interp1`` does.
    """
    tab = optics_coef_table()
    return interp1(tab[:, 0], tab[:, 1], wl, method=method)


def bb_water(wl, model="morel1974", bbw_table=None):
    """Pure-seawater backscattering b_bw (m^-1). THEORY.md G6.

    ``model='morel1974'`` (default, and what upstream uses in ``get_bbw.m``)::

        b_bw = 0.0038 * (400 / wl) ** 4.32

    ``model='table'`` interpolates a user-supplied ``bbw_table`` of shape (N, 2)
    holding [wavelength_nm, b_bw]. This is the supported route for using the modern
    salinity- and temperature-dependent seawater backscattering of Zhang, Hu & He
    (2009), doi:10.1364/OE.17.005698, which runs above the Morel fit in the blue at
    oceanic salinity (THEORY.md A8). That formulation is **not** reimplemented here:
    it needs the density derivative of the refractive index, the isothermal
    compressibility and the water activity, and a paraphrase of it would be a
    fabrication wearing a citation. Generate the table with a package that implements
    it, or supply measured values, and pass it in.
    """
    wl = np.asarray(wl, dtype=float)
    if model == "morel1974":
        return 0.0038 * (400.0 / wl) ** 4.32
    if model == "table":
        if bbw_table is None:
            raise ValueError("model='table' requires bbw_table=[[wl, bbw], ...]")
        t = np.asarray(bbw_table, dtype=float)
        return interp1(t[:, 0], t[:, 1], wl, method="pchip")
    raise ValueError(f"unknown b_bw model {model!r}; use 'morel1974' or 'table'")


#: mW cm^-2 um^-1  ->  W m^-2 nm^-1
_F0_TO_SI = 0.01


def f0_solar(wl, method="pchip"):
    """Extraterrestrial solar irradiance F0 at mean Earth-Sun distance, W m^-2 nm^-1.

    Read from ``optics_coef.txt`` column 7 and converted from the mW cm^-2 um^-1 the
    table uses. Sanity: this gives F0(443) = 1.945 W m^-2 nm^-1, against a published
    value near 1.90.

    F0 is what converts a reflectance into a radiance. The standard satellite
    ocean-colour product is normalised water-leaving radiance, nLw = R_rs x F0, so
    without F0 a spectroradiometer's absolute calibration cannot be turned into the
    quantity the satellite community actually exchanges.

    NOTE: no Earth-Sun distance correction is applied. F0 here is the mean-distance
    value, which is the convention for normalised products; the annual variation is
    about +/-3.4 %.
    """
    tab = optics_coef_table()
    return interp1(tab[:, 0], tab[:, 6] * _F0_TO_SI, wl, method=method)
