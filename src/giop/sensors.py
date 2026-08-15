"""Satellite band sets and spectral convolution: what each sensor would see.

A hyperspectral field spectrum is not what a satellite measures. The satellite integrates
over a band with a finite spectral response, so comparing a 1 nm field R_rs to a satellite
product without convolving first compares two different quantities. This module does the
convolution, and is explicit about how well it can do it for each sensor.

TWO TIERS OF FIDELITY, and the difference matters:

* **MODIS-Aqua** ships a MEASURED spectral response function (16 bands, 380-2199 nm) in
  ``spectralresponse_modisa.mat``, distributed with the reference GIOP code. Convolution
  for MODIS uses that measured response.
* **Every other sensor** here is a NOMINAL centre and bandwidth from mission
  documentation, convolved with a Gaussian or a boxcar. That is a good approximation for
  the ocean-colour bands, which are near-rectangular and narrow, but it is not the real
  instrument response and is labelled as such in :data:`SENSORS` and in the GUI.

Nothing here is fitted or tuned. If you need the real SRF for OLCI or OCI, download it
from the mission and pass it to :func:`convolve_with_srf`.
"""

from __future__ import annotations

import functools
from importlib.resources import files

import numpy as np
import scipy.io as sio

from .matlab_compat import interp1

__all__ = ["SENSORS", "SensorBand", "Sensor", "modis_srf", "convolve",
           "convolve_with_srf", "band_table"]


class SensorBand(object):
    __slots__ = ("centre", "fwhm", "label")

    def __init__(self, centre, fwhm, label=None):
        self.centre = float(centre)
        self.fwhm = float(fwhm)
        self.label = label or ("%g" % centre)

    def __repr__(self):
        return "SensorBand(%g nm, FWHM %g nm)" % (self.centre, self.fwhm)


class Sensor(object):
    def __init__(self, name, bands, response, note, giop_bands=None):
        self.name = name
        self.bands = [b if isinstance(b, SensorBand) else SensorBand(*b) for b in bands]
        self.response = response          # 'measured' or 'gaussian'
        self.note = note
        #: The subset used for GIOP retrievals (visible, water-leaving).
        self.giop_bands = giop_bands or [b.centre for b in self.bands if b.centre <= 720]

    @property
    def centres(self):
        return [b.centre for b in self.bands]

    def __repr__(self):
        return "Sensor(%s, %d bands, %s response)" % (
            self.name, len(self.bands), self.response)


# --- MODIS-Aqua: MEASURED response, columns 1-16 of the shipped table ---------------
# Column order verified against the file: centroids 416.3, 442.6, 466.1, 487.5, 530.2,
# 547.2, 553.9, 645.8, 667.2, 678.5, 745.3, 856.9, 866.9, 1241.5, 1628.1, 2114.0 nm.
_MODIS_LABELS = [412, 443, 469, 488, 531, 547, 555, 645, 667, 678, 748, 859, 869,
                 1240, 1640, 2130]


@functools.lru_cache(maxsize=1)
def modis_srf():
    """(wavelength, response[n_wl, 16], labels) for MODIS-Aqua. Measured, not modelled."""
    a = sio.loadmat(files("giop.data").joinpath("spectralresponse_modisa.mat"))
    a = a["spectralresponsemodisa"]
    wl = a[:, 0]
    ok = np.isfinite(wl)
    return wl[ok], a[ok, 1:], list(_MODIS_LABELS)


SENSORS = {}


def _add(name, bands, response, note, giop_bands=None):
    SENSORS[name] = Sensor(name, bands, response, note, giop_bands)


_add("MODIS-Aqua",
     [(412, 15), (443, 10), (469, 20), (488, 10), (531, 10), (547, 10), (555, 20),
      (645, 50), (667, 10), (678, 10), (748, 10), (859, 35), (869, 15),
      (1240, 20), (1640, 25), (2130, 50)],
     "measured",
     "Measured SRF shipped with the reference GIOP code. The FWHM values listed here "
     "are indicative only; the convolution uses the full measured response curve.",
     giop_bands=[412, 443, 488, 531, 547, 667])

_add("Sentinel-3 OLCI",
     [(400, 15), (412.5, 10), (442.5, 10), (490, 10), (510, 10), (560, 10), (620, 10),
      (665, 10), (673.75, 7.5), (681.25, 7.5), (708.75, 10), (753.75, 7.5),
      (778.75, 15), (865, 20), (885, 10), (900, 10)],
     "gaussian",
     "Nominal Oa1-Oa21 band centres and widths from ESA mission documentation, "
     "convolved with a Gaussian. Oa13-Oa15 (the O2-A band trio) are omitted: they are "
     "atmospheric, very narrow, and not used for water-leaving retrievals.",
     giop_bands=[412.5, 442.5, 490, 510, 560, 665])

_add("PACE OCI (hyperspectral)",
     [(w, 5.0) for w in range(400, 721, 5)],
     "gaussian",
     "OCI is hyperspectral: ~5 nm FWHM sampled about every 2.5 nm from the UV to the "
     "NIR. Represented here on a 5 nm grid over 400-720 nm, which is the range GIOP's "
     "aph* eigenvectors cover. Nominal Gaussian response, not a measured SRF.",
     giop_bands=[412, 443, 490, 510, 555, 670])

_add("PACE OCI (blue-band subset)",
     [(412, 5), (443, 5), (490, 5), (510, 5), (555, 5), (670, 5)],
     "gaussian",
     "The heritage six-band subset OBPG also produces from OCI, for continuity with "
     "SeaWiFS and MODIS. Use this to compare like with like against older products.",
     giop_bands=[412, 443, 490, 510, 555, 670])

_add("VIIRS-SNPP",
     [(410, 20), (443, 15), (486, 20), (551, 20), (671, 20), (745, 15), (862, 39)],
     "gaussian",
     "Nominal M1-M7 band centres and widths. Note VIIRS has no 510 nm band, so the "
     "GIOP band set is one narrower than SeaWiFS or MODIS.",
     giop_bands=[410, 443, 486, 551, 671])

_add("SeaWiFS",
     [(412, 20), (443, 20), (490, 20), (510, 20), (555, 20), (670, 20), (765, 40),
      (865, 40)],
     "gaussian",
     "Nominal band centres and widths. This is the band set the published GIOP demo "
     "uses, so it is the one to pick when reproducing the reference numbers.",
     giop_bands=[412, 443, 490, 510, 555, 670])

_add("Field hyperspectral (native)",
     [], "none",
     "No convolution: invert the spectrum on its own wavelength grid. Use this for "
     "NaturaSpec data when you want the full spectral information rather than a "
     "satellite simulation.")


def convolve(wl, rrs, sensor_name, use_giop_bands=True, min_coverage=0.5):
    """Convolve a spectrum to a sensor's bands.

    Returns ``(centres, values, info)``. Bands whose response is more than
    ``1 - min_coverage`` outside the measured spectral range come back NaN rather than
    being extrapolated, and ``info`` records which and why.
    """
    sensor = SENSORS[sensor_name]
    wl = np.asarray(wl, dtype=float)
    rrs = np.asarray(rrs, dtype=float)

    if sensor.response == "none":
        return wl.copy(), rrs.copy(), {"method": "native", "dropped": []}

    if sensor.response == "measured" and sensor_name == "MODIS-Aqua":
        centres, values, dropped = _convolve_modis(wl, rrs, sensor, use_giop_bands,
                                                   min_coverage)
        return centres, values, {"method": "measured SRF", "dropped": dropped}

    wanted = sensor.giop_bands if use_giop_bands else sensor.centres
    bands = [b for b in sensor.bands if b.centre in wanted] or sensor.bands
    centres, values, dropped = [], [], []
    for b in bands:
        v, why = _gaussian_band(wl, rrs, b.centre, b.fwhm, min_coverage)
        centres.append(b.centre)
        values.append(v)
        if why:
            dropped.append((b.centre, why))
    return (np.array(centres), np.array(values),
            {"method": "nominal Gaussian", "dropped": dropped})


def _convolve_modis(wl, rrs, sensor, use_giop_bands, min_coverage):
    srf_wl, srf, labels = modis_srf()
    wanted = sensor.giop_bands if use_giop_bands else [float(x) for x in labels]
    centres, values, dropped = [], [], []
    for j, lab in enumerate(labels):
        if float(lab) not in [float(x) for x in wanted]:
            continue
        r = srf[:, j]
        m = np.isfinite(r) & (r > 0)
        if m.sum() < 3:
            continue
        w_srf, r_srf = srf_wl[m], r[m]
        inside = (w_srf >= wl.min()) & (w_srf <= wl.max())
        coverage = r_srf[inside].sum() / r_srf.sum() if r_srf.sum() > 0 else 0.0
        centres.append(float(lab))
        if coverage < min_coverage:
            values.append(np.nan)
            dropped.append((float(lab),
                            "only %.0f%% of the response falls inside the measured "
                            "range" % (100 * coverage)))
            continue
        interp = interp1(wl, rrs, w_srf[inside], method="pchip")
        good = np.isfinite(interp)
        if good.sum() < 3:
            values.append(np.nan)
            dropped.append((float(lab), "no finite data under the response"))
            continue
        wgt = r_srf[inside][good]
        values.append(float(np.sum(wgt * interp[good]) / np.sum(wgt)))
    return np.array(centres), np.array(values), dropped


def _gaussian_band(wl, rrs, centre, fwhm, min_coverage):
    from math import erf, log, sqrt

    sigma = fwhm / (2.0 * sqrt(2.0 * log(2.0)))
    lo, hi = wl.min(), wl.max()
    coverage = 0.5 * (erf((hi - centre) / (sigma * sqrt(2.0)))
                      - erf((lo - centre) / (sigma * sqrt(2.0))))
    if coverage < min_coverage:
        return np.nan, "only %.0f%% of the band falls inside the spectrum" % (
            100 * coverage)
    w = np.exp(-0.5 * ((wl - centre) / sigma) ** 2)
    good = np.isfinite(rrs)
    if not good.any():
        return np.nan, "no finite data"
    return float(np.sum(w[good] * rrs[good]) / np.sum(w[good])), None


def convolve_with_srf(wl, rrs, srf_wl, srf, min_coverage=0.5):
    """Convolve with a user-supplied response curve, e.g. a real OLCI or OCI SRF."""
    srf_wl = np.asarray(srf_wl, dtype=float)
    srf = np.asarray(srf, dtype=float)
    inside = (srf_wl >= np.min(wl)) & (srf_wl <= np.max(wl))
    if srf.sum() <= 0 or srf[inside].sum() / srf.sum() < min_coverage:
        return np.nan
    v = interp1(wl, rrs, srf_wl[inside], method="pchip")
    good = np.isfinite(v)
    return float(np.sum(srf[inside][good] * v[good]) / np.sum(srf[inside][good]))


def band_table(sensor_name):
    """Rows of (label, centre, fwhm, used_by_giop) for display."""
    s = SENSORS[sensor_name]
    return [(b.label, b.centre, b.fwhm, b.centre in s.giop_bands) for b in s.bands]
