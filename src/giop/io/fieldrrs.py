"""Read the output of the `fieldrrs` field package directly, metadata included.

`fieldrrs` (https://github.com/tmittal22/fieldrrs) turns NaturaSpec `.sed` scans into
R_rs. It writes two things, and both are read here:

* **per-spectrum CSV** — a `#`-commented header carrying rho, the residual-glint method,
  the viewing and solar geometry, the wind speed, the panel reflectance and the
  instrument footprint, then a wavelength/R_rs table;
* **batch CSV** (`rrs_all_stations.csv`) — one row per station, one column per band.

WHY THE METADATA MATTERS ENOUGH TO PARSE IT. An R_rs is not interpretable without the
conditions it was measured under. rho depends on wind speed; the retrieval is at the
measurement geometry rather than nadir-normalised (THEORY.md A12), so the BRDF correction
needs the solar and viewing angles; and `nir_zero` is invalid in turbid water. Carrying
those fields through means the inversion can warn about them instead of the operator
having to remember. A hand-copied column of numbers loses all of it.
"""

from __future__ import annotations

import csv
import os

import numpy as np

__all__ = ["FieldSpectrum", "read_fieldrrs_csv", "read_fieldrrs_batch",
           "is_fieldrrs_csv", "brdf_args_from_meta"]


class FieldSpectrum(object):
    def __init__(self, wavelength, rrs, meta, warnings, path=None, name=None):
        self.wavelength = np.asarray(wavelength, dtype=float)
        self.rrs = np.asarray(rrs, dtype=float)
        self.meta = meta
        self.warnings = warnings
        self.path = path
        self.name = name or (os.path.basename(path) if path else "spectrum")

    # -- the fields worth acting on -----------------------------------------
    def _num(self, key):
        v = self.meta.get(key)
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @property
    def rho(self):
        return self._num("rho")

    @property
    def wind_ms(self):
        return self._num("wind_speed_ms")

    @property
    def solar_zenith(self):
        return self._num("solar_zenith_deg")

    @property
    def view_zenith(self):
        return self._num("view_zenith_from_nadir_deg")

    @property
    def relative_azimuth(self):
        return self._num("relative_azimuth_from_sun_deg")

    @property
    def residual_method(self):
        return self.meta.get("residual_method")

    @property
    def footprint_area_m2(self):
        return self._num("footprint_area_m2")

    def review(self):
        """Everything about this spectrum that should change how it is interpreted.

        Returns a list of strings. Empty means nothing to flag, which is rare.
        """
        out = list(self.warnings)
        w = self.wind_ms
        if w is None:
            out.append("Wind speed was not recorded, so rho cannot be checked. rho = "
                       "0.028 assumes wind below ~5 m/s.")
        elif w > 5.0 and (self.rho or 0.028) <= 0.030:
            out.append("Wind was %.1f m/s but rho = %.3f, which is only valid below "
                       "~5 m/s. Blue R_rs is biased HIGH." % (w, self.rho or 0.028))
        if self.residual_method == "nir_zero":
            out.append("nir_zero was applied: valid in clear oceanic water, and it "
                       "DELETES real signal in turbid water.")
        if self.solar_zenith is None:
            out.append("No solar geometry recorded, so the BRDF correction (G18) "
                       "cannot be applied.")
        vis = (self.wavelength >= 400) & (self.wavelength <= 700)
        neg = int(np.sum(self.rrs[vis] < 0))
        if neg:
            out.append("%d visible bands are negative: the glint subtraction "
                       "over-corrected." % neg)
        return out

    def __repr__(self):
        return "FieldSpectrum(%s, %d bands %.0f-%.0f nm)" % (
            self.name, len(self.wavelength), self.wavelength.min(),
            self.wavelength.max())


def is_fieldrrs_csv(path):
    """True if this looks like a per-spectrum fieldrrs export."""
    try:
        with open(path, "r") as fh:
            head = fh.read(400)
    except OSError:
        return False
    return "fieldrrs" in head and "Rrs" in head


def read_fieldrrs_csv(path):
    """Read one per-spectrum fieldrrs CSV into a :class:`FieldSpectrum`."""
    meta, warnings, rows, header = {}, [], [], None
    with open(path, "r") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("#"):
                cells = [c.strip() for c in next(csv.reader([line]))]
                cells = [c.lstrip("#").strip() for c in cells]
                cells = [c for c in cells if c]
                if not cells:
                    continue
                if cells[0].upper() == "WARNING":
                    warnings.append(",".join(cells[1:]))
                elif cells[0] == "rho" or len(cells) >= 2:
                    # the rho line is written as: # rho,<v>,residual_method,<m>,offset,<o>
                    for i in range(0, len(cells) - 1, 2):
                        meta[cells[i]] = cells[i + 1]
                continue
            cells = next(csv.reader([line]))
            if header is None:
                header = [c.strip() for c in cells]
                continue
            try:
                rows.append([float(c) if c.strip() else np.nan for c in cells])
            except ValueError:
                continue

    if header is None or not rows:
        raise ValueError("%s: no data table found. Is this a fieldrrs export?"
                         % os.path.basename(path))

    arr = np.asarray(rows, dtype=float)
    cols = {name: arr[:, i] for i, name in enumerate(header) if i < arr.shape[1]}

    wl_key = next((k for k in cols if k.lower().startswith("wavelength")), None)
    rrs_key = next((k for k in cols if k.lower().startswith("rrs_sr")
                    or k.lower() == "rrs"), None)
    if wl_key is None or rrs_key is None:
        raise ValueError("%s: expected 'wavelength_nm' and 'Rrs_sr-1' columns; got %s"
                         % (os.path.basename(path), header))

    return FieldSpectrum(cols[wl_key], cols[rrs_key], meta, warnings, path=path)


def read_fieldrrs_batch(path):
    """Read `rrs_all_stations.csv`: one row per station, one column per band.

    Returns a list of :class:`FieldSpectrum`, one per station. The batch file carries no
    per-station metadata, so ``meta`` is empty and :meth:`FieldSpectrum.review` will say
    so. Prefer the per-spectrum files when the conditions matter.
    """
    with open(path, "r") as fh:
        rows = [r for r in csv.reader(fh) if r and any(c.strip() for c in r)]
    if len(rows) < 2:
        raise ValueError("%s: needs a header row and at least one station"
                         % os.path.basename(path))

    try:
        wl = np.array([float(c) for c in rows[0][1:]])
    except ValueError:
        raise ValueError("%s: the first row must be band centres in nm after a label "
                         "cell; got %s" % (os.path.basename(path), rows[0][:4]))

    out = []
    for r in rows[1:]:
        vals = []
        for c in r[1:]:
            try:
                vals.append(float(c))
            except ValueError:
                vals.append(np.nan)
        if len(vals) != len(wl):
            continue
        out.append(FieldSpectrum(wl, vals, {}, [], path=path, name=r[0]))
    if not out:
        raise ValueError("%s: no station rows parsed" % os.path.basename(path))
    return out


def brdf_args_from_meta(spec, default_chl=0.5):
    """Pull the BRDF arguments (G18) straight out of a fieldrrs header.

    Returns ``(chl, solz, senz, relaz)`` or None if the geometry was not recorded, which
    is the honest outcome: the correction cannot be applied without it.
    """
    solz, senz, relaz = spec.solar_zenith, spec.view_zenith, spec.relative_azimuth
    if solz is None or senz is None or relaz is None:
        return None
    return float(default_chl), solz, senz, relaz
