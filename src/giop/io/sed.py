"""Reader for Spectral Evolution ``.sed`` files (DARWin SP).

The column vocabulary below was taken from the string table of ``DARWin2.exe`` in the
NaturaSpec Plus installer, not from a third-party parser, because the generic parsers
in circulation assume ``Irrad. (Ref.)`` where DARWin actually writes ``Irr. (Ref.)``
and therefore drop the irradiance columns silently.

File layout::

    Version: 2.1
    File Name: C:\\...\\scan0001.sed
    Instrument: PSR+3500_SN...
    ...
    Data:
    Wvl\tRad. (Ref.)\tRad. (Target)\tReflect. %
    350.0\t1.234e+000\t5.678e-002\t4.601
    ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["SedSpectrum", "read_sed", "read_sed_dir", "COLUMN_ALIASES"]

#: Canonical name -> the labels DARWin may write for it.
COLUMN_ALIASES = {
    "wavelength":     ("Wvl",),
    "rad_ref":        ("Rad. (Ref.)",),
    "rad_target":     ("Rad. (Target)",),
    "irr_ref":        ("Irr. (Ref.)", "Irrad. (Ref.)"),
    "irr_target":     ("Irr. (Target)", "Irrad. (Target)"),
    "dn_ref":         ("DN (Ref.)", "Ref. DN"),
    "dn_target":      ("DN (Target)", "Tgt. DN"),
    "reflectance":    ("Reflect. %", "Reflect. [1.0]", "Reflect."),
}

_DATA_MARKER = re.compile(r"^\s*Data:\s*$", re.IGNORECASE)


@dataclass
class SedSpectrum:
    """One ``.sed`` scan: the header as parsed key/values, plus the data columns."""

    path: Path
    header: dict
    columns: dict          # canonical name -> np.ndarray
    raw_columns: dict      # original label -> np.ndarray
    reflectance_scale: float | None = None  # 100 for '%' variant, 1 for '[1.0]'

    @property
    def wavelength(self):
        return self.columns["wavelength"]

    @property
    def reflectance(self):
        """Target/reference reflectance factor as a fraction (never percent)."""
        if "reflectance" not in self.columns:
            raise KeyError(
                f"{self.path.name} has no reflectance column; available: "
                f"{sorted(self.raw_columns)}"
            )
        return self.columns["reflectance"] / (self.reflectance_scale or 1.0)

    @property
    def radiance_target(self):
        return self._need("rad_target", "W m^-2 sr^-1 nm^-1")

    @property
    def radiance_reference(self):
        return self._need("rad_ref", "W m^-2 sr^-1 nm^-1")

    @property
    def irradiance_reference(self):
        return self._need("irr_ref", "W m^-2 nm^-1")

    def _need(self, key, units):
        if key not in self.columns:
            raise KeyError(
                f"{self.path.name} has no {key} column (expected one of "
                f"{COLUMN_ALIASES[key]}, units {units}); available: "
                f"{sorted(self.raw_columns)}. Re-export from DARWin with that column "
                "enabled, or use a different measurement path."
            )
        return self.columns[key]

    # -- convenience metadata ------------------------------------------------
    @property
    def comment(self):
        return self.header.get("Comment", "")

    @property
    def instrument(self):
        return self.header.get("Instrument", "")

    @property
    def latitude(self):
        return _parse_coord(self.header.get("Latitude"))

    @property
    def longitude(self):
        return _parse_coord(self.header.get("Longitude"))

    @property
    def datetime_str(self):
        d, t = self.header.get("Date", ""), self.header.get("Time", "")
        return f"{d} {t}".strip()

    def clip(self, lo, hi):
        """Restrict to ``lo <= wl <= hi`` nm, returning a new SedSpectrum."""
        m = (self.wavelength >= lo) & (self.wavelength <= hi)
        return SedSpectrum(
            path=self.path, header=self.header,
            columns={k: v[m] for k, v in self.columns.items()},
            raw_columns={k: v[m] for k, v in self.raw_columns.items()},
            reflectance_scale=self.reflectance_scale,
        )

    def __repr__(self):
        wl = self.wavelength
        return (f"SedSpectrum({self.path.name!r}, {len(wl)} bands "
                f"{wl.min():.1f}-{wl.max():.1f} nm, cols={sorted(self.columns)})")


def read_sed(path, encoding="latin-1"):
    """Parse one ``.sed`` file.

    ``encoding`` defaults to latin-1 because DARWin writes Windows-encoded degree and
    micro signs into the header, and utf-8 raises on them.
    """
    path = Path(path)
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):          # UTF-8 BOM
        raw = raw[3:]
    # Real NaturaSpec exports are UTF-8 and put degree signs in a <Metadata> block, so
    # decoding latin-1 first turns "4.8 deg" into mojibake. Verified against a genuine
    # NaturaSpecPlus_SN25494G1 file.
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode(encoding, errors="replace")
    lines = text.splitlines()

    marker = None
    header = {}
    for i, line in enumerate(lines):
        if _DATA_MARKER.match(line):
            marker = i
            break
        if ":" in line:
            k, _, v = line.partition(":")
            header[k.strip()] = v.strip()

    if marker is None:
        raise ValueError(
            f"{path.name} has no 'Data:' marker, so it is not a DARWin .sed export. "
            f"First line was: {lines[0][:80] if lines else '<empty file>'!r}"
        )

    body = [ln for ln in lines[marker + 1:] if ln.strip()]
    if len(body) < 2:
        raise ValueError(f"{path.name}: no data rows after the 'Data:' marker")

    labels = [c.strip() for c in body[0].split("\t")]
    if len(labels) < 2:
        labels = re.split(r"\s{2,}|\t", body[0].strip())
        labels = [c.strip() for c in labels]

    rows = []
    for ln in body[1:]:
        parts = ln.split("\t") if "\t" in ln else ln.split()
        if len(parts) != len(labels):
            continue  # trailing junk or a wrapped line; skip rather than misalign
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    if not rows:
        raise ValueError(
            f"{path.name}: found the header {labels} but no parseable numeric rows"
        )

    arr = np.asarray(rows, dtype=float)
    raw = {lab: arr[:, i] for i, lab in enumerate(labels)}

    columns, scale = {}, None
    for canon, aliases in COLUMN_ALIASES.items():
        for lab in labels:
            if lab in aliases:
                columns[canon] = raw[lab]
                if canon == "reflectance":
                    scale = 100.0 if "%" in lab else 1.0
                break

    if "wavelength" not in columns:
        raise ValueError(
            f"{path.name}: no wavelength column. Labels found: {labels}. "
            "Expected a column named 'Wvl'."
        )

    order = np.argsort(columns["wavelength"])
    if not np.all(np.diff(order) == 1):
        columns = {k: v[order] for k, v in columns.items()}
        raw = {k: v[order] for k, v in raw.items()}

    return SedSpectrum(path=path, header=header, columns=columns,
                       raw_columns=raw, reflectance_scale=scale)


def read_sed_dir(directory, pattern="*.sed"):
    """Read every ``.sed`` in a directory, sorted by filename."""
    directory = Path(directory)
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files matching {pattern!r} in {directory}")
    return [read_sed(p) for p in paths]


def _parse_coord(value):
    if not value or value.strip().lower() in ("n/a", "", "none"):
        return None
    try:
        return float(re.sub(r"[^0-9.eE+-]", "", value.strip()))
    except ValueError:
        return None
