"""Generic CSV in / CSV out, for data that is already R_rs."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

__all__ = ["read_csv_spectra", "write_result_csv"]


def read_csv_spectra(path, wl_col=0, delimiter=None):
    """Read a CSV of spectra.

    Two layouts are accepted and detected automatically:

    - **long**: first column wavelength, each remaining column one spectrum;
    - **wide**: first row wavelengths, each remaining row one spectrum, with an
      optional leading label column.

    Returns ``(wavelength, spectra, labels)`` where ``spectra`` is (n_spectra, n_bands).
    """
    path = Path(path)
    text = path.read_text().splitlines()
    if not text:
        raise ValueError(f"{path.name} is empty")

    if delimiter is None:
        sample = "\n".join(text[:5])
        delimiter = "\t" if sample.count("\t") > sample.count(",") else ","

    rows = [r for r in csv.reader(text, delimiter=delimiter) if any(c.strip() for c in r)]
    header = rows[0]

    def numeric(cells):
        out = []
        for c in cells:
            try:
                out.append(float(c))
            except (ValueError, TypeError):
                out.append(np.nan)
        return np.array(out)

    first = numeric(header)
    if np.isfinite(first[1:]).sum() >= max(2, len(first) // 2) and np.isnan(first[0]):
        # wide: header row is wavelengths after a label cell
        wl = first[1:]
        labels, data = [], []
        for r in rows[1:]:
            labels.append(r[0])
            data.append(numeric(r[1:]))
        return wl, np.vstack(data), labels

    # long layout
    body = rows[1:] if not np.isfinite(first).all() else rows
    labels = header[1:] if not np.isfinite(first).all() else [
        f"spectrum_{i}" for i in range(len(header) - 1)
    ]
    arr = np.vstack([numeric(r) for r in body])
    wl = arr[:, wl_col]
    spectra = np.delete(arr, wl_col, axis=1).T
    order = np.argsort(wl)
    return wl[order], spectra[:, order], list(labels)


def write_result_csv(path, results, labels=None):
    """Write per-spectrum GIOP scalars to CSV.

    One row per inversion, with the three eigenvalues, the prescribed shape
    parameters, the seed chlorophyll and the convergence/QC flags. Everything a
    downstream analysis needs to know whether a number is usable.
    """
    path = Path(path)
    results = list(results)
    labels = labels or [f"spectrum_{i}" for i in range(len(results))]

    fields = ["label", "adg443_m-1", "bbp443_m-1", "aph_amplitude_mg_m-3",
              "chl_seed_mg_m-3", "sdg_nm-1", "eta", "cost", "converged", "qc_passed",
              "n_bands", "wl_min_nm", "wl_max_nm"]

    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for lab, r in zip(labels, results):
            w.writerow([
                lab,
                f"{r.x[0]:.6g}", f"{r.x[1]:.6g}", f"{r.x[2]:.6g}",
                f"{r.chl_seed:.6g}", f"{r.sdg:.6g}", f"{r.eta:.6g}",
                f"{r.cost:.6g}", int(r.converged),
                "" if r.qc_passed is None else int(r.qc_passed),
                len(r.wl), f"{r.wl.min():.1f}", f"{r.wl.max():.1f}",
            ])
    return path
