"""Uncertainty estimators. THEORY.md sect. 7.

Upstream GIOP returns no uncertainty at all. Neither estimator here is a posterior, and
they measure different things:

- :func:`linearised_covariance` is conditional on the prescribed shapes being correct;
- :func:`shape_ensemble` measures the error those prescriptions contribute, which is
  usually the larger term.

Reporting only the first is how a retrieval acquires a credible-looking error bar that
excludes its own dominant error.
"""

from __future__ import annotations

import itertools

import numpy as np

from .diagnostics import design_matrix
from .inversion import giop
from .model import GiopConfig

__all__ = ["linearised_covariance", "shape_ensemble"]


def linearised_covariance(result, sigma_rrs):
    """Gaussian covariance of the three amplitudes at the solution.

    ``sigma_rrs`` is the per-band 1-sigma uncertainty on **subsurface** r_rs, either a
    scalar or an array over the band set. For field data a defensible source is the
    standard deviation across replicate scans propagated through G17.

    Returns ``(cov, sigma)`` with ``sigma`` the square root of the diagonal, in the
    parameter order (M_dg, M_bp, M_phi).

    The Jacobian is computed analytically from the model, not by finite differences::

        d r_rs / d x_k = (g0 + 2 g1 u) * du/dx_k

    with, writing D = a + b_b,
    du/dM_dg = -u/D * e_dg, du/dM_phi = -u/D * aph*, du/dM_bp = (1-u)/D * e_bp.
    """
    r = result
    a_tot = r.aw + r.adg + r.aph
    bb_tot = r.bbw + r.bbp
    D = a_tot + bb_tot
    u = bb_tot / D

    du = np.column_stack([
        -u / D * r.adgs,
        (1.0 - u) / D * r.bbps,
        -u / D * r.aphs,
    ])
    J = (r.g0 + 2.0 * r.g1 * u)[:, None] * du

    sigma_rrs = np.asarray(sigma_rrs, dtype=float)
    if sigma_rrs.ndim == 0:
        sigma_rrs = np.full(len(r.wl), float(sigma_rrs))
    if np.any(sigma_rrs <= 0):
        raise ValueError("sigma_rrs must be positive")

    W = np.diag(1.0 / sigma_rrs**2)
    JtWJ = J.T @ W @ J
    try:
        cov = np.linalg.inv(JtWJ)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "the normal matrix is singular: the three constituents are not separated "
            "by this band set. See diagnostics.report()."
        ) from exc
    return cov, np.sqrt(np.diag(cov))


def shape_ensemble(
    wl, rrs, chl,
    sdg_values=(0.010, 0.014, 0.018, 0.022),
    eta_values=("qaa", 0.5, 1.0, 1.5, 2.0),
    aph_values=("bricaud", "ciotti", "gsm"),
    base_cfg=None,
):
    """Re-invert across the grid of allowed prescriptions and report the spread.

    This is the estimator that captures assumptions A2, A3 and A4. ``eta_expts.m`` is
    upstream's manual version of it for eta alone.

    Returns a dict with the per-parameter median, 16th and 84th percentiles, full range,
    the number of converged members, and the members themselves. Non-converged and
    non-finite members are dropped and counted, not silently included.
    """
    base = base_cfg or GiopConfig()
    rows, failed = [], 0

    for sdg, eta, aph in itertools.product(sdg_values, eta_values, aph_values):
        cfg = GiopConfig(**{**base.__dict__, "sdg": sdg, "eta": eta, "aph": aph})
        try:
            res = giop(wl, rrs, chl, cfg=cfg)
        except Exception:
            failed += 1
            continue
        if res.converged and np.all(np.isfinite(res.x)):
            rows.append((sdg, eta, aph, res.x))
        else:
            failed += 1

    if not rows:
        raise RuntimeError("no ensemble member converged")

    X = np.array([r[3] for r in rows])
    names = ("adg443", "bbp443", "aph_amplitude")
    out = {
        "n_members": len(rows),
        "n_failed": failed,
        "members": rows,
        "negative_fraction": {
            n: float(np.mean(X[:, k] < 0)) for k, n in enumerate(names)
        },
    }
    for k, n in enumerate(names):
        out[n] = {
            "median": float(np.median(X[:, k])),
            "p16": float(np.percentile(X[:, k], 16)),
            "p84": float(np.percentile(X[:, k], 84)),
            "min": float(X[:, k].min()),
            "max": float(X[:, k].max()),
        }
    return out
