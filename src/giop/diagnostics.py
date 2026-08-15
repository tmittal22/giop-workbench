"""Identifiability diagnostics. THEORY.md sect. 6.3.

The question these answer is not "did the fit converge" but "is the band set capable of
separating these constituents at all". A converged fit on a degenerate design matrix is
a converged fit to an arbitrary point along a valley.
"""

from __future__ import annotations

import numpy as np

from .model import u_from_rrs

__all__ = ["design_matrix", "eigenvector_angles", "condition_number", "report"]

#: Below this angle a pair is not meaningfully separated by the band set in use.
DEGENERACY_THRESHOLD_DEG = 15.0

_NAMES = ("adg", "bbp", "aph")


def design_matrix(result):
    """The N x 3 matrix whose columns are the three constituents' contributions.

    This is exactly **A** from THEORY.md G13, i.e. the linearisation the inversion sees.
    Built from a :class:`~giop.inversion.GiopResult`.
    """
    if result.g1 == 0:
        # Morel option: r_rs = g0 * u, so u is linear in r_rs.
        u = np.asarray(result.rrs_obs_subsurface) / np.asarray(result.g0)
    else:
        u = u_from_rrs(result.rrs_obs_subsurface, result.g0, result.g1)
    return np.column_stack([result.adgs * u, result.bbps * (u - 1.0), result.aphs * u])


def eigenvector_angles(result):
    """Pairwise angles (degrees) between the three columns of the design matrix.

    Returns a dict keyed by pair name. Small angles mean the two constituents make
    nearly the same change in r_rs over this band set, so their amplitudes trade off
    against each other and only their combination is constrained.
    """
    A = design_matrix(result)
    out = {}
    for i in range(3):
        for j in range(i + 1, 3):
            ni, nj = np.linalg.norm(A[:, i]), np.linalg.norm(A[:, j])
            if ni == 0 or nj == 0:
                out[f"{_NAMES[i]}-{_NAMES[j]}"] = np.nan
                continue
            c = float(np.clip(abs(A[:, i] @ A[:, j] / (ni * nj)), 0.0, 1.0))
            out[f"{_NAMES[i]}-{_NAMES[j]}"] = float(np.degrees(np.arccos(c)))
    return out


def condition_number(result):
    """2-norm condition number of the design matrix."""
    return float(np.linalg.cond(design_matrix(result)))


def report(result):
    """Human-readable identifiability summary.

    Reports the angles and the condition number, and states the part that the angles do
    **not** capture: they measure conditioning at a *fixed* set of eigenvectors, so a
    well-conditioned design can still sit on a badly wrong answer if the prescribed
    S_dg, eta or aph* shape is wrong. That error is measured by
    :func:`giop.uncertainty.shape_ensemble`, not by this function.
    """
    ang = eigenvector_angles(result)
    lines = [
        f"condition number      : {condition_number(result):.3g}",
        f"eigenvector angles    : " + ", ".join(f"{k} {v:.1f} deg" for k, v in ang.items()),
    ]
    weak = [k for k, v in ang.items() if np.isfinite(v) and v < DEGENERACY_THRESHOLD_DEG]
    if weak:
        lines.append(
            f"DEGENERATE PAIRS      : {', '.join(weak)} (below "
            f"{DEGENERACY_THRESHOLD_DEG} deg). Only the sum of each pair is "
            "constrained; the split between them is set by the prescribed shapes."
        )
    else:
        lines.append(
            "no pair below the degeneracy threshold AT THE PRESCRIBED SHAPES. This "
            "does not mean the retrieval is accurate: the dominant error is normally "
            "the choice of S_dg, eta and aph* itself, which this diagnostic holds "
            "fixed. Use uncertainty.shape_ensemble for that."
        )
    return "\n".join(lines)
