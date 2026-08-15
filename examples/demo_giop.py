"""Worked example: reproduce the upstream GIOP demo and show what it is conditional on.

Run::

    PYTHONPATH=src python examples/demo_giop.py

Produces ``figures/demo_giop.png`` and prints the numbers behind every panel.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from giop import GiopConfig, get_oc, giop
from giop.model import rrs_below_to_above

WL = np.array([412.0, 443, 490, 510, 555, 670])
RRS = np.array([0.003478, 0.004074, 0.004465, 0.003588, 0.002494, 0.000051])
GOLDEN_FMIN = np.array([0.0441, 0.0033, 0.3693])
GOLDEN_LMI = np.array([0.0414, 0.0022, 0.1058])

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)


def main():
    chl = float(get_oc(RRS[1], RRS[2], RRS[3], RRS[4], "oc4"))
    print(f"OC4 seed chlorophyll = {chl:.6f} mg m^-3\n")

    fmin = giop(WL, RRS, chl, qc=0.33)
    lmi = giop(WL, RRS, chl, inv="lmi", qc=0.33)

    print("Eigenvalues (a_dg(443) m^-1, b_bp(443) m^-1, aph amplitude):")
    for name, res, golden in (("fmin", fmin, GOLDEN_FMIN), ("lmi", lmi, GOLDEN_LMI)):
        print(f"  {name:5s} port   {np.array2string(res.x, precision=6)}")
        print(f"  {name:5s} MATLAB {golden}   max|diff| = "
              f"{np.max(np.abs(res.x - golden)):.2e}")
    print(f"\nPrescribed shapes: S_dg = {fmin.sdg:.5f} nm^-1, eta = {fmin.eta:.5f}")

    # Sensitivity to the prescribed shapes: the assumption that dominates the answer.
    etas = np.linspace(0.5, 2.0, 16)
    bbp_vs_eta = np.array(
        [giop(WL, RRS, chl, eta=float(e)).x[1] for e in etas])
    sdgs = np.linspace(0.010, 0.025, 16)
    adg_vs_sdg = np.array(
        [giop(WL, RRS, chl, sdg=float(s)).x[0] for s in sdgs])
    aph_vs_sdg = np.array(
        [giop(WL, RRS, chl, sdg=float(s)).x[2] for s in sdgs])

    print(f"\nb_bp(443) across eta 0.5-2.0 : {bbp_vs_eta.min():.5f} to "
          f"{bbp_vs_eta.max():.5f} m^-1, a factor {bbp_vs_eta.max()/bbp_vs_eta.min():.2f}")
    print(f"a_dg(443) across S_dg 0.010-0.025: {adg_vs_sdg.min():.5f} to "
          f"{adg_vs_sdg.max():.5f} m^-1, a factor {adg_vs_sdg.max()/adg_vs_sdg.min():.2f}")
    print(f"aph amp   across S_dg 0.010-0.025: {aph_vs_sdg.min():.5f} to "
          f"{aph_vs_sdg.max():.5f}")
    if np.any(aph_vs_sdg < 0):
        zero_cross = np.interp(0.0, aph_vs_sdg, sdgs)
        print(f"  -> aph amplitude is NEGATIVE below S_dg = {zero_cross:.5f} nm^-1. "
              "The cost function has no positivity constraint (THEORY.md sect. 6.1), "
              "so the CDOM/phytoplankton degeneracy can push a_phi through zero "
              "inside the plausible S_dg range.")

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    a = ax[0, 0]
    a.plot(WL, RRS, "ko-", lw=1.4, ms=5, label="measured $R_{rs}$")
    a.plot(WL, fmin.rrs_model_above, "b--o", ms=4, label="GIOP fmin (above water)")
    a.plot(WL, lmi.rrs_model_above, "r:s", ms=4, label="GIOP lmi (above water)")
    a.plot(WL, fmin.rrs_model_subsurface, color="0.6", ls="-.",
           label="model $r_{rs}$ below water\n(what upstream plots here)")
    a.set_xlabel("wavelength (nm)")
    a.set_ylabel("$R_{rs}$ (sr$^{-1}$)")
    a.set_title("Fit, and the above/below-water distinction", fontsize=10)
    a.legend(fontsize=7)

    a = ax[0, 1]
    a.plot(WL, fmin.aw, color="0.5", label="$a_w$ (prescribed)")
    a.plot(WL, fmin.aph, "g-o", ms=4, label="$a_\\varphi$")
    a.plot(WL, fmin.adg, "c-o", ms=4, label="$a_{dg}$")
    a.plot(WL, fmin.apg, "m:", lw=2, label="$a_{pg}=a_\\varphi+a_{dg}$")
    a.set_xlabel("wavelength (nm)")
    a.set_ylabel("$a$ (m$^{-1}$)")
    a.set_title("Retrieved absorption", fontsize=10)
    a.legend(fontsize=8)

    a = ax[1, 0]
    a.plot(etas, bbp_vs_eta, "r-o", ms=4)
    a.axvline(fmin.eta, color="k", ls="--", lw=1,
              label=f"QAA value used: {fmin.eta:.2f}")
    a.set_xlabel("prescribed $\\eta$")
    a.set_ylabel("retrieved $b_{bp}(443)$ (m$^{-1}$)")
    a.set_title("$b_{bp}$ is conditional on the assumed slope", fontsize=10)
    a.legend(fontsize=8)

    # Absolute values, not ratios: the aph amplitude changes sign here, and
    # normalising by a negative reference would flip the curve and read as nonsense.
    a = ax[1, 1]
    a.plot(sdgs, adg_vs_sdg, "c-o", ms=4, label="$a_{dg}(443)$ (m$^{-1}$)")
    a.plot(sdgs, aph_vs_sdg, "g-s", ms=4, label="aph amplitude (mg m$^{-3}$)")
    a.axhline(0, color="k", lw=0.8)
    a.axvline(0.018, color="k", ls="--", lw=1, label="GIOP-DC 0.018")
    neg = aph_vs_sdg < 0
    if neg.any():
        a.axvspan(sdgs[0], sdgs[neg].max(), color="red", alpha=0.12,
                  label="aph amplitude < 0 (unphysical)")
    a.set_xlabel("prescribed $S_{dg}$ (nm$^{-1}$)")
    a.set_ylabel("retrieved value")
    a.set_title("CDOM/phytoplankton split follows the assumption,\n"
                "and can go unphysical", fontsize=10)
    a.legend(fontsize=7)

    fig.suptitle(
        "GIOP demo spectrum (run_giop.m). Port reproduces the published "
        "eigenvalues; lower panels show what they are conditional on.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = OUT / "demo_giop.png"
    fig.savefig(path, dpi=130)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
