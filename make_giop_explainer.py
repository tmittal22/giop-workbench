"""One-page infographic: what GIOP solves, what S_dg and eta are, and what we changed.

    python make_giop_explainer.py [--out .]
"""
import argparse, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from giop.water import a_water, bb_water

INK, OLD, NEW, ACC = "#1a1a1a", "#c0392b", "#2e7d32", "#2c6f9b"


def box(ax, x, y, w, h, title, col, fc, body, tsize=11, bsize=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008", fc=fc,
                                ec=col, lw=2.0, transform=ax.transAxes, zorder=2))
    ax.text(x + w / 2, y + h - 0.028, title, fontsize=tsize, weight="bold", color=col,
            ha="center", transform=ax.transAxes, zorder=3)
    ax.text(x + 0.014, y + h - 0.058, body, fontsize=bsize, color=INK, va="top",
            ha="left", transform=ax.transAxes, linespacing=1.45, zorder=3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=".")
    a = ap.parse_args()
    fig = plt.figure(figsize=(16.5, 11.7))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.text(0.5, 0.972, "GIOP — what it solves, and what the two shape constants are",
             fontsize=20, weight="bold", ha="center")
    fig.text(0.5, 0.949, "Werdell et al. (2013) doi:10.1364/AO.52.002019   ·   "
             "left: the original   ·   right: what our port adds", fontsize=11,
             ha="center", color="#555")

    # ---- the model
    ax.add_patch(FancyBboxPatch((0.035, 0.775), 0.93, 0.155, boxstyle="round,pad=0.01",
                                fc="#eef5ff", ec="#8aa8c8", lw=1.8,
                                transform=ax.transAxes))
    ax.text(0.5, 0.912, "THE MODEL: split absorption and backscatter into a known part "
            "and three unknown amplitudes", fontsize=12, weight="bold", ha="center",
            color="#33506e")
    ax.text(0.5, 0.879,
            r"$a(\lambda)=a_w(\lambda)+M_\phi\,a^*_\phi(\lambda)"
            r"+M_{dg}\,e^{-S_{dg}(\lambda-443)}$", fontsize=15, ha="center")
    ax.text(0.5, 0.842,
            r"$b_b(\lambda)=b_{bw}(\lambda)+M_{bp}\,(443/\lambda)^{\eta}$",
            fontsize=15, ha="center")
    ax.text(0.5, 0.805,
            r"solve for $M_\phi,\;M_{dg},\;M_{bp}$   from   "
            r"$u=b_b/(a+b_b)$,   $r_{rs}=g_0u+g_1u^2$", fontsize=12.5, ha="center",
            color="#33506e")

    # ---- S_dg and eta explained, with plots
    axd = fig.add_axes([0.075, 0.565, 0.36, 0.185])
    lam = np.linspace(400, 700, 300)
    for s_, c in ((0.010, "#f5a623"), (0.018, OLD), (0.025, "#6a3d9a")):
        axd.plot(lam, np.exp(-s_ * (lam - 443)), lw=2.4, color=c,
                 label="$S_{dg}$ = %.3f" % s_)
    axd.axvline(443, color="#888", ls=":")
    axd.set_xlabel("wavelength (nm)"); axd.set_ylabel("$a_{dg}$ shape (rel. to 443)")
    axd.legend(fontsize=9); axd.grid(alpha=0.25)
    axd.set_title("$S_{dg}$ — how FAST yellow-substance absorption\nfalls toward the "
                  "red.  Units nm$^{-1}$.", fontsize=10.5, loc="left")

    axe = fig.add_axes([0.565, 0.565, 0.36, 0.185])
    for e_, c in ((0.0, "#f5a623"), (1.0, ACC), (2.0, "#6a3d9a")):
        axe.plot(lam, (443.0 / lam) ** e_, lw=2.4, color=c, label="$\\eta$ = %.1f" % e_)
    axe.axvline(443, color="#888", ls=":")
    axe.set_xlabel("wavelength (nm)"); axe.set_ylabel("$b_{bp}$ shape (rel. to 443)")
    axe.legend(fontsize=9); axe.grid(alpha=0.25)
    axe.set_title("$\\eta$ — the PARTICLE SIZE proxy. Large $\\eta$ = small\n"
                  "particles (steep, blue-scattering); $\\eta\\to0$ = large, flat.",
                  fontsize=10.5, loc="left")

    box(ax, 0.035, 0.265, 0.45, 0.245, "THE ORIGINAL (Werdell 2013 / run_giop.m)", OLD,
        "#fdecea",
        "SOLVER   unconstrained linear matrix inversion, or\n"
        "         Nelder-Mead. Nothing forbids a NEGATIVE\n"
        "         absorption, and it returns them: $a_\\phi$ = -0.275\n"
        "         at $S_{dg}$ = 0.010 on the shipped demo data.\n\n"
        "SHAPES   $S_{dg}$ = 0.018 nm$^{-1}$ and $\\eta$ from a band ratio,\n"
        "         both FIXED. Never fitted, never reported\n"
        "         with an uncertainty.\n\n"
        "WEIGHTS  every band counted equally; no per-band\n"
        "         noise model.\n\n"
        "BANDS    demoed on the 6 SeaWiFS bands, though the\n"
        "         maths is not limited to them.", bsize=8.5)

    box(ax, 0.515, 0.265, 0.45, 0.245, "OUR PORT (tmittal22/giop-workbench)", NEW,
        "#eaf6ea",
        "SOLVER   + inv='bounded': positivity enforced and a\n"
        "         PER-BAND sigma. On this Arctic water the\n"
        "         original returns $M_\\phi$ = 8272, $b_{bp}$ = 22.8;\n"
        "         bounded returns 11.5 and 0.084.\n\n"
        "SHAPES   + fit_shapes=True actually FITS $S_{dg}$ and $\\eta$\n"
        "         (fmin path). On LOC1 both RAIL, which is\n"
        "         the honest answer: this spectrum cannot\n"
        "         determine them.\n\n"
        "ALSO     9 upstream defects catalogued with line\n"
        "         numbers; measured MODIS SRF; BRDF\n"
        "         normalisation; $F_0$ and nLw; 139 tests.", bsize=8.5)

    box(ax, 0.035, 0.030, 0.93, 0.205,
        "WHY $S_{dg}$ MATTERS MORE THAN ANYTHING ELSE IN THE RETRIEVAL", "#8a6000",
        "#fff6d5",
        "$a_{dg}$ lumps CDOM and detritus, and $S_{dg}$ sets its slope. Phytoplankton "
        "absorption ALSO rises toward the blue, so the\ntwo are nearly collinear: move "
        "$S_{dg}$ and the fit trades $a_{dg}$ against $a_\\phi$ almost freely. "
        "Measured on the LOC1 mean:\n\n"
        "     $S_{dg}$ = 0.014        0.018 (default)        0.022\n"
        "     6 bands          chl 0.00 (railed)       chl 23.7               chl 297.1"
        "        <- unusable\n"
        "     hyperspectral    chl 2.37                chl 11.5               chl 20.9"
        "         <- a factor 8.8, still the largest single error\n\n"
        "$S_{dg}$ is NOT measured by the radiometer. It is measured on a filtered water "
        "sample with a bench spectrophotometer,\nwhich is why one 5-minute CDOM sample "
        "per station is the highest-value change to the field protocol.",
        tsize=11.5, bsize=9)

    p = os.path.join(a.out, "GIOP_explainer.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote %s" % p)


if __name__ == "__main__":
    main()
