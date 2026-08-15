"""GIOP Workbench: a Streamlit front end for the GIOP ocean-colour inversion.

    streamlit run app/giop_app.py

Design principle: this tool refuses to show you a number without also showing you what
that number is conditional on. GIOP as published returns three amplitudes and no
uncertainty at all, and the dominant error is not measurement noise but the spectral
shapes the model prescribes. Every panel here is built around making that visible.

Documentation: docs/STREAMLIT_GUIDE.md
Physics: THEORY.md      Deviations from upstream: PORTING_NOTES.md      Evidence: VALIDATION.md
"""

from __future__ import annotations

import io
import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from giop import GiopConfig, get_oc, giop                       # noqa: E402
from giop.diagnostics import (                                   # noqa: E402
    DEGENERACY_THRESHOLD_DEG, condition_number, eigenvector_angles,
)
from giop.io import RHO_MOBLEY1999, read_sed, rrs_from_sed_triplet  # noqa: E402
from giop.model import ConfigurationError                        # noqa: E402
from giop.sensors import SENSORS, convolve           # noqa: E402
from giop.uncertainty import linearised_covariance, shape_ensemble  # noqa: E402

st.set_page_config(page_title="GIOP Workbench", layout="wide",
                   initial_sidebar_state="expanded")

DEMO_WL = np.array([412.0, 443, 490, 510, 555, 670])
DEMO_RRS = np.array([0.003478, 0.004074, 0.004465, 0.003588, 0.002494, 0.000051])

S = st.session_state
S.setdefault("wl", None)
S.setdefault("rrs", None)
S.setdefault("source", None)
S.setdefault("result", None)
S.setdefault("notes", [])


# ----------------------------------------------------------------------------------
# helpers

def fig_axes(h=3.6):
    fig, ax = plt.subplots(figsize=(9, h))
    return fig, ax


def show(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def caveat(md):
    st.markdown(
        f"<div style='background:#fff8e5;border-left:4px solid #d0b000;"
        f"padding:0.6em 0.9em;border-radius:4px;font-size:0.92em'>{md}</div>",
        unsafe_allow_html=True)


def save_uploads(files):
    paths = []
    tmp = tempfile.mkdtemp()
    for f in files:
        p = os.path.join(tmp, f.name)
        with open(p, "wb") as fh:
            fh.write(f.getbuffer())
        paths.append(p)
    return paths


# ----------------------------------------------------------------------------------
# sidebar: model configuration

st.sidebar.title("GIOP configuration")
st.sidebar.caption("Defaults reproduce GIOP-DC (Werdell et al. 2013). "
                   "Anything marked EXT is an addition, not the published model.")

solver = st.sidebar.selectbox(
    "Inversion", ["fmin (GIOP-DC)", "bounded  [EXT]", "lmi"], index=0,
    help="fmin is the published unconstrained Nelder-Mead. bounded adds positivity and "
         "per-band noise weighting. lmi is the linear matrix inversion.")
inv = {"fmin (GIOP-DC)": "fmin", "bounded  [EXT]": "bounded", "lmi": "lmi"}[solver]

aph_opt = st.sidebar.selectbox("aph* eigenvector",
                               ["bricaud", "ciotti", "gsm", "chase"], index=0)
sf = st.sidebar.slider("Ciotti size fraction S_f", 0.0, 1.0, 0.5, 0.05,
                       disabled=(aph_opt != "ciotti"))

sdg_mode = st.sidebar.selectbox("S_dg (a_dg slope)",
                                ["0.018 (GIOP-DC)", "qaa", "obpg", "gsm", "custom"])
sdg_custom = st.sidebar.number_input("S_dg value (nm⁻¹)", 0.005, 0.030, 0.018, 0.001,
                                     format="%.4f",
                                     disabled=(sdg_mode != "custom"))
sdg = {"0.018 (GIOP-DC)": 0.018, "qaa": "qaa", "obpg": "obpg", "gsm": "gsm",
       "custom": sdg_custom}[sdg_mode]

eta_mode = st.sidebar.selectbox("η (b_bp slope)", ["qaa (GIOP-DC)", "gsm", "custom"])
eta_custom = st.sidebar.number_input("η value", -1.0, 4.0, 1.0, 0.05,
                                     disabled=(eta_mode != "custom"))
eta = {"qaa (GIOP-DC)": "qaa", "gsm": "gsm", "custom": eta_custom}[eta_mode]

fq = st.sidebar.selectbox("AOP–IOP relationship", ["gordon", "morel"], index=0)
trans = st.sidebar.selectbox("Air–sea transmission", ["lee", "flat"], index=0)

st.sidebar.markdown("---")
qc_on = st.sidebar.checkbox("Apply QC (|Δ| ≤ 33 % over 400–600 nm)", value=True)
wl_lo, wl_hi = st.sidebar.slider("Inversion window (nm)", 350, 900, (400, 700), 5,
                                 help="aph* is tabulated over 400–700 nm only.")
sigma_rel = st.sidebar.number_input(
    "Per-band σ, relative", 0.0, 0.5, 0.05, 0.01,
    help="Used by the bounded solver and by the uncertainty estimates. "
         "0.05 means 5 % of R_rs per band.")
sigma_floor = st.sidebar.number_input("Per-band σ, floor (sr⁻¹)", 0.0, 1e-2, 2e-4,
                                      1e-4, format="%.5f")


def build_cfg():
    return GiopConfig(fq=fq, trans=trans, eta=eta, sdg=sdg, aph=aph_opt, sf=sf,
                      inv=inv, qc=0.33 if qc_on else None)


def sigma_for(rrs):
    return np.sqrt((sigma_rel * np.abs(rrs)) ** 2 + sigma_floor ** 2)


# ----------------------------------------------------------------------------------

st.title("GIOP Workbench")
st.caption("Generalized Ocean Colour Inversion · Werdell et al. (2013), "
           "doi:10.1364/AO.52.002019 · ported, validated and extended")

tabs = st.tabs(["1 · Data", "2 · Sensor view", "3 · Inversion", "4 · Uncertainty",
                "5 · Identifiability", "6 · Export", "ℹ Guide"])

# ================================================================== 1 · DATA
with tabs[0]:
    st.subheader("Load a spectrum")
    mode = st.radio("Source", ["Demo (published GIOP example)",
                               "NaturaSpec .sed scans", "CSV of R_rs"],
                    horizontal=True)

    if mode.startswith("Demo"):
        if st.button("Load demo spectrum", type="primary"):
            S.wl, S.rrs, S.source, S.notes = DEMO_WL, DEMO_RRS, "GIOP demo (run_giop.m)", []
        st.info("The six-band example from upstream `run_giop.m`. GIOP-DC should return "
                "a_dg(443)=0.0441, b_bp(443)=0.0033, aph amplitude=0.3693.")

    elif mode.startswith("NaturaSpec"):
        st.markdown("Upload the **water** and **sky** scans. The panel is optional: if "
                    "omitted, the panel radiance is read from the water file's own "
                    "`Rad. (Ref.)` column, which is the DARWin reference-scan workflow.")
        c1, c2, c3 = st.columns(3)
        f_water = c1.file_uploader("Water (L_t)", type=["sed"])
        f_sky = c2.file_uploader("Sky (L_sky)", type=["sed"])
        f_panel = c3.file_uploader("Panel (optional)", type=["sed"])

        c4, c5, c6 = st.columns(3)
        panel_r = c4.number_input("Panel reflectance", 0.1, 1.0, 0.99, 0.01)
        rho = c5.number_input("ρ (sky glint)", 0.0, 0.2, RHO_MOBLEY1999, 0.001,
                              format="%.3f")
        residual = c6.selectbox("Residual glint", ["none", "nir_zero", "nir_similarity"])
        caveat(
            "<b>ρ = 0.028</b> assumes wind below ~5 m s⁻¹, 40°/135° geometry, clear sky "
            "(Mobley 1999). It is the largest single error above water and it is "
            "sea-state dependent. <b>nir_zero deletes real signal in turbid water.</b>")

        if st.button("Build R_rs", type="primary", disabled=not (f_water and f_sky)):
            try:
                paths = save_uploads([x for x in (f_water, f_sky, f_panel) if x])
                w = read_sed(paths[0]); sk = read_sed(paths[1])
                pn = read_sed(paths[2]) if f_panel else None
                res = rrs_from_sed_triplet(w, sk, pn, panel_reflectance=panel_r,
                                           rho=rho, residual=residual)
                S.wl, S.rrs = res.wavelength, res.rrs
                S.source = f"sed: {f_water.name}"
                S.notes = res.notes
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")

    else:
        up = st.file_uploader("CSV: first column wavelength (nm), second R_rs (sr⁻¹)",
                              type=["csv", "txt"])
        if up is not None and st.button("Load CSV", type="primary"):
            try:
                arr = np.genfromtxt(io.StringIO(up.getvalue().decode()),
                                    delimiter=None if b"\t" in up.getvalue() else ",",
                                    comments="#")
                arr = arr[np.isfinite(arr).all(axis=1)]
                S.wl, S.rrs, S.source, S.notes = arr[:, 0], arr[:, 1], f"csv: {up.name}", []
            except Exception as exc:
                st.error(f"Could not parse: {exc}")

    if S.wl is not None:
        st.success(f"Loaded **{S.source}** — {len(S.wl)} bands, "
                   f"{S.wl.min():.1f}–{S.wl.max():.1f} nm")
        for n in S.notes:
            st.warning(n)
        fig, ax = fig_axes()
        ax.plot(S.wl, S.rrs, "k-", lw=1.4)
        ax.axhline(0, color="0.7", lw=0.7)
        ax.axvspan(wl_lo, wl_hi, color="#0a6", alpha=0.10, label="inversion window")
        ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("$R_{rs}$ (sr$^{-1}$)")
        ax.legend(fontsize=8); show(fig)

        neg = int(np.sum((S.rrs < 0) & (S.wl >= 400) & (S.wl <= 700)))
        if neg:
            st.error(f"{neg} visible bands are NEGATIVE. R_rs cannot be negative; this "
                     "indicates over-subtraction in the glint correction, not water "
                     "with negative reflectance.")

# ============================================================ 2 · SENSOR VIEW
with tabs[1]:
    st.subheader("What each satellite would see")
    st.markdown(
        "A satellite integrates over a band with a finite spectral response. Comparing "
        "a 1 nm field spectrum to a satellite product without convolving first compares "
        "two different quantities.")

    if S.wl is None:
        st.info("Load a spectrum first.")
    else:
        pick = st.multiselect("Sensors", list(SENSORS),
                              default=["MODIS-Aqua", "Sentinel-3 OLCI",
                                       "PACE OCI (blue-band subset)"])
        giop_only = st.checkbox("Show only the bands GIOP uses", value=True)

        fig, ax = fig_axes(4.2)
        ax.plot(S.wl, S.rrs, color="0.4", lw=1, label="field spectrum")
        rows = []
        for i, name in enumerate(pick):
            c, v, info = convolve(S.wl, S.rrs, name, use_giop_bands=giop_only)
            ax.plot(c, v, "o--", ms=7, lw=1, label=f"{name} ({info['method']})")
            for cc, vv in zip(c, v):
                rows.append({"sensor": name, "band (nm)": round(float(cc), 2),
                             "R_rs (sr⁻¹)": None if not np.isfinite(vv) else round(float(vv), 6)})
            for band, why in info["dropped"]:
                st.warning(f"{name} {band:.1f} nm dropped: {why}")
        ax.set_xlim(max(350, S.wl.min()), min(950, S.wl.max()))
        ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("$R_{rs}$ (sr$^{-1}$)")
        ax.legend(fontsize=8); show(fig)

        if rows:
            st.dataframe(rows, width='stretch', height=260)

        with st.expander("Band definitions and how good they are"):
            for name in pick:
                s = SENSORS[name]
                st.markdown(f"**{name}** — response: `{s.response}`")
                st.caption(s.note)
            caveat(
                "Only <b>MODIS-Aqua</b> uses a measured spectral response function, "
                "shipped with the reference GIOP code. Every other sensor here is a "
                "nominal centre and width convolved with a Gaussian: good for the "
                "near-rectangular ocean-colour bands, but not the real instrument "
                "response. For publication, use the mission's own SRF.")

        st.markdown("#### BRDF normalisation  [EXT]")
        st.markdown(
            "Satellite ocean-colour products are **exact normalised** water-leaving "
            "reflectance: what you would see looking straight down with the sun "
            "overhead. Field data taken at 40° from nadir with the sun at 30–60° is a "
            "different quantity. Correcting for that is not optional if you are "
            "validating a satellite product.")
        b1, b2, b3, b4 = st.columns(4)
        brdf_on = b1.checkbox("Apply BRDF normalisation", value=False)
        solz_b = b2.number_input("Solar zenith θ_s (°)", 0.0, 80.0, 40.0, 1.0)
        senz_b = b3.number_input("View zenith θ_v (°)", 0.0, 60.0, 40.0, 1.0)
        relaz_b = b4.number_input("Rel. azimuth Δφ (°)", 0.0, 180.0, 135.0, 5.0)
        chl_b = st.number_input("Chlorophyll for the f/Q table (mg m⁻³)",
                                0.03, 10.0, 0.5, 0.05,
                                help="The Morel f/Q tables are indexed by chlorophyll; "
                                     "0.03–10 is their range.")
        try:
            from giop.aopiop import normalize_brdf
            rrs_n, fac = normalize_brdf(S.wl, S.rrs, chl_b, solz_b, senz_b, relaz_b)
            fig, ax = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
            ax[0].plot(S.wl, S.rrs, "k-", lw=1.3, label="measured (at this geometry)")
            ax[0].plot(S.wl, rrs_n, "b--", lw=1.3,
                       label="BRDF-normalised (nadir, sun overhead)")
            ax[0].set_ylabel("$R_{rs}$ (sr$^{-1}$)"); ax[0].legend(fontsize=8)
            ax[1].plot(S.wl, 100 * (fac - 1), "r-", lw=1.4)
            ax[1].axhline(0, color="0.6", lw=0.8)
            ax[1].set_ylabel("correction (%)"); ax[1].set_xlabel("wavelength (nm)")
            fig.tight_layout(); show(fig)
            st.caption(f"Correction spans **{100*(fac.min()-1):+.1f} % to "
                       f"{100*(fac.max()-1):+.1f} %** across the spectrum.")
            caveat(
                "This is the Morel f/Q ratio only. The full normalisation also carries "
                "an air–water transmittance and refraction term that depends on viewing "
                "angle: a few percent at 40°, against an f/Q ratio that reaches 10 % or "
                "more. The dominant part is here; the remainder is not. Do not call this "
                "a complete NASA-style exact normalisation.")
            if brdf_on and st.button("Apply BRDF to the working spectrum"):
                S.rrs = rrs_n
                S.source = f"{S.source} + BRDF({solz_b:.0f}/{senz_b:.0f}/{relaz_b:.0f})"
                st.success("Applied. The working spectrum is now nadir-normalised.")
        except Exception as exc:
            st.warning(f"BRDF factor unavailable: {type(exc).__name__}: {exc}")

        st.markdown("#### Use a sensor's bands for the inversion")
        use_sensor = st.selectbox("Resample the spectrum to", ["(keep native)"] + list(SENSORS))
        if use_sensor != "(keep native)" and st.button("Apply resampling"):
            c, v, info = convolve(S.wl, S.rrs, use_sensor, use_giop_bands=giop_only)
            good = np.isfinite(v)
            if good.sum() < 4:
                st.error("Fewer than 4 finite bands survive; cannot invert.")
            else:
                S.wl, S.rrs = c[good], v[good]
                S.source = f"{S.source} → {use_sensor}"
                st.success(f"Resampled to {use_sensor}: {good.sum()} bands.")

# ============================================================== 3 · INVERSION
with tabs[2]:
    st.subheader("Invert")
    if S.wl is None:
        st.info("Load a spectrum first.")
    else:
        m = (S.wl >= wl_lo) & (S.wl <= wl_hi) & np.isfinite(S.rrs)
        wl, rrs = S.wl[m], S.rrs[m]
        st.caption(f"{m.sum()} bands in the {wl_lo}–{wl_hi} nm window.")

        chl_mode = st.radio("Seed chlorophyll", ["OC4 band ratio", "manual"],
                            horizontal=True)
        if chl_mode == "manual":
            chl = st.number_input("Chl (mg m⁻³)", 0.001, 200.0, 0.2, 0.05)
        else:
            try:
                pick = lambda t: float(rrs[int(np.argmin(np.abs(wl - t)))])  # noqa: E731
                chl = float(get_oc(pick(443), pick(490), pick(510), pick(555), "oc4"))
                st.caption(f"OC4 seed = **{chl:.4f}** mg m⁻³")
            except Exception:
                chl = 0.2

        if st.button("Run inversion", type="primary"):
            try:
                cfg = build_cfg()
                kw = {}
                if inv == "bounded":
                    kw["sigma"] = sigma_for(rrs)
                S.result = giop(wl, rrs, chl, cfg=cfg, **kw)
                S.fit_sigma = sigma_for(rrs)
            except ConfigurationError as exc:
                st.error(f"Configuration refused: {exc}")
                S.result = None
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
                S.result = None

        r = S.result
        if r is not None:
            if not r.converged:
                st.error("Did not converge. Upstream returns −999 here.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("a_dg(443)  m⁻¹", f"{r.adg443:.4g}")
                c2.metric("b_bp(443)  m⁻¹", f"{r.bbp443:.4g}")
                c3.metric("aph amplitude", f"{r.chl:.4g}")
                chi2 = float(np.sum(((r.rrs_model_above - r.rrs_obs) / S.fit_sigma) ** 2))
                c4.metric("χ²_ν", f"{chi2 / max(len(r.wl) - 3, 1):.3f}")

                if r.x[2] < 0 or r.x[0] < 0:
                    st.error(
                        "A retrieved amplitude is NEGATIVE. Absorption cannot be "
                        "negative: the prescribed S_dg / η / aph* are wrong for this "
                        "water, or the spectrum is contaminated. The `bounded` solver "
                        "prevents this, but preventing it does not make the shapes right.")

                st.caption(f"Prescribed: S_dg = {r.sdg:.5g} nm⁻¹, η = {r.eta:.4g}. "
                           f"Seed chl {r.chl_seed:.4g}. QC: {r.qc_passed}.")
                if abs(r.chl_seed - r.chl) > 3 * max(r.chl, 1e-6):
                    st.warning(
                        f"The OC4 seed ({r.chl_seed:.3g}) and the retrieved aph "
                        f"amplitude ({r.chl:.3g}) disagree by more than a factor 3. "
                        "That is the signature of a blue-green ratio driven by "
                        "something other than phytoplankton.")

                fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
                ax[0].plot(r.wl, r.rrs_obs, "k-o", ms=4, label="measured")
                ax[0].plot(r.wl, r.rrs_model_above, "b--o", ms=3, label="GIOP model")
                ax[0].axhline(0, color="0.8", lw=0.7)
                ax[0].set_ylabel("$R_{rs}$ (sr$^{-1}$)"); ax[0].legend(fontsize=8)
                resid = (r.rrs_obs - r.rrs_model_above)
                ax[1].plot(r.wl, resid / S.fit_sigma, "r-o", ms=3)
                ax[1].axhline(0, color="0.5", lw=0.8)
                for k in (-2, 2):
                    ax[1].axhline(k, color="0.7", ls=":", lw=0.8)
                ax[1].set_ylabel("residual / σ")
                ax[2].plot(r.wl, r.aw, color="0.5", lw=1, label="$a_w$ (fixed)")
                ax[2].plot(r.wl, r.aph, "g-", label="$a_\\varphi$")
                ax[2].plot(r.wl, r.adg, "c-", label="$a_{dg}$")
                ax[2].plot(r.wl, r.apg, "m:", lw=2, label="$a_{pg}$ (better constrained)")
                ax[2].plot(r.wl, r.bbp, "r-", label="$b_{bp}$")
                ax[2].set_ylabel("m$^{-1}$"); ax[2].set_xlabel("wavelength (nm)")
                ax[2].legend(fontsize=8, ncol=2)
                fig.tight_layout(); show(fig)

# ============================================================ 4 · UNCERTAINTY
with tabs[3]:
    st.subheader("Uncertainty")
    st.markdown(
        "GIOP as published returns **no uncertainty at all**. Two estimators are "
        "offered here, and they measure different things. Reporting only the first is "
        "how a retrieval acquires an error bar that excludes its own dominant error.")

    r = S.result
    if r is None or not r.converged:
        st.info("Run a converged inversion first.")
    else:
        st.markdown("#### 1 · Linearised covariance — conditional on the shapes")
        try:
            cov, sig = linearised_covariance(r, S.fit_sigma)
            names = ["a_dg(443)", "b_bp(443)", "aph amplitude"]
            st.dataframe([{"parameter": n, "value": float(r.x[i]),
                           "1σ": float(sig[i]),
                           "relative": f"{100*sig[i]/abs(r.x[i]):.1f} %"
                                       if r.x[i] else "n/a"}
                          for i, n in enumerate(names)], width='stretch')
            rho_ac = cov[0, 2] / np.sqrt(cov[0, 0] * cov[2, 2])
            st.caption(f"a_dg ↔ aph correlation = **{rho_ac:+.2f}**. Strongly negative "
                       "means the two exchange amplitude while their sum stays fixed, "
                       "which is why a_pg is the defensible product.")
        except Exception as exc:
            st.warning(f"Could not compute: {exc}")

        st.markdown("#### 2 · Shape ensemble — the error the first one excludes")
        if st.button("Run shape ensemble (re-inverts over S_dg × η × aph*)"):
            with st.spinner("inverting across the prescription grid…"):
                try:
                    ens = shape_ensemble(r.wl, r.rrs_obs, r.chl_seed,
                                         base_cfg=build_cfg())
                    S.ens = ens
                except Exception as exc:
                    st.error(f"{exc}")
                    S.ens = None

        ens = S.get("ens")
        if ens:
            st.caption(f"{ens['n_members']} members converged, {ens['n_failed']} failed.")
            rows = []
            for key, label in (("adg443", "a_dg(443)"), ("bbp443", "b_bp(443)"),
                               ("aph_amplitude", "aph amplitude")):
                d = ens[key]
                rows.append({"parameter": label, "median": d["median"],
                             "16th": d["p16"], "84th": d["p84"],
                             "min": d["min"], "max": d["max"],
                             "fraction negative":
                                 f"{100*ens['negative_fraction'][key]:.0f} %"})
            st.dataframe(rows, width='stretch')
            fneg = ens["negative_fraction"]["aph_amplitude"]
            if fneg > 0:
                caveat(f"<b>{100*fneg:.0f} % of ensemble members return a negative aph "
                       "amplitude.</b> The spread here is the error contributed by "
                       "prescribing S_dg, η and aph*, and it is typically one to two "
                       "orders of magnitude larger than the linearised σ above.")

# ========================================================= 5 · IDENTIFIABILITY
with tabs[4]:
    st.subheader("Can this band set separate these constituents?")
    r = S.result
    if r is None or not r.converged:
        st.info("Run a converged inversion first.")
    else:
        ang = eigenvector_angles(r)
        cond = condition_number(r)
        c1, c2 = st.columns([2, 1])
        with c1:
            st.dataframe([{"pair": k, "angle (deg)": round(v, 1),
                           "verdict": "DEGENERATE" if v < DEGENERACY_THRESHOLD_DEG
                                      else "separated"}
                          for k, v in ang.items()], width='stretch')
        c2.metric("cond(A)", f"{cond:.3g}")

        worst = min(ang.values())
        if worst < DEGENERACY_THRESHOLD_DEG:
            st.error(f"Smallest angle {worst:.1f}° is below the {DEGENERACY_THRESHOLD_DEG}° "
                     "threshold: only the SUM of that pair is constrained by these bands.")
        else:
            caveat(
                f"No pair is below {DEGENERACY_THRESHOLD_DEG}° <b>at the prescribed "
                "shapes</b>. That does not mean the retrieval is accurate: this "
                "diagnostic holds S_dg, η and aph* fixed, and their choice is normally "
                "the larger error. Use the shape ensemble for that.")

        st.markdown(
            "**Why more bands do not help.** A hyperspectral spectrum has hundreds of "
            "channels but the model still has three smooth eigenvectors, so the "
            "effective rank stays 3. Extra bands buy noise averaging, not new degrees "
            "of freedom, and they do not relieve the a_dg / a_φ degeneracy.")

# ================================================================= 6 · EXPORT
with tabs[5]:
    st.subheader("Export")
    r = S.result
    if r is None:
        st.info("Nothing to export yet.")
    else:
        spec = np.column_stack([r.wl, r.rrs_obs, r.rrs_model_above, r.aph, r.adg,
                                r.apg, r.bbp, r.aw, r.bbw])
        hdr = ("wavelength_nm,rrs_obs_sr-1,rrs_model_above_sr-1,aph_m-1,adg_m-1,"
               "apg_m-1,bbp_m-1,aw_m-1,bbw_m-1")
        buf = io.StringIO()
        buf.write(f"# GIOP retrieval, source: {S.source}\n")
        buf.write(f"# solver={inv} aph={aph_opt} S_dg={r.sdg:.6g} eta={r.eta:.6g} "
                  f"fq={fq} trans={trans}\n")
        buf.write(f"# adg443={r.adg443:.6g} bbp443={r.bbp443:.6g} "
                  f"aph_amp={r.chl:.6g} chl_seed={r.chl_seed:.6g}\n")
        buf.write("# NOTE: these amplitudes are conditional on the prescribed S_dg, "
                  "eta and aph*. See THEORY.md sect. 11.\n")
        np.savetxt(buf, spec, delimiter=",", header=hdr, comments="")
        st.download_button("Download spectra CSV", buf.getvalue(),
                           file_name="giop_retrieval.csv", mime="text/csv")
        st.code(buf.getvalue()[:600] + " …", language="text")

# ================================================================== ℹ GUIDE
with tabs[6]:
    st.subheader("Guide")
    guide = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "docs", "STREAMLIT_GUIDE.md")
    if os.path.exists(guide):
        st.markdown(open(guide).read())
    else:
        st.info("docs/STREAMLIT_GUIDE.md not found.")
