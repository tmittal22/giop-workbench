# GIOP Workbench — user guide

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate claude-science-env
cd giop_python
streamlit run app/giop_app.py
```

Opens at <http://localhost:8501>.

The tool is built around one idea: **GIOP returns three numbers and no uncertainty, and
the dominant error is not measurement noise but the spectral shapes the model
prescribes.** Every panel exists to make that visible rather than to hide it behind a
clean-looking answer.

---

## 1 · Data

Four ways in.

**Demo** — the six-band example from upstream `run_giop.m`. Use it to confirm the install:
GIOP-DC must return a_dg(443) = 0.0441, b_bp(443) = 0.0033, aph amplitude = 0.3693.

**NaturaSpec `.sed`** — upload the water and sky scans. The panel is optional: leave it
out and the panel radiance comes from the water file's own `Rad. (Ref.)` column, which is
the DARWin reference-scan workflow. Set the panel reflectance, ρ and the residual glint
correction. See `fieldrrs/FIELD_PROTOCOL.md` for how to collect the scans.

**fieldrrs output** — a CSV written by
[`fieldrrs`](https://github.com/tmittal22/fieldrrs), read **with its metadata**. Either a
per-station file or `rrs_all_stations.csv`.

This is the one to use if the data came off a NaturaSpec, because the per-station file
carries the conditions: ρ, the residual-glint method, the solar and viewing geometry, the
wind speed, the panel reflectance and the instrument footprint. The panel then tells you
what should change your interpretation rather than leaving you to remember it:

- wind above ~5 m s⁻¹ while ρ is still 0.028, which biases blue R_rs high;
- `nir_zero` applied, which deletes real signal in turbid water;
- no geometry recorded, so the BRDF correction cannot be applied;
- negative visible bands, meaning the glint subtraction over-corrected.

If the geometry is present you get one-click BRDF normalisation using **the angles that
were actually recorded**. If it is absent the button does not appear, because applying a
geometry correction with a guessed geometry is worse than not applying one.

The batch file carries bands only and no conditions, so it comes with that caveat
attached.

**CSV** — two columns, wavelength (nm) and R_rs (sr⁻¹).

The green band on the plot is the inversion window. Negative R_rs in the visible is
flagged as an error, not a warning: R_rs cannot be negative, so it means the glint
subtraction over-corrected.

---

## 2 · Sensor view — what MODIS, OLCI and PACE would see

A satellite integrates over a band with a finite spectral response. Comparing a 1 nm
field spectrum to a satellite product without convolving first compares two different
quantities.

| sensor | response used | fidelity |
|---|---|---|
| MODIS-Aqua | **measured SRF**, 16 bands, shipped with the reference GIOP code | real instrument response |
| Sentinel-3 OLCI | nominal Oa1–Oa21 centres and widths, Gaussian | approximation |
| PACE OCI | ~5 nm FWHM, hyperspectral or the heritage 6-band subset | approximation |
| VIIRS-SNPP | nominal M1–M7, Gaussian | approximation |
| SeaWiFS | nominal, Gaussian | approximation |

Only MODIS-Aqua uses a real spectral response function. The others are near-rectangular
narrow bands so a Gaussian is a good stand-in, but for publication use the mission's own
SRF and pass it to `giop.sensors.convolve_with_srf`.

**Why this panel earns its place.** On the same water, OLCI's 560 nm band and SeaWiFS's
555 nm band differ by several percent purely from band placement. If you are validating a
satellite product against field data, that difference is not noise, it is the thing you
must remove before you can say anything about agreement.

Bands whose response falls more than half outside your measured spectrum come back NaN
and are reported, rather than being silently extrapolated.

**Apply resampling** replaces the working spectrum with the sensor's bands, so you can
ask "what would GIOP retrieve if this were a MODIS pixel?"

### BRDF normalisation [EXT]

Satellite ocean-colour products are **exact normalised** water-leaving reflectance: what
you would see looking straight down with the sun overhead. Field data taken at 40° from
nadir with the sun at 30–60° is a different quantity, and comparing the two without
correcting compares two different things.

Because r_rs = (f/Q)·u and u is a property of the water alone, the ratio between two
geometries is just the ratio of their f/Q (THEORY.md G18). Measured:

| geometry | correction |
|---|---|
| nadir, sun overhead | **exactly 1.000** (the definitional check) |
| θ_s 30°, 40°/135° | −7.5 % |
| θ_s 45°, 40°/135° | −10.5 % |
| θ_s 60°, 40°/135° | −12.7 % |
| θ_s 30°, 40°/90° | −4.7 % |

So a field spectrum sits 8–13 % away from what a satellite reports on the same water,
from geometry alone. The panel plots the corrected spectrum against the measured one and
the percent correction beneath it.

⚠ **This is the Morel f/Q ratio only.** The full normalisation also carries an air–water
transmittance and refraction term that depends on viewing angle: a few percent at 40°,
against an f/Q ratio that reaches 10 % or more. The dominant part is here and the
remainder is not. Do not present this as a complete NASA-style exact normalisation.

---

## 3 · Inversion

Three solvers.

**`fmin` (GIOP-DC)** — the published model: unweighted least squares, unconstrained
Nelder–Mead. Use this to reproduce reference numbers.

**`bounded` [EXT]** — an addition, not GIOP-DC. Two changes:

- *Positivity.* The amplitudes are bounded to ≥ 0. GIOP-DC is unconstrained, and on the
  demo spectrum 25 % of the shape ensemble returns a **negative** aph amplitude. A
  negative absorption coefficient is not a retrieval.
- *Weighting.* Residuals are divided by a per-band σ, so the bright blue-green bands stop
  dominating and χ²_ν becomes interpretable.

Measured on the demo spectrum: where the answer is physical the two agree to about 0.5 %,
and where GIOP-DC goes negative the bounded solver rails at zero and redistributes into
a_dg:

| S_dg (nm⁻¹) | fmin aph | bounded aph | fmin a_dg | bounded a_dg |
|---|---|---|---|---|
| 0.010 | **−0.2752** | 0.0000 | 0.0970 | 0.0829 |
| 0.012 | **−0.0233** | 0.0000 | 0.0748 | 0.0734 |
| 0.018 | +0.3693 | 0.3709 | 0.0441 | 0.0444 |
| 0.025 | +0.5772 | 0.5838 | 0.0299 | 0.0301 |

Bounding stops the retrieval being unphysical. **It does not make the prescribed shapes
right** — a railed zero is still telling you the model is wrong for that water.

**`lmi`** — linear matrix inversion, exact given u, but noise enters through a square root
with no smoothing.

The residual panel is in units of σ. Structure there (a run of same-sign residuals rather
than scatter) means a missing constituent or a wrong shape, not noise.

The tool warns when the OC4 seed chlorophyll and the retrieved aph amplitude disagree by
more than a factor of 3. That is the signature of a blue-green ratio driven by something
other than phytoplankton, and it is exactly what happens on turbid or mineral-rich water.

---

## 4 · Uncertainty

Two estimators that measure **different things**.

**Linearised covariance** — from the Jacobian at the solution with your stated per-band σ.
This is the shape-*conditional* error: it assumes S_dg, η and aph* are correct. On the
demo spectrum it gives 1.4 %, 1.4 % and 4.2 %.

**Shape ensemble** — re-inverts across the grid of allowed S_dg, η and aph*
parameterisations and reports the spread. On the same spectrum, 60 members give a_dg(443)
spanning 0.029–0.118 m⁻¹ and the aph amplitude spanning −0.43 to +0.61, with 25 % of
members negative.

**The prescription error is one to two orders of magnitude larger than the linearised
error.** Quoting the linearised σ alone produces a credible-looking error bar that
excludes the dominant term. Neither estimator is a posterior.

The a_dg ↔ aph correlation is reported because it is the mechanism: on the demo it is
−0.52, meaning the two absorbers trade amplitude while their sum stays nearly fixed.
**Report a_pg unless you can defend the split.**

---

## 5 · Identifiability

Pairwise angles between the columns of the design matrix, and its condition number. Below
about 15° a pair is not separated by the band set: only its sum is constrained.

The important subtlety, stated in the panel itself: these angles hold the shapes **fixed**.
A well-conditioned design can still sit on a badly wrong answer if the prescribed shapes
are wrong. On the demo spectrum no pair is below 15° and the condition number is 837, yet
the shape ensemble spans a factor of 4 in a_dg. Conditioning and accuracy are different
questions.

**Hyperspectral data does not fix this.** The model has three smooth eigenvectors no
matter how many channels you feed it, so the effective rank stays 3. Extra bands buy noise
averaging, not new degrees of freedom.

---

## 6 · Export

CSV of every spectrum plus a header recording the solver, the eigenvector choices, the
prescribed S_dg and η, the seed chlorophyll and a reminder that the amplitudes are
conditional on those choices. A retrieval without its configuration cannot be reproduced.

---

## What this tool will not do for you

- It will not tell you your water is Case 1. GIOP assumes exactly three optically active
  constituents plus pure water (THEORY.md A1). A fourth absorber, mineral, ash or iron,
  gets projected onto a_dg or b_bp and comes back as a plausible number.
- It has **no Raman or fluorescence term**. `giop.empirical.raman_correction` implements
  Lee et al. (2013) as an optional preprocessing step at six specific bands.
- Pure-water backscattering is the Morel power law with no salinity or temperature
  dependence, which runs low in the blue at oceanic salinity (THEORY.md A8).
- R_rs from `fieldrrs` is at the **measurement geometry**, not normalised to nadir.
  Satellite products are BRDF-normalised via Morel f/Q. That correction exists in
  `giop.aopiop` but is not applied automatically.

## Further reading

- `THEORY.md` — every equation, symbol, unit and assumption, with the ledger in §11
- `PORTING_NOTES.md` — nine defects found in the upstream MATLAB
- `VALIDATION.md` — the evidence, with what could not be checked stated first
- Werdell, P.J., et al. (2013), *Applied Optics* 52(10), 2019–2037,
  doi:10.1364/AO.52.002019
