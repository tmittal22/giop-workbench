# giop — GIOP ocean-colour inversion in Python, with field-spectrometer ingestion

A careful Python port of the reference MATLAB GIOP (Werdell et al. 2013), plus the layer
that turns Spectral Evolution NaturaSpec `.sed` scans into the R_rs that GIOP needs, plus a
GUI.

- **What it computes.** Given remote-sensing reflectance R_rs(λ), it decomposes the water
  column into phytoplankton absorption a_φ, combined CDOM + detrital absorption a_dg, and
  particulate backscattering b_bp.
- **The physics is in [`THEORY.md`](THEORY.md)** — every equation, symbol, unit, assumption,
  and a map from each equation to the function and the upstream MATLAB line.
- **The port decisions are in [`PORTING_NOTES.md`](PORTING_NOTES.md)** — including eight
  defects and inconsistencies found in the upstream MATLAB.
- **The evidence is in [`VALIDATION.md`](VALIDATION.md)** — with numbers, and with what
  could not be checked stated first.

Upstream: <https://github.com/kelseybisson/GIOP> @ `ef9b93f`, kept unmodified in
`upstream_matlab/`.

---

## Install

```bash
git clone https://github.com/tmittal22/giop-workbench
cd giop-workbench
pip install -e .
python scripts/fetch_data.py      # one time: gets the reference optical tables
```

**The reference optical tables are not in this repository.** The upstream GIOP repo
carries no licence file, so its terms are unresolved, and redistributing its tables would
pass that question on to you. `scripts/fetch_data.py` downloads them from the original
source, pinned to commit `ef9b93f`. See
[`src/giop/data/PROVENANCE.md`](src/giop/data/PROVENANCE.md).

Dependencies: numpy, scipy, matplotlib. The Streamlit workbench adds streamlit; the
desktop GUI uses tkinter, which ships with CPython.

## The Streamlit workbench

```bash
streamlit run app/giop_app.py          # or double-click run_app.bat on Windows
```

Seven panels: data in (`.sed`, CSV, or the published demo), what each satellite would
see, the inversion, uncertainty, identifiability, export, and the guide. Full walkthrough
in [`docs/STREAMLIT_GUIDE.md`](docs/STREAMLIT_GUIDE.md).

## Check it works

```bash
export PYTHONPATH=$PWD/src
python -m pytest tests/ -q
python examples/demo_giop.py      # writes figures/demo_giop.png
```

The port reproduces the eigenvalues published in upstream's `run_giop.m` to every digit
those are quoted at: `[0.0441, 0.0033, 0.3693]` for the nonlinear solver and
`[0.0414, 0.0022, 0.1058]` for the linear matrix inversion.

## Use it

```python
import numpy as np
from giop import giop, get_oc

wl  = np.array([412., 443, 490, 510, 555, 670])          # nm
rrs = np.array([0.003478, 0.004074, 0.004465,            # sr^-1, ABOVE water
                0.003588, 0.002494, 0.000051])

chl = get_oc(rrs[1], rrs[2], rrs[3], rrs[4], 'oc4')      # seed chlorophyll
res = giop(wl, rrs, chl, qc=0.33)

res.adg443      # a_dg(443)  m^-1
res.bbp443      # b_bp(443)  m^-1
res.chl         # a_phi amplitude; equals chlorophyll only for Bricaud aph*
res.apg         # a_phi + a_dg spectrum, better constrained than either alone
res.rrs_model_above   # modelled R_rs above water, comparable with the input
```

Everything upstream's `gopt` struct controlled is a `GiopConfig` field:

```python
from giop import GiopConfig
cfg = GiopConfig(aph='ciotti', sf=0.3, sdg='qaa', eta=1.2, inv='lmi', trans='flat')
res = giop(wl, rrs, chl, cfg=cfg)
```

## Ingest NaturaSpec data

A spectroradiometer does not measure R_rs. The standard above-water method needs **three**
scans with the same foreoptic (THEORY.md §10):

| scan | what to point at |
|---|---|
| target | the water, ~40° from nadir, ~135° in azimuth from the sun |
| sky | the sky at the mirror angle (same azimuth, 40° from zenith) |
| panel | a horizontal Spectralon panel of known reflectance |

```python
from giop.io import read_sed, rrs_from_sed_triplet
from giop import giop, get_oc

t, s, p = (read_sed(f) for f in ('water.sed', 'sky.sed', 'panel.sed'))
out = rrs_from_sed_triplet(t, s, p, panel_reflectance=0.99, rho=0.028)
print(out.notes)                      # read these; they flag over-subtraction

m = (out.wavelength >= 400) & (out.wavelength <= 700)     # aph* is only defined here
wl, rrs = out.wavelength[m], out.rrs[m]
res = giop(wl, rrs, get_oc(*[rrs[np.argmin(abs(wl - x))] for x in (443, 490, 510, 555)], 'oc4'))
```

`ρ = 0.028` is Mobley (1999) for that geometry in wind below ~5 m s⁻¹. It is the single
largest error term above water and it is sea-state dependent. If the wind was up, say so and
change it.

If your data is already R_rs, `giop.io.read_csv_spectra` reads CSV in either layout.

## GUI

```bash
giop-gui           # or: python -m giop.gui
```

Load a `.sed` triplet or a CSV, set the panel reflectance / ρ / residual correction, pick the
wavelength window and the GIOP parameterisation, invert, and export. The assumptions panel is
always visible and not dismissible, because the numbers are not interpretable without it.

## What to be careful about

These are not hypothetical; each is measured in `VALIDATION.md`.

1. **GIOP returns no uncertainty.** The cost function is unweighted least squares with no
   covariance. `giop.uncertainty` adds two estimators that upstream does not have.
2. **The shape parameters are prescribed, not fitted.** On the demo spectrum, sweeping S_dg
   across its plausible range (0.010–0.025 nm⁻¹) moves a_dg(443) by a factor 3.2 and takes the
   a_φ amplitude from −0.275 to +0.577, crossing zero at 0.01225. Across the full prescription
   ensemble, 25 % of members return a **negative** a_φ.
3. **Report a_pg unless you can defend the split.** a_φ and a_dg trade off against each other
   (correlation −0.52); their sum is much better constrained than either.
4. **There is no positivity constraint.** A negative retrieval is a sign the prescription is
   wrong, not a measurement.
5. **aph\* is tabulated 400–700 nm only.** A NaturaSpec spectrum runs far past that; the
   inversion window has to be trimmed and the code will say so rather than extrapolate.
6. **No Raman or fluorescence** is in the forward model. `giop.empirical.raman_correction`
   implements Lee et al. (2013) as an optional preprocessing step at six specific bands.

## Licensing

This port is the user's own work. The upstream MATLAB repository carries **no licence file**,
so its terms are unresolved; see [`src/giop/data/PROVENANCE.md`](src/giop/data/PROVENANCE.md)
for the reference tables, their original published sources, and what that means for
redistribution. The underlying optical data are published science (Pope & Fry 1997,
Bricaud et al. 1998, Morel et al. 2002, Ciotti & Bricaud 2006); the `.txt` encodings came
through NASA GSFC.

## Citing

Cite the model, not this port:

> Werdell, P.J., et al. (2013). Generalized ocean color inversion model for retrieving marine
> inherent optical properties. *Applied Optics*, 52(10), 2019–2037. doi:10.1364/AO.52.002019
