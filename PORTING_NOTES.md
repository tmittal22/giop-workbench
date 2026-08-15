# PORTING NOTES — MATLAB to Python, decision by decision

Upstream: <https://github.com/kelseybisson/GIOP> @ `ef9b93f`, pristine copy in
`upstream_matlab/`. Read `THEORY.md` first; this file only covers where the Python
departs from the MATLAB, and why.

No MATLAB or Octave is installed on this host, so the port could **not** be validated by
running the original side by side. It is validated against the eigenvalues published in
`run_giop.m` comments, against a second in-repo copy of a_w, and against the worked
examples in `fastsmooth.m`'s own header. That is stated plainly in `VALIDATION.md`
rather than implied to be stronger than it is.

---

## 1. Defects and inconsistencies found in the upstream MATLAB

These are properties of the original, found by reading it and by writing tests for paths it
ships but never exercises. None is fixed silently.

### D1 — a subsurface model spectrum is plotted as if it were above water

`giop.m:235` computes `mrrs = g0*u + g1*u^2`, which is **subsurface** r_rs. `run_giop.m:96`
then plots it against the above-surface input `rrs` and labels both `R_rs`. The two differ
by the interface factor, about 1.9. The retrieval is unaffected because the internal QC
test (`giop.m:247`) correctly compares `mrrs` against `rin`.

**Port:** `GiopResult` carries both `rrs_model_subsurface` (upstream's `mrrs`, bit-for-bit)
and `rrs_model_above` (back-transformed through G4c). The demo figure and the GUI plot the
above-water one against the above-water observation, and the demo figure shows the
subsurface curve too, labelled, so the size of the discrepancy is visible.

### D2 — Morel f/Q combined with LMI is a division by zero

The Morel option sets `g1 = 0` (`giop.m:67`). The LMI branch computes
`q = (-g0 + sqrt(g0^2 + 4*g1*rin)) / (2*g1)` at `giop.m:190`. MATLAB divides by zero
without raising, yielding Inf or NaN, which then propagates into the QR solve and comes
back as garbage rather than as an error.

**Port:** `GiopConfig.validate()` raises `ConfigurationError` naming the reason.
Pinned by `tests/test_physics.py::TestConfigGuards::test_morel_plus_lmi_is_refused`.

### D3 — the S_dg options mix above-surface and subsurface reflectance

`'qaa'` uses `rin` (subsurface) at `giop.m:118` while `'obpg'` uses `rrs` (above surface)
at `giop.m:121`. Both are described as blue/green ratios. Whether this is intentional
(each parameterisation was fitted against whichever quantity its author used) or an
oversight cannot be determined from the source.

**Port:** preserved exactly, and flagged in the docstring of `model.sdg_from_option` so it
cannot be "cleaned up" by a later reader.

### D4 — `read_merged_files.m` computes eta from above-surface reflectance

`read_merged_files.m:42` uses `Rrs_r(:,2)./Rrs_r(:,5)` in the QAA eta expression, where
QAA and `giop.m:97` both use the subsurface ratio. This is in a personal analysis script,
not in the model.

**Port:** not reproduced. `model.eta_qaa` takes subsurface r_rs, as `giop.m` does.

### D5 — anchor bands are selected as "first in a window", not "nearest"

`giop.m:30-32` selects `find(wl >= 410 & wl <= 415)` and later takes element `(1)`. On the
6-band satellite grid that window contains exactly one band. On a NaturaSpec 1 nm grid it
contains six, and the first is 410.0 nm, so "R_rs(412)" silently becomes R_rs(410). This
propagates into eta (G10) and into the `'obpg'` S_dg.

**Port:** `matlab_compat.nearest_band` picks the band nearest the nominal wavelength.
`tests/test_matlab_compat.py::TestBandSelection` pins both that the two rules agree on the
6-band grid (so the golden test is unaffected) and that they differ on a 1 nm grid.

### D6 — `fastsmooth.m` has CR-only line endings

The file is a single line as far as any Unix tool is concerned, and the sequence
`end` followed by `s(k+halfw)=...` reads as a nonsense token `ends(k+halfw)=...`.

**Port:** reconstructed as the standard published fastsmooth and validated against the two
worked examples in the file's own header comments
(`tests/test_matlab_compat.py::TestFastsmooth`), which is an independent check that the
reconstruction is right.

### D7 — `get_oc.m` silently falls through to OC4

Its `switch` has `otherwise: a = c(1,:)`, so a typo in the algorithm name returns SeaWiFS
OC4 chlorophyll instead of an error.

**Port:** `empirical.get_oc` raises on an unknown name.

### D8 — `get_chase_ap` supplies a_p where the model wants a_phi

`get_chase_ap.m` returns Chase et al. (2017) **total particulate** absorption, and
`giop_kb.m:132` assigns it to `gopt.aphs`, the phytoplankton eigenvector. The detrital
fraction is then represented twice, once there and once in a_dg. `giop_kb.m:190-192`
contains a commented-out attempt to address this.

**Port:** exposed as `aph='chase'` because upstream exposes it, but it emits a
`UserWarning` naming the double-count.

### D9 — the Morel full-geometry branch returns NaN outside 412.5–660 nm

`morel_fq.m:26-27` interpolates the f/Q lookup table, which is defined at seven wavelengths
from 412.5 to 660 nm, straight onto the 380–700 nm output grid:

```matlab
fq = interp1(w,fq,wvl,'cubic');
```

MATLAB's `interp1` returns NaN outside the data range, so this branch produces NaN below
412.5 nm and above 660 nm and never fills them. The sibling function `morel_read.m:82-84`
*does* fill its trailing NaNs, and the Appendix-B branch `morel_fq_appb.m:21-24` avoids the
problem entirely by clamping its query wavelengths into the table span first. So the three
Morel code paths handle their edges three different ways, and only the full-geometry one is
left broken. A single NaN in g₀ makes the entire cost function NaN.

Found by writing a test for a path that upstream ships but that nothing in the repository
exercises: `run_giop.m` and `run_giop_cr.m` both use the no-geometry branch.

**Port:** the query wavelengths are clamped to the table edge, matching what upstream's own
Appendix-B branch does, so both Morel branches now behave the same way. Pinned by
`tests/test_aopiop.py::TestGeometryDependence::test_full_geometry_branch_runs_and_differs_from_the_appendix_branch`.

---

## 2. MATLAB semantics that required a decision

### P1 — `interp1(..., 'cubic')`

Discussed in THEORY.md sect. 9.1. Both interpretations are implemented
(`matlab_compat.interp1`, `method='pchip'` or `'v5cubic'`), default `pchip`.

The published golden numbers cannot discriminate between them, because every `'cubic'`
call in the model path acts on a uniform grid where the demo wavelengths land exactly on
nodes. Measured difference on an off-node grid: see `VALIDATION.md` sect. 3.

### P2 — `fminsearch` to `scipy.optimize.minimize(method='Nelder-Mead')`

Same algorithm, same initial-simplex rule (5 % relative, 0.00025 absolute for a zero
component), same paired TolX/TolFun stopping test. MATLAB's `exitflag < 1` maps to
SciPy's `status != 0`, and upstream's `x = [-999,-999,-999]` on non-convergence is
reproduced via `FILL`.

### P3 — the LMI QR expression

`giop.m:203-207` is the semi-normal equations plus one refinement step. THEORY.md sect. 6.2
derives that it collapses to `(A'A)^-1 A'b` computed through the QR factor; the port
implements that directly on the economy factor. Pinned against `scipy.linalg.lstsq` in
`tests/test_physics.py::TestLmiSolver`, with a control showing the naive
`inv(A'A)` route is numerically distinct on an ill-conditioned system.

### P4 — 1-based to 0-based indexing

Every `find(...)(1)` became an explicit index. The Bricaud normalisation row is selected by
exact value match on 442 nm (`aphstar.BRICAUD_NORM_WL`), not by position, so a change in
the table cannot silently shift it.

### P5 — MATLAB's silent NaN propagation

`interp1` returns NaN outside the data range, which upstream then carries into the fit
without comment. The port raises with a message naming the wavelength range at fault, for
the two cases that actually occur with field data: a_w outside 380-1150 nm, and aph*
outside 400-700 nm.

---

## 3. Additions that are not in upstream

Each is off by default, so the default path is GIOP-DC.

| addition | where | why |
|---|---|---|
| Above-water R_rs from field scans | `io.rrs` | upstream starts from R_rs; a spectroradiometer does not produce it |
| `.sed` reader | `io.sed` | the instrument's native format |
| Spectral resampling | `io.resample` | hyperspectral field data onto an inversion grid |
| `rrs_model_above` | `inversion` | D1 |
| Unit and range guards | `inversion._sanity_check_units` | percent reflectance and micrometre wavelengths are the two field-data unit errors that otherwise fail silently |
| `n_starts` multistart | `inversion._invert_fmin` | `gsm_invert.m:26-28` records that the retrieval is start-point sensitive; this measures it |
| `fit_shapes` (5-parameter) | `inversion._invert_fmin_shapes` | prescribing S_dg and eta is the largest structural assumption (A2, A3); hyperspectral data is the one case where relaxing it might be supportable. **Not GIOP-DC.** |
| GUI | `gui` | the requested deliverable |

---

## 4. Files not ported, and why

| file | status | reason |
|---|---|---|
| `run_giop.m`, `run_giop_cr.m` | ported as `examples/demo_giop.py` and tests | demo drivers |
| `single_wv_comps.m` | **not portable** | hard-codes `/Volumes/bissonk/...` paths, needs float-matchup CSVs not in the repo, and calls `bossbisson_fmin`, which **does not exist anywhere in the repository**. The reusable physics (the Lee et al. 2013 Raman correction, lines 105-117) was extracted into `empirical.raman_correction`. |
| `eta_expts.m` | **not portable** | depends on `Rrs412...` variables from a workspace that is not in the repo. Its experiment (sweep eta, look at the b_bp spread) is reproduced in `examples/demo_giop.py`. |
| `goci_play.m`, `read_merged_files.m` | **not portable** | hard-coded absolute paths to the author's machine and data |
| `notes.m` | not code | a list of research questions and an index array |
| `psrf.m` | not ported | Gelman-Rubin PSRF, unrelated to GIOP and unused by it. `arviz.rhat` is the maintained equivalent. |
| `gsm_invert.m`, `gsm_cost.m` | not ported | GSM is a different model (its own fixed eigenvectors and a different parameter order). GIOP reproduces GSM's *parameterisations* through `aph='gsm'`, `sdg='gsm'`, `eta='gsm'`, which is the comparison GIOP was designed to support. Porting the standalone GSM solver would add a second inversion path with no golden values to check it against. |
| `morel_fq.m` etc. | ported | `aopiop.py` |
| `estimate_bbp_from_Rrs.m` | ported | `empirical.qaa_bbp` |
| `fastsmooth.m` | ported | `matlab_compat.fastsmooth` |

## 5. Data files not carried over

`chase_ap17.mat`, `pureH2O_iop.mat`, `spectralresponse_modisa.mat` and the Morel tables are
shipped. The remaining `.mat` files (`inputsnums`, `longhurst`, `longviirs`, `ls2results`,
`model_giopbbp700s`, `modis_lat`, `modishr`, `paramcheck_20mar19`, `v_giop`, `v_resid`) are
outputs and inputs of Bisson's float/lidar matchup analysis, not model reference data, and
are not GIOP validation targets. They stay in `upstream_matlab/` only.
