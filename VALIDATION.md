# VALIDATION — what was checked, against what, with the numbers

Every number here was produced on this host on 2026-08-14 by the code in this repository.
Reproduce with:

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate claude-science-env
cd giop_python && export PYTHONPATH=$PWD/src
python -m pytest tests/ -q          # 89 passed, 1 skipped
python examples/demo_giop.py        # prints sect. 1 and 5, writes figures/demo_giop.png
```

## 0. What could NOT be done, stated first

**No MATLAB or Octave is installed on this host**, so the port was not validated by running
the original side by side. Nothing below should be read as "bit-identical to MATLAB". The
anchors that do exist are:

| anchor | strength |
|---|---|
| eigenvalues published in `run_giop.m` comments | strong, but only 4 decimal places and only one spectrum |
| `fastsmooth.m`'s own worked examples | strong for that function |
| `pureH2O_iop.mat` as a second in-repo copy of a_w | strong, and genuinely independent of the code path |
| forward/inverse self-consistency | weak (circular by construction) |
| analytic identity for the LMI solver | strong, but about the solver, not about GIOP |

A full MATLAB cross-check remains the one open validation item, and it needs a MATLAB licence.

---

## 1. Golden master: the published eigenvalues

`run_giop.m:88-89` and `:103-104` state the expected outputs for the demo spectrum
wl = [412, 443, 490, 510, 555, 670] nm,
R_rs = [0.003478, 0.004074, 0.004465, 0.003588, 0.002494, 0.000051] sr⁻¹,
seed chlorophyll from OC4.

| solver | quantity | MATLAB (published) | this port | abs. difference |
|---|---|---|---|---|
| fmin | a_dg(443) m⁻¹ | 0.0441 | 0.044119 | 1.9e-5 |
| fmin | b_bp(443) m⁻¹ | 0.0033 | 0.003278 | 2.2e-5 |
| fmin | a_φ amplitude | 0.3693 | 0.369303 | 3.4e-6 |
| lmi | a_dg(443) m⁻¹ | 0.0414 | 0.041351 | 4.9e-5 |
| lmi | b_bp(443) m⁻¹ | 0.0022 | 0.002162 | 3.8e-5 |
| lmi | a_φ amplitude | 0.1058 | 0.105784 | 1.6e-5 |

Every value agrees to the precision at which the reference is quoted. The OC4 seed
chlorophyll is 0.527107 mg m⁻³; it is not separately published, but it is load-bearing,
because the Bricaud a*_φ eigenvector is a function of it and the a_φ amplitude could not
land on 0.3693 if it were wrong.

**Prove-it-can-fail.** Four controls each break one piece of the physics and confirm the
golden test then fails (`tests/test_golden.py`):

| control | what it breaks |
|---|---|
| renormalise Bricaud a*_φ at 443 nm instead of 442 | the anchor at `get_bricaud_aph.m:24` |
| perturb g₀ by 10 % | the AOP–IOP closure (G1) |
| use the flat Gordon 2007 air-sea transmission | (G4b) instead of (G4a) |
| assert fmin and lmi disagree | catches one solver silently aliasing the other |

---

## 2. `fastsmooth` against its own published examples

`fastsmooth.m`'s header states two worked results. The upstream file has CR-only line
endings and renders as a single line with a mangled `ends(k+halfw)=` token, so the port is
a reconstruction; these examples are what confirms the reconstruction is correct.

| call | published | port |
|---|---|---|
| `fastsmooth([1 1 1 10 10 10 1 1 1 1], 3)` | `[0 1 4 7 10 7 4 1 1 0]` | exact match |
| `fastsmooth([1 1 1 10 10 10 1 1 1 1], 3, 1, 1)` | `[1 1 4 7 10 7 4 1 1 1]` | exact match |

Control: a width-5 smooth does not reproduce the width-3 answer.

---

## 3. Pure-water absorption, against the second in-repo copy

`optics_coef.txt` column 2 (what `get_aw.m` reads) against `pureH2O_iop.mat`, which
documents its own provenance in a NOTES field (Pope & Fry 1997 over 380–700 nm) and is not
read by any code path.

- 400–700 nm, 301 points: **max relative difference 1.76e-6**, median 0.0.

The two are the same table to round-off. b_bw from (G6) at 443 nm is 0.0024447 m⁻¹ against
0.0024362 from `optics_coef.txt` column 3 divided by two, a 0.35 % difference, consistent
with (G6) being a power-law fit to the same underlying Morel data rather than a lookup.

---

## 4. The LMI solver identity

THEORY.md §6.2 derives that upstream's four-line QR expression equals
(**A**ᵀ**A**)⁻¹**A**ᵀ**b** computed through the QR factor, with one refinement step.
Verified numerically against `scipy.linalg.lstsq` on a random 6×3 system: agreement to
rtol 1e-9.

Control: explicitly forming and inverting **A**ᵀ**A** on a near-collinear system
(columns separated by 1e-8) does **not** match `lstsq` to rtol 1e-6, so the two routes are
numerically distinct and the identity is not vacuous.

---

## 5. Identifiability and uncertainty on the demo spectrum

`giop.diagnostics`, 6-band demo, GIOP-DC settings:

| quantity | value |
|---|---|
| angle a_dg–b_bp | 33.4° |
| angle a_dg–a_φ | 27.5° |
| angle b_bp–a_φ | 28.5° |
| cond(**A**) | 837 |

No pair falls below the 15° degeneracy threshold, so **at the prescribed shapes** this
design is not degenerate. Linearised 1σ at a per-band uncertainty of 5e-5 sr⁻¹:
1.4 % on a_dg(443), 1.4 % on b_bp(443), 4.2 % on the a_φ amplitude, with an a_dg–a_φ
correlation of −0.52.

That is the shape-conditional error only. `giop.uncertainty.shape_ensemble` over
4 S_dg × 5 η × 3 a*_φ = 60 members, all converged:

| parameter | median | p16–p84 | full range |
|---|---|---|---|
| a_dg(443) m⁻¹ | 0.05102 | 0.03649 – 0.08577 | 0.02852 – 0.11781 |
| b_bp(443) m⁻¹ | 0.00329 | 0.00261 – 0.00439 | 0.00205 – 0.00637 |
| a_φ amplitude | 0.17993 | −0.16195 – 0.44845 | −0.43449 – 0.60951 |

**25 % of ensemble members return a negative a_φ amplitude.** The prescription error is one
to two orders of magnitude larger than the shape-conditional error. Any GIOP uncertainty
quoted from the linearised covariance alone excludes its own dominant term.

### 5b. Sensitivity to each prescribed shape separately

From `examples/demo_giop.py`, holding everything else at GIOP-DC:

- b_bp(443) across η ∈ [0.5, 2.0]: 0.00262 → 0.00376 m⁻¹, a factor 1.44.
- a_dg(443) across S_dg ∈ [0.010, 0.025] nm⁻¹: 0.09700 → 0.02995 m⁻¹, a factor 3.24.
- a_φ amplitude across the same S_dg range: −0.27518 → +0.57721, **crossing zero at
  S_dg = 0.01225 nm⁻¹**, i.e. inside the plausible range and below the GIOP-DC default of
  0.018.

`figures/demo_giop.png` shows all of this. The figure was generated and inspected; its
lower-right panel plots absolute values rather than ratios, because the a_φ amplitude
changes sign and normalising by a negative reference inverts the curve.

---

## 6. Interpolation semantics (THEORY.md §9.1)

The MATLAB `'cubic'` ambiguity, measured on a_w over an off-node 1 nm grid
(400.25, 401.25, ... nm), pchip against v5cubic:

- **max relative difference 5.70e-4 (0.057 %)**, median 2.4e-5, worst at 565.25 nm.

So the choice is real but small for a_w. It cannot be resolved by the golden test, because
every `'cubic'` call in the model path acts on a uniform table whose nodes the demo
wavelengths land on exactly; verified in
`tests/test_matlab_compat.py::TestInterp1::test_pchip_and_v5cubic_agree_on_grid_nodes`.
Default is `pchip`.

---

## 7. Field-data ingestion

Closure test: a synthetic `.sed` triplet is built from a **known** R_rs through (G16)/(G17),
written in DARWin's own format, read back, and inverted through the ingestion chain. The
recovered R_rs matches the input to `atol=1e-9` (`tests/test_io.py`).

Controls that make that test meaningful:

| control | result |
|---|---|
| use ρ = 0.05 instead of 0.028 | recovery fails, and biases low, as over-subtraction must |
| radiance path vs ratio path | agree to rtol 1e-8, limited only by file write precision |
| panel reflectance 0.99 → 0.50 | output scales exactly linearly, as (G16) requires |
| L_t < ρ·L_sky | reported in `notes`, not silently returned as negative R_rs |
| `nir_zero` on flat turbid water | removes the full 2e-3 sr⁻¹ signal, the documented failure mode |

`test_reads_a_real_file` is **skipped**: no real `.sed` file has been supplied yet. Drop one
into `tests/data/` and it runs. Until then the reader is validated only against synthetic
files written to the format extracted from `DARWin2.exe`, which is a real constraint on how
much the reader has been exercised.

---

## 8. GUI

The GUI was driven programmatically through its own event loop on a live X display: construct
the application, load the demo spectrum, run the inversion, read the status line, export.

- It returns `[0.044119, 0.003278, 0.369303]`, i.e. the golden values, so the GUI path and the
  API path are the same calculation and cannot drift apart silently
  (`tests/test_gui.py::test_gui_reproduces_the_golden_eigenvalues`).
- Its status line reports the prescribed S_dg and eta and the QC verdict, so a user can see
  what the retrieval was conditional on.
- Selecting Morel + LMI produces the `ConfigurationError` message rather than raising out of
  the event loop (PORTING_NOTES D2).
- The rendered three-panel figure was inspected: the model tracks the measurement, b_bp sits
  above b_bw across the band, and a_w takes over past 555 nm.

CSV export was exercised and produces the documented columns.

## 9. Suite status

```
89 passed, 1 skipped in 1.43 s
```

Skipped: the real-`.sed` test in §7. GUI tests skip automatically without a display.

## 10. Open items

1. **MATLAB cross-check.** The strongest available validation, blocked on a licence. It would
   settle §6 (the `'cubic'` ambiguity) definitively rather than by inference.
2. **A real `.sed` file** from the NaturaSpec Plus, to exercise the reader against the
   instrument rather than against a synthetic file.
3. **Field validation of the R_rs chain.** The closure test in §7 proves the arithmetic, not
   the physics. Only a match-up against an independent R_rs measurement tests ρ, the panel
   calibration and the residual correction together.
4. **b_bw for seawater.** (G6) is pure water with no salinity dependence (A8). Supplying a
   Zhang et al. (2009) table through `bb_water(model='table')` is supported and not done.
