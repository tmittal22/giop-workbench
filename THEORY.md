# THEORY — GIOP semi-analytical ocean-colour inversion, and the above-water field protocol that feeds it

Single source of truth for what `src/giop/` is supposed to implement mathematically. Every
equation is labelled `(Gn)`, every symbol is defined with units, and every equation carries a
pointer to the function that implements it and to the MATLAB line it was ported from.

Upstream: <https://github.com/kelseybisson/GIOP>, commit `ef9b93fe948420a2a153853b80ec293fd972a829`
(2021-11-29), a fork of the reference MATLAB GIOP released by J. Werdell (NASA GSFC, July 2013)
with additions by K. Bisson (OSU, 2019). Pristine copy in `upstream_matlab/`, never edited.

Primary reference: Werdell, P.J., and 18 co-authors (2013), *Generalized ocean color inversion
model for retrieving marine inherent optical properties*, Applied Optics 52(10), 2019–2037,
doi:10.1364/AO.52.002019.

---

## 1. Symbols and units

| symbol | code | meaning | units |
|---|---|---|---|
| λ | `wl` | wavelength in vacuum | nm |
| R_rs | `rrs` | remote-sensing reflectance, **above** the sea surface | sr⁻¹ |
| r_rs | `rin` | remote-sensing reflectance, **just below** the sea surface | sr⁻¹ |
| a | `atot` | total absorption coefficient | m⁻¹ |
| a_w | `aw` | pure-seawater absorption | m⁻¹ |
| a_φ | `aph` | phytoplankton absorption | m⁻¹ |
| a_dg | `adg` | combined CDOM + non-algal detrital absorption | m⁻¹ |
| a_pg | `apg` | a_φ + a_dg | m⁻¹ |
| b_b | `bbtot` | total backscattering coefficient | m⁻¹ |
| b_bw | `bbw` | pure-seawater backscattering | m⁻¹ |
| b_bp | `bbp` | particulate backscattering | m⁻¹ |
| u | `u`, `modx` | b_b / (a + b_b), dimensionless | – |
| a*_φ | `aphs` | chlorophyll-specific phytoplankton absorption (eigenvector) | m² mg⁻¹ |
| Chl | `chl` | chlorophyll-a concentration | mg m⁻³ |
| M_dg | `x[0]` | a_dg eigenvalue = a_dg(443) | m⁻¹ |
| M_bp | `x[1]` | b_bp eigenvalue = b_bp(443) | m⁻¹ |
| M_φ | `x[2]` | a_φ eigenvalue (= Chl when a*_φ is Bricaud-normalised) | mg m⁻³ |
| S_dg | `sdg` | a_dg exponential slope | nm⁻¹ |
| η | `eta` | b_bp power-law slope | – |
| g₀, g₁ | `g0`, `g1` | AOP–IOP coefficients | sr⁻¹, sr⁻¹ |
| θ_s | `solz` | solar zenith angle | deg |
| θ_v | `senz` | sensor (viewing) zenith angle, in air | deg |
| Δφ | `relaz` | relative azimuth, sun to sensor | deg |

Wavelength is always nm. Absorption and backscattering are always m⁻¹. Reflectance is always
sr⁻¹. There is no unit conversion anywhere in the model; a caller supplying R_rs in
percent, or λ in µm, gets silently wrong answers, so `giop.inversion.giop` range-checks both
(§9.3).

---

## 2. The forward model

### G1 — radiative-transfer closure (Gordon et al. 1988)

The subsurface remote-sensing reflectance is a polynomial in the single scattering-to-extinction
ratio u:

$$ r_{rs}(\lambda) \;=\; g_0\,u(\lambda) \;+\; g_1\,u(\lambda)^2 , \qquad
u(\lambda) \;=\; \frac{b_b(\lambda)}{a(\lambda) + b_b(\lambda)} \tag{G1} $$

with the GIOP-DC default (g₀, g₁) = (0.0949, 0.0794) sr⁻¹ from Gordon et al. (1988),
*A semianalytic radiance model of ocean color*, JGR 93(D9), 10909–10924,
doi:10.1029/JD093iD09p10909.

**Assumptions carried by (G1):** quasi-single-scattering with a fixed particle phase function;
no inelastic scattering (no Raman, no chlorophyll or CDOM fluorescence); a vertically
homogeneous, optically deep, plane-parallel water column; and a fixed sun-sensor geometry
absorbed into g₀, g₁. All four are violated to some degree by real field data. Raman is the
largest neglected term in clear blue water (it can reach 5–20 % of r_rs in the green/red where
water absorption is high); §8.3 provides the Lee et al. (2013) correction as a preprocessing
step, and it is *not* applied by default because the upstream code does not apply it.

Implemented in `giop.model.rrs_from_iops`. Ported from `giop_cost.m:5-8` and `giop.m:231-235`.

### G2 — additive decomposition of the IOPs

$$ a(\lambda) = a_w(\lambda) + a_\varphi(\lambda) + a_{dg}(\lambda), \qquad
   b_b(\lambda) = b_{bw}(\lambda) + b_{bp}(\lambda) \tag{G2} $$

a_w and b_bw are taken as known constants of pure seawater (§3), not retrieved. This is the
central structural assumption of GIOP: the water column is exactly three optically active
constituents plus water.

### G3 — eigenvector / eigenvalue expansion

Each unknown constituent is factored into a **fixed spectral shape** (the eigenvector, chosen by
parameterisation) times a **scalar amplitude** (the eigenvalue, retrieved):

$$ a_\varphi(\lambda) = M_\varphi \; a^*_\varphi(\lambda), \qquad
   a_{dg}(\lambda) = M_{dg} \; e^{-S_{dg}(\lambda - 443)}, \qquad
   b_{bp}(\lambda) = M_{bp} \left(\frac{443}{\lambda}\right)^{\eta} \tag{G3} $$

Because both non-φ eigenvectors are normalised to unity at 443 nm, the eigenvalues are the
constituent values at 443 nm: M_dg = a_dg(443), M_bp = b_bp(443). With the Bricaud a*_φ
normalised to 0.055 m² mg⁻¹ at 442 nm (§4.1), M_φ carries units of mg m⁻³ and is interpreted as
chlorophyll, though it is an absorption amplitude and inherits every error in a*_φ.

Implemented in `giop.model.eigenvectors`. Ported from `giop.m:163-164, 214-216`.

This is a **linear-in-amplitude, nonlinear-in-shape** model. The three shape parameters
(S_dg, η, and the choice of a*_φ) are *prescribed*, not fitted. That is the defining design
choice of GIOP, and it is what makes the inversion well-posed with only 5–6 bands: the shapes
carry the information that the bands cannot.

### G4 — air-sea interface

Field and satellite R_rs are measured above the surface; (G1) lives below it. The default
transformation is Lee et al. (2002), QAA:

$$ r_{rs} = \frac{R_{rs}}{0.52 + 1.7\,R_{rs}} \tag{G4a} $$

The alternative (`trans='flat'`) is the spectrally flat form of Gordon (2007):

$$ r_{rs} = R_{rs}/0.529 \tag{G4b} $$

(G4a) is nonlinear in R_rs and its inverse is

$$ R_{rs} = \frac{0.52\,r_{rs}}{1 - 1.7\,r_{rs}} \tag{G4c} $$

with a pole at r_rs = 1/1.7 = 0.588 sr⁻¹, far outside any ocean value.

Implemented in `giop.model.rrs_above_to_below` / `rrs_below_to_above`. Ported from `giop.m:86-92`.

> **Finding, upstream inconsistency.** `giop.m` returns `mrrs` from (G1), which is *subsurface*
> r_rs, but `run_giop.m:96` plots it directly against the *above-surface* input `rrs` and labels
> both "R_rs". The two differ by a factor of ~1.9. The internal QC test (`giop.m:247`) correctly
> compares `mrrs` against `rin`, so the retrieval is unaffected; only the demo figure is
> misleading. The Python port returns **both** `rrs_model_subsurface` and
> `rrs_model_above` (via G4c) with explicit names, and the GUI plots the above-water one against
> the above-water observation. See `PORTING_NOTES.md` D1.

---

## 3. Pure-seawater optical properties

### G5 — absorption of pure seawater

a_w(λ) is read from `optics_coef.txt` column 2, a 1 nm table spanning 380–1150 nm, and
interpolated to the target wavelengths. The companion file `pureH2O_iop.mat` documents the
provenance of the same quantity: Pope & Fry (1997) over 380–700 nm, Smith & Baker (1981) below
380 nm, Kou et al. (1993) above 700 nm. §6 of `VALIDATION.md` reports the measured agreement
between the two in-repo copies.

Implemented in `giop.water.a_water`. Ported from `get_aw.m`.

Pope & Fry (1997), Applied Optics 36(33), 8710–8723, doi:10.1364/AO.36.008710.

### G6 — backscattering of pure seawater

$$ b_{bw}(\lambda) = 0.0038 \left(\frac{400}{\lambda}\right)^{4.32} \tag{G6} $$

A power-law fit to Morel (1974). The exponent 4.32 is below the Rayleigh 4.0 + dispersion value
because it absorbs the wavelength dependence of the refractive index and of the density
fluctuation term.

Implemented in `giop.water.bb_water`. Ported from `get_bbw.m`.

**Known limitation, not fixed here.** (G6) is a *pure-water* fit with no salinity or temperature
dependence. The modern standard is Zhang, Hu & He (2009), doi:10.1364/OE.17.005698, which gives
b_bw for seawater as a function of S and T and is 15–30 % higher than (G6) in the blue for
S = 35. Bisson's own `single_wv_comps.m:135` carries the comment "update from Zhang!", so
upstream knew. The port keeps (G6) as the default for bit-comparability with upstream and
accepts a measured or externally computed table through `bb_water(..., model='table',
bbw_table=[[wl, bbw], ...])`. The Zhang formulation is deliberately **not** reimplemented:
it requires the density derivative of the refractive index, the isothermal compressibility
and the water activity, and a paraphrase of it would be a fabrication wearing a citation.
For a coastal or estuarine NaturaSpec deployment the difference is negligible against the
particulate signal; for clear oligotrophic water it is not.

---

## 4. Eigenvector parameterisations

### 4.1 a*_φ — Bricaud et al. (1998), the GIOP-DC default

Bricaud et al. (1998), JGR 103(C13), 31033–31044, doi:10.1029/98JC02712, give a power law in
chlorophyll at each wavelength on a 2 nm grid over 400–700 nm:

$$ a_\varphi(\lambda) = A_\varphi(\lambda)\,\mathrm{Chl}^{E_\varphi(\lambda)}, \qquad
   a^*_\varphi(\lambda) = A_\varphi(\lambda)\,\mathrm{Chl}^{E_\varphi(\lambda) - 1} \tag{G7} $$

GIOP-DC then renormalises so that a*_φ(442) = 0.055 m² mg⁻¹:

$$ a^*_\varphi \leftarrow a^*_\varphi \cdot \frac{0.055}{a^*_\varphi(442)} \tag{G8} $$

**Note the 442, not 443.** The Bricaud table is on a 2 nm grid containing 442 and not 443, and
`get_bricaud_aph.m:24` selects the exact row `dat(:,1)==442`. Every other 443 in GIOP is a true
443. Preserving this is required to reproduce the published eigenvalues; §5 of `VALIDATION.md`
shows what happens if you "fix" it to 443.

a*_φ depends on Chl, so the eigenvector depends on the *initial* Chl estimate passed in (from
OC4/OC3M, §8.1). GIOP does **not** iterate this: the a*_φ shape is frozen at the seed Chl while
M_φ is retrieved. The retrieved M_φ and the seed Chl are therefore not required to agree, and
their disagreement is a useful diagnostic (`giop.inversion.GiopResult.chl_seed` vs `.chl`).

Implemented in `giop.aphstar.bricaud1998`. Ported from `get_bricaud_aph.m`.

### 4.2 a*_φ — Ciotti & Bricaud (2006), size-fraction mixing

$$ a^*_\varphi(\lambda) = S_f\, a^*_{pico}(\lambda) + (1 - S_f)\, a^*_{micro}(\lambda) \tag{G9} $$

with S_f ∈ [0, 1] the picoplankton size fraction (default 0.5), and the two end-member vectors
tabulated on a 2 nm grid over 400–700 nm, each rescaled to a fixed a*(443)
(pico × 0.023/0.891, micro × 0.0086/1.249). Unlike Bricaud, this eigenvector does **not** depend
on Chl. Ciotti & Bricaud (2006), L&O Methods 4, 237–253, doi:10.4319/lom.2006.4.237.

Implemented in `giop.aphstar.ciotti2006`. Ported from `get_ciotti_aph.m`.

### 4.3 a*_φ — GSM

Six fixed values at [412, 443, 490, 510, 555, 670] nm, interpolated to the target grid.
Maritorena et al. (2002), Applied Optics 41(15), 2705–2714, doi:10.1364/AO.41.002705.

> `giop.m:143` interpolates these with `'cubic'` while `giop_kb.m:125` uses `'pchip'`. See §9.1;
> this is the single most consequential MATLAB-semantics decision in the port.

### 4.4 a_p — Chase et al. (2017), hyperspectral

$$ a_p(\lambda) = A_p(\lambda)\,\mathrm{Chl}^{E_p(\lambda)} $$

on a 148-point grid from `chase_ap17.mat`. Added by Bisson (`get_chase_ap.m`) and reachable only
through `giop_kb.m`. Note this is **a_p, total particulate absorption**, not a_φ: using it as the
a*_φ eigenvector double-counts the detrital fraction already carried by a_dg. `giop_kb.m:190-192`
contains a commented-out attempt to handle exactly this. The port exposes it as
`aph='chase'` with a runtime warning naming the double-count, because upstream exposes it.
Chase et al. (2017), JGR Oceans 122, 9725–9743, doi:10.1002/2017JC012859.

### 4.5 η — b_bp slope

Default (QAA v5, Lee et al. 2002):

$$ \eta = 2.0\left[1 - 1.2\exp\!\left(-0.9\,\frac{r_{rs}(443)}{r_{rs}(555)}\right)\right] \tag{G10} $$

Alternatives: GSM fixed η = 1.03373, or user-supplied. Note (G10) uses the **subsurface** ratio,
whereas the visually similar expression in `estimate_bbp_from_Rrs.m:98` also uses subsurface, but
`read_merged_files.m:42` uses the **above-surface** ratio. That last one is inconsistent with
QAA; flagged in `PORTING_NOTES.md` D4 and not reproduced.

### 4.6 S_dg — a_dg slope

Four options (`giop.m:112-131`):

| option | expression | source |
|---|---|---|
| default | 0.018 nm⁻¹ | GIOP-DC |
| `'qaa'` | 0.015 + 0.002 / (0.6 + r_rs(443)/r_rs(555)) | Lee et al. 2002 |
| `'obpg'` | 0.015 + 0.0038 log₁₀(R_rs(412)/R_rs(555)), clipped to [0.01, 0.02] | NASA OBPG, unpublished |
| `'gsm'` | 0.02061 nm⁻¹ | Maritorena et al. 2002 |

The `'obpg'` form uses **above-surface** R_rs while `'qaa'` uses subsurface r_rs. That asymmetry
is in the upstream source and is preserved deliberately (`PORTING_NOTES.md` D3).

---

## 5. The AOP–IOP relationship

Two ways to set (g₀, g₁):

1. **Gordon (default).** g₀ = 0.0949, g₁ = 0.0794, geometry-independent.
2. **Morel f/Q.** g₀ = f/Q(λ; Chl, θ_s, θ_v, Δφ) from the Morel et al. (2002) lookup tables, and
   **g₁ = 0**, which reduces (G1) to r_rs = (f/Q)·u. Morel et al. (2002), Applied Optics 41(30),
   6289–6306, doi:10.1364/AO.41.006289.

The Morel path has two sub-branches in `giop.m:57-67`: with no geometry supplied it uses the
Appendix-B polynomial (`morel_fq_appb.m`) and the tabulated f′ (`morel_read.m`), forming f′/Q;
with geometry supplied it interpolates the full 5-D f/Q table (7 λ × 6 θ_s × 6 Chl × 17 nadir ×
13 azimuth) from `morel_fq.dat` via `read_fq.m` and `get_fq.m`, with in-water refraction
θ′ = asin(sin θ_v / 1.34).

Implemented in `giop.aopiop`. Ported from `morel_fq.m`, `morel_fq_appb.m`, `morel_read.m`,
`read_fq.m`, `get_fq.m`.

> **Structural warning.** Under the Morel option g₁ = 0, and the LMI branch (§6.2) divides by g₁
> at `giop.m:190`. Morel + LMI is a division by zero in MATLAB (silently returning ±Inf/NaN, no
> error). The port raises `ConfigurationError` on that combination. `PORTING_NOTES.md` D2.

---

## 6. The inverse problem

Given measured r_rs(λ) at N bands, find the three amplitudes
**x** = (M_dg, M_bp, M_φ).

### 6.1 Nonlinear inversion (`inv='fmin'`, GIOP-DC default)

Minimise the unweighted sum of squared residuals in reflectance space:

$$ \chi^2(\mathbf{x}) = \sum_{i=1}^{N}\Big[r_{rs}(\lambda_i) - \hat r_{rs}(\lambda_i;\mathbf{x})\Big]^2 \tag{G11} $$

with $\hat r_{rs}$ from (G1)–(G3), by Nelder–Mead simplex from the start point
**x**₀ = (0.01, 0.001, Chl_seed).

Implemented in `giop.inversion._invert_fmin`, cost in `giop.model.cost`. Ported from
`giop_cost.m` and `giop.m:173-181`.

Three properties of (G11) worth stating plainly, because they bound everything downstream:

- **It is unweighted.** Each band contributes its squared residual in absolute sr⁻¹, so bands
  where r_rs is large (blue-green) dominate, and the red bands, where a_φ has its second peak and
  where b_bp is most cleanly separable, contribute almost nothing. There is no per-band σ, so
  no χ²_ν, and no covariance is produced. The retrieval has **no uncertainty estimate at all**.
  This is a property of GIOP as published, not of the port. `giop.uncertainty` adds an optional
  Jacobian-based covariance (§7) that upstream does not have; it is off by default.
- **Positivity is not enforced.** Nelder–Mead is unconstrained, so any of the three amplitudes
  can go negative, and does in absorbing/turbid water. Upstream handles this only after the fact,
  by flagging reconstructed spectra outside [−0.005, 5] as −999 (`giop.m:267-277`).
- **The start point matters.** Nelder–Mead on a 3-parameter problem with a curved valley is not
  guaranteed to find the global minimum. `gsm_invert.m:26-28` records exactly this experience:
  "NOTE THAT THE CHL RETRIEVALS ARE ESPECIALLY SENSITIVE TO ITS INITIAL GUESS." The port exposes
  `n_starts` for multistart and reports the spread; default 1, matching upstream.

### 6.2 Linear matrix inversion (`inv='lmi'`)

Because (G1) is a quadratic in u, it can be inverted analytically for u:

$$ u = \frac{-g_0 + \sqrt{g_0^2 + 4 g_1 r_{rs}}}{2 g_1} \tag{G12} $$

taking the positive root. Then, from u = b_b/(a + b_b),

$$ u\,(a_w + a_{dg} + a_\varphi) - (1-u)(b_{bw} + b_{bp}) = 0 $$

and substituting (G3) gives a linear system in **x**, **A x** = **b**, with

$$ \mathbf{A} = \Big[\, u\,\mathbf{e}_{dg} \;\;\big|\;\; (u-1)\,\mathbf{e}_{bp} \;\;\big|\;\; u\,\mathbf{a}^*_\varphi \,\Big], \qquad
   \mathbf{b} = (1-u)\,b_{bw} - u\,a_w \tag{G13} $$

an N×3 overdetermined system solved in least squares. This branch is **exact given u**: it makes
no linearisation, it simply exploits that the model is linear in the amplitudes once u is known.
The cost is that measurement noise enters through the square root in (G12) with no smoothing.

Implemented in `giop.inversion._invert_lmi`. Ported from `giop.m:186-208`.

**On the upstream solver.** `giop.m:203-207` computes, with `[Q,R] = qr(A)` returning the *full*
6×3 R:

```matlab
x = R \ (R' \ (A' * b));   r = b - A*x;   err = R \ (R' \ (A' * r));   x = x + err;
```

This looks unusual but is the **semi-normal equations with one step of iterative refinement**.
Writing R = [R₁; 0] with R₁ the 3×3 upper-triangular factor, the MATLAB basic solution of the
underdetermined `R' \ y` puts zeros in the last three components, and the subsequent least-squares
`R \ ·` discards them, so the composite operator is exactly

$$ \mathbf{x} = R_1^{-1} R_1^{-\!\top} \mathbf{A}^\top \mathbf{b} = (\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top\mathbf{b} $$

i.e. the normal-equations solution, computed through the QR factor, plus one refinement pass.
The port implements this identity directly on the economy factor and pins the equivalence with a
test against `scipy.linalg.lstsq` (`VALIDATION.md` §4). Refinement is retained because it changes
the answer in the 4th decimal on ill-conditioned band sets, and the published golden numbers
include it.

### 6.3 Identifiability

With 6 bands and 3 amplitudes the system is nominally overdetermined 2:1. The important point
is that conditioning **at fixed eigenvectors** and sensitivity **to the choice of eigenvectors**
are two different things, and for GIOP the second dominates.

Measured on the demo spectrum (`giop.diagnostics`, numbers in `VALIDATION.md` §5): the pairwise
angles between the three columns of **A** in (G13) are a_dg–b_bp 33.4°, a_dg–a_φ 27.5°,
b_bp–a_φ 28.5°, and cond(**A**) = 837. No pair is below the ~15° threshold at which a pair
stops being separable, so at the prescribed shapes this particular design is not degenerate,
and the linearised 1σ on the three amplitudes is only 1.4 %, 1.4 % and 4.2 % at a per-band
reflectance uncertainty of 5×10⁻⁵ sr⁻¹.

That number is misleading on its own. Re-inverting the same spectrum across the allowed grid of
S_dg, η and a*_φ prescriptions (60 members, all converged) gives a_dg(443) spanning
0.029–0.118 m⁻¹, b_bp(443) spanning 0.0021–0.0064 m⁻¹, and the a_φ amplitude spanning
−0.43 to +0.61, with **25 % of members returning a negative a_φ amplitude**. The prescription
error is therefore one to two orders of magnitude larger than the shape-conditional error, and
it is the CDOM/phytoplankton trade-off that carries it: the a_dg–a_φ correlation coefficient in
the linearised covariance is −0.52, so the two absorbers exchange amplitude while their sum
a_pg stays nearly fixed.

Two consequences that should travel with any GIOP number:

- **a_pg is far better constrained than its split into a_dg and a_φ.** Report the total unless
  the split is defended.
- **The retrieval has no positivity constraint** (§6.1), so a wrong S_dg can and does push a_φ
  negative rather than to zero. A negative a_φ is a signal that the prescription is wrong, not a
  measurement of negative absorption.

**Consequence for hyperspectral field data.** A NaturaSpec spectrum has hundreds of bands, but
they are not hundreds of independent constraints: the three eigenvectors remain three smooth
shapes, so the effective rank stays 3 and the extra bands buy noise averaging, not new degrees of
freedom. Adding bands does **not** relieve the a_dg/a_φ degeneracy. What would relieve it is
letting S_dg and η float, which converts the problem to 5 parameters and requires the
hyperspectral sampling to be worth anything. `giop.inversion.giop(..., fit_shapes=True)` provides
that as an explicit, off-by-default extension, listed in `PORTING_NOTES.md` §3 and **not** part
of GIOP-DC.

---

## 7. Uncertainty (an addition, not upstream)

Upstream GIOP returns no uncertainty. For a field instrument where replicate scans are available,
two estimators are provided in `giop.uncertainty`, both opt-in:

1. **Linearised covariance.** With J the N×3 Jacobian ∂r̂_rs/∂x at the solution and σ the
   per-band reflectance uncertainty,
   $$ \mathrm{Cov}(\mathbf{x}) \approx (\mathbf{J}^\top \Sigma^{-1} \mathbf{J})^{-1} $$
   This is a local, Gaussian, *shape-conditional* uncertainty. It does **not** include the error
   from prescribing S_dg, η and a*_φ, which is generally the larger term.
2. **Shape-ensemble spread.** Re-invert over the grid of allowed S_dg, η and a*_φ
   parameterisations and report the spread of the retrieved amplitudes. This captures the
   prescription error that (1) misses. `eta_expts.m` is upstream's manual version of exactly this
   experiment for η alone.

Neither is a posterior. Both are reported as ranges with the estimator named.

---

## 8. Empirical algorithms carried alongside

### 8.1 OC / KD band-ratio algorithms

$$ \log_{10}(\text{product}) = a_0 + a_1 X + a_2 X^2 + a_3 X^3 + a_4 X^4, \qquad
   X = \log_{10}\!\left(\frac{\max(R_{rs}^{blue})}{R_{rs}^{green}}\right) \tag{G14} $$

21 coefficient sets (OC4/OC3/OC2 variants per sensor, plus KD2). Used to produce the seed Chl for
the a*_φ eigenvector. Implemented in `giop.empirical.get_oc`. Ported from `get_oc.m`.

### 8.2 QAA v6 particulate backscattering

`estimate_bbp_from_Rrs.m` (N. Haentjens, U. Maine) implements QAA v6 to get b_bp at an arbitrary
wavelength, with the reference band switching from 555 to 670 nm when R_rs(670) ≥ 0.0015 sr⁻¹.
Independent of GIOP; retained as a cross-check on M_bp. Implemented in `giop.empirical.qaa_bbp`.

### 8.3 Raman correction, Lee et al. (2013)

$$ R_{rs}^{corr}(\lambda) = \frac{R_{rs}(\lambda)}{1 + RF(\lambda)}, \qquad
   RF(\lambda) = \alpha(\lambda)\frac{R_{rs}(443)}{R_{rs}(555)} + \beta_1(\lambda) R_{rs}(555)^{\beta_2(\lambda)} \tag{G15} $$

with coefficients tabulated at [412, 443, 488, 531, 555, 667] nm. Lee et al. (2013),
JGR Oceans 118, 4241–4255, doi:10.1002/jgrc.20308. Extracted from `single_wv_comps.m:105-117`,
which is otherwise an unportable personal analysis script. Implemented in
`giop.empirical.raman_correction`. **Defined only at those 6 bands**; applying it to a
hyperspectral grid requires interpolating α, β₁, β₂, which the port will do but flags, because
the published coefficients are band-specific fits and not a continuous function.

---

## 9. MATLAB semantics that change numbers

Full treatment in `PORTING_NOTES.md`. The three that alter results:

### 9.1 `interp1(..., 'cubic')`

MATLAB's `'cubic'` for `interp1` means shape-preserving PCHIP in current releases, but historically
selected `'v5cubic'` (Keys cubic convolution, which requires a uniformly spaced grid). The two
differ by up to a few tenths of a percent on smooth optical tables, and `v5cubic` overshoots where
PCHIP cannot. Evidence that upstream hit the difference: `giop_kb.m:125` changed exactly the one
`'cubic'` call that operates on a **non-uniform** grid (the 6-band GSM a*_φ) to `'pchip'`, which is
the change you make when `v5cubic` refuses non-uniform input.

Every other `'cubic'` in the codebase acts on a uniform grid (`optics_coef.txt` at 1 nm,
Ciotti at 2 nm, Morel at 1 nm), where the demo wavelengths 412/443/490/510/555/670 land exactly
on nodes, so both interpolants return the node value and the published golden numbers cannot
discriminate. The port therefore implements **both** (`giop.matlab_compat.interp1`), defaults to
`pchip`, and `VALIDATION.md` §3 measures the difference on a real hyperspectral grid where it does
bite.

### 9.2 `fminsearch` vs `scipy.optimize.minimize(method='Nelder-Mead')`

Same algorithm and, as it happens, the same initial-simplex rule (5 % relative perturbation per
coordinate, 0.00025 absolute for a zero coordinate). Termination differs in detail: MATLAB
requires simplex spread ≤ TolX **and** function spread ≤ TolFun; SciPy uses `xatol`/`fatol` the
same way. MATLAB's `exitflag < 1` (iteration/evaluation limit hit) maps to SciPy's
`result.status != 0`, and upstream returns −999 in that case. Reproduced exactly.

### 9.3 Indexing, empty tests, and silent broadcasting

`find(wl >= 410 & wl <= 415)` returns *all* matching indices and upstream takes `(1)`, the first.
On a hyperspectral grid that window contains ~5 bands, so "R_rs(412)" is whichever band is
lowest-indexed in [410, 415], not the closest to 412. The port selects the **nearest** band to
the nominal wavelength and records which one it used, because on a 1 nm field grid "first in
window" is arbitrary. This changes η and S_dg slightly for hyperspectral input and not at all for
the 6-band demo. `PORTING_NOTES.md` D5.

---

## 10. Above-water field radiometry: from a NaturaSpec `.sed` file to R_rs

This section is **not** in upstream GIOP, which starts from R_rs. A spectroradiometer does not
measure R_rs, so this is the layer that has to be right for the field data to mean anything.

### 10.1 What the instrument records

A Spectral Evolution `.sed` file (DARWin SP) carries a `Key: value` header, then a line `Data:`,
then a tab-separated table. The column vocabulary emitted by DARWin (verified by extracting the
string table from `DARWin2.exe` in the NaturaSpec Plus installer, not guessed) is:

| column | meaning | units |
|---|---|---|
| `Wvl` | wavelength | nm |
| `Rad. (Ref.)` / `Rad. (Target)` | calibrated radiance | W m⁻² sr⁻¹ nm⁻¹ |
| `Irr. (Ref.)` / `Irr. (Target)` | calibrated irradiance | W m⁻² nm⁻¹ |
| `DN (Ref.)` / `DN (Target)` / `Ref. DN` / `Tgt. DN` | raw counts | – |
| `Reflect. %` | 100 × target/reference | % |
| `Reflect. [1.0]` | target/reference | – |

Note `Irr.`, not `Irrad.`; generic third-party parsers assume the latter and drop the column.

### 10.2 The three-measurement above-water protocol

The standard method (Mobley 1999; NASA Ocean Optics Protocols; Ruddick et al. 2006) takes three
radiance scans with the same foreoptic:

- L_t(λ), total upwelling radiance from the water, viewed at θ_v ≈ 40° from nadir and
  Δφ ≈ 90–135° from the sun azimuth (the geometry that minimises specular sun glint);
- L_sky(λ), sky radiance at the mirror-image angle (θ_v from **zenith**, same azimuth);
- L_p(λ), radiance from a horizontal Spectralon reference panel of known reflectance R_p(λ).

Downwelling irradiance follows from the panel under the Lambertian assumption:

$$ E_d(\lambda) = \frac{\pi\,L_p(\lambda)}{R_p(\lambda)} \tag{G16} $$

and the water-leaving reflectance is

$$ R_{rs}(\lambda) = \frac{L_t(\lambda) - \rho\,L_{sky}(\lambda)}{E_d(\lambda)} - \Delta \tag{G17} $$

where ρ is the effective air-water interface reflectance for **radiance** at the viewing geometry
and Δ is a residual-correction offset.

**ρ is not the Fresnel coefficient.** It is a sea-state-, wind-, and geometry-dependent
effective factor that accounts for the distribution of wave facets, and its uncertainty is the
dominant error term in above-water radiometry. What `giop.io.rrs` implements:

- `rho` is a **float**, defaulting to `RHO_MOBLEY1999 = 0.028`, valid for wind below about
  5 m s⁻¹, θ_v = 40°, Δφ = 135°, clear sky.
- Any other value is supplied by the caller.

**Not implemented:** the wind- and geometry-dependent Mobley (2015) ρ lookup table. It is not
bundled and no interpolator for it exists here, so a deployment in wind above roughly
5 m s⁻¹ is using a ρ that does not describe its own sea state. At 10 m s⁻¹ that is a tens of
percent error in blue R_rs, and it is the largest single uncertainty in the field chain.
Treat the default as a placeholder to be replaced with a measured or tabulated value, not as
a calibration.

Mobley (1999), Applied Optics 38(36), 7442–7455, doi:10.1364/AO.38.007442.

### 10.3 Residual glint / offset correction Δ

Even at the optimal geometry, (G17) leaves a residual from sky-glint mismatch and from whitecaps.
Three options, all standard, none universally right:

| option | assumption | reference |
|---|---|---|
| `none` | Δ = 0 | – |
| `nir_zero` | R_rs(750–800 nm) = 0; subtract its mean | Ruddick et al. 2006 |
| `nir_similarity` | fixed R_rs(780)/R_rs(870) similarity ratio, 1.912 | Ruddick et al. 2006, L&O 51(2), 1167–1179, doi:10.4319/lo.2006.51.2.1167 |

`nir_zero` is invalid in turbid or sediment-laden water, where NIR R_rs is genuinely non-zero;
in that case the correction removes real signal. The port defaults to `none` and requires the
user to choose, because the right choice is a property of the water body, not of the software.

### 10.4 Spectral resampling to the inversion grid

The GIOP eigenvectors are defined over 400–700 nm; a NaturaSpec Plus reaches far beyond that. Two
resampling paths in `giop.io.resample`:

- **Band convolution** with a sensor spectral response function, for matching a satellite band
  set (the response for MODIS-Aqua ships upstream as `spectralresponse_modisa.mat`);
- **Gaussian/boxcar binning** to a chosen grid, for running the inversion hyperspectrally.

Both must be applied to R_rs, never to the retrieved IOPs, because the forward model is nonlinear
in the IOPs and band-averaging does not commute with (G1).

---

## 11. Assumption ledger

Everything the retrieved numbers are conditional on, in one place. A result quoted without these
is a result quoted without its error bar.

| # | assumption | where it enters | consequence if wrong |
|---|---|---|---|
| A1 | Water is 3 constituents + pure water | (G2) | any 4th absorber (mineral, ash, iron) is projected onto a_dg or b_bp |
| A2 | a_dg is a single exponential with prescribed S_dg | (G3) | biases the CDOM/detritus split and, through it, M_φ |
| A3 | b_bp is a single power law with prescribed η | (G3) | biases b_bp at all λ off 443; `eta_expts.m` measured this spread |
| A4 | a*_φ shape is known from Chl alone | (G7)–(G9) | pigment-packaging and community-composition error goes straight into M_φ |
| A5 | (g₀, g₁) fixed, geometry-independent | (G1) | 5–15 % in r_rs across the realistic solar/viewing envelope |
| A6 | No inelastic scattering | (G1) | Raman is up to 5–20 % in green/red clear water; §8.3 correction is opt-in |
| A7 | Vertically homogeneous, optically deep | (G1) | invalid over shallow bottom or in a surface bloom/slick layer |
| A8 | Pure-water IOPs known and S/T-independent | (G5), (G6) | b_bw 15–30 % low in the blue vs Zhang 2009 at S = 35 |
| A9 | Unweighted least squares in reflectance | (G11) | blue-green bands dominate; no uncertainty is produced |
| A10 | ρ is a single scalar (field data only) | (G17) | dominant error term above water; sea-state dependent |
| A11 | Reference panel is Lambertian, R_p known | (G16) | direct multiplicative bias on all R_rs |
| A12 | R_rs is reported at the MEASUREMENT geometry, not nadir-normalised | (G1) | 8-13 % against a satellite product; correctable with G18, which itself omits the transmittance term |
| A13 | Satellite band response is a nominal Gaussian, except MODIS-Aqua | (G19) | band placement alone moves R_rs by several percent between sensors |

---

## 12. Equation → code → upstream map

| eq | function | upstream |
|---|---|---|
| G1 | `model.rrs_from_iops` | `giop_cost.m:5-8`, `giop.m:231-235` |
| G2 | `model.total_iops` | `giop_cost.m:5-6` |
| G3 | `model.eigenvectors` | `giop.m:163-164, 214-216` |
| G4a–c | `model.rrs_above_to_below`, `rrs_below_to_above` | `giop.m:86-92` |
| G5 | `water.a_water` | `get_aw.m` |
| G6 | `water.bb_water` | `get_bbw.m` |
| G7, G8 | `aphstar.bricaud1998` | `get_bricaud_aph.m` |
| G9 | `aphstar.ciotti2006` | `get_ciotti_aph.m` |
| G10 | `model.eta_qaa` | `giop.m:97` |
| G11 | `model.cost`, `inversion._invert_fmin` | `giop_cost.m:12`, `giop.m:173-181` |
| G12, G13 | `inversion._invert_lmi` | `giop.m:186-208` |
| G14 | `empirical.get_oc` | `get_oc.m` |
| G15 | `empirical.raman_correction` | `single_wv_comps.m:105-117` |
| G16, G17 | `io.rrs.rrs_above_water` | none (new) |
| G18 | `aopiop.brdf_factor`, `normalize_brdf` | derived from `morel_fq.m`'s `fc` |
| G19 | `sensors.convolve` | `spectralresponse_modisa.mat` |
| G20 | `inversion._invert_bounded` | none (new) |

---

## 12b. Additions beyond GIOP-DC

All opt-in; the defaults still reproduce the published model exactly.

### G18 — BRDF normalisation to nadir viewing with the sun overhead

Because r_rs = (f/Q) u and u is a property of the water alone, the ratio between two
sun-sensor geometries is the ratio of their f/Q:

$$ R_{rs}(0,0) = R_{rs}(\theta_s,\theta_v,\Delta\varphi)\;
   \frac{f/Q(0,0)}{f/Q(\theta_s,\theta_v',\Delta\varphi)} \tag{G18} $$

with θ_v′ the in-water angle. That ratio is exactly the `fc` already returned by
`morel_fq_geometry`. **The air-water transmittance and refraction term is not
included** (A12).

### G19 — spectral convolution to a sensor band

$$ R_{rs}^{band} = \frac{\int S(\lambda) R_{rs}(\lambda)\,d\lambda}
                        {\int S(\lambda)\,d\lambda} \tag{G19} $$

with S the spectral response: measured for MODIS-Aqua, a nominal Gaussian otherwise
(A13). Applied to R_rs only, never to retrieved IOPs, because (G1) is nonlinear in the
IOPs and band-averaging does not commute with it.

### G20 — bounded, noise-weighted inversion

$$ \hat{\mathbf{x}} = \arg\min_{\mathbf{0}\le\mathbf{x}\le\mathbf{u}}
   \sum_i \left[\frac{r_{rs}(\lambda_i)-\hat r_{rs}(\lambda_i;\mathbf{x})}
   {\sigma_i}\right]^2 \tag{G20} $$

Two changes from (G11): amplitudes are bounded below at zero, and residuals are weighted
by a per-band σ so χ²_ν is interpretable. Bounding stops the retrieval being unphysical;
it does not make the prescribed shapes right, and a railed zero is itself a statement
that the model is wrong for that water.

## 13. Changelog

- **2026-08-14** — Initial theory document, written from a full read of the upstream MATLAB
  source before any Python was written. Documents the upstream defects found during that read
  (D1 subsurface-vs-above-surface plot, D2 Morel+LMI division by zero, D3 mixed R_rs/r_rs in
  the S_dg options, D5 first-in-window band selection) and the semi-normal-equations identity
  behind the LMI solver. Nine defects in total are catalogued in `PORTING_NOTES.md` §1; D9
  (the Morel full-geometry branch returning NaN outside 412.5–660 nm) was found later, by
  writing a test for a path upstream ships but never exercises.
- **2026-08-14, revision** — §3 corrected: the seawater b_bw alternative is a user-supplied
  table, not a reimplementation of Zhang et al. (2009). §10.2 corrected: the Mobley (2015)
  wind-dependent ρ lookup is **not** implemented, and ρ is a plain float. Both were
  descriptions of an intended API that the code does not provide.
