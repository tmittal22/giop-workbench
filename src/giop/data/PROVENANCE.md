# Reference tables — provenance and licensing status

Every file here is copied by `scripts/fetch_data.py` out of `upstream_matlab/`, the
verbatim vendored copy of <https://github.com/kelseybisson/GIOP> @ `ef9b93f`, itself a
fork of the reference GIOP distribution released by J. Werdell (NASA GSFC, July 2013).
The copies are byte-identical, and verified so: the SHA-256 prefixes printed by the fetch
script match whether the files are taken from `upstream_matlab/` or downloaded fresh.

This directory is **populated, not committed**. See `upstream_matlab/README.md` for the
source, the commit, and the licence position.

## ⚠ Licensing is unresolved

**The upstream repository contains no LICENSE or COPYING file.** Absent an explicit grant, the
default is all rights reserved, and that status is inherited by anything that redistributes
these files. This is stated rather than assumed away.

Mitigating facts, none of which is a substitute for a licence:

- The **underlying optical data are published science** with their own citations (below), not
  original to the repository. The measurements are attributable to their authors.
- The reference GIOP code and tables were produced by **NASA Goddard Space Flight Center** and
  distributed publicly through the Ocean Biology Processing Group. Work by US federal
  employees is generally not subject to domestic copyright, but that turns on authorship
  details not established here.
- The same tables are distributed inside **SeaDAS/l2gen**, which is publicly released software.

If this package is redistributed, the honest options are: obtain written permission from
P. J. Werdell (NASA GSFC) and/or K. Bisson; or replace each table with a copy generated
directly from the cited publication or from SeaDAS under its own terms. Until then, treat this
directory as third-party content of undetermined licence, and note that the MIT-style terms of
the port itself do **not** extend to it.

## Files

| file | contents | used by | original source |
|---|---|---|---|
| `optics_coef.txt` | 1 nm grid, 380–1150 nm. Col 1 wavelength; **col 2 pure-water absorption a_w** (the only column the model reads); col 3 pure-water scattering b_w (verified: b_w/2 at 443 nm = 0.0024362 against 0.0024447 from the b_bw power law, 0.35 % apart). Remaining columns are not used and are deliberately not identified here rather than guessed. | `water.a_water` | Pope & Fry (1997) 380–700 nm; Smith & Baker (1981) below 380 nm; Kou et al. (1993) above 700 nm, per the NOTES field of `pureH2O_iop.mat` |
| `pureH2O_iop.mat` | `wave`, `aw`, `bw`, and a `NOTES` string giving the provenance above | validation only; no code path reads it | as above |
| `bricaud_1998_aph.txt` | comma-delimited, 2 nm grid 400–700 nm. Cols: wavelength, A_p, E_p, A_ph, E_ph. **Contains 442 nm and not 443 nm**, which is why the GIOP-DC normalisation is anchored at 442 | `aphstar.bricaud1998` | Bricaud et al. (1998), JGR 103(C13), 31033–31044, doi:10.1029/98JC02712 |
| `chase_ap17.mat` | 148 × 3: wavelength, A_p, E_p for total particulate absorption | `aphstar.chase2017` | Chase et al. (2017), JGR Oceans 122, 9725–9743, doi:10.1002/2017JC012859 |
| `morel_fq_appb.txt` | 24 × 7: f0, sf, q0, sq blocks over 7 wavelengths × 6 chlorophylls | `aopiop.morel_fq_appb` | Morel et al. (2002) Appendix B, Applied Optics 41(30), 6289–6306, doi:10.1364/AO.41.006289 |
| `morel_f.txt`, `morel_fp.txt`, `morel_mud.txt` | 490 × 8: solar zenith, wavelength (70 values, 5 nm, 352.5–697.5 nm), then 6 chlorophyll columns [0.03 … 10] | `aopiop.morel_read` | lookup tables provided by A. Morel, per `morel_read.m` header |
| `morel_fq.dat` | 4284 × 13 = (7 λ × 6 θ_s × 6 Chl × 17 nadir) rows × 13 azimuths | `aopiop.read_fq` | translated from SeaDAS/l2gen, per `read_fq.m` header |
| `spectralresponse_modisa.mat` | 1821 × 17 MODIS-Aqua spectral response functions | `sensors.modis_srf`, `sensors.convolve` (the measured MODIS-Aqua response) | NASA OBPG |

## Not carried over

`inputsnums.mat`, `longhurst.mat`, `longviirs.mat`, `ls2results.mat`, `model_giopbbp700s.mat`,
`modis_lat.mat`, `modishr.mat`, `paramcheck_20mar19.mat`, `v_giop.mat`, `v_resid.mat` are inputs
and outputs of K. Bisson's Argo-float / lidar match-up analysis, not model reference data.
They remain in `upstream_matlab/` only.

## Full citations

- Pope, R.M., and Fry, E.S. (1997). Absorption spectrum (380–700 nm) of pure water. II.
  Integrating cavity measurements. *Applied Optics* 36(33), 8710–8723. doi:10.1364/AO.36.008710
- Smith, R.C., and Baker, K.S. (1981). Optical properties of the clearest natural waters
  (200–800 nm). *Applied Optics* 20(2), 177–184. doi:10.1364/AO.20.000177
- Kou, L., Labrie, D., and Chylek, P. (1993). Refractive indices of water and ice in the
  0.65–2.5 µm spectral range. *Applied Optics* 32(19), 3531–3540. doi:10.1364/AO.32.003531
- Morel, A. (1974). Optical properties of pure water and pure sea water. In *Optical Aspects of
  Oceanography*, Academic Press, 1–24.
- Bricaud, A., Morel, A., Babin, M., Allali, K., and Claustre, H. (1998). Variations of light
  absorption by suspended particles with chlorophyll a concentration in oceanic (case 1)
  waters. *JGR* 103(C13), 31033–31044. doi:10.1029/98JC02712
- Morel, A., Antoine, D., and Gentili, B. (2002). Bidirectional reflectance of oceanic waters:
  accounting for Raman emission and varying particle scattering phase function.
  *Applied Optics* 41(30), 6289–6306. doi:10.1364/AO.41.006289
- Ciotti, A.M., and Bricaud, A. (2006). Retrievals of a size parameter for phytoplankton and
  spectral light absorption by colored detrital matter from water-leaving radiances at SeaWiFS
  channels in a continental shelf region off Brazil. *L&O Methods* 4, 237–253.
  doi:10.4319/lom.2006.4.237
- Chase, A.P., Boss, E., Cetinić, I., and Slade, W. (2017). Estimation of phytoplankton
  accessory pigments from hyperspectral reflectance spectra: toward a global algorithm.
  *JGR Oceans* 122, 9725–9743. doi:10.1002/2017JC012859
