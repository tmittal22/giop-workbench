# upstream_matlab — the original MATLAB GIOP, vendored verbatim

**This directory is not our code.** It is an unmodified copy of the reference MATLAB
implementation of GIOP, kept here so the Python port can be checked line by line against
the thing it was ported from.

## Source

| | |
|---|---|
| repository | **`kelseybisson/GIOP`** |
| URL | <https://github.com/kelseybisson/GIOP> |
| commit | `ef9b93fe948420a2a153853b80ec293fd972a829` |
| commit date | 2021-11-29 |
| commit author | Kelsey Bisson |
| retrieved | 2026-08-14 |
| files | 48 (26 `.m`, plus data tables and figures) |

Reproduce this directory exactly:

```bash
git clone https://github.com/kelseybisson/GIOP upstream_matlab
cd upstream_matlab && git checkout ef9b93fe948420a2a153853b80ec293fd972a829
rm -rf .git      # flattened here so it is not an embedded repository
```

The only change made to the contents is the removal of `.git/` and the addition of this
README. No `.m` file, data table or figure has been edited.

## Authorship

The reference GIOP code was written by **J. Werdell, NASA Goddard Space Flight Center**
(July 2013), and this fork carries additions by **K. Bisson** (Oregon State University,
2019). Individual file headers name their authors; `fastsmooth.m` is by T. C. O'Haver
(2008) and carries its own BSD-style notice, and `psrf.m` is by S. Särkkä and A. Vehtari
and carries a GPL notice.

The model is published as:

> Werdell, P.J., and 18 co-authors (2013). Generalized ocean color inversion model for
> retrieving marine inherent optical properties. *Applied Optics* **52**(10), 2019–2037.
> doi:[10.1364/AO.52.002019](https://doi.org/10.1364/AO.52.002019)

**Cite the paper, not this port.**

## ⚠ Licence status: unresolved

**The upstream repository contains no LICENSE or COPYING file.** Absent an explicit
grant, the default is all rights reserved, and that status is inherited by anything that
redistributes these files, including this repository. This is stated rather than assumed
away.

Mitigating facts, none of which is a substitute for a licence:

- The **optical data are published science** with their own citations (Pope & Fry 1997,
  Bricaud et al. 1998, Morel et al. 2002, Ciotti & Bricaud 2006, Chase et al. 2017). The
  measurements are attributable to their authors and are not original to this repository.
- The reference code and tables were produced by **NASA GSFC** and distributed publicly
  through the Ocean Biology Processing Group. Work by US federal employees is generally
  not subject to domestic copyright, but that turns on authorship details not
  established here.
- The same tables ship inside **SeaDAS/l2gen**, which is publicly released.
- Two files carry their own explicit licences (`fastsmooth.m` BSD-style, `psrf.m` GPL),
  which those files' terms govern regardless of the repository's silence.

If you intend to redistribute, the clean options are to obtain written permission from
P. J. Werdell (NASA GSFC) and/or K. Bisson, or to regenerate each table from the cited
publications or from SeaDAS under its own terms. Until then treat this directory as
third-party content of undetermined licence. **The licence of the Python port does not
extend to it.**

## How this directory is used

- **Reference for the port.** `PORTING_NOTES.md` cites these files by line number for
  every deviation, and `THEORY.md` §12 maps each equation to both the Python function and
  the MATLAB line it came from.
- **Source of the optical tables.** `scripts/fetch_data.py` copies the tables from here
  into `src/giop/data/` (falling back to downloading them if this directory is absent).
  The package data directory is gitignored so the same bytes do not sit in two places.
- **Golden values.** The comments in `run_giop.m` state the expected eigenvalues
  (`0.0441, 0.0033, 0.3693` for the nonlinear solver; `0.0414, 0.0022, 0.1058` for the
  linear matrix inversion). Those are the primary validation anchor for the port and are
  checked on every CI run.

## What is here, and what the port does with it

| upstream file | ported to | note |
|---|---|---|
| `giop.m`, `giop_cost.m` | `giop.inversion`, `giop.model` | the model itself |
| `giop_kb.m` | folded into the above | Bisson's variant; differences noted in PORTING_NOTES |
| `get_aw.m`, `get_bbw.m` | `giop.water` | |
| `get_bricaud_aph.m`, `get_ciotti_aph.m`, `get_chase_ap.m` | `giop.aphstar` | |
| `morel_fq*.m`, `morel_read.m`, `read_fq.m`, `get_fq.m` | `giop.aopiop` | includes the BRDF factor |
| `get_oc.m` | `giop.empirical.get_oc` | |
| `estimate_bbp_from_Rrs.m` | `giop.empirical.qaa_bbp` | QAA v6, by N. Haentjens |
| `fastsmooth.m` | `giop.matlab_compat.fastsmooth` | CR-only line endings; reconstructed |
| `run_giop.m`, `run_giop_cr.m` | `examples/demo_giop.py`, tests | |
| `single_wv_comps.m`, `eta_expts.m`, `goci_play.m`, `read_merged_files.m` | **not ported** | hard-coded paths to the author's machine, missing input data, and one call to a function that does not exist anywhere in the repository |
| `gsm_invert.m`, `gsm_cost.m` | **not ported** | GSM is a different model; GIOP reproduces its *parameterisations* instead |
| `psrf.m` | **not ported** | Gelman–Rubin diagnostic, unrelated to GIOP and unused by it |

Nine defects found in this code while porting it are catalogued in
[`../PORTING_NOTES.md`](../PORTING_NOTES.md) §1. They are reported there as properties of
the original, with the line numbers, and none is fixed silently.
