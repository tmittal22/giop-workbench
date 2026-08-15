"""Fetch the reference optical tables from their original source.

    python scripts/fetch_data.py

WHY THIS EXISTS INSTEAD OF THE FILES THEMSELVES. The upstream GIOP repository carries
**no licence file**, so its terms are unresolved and redistributing its tables inside
this repository would pass that unresolved question on to everyone who clones it. The
underlying measurements are published science (Pope & Fry 1997, Bricaud et al. 1998,
Morel et al. 2002, Ciotti & Bricaud 2006, Chase et al. 2017) and the encodings came
through NASA GSFC, but neither fact is a licence.

So this script downloads them from the original public repository, which means you
obtain them from the source on the same terms the source offers, rather than from a
redistribution by someone with no right to redistribute.

See src/giop/data/PROVENANCE.md for each file's citation and status.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.error
import urllib.request

RAW = "https://raw.githubusercontent.com/kelseybisson/GIOP/{commit}/{name}"
COMMIT = "ef9b93fe948420a2a153853b80ec293fd972a829"   # pinned; see PORTING_NOTES.md

FILES = [
    "optics_coef.txt",            # pure-water absorption (G5)
    "bricaud_1998_aph.txt",       # aph* power-law coefficients (G7)
    "morel_fq_appb.txt",          # Morel Appendix B
    "morel_f.txt",
    "morel_fp.txt",
    "morel_mud.txt",
    "morel_fq.dat",               # the 5-D f/Q table
    "chase_ap17.mat",             # Chase et al. 2017 a_p
    "pureH2O_iop.mat",            # independent a_w copy, used for validation
    "spectralresponse_modisa.mat",  # measured MODIS-Aqua SRF
]

DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "src", "giop", "data")


def fetch(name, dest_dir, timeout=60):
    url = RAW.format(commit=COMMIT, name=name)
    out = os.path.join(dest_dir, name)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return "present", out, os.path.getsize(out)
    req = urllib.request.Request(url, headers={"User-Agent": "giop-fetch"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    if not data:
        raise RuntimeError("empty response for %s" % name)
    with open(out, "wb") as fh:
        fh.write(data)
    return "downloaded", out, len(data)


def main():
    if not os.path.isdir(DEST):
        os.makedirs(DEST)
    print("Fetching %d reference tables from kelseybisson/GIOP @ %s\n"
          % (len(FILES), COMMIT[:8]))
    failed = []
    for name in FILES:
        try:
            status, path, size = fetch(name, DEST)
            digest = hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]
            print("  %-30s %-11s %8d bytes  sha256:%s" % (name, status, size, digest))
        except (urllib.error.URLError, RuntimeError, OSError) as exc:
            print("  %-30s FAILED  %s" % (name, exc))
            failed.append(name)

    if failed:
        print("\n%d file(s) failed. Without them the package cannot run." % len(failed))
        print("If you have no internet, copy them from a checkout of")
        print("  https://github.com/kelseybisson/GIOP")
        print("into %s" % DEST)
        return 1

    print("\nDone. Now: python -m pytest tests/ -q")
    return 0


if __name__ == "__main__":
    sys.exit(main())
