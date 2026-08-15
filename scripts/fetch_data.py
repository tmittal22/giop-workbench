"""Fetch the reference optical tables from their original source.

    python scripts/fetch_data.py

The tables live in the vendored upstream_matlab/ directory, and this copies them
into src/giop/data/ where the package looks for them. If upstream_matlab/ is
absent it downloads them instead, from the original repository at a pinned commit.

Two copies of the same bytes in one repository drift apart, so the package data
directory is gitignored and populated by this script rather than committed.

LICENCE. The upstream repository carries no licence file, so the status of these tables
is unresolved. See upstream_matlab/README.md and src/giop/data/PROVENANCE.md.
"""

from __future__ import annotations

import hashlib
import os
import shutil
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


VENDORED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "upstream_matlab")


def fetch(name, dest_dir, timeout=60):
    """Copy from the vendored upstream_matlab/ if present, else download."""
    out = os.path.join(dest_dir, name)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return "present", out, os.path.getsize(out)

    local = os.path.join(VENDORED, name)
    if os.path.exists(local) and os.path.getsize(local) > 0:
        shutil.copyfile(local, out)
        return "vendored", out, os.path.getsize(out)

    url = RAW.format(commit=COMMIT, name=name)
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
    src = ("the vendored upstream_matlab/" if os.path.isdir(VENDORED)
           else "kelseybisson/GIOP @ %s (download)" % COMMIT[:8])
    print("Populating src/giop/data/ with %d reference tables from %s\n"
          % (len(FILES), src))
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
