"""Reading `fieldrrs` output directly, metadata and all.

The point of parsing the header rather than just the two data columns: an R_rs is not
interpretable without the conditions it was measured under. rho depends on wind, the
BRDF correction needs the geometry, and nir_zero is invalid in turbid water.
"""

import os

import numpy as np
import pytest

from giop.io import (
    brdf_args_from_meta, is_fieldrrs_csv, read_fieldrrs_batch, read_fieldrrs_csv,
)

HEADER = """# fieldrrs above-water Rrs
# rho,0.028,residual_method,{method},offset_sr-1,0.0
#,solar_zenith_deg,45.00
#,view_zenith_from_nadir_deg,40.0
#,relative_azimuth_from_sun_deg,135.0
#,wind_speed_ms,{wind}
#,footprint_area_m2,0.300
{warn}wavelength_nm,Rrs_sr-1
400.0,0.00300
500.0,0.00420
600.0,0.00250
700.0,{red}
"""


def write(tmp, method="none", wind="3.5", warn="", red="0.00010"):
    p = os.path.join(tmp, "station.csv")
    with open(p, "w") as fh:
        fh.write(HEADER.format(method=method, wind=wind, warn=warn, red=red))
    return p


def test_detects_a_fieldrrs_file(tmp_path):
    assert is_fieldrrs_csv(write(str(tmp_path)))


def test_reads_spectrum_and_metadata(tmp_path):
    s = read_fieldrrs_csv(write(str(tmp_path)))
    np.testing.assert_allclose(s.wavelength, [400, 500, 600, 700])
    np.testing.assert_allclose(s.rrs, [0.003, 0.0042, 0.0025, 0.0001])
    assert s.rho == 0.028
    assert s.wind_ms == 3.5
    assert s.solar_zenith == 45.0
    assert s.view_zenith == 40.0
    assert s.relative_azimuth == 135.0
    assert s.footprint_area_m2 == 0.3


def test_clean_station_flags_nothing(tmp_path):
    assert read_fieldrrs_csv(write(str(tmp_path))).review() == []


def test_high_wind_with_default_rho_is_flagged(tmp_path):
    """rho = 0.028 is only valid below ~5 m/s; using it above that biases blue high."""
    s = read_fieldrrs_csv(write(str(tmp_path), wind="11.0"))
    assert any("only valid below" in m for m in s.review())


def test_missing_wind_is_flagged(tmp_path):
    s = read_fieldrrs_csv(write(str(tmp_path), wind="NOT RECORDED"))
    assert any("not recorded" in m.lower() for m in s.review())


def test_nir_zero_carries_its_turbid_water_caveat(tmp_path):
    s = read_fieldrrs_csv(write(str(tmp_path), method="nir_zero"))
    assert any("DELETES real signal" in m for m in s.review())


def test_negative_visible_bands_are_flagged(tmp_path):
    s = read_fieldrrs_csv(write(str(tmp_path), red="-0.00050"))
    assert any("negative" in m for m in s.review())


def test_warnings_in_the_header_are_carried_through(tmp_path):
    p = write(str(tmp_path), warn="# WARNING,glint subtraction over-corrected\n")
    s = read_fieldrrs_csv(p)
    assert any("over-corrected" in w for w in s.warnings)
    assert any("over-corrected" in m for m in s.review())


def test_brdf_args_come_from_the_header(tmp_path):
    s = read_fieldrrs_csv(write(str(tmp_path)))
    chl, solz, senz, relaz = brdf_args_from_meta(s)
    assert (solz, senz, relaz) == (45.0, 40.0, 135.0)


def test_brdf_args_are_none_without_geometry(tmp_path):
    """Refusing is the honest outcome: the correction cannot be applied blind."""
    p = os.path.join(str(tmp_path), "bare.csv")
    with open(p, "w") as fh:
        fh.write("# fieldrrs above-water Rrs\nwavelength_nm,Rrs_sr-1\n400.0,0.003\n"
                 "500.0,0.004\n")
    assert brdf_args_from_meta(read_fieldrrs_csv(p)) is None


def test_batch_file_gives_one_spectrum_per_station(tmp_path):
    p = os.path.join(str(tmp_path), "rrs_all_stations.csv")
    with open(p, "w") as fh:
        fh.write("station,443.0,490.0,555.0\nst1,0.003,0.004,0.002\n"
                 "st2,0.005,0.006,0.003\n")
    out = read_fieldrrs_batch(p)
    assert [x.name for x in out] == ["st1", "st2"]
    np.testing.assert_allclose(out[1].rrs, [0.005, 0.006, 0.003])
    assert any("not recorded" in m.lower() or "geometry" in m.lower()
               for m in out[0].review())


def test_a_non_fieldrrs_csv_is_refused_clearly(tmp_path):
    p = os.path.join(str(tmp_path), "other.csv")
    with open(p, "w") as fh:
        fh.write("a,b\n1,2\n")
    with pytest.raises(ValueError):
        read_fieldrrs_csv(p)
