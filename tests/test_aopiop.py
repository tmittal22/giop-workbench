"""Morel et al. (2002) f/Q AOP-IOP option. THEORY.md sect. 5.

There is no published reference output for this path in upstream, so these tests pin
structural and physical properties rather than values: table shapes, monotonicity in solar
zenith, the g1 = 0 collapse, and that the resulting g0 lands in the range Morel's tables
actually span.
"""

import numpy as np
import pytest

from giop import get_oc, giop
from giop.aopiop import morel_fq_appb, morel_g0, morel_read, read_fq

WL = np.array([412.0, 443, 490, 510, 555, 670])
RRS = np.array([0.003478, 0.004074, 0.004465, 0.003588, 0.002494, 0.000051])


class TestTables:
    def test_fq_table_has_the_documented_shape(self):
        """read_fq.m declares 7 wl x 6 solz x 6 chl x 17 nadir x 13 azimuth = 4284 rows."""
        foq = read_fq()
        assert foq.shape == (7, 6, 6, 17, 13)
        assert np.all(np.isfinite(foq)) and np.all(foq > 0)

    def test_morel_read_covers_the_full_output_grid(self):
        """The tables stop at 697.5 nm while the output grid runs to 700 nm; upstream
        back-fills the trailing NaNs (morel_read.m:82-84) and so must the port."""
        val = morel_read(0.5, 30.0, "fp")
        assert val.shape == (321,)                       # 380-700 nm at 1 nm
        assert np.all(np.isfinite(val)), "trailing NaNs above 697.5 nm were not filled"

    def test_appb_returns_smoothed_f_and_q(self):
        f, q = morel_fq_appb(0.5, 30.0)
        assert f.shape == q.shape == (321,)
        assert np.all(np.isfinite(f)) and np.all(np.isfinite(q))
        assert np.all(q > 0)

    def test_f_over_q_is_physically_plausible(self):
        """f/Q is the AOP-IOP factor that replaces Gordon's g0 = 0.0949, so it has to
        land in the same neighbourhood or the whole option is misconfigured."""
        g0 = morel_g0(WL, 0.5, solz=30.0)
        assert np.all(g0 > 0.05) and np.all(g0 < 0.20)


class TestGeometryDependence:
    def test_g0_varies_with_solar_zenith(self):
        """If it did not, the Morel option would be an expensive constant and the whole
        point of using it (geometry dependence) would be absent."""
        a = morel_g0(WL, 0.5, solz=0.0)
        b = morel_g0(WL, 0.5, solz=60.0)
        assert np.max(np.abs(a - b) / a) > 0.01

    def test_g0_varies_with_chlorophyll(self):
        a = morel_g0(WL, 0.05, solz=30.0)
        b = morel_g0(WL, 5.0, solz=30.0)
        assert np.max(np.abs(a - b) / a) > 0.01

    def test_full_geometry_branch_runs_and_differs_from_the_appendix_branch(self):
        """giop.m:57-67 takes different code paths depending on whether viewing geometry
        was supplied. Both must work, and they are not the same calculation."""
        no_geom = morel_g0(WL, 0.5, solz=30.0)
        with_geom = morel_g0(WL, 0.5, solz=30.0, senz=20.0, relaz=90.0)
        assert np.all(np.isfinite(with_geom))
        assert not np.allclose(no_geom, with_geom, rtol=1e-3)


class TestInversionUnderMorel:
    def test_g1_collapses_to_zero_and_the_fit_converges(self):
        chl = float(get_oc(RRS[1], RRS[2], RRS[3], RRS[4], "oc4"))
        res = giop(WL, RRS, chl, fq="morel", solz=30.0)
        assert res.g1 == 0.0
        assert np.size(res.g0) == len(WL)          # g0 is spectral here, not scalar
        assert res.converged

    def test_morel_and_gordon_give_similar_but_distinct_answers(self):
        """A sanity band: the two AOP-IOP closures should agree to order tens of percent
        on the same spectrum. Order-of-magnitude disagreement would mean one is wrong."""
        chl = float(get_oc(RRS[1], RRS[2], RRS[3], RRS[4], "oc4"))
        g = giop(WL, RRS, chl).x
        m = giop(WL, RRS, chl, fq="morel", solz=30.0).x
        ratio = m / g
        assert np.all(ratio > 0.5) and np.all(ratio < 2.0)
        assert not np.allclose(m, g, rtol=1e-3)


class TestBrdfNormalisation:
    """R_rs(nadir, sun overhead) = R_rs(geometry) x f/Q(0,0) / f/Q(geometry).

    Satellite ocean-colour products are exact-normalised; field data is not. These pin
    the correction that converts one to the other.
    """

    WL = np.array([412.0, 443, 490, 510, 555, 670])

    def test_identity_at_nadir_with_the_sun_overhead(self):
        """The definitional check: correcting the reference geometry to itself is 1."""
        from giop.aopiop import brdf_factor
        f = brdf_factor(self.WL, 0.5, solz=0.0, senz=0.0, relaz=0.0)
        np.testing.assert_allclose(f, 1.0, atol=1e-12)

    def test_correction_grows_with_solar_zenith(self):
        from giop.aopiop import brdf_factor
        dev = [abs(np.mean(brdf_factor(self.WL, 0.5, s, 40.0, 135.0)) - 1.0)
               for s in (10.0, 30.0, 60.0)]
        assert dev[0] < dev[1] < dev[2]

    def test_correction_is_material_at_field_geometry(self):
        """If it were sub-percent it would not be worth correcting."""
        from giop.aopiop import brdf_factor
        f = brdf_factor(self.WL, 0.5, 45.0, 40.0, 135.0)
        assert 0.05 < abs(np.mean(f) - 1.0) < 0.30

    def test_normalize_applies_exactly_the_factor(self):
        from giop.aopiop import brdf_factor, normalize_brdf
        rrs = np.full_like(self.WL, 3e-3)
        out, f = normalize_brdf(self.WL, rrs, 0.5, 45.0, 40.0, 135.0)
        np.testing.assert_allclose(out, rrs * f, rtol=1e-12)
        np.testing.assert_allclose(f, brdf_factor(self.WL, 0.5, 45.0, 40.0, 135.0))

    def test_depends_on_chlorophyll(self):
        from giop.aopiop import brdf_factor
        a = brdf_factor(self.WL, 0.05, 45.0, 40.0, 135.0)
        b = brdf_factor(self.WL, 5.0, 45.0, 40.0, 135.0)
        assert np.max(np.abs(a - b)) > 0.01
