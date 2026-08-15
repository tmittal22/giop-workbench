"""Data ingestion: field spectroradiometer files in, R_rs on an inversion grid out."""

from .fieldrrs import (
    FieldSpectrum,
    brdf_args_from_meta,
    is_fieldrrs_csv,
    read_fieldrrs_batch,
    read_fieldrrs_csv,
)
from .generic import read_csv_spectra, write_result_csv
from .resample import SATELLITE_BANDS, align, bin_spectrum, gaussian_resample
from .rrs import (
    RHO_MOBLEY1999,
    RrsResult,
    reflectance_factor_to_rrs,
    residual_correction,
    rrs_above_water,
    rrs_from_sed_triplet,
)
from .sed import SedSpectrum, read_sed, read_sed_dir

__all__ = [
    "read_sed", "read_sed_dir", "SedSpectrum",
    "rrs_above_water", "rrs_from_sed_triplet", "reflectance_factor_to_rrs",
    "residual_correction", "RrsResult", "RHO_MOBLEY1999",
    "align", "bin_spectrum", "gaussian_resample", "SATELLITE_BANDS",
    "read_csv_spectra", "write_result_csv",
    "read_fieldrrs_csv", "read_fieldrrs_batch", "is_fieldrrs_csv",
    "FieldSpectrum", "brdf_args_from_meta",
]
