import os

import h5py

from refractor.executor.testing import ComparisonExecutor

config_base_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../../config")
expt_results_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../expected/rtr_comparison")

def compare_fm(config_filename, expt_results_filename):

    expt_data = h5py.File(expt_results_filename)

    exc = ComparisonExecutor(config_filename)
    exc.execute_simulation()

    for spec_idx, (conv_spec, hr_spec) in zip(exc.captured_radiances.convolved_spectrum, exc.captured_radiances.high_res_spectrum):
        expt_conv_rad = expt_data['Spectrum_{}/convolved/radiance'.format(spec_idx+1)][:]
        expt_hr_rad = expt_data['Spectrum_{}/monochromatic/radiance'.format(spec_idx+1)][:]

        calc_conv_rad = conv_spec.spectral_domain.data
        calc_hr_rad = hr_spec.spectral_domain.data

        # convolved radiances are on the order of magnitude of ~1e19
        # Use this method so we can better control absolute tolerance
        npt.assert_allclose(expt_conv_rad, calc_conv_rad, rtol=1e-3, atol=1e15)

        # monochromatic radiance should be the same to almost machine precision
        npt.assert_almost_equal(expt_hr_rad, calc_hr_rad, decimal=9)

# Expected results captured from RtRetrievalFramework revision e096dc
# using notebooks/computed_expected_radiances.ipynb from
# git@github.jpl.nasa.gov:OCO/fp_lua_notebooks.git

def test_clear():

    config_filename = os.path.join(config_base_dir, "rtr_comparison_clear.py")
    expt_results_filename = os.path.join(expt_results_dir, "rtr_expected_radiances_clear.h5")
    compare_fm(config_filename, expt_results_filename)

def test_aerosols():

    config_filename = os.path.join(config_base_dir, "rtr_comparison_aerosols.py")
    expt_results_filename = os.path.join(expt_results_dir, "rtr_expected_radiances_aerosols.h5")
    compare_fm(config_filename, expt_results_filename)
