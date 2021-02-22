import os

import numpy as np
import numpy.testing as npt
import h5py

from refractor.executor.testing import ComparisonExecutor

config_base_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../../config")
expt_results_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../expected/rtr_comparison")

def compare_fm(config_filename, expt_results_filename):

    expt_data = h5py.File(expt_results_filename)

    exc = ComparisonExecutor(config_filename)
    exc.execute_simulation()

    if len(exc.captured_radiances.convolved_spectrum) == 0:
        raise Exception("No convolved radiances captured")

    if len(exc.captured_radiances.high_res_spectrum) == 0:
        raise Exception("No high resolution spectrum captured")

    for sensor_idx, (conv_spec, hr_spec) in enumerate(zip(exc.captured_radiances.convolved_spectrum, exc.captured_radiances.high_res_spectrum)):

        expt_hr_grid = expt_data['Spectrum_{}/monochromatic/grid'.format(sensor_idx+1)][:]

        expt_conv_rad = expt_data['Spectrum_{}/convolved/radiance'.format(sensor_idx+1)][:]
        expt_hr_rad = expt_data['Spectrum_{}/monochromatic/radiance'.format(sensor_idx+1)][:]

        calc_hr_grid = hr_spec.spectral_domain.data

        calc_conv_rad = conv_spec.spectral_range.data
        calc_hr_rad = hr_spec.spectral_range.data

        # convolved radiances are on the order of magnitude of ~1e19
        # Use this method so we can better control absolute tolerance
        npt.assert_allclose(expt_conv_rad, calc_conv_rad, rtol=2e-3)

        # monochromatic radiance should have a higher degree of similarity
        # Grids are not always 100% the same exact length, may be off by 1 point, so 
        # check only comparable regions
        w_compare_calc = np.where(np.logical_and(calc_hr_grid >= expt_hr_grid[0], 
                                                 calc_hr_grid <= expt_hr_grid[-1]))
        w_compare_expt = np.where(np.logical_and(expt_hr_grid >= calc_hr_grid[w_compare_calc][0], 
                                                 expt_hr_grid <= calc_hr_grid[w_compare_calc][-1]))

        npt.assert_allclose(expt_hr_rad[w_compare_expt], calc_hr_rad[w_compare_calc], rtol=2e-5)

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
