import os
from bisect import bisect

import h5py
import numpy as np

from refractor import framework as rf

from .retrieval_config import retrieval_config_definition

test_in_dir = os.path.join(os.path.realpath(os.path.dirname(os.path.realpath(__file__))), "../test/in")

sounding_id = "2015101019275775"

def initial_guess_values(use_aerosols):
    """Pull initial guess values from the retrieved results of an OCO-2 project data product"""

    ret_data_fn = os.path.join(test_in_dir, "oco2_L2DiaTG_06779a_151010_B8100r_170730115815-selected_ids.h5")

    def ig_helper():
        ret_data = h5py.File(ret_data_fn, "r")

        snd_id_list = [ "%d" % snd_id for snd_id in ret_data['/RetrievalHeader/sounding_id'] ]
        sounding_index = bisect(snd_id_list, sounding_id) - 1

        assert(snd_id_list[sounding_index] == sounding_id)

        ret_state = ret_data["RetrievedStateVector/state_vector_result"][sounding_index, :]
        ret_names = ret_data["RetrievedStateVector/state_vector_names"][sounding_index, :]

        ig_vals = []
        if use_aerosols:
            ig_vals = ret_state
        else:
            for name, val in zip(ret_names, ret_state):
                name = name.decode('utf-8')
                if not name.find("Aerosol") >= 0:
                    ig_vals.append(val)

        return np.array(ig_vals)
    
    return ig_helper

def rtr_comparison_base_config(**kwargs):
    """Make modifications to configuration so that ReFRACtor code matches the behavior of the
       OCO-2 software.
    """

    l1b_file = os.path.join(test_in_dir, "oco2_L1bScTG_06779a_151010_B8000r_170717081341-selected_ids.h5")
    met_file = os.path.join(test_in_dir, "oco2_ECMWFTG_06779a_151010_B8000r_170703073334-selected_ids.h5")
     

    config_def = retrieval_config_definition(l1b_file, met_file, sounding_id, **kwargs)

    # Reduce stopping point of window ranges by one to account for differences in behavior with old code
    mw = config_def['spec_win']['window_ranges']['value']
    for chan_idx in range(mw.shape[0]):
        mw[chan_idx, 0, 1] -= 1
    config_def['spec_win']['window_ranges']['value'] = mw

    # Use BRDF sufrace type
    config_def['atmosphere']['ground']['child'] = 'brdf'

    # Set to 1.0 to avoid difference where ReFRACtor has fixed the bug where table scale in
    # OCO code only applies itself within the low res, not high res spectral bounds
    config_def['atmosphere']['absorber']['CO2']['absorption']['table_scale'] = 1.0

    # Set z_matrix interpolation wavenumber ranges to what is used in expected daa
    rt_interp_wn = np.array([[[12962.5, 13172.1]],
                             [[6181.26, 6257.9]],
                             [[4808.22, 4883.57]]])
    rt_interp_win = rf.SpectralWindowRange(rf.ArrayWithUnit(rt_interp_wn, "cm^-1"))

    config_def['radiative_transfer']['spec_win'] = rt_interp_win

    return config_def
